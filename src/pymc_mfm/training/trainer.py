"""Canonical model training with structured records and atomic checkpoints.

``train`` seeds model construction, optimizes every active objective with
AdamW, applies the configured learning-rate schedule and gradient clipping,
writes milestone and evaluation checkpoints, and exports terminal weights
after a completed training loop. It records every active loss, evaluation,
selection decision, and checkpoint identity locally.

Evaluation keeps joint-router and graph-conditional decoder readouts separate.
Checkpoint selection uses the held-out joint-query readout; it is explicitly
not a conditional-decoder quality claim. Missing, non-finite, or partially
defined gate metrics block promotion and remain visible in the records.
"""

from __future__ import annotations

import json
import math
import platform
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pymc_mfm.inference.checkpoint import save_model
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.model.model import MFM, objective_trace_to_jsonable
from pymc_mfm.training.checkpoint import save_checkpoint
from pymc_mfm.training.config import TrainingConfig
from pymc_mfm.training.evaluation import (
    GATE_METRICS,
    MONITOR_METRIC,
    evaluate_rows,
    flatten_evaluation,
)
from pymc_mfm.training.observer import TrainingObserver
from pymc_mfm.training.stream import CorpusStream, PrefetchingStream

RUN_FORMAT_VERSION = 1
LOSS_TERMS = (
    "loss",
    "flow_loss",
    "context_flow_loss",
    "graph_conditional_flow_loss",
    "graph_conditional_context_flow_loss",
    "task_loss",
    "struct_loss",
)
SELECTION_READOUT = (
    "joint router head on the held-out query window; not a conditional-decoder quality claim"
)


class TrainingError(RuntimeError):
    """A run stopped for a reason the records already explain."""


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    run: dict[str, Any]
    model: MFM

    @property
    def status(self) -> str:
        return str(self.run["status"])

    @property
    def final_step(self) -> int:
        return int(self.run["final_step"])


def resolve_device(device: str | torch.device) -> torch.device:
    """``auto`` prefers CUDA, then MPS, then CPU; anything else must exist."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but CUDA is unavailable")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device 'mps' requested but MPS is unavailable")
    return resolved


def corpus_geometry(corpora: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Shared padded ``(K, M, J, T)`` of the corpora plus derived slot count."""
    if not corpora:
        raise ValueError("training requires at least one corpus")
    shapes = set()
    for corpus in corpora:
        k = int(np.shape(corpus["treatment_raw"])[2])
        m = int(np.shape(corpus["covariates"])[2])
        j = int(np.shape(corpus["latent_active_mask"])[1])
        t = int(np.shape(corpus["treatment_raw"])[1])
        shapes.add((k, m, j, t))
    if len(shapes) != 1:
        raise ValueError(
            f"corpora must share one padded geometry (K, M, J, T), got {sorted(shapes)}"
        )
    k, m, j, t = shapes.pop()
    return {
        "n_channels": k,
        "n_controls": m,
        "n_demand": j,
        "n_weeks": t,
        "n_graph_slots": EdgeSlotLayout(k, m, j).n_slots,
    }


def default_corpus_provenance(corpora: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Counts and geometry per corpus when the caller supplies no richer record."""
    records = []
    for index, corpus in enumerate(corpora):
        is_val = np.asarray(corpus["is_val"]).astype(bool)
        lengths = np.asarray(corpus["support_mask"]).sum(axis=1)
        records.append(
            {
                "index": index,
                "n_worlds": int(is_val.size),
                "n_train": int((~is_val).sum()),
                "n_holdout": int(is_val.sum()),
                "n_cells": int(np.unique(np.asarray(corpus["cell_id"])).size),
                "context_lengths": [int(v) for v in np.unique(lengths)],
            }
        )
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True))
    tmp.replace(path)


def _finite(value: float | None) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _select_checkpoint(
    flat: Mapping[str, float | None],
    undefined: Mapping[str, str],
    best_value: float | None,
    best_step: int | None,
    step: int,
) -> dict[str, Any]:
    """Promote only when every gate passes and the monitor improves.

    The held-out joint-query energy score and auxiliary ``cy`` log score must
    be finite, and padded-channel leakage must be zero. Missing, non-finite,
    or partially defined gate metrics block promotion with a recorded reason;
    the terminal checkpoint remains available.
    """

    def gate(key: str, threshold: float | None = None) -> dict[str, Any]:
        value = flat.get(key)
        if key in undefined:
            reason = f"metric_partially_defined:{undefined[key]}"
        elif not _finite(value):
            reason = "metric_missing_or_nonfinite"
        elif threshold is not None and value > threshold:
            reason = "threshold_not_met"
        else:
            reason = None
        return {"value": value, "passing": reason is None, "reason": reason}

    log_score_key, leakage_key = GATE_METRICS
    gates = {
        MONITOR_METRIC: gate(MONITOR_METRIC),
        log_score_key: gate(log_score_key),
        leakage_key: gate(leakage_key, threshold=0.0),
    }
    monitor = flat.get(MONITOR_METRIC)
    gates_passing = all(entry["passing"] for entry in gates.values())
    improved = gates_passing and (best_value is None or monitor < best_value)
    return {
        "monitor": MONITOR_METRIC,
        "direction": "min",
        "value": monitor,
        "gates": gates,
        "gates_passing": gates_passing,
        "promoted": improved,
        "best_step": step if improved else best_step,
        "best_value": monitor if improved else best_value,
        "readout": SELECTION_READOUT,
        "terminal_checkpoint": "checkpoint_last.pt",
    }


def train(
    corpora: Sequence[Mapping[str, Any]],
    *,
    out_dir: str | Path,
    model_config: ModelConfig | None = None,
    training: TrainingConfig | None = None,
    device: str | torch.device = "auto",
    corpus_provenance: Sequence[Mapping[str, Any]] | None = None,
    observers: Sequence[TrainingObserver] = (),
    run_name: str | None = None,
) -> TrainingResult:
    """Train the canonical model on loaded schema-4 corpora.

    ``out_dir`` must not already contain a run. It receives ``run.json``
    (rewritten at the end with the final status), ``objective_trace.json``
    (the first update's executed objective), ``metrics.jsonl`` (one record per
    ``log_every`` updates and at update 1), ``evaluations.jsonl`` (one record
    per evaluation event), ``checkpoint_last.pt``/``checkpoint_best.pt``/
    ``checkpoint_step_<N>.pt`` and, on completion, ``model.pt`` in the
    :func:`pymc_mfm.inference.checkpoint.save_model` format holding the
    terminal weights.

    ``model_config`` defaults to the A/B architecture sized to the corpus
    controls/latents; a supplied one must match the corpus geometry. A
    nonfinite loss or gradient norm at a logging step fails the run, and an
    interrupted or failed run is recorded as such rather than as finished.
    """
    training = TrainingConfig() if training is None else training
    out_dir = Path(out_dir)
    if (out_dir / "run.json").exists():
        raise FileExistsError(f"{out_dir} already holds a training run")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    geometry = corpus_geometry(corpora)
    if model_config is None:
        model_config = ModelConfig(n_controls=geometry["n_controls"], n_demand=geometry["n_demand"])
    if (
        model_config.n_controls != geometry["n_controls"]
        or model_config.n_demand != geometry["n_demand"]
    ):
        raise ValueError(
            "model_config n_controls/n_demand "
            f"({model_config.n_controls}, {model_config.n_demand}) must match the corpus "
            f"({geometry['n_controls']}, {geometry['n_demand']})"
        )
    provenance = (
        default_corpus_provenance(corpora)
        if corpus_provenance is None
        else [dict(record) for record in corpus_provenance]
    )
    if len(provenance) != len(corpora):
        raise ValueError("corpus_provenance must have one record per corpus")

    # Batches are encoded on the CPU (possibly on the prefetch thread) and
    # moved to the device by this thread only; accelerator work must not be
    # issued from two threads.
    stream = CorpusStream(
        corpora,
        batch_size=training.batch_size,
        seed=training.seed,
        split="train",
        device="cpu",
    )
    n_holdout = sum(int(np.asarray(corpus["is_val"]).sum()) for corpus in corpora)
    if n_holdout < 1:
        raise ValueError("training requires held-out worlds (is_val == 1) for evaluation")
    eval_holdout = min(training.eval_worlds, n_holdout)

    # Seed before construction: weight initialization consumes the global RNG.
    torch.manual_seed(training.seed)
    model = MFM(model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)

    started = datetime.now(UTC)
    run: dict[str, Any] = {
        "format_version": RUN_FORMAT_VERSION,
        "run_id": f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}",
        "run_name": run_name,
        "status": "running",
        "created_at": started.isoformat(),
        "finished_at": None,
        "final_step": 0,
        "error": None,
        "environment": {
            "pymc_mfm": package_version("pymc-mfm"),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(resolved_device),
        },
        "model_config": model_config.to_dict(),
        "training_config": training.to_dict(),
        "optimizer": {
            "type": "AdamW",
            "lr": training.lr,
            "betas": list(TrainingConfig.adam_betas),
            "eps": TrainingConfig.adam_eps,
            "weight_decay": training.weight_decay,
            "grad_clip_norm": training.grad_clip,
        },
        "objective_weights": {
            "struct_weight": model_config.struct_weight,
            "struct_cy_weight": model_config.struct_cy_weight,
            "graph_conditional_weight": model_config.graph_conditional_weight,
            "context_decomposition_weight": model_config.context_decomposition_weight,
            "context_graph_weight": model_config.context_graph_weight,
            "graph_conditional_context_weight": (
                model_config.graph_conditional_weight * model_config.context_decomposition_weight
            ),
        },
        "geometry": geometry,
        "parameter_count": parameter_count,
        "corpus": provenance,
        "extraction_source": {
            "repository": "https://github.com/pymc-labs/pymc-pfn",
            "revision": "52687cf25cca9be13094a397fabdf016367c0a52",
            "training_entrypoint": "scripts/train_mmm_decomposition.py",
        },
        "consumption_budget": {
            "steps": training.steps,
            "batch_size": training.batch_size,
            "train_rows": stream.n_rows,
            "holdout_rows": n_holdout,
            "rows_per_epoch": stream.rows_per_epoch,
            "dropped_rows_per_epoch": stream.n_rows - stream.rows_per_epoch,
            "epochs": training.steps * training.batch_size / stream.rows_per_epoch,
        },
        "evaluation_policy": {
            "every": training.eval_every,
            "holdout_worlds": eval_holdout,
            "seen_worlds": training.eval_worlds,
            "samples": training.eval_samples,
            "seed": training.eval_seed,
            "legs": ["holdout", "seen"],
        },
        "selection_policy": {
            "monitor": MONITOR_METRIC,
            "direction": "min",
            "gates": [MONITOR_METRIC, *GATE_METRICS],
            "readout": SELECTION_READOUT,
            "terminal_checkpoint": "checkpoint_last.pt",
        },
        "checkpoints": [],
        "best": None,
    }
    _write_json(out_dir / "run.json", run)

    metrics_path = out_dir / "metrics.jsonl"
    evaluations_path = out_dir / "evaluations.jsonl"
    source = None
    best_value: float | None = None
    best_step: int | None = None
    t0 = time.perf_counter()
    # Count only completed optimizer updates; the prefetch thread can build
    # later batches without those worlds having reached the optimizer.
    first_seen: dict[tuple[int, int], None] = {}
    consumed = {
        "updates": 0,
        "rows": 0,
        "distinct_rows": 0,
        "epoch": 0,
        "train_rows": stream.n_rows,
        "rows_per_epoch": stream.rows_per_epoch,
        "bucket_sizes": stream.state()["bucket_sizes"],
    }

    def note_consumed(step: int, provenance_record) -> None:
        for item in provenance_record.rows:
            first_seen.setdefault(item, None)
        consumed.update(
            updates=step,
            rows=step * training.batch_size,
            distinct_rows=len(first_seen),
            epoch=provenance_record.epoch,
        )

    def record_checkpoint(kind: str, step: int, name: str, evaluation, selection) -> None:
        nonlocal best_value, best_step
        path = out_dir / name
        digest = save_checkpoint(
            path,
            model=model,
            kind=kind,
            step=step,
            training_config=run["training_config"],
            run={
                key: run[key]
                for key in (
                    "run_id",
                    "run_name",
                    "created_at",
                    "environment",
                    "corpus",
                    "geometry",
                    "extraction_source",
                )
            },
            consumed=consumed,
            evaluation=evaluation,
            selection=selection,
        )
        entry = {"kind": kind, "step": step, "file": name, "sha256": digest}
        run["checkpoints"] = [
            existing for existing in run["checkpoints"] if existing["file"] != name
        ] + [entry]
        if kind == "best":
            best_value, best_step = selection["best_value"], selection["best_step"]
        for observer in observers:
            observer.on_checkpoint({**entry, "path": str(path), "selection": selection})

    def evaluate(step: int) -> dict[str, Any]:
        legs: dict[str, Any] = {}
        # Fixed seeds per leg keep worlds and draws comparable across events.
        holdout_rows = stream.holdout_rows(eval_holdout, seed=training.eval_seed)
        legs["holdout"] = evaluate_rows(
            model,
            corpora,
            holdout_rows,
            n_samples=training.eval_samples,
            seed=training.eval_seed + 1,
            device=resolved_device,
            leg="holdout",
        )
        seen_count = min(training.eval_worlds, len(first_seen))
        if seen_count:
            legs["seen"] = evaluate_rows(
                model,
                corpora,
                list(first_seen)[:seen_count],
                n_samples=training.eval_samples,
                seed=training.eval_seed,
                device=resolved_device,
                leg="seen",
            )
        flat: dict[str, float | None] = {}
        for record in legs.values():
            flat.update(flatten_evaluation(record))
        undefined = {
            f"{leg}/{key}": reason
            for leg, record in legs.items()
            for key, reason in record["undefined"].items()
        }
        selection = _select_checkpoint(flat, undefined, best_value, best_step, step)
        record = {
            "step": step,
            "epoch": consumed["epoch"],
            "consumed_rows": consumed["rows"],
            "n_samples": training.eval_samples,
            "legs": legs,
            "flat": flat,
            "selection": selection,
        }
        with evaluations_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        for observer in observers:
            observer.on_evaluation(record)
        # Sampling folds draws into the batch axis; hand its blocks back so the
        # allocator does not keep training steps slow for the rest of the run.
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
        elif resolved_device.type == "mps":
            torch.mps.empty_cache()
        return record

    status = "failed"
    error: str | None = None
    step = 0
    completed_updates = 0
    try:
        for observer in observers:
            observer.on_run_start(run)
        if training.prefetch_batches > 0:
            source = PrefetchingStream(stream, depth=training.prefetch_batches)
        with metrics_path.open("a") as metrics_handle:
            for step in range(1, training.steps + 1):
                model.train()
                wait_start = time.perf_counter()
                batch, provenance_record = source.next_batch() if source else stream.next_batch()
                if resolved_device.type != "cpu":
                    batch = {key: value.to(resolved_device) for key, value in batch.items()}
                batch_wait = time.perf_counter() - wait_start
                lr = training.learning_rate(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                step_start = time.perf_counter()
                if step == 1:
                    terms, trace = model.loss_terms(batch, return_trace=True)
                    (out_dir / "objective_trace.json").write_text(
                        json.dumps(objective_trace_to_jsonable(trace), indent=2, allow_nan=False)
                    )
                else:
                    terms = model.loss_terms(batch)
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, training.grad_clip)
                optimizer.step()
                completed_updates = step
                note_consumed(step, provenance_record)
                step_seconds = time.perf_counter() - step_start

                if step in training.milestone_steps:
                    record_checkpoint(
                        "milestone",
                        step,
                        f"checkpoint_step_{step}.pt",
                        None,
                        {"promoted": False, "rationale": "preregistered_exact_step_retention"},
                    )

                if step % training.log_every == 0 or step == 1:
                    values = torch.stack([terms[name].detach() for name in LOSS_TERMS])
                    values = torch.cat([values, grad_norm.detach().reshape(1)]).tolist()
                    *losses, grad_norm_value = values
                    elapsed = time.perf_counter() - t0
                    record = {
                        "step": step,
                        "epoch": provenance_record.epoch,
                        "consumed_rows": consumed["rows"],
                        "lr": lr,
                        **dict(zip(LOSS_TERMS, losses, strict=True)),
                        "grad_norm": grad_norm_value,
                        "grad_clipped": grad_norm_value > training.grad_clip,
                        "batch_context_length": provenance_record.context_length,
                        "elapsed_seconds": elapsed,
                        "step_seconds": step_seconds,
                        "batch_wait_seconds": batch_wait,
                        "worlds_per_second": step * training.batch_size / elapsed,
                    }
                    nonfinite = [
                        name
                        for name, value in zip((*LOSS_TERMS, "grad_norm"), values, strict=True)
                        if not math.isfinite(value)
                    ]
                    if nonfinite:
                        raise TrainingError(f"non-finite {', '.join(nonfinite)} at update {step}")
                    metrics_handle.write(json.dumps(record, allow_nan=False) + "\n")
                    metrics_handle.flush()
                    for observer in observers:
                        observer.on_step(record)

                if step % training.eval_every == 0 or step == training.steps:
                    evaluation = evaluate(step)
                    record_checkpoint(
                        "last", step, "checkpoint_last.pt", evaluation, evaluation["selection"]
                    )
                    if evaluation["selection"]["promoted"]:
                        record_checkpoint(
                            "best", step, "checkpoint_best.pt", evaluation, evaluation["selection"]
                        )
        save_model(model, out_dir / "model.pt")
        status = "finished"
    except KeyboardInterrupt:
        status, error = "interrupted", f"KeyboardInterrupt at update {step}"
        raise
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if source is not None:
            source.stop()
        run["status"] = status
        run["error"] = error
        run["final_step"] = completed_updates
        run["finished_at"] = datetime.now(UTC).isoformat()
        run["consumed"] = dict(consumed)
        run["best"] = None if best_step is None else {"step": best_step, "value": best_value}
        run["elapsed_seconds"] = time.perf_counter() - t0
        _write_json(out_dir / "run.json", run)
        try:
            for observer in observers:
                observer.on_run_end(run)
        except BaseException as exc:
            if status == "finished":
                run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            detail = f"on_run_end: {type(exc).__name__}: {exc}"
            run["error"] = f"{error}; {detail}" if error else detail
            _write_json(out_dir / "run.json", run)
            raise
    return TrainingResult(run_dir=out_dir, run=run, model=model.eval())
