"""Canonical model training with structured records and atomic checkpoints.

``train`` seeds model construction, optimizes every active objective with
AdamW, applies the configured learning-rate schedule and gradient clipping,
writes milestone and evaluation checkpoints, and exports terminal weights
after a completed training loop. It records every active loss, evaluation,
selection decision, and checkpoint identity locally.

Evaluation keeps joint-router and graph-conditional decoder readouts separate.
Checkpoint selection minimizes the held-out query energy score of
``TrainingConfig.selection_head`` (the conditional decoder by default) and
also gates that head's context energy score. Missing, non-finite, partially
defined, or out-of-bounds gate metrics block promotion, as does any held-out
metric of the head with non-finite worlds; all remain visible in the records.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pymc_mfm.inference.checkpoint import _export_environment, save_pretrained
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.model.model import MFM, objective_trace_to_jsonable
from pymc_mfm.training._provenance import (
    code_record,
    corpus_identity,
    environment_record,
    export_lineage,
)
from pymc_mfm.training.checkpoint import file_sha256, save_checkpoint
from pymc_mfm.training.config import TrainingConfig
from pymc_mfm.training.evaluation import (
    BOUNDS_SLACK,
    evaluate_rows,
    flatten_evaluation,
    metric_bounds,
    selection_metrics,
    within_bounds,
)
from pymc_mfm.training.observer import TrainingObserver
from pymc_mfm.training.stream import CorpusStream, PrefetchingStream
from pymc_mfm.training.summary import corpus_summary, corpus_support, parameter_counts
from pymc_mfm.training.tracking import (
    MlflowTrainingObserver,
    TrackingConfig,
    evidence_value,
    preserve_rng,
)

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
#: What the selection monitor measures, per ``TrainingConfig.selection_head``.
SELECTION_READOUT = {
    "conditional": "graph-conditional decoder (model_average) on the held-out query window",
    "joint": (
        "joint router head on the held-out query window; not a conditional-decoder quality claim"
    ),
}
#: Undefined reason under which the leakage gate passes: no world has a padded channel.
_VACUOUS_LEAKAGE = "no_inactive_channels"
#: Undefined-reason prefix of a metric that dropped non-finite worlds or slots.
_NONFINITE_REASON = "nonfinite_in_"


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

    def records(self):
        """This run's recorded history, a :class:`pymc_mfm.diagnostics.TrainingRun`.

        Read from ``run_dir``; the model is not used.
        """
        from pymc_mfm.diagnostics.accessors import TrainingRun

        return TrainingRun.load(self.run_dir)


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
    tmp.write_text(json.dumps(evidence_value(payload), indent=2, allow_nan=False, sort_keys=True))
    tmp.replace(path)


def _finite(value: float | None) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _bounds_record(key: str) -> list[float | None]:
    """JSON form of :func:`metric_bounds`; an infinite bound is ``None``."""
    return [bound if math.isfinite(bound) else None for bound in metric_bounds(key)]


def _select_checkpoint(
    flat: Mapping[str, float | None],
    undefined: Mapping[str, str],
    best_value: float | None,
    best_step: int | None,
    step: int,
    *,
    head: str,
) -> dict[str, Any]:
    """Promote only when every gate passes and the monitor improves.

    ``head`` is ``TrainingConfig.selection_head``. Its held-out query energy
    score (the minimized monitor), its context energy score and the auxiliary
    ``cy`` log score must be finite, fully defined and within
    :data:`~pymc_mfm.training.evaluation.METRIC_BOUNDS`; the head's padded-channel
    leakage must also be at most zero, and passes vacuously when no world has a
    padded channel. Missing, non-finite, partially defined, or out-of-bounds gate
    metrics block promotion with a recorded reason. A held-out metric of the head
    that dropped non-finite worlds, gated or not, also blocks promotion:
    ``nonfinite_keys`` maps each such key to its ``nonfinite_in_*`` reason, and
    ``gates_passing`` requires it to be empty. Other out-of-bounds metrics are
    listed under ``bounds_violations``. The terminal checkpoint remains available.
    """
    keys = selection_metrics(head)

    def gate(key: str, threshold: float | None = None, vacuous: str | None = None):
        value = flat.get(key)
        if vacuous is not None and undefined.get(key) == vacuous:
            reason = f"vacuous:{vacuous}"
        elif key in undefined:
            reason = f"metric_partially_defined:{undefined[key]}"
        elif not _finite(value):
            reason = "metric_missing_or_nonfinite"
        elif not within_bounds(key, value):
            reason = "metric_outside_bounds"
        elif threshold is not None and value > threshold:
            reason = "threshold_not_met"
        else:
            reason = None
        return {
            "value": value,
            "passing": reason is None or reason.startswith("vacuous:"),
            "reason": reason,
            "bounds": _bounds_record(key),
        }

    gates = {
        keys.monitor: gate(keys.monitor),
        keys.context_gate: gate(keys.context_gate),
        keys.structure_gate: gate(keys.structure_gate),
        keys.leakage_gate: gate(keys.leakage_gate, threshold=0.0, vacuous=_VACUOUS_LEAKAGE),
    }
    prefix = f"holdout/{head}/"
    nonfinite_keys = {
        key: reason
        for key, reason in undefined.items()
        if key.startswith(prefix) and reason.startswith(_NONFINITE_REASON)
    }
    monitor = flat.get(keys.monitor)
    gates_passing = not nonfinite_keys and all(entry["passing"] for entry in gates.values())
    improved = gates_passing and (best_value is None or monitor < best_value)
    return {
        "head": head,
        "monitor": keys.monitor,
        "direction": "min",
        "value": monitor,
        "gates": gates,
        "nonfinite_keys": nonfinite_keys,
        "gates_passing": gates_passing,
        "bounds_violations": [
            {"metric": key, "value": value, "bounds": _bounds_record(key)}
            for key, value in flat.items()
            if key not in gates and _finite(value) and not within_bounds(key, value)
        ],
        "promoted": improved,
        "best_step": step if improved else best_step,
        "best_value": monitor if improved else best_value,
        "readout": SELECTION_READOUT[head],
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
    tracking: TrackingConfig | None = None,
) -> TrainingResult:
    """Train with durable local evidence and default-enabled optional MLflow.

    Use ``TrackingConfig(enabled=False)`` for explicit tracking opt-out.
    Numerical configuration, optimizer updates and evaluation sampling do not
    depend on tracking. The lifecycle starts before device/model/stream setup,
    so failed initialization is also an identified, honestly finalized run.
    The terminal model is a safe directory bundle at ``out_dir / "model"``. An
    environment that export would refuse (``save_pretrained``: package versions
    outside this release's ``COMPATIBILITY``, or not release versions) is refused
    before anything is written.
    """
    training = TrainingConfig() if training is None else training
    tracking = TrackingConfig() if tracking is None else tracking
    out_dir = Path(out_dir)
    if (out_dir / "run.json").exists():
        raise FileExistsError(f"{out_dir} already holds a training run")
    _export_environment()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    run: dict[str, Any] = {
        "format_version": RUN_FORMAT_VERSION,
        "run_id": f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}",
        "run_name": run_name,
        "profile": tracking.profile,
        "status": "running",
        "created_at": started.isoformat(),
        "finished_at": None,
        "final_step": 0,
        "error": None,
        "training_config": training.to_dict(),
        "requested_device": str(device),
        "requested_model_config": None if model_config is None else model_config.to_dict(),
        "consumed": {"updates": 0, "rows": 0, "distinct_rows": 0, "epoch": 0},
        "checkpoints": [],
        "best": None,
    }
    tracker = MlflowTrainingObserver(out_dir, tracking)
    run["tracking"] = tracker.summary
    _write_json(out_dir / "run.json", run)
    try:
        tracker.on_run_start(run)
        run["tracking"] = tracker.summary
        _write_json(out_dir / "run.json", run)
        return _train_run(
            corpora,
            out_dir=out_dir,
            model_config=model_config,
            training=training,
            device=device,
            corpus_provenance=corpus_provenance,
            observers=observers,
            run=run,
            tracker=tracker,
        )
    except BaseException as exc:
        if run["status"] == "running":
            run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            run["error"] = f"{type(exc).__name__} during initialization"
            run["error_type"] = type(exc).__name__
            run["finished_at"] = datetime.now(UTC).isoformat()
        raise
    finally:
        run["tracking"] = tracker.summary
        run["tracking"]["lifecycle_status"] = run["status"]
        if run["tracking"]["status"] not in ("disabled", "degraded"):
            run["tracking"]["status"] = run["status"]
        _write_json(out_dir / "run.json", run)
        try:
            tracker.on_run_end(run)
        except BaseException as exc:
            run["status"] = tracker.lifecycle_status
            run["error"] = f"{type(exc).__name__} during tracking finalization"
            run["error_type"] = type(exc).__name__
            run["finished_at"] = datetime.now(UTC).isoformat()
            raise
        finally:
            run["tracking"] = tracker.summary
            _write_json(out_dir / "run.json", run)


def _train_run(
    corpora: Sequence[Mapping[str, Any]],
    *,
    out_dir: str | Path,
    model_config: ModelConfig | None = None,
    training: TrainingConfig | None = None,
    device: str | torch.device = "auto",
    corpus_provenance: Sequence[Mapping[str, Any]] | None = None,
    observers: Sequence[TrainingObserver] = (),
    run: dict[str, Any],
    tracker: MlflowTrainingObserver,
) -> TrainingResult:
    """Train the canonical model on loaded schema-4 corpora.

    ``out_dir`` must not already contain a run. It receives ``run.json``
    (rewritten at the end with the final status), ``objective_trace.json``
    (the first update's executed objective), ``metrics.jsonl`` (one record per
    ``log_every`` updates and at update 1), ``evaluations.jsonl`` (one record
    per evaluation event), ``evaluation_tables/step_<9d>.json`` (each event's
    reliability and threshold tables, written before its ``evaluations.jsonl``
    line), ``checkpoint_last.pt``/``checkpoint_best.pt``/
    ``checkpoint_step_<N>.pt`` and, on completion, ``model`` in the
    :func:`pymc_mfm.inference.checkpoint.save_pretrained` directory format
    holding the terminal weights.

    ``model_config`` defaults to the A/B architecture sized to the corpus
    controls/latents; a supplied one must match the corpus geometry. A
    nonfinite loss or gradient norm at a logging step fails the run, and an
    interrupted or failed run is recorded as such rather than as finished.
    """
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
    for corpus, record in zip(corpora, provenance, strict=True):
        record["actual_inputs"] = corpus_identity(corpus)

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
    loss_weights = training.loss_weights
    selection_keys = selection_metrics(training.selection_head)

    run.update(
        {
            "environment": environment_record(str(resolved_device)),
            "code": code_record(),
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
                **loss_weights.to_dict(),
                "graph_conditional_context_weight": (
                    loss_weights.graph_conditional_weight
                    * loss_weights.context_decomposition_weight
                ),
            },
            "geometry": geometry,
            "parameter_count": parameter_count,
            "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_counts": parameter_counts(model),
            "corpus": evidence_value(provenance),
            "corpus_summary": corpus_summary(corpora),
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
                "diagnostics": {
                    "enabled": tracker.config.enabled and tracker.config.diagnostics,
                    "cadence": "each evaluation event, all existing conditional world/window draws",
                    "resampling": False,
                    "reference_fits": "not_scheduled",
                    "stress_comparisons": "not_scheduled",
                    "posterior_sbc": "not_scheduled",
                },
            },
            "selection_policy": {
                "head": training.selection_head,
                "monitor": selection_keys.monitor,
                "direction": "min",
                "gates": list(selection_keys),
                "bounds": {key: _bounds_record(key) for key in selection_keys},
                "bounds_slack": BOUNDS_SLACK,
                "leakage_threshold": 0.0,
                "vacuous": {selection_keys.leakage_gate: _VACUOUS_LEAKAGE},
                "nonfinite_block": {
                    "prefix": f"holdout/{training.selection_head}/",
                    "reason_prefix": _NONFINITE_REASON,
                },
                "readout": SELECTION_READOUT[training.selection_head],
                "terminal_checkpoint": "checkpoint_last.pt",
            },
            "checkpoints": [],
            "diagnostics": [],
            "best": None,
        }
    )
    _write_json(out_dir / "run.json", run)
    _write_json(out_dir / "corpus_support.json", corpus_support(corpora))
    tracker.on_run_config(run)
    event_observers = (tracker, *observers)

    metrics_path = out_dir / "metrics.jsonl"
    evaluations_path = out_dir / "evaluations.jsonl"
    tables_dir = out_dir / "evaluation_tables"
    source = None
    best_value: float | None = None
    best_step: int | None = None
    t0 = time.perf_counter()
    # Count only completed optimizer updates; the prefetch thread can build
    # later batches without those worlds having reached the optimizer.
    first_seen: dict[tuple[int, int], None] = {}
    consumption_digest = hashlib.sha256()
    diagnostic_inputs: list[tuple[Any, Any, Any, Any]] = []
    training_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
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
        consumption_digest.update(
            json.dumps(provenance_record.rows, separators=(",", ":")).encode()
        )
        consumed.update(
            updates=step,
            rows=step * training.batch_size,
            distinct_rows=len(first_seen),
            epoch=provenance_record.epoch,
            row_sequence_sha256=consumption_digest.hexdigest(),
        )

    def record_checkpoint(kind: str, step: int, name: str, evaluation, selection) -> dict[str, Any]:
        nonlocal best_value, best_step
        path = out_dir / name
        run["tracking"] = tracker.summary
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
                    "code",
                    "profile",
                    "tracking",
                )
            },
            consumed=consumed,
            evaluation=evaluation,
            selection=selection,
        )
        entry = {"kind": kind, "step": step, "file": name, "sha256": digest}
        if evaluation is not None:
            entry["evaluation_sha256"] = hashlib.sha256(
                json.dumps(evaluation, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        run["checkpoints"] = [
            existing for existing in run["checkpoints"] if existing["file"] != name
        ] + [entry]
        if kind == "best":
            best_value, best_step = selection["best_value"], selection["best_step"]
        for observer in event_observers:
            observer.on_checkpoint({**entry, "path": str(path), "selection": selection})
        return entry

    def record_diagnostics(step: int, checkpoint: Mapping[str, Any]) -> None:
        for index, (result, truth, graph_truth, observed) in enumerate(diagnostic_inputs):
            provenance = {
                **result.provenance,
                "checkpoint_sha256": checkpoint["sha256"],
                "source_step": step,
                "training_local_run_id": run["run_id"],
                "tracking_run_id": tracker.run_id,
                "profile": tracker.config.profile,
                "evaluation_data_origin": "training_corpus",
                "synthetic_training_evaluation": all(
                    item.get("data_origin") == "synthetic" for item in run["corpus"]
                ),
                "world_id": result.provenance.get("world_id", index),
            }
            report_dir = (
                out_dir
                / "diagnostics"
                / f"step_{step:09d}"
                / checkpoint["sha256"]
                / f"world_{index:04d}_{result.window}_{result.mode}"
            )
            try:
                from pymc_mfm.diagnostics import build_report

                with preserve_rng():
                    manifest = build_report(
                        replace(result, provenance=provenance),
                        report_dir,
                        truth=truth,
                        graph_truth=graph_truth,
                        observed=observed,
                        training_records=training_records,
                        evaluation_records=evaluation_records,
                        strict=False,
                    )
                run["diagnostics"].append(
                    {
                        "name": f"step_{step}_world_{index}_{result.window}_{result.mode}",
                        "sha256": file_sha256(report_dir / "manifest.json"),
                        "checkpoint_sha256": checkpoint["sha256"],
                        "status": manifest.get("status", "recorded"),
                        "recipe": "saved_conditional_evaluation_report",
                    }
                )
                event = {
                    "step": step,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "path": str(report_dir),
                    "manifest": manifest,
                }
            except Exception as exc:
                event = {
                    "step": step,
                    "checkpoint_sha256": checkpoint["sha256"],
                    "path": str(report_dir),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "provenance": provenance,
                }
                tracker.on_diagnostic_failure(event)
                run["diagnostics"].append(
                    {
                        "name": f"step_{step}_world_{index}_{result.window}_{result.mode}",
                        "checkpoint_sha256": checkpoint["sha256"],
                        "status": "failed",
                        "recipe": "saved_conditional_evaluation_report",
                    }
                )
            with (out_dir / "diagnostics.jsonl").open("a") as handle:
                handle.write(json.dumps(evidence_value(event), allow_nan=False) + "\n")
            if "manifest" in event:
                for observer in event_observers:
                    observer.on_diagnostic(event["manifest"], report_dir)
        diagnostic_inputs.clear()

    def evaluate(step: int) -> dict[str, Any]:
        legs: dict[str, Any] = {}
        diagnostic_inputs.clear()
        result_sink = None
        if tracker.config.enabled and tracker.config.diagnostics:

            def result_sink(result, truth, graph_truth, observed):
                diagnostic_inputs.append((result, truth, graph_truth, observed))

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
            result_sink=result_sink,
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
                result_sink=result_sink,
            )
        flat: dict[str, float | None] = {}
        for record in legs.values():
            flat.update(flatten_evaluation(record))
        undefined = {
            f"{leg}/{key}": reason
            for leg, record in legs.items()
            for key, reason in record["undefined"].items()
        }
        for leg, values in legs.items():
            values["rows"] = holdout_rows if leg == "holdout" else list(first_seen)[:seen_count]
        selection = _select_checkpoint(
            flat, undefined, best_value, best_step, step, head=training.selection_head
        )
        record = {
            "step": step,
            "epoch": consumed["epoch"],
            "consumed_rows": consumed["rows"],
            "n_samples": training.eval_samples,
            "legs": legs,
            "flat": flat,
            "selection": selection,
        }
        # The tables file lands before its evaluations.jsonl line, so every
        # recorded evaluation has its reliability and threshold tables.
        tables_dir.mkdir(exist_ok=True)
        _write_json(
            tables_dir / f"step_{step:09d}.json",
            {"step": step, "legs": {leg: values["tables"] for leg, values in legs.items()}},
        )
        with evaluations_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        evaluation_records.append(record)
        for observer in event_observers:
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
            observer.on_run_config(run)
        if training.prefetch_batches > 0:
            source = PrefetchingStream(stream, depth=training.prefetch_batches)
        with metrics_path.open("a") as metrics_handle:
            for step in range(1, training.steps + 1):
                model.train()
                wait_start = time.perf_counter()
                batch, provenance_record = source.next_batch() if source else stream.next_batch()
                if resolved_device.type != "cpu":
                    batch = {
                        key: value if key == "sales_scale" else value.to(resolved_device)
                        for key, value in batch.items()
                    }
                batch_wait = time.perf_counter() - wait_start
                lr = training.learning_rate(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                step_start = time.perf_counter()
                if step == 1:
                    terms, trace = model.loss_terms(batch, weights=loss_weights, return_trace=True)
                    _write_json(
                        out_dir / "objective_trace.json", objective_trace_to_jsonable(trace)
                    )
                else:
                    terms = model.loss_terms(batch, weights=loss_weights)
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
                        "row_sequence_sha256": consumed["row_sequence_sha256"],
                        "batch_rows": provenance_record.rows,
                        "lr": lr,
                        **dict(zip(LOSS_TERMS, losses, strict=True)),
                        "objective_weights": run["objective_weights"],
                        # Weights are strictly positive, so the raw term is always recoverable.
                        "struct_loss_raw": (
                            losses[LOSS_TERMS.index("struct_loss")] / loss_weights.struct_weight
                        ),
                        "struct_loss_raw_status": "reconstructed_from_weighted_term",
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
                        record["nonfinite"] = {name: repr(record[name]) for name in nonfinite}
                        for name in nonfinite:
                            record[name] = None
                    metrics_handle.write(json.dumps(evidence_value(record), allow_nan=False) + "\n")
                    metrics_handle.flush()
                    training_records.append(record)
                    for observer in event_observers:
                        observer.on_step(record)
                    if nonfinite:
                        raise TrainingError(f"non-finite {', '.join(nonfinite)} at update {step}")

                if step % training.eval_every == 0 or step == training.steps:
                    evaluation = evaluate(step)
                    checkpoint = record_checkpoint(
                        "last", step, "checkpoint_last.pt", evaluation, evaluation["selection"]
                    )
                    if evaluation["selection"]["promoted"]:
                        record_checkpoint(
                            "best", step, "checkpoint_best.pt", evaluation, evaluation["selection"]
                        )
                    record_diagnostics(step, checkpoint)
        terminal = next(entry for entry in run["checkpoints"] if entry["kind"] == "last")
        export_provenance = {
            "source_checkpoint_sha256": terminal["sha256"],
            "source_step": completed_updates,
            "tracking_local_run_id": run["run_id"],
            "tracking_run_id": tracker.run_id,
            "tracking_experiment_id": tracker.experiment_id,
            "training_seed": training.seed,
            "source_revision": run["code"]["revision"],
            **export_lineage(run, terminal["sha256"]),
        }
        save_pretrained(
            model,
            out_dir / "model",
            provenance={
                key: value for key, value in export_provenance.items() if value is not None
            },
            profile=tracker.config.profile,
        )
        run["export"] = {
            "path": "model",
            "source_checkpoint_sha256": terminal["sha256"],
            "source_step": completed_updates,
            "files": ["config.json", "model.safetensors", "manifest.json"],
        }
        status = "finished"
    except KeyboardInterrupt:
        status, error = "interrupted", f"KeyboardInterrupt at update {step}"
        run["error_type"] = "KeyboardInterrupt"
        raise
    except BaseException as exc:
        error = f"{type(exc).__name__} at update {step}"
        run["error_type"] = type(exc).__name__
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
            detail = f"on_run_end: {type(exc).__name__}"
            run["error_type"] = type(exc).__name__
            run["error"] = f"{error}; {detail}" if error else detail
            _write_json(out_dir / "run.json", run)
            raise
    return TrainingResult(run_dir=out_dir, run=run, model=model.eval())
