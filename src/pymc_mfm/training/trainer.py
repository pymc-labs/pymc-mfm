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
defined, or out-of-bounds gate metrics block promotion, as does any legacy
held-out metric of the head with non-finite worlds. Additive simultaneous
coverage never changes promotion; its unavailable reasons remain recorded.
"""

from __future__ import annotations

import json
import math
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
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
from pymc_mfm.training.archive import (
    _corpus_records,
    _evaluation_block,
    _model_identity,
    _world_records,
)
from pymc_mfm.training.checkpoint import file_sha256, save_checkpoint
from pymc_mfm.training.config import TrainingConfig
from pymc_mfm.training.evaluation import (
    BOUNDS_SLACK,
    DECOMPOSITION_HEADS,
    evaluate_rows,
    flatten_evaluation,
    metric_bounds,
    selection_metrics,
    within_bounds,
)
from pymc_mfm.training.observer import TrainingObserver
from pymc_mfm.training.precision import (
    autocast_context,
    build_loss_fn,
    check_acceleration_allowed,
    compile_counts,
    compile_record,
)
from pymc_mfm.training.resume import (
    EMPTY_ROW_SEQUENCE,
    ROW_SEQUENCE_SCHEME,
    StartPoint,
    build_resume_block,
    chain_row_sequence,
    checkpoint_entry,
    continue_run_record,
    copy_best_checkpoint,
    copy_run_history,
    evaluation_sha256,
    load_start_point,
    provenance_of_start,
    read_jsonl,
    restore_rng_state,
    tracking_tags,
    truncate_run_files,
)
from pymc_mfm.training.stream import CorpusStream, PrefetchingStream, SeenRows
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
#: Evaluation events ``train(holdout_archives=...)`` can archive: ``"best"``, every event
#: that promotes ``checkpoint_best.pt``, and ``"terminal"``, the final event.
HOLDOUT_ARCHIVE_KINDS = ("best", "terminal")
#: Undefined reason under which the leakage gate passes: no world has a padded channel.
_VACUOUS_LEAKAGE = "no_inactive_channels"
#: Undefined-reason prefix of a metric that dropped non-finite worlds or slots.
_NONFINITE_REASON = "nonfinite_in_"
# Additive joint-coverage diagnostics must not extend the legacy selection guard.
_NONFINITE_EXCLUDED_SUFFIXES = (
    "/timeavg_contribution_simcov80_live",
    "/timeavg_control_contribution_simcov80_live",
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


def _step_clock(device: torch.device, *, measure: bool) -> float:
    if measure and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


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


def _resolve_inputs(
    corpora: Sequence[Mapping[str, Any]],
    model_config: ModelConfig | None,
    corpus_provenance: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, int], ModelConfig, list[dict[str, Any]]]:
    """Corpus geometry, the model configuration and per-corpus provenance with identities."""
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
    return geometry, model_config, provenance


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
    metrics block promotion with a recorded reason. A legacy held-out metric of
    the head that dropped non-finite worlds, gated or not, also blocks promotion:
    ``nonfinite_keys`` maps each such key to its ``nonfinite_in_*`` reason, and
    ``gates_passing`` requires it to be empty. The additive simultaneous-coverage
    metrics are excluded from this guard, preserving checkpoint selection.
    Other out-of-bounds metrics are listed under ``bounds_violations``. The
    terminal checkpoint remains available.
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
        if key.startswith(prefix)
        and reason.startswith(_NONFINITE_REASON)
        and not key.endswith(_NONFINITE_EXCLUDED_SUFFIXES)
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


def _holdout_archive_kinds(value: Sequence[str] | None) -> tuple[str, ...]:
    """``holdout_archives`` as distinct :data:`HOLDOUT_ARCHIVE_KINDS`, in canonical order.

    ``None`` and an empty sequence disable archives. A string or another value that is
    not an iterable of names (``True``), a name outside :data:`HOLDOUT_ARCHIVE_KINDS` or
    a repeated name raises ``ValueError``.
    """
    if value is None:
        return ()
    names = None if isinstance(value, str) or not isinstance(value, Iterable) else list(value)
    if (
        names is None
        or any(name not in HOLDOUT_ARCHIVE_KINDS for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            f"holdout_archives must be distinct names from {HOLDOUT_ARCHIVE_KINDS}, got {value!r}"
        )
    return tuple(kind for kind in HOLDOUT_ARCHIVE_KINDS if kind in names)


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
    holdout_archives: Sequence[str] | None = (),
    resume_from: str | Path | None = None,
    init_from: str | Path | None = None,
    measure_unadopted_acceleration: bool = False,
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

    ``resume_from`` continues a run at ``step + 1`` of its original schedule from a
    ``last`` or ``milestone`` format 3 checkpoint, restoring optimizer, RNG, batch
    stream and consumption so that later records equal an uninterrupted run's. The
    corpora, model configuration and ``training_config`` must equal the checkpoint's
    (``prefetch_batches`` may differ); anything else, a format 2 file, a ``best``
    checkpoint or a dimensional checkpoint raises ``ValueError`` before any file is
    written. With ``out_dir`` the directory of the checkpoint, only its
    ``checkpoint_last.pt``, the run continues in place with the same ``run_id``.
    It reopens the MLflow run when a run id exists. Records after the checkpoint
    are dropped and ``run.json`` gains a ``resumes`` entry. In a new ``out_dir``
    the run gets a new ``run_id``, a ``resumed_from``
    record, and a copy of the source's metrics and evaluations up to the checkpoint.
    ``init_from`` loads weights only from a format 2 or 3 checkpoint into a new run
    at step 1 with a fresh optimizer; ``run.json`` records ``initialized_from``.
    The two are exclusive. A resume cannot rebuild the diagnostics and archives of
    the interrupted step (the checkpoint precedes them).

    ``holdout_archives`` selects evaluation events whose held-out leg is kept as
    a :class:`pymc_mfm.diagnostics.EvaluationArchive` at
    ``out_dir / "archives" / "step_<N:09d>"``: ``"best"``, every event that
    promotes ``checkpoint_best.pt``, and ``"terminal"``, the final event. An
    archive holds that leg's existing conditional captures, so no inference is
    added and draws, scores, selection, weights and random streams are
    unchanged; the evaluation records of events that can be archived gain the
    holdout leg's ``result_capture``. An archive written at the final step names
    that step's ``checkpoint_last.pt``, the exported bundle's source checkpoint,
    and any other names its step's ``checkpoint_best.pt``. Each archive gets a
    ``run.json`` ``diagnostics`` entry (recipe ``holdout_evaluation_archive``),
    which the bundle references for the final step. A failed archive is
    recorded as failed and training continues; a configuration that cannot be
    archived fails the run before training, and an invalid value raises
    ``ValueError`` before anything is written. Unset (``()`` or ``None``),
    nothing in the run changes.

    ``precision="bf16"`` and ``compile=True`` are measured and not adopted
    (:mod:`pymc_mfm.training.precision`); they raise ``ValueError`` before anything
    is written unless ``measure_unadopted_acceleration`` is set. That flag also
    synchronizes CUDA before and after each timed step.
    """
    training = TrainingConfig() if training is None else training
    check_acceleration_allowed(training, measure=measure_unadopted_acceleration)
    tracking = TrackingConfig() if tracking is None else tracking
    archives = _holdout_archive_kinds(holdout_archives)
    out_dir = Path(out_dir)
    if resume_from is not None and init_from is not None:
        raise ValueError("resume_from and init_from are exclusive")
    start: StartPoint | None = None
    inputs = None
    if resume_from is not None or init_from is not None:
        # Validate checkpoint compatibility before creating or truncating run files.
        inputs = _resolve_inputs(corpora, model_config, corpus_provenance)
        start = load_start_point(
            resume_from if resume_from is not None else init_from,
            mode="resume" if resume_from is not None else "init",
            out_dir=out_dir,
            device=resolve_device(device),
            model_config=inputs[1],
            training=training,
            corpus_records=inputs[2],
        )
    in_place = start is not None and start.mode == "resume" and start.same_directory
    if (out_dir / "run.json").exists() and not in_place:
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
    tracker_tags = tracking_tags(start) if start is not None and not in_place else {}
    if start is not None:
        if in_place:
            truncate_run_files(out_dir, start.step, final_step=training.steps)
            run = continue_run_record(
                json.loads((out_dir / "run.json").read_text()), start, out_dir
            )
        else:
            record = "resumed_from" if start.mode == "resume" else "initialized_from"
            run[record] = provenance_of_start(start)
            if start.mode == "resume":
                run[record]["history_copied"] = (start.source_dir / "metrics.jsonl").is_file()
                copy_run_history(
                    start.source_dir,
                    out_dir,
                    step=start.step,
                    evaluation_step=start.step - int(_evaluation_pending(start, training)),
                )
    tracker = MlflowTrainingObserver(
        out_dir,
        tracking,
        continue_run_id=start.source_tracking_run_id if in_place else None,
        continue_local=in_place,
        tags=tracker_tags,
    )
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
            holdout_archives=archives,
            start=start,
            inputs=inputs,
            measure_unadopted_acceleration=measure_unadopted_acceleration,
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


def _evaluation_pending(start: StartPoint, training: TrainingConfig) -> bool:
    """A milestone checkpoint precedes its own step's evaluation, which a resume must run."""
    step = start.step
    return start.kind == "milestone" and (step % training.eval_every == 0 or step == training.steps)


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
    holdout_archives: tuple[str, ...] = (),
    start: StartPoint | None = None,
    inputs: tuple[dict[str, int], ModelConfig, list[dict[str, Any]]] | None = None,
    measure_unadopted_acceleration: bool,
) -> TrainingResult:
    """Train the canonical model on loaded schema-4 corpora.

    ``out_dir`` must not already contain a run, except for a same-directory resume
    (``start``, see :func:`train`). It receives ``run.json``
    (rewritten at the end with the final status), ``objective_trace.json``
    (the first update's executed objective), ``metrics.jsonl`` (one record per
    ``log_every`` updates and at update 1), ``evaluations.jsonl`` (one record
    per evaluation event), ``evaluation_tables/step_<9d>.json`` (each event's
    reliability, CORP and threshold tables and its per-effect ``strata``
    tables, written before its ``evaluations.jsonl`` line; the strata tables
    are only there), ``checkpoint_last.pt``/``checkpoint_best.pt``/
    ``checkpoint_step_<N>.pt``, ``archives/step_<N:09d>`` (an evaluation
    archive of each event that ``holdout_archives`` selects, see :func:`train`)
    and, on completion, ``model`` in the
    :func:`pymc_mfm.inference.checkpoint.save_pretrained` directory format
    holding the terminal weights.

    ``model_config`` defaults to the A/B architecture sized to the corpus
    controls/latents; a supplied one must match the corpus geometry. A
    nonfinite loss or gradient norm at a logging step fails the run, and an
    interrupted or failed run is recorded as such rather than as finished.
    ``holdout_archives`` holds canonical kinds (:func:`_holdout_archive_kinds`);
    their corpora and evaluation identity is derived before weight
    initialization and draws nothing, so a configuration that cannot be
    archived fails the run before training.
    """
    resolved_device = resolve_device(device)

    geometry, model_config, provenance = (
        _resolve_inputs(corpora, model_config, corpus_provenance) if inputs is None else inputs
    )
    resuming = start is not None and start.mode == "resume"

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
    archive_static: dict[str, Any] | None = None
    if holdout_archives:
        # The held-out worlds are the same at every event; deriving their
        # identity encodes them but draws nothing, so initialization is unchanged.
        archive_static = {
            "corpora": _corpus_records(
                corpora, provenance, identities=[record["actual_inputs"] for record in provenance]
            ),
            "evaluation": _evaluation_block(
                _world_records(corpora, stream.holdout_rows(eval_holdout, seed=training.eval_seed)),
                leg="holdout",
                heads=DECOMPOSITION_HEADS,
                seed=training.eval_seed + 1,
                n_samples=training.eval_samples,
                device=resolved_device,
            ),
        }

    # Seed before construction: weight initialization consumes the global RNG.
    torch.manual_seed(training.seed)
    model = MFM(model_config).to(resolved_device)
    if start is not None:
        model.load_state_dict(start.model.state_dict(), strict=True)
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
                    "excluded_suffixes": list(_NONFINITE_EXCLUDED_SUFFIXES),
                },
                "readout": SELECTION_READOUT[training.selection_head],
                "terminal_checkpoint": "checkpoint_last.pt",
            },
            "checkpoints": run["checkpoints"] if resuming else [],
            "diagnostics": run.get("diagnostics", []) if resuming else [],
            "best": None,
        }
    )
    if holdout_archives:
        run["evaluation_policy"]["holdout_archives"] = {
            "checkpoints": list(holdout_archives),
            "leg": "holdout",
            "path": "archives/step_<N:09d>",
            "resampling": False,
        }
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
    seen = SeenRows(
        [int(np.asarray(corpus["is_val"]).size) for corpus in corpora],
        prefix_limit=training.eval_worlds,
    )
    row_sequence = EMPTY_ROW_SEQUENCE
    diagnostic_inputs: list[tuple[Any, Any, Any, Any]] = []
    archive_inputs: list[tuple[Any, Any, Any, Any]] = []
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
        "row_sequence_scheme": ROW_SEQUENCE_SCHEME,
        "row_sequence_sha256": EMPTY_ROW_SEQUENCE,
    }
    first_step = 1
    if resuming:
        # Restore before the prefetch thread starts, so no batch is built from the
        # initial stream position. The RNG goes last: nothing above may draw from it.
        state = start.payload["resume"]
        optimizer.load_state_dict(state["optimizer"])
        stream.restore(state["stream"])
        seen.restore(state["consumption"]["seen_rows"])
        row_sequence = state["consumption"]["row_sequence_sha256"]
        consumed.update(start.payload["consumed"])
        if (
            consumed["row_sequence_sha256"] != row_sequence
            or consumed["distinct_rows"] != len(seen)
            or consumed["updates"] != start.step
        ):
            raise ValueError("resume_from: the checkpoint's consumption records disagree")
        best_value, best_step = state["best_value"], state["best_step"]
        training_records.extend(read_jsonl(metrics_path))
        evaluation_records.extend(read_jsonl(evaluations_path))
        restore_rng_state(state)
        first_step = start.step + 1
        run["consumed"] = dict(consumed)

    def note_consumed(step: int, provenance_record) -> None:
        nonlocal row_sequence
        seen.add(provenance_record.rows)
        row_sequence = chain_row_sequence(row_sequence, provenance_record.rows)
        consumed.update(
            updates=step,
            rows=step * training.batch_size,
            distinct_rows=len(seen),
            epoch=provenance_record.epoch,
            row_sequence_sha256=row_sequence,
        )

    def record_checkpoint(kind: str, step: int, name: str, evaluation, selection) -> dict[str, Any]:
        nonlocal best_value, best_step
        path = out_dir / name
        run["tracking"] = tracker.summary
        resume = None
        if kind != "best":
            pending = (
                selection
                if kind == "last"
                else {
                    "best_value": best_value,
                    "best_step": best_step,
                }
            )
            resume = build_resume_block(
                device=resolved_device,
                optimizer=optimizer,
                step=step,
                stream_state=source.consumed_state if source else stream.resume_state(),
                row_sequence=row_sequence,
                seen=seen,
                best_value=pending["best_value"],
                best_step=pending["best_step"],
            )
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
            resume=resume,
        )
        entry = {"kind": kind, "step": step, "file": name, "sha256": digest}
        if evaluation is not None:
            entry["evaluation_sha256"] = evaluation_sha256(evaluation)
        run["checkpoints"] = [
            existing for existing in run["checkpoints"] if existing["file"] != name
        ] + [entry]
        if kind == "best":
            best_value, best_step = selection["best_value"], selection["best_step"]
        for observer in event_observers:
            observer.on_checkpoint({**entry, "path": str(path), "selection": selection})
        return entry

    def record_archive(step: int, last: Mapping[str, Any], best: Mapping[str, Any] | None) -> None:
        """Archive this event's held-out captures when ``holdout_archives`` selects it.

        The archive names ``last``, this event's ``checkpoint_last.pt``, when it
        includes ``"terminal"`` or is written at the final step, and ``best``
        otherwise. A failure is recorded by its type and training continues.
        """
        if not holdout_archives:
            return
        captured = list(archive_inputs)
        archive_inputs.clear()
        kinds = [
            kind
            for kind in holdout_archives
            if (kind == "best" and best is not None)
            or (kind == "terminal" and step == training.steps)
        ]
        if not kinds:
            return
        relative = f"archives/step_{step:09d}"
        bound = last if ("terminal" in kinds or step == training.steps) else best
        entry = {
            "name": f"step_{step}_holdout_archive",
            "checkpoint_sha256": last["sha256"],
            "status": "failed",
            "recipe": "holdout_evaluation_archive",
            "path": relative,
            "kinds": kinds,
        }
        if best is not None:
            entry["best_checkpoint_sha256"] = best["sha256"]
        try:
            from pymc_mfm.diagnostics.archive import EvaluationArchive

            with preserve_rng():
                identity = {
                    "model": _model_identity(
                        model,
                        checkpoint_sha256=bound["sha256"],
                        source_step=step,
                        profile=tracker.config.profile,
                    ),
                    **archive_static,
                }
                # absolute(): create expands a leading "~", which the run directory keeps.
                target = out_dir.absolute() / relative
                with EvaluationArchive.create(target, identity) as writer:
                    for item in captured:
                        writer.add(*item)
            # Hashed where create published it, not at a re-derived spelling.
            entry.update(
                status="recorded",
                sha256=file_sha256(writer.path / "manifest.json"),
                archive_sha256=writer.archive_sha256,
                state_sha256=identity["model"]["state_sha256"],
            )
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
        run["diagnostics"].append(entry)

    def record_diagnostics(step: int, checkpoint: Mapping[str, Any]) -> None:
        if not diagnostic_inputs:
            return
        from pymc_mfm.diagnostics.coverage import evaluation_coverage_intervals

        coverage_snapshots = []
        for record in evaluation_records:
            path = tables_dir / f"step_{int(record['step']):09d}.json"
            tables = json.loads(path.read_text()) if path.is_file() else None
            coverage_snapshots.append(
                {
                    "step": record["step"],
                    "coverage_intervals": evaluation_coverage_intervals(record, tables),
                }
            )
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
                        coverage_intervals=coverage_snapshots,
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
        archive_inputs.clear()
        result_sink = None
        if tracker.config.enabled and tracker.config.diagnostics:

            def result_sink(result, truth, graph_truth, observed):
                diagnostic_inputs.append((result, truth, graph_truth, observed))

        # Only events that can be archived collect held-out captures: a sink adds
        # result_capture to the holdout leg's record but changes no draw or score.
        holdout_sink = result_sink
        if holdout_archives and ("best" in holdout_archives or step == training.steps):

            def holdout_sink(result, truth, graph_truth, observed, diagnose=result_sink):
                archive_inputs.append((result, truth, graph_truth, observed))
                if diagnose is not None:
                    diagnose(result, truth, graph_truth, observed)

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
            result_sink=holdout_sink,
            per_world=True,
        )
        seen_count = min(training.eval_worlds, len(seen))
        if seen_count:
            legs["seen"] = evaluate_rows(
                model,
                corpora,
                seen.prefix(seen_count),
                n_samples=training.eval_samples,
                seed=training.eval_seed,
                device=resolved_device,
                leg="seen",
                result_sink=result_sink,
                per_world=True,
            )
        flat: dict[str, float | None] = {}
        for record in legs.values():
            flat.update(flatten_evaluation(record))
        undefined = {
            f"{leg}/{key}": reason
            for leg, record in legs.items()
            for key, reason in record["undefined"].items()
        }
        from pymc_mfm.diagnostics.coverage import coverage_evidence

        # Per-effect strata and coverage/cell evidence stay in evaluation-table
        # artifacts, never evaluations.jsonl, checkpoints or evaluation events.
        strata = {leg: values.pop("strata") for leg, values in legs.items()}
        coverage = {leg: coverage_evidence(values.pop("per_world")) for leg, values in legs.items()}
        for leg, values in legs.items():
            values["rows"] = holdout_rows if leg == "holdout" else seen.prefix(seen_count)
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
        # Tables land before the evaluation line, including report-only coverage
        # evidence for bootstrap intervals without replaying model inference.
        tables_dir.mkdir(exist_ok=True)
        _write_json(
            tables_dir / f"step_{step:09d}.json",
            {
                "step": step,
                "legs": {
                    leg: {
                        **values["tables"],
                        "strata": strata[leg],
                        "coverage": coverage[leg],
                    }
                    for leg, values in legs.items()
                },
            },
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

    def evaluate_and_checkpoint(step: int) -> None:
        evaluation = evaluate(step)
        checkpoint = record_checkpoint(
            "last", step, "checkpoint_last.pt", evaluation, evaluation["selection"]
        )
        best = None
        if evaluation["selection"]["promoted"]:
            best = record_checkpoint(
                "best", step, "checkpoint_best.pt", evaluation, evaluation["selection"]
            )
        record_archive(step, checkpoint, best)
        record_diagnostics(step, checkpoint)

    def carry_terminal_checkpoint() -> None:
        """A new directory resumed at the final step has no update left to write its own.

        The exported bundle's source is ``checkpoint_last.pt``, so the start
        checkpoint (a ``last`` at the final step) is copied under that name.
        """
        if start.step == training.steps and start.kind == "last" and not start.same_directory:
            shutil.copyfile(start.path, out_dir / "checkpoint_last.pt")
            run["checkpoints"] = [
                *run["checkpoints"],
                checkpoint_entry(out_dir / "checkpoint_last.pt"),
            ]

    def carry_best_checkpoint() -> None:
        """Carry the recorded best checkpoint when its weights are available."""
        if best_step is None or any(
            entry["kind"] == "best" and entry["step"] == best_step for entry in run["checkpoints"]
        ):
            return
        payload = start.payload
        if best_step == start.step and start.kind == "last":
            record_checkpoint(
                "best",
                start.step,
                "checkpoint_best.pt",
                payload["evaluation"],
                payload["selection"],
            )
            return
        entry = copy_best_checkpoint(start, out_dir, best_step)
        if entry is not None:
            run["checkpoints"].append(entry)
        if "resumed_from" in run:
            run["resumed_from"]["best_checkpoint_copied"] = entry is not None
            if entry is None:
                run["resumed_from"]["best_checkpoint_reason"] = (
                    "source best checkpoint is unavailable"
                )

    loss_fn = build_loss_fn(model, loss_weights, training, resolved_device.type)
    run["compile"] = compile_record(training)

    def note_compiled_step(seconds: float) -> None:
        """Record the first compiled step's duration (forward, backward, update) and the counts."""
        if run["compile"]["first_call_seconds"] is None:
            run["compile"]["first_call_seconds"] = seconds
        run["compile"].update(compile_counts())

    status = "failed"
    error: str | None = None
    step = first_step - 1
    completed_updates = step
    try:
        for observer in observers:
            observer.on_run_start(run)
            observer.on_run_config(run)
        if resuming:
            carry_terminal_checkpoint()
            carry_best_checkpoint()
        if training.prefetch_batches > 0:
            source = PrefetchingStream(stream, depth=training.prefetch_batches)
        if resuming and _evaluation_pending(start, training):
            evaluate_and_checkpoint(start.step)
        with metrics_path.open("a") as metrics_handle:
            for step in range(first_step, training.steps + 1):
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
                step_start = _step_clock(resolved_device, measure=measure_unadopted_acceleration)
                if step == 1:
                    with autocast_context(training, resolved_device.type):
                        terms, trace = model.loss_terms(
                            batch, weights=loss_weights, return_trace=True
                        )
                    _write_json(
                        out_dir / "objective_trace.json", objective_trace_to_jsonable(trace)
                    )
                else:
                    terms = loss_fn(batch)
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, training.grad_clip)
                optimizer.step()
                step_seconds = (
                    _step_clock(resolved_device, measure=measure_unadopted_acceleration)
                    - step_start
                )
                if training.compile and step != 1:
                    note_compiled_step(step_seconds)
                completed_updates = step
                note_consumed(step, provenance_record)

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
                        "worlds_per_second": (step - first_step + 1)
                        * training.batch_size
                        / elapsed,
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
                    evaluate_and_checkpoint(step)
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
