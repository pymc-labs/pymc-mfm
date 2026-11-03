"""Exact resume and weights-only warm start for training runs.

``train(resume_from=...)`` continues a run at ``step + 1`` of its original
schedule from a ``last`` or ``milestone`` checkpoint (format 3). It restores
the optimizer, the torch RNG, the batch stream and the consumption record, so
the later losses, gradient norms and weights equal those of an uninterrupted
run. ``train(init_from=...)`` loads weights only, from a format 2 or 3
checkpoint, and starts a new run at step 1 with a fresh optimizer.

This module holds the pieces that do not need the training loop: refusal
checks that run before any file is written, the chained consumption digest,
RNG capture and its device check, and the reconciliation of a run directory's records with a
checkpoint step.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM
from pymc_mfm.training.checkpoint import file_sha256, load_checkpoint
from pymc_mfm.training.config import TrainingConfig
from pymc_mfm.training.stream import SeenRows

#: Name of the checkpoint a same-directory resume may start from.
LAST_CHECKPOINT = "checkpoint_last.pt"
#: Recorded with every consumed record; names how ``row_sequence_sha256`` is chained.
ROW_SEQUENCE_SCHEME = "chained_sha256_v1"
#: ``row_sequence_sha256`` before any batch: the digest of no bytes.
EMPTY_ROW_SEQUENCE = hashlib.sha256(b"").hexdigest()
#: Training-config fields that do not change numerics; a resume may differ in them.
RESUME_FREE_FIELDS = ("prefetch_batches",)

_JSONL_FILES = ("metrics.jsonl", "evaluations.jsonl", "diagnostics.jsonl")
_KIND_ORDER = {"milestone": 0, "last": 1, "best": 2}
_STEP_IN_NAME = re.compile(r"step_0*(\d+)")


def chain_row_sequence(previous: str, rows: Sequence[Sequence[int]]) -> str:
    """Next link of the consumption digest.

    ``d_n = sha256(bytes.fromhex(d_{n-1}) + json(rows_n))``. The running value is
    the whole state, so it serializes and resumes without a consumed-row log.
    """
    encoded = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(bytes.fromhex(previous) + encoded).hexdigest()


def evaluation_sha256(evaluation: Mapping[str, Any]) -> str:
    """Digest of an evaluation record as a checkpoint entry lists it."""
    return hashlib.sha256(
        json.dumps(evaluation, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """RNG state of the training device, plus the CPU torch RNG.

    ``device_type`` names the accelerator whose state is held in ``accelerator_rng``:
    all CUDA devices' states for ``cuda``, the MPS state for ``mps``, ``None`` for
    ``cpu``. The training path draws no NumPy or Python global random numbers: the
    stream owns its generator and evaluation forks and restores the torch RNG.
    """
    accelerator_rng = None
    if device.type == "cuda":
        accelerator_rng = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        accelerator_rng = torch.mps.get_rng_state()
    return {
        "torch_rng": torch.random.get_rng_state(),
        "device_type": device.type,
        "accelerator_rng": accelerator_rng,
    }


def check_rng_fits_device(resume: Mapping[str, Any], device: torch.device) -> None:
    """Refuse a resume whose saved accelerator RNG cannot continue on ``device``."""
    saved = resume["device_type"]
    if saved != device.type:
        raise ValueError(
            f"resume_from: the checkpoint was written on a {saved} device and cannot "
            f"continue its RNG on a {device.type} device"
        )
    if saved == "cuda" and len(resume["accelerator_rng"]) != torch.cuda.device_count():
        raise ValueError(
            f"resume_from: the checkpoint holds RNG states of {len(resume['accelerator_rng'])} "
            f"CUDA devices, but {torch.cuda.device_count()} are visible"
        )


def restore_rng_state(resume: Mapping[str, Any]) -> None:
    """Set the torch RNGs from a resume block that :func:`check_rng_fits_device` accepted."""
    torch.random.set_rng_state(resume["torch_rng"])
    if resume["device_type"] == "cuda":
        torch.cuda.set_rng_state_all(list(resume["accelerator_rng"]))
    elif resume["device_type"] == "mps":
        torch.mps.set_rng_state(resume["accelerator_rng"])


def build_resume_block(
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    step: int,
    stream_state: Mapping[str, Any],
    row_sequence: str,
    seen: SeenRows,
    best_value: float | None,
    best_step: int | None,
) -> dict[str, Any]:
    """The checkpoint ``resume`` block at the end of ``step``'s update."""
    return {
        "optimizer": optimizer.state_dict(),
        "step": step,
        **capture_rng_state(device),
        "stream": dict(stream_state),
        "consumption": {
            "scheme": ROW_SEQUENCE_SCHEME,
            "row_sequence_sha256": row_sequence,
            "seen_rows": seen.state(),
        },
        "best_value": best_value,
        "best_step": best_step,
    }


@dataclass(frozen=True)
class StartPoint:
    """A validated checkpoint that a run starts from."""

    mode: str  # "resume" or "init"
    path: Path
    sha256: str
    payload: dict[str, Any]
    model: MFM
    same_directory: bool

    @property
    def step(self) -> int:
        return int(self.payload["step"])

    @property
    def kind(self) -> str:
        return str(self.payload["kind"])

    @property
    def source_dir(self) -> Path:
        return self.path.parent

    @property
    def source_run_id(self) -> str | None:
        return self.payload["run"].get("run_id")

    @property
    def source_tracking_run_id(self) -> str | None:
        tracking = self.payload["run"].get("tracking") or {}
        return tracking.get("run_id")


def _differing(recorded: Mapping[str, Any], current: Mapping[str, Any], ignore=()) -> list[str]:
    return sorted(
        str(key)
        for key in recorded.keys() | current.keys()
        if key not in ignore and recorded.get(key) != current.get(key)
    )


def _describe(recorded: Mapping[str, Any], current: Mapping[str, Any], names: Iterable[str]) -> str:
    return ", ".join(
        f"{name} (checkpoint {recorded.get(name)!r}, requested {current.get(name)!r})"
        for name in names
    )


def load_start_point(
    path: str | Path,
    *,
    mode: str,
    out_dir: Path,
    device: torch.device,
    model_config: ModelConfig,
    training: TrainingConfig,
    corpus_records: Sequence[Mapping[str, Any]],
) -> StartPoint:
    """Load and check a checkpoint for ``mode`` (``"resume"`` or ``"init"``).

    Nothing is written. Raises ``ValueError`` when the checkpoint cannot serve the
    mode: a dimensional or unreadable file, a model configuration that differs, and
    for a resume also a format 2 file, a checkpoint without resume state, a
    ``training_config`` that differs, a different corpus, RNG state that does not fit
    ``device``, or a same-directory
    request that does not start from the directory's own ``checkpoint_last.pt``.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{mode} checkpoint {path} does not exist")
    model, payload = load_checkpoint(path)
    label = "resume_from" if mode == "resume" else "init_from"
    differing = _differing(payload["config"], model_config.to_dict())
    if differing:
        raise ValueError(
            f"{label} model config differs: "
            + _describe(payload["config"], model_config.to_dict(), differing)
        )
    same_directory = path.resolve().parent == out_dir.resolve()
    start = StartPoint(
        mode=mode,
        path=path,
        sha256=file_sha256(path),
        payload=payload,
        model=model,
        same_directory=same_directory and mode == "resume",
    )
    if mode == "resume":
        _check_resumable(start, out_dir, device, training, corpus_records)
    return start


def _check_resumable(
    start: StartPoint,
    out_dir: Path,
    device: torch.device,
    training: TrainingConfig,
    corpus_records: Sequence[Mapping[str, Any]],
) -> None:
    payload = start.payload
    if payload["format_version"] < 3:
        raise ValueError(
            f"resume_from: {start.path} is a format {payload['format_version']} checkpoint, "
            "which holds no optimizer, RNG or stream state; use init_from for a "
            "weights-only warm start"
        )
    if payload["resume"] is None:
        raise ValueError(
            f"resume_from: {start.path} is a {payload['kind']!r} checkpoint without resume "
            "state; resume from a 'last' or 'milestone' checkpoint"
        )
    check_rng_fits_device(payload["resume"], device)
    recorded, current = payload["training_config"], training.to_dict()
    differing = _differing(recorded, current, ignore=RESUME_FREE_FIELDS)
    if differing:
        raise ValueError(
            "resume_from: training_config differs from the checkpoint's: "
            + _describe(recorded, current, differing)
        )
    recorded_corpus = [
        record["actual_inputs"]["content_sha256"] for record in payload["run"]["corpus"]
    ]
    current_corpus = [record["actual_inputs"]["content_sha256"] for record in corpus_records]
    if None in recorded_corpus or None in current_corpus or recorded_corpus != current_corpus:
        raise ValueError(
            "resume_from: the corpora differ from the checkpoint's (content SHA-256 "
            f"{recorded_corpus} recorded, {current_corpus} given)"
        )
    if start.same_directory:
        if start.path.name != LAST_CHECKPOINT or payload["kind"] != "last":
            raise ValueError(
                f"resume_from: resuming into the checkpoint's own directory starts only from "
                f"its {LAST_CHECKPOINT}, not {start.path.name}"
            )
        run = _read_run_json(out_dir)
        if run is None or run.get("run_id") != payload["run"]["run_id"]:
            raise ValueError(
                f"resume_from: {out_dir / 'run.json'} is missing or belongs to another run"
            )
    elif (out_dir / "run.json").exists():
        raise FileExistsError(f"{out_dir} already holds a training run")


def _read_run_json(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "run.json"
    return json.loads(path.read_text()) if path.is_file() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parsed records of a JSON-lines file; a missing file is empty and a torn line is skipped."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a write cut short by a crash
    return records


def _rewrite_jsonl(path: Path, keep: Callable[[dict[str, Any]], bool]) -> None:
    if not path.is_file():
        return
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in read_jsonl(path) if keep(r))
    )
    tmp.replace(path)


def _step_of(name: str) -> int | None:
    match = _STEP_IN_NAME.search(name)
    return None if match is None else int(match[1])


def truncate_run_files(out_dir: Path, step: int) -> None:
    """Drop every record and artifact after ``step`` from a run directory.

    Covers the JSONL logs, evaluation tables, diagnostics, archives and milestone
    checkpoints. ``tracking/`` is append-only delivery evidence and is kept.
    """
    for name in _JSONL_FILES:
        _rewrite_jsonl(out_dir / name, lambda record: int(record.get("step", 0)) <= step)
    for directory, pattern in (
        ("evaluation_tables", "step_*.json"),
        ("diagnostics", "step_*"),
        ("archives", "step_*"),
        (".", "checkpoint_step_*.pt"),
    ):
        for path in (out_dir / directory).glob(pattern):
            if (found := _step_of(path.name)) is not None and found > step:
                shutil.rmtree(path) if path.is_dir() else path.unlink()


def copy_run_history(source: Path, out_dir: Path, *, step: int, evaluation_step: int) -> bool:
    """Copy a source run's metrics, evaluations, tables and objective trace up to the step.

    Metrics keep ``step`` and earlier; evaluations and their tables keep
    ``evaluation_step`` and earlier. Returns whether the source had a metrics log.
    Diagnostics reports and archives stay in the source directory.
    """
    for name, limit in (("metrics.jsonl", step), ("evaluations.jsonl", evaluation_step)):
        records = [r for r in read_jsonl(source / name) if int(r.get("step", 0)) <= limit]
        if records:
            (out_dir / name).write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
            )
    tables = source / "evaluation_tables"
    if tables.is_dir():
        (out_dir / "evaluation_tables").mkdir(exist_ok=True)
        for path in tables.glob("step_*.json"):
            if (found := _step_of(path.name)) is not None and found <= evaluation_step:
                shutil.copyfile(path, out_dir / "evaluation_tables" / path.name)
    if (source / "objective_trace.json").is_file():
        shutil.copyfile(source / "objective_trace.json", out_dir / "objective_trace.json")
    return (source / "metrics.jsonl").is_file()


def checkpoint_entry(path: Path) -> dict[str, Any]:
    """A ``run.json`` checkpoint entry for a checkpoint file, read from the file."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    entry = {
        "kind": payload["kind"],
        "step": int(payload["step"]),
        "file": path.name,
        "sha256": file_sha256(path),
    }
    if payload["evaluation"] is not None:
        entry["evaluation_sha256"] = evaluation_sha256(payload["evaluation"])
    return entry


def reconcile_checkpoints(
    out_dir: Path, listed: Sequence[Mapping[str, Any]], step: int
) -> list[dict[str, Any]]:
    """Checkpoint entries of the files in ``out_dir`` at ``step`` or earlier.

    ``listed`` is ``run.json``'s own list, trusted per file only while the file's
    SHA-256 still matches; a run killed before its final ``run.json`` write lists
    nothing, so other files are read. Files are ordered by step, then kind.
    """
    by_file = {entry["file"]: dict(entry) for entry in listed}
    entries = []
    for path in sorted(out_dir.glob("checkpoint_*.pt")):
        entry = by_file.get(path.name)
        if entry is None or entry["sha256"] != file_sha256(path):
            entry = checkpoint_entry(path)
        if entry["step"] <= step:
            entries.append(entry)
    return sorted(entries, key=lambda e: (e["step"], _KIND_ORDER[e["kind"]]))


def continue_run_record(
    existing: Mapping[str, Any], start: StartPoint, out_dir: Path
) -> dict[str, Any]:
    """The ``run.json`` content a same-directory resume continues from.

    Keeps the run identity; clears the end-of-run fields; keeps only the
    checkpoints and diagnostics entries at or before the checkpoint step; appends
    a ``resumes`` entry.
    """
    run = {
        key: value
        for key, value in existing.items()
        if key not in ("error_type", "export", "elapsed_seconds")
    }
    run.update(status="running", finished_at=None, error=None)
    run["checkpoints"] = reconcile_checkpoints(out_dir, existing.get("checkpoints", ()), start.step)
    run["diagnostics"] = [
        entry
        for entry in existing.get("diagnostics", ())
        if (_step_of(entry.get("name", "")) or 0) <= start.step
    ]
    run["resumes"] = [
        *existing.get("resumes", ()),
        {
            "from_step": start.step,
            "checkpoint_sha256": start.sha256,
            "resumed_at": datetime.now(UTC).isoformat(),
        },
    ]
    return run


def provenance_of_start(start: StartPoint) -> dict[str, Any]:
    """The ``resumed_from`` or ``initialized_from`` record of a run that starts elsewhere."""
    if start.mode == "init":
        return {
            "path": str(start.path),
            "sha256": start.sha256,
            "step": start.step,
            "format_version": start.payload["format_version"],
        }
    return {
        "path": str(start.path),
        "step": start.step,
        "sha256": start.sha256,
        "run_id": start.source_run_id,
        "kind": start.kind,
    }
