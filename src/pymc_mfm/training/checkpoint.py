"""Atomic training checkpoints with run provenance and strict reload.

A training checkpoint contains the model configuration and weights together
with the resolved training configuration, run identity, corpus provenance,
consumption, evaluation, and selection records needed to interpret it.

Format 3 adds a nullable ``resume`` block: optimizer state, RNG states, the
stream position at the consumed-batch boundary and the consumption record.
``last`` and ``milestone`` checkpoints carry it; ``best`` checkpoints hold
``None``. Format 2 (no ``resume`` key) is still read, for weights and scoring,
but cannot resume a run. Format 1 and dimensional checkpoints are refused.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pymc_mfm.inference._artifact_schema import semantic_schema, validate_semantic_schema
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM

CHECKPOINT_FORMAT_VERSION = 3
#: Formats ``load_checkpoint`` reads; each has its own exact key set.
READABLE_FORMAT_VERSIONS = (2, 3)
CHECKPOINT_KINDS = ("last", "best", "milestone")

_PAYLOAD_KEYS_V2 = frozenset(
    {
        "format_version",
        "kind",
        "step",
        "config",
        "schemas",
        "state_dict",
        "training_config",
        "run",
        "consumed",
        "evaluation",
        "selection",
    }
)
_PAYLOAD_KEYS = {2: _PAYLOAD_KEYS_V2, 3: _PAYLOAD_KEYS_V2 | {"resume"}}
_RESUME_KEYS = frozenset(
    {
        "optimizer",
        "step",
        "torch_rng",
        "device_type",
        "accelerator_rng",
        "stream",
        "consumption",
        "best_value",
        "best_step",
    }
)


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write ``payload`` to a sibling temporary file, then rename into place.

    A crash mid-write leaves either the previous file or nothing, never a
    truncated checkpoint under the final name.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    *,
    model: MFM,
    kind: str,
    step: int,
    training_config: Mapping[str, Any],
    run: Mapping[str, Any],
    consumed: Mapping[str, Any],
    evaluation: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
    resume: Mapping[str, Any] | None = None,
) -> str:
    """Atomically write a training checkpoint and return its SHA-256.

    ``resume`` is the restart block (exactly :data:`_RESUME_KEYS`) or ``None``, as
    for ``best`` checkpoints, which serve weights and scoring only.
    ``kind`` is ``last`` for the latest evaluation checkpoint, ``best`` for a
    promoted checkpoint, or ``milestone`` for an exact configured step.
    ``evaluation`` is the accompanying record, or ``None`` for a milestone.
    """
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer")
    if resume is not None:
        _check_resume_block(resume, step)
    previous = getattr(model, "artifact_metadata", {})
    if previous:
        validate_semantic_schema(previous.get("schemas"), model.config, label="loaded artifact")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "step": step,
        "config": model.config.to_dict(),
        "schemas": semantic_schema(model.config),
        "state_dict": model.state_dict(),
        "training_config": dict(training_config),
        "run": dict(run),
        "consumed": dict(consumed),
        "evaluation": None if evaluation is None else dict(evaluation),
        "selection": dict(selection),
        "resume": None if resume is None else dict(resume),
    }
    atomic_torch_save(payload, path)
    return file_sha256(path)


def _check_resume_block(resume: Mapping[str, Any], step: int) -> None:
    if set(resume) != _RESUME_KEYS:
        raise ValueError(f"resume must contain exactly {sorted(_RESUME_KEYS)}")
    if resume["step"] != step:
        raise ValueError(f"resume step {resume['step']!r} does not match checkpoint step {step}")


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[MFM, dict]:
    """Strictly load a training checkpoint into an evaluation-mode model.

    Returns the model and the complete non-tensor payload (with ``state_dict``
    removed). Only tensors and primitives are deserialized; configuration
    fields are never defaulted. Formats 2 and 3 are read; a format 3 payload holds
    its ``resume`` block (``None`` for ``best``), a format 2 payload has no such key.
    Checkpoints from dimensional preprocessing must be retrained, not converted by
    preserving their equal-shaped weights.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or type(payload.get("format_version")) is not int
        or payload["format_version"] not in READABLE_FORMAT_VERSIONS
        or "schemas" not in payload
    ):
        raise ValueError(
            "training checkpoint has an incompatible input representation or format; "
            "retrain with the current context-only unit-invariant preprocessing. "
            "Old dimensional checkpoints, including A/B weights, cannot be converted."
        )
    expected = _PAYLOAD_KEYS[payload["format_version"]]
    if payload.keys() != expected:
        raise ValueError(f"training checkpoint must contain exactly {sorted(expected)}")
    if payload.get("resume") is not None:
        _check_resume_block(payload["resume"], payload["step"])
    validate_semantic_schema(payload["schemas"], label="training checkpoint schemas")
    config = ModelConfig.from_dict(payload["config"])
    validate_semantic_schema(payload["schemas"], config, label="training checkpoint schemas")
    with torch.device("meta"):
        model = MFM(config)
    model.load_state_dict(payload.pop("state_dict"), strict=True, assign=True)
    return model.to(device).eval(), payload
