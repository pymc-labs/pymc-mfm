"""Atomic training checkpoints with run provenance and strict reload.

A training checkpoint contains the model configuration and weights together
with the resolved training configuration, run identity, corpus provenance,
consumption, evaluation, and selection records needed to interpret it.
Optimizer state is intentionally absent, so checkpoints reproduce model
weights rather than resumable optimizer state.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM

CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_KINDS = ("last", "best", "milestone")

_PAYLOAD_KEYS = frozenset(
    {
        "format_version",
        "kind",
        "step",
        "config",
        "state_dict",
        "training_config",
        "run",
        "consumed",
        "evaluation",
        "selection",
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
) -> str:
    """Atomically write a training checkpoint and return its SHA-256.

    ``kind`` is ``last`` for the latest evaluation checkpoint, ``best`` for a
    promoted checkpoint, or ``milestone`` for an exact configured step.
    ``evaluation`` is the accompanying record, or ``None`` for a milestone.
    """
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
    if type(step) is not int or step < 1:
        raise ValueError("step must be a positive integer")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "step": step,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "training_config": dict(training_config),
        "run": dict(run),
        "consumed": dict(consumed),
        "evaluation": None if evaluation is None else dict(evaluation),
        "selection": dict(selection),
    }
    atomic_torch_save(payload, path)
    return file_sha256(path)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[MFM, dict]:
    """Strictly load a training checkpoint into an evaluation-mode model.

    Returns the model and the complete non-tensor payload (with ``state_dict``
    removed). Only tensors and primitives are deserialized; configuration
    fields are never defaulted.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.keys() != _PAYLOAD_KEYS:
        raise ValueError(f"training checkpoint must contain exactly {sorted(_PAYLOAD_KEYS)}")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported training checkpoint format_version: {payload['format_version']!r}"
        )
    config = ModelConfig.from_dict(payload["config"])
    with torch.device("meta"):
        model = MFM(config)
    model.load_state_dict(payload.pop("state_dict"), strict=True, assign=True)
    return model.to(device).eval(), payload
