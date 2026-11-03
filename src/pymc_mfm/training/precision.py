"""Training-step forward pass: optional bf16 autocast and ``torch.compile``.

Only the forward pass and loss (``MFM.loss_terms``) run under autocast or in
the compiled callable. Forward and loss use bf16 autocast. Parameters and
accumulated parameter gradients remain fp32. Backward runs outside autocast.
Evaluation never uses either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch._dynamo.utils import counters

from pymc_mfm.model.config import LossWeights
from pymc_mfm.model.model import MFM
from pymc_mfm.training.config import TrainingConfig

LossFn = Callable[[Mapping[str, Any]], dict[str, torch.Tensor]]


def autocast_context(training: TrainingConfig, device_type: str):
    """Autocast to bfloat16 for ``precision="bf16"``; a no-op for ``"fp32"``."""
    return torch.autocast(device_type, dtype=torch.bfloat16, enabled=training.precision == "bf16")


def build_loss_fn(
    model: MFM, weights: LossWeights, training: TrainingConfig, device_type: str
) -> LossFn:
    """Return ``batch -> loss terms`` for every step after the traced first one.

    ``MFM`` has no custom ``forward``, so the compiled callable wraps
    ``loss_terms`` instead of the module. ``fallback_random`` makes compiled
    code draw its noise from the eager RNG stream, so compiled and eager
    losses agree to float32 round-off while every parameter takes part.

    Two differences remain, so compiled training is not optimizer-equivalent to eager.
    A parameter that a batch leaves unused gets a zero gradient instead of ``None``,
    so AdamW still applies momentum and weight decay to it and the runs drift apart.
    Turning exact-zero gradients back into ``None`` was tried and made compiled
    steps slower than eager on an L4, so it is not done. Compiled steps also skip
    the mask value checks, which step 1 and evaluation still run eagerly.
    """

    def loss_fn(batch):
        with autocast_context(training, device_type):
            return model.loss_terms(batch, weights=weights)

    if not training.compile:
        return loss_fn
    torch._inductor.config.fallback_random = True
    # Drop graphs compiled for an earlier run's model, and count from zero.
    torch._dynamo.reset()
    counters.clear()
    return torch.compile(loss_fn, backend="inductor", dynamic=None)


def compile_record(training: TrainingConfig) -> dict[str, Any]:
    """The ``run.json`` ``compile`` block, before any compiled call has run."""
    return {
        "enabled": training.compile,
        "backend": "inductor" if training.compile else None,
        "fallback_random": True if training.compile else None,
        "first_call_seconds": None,
        "graphs": None,
        "graph_breaks": None,
    }


def compile_counts() -> dict[str, int]:
    """Graphs captured and graph breaks seen since the last ``counters.clear()``.

    A new input shape (a context-length bucket) adds graphs; dynamo keeps no separate
    recompile counter.
    """
    return {
        "graphs": counters["stats"]["unique_graphs"],
        "graph_breaks": sum(counters["graph_break"].values()),
    }
