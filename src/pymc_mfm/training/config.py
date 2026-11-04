"""Training configuration: update budget, optimizer, objective, schedule, and cadence.

``TrainingConfig()`` defines the canonical full run: 62,500 AdamW updates of
64 worlds, a peak learning rate of 2.8e-3 after 3,125 warmup updates, cosine
decay to zero, weight decay 0.01, global gradient clipping at 1.0, the A/B
loss weights (the :class:`~pymc_mfm.model.config.LossWeights` defaults),
evaluation every 6,250 updates on 16 held-out worlds with 128 draws,
milestones at 25,000 and 50,000, and seed 31. Corpus selection does not alter
this execution contract.

Architecture size and geometry are configured by
:class:`pymc_mfm.model.config.ModelConfig`; paths, device, and corpus selection
are arguments to :func:`pymc_mfm.training.train`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import Any, ClassVar, Self

from pymc_mfm.model.config import LossWeights

CANONICAL_STEPS = 62_500
CANONICAL_BATCH_SIZE = 64
SELECTION_HEADS = ("conditional", "joint")
_LOSS_WEIGHT_FIELDS = tuple(field.name for field in fields(LossWeights))


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingConfig:
    """Run budget and optimization settings, recorded verbatim in every run."""

    #: PyTorch AdamW defaults.
    adam_betas: ClassVar[tuple[float, float]] = (0.9, 0.999)
    adam_eps: ClassVar[float] = 1e-8

    steps: int = CANONICAL_STEPS
    batch_size: int = CANONICAL_BATCH_SIZE
    lr: float = 2.8e-3
    lr_schedule: str = "cosine"
    warmup_steps: int = 3_125
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    #: Objective coefficients passed to the model as :attr:`loss_weights`.
    struct_weight: float = 1.0
    struct_cy_weight: float = 1.0
    graph_conditional_weight: float = 1.0
    context_decomposition_weight: float = 0.5
    context_graph_weight: float = 0.1
    seed: int = 31
    eval_every: int = 6_250
    eval_worlds: int = 16
    eval_samples: int = 128
    eval_seed: int = 616_161
    #: Head whose held-out query readout selects checkpoints.
    selection_head: str = "conditional"
    log_every: int = 100
    milestone_steps: tuple[int, ...] = (25_000, 50_000)
    prefetch_batches: int = 4

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "eval_every", "eval_worlds", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "prefetch_batches"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("seed", "eval_seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("lr", "grad_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        # LossWeights is the one validation path for the objective coefficients.
        for name, value in self.loss_weights.to_dict().items():
            object.__setattr__(self, name, value)
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if self.selection_head not in SELECTION_HEADS:
            raise ValueError(f"selection_head must be one of {SELECTION_HEADS}")
        # The held-out energy score that selects checkpoints needs two draws.
        if type(self.eval_samples) is not int or self.eval_samples < 2:
            raise ValueError("eval_samples must be an integer >= 2")
        milestones = tuple(self.milestone_steps)
        if any(type(step) is not int for step in milestones):
            raise ValueError("milestone_steps must contain integers")
        if milestones != tuple(sorted(set(milestones))):
            raise ValueError("milestone_steps must be strictly increasing")
        if milestones and (milestones[0] < 1 or milestones[-1] > self.steps):
            raise ValueError("milestone_steps must lie within [1, steps]")
        object.__setattr__(self, "milestone_steps", milestones)

    @property
    def budget(self) -> str:
        """``"canonical"`` for the recorded full-run budget, else ``"reduced"``."""
        if self.steps == CANONICAL_STEPS and self.batch_size == CANONICAL_BATCH_SIZE:
            return "canonical"
        return "reduced"

    @property
    def loss_weights(self) -> LossWeights:
        """The objective coefficients this run passes to ``MFM.loss_terms``."""
        return LossWeights(**{name: getattr(self, name) for name in _LOSS_WEIGHT_FIELDS})

    def learning_rate(self, step: int) -> float:
        """Return the learning rate for a 1-indexed update before it runs.

        The cosine schedule warms linearly to ``lr`` over ``warmup_steps``,
        then decays to exactly zero at ``steps``. It has no floor.
        """
        if step < 1 or step > self.steps:
            raise ValueError(f"step must lie within [1, {self.steps}], got {step}")
        if self.lr_schedule == "constant":
            return self.lr
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.steps - self.warmup_steps)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def to_dict(self) -> dict[str, object]:
        """Complete JSON-safe record including the fixed optimizer constants."""
        payload = asdict(self)
        payload["milestone_steps"] = list(self.milestone_steps)
        payload["adam_betas"] = list(self.adam_betas)
        payload["adam_eps"] = self.adam_eps
        payload["budget"] = self.budget
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Rebuild from :meth:`to_dict`; derived and constant fields are checked."""
        names = {field.name for field in fields(cls)}
        derived = {"adam_betas", "adam_eps", "budget"}
        missing = names - payload.keys()
        unknown = payload.keys() - names - derived
        if missing or unknown:
            raise ValueError(
                f"invalid training configuration fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown, key=str)}"
            )
        values = {name: payload[name] for name in names}
        values["milestone_steps"] = tuple(values["milestone_steps"])
        config = cls(**values)
        if "adam_betas" in payload and tuple(payload["adam_betas"]) != cls.adam_betas:
            raise ValueError("adam_betas differ from the fixed AdamW defaults")
        if "adam_eps" in payload and payload["adam_eps"] != cls.adam_eps:
            raise ValueError("adam_eps differs from the fixed AdamW default")
        return config


#: Named training configurations. ``canonical`` is ``TrainingConfig()``. ``mini``
#: and ``smoke`` are development presets for the ``mini`` corpus plans; they are
#: not validated surrogates for the canonical run, so results from them do not
#: rank full-scale choices.
PRESETS: Mapping[str, TrainingConfig] = MappingProxyType(
    {
        "canonical": TrainingConfig(),
        "mini": TrainingConfig(
            steps=8_000,
            batch_size=64,
            warmup_steps=400,
            eval_every=1_000,
            milestone_steps=(),
        ),
        "smoke": TrainingConfig(
            steps=50,
            batch_size=4,
            warmup_steps=10,
            eval_every=50,
            eval_worlds=4,
            eval_samples=8,
            log_every=10,
            milestone_steps=(),
        ),
    }
)


def training_cli_fields() -> frozenset[str]:
    """The ``TrainingConfig`` fields that the training CLI and launch files accept."""
    return frozenset(field.name for field in fields(TrainingConfig))


def resolve_training(preset: str, overrides: Mapping[str, Any]) -> TrainingConfig:
    """The config the training CLI builds from ``preset`` and explicit overrides.

    Without a ``milestone_steps`` override, the preset's milestones within ``steps`` are kept.
    Raises ``ValueError`` for a value the config refuses.
    """
    values = {name: getattr(PRESETS[preset], name) for name in training_cli_fields()}
    values.update(overrides)
    if "milestone_steps" not in overrides and type(values["steps"]) is int:
        values["milestone_steps"] = tuple(
            step for step in PRESETS[preset].milestone_steps if step <= values["steps"]
        )
    values["milestone_steps"] = tuple(values["milestone_steps"])
    return TrainingConfig(**values)
