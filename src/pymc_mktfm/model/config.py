"""Construction inputs and objective weights of the one graph-conditional architecture."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import ClassVar, Self


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelConfig:
    """Size and corpus geometry of the canonical architecture.

    Version 2 holds only what differs between trainings of the same network:
    model size (``d_model``, ``n_layers``, ``n_heads``, ``d_ff``,
    ``flow_hidden``) and corpus geometry (``n_controls``, ``n_demand``).
    Everything else is fixed: causal bi-attention, all five observation feature
    blocks, signed sales-scale targets, the eight-family graph router, the
    support-projected conditional decoder, coherent context-state flows with
    fixed sampling numerics (:mod:`pymc_mktfm.model.flow` constants), and the
    extended control-equivariant structure objective.

    Loss weights are part of the training recipe (:class:`LossWeights`). Paths,
    corpus identity, device, seed, optimizer, and run budget are not model
    inputs either. Channels are count-agnostic; ``n_controls`` records training
    geometry and ``n_demand`` also sizes router demand embeddings.
    """

    format_version: ClassVar[int] = 2

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    flow_hidden: int = 256
    n_controls: int = 6
    n_demand: int = 1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

    def to_dict(self) -> dict[str, int]:
        """Return the complete JSON-safe configuration, never a default override."""
        return {"format_version": self.format_version, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Load an exact versioned configuration without reconstructing defaults.

        Missing fields are errors: filling them with today's defaults could
        silently change the meaning of already-trained weights.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("model configuration must be a mapping")
        version = payload.get("format_version")
        if type(version) is not int or version != cls.format_version:
            raise ValueError(f"unsupported model configuration format_version: {version!r}")
        names = {field.name for field in fields(cls)}
        missing = names - payload.keys()
        unknown = payload.keys() - names - {"format_version"}
        if missing or unknown:
            raise ValueError(
                f"invalid model configuration fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown, key=str)}"
            )
        return cls(**{name: payload[name] for name in names})


@dataclass(frozen=True, slots=True, kw_only=True)
class LossWeights:
    """Coefficients of the training objective; the defaults are the A/B recipe.

    ``struct_weight`` scales the auxiliary structure BCE, whose ``cy`` family
    enters the equal-family mean with weight ``struct_cy_weight``.
    ``context_graph_weight`` is the context router's graph coefficient relative
    to its decomposition term, not a whole-term multiplier. With
    ``cdw = context_decomposition_weight`` and ``gcw = graph_conditional_weight``::

        task_loss = flow_loss + cdw * context_flow_loss
                    + gcw * graph_conditional_flow_loss
                    + gcw * cdw * graph_conditional_context_flow_loss
        loss = task_loss + struct_loss

    Every weight is finite and strictly positive: each objective stays active.
    """

    struct_weight: float = 1.0
    struct_cy_weight: float = 1.0
    graph_conditional_weight: float = 1.0
    context_decomposition_weight: float = 0.5
    context_graph_weight: float = 0.1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field.name} must be finite and positive")
            object.__setattr__(self, field.name, float(value))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Rebuild exactly from :meth:`to_dict`; a missing weight is an error."""
        if not isinstance(payload, Mapping):
            raise ValueError("loss weights must be a mapping")
        names = {field.name for field in fields(cls)}
        missing = names - payload.keys()
        unknown = payload.keys() - names
        if missing or unknown:
            raise ValueError(
                f"invalid loss weight fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown, key=str)}"
            )
        return cls(**{name: payload[name] for name in names})
