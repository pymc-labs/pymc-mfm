"""Construction inputs for the single trainable graph-conditional architecture."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import ClassVar, Self


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelConfig:
    """Shared construction and checkpoint configuration, resolved from A and B.

    Version 1 fixes causal bi-attention, all five observation feature blocks,
    signed sales-scale targets, the eight-family graph router, support-projected
    conditional decoder, coherent context-state flows, and the extended
    control-equivariant structure objective. These are not optional architectures.

    Paths, corpus identity, device, seed, optimizer and run budget are not model
    inputs. Channels are count-agnostic; ``n_controls`` records the training
    geometry and ``n_demand`` also sizes the router's demand embeddings.
    """

    format_version: ClassVar[int] = 1

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 512
    flow_hidden: int = 256
    flow_n_freq: int = 2
    flow_steps_sample: int = 16
    flow_mc_samples: int = 32
    flow_ode_method: str = "euler"
    flow_trajectory_correlation: float = 0.98
    n_controls: int = 6
    n_demand: int = 1
    struct_weight: float = 1.0
    struct_cy_weight: float = 1.0
    graph_conditional_weight: float = 1.0
    context_decomposition_weight: float = 0.5
    context_graph_weight: float = 0.1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is int:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{field.name} must be a positive integer")
            elif field.type is float:
                if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                    raise ValueError(f"{field.name} must be finite and positive")
                object.__setattr__(self, field.name, float(value))
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not isinstance(self.flow_ode_method, str) or self.flow_ode_method not in (
            "euler",
            "heun",
        ):
            raise ValueError("flow_ode_method must be 'euler' or 'heun'")
        if self.flow_trajectory_correlation >= 1:
            raise ValueError("flow_trajectory_correlation must be strictly between zero and one")

    def to_dict(self) -> dict[str, int | float | str]:
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
