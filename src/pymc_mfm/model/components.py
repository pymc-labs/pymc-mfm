"""Contracts for the swappable parts of :class:`~pymc_mfm.model.model.MFM`.

Three parts of the model are components: the encoder, the router head and the
conditional head. The structure head (``struct_head``) and the graph conditioner
are not components. Their shape is fixed by the graph geometry, and
``training/evaluation.py`` reads ``struct_head.predictive_probs`` directly.

Each contract is an abstract base class plus a member list that
:func:`missing_members` checks at registration time. The checks and the base
classes ask for the same names. A member that is an instance attribute (for
example ``n_controls``) is declared as a class-level annotation, because the class
has no value for it before construction.

Seeded draws are part of every head contract. ``evaluate_rows`` draws under an
isolated RNG in a fixed call order. A head that consumes a different number of random
values, or the same values in a different order, changes every later draw. The
router's ``sample`` and the conditional head's ``_draw_noise`` define that order.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from .encoder import TimelineRepresentations
    from .flow import ZeroAtomGate

#: Name of the component slots in :class:`~pymc_mfm.model.model.MFM`, in build order.
COMPONENT_SLOTS = ("encoder", "head", "conditional_head")


class ResearchComponentError(ValueError):
    """A research component reached a release path (bundle export or load)."""


class Encoder(ABC):
    """Reads context and query covariates into per-window representations.

    Constructed as ``cls(config)``. ``assert_batch_alignment`` is called by the
    inference entry point before any forward pass.
    """

    @abstractmethod
    def forward_timeline(self, batch: dict[str, Tensor]) -> TimelineRepresentations:
        """Context and query readouts of one timeline (separate marginals)."""

    @abstractmethod
    def forward_with_controls(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Query ``(weekly, channels, controls)`` readouts."""

    @abstractmethod
    def assert_batch_alignment(self, batch: dict[str, Tensor]) -> int:
        """Refuse a batch whose feature layout the encoder cannot read."""


class RouterHead(ABC):
    """Joint flow over decomposition and graph slots; also the shape of a conditional head.

    Constructed as ``cls(config, graph_router=bool)``. ``MFM`` reads ``n_fixed``,
    ``n_demand`` and ``n_controls`` for state layout and ``zero_atom_gate_slices``
    for support gating.
    """

    n_controls: int
    n_demand: int

    @property
    @abstractmethod
    def n_fixed(self) -> int:
        """Number of fixed (confounder and indirect) decomposition coordinates."""

    @abstractmethod
    def state_dim(self, c: int, m: int | None = None) -> int:
        """State width for ``c`` channels and ``m`` controls."""

    @abstractmethod
    def zero_atom_gate_slices(
        self, n_channels: int, n_controls: int
    ) -> tuple[tuple[ZeroAtomGate, slice, slice], ...]:
        """Contribution and absolute edge slices for the support gates."""

    @abstractmethod
    def loss(
        self,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        targets_z: Tensor,
        control_repr: Tensor | None = None,
        state_mask: Tensor | None = None,
        *,
        graph_loss_weight: float | None = None,
    ) -> Tensor:
        """Training loss for one window."""

    @abstractmethod
    def sample(
        self,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        n_samples: int = ...,
        control_repr: Tensor | None = None,
        state_mask: Tensor | None = None,
        apply_zero_atom: bool = True,
    ) -> Tensor:
        """Draw states ``[S, ..., T, D]``. The draw order is part of the contract."""


class ConditionalHead(RouterHead):
    """A head that ``MFM.sample_disentangled`` drives step by step.

    ``sample_disentangled`` draws noise in chunks with ``_draw_noise``, mixes it
    with ``_mix_noise`` and integrates with ``_transport``. It does not call
    ``sample``, so these three set the seeded draw order for conditional draws.
    """

    @staticmethod
    @abstractmethod
    def _draw_noise(shape: tuple[int, ...], *, device, dtype) -> tuple[Tensor, Tensor]:
        """The random part of the base noise: ``(white, latents)``."""

    @abstractmethod
    def _mix_noise(self, white: Tensor, z: Tensor, weekly_repr: Tensor) -> tuple[Tensor, Tensor]:
        """The deterministic part of the base noise: ``(x0, trajectory_latent)``."""

    @abstractmethod
    def _transport(
        self,
        x: Tensor,
        trajectory_latent: Tensor,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor | None,
        state_mask: Tensor | None,
        *,
        apply_zero_atom: bool,
    ) -> Tensor:
        """Integrate base draws to samples."""


#: Required members of each contract, in the order they are reported.
REQUIRED_MEMBERS: dict[type, tuple[str, ...]] = {
    Encoder: ("forward_timeline", "forward_with_controls", "assert_batch_alignment"),
    RouterHead: (
        "loss",
        "sample",
        "state_dim",
        "zero_atom_gate_slices",
        "n_fixed",
        "n_demand",
        "n_controls",
    ),
    ConditionalHead: (
        "loss",
        "sample",
        "state_dim",
        "zero_atom_gate_slices",
        "n_fixed",
        "n_demand",
        "n_controls",
        "_draw_noise",
        "_mix_noise",
        "_transport",
    ),
}
_CONTRACTS = tuple(REQUIRED_MEMBERS)


def missing_members(cls: type, contract: type) -> list[str]:
    """Required members that ``cls`` does not define itself.

    A member counts when a class between ``cls`` and the contract classes defines
    it or annotates it. Inheriting the contract's own abstract stub or annotation
    does not count.
    """
    provided: set[str] = set()
    for klass in cls.__mro__:
        if klass in _CONTRACTS or klass is object:
            continue
        provided |= vars(klass).keys() | inspect.get_annotations(klass).keys()
    return [name for name in REQUIRED_MEMBERS[contract] if name not in provided]


def check_component_class(cls: object, contract: type, *, label: str) -> None:
    """Refuse ``cls`` unless it subclasses ``contract`` and defines every member."""
    if not (isinstance(cls, type) and issubclass(cls, contract)):
        raise TypeError(f"{label} must subclass {contract.__name__}")
    missing = missing_members(cls, contract)
    if missing:
        raise TypeError(f"{label} is missing required members of {contract.__name__}: {missing}")
