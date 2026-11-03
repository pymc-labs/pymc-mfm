"""Registries of model components and the ``components`` mapping that picks them.

``components`` lives beside :class:`~pymc_mfm.model.config.ModelConfig`, never
inside it, so config dictionaries, config hashes and release bundles do not change.
A component is one of the registered classes below. Built-in classes are release
components. Every class registered through :func:`register_encoder` or
:func:`register_head` is research: a model that uses one has ``is_research`` set
and cannot be exported or loaded as a release bundle.

One registry, ``HEADS``, serves both heads. The router slot needs a
:class:`RouterHead`. The conditional slot needs a :class:`ConditionalHead`, which
``MFM.sample_disentangled`` drives step by step.
"""

from __future__ import annotations

from collections.abc import Mapping

from .components import (
    COMPONENT_SLOTS,
    ConditionalHead,
    Encoder,
    RouterHead,
    check_component_class,
)
from .encoder import BiAttentionEncoder
from .flow import JointFlowHead

DEFAULT_COMPONENTS: dict[str, str] = {
    "encoder": "bi_attention",
    "head": "joint_flow",
    "conditional_head": "joint_flow",
}

ENCODERS: dict[str, type[Encoder]] = {}
HEADS: dict[str, type[RouterHead]] = {}
#: Names of the release components. Only this module adds to it.
_RELEASE: dict[str, set[str]] = {"encoder": set(), "head": set(), "conditional_head": set()}


def _add(registry: dict, name: str, cls: type, contract: type, *, kind: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} name must be a nonempty string")
    if name in registry:
        raise ValueError(f"{kind} {name!r} is already registered")
    check_component_class(cls, contract, label=f"{kind} {name!r}")
    registry[name] = cls


def register_encoder(name: str, cls: type[Encoder]) -> type[Encoder]:
    """Register a research encoder class under ``name``; refuse a broken contract."""
    _add(ENCODERS, name, cls, Encoder, kind="encoder")
    return cls


def register_head(name: str, cls: type[RouterHead]) -> type[RouterHead]:
    """Register a research head class under ``name``; refuse a broken contract.

    A class that is only a :class:`RouterHead` can fill the router slot. A
    :class:`ConditionalHead` can fill both slots, and is checked against that
    stricter contract.
    """
    contract = (
        ConditionalHead
        if isinstance(cls, type) and issubclass(cls, ConditionalHead)
        else RouterHead
    )
    _add(HEADS, name, cls, contract, kind="head")
    return cls


ENCODERS["bi_attention"] = BiAttentionEncoder
HEADS["joint_flow"] = JointFlowHead
_RELEASE["encoder"].add("bi_attention")
_RELEASE["head"].add("joint_flow")
_RELEASE["conditional_head"].add("joint_flow")


def resolve_components(components: Mapping[str, str] | None = None) -> dict[str, str]:
    """Fill defaults and refuse unknown slots, unregistered names and unfit classes."""
    chosen = dict(DEFAULT_COMPONENTS)
    if components is not None:
        unknown = sorted(set(components) - set(COMPONENT_SLOTS))
        if unknown:
            raise ValueError(f"unknown component slots {unknown}; expected {list(COMPONENT_SLOTS)}")
        chosen.update(components)
    for slot, registry in (("encoder", ENCODERS), ("head", HEADS), ("conditional_head", HEADS)):
        name = chosen[slot]
        if name not in registry:
            raise ValueError(f"{slot} {name!r} is not registered; known: {sorted(registry)}")
    check_component_class(
        HEADS[chosen["conditional_head"]], ConditionalHead, label="conditional_head"
    )
    return {slot: chosen[slot] for slot in COMPONENT_SLOTS}


def is_research(components: Mapping[str, str] | None = None) -> bool:
    """True when any resolved component is not a release component."""
    resolved = resolve_components(components)
    return any(name not in _RELEASE[slot] for slot, name in resolved.items())


def build_encoder(name: str, config) -> Encoder:
    return ENCODERS[name](config)


def build_head(name: str, config, *, graph_router: bool) -> RouterHead:
    return HEADS[name](config, graph_router=graph_router)
