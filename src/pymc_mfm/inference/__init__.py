"""Checkpoint loading and observation-based inference interfaces.

Names load on first use (PEP 562). ``DecompositionResult`` needs no PyTorch; the
other names import modules that do. Importing this package imports none of them.
"""

from __future__ import annotations

from pkgutil import iter_modules
from typing import TYPE_CHECKING

from pymc_mfm._lazy import import_local

if TYPE_CHECKING:
    from .api import decompose, observation_batch
    from .checkpoint import save_pretrained
    from .hub import from_pretrained, list_profiles, push_to_hub
    from .result import DecompositionResult
    from .warmup import WarmUp, warm_up_observations, warm_up_sampling

#: Public name -> submodule that defines it.
_EXPORTS = {
    "DecompositionResult": "result",
    "decompose": "api",
    "observation_batch": "api",
    "save_pretrained": "checkpoint",
    "from_pretrained": "hub",
    "list_profiles": "hub",
    "push_to_hub": "hub",
    "WarmUp": "warmup",
    "warm_up_observations": "warmup",
    "warm_up_sampling": "warmup",
}

#: Submodules that load on attribute access, as the eager package left them bound.
_SUBMODULES = frozenset(info.name for info in iter_modules(__path__))

__all__ = [
    "DecompositionResult",
    "decompose",
    "from_pretrained",
    "list_profiles",
    "observation_batch",
    "push_to_hub",
    "save_pretrained",
    "WarmUp",
    "warm_up_observations",
    "warm_up_sampling",
]


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        # A submodule the eager package used to import (``package.calibration``).
        if name in _SUBMODULES:
            return import_local(f"{__name__}.{name}", None, f"{__name__}.{name}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_local(f"{__name__}.{submodule}", name, f"{__name__}.{name}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})
