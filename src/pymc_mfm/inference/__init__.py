"""Checkpoint loading and observation-based inference interfaces.

Names load on first use (PEP 562), so importing this package imports no submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymc_mfm._lazy import lazy_exports

if TYPE_CHECKING:
    from .api import decompose, observation_batch
    from .checkpoint import save_pretrained
    from .curves import ResponseCurves
    from .hub import from_pretrained, list_profiles, push_to_hub
    from .result import DecompositionResult
    from .warmup import WarmUp, warm_up_observations, warm_up_sampling

#: Public name -> module that defines it.
_EXPORTS = {
    "DecompositionResult": "pymc_mfm.inference.result",
    "ResponseCurves": "pymc_mfm.inference.curves",
    "decompose": "pymc_mfm.inference.api",
    "from_pretrained": "pymc_mfm.inference.hub",
    "list_profiles": "pymc_mfm.inference.hub",
    "observation_batch": "pymc_mfm.inference.api",
    "push_to_hub": "pymc_mfm.inference.hub",
    "save_pretrained": "pymc_mfm.inference.checkpoint",
    "WarmUp": "pymc_mfm.inference.warmup",
    "warm_up_observations": "pymc_mfm.inference.warmup",
    "warm_up_sampling": "pymc_mfm.inference.warmup",
}

__all__ = [
    "DecompositionResult",
    "ResponseCurves",
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

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
