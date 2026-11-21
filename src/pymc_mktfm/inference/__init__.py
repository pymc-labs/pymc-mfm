"""Checkpoint loading and observation-based inference interfaces.

Names load on first use (PEP 562), so importing this package imports no submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymc_mktfm._lazy import lazy_exports

if TYPE_CHECKING:
    from .api import decompose, observation_batch
    from .checkpoint import save_pretrained
    from .curves import ResponseCurves
    from .hub import from_pretrained, list_profiles, push_to_hub
    from .result import DecompositionResult
    from .warmup import WarmUp, warm_up_observations, warm_up_sampling

#: Public name -> module that defines it.
_EXPORTS = {
    "DecompositionResult": "pymc_mktfm.inference.result",
    "ResponseCurves": "pymc_mktfm.inference.curves",
    "decompose": "pymc_mktfm.inference.api",
    "from_pretrained": "pymc_mktfm.inference.hub",
    "list_profiles": "pymc_mktfm.inference.hub",
    "observation_batch": "pymc_mktfm.inference.api",
    "push_to_hub": "pymc_mktfm.inference.hub",
    "save_pretrained": "pymc_mktfm.inference.checkpoint",
    "WarmUp": "pymc_mktfm.inference.warmup",
    "warm_up_observations": "pymc_mktfm.inference.warmup",
    "warm_up_sampling": "pymc_mktfm.inference.warmup",
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
