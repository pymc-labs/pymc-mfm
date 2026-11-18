"""PyMC Marketing Foundation Models.

The base install is a light HTTP client, :mod:`pymc_mfm.client`. The model, inference
and training code needs PyTorch: install it with ``pip install "pymc-mfm[local]"``.
Those names load on first use, so ``import pymc_mfm`` never imports PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from ._lazy import lazy_exports

if TYPE_CHECKING:
    from .inference import (
        DecompositionResult,
        decompose,
        from_pretrained,
        list_profiles,
        observation_batch,
        push_to_hub,
        save_pretrained,
    )
    from .model import MFM, ModelConfig
    from .plotting.charts import Plots, set_backend

__version__ = version("pymc-mfm")

#: Public name -> module that defines it. Every module here imports PyTorch.
_LOCAL_EXPORTS = {
    "MFM": "pymc_mfm.model",
    "ModelConfig": "pymc_mfm.model",
    "DecompositionResult": "pymc_mfm.inference.result",
    "decompose": "pymc_mfm.inference.api",
    "observation_batch": "pymc_mfm.inference.api",
    "save_pretrained": "pymc_mfm.inference.checkpoint",
    "from_pretrained": "pymc_mfm.inference.hub",
    "push_to_hub": "pymc_mfm.inference.hub",
    "list_profiles": "pymc_mfm.inference.hub",
}

#: Public name -> module that defines it. These read a model tree and need no PyTorch.
_LIGHT_EXPORTS = {
    "Plots": "pymc_mfm.plotting.charts",
    "set_backend": "pymc_mfm.plotting.charts",
}

__all__ = [
    "MFM",
    "ModelConfig",
    "DecompositionResult",
    "decompose",
    "observation_batch",
    "save_pretrained",
    "from_pretrained",
    "push_to_hub",
    "list_profiles",
    "Plots",
    "set_backend",
]

_lazy_getattr, __dir__ = lazy_exports(__name__, {**_LOCAL_EXPORTS, **_LIGHT_EXPORTS}, globals())


def __getattr__(name: str):
    if name == "client":  # torch-free; `import pymc_mfm` then `pymc_mfm.client` works
        return import_module(f"{__name__}.client")
    return _lazy_getattr(name)
