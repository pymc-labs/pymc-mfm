"""PyMC Marketing Foundation Models.

The base install is a light HTTP client, :mod:`pymc_mktfm.client`. The model, inference
and training code needs PyTorch: install it with ``pip install "pymc-mktfm[local]"``.
Those names load on first use, so ``import pymc_mktfm`` never imports PyTorch.
"""

from __future__ import annotations

from functools import partial
from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

from ._deprecated import deprecated_names
from ._lazy import lazy_exports

if TYPE_CHECKING:
    # The `X as X` imports keep editor types for the six deprecated names (see
    # `_DEPRECATED_EXPORTS`); ruff reads that form as a re-export. Remove them with the shims.
    from .idata import load_idata, save_idata, validate_idata
    from .inference import from_pretrained, list_profiles
    from .inference.api import decompose as decompose
    from .inference.api import observation_batch as observation_batch
    from .inference.checkpoint import save_pretrained as save_pretrained
    from .inference.hub import push_to_hub as push_to_hub
    from .inference.result import DecompositionResult as DecompositionResult
    from .model import MFM
    from .model import ModelConfig as ModelConfig
    from .plotting.charts import Plots, set_backend

__version__ = version("pymc-mktfm")

#: Public name -> module that defines it. Every module here imports PyTorch.
_LOCAL_EXPORTS = {
    "MFM": "pymc_mktfm.model",
    "from_pretrained": "pymc_mktfm.inference.hub",
    "list_profiles": "pymc_mktfm.inference.hub",
}

#: Names that moved to a subpackage: name -> (module that defines it, text for the warning).
_DEPRECATED_EXPORTS = {
    "ModelConfig": ("pymc_mktfm.model", "use pymc_mktfm.model.ModelConfig"),
    "DecompositionResult": (
        "pymc_mktfm.inference.result",
        "use pymc_mktfm.inference.DecompositionResult",
    ),
    "decompose": ("pymc_mktfm.inference.api", "use pymc_mktfm.inference.decompose"),
    "observation_batch": ("pymc_mktfm.inference.api", "use pymc_mktfm.inference.observation_batch"),
    "save_pretrained": (
        "pymc_mktfm.inference.checkpoint",
        "use pymc_mktfm.inference.save_pretrained",
    ),
    "push_to_hub": ("pymc_mktfm.inference.hub", "use pymc_mktfm.inference.push_to_hub"),
}

#: Public name -> module that defines it. These read a model tree and need no PyTorch.
_LIGHT_EXPORTS = {
    "Plots": "pymc_mktfm.plotting.charts",
    "set_backend": "pymc_mktfm.plotting.charts",
    "save_idata": "pymc_mktfm.idata",
    "load_idata": "pymc_mktfm.idata",
    "validate_idata": "pymc_mktfm.idata",
}

__all__ = [
    "MFM",
    "from_pretrained",
    "list_profiles",
    "Plots",
    "set_backend",
    "save_idata",
    "load_idata",
    "validate_idata",
]

_lazy_getattr, _lazy_dir = lazy_exports(__name__, {**_LOCAL_EXPORTS, **_LIGHT_EXPORTS}, globals())
_load_deprecated, _ = lazy_exports(
    __name__, {name: module for name, (module, _) in _DEPRECATED_EXPORTS.items()}, {}
)
_deprecated_getattr = deprecated_names(
    __name__,
    {
        name: (replacement, partial(_load_deprecated, name))
        for name, (_, replacement) in _DEPRECATED_EXPORTS.items()
    },
    globals(),
    stacklevel=3,
)


def __dir__() -> list[str]:
    # A deprecated name is cached in globals() after its first use; keep it out of dir().
    return [name for name in _lazy_dir() if name not in _DEPRECATED_EXPORTS]


def __getattr__(name: str):
    if name in _DEPRECATED_EXPORTS:
        return _deprecated_getattr(name)
    if name == "client":  # torch-free; `import pymc_mktfm` then `pymc_mktfm.client` works
        return import_module(f"{__name__}.client")
    return _lazy_getattr(name)
