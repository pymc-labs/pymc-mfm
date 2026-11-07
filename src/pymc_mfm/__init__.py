"""PyMC Marketing Foundation Models.

The base install is a light HTTP client, :mod:`pymc_mfm.client`. The model, inference
and training code needs PyTorch: install it with ``pip install "pymc-mfm[local]"``.
Those names load on first use, so ``import pymc_mfm`` never imports PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

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

__version__ = version("pymc-mfm")

#: Public name -> module that defines it. Every module here imports PyTorch.
_LOCAL_EXPORTS = {
    "MFM": "pymc_mfm.model",
    "ModelConfig": "pymc_mfm.model",
    "DecompositionResult": "pymc_mfm.inference",
    "decompose": "pymc_mfm.inference",
    "observation_batch": "pymc_mfm.inference",
    "save_pretrained": "pymc_mfm.inference",
    "from_pretrained": "pymc_mfm.inference",
    "push_to_hub": "pymc_mfm.inference",
    "list_profiles": "pymc_mfm.inference",
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
]

_LOCAL_DEPENDENCIES = ("torch", "safetensors")


def __getattr__(name: str):
    if name == "client":  # torch-free; `import pymc_mfm` then `pymc_mfm.client` works
        return import_module(f"{__name__}.client")
    module_name = _LOCAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(module_name), name)
    except ModuleNotFoundError as error:
        # Only a missing PyTorch dependency becomes the install hint. Any other
        # missing module is a real bug and keeps its own error.
        if (error.name or "").partition(".")[0] not in _LOCAL_DEPENDENCIES:
            raise
        raise ImportError(
            f"pymc_mfm.{name} runs the model in this process and needs PyTorch. "
            'Install it with: pip install "pymc-mfm[local]". '
            "To call a hosted service without PyTorch, use pymc_mfm.client."
        ) from error
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LOCAL_EXPORTS})
