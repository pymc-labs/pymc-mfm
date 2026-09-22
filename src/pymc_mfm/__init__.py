"""PyMC Marketing Foundation Models."""

from importlib.metadata import version

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
