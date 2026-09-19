"""PyMC Marketing Foundation Models."""

from importlib.metadata import version

from .inference.checkpoint import load_model, save_model
from .model import MFM, ModelConfig

__version__ = version("pymc-mfm")

__all__ = ["MFM", "ModelConfig", "load_model", "save_model"]
