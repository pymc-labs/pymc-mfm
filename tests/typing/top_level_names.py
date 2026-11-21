"""Static check, not a test: pyright reads this file and never runs it.

The six deprecated top-level names must keep their real types for editors. Without the
`TYPE_CHECKING` declarations in `pymc_mktfm/__init__.py`, each name is `Any` and every
`assert_type` below fails. Run: `pyright tests/typing`. Remove this file with the shims.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, assert_type

import torch

import pymc_mktfm
from pymc_mktfm.inference.result import DecompositionResult
from pymc_mktfm.model import MFM, ModelConfig

T = TypeVar("T")


def returns(function: Callable[..., T]) -> T:
    """The return type of ``function``; ``Any`` stays ``Any`` and fails ``assert_type``."""
    raise NotImplementedError


def model_arg(function: Callable[[T, str], object]) -> T:
    """The type of the first parameter of ``function``."""
    raise NotImplementedError


assert_type(pymc_mktfm.ModelConfig, type[ModelConfig])
assert_type(pymc_mktfm.DecompositionResult, type[DecompositionResult])
assert_type(returns(pymc_mktfm.decompose), DecompositionResult)
assert_type(returns(pymc_mktfm.save_pretrained), Path)
assert_type(returns(pymc_mktfm.observation_batch), dict[str, torch.Tensor])
assert_type(model_arg(pymc_mktfm.push_to_hub), MFM)  # no return annotation to check
