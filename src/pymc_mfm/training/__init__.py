"""Canonical training: corpus consumption, optimization, evaluation, checkpoints.

Names load on first use (PEP 562), so ``pymc_mfm.training.metrics`` and the other
numeric helpers import without PyTorch.
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .archive import archive_identity, evaluate_into_archive
    from .checkpoint import load_checkpoint, save_checkpoint
    from .compatibility import check_recorded_evaluation
    from .config import TrainingConfig
    from .evaluation import (
        METRIC_BOUNDS,
        evaluate_rows,
        flatten_evaluation,
        metric_aggregation,
        selection_metrics,
    )
    from .observer import TrainingObserver
    from .stream import CorpusStream
    from .tracking import MlflowTrainingObserver, TrackingConfig
    from .trainer import TrainingError, TrainingResult, train

#: Public name -> submodule that defines it.
_EXPORTS = {
    "archive_identity": "archive",
    "evaluate_into_archive": "archive",
    "load_checkpoint": "checkpoint",
    "save_checkpoint": "checkpoint",
    "check_recorded_evaluation": "compatibility",
    "TrainingConfig": "config",
    "METRIC_BOUNDS": "evaluation",
    "evaluate_rows": "evaluation",
    "flatten_evaluation": "evaluation",
    "metric_aggregation": "evaluation",
    "selection_metrics": "evaluation",
    "TrainingObserver": "observer",
    "CorpusStream": "stream",
    "MlflowTrainingObserver": "tracking",
    "TrackingConfig": "tracking",
    "TrainingError": "trainer",
    "TrainingResult": "trainer",
    "train": "trainer",
}

#: Submodules that load on attribute access, as the eager package left them bound.
_SUBMODULES = frozenset(info.name for info in iter_modules(__path__))

__all__ = [
    "METRIC_BOUNDS",
    "CorpusStream",
    "MlflowTrainingObserver",
    "TrackingConfig",
    "TrainingConfig",
    "TrainingError",
    "TrainingObserver",
    "TrainingResult",
    "archive_identity",
    "check_recorded_evaluation",
    "evaluate_into_archive",
    "evaluate_rows",
    "flatten_evaluation",
    "load_checkpoint",
    "metric_aggregation",
    "save_checkpoint",
    "selection_metrics",
    "train",
]


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        # A submodule the eager package used to import (``package.calibration``).
        if name in _SUBMODULES:
            return import_module(f"{__name__}.{name}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})
