"""Canonical training: corpus consumption, optimization, evaluation, checkpoints."""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .evaluation import MONITOR_METRIC, evaluate_rows, flatten_evaluation
from .observer import TrainingObserver
from .stream import CorpusStream
from .trainer import TrainingError, TrainingResult, train

__all__ = [
    "MONITOR_METRIC",
    "CorpusStream",
    "TrainingConfig",
    "TrainingError",
    "TrainingObserver",
    "TrainingResult",
    "evaluate_rows",
    "flatten_evaluation",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
