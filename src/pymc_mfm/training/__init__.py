"""Canonical training: corpus consumption, optimization, evaluation, checkpoints."""

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
