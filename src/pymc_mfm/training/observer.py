"""Lifecycle seam through which a run's structured records leave the trainer.

The trainer always writes its own local evidence (``run.json``,
``metrics.jsonl``, ``evaluations.jsonl``, checkpoints). Observers receive the
same records, in order, for delivery elsewhere; MLflow tracking (#16) will be
one such observer. Passing no observer is the explicit opt-out: nothing here
imports or requires a tracking service, and training semantics do not depend
on which observers are attached.

An observer that raises stops the run: a delivery layer that wants fail-safe
training must catch its own errors and report degraded delivery in its own
records, never by mutating the trainer's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TrainingObserver:
    """No-op hooks; override any subset."""

    def on_run_start(self, run: Mapping[str, Any]) -> None:
        """The run record after ``run.json`` is first written (status ``running``)."""

    def on_step(self, record: Mapping[str, Any]) -> None:
        """One ``metrics.jsonl`` line: every active loss term, lr, gradient health."""

    def on_evaluation(self, record: Mapping[str, Any]) -> None:
        """One ``evaluations.jsonl`` line: all legs, heads, windows and selection."""

    def on_checkpoint(self, record: Mapping[str, Any]) -> None:
        """A checkpoint file written: kind, step, path, SHA-256, selection."""

    def on_run_end(self, run: Mapping[str, Any]) -> None:
        """The final run record (status ``finished``, ``failed`` or ``interrupted``)."""
