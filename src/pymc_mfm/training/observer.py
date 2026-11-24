"""Lifecycle seam through which a run's structured records leave the trainer.

The trainer always writes its own local evidence (``run.json``,
``metrics.jsonl``, ``evaluations.jsonl``, checkpoints). Observers receive the
same records, in order, for delivery elsewhere. Optional MLflow delivery is
enabled by default; ``TrackingConfig(enabled=False)`` is the explicit opt-out.
No observer API requires MLflow and numerical configuration is independent
of delivery settings.

An observer that raises stops the run: a delivery layer that wants fail-safe
training must catch its own errors and report degraded delivery in its own
records, never by mutating the trainer's.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TrainingObserver:
    """No-op hooks; override any subset."""

    def on_run_start(self, run: Mapping[str, Any]) -> None:
        """The run record after ``run.json`` is first written (status ``running``)."""

    def on_run_config(self, run: Mapping[str, Any]) -> None:
        """Resolved configuration after model/stream setup, before updates."""

    def on_step(self, record: Mapping[str, Any]) -> None:
        """One ``metrics.jsonl`` line: every active loss term, lr, gradient health."""

    def on_evaluation(self, record: Mapping[str, Any]) -> None:
        """One ``evaluations.jsonl`` line: all legs, heads, windows and selection."""

    def on_checkpoint(self, record: Mapping[str, Any]) -> None:
        """A checkpoint file written: kind, step, path, SHA-256, selection."""

    def on_diagnostic(self, manifest: Mapping[str, Any], path: str | Path) -> None:
        """A saved report manifest and directory; never requests new inference."""

    def on_run_end(self, run: Mapping[str, Any]) -> None:
        """The final run record (status ``finished``, ``failed`` or ``interrupted``)."""
