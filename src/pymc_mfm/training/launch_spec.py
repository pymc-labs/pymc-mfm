"""Launch file of ``python -m pymc_mfm.training.launch``: validation and command lines.

The launch file is a JSON list of runs::

    [{"name": "lr-high", "preset": "mini", "training": {"lr": 0.004},
      "corpus": ["mini/shard_550000.npz"], "device": "cuda", "timeout_seconds": 1800,
      "score": {"mode": "eval-set", "corpus": ["eval/shard_551000.npz"],
                "seed": 7, "draws": 16}}]

``preset`` (default ``canonical``), ``device`` (default ``auto``) and ``score`` are
optional; ``training`` holds :class:`~pymc_mfm.training.config.TrainingConfig` fields.
``score.corpus`` defaults to the run's corpus. ``score.seed`` and ``score.draws`` are
for ``eval-set`` mode only; ``score`` keeps its canonical 128 draws when they are
absent, which is slow on CPU, so local smokes should set a small ``draws``.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pymc_mfm.training.config import PRESETS, TrainingConfig
from pymc_mfm.training.resume import LAST_CHECKPOINT
from pymc_mfm.training.score import MODES

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RUN_KEYS = {"name", "preset", "training", "corpus", "device", "timeout_seconds", "score"}
_SCORE_KEYS = {"mode", "corpus", "seed", "draws"}
_TRAINING_FIELDS = {field.name for field in fields(TrainingConfig)}


class LaunchFileError(ValueError):
    """The launch file is invalid; nothing was started."""


@dataclass(frozen=True)
class RunSpec:
    name: str
    preset: str
    training: dict[str, Any]
    corpus: tuple[str, ...]
    device: str
    timeout_seconds: float
    #: ``None`` when the run is not scored; otherwise ``{"mode": ..., "corpus": [...]}``.
    score: dict[str, Any] | None


def parse_launch_file(path: Path) -> list[RunSpec]:
    """The runs of a launch file; raises :class:`LaunchFileError` naming the first problem."""
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchFileError(f"{path}: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise LaunchFileError(f"{path}: expected a non-empty JSON list of runs")
    specs = [_parse_run(index, entry) for index, entry in enumerate(entries)]
    names = [spec.name for spec in specs]
    for name in sorted(set(names)):
        if names.count(name) > 1:
            raise LaunchFileError(f"run name {name!r} is used more than once")
    return specs


def _parse_run(index: int, entry: Any) -> RunSpec:
    label = f"run {index}"
    if not isinstance(entry, dict):
        raise LaunchFileError(f"{label}: expected an object")
    if unknown := entry.keys() - _RUN_KEYS:
        raise LaunchFileError(f"{label}: unknown keys {sorted(unknown)}")
    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise LaunchFileError(f"{label}: name must match {_NAME.pattern}, got {name!r}")
    label = f"run {name!r}"
    preset = entry.get("preset", "canonical")
    if preset not in PRESETS:
        raise LaunchFileError(f"{label}: unknown preset {preset!r}; known: {sorted(PRESETS)}")
    training = entry.get("training", {})
    if not isinstance(training, dict):
        raise LaunchFileError(f"{label}: training must be an object of TrainingConfig fields")
    if unknown := training.keys() - _TRAINING_FIELDS:
        raise LaunchFileError(f"{label}: unknown training fields {sorted(unknown)}")
    try:
        resolved_training(preset, training)
    except (TypeError, ValueError) as exc:
        raise LaunchFileError(f"{label}: {exc}") from exc
    corpus = _existing_files(label, entry.get("corpus"))
    timeout = entry.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise LaunchFileError(f"{label}: timeout_seconds must be a positive number")
    score = _parse_score(label, entry.get("score"), corpus)
    device = entry.get("device", "auto")
    if not isinstance(device, str):
        raise LaunchFileError(f"{label}: device must be a string")
    return RunSpec(name, preset, training, corpus, device, float(timeout), score)


def resolved_training(preset: str, overrides: dict[str, Any]) -> TrainingConfig:
    """The config the training CLI builds from ``preset`` and the explicit flags.

    Without ``milestone_steps`` the CLI keeps the preset's milestones within ``steps``.
    Raises ``ValueError`` for a value the config refuses.
    """
    values = {f.name: getattr(PRESETS[preset], f.name) for f in fields(TrainingConfig)}
    values.update(overrides)
    if "milestone_steps" not in overrides and type(values["steps"]) is int:
        values["milestone_steps"] = tuple(
            step for step in PRESETS[preset].milestone_steps if step <= values["steps"]
        )
    values["milestone_steps"] = tuple(values["milestone_steps"])
    return TrainingConfig(**values)


def _parse_score(label: str, score: Any, run_corpus: tuple[str, ...]) -> dict[str, Any] | None:
    """The validated ``score`` block, with its corpus defaulted to the run's."""
    if score is None:
        return None
    if not isinstance(score, dict) or score.get("mode") not in MODES:
        raise LaunchFileError(f"{label}: score.mode must be one of {MODES}")
    if unknown := score.keys() - _SCORE_KEYS:
        raise LaunchFileError(f"{label}: unknown score keys {sorted(unknown)}")
    for key in ("seed", "draws"):
        if key in score and score["mode"] != "eval-set":
            raise LaunchFileError(f"{label}: score.{key} is for eval-set mode only")
        if key in score and (type(score[key]) is not int or score[key] < 1):
            raise LaunchFileError(f"{label}: score.{key} must be a positive integer")
    corpus = _existing_files(label, score.get("corpus", list(run_corpus)))
    return {**score, "corpus": corpus}


def _existing_files(label: str, paths: Any) -> tuple[str, ...]:
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise LaunchFileError(f"{label}: corpus must be a non-empty list of paths")
    for path in paths:
        if not Path(path).is_file():
            raise LaunchFileError(f"{label}: corpus file {path} does not exist")
    return tuple(str(Path(path).resolve()) for path in paths)


def train_command(spec: RunSpec, run_dir: Path, *, resume: bool = False) -> list[str]:
    """The ``python -m pymc_mfm.training`` command line of a run.

    Tracking and diagnostic reports are off: a launched run is for its scores, and both
    would slow it. ``resume`` continues the run from its own ``checkpoint_last.pt``.
    """
    command = [
        sys.executable,
        "-m",
        "pymc_mfm.training",
        "--preset",
        spec.preset,
        "--corpus",
        *spec.corpus,
        "--out-dir",
        str(run_dir),
        "--device",
        spec.device,
        "--run-name",
        spec.name,
        "--no-tracking",
        "--no-diagnostics",
    ]
    for key, value in spec.training.items():
        if key == "milestone_steps":
            value = ",".join(map(str, value))
        command += ["--" + key.replace("_", "-"), str(value)]
    if resume:
        command += ["--resume-from", str(run_dir / LAST_CHECKPOINT)]
    return command


def score_command(spec: RunSpec, run_dir: Path, results: Path) -> list[str]:
    """The ``python -m pymc_mfm.training score`` command line of a run's terminal checkpoint."""
    command = [
        sys.executable,
        "-m",
        "pymc_mfm.training",
        "score",
        "--checkpoint",
        str(run_dir / LAST_CHECKPOINT),
        "--corpus",
        *spec.score["corpus"],
        "--replay" if spec.score["mode"] == "replay" else "--eval-set",
        "--results",
        str(results),
        "--device",
        spec.device,
    ]
    for key in ("seed", "draws"):
        if key in spec.score:
            command += [f"--{key}", str(spec.score[key])]
    return command
