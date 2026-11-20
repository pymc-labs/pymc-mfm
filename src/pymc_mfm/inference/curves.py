"""Response curves: each media covariate's contribution as its level in the forecast scales."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_mfm.inference.dataframe import _PredictionSnapshot
    from pymc_mfm.inference.result import DecompositionResult
    from pymc_mfm.model.model import MFM


@dataclass(frozen=True, slots=True)
class ResponseCurves:
    """Each media covariate's contribution over the first forecast weeks, by level.

    ``draws[k, l]`` are the ``n_samples`` draws of media covariate ``channels[k]``'s
    contribution summed over ``dates``, the first ``weeks`` forecast weeks, in ``units``,
    with that covariate in those weeks at ``levels[l]`` (sorted multipliers) times its
    observed values. Each run decomposes the history and a plan of just those weeks,
    every other input as given to the last complete ``inference(history, future, ...)``.
    ``predict`` does not change that base run. ``media`` is each covariate's observed
    total over those weeks. Each level is a separate sample: draws are not paired across
    levels.
    """

    channels: tuple[str, ...]
    levels: np.ndarray
    media: np.ndarray
    draws: np.ndarray
    weeks: int
    dates: np.ndarray
    units: str
    n_samples: int

    def to_frame(self, interval: float = 0.9) -> pd.DataFrame:
        """One row per media covariate and level: the mean and central ``interval`` of its draws.

        Columns: ``channel``, ``level``, ``media`` (the level times the observed total),
        ``mean``, ``low`` and ``high``.
        """
        # The charts summarize with the same function, so they draw exactly these numbers.
        from pymc_mfm.plotting.charts._data import summarize

        mean, low, high = summarize(self.draws, interval, axis=2)
        return pd.DataFrame(
            {
                "channel": np.repeat(self.channels, len(self.levels)),
                "level": np.tile(self.levels, len(self.channels)),
                "media": (self.media[:, None] * self.levels).ravel(),
                "mean": mean.ravel(),
                "low": low.ravel(),
                "high": high.ravel(),
            }
        )


def response_curves(
    model: MFM,
    *,
    levels: Iterable[float] | None = None,
    channels: Iterable[str] | None = None,
    weeks: int | None = None,
    n_samples: int | None = None,
    random_seed: int | None = None,
) -> ResponseCurves:
    """Curves of ``model``'s last complete inference; see ``MFM.response_curves``."""
    from pymc_mfm.inference.api import decompose
    from pymc_mfm.model.flow import _sampling_generator

    prediction = model._prediction_snapshot
    if prediction is None:
        raise RuntimeError(
            "call inference(history, future, ...) successfully before response_curves()"
        )
    history = prediction.fitted_context
    start = len(history.dates)
    levels = _levels(levels)
    names = select_channels(channels, history.channel_columns)
    weeks = _weeks(weeks, len(prediction.time) - start)
    n_samples = _sample_size(n_samples, prediction)
    # One generator serves every run, so each run continues the stream of the one before.
    generator = _sampling_generator(random_seed, next(model.parameters()).device)
    stop = start + weeks
    observations = {
        **prediction.observations,
        "spend": prediction.observations["spend"][:stop],
        "controls": prediction.observations["controls"][:stop],
    }
    graph = prediction.query.graphs[:, None, :] if prediction.mode == "fixed" else None

    def sample(spend: np.ndarray) -> DecompositionResult:
        return decompose(
            model,
            {**observations, "spend": spend},
            n_samples=n_samples,
            window="query",
            mode=prediction.mode,
            graph=graph,
            router_samples=prediction.router_samples,
            channel_names=prediction.query.channel_names,
            control_names=prediction.query.control_names,
            time=prediction.time[:stop],
            units=prediction.units,
            provenance=prediction.annotations,
            _generator=generator,
        )

    actual = observations["spend"]
    columns = [history.channel_columns.index(name) for name in names]
    # Level 1 is one run on the unchanged inputs, shared by every channel.
    reference = sample(actual) if 1.0 in levels else None
    draws = np.empty((len(names), len(levels), n_samples))
    for row, (name, column) in enumerate(zip(names, columns, strict=True)):
        # Scaling a zero media covariate changes no input: its curve is the unchanged run.
        idle = not actual[start:, column].any()
        if idle and reference is None:
            reference = sample(actual)
        for index, level in enumerate(levels):
            if level == 1.0 or idle:
                result = reference
            else:
                scenario = actual.copy()
                scenario[start:, column] *= level
                result = sample(scenario)
            component = result.component_names.index(f"channel:{name}")
            draws[row, index] = result.components[:, :, component].sum(axis=1)
    media = actual[start:, columns].sum(axis=0)
    for array in (levels, media, draws):
        array.setflags(write=False)
    return ResponseCurves(
        channels=names,
        levels=levels,
        media=media,
        draws=draws,
        weeks=weeks,
        dates=prediction.time[start:stop],
        units=prediction.units,
        n_samples=n_samples,
    )


def select_channels(channels: Iterable[str] | None, available: tuple[str, ...]) -> tuple[str, ...]:
    """``channels``, checked against ``available``; all of ``available`` when ``None``."""
    if channels is None:
        return available
    if isinstance(channels, (str, bytes)):
        raise ValueError("channels must be a sequence of channel names, not a string")
    try:
        names = tuple(channels)
    except TypeError:
        raise ValueError("channels must be a sequence of channel names") from None
    if not names:
        raise ValueError(f"channels must name at least one of {list(available)}")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; choose from {list(available)}")
    if len(set(names)) != len(names):
        raise ValueError(f"channels must not repeat a name, got {list(names)}")
    return names


def _levels(levels: Iterable[float] | None) -> np.ndarray:
    """Sorted distinct multipliers, at least two; 0, 0.25, ..., 2 when ``None``."""
    if levels is None:
        return np.linspace(0.0, 2.0, 9)
    values = np.asarray(levels)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError("levels must be a nonempty sequence of multipliers, e.g. [0, 1, 2]")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"levels must be finite and nonnegative, got {values.tolist()}")
    values = np.unique(values)
    if values.size < 2:
        raise ValueError(
            f"a curve needs at least two distinct levels, got {values.tolist()}; "
            "use inference(history, future, ...) for one complete plan"
        )
    return values


def _weeks(weeks: int | None, available: int) -> int:
    if weeks is None:
        return available
    if isinstance(weeks, bool) or not isinstance(weeks, Integral) or not 1 <= weeks <= available:
        raise ValueError(
            f"weeks must be a whole number from 1 to {available}, the forecast weeks of the "
            f"latest inference; got {weeks!r}"
        )
    return int(weeks)


def _sample_size(n_samples: int | None, prediction: _PredictionSnapshot) -> int:
    if n_samples is None:
        return prediction.n_samples
    from pymc_mfm.model.flow import _sample_count

    count = _sample_count(n_samples)
    if prediction.mode == "fixed" and count != prediction.n_samples:
        raise ValueError(
            f"response curves reuse the {prediction.n_samples} graphs of the latest "
            f"mode='fixed' inference, one per draw: n_samples must be "
            f"{prediction.n_samples} (or None), got {count}"
        )
    return count
