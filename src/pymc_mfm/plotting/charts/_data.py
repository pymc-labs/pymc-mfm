"""The model state charts draw: history, the latest prediction and named window draws.

:class:`ModelSource` is the one place the charts read a model. It never samples,
except through ``MFM._context_result``, which samples the context marginal once per
prediction and caches it for ``build_idata`` too.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pymc_mfm.inference.curves import ResponseCurves
    from pymc_mfm.inference.dataframe import _FittedContext, _PredictionSnapshot
    from pymc_mfm.inference.result import DecompositionResult
    from pymc_mfm.model.model import MFM

WINDOWS = ("context", "query")


@dataclass(frozen=True, slots=True)
class Window:
    """Draws of one window, active components only, with what was observed there.

    ``draws`` is ``[sample, week, component]`` in outcome units; ``components`` are
    display names (``"tv"``, not ``"channel:tv"``) and ``families`` their component
    families. ``outcome`` is the observed outcome by week (NaN where unobserved), or
    ``None`` when none was supplied. ``spend`` is raw weekly spend, one column per
    name in ``channels``.
    """

    name: str
    dates: np.ndarray
    components: tuple[str, ...]
    families: tuple[str, ...]
    draws: np.ndarray
    outcome: np.ndarray | None
    channels: tuple[str, ...]
    spend: np.ndarray
    target: str
    units: str

    @property
    def n_samples(self) -> int:
        return self.draws.shape[0]

    @property
    def expected(self) -> np.ndarray:
        """Expected outcome draws ``[sample, week]``: the sum of every component."""
        return self.draws.sum(axis=2)

    def contribution(self, name: str) -> np.ndarray:
        """Draws ``[sample, week]`` of the component named ``name``."""
        return self.draws[:, :, self.components.index(name)]


def display_names(result: DecompositionResult) -> tuple[np.ndarray, tuple[str, ...]]:
    """Indices and display names of the active components of ``result``.

    Channels and controls drop their family prefix. A name that would then appear
    twice (a channel named ``baseline``, say) keeps its full component name, until
    every name is unique; full component names are unique, so this ends.
    """
    channels, controls = iter(result.channel_names), iter(result.control_names)
    plain = []
    for name, family in zip(result.component_names, result.component_families, strict=True):
        if family == "channel":
            plain.append(next(channels))
        elif family == "control":
            plain.append(next(controls))
        else:
            plain.append(name)
    active = np.flatnonzero(result.component_mask)
    names = {index: plain[index] for index in active}
    while True:
        counts = Counter(names.values())
        repeated = [index for index, name in names.items() if counts[name] > 1]
        if not repeated:
            return active, tuple(names.values())
        for index in repeated:
            names[index] = result.component_names[index]


def interval_quantiles(interval: float) -> tuple[float, float]:
    """Lower and upper quantiles of a central ``interval`` (a fraction)."""
    if isinstance(interval, bool) or not 0 < interval < 1:
        raise ValueError(f"interval must be a fraction between 0 and 1, got {interval!r}")
    return (1 - interval) / 2, (1 + interval) / 2


def summarize(
    values: np.ndarray, interval: float, axis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and central ``interval`` bounds of ``values`` over ``axis`` (the draws)."""
    low, high = interval_quantiles(interval)
    bounds = np.quantile(values, [low, high], axis=axis)
    return values.mean(axis=axis), bounds[0], bounds[1]


def window_weeks(window: Window) -> str:
    """``"24 history weeks"`` or ``"1 forecast week"``: the weeks ``window`` spans."""
    count = len(window.dates)
    kind = "history" if window.name == "context" else "forecast"
    return f"{count} {kind} week{'' if count == 1 else 's'}"


def checked_names(names: Sequence[str], available: Sequence[str], role: str) -> tuple[str, ...]:
    """``names`` as a tuple, each one of ``available`` once; ``role`` names them in errors."""
    if isinstance(names, str):
        raise ValueError(f"{role} must be a sequence of names, not a string")
    names = tuple(names)
    if not names:
        raise ValueError(f"{role} must name at least one of {list(available)}")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown {role} {unknown}; available: {list(available)}")
    repeated = [name for name, count in Counter(names).items() if count > 1]
    if repeated:
        raise ValueError(f"{role} repeat {repeated}; name each once")
    return names


def truth_by_week(truth: pd.DataFrame, window: Window) -> dict[str, np.ndarray]:
    """Each column of ``truth`` on the weeks of ``window``: NaN where it has no value.

    ``truth`` holds known contributions, from a simulation say: a DataFrame indexed
    by date with one column per component display name. Its other weeks are ignored.
    """
    if not isinstance(truth, pd.DataFrame) or not isinstance(truth.index, pd.DatetimeIndex):
        raise ValueError("truth must be a pandas DataFrame indexed by date (a DatetimeIndex)")
    if not truth.columns.is_unique:
        raise ValueError("truth columns must be unique")
    unknown = [column for column in truth.columns if column not in window.components]
    if unknown:
        raise ValueError(
            f"truth columns {unknown} are not components; components: {list(window.components)}"
        )
    weekly = truth.reindex(window.dates)
    return {
        column: weekly[column].to_numpy(dtype=np.float64, na_value=np.nan)
        for column in weekly.columns
    }


class ModelSource:
    """Read access to one model's latest ``add_context`` and ``inference``."""

    __slots__ = ("_model",)

    def __init__(self, model: MFM):
        self._model = model

    def history(self) -> _FittedContext:
        """The context of the latest ``add_context``; raises before one."""
        history = self._model._inference_context
        if history is None:
            raise RuntimeError("call model.add_context(history, ...) before plotting")
        return history

    def prediction(self) -> _PredictionSnapshot | None:
        """The latest successful ``inference`` after the current context, or ``None``."""
        self.history()
        return self._model._prediction_snapshot

    def require_prediction(self) -> _PredictionSnapshot:
        prediction = self.prediction()
        if prediction is None:
            raise RuntimeError(
                "this chart draws inference draws: call model.inference(future_data) first"
            )
        return prediction

    def response_curves(self) -> ResponseCurves:
        """The latest ``model.response_curves`` of the current prediction; raises before one."""
        self.require_prediction()
        curves = self._model._response_curves
        if curves is None:
            raise RuntimeError("call model.response_curves(...) first")
        return curves

    def window(self, name: str) -> Window:
        """Named draws of the ``"context"`` or ``"query"`` window.

        The first context window of a prediction samples the context marginal once.
        """
        if name not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
        prediction = self.require_prediction()
        history = prediction.fitted_context
        length = len(history.dates)
        if name == "context":
            result = self._model._context_result()
            dates, outcome, spend = history.dates, history.outcome, history.spend
        else:
            result = prediction.query
            dates = prediction.time[length:]
            outcome = prediction.query_outcome
            spend = prediction.observations["spend"][length:]
        active, names = display_names(result)
        return Window(
            name=name,
            dates=dates,
            components=names,
            families=tuple(result.component_families[index] for index in active),
            draws=result.components[:, :, active],
            outcome=outcome,
            channels=history.channel_columns,
            spend=spend,
            target=history.target_column,
            units=result.units,
        )

    def windows(self, window: str | None) -> tuple[Window, ...]:
        """Both windows in time order when ``window`` is ``None``, else that one."""
        if window is None:
            return tuple(self.window(name) for name in WINDOWS)
        return (self.window(window),)
