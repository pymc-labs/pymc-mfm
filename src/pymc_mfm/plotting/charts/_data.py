"""The model state charts draw: history, the latest prediction and named window draws.

:class:`ModelSource` is the one place the charts read a model. It never samples,
except through ``MFM._context_result``, which samples the context marginal once per
prediction and caches it for ``build_idata`` too.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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
    twice (a channel named ``baseline``, say) keeps its full component name.
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
    counts = Counter(plain[index] for index in active)
    names = tuple(
        plain[index] if counts[plain[index]] == 1 else result.component_names[index]
        for index in active
    )
    return active, names


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
        self.history()
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
