"""What the model reads: outcome, spend and controls by week (``model.plot.inputs``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.plotting.panels import COLORS, _suptitle

from ._backend import check, draw
from ._mpl import compact_axis, figure, forecast_span, time_positions
from ._style import ACTUAL

if TYPE_CHECKING:
    from ._data import ModelSource

_ROLE_COLORS = {"outcome": ACTUAL, "channel": COLORS["channel"], "control": COLORS["control"]}
_ROLE_LABELS = {"outcome": "outcome", "channel": "spend", "control": "control"}
_SPAN_NOTE = "Shaded: forecast weeks, whose outcome the model did not see"


@dataclass(frozen=True, slots=True)
class Inputs:
    """Weekly series in drawing order: ``(name, role, values)``; roles as in ``_ROLE_COLORS``."""

    dates: np.ndarray
    first_query: int | None
    series: tuple[tuple[str, str, np.ndarray], ...]
    title: str


def prepare(source: ModelSource, *, title: str | None = None) -> Inputs:
    """The context, and the forecast weeks of the latest ``inference`` if there is one.

    Forecast-week outcomes are the target column given to ``inference`` (NaN where
    it had none); the model never read them.
    """
    history = source.history()
    prediction = source.prediction()
    length = len(history.dates)
    if prediction is None:
        dates, first = history.dates, None
        spend, controls, outcome = history.spend, history.controls, history.outcome
    else:
        dates, first = prediction.time, length
        spend = prediction.observations["spend"]
        controls = prediction.observations["controls"]
        query = prediction.query_outcome
        future = np.full(len(dates) - length, np.nan) if query is None else query
        outcome = np.concatenate((history.outcome, future))
    channels, controls_named = history.channel_columns, history.control_columns
    series = (
        (history.target_column, "outcome", outcome),
        *((name, "channel", spend[:, index]) for index, name in enumerate(channels)),
        *((name, "control", controls[:, index]) for index, name in enumerate(controls_named)),
    )
    return Inputs(dates, first, series, title or "What the model reads")


def _matplotlib(data: Inputs) -> Any:
    fig, axes = figure(len(data.series), 1, width=11, row_height=1.5, sharex=True)
    for ax, (name, role, values) in zip(axes[:, 0], data.series, strict=True):
        positions = time_positions(ax, data.dates)
        ax.plot(positions, values, color=_ROLE_COLORS[role], linewidth=1.2)
        if data.first_query is not None:
            forecast_span(ax, positions, data.first_query, label=None)
        ax.set_title(f"{name} ({_ROLE_LABELS[role]})", loc="left", fontsize=9)
        compact_axis(ax.yaxis)
        ax.grid(axis="y", alpha=0.15)
    for ax in axes[:-1, 0]:  # the rows share one date axis, labelled once at the bottom
        ax.tick_params(labelbottom=False)
    _suptitle(fig, data.title, _SPAN_NOTE if data.first_query is not None else None)
    return fig


def _plotly(data: Inputs) -> Any:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    rows = len(data.series)
    fig = plotting.figure(
        rows,
        1,
        shared_xaxes=True,
        vertical_spacing=min(0.06, 0.4 / rows),
        subplot_titles=[
            plotting.text(f"{name} ({_ROLE_LABELS[role]})") for name, role, _ in data.series
        ],
    )
    for row, (name, role, values) in enumerate(data.series, start=1):
        fig.add_trace(
            go.Scatter(
                x=data.dates,
                y=values,
                mode="lines",
                name=name,
                line={"color": _ROLE_COLORS[role], "width": 1.5},
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:,.4~r}}<extra>{plotting.text(name)}</extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    if data.first_query is not None:
        plotting.forecast_span(fig, data.dates, data.first_query, row="all", col=1)
    caption = _SPAN_NOTE if data.first_query is not None else None
    return plotting.finish(fig, title=data.title, caption=caption, height=110 * rows + 50)


def inputs(source: ModelSource, *, backend: str | None = None, title: str | None = None) -> Any:
    check(backend)
    data = prepare(source, title=title)
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly)
