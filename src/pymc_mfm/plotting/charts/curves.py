"""Each channel's contribution as its spend scales (``model.plot.response_curves``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymc_mfm.inference.curves import select_channels
from pymc_mfm.plotting.panels import COLORS, _suptitle

from ._backend import check, draw
from ._data import summarize
from ._mpl import compact_axis, figure, percent_axis
from ._style import ACTUAL, BAND_ALPHA, CAVEAT, caption, compact, percent

if TYPE_CHECKING:
    from ._data import ModelSource

_TITLE = "Where does extra spend stop paying off?"
#: Panels per row before the grid wraps.
_COLUMNS = 5
_CURVE = COLORS["channel"]
#: The dashed line at the actual spend.
_REFERENCE = "#737373"
_REFERENCE_LABEL = "Actual spend (100%)"
_XLABEL = "Spend, % of actual"


@dataclass(frozen=True, slots=True)
class Curves:
    """One panel per channel, in drawing order.

    ``spend`` (the level times the actual spend), ``mean``, ``low``, ``high`` and
    ``truth`` are ``[channel, level]``; ``truth`` is NaN where none was given and
    ``None`` without truth. ``actual`` indexes level 1 in ``levels``, if listed.
    """

    channels: tuple[str, ...]
    panel_titles: tuple[str, ...]
    levels: np.ndarray
    spend: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    actual: int | None
    truth: np.ndarray | None
    band: str
    ylabel: str
    title: str
    caption: tuple[str, ...]


def prepare(
    source: ModelSource,
    *,
    channels: Sequence[str] | None = None,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    title: str | None = None,
) -> Curves:
    """Mean and central ``interval`` of the latest ``model.response_curves``, by channel."""
    curves = source.response_curves()
    names = select_channels(channels, curves.channels)
    rows = [curves.channels.index(name) for name in names]
    mean, low, high = summarize(curves.draws[rows], interval, axis=2)
    totals = curves.spend[rows]
    (actual,) = np.nonzero(curves.levels == 1.0)
    weeks = curves.weeks
    span = "a one-week plan" if weeks == 1 else f"a plan of the first {weeks} forecast weeks"
    return Curves(
        channels=names,
        panel_titles=tuple(
            f"{name} · spend {compact(total)}" if total > 0 else f"{name} · no spend in these weeks"
            for name, total in zip(names, totals, strict=True)
        ),
        levels=curves.levels,
        spend=totals[:, None] * curves.levels,
        mean=mean,
        low=low,
        high=high,
        actual=int(actual[0]) if actual.size else None,
        truth=_truth(truth, curves.levels, curves.channels, names),
        band=f"{percent(interval)} interval",
        ylabel=f"Contribution over {weeks} week{'s' if weeks > 1 else ''} ({curves.target})",
        title=title or _TITLE,
        caption=(
            caption(interval, curves.n_samples),
            f"Each channel varied separately over {span}; history and other inputs fixed.",
            CAVEAT,
        ),
    )


def _truth(
    truth: pd.DataFrame | None,
    levels: np.ndarray,
    available: tuple[str, ...],
    names: tuple[str, ...],
) -> np.ndarray | None:
    """``truth`` as ``[channel, level]`` values of ``names``; NaN where it has none."""
    if truth is None:
        return None
    if not isinstance(truth, pd.DataFrame):
        raise ValueError("truth must be a DataFrame indexed by level, one column per channel")
    columns = list(truth.columns)
    if truth.columns.has_duplicates or any(name not in available for name in columns):
        raise ValueError(f"truth columns must be distinct channels of {list(available)}: {columns}")
    try:
        index = truth.index.to_numpy(dtype=np.float64)
        values = truth.to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("truth must hold numbers, indexed by spend level") from None
    matches = np.isclose(index[:, None], levels, rtol=1e-9, atol=1e-12)
    found = matches.any(axis=1)
    if not found.all():
        raise ValueError(
            f"truth is indexed by levels {index[~found].tolist()} that the curves do not "
            f"have; their levels are {levels.tolist()}"
        )
    positions = matches.argmax(axis=1)
    if np.unique(positions).size != positions.size:
        raise ValueError("truth must list each level once")
    result = np.full((len(names), len(levels)), np.nan)
    for row, name in enumerate(names):
        if name in columns:
            result[row, positions] = values[:, columns.index(name)]
    return result


def _grid(count: int) -> tuple[int, int]:
    """Rows and columns of ``count`` panels, at most ``_COLUMNS`` to a row."""
    columns = min(count, _COLUMNS)
    return math.ceil(count / columns), columns


def _matplotlib(data: Curves) -> Any:
    count = len(data.channels)
    rows, columns = _grid(count)
    # The title, its three caption lines and the legend take about 1.5 inches.
    height = 3.0 + 1.5 / rows
    fig, axes = figure(rows, columns, width=11, row_height=height, sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.set_visible(False)
            continue
        ax.plot(data.levels, data.mean[index], color=_CURVE, linewidth=2, label="Mean")
        ax.fill_between(
            data.levels,
            data.low[index],
            data.high[index],
            color=_CURVE,
            alpha=BAND_ALPHA,
            linewidth=0,
            label=data.band,
        )
        if data.actual is not None:
            ax.scatter(
                data.levels[data.actual],
                data.mean[index, data.actual],
                color=_CURVE,
                edgecolor="white",
                zorder=3,
            )
        ax.axvline(1.0, color=_REFERENCE, linestyle="--", linewidth=1, label=_REFERENCE_LABEL)
        if data.truth is not None:
            ax.scatter(data.levels, data.truth[index], color=ACTUAL, s=14, zorder=4, label="Truth")
        ax.set_title(data.panel_titles[index], loc="left", fontsize=9)
        percent_axis(ax.xaxis)
        compact_axis(ax.yaxis)
        ax.grid(axis="y", alpha=0.15)
        if index % columns:  # the panels share the y axis, labelled on the left
            ax.tick_params(labelleft=False)
        if index + columns >= count:  # the lowest panel of its column
            ax.set_xlabel(_XLABEL, fontsize=9)
    fig.supylabel(data.ylabel, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncols=len(labels), frameon=False, fontsize=9
    )
    _suptitle(fig, data.title, *data.caption)
    return fig


def _plotly(data: Curves) -> Any:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    count = len(data.channels)
    rows, columns = _grid(count)
    fig = plotting.figure(
        rows,
        columns,
        shared_yaxes="all",
        specs=[
            [{} if row * columns + column < count else None for column in range(columns)]
            for row in range(rows)
        ],
        subplot_titles=[plotting.text(title) for title in data.panel_titles],
        horizontal_spacing=0.06,
        vertical_spacing=0.3 / rows,
    )
    for index, name in enumerate(data.channels):
        where = {"row": index // columns + 1, "col": index % columns + 1}
        first = index == 0
        fig.add_trace(
            go.Scatter(
                x=data.levels,
                y=data.mean[index],
                mode="lines",
                name="Mean",
                line={"color": _CURVE, "width": 2},
                customdata=np.column_stack((data.spend[index], data.low[index], data.high[index])),
                hovertemplate=(
                    f"{plotting.text(name)} at %{{x:.1%}} of its spend<br>"
                    "Spend %{customdata[0]:,.4~r}<br>"
                    f"Mean contribution %{{y:,.4~r}} · {data.band} "
                    "%{customdata[1]:,.4~r} to %{customdata[2]:,.4~r}<extra></extra>"
                ),
                legendgroup="mean",
                showlegend=first,
            ),
            **where,
        )
        plotting.band(
            fig,
            data.levels,
            data.low[index],
            data.high[index],
            color=_CURVE,
            name=data.band,
            legendgroup="interval",
            showlegend=first,
            **where,
        )
        if data.actual is not None:
            fig.add_trace(
                go.Scatter(
                    x=[data.levels[data.actual]],
                    y=[data.mean[index, data.actual]],
                    mode="markers",
                    marker={"color": _CURVE, "size": 9, "line": {"color": "white", "width": 1}},
                    hoverinfo="skip",
                    legendgroup="mean",
                    showlegend=False,
                ),
                **where,
            )
        fig.add_vline(
            x=1.0,
            line={"color": _REFERENCE, "dash": "dash", "width": 1},
            name=_REFERENCE_LABEL,
            legendgroup="actual",
            showlegend=first,
            **where,
        )
        if data.truth is not None:
            fig.add_trace(
                go.Scatter(
                    x=data.levels,
                    y=data.truth[index],
                    mode="markers",
                    name="Truth",
                    marker={"color": ACTUAL, "size": 6},
                    hovertemplate=(
                        f"Truth for {plotting.text(name)} at %{{x:.1%}} of its spend<br>"
                        "%{y:,.4~r}<extra></extra>"
                    ),
                    legendgroup="truth",
                    showlegend=first,
                ),
                **where,
            )
        if index + columns >= count:  # the lowest panel of its column
            fig.update_xaxes(title={"text": _XLABEL, "font": {"size": 11}}, **where)
    # Narrow panels get fewer ticks, which then stay level instead of rotating.
    fig.update_xaxes(tickformat=".0%", nticks=4 if columns > 2 else 0)
    fig.update_yaxes(title={"text": plotting.text(data.ylabel), "font": {"size": 11}}, col=1)
    return plotting.finish(fig, title=data.title, caption=data.caption, height=300 * rows + 50)


def response_curves(
    source: ModelSource,
    *,
    channels: Sequence[str] | None = None,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    backend: str | None = None,
    title: str | None = None,
) -> Any:
    check(backend)
    data = prepare(source, channels=channels, interval=interval, truth=truth, title=title)
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly)
