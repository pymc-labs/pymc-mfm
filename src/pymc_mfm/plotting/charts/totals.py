"""Distribution over draws of each component's window total (``model.plot.contribution_totals``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.plotting.panels import COLORS, _suptitle

from ._backend import check, draw
from ._data import checked_names, interval_quantiles, summarize, truth_by_week, window_weeks
from ._mpl import compact_axis, figure, figure_legend
from ._style import ACTUAL, CAVEAT, caption, panel_grid, percent, rgba

if TYPE_CHECKING:
    import pandas as pd

    from ._data import IdataReader, Window

#: Families drawn unless ``components=`` names others.
_FAMILIES = ("channel", "control")
_HISTOGRAM_ALPHA = 0.55
_SPAN_ALPHA = 0.15
#: Weeks an error names by date before it counts the rest.
_LISTED = 6


@dataclass(frozen=True, slots=True)
class Total:
    """One component's per-draw totals: histogram, mean, central interval and truth.

    ``counts`` and ``edges`` are ``np.histogram`` of the totals; ``truth`` is the
    sum of the component's truth over the window, or ``None`` without one.
    """

    name: str
    family: str
    counts: np.ndarray
    edges: np.ndarray
    mean: float
    low: float
    high: float
    truth: float | None


@dataclass(frozen=True, slots=True)
class Totals:
    """One panel per component, in drawing order."""

    totals: tuple[Total, ...]
    xlabel: str
    interval: float
    n_samples: int
    title: str


def _truth_total(weekly: dict[str, np.ndarray], name: str, window: Window) -> float | None:
    """The sum of ``name``'s truth over every week of ``window``, or ``None`` without one."""
    values = weekly.get(name)
    if values is None:
        return None
    missing = np.isnan(values)
    if missing.any():
        weeks = np.datetime_as_string(window.dates[missing], unit="D")
        listed = ", ".join(weeks[:_LISTED])
        if len(weeks) > _LISTED:
            listed += f" and {len(weeks) - _LISTED} more"
        raise ValueError(
            f"truth[{name!r}] has no value for {listed}; a total over part of the "
            f"{window_weeks(window)} would mislead"
        )
    return float(values.sum())


def prepare(
    source: IdataReader,
    *,
    window: str = "context",
    components: Sequence[str] | None = None,
    interval: float = 0.9,
    bins: int = 40,
    truth: pd.DataFrame | None = None,
    title: str | None = None,
) -> Totals:
    """Each component's per-draw total over ``window``, binned once for both backends.

    A draw's total sums its weekly contributions; the mean and central ``interval``
    summarize the totals. ``interval`` is checked before any window is read.
    """
    interval_quantiles(interval)
    data = source.window(window)
    if components is None:
        names = tuple(
            name
            for name, family in zip(data.components, data.families, strict=True)
            if family in _FAMILIES
        )
    else:
        names = checked_names(components, data.components, "components")
    weekly = {} if truth is None else truth_by_week(truth, data)
    totals = []
    for name in names:
        values = data.contribution(name).sum(axis=1)
        counts, edges = np.histogram(values, bins)
        mean, low, high = summarize(values, interval)
        totals.append(
            Total(
                name=name,
                family=data.families[data.components.index(name)],
                counts=counts,
                edges=edges,
                mean=float(mean),
                low=float(low),
                high=float(high),
                truth=_truth_total(weekly, name, data),
            )
        )
    weeks = window_weeks(data)
    return Totals(
        totals=tuple(totals),
        xlabel=f"Total over {weeks} (outcome)",
        interval=interval,
        n_samples=data.n_samples,
        title=title or f"Total contribution over the {weeks}, by component",
    )


def _matplotlib(data: Totals) -> Any:
    from matplotlib.ticker import MaxNLocator

    count = len(data.totals)
    rows, columns = panel_grid(count)
    # Each row's panels, plus room for the title lines and the legend.
    fig, axes = figure(rows, columns, width=11, row_height=2.8 + 1.2 / rows)
    band = f"{percent(data.interval)} interval"
    for ax, total in zip(axes.flat, data.totals):
        color = COLORS[total.family]
        ax.stairs(total.counts, total.edges, fill=True, color=color, alpha=_HISTOGRAM_ALPHA)
        ax.axvspan(
            total.low, total.high, color=color, alpha=_SPAN_ALPHA, linewidth=0, zorder=0, label=band
        )
        ax.axvline(total.mean, color=color, linewidth=2, label="Mean")
        if total.truth is not None:
            ax.axvline(total.truth, color=ACTUAL, linestyle="--", linewidth=1.5, label="Truth")
        ax.set_title(total.name, loc="left", fontsize=10)
        ax.set_xlabel(data.xlabel, fontsize=9)
        compact_axis(ax.xaxis)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # draw counts
    for ax in axes[:, 0]:
        ax.set_ylabel("Draws", fontsize=9)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    figure_legend(fig, axes.flat[:count])
    lines = caption(data.interval, data.n_samples, pointwise=False)
    _suptitle(fig, data.title, lines, CAVEAT)
    return fig


def _plotly(data: Totals) -> Any:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    count = len(data.totals)
    rows, columns = panel_grid(count)
    fig = plotting.figure(
        rows,
        columns,
        specs=plotting.subplot_specs(count),
        subplot_titles=[plotting.text(total.name) for total in data.totals],
        horizontal_spacing=0.08,
        vertical_spacing=0.36 / rows,
    )
    band = f"{percent(data.interval)} interval"
    truth_shown = False
    for index, total in enumerate(data.totals):
        where = {"row": index // columns + 1, "col": index % columns + 1}
        color, first = COLORS[total.family], index == 0
        # Each bar's hover also states the panel's summary: the lines have no hover.
        summary = [total.mean, total.low, total.high]
        hover = [
            "%{x:,.4~r} to %{customdata[0]:,.4~r}: %{y} draws",
            f"Mean %{{customdata[1]:,.4~r}} · {band} "
            "%{customdata[2]:,.4~r} to %{customdata[3]:,.4~r}",
        ]
        if total.truth is not None:
            summary.append(total.truth)
            hover.append("Truth %{customdata[4]:,.4~r}")
        fig.add_trace(
            go.Bar(
                x=total.edges[:-1],
                y=total.counts,
                width=np.diff(total.edges),
                offset=0,
                customdata=np.column_stack(
                    (total.edges[1:], np.tile(summary, (len(total.counts), 1)))
                ),
                marker={"color": rgba(color, _HISTOGRAM_ALPHA), "line": {"width": 0}},
                name=total.name,
                showlegend=False,
                hovertemplate="<br>".join(hover) + f"<extra>{plotting.text(total.name)}</extra>",
            ),
            **where,
        )
        fig.add_vrect(
            x0=total.low,
            x1=total.high,
            fillcolor=color,
            opacity=_SPAN_ALPHA,
            line_width=0,
            layer="below",
            name=band,
            legendgroup="interval",
            showlegend=first,
            **where,
        )
        fig.add_vline(
            x=total.mean,
            line={"color": color, "width": 2},
            name="Mean",
            legendgroup="mean",
            showlegend=first,
            **where,
        )
        if total.truth is not None:
            fig.add_vline(
                x=total.truth,
                line={"color": ACTUAL, "width": 1.5, "dash": "dash"},
                name="Truth",
                legendgroup="truth",
                showlegend=not truth_shown,
                **where,
            )
            truth_shown = True
    fig.update_xaxes(title={"text": plotting.text(data.xlabel), "font": {"size": 11}})
    fig.update_yaxes(title={"text": "Draws", "font": {"size": 11}}, col=1)
    fig.update_layout(showlegend=True)  # the legend lists shapes, not traces
    return plotting.finish(
        fig,
        title=data.title,
        caption=(caption(data.interval, data.n_samples, pointwise=False), CAVEAT),
        height=300 * rows + 50,
    )


def contribution_totals(
    source: IdataReader,
    *,
    window: str = "context",
    components: Sequence[str] | None = None,
    interval: float = 0.9,
    bins: int = 40,
    truth: pd.DataFrame | None = None,
    backend: str | None = None,
    title: str | None = None,
) -> Any:
    check(backend)
    data = prepare(
        source,
        window=window,
        components=components,
        interval=interval,
        bins=bins,
        truth=truth,
        title=title,
    )
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly)
