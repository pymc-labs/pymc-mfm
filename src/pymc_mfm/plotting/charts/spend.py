"""Weekly contribution against weekly spend, per channel (``model.plot.contribution_vs_spend``)."""

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

    from ._data import ModelSource

_COLOR = COLORS["channel"]
_ERROR_ALPHA = 0.35
_CARRYOVER = (
    "A week's contribution also depends on earlier weeks' spend (carryover), "
    "so the points need not lie on one curve"
)


@dataclass(frozen=True, slots=True)
class ChannelWeeks:
    """One channel by week: raw spend and the mean and central interval of its contribution.

    ``truth`` is the channel's truth by week (NaN where it has none), or ``None``.
    """

    name: str
    spend: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    truth: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ContributionVsSpend:
    """One panel per channel, in drawing order, over the weeks ``dates``."""

    dates: np.ndarray
    channels: tuple[ChannelWeeks, ...]
    ylabel: str
    interval: float
    n_samples: int
    title: str


def prepare(
    source: ModelSource,
    *,
    window: str = "context",
    channels: Sequence[str] | None = None,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    title: str | None = None,
) -> ContributionVsSpend:
    """Each channel's weekly spend beside the mean and interval of its contribution draws.

    ``interval`` is checked before any window is read. A drawn channel's truth may
    miss some weeks, which get no cross, but not every week: its dates would not be
    the window's (Sunday- rather than Monday-dated weeks, say).
    """
    interval_quantiles(interval)
    data = source.window(window)
    # Channel components come in the order of the spend columns.
    names = tuple(
        name
        for name, family in zip(data.components, data.families, strict=True)
        if family == "channel"
    )
    spend = dict(zip(names, data.spend.T, strict=True))
    selected = names if channels is None else checked_names(channels, names, "channels")
    weekly = {} if truth is None else truth_by_week(truth, data)
    weeks = window_weeks(data)
    panels = []
    for name in selected:
        known = weekly.get(name)
        if known is not None and np.isnan(known).all():
            raise ValueError(
                f"truth[{name!r}] has no value in any of the {weeks}; "
                "index it by the dates of the data given to the model"
            )
        mean, low, high = summarize(data.contribution(name), interval)
        panels.append(ChannelWeeks(name, spend[name], mean, low, high, known))
    default = f"Weekly contribution against weekly spend over the {weeks}, by channel"
    return ContributionVsSpend(
        dates=data.dates,
        channels=tuple(panels),
        ylabel=f"Weekly contribution ({data.target})",
        interval=interval,
        n_samples=data.n_samples,
        title=title or default,
    )


def _matplotlib(data: ContributionVsSpend) -> Any:
    count = len(data.channels)
    rows, columns = panel_grid(count)
    # Each row's panels, plus room for the title lines and the legend.
    fig, axes = figure(rows, columns, width=11, row_height=3.2 + 1.4 / rows, sharey=True)
    band = f"{percent(data.interval)} interval"
    for ax, channel in zip(axes.flat, data.channels):
        ax.vlines(
            channel.spend,
            channel.low,
            channel.high,
            color=_COLOR,
            alpha=_ERROR_ALPHA,
            linewidth=1.5,
            label=band,
        )
        ax.scatter(channel.spend, channel.mean, color=_COLOR, s=16, zorder=3, label="Mean")
        if channel.truth is not None:
            ax.scatter(
                channel.spend,
                channel.truth,
                color=ACTUAL,
                marker="x",
                s=30,
                linewidths=1.2,
                zorder=4,
                label="Truth",
            )
        ax.set_title(channel.name, loc="left", fontsize=10)
        ax.set_xlabel("Weekly spend", fontsize=9)
        compact_axis(ax.xaxis)
        compact_axis(ax.yaxis)
        ax.grid(alpha=0.15)
    for ax in axes[:, 0]:
        ax.set_ylabel(data.ylabel, fontsize=9)
    for ax in axes[:, 1:].flat:  # the panels share one y axis, labelled on the left
        ax.tick_params(labelleft=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    figure_legend(fig, axes.flat[:count])
    _suptitle(fig, data.title, _CARRYOVER, caption(data.interval, data.n_samples), CAVEAT)
    return fig


def _plotly(data: ContributionVsSpend) -> Any:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    count = len(data.channels)
    rows, columns = panel_grid(count)
    fig = plotting.figure(
        rows,
        columns,
        specs=plotting.subplot_specs(count),
        shared_yaxes="all",
        subplot_titles=[plotting.text(channel.name) for channel in data.channels],
        horizontal_spacing=0.06,
        vertical_spacing=0.36 / rows,
    )
    weeks = np.datetime_as_string(data.dates, unit="D")
    band = f"{percent(data.interval)} interval"
    truth_shown = False
    for index, channel in enumerate(data.channels):
        where = {"row": index // columns + 1, "col": index % columns + 1}
        extra = f"<extra>{plotting.text(channel.name)}</extra>"
        fig.add_trace(
            go.Scatter(
                x=channel.spend,
                y=channel.mean,
                mode="markers",
                marker={"color": _COLOR, "size": 6},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": channel.high - channel.mean,
                    "arrayminus": channel.mean - channel.low,
                    "color": rgba(_COLOR, _ERROR_ALPHA),
                    "thickness": 1.5,
                    "width": 0,
                },
                customdata=np.column_stack((channel.low, channel.high)),
                text=weeks,
                name=f"Mean and {band}",
                legendgroup="mean",
                showlegend=index == 0,
                hovertemplate=(
                    "%{text}<br>Spend %{x:,.4~r}<br>Mean contribution %{y:,.4~r}<br>"
                    f"{band} %{{customdata[0]:,.4~r}} to %{{customdata[1]:,.4~r}}{extra}"
                ),
            ),
            **where,
        )
        if channel.truth is not None:
            fig.add_trace(
                go.Scatter(
                    x=channel.spend,
                    y=channel.truth,
                    mode="markers",
                    marker={"color": ACTUAL, "symbol": "x", "size": 7},
                    text=weeks,
                    name="Truth",
                    legendgroup="truth",
                    showlegend=not truth_shown,
                    hovertemplate=f"%{{text}}<br>Spend %{{x:,.4~r}}<br>Truth %{{y:,.4~r}}{extra}",
                ),
                **where,
            )
            truth_shown = True
    fig.update_xaxes(title={"text": "Weekly spend", "font": {"size": 11}})
    fig.update_yaxes(title={"text": plotting.text(data.ylabel), "font": {"size": 11}}, col=1)
    return plotting.finish(
        fig,
        title=data.title,
        caption=(_CARRYOVER, caption(data.interval, data.n_samples), CAVEAT),
        height=320 * rows + 50,
    )


def contribution_vs_spend(
    source: ModelSource,
    *,
    window: str = "context",
    channels: Sequence[str] | None = None,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    backend: str | None = None,
    title: str | None = None,
) -> Any:
    check(backend)
    data = prepare(
        source, window=window, channels=channels, interval=interval, truth=truth, title=title
    )
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly)
