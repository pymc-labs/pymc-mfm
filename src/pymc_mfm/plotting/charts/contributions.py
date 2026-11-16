"""What each component contributed, week by week (``model.plot.contributions_over_time``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.plotting.panels import COLORS, _single, _suptitle, _title

from ._backend import check, draw
from ._data import checked_names, interval_quantiles, summarize, truth_by_week
from ._mpl import compact_axis, figure, forecast_span, time_positions
from ._style import ACTUAL, BAND_ALPHA, CAVEAT, caption, percent, rgba, series_colors

if TYPE_CHECKING:
    import pandas as pd

    from ._data import ModelSource, Window

#: Component families from the bottom of the stack up: channels end on top.
FAMILY_ORDER = ("baseline", "latent", "indirect", "control", "channel")
#: Panels per row when ``stacked=False``.
COLUMNS = 3
_STACK_ALPHA = 0.85
#: Line width, in points, of a one-week window's interval: one week has no width to fill.
_BAR_WIDTH = 8


@dataclass(frozen=True, slots=True)
class Segment:
    """One drawn window's numbers ``[component, week]``, components in drawing order.

    ``mean`` is each component's weekly mean over the draws. Stacked, ``lower`` and
    ``upper`` bound the component's layer of the stack, flat across each week from one
    of its ``edges`` to the next (``len(dates) + 1`` dates halfway between weeks); in
    panels they bound the central interval of its draws.
    """

    dates: np.ndarray
    edges: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True, slots=True)
class Contributions:
    """Every number the chart draws.

    ``components`` and their ``colors`` are in drawing order, the bottom of the stack
    first. ``segments`` are the drawn windows in time order over ``dates``;
    ``first_query`` indexes the first forecast week in ``dates`` when both windows are
    drawn. ``outcome`` is the observed outcome on ``dates``, drawn over the stack
    (``None`` in panels, or when no drawn week was observed). ``truth`` has one entry
    per component: its known contribution on ``dates`` (NaN where none was given), or
    ``None``.
    """

    components: tuple[str, ...]
    colors: tuple[str, ...]
    stacked: bool
    segments: tuple[Segment, ...]
    dates: np.ndarray
    first_query: int | None
    outcome: np.ndarray | None
    truth: tuple[np.ndarray | None, ...]
    target: str
    interval: float
    title: str
    caption: tuple[str, ...]


def _layers(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(lower, upper)`` of each component's layer ``[component, week]`` of the stack.

    Positive means stack up from zero and negative means down from zero, each in
    drawing order, so no two layers overlap.
    """
    zero = np.zeros((1, mean.shape[1]))
    up = np.cumsum(np.concatenate((zero, np.maximum(mean, 0))), axis=0)
    down = np.cumsum(np.concatenate((zero, np.minimum(mean, 0))), axis=0)
    rising = mean >= 0
    return np.where(rising, up[:-1], down[1:]), np.where(rising, up[1:], down[:-1])


def _edges(dates: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Where each of ``dates`` starts and ends: halfway to its neighbours.

    The first starts, and the last ends, half the ``history`` cadence (its median step)
    away from its date.
    """
    half = np.timedelta64(round(np.median(np.diff(history).astype(np.int64)) / 2), "ns")
    middle = dates[:-1] + (dates[1:] - dates[:-1]) / 2
    return np.concatenate(([dates[0] - half], middle, [dates[-1] + half]))


def _segment(
    data: Window, edges: np.ndarray, order: list[int], stacked: bool, interval: float
) -> Segment:
    if stacked:
        mean = data.draws.mean(axis=0)[:, order].T
        return Segment(data.dates, edges, mean, *_layers(mean))
    mean, low, high = (values[:, order].T for values in summarize(data.draws, interval))
    return Segment(data.dates, edges, mean, low, high)


def _outcome(windows: tuple[Window, ...]) -> np.ndarray | None:
    """The observed outcome over every drawn week, or ``None`` when none was observed."""
    observed = np.concatenate(
        [
            np.full(len(part.dates), np.nan) if part.outcome is None else part.outcome
            for part in windows
        ]
    )
    return observed if np.isfinite(observed).any() else None


def _truth(
    truth: pd.DataFrame | None, windows: tuple[Window, ...], names: tuple[str, ...]
) -> tuple[np.ndarray | None, ...]:
    """Each component's ``truth`` over every drawn week (NaN where it has none), or ``None``.

    A drawn component's truth with no value on any drawn date raises: its dates (a
    Sunday instead of a Monday, say) would otherwise draw nothing without a word.
    """
    if truth is None:
        return (None,) * len(names)
    weekly = [truth_by_week(truth, part) for part in windows]
    drawn = []
    for name in names:
        if name not in weekly[0]:
            drawn.append(None)
            continue
        values = np.concatenate([part[name] for part in weekly])
        if np.isnan(values).all():
            raise ValueError(
                f"truth[{name!r}] has no value on any drawn date; "
                "index it by the dates of the data given to the model"
            )
        drawn.append(values)
    return tuple(drawn)


def prepare(
    source: ModelSource,
    *,
    window: str | None = None,
    components: Sequence[str] | None = None,
    stacked: bool = True,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    title: str | None = None,
) -> Contributions:
    """Each chosen component's weekly numbers in every drawn window.

    ``truth`` with ``stacked=True`` and a bad ``interval`` with ``stacked=False`` raise
    before any window is read; ``window="query"`` never samples the context.
    """
    if truth is not None and stacked:
        raise ValueError("truth is drawn only with stacked=False, in each component's panel")
    if not stacked:
        interval_quantiles(interval)
    windows = source.windows(window)
    first = windows[0]
    ranked = sorted(
        range(len(first.components)), key=lambda index: FAMILY_ORDER.index(first.families[index])
    )
    chosen = first.components if components is None else components
    chosen = checked_names(chosen, first.components, "components")
    if stacked or components is None:
        order = [index for index in ranked if first.components[index] in chosen]
    else:  # panels follow the order the caller listed them in
        order = [first.components.index(name) for name in dict.fromkeys(chosen)]
    names = tuple(first.components[index] for index in order)
    # Colored over every component, so a component keeps its color in any selection.
    palette = series_colors(
        first.components[index] for index in ranked if first.families[index] != "baseline"
    )
    colors = tuple(
        COLORS["baseline"] if first.families[index] == "baseline" else palette[name]
        for index, name in zip(order, names, strict=True)
    )
    dates = np.concatenate([part.dates for part in windows])
    # The windows share the edge between them, so their stacks meet where forecasts start.
    edges = _edges(dates, source.history().dates)
    segments, start = [], 0
    for part in windows:
        stop = start + len(part.dates)
        segments.append(_segment(part, edges[start : stop + 1], order, stacked, interval))
        start = stop
    draws = first.n_samples
    if stacked:
        lines = (
            f"Stacked weekly means over {draws:,} draws: positive means up from zero, "
            "negative means down",
            f"Intervals do not add up, so the stack shows none · {CAVEAT}",
        )
    else:
        lines = (caption(interval, draws), CAVEAT)
    return Contributions(
        components=names,
        colors=colors,
        stacked=stacked,
        segments=tuple(segments),
        dates=dates,
        first_query=len(first.dates) if len(windows) > 1 else None,
        outcome=_outcome(windows) if stacked else None,
        truth=_truth(truth, windows, names),
        target=first.target,
        interval=interval,
        title=title or f"What each component contributed to {first.target}, week by week",
        caption=lines,
    )


def _matplotlib_stack(data: Contributions, ax) -> Any:
    fig, ax, owned = _single(ax, width=11, row_height=4.6)
    for number, segment in enumerate(data.segments):
        edges = time_positions(ax, segment.edges)
        for name, color, lower, upper in zip(
            data.components, data.colors, segment.lower, segment.upper, strict=True
        ):
            layer = ax.stairs(
                upper,
                edges,
                baseline=lower,
                fill=True,
                color=color,
                alpha=_STACK_ALPHA,
                label=name if number == 0 else None,
            )
            layer.sticky_edges.y.clear()  # keep a margin under the stack, as above it
    positions = time_positions(ax, data.dates)
    ax.axhline(0, color="0.3", linewidth=0.8)
    if data.outcome is not None:
        ax.plot(
            positions,
            data.outcome,
            color=ACTUAL,
            linewidth=1.1,
            marker="o",
            markersize=2.5,
            label=f"Actual {data.target}",
        )
    if data.first_query is not None:
        forecast_span(ax, positions, data.first_query)
    ax.set_ylabel(data.target, fontsize=9)
    compact_axis(ax.yaxis)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _title(fig, ax, "\n".join(data.caption), data.title, owned)
    return fig


def _matplotlib_panels(data: Contributions) -> Any:
    count = len(data.components)
    columns = min(COLUMNS, count)
    rows = math.ceil(count / columns)
    # Each row's panels, plus room for the title lines and the legend.
    fig, axes = figure(rows, columns, width=11, row_height=2.2 + 1.4 / rows, sharex=True)
    band, truth_label = f"{percent(data.interval)} interval", "Truth"
    for index, (ax, name, color, truth) in enumerate(
        zip(axes.flat, data.components, data.colors, data.truth)
    ):
        first = index == 0
        for number, segment in enumerate(data.segments):
            x = time_positions(ax, segment.dates)
            shade = {"color": color, "alpha": BAND_ALPHA}
            shade["label"] = band if first and number == 0 else None
            low, high = segment.lower[index], segment.upper[index]
            if len(x) == 1:  # one week has no width to fill: its interval is a bar
                ax.vlines(x, low, high, linewidth=_BAR_WIDTH, **shade)
            else:
                ax.fill_between(x, low, high, linewidth=0, **shade)
            ax.plot(
                x,
                segment.mean[index],
                color=color,
                linewidth=1.5,
                marker="o" if len(x) == 1 else None,
                label="Weekly mean" if first and number == 0 else None,
            )
        positions = time_positions(ax, data.dates)
        if truth is not None:
            ax.plot(
                positions,
                truth,
                color=ACTUAL,
                linestyle="--",
                linewidth=1.2,
                marker="o" if len(positions) == 1 else None,
                label=truth_label,
            )
            truth_label = None
        if data.first_query is not None:
            forecast_span(
                ax, positions, data.first_query, label="Forecast weeks" if first else None
            )
        ax.set_title(name, loc="left", fontsize=9)
        compact_axis(ax.yaxis)
        ax.grid(axis="y", alpha=0.15)
        if index + columns < count:  # the panel below labels the shared dates
            ax.tick_params(labelbottom=False)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    entries = {}
    for ax in axes.flat[:count]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, handle)
    fig.legend(
        list(entries.values()),
        list(entries),
        loc="outside lower center",
        ncols=len(entries),
        frameon=False,
        fontsize=9,
    )
    _suptitle(fig, data.title, *data.caption)
    return fig


def _matplotlib(data: Contributions, ax=None) -> Any:
    if data.stacked:
        return _matplotlib_stack(data, ax)
    return _matplotlib_panels(data)


def _plotly_stack(fig, data: Contributions) -> None:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    for number, segment in enumerate(data.segments):
        for name, color, mean, lower, upper in zip(
            data.components, data.colors, segment.mean, segment.lower, segment.upper, strict=True
        ):
            label = plotting.text(name)
            # "hv" holds each week's value flat up to the next edge; the last edge repeats it.
            edge = {"x": segment.edges, "mode": "lines", "line": {"width": 0, "shape": "hv"}}
            edge |= {"hoverinfo": "skip", "legendgroup": name}
            fig.add_trace(go.Scatter(y=np.append(lower, lower[-1]), showlegend=False, **edge))
            fig.add_trace(
                go.Scatter(
                    y=np.append(upper, upper[-1]),
                    fill="tonexty",
                    fillcolor=rgba(color, _STACK_ALPHA),
                    name=label,
                    showlegend=number == 0,
                    **edge,
                )
            )
            # Hovering a layer anywhere near its middle names it, with its weekly mean.
            fig.add_trace(
                go.Scatter(
                    x=segment.dates,
                    y=(lower + upper) / 2,
                    customdata=mean,
                    mode="markers",
                    marker={"opacity": 0, "color": color},
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%Y-%m-%d}}<br>{label}: mean %{{customdata:,.4~r}}<extra></extra>"
                    ),
                )
            )
    if data.outcome is not None:
        actual = plotting.text(f"Actual {data.target}")
        fig.add_trace(
            go.Scatter(
                x=data.dates,
                y=data.outcome,
                mode="lines+markers",
                line={"color": ACTUAL, "width": 1.2},
                marker={"size": 4},
                name=actual,
                legendgroup="actual",
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{actual} %{{y:,.4~r}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": "#4d4d4d", "width": 1})
    if data.first_query is not None:
        plotting.forecast_span(
            fig, data.dates, data.first_query, name="Forecast weeks", showlegend=True
        )
    fig.update_yaxes(title_text=plotting.text(data.target))


def _plotly_panels(fig, data: Contributions, columns: int) -> None:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    band = f"{percent(data.interval)} interval"
    truth_shown = False
    for index, (name, color, truth) in enumerate(
        zip(data.components, data.colors, data.truth, strict=True)
    ):
        where = {"row": index // columns + 1, "col": index % columns + 1}
        label, first = plotting.text(name), index == 0
        hover = (
            f"%{{x|%Y-%m-%d}}<br>{label}: mean %{{y:,.4~r}}<br>{band} "
            "%{customdata[0]:,.4~r} to %{customdata[1]:,.4~r}<extra></extra>"
        )
        for number, segment in enumerate(data.segments):
            low, high = segment.lower[index], segment.upper[index]
            single, shown = len(segment.dates) == 1, first and number == 0
            if single:  # one week has no width to fill: its interval is a bar
                fig.add_trace(
                    go.Scatter(
                        x=np.repeat(segment.dates, 2),
                        y=np.concatenate((low, high)),
                        mode="lines",
                        line={"color": rgba(color, BAND_ALPHA), "width": _BAR_WIDTH},
                        hoverinfo="skip",
                        name=band,
                        legendgroup="interval",
                        showlegend=shown,
                    ),
                    **where,
                )
            else:
                plotting.band(
                    fig,
                    segment.dates,
                    low,
                    high,
                    color=color,
                    name=band,
                    legendgroup="interval",
                    showlegend=shown,
                    **where,
                )
            fig.add_trace(
                go.Scatter(
                    x=segment.dates,
                    y=segment.mean[index],
                    customdata=np.column_stack((low, high)),
                    mode="lines+markers" if single else "lines",
                    line={"color": color, "width": 1.6},
                    name="Weekly mean",
                    legendgroup="mean",
                    showlegend=shown,
                    hovertemplate=hover,
                ),
                **where,
            )
        if truth is not None:
            fig.add_trace(
                go.Scatter(
                    x=data.dates,
                    y=truth,
                    mode="lines+markers" if len(data.dates) == 1 else "lines",
                    line={"color": ACTUAL, "width": 1.2, "dash": "dash"},
                    name="Truth",
                    legendgroup="truth",
                    showlegend=not truth_shown,
                    hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}: truth %{{y:,.4~r}}<extra></extra>",
                ),
                **where,
            )
            truth_shown = True
        if data.first_query is not None:
            plotting.forecast_span(
                fig,
                data.dates,
                data.first_query,
                name="Forecast weeks",
                legendgroup="forecast",
                showlegend=first,
                **where,
            )
    fig.update_xaxes(matches="x")  # every panel pans and zooms the same weeks


def _plotly(data: Contributions) -> Any:
    from . import _plotly as plotting

    if not data.stacked:
        count = len(data.components)
        columns = min(COLUMNS, count)
        rows = math.ceil(count / columns)
        fig = plotting.figure(
            rows,
            columns,
            shared_xaxes=True,
            specs=[
                [{} if row * columns + column < count else None for column in range(columns)]
                for row in range(rows)
            ],
            subplot_titles=[plotting.text(name) for name in data.components],
            horizontal_spacing=0.06,
            vertical_spacing=0.3 / rows,
        )
        _plotly_panels(fig, data, columns)
        fig.update_layout(showlegend=True)
        return plotting.finish(fig, title=data.title, caption=data.caption, height=210 * rows + 50)
    fig = plotting.figure()
    _plotly_stack(fig, data)
    fig = plotting.finish(fig, title=data.title, caption=data.caption, height=450)
    # One legend row cannot hold every component: list them down the right, as matplotlib does.
    fig.update_layout(
        legend={"orientation": "v", "yref": "paper", "y": 1, "x": 1.02, "xanchor": "left"}
    )
    return fig


def contributions_over_time(
    source: ModelSource,
    *,
    window: str | None = None,
    components: Sequence[str] | None = None,
    stacked: bool = True,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    backend: str | None = None,
    title: str | None = None,
    ax: Any = None,
) -> Any:
    check(backend, ax)
    if ax is not None and not stacked:
        raise ValueError(
            "stacked=False draws one panel per component; ax= draws only the single-axes "
            "stacked chart"
        )
    data = prepare(
        source,
        window=window,
        components=components,
        stacked=stacked,
        interval=interval,
        truth=truth,
        title=title,
    )
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly, ax=ax)
