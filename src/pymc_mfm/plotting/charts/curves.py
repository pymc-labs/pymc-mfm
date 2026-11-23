"""Each media covariate's contribution by its level (``model.plot.response_curves``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pymc_mfm.inference.curves import select_channels
from pymc_mfm.plotting.panels import COLORS, _fit, _suptitle

from ._backend import check, draw
from ._data import CurveSummary, summarize
from ._mpl import compact_axis, figure, figure_legend, percent_axis
from ._style import ACTUAL, BAND_ALPHA, CAVEAT, caption, compact, percent, rgba

if TYPE_CHECKING:
    from ._data import IdataReader

_TITLE = "Response curves: contribution by media covariate level"
#: Panels per row before the grid wraps.
_COLUMNS = 5
#: Figure width in inches and the font size of the notes under the panels (points).
_WIDTH = 11
_NOTE_SIZE = 9
_CURVE = COLORS["channel"]
#: The dashed line at the observed level.
_REFERENCE = "#737373"
_REFERENCE_LABEL = "Observed level (100%)"
_XLABEL = "Media covariate, % of observed"
_WARNING_LABEL = "Level has its own warning (see notes)"
#: Width of the interval bar at a level whose neighbours are missing, where a band has no area.
_BAR_WIDTH = 6


@dataclass(frozen=True, slots=True)
class Curves:
    """One panel per media covariate, in drawing order.

    ``media`` (the level times the observed total), ``mean``, ``low``, ``high``,
    ``warnings`` and ``truth`` are ``[channel, level]``; ``warnings`` holds each level's
    own hosted warnings, without those every drawn level-1 point shares (the unchanged
    plan's; ``""`` for none, always for local draws); ``truth`` is NaN where none was
    given and ``None`` without truth. ``actual`` indexes level 1 in ``levels``, if listed.
    ``notes`` go under the panels.
    """

    channels: tuple[str, ...]
    panel_titles: tuple[str, ...]
    levels: np.ndarray
    media: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    actual: int | None
    truth: np.ndarray | None
    warnings: np.ndarray
    band: str
    ylabel: str
    title: str
    caption: tuple[str, ...]
    notes: tuple[str, ...]


def prepare(
    source: IdataReader,
    *,
    channels: Sequence[str] | None = None,
    interval: float = 0.9,
    truth: pd.DataFrame | None = None,
    title: str | None = None,
) -> Curves:
    """Mean and central ``interval`` of the latest ``model.response_curves``, by media covariate."""
    stored = source.curve_summary(fallback=True)
    curves = source.response_curves() if stored is None else stored
    names = select_channels(channels, curves.channels)
    rows = (
        slice(None) if names == curves.channels else [curves.channels.index(name) for name in names]
    )
    missing_note = ()
    if isinstance(curves, CurveSummary):
        if interval != curves.interval:
            from pymc_mfm._product import NotSupported

            raise NotSupported(
                f"plot.response_curves: hosted curves only support interval={curves.interval}; "
                f"got {interval!r}"
            )
        mean, low, high = curves.mean[rows], curves.low[rows], curves.high[rows]
        media, totals = curves.media[rows], curves.total_media[rows]
        if np.isnan(mean).any():
            missing_note = ("Gaps mark levels with no hosted curve summary.",)
        warnings = curves.warnings[rows]
    else:
        mean, low, high = summarize(curves.draws[rows], interval, axis=2)
        media, totals = source.curve_media()[rows], curves.media[rows]
        warnings = np.full(mean.shape, "")
    own, notes = _warning_notes(names, curves.levels, warnings, mean)
    (actual,) = np.nonzero(curves.levels == 1.0)
    weeks = curves.weeks
    span = "a one-week plan" if weeks == 1 else f"a plan of the first {weeks} forecast weeks"
    return Curves(
        channels=names,
        panel_titles=tuple(
            f"{name} · media covariate total {compact(total)}"
            if total > 0
            else f"{name} · zero in these weeks"
            for name, total in zip(names, totals, strict=True)
        ),
        levels=curves.levels,
        media=media,
        mean=mean,
        low=low,
        high=high,
        actual=int(actual[0]) if actual.size else None,
        truth=_truth(truth, curves.levels, curves.channels, names),
        warnings=own,
        band=f"{percent(interval)} interval",
        ylabel=f"Contribution over {weeks} week{'s' if weeks > 1 else ''} (outcome)",
        title=title or _TITLE,
        caption=(
            caption(interval, curves.n_samples),
            f"Each media covariate varied separately over {span}; history and other inputs fixed.",
            *missing_note,
            CAVEAT,
        ),
        notes=notes,
    )


def _warning_notes(
    names: tuple[str, ...], levels: np.ndarray, warnings: np.ndarray, mean: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Each drawn level's own warnings, and the notes under the chart that list them all.

    Level 1 is the unchanged plan in every panel, so a warning that every drawn level-1
    point carries is about that plan, not one level: it gets a single ``Base plan`` note
    and no ring. Without a drawn level-1 point nothing is the plan's. Each other warning
    gets one note naming its channels, or ``all channels`` when every drawn panel has it.
    At most ``MAX_WARNING_LINES`` notes, as under ``client.plots.plot_curves``, then one
    counting the rest: a saved image keeps the warnings. Levels without a mean draw
    nothing, so their text is not listed.
    """
    present = ~np.isnan(mean)
    messages = {
        (row, column): [message for message in str(warnings[row, column]).split("\n") if message]
        for row, column in zip(*np.nonzero(present), strict=True)
    }
    base = [items for (_, column), items in messages.items() if levels[column] == 1.0]
    plan = set.intersection(*map(set, base)) if base else set()
    own = np.full(mean.shape, "", dtype=object)
    shared: dict[str, None] = {}
    found: dict[str, dict[str, None]] = {}
    for (row, column), items in messages.items():
        own[row, column] = "\n".join(message for message in items if message not in plan)
        for message in items:
            if message in plan:
                shared.setdefault(message, None)
            else:
                found.setdefault(message, {})[names[row]] = None
    notes = [f"Base plan: {message}" for message in shared]
    for message, where in found.items():
        label = "all channels" if len(names) > 1 and len(where) == len(names) else ", ".join(where)
        notes.append(f"Warning ({label}): {message}")
    if notes:
        from pymc_mfm.client.plots import MAX_WARNING_LINES

        if len(notes) > MAX_WARNING_LINES:
            rest = len(notes) - MAX_WARNING_LINES
            plural = "s" if rest > 1 else ""
            notes = [
                *notes[:MAX_WARNING_LINES],
                f"{rest} more warning{plural}: see /response_curves/summary/warning.",
            ]
    return own.astype(str), tuple(notes)


def _isolated(mean: np.ndarray) -> np.ndarray:
    """Levels with a mean but no neighbour with one, where a band cannot form an area."""
    present = np.pad(~np.isnan(mean), 1)
    return present[1:-1] & ~present[:-2] & ~present[2:]


def _flagged(data: Curves, index: int) -> np.ndarray:
    """Levels of panel ``index`` with a mean and a warning."""
    return (data.warnings[index] != "") & ~np.isnan(data.mean[index])


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
        raise ValueError("truth must hold numbers, indexed by level") from None
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
    # Wrap the notes narrower than the figure: the estimate can undercount wide letters.
    notes = _fit("\n".join(data.notes), 0.9 * _WIDTH, _NOTE_SIZE) if data.notes else None
    lines = notes.count("\n") + 1 if notes else 0
    # Allow room for title, caption and legend; the notes add their own height.
    height = 3.0 + 1.5 / rows + (lines * _NOTE_SIZE * 1.3 / 72 + 0.1 if lines else 0) / rows
    fig, axes = figure(rows, columns, width=_WIDTH, row_height=height, sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat):
        if index >= count:
            ax.set_visible(False)
            continue
        mean, low, high = data.mean[index], data.low[index], data.high[index]
        ax.plot(
            data.levels,
            mean,
            color=_CURVE,
            linewidth=2,
            marker="." if np.isnan(mean).any() else None,
            label="Mean",
        )
        ax.fill_between(
            data.levels, low, high, color=_CURVE, alpha=BAND_ALPHA, linewidth=0, label=data.band
        )
        isolated = _isolated(mean)
        if isolated.any():
            ax.vlines(
                data.levels[isolated],
                low[isolated],
                high[isolated],
                color=_CURVE,
                alpha=BAND_ALPHA,
                linewidth=_BAR_WIDTH,
            )
        if data.actual is not None and not np.isnan(mean[data.actual]):
            ax.scatter(
                data.levels[data.actual],
                mean[data.actual],
                color=_CURVE,
                edgecolor="white",
                zorder=3,
            )
        ax.axvline(1.0, color=_REFERENCE, linestyle="--", linewidth=1, label=_REFERENCE_LABEL)
        if data.truth is not None:
            ax.scatter(data.levels, data.truth[index], color=ACTUAL, s=14, zorder=4, label="Truth")
        flagged = _flagged(data, index)
        if flagged.any():
            from pymc_mfm.client.plots import WARNING_COLOR

            ax.scatter(
                data.levels[flagged],
                mean[flagged],
                s=70,
                facecolors="none",
                edgecolors=WARNING_COLOR,
                linewidths=1.5,
                zorder=5,
                label=_WARNING_LABEL,
            )
        ax.set_title(data.panel_titles[index], loc="left", fontsize=9)
        percent_axis(ax.xaxis)
        compact_axis(ax.yaxis)
        ax.grid(axis="y", alpha=0.15)
        if index % columns:  # the panels share the y axis, labelled on the left
            ax.tick_params(labelleft=False)
        if index + columns >= count:  # the lowest panel of its column
            ax.set_xlabel(_XLABEL, fontsize=9)
    fig.supylabel(data.ylabel, fontsize=9)
    figure_legend(fig, axes.flat, notes=notes)
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
    band_legend = warning_legend = True
    for index, name in enumerate(data.channels):
        where = {"row": index // columns + 1, "col": index % columns + 1}
        first = index == 0
        mean, low, high = data.mean[index], data.low[index], data.high[index]
        has_gaps = np.isnan(mean).any()
        fig.add_trace(
            go.Scatter(
                x=data.levels,
                y=mean,
                mode="lines+markers" if has_gaps else "lines",
                name="Mean",
                line={"color": _CURVE, "width": 2},
                customdata=np.column_stack((data.media[index], low, high)),
                hovertemplate=(
                    f"{plotting.text(name)} at %{{x:.1%}} of its observed level<br>"
                    "Media covariate %{customdata[0]:,.4~r}<br>"
                    f"Mean contribution %{{y:,.4~r}} · {data.band} "
                    "%{customdata[1]:,.4~r} to %{customdata[2]:,.4~r}<extra></extra>"
                ),
                legendgroup="mean",
                showlegend=first,
            ),
            **where,
        )
        # Plotly fills across missing values even when its lines have gaps.
        spans = (
            np.ma.clump_unmasked(np.ma.masked_invalid(mean, copy=False))
            if has_gaps
            else (slice(None),)
        )
        isolated = _isolated(mean)
        for span in spans:
            if isolated[span].all():
                continue  # one level: a band of it has no area, so it gets a bar below
            plotting.band(
                fig,
                data.levels[span],
                low[span],
                high[span],
                color=_CURVE,
                name=data.band,
                legendgroup="interval",
                showlegend=band_legend,
                **where,
            )
            band_legend = False
        if isolated.any():
            # Vertical bars from low to high, one per level, split by NaN gaps.
            levels = data.levels[isolated]
            gaps = np.full(len(levels), np.nan)
            fig.add_trace(
                go.Scatter(
                    x=np.column_stack((levels, levels, gaps)).ravel(),
                    y=np.column_stack((low[isolated], high[isolated], gaps)).ravel(),
                    mode="lines",
                    line={"color": rgba(_CURVE, BAND_ALPHA), "width": _BAR_WIDTH},
                    hoverinfo="skip",
                    name=data.band,
                    legendgroup="interval",
                    showlegend=band_legend,
                ),
                **where,
            )
            band_legend = False
        if data.actual is not None and not np.isnan(mean[data.actual]):
            fig.add_trace(
                go.Scatter(
                    x=[data.levels[data.actual]],
                    y=[mean[data.actual]],
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
                        f"Truth for {plotting.text(name)} at %{{x:.1%}} of its observed level<br>"
                        "%{y:,.4~r}<extra></extra>"
                    ),
                    legendgroup="truth",
                    showlegend=first,
                ),
                **where,
            )
        flagged = _flagged(data, index)
        if flagged.any():
            from pymc_mfm.client.plots import WARNING_COLOR

            fig.add_trace(
                go.Scatter(
                    x=data.levels[flagged],
                    y=mean[flagged],
                    mode="markers",
                    name=_WARNING_LABEL,
                    marker={
                        "symbol": "circle-open",
                        "color": WARNING_COLOR,
                        "size": 12,
                        "line": {"width": 2},
                    },
                    hovertext=[
                        plotting.text(text).replace("\n", "<br>")
                        for text in data.warnings[index][flagged]
                    ],
                    hovertemplate=(
                        f"Warning for {plotting.text(name)} at %{{x:.1%}} of its observed level"
                        "<br>%{hovertext}<extra></extra>"
                    ),
                    legendgroup="warning",
                    showlegend=warning_legend,
                ),
                **where,
            )
            warning_legend = False
        if index + columns >= count:  # the lowest panel of its column
            fig.update_xaxes(title={"text": _XLABEL, "font": {"size": 11}}, **where)
    # Narrow panels get fewer ticks, which then stay level instead of rotating.
    fig.update_xaxes(tickformat=".0%", nticks=4 if columns > 2 else 0)
    fig.update_yaxes(title={"text": plotting.text(data.ylabel), "font": {"size": 11}}, col=1)
    return plotting.finish(
        fig, title=data.title, caption=data.caption, height=300 * rows + 50, notes=data.notes
    )


def response_curves(
    source: IdataReader,
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
