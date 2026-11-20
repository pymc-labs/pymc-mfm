"""How well the expected outcome follows the observed outcome (``model.plot.fit``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.plotting.panels import _figure, _suptitle

from ._backend import draw
from ._data import interval_quantiles, summarize
from ._mpl import compact_axis, forecast_span, time_positions
from ._style import ACTUAL, BAND_ALPHA, EXPECTED, caption, rgba

if TYPE_CHECKING:
    from ._data import ModelSource, Window

_GROUPS = {"context": "History (context)", "query": "Forecast (query)"}
_CARD_FACE, _CARD_EDGE, _CARD_LABEL = "#f6f6f6", "#d9d9d9", "#555555"


@dataclass(frozen=True, slots=True)
class Segment:
    """One window's expected outcome by week: the mean and central interval of its draws."""

    window: str
    dates: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True, slots=True)
class Score:
    """How well one window's expected outcome follows its ``weeks`` finite observed weeks.

    ``mape`` and ``r2`` score the mean line; ``mape_sd`` and ``r2_sd`` are the
    standard deviation over draws of each draw's score. MAPE averages
    ``|y - mean| / |y|`` over the weeks whose outcome is not zero (``None`` without
    one); R² is ``1 - SS_res / SS_tot`` (``None`` for a constant outcome, whose
    ``SS_tot`` is zero). ``inside`` counts the weeks with ``low <= y <= high``.
    """

    window: str
    weeks: int
    mape: float | None
    mape_sd: float | None
    r2: float | None
    r2_sd: float | None
    inside: int

    @property
    def coverage(self) -> float:
        """Share of the observed weeks inside the interval."""
        return self.inside / self.weeks

    @property
    def label(self) -> str:
        """The heading of this window's scores: ``"History (context), 24 weeks"``."""
        return f"{_GROUPS[self.window]}, {self.weeks} weeks"


@dataclass(frozen=True, slots=True)
class Fit:
    """Every number the chart draws.

    ``segments`` are the drawn windows in time order over ``dates``, each its own
    line and band. ``observed`` is the outcome on ``dates`` (NaN where unobserved),
    or ``None`` when no drawn week was observed. ``first_query`` indexes the first
    forecast week in ``dates`` when both windows are drawn. ``scores`` has one
    entry per window with at least two observed weeks.
    """

    dates: np.ndarray
    segments: tuple[Segment, ...]
    observed: np.ndarray | None
    first_query: int | None
    scores: tuple[Score, ...]
    target: str
    interval: float
    title: str
    caption: str


def _mape(y: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Mean absolute percentage error of ``fit`` (``[..., week]``) against ``y``."""
    return np.mean(np.abs(y - fit) / np.abs(y), axis=-1)


def _r2(y: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """``1 - SS_res / SS_tot`` of ``fit`` (``[..., week]``) against ``y``."""
    return 1 - np.sum((y - fit) ** 2, axis=-1) / np.sum((y - y.mean()) ** 2)


def _score(
    window: Window, expected: np.ndarray, mean: np.ndarray, low: np.ndarray, high: np.ndarray
) -> Score | None:
    """``window``'s score over its finite observed weeks; ``None`` with fewer than two."""
    if window.outcome is None:
        return None
    seen = np.isfinite(window.outcome)
    weeks = int(seen.sum())
    if weeks < 2:
        return None
    y, draws, mean = window.outcome[seen], expected[:, seen], mean[seen]
    mape = mape_sd = r2 = r2_sd = None
    if (nonzero := y != 0).any():
        mape = float(_mape(y[nonzero], mean[nonzero]))
        mape_sd = float(_mape(y[nonzero], draws[:, nonzero]).std())
    if np.ptp(y) > 0:  # SS_tot is zero only for a constant outcome
        r2, r2_sd = float(_r2(y, mean)), float(_r2(y, draws).std())
    inside = int(np.count_nonzero((low[seen] <= y) & (y <= high[seen])))
    return Score(window.name, weeks, mape, mape_sd, r2, r2_sd, inside)


def prepare(
    source: ModelSource,
    *,
    window: str | None = None,
    interval: float = 0.9,
    title: str | None = None,
) -> Fit:
    """Each window's expected-outcome draws, summarized and scored against the outcome.

    ``interval`` is checked before any window is read, and ``window="query"``
    never samples the context.
    """
    interval_quantiles(interval)
    windows = source.windows(window)
    segments, scores = [], []
    for part in windows:
        expected = part.expected
        mean, low, high = summarize(expected, interval)
        segments.append(Segment(part.name, part.dates, mean, low, high))
        if (score := _score(part, expected, mean, low, high)) is not None:
            scores.append(score)
    observed = None
    if any(part.outcome is not None for part in windows):
        observed = np.concatenate(
            [
                np.full(len(part.dates), np.nan) if part.outcome is None else part.outcome
                for part in windows
            ]
        )
    first, target = windows[0], windows[0].target
    return Fit(
        dates=np.concatenate([part.dates for part in windows]),
        segments=tuple(segments),
        observed=observed,
        first_query=len(first.dates) if len(windows) > 1 else None,
        scores=tuple(scores),
        target=target,
        interval=interval,
        title=title or f"How well the expected outcome follows actual {target}",
        caption=caption(interval, first.n_samples),
    )


def _metrics(score: Score, interval: float) -> tuple[tuple[str, str], ...]:
    """``(name, value)`` text of each metric both backends show; ``""`` when undefined."""
    return (
        ("MAPE", "" if score.mape is None else f"{score.mape:.1%} ± {score.mape_sd:.1%}"),
        ("R²", "" if score.r2 is None else f"{score.r2:.2f} ± {score.r2_sd:.2f}"),
        (
            f"Inside {interval:.0%} interval",
            f"{score.coverage:.0%} ({score.inside} of {score.weeks})",
        ),
    )


def _cards(strip, data: Fit) -> None:
    """One card of metrics per scored window, side by side in ``strip``."""
    strip.axis("off")
    place = {
        "xy": (0, 1),
        "xycoords": "axes fraction",
        "textcoords": "offset points",
        "va": "top",
        "fontsize": 9,
    }
    for index, score in enumerate(data.scores):
        card = strip.inset_axes((index * 0.505, 0, 0.495, 1))
        card.set_xticks([])
        card.set_yticks([])
        card.set_facecolor(_CARD_FACE)
        for spine in card.spines.values():
            spine.set_color(_CARD_EDGE)
        card.annotate(score.label, xytext=(10, -8), fontweight="bold", **place)
        shown = [(name, value) for name, value in _metrics(score, data.interval) if value]
        for row, (name, value) in enumerate(shown, start=1):
            card.annotate(name, xytext=(10, -8 - 16 * row), color=_CARD_LABEL, **place)
            card.annotate(value, xytext=(150, -8 - 16 * row), **place)


def _matplotlib(data: Fit) -> Any:
    if data.scores:
        fig, axes = _figure(2, 1, width=11, height=5.8, height_ratios=(4, 1.4))
        _cards(axes[1, 0], data)
    else:
        fig, axes = _figure(1, 1, width=11, row_height=4.4)
    ax = axes[0, 0]
    positions = time_positions(ax, data.dates)
    start = 0
    for index, segment in enumerate(data.segments):
        x = positions[start : start + len(segment.dates)]
        start += len(x)
        first, single = index == 0, len(x) == 1
        label = f"{data.interval:.0%} interval" if first else None
        band = {"color": EXPECTED, "alpha": BAND_ALPHA, "label": label}
        if single:  # one week has no width to fill: its interval is a bar
            ax.vlines(x, segment.low, segment.high, linewidth=8, **band)
        else:
            ax.fill_between(x, segment.low, segment.high, linewidth=0, **band)
        ax.plot(
            x,
            segment.mean,
            color=EXPECTED,
            linewidth=1.8,
            marker="o" if single else None,
            label="Expected" if first else None,
        )
    if data.observed is not None:
        ax.plot(
            positions,
            data.observed,
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
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=4, frameon=False, fontsize=8)
    _suptitle(fig, data.title, data.caption)
    return fig


def _plotly(data: Fit) -> Any:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    if data.scores:
        table = {"specs": [[{"type": "xy"}], [{"type": "table"}]], "row_heights": [0.76, 0.24]}
        fig = plotting.figure(2, 1, vertical_spacing=0.1, **table)
    else:
        fig = plotting.figure()
    band = f"{data.interval:.0%} interval"
    hover = (
        f"%{{x|%Y-%m-%d}}<br>Expected %{{y:,.4~r}}<br>{band} "
        "%{customdata[0]:,.4~r} to %{customdata[1]:,.4~r}<extra></extra>"
    )
    for index, segment in enumerate(data.segments):
        first, single = index == 0, len(segment.dates) == 1
        if single:  # one week has no width to fill: its interval is a bar
            fig.add_trace(
                go.Scatter(
                    x=np.repeat(segment.dates, 2),
                    y=np.concatenate((segment.low, segment.high)),
                    mode="lines",
                    line={"color": rgba(EXPECTED, BAND_ALPHA), "width": 8},
                    hoverinfo="skip",
                    name=band,
                    legendgroup="interval",
                    showlegend=first,
                ),
                row=1,
                col=1,
            )
        else:
            plotting.band(
                fig,
                segment.dates,
                segment.low,
                segment.high,
                color=EXPECTED,
                name=band,
                legendgroup="interval",
                showlegend=first,
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=segment.dates,
                y=segment.mean,
                customdata=np.column_stack((segment.low, segment.high)),
                mode="lines+markers" if single else "lines",
                line={"color": EXPECTED, "width": 2},
                name="Expected",
                legendgroup="expected",
                showlegend=first,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )
    if data.observed is not None:
        actual = plotting.text(f"Actual {data.target}")
        fig.add_trace(
            go.Scatter(
                x=data.dates,
                y=data.observed,
                mode="lines+markers",
                line={"color": ACTUAL, "width": 1.2},
                marker={"size": 4},
                name=actual,
                legendgroup="actual",
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{actual} %{{y:,.4~r}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if data.first_query is not None:
        plotting.forecast_span(
            fig, data.dates, data.first_query, row=1, col=1, name="Forecast weeks", showlegend=True
        )
    if data.scores:
        rows = [
            (score.label, *(value for _, value in _metrics(score, data.interval)))
            for score in data.scores
        ]
        names = [name for name, _ in _metrics(data.scores[0], data.interval)]
        fig.add_trace(
            go.Table(
                header={"values": ["Window", *names], "align": "left", "fill_color": "#f0f0f0"},
                cells={"values": [list(column) for column in zip(*rows)], "align": "left"},
                columnwidth=[2.2, 1.6, 1.6, 1.6],
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text=plotting.text(data.target), row=1, col=1)
    height = 560 if data.scores else 410
    return plotting.finish(fig, title=data.title, caption=data.caption, height=height)


def fit(
    source: ModelSource,
    *,
    window: str | None = None,
    interval: float = 0.9,
    backend: str | None = None,
    title: str | None = None,
) -> Any:
    data = prepare(source, window=window, interval=interval, title=title)
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly)
