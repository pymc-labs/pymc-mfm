"""Each channel's and control's share of the expected outcome (``model.plot.shares``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mktfm.plotting.panels import COLORS, _single, _title

from ._backend import check, draw
from ._data import interval_quantiles, summarize, window_weeks
from ._mpl import percent_axis
from ._style import CAVEAT, caption, percent, rgba, series_colors

if TYPE_CHECKING:
    from ._data import IdataReader, Window

SHARES = ("mean", "total")
VIEWS = ("bars", "dist")
#: Points of the grid every kernel density is evaluated on.
GRID_POINTS = 400
#: Bandwidth, as a fraction of ``max(1, |share|)``, when every draw has the same share.
FLAT_BANDWIDTH = 1e-3
_DRIVERS = ("channel", "control")
_GROUPS = {"channel": "media covariates", "control": "control covariates"}
_BAR_HEIGHT = 0.6
_FILL_ALPHA = 0.25
_DIST_NOTE = "Kernel densities, each scaled to a peak of 1"


@dataclass(frozen=True, slots=True)
class Shares:
    """Each series' per-draw share of the expected outcome and its summary, in drawing order.

    The series are the channels and controls. ``shares`` is ``[kept draw, series]``, as
    fractions of the expected outcome; ``mean``, ``low`` and ``high`` summarize it over
    the kept draws. ``left_out_draws`` and ``left_out_draw_weeks`` count what was left
    out because its expected outcome was not positive. The ``dist`` view adds the shared
    ``grid`` and each series' kernel density on it, ``density`` ``[series, point]``,
    scaled to peak at 1. ``caption`` holds the caption's lines.
    """

    view: str
    names: tuple[str, ...]
    families: tuple[str, ...]
    colors: tuple[str, ...]
    shares: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    interval: float
    left_out_draws: int
    left_out_draw_weeks: int
    grid: np.ndarray | None
    density: np.ndarray | None
    xlabel: str
    caption: tuple[str, ...]
    title: str

    @property
    def separator(self) -> float | None:
        """Bar position of the dashed line between channels and controls, when both show."""
        channels = self.families.count("channel")
        return channels - 0.5 if 0 < channels < len(self.names) else None


def _per_draw(data: Window, drivers: list[int], share: str) -> tuple[np.ndarray, int, int]:
    """Kept per-draw shares ``[draw, driver]``, then the left-out draws and draw-weeks."""
    expected = data.expected
    parts = data.draws[:, :, drivers]
    if share == "total":
        totals = expected.sum(axis=1)
        kept = totals > 0
        return parts[kept].sum(axis=1) / totals[kept, None], int(np.count_nonzero(~kept)), 0
    positive = expected > 0
    weeks = positive.sum(axis=1)
    kept = weeks > 0
    ratios = np.divide(
        parts, expected[:, :, None], out=np.zeros(parts.shape), where=positive[:, :, None]
    )
    return (
        ratios[kept].sum(axis=1) / weeks[kept, None],
        int(np.count_nonzero(~kept)),
        int(np.count_nonzero(~positive)),
    )


def _bandwidth(values: np.ndarray) -> float:
    """Gaussian kernel bandwidth of ``values``: Scott's rule, sample ``std * n ** (-1/5)``.

    When every value is the same, ``FLAT_BANDWIDTH * max(1, |value|)``.
    """
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if spread > 0:
        return spread * len(values) ** -0.2
    return FLAT_BANDWIDTH * max(1.0, abs(float(values[0])))


def _densities(shares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A shared grid and each column's Gaussian kernel density on it, scaled to peak at 1.

    The grid runs three of the widest bandwidths past the extreme shares. Kernels are
    summed in log space, so a density narrower than the grid step still peaks at 1.
    """
    bandwidths = [_bandwidth(column) for column in shares.T]
    reach = 3 * max(bandwidths)
    grid = np.linspace(shares.min() - reach, shares.max() + reach, GRID_POINTS)
    density = np.empty((shares.shape[1], GRID_POINTS))
    for row, (column, bandwidth) in enumerate(zip(shares.T, bandwidths, strict=True)):
        exponent = -0.5 * ((grid[:, None] - column) / bandwidth) ** 2
        top = exponent.max(axis=1)
        log_density = top + np.log(np.exp(exponent - top[:, None]).sum(axis=1))
        density[row] = np.exp(log_density - log_density.max())
    return grid, density


def _left_out(data: Window, share: str, draws: int, draw_weeks: int) -> tuple[str, ...]:
    """Caption lines saying what was left out of the shares, and why."""
    samples, weeks = data.draws.shape[:2]
    lines = ()
    if draw_weeks:
        lines += (
            f"{draw_weeks:,} of {samples * weeks:,} draw-weeks left out: expected outcome ≤ 0",
        )
    if draws:
        why = (
            "total expected outcome ≤ 0"
            if share == "total"
            else "expected outcome ≤ 0 in every week"
        )
        lines += (f"{draws:,} of {samples:,} draws left out: {why}",)
    return lines


def prepare_window(
    data: Window, *, share: str, view: str, interval: float, title: str | None = None
) -> Shares:
    """Every number the chart of ``data`` draws; :func:`prepare` checks the arguments.

    The expected outcome of a draw and week is the sum of every component.
    ``share="total"`` divides each series' window total by the expected total, per
    draw; ``"mean"`` averages each week's share over the weeks with a positive expected
    outcome. A draw whose expected total (``"total"``) or every week (``"mean"``) is not
    positive is left out, as is each such draw-week from the ``"mean"`` average. Bars
    run channels, then controls, each largest mean first; ``dist`` keeps the
    component order and draws each series' per-draw shares as a Gaussian kernel
    density, with Scott's bandwidth or, when every draw has the same share,
    ``FLAT_BANDWIDTH * max(1, |share|)``.
    """
    drivers = [index for index, family in enumerate(data.families) if family in _DRIVERS]
    shares, left_out_draws, left_out_draw_weeks = _per_draw(data, drivers, share)
    weeks = window_weeks(data)
    if not len(shares):
        expected = "total expected" if share == "total" else "expected"
        raise ValueError(
            f"every draw's {expected} outcome is ≤ 0 over the {weeks}, so no share is defined"
        )
    mean, low, high = summarize(shares, interval)
    names = tuple(data.components[index] for index in drivers)
    families = tuple(data.families[index] for index in drivers)
    if view == "bars":
        # Channels, then controls; each largest mean first, ties in component order.
        order = np.lexsort((-mean, [_DRIVERS.index(family) for family in families]))
        names = tuple(names[index] for index in order)
        families = tuple(families[index] for index in order)
        shares, mean, low, high = shares[:, order], mean[order], low[order], high[order]
        colors = tuple(COLORS[family] for family in families)
        grid = density = None
    else:
        palette = series_colors(names)
        colors = tuple(palette[name] for name in names)
        grid, density = _densities(shares)
    # The distribution view draws densities, not intervals: its caption names only the draws.
    if view == "bars":
        lead = (caption(interval, len(shares), pointwise=False), CAVEAT)
    else:
        lead = (f"{len(shares):,} draws · expected outcome, without observation noise", CAVEAT)
        lead += (_DIST_NOTE,)
    notes = _left_out(data, share, left_out_draws, left_out_draw_weeks)
    return Shares(
        view=view,
        names=names,
        families=families,
        colors=colors,
        shares=shares,
        mean=mean,
        low=low,
        high=high,
        interval=interval,
        left_out_draws=left_out_draws,
        left_out_draw_weeks=left_out_draw_weeks,
        grid=grid,
        density=density,
        xlabel=(
            f"Average of each week's share of the expected outcome, over {weeks}"
            if share == "mean"
            else f"Share of the expected outcome over the {weeks}"
        ),
        caption=(*lead, *notes),
        title=title or "Share of the expected outcome by media and control covariate",
    )


def prepare(
    source: IdataReader,
    *,
    share: str = "mean",
    window: str = "context",
    view: str = "bars",
    interval: float = 0.9,
    title: str | None = None,
) -> Shares:
    """Check every argument, then the numbers of ``window``; see :func:`prepare_window`.

    Both windows read the last complete inference without sampling.
    """
    if share not in SHARES:
        raise ValueError(f"share must be one of {SHARES}, got {share!r}")
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    interval_quantiles(interval)
    data = source.window(window)
    return prepare_window(data, share=share, view=view, interval=interval, title=title)


def _matplotlib_bars(ax, data: Shares) -> None:
    positions = np.arange(len(data.names))
    for family, label in _GROUPS.items():
        rows = [row for row, name in enumerate(data.families) if name == family]
        if rows:
            ax.barh(
                positions[rows],
                data.mean[rows],
                height=_BAR_HEIGHT,
                color=[data.colors[row] for row in rows],
                label=label,
            )
    # Centred on the interval, so it spans low..high even when the mean lies outside.
    ax.errorbar(
        (data.low + data.high) / 2,
        positions,
        xerr=(data.high - data.low) / 2,
        fmt="none",
        ecolor="0.15",
        elinewidth=1.1,
        capsize=3,
    )
    ax.axvline(0, color="0.3", linewidth=0.8)
    if data.separator is not None:
        ax.axhline(data.separator, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_yticks(positions, data.names, fontsize=9)
    ax.invert_yaxis()  # the first series on top
    ax.grid(axis="x", alpha=0.15)


def _matplotlib_dist(ax, data: Shares) -> None:
    for name, color, density in zip(data.names, data.colors, data.density, strict=True):
        ax.fill_between(data.grid, density, color=color, alpha=_FILL_ALPHA, linewidth=0)
        ax.plot(data.grid, density, color=color, linewidth=1.4, label=name)
    ax.set_ylim(0, 1.06)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)


def _matplotlib(data: Shares, ax=None) -> Any:
    bars = data.view == "bars"
    # Room for the title and each caption line above the axes.
    height = (1.4 + 0.35 * len(data.names) if bars else 3.0) + 0.22 * (1 + len(data.caption))
    fig, ax, owned = _single(ax, width=10, row_height=height)
    (_matplotlib_bars if bars else _matplotlib_dist)(ax, data)
    percent_axis(ax.xaxis)
    ax.set_xlabel(data.xlabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _title(fig, ax, "\n".join(data.caption), data.title, owned)
    return fig


def _plotly(data: Shares) -> Any:
    import plotly.graph_objects as go

    from . import _plotly as plotting

    fig = plotting.figure()
    interval = f"{percent(data.interval)} interval"
    if data.view == "bars":
        positions = np.arange(len(data.names))
        hover = (
            "%{hovertext}<br>mean %{x:.1%}<br>"
            f"{interval} %{{customdata[0]:.1%}} to %{{customdata[1]:.1%}}<extra></extra>"
        )
        for family, label in _GROUPS.items():
            rows = [row for row, name in enumerate(data.families) if name == family]
            if not rows:
                continue
            fig.add_trace(
                go.Bar(
                    x=data.mean[rows],
                    y=positions[rows],
                    orientation="h",
                    width=_BAR_HEIGHT,
                    marker_color=[data.colors[row] for row in rows],
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": (data.high - data.mean)[rows],
                        "arrayminus": (data.mean - data.low)[rows],
                        "color": "#262626",
                        "thickness": 1.1,
                        "width": 4,
                    },
                    customdata=np.column_stack((data.low, data.high))[rows],
                    hovertext=[plotting.text(data.names[row]) for row in rows],
                    hovertemplate=hover,
                    name=label,
                    legendgroup=family,
                )
            )
        if data.separator is not None:
            fig.add_hline(y=data.separator, line={"color": "#808080", "dash": "dash", "width": 1})
        fig.add_vline(x=0, line={"color": "#4d4d4d", "width": 1})
        fig.update_yaxes(
            tickvals=positions,
            ticktext=[plotting.text(name) for name in data.names],
            autorange="reversed",
            showgrid=False,
        )
        fig.update_layout(barmode="overlay")
        height = 40 * len(data.names) + 120
    else:
        for name, color, density, mean, low, high in zip(
            data.names, data.colors, data.density, data.mean, data.low, data.high, strict=True
        ):
            label = plotting.text(name)
            fig.add_trace(
                go.Scatter(
                    x=data.grid,
                    y=density,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=rgba(color, _FILL_ALPHA),
                    line={"color": color, "width": 1.5},
                    name=label,
                    legendgroup=name,
                    hovertemplate=(
                        f"{label}<br>share %{{x:.1%}}<br>"
                        f"mean {mean:.1%} · {interval} {low:.1%} to {high:.1%}<extra></extra>"
                    ),
                )
            )
        fig.update_yaxes(visible=False, range=[0, 1.06])
        height = 340
    fig.update_xaxes(tickformat=".0%", title_text=plotting.text(data.xlabel), zeroline=False)
    fig = plotting.finish(fig, title=data.title, caption=data.caption, height=height)
    if data.view == "dist":  # one entry per series: a column at the right, as in matplotlib
        fig.update_layout(
            legend={
                "orientation": "v",
                "yref": "paper",
                "y": 1,
                "yanchor": "top",
                "x": 1.02,
                "xanchor": "left",
            }
        )
    return fig


def shares(
    source: IdataReader,
    *,
    share: str = "mean",
    window: str = "context",
    view: str = "bars",
    interval: float = 0.9,
    backend: str | None = None,
    title: str | None = None,
    ax: Any = None,
) -> Any:
    check(backend, ax)
    data = prepare(source, share=share, window=window, view=view, interval=interval, title=title)
    return draw(data, backend=backend, matplotlib=_matplotlib, plotly=_plotly, ax=ax)
