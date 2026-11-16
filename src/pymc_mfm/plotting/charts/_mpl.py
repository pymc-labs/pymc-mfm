"""Matplotlib pieces shared by the charts; only renderers call them, so matplotlib loads late."""

from __future__ import annotations

import numpy as np

from pymc_mfm.plotting.panels import _figure, _plot_time

from ._style import FORECAST, FORECAST_ALPHA, compact


def figure(rows, columns=1, *, width=11.0, row_height=3.2, sharex=False, sharey=False):
    """A pyplot-free figure and its ``[rows, columns]`` axes, optionally sharing axes."""
    fig, axes = _figure(rows, columns, width=width, row_height=row_height)
    first, *others = axes.flat
    for ax in others:
        if sharex:
            ax.sharex(first)
        if sharey:
            ax.sharey(first)
    return fig, axes


def time_positions(ax, dates: np.ndarray) -> np.ndarray:
    """Positions of ``dates`` on ``ax``, whose x axis then labels them as dates."""
    return np.asarray(_plot_time(ax, dates), dtype=float)


def _half_step(positions: np.ndarray) -> float:
    return float(np.median(np.diff(positions))) / 2 if len(positions) > 1 else 0.5


def forecast_span(ax, positions: np.ndarray, first: int, *, label: str | None = "Forecast weeks"):
    """Shade the weeks from ``positions[first]`` on: the weeks the model did not see."""
    half = _half_step(positions)
    ax.axvspan(
        positions[first] - half,
        positions[-1] + half,
        color=FORECAST,
        alpha=FORECAST_ALPHA,
        linewidth=0,
        label=label,
    )


def compact_axis(axis) -> None:
    """Label ``axis`` ticks as ``12.5k``, ``3.2M`` and so on."""
    from matplotlib.ticker import FuncFormatter

    axis.set_major_formatter(FuncFormatter(lambda value, _: compact(value)))


def percent_axis(axis) -> None:
    """Label ``axis`` ticks, given as fractions, as percentages."""
    from matplotlib.ticker import PercentFormatter

    axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
