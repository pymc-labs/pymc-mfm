"""Colors, number formats and captions shared by both backends."""

from __future__ import annotations

import math
from collections.abc import Iterable

from pymc_mfm.plotting.panels import COLORS

ACTUAL = "#000000"
EXPECTED = COLORS["channel"]
FORECAST = "#D55E00"
FORECAST_ALPHA = 0.08
BAND_ALPHA = 0.22
#: Matplotlib's ``tab20``, which is also D3's: both backends color a series alike.
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
)
#: Shown with contributions; :func:`caption` already says the draws exclude observation noise.
CAVEAT = "Conditional contributions, not causal lift."


def series_colors(names: Iterable[str]) -> dict[str, str]:
    """One palette color per name, in order; the palette repeats past 20 names."""
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(names)}


def rgba(color: str, alpha: float) -> str:
    """A ``#rrggbb`` color as a CSS ``rgba()`` string with ``alpha``."""
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def compact(value: float) -> str:
    """A short label: ``950``, ``12.5k``, ``3.2M``, ``1.1B``; signs kept."""
    if not math.isfinite(value):
        return str(value)
    size = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if size >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"


def caption(interval: float, n_samples: int, *, pointwise: bool = True) -> str:
    """How to read the draws a chart summarizes.

    ``pointwise`` intervals are drawn per week or per level, not as a joint band; a
    chart of one quantity per series (a share, a window total) passes ``False``.
    """
    kind = "pointwise intervals" if pointwise else "central intervals"
    return (
        f"{interval:.0%} {kind} over {n_samples:,} draws · "
        "expected outcome, without observation noise"
    )
