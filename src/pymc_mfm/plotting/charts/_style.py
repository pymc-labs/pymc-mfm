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
#: Matplotlib's ``tab20`` (also D3's) without its two greys, which stay the baseline's,
#: then ``tab20b``: 38 colors, the same in both backends.
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#5254a3",
    "#8ca252",
    "#bd9e39",
    "#ad494a",
    "#a55194",
    "#6b6ecf",
    "#b5cf6b",
    "#e7ba52",
    "#d6616b",
    "#ce6dbd",
    "#9c9ede",
    "#cedb9c",
    "#e7cb94",
    "#e7969c",
    "#de9ed6",
)
#: Shown with contributions; :func:`caption` already says the draws exclude observation noise.
CAVEAT = "Conditional contributions, not causal lift."


def series_colors(names: Iterable[str]) -> dict[str, str]:
    """One palette color per name, in order; the palette repeats past 38 names."""
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(names)}


def rgba(color: str, alpha: float) -> str:
    """A ``#rrggbb`` color as a CSS ``rgba()`` string with ``alpha``."""
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def panel_grid(count: int, columns: int = 2) -> tuple[int, int]:
    """Rows and columns of ``count`` panels, at most ``columns`` to a row."""
    columns = min(columns, count)
    return math.ceil(count / columns), columns


def compact(value: float) -> str:
    """A short label: ``950``, ``12.5k``, ``3.2M``, ``1.1B``; signs kept.

    The value is rounded to three significant digits before its unit is chosen, so
    ``999_999`` reads ``1M``, not ``1e+03k``.
    """
    if not math.isfinite(value):
        return str(value)
    rounded = float(f"{value:.3g}")
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(rounded) >= limit:
            return f"{rounded / limit:.3g}{suffix}"
    return f"{rounded:.3g}"


def percent(fraction: float) -> str:
    """``fraction`` as a percent with up to three significant digits: ``90%``, ``97.5%``."""
    return f"{fraction * 100:.3g}%"


def caption(interval: float, n_samples: int, *, pointwise: bool = True) -> str:
    """How to read the draws a chart summarizes.

    ``pointwise`` intervals are drawn per week or per level, not as a joint band; a
    chart of one quantity per series (a share, a window total) passes ``False``.
    """
    kind = "pointwise intervals" if pointwise else "central intervals"
    return (
        f"{percent(interval)} {kind} over {n_samples:,} draws · "
        "expected outcome, without observation noise"
    )
