"""Plots of an analysis result table. They need matplotlib (the ``plot`` extra)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CHANNEL_COLOR = "#1f77b4"
CAVEAT = (
    "Each channel varied separately; history and other inputs fixed. "
    "Conditional contributions, not causal lift; observation noise excluded."
)


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Plots need matplotlib. Install it with: pip install "pymc-mfm[plot]"'
        ) from error
    return plt


def curve_points(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of each response curve, with the unrun level 1 filled in from its base.

    One row per channel, base plan and level, with the channel's own ``mean``, ``p05`` and
    ``p95``, in the order channels first appear in the table. Failed runs are dropped.
    """
    curves = table[(table["kind"] == "curve") & (table["component"] == table["varied_channel"])]
    columns = ["varied_channel", "base_scenario_id", "level", "spend_total", "mean", "p05", "p95"]
    points = curves[columns].copy()
    for (channel, base), group in points.groupby(["varied_channel", "base_scenario_id"]):
        if (group["level"] == 1).any():
            continue
        base_row = table[(table["scenario_id"] == base) & (table["component"] == channel)]
        if base_row.empty:
            continue
        row = base_row.iloc[0]
        points.loc[len(points)] = [
            channel,
            base,
            1.0,
            row["spend_total"],
            *row[["mean", "p05", "p95"]],
        ]
    points = points.dropna(subset=["mean"])
    position = {name: i for i, name in enumerate(dict.fromkeys(curves["varied_channel"]))}
    points["channel_position"] = points["varied_channel"].map(position)
    points = points.sort_values(["channel_position", "base_scenario_id", "level"])
    return points.drop(columns="channel_position").reset_index(drop=True)


def plot_curves(table: pd.DataFrame, *, title: str | None = None) -> Any:
    """One panel per channel: mean and 90% pointwise band of its contribution by spend level."""
    plt = _pyplot()
    points = curve_points(table)
    if points.empty:
        raise ValueError("The result has no curve rows. Ask for curves or use client.explore().")
    channels = list(dict.fromkeys(points["varied_channel"]))
    figure, axes = plt.subplots(
        1,
        len(channels),
        figsize=(6 * len(channels), 5),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for axis, channel in zip(axes[0], channels, strict=True):
        shown = points[points["varied_channel"] == channel]
        bases = list(dict.fromkeys(shown["base_scenario_id"]))
        for base in bases:
            line = shown[shown["base_scenario_id"] == base]
            x = 100 * line["level"].to_numpy()
            color = CHANNEL_COLOR if len(bases) == 1 else None
            label = "Mean" if len(bases) == 1 else f"Mean, base {base}"
            (mean_line,) = axis.plot(x, line["mean"], color=color, linewidth=2, label=label)
            axis.fill_between(
                x,
                line["p05"],
                line["p95"],
                color=mean_line.get_color(),
                alpha=0.2,
                label="90% pointwise interval" if len(bases) == 1 else None,
            )
        axis.axvline(100, color="0.45", linestyle="--", linewidth=1, label="Base spend (100%)")
        axis.set(
            title=channel,
            xlabel="Spend relative to the base plan in the decision window",
            xlim=(0, None),
        )
        axis.yaxis.set_major_formatter("{x:,.0f}")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.legend(fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("Channel contribution over the forecast")
    figure.suptitle(title or "River: contribution by spend level")
    figure.supxlabel(CAVEAT, fontsize=9)
    return figure


def plot_compare(
    table: pd.DataFrame,
    *,
    component: str = "outcome",
    error_bars: float = 2.0,
    include_curves: bool = False,
    title: str | None = None,
) -> Any:
    """Mean difference of each plan against its base, with simulation error bars.

    ``error_bars`` is how many simulation standard errors each bar spans on each side.
    The service gives no interval for a difference between plans, and neither does this plot.
    """
    plt = _pyplot()
    kinds = ["scenario", "curve"] if include_curves else ["scenario"]
    rows = table[table["kind"].isin(kinds) & (table["component"] == component)]
    rows = rows.dropna(subset=["mean_difference"]).sort_values("mean_difference")
    if rows.empty:
        raise ValueError(f"No comparable rows for component {component!r}")
    figure, axis = plt.subplots(figsize=(8, 0.5 * len(rows) + 2), layout="constrained")
    errors = error_bars * rows["simulation_se"].fillna(0.0).to_numpy()
    axis.errorbar(
        rows["mean_difference"],
        np.arange(len(rows)),
        xerr=errors,
        fmt="o",
        color=CHANNEL_COLOR,
        capsize=3,
    )
    axis.axvline(0, color="0.45", linestyle="--", linewidth=1)
    axis.set_yticks(np.arange(len(rows)), rows["scenario_id"])
    axis.set(
        xlabel=f"Mean difference of {component} versus the base plan",
        title=title or "River: scenarios compared with the base plan",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.15)
    figure.supxlabel(
        f"Error bars: ±{error_bars:g} standard errors of the mean difference: "
        "simulation error, not model uncertainty.",
        fontsize=9,
    )
    return figure
