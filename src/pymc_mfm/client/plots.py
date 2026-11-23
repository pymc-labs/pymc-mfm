"""Matplotlib plots of an analysis result table."""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd

CHANNEL_COLOR = "#1f77b4"
CAVEAT = (
    "Each channel varied separately; history and other inputs fixed. "
    "Conditional contributions, not causal lift; observation noise excluded."
)


WARNING_COLOR = "#d62728"
MAX_WARNING_LINES = 8


def _warning_list(value: Any) -> tuple[str, ...]:
    """The warnings of one result row, whether a list, a tuple or missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(value)


def plan_warnings(table: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """The distinct warnings of each plan, by ``scenario_id``; plans with none are left out."""
    found: dict[str, tuple[str, ...]] = {}
    for scenario_id, group in table.groupby("scenario_id", sort=False):
        merged = dict.fromkeys(w for value in group["warnings"] for w in _warning_list(value))
        if merged:
            found[str(scenario_id)] = tuple(merged)
    return found


def _warning_footnote(warnings: dict[str, tuple[str, ...]]) -> str:
    """One wrapped text for the foot of a chart, so an exported image keeps the warnings."""
    lines = []
    for scenario_id, items in warnings.items():
        lines += textwrap.wrap(f"Warning, {scenario_id}: " + "; ".join(items), width=130)
    if len(lines) > MAX_WARNING_LINES:
        lines = [*lines[:MAX_WARNING_LINES], "More warnings: see the result table."]
    return "\n".join(lines)


def curve_points(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of each response curve, with the unrun level 1 filled in from its base.

    One row per channel, base plan and level, with the channel's own ``mean``, ``p05`` and
    ``p95`` and the plan's ``warnings`` (a tuple), in the order channels first appear in the
    table. Failed runs are dropped.
    """
    curves = table[(table["kind"] == "curve") & (table["component"] == table["varied_channel"])]
    columns = ["varied_channel", "base_scenario_id", "level", "spend_total", "mean", "p05", "p95"]
    points = curves[columns].reset_index(drop=True)
    points["warnings"] = [_warning_list(value) for value in curves["warnings"]]
    baselines = []
    for (channel, base), group in points.groupby(["varied_channel", "base_scenario_id"]):
        if (group["level"] == 1).any():
            continue
        base_row = table[(table["scenario_id"] == base) & (table["component"] == channel)]
        if base_row.empty:
            continue
        row = base_row.iloc[0]
        baselines.append(
            {
                "varied_channel": channel,
                "base_scenario_id": base,
                "level": 1.0,
                "spend_total": row["spend_total"],
                **row[["mean", "p05", "p95"]].to_dict(),
                "warnings": _warning_list(row["warnings"]),
            }
        )
    if baselines:
        points = pd.concat([points, pd.DataFrame(baselines)], ignore_index=True)
    points = points.dropna(subset=["mean"])
    position = {name: i for i, name in enumerate(dict.fromkeys(curves["varied_channel"]))}
    points["channel_position"] = points["varied_channel"].map(position)
    points = points.sort_values(["channel_position", "base_scenario_id", "level"])
    return points.drop(columns="channel_position").reset_index(drop=True)


def plot_curves(table: pd.DataFrame, *, title: str | None = None) -> Any:
    """One panel per channel: mean and 90% pointwise band of its contribution by spend level.

    A level whose plan has warnings, such as spend outside the history's 5th to 95th
    percentile, gets a red open marker: the history gives that level no support. The
    warnings are also printed under the chart, so a saved image keeps them.
    """
    import matplotlib.pyplot as plt

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
        flagged = shown[shown["warnings"].map(len) > 0]
        if not flagged.empty:
            axis.scatter(
                100 * flagged["level"],
                flagged["mean"],
                s=70,
                facecolors="none",
                edgecolors=WARNING_COLOR,
                linewidths=1.5,
                zorder=3,
                label="Plan has a warning (see below)",
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
    footnote = _warning_footnote(plan_warnings(_plotted_rows(table, points)))
    figure.supxlabel(f"{CAVEAT}\n{footnote}" if footnote else CAVEAT, fontsize=9)
    return figure


def _plotted_rows(table: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """The result rows of the plans drawn in a curve chart: the curves and their bases."""
    drawn = set(points["base_scenario_id"])
    curve_ids = table["kind"] == "curve"
    return table[curve_ids | table["scenario_id"].isin(drawn)]


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
    A plan with no simulation standard error (for example from one draw) gets an open marker,
    no bar, and a "simulation error unavailable" label. A plan with warnings is marked ``*``
    and its warnings are printed under the chart, so a saved image keeps them.
    """
    import matplotlib.pyplot as plt

    kinds = ["scenario", "curve"] if include_curves else ["scenario"]
    rows = table[table["kind"].isin(kinds) & (table["component"] == component)]
    rows = rows.dropna(subset=["mean_difference"]).sort_values("mean_difference")
    if rows.empty:
        raise ValueError(f"No comparable rows for component {component!r}")
    figure, axis = plt.subplots(figsize=(8, 0.5 * len(rows) + 2), layout="constrained")
    positions = np.arange(len(rows))
    known = rows["simulation_se"].notna().to_numpy()
    axis.errorbar(
        rows["mean_difference"].to_numpy()[known],
        positions[known],
        xerr=error_bars * rows["simulation_se"].to_numpy()[known],
        fmt="o",
        color=CHANNEL_COLOR,
        capsize=3,
    )
    if not known.all():
        axis.plot(
            rows["mean_difference"].to_numpy()[~known],
            positions[~known],
            "o",
            color=CHANNEL_COLOR,
            markerfacecolor="none",
        )
    axis.axvline(0, color="0.45", linestyle="--", linewidth=1)
    warnings = plan_warnings(rows)
    labels = [
        f"{name}{' *' if name in warnings else ''}"
        + ("" if is_known else " (simulation error unavailable)")
        for name, is_known in zip(rows["scenario_id"], known, strict=True)
    ]
    axis.set_yticks(positions, labels)
    axis.set(
        xlabel=f"Mean difference of {component} versus the base plan",
        title=title or "River: scenarios compared with the base plan",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.15)
    note = (
        f"Error bars: ±{error_bars:g} standard errors of the mean difference: "
        "simulation error, not model uncertainty."
    )
    footnote = _warning_footnote(warnings)
    figure.supxlabel(f"{note}\n* {footnote}" if footnote else note, fontsize=9)
    return figure
