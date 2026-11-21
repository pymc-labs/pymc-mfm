"""Readable headless figures from computed, persisted diagnostic summaries.

Most functions return a matplotlib Figure for notebooks or ``savefig``;
``plot_dag`` returns a native GraphvizGraph instead. Matplotlib is imported only
when a Matplotlib figure is requested; pyplot/backend globals are never changed,
and no function imports or calls an inference model. Notebooks display either
artifact as a PNG cell result without pyplot.
Single-panel Matplotlib functions accept ``ax=``: they then draw into the caller's
axes, put the labels in its title and return the root figure holding those axes
(also when the axes sit on a subfigure); the caller's layout engine, or
``bbox_inches="tight"``, makes room for that title. Functions whose Matplotlib
figure has several axes raise ``ValueError`` for ``ax=``, stating how many axes
they need.
"""

from __future__ import annotations

import copy
import functools
import io
import math
import re
import textwrap
from typing import TYPE_CHECKING

import numpy as np

from pymc_mktfm.inference._time import _restore_time, _time_to_json

from ._graphs import ranked_graphs, select_graph
from ._graphviz import GraphvizGraph
from .dot import _graph_statements, _quoted_label, graph_to_dot

if TYPE_CHECKING:
    from matplotlib.figure import Figure

COLORS = {
    "baseline": "#666666",
    "channel": "#0072B2",
    "control": "#009E73",
    "latent": "#D55E00",
    "indirect": "#CC79A7",
    "pooled": "#333333",
}


def _png(fig):
    """The figure as PNG bytes: what a notebook displays for a figure cell result."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


@functools.cache
def _figure_type():
    """The class of every figure here, built on first use because matplotlib loads lazily."""
    from matplotlib.figure import Figure

    class LibraryFigure(Figure):
        """A Figure that notebooks display as PNG and that pickles as a plain Figure."""

        def _repr_png_(self):
            # Notebooks render a result through ``_repr_png_``; without pyplot's
            # inline backend a bare Figure would show only its text repr.
            return _png(self)

        def __reduce_ex__(self, protocol):
            # A plain Figure, so the pickle loads wherever matplotlib does.
            return Figure, (), self.__getstate__()

        def __deepcopy__(self, memo):
            # Unlike a pickle, a copy keeps the notebook display.
            clone = type(self).__new__(type(self))
            memo[id(self)] = clone
            clone.__setstate__(copy.deepcopy(self.__getstate__(), memo))
            return clone

    return LibraryFigure


def _figure(
    rows=1,
    columns=1,
    *,
    width=11,
    row_height=3.2,
    height=None,
    height_ratios=None,
    width_ratios=None,
):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    height = row_height * rows if height is None else height
    fig = _figure_type()(figsize=(width, height), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = np.asarray(
        fig.subplots(
            rows,
            columns,
            squeeze=False,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    return fig, axes


def _labels(metadata):
    """Head, window, mode, units and draw count of the plotted draws."""
    return (
        f"{metadata.get('head', 'conditional')} head · {metadata['window']} window · "
        f"{metadata['mode']} mode · units: {metadata['units']} · {metadata['n_samples']} draws"
    )


#: Longest unbroken word of an ``ax=`` title. The caller's layout can narrow its
#: axes after the title is set, and draw-time wrapping breaks only at spaces, so
#: longer words (a run label, a metric name) break after ``_``, ``/`` or ``-``.
_LONGEST_WORD = 24


def _pieces(word, longest):
    """``word`` in pieces of at most ``longest`` characters, cut after ``_``, ``/`` or ``-``."""
    pieces = []
    while len(word) > longest:
        # Cut after the last separator inside the limit, else at the limit.
        cut = max(word.rfind(mark, 0, longest) for mark in "_/-") + 1 or longest
        pieces.append(word[:cut])
        word = word[cut:]
    return [*pieces, word]


def _fit(text, width_inches, fontsize, *, longest=None):
    """Wrap each line of ``text`` to ``width_inches``, breaking at `` · `` separators first.

    A part longer than a whole line (a long title or unit) breaks between words.
    ``longest`` also puts the pieces of every longer word on lines of their own,
    whatever the width.
    """
    if longest is not None:
        text = re.sub(
            rf"\S{{{longest + 1},}}", lambda word: "\n".join(_pieces(word[0], longest)), text
        )
    # About 0.55 em per character in matplotlib's default sans-serif font.
    capacity = max(20, int(width_inches * 72 / (0.55 * fontsize)))
    lines = []
    for line in text.split("\n"):
        current = ""
        for piece in line.split(" · "):
            candidate = piece if not current else f"{current} · {piece}"
            if current and len(candidate) > capacity:
                lines.extend(textwrap.wrap(current, capacity))
                current = piece
            else:
                current = candidate
        lines.extend(textwrap.wrap(current, capacity) or [current])
    return "\n".join(lines)


def _suptitle(fig, *lines):
    """The given lines as the figure title, each wrapped to the figure width.

    The width estimate assumes Latin text; wider glyphs (CJK units, for example)
    rewrap at draw time.
    """
    fontsize = 12
    text = "\n".join(line for line in lines if line)
    fig.suptitle(_fit(text, fig.get_figwidth(), fontsize), fontsize=fontsize, wrap=True)


def _header(fig, summary, title):
    _suptitle(fig, title, _labels(summary["metadata"]))


def _root_figure(ax):
    """The top-level Figure holding ``ax``, which may sit on (nested) subfigures."""
    try:
        return ax.get_figure(root=True)
    except TypeError:  # matplotlib 3.9: ``get_figure`` takes no ``root``
        figure = ax.figure
        while figure.figure is not figure:  # a subfigure's ``figure`` is its root
            figure = figure.figure
        return figure


def _single(ax, **figure):
    """``(figure, axes, owned)``: the caller's axes on their root figure, or a new figure."""
    if ax is not None:
        return _root_figure(ax), ax, False
    fig, axes = _figure(**figure)
    return fig, axes[0, 0], True


def _title(fig, ax, labels, title, owned, detail=None):
    """Title and labels as the figure title, or with ``detail`` as the ``ax=`` axes title.

    Every line wraps to the width it is drawn across. The caller's layout may
    still narrow its axes, so their title also rewraps at draw time to stay
    inside the figure, and no word in it is longer than :data:`_LONGEST_WORD`
    characters. The caller's layout engine (or ``bbox_inches="tight"``) makes
    room for the title above the axes.
    """
    width = ax.get_window_extent().width / fig.dpi
    if owned:
        _suptitle(fig, title, labels)
        if detail:
            ax.set_title(_fit(detail, width, 10), fontsize=10)
    else:
        text = "\n".join(line for line in (title, labels, detail) if line)
        ax.set_title(_fit(text, width, 9, longest=_LONGEST_WORD), fontsize=9, wrap=True)


def _plot_time(ax, values):
    """Date positions with readable labels, including dates beyond Matplotlib's calendar."""
    time = _restore_time(values)
    if time.dtype.kind != "M":
        return time
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, date2num
    from matplotlib.ticker import FixedFormatter, FixedLocator

    time = time.astype(time.dtype.newbyteorder("="), copy=False)
    positions = date2num(time)
    padding = max(float(np.ptp(positions)), 1.0)
    calendar_min = date2num(np.datetime64("0001-01-01", "D"))
    calendar_stop = date2num(np.datetime64("10000-01-01", "D"))
    if positions[0] - padding < calendar_min or positions[-1] + padding >= calendar_stop:
        # Fixed ISO labels avoid calendar conversion during drawing or autoscaling.
        ticks = np.linspace(0, len(time) - 1, min(len(time), 3), dtype=int)
        ax.xaxis.set_major_locator(FixedLocator(positions[ticks]))
        ax.xaxis.set_major_formatter(FixedFormatter(_time_to_json(time[ticks])))
        ax.tick_params(axis="x", labelrotation=25)
    else:
        ax.xaxis_date()
        locator = AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    return positions


def _boundary(ax, metadata):
    time = _plot_time(ax, metadata["time"])
    spacing = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    if metadata["window"] == "query":
        boundary = time[0] - spacing / 2
        label = "query begins"
    else:
        boundary = time[-1] + spacing / 2
        label = "context ends"
    ax.axvline(boundary, color="0.6", linestyle=":", linewidth=1, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel(metadata["units"])


def _band(ax, time, series, color, *, label="Posterior mean"):
    time = _plot_time(ax, time)
    ax.plot(time, series["mean"], color=color, linewidth=1.6, label=label)
    ax.fill_between(
        time, series["low"], series["high"], color=color, alpha=0.23, label="Central 80% interval"
    )
    if series.get("truth") is not None:
        ax.plot(time, series["truth"], color="black", linewidth=1.3, linestyle="--", label="Truth")


def _components(summary, families):
    return [
        entry
        for entry in summary["arrays"]["components"]
        if entry["active"] and entry["family"] in families
    ]


def plot_component_trajectories(summary, *, families=None):
    """Per-entity trajectories, retaining entity labels and the original window."""
    families = tuple(COLORS) if families is None else families
    entries = _components(summary, families)
    if not entries:
        raise ValueError("no active components in requested families")
    columns = min(3, len(entries))
    fig, axes = _figure(math.ceil(len(entries) / columns), columns, width=4.4 * columns)
    metadata = summary["metadata"]
    for ax, entry in zip(axes.flat, entries):
        _band(ax, metadata["time"], entry, COLORS[entry["family"]])
        ax.set_title(entry["name"], fontsize=10)
        _boundary(ax, metadata)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axes.flat)[len(entries) :]:
        ax.set_visible(False)
    _header(fig, summary, "Component trajectories — latent quantities, not observation noise")
    return fig


def plot_family_totals(summary):
    entries = [entry for entry in summary["arrays"]["families"] if entry["status"] == "ok"]
    fig, axes = _figure(math.ceil(len(entries) / 2), 2)
    metadata = summary["metadata"]
    for ax, entry in zip(axes.flat, entries):
        _band(ax, metadata["time"], entry, COLORS[entry["name"]])
        ax.set_title(f"{entry['name']} total", fontsize=10)
        _boundary(ax, metadata)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(entries) :]:
        ax.set_visible(False)
    _header(fig, summary, "Family totals — aggregate draws before interval calculation")
    return fig


def plot_target_recovery(summary):
    fig, axes = _figure(2, 1, row_height=3)
    ax, residual = axes[:, 0]
    data = summary["arrays"]["total"]
    metadata = summary["metadata"]
    time = _plot_time(ax, metadata["time"])
    _band(ax, time, data, COLORS["channel"], label="Latent total posterior mean")
    if data["observed"] is not None:
        ax.plot(time, data["observed"], ".", color="0.3", label="Noisy observations")
        residual.plot(
            time,
            data["observed_minus_latent_mean"],
            color=COLORS["channel"],
            label="Observed − posterior-mean latent total",
        )
    if data["realized_noise"] is not None:
        residual.plot(
            time,
            data["realized_noise"],
            "--",
            color="0.2",
            label="Realized observation noise (observed − latent truth)",
        )
    if data["observed"] is None:
        residual.text(
            0.5,
            0.5,
            "Observation residual unavailable: observations not supplied",
            transform=residual.transAxes,
            ha="center",
            va="center",
        )
    residual.axhline(0, color="0.7", linewidth=0.7)
    for axis in (ax, residual):
        _boundary(axis, metadata)
        axis.legend(fontsize=8)
    residual.set_title("Residual is not a predictive-noise draw", fontsize=10)
    _header(fig, summary, "Reconstruction / closure — latent sum excludes observation noise")
    return fig


def plot_channel_totals(summary, *, ax=None):
    entries = _components(summary, ("channel",))
    if not entries:
        raise ValueError("no active channels")
    fig, ax, owned = _single(ax, width=max(8, len(entries) * 0.8), row_height=4)
    for index, entry in enumerate(entries):
        data = entry["time_mean"]
        ax.vlines(index, data["low"], data["high"], color=COLORS["channel"], linewidth=3)
        ax.scatter(
            index,
            data["mean"],
            color=COLORS["channel"],
            label="Mean / central 80%" if index == 0 else None,
        )
        if data["truth"] is not None:
            ax.scatter(
                index,
                data["truth"],
                color="black",
                marker="x",
                s=50,
                label="Truth" if index == 0 else None,
            )
    ax.set_xticks(
        range(len(entries)), [entry["name"] for entry in entries], rotation=35, ha="right"
    )
    ax.set_ylabel(f"Time-mean contribution ({summary['metadata']['units']})")
    ax.legend(fontsize=8)
    _title(
        fig,
        ax,
        _labels(summary["metadata"]),
        "Channel attribution — aggregate each paired trajectory before intervals",
        owned,
    )
    return fig


def plot_recovery_scatter(summary, *, ax=None):
    fig, ax, owned = _single(ax, width=7, row_height=6)
    seen = set()
    extrema = []
    for entry in _components(summary, tuple(COLORS)):
        if entry["truth"] is None:
            raise ValueError("recovery scatter requires truth")
        ax.scatter(
            entry["truth"],
            entry["mean"],
            s=12,
            alpha=0.55,
            color=COLORS[entry["family"]],
            label=entry["family"] if entry["family"] not in seen else None,
        )
        seen.add(entry["family"])
        extrema.extend(
            [min(entry["truth"]), max(entry["truth"]), min(entry["mean"]), max(entry["mean"])]
        )
    low, high = min(extrema), max(extrema)
    if low == high:
        low, high = low - 1, high + 1
    ax.plot([low, high], [low, high], "k--", linewidth=1, label="Identity")
    ax.set_xlabel(f"True component ({summary['metadata']['units']})")
    ax.set_ylabel(f"Posterior mean ({summary['metadata']['units']})")
    ax.legend(fontsize=8)
    _title(fig, ax, _labels(summary["metadata"]), "Component recovery — active slots only", owned)
    return fig


#: Display names of the per-effect attribution scores.
_ATTRIBUTION_LABELS = {
    "crps": "CRPS",
    "mean_bias": "Bias",
    "mean_abs_error": "Absolute bias",
    "demeaned_rmse": "Demeaned RMSE",
    "leakage_scaled_mae": "Posterior-mean leakage",
    "draw_leakage_scaled_mae": "Per-draw leakage",
}
#: Channel colors by graph-truth label; ``None`` is a channel without graph truth.
_LIVE_COLORS = {True: COLORS["channel"], False: "#999999", None: COLORS["channel"]}


def _bar_offsets(bars):
    """Width and centre offsets of ``bars`` side-by-side bars in one group."""
    width = 0.8 / bars
    return width, (np.arange(bars) - (bars - 1) / 2) * width


def plot_attribution(summary):
    """Live and dead attribution errors of each effect, in sales-scale units.

    A channel or control is live when graph truth holds its direct ``cy`` or
    ``zb`` edge. Live effects show CRPS, bias, absolute bias and demeaned RMSE;
    dead effects their posterior-mean and per-draw leakage. Each dashed line is
    the metric table's world value, the mean of its bars.
    """
    data = summary["arrays"].get("attribution")
    if data is None:
        raise ValueError("live/dead attribution requires truth, graph truth and sales_scale")
    groups = [(effect, record) for effect, record in data["effects"].items() if record["effects"]]
    fig, axes = _figure(len(groups), 2, width=12, row_height=3.8)
    for row, (effect, record) in zip(axes, groups):
        for ax, live in zip(row, (True, False)):
            status = "live" if live else "dead"
            relation = "in" if live else "absent from"
            ax.set_title(
                f"{status.capitalize()} {effect} effects: {record['edge']} edge {relation} "
                "graph truth",
                fontsize=10,
            )
            entries = [entry for entry in record["effects"] if entry["live"] is live]
            if not entries:
                ax.text(
                    0.5,
                    0.5,
                    f"No {status} {effect} effects",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                ax.set_axis_off()
                continue
            keys = [key for key in entries[0] if key not in ("name", "index", "live")]
            width, offsets = _bar_offsets(len(keys))
            positions = np.arange(len(entries))
            for number, (key, offset) in enumerate(zip(keys, offsets, strict=True)):
                metric = f"attribution/{status}_{effect}/{key}"
                ax.bar(
                    positions + offset,
                    [entry[key] for entry in entries],
                    width,
                    color=f"C{number}",
                    label=_ATTRIBUTION_LABELS.get(key, key),
                    gid=metric,
                )
                if summary["metrics"].get(metric) is not None:
                    ax.axhline(
                        summary["metrics"][metric],
                        color=f"C{number}",
                        linestyle="--",
                        linewidth=1,
                        gid=metric,
                    )
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_xticks(
                positions, [entry["name"] for entry in entries], rotation=30, ha="right", fontsize=8
            )
            ax.set(xlim=(-0.75, len(entries) - 0.25), ylabel="Error (sales-scale units)")
            if not live:
                # Leakage is non-negative; all-zero bars would otherwise autoscale
                # to a sliver below zero and read as negative.
                peak = max(
                    (entry[key] for entry in entries for key in keys if entry[key] is not None),
                    default=0.0,
                )
                ax.set_ylim(0.0, peak * 1.1 if peak > 0 else 1.0)
                if peak == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "Leakage is exactly 0 for every dead effect",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
            ax.legend(fontsize=7)
    _header(
        fig,
        summary,
        "Live/dead attribution per effect — dashed lines are world values (means over effects)",
    )
    return fig


def plot_window_totals(summary):
    """Pointwise window-mean channel coverage, joint-live coverage and latent-total fit.

    A channel's interval is the central 80% of its per-draw window means; a
    vermillion cross marks window-mean truth outside it. The middle panel draws
    the recorded pointwise coverage shares beside the studentized simultaneous
    80% event for every truth-live active channel. Unavailable events state their
    reason; older summaries say ``not recorded`` and are never rescored. The
    right panel compares posterior-mean latent totals with structural truth.
    """
    data = summary["arrays"].get("window_totals")
    if data is None:
        raise ValueError("window totals require simulation truth")
    metrics = summary["metrics"]
    units = summary["metadata"]["units"]
    entries = data["channels"]
    means = [summary["arrays"]["components"][entry["index"]]["time_mean"] for entry in entries]
    fig, axes = _figure(1, 3, width=15, row_height=4.6, width_ratios=(3, 2, 2))
    channels, rates, fit = axes[0]
    for live, label in ((True, "Live channel"), (False, "Dead channel"), (None, "Channel")):
        members = [slot for slot, entry in enumerate(entries) if entry["live"] is live]
        if members:
            channels.vlines(
                members,
                [means[slot]["low"] for slot in members],
                [means[slot]["high"] for slot in members],
                color=_LIVE_COLORS[live],
                linewidth=5,
                alpha=0.8,
                label=f"{label}: central 80% of window means",
                gid="window_totals/interval",
            )
    channels.scatter(
        range(len(entries)),
        [mean["mean"] for mean in means],
        color="black",
        s=14,
        zorder=3,
        label="Posterior mean",
        gid="window_totals/mean",
    )
    for covered, marker, color, label in (
        (True, "x", "black", "Truth inside"),
        (False, "X", "#D55E00", "Truth outside"),
    ):
        members = [slot for slot, entry in enumerate(entries) if entry["covered"] is covered]
        if members:
            channels.scatter(
                members,
                [means[slot]["truth"] for slot in members],
                marker=marker,
                color=color,
                s=60,
                zorder=4,
                label=label,
                gid=f"window_totals/truth_{'covered' if covered else 'missed'}",
            )
    channels.set_xticks(
        range(len(entries)),
        [entry["name"] for entry in entries],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    channels.set(
        ylabel=f"Window-mean contribution ({units})", title="Time-average 80% coverage per channel"
    )
    channels.legend(fontsize=7)
    groups = (
        ("timeavg/channel/cov80", "Pointwise\nall", COLORS["pooled"]),
        ("timeavg/channel/cov80_live", "Pointwise\nlive", _LIVE_COLORS[True]),
        ("timeavg/channel/cov80_dead", "Pointwise\ndead", _LIVE_COLORS[False]),
        ("window_totals/simultaneous_cov80", "Joint-live\nevent", _LIVE_COLORS[True]),
    )
    for position, (metric, label, color) in enumerate(groups):
        value = metrics.get(metric)
        if value is None:
            rates.text(position, 0.02, "n/a", ha="center", fontsize=8)
            status = summary.get("statuses", {}).get(f"metric/{metric}", {})
            reason = status.get("reason") or (
                "unavailable" if metric in metrics else "not recorded"
            )
            rates.plot(
                [],
                [],
                " ",
                label=_fit(
                    f"{' '.join(label.split())} n/a: {reason}",
                    fig.get_figwidth() * 2 / 7,
                    7,
                    longest=_LONGEST_WORD,
                ),
            )
        else:
            rates.bar(
                position,
                value,
                color=color,
                hatch="//" if metric == "window_totals/simultaneous_cov80" else None,
                gid=metric,
            )
    rates.axhline(0.8, color="black", linestyle="--", linewidth=1, label="Nominal 80%")
    rates.set_xticks(range(len(groups)), [label for _, label, _ in groups], fontsize=8)
    rates.set(
        ylim=(0, 1.05), ylabel="Coverage share / indicator", title="Pointwise shares / joint event"
    )
    rates.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    total = summary["arrays"]["total"]
    fit.scatter(
        total["truth"],
        total["mean"],
        color=COLORS["latent"],
        s=16,
        label="Weeks",
        gid="window_totals/latent_fit",
    )
    low, high = min(total["truth"] + total["mean"]), max(total["truth"] + total["mean"])
    if low == high:
        low, high = low - 1, high + 1
    fit.plot([low, high], [low, high], "k--", linewidth=1, label="Identity")
    r2 = metrics.get("latent_total/r2")
    fit.set(
        xlabel=f"Structural latent total ({units})",
        ylabel=f"Posterior-mean latent total ({units})",
        title=f"Latent-sales R² {r2:.3f}" if r2 is not None else "Latent-sales R² undefined",
    )
    fit.legend(fontsize=7)
    _header(
        fig, summary, "Window totals — pointwise coverage, joint-live event and latent-total fit"
    )
    return fig


def plot_cumulative(summary):
    """Running totals of each channel's contribution across the window, with truth.

    Bands are central 80% intervals of the per-draw running totals. The title
    states the table's energy score of those running totals.
    """
    data = summary["arrays"].get("cumulative")
    if data is None:
        raise ValueError("cumulative contributions require simulation truth")
    entries = data["channels"]
    metadata = summary["metadata"]
    columns = min(3, len(entries))
    fig, axes = _figure(math.ceil(len(entries) / columns), columns, width=4.4 * columns)
    for ax, entry in zip(axes.flat, entries):
        _band(ax, metadata["time"], entry, _LIVE_COLORS[entry["live"]])
        status = {True: " · live", False: " · dead", None: ""}[entry["live"]]
        ax.set_title(f"{entry['name']}{status}", fontsize=10)
        _boundary(ax, metadata)
        ax.set_ylabel(f"Cumulative {metadata['units']}")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    for ax in list(axes.flat)[len(entries) :]:
        ax.set_visible(False)
    energy = summary["metrics"].get("cumulative/channel/energy_score")
    score = "undefined" if energy is None else f"{energy:.4g} (sales-scale units)"
    _header(
        fig,
        summary,
        f"Cumulative channel contributions across the window — energy score {score}",
    )
    return fig


#: Correlation-valued temporal scores in plotting order, with display names.
_TEMPORAL_SCORES = (
    ("truth_lag1", "Truth lag-1"),
    ("draw_lag1", "Draw lag-1"),
    ("truth_error_residual_lag1", "Truth-error\nresidual lag-1"),
    ("posterior_centered_residual_lag1", "Posterior-centred\nresidual lag-1"),
    ("adjacent_draw_correlation", "Adjacent-week\ncorrelation over draws"),
)
_TEMPORAL_NAMES = {
    "baseline": "Baseline",
    "channel": "Channels",
    "control": "Controls",
    "latent_total": "Latent total",
    "indirect": "Indirect sources",
}


def plot_temporal(summary):
    """Lag-1 autocorrelation of truth, draws and residuals for each trajectory family.

    Dots are the family's slots; each black bar is the metric table's family
    value, the mean over slots where the score is defined (constant series
    have none). The lag-1 tracking error ``|draw - truth lag-1|`` is in the title.
    """
    data = summary["arrays"].get("temporal")
    if data is None:
        raise ValueError("temporal coherence requires simulation truth")
    records = data["families"]
    fig, axes = _figure(len(records), 1, width=11, row_height=3.1)
    for ax, record in zip(axes[:, 0], records):
        family = record["family"]
        dots_x, dots_y, first = [], [], True
        for position, (key, _) in enumerate(_TEMPORAL_SCORES):
            values = [slot["values"][key] for slot in record["slots"]]
            values = [value for value in values if value is not None]
            # Spread a family's slots around the score's position; one dot stays centred.
            dots_x.extend(position + np.linspace(-0.15, 0.15, len(values)) * (len(values) > 1))
            dots_y.extend(values)
            metric = f"temporal/{family}/{key}"
            if summary["metrics"][metric] is not None:
                ax.hlines(
                    summary["metrics"][metric],
                    position - 0.3,
                    position + 0.3,
                    color="black",
                    linewidth=2.2,
                    gid=metric,
                    label="Family value (table)" if first else None,
                )
                first = False
        ax.scatter(
            dots_x,
            dots_y,
            s=18,
            alpha=0.7,
            color=COLORS.get(family, COLORS["latent"]),
            label="Slot values",
            gid=f"temporal/{family}/slots",
        )
        ax.axhline(0, color="0.7", linewidth=0.7)
        ax.set_xticks(range(len(_TEMPORAL_SCORES)), [label for _, label in _TEMPORAL_SCORES])
        ax.set(xlim=(-0.5, len(_TEMPORAL_SCORES) - 0.5), ylim=(-1.1, 1.1), ylabel="Correlation")
        tracking = summary["metrics"][f"temporal/{family}/lag1_tracking_abs_error"]
        detail = "undefined" if tracking is None else f"{tracking:.3f}"
        ax.set_title(
            f"{_TEMPORAL_NAMES.get(family, family)} · {len(record['slots'])} "
            f"{'slot' if len(record['slots']) == 1 else 'slots'} · "
            f"lag-1 tracking error {detail}",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="lower left")
    _header(
        fig,
        summary,
        "Temporal coherence — lag-1 autocorrelation of truth, draws and residuals",
    )
    return fig


#: Families whose edges connect two node banks with several nodes each.
_MATRIX_FAMILIES = ("cc", "zc", "zz")
_STRIP_EDGES = 24


def _edge_groups(edges):
    """One family per group: cc/zc/zz as matrices, other families as edge strips."""
    families = {}
    for entry in edges:
        families.setdefault(entry["family"], []).append(entry)
    groups = []
    for family, entries in families.items():
        if family in _MATRIX_FAMILIES and all(entry["endpoints"] for entry in entries):
            groups.append((family, entries, True))
        else:
            groups.extend(
                (family, entries[start : start + _STRIP_EDGES], False)
                for start in range(0, len(entries), _STRIP_EDGES)
            )
    return groups


def _family_title(family, entries):
    if family is None:
        return "Edges without recorded family labels"
    banks = {
        tuple(node.split(":", 1)[0] for node in entry["endpoints"])
        for entry in entries
        if entry["endpoints"]
    }
    return f"{family}: {' → '.join(banks.pop())}" if len(banks) == 1 else family


def plot_structure_heatmap(summary):
    """Edge frequencies grouped by edge family; empty families are not drawn.

    cc, zc and zz edges are source × target matrices whose rows and columns
    share one node order (blank cells are self-loops, which are not graph
    slots); every other family is a strip of its edges.
    """
    edges = summary["arrays"]["graphs"]["edges"]
    if not edges:
        raise ValueError("no active edges")
    from matplotlib import colormaps

    colormap = colormaps["Blues"].with_extremes(bad="0.93")
    frequency_label = (
        "Paired router frequency"
        if summary["metadata"]["mode"] == "model_average"
        else "Selected hypothesis indicators"
    )
    groups = _edge_groups(edges)
    heights = [
        1.6 + 0.28 * len({entry["endpoints"][0] for entry in entries}) if matrix else 2.2
        for _, entries, matrix in groups
    ]
    # The figure title and rotated tick labels need room beyond the families'
    # own rows; without it one short strip collapses the constrained layout.
    height = max(sum(heights) + 1.5, 4)
    fig, axes = _figure(len(groups), 1, height=height, height_ratios=heights)
    for ax, (family, entries, matrix) in zip(axes[:, 0], groups):
        has_truth = all(entry["truth"] is not None for entry in entries)
        title = _family_title(family, entries)
        if matrix:
            nodes = list(dict.fromkeys(node for entry in entries for node in entry["endpoints"]))
            ends = [{entry["endpoints"][end] for entry in entries} for end in (0, 1)]
            sources, targets = ([node for node in nodes if node in members] for members in ends)
            cells = [
                (sources.index(entry["endpoints"][0]), targets.index(entry["endpoints"][1]))
                for entry in entries
            ]
            values = np.full((len(sources), len(targets)), np.nan)
            for (row, column), entry in zip(cells, entries):
                values[row, column] = entry["probability"]
            image = ax.imshow(
                np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=colormap, aspect="auto"
            )
            if has_truth:
                true_cells = [cell for cell, entry in zip(cells, entries) if entry["truth"]]
                ax.scatter(
                    [column for _, column in true_cells],
                    [row for row, _ in true_cells],
                    marker="s",
                    s=90,
                    facecolors="none",
                    edgecolors="#D55E00",
                    linewidths=1.4,
                    label="Graph truth",
                )
                title = f"{title} — squares mark graph truth"
            ax.set_yticks(range(len(sources)), sources, fontsize=7)
            ax.set_xticks(range(len(targets)), targets, rotation=70, ha="right", fontsize=7)
            ax.set(ylabel="Source", xlabel="Target")
        else:
            rows, names = [[entry["probability"] for entry in entries]], [frequency_label]
            if has_truth:
                rows.append([entry["truth"] for entry in entries])
                names.append("Graph truth")
            image = ax.imshow(rows, vmin=0, vmax=1, cmap=colormap, aspect="auto")
            ax.set_yticks(range(len(rows)), names)
            prefix = f"{family}:"
            ax.set_xticks(
                range(len(entries)),
                [entry["name"].removeprefix(prefix) for entry in entries],
                rotation=70,
                ha="right",
                fontsize=7,
            )
        ax.set_title(title, fontsize=10)
    fig.colorbar(
        image, ax=axes[:, 0].tolist(), fraction=0.02, label="Edge frequency (not auxiliary logits)"
    )
    _header(fig, summary, summary["arrays"]["graphs"]["label"])
    return fig


def plot_dag(summary, *, graph=None) -> GraphvizGraph:
    """Render Average or a zero-based complete graph rank with native ``dot`` defaults.

    Without graph truth, the source is exactly the prediction-only DOT export.
    Supplied truth is compared in a separate, semantically labeled native cluster;
    it never changes the prediction export. Rendering requires Graphviz.
    """
    edges = summary["arrays"]["graphs"]["edges"]
    if not edges or not all(entry["truth"] is not None for entry in edges):
        return GraphvizGraph(graph_to_dot(summary, graph=graph))
    selected = select_graph(summary, graph=graph)
    nodes, average, individual = _graph_statements(edges)
    truth_nodes, _, truth_edges = _graph_statements(edges, prefix="t")
    prediction_label = "Average" if selected is None else f"Complete graph rank {graph}"
    values = [entry["probability"] for entry in edges] if selected is None else selected["graph"]
    prediction_edges = average if selected is None else individual
    lines = ["digraph {", "  subgraph cluster_truth {", '    label="True graph";']
    lines.extend(f"  {line}" for line in truth_nodes)
    lines.extend(
        f"  {line}" for line, entry in zip(truth_edges, edges, strict=True) if entry["truth"]
    )
    lines.extend(["  }", "  subgraph cluster_prediction {"])
    lines.append(f"    label={_quoted_label(prediction_label)};")
    lines.extend(f"  {line}" for line in nodes)
    lines.extend(f"  {line}" for line, value in zip(prediction_edges, values, strict=True) if value)
    lines.extend(["  }", "}", ""])
    return GraphvizGraph("\n".join(lines))


def plot_graph_posterior(summary, *, ax=None):
    graph = summary["arrays"]["graphs"]
    records = ranked_graphs(summary)
    selected = records[:15]
    probabilities = [entry["frequency"] for entry in selected]
    labels = ["".join(map(str, entry["graph"])) or "empty graph" for entry in selected]
    labels = [
        label if len(label) <= 28 else f"Graph {index} (full vector in evidence)"
        for index, label in enumerate(labels)
    ]
    colors = ["#D55E00" if entry.get("is_truth", False) else "#0072B2" for entry in selected]
    if len(records) > len(selected):
        labels.append(f"Other {len(records) - len(selected)} graphs")
        probabilities.append(sum(entry["frequency"] for entry in records[len(selected) :]))
        colors.append("#999999")
    fig, ax, owned = _single(ax, width=11, row_height=max(4, 0.32 * len(labels)))
    ax.barh(range(len(labels)), probabilities, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Empirical complete-graph frequency; orange = true graph when supplied")
    ax.set_xlim(0, 1)
    _title(fig, ax, _labels(summary["metadata"]), graph["label"], owned)
    return fig


def plot_cardinality(summary):
    graph = summary["arrays"]["graphs"]
    support = summary["arrays"].get("graph_support")
    entries = [
        ("All active edges", graph["cardinality_probabilities"], graph["true_cardinality"], None)
    ]
    if support is not None:
        contract = support["arrays"]["support"]
        budgets = {} if contract is None else contract["cardinality"]
        for family, counts in support["arrays"]["edge_count_histograms"].items():
            entries.append(
                (
                    family,
                    np.asarray(counts) / sum(counts),
                    support["metrics"][f"family/{family}/true_count"],
                    budgets.get(family),
                )
            )
    columns = min(3, len(entries))
    fig, axes = _figure(math.ceil(len(entries) / columns), columns, width=4.2 * columns)
    for ax, (name, probabilities, truth, budget) in zip(axes.flat, entries):
        ax.bar(
            np.arange(len(probabilities)),
            probabilities,
            color=COLORS["channel"],
            label="Draw frequency",
        )
        if truth is not None:
            ax.axvline(truth, color="black", linestyle="--", label="True edge count")
        if budget is not None:
            ax.axvspan(
                budget[0] - 0.4,
                budget[1] + 0.4,
                alpha=0.15,
                color="#009E73",
                label="Declared prior range",
            )
        ax.legend(fontsize=7)
        ax.set(xlabel="Number of active edges", ylabel="Empirical frequency", title=name)
    for ax in list(axes.flat)[len(entries) :]:
        ax.set_visible(False)
    _header(
        fig,
        summary,
        "Graph cardinality — decoded complete graphs, not independent-edge approximation",
    )
    return fig


def _stated(summary, name, template="{:.3g}"):
    """A table metric through ``template``, or ``undefined (<reason>)`` without a value."""
    value = summary["metrics"][name]
    if value is None:
        return f"undefined ({summary['statuses'][f'metric/{name}']['reason']})"
    return template.format(value)


#: Whole-graph scores drawn by :func:`plot_graph_statistics`, as (metric key, label).
_GRAPH_SCORES = (
    ("normalized_shd_mean", "Normalized SHD, mean"),
    ("energy_score", "Energy score"),
    ("categorical_brier_score", "Categorical Brier score"),
    ("pairwise_comoment_abs_err", "Pairwise edge co-moment, mean"),
    ("pairwise_comoment_max_abs_err", "Pairwise edge co-moment, max"),
)


def _share_axis(ax):
    """A share-of-draws y axis ticked up to 100%, with room above the bars for the legend."""
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Share of draws")


def plot_graph_statistics(summary):
    """Distinct complete graphs, their distance to graph truth and whole-graph scores.

    From the stored draws: the distinct graphs ranked by frequency, and the
    share of draws at each structural Hamming distance (SHD) to graph truth
    with the table's mean and median. The last axes draws the table's other
    whole-graph scores (``graph/*``, the evaluator's ``joint_router/graph``
    statistics); undefined ones state why. Rank and distance axes tick whole
    numbers only.
    """
    from matplotlib.ticker import MaxNLocator

    graph = summary["arrays"]["graphs"]
    metrics = summary["metrics"]
    records = sorted(graph["complete_graphs"], key=lambda item: (-item["count"], item["graph"]))
    frequencies = np.asarray([record["frequency"] for record in records])
    rank = np.arange(1, len(records) + 1)
    fig, axes = _figure(1, 3, width=16, row_height=4.2)
    ranks, distance, scores = axes[0]
    ranks.bar(
        rank,
        frequencies,
        color=["#D55E00" if record.get("is_truth") else COLORS["channel"] for record in records],
        label="Graph frequency (orange = true graph)",
    )
    ranks.plot(
        rank,
        np.cumsum(frequencies),
        drawstyle="steps-mid",
        marker="o",
        markersize=4,
        color="black",
        label="Cumulative mass",
    )
    ranks.set(xlabel="Distinct complete graph, by frequency", xlim=(0.5, len(records) + 0.5))
    _share_axis(ranks)
    # min_n_ticks=1: one graph leaves only the integer 1 in view, and the default of
    # two integers would fall back to fractional ticks.
    ranks.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ranks.set_title(
        f"distinct graphs {_stated(summary, 'graph/n_distinct', '{:g}')} · "
        f"mode {metrics['uncertainty/mode_frequency']:.1%} · "
        f"entropy {_stated(summary, 'graph/entropy', '{:.3g} nats')}",
        fontsize=10,
    )
    ranks.legend(fontsize=8, loc="upper right")
    if graph["truth"] is None:
        distance.text(
            0.5,
            0.5,
            "Distance to graph truth unavailable: graph truth not supplied",
            transform=distance.transAxes,
            ha="center",
            va="center",
        )
        distance.set_axis_off()
    else:
        shd = np.asarray([record["shd"] for record in records])
        mass = np.bincount(shd, weights=frequencies)
        distance.bar(np.arange(len(mass)), mass, color=COLORS["channel"], label="Share of draws")
        for key, linestyle, label in (("shd_mean", "--", "Mean"), ("shd_median", ":", "Median")):
            value = metrics[f"graph/{key}"]
            if value is not None:
                distance.axvline(
                    value, color="black", linestyle=linestyle, label=label, gid=f"graph/{key}"
                )
        distance.set(
            xlabel="Structural Hamming distance to graph truth (active edges)",
            xlim=(-0.5, len(mass) - 0.5),
        )
        _share_axis(distance)
        distance.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        distance.set_title(
            f"exact recovery {_stated(summary, 'graph/exact_recovery', '{:.1%}')} · "
            f"mean {_stated(summary, 'graph/shd_mean')} · "
            f"median {_stated(summary, 'graph/shd_median', '{:g}')} edges",
            fontsize=10,
        )
        distance.legend(fontsize=8, loc="upper center", ncol=3)
    labels, defined = [], [1.0]
    for position, (key, label) in enumerate(_GRAPH_SCORES):
        name = f"graph/{key}"
        if metrics[name] is not None:
            scores.barh(position, metrics[name], color=COLORS["channel"], gid=name)
            defined.append(metrics[name])
        labels.append(f"{label}\n{_stated(summary, name)}")
    scores.set_yticks(range(len(labels)), labels, fontsize=8)
    scores.set(
        xlim=(0, max(defined)), ylim=(len(labels) - 0.5, -0.5), xlabel="Score (dimensionless)"
    )
    scores.set_title("Whole-graph scores", fontsize=10)
    _header(fig, summary, f"Complete-graph statistics — {graph['label']}")
    return fig


#: Rates drawn by :func:`plot_cycles`, as (metric key, label, color).
_CYCLE_RATES = (
    ("cyclic_draw_fraction", "Complete graph: draws with a cycle", COLORS["pooled"]),
    ("cc_cycle_rate", "cc: draws with a cycle", COLORS["channel"]),
    ("cc_two_cycle_rate", "cc: two-cycle rate", COLORS["channel"]),
    ("zz_cycle_rate", "zz: draws with a cycle", COLORS["control"]),
    ("zz_two_cycle_rate", "zz: two-cycle rate", COLORS["control"]),
)


def plot_cycles(summary, *, ax=None):
    """Directed-cycle rates of the drawn graphs; the graph prior forbids any cycle.

    Bars are the table's ``graph/*`` rates: the share of draws whose complete
    graph (every edge family) has a directed cycle, then per ``cc`` and ``zz``
    family the share of draws whose family edges form one and the two-cycle
    rate (mutual edges, over draws and node pairs). Undefined family rates
    state their reason.
    """
    metrics = summary["metrics"]
    records = summary["arrays"]["graphs"]["complete_graphs"]
    if metrics.get("graph/cyclic_draw_fraction") is None or any(
        "cyclic" not in record for record in records
    ):
        raise ValueError("cycle check requires edge endpoint metadata")
    fig, ax, owned = _single(ax, width=9, row_height=3.4)
    labels = []
    for position, (key, label, color) in enumerate(_CYCLE_RATES):
        name = f"graph/{key}"
        if metrics[name] is not None:
            ax.barh(position, metrics[name], color=color, gid=name)
        labels.append(f"{label}: {_stated(summary, name, '{:.1%}')}")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set(xlim=(0, 1), ylim=(len(labels) - 0.5, -0.5), xlabel="Rate (the graph prior requires 0)")
    cyclic = sum(record["cyclic"] for record in records)
    _title(
        fig,
        ax,
        _labels(summary["metadata"]),
        "Directed-cycle check — the graph prior forbids cycles",
        owned,
        detail=f"{cyclic} of {len(records)} distinct complete graphs contain a directed cycle",
    )
    return fig


#: Views of :func:`plot_reliability`: the CORP isotonic curve or equal-width bins.
RELIABILITY_KINDS = ("corp", "binned")


def plot_reliability(summary, *, kind="corp", ax=None):
    """Edge reliability: how often graph truth holds edges of each router frequency.

    ``kind="corp"`` draws the CORP curve, the isotonic (pool-adjacent-violators)
    fit of graph truth on edge frequency: each block's calibrated probability
    across its frequency range, joined over the gaps between blocks, with
    pooled block markers sized by edge count. The title states the pooled
    Brier decomposition ``MCB − DSC + UNC``. ``kind="binned"`` draws equal-width
    bins (points at each bin's mean frequency and truth share, pooled points
    sized by edge count) and the pooled ECE. Black pools every active edge and
    colors split the families. An edge set with one truth class draws nothing:
    its legend entry states that its calibration is not evidenced.
    """
    from pymc_mktfm.training.metrics import one_class_notice

    if kind not in RELIABILITY_KINDS:
        raise ValueError(f"reliability kind must be one of {RELIABILITY_KINDS}, got {kind!r}")
    data = summary["arrays"].get("reliability")
    if data is None:
        raise ValueError("edge reliability requires graph truth")
    if kind == "corp" and "corp" not in data["pooled"]:
        raise ValueError("this saved reliability evidence has no CORP fit; draw kind='binned'")
    fig, ax, owned = _single(ax, width=9, row_height=6)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly reliable")
    sets = [("pooled", "All active edges", data["pooled"], "black")]
    sets += [
        (family, family, record, f"C{number}")
        for number, (family, record) in enumerate(data["families"].items())
    ]
    for key, name, record, color in sets:
        pooled = key == "pooled"
        gid = f"reliability/{key}"
        n, n_pos = record["n"], record["n_pos"]
        label = f"{name}: {n} {'edge' if n == 1 else 'edges'} ({n_pos} in graph truth)"
        notice = one_class_notice(n, n_pos)
        if notice is not None:
            # Legend only: one truth class draws no calibration curve or points.
            ax.plot([], [], "x", color=color, gid=gid, label=f"{label}\n{notice}")
            continue
        if kind == "corp":
            rows = record["corp"]
            x = [row["mean_probability"] for row in rows]
            y = [row["calibrated_probability"] for row in rows]
            # Markers keep a block visible when its frequencies coincide (lower == upper).
            ax.plot(
                [bound for row in rows for bound in (row["lower"], row["upper"])],
                np.repeat(y, 2),
                "-o",
                color=color,
                linewidth=1.6 if pooled else 1.1,
                markersize=2.5,
                gid=gid,
                label=f"{label}\nMCB {record['decomposition']['mcb']:.3g}",
            )
        else:
            rows = record["rows"]
            x = [row["mean_probability"] for row in rows]
            y = [row["observed_frequency"] for row in rows]
            if pooled:
                ax.plot(x, y, "-o", color=color, linewidth=1.2, markersize=3, gid=gid, label=label)
            else:
                ax.scatter(x, y, s=22, color=color, alpha=0.8, gid=gid, label=label)
        if pooled:
            counts = np.asarray([row["count"] for row in rows], dtype=float)
            ax.scatter(x, y, s=20 + 180 * counts / counts.max(), color=color, alpha=0.3)
    if kind == "binned":
        ece = summary["metrics"].get("graph/ece")
        xlabel = "Mean edge frequency in bin"
        ylabel = "Share of the bin's edges in graph truth"
        detail = f"{data['bins']} equal-width bins · pooled ECE " + (
            "undefined" if ece is None else f"{ece:.3g}"
        )
        pooled_notice = one_class_notice(data["pooled"]["n"], data["pooled"]["n_pos"])
        if pooled_notice is not None:
            detail += f" ({pooled_notice})"
    else:
        xlabel = "Edge frequency"
        ylabel = "Share of edges in graph truth (isotonic fit)"
        if summary["metrics"]["graph/mcb"] is None:
            split = f"decomposition {_stated(summary, 'graph/mcb')}"
        else:
            split = (
                f"Brier {_stated(summary, 'graph/brier')} = MCB {_stated(summary, 'graph/mcb')}"
                f" − DSC {_stated(summary, 'graph/dsc')} + UNC {_stated(summary, 'graph/unc')}"
            )
        detail = f"CORP isotonic fit, no bins · pooled {split}"
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    _title(
        fig,
        ax,
        _labels(summary["metadata"]),
        f"Edge reliability — {data['label']} against graph truth",
        owned,
        detail=detail,
    )
    return fig


#: Cell values :func:`plot_strata_table` draws, with their axis labels.
STRATA_METRICS = {
    "coverage": "Window totals inside their central 80% interval",
    "crps": "Window-total CRPS",
    "signed_residual": "Signed residual (posterior mean − truth)",
    "abs_residual": "Absolute residual |posterior mean − truth|",
}
#: What each stratification axis groups effects by.
_STRATA_AXIS_LABELS = {
    "true_size_bin": "true size bin (|true window total|)",
    "predicted_size_bin": "predicted size bin (|posterior-mean window total|)",
    "edge_probability_bin": "direct-edge frequency bin",
    "active_channels": "active channels in the world",
    "live_channels": "truly live channels in the world",
}
#: Views of :func:`plot_residual_by_magnitude`: the size its bins take.
RESIDUAL_AXES = ("true", "predicted")


def _stratum_tick(cell, by):
    """A size bin's bounds, or the stratum name of any other axis."""
    if by not in ("true_size_bin", "predicted_size_bin"):
        return cell["stratum"]
    if cell["stratum"] == "zero":
        return "0"
    if cell["upper"] is None:
        return f"> {cell['lower']:.3g}"
    return f"({cell['lower']:.3g}, {cell['upper']:.3g}]"


def _row_residuals(ax, rows, table, by, metric, strata, offsets, width):
    """Each effect's residual at its stratum, ordered by size within it; hollow if uncovered."""
    from pymc_mktfm.diagnostics.strata import row_strata, scored

    position = {stratum: index for index, stratum in enumerate(strata)}
    size = "mean" if by == "predicted_size_bin" else "truth"
    places = row_strata(rows, by, cuts=table["cuts"])
    for family, offset in offsets.items():
        groups = {}
        for row, place in zip(rows, places, strict=True):
            if row["family"] == family and place is not None and scored(row):
                groups.setdefault(place, []).append(row)
        points = {True: ([], []), False: ([], [])}
        for place, members in groups.items():
            members.sort(key=lambda row: abs(row[size]))
            shifts = np.linspace(-0.35, 0.35, len(members)) * width if len(members) > 1 else [0.0]
            for row, shift in zip(members, shifts, strict=True):
                residual = row["mean"] - row["truth"]
                x, y = points[row["low"] <= row["truth"] <= row["high"]]
                x.append(position[place] + offset + shift)
                y.append(abs(residual) if metric == "abs_residual" else residual)
        color = COLORS[family]
        for covered, (x, y) in points.items():
            if not x:
                continue
            ax.scatter(
                x,
                y,
                s=9,
                color=color if covered else "none",
                edgecolors=color,
                linewidths=0.6,
                alpha=0.35,
                gid=f"strata/rows/{family}/{'covered' if covered else 'uncovered'}",
                label=f"{family} effects, {'inside' if covered else 'outside'} their 80% interval",
            )


def plot_strata_table(
    table, *, metric="coverage", by="true_size_bin", rows=None, header=None, ax=None
):
    """One cell value per stratum of ``by`` and family, from a strata summary.

    ``table`` is a :func:`pymc_mktfm.diagnostics.strata_summary`, as the
    evaluator records it (``evaluation_tables/step_<N>.json``) or diagnostics
    compute it. ``metric`` is a key of :data:`STRATA_METRICS`; the signed
    residual adds each stratum's 10%–90% residual range and median, and ``n=``
    counts the scored effects behind each value. A stratum
    without a value is marked ``n/a`` at the bottom, never drawn as zero. With
    ``rows`` (the summary's :func:`pymc_mktfm.diagnostics.effect_rows`), a
    residual metric also shows every effect, ordered by size within its
    stratum and hollow when its interval misses the truth. ``header`` states
    what the table describes.
    """
    if metric not in STRATA_METRICS:
        raise ValueError(f"strata metric must be one of {tuple(STRATA_METRICS)}, got {metric!r}")
    if by not in table["by"]:
        raise ValueError(f"strata by must be one of {tuple(table['by'])}, got {by!r}")
    families = table["by"][by]
    strata = [cell["stratum"] for cell in next(iter(families.values()))["cells"]]
    if not strata:
        raise ValueError(f"no {by} strata: no effect has a value on this axis")
    fig, ax, owned = _single(ax, width=10, row_height=5)
    width, shifts = _bar_offsets(len(families))
    offsets = dict(zip(families, shifts, strict=True))
    residual = metric in ("signed_residual", "abs_residual")
    if rows is not None and residual:
        _row_residuals(ax, rows, table, by, metric, strata, offsets, width)
    reasons = {}
    for family, record in families.items():
        color, offset = COLORS[family], offsets[family]
        drawn = [
            (index, cell) for index, cell in enumerate(record["cells"]) if cell[metric] is not None
        ]
        for index, cell in enumerate(record["cells"]):
            if cell[metric] is None:
                reasons.setdefault(cell["reason"], []).append(index + offset)
        x = [index + offset for index, _ in drawn]
        y = [cell[metric] for _, cell in drawn]
        placed = sum(cell["n_effects"] for cell in record["cells"])
        unplaced = sum(entry["n_effects"] for entry in record["excluded"])
        label = f"{family}: {placed} {'effect' if placed == 1 else 'effects'}"
        if unplaced:
            label += f" ({unplaced} without a {by} stratum)"
        if metric == "signed_residual" and drawn:
            median = np.asarray([cell["residual_median"] for _, cell in drawn])
            ax.errorbar(
                x,
                median,
                yerr=[
                    median - [cell["residual_q10"] for _, cell in drawn],
                    np.asarray([cell["residual_q90"] for _, cell in drawn]) - median,
                ],
                fmt="_",
                color=color,
                markersize=10,
                capsize=3,
                linewidth=1.2,
                gid=f"strata/range/{family}",
            )
        ax.plot(x, y, "o", color=color, markersize=6, gid=f"strata/{family}", label=label)
        # Counts sit in a band along the top, clear of the markers and effect points.
        for (_, cell), position in zip(drawn, x, strict=True):
            ax.text(
                position,
                0.99,
                f"n={cell['n_scored']}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6,
                color=color,
            )
    for reason, positions in reasons.items():
        for x in positions:
            ax.text(
                x,
                0.02,
                "n/a",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#777777",
            )
        ax.plot([], [], " ", label=f"n/a: {reason}")
    if metric == "coverage":
        ax.axhline(0.8, color="black", linestyle="--", linewidth=1, label="Nominal 80%")
        ax.set_ylim(-0.02, 1.08)
    else:
        if metric == "signed_residual":
            ax.axhline(0.0, color="black", linewidth=0.8)
        # Room for the count band above and the n/a marks below the values.
        low, high = ax.get_ylim()
        ax.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))
    unit = "" if metric == "coverage" else f" ({table['unit']})"
    ticks = [_stratum_tick(cell, by) for cell in next(iter(families.values()))["cells"]]
    ax.set_xticks(range(len(strata)), ticks, fontsize=8, rotation=20 if len(strata) > 6 else 0)
    ax.set(
        xlim=(-0.6, len(strata) - 0.4),
        xlabel=_STRATA_AXIS_LABELS[by].capitalize(),
        ylabel=f"{STRATA_METRICS[metric]}{unit}",
    )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    detail = f"per-effect window totals · units: {table['unit']} · {table['interval']}"
    if by in ("true_size_bin", "predicted_size_bin"):
        cuts = ", ".join(f"{cut:.3g}" for cut in table["cuts"]) or "none"
        detail += f" · size cuts {cuts}; the same cuts bin true and predicted size"
    if metric == "signed_residual":
        detail += " · dot: mean · bar: median, 10%–90% of effects"
    _title(
        fig,
        ax,
        header,
        f"{STRATA_METRICS[metric]} by {_STRATA_AXIS_LABELS[by]}",
        owned,
        detail=detail,
    )
    return fig


def plot_strata(summary, *, metric="coverage", by="true_size_bin", ax=None):
    """:func:`plot_strata_table` of one result's effects, from its diagnostic summary.

    The summary's cuts come from this world's effects; residual metrics also
    show each effect. Needs simulation truth and the result's ``sales_scale``.
    """
    data = summary["arrays"].get("strata")
    if data is None:
        raise ValueError("stratified readouts require simulation truth and sales_scale provenance")
    return plot_strata_table(
        data["table"],
        metric=metric,
        by=by,
        rows=data["rows"],
        header=_labels(summary["metadata"]),
        ax=ax,
    )


def plot_residual_by_magnitude(summary, *, axis="true", ax=None):
    """Signed window-total residual (posterior mean − truth) against effect size.

    ``axis="true"`` bins effects by true size and ``"predicted"`` by posterior
    mean size, with the same cuts: each bin's residual mean, median and
    10%–90% range, and every effect ordered by size within its bin. Residuals
    are in ``sales_scale`` units and not divided by the truth.
    """
    if axis not in RESIDUAL_AXES:
        raise ValueError(f"residual axis must be one of {RESIDUAL_AXES}, got {axis!r}")
    return plot_strata(summary, metric="signed_residual", by=f"{axis}_size_bin", ax=ax)


def plot_zero_atoms(summary, *, ax=None):
    entries = _components(summary, tuple(COLORS))
    fig, ax, owned = _single(ax, width=11, row_height=max(4, len(entries) * 0.3))
    ax.barh(
        range(len(entries)),
        [entry["zero_atom_probability"] for entry in entries],
        color=[COLORS[entry["family"]] for entry in entries],
    )
    ax.set_yticks(range(len(entries)), [entry["name"] for entry in entries], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Fraction of draws with an exactly zero full trajectory")
    _title(
        fig,
        ax,
        _labels(summary["metadata"]),
        (
            "Zero atoms — not inferred causal absence; graph false-exclusion "
            "metrics reported separately"
        ),
        owned,
    )
    return fig


def plot_uncertainty(summary, *, ax=None):
    uncertainty = summary["arrays"]["total_uncertainty"]
    time = summary["metadata"]["time"]
    within = np.asarray(uncertainty["within_graph_variance"])
    between = np.asarray(uncertainty["between_graph_variance"])
    fig, ax, owned = _single(ax, width=11, row_height=4)
    time = _plot_time(ax, time)
    ax.stackplot(
        time,
        within,
        between,
        labels=["Within graph (ddof=0)", "Between graph (ddof=0)"],
        colors=["#56B4E9", "#E69F00"],
        alpha=0.75,
    )
    ax.plot(time, uncertainty["total_variance"], "k--", label="Total empirical variance")
    _boundary(ax, summary["metadata"])
    ax.set_ylabel(f"Variance ({summary['metadata']['units']})²")
    ax.legend(fontsize=8)
    _title(
        fig,
        ax,
        _labels(summary["metadata"]),
        "Finite paired-draw uncertainty of the latent total — singleton zero variance is not "
        "certainty",
        owned,
        detail=(
            f"{uncertainty['n_unique_graphs']} graphs; "
            f"{uncertainty['n_singleton_groups']} singleton groups; "
            f"rare draw mass {uncertainty['rare_draw_mass']:.1%}"
        ),
    )
    return fig


#: Marker of each coverage line in drawing order. Markers are hollow and each
#: later one is smaller, so families with equal coverage nest visibly.
_COVERAGE_MARKERS = ("o", "s", "D", "^", "v", "P")


def plot_calibration(summary, *, kind="coverage", ax=None):
    """One-world screening: interval ``"coverage"``, or ``"ranks"``/``"ecdf"`` per family.

    Coverage draws every family on one axes, so it accepts ``ax=``; the rank
    and ECDF grids draw one axes per family and refuse it. Any other ``kind``
    raises ``ValueError``.
    """
    if kind not in ("coverage", "ranks", "ecdf"):
        raise ValueError(f"calibration kind must be 'coverage', 'ranks' or 'ecdf', got {kind!r}")
    groups = summary["arrays"]["calibration"]
    if not groups:
        raise ValueError("calibration requires simulation truth")
    title = "Dependent one-world screening — not independent-world SBC; no pass/fail"
    if kind == "coverage":
        fig, ax, owned = _single(ax, width=8, row_height=5)
        for number, (name, record) in enumerate(groups.items()):
            ax.plot(
                [row["level"] for row in record["intervals"]],
                [row["coverage"] for row in record["intervals"]],
                marker=_COVERAGE_MARKERS[number % len(_COVERAGE_MARKERS)],
                markersize=max(3.5, 13 - 1.9 * number),
                markerfacecolor="none",
                markeredgewidth=1.5,
                color=COLORS[name],
                label=name,
                clip_on=False,
            )
        ax.plot([0, 1], [0, 1], "k--", label="Nominal", zorder=1)
        # Margins keep markers at coverage 0 and 1 clear of the spines.
        ax.set(
            xlabel="Central interval level",
            ylabel="Observed coordinate coverage",
            xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=8)
        _title(fig, ax, _labels(summary["metadata"]), title, owned)
        return fig
    if ax is not None:
        raise ValueError(
            f"{kind} calibration needs one axes per family ({len(groups)} families); "
            "ax= draws only kind='coverage'"
        )
    fig, axes = _figure(math.ceil(len(groups) / 2), 2)
    for ax, (name, record) in zip(axes.flat, groups.items()):
        if kind == "ranks":
            edges = np.asarray(record["rank_bin_edges"])
            ax.bar(
                edges[:-1],
                record["rank_histogram"],
                width=np.diff(edges),
                align="edge",
                color=COLORS[name],
                alpha=0.65,
            )
            ax.stairs(
                record["expected_rank_histogram"],
                edges,
                color="black",
                label="Discrete uniform expectation",
            )
            ax.set(xlabel="Randomized rank (inclusive zero ties)", ylabel="Coordinate count")
        else:
            x = record["ecdf_grid"]
            ax.step(
                x,
                record["ecdf"],
                where="post",
                color=COLORS[name],
                label="Observed mid-PIT ECDF",
            )
            ax.step(
                x,
                record["null_ecdf"],
                where="post",
                color="black",
                linestyle="--",
                label="Discrete uniform rank ECDF",
            )
            ax.set(xlabel="Mid-PIT", ylabel="ECDF", xlim=(0, 1), ylim=(0, 1))
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(groups) :]:
        ax.set_visible(False)
    _header(fig, summary, title)
    return fig


def plot_support(summary):
    support = summary["arrays"]["support"]
    fig, axes = _figure(2, 1, row_height=3)
    ax, note = axes[:, 0]
    values = [
        sum(support["component_mask"]),
        len(support["component_mask"]) - sum(support["component_mask"]),
        sum(support["edge_mask"]),
        len(support["edge_mask"]) - sum(support["edge_mask"]),
    ]
    ax.bar(
        ["Active components", "Padded components", "Active edges", "Padded edges"],
        values,
        color=["#0072B2", "#999999", "#009E73", "#999999"],
    )
    ax.set_ylabel("Slot count")
    descriptions = [support["qualification"]]
    inputs = support["input_support"]
    if isinstance(inputs, dict):
        for name, record in inputs.items():
            if isinstance(record, dict) and all(key in record for key in ("mean", "std")):
                descriptions.append(
                    f"{name}: mean={np.array2string(np.asarray(record['mean']), precision=2)}; "
                    f"std={np.array2string(np.asarray(record['std']), precision=2)}; "
                    f"units={record.get('units', 'not recorded')}"
                )
    else:
        descriptions.append(
            "Input support descriptors unavailable; observations were not supplied in provenance."
        )
    descriptions.append(
        "Prior/corpus provenance: recorded in evidence"
        if support["prior"] is not None
        else "Prior/corpus provenance unavailable; no OOD certification."
    )
    note.text(
        0, 1, "\n".join(descriptions), transform=note.transAxes, va="top", fontsize=9, wrap=True
    )
    note.set_axis_off()
    _header(fig, summary, "Padding, entity support, and recorded input descriptors")
    return fig


def plot_records(records, *, kind="training", metric_names=None, header=None, ax=None):
    """Plot recorded scalar trajectories, optionally selecting exact nested names.

    ``records`` is a :func:`pymc_mktfm.diagnostics.record_series` summary. By
    default evaluation shows a labeled overview. ``metric_names`` makes any
    recorded non-headline objective, auxiliary-head readout, uncertainty or
    calibration series plottable without recomputation. ``header`` states what
    the records describe (for example the run); decomposition-result labels
    never describe training curves, so none are added. Each axes holds up to
    six series of one group; with ``ax=`` the series must fit one axes, which
    is the caller's, and more raise ``ValueError``. The ``ax=`` title is
    ``header`` alone when that names the series group, as a run's evaluation
    header names its metric.
    """
    series = records["series"]
    if metric_names is not None:
        missing = set(metric_names) - series.keys()
        if missing:
            raise KeyError(f"unknown {kind} metric names: {sorted(missing)}")
        series = {name: series[name] for name in metric_names}
    elif kind == "evaluation":
        flat = {name: values for name, values in series.items() if name.startswith("flat/")}
        if flat:
            series = flat
        # Whole trailing path segments, so "shd_mean" never selects "normalized_shd_mean".
        headlines = tuple(
            f"/{metric}"
            for metric in (
                "decomposition_energy_score",
                "closure_rel_err",
                "contribution_crps",
                "control_contribution_crps",
                "cy/brier",
                "cy/log_score",
                "shd_mean",
                "exact_recovery",
            )
        )
        selected = {
            name: values for name, values in series.items() if f"/{name}".endswith(headlines)
        }
        if selected:
            series = selected
    if not series:
        raise ValueError(f"no numeric {kind} trajectories")
    groups = {}
    for name, values in series.items():
        if kind == "training":
            group = (
                "Objectives"
                if "loss" in name or name.startswith("raw_")
                else ("Learning rate" if name == "lr" else "Health / timing / data")
            )
        else:
            group = name.rsplit("/", 1)[-1]
        groups.setdefault(group, []).append((name, values))
    pages = [
        (group, rows[start : start + 6])
        for group, rows in groups.items()
        for start in range(0, len(rows), 6)
    ]
    if len(pages) == 1:
        fig, ax, owned = _single(ax, width=12, row_height=3.4)
        axes = [ax]
    elif ax is not None:
        raise ValueError(
            f"these {kind} series need {len(pages)} axes; ax= draws only single-axes figures"
        )
    else:
        fig, grid = _figure(len(pages), 1, width=12, row_height=3.4)
        axes, owned = grid[:, 0], True
    for ax, (group, rows) in zip(axes, pages):
        for name, values in rows:
            ax.plot(
                [row["step"] if row["step"] is not None else row["record_index"] for row in values],
                [np.nan if row["value"] is None else row["value"] for row in values],
                label=name,
                marker="." if len(values) < 10 else None,
                linewidth=1.2,
            )
        xlabel = (
            "Producing training step"
            if records["coordinate"] == "step"
            else "Record index (training step not supplied)"
        )
        ax.set(xlabel=xlabel, ylabel="Recorded value", title=group)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    title = (
        "Recorded training objectives / schedule / health"
        if kind == "training"
        else (
            "Checkpoint trajectories — selected recovery / structure scores; "
            "all values retained in evidence"
        )
    )
    if owned:
        _suptitle(fig, title, header)
    elif header is not None and pages[0][0] in header.split():
        _title(fig, axes[0], header, None, owned)
    else:
        # The caller's axes holds the whole heading, its series group included.
        _title(fig, axes[0], header, title, owned, detail=pages[0][0])
    return fig


#: Per-world histograms of a ``corpus_summary`` split, with their axis labels.
_CORPUS_HISTOGRAMS = (
    ("context_lengths", "Context length (weeks)"),
    ("active_channels", "Active channels"),
    ("active_controls", "Active controls"),
    ("active_latents", "Active latent sources"),
)
_SPLIT_COLORS = {"train": COLORS["channel"], "holdout": COLORS["latent"]}


def plot_corpus_summary(summary, *, header=None):
    """Training-corpus composition, as recorded in ``run.json`` ``corpus_summary``.

    Split sizes; each split's share of worlds by context length and by active
    channel, control and latent count; and each edge family's edges per world
    (bar: mean over the split's worlds, whiskers: fewest to most). ``header``
    states the run the corpus trained.
    """
    splits = [split for split in ("train", "holdout") if summary.get(split, {}).get("worlds")]
    if not splits:
        raise ValueError("corpus summary records no worlds")
    width, offsets = _bar_offsets(len(splits))
    fig, axes = _figure(3, 2, width=12, row_height=3.3)
    sizes = axes[0, 0]
    quantities = (("worlds", "Worlds"), ("cells", "Cells"), ("distinct_graphs", "Distinct graphs"))
    for split, offset in zip(splits, offsets, strict=True):
        sizes.bar(
            np.arange(len(quantities)) + offset,
            [summary[split][key] for key, _ in quantities],
            width,
            color=_SPLIT_COLORS[split],
            label=split,
        )
    sizes.set_xticks(range(len(quantities)), [label for _, label in quantities])
    shared = summary["shared_graphs"]
    sizes.set(ylabel="Count", title=f"Split sizes · {shared} graphs in both splits")
    sizes.legend(fontsize=8)
    for ax, (name, label) in zip(axes.flat[1:5], _CORPUS_HISTOGRAMS, strict=True):
        values = sorted({int(value) for split in splits for value in summary[split][name]})
        for split, offset in zip(splits, offsets, strict=True):
            histogram, worlds = summary[split][name], summary[split]["worlds"]
            ax.bar(
                np.arange(len(values)) + offset,
                [histogram.get(str(value), 0) / worlds for value in values],
                width,
                color=_SPLIT_COLORS[split],
                label=split,
            )
        ax.set_xticks(range(len(values)), [str(value) for value in values])
        ax.set(xlim=(-0.75, len(values) - 0.25), xlabel=label, ylabel="Share of split worlds")
        ax.legend(fontsize=8)
    edges = axes[2, 1]
    families = list(summary[splits[0]]["edges"])
    for split, offset in zip(splits, offsets, strict=True):
        histograms = [
            {int(count): worlds for count, worlds in summary[split]["edges"][family].items()}
            for family in families
        ]
        means = np.asarray(
            [
                sum(count * n for count, n in histogram.items()) / sum(histogram.values())
                for histogram in histograms
            ]
        )
        fewest = np.asarray([min(histogram) for histogram in histograms])
        most = np.asarray([max(histogram) for histogram in histograms])
        edges.bar(
            np.arange(len(families)) + offset,
            means,
            width,
            yerr=[means - fewest, most - means],
            capsize=2,
            color=_SPLIT_COLORS[split],
            label=split,
        )
    edges.set_xticks(range(len(families)), families)
    edges.set(xlabel="Edge family", ylabel="Edges per world")
    edges.legend(fontsize=8)
    title = (
        f"Training corpus — {summary['corpora']} "
        f"{'corpus' if summary['corpora'] == 1 else 'corpora'}: split sizes, context lengths, "
        "active nodes and edges per family"
    )
    _suptitle(fig, title, header)
    return fig


def plot_set_posterior(summary):
    """Render explicitly supplied cardinality-conditioned marginal approximations."""
    groups = summary["arrays"]["set_posterior"]
    if not groups:
        raise ValueError("set posterior requires explicit design cardinalities")
    fig, axes = _figure(len(groups), 1, row_height=4)
    for ax, (family, record) in zip(axes[:, 0], groups.items()):
        probabilities = np.exp(record["log_probabilities"])
        order = np.argsort(-probabilities, kind="stable")[:15]
        values = probabilities[order].tolist()
        names = [str(record["subsets"][index]) for index in order]
        colors = ["#D55E00" if index == record["true_index"] else "#0072B2" for index in order]
        if len(order) < len(probabilities):
            values.append(float(1 - probabilities[order].sum()))
            names.append("All other sets")
            colors.append("#999999")
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Approximate conditional set probability")
        ax.set_title(
            f"{family}: design k={record['cardinality']}; source={record['source']}", fontsize=10
        )
    _header(fig, summary, "Factorized cardinality approximation — NOT complete-graph posterior")
    return fig


def _saved_reliability_kind(summary):
    """The CORP view, or the bins of reliability evidence saved before CORP fits were recorded."""
    data = summary["arrays"].get("reliability")
    return "binned" if data is not None and "corp" not in data["pooled"] else "corp"


def plot_panel(name, summary) -> Figure | GraphvizGraph:
    """Render a named applicable catalog panel solely from computed arrays."""
    dispatch = {
        "target_recovery": plot_target_recovery,
        "channel_contributions": lambda data: plot_component_trajectories(
            data, families=("channel",)
        ),
        "channel_totals": plot_channel_totals,
        "components": plot_family_totals,
        "indirect_sources": lambda data: plot_component_trajectories(data, families=("indirect",)),
        "controls": lambda data: plot_component_trajectories(data, families=("control",)),
        "recovery_scatter": plot_recovery_scatter,
        "attribution": plot_attribution,
        "window_totals": plot_window_totals,
        "cumulative": plot_cumulative,
        "temporal": plot_temporal,
        "residual_true_size": lambda data: plot_residual_by_magnitude(data, axis="true"),
        "residual_predicted_size": lambda data: plot_residual_by_magnitude(data, axis="predicted"),
        "strata": plot_strata,
        "structure_heatmap": plot_structure_heatmap,
        "dag_recovery": plot_dag,
        "dag_posterior": plot_graph_posterior,
        "graph_cardinality": plot_cardinality,
        "graph_statistics": plot_graph_statistics,
        "cycles": plot_cycles,
        "reliability": lambda data: plot_reliability(data, kind=_saved_reliability_kind(data)),
        "support": plot_support,
        "zero_atoms": plot_zero_atoms,
        "graph_uncertainty": plot_uncertainty,
        "calibration_pit": lambda data: plot_calibration(data, kind="ranks"),
        "calibration_ecdf": lambda data: plot_calibration(data, kind="ecdf"),
        "coverage": plot_calibration,
        "training": lambda data: plot_records(data["arrays"]["training"]),
        "evaluation": lambda data: plot_records(data["arrays"]["evaluation"], kind="evaluation"),
        "set_posterior": plot_set_posterior,
    }
    if name == "reference_comparison":
        from .comparisons import plot_reference_comparison

        return plot_reference_comparison(summary["arrays"]["reference"])
    if name == "input_support":
        from .comparisons import plot_input_support

        return plot_input_support(summary["arrays"]["input_support"])
    if name in ("posterior_sbc", "c2st"):
        from .comparisons import plot_c2st, plot_posterior_sbc

        renderer = plot_posterior_sbc if name == "posterior_sbc" else plot_c2st
        return renderer(summary["arrays"]["reference"][name], metadata=summary["metadata"])
    if name not in dispatch:
        raise KeyError(f"unknown diagnostic panel {name!r}")
    return dispatch[name](summary)


def _plot_conditional_sbc_report(report, *, kind):
    """Draw saved live-effect conditional and graph-averaged PIT evidence."""
    if kind not in ("ranks", "ecdf", "coverage"):
        raise ValueError("expected ranks/ecdf/coverage kind for conditional SBC")
    families = [(family, report["families"].get(family) or {}) for family in ("channel", "control")]
    entries = [(f"{family} · one effect per cell", group) for family, group in families]
    for family, group in families:
        entries.extend(
            (f"{family} · {name}", entity) for name, entity in group.get("entities", {}).items()
        )
    fig, axes = _figure(math.ceil(len(entries) / 2), 2, width=12, row_height=3.8)
    for ax, (name, group) in zip(axes.flat, entries):
        if kind == "coverage":
            ax.plot([0, 1], [0, 1], "k--", label="Nominal")
            ax.set(
                xlabel="Central PIT level",
                ylabel="PIT-implied coverage",
                xlim=(0, 1),
                ylim=(0, 1),
            )
        unavailable = []
        for mode, label, color in (
            ("conditional", "Edge-on conditional", COLORS["channel"]),
            ("model_average", "Graph-averaged", COLORS["control"]),
        ):
            record = group.get(mode)
            if record is None:
                unavailable.append(label)
                continue
            label = f"{label} ({record['n_ranks']} worlds)"
            if kind == "ranks":
                edges = np.asarray(record["rank_bin_edges"])
                ax.stairs(
                    record["rank_histogram"],
                    edges,
                    fill=True,
                    color=color,
                    alpha=0.25,
                    label=label,
                )
                ax.stairs(
                    record["expected_rank_histogram"],
                    edges,
                    color=color,
                    linestyle=":",
                    label=f"{mode.replace('_', ' ')} null expectation",
                )
                ax.set(xlabel="Saved PIT bin", ylabel="Independent-world count", xlim=(0, 1))
            elif kind == "ecdf":
                grid = record["ecdf_grid"]
                ax.step(grid, record["ecdf"], where="post", color=color, label=label)
                ax.step(
                    grid,
                    record["null_ecdf"],
                    where="post",
                    linestyle="--",
                    color=color,
                    label=f"{mode.replace('_', ' ')} saved null",
                )
                if record.get("ecdf_lower") is not None and record.get("ecdf_upper") is not None:
                    ax.fill_between(
                        grid,
                        record["ecdf_lower"],
                        record["ecdf_upper"],
                        step="post",
                        color=color,
                        alpha=0.15,
                    )
                ax.set(xlabel="PIT", ylabel="Saved ECDF", xlim=(0, 1), ylim=(0, 1))
            else:
                coverage = sorted(
                    (float(key.removeprefix("cov")) / 100, value)
                    for key, value in record["rank_coverage"].items()
                )
                ax.plot(
                    [level for level, _ in coverage],
                    [value for _, value in coverage],
                    "o-",
                    color=color,
                    label=label,
                )
        if unavailable:
            ax.text(
                0.02,
                0.02,
                f"Unavailable: {', '.join(unavailable)}",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
            )
        ax.set_title(_fit(name, fig.get_figwidth() / 2, 10, longest=_LONGEST_WORD), fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, ncol=2)
    for ax in list(axes.flat)[len(entries) :]:
        ax.set_visible(False)
    provenance = report.get("provenance", {})
    fig.suptitle(
        "Independent-world live-effect SBC · conditional versus graph-averaged PIT\n"
        "Conditional: live truth edges only · graph-averaged: all eligible truth edges\n"
        "A non-rejection is not proof of calibration or OOD validity\n"
        f"{provenance.get('head', 'unspecified')} head · "
        f"{provenance.get('window', 'unspecified')} window · "
        f"{provenance.get('mode', 'unspecified')} mode · "
        f"{len(report['world_ids'])} worlds / {len(report['cell_ids'])} cells",
        fontsize=12,
    )
    return fig


def plot_sbc_report(report, *, kind="ecdf"):
    """Independent-world SBC rank/ECDF/coverage figures from a saved report card.

    ECDF bands are the stored Monte Carlo-calibrated discrete-null bands.
    Coverage is rank-implied, not sample-quantile coverage, and each plotted
    quantity contributes exactly one rank per declared independent world.

    Conditional live-effect cards instead plot saved PIT evidence for both
    edge-on and graph-averaged ranks, with family rows before per-entity rows.
    """
    if report.get("kind") == "independent_world_conditional_sbc":
        return _plot_conditional_sbc_report(report, kind=kind)
    if report.get("kind") != "independent_world_sbc" or kind not in ("ranks", "ecdf", "coverage"):
        raise ValueError("expected an independent_world_sbc report and ranks/ecdf/coverage kind")
    quantities = report["quantities"]
    fig, axes = _figure(math.ceil(len(quantities) / 2), 2, width=11)
    for ax, (name, record) in zip(axes.flat, quantities.items()):
        if kind == "ranks":
            edges = np.asarray(record["rank_bin_edges"])
            ax.bar(
                edges[:-1],
                record["rank_histogram"],
                width=np.diff(edges),
                align="edge",
                color="#0072B2",
                alpha=0.6,
            )
            ax.stairs(
                record["expected_rank_histogram"],
                edges,
                color="black",
                label="Discrete uniform expectation",
            )
            ax.set(xlabel="Randomized rank", ylabel="Independent-world count")
        elif kind == "ecdf":
            grid = record["ecdf_grid"]
            ax.step(grid, record["ecdf"], where="post", color="#0072B2", label="Saved rank ECDF")
            ax.step(
                grid,
                record["null_ecdf"],
                where="post",
                color="black",
                linestyle="--",
                label="Discrete null ECDF",
            )
            if "ecdf_lower" in record:
                ax.fill_between(
                    grid,
                    record["ecdf_lower"],
                    record["ecdf_upper"],
                    step="post",
                    color="#999999",
                    alpha=0.3,
                    label="Simultaneous null band",
                )
            ax.set(xlabel="Mid-PIT", ylabel="ECDF", xlim=(0, 1), ylim=(0, 1))
        else:
            coverage = sorted(
                (float(key.removeprefix("cov")) / 100, value)
                for key, value in record["rank_coverage"].items()
            )
            ax.plot(
                [level for level, _ in coverage],
                [value for _, value in coverage],
                "o-",
                color="#0072B2",
                label="Rank-implied coverage",
            )
            ax.plot([0, 1], [0, 1], "k--", label="Nominal")
            ax.set(xlabel="Central level", ylabel="Rank-implied coverage", xlim=(0, 1), ylim=(0, 1))
        ax.set_title(
            f"{name} · {record['n_ranks']} worlds · {record['n_post_samples']} draws", fontsize=10
        )
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(quantities) :]:
        ax.set_visible(False)
    provenance = report.get("provenance", {})
    fig.suptitle(
        "Independent-world SBC (declared sampling contract)\n"
        "A non-rejection is not proof of calibration or OOD validity\n"
        f"{provenance.get('head', 'unspecified')} head · "
        f"{provenance.get('window', 'unspecified')} window · "
        f"{provenance.get('mode', 'unspecified')} mode · "
        f"units: {provenance.get('units', 'rank')} · {len(report['world_ids'])} worlds",
        fontsize=12,
    )
    return fig
