"""Readable comparison figures drawn only from stored numerical summaries.

Matplotlib is imported on demand. Rendering never invokes an inference model,
generator, or reference sampler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, title, *, note=None):
    fig.suptitle(title, fontsize=13)
    handles, labels = [], []
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
        found, names = ax.get_legend_handles_labels()
        for handle, name in zip(found, names, strict=True):
            if name not in labels:
                handles.append(handle)
                labels.append(name)
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9, frameon=False
        )
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.1 if note else 0.07, 1, 0.96))
    return fig


def _reference_note(summary):
    if summary.get("reference_fit_summary") or any(
        row.get("reference_health") is not None for row in summary.get("records", ())
    ):
        return (
            " References are approximate; failures and incomplete fit "
            "diagnostics remain included. See reference fit-health panel."
        )
    return ""


def _aggregate_figure(
    summary, metric, title, *, window="query", families=None, reference=None, spread=False
):
    rows = [
        row
        for row in summary["aggregate"]
        if row["metric"] == metric
        and row["window"] == window
        and (families is None or row["family"] in families)
    ]
    if not rows:
        raise ValueError(f"No saved {metric} evidence for {window}")
    families = list(dict.fromkeys(row["family"] for row in rows))
    cases = list(dict.fromkeys(row["case"] for row in rows))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    plt = _pyplot()
    fig, axes = plt.subplots(
        1, len(families), figsize=(5 * len(families), max(4, 0.7 * len(cases) + 2)), squeeze=False
    )
    offsets = np.linspace(-0.3, 0.3, len(methods)) if len(methods) > 1 else [0]
    for ax, family in zip(axes[0], families, strict=True):
        for j, method in enumerate(methods):
            selected = [row for row in rows if row["family"] == family and row["method"] == method]
            for row in selected:
                y = cases.index(row["case"]) + offsets[j]
                color = f"C{j % 10}"
                if spread:
                    ax.plot([row["median"], row["p90"]], [y, y], color=color, alpha=0.5)
                    ax.plot(row["p90"], y, "|", color=color)
                    x = row["median"]
                else:
                    if row["ci_low"] is not None and row["ci_high"] is not None:
                        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color)
                    x = row["mean"]
                ax.plot(x, y, "o", color=color, label=method)
        if reference is not None:
            ax.axvline(reference, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(cases)), labels=cases)
        ax.invert_yaxis()
        ax.set_title(family.replace("_", " "))
        ax.set_xlabel(metric.replace("_", " "))
    note = (
        "Median to 90th percentile across worlds."
        if spread
        else (
            "World means and 95% world-bootstrap intervals where retained; "
            "not intervals over training seeds."
        )
    )
    return _finish(fig, f"{title} — {window}", note=note + _reference_note(summary))


def plot_performance(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "crps",
        "Window-mean CRPS / recorded outcome scale",
        window=window,
        families=("channel", "control", "total"),
    )


def plot_coverage(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "coverage",
        "Coverage of equal-tailed window-mean intervals",
        window=window,
        families=("channel", "control", "total"),
        reference=summary["settings"]["mass"],
    )


def plot_recovery(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "recovered",
        "All live effects covered and none falsely excluded",
        window=window,
        families=("all",),
    )


def plot_failure_size(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "crps",
        "Typical and tail CRPS",
        window=window,
        families=("channel", "control", "total"),
        spread=True,
    )


def plot_false_effects(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "false_effect_rate",
        (
            "Attribution to inactive inputs ≥ "
            f"{summary['settings']['false_effect_threshold']:g}"
            " × outcome scale"
        ),
        window=window,
        families=("channel", "control"),
    )


def plot_five_controls(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "five_control_false_exclusions",
        "False exclusions among five generating-live controls",
        window=window,
        families=("all",),
        reference=0,
    )


def plot_error_decomposition(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "total_signed_error",
        "Signed baseline and generating-live family totals / outcome scale",
        window=window,
        families=("baseline", "channel", "control"),
        reference=0,
    )


def plot_component_hit_rate(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "component_hit_rate",
        "Fraction of live components covered",
        window=window,
        reference=summary["settings"]["mass"],
    )


def plot_control_count_recovery(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "five_control_modal_count_hit",
        "Modal full-graph count recovers five live controls",
        window=window,
        reference=1,
    )


def plot_inactive_attribution(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "inactive_mean_attribution",
        "Absolute attribution to truly inactive inputs / outcome scale",
        window=window,
        reference=0,
    )


def plot_false_effect_worlds(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "worlds_with_false_effect",
        "Worlds with any effect-sized false attribution",
        window=window,
        reference=0,
    )


def plot_absorbed_share(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "absorbed_share",
        "Omitted contribution absorbed elsewhere (0 missing, 1 fully absorbed)",
        window=window,
        reference=0,
    )


def plot_structure_recovery(summary, *, window="query"):
    return _aggregate_figure(
        summary,
        "structure_exact_hit",
        "Empirical complete-graph mode exactly recovers truth",
        window=window,
        reference=1,
    )


def plot_graph_count_recovery(summary, *, family, window="query"):
    return _aggregate_figure(
        summary,
        f"family/{family}/modal_count_hit",
        f"Empirical full-graph mode recovers {family} edge count",
        window=window,
        reference=1,
    )


def _paired_figure(rows, title, *, window, label_keys, note=""):
    rows = [row for row in rows if row["window"] == window]
    if not rows:
        raise ValueError(f"No saved paired evidence for {window}")
    families = list(dict.fromkeys(row["family"] for row in rows))
    plt = _pyplot()
    height = max(
        4, max(sum(row["family"] == family for row in rows) for family in families) * 0.4 + 2
    )
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), height), squeeze=False)
    for ax, family in zip(axes[0], families, strict=True):
        selected = [row for row in rows if row["family"] == family]
        for index, row in enumerate(selected):
            ax.plot([row["ci_low"], row["ci_high"]], [index, index], color="C0")
            ax.plot(row["mean"], index, "o", color="C0")
        ax.set_yticks(
            range(len(selected)),
            labels=[" · ".join(str(row[key]) for key in label_keys) for row in selected],
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.invert_yaxis()
        ax.set_xlabel("Paired CRPS difference / outcome scale")
        ax.set_title(family)
    return _finish(
        fig,
        f"{title} — {window}",
        note="Paired within world; intervals bootstrap worlds, never individual components." + note,
    )


def plot_ab_comparison(summary, *, window="query"):
    rows = [row for row in summary["paired"] if row.get("kind") == "method_comparison"]
    return _paired_figure(
        rows,
        "Method comparison: second minus first",
        window=window,
        label_keys=("case", "first", "second"),
        note=_reference_note(summary),
    )


def plot_oracle_comparison(summary, *, window="query"):
    rows = [row for row in summary["paired"] if row.get("kind") == "oracle_comparison"]
    return _paired_figure(
        rows,
        "Method minus privileged approximate oracle — normalized CRPS",
        window=window,
        label_keys=("case", "second"),
        note=_reference_note(summary),
    )


def plot_graph_averaging(summary, *, window="query"):
    rows = [row for row in summary["paired"] if row.get("kind") == "graph_averaging"]
    return _paired_figure(
        rows,
        "Model average minus graph view (negative favors averaging)",
        window=window,
        label_keys=("case", "first", "second"),
    )


def plot_matched_omission(summary, *, window="query"):
    return _paired_figure(
        summary["omission_summary"],
        "Matched omission: after minus before on retained identities",
        window=window,
        label_keys=("case", "method"),
    )


def plot_world_comparison(summary, *, case, world, window="query", projection="trajectory"):
    """Compare stored trajectories or distributions without resampling either window."""
    if projection not in ("trajectory", "window_mean"):
        raise ValueError("projection must be trajectory or window_mean")
    records = [
        row
        for row in summary["records"]
        if (row["case"], row["world"], row["window"]) == (case, world, window)
    ]
    if not records:
        raise ValueError("No saved records match the requested world and window")
    names = ["Latent total"] + [row["name"] for row in records[0]["components"]]
    plt = _pyplot()
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows + 1), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for method_index, record in enumerate(records):
            component = (
                record["total"]
                if name == "Latent total"
                else next(row for row in record["components"] if row["name"] == name)
            )
            color = f"C{method_index % 10}"
            label = f"{record['method']} ({record['mode']})"
            if projection == "trajectory":
                values = component["weekly"]
                ax.plot(record["time"], values["mean"], color=color, label=label)
                ax.fill_between(
                    record["time"], values["low"], values["high"], color=color, alpha=0.12
                )
                if method_index == 0 and component["truth"] is not None:
                    ax.plot(
                        record["time"],
                        component["truth"],
                        color="black",
                        linestyle="--",
                        label="Generating truth",
                    )
                if method_index == 0 and name == "Latent total" and record["observed"] is not None:
                    ax.scatter(
                        record["time"],
                        record["observed"],
                        color="black",
                        s=9,
                        alpha=0.5,
                        label="Observed outcome (includes noise)",
                    )
                ax.set_xlabel(f"Time — {window} marginal only")
                ax.set_ylabel(record["units"])
            else:
                values = component["window_mean"]
                distribution = values["distribution"]
                ax.stairs(distribution["density"], distribution["edges"], color=color, label=label)
                ax.axvline(values["mean"], color=color, linewidth=1)
                ax.axvspan(values["low"], values["high"], color=color, alpha=0.08)
                if method_index == 0 and component["truth"] is not None:
                    ax.axvline(
                        values["truth"], color="black", linestyle="--", label="Generating truth"
                    )
                ax.set_xlabel(f"Window mean ({record['units']})")
                ax.set_ylabel("Posterior density (atoms retained)")
        ax.set_title(name)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    warnings = [
        row["method"]
        for row in records
        if row["reference_health"] is not None and row["reference_health"]["healthy"] is not True
    ]
    note = (
        "Equal-tailed "
        f"{records[0]['mass']:.0%}"
        " intervals; latent totals are not noisy posterior predictions."
    )
    if warnings:
        note += " Reference health warning: " + ", ".join(warnings)
    return _finish(
        fig, f"{case} · world {world} · {window} · {projection.replace('_', ' ')}", note=note
    )


def plot_reference_comparison(summary):
    """Render the reference report hook's saved values and explicit health warning."""
    arrays = summary["arrays"]
    names = arrays["names"]
    plt = _pyplot()
    ncols = min(3, len(names))
    fig, axes = plt.subplots(
        (len(names) + ncols - 1) // ncols,
        ncols,
        squeeze=False,
        figsize=(5 * ncols, 3 * ((len(names) + ncols - 1) // ncols) + 1),
    )
    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        for source, color, label in (
            ("model", "C0", "Conditional model"),
            ("reference", "C1", summary["label"]),
        ):
            values = arrays[source]
            ax.plot(arrays["time"], np.asarray(values["mean"])[:, i], color=color, label=label)
            ax.fill_between(
                arrays["time"],
                np.asarray(values["low"])[:, i],
                np.asarray(values["high"])[:, i],
                color=color,
                alpha=0.15,
            )
        ax.set_title(name)
        ax.set_xlabel(f"Time — {summary['window']}")
        ax.set_ylabel(summary["units"])
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    health = summary["reference_health"]
    note = "Reference health: " + (
        "recorded thresholds met" if health["healthy"] is True else "; ".join(health["reasons"])
    )
    return _finish(
        fig, "Saved conditional/reference comparison — equal-tailed 80% intervals", note=note
    )


def plot_reference_health(summary, *, window="query"):
    rows = [
        row
        for row in summary["records"]
        if row["window"] == window and row["reference_health"] is not None
    ]
    if not rows and summary.get("reference_fit_summary"):
        diagnostics = summary["reference_fit_summary"]
        world_count = summary["source_summary"]["population"]["worlds_per_case"]
        rows = [
            {
                "case": case,
                "world": "all",
                "method": method,
                "reference_health": {
                    "healthy": count == world_count,
                    "reasons": [
                        f"{count}/{world_count} fits meet thresholds; "
                        f"maximum R-hat {detail['max_rhat_by_case'][case]['max']:.3f}; "
                        f"{detail['divergences_by_case'][case]['n_worlds_any']}"
                        " worlds with divergences"
                    ],
                },
            }
            for method, detail in diagnostics.items()
            for case, count in detail.get("healthy_worlds_by_case", {}).items()
        ]
    if not rows:
        raise ValueError("No saved reference fit-health evidence")
    labels = [f"{row['case']} / {row['world']} / {row['method']}" for row in rows]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(11, max(3, len(rows) * 0.35)))
    for index, row in enumerate(rows):
        health = row["reference_health"]
        ax.text(
            0,
            index,
            "Thresholds met" if health["healthy"] is True else "; ".join(health["reasons"]),
            color="C2" if health["healthy"] is True else "C3",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(rows)), labels=labels)
    ax.set_xlim(-0.02, 1)
    ax.set_xticks([])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    return _finish(
        fig,
        f"Reference fit health — {window}",
        note=(
            "Unhealthy/incomplete fits are retained in every comparison, not "
            "silently dropped or rerun."
        ),
    )


def render_comparisons(summary, out_dir):
    """Save every applicable comparison panel; missing evidence is an explicit status."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    windows = list(
        dict.fromkeys(row["window"] for row in (*summary["records"], *summary["aggregate"]))
    )
    panels = []
    functions = (
        ("performance", plot_performance, "crps"),
        ("coverage", plot_coverage, "coverage"),
        ("recovery", plot_recovery, "recovered"),
        ("failure_size", plot_failure_size, "crps"),
        ("false_effects", plot_false_effects, "false_effect_rate"),
        ("five_controls", plot_five_controls, "five_control_false_exclusions"),
        ("error_decomposition", plot_error_decomposition, "total_signed_error"),
        ("component_hit_rate", plot_component_hit_rate, "component_hit_rate"),
        ("control_count_recovery", plot_control_count_recovery, "five_control_modal_count_hit"),
        ("inactive_attribution", plot_inactive_attribution, "inactive_mean_attribution"),
        ("false_effect_worlds", plot_false_effect_worlds, "worlds_with_false_effect"),
        ("absorbed_share", plot_absorbed_share, "absorbed_share"),
        ("structure_recovery", plot_structure_recovery, "structure_exact_hit"),
        (
            "channel_graph_count",
            lambda table, **kw: plot_graph_count_recovery(table, family="cy", **kw),
            "family/cy/modal_count_hit",
        ),
        (
            "control_graph_count",
            lambda table, **kw: plot_graph_count_recovery(table, family="zb", **kw),
            "family/zb/modal_count_hit",
        ),
    )

    def save(name, make_figure):
        path = root / f"{name}.png"
        figure = None
        try:
            figure = make_figure()
            figure.savefig(path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            panels.append(
                {"name": name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            )
        else:
            panels.append({"name": name, "status": "available", "path": path.name})
        finally:
            if figure is not None:
                _pyplot().close(figure)

    for window in windows:
        for name, function, metric in functions:
            family_filter = (
                ("channel", "control", "total")
                if name in ("performance", "coverage", "failure_size")
                else ("baseline", "channel", "control")
                if name == "error_decomposition"
                else None
            )
            if any(
                row["window"] == window
                and row["metric"] == metric
                and (family_filter is None or row["family"] in family_filter)
                for row in summary["aggregate"]
            ):
                save(f"{name}_{window}", lambda: function(summary, window=window))
            else:
                panels.append(
                    {
                        "name": f"{name}_{window}",
                        "status": "unavailable",
                        "reason": f"No saved {metric} values; required truth/support may be absent",
                    }
                )
        for name, function, rows in (
            (
                "method_comparison",
                plot_ab_comparison,
                [row for row in summary["paired"] if row.get("kind") == "method_comparison"],
            ),
            (
                "oracle_comparison",
                plot_oracle_comparison,
                [row for row in summary["paired"] if row.get("kind") == "oracle_comparison"],
            ),
            (
                "graph_averaging",
                plot_graph_averaging,
                [row for row in summary["paired"] if row.get("kind") == "graph_averaging"],
            ),
            ("matched_omission", plot_matched_omission, summary["omission_summary"]),
        ):
            if any(row["window"] == window for row in rows):
                save(f"{name}_{window}", lambda: function(summary, window=window))
            else:
                panels.append(
                    {
                        "name": f"{name}_{window}",
                        "status": "unavailable",
                        "reason": "Required verified paired evidence not supplied",
                    }
                )
        if summary.get("reference_fit_summary") or any(
            row["window"] == window and row["reference_health"] is not None
            for row in summary["records"]
        ):
            save(
                f"reference_health_{window}", lambda: plot_reference_health(summary, window=window)
            )
    keys = list(
        dict.fromkeys((row["case"], row["world"], row["window"]) for row in summary["records"])
    )
    for index, (case, world, window) in enumerate(keys):
        for projection in ("trajectory", "window_mean"):
            save(
                f"world_{index:04d}_{projection}",
                lambda: plot_world_comparison(
                    summary, case=case, world=world, window=window, projection=projection
                ),
            )
    if not keys:
        panels.append(
            {
                "name": "world_trajectories",
                "status": "unavailable",
                "reason": "Derived aggregate evidence contains no full per-world trajectories",
            }
        )
    return panels


def plot_posterior_sbc(summary):
    """Render saved posterior-SBC ranks, ECDFs and coverage without new sampling."""
    names = list(summary["ranks"])
    if not names:
        raise ValueError("No saved posterior-SBC quantities")
    plt = _pyplot()
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 3.2 * len(names) + 1), squeeze=False)
    bins = np.arange(summary["n_post_samples"] + 2) - 0.5
    for row, name in zip(axes, names, strict=True):
        row[0].hist(summary["ranks"][name], bins=bins, color="C0", alpha=0.8)
        row[0].axhline(
            summary["n_replications"] / (summary["n_post_samples"] + 1),
            color="black",
            linestyle="--",
        )
        row[0].set(xlabel="Inclusive randomized rank", ylabel=f"{name}\nReplications")
        ecdf = summary["ecdf"][name]
        row[1].step(ecdf["x"], ecdf["y"], where="post", label="Saved randomized PIT")
        row[1].plot([0, 1], [0, 1], "--", color="black")
        row[1].set(xlabel="PIT", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1))
        levels = sorted(summary["coverage"][name], key=float)
        row[2].plot(
            [float(level) for level in levels],
            [summary["coverage"][name][level] for level in levels],
            "o-",
            label="Rank-central coverage",
        )
        row[2].plot([0, 1], [0, 1], "--", color="black")
        row[2].set(xlabel="Nominal mass", ylabel="Observed coverage", xlim=(0, 1), ylim=(0, 1))
    return _finish(
        fig, "Posterior-SBC — saved conditional-updating experiment", note=summary["qualification"]
    )


def plot_c2st(summary):
    """Render frozen world-held-out classifier predictions and paired statistics."""
    labels = np.asarray(summary["labels"])
    probabilities = np.asarray(summary["probabilities"])
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, name in ((0, "Reference"), (1, "Conditional model")):
        axes[0].hist(
            probabilities[labels == label], bins=np.linspace(0, 1, 11), alpha=0.5, label=name
        )
    axes[0].set(xlabel="Predicted conditional-model probability", ylabel="Held-out rows")
    axes[1].bar(["Accuracy"], [summary["accuracy"]], color="C0")
    axes[1].axhline(0.5, color="black", linestyle="--", label="Chance accuracy")
    axes[1].set(ylim=(0, 1), ylabel="Held-out accuracy")
    axes[1].text(
        0.05,
        0.95,
        f"Paired-swap p = {summary['p_value']:.3g}\n{summary['n_test_worlds']} test worlds",
        transform=axes[1].transAxes,
        va="top",
    )
    axes[2].bar(
        ["Cross entropy", "Descriptive\nlog(2) − loss"],
        [summary["cross_entropy"], summary["divergence_lb"]],
        color=["C0", "C1"],
    )
    axes[2].axhline(np.log(2), color="black", linestyle="--", label="Chance log loss")
    axes[2].set_ylabel("Nats")
    return _finish(fig, f"Saved C2ST — {summary['classifier']}", note=summary["qualification"])


def plot_input_support(summary):
    """Display raw context/query location and spread without asserting OOD status."""
    rows = summary["inputs"]
    if not rows:
        raise ValueError("No raw input support descriptors")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.4 * len(rows) + 2)), squeeze=False)
    for axis, statistic in zip(axes[0], ("mean", "std"), strict=True):
        for index, row in enumerate(rows):
            for j, window in enumerate(("context", "query")):
                if row[window] is not None:
                    axis.plot(
                        row[window][statistic],
                        index + (j - 0.5) * 0.2,
                        "o",
                        color=f"C{j}",
                        label=window,
                    )
        axis.set_yticks(range(len(rows)), labels=[row["name"] for row in rows])
        axis.set_xlabel(f"Raw input {statistic}")
        axis.invert_yaxis()
    return _finish(fig, "Raw input support — location and scale", note=summary["qualification"])


def plot_corpus_support(summary):
    """Plot the original mean/std world cloud and both marginal distributions."""
    worlds = summary["worlds"]
    if not worlds:
        raise ValueError("No saved world-level support descriptors")
    means = [world["mean"] for world in worlds]
    stds = [world["std"] for world in worlds]
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(means, stds, alpha=0.65)
    axes[0].set(xlabel="Context outcome mean", ylabel="Context outcome standard deviation")
    bins = min(30, max(1, int(np.sqrt(len(worlds)))))
    axes[1].hist(means, bins=bins, color="C0", alpha=0.8)
    axes[1].set(xlabel="Context outcome mean", ylabel="Worlds")
    axes[2].hist(stds, bins=bins, color="C1", alpha=0.8)
    axes[2].set(xlabel="Context outcome standard deviation", ylabel="Worlds")
    return _finish(
        fig, "Saved corpus support — world-level location and scale", note=summary["qualification"]
    )
