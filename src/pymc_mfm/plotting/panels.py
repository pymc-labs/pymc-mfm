"""Readable headless figures from computed, persisted diagnostic summaries.

Every function returns a matplotlib Figure for notebooks or ``savefig``.
Matplotlib is imported only when a figure is requested; pyplot/backend globals
are never changed, and no function imports or calls an inference model.
"""

from __future__ import annotations

import math

import numpy as np

COLORS = {
    "baseline": "#666666",
    "channel": "#0072B2",
    "control": "#009E73",
    "latent": "#D55E00",
    "indirect": "#CC79A7",
    "pooled": "#333333",
}


def _figure(rows=1, columns=1, *, width=11, row_height=3.2):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, row_height * rows), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(rows, columns, squeeze=False))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    return fig, axes


def _header(fig, summary, title):
    metadata = summary["metadata"]
    fig.suptitle(
        f"{title}\nConditional · {metadata['window']} · {metadata['mode']} · "
        f"{metadata['n_samples']} paired draws",
        fontsize=12,
    )


def _boundary(ax, metadata):
    time = np.asarray(metadata["time"])
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
    _band(ax, metadata["time"], data, COLORS["channel"], label="Latent total posterior mean")
    if data["observed"] is not None:
        ax.plot(metadata["time"], data["observed"], ".", color="0.3", label="Noisy observations")
        residual.plot(
            metadata["time"],
            data["observed_minus_latent_mean"],
            color=COLORS["channel"],
            label="Observed − posterior-mean latent total",
        )
    if data["realized_noise"] is not None:
        residual.plot(
            metadata["time"],
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


def plot_channel_totals(summary):
    entries = _components(summary, ("channel",))
    if not entries:
        raise ValueError("no active channels")
    fig, axes = _figure(width=max(8, len(entries) * 0.8), row_height=4)
    ax = axes[0, 0]
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
    _header(fig, summary, "Channel attribution — aggregate each paired trajectory before intervals")
    return fig


def plot_recovery_scatter(summary):
    fig, axes = _figure(width=7, row_height=6)
    ax = axes[0, 0]
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
    _header(fig, summary, "Component recovery — active slots only")
    return fig


def plot_structure_heatmap(summary):
    edges = summary["arrays"]["graphs"]["edges"]
    if not edges:
        raise ValueError("no active edges")
    chunk_size = 24
    chunks = [edges[index : index + chunk_size] for index in range(0, len(edges), chunk_size)]
    fig, axes = _figure(len(chunks), 1, row_height=3.5)
    for ax, chunk in zip(axes[:, 0], chunks):
        rows = [[entry["probability"] for entry in chunk]]
        names = [
            "Paired router frequency"
            if summary["metadata"]["mode"] == "model_average"
            else "Selected hypothesis indicators"
        ]
        if all(entry["truth"] is not None for entry in chunk):
            rows.append([entry["truth"] for entry in chunk])
            names.append("Graph truth")
        image = ax.imshow(rows, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(rows)), names)
        ax.set_xticks(
            range(len(chunk)),
            [entry["name"] for entry in chunk],
            rotation=70,
            ha="right",
            fontsize=7,
        )
        fig.colorbar(image, ax=ax, fraction=0.03, label="Edge frequency (not auxiliary logits)")
    _header(fig, summary, summary["arrays"]["graphs"]["label"])
    return fig


def plot_dag(summary):
    edges = summary["arrays"]["graphs"]["edges"]
    if not edges or any(entry["endpoints"] is None for entry in edges):
        raise ValueError("DAG view requires aligned edge endpoints")
    from matplotlib.patches import FancyArrowPatch

    nodes = sorted({node for entry in edges for node in entry["endpoints"]})
    banks = {}
    for node in nodes:
        bank = node.split(":", 1)[0]
        banks.setdefault(bank, []).append(node)
    order = sorted(
        banks,
        key=lambda bank: (
            {"latent": 0, "control": 1, "channel": 2, "outcome": 3, "sales": 4}.get(bank, 5),
            bank,
        ),
    )
    positions = {}
    for column, bank in enumerate(order):
        ys = np.linspace(0.1, 0.9, len(banks[bank])) if len(banks[bank]) > 1 else [0.5]
        for node, y in zip(banks[bank], ys):
            positions[node] = (column, y)
    has_truth = all(entry["truth"] is not None for entry in edges)
    fig, axes = _figure(1, 2 if has_truth else 1, width=15 if has_truth else 9, row_height=5.5)
    for column, ax in enumerate(axes[0]):
        true_panel = has_truth and column == 0
        for entry in edges:
            value = entry["truth"] if true_panel else entry["probability"]
            if not value:
                continue
            source, target = entry["endpoints"]
            arrow = FancyArrowPatch(
                positions[source],
                positions[target],
                arrowstyle="-|>",
                mutation_scale=11,
                shrinkA=14,
                shrinkB=14,
                linewidth=0.6 + 2 * value,
                alpha=0.3 + 0.7 * value,
                color="#0072B2",
                connectionstyle="arc3,rad=0.12",
            )
            ax.add_patch(arrow)
        for node, (x, y) in positions.items():
            ax.scatter(x, y, color="white", edgecolor="0.3", s=240, zorder=3)
            ax.text(x, y + 0.045, node, ha="center", va="bottom", fontsize=8)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_axis_off()
        ax.set_title(
            "True graph" if true_panel else "Marginal routing view (may contain cycles)",
            fontsize=10,
        )
    _header(
        fig, summary, "Directed graph views — native renderer; width / opacity = edge frequency"
    )
    return fig


def plot_graph_posterior(summary):
    graph = summary["arrays"]["graphs"]
    records = sorted(graph["complete_graphs"], key=lambda item: (-item["count"], item["graph"]))
    selected = records[:15]
    probabilities = [entry["frequency"] for entry in selected]
    labels = ["".join(map(str, entry["graph"])) or "empty graph" for entry in selected]
    labels = [
        label if len(label) <= 28 else f"Graph {index + 1} (full vector in evidence)"
        for index, label in enumerate(labels)
    ]
    colors = ["#D55E00" if entry.get("is_truth", False) else "#0072B2" for entry in selected]
    if len(records) > len(selected):
        labels.append(f"Other {len(records) - len(selected)} graphs")
        probabilities.append(sum(entry["frequency"] for entry in records[len(selected) :]))
        colors.append("#999999")
    fig, axes = _figure(width=11, row_height=max(4, 0.32 * len(labels)))
    ax = axes[0, 0]
    ax.barh(range(len(labels)), probabilities, color=colors)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Empirical complete-graph frequency; orange = true graph when supplied")
    ax.set_xlim(0, 1)
    _header(fig, summary, graph["label"])
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


def plot_zero_atoms(summary):
    entries = _components(summary, tuple(COLORS))
    fig, axes = _figure(width=11, row_height=max(4, len(entries) * 0.3))
    ax = axes[0, 0]
    ax.barh(
        range(len(entries)),
        [entry["zero_atom_probability"] for entry in entries],
        color=[COLORS[entry["family"]] for entry in entries],
    )
    ax.set_yticks(range(len(entries)), [entry["name"] for entry in entries], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Fraction of draws with an exactly zero full trajectory")
    _header(
        fig,
        summary,
        (
            "Zero atoms — not inferred causal absence; graph false-exclusion "
            "metrics reported separately"
        ),
    )
    return fig


def plot_uncertainty(summary):
    uncertainty = summary["arrays"]["total_uncertainty"]
    time = summary["metadata"]["time"]
    within = np.asarray(uncertainty["within_graph_variance"])
    between = np.asarray(uncertainty["between_graph_variance"])
    fig, axes = _figure(width=11, row_height=4)
    ax = axes[0, 0]
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
    ax.set_title(
        f"{uncertainty['n_unique_graphs']} graphs; {
            uncertainty[('n_singleton_groups')]
        } singleton groups; "
        f"rare draw mass {uncertainty['rare_draw_mass']:.1%}",
        fontsize=10,
    )
    _header(
        fig, summary, "Finite paired-draw uncertainty — singleton zero variance is not certainty"
    )
    return fig


def plot_calibration(summary, *, kind="coverage"):
    groups = summary["arrays"]["calibration"]
    if not groups:
        raise ValueError("calibration requires simulation truth")
    if kind == "coverage":
        fig, axes = _figure(width=8, row_height=5)
        ax = axes[0, 0]
        for name, record in groups.items():
            ax.plot(
                [row["level"] for row in record["intervals"]],
                [row["coverage"] for row in record["intervals"]],
                "o-",
                color=COLORS[name],
                label=name,
            )
        ax.plot([0, 1], [0, 1], "k--", label="Nominal")
        ax.set(
            xlabel="Central interval level",
            ylabel="Observed coordinate coverage",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
    else:
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
    _header(fig, summary, "Dependent one-world screening — not independent-world SBC; no pass/fail")
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


def plot_records(summary, *, kind="training", metric_names=None):
    """Plot saved scalar trajectories, optionally selecting exact nested names.

    By default evaluation shows a labeled overview. ``metric_names`` makes any
    recorded non-headline objective, auxiliary-head readout, uncertainty or
    calibration series plottable without recomputation.
    """
    records = summary["arrays"][kind]
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
        suffixes = (
            "decomposition_energy_score",
            "closure_rel_err",
            "contribution_crps",
            "control_contribution_crps",
            "cy/brier",
            "cy/log_score",
            "shd_mean",
            "exact_recovery",
        )
        selected = {name: values for name, values in series.items() if name.endswith(suffixes)}
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
    fig, axes = _figure(len(pages), 1, width=12, row_height=3.4)
    for ax, (group, rows) in zip(axes[:, 0], pages):
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
    _header(fig, summary, title)
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


def plot_panel(name, summary):
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
        "structure_heatmap": plot_structure_heatmap,
        "dag_recovery": plot_dag,
        "dag_posterior": plot_graph_posterior,
        "graph_cardinality": plot_cardinality,
        "support": plot_support,
        "zero_atoms": plot_zero_atoms,
        "graph_uncertainty": plot_uncertainty,
        "calibration_pit": lambda data: plot_calibration(data, kind="ranks"),
        "calibration_ecdf": lambda data: plot_calibration(data, kind="ecdf"),
        "coverage": plot_calibration,
        "training": plot_records,
        "evaluation": lambda data: plot_records(data, kind="evaluation"),
        "set_posterior": plot_set_posterior,
    }
    if name == "reference_comparison":
        from .comparisons import plot_reference_comparison

        return plot_reference_comparison(summary["arrays"]["reference"])
    if name in ("posterior_sbc", "c2st"):
        from .comparisons import plot_c2st, plot_posterior_sbc

        renderer = plot_posterior_sbc if name == "posterior_sbc" else plot_c2st
        return renderer(summary["arrays"]["reference"][name])
    if name not in dispatch:
        raise KeyError(f"unknown diagnostic panel {name!r}")
    return dispatch[name](summary)


def plot_sbc_report(report, *, kind="ecdf"):
    """Independent-world SBC rank/ECDF/coverage figures from a saved report card.

    ECDF bands are the stored Monte Carlo-calibrated discrete-null bands.
    Coverage is rank-implied, not sample-quantile coverage, and each plotted
    quantity contributes exactly one rank per declared independent world.
    """
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
    fig.suptitle(
        "Independent-world SBC (declared sampling contract)\n"
        "A non-rejection is not proof of calibration or OOD validity",
        fontsize=12,
    )
    return fig
