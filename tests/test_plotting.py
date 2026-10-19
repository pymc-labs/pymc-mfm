"""Plot values and labels share the persisted diagnostic definitions."""

import itertools
import json
import warnings
from dataclasses import replace

import numpy as np
import pytest
from test_diagnostics import make_result

from pymc_mfm.diagnostics import compute_diagnostics, record_series
from pymc_mfm.inference.result import result_from_batch
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.plotting import (
    plot_attribution,
    plot_calibration,
    plot_channel_totals,
    plot_component_trajectories,
    plot_cumulative,
    plot_cycles,
    plot_graph_posterior,
    plot_graph_statistics,
    plot_panel,
    plot_records,
    plot_recovery_scatter,
    plot_reliability,
    plot_structure_heatmap,
    plot_target_recovery,
    plot_temporal,
    plot_uncertainty,
    plot_window_totals,
    plot_zero_atoms,
)
from pymc_mfm.training.evaluation import GRAPH_METRICS

pytest.importorskip("matplotlib")


def layout_result(graphs=None, *, draws=6, seed=0, weeks=3, active=None):
    """Two live channels (plus a padded one), two controls and an inactive latent.

    Labeled by ``result_from_batch``, so edge names, endpoints and families
    follow the model's slot layout; the latent families have no active edge.
    ``active`` replaces the channel, control and latent masks.
    """
    if active is None:
        active = {"c": np.array([1, 1, 0]), "m": np.array([1, 1]), "j": np.array([0])}
    layout = EdgeSlotLayout(3, 2, 1)
    slots = layout.active_slot_mask(active["c"], active["m"], active["j"]).astype(bool)
    rng = np.random.default_rng(seed)
    graphs = rng.integers(0, 2, (draws, layout.n_slots)) * slots if graphs is None else graphs
    graphs = np.asarray(graphs)
    component_mask = np.r_[True, active["c"], active["m"], active["j"].any(), True, True, True]
    components = rng.normal(size=(len(graphs), 1, weeks, component_mask.size)) * component_mask
    batch = {
        **{f"active_{key}_mask": value[None].astype(float) for key, value in active.items()},
        "ctx_spend_z": np.zeros((1, 4, 3)),
        "qry_spend_z": np.zeros((1, weeks, 3)),
        "sales_scale": np.array([2.0]),
    }
    return result_from_batch(
        batch, components, graphs[:, None], model_config=ModelConfig(n_controls=2, n_demand=1)
    )


def has_directed_cycle(pairs):
    successors = {}
    for source, target in pairs:
        successors.setdefault(source, []).append(target)
    state = {}

    def reaches_open_node(node):
        state[node] = "open"
        for target in successors.get(node, ()):
            if state.get(target) == "open" or (target not in state and reaches_open_node(target)):
                return True
        state[node] = "done"
        return False

    return any(node not in state and reaches_open_node(node) for node in list(successors))


def test_component_lines_bands_units_and_padding_match_stored_arrays():
    result = make_result()
    summary = compute_diagnostics(result, truth=result.components.mean(0))
    figure = plot_component_trajectories(summary, families=("channel",))
    ax = figure.axes[0]
    lines = {line.get_label(): line for line in ax.lines}
    expected = result.components[..., 1]
    np.testing.assert_allclose(lines["Posterior mean"].get_ydata(), expected.mean(axis=0))
    np.testing.assert_array_equal(lines["Posterior mean"].get_xdata(), result.time)
    np.testing.assert_allclose(lines["Truth"].get_ydata(), expected.mean(axis=0))
    band = ax.collections[0].get_paths()[0].vertices
    low, high = np.quantile(expected, [0.1, 0.9], axis=0)
    for time, lower, upper in zip(result.time, low, high):
        heights = band[band[:, 0] == time, 1]
        assert min(heights) == pytest.approx(lower)
        assert max(heights) == pytest.approx(upper)
    assert ax.get_title() == "channel:paid search"
    assert ax.get_ylabel() == "revenue"
    assert "padding" not in " ".join(axis.get_title() for axis in figure.axes)
    assert lines["query begins"].get_xdata()[0] == pytest.approx(3.5)


def test_target_reconstruction_distinguishes_noisy_observations_and_latent_truth():
    result = make_result()
    truth = result.components.mean(0)
    observed = truth.sum(-1) + [1, 2, -3]
    summary = compute_diagnostics(result, truth=truth, observed=observed)
    figure = plot_target_recovery(summary)
    main, residual = figure.axes
    lines = {line.get_label(): line for line in main.lines}
    np.testing.assert_allclose(lines["Latent total posterior mean"].get_ydata(), truth.sum(-1))
    np.testing.assert_allclose(lines["Noisy observations"].get_ydata(), observed)
    residual_lines = {line.get_label(): line for line in residual.lines}
    np.testing.assert_allclose(
        residual_lines["Realized observation noise (observed − latent truth)"].get_ydata(),
        [1, 2, -3],
    )
    assert main.get_ylabel() == "revenue"


def test_time_aggregate_intervals_aggregate_draws_not_weekly_quantiles():
    result = make_result()
    draws = result.components.copy()
    draws[..., 1] = [[0, 0, 9], [0, 9, 0], [9, 0, 0], [0, 0, 9], [0, 9, 0]]
    result = replace(result, components=draws)
    summary = compute_diagnostics(result)
    figure = plot_channel_totals(summary)
    intervals = figure.axes[0].collections[0].get_segments()[0]
    np.testing.assert_allclose(intervals[:, 1], [3, 3])
    assert not np.isclose(np.quantile(draws[..., 1], 0.1, axis=0).mean(), 3)
    assert figure.axes[0].get_xticklabels()[0].get_text() == "channel:paid search"


def test_empirical_graph_plot_preserves_complete_graph_mass_and_truth_marker():
    result = make_result()
    summary = compute_diagnostics(result, graph_truth=[1, 0, 1])
    figure = plot_graph_posterior(summary)
    bars = figure.axes[0].patches
    np.testing.assert_allclose([bar.get_width() for bar in bars], [0.4, 0.4, 0.2])
    assert sum(bar.get_width() for bar in bars) == pytest.approx(1)
    assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == ["00", "11", "10"]
    assert bars[0].get_facecolor() != bars[1].get_facecolor()


def heatmap_axes(figure):
    """Heatmap axes; the shared colorbar axes holds no image."""
    return [ax for ax in figure.axes if ax.images]


def test_fixed_hypothesis_heatmap_draws_only_active_edges_per_family():
    result = make_result()
    result = replace(
        result,
        mode="fixed",
        graphs=np.tile([1, 0, 1], (5, 1)),
        provenance={**result.provenance, "graph_source": "supplied_hypothesis"},
    )
    summary = compute_diagnostics(result, graph_truth=[1, 0, 0])
    strips = heatmap_axes(plot_structure_heatmap(summary))
    # cy and zb hold one active edge each: indicator row, then graph truth.
    assert [ax.images[0].get_array().tolist() for ax in strips] == [[[1], [1]], [[1], [0]]]
    labels = [label.get_text() for ax in strips for label in ax.get_xticklabels()]
    assert all("padding_slot" not in label for label in labels)


def test_heatmap_groups_one_family_per_axes_with_node_matrices():
    result = layout_result()
    truth = result.graphs[0]
    summary = compute_diagnostics(result, graph_truth=truth)
    frequency = result.graphs.mean(axis=0)
    families = result.provenance["edge_families"]
    active = np.flatnonzero(result.edge_mask)
    endpoints = {tuple(result.provenance["edge_endpoints"][i]): i for i in active}
    by_label = {result.edge_names[i].split(":", 1)[1]: i for i in active}
    drawn, groups, matrices = [], [], set()
    for ax in heatmap_axes(plot_structure_heatmap(summary)):
        values = ax.images[0].get_array()
        rows = [label.get_text() for label in ax.get_yticklabels()]
        columns = [label.get_text() for label in ax.get_xticklabels()]
        if "->" in columns[0]:
            edges = [by_label[label] for label in columns]
            np.testing.assert_allclose(values[0], frequency[edges])
            np.testing.assert_array_equal(values[1], truth[edges])
        else:
            missing = np.ma.getmaskarray(values)
            cells = {
                (row, column): endpoints[rows[row], columns[column]]
                for row, column in zip(*np.nonzero(~missing))
            }
            edges = list(cells.values())
            for (row, column), edge in cells.items():
                assert values[row, column] == pytest.approx(frequency[edge])
            assert all(rows[row] == columns[column] for row, column in zip(*np.nonzero(missing)))
            offsets = ax.collections[0].get_offsets()
            marked = {cells[int(row), int(column)] for column, row in offsets}
            assert marked == {edge for edge in edges if truth[edge]}
            matrices.add(families[edges[0]])
        assert len({families[edge] for edge in edges}) == 1
        groups.append(families[edges[0]])
        drawn.extend(edges)
    assert sorted(drawn) == active.tolist()
    # Latent families touch only the inactive latent node, so they are not drawn.
    assert sorted(groups) == ["cc", "cy", "zb", "zc", "zz"]
    assert matrices == {"cc", "zc", "zz"}


def test_a_heatmap_of_one_edge_strip_keeps_its_constrained_layout():
    # One active channel and no active control or latent: only the cy strip.
    active = {"c": np.array([1, 0, 0]), "m": np.array([0, 0]), "j": np.array([0])}
    figure = plot_structure_heatmap(compute_diagnostics(layout_result(active=active)))
    (strip,) = heatmap_axes(figure)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.canvas.draw()
    assert not [str(item.message) for item in caught if "constrained_layout" in str(item.message)]
    renderer = figure.canvas.get_renderer()
    header = figure._suptitle.get_window_extent(renderer)
    assert header.y0 >= strip.title.get_window_extent(renderer).y1


def test_scatter_and_uncertainty_share_numeric_evidence():
    result = make_result()
    truth = result.components.mean(0) + np.array([1, 2, 0, 3, 4, 1, 1, 1])
    summary = compute_diagnostics(result, truth=truth)
    scatter = plot_recovery_scatter(summary)
    offsets = scatter.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], truth[:, 0])
    np.testing.assert_allclose(offsets[:, 1], result.components[..., 0].mean(0))
    assert len(scatter.axes[0].collections) == int(result.component_mask.sum())
    uncertainty = plot_uncertainty(summary)
    line = next(
        line for line in uncertainty.axes[0].lines if line.get_label() == "Total empirical variance"
    )
    np.testing.assert_allclose(
        line.get_ydata(), result.components[..., result.component_mask].sum(-1).var(0)
    )


def test_calibration_screening_coverage_coordinates_are_stored_values():
    result = make_result()
    summary = compute_diagnostics(result, truth=result.components.mean(0))
    figure = plot_calibration(summary)
    line = next(line for line in figure.axes[0].lines if line.get_label() == "channel")
    expected = summary["arrays"]["calibration"]["channel"]["intervals"]
    np.testing.assert_array_equal(line.get_xdata(), [row["level"] for row in expected])
    np.testing.assert_array_equal(line.get_ydata(), [row["coverage"] for row in expected])
    with pytest.raises(ValueError, match="truth"):
        plot_calibration(compute_diagnostics(result))
    # "pit" names the rank figure only in ``diag.plot.calibration``; the renderer refuses it.
    with pytest.raises(ValueError):
        plot_calibration(summary, kind="pit")


def test_arbitrary_saved_auxiliary_trajectories_preserve_undefined_gaps():
    name = "heads/auxiliary/zb/set_ece"
    records = record_series(
        [
            {"step": 1, "heads": {"auxiliary": {"zb": {"set_ece": 0.2}}}},
            {"step": 2, "heads": {"auxiliary": {"zb": {"set_ece": None}}}},
            {"step": 3, "heads": {"auxiliary": {"zb": {"set_ece": 0.1}}}},
        ],
        "evaluation",
    )
    figure = plot_records(records, kind="evaluation", metric_names=[name], header="run r7")
    line = figure.axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), [0.2, np.nan, 0.1], equal_nan=True)
    assert line.get_label() == name
    assert figure._suptitle.get_text().endswith("run r7")
    with pytest.raises(KeyError, match="unknown"):
        plot_records(records, kind="evaluation", metric_names=["absent"])


def test_default_evaluation_overview_selects_whole_headline_metric_names():
    flat = {
        "holdout/joint_router/graph/shd_mean": 1.5,
        "holdout/joint_router/graph/normalized_shd_mean": 0.01,
        "holdout/structure_head/cy/brier": 0.1,
        "holdout/structure_head/zb/brier": 0.2,
    }
    records = record_series([{"step": 2, "flat": flat}], "evaluation")
    figure = plot_records(records, kind="evaluation")
    plotted = {line.get_label() for ax in figure.axes for line in ax.lines}
    assert plotted == {
        "flat/holdout/joint_router/graph/shd_mean",
        "flat/holdout/structure_head/cy/brier",
    }


def test_unknown_panel_is_not_silently_skipped():
    with pytest.raises(KeyError, match="unknown diagnostic panel"):
        plot_panel("missing_panel", compute_diagnostics(make_result()))


C0, C1 = "channel:channel_0", "channel:channel_1"
Z0, Z1 = "control:control_0", "control:control_1"
#: Graph truth of the planted draws: channel 0 and control 1 are live.
TRUE_EDGES = (f"cy:{C0}->outcome", f"zb:{Z1}->outcome")
#: Whole-graph scores that compare the draws with graph truth.
TRUTH_SCORES = (
    "shd_mean",
    "exact_recovery",
    "shd_median",
    "normalized_shd_mean",
    "energy_score",
    "categorical_brier_score",
)


def edge_indicator(edges):
    """Indicator over ``layout_result``'s edges of exactly the named ``edges``."""
    names = layout_result().edge_names
    indicator = np.zeros(len(names), dtype=int)
    indicator[[names.index(edge) for edge in edges]] = 1
    return indicator


def planted_result(*draws):
    """``layout_result`` whose draw ``s`` holds exactly the edges named in ``draws[s]``."""
    return layout_result(np.array([edge_indicator(edges) for edges in draws]))


def test_graph_statistics_draw_the_tables_whole_graph_scores_with_and_without_truth():
    # Draw 0 adds a cc two-cycle and drops the live control, draw 1 adds one cc
    # and one zz edge (no cycle), draws 2 and 3 are the true graph.
    result = planted_result(
        (f"cy:{C0}->outcome", f"cc:{C0}->{C1}", f"cc:{C1}->{C0}"),
        (*TRUE_EDGES, f"cc:{C0}->{C1}", f"zz:{Z0}->{Z1}"),
        TRUE_EDGES,
        TRUE_EDGES,
    )
    truth = edge_indicator(TRUE_EDGES)
    draws, target = result.graphs[:, result.edge_mask], truth[result.edge_mask]
    assert draws.shape == (4, 12)  # the padded channel's and the latent's edges are not scored
    comoment = [
        abs(np.mean(draws[:, i] * draws[:, j]) - draws[:, i].mean() * draws[:, j].mean())
        for i, j in itertools.combinations(range(draws.shape[1]), 2)
    ]
    # Distances to graph truth: 3, 2, 0 and 0 of 12 edges.
    expected = {
        "shd_mean": 1.25,
        "exact_recovery": 0.5,
        "shd_median": 1.0,
        "normalized_shd_mean": 1.25 / 12,
        "energy_score": reference_energy(draws, target),
        # 1 - 2 P(draw = truth) + the collision probability of the draw counts 2, 1, 1.
        "categorical_brier_score": 1 - 2 * 0.5 + 2 / 12,
        "entropy": 1.5 * np.log(2),
        "n_distinct": 3,
        "pairwise_comoment_abs_err": np.mean(comoment),
        "pairwise_comoment_max_abs_err": np.max(comoment),
        # Draw 0's mutual cc pair is the only cycle; draw 1's single zz edge is none.
        "cc_two_cycle_rate": 0.25,
        "cc_cycle_rate": 0.25,
        "zz_two_cycle_rate": 0.0,
        "zz_cycle_rate": 0.0,
    }
    assert set(expected) == set(GRAPH_METRICS)
    summary = compute_diagnostics(result, graph_truth=truth)
    assert {key: summary["metrics"][f"graph/{key}"] for key in expected} == pytest.approx(expected)
    ranks, distance, scores = plot_graph_statistics(summary).axes
    np.testing.assert_allclose([bar.get_height() for bar in ranks.patches], [0.5, 0.25, 0.25])
    np.testing.assert_allclose(ranks.lines[0].get_ydata(), [0.5, 0.75, 1.0])
    np.testing.assert_allclose([bar.get_height() for bar in distance.patches], [0.5, 0, 0.25, 0.25])
    lines = {line.get_gid(): line.get_xdata()[0] for line in distance.lines if line.get_gid()}
    assert lines == pytest.approx({"graph/shd_mean": 1.25, "graph/shd_median": 1.0})
    bars = {bar.get_gid(): bar.get_width() for bar in scores.patches}
    assert bars == pytest.approx(
        {
            f"graph/{key}": expected[key]
            for key in (
                "normalized_shd_mean",
                "energy_score",
                "categorical_brier_score",
                "pairwise_comoment_abs_err",
                "pairwise_comoment_max_abs_err",
            )
        }
    )
    drawn = {**lines, **bars}
    assert drawn == pytest.approx({name: summary["metrics"][name] for name in drawn})

    # Without graph truth the truth-free statistics keep their values; the others
    # are unavailable for the reason every graph-truth score states.
    unlabeled = compute_diagnostics(result)
    missing = unlabeled["statuses"]["metric/graph/cardinality_absolute_error"]
    for key, value in expected.items():
        if key in TRUTH_SCORES:
            assert unlabeled["metrics"][f"graph/{key}"] is None, key
            assert unlabeled["statuses"][f"metric/graph/{key}"] == missing, key
        else:
            assert unlabeled["metrics"][f"graph/{key}"] == pytest.approx(value), key
    scores = plot_graph_statistics(unlabeled).axes[-1]
    assert {bar.get_gid(): bar.get_width() for bar in scores.patches} == pytest.approx(
        {
            "graph/pairwise_comoment_abs_err": expected["pairwise_comoment_abs_err"],
            "graph/pairwise_comoment_max_abs_err": expected["pairwise_comoment_max_abs_err"],
        }
    )


def test_cycle_panel_draws_the_complete_graph_and_per_family_cycle_rates():
    result = planted_result(
        (f"cc:{C0}->{C1}", f"cc:{C1}->{C0}"),
        (f"zz:{Z0}->{Z1}", f"zz:{Z1}->{Z0}"),
        (f"cc:{C0}->{C1}", f"zc:{Z0}->{C0}"),
        (),
        (*TRUE_EDGES, f"cc:{C1}->{C0}"),
        (*TRUE_EDGES, f"zz:{Z0}->{Z1}", f"zz:{Z1}->{Z0}"),
    )
    endpoints = result.provenance["edge_endpoints"]
    cyclic = [has_directed_cycle(endpoints[i] for i in np.flatnonzero(g)) for g in result.graphs]
    assert cyclic == [True, True, False, False, False, True]
    # The cc mutual pair is in one of six draws, the zz pair in two; the
    # complete graph counts both.
    expected = {
        "graph/cyclic_draw_fraction": np.mean(cyclic),
        "graph/cc_cycle_rate": 1 / 6,
        "graph/cc_two_cycle_rate": 1 / 6,
        "graph/zz_cycle_rate": 2 / 6,
        "graph/zz_two_cycle_rate": 2 / 6,
    }
    summary = compute_diagnostics(result)
    ax = plot_cycles(summary).axes[0]
    drawn = {bar.get_gid(): bar.get_width() for bar in ax.patches}
    assert drawn == pytest.approx(expected)
    assert {name: summary["metrics"][name] for name in expected} == pytest.approx(expected)
    # Each bar's tick label states that bar's rate and, for a family rate, its family.
    labels = [label.get_text() for label in ax.get_yticklabels()]
    for bar in ax.patches:
        label = labels[round(bar.get_y() + bar.get_height() / 2)]
        assert f"{bar.get_width():.1%}" in label, label
        family = bar.get_gid().removeprefix("graph/").split("_")[0]
        if family in ("cc", "zz"):
            assert label.startswith(f"{family}:"), label


def test_graph_statistics_are_unavailable_without_active_edges():
    # No active channel, control or latent leaves no active edge slot.
    active = {"c": np.zeros(3, dtype=int), "m": np.zeros(2, dtype=int), "j": np.zeros(1, dtype=int)}
    result = layout_result(np.zeros((4, EdgeSlotLayout(3, 2, 1).n_slots), dtype=int), active=active)
    summary = compute_diagnostics(result, graph_truth=np.zeros(result.graphs.shape[1]))
    assert summary["metrics"]["graph/shd_mean"] is None
    # Every whole-graph statistic is undefined, so the panel is unavailable like the heatmap.
    assert summary["panels"]["graph_statistics"]["status"] == "unavailable"
    assert summary["panels"]["graph_statistics"] == summary["panels"]["structure_heatmap"]
    with pytest.raises(ValueError, match="graph_statistics"):
        result.plot.graph_statistics()


def drawn_ticks(ax):
    """Tick positions inside the view limits: the ticks matplotlib draws."""
    low, high = sorted(ax.get_xlim())
    return [tick for tick in ax.get_xticks() if low <= tick <= high]


def test_one_graph_posterior_ticks_only_whole_non_negative_counts():
    truth = edge_indicator(TRUE_EDGES)
    result = layout_result(np.tile(truth, (4, 1)))  # every draw is the true graph
    for graph_truth in (truth, None):
        figure = plot_graph_statistics(compute_diagnostics(result, graph_truth=graph_truth))
        figure.canvas.draw()
        ranks, distance, _ = figure.axes
        # Graph ranks start at 1 and distances at 0; without truth no distance is drawn.
        counted = [(ranks, 1)] if graph_truth is None else [(ranks, 1), (distance, 0)]
        for ax, lowest in counted:
            ticks = drawn_ticks(ax)
            assert ticks, ax.get_xlabel()
            assert all(float(tick).is_integer() and tick >= lowest for tick in ticks), ticks


def test_latent_total_uncertainty_entries_summarize_the_drawn_series():
    result = make_result()
    summary = compute_diagnostics(result)
    ax = plot_uncertainty(summary).axes[0]
    latent = result.components[..., result.component_mask].sum(axis=-1)
    _, group = np.unique(result.graphs[:, result.edge_mask], axis=0, return_inverse=True)
    group = group.reshape(-1)
    weights = np.bincount(group) / len(group)
    within = sum(w * latent[group == g].var(axis=0) for g, w in enumerate(weights))
    between = sum(
        w * (latent[group == g].mean(axis=0) - latent.mean(axis=0)) ** 2
        for g, w in enumerate(weights)
    )
    for band, top in zip(ax.collections[:2], (within, within + between)):
        vertices = band.get_paths()[0].vertices
        for time, height in zip(result.time, top):
            assert vertices[vertices[:, 0] == time, 1].max() == pytest.approx(height)
    for name, series in (
        ("within_graph_variance", within),
        ("between_graph_variance", between),
        ("total_variance", latent.var(axis=0)),
    ):
        assert summary["metrics"][f"uncertainty/latent_total/{name}_mean"] == pytest.approx(
            series.mean()
        )


def test_summaries_without_a_recorded_head_are_labeled_conditional():
    summary = compute_diagnostics(make_result())
    del summary["metadata"]["head"]
    assert "conditional" in plot_panel("components", summary)._suptitle.get_text()


def reference_crps(draws, truth):
    """Mean CRPS ``E|X - y| - E|X - X'| / 2`` over trailing coordinates, from all draw pairs."""
    pairs = np.abs(draws[:, None] - draws[None, :]).mean(axis=(0, 1))
    return float(np.mean(np.abs(draws - truth).mean(axis=0) - pairs / 2))


def reference_energy(draws, truth):
    """Energy score of draws ``[S, ...]``: mean distance to truth minus half the pair mean."""
    flat, target = draws.reshape(len(draws), -1), truth.ravel()
    pairs = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).sum() / (len(flat) - 1)
    to_truth = np.linalg.norm(flat - target, axis=1).sum()
    return float((to_truth - pairs / 2) / len(flat) / np.sqrt(target.size))


def reference_correlation(left, right):
    """Pearson correlation, or ``None`` when either side is constant."""
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def mean_of_defined(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def reference_temporal(draws, truth):
    """Temporal scores of one slot's draws ``[S, T]`` against its truth ``[T]``."""

    def lag1(series):
        return reference_correlation(series[:-1], series[1:])

    centered = draws - draws.mean(axis=0)
    weeks = draws.shape[1]
    return {
        "truth_lag1": lag1(truth),
        "draw_lag1": mean_of_defined([lag1(draw) for draw in draws]),
        "truth_error_residual_lag1": mean_of_defined([lag1(draw - truth) for draw in draws]),
        "posterior_centered_residual_lag1": mean_of_defined([lag1(row) for row in centered]),
        "adjacent_draw_correlation": mean_of_defined(
            [reference_correlation(draws[:, t], draws[:, t + 1]) for t in range(weeks - 1)]
        ),
    }


LIVE_EFFECTS = {
    "channel:channel_0": True,
    "channel:channel_1": False,
    "control:control_0": False,
    "control:control_1": True,
}


def evidence_world(*, weeks=6, draws=8, seed=3):
    """A labeled result with random truth; channel 0 and control 1 are live, the rest dead."""
    result = layout_result(draws=draws, seed=seed, weeks=weeks)
    truth = np.random.default_rng(seed + 1).normal(size=(weeks, len(result.component_names)))
    truth *= result.component_mask
    graph_truth = np.zeros(len(result.edge_names), dtype=int)
    for name in ("cy:channel:channel_0->outcome", "zb:control:control_1->outcome"):
        graph_truth[result.edge_names.index(name)] = 1
    return result, truth, graph_truth


def tick_names(ax):
    return [label.get_text() for label in ax.get_xticklabels()]


def test_attribution_bars_are_each_effects_scores_and_lines_their_table_means():
    result, truth, graph_truth = evidence_world()
    summary = compute_diagnostics(result, truth=truth, graph_truth=graph_truth)
    scale, names = result.provenance["sales_scale"], list(result.component_names)
    expected = {}
    for name, live in LIVE_EFFECTS.items():
        draws = result.components[..., names.index(name)] / scale
        target = truth[:, names.index(name)] / scale
        error = draws.mean(axis=0) - target
        group = (
            f"{'live' if live else 'dead'}_{'media' if name.startswith('channel') else 'control'}"
        )
        scores = (
            {
                "crps": reference_crps(draws, target),
                "mean_bias": error.mean(),
                "mean_abs_error": abs(error.mean()),
                "demeaned_rmse": np.sqrt(np.mean((error - error.mean()) ** 2)),
            }
            if live
            else {
                "leakage_scaled_mae": np.abs(error).mean(),
                "draw_leakage_scaled_mae": np.abs(draws - target).mean(),
            }
        )
        for key, value in scores.items():
            expected.setdefault(f"attribution/{group}/{key}", {})[name] = value
    drawn, lines = {}, 0
    for ax in plot_attribution(summary).axes:
        for bar in ax.patches:
            position = round(bar.get_x() + bar.get_width() / 2)
            drawn.setdefault(bar.get_gid(), {})[tick_names(ax)[position]] = bar.get_height()
        for line in (line for line in ax.lines if line.get_gid()):
            value = np.mean(list(expected[line.get_gid()].values()))
            assert line.get_ydata()[0] == pytest.approx(value)
            assert summary["metrics"][line.get_gid()] == pytest.approx(value)
            lines += 1
    assert drawn.keys() == expected.keys()
    for metric, values in expected.items():
        assert drawn[metric] == pytest.approx(values), metric
    assert lines == len(expected)


def test_window_totals_draw_window_mean_intervals_coverage_rates_and_latent_fit():
    result, truth, graph_truth = evidence_world()
    names = list(result.component_names)
    # Live channel 0 sits at the median draw window mean, dead channel 1 far above all.
    window_means = result.components.mean(axis=1)
    live, dead = names.index("channel:channel_0"), names.index("channel:channel_1")
    truth[:, live] = np.median(window_means[:, live])
    truth[:, dead] = window_means[:, dead].max() + 10
    summary = compute_diagnostics(result, truth=truth, graph_truth=graph_truth)
    channels, rates, fit = plot_window_totals(summary).axes
    assert tick_names(channels) == ["channel:channel_0", "channel:channel_1"]
    intervals = {
        tick_names(channels)[round(x)]: (low, high)
        for collection in channels.collections
        if collection.get_gid() == "window_totals/interval"
        for (x, low), (_, high) in collection.get_segments()
    }
    covered = {}
    for name in tick_names(channels):
        low, high = np.quantile(result.components[..., names.index(name)].mean(axis=1), [0.1, 0.9])
        assert intervals[name] == pytest.approx((low, high))
        covered[name] = bool(low <= truth[:, names.index(name)].mean() <= high)
    assert covered == {"channel:channel_0": True, "channel:channel_1": False}
    marked = {}
    for collection in channels.collections:
        if collection.get_gid() in ("window_totals/truth_covered", "window_totals/truth_missed"):
            for x, level in collection.get_offsets():
                name = tick_names(channels)[round(x)]
                assert level == pytest.approx(truth[:, names.index(name)].mean())
                marked[name] = collection.get_gid() == "window_totals/truth_covered"
    assert marked == covered
    rates_drawn = {bar.get_gid(): bar.get_height() for bar in rates.patches}
    assert rates_drawn == pytest.approx(
        {
            "timeavg/channel/cov80": np.mean(list(covered.values())),
            "timeavg/channel/cov80_live": float(covered["channel:channel_0"]),
            "timeavg/channel/cov80_dead": float(covered["channel:channel_1"]),
        }
    )
    assert rates_drawn == pytest.approx({key: summary["metrics"][key] for key in rates_drawn})
    latent = result.components[..., result.component_mask].sum(axis=-1)
    structural = truth[:, result.component_mask].sum(axis=-1)
    points = next(item for item in fit.collections if item.get_gid() == "window_totals/latent_fit")
    np.testing.assert_allclose(points.get_offsets(), np.c_[structural, latent.mean(axis=0)])
    residual = np.sum((latent.mean(axis=0) - structural) ** 2)
    r2 = 1 - residual / np.sum((structural - structural.mean()) ** 2)
    assert summary["metrics"]["latent_total/r2"] == pytest.approx(r2)


def test_cumulative_panel_draws_running_totals_and_the_table_scores_them():
    result, truth, graph_truth = evidence_world()
    summary = compute_diagnostics(result, truth=truth, graph_truth=graph_truth)
    names = list(result.component_names)
    channels = ["channel:channel_0", "channel:channel_1"]
    axes = [ax for ax in plot_cumulative(summary).axes if ax.get_visible()]
    assert len(axes) == len(channels)
    for ax, name in zip(axes, channels):
        running = np.cumsum(result.components[..., names.index(name)], axis=1)
        lines = {line.get_label(): line for line in ax.lines}
        np.testing.assert_allclose(lines["Posterior mean"].get_ydata(), running.mean(axis=0))
        np.testing.assert_array_equal(lines["Posterior mean"].get_xdata(), result.time)
        np.testing.assert_allclose(
            lines["Truth"].get_ydata(), np.cumsum(truth[:, names.index(name)])
        )
        band = ax.collections[0].get_paths()[0].vertices
        low, high = np.quantile(running, [0.1, 0.9], axis=0)
        for time, lower, upper in zip(result.time, low, high):
            heights = band[band[:, 0] == time, 1]
            assert (min(heights), max(heights)) == pytest.approx((lower, upper))
        assert ax.get_title().startswith(name)
    scale = result.provenance["sales_scale"]
    index = [names.index(name) for name in channels]
    energy = reference_energy(
        np.cumsum(result.components[..., index], axis=1) / scale,
        np.cumsum(truth[:, index], axis=0) / scale,
    )
    assert summary["metrics"]["cumulative/channel/energy_score"] == pytest.approx(energy)


def test_temporal_dots_are_slot_scores_and_bars_the_table_family_values():
    result, truth, _ = evidence_world(weeks=7)
    summary = compute_diagnostics(result, truth=truth)
    names, families = list(result.component_names), list(result.component_families)
    active = result.component_mask
    slots = {
        family: [
            (result.components[..., index], truth[:, index])
            for index, name in enumerate(names)
            if families[index] == family and active[index]
        ]
        for family in ("baseline", "channel", "control", "indirect")
    }
    slots["latent_total"] = [
        (result.components[..., active].sum(axis=-1), truth[:, active].sum(axis=-1))
    ]
    drawn = set()
    for ax in plot_temporal(summary).axes:
        dots = next(item for item in ax.collections if str(item.get_gid()).endswith("/slots"))
        family = dots.get_gid().split("/")[1]
        references = [reference_temporal(*slot) for slot in slots[family]]
        positions = {}
        for bar in (item for item in ax.collections if item.get_gid() != dots.get_gid()):
            (start, value), (end, _) = bar.get_segments()[0]
            key = bar.get_gid().rsplit("/", 1)[1]
            positions[round((start + end) / 2)] = key
            family_value = mean_of_defined([reference[key] for reference in references])
            assert value == pytest.approx(family_value), bar.get_gid()
            assert summary["metrics"][bar.get_gid()] == pytest.approx(family_value)
        for position, key in positions.items():
            plotted = [y for x, y in dots.get_offsets() if round(x) == position]
            defined = [reference[key] for reference in references if reference[key] is not None]
            assert sorted(plotted) == pytest.approx(sorted(defined)), (family, key)
        tracking = mean_of_defined(
            [
                abs(reference["draw_lag1"] - reference["truth_lag1"])
                for reference in references
                if reference["draw_lag1"] is not None and reference["truth_lag1"] is not None
            ]
        )
        metric = summary["metrics"][f"temporal/{family}/lag1_tracking_abs_error"]
        assert metric == pytest.approx(tracking)
        drawn.add(family)
    assert drawn == set(slots)


NOT_EVIDENCED = "no positive edges — calibration not evidenced"


def reliability_world():
    """Router frequencies, graph truth and families over the active edges of a labeled world.

    ``cy`` and ``zb`` hold both truth classes; ``zc``, ``cc`` and ``zz`` hold no positives.
    """
    result, _, graph_truth = evidence_world(draws=7)
    probability = result.graphs.mean(axis=0)[result.edge_mask]
    labels = graph_truth[result.edge_mask]
    families = np.asarray(result.provenance["edge_families"])[result.edge_mask]
    return result, graph_truth, probability, labels, families


def legend_labels(ax):
    return dict(zip(*reversed(ax.get_legend_handles_labels()), strict=True))


def one_class_families(labels, families):
    return {family for family in set(families) if labels[families == family].sum() == 0}


def test_reliability_bins_draw_router_frequencies_against_graph_truth():
    result, graph_truth, probability, labels, families = reliability_world()
    summary = compute_diagnostics(result, graph_truth=graph_truth)
    ax = plot_reliability(summary, kind="binned").axes[0]
    edges = np.linspace(0, 1, 11)

    def bins(keep):
        index = np.clip(np.searchsorted(edges, probability[keep], side="right") - 1, 0, 9)
        return np.array(
            [
                (probability[keep][index == b].mean(), labels[keep][index == b].mean())
                for b in range(10)
                if (index == b).any()
            ]
        )

    everything = np.ones(probability.size, dtype=bool)
    pooled = next(line for line in ax.lines if line.get_gid() == "reliability/pooled")
    np.testing.assert_allclose(np.c_[pooled.get_xdata(), pooled.get_ydata()], bins(everything))
    index = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, 9)
    ece = sum(
        (index == b).mean() * abs(labels[index == b].mean() - probability[index == b].mean())
        for b in range(10)
        if (index == b).any()
    )
    assert summary["metrics"]["graph/ece"] == pytest.approx(ece)
    points = {item.get_gid(): item.get_offsets() for item in ax.collections if item.get_gid()}
    one_class = one_class_families(labels, families)
    assert set(points) == {f"reliability/{family}" for family in set(families) - one_class}
    for family in set(families) - one_class:
        np.testing.assert_allclose(points[f"reliability/{family}"], bins(families == family))
    # A family without positive edges draws no bins: its legend entry says why.
    drawn = legend_labels(ax)
    for family in one_class:
        [label] = [text for text in drawn if text.startswith(f"{family}:")]
        assert label.endswith(NOT_EVIDENCED)
        assert not len(drawn[label].get_xdata())


def isotonic_reference(p, y):
    """The max-min formula of the isotonic fit, per distinct probability (independent of PAVA)."""
    values = np.unique(p)
    hits = np.array([y[p == v].sum() for v in values])
    counts = np.array([(p == v).sum() for v in values])
    return values, np.array(
        [
            max(
                min(hits[j : k + 1].sum() / counts[j : k + 1].sum() for k in range(i, values.size))
                for j in range(i + 1)
            )
            for i in range(values.size)
        ]
    )


def test_reliability_corp_curve_is_the_isotonic_fit_of_truth_on_frequency():
    result, graph_truth, probability, labels, families = reliability_world()
    summary = compute_diagnostics(result, graph_truth=graph_truth)
    ax = plot_reliability(summary).axes[0]
    curves = {line.get_gid(): line for line in ax.lines if line.get_gid()}
    one_class = one_class_families(labels, families)
    assert one_class and set(families) - one_class
    for name, keep in (
        ("pooled", np.ones(probability.size, dtype=bool)),
        *((family, families == family) for family in set(families) - one_class),
    ):
        x, y = curves[f"reliability/{name}"].get_data()
        # Each block is a horizontal segment (lower, c) to (upper, c) over observed frequencies.
        lower, upper, level = np.asarray(x[::2]), np.asarray(x[1::2]), np.asarray(y[::2])
        np.testing.assert_array_equal(y[1::2], level)
        values, fitted = isotonic_reference(probability[keep], labels[keep])
        block = np.searchsorted(lower, values, side="right") - 1
        assert np.all((lower[block] <= values) & (values <= upper[block]))
        np.testing.assert_allclose(level[block], fitted, rtol=0, atol=1e-12)
    for family in one_class:
        assert not len(curves[f"reliability/{family}"].get_xdata())
        assert curves[f"reliability/{family}"].get_label().endswith(NOT_EVIDENCED)
    metrics = summary["metrics"]
    assert metrics["graph/mcb"] - metrics["graph/dsc"] + metrics["graph/unc"] == pytest.approx(
        metrics["graph/brier"]
    )
    assert f"MCB {metrics['graph/mcb']:.3g}" in ax.get_title().replace("\n", " ")


@pytest.mark.parametrize("mode", ["model_average", "fixed"])
def test_one_class_edges_leave_the_brier_decomposition_undefined(mode):
    result, graph_truth, *_ = reliability_world()
    provenance = {
        **result.provenance,
        "inference_config": {**result.provenance.get("inference_config", {}), "mode": mode},
        "graph_source": "supplied_hypothesis" if mode == "fixed" else "joint_router",
    }
    result = replace(result, mode=mode, provenance=provenance)
    summary = compute_diagnostics(result, graph_truth=np.zeros_like(graph_truth))
    for key in ("mcb", "dsc", "unc"):
        assert summary["metrics"][f"graph/{key}"] is None
        assert summary["statuses"][f"metric/graph/{key}"]["reason"] == NOT_EVIDENCED
    # The historical ECE keeps its value but is not calibration evidence in any graph
    # mode; a supplied hypothesis adds its own qualification.
    assert summary["metrics"]["graph/ece"] is not None
    for status in (summary["statuses"]["metric/graph/ece"], summary["panels"]["reliability"]):
        assert status["status"] == "not_evidenced"
        assert status["reason"].startswith(NOT_EVIDENCED)
    for kind in ("corp", "binned"):
        ax = plot_reliability(summary, kind=kind).axes[0]
        assert NOT_EVIDENCED in ax.get_title().replace("\n", " ")
        assert all(not len(line.get_xdata()) for line in ax.lines if line.get_gid())
        assert not [item for item in ax.collections if item.get_gid()]


def test_reliability_evidence_saved_without_corp_rerenders_its_bins():
    result, graph_truth, *_ = reliability_world()
    summary = compute_diagnostics(result, graph_truth=graph_truth)
    saved = json.loads(json.dumps(summary))
    for record in (
        saved["arrays"]["reliability"]["pooled"],
        *saved["arrays"]["reliability"]["families"].values(),
    ):
        for key in ("corp", "decomposition", "notice"):
            del record[key]
    with pytest.raises(ValueError, match="kind='binned'"):
        plot_reliability(saved)
    assert "equal-width bins" in plot_panel("reliability", saved).axes[0].get_title()
    assert "CORP" in plot_panel("reliability", summary).axes[0].get_title()
    with pytest.raises(ValueError, match="graph truth"):
        plot_panel("reliability", compute_diagnostics(result))


def dead_leakage_axes(figure):
    return [
        ax
        for ax in figure.axes
        if any(str(bar.get_gid()).startswith("attribution/dead_") for bar in ax.patches)
    ]


@pytest.mark.parametrize("zero_leakage", [True, False])
def test_dead_leakage_axes_start_at_zero_and_show_every_bar(zero_leakage):
    result, truth, graph_truth = evidence_world()
    names = list(result.component_names)
    components = result.components.copy()
    for name, live in LIVE_EFFECTS.items():
        if not live and zero_leakage:
            # Draws and truth both exactly zero: every leakage value is exactly 0.
            components[..., names.index(name)] = 0.0
            truth[:, names.index(name)] = 0.0
    result = replace(result, components=components)
    summary = compute_diagnostics(result, truth=truth, graph_truth=graph_truth)
    axes = dead_leakage_axes(plot_attribution(summary))
    assert len(axes) == 2  # dead media and dead controls
    for ax in axes:
        heights = [bar.get_height() for bar in ax.patches]
        bottom, top = ax.get_ylim()
        assert bottom == 0.0
        assert top > 0.0 and top >= max(heights)
        assert bool(max(heights) == 0.0) == zero_leakage


#: A recorded metric name longer than any word an ``ax=`` heading keeps whole.
LONG_METRIC = "contribution_lag1_tracking_abs_error"


def unnamed_run(root):
    """A two-evaluation run recorded without a name, as ``train()`` and the CLI default to."""
    from test_training_run import write_run

    from pymc_mfm.diagnostics import TrainingRun

    write_run(root, evaluations=(2, 4))
    record = json.loads((root / "run.json").read_text())
    record.update(run_name=None, run_id="20260924T090000Z-" + "3f1c9a7e" * 4)
    (root / "run.json").write_text(json.dumps(record))
    events = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
    for event in events:
        event["flat"].update(
            {f"{leg}/conditional/query/{LONG_METRIC}": 0.5 for leg in ("holdout", "seen")}
        )
    (root / "evaluations.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return TrainingRun.load(root)


def test_single_axes_headers_fit_inside_their_figures_and_beside_each_other(tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def drawn_titles(figure, axes):
        figure.canvas.draw()
        return [ax.title.get_window_extent(figure.canvas.get_renderer()) for ax in axes]

    def grid_headings(size, shape, draws):
        """Headings of ``draws`` in a caller's constrained grid, checked to fit it."""
        grid = Figure(figsize=size, layout="constrained")
        FigureCanvasAgg(grid)
        axes = list(grid.subplots(*shape).flat)
        for draw, ax in zip(draws, axes, strict=True):
            draw(ax)
        titles = drawn_titles(grid, axes)
        for box, ax in zip(titles, axes):
            assert 0 <= box.x0 and box.x1 <= grid.bbox.width, ax.get_title()
            assert 0 <= box.y0 and box.y1 <= grid.bbox.height, ax.get_title()
        assert not any(
            first.overlaps(second) for first, second in itertools.combinations(titles, 2)
        )
        return [ax.get_title() for ax in axes]

    result = replace(make_result(), units="sales units in the outcome's natural scale")
    truth = (result.components.mean(0) + 1.0) * result.component_mask
    summary = compute_diagnostics(result, truth=truth, graph_truth=[1, 0, 1])
    for figure in (
        plot_recovery_scatter(summary),
        plot_channel_totals(summary),
        plot_graph_posterior(summary),
        plot_uncertainty(summary),
    ):
        figure.canvas.draw()
        header = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
        assert header.x0 >= 0 and header.x1 <= figure.bbox.width, figure._suptitle.get_text()
    # ax= panels in a caller's constrained grid, which narrows each axes after
    # its title is written: every title stays inside the figure and clear of
    # the others.
    grid_headings(
        (10, 7),
        (2, 2),
        [
            lambda ax, draw=draw: draw(summary, ax=ax)
            for draw in (plot_uncertainty, plot_zero_atoms, plot_reliability, plot_cycles)
        ],
    )
    run = unnamed_run(tmp_path / "run")
    run_figures = (
        lambda ax: run.plot.evaluation(LONG_METRIC, ax=ax),
        lambda ax: run.plot.heads("decomposition_energy_score", ax=ax),
        lambda ax: run.plot.learning_rate(ax=ax),
    )
    headings = grid_headings((15, 6), (1, 3), run_figures)
    # An unnamed run is headed by its start time and first id digits; run.json keeps the rest.
    for heading in headings:
        joined = heading.replace("\n", "")
        assert run.run["run_id"][:25] in joined and run.run["run_id"] not in joined, heading
    # Axes on a subfigure of a subfigure: each run heading stays inside its subfigure.
    for draw in run_figures:
        nested = Figure(figsize=(10, 6), layout="constrained")
        FigureCanvasAgg(nested)
        left, right = nested.subfigures(1, 2)
        top, bottom = right.subfigures(2, 1)
        left.subplots()
        bottom.subplots()
        ax = top.subplots()
        draw(ax)
        (box,) = drawn_titles(nested, [ax])
        assert top.bbox.x0 - 1 <= box.x0 and box.x1 <= top.bbox.x1 + 1, ax.get_title()
        assert box.y1 <= top.bbox.y1 + 1, ax.get_title()


@pytest.mark.parametrize("render", [plot_uncertainty, plot_graph_posterior])
def test_headers_of_wide_glyphs_wrap_inside_their_figures(render):
    # Glyphs wider than the width estimate still wrap at draw time.
    units = "売上高（百万円）週次の自然な単位での売上高と予算配分の比較"
    summary = compute_diagnostics(replace(make_result(), units=units))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the default font has no CJK glyphs; boxes keep a width
        figure = render(summary)
        figure.canvas.draw()
    box = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
    assert 0 <= box.x0 and box.x1 <= figure.bbox.width, (box.x0, box.x1, figure.bbox.width)
