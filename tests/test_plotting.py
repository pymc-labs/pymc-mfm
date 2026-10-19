"""Plot values and labels share the persisted diagnostic definitions."""

from dataclasses import replace

import numpy as np
import pytest
from test_diagnostics import make_result

from pymc_mfm.diagnostics import compute_diagnostics
from pymc_mfm.plotting import (
    plot_calibration,
    plot_channel_totals,
    plot_component_trajectories,
    plot_graph_posterior,
    plot_panel,
    plot_records,
    plot_recovery_scatter,
    plot_structure_heatmap,
    plot_target_recovery,
    plot_uncertainty,
)

pytest.importorskip("matplotlib")


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


def test_edge_heatmap_uses_active_aligned_edges_and_fixed_hypothesis_label():
    result = make_result()
    result = replace(
        result,
        mode="fixed",
        graphs=np.tile([1, 0, 1], (5, 1)),
        provenance={**result.provenance, "graph_source": "supplied_hypothesis"},
    )
    summary = compute_diagnostics(result, graph_truth=[1, 0, 0])
    figure = plot_structure_heatmap(summary)
    ax = figure.axes[0]
    np.testing.assert_array_equal(ax.images[0].get_array(), [[1, 1], [1, 0]])
    assert ax.get_yticklabels()[0].get_text() == "Selected hypothesis indicators"
    assert all("padding_slot" not in label.get_text() for label in ax.get_xticklabels())


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
    assert "not independent-world SBC" in figure._suptitle.get_text()
    with pytest.raises(ValueError, match="truth"):
        plot_calibration(compute_diagnostics(result))


def test_arbitrary_saved_auxiliary_trajectories_preserve_undefined_gaps():
    name = "heads/auxiliary/zb/set_ece"
    summary = compute_diagnostics(
        make_result(),
        evaluation_records=[
            {"step": 1, "heads": {"auxiliary": {"zb": {"set_ece": 0.2}}}},
            {"step": 2, "heads": {"auxiliary": {"zb": {"set_ece": None}}}},
            {"step": 3, "heads": {"auxiliary": {"zb": {"set_ece": 0.1}}}},
        ],
    )
    figure = plot_records(summary, kind="evaluation", metric_names=[name])
    line = figure.axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), [0.2, np.nan, 0.1], equal_nan=True)
    assert line.get_label() == name
    with pytest.raises(KeyError, match="unknown"):
        plot_records(summary, kind="evaluation", metric_names=["absent"])


def test_unknown_panel_is_not_silently_skipped():
    with pytest.raises(KeyError, match="unknown diagnostic panel"):
        plot_panel("missing_panel", compute_diagnostics(make_result()))
