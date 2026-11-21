"""Comparison and calibration accessors redraw saved evidence of one world and window."""

import json
import re
from dataclasses import replace

import numpy as np
import pytest
from test_comparisons import omission_pair, posterior_evidence, record, reference

from pymc_mktfm.diagnostics import CalibrationReport, Comparison, sbc_report

NAMES = ("baseline", "channel:a", "channel:off") + tuple(f"control:z{i}" for i in range(5))
FAMILIES = ("baseline", "channel", "channel") + ("control",) * 5
# Two weeks; channel:off is generating-inactive and all five controls are live.
TRUTH = np.tile([10.0, 2.0, 0.0] + [1.0] * 5, (2, 1))
LIVE = np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
GRAPH_TRUTH = np.array([1, 0, 1, 1, 1, 1, 1])  # cy:a, cy:off, zb:z0 ... zb:z4
HEALTHY = {
    "divergences": 0,
    "max_rhat": 1.0,
    "min_ess_bulk": 500,
    "min_ess_tail": 500,
    "tree_depth_hits": 0,
    "min_bfmi": 0.5,
}
C2ST = {
    "labels": [0, 0, 1, 1],
    "probabilities": [0.1, 0.2, 0.8, 0.9],
    "world_ids": ["a", "b", "a", "b"],
    "training_world_ids": ["train"],
    "classifier": "frozen",
}


def draws(channel, off=0.0):
    """Truth in every week, except per-draw live-channel values and the inactive channel."""
    values = np.tile(TRUTH, (len(channel), 1, 1))
    values[:, :, 1] = np.asarray(channel, dtype=float)[:, None]
    values[:, :, 2] = off
    return values


def result(samples, *, mode="model_average", window="query", **provenance):
    base = record(draws=samples, names=NAMES, families=FAMILIES).result
    if window == "context":
        base = replace(base, window="context", time=np.arange(2))
    return replace(base, mode=mode, provenance=dict(base.provenance, sales_scale=2.0, **provenance))


def results(**options):
    """A model average and B map view of one checkpoint, and an exact oracle of another."""
    return {
        "A": result(draws([1, 1, 3, 3]), **options),
        "B": result(draws([4] * 8, off=0.1), mode="map", **options),
        "oracle": result(draws([2] * 4), checkpoint_sha256="c" * 64, **options),
    }


def full_comparison(order=("A", "B", "oracle"), **settings):
    models = results()
    return Comparison(
        {name: models[name] for name in order},
        truth=TRUTH,
        live=LIVE,
        graph_truth=GRAPH_TRUTH,
        oracle_pairs=[("oracle", "A"), ("oracle", "B")],
        bootstrap_samples=20,
        **settings,
    )


def omission_comparison():
    before, after = omission_pair()
    # Omitting channel:b changes the encoded observations and therefore their identity.
    provenance = dict(after.result.provenance, data_sha256="c" * 64)
    after = replace(after, result=replace(after.result, provenance=provenance))
    return before, after


def markers(figure):
    """Plotted point estimate per (family axis, method) of an aggregate figure."""
    return {
        (ax.get_title(), line.get_label()): float(line.get_xdata()[0])
        for ax in figure.axes
        for line in ax.lines
        if line.get_marker() == "o"
    }


def paired_means(figure):
    """Plotted paired mean per (family axis, row), rows in plotted order."""
    return {
        (ax.get_title(), row): float(line.get_xdata()[0])
        for ax in figure.axes
        for row, line in enumerate(line for line in ax.lines if line.get_marker() == "o")
    }


def per_family(**rows):
    """Expected paired means, ``family=[row 0, row 1, ...]``."""
    return pytest.approx(
        {
            (family, row): value
            for family, values in rows.items()
            for row, value in enumerate(values)
        }
    )


def test_mapping_comparison_plots_hand_scored_rows_and_pair_directions():
    pytest.importorskip("matplotlib")
    comparison = full_comparison()
    # Window-mean CRPS / sales_scale 2: A draws 1,1,3,3 against 2 score 1 - 0.5 = 0.5;
    # B is a point mass at 4; B's total also carries its 0.1 false channel effect.
    assert markers(comparison.plot.performance()) == pytest.approx(
        {
            ("channel", "A"): 0.25,
            ("channel", "B"): 1.0,
            ("channel", "oracle"): 0.0,
            ("control", "A"): 0.0,
            ("control", "B"): 0.0,
            ("control", "oracle"): 0.0,
            ("total", "A"): 0.25,
            ("total", "B"): 1.05,
            ("total", "oracle"): 0.0,
        }
    )
    # A's model average and B's map view share a checkpoint, so the summary pairs
    # them itself as A minus B in either mapping order; the default method pairs
    # leave that contrast to graph_averaging and the declared oracles to oracle.
    for order in (("A", "B", "oracle"), ("B", "A", "oracle")):
        each = full_comparison(order)
        kinds = {(row["first"], row["second"], row["kind"]) for row in each.summary["paired"]}
        assert kinds == {
            ("B", "A", "graph_averaging"),
            ("oracle", "A", "oracle_comparison"),
            ("oracle", "B", "oracle_comparison"),
        }, order
        assert paired_means(each.plot.graph_averaging()) == per_family(
            baseline=[0], channel=[-0.75], control=[0], total=[-0.8]
        )
        with pytest.raises(ValueError, match="method_pairs"):
            each.plot.ab()
    # B of another checkpoint makes B minus A a default method pair.
    distinct = Comparison(
        {"A": results()["A"], "B": result(draws([4] * 8, off=0.1), checkpoint_sha256="d" * 64)},
        truth=TRUTH,
        live=LIVE,
        bootstrap_samples=20,
    )
    assert paired_means(distinct.plot.ab()) == per_family(
        baseline=[0], channel=[0.75], control=[0], total=[0.8]
    )
    assert paired_means(comparison.plot.oracle()) == per_family(
        baseline=[0, 0], channel=[0.25, 1.0], control=[0, 0], total=[0.25, 1.05]
    )
    assert markers(comparison.plot.inactive_attribution())[("channel", "B")] == pytest.approx(0.05)
    world = comparison.plot.world()
    channel = next(ax for ax in world.axes if ax.get_title() == "channel:a")
    lines = {line.get_label(): line.get_ydata() for line in channel.lines}
    np.testing.assert_allclose(lines["A (model_average)"], [2, 2])
    np.testing.assert_allclose(lines["B (map)"], [4, 4])
    np.testing.assert_allclose(lines["Generating truth"], TRUTH[:, 1])
    with pytest.raises(ValueError, match="oracle_pairs"):
        Comparison(
            results(), truth=TRUTH, method_pairs=[("oracle", "A")], oracle_pairs=[("oracle", "A")]
        )


def test_comparison_draws_its_common_window_and_states_each_methods_dimensions():
    pytest.importorskip("matplotlib")
    comparison = Comparison(
        {
            "A": result(draws([1, 1, 3, 3]), window="context"),
            "B": result(draws([4] * 8), mode="map", window="context", head="joint"),
        },
        truth=TRUTH,
        live=LIVE,
        bootstrap_samples=20,
    )
    assert comparison.window == "context"
    coverage = comparison.plot.coverage()
    assert markers(coverage)[("channel", "A")] == 1
    assert markers(coverage)[("channel", "B")] == 0
    for figure, units in (
        (coverage, "fraction"),
        (comparison.plot.performance(), "fraction_of_outcome_scale"),
    ):
        stated = set(re.findall(r"\w+", figure.get_suptitle().splitlines()[-1]))
        assert {"conditional", "joint", "context", "model_average", "map", "4", "8"} <= stated
        assert units in stated


def test_figures_name_the_evidence_they_lack_and_truth_free_figures_still_draw():
    pytest.importorskip("matplotlib")
    comparison = Comparison(results(), bootstrap_samples=20)
    with pytest.raises(ValueError, match="truth"):
        comparison.plot.performance()
    with pytest.raises(ValueError, match="graph_truth"):
        comparison.plot.structure_recovery()
    with pytest.raises(ValueError, match="oracle_pairs"):
        Comparison(results(), truth=TRUTH, bootstrap_samples=20).plot.oracle()
    with pytest.raises(ValueError, match="live"):
        Comparison(results(), truth=TRUTH, bootstrap_samples=20).plot.recovery()
    channel = next(ax for ax in comparison.plot.world().axes if ax.get_title() == "channel:a")
    lines = {line.get_label(): line.get_ydata() for line in channel.lines}
    assert "Generating truth" not in lines
    np.testing.assert_allclose(lines["A (model_average)"], draws([1, 1, 3, 3])[:, :, 1].mean(0))


def test_comparison_requires_one_window_world_and_data_identity():
    a, b = result(draws([1, 1, 3, 3])), result(draws([4] * 8))
    with pytest.raises(ValueError, match="window"):
        Comparison({"A": a, "B": result(draws([4] * 8), window="context")})
    with pytest.raises(ValueError, match="data_sha256"):
        Comparison({"A": a, "B": replace(b, provenance=dict(b.provenance, data_sha256="c" * 64))})
    first = record()
    with pytest.raises(ValueError, match="world"):
        Comparison([first, replace(first, method="B", world="1")])
    with pytest.raises(ValueError, match="truth"):
        Comparison([first], truth=first.truth)


@pytest.mark.parametrize("scale", [None, True, "2", 0])
def test_mapping_results_must_record_the_sales_scale_that_scales_them(scale):
    recorded = result(draws([1, 1, 3, 3]))
    provenance = {key: value for key, value in recorded.provenance.items() if key != "sales_scale"}
    if scale is not None:
        provenance["sales_scale"] = scale
    with pytest.raises(ValueError, match="'A'.*sales_scale"):
        Comparison({"A": replace(recorded, provenance=provenance)}, truth=TRUTH)


def test_a_declared_oracle_replaces_the_automatic_pair_of_one_checkpoint():
    pytest.importorskip("matplotlib")
    # A fixed true-graph view of the model average's own checkpoint, declared its oracle.
    views = {
        "true_graph": result(draws([2] * 4), mode="fixed"),
        "average": result(draws([1, 1, 3, 3])),
    }
    settings = {"truth": TRUTH, "live": LIVE, "bootstrap_samples": 20}
    for oracle in (("true_graph", "average"), ("average", "true_graph")):
        comparison = Comparison(views, oracle_pairs=[oracle], **settings)
        kinds = {(row["first"], row["second"], row["kind"]) for row in comparison.summary["paired"]}
        assert kinds == {(*oracle, "oracle_comparison")}, oracle
    # Average minus the exact oracle: draws 1,1,3,3 against 2 score 1 - 0.5, over sales_scale 2.
    oracle = Comparison(views, oracle_pairs=[("true_graph", "average")], **settings)
    assert paired_means(oracle.plot.oracle()) == per_family(
        baseline=[0], channel=[0.25], control=[0], total=[0.25]
    )


def test_omitted_case_may_change_data_only_under_declared_omission_pairs():
    pytest.importorskip("matplotlib")
    before, after = omission_comparison()
    with pytest.raises(ValueError, match="data_sha256"):
        Comparison([before, after])
    comparison = Comparison([before, after], omission_pairs=[("base", "omitted")])
    # Retained channel:a CRPS falls from |4 - 2| to |3 - 2| once channel:b is omitted.
    assert paired_means(comparison.plot.matched_omission()) == per_family(channel=[-1])
    # Latent total 17 hides 5 omitted units; the omitted fit's total 13 absorbs 1 - 4/5.
    assert markers(comparison.plot.absorbed_share())[("all", "A")] == pytest.approx(0.2)


def test_every_comparison_figure_and_evidence_report_stays_off_pyplot(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    from matplotlib.figure import Figure

    from pymc_mktfm.diagnostics import build_report
    from pymc_mktfm.diagnostics.comparisons import (
        corpus_support_summary,
        input_support_summary,
        reference_diagnostics,
    )
    from pymc_mktfm.plotting import comparisons as plots

    before = plt.get_fignums()
    # An explicit method pair beside the automatic graph-averaging one draws ab.
    full = full_comparison(method_pairs=[("A", "B")])
    figures = [
        getattr(full.plot, name)()
        for name in (
            "performance",
            "coverage",
            "recovery",
            "failure_size",
            "false_effects",
            "five_controls",
            "error_decomposition",
            "component_hit_rate",
            "control_count_recovery",
            "inactive_attribution",
            "false_effect_worlds",
            "structure_recovery",
            "ab",
            "oracle",
            "graph_averaging",
            "world",
        )
    ]
    figures += [full.plot.graph_count_recovery("zb"), full.plot.world("window_mean")]
    omission = Comparison(list(omission_comparison()), omission_pairs=[("base", "omitted")])
    figures += [omission.plot.matched_omission(), omission.plot.absorbed_share()]
    first = record()
    references = Comparison(
        [first, replace(first, method="mcmc", kind="reference", health=HEALTHY)],
        bootstrap_samples=20,
    )
    figures.append(references.plot.reference_health())
    evidence = reference(first.result, posterior_sbc=posterior_evidence(), c2st=C2ST)
    saved = reference_diagnostics(first.result, evidence)
    figures += [
        plots.plot_reference_comparison(saved),
        plots.plot_posterior_sbc(saved["posterior_sbc"]),
        plots.plot_c2st(saved["c2st"]),
        plots.plot_input_support(input_support_summary({"control:z": [1, 3]}, {"control:z": [2]})),
        plots.plot_corpus_support(corpus_support_summary({"first": [98, 102]})),
    ]
    assert all(isinstance(figure, Figure) for figure in figures)
    panels = plots.render_comparisons(full.summary, tmp_path / "comparison")
    assert "failed" not in {panel["status"] for panel in panels}
    manifest = build_report(first.result, tmp_path / "report", reference=evidence)
    assert {
        manifest["panels"][name]["render_status"]
        for name in ("reference_comparison", "posterior_sbc", "c2st")
    } == {"ok"}
    assert plt.get_fignums() == before


#: The renderer each documented ``Comparison.plot`` method draws.
COMPARISON_FIGURES = {
    "performance": "plot_performance",
    "coverage": "plot_coverage",
    "recovery": "plot_recovery",
    "failure_size": "plot_failure_size",
    "false_effects": "plot_false_effects",
    "five_controls": "plot_five_controls",
    "error_decomposition": "plot_error_decomposition",
    "component_hit_rate": "plot_component_hit_rate",
    "control_count_recovery": "plot_control_count_recovery",
    "inactive_attribution": "plot_inactive_attribution",
    "false_effect_worlds": "plot_false_effect_worlds",
    "absorbed_share": "plot_absorbed_share",
    "structure_recovery": "plot_structure_recovery",
    "graph_count_recovery": "plot_graph_count_recovery",
    "ab": "plot_ab_comparison",
    "oracle": "plot_oracle_comparison",
    "graph_averaging": "plot_graph_averaging",
    "matched_omission": "plot_matched_omission",
    "world": "plot_world_comparison",
    "reference_health": "plot_reference_health",
}


def test_each_comparison_method_draws_its_documented_figure(monkeypatch):
    from pymc_mktfm.diagnostics.comparison_accessors import ComparisonPlots
    from pymc_mktfm.plotting import comparisons as plots

    drawn = []
    for function in COMPARISON_FIGURES.values():
        monkeypatch.setattr(plots, function, lambda summary, _f=function, **_: drawn.append(_f))
    first = record()
    omission = Comparison(list(omission_comparison()), omission_pairs=[("base", "omitted")])
    comparisons = {
        "absorbed_share": omission,
        "matched_omission": omission,
        "reference_health": Comparison(
            [first, replace(first, method="mcmc", kind="reference", health=HEALTHY)],
            bootstrap_samples=20,
        ),
    }
    full = full_comparison(method_pairs=[("A", "B")])
    assert set(COMPARISON_FIGURES) == {
        name for name in dir(ComparisonPlots) if not name.startswith("_")
    }
    for method, function in COMPARISON_FIGURES.items():
        drawn.clear()
        options = {"family": "zb"} if method == "graph_count_recovery" else {}
        getattr(comparisons.get(method, full).plot, method)(**options)
        assert drawn == [function], method


def test_calibration_report_redraws_and_persists_the_saved_sbc_card(tmp_path):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from pymc_mktfm.diagnostics import render_report

    card = sbc_report(
        np.zeros((20, 4, 1)),
        np.ones((20, 1)),
        world_ids=[str(index) for index in range(20)],
        quantity_names=["total"],
        provenance={
            "independent_worlds": True,
            "synthetic_training_evaluation": True,
            "data_sha256": "c" * 64,
            "prior": "test point mass",
        },
        simulations=200,
    )
    calibration = CalibrationReport(card)
    saved = card["quantities"]["total"]
    ranks = calibration.plot.card("ranks")
    assert [bar.get_height() for bar in ranks.axes[0].containers[0]] == saved["rank_histogram"]
    ecdf = calibration.plot.card("ecdf")
    line = next(line for line in ecdf.axes[0].lines if line.get_label() == "Saved rank ECDF")
    np.testing.assert_array_equal(line.get_ydata(), saved["ecdf"])
    manifest = calibration.report(tmp_path / "sbc")
    stored = json.loads((tmp_path / "sbc" / "calibration.json").read_text())
    assert stored["ranks"] == card["ranks"]
    assert render_report(tmp_path / "sbc", tmp_path / "replayed")["metrics"] == manifest["metrics"]
    with pytest.raises(ValueError, match="sbc_report"):
        CalibrationReport(manifest)
