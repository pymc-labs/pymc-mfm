"""Finite-draw oracles, scientific qualifications and persisted report failures."""

import hashlib
import json
from dataclasses import replace
from html.parser import HTMLParser
from urllib.request import urlopen

import numpy as np
import pytest

from pymc_mfm.diagnostics import (
    DiagnosticError,
    build_report,
    calibration_summary,
    cardinality_decode,
    compute_diagnostics,
    factorized_set_diagnostics,
    graph_support_diagnostics,
    paired_graph_uncertainty,
    randomized_ranks,
    randomized_set_pit,
    render_report,
    sbc_report,
    write_calibration_report,
)
from pymc_mfm.inference.result import DecompositionResult


def make_result():
    graphs = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]])
    draws = np.zeros((5, 3, 8))
    draws[..., 0] = 10 + np.arange(5)[:, None] + np.arange(3)
    draws[..., 1] = graphs[:, 0, None] * np.array([1, 2, 3])
    draws[..., 3] = graphs[:, 2, None] * np.array([-1, 0, 1])
    draws[..., 4] = np.arange(5)[:, None] / 4
    draws[..., 5:] = np.arange(3)[None, :, None] / 10
    return DecompositionResult(
        components=draws,
        graphs=graphs,
        component_names=(
            "baseline",
            "channel:paid search",
            "channel:padding_slot",
            "control:weather",
            "latent",
            "indirect:cc",
            "indirect:zc",
            "indirect:dc",
        ),
        component_families=(
            "baseline",
            "channel",
            "channel",
            "control",
            "latent",
            "indirect",
            "indirect",
            "indirect",
        ),
        component_mask=np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool),
        edge_names=(
            "cy:channel:paid search->sales",
            "cy:channel:padding_slot->sales",
            "zb:control:weather->outcome",
        ),
        edge_mask=np.array([1, 0, 1], dtype=bool),
        channel_names=("paid search", "padding_slot"),
        control_names=("weather",),
        time=np.arange(3) + 4,
        window="query",
        mode="model_average",
        units="revenue",
        context_length=4,
        provenance={
            "checkpoint_sha256": "a" * 64,
            "data_sha256": "b" * 64,
            "source_step": 7,
            "data_split": "heldout",
            "profile": "test",
            "sales_scale": 10,
            "graph_source": "joint_router",
            "edge_endpoints": [
                ["channel:paid search", "sales"],
                ["channel:padding_slot", "sales"],
                ["control:weather", "outcome"],
            ],
            "edge_families": ["cy", "cy", "zb"],
            "synthetic_training_evaluation": True,
        },
    )


def test_paired_mixture_variance_and_covariance_match_finite_oracle():
    graphs = np.array([[0], [0], [0], [1], [1]])
    values = np.array([[1, 5], [2, 4], [4, 7], [10, -1], [12, 1]], dtype=float)
    report = paired_graph_uncertainty(graphs, values, covariance=True, rare_count=3)
    group0, group1 = values[:3], values[3:]
    within = 3 / 5 * np.var(group0, axis=0) + 2 / 5 * np.var(group1, axis=0)
    mean = values.mean(axis=0)
    between = 3 / 5 * (group0.mean(0) - mean) ** 2 + 2 / 5 * (group1.mean(0) - mean) ** 2
    np.testing.assert_allclose(report["within_graph_variance"], within)
    np.testing.assert_allclose(report["between_graph_variance"], between)
    np.testing.assert_allclose(report["total_variance"], np.var(values, axis=0))
    np.testing.assert_allclose(report["variance_closure_error"], 0, atol=1e-14)
    centered = values - mean
    np.testing.assert_allclose(report["total_covariance"], centered.T @ centered / len(values))
    np.testing.assert_allclose(report["covariance_closure_error"], 0, atol=1e-14)
    assert report["n_rare_groups"] == 1
    assert report["rare_draw_mass"] == pytest.approx(0.4)


def test_complete_graph_mode_is_not_factorized_marginal_mode():
    graphs = np.array([[0, 1]] * 4 + [[1, 0]] * 4 + [[1, 1]] * 2)
    report = paired_graph_uncertainty(graphs, np.arange(10.0))
    assert report["mode_graph"] == [0, 1]
    assert report["tied_mode_graphs"] == [[0, 1], [1, 0]]
    assert report["mode_frequency"] == pytest.approx(0.4)
    assert (graphs.mean(0) >= 0.5).tolist() == [True, True]
    assert report["graph_frequencies"][2]["frequency"] == pytest.approx(0.2)


def test_singleton_uncertainty_and_absent_graph_are_not_fabricated():
    report = paired_graph_uncertainty([[0, 0], [1, 0]], [[3, 4], [7, 8]], selected_graph=[0, 1])
    assert report["n_singleton_groups"] == 2
    np.testing.assert_array_equal(report["within_graph_variance"], [0, 0])
    np.testing.assert_array_equal(report["between_graph_variance"], [4, 4])
    assert report["selected_group"]["status"] == "absent"
    assert report["selected_group"]["mean"] is None
    assert report["selected_group"]["sd"] is None


def test_paired_graph_alignment_binary_and_finite_contracts():
    with pytest.raises(ValueError, match="aligned"):
        paired_graph_uncertainty([[0], [1]], [1])
    with pytest.raises(ValueError, match="binary"):
        paired_graph_uncertainty([[0.1], [1]], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        paired_graph_uncertainty([[0], [1]], [1, np.nan])
    with pytest.raises(ValueError, match="aggregates"):
        paired_graph_uncertainty([[0], [1]], np.zeros((2, 2, 2)), covariance=True)
    report = paired_graph_uncertainty(np.empty((2, 0)), [1, 3])
    assert report["graph_frequencies"] == [{"graph": [], "count": 2, "frequency": 1.0}]
    assert report["within_graph_variance"] == 1
    assert report["between_graph_variance"] == 0


def test_exact_zero_ties_cover_all_inclusive_randomized_ranks_without_global_rng():
    before = np.random.get_state()
    ranks = randomized_ranks(np.zeros((4, 200)), np.zeros(200), seed=0)
    np.testing.assert_array_equal(
        ranks,
        randomized_ranks(np.zeros((4, 200)), np.zeros(200), seed=0),
    )
    assert set(ranks) == {0, 1, 2, 3, 4}
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    summary = calibration_summary(np.zeros((4, 200)), np.zeros(200), seed=0)
    assert summary["status"] == "screening_only"
    assert summary["ks_pvalue"] is None
    assert summary["ecdf_band_reject"] is None
    assert summary["intervals"][1]["coverage"] == 1
    assert summary["intervals"][1]["mean_width"] == 0
    np.testing.assert_allclose(summary["pit"], (ranks + 0.5) / 5)


def test_interval_coverage_is_not_rank_coverage_at_atoms():
    summary = calibration_summary(np.zeros((3, 100)), np.zeros(100), seed=0)
    assert summary["intervals"][0]["coverage"] == 1
    assert summary["rank_coverage"]["cov50"] < 1
    with pytest.raises(ValueError, match="exactly"):
        calibration_summary(np.ones((2, 3)), np.ones((3, 1)))


def test_sbc_requires_independent_identified_worlds_and_detects_extreme_miscalibration():
    pytest.importorskip("scipy")
    draws = np.zeros((30, 8, 1))
    truth = np.full((30, 1), 10.0)
    kwargs = dict(world_ids=[str(i) for i in range(30)], quantity_names=["total"], simulations=300)
    with pytest.raises(ValueError, match="independent_worlds"):
        sbc_report(draws, truth, provenance={}, **kwargs)
    with pytest.raises(ValueError, match="uniquely"):
        sbc_report(
            draws,
            truth,
            provenance={"independent_worlds": True},
            **{**kwargs, "world_ids": ["same"] * 30},
        )
    report = sbc_report(
        draws,
        truth,
        provenance={
            "independent_worlds": True,
            "prior": "test point mass",
            "data_sha256": "c" * 64,
        },
        **kwargs,
    )
    assert report["quantities"]["total"]["ecdf_band_reject"] == 1
    assert report["family_wise"]["rejecting_quantities"] == ["total"]


def test_masked_components_do_not_dilute_recovery_and_noise_is_not_closure():
    result = make_result()
    truth = result.components.mean(axis=0)
    observed = truth.sum(axis=-1) + np.array([2.0, -3.0, 4.0])
    report = compute_diagnostics(result, truth=truth, graph_truth=[1, 0, 1], observed=observed)
    assert report["metrics"]["closure/latent_relative_error"] == pytest.approx(0)
    assert report["metrics"]["observation/latent_mean_absolute_residual"] == pytest.approx(3)
    np.testing.assert_allclose(report["arrays"]["total"]["realized_noise"], [2, -3, 4])
    np.testing.assert_allclose(report["arrays"]["total"]["mean"], truth.sum(axis=-1))
    assert report["arrays"]["calibration"]["pooled"]["n_ranks"] == 3 * 7
    assert report["metrics"]["support/padded_components"] == 1
    assert report["metrics"]["graph/false_exclusion_mass"] == pytest.approx(0.5)
    assert report["metrics"][
        "component/channel%3Apaid%20search/zero_atom_probability"
    ] == pytest.approx(0.4)
    for metric in report["metrics"]:
        assert f"metric/{metric}" in report["statuses"]
    assert report["arrays"]["reference"] is None
    assert report["panels"]["reference_comparison"]["status"] == "unavailable"
    assert report["panels"]["coverage"]["status"] == "screening_only"


def test_observation_only_keeps_recovery_unavailable_without_fabricated_truth():
    result = make_result()
    report = compute_diagnostics(result, observed=[12, 13, 14])
    assert report["metrics"]["latent_total/mae"] is None
    assert report["panels"]["recovery_scatter"]["status"] == "unavailable"
    assert report["panels"]["calibration_pit"]["status"] == "unavailable"
    assert report["panels"]["target_recovery"]["status"] == "ok"
    assert report["arrays"]["total"]["truth"] is None
    assert report["arrays"]["total"]["realized_noise"] is None
    assert all(row["truth"] is None for row in report["arrays"]["components"])
    for name in ("reference_comparison", "training", "evaluation", "set_posterior"):
        assert report["panels"][name]["status"] == "unavailable"


def test_truth_alignment_and_padded_support_are_rejected():
    result = make_result()
    with pytest.raises(ValueError, match="shape"):
        compute_diagnostics(result, truth=np.zeros((8, 3)))
    bad_truth = np.zeros((3, 8))
    bad_truth[:, 2] = 1
    with pytest.raises(ValueError, match="padded"):
        compute_diagnostics(result, truth=bad_truth)
    with pytest.raises(ValueError, match="support"):
        compute_diagnostics(result, graph_truth=[1, 1, 0])
    with pytest.raises(ValueError, match="aligned"):
        replace(result, graphs=result.graphs[:-1])


def test_fixed_graph_frequency_is_labeled_as_hypothesis_not_posterior():
    result = make_result()
    result = replace(
        result,
        mode="fixed",
        graphs=np.tile([1, 0, 1], (5, 1)),
        provenance={**result.provenance, "graph_source": "supplied_hypothesis"},
    )
    report = compute_diagnostics(result)
    assert report["statuses"]["array/graphs"]["status"] == "hypothesis_conditional"
    edge_key = next(key for key in report["metrics"] if key.startswith("edge/"))
    assert report["statuses"][f"metric/{edge_key}"]["status"] == "hypothesis_conditional"
    assert (
        report["statuses"]["metric/uncertainty/mode_frequency"]["status"]
        == "hypothesis_conditional"
    )
    assert report["metrics"]["uncertainty/between_graph_variance_mean"] == 0


def test_set_approximation_cardinality_zero_truth_mismatch_and_probability_ties():
    sets, logp = cardinality_decode([0.3, 0.7], 0)
    assert sets.shape == (1, 0)
    np.testing.assert_array_equal(logp, [0])
    sets, logp = cardinality_decode([0.5, 0.5], 1)
    np.testing.assert_array_equal(sets, [[0], [1]])
    np.testing.assert_allclose(np.exp(logp), [0.5, 0.5])
    expected = np.random.default_rng(7).random()
    assert randomized_set_pit(logp, 0, seed=7) == pytest.approx(expected)
    report = factorized_set_diagnostics(
        [0.5, 0.5], 1, truth=[1, 1], source="auxiliary structure head"
    )
    assert report["metrics"]["true_set_log_score"] is None
    assert report["statuses"]["true_set_log_score"]["status"] == "unavailable"
    assert report["true_index"] is None
    assert report["map_ambiguous"] is True
    with pytest.raises(ValueError, match="finite probabilities"):
        cardinality_decode([np.nan, 1], 1)


def test_training_and_auxiliary_metrics_preserve_names_and_undefined_records():
    report = compute_diagnostics(
        make_result(),
        training_records=[{"step": 1, "loss": 4, "loss_conditional": 3, "lr": 0.01}],
        evaluation_records=[{"step": 1, "heads": {"structure_head": {"cy": {"auroc": None}}}}],
    )
    assert report["metrics"]["training/step/1/loss_conditional"] == 3
    assert report["metrics"]["evaluation/step/1/heads/structure_head/cy/auroc"] is None
    assert report["arrays"]["evaluation"]["series"]["heads/structure_head/cy/auroc"] == [
        {"step": 1, "value": None}
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_diagnostics(make_result(), training_records=[{"step": 1}, {"step": 1}])


def test_report_rerender_uses_stored_values_without_scoring(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from pymc_mfm.diagnostics import report as reporting

    result = make_result()
    truth = result.components.mean(0)
    first = build_report(result, tmp_path / "first", truth=truth, graph_truth=[1, 0, 1])
    saved = DecompositionResult.load(tmp_path / "first" / "result")
    np.testing.assert_array_equal(saved.components, result.components)

    def forbidden(*args, **kwargs):
        raise AssertionError("render must not compute diagnostic values or sample")

    monkeypatch.setattr(reporting, "compute_diagnostics", forbidden)
    second = render_report(tmp_path / "first", tmp_path / "second")
    assert second["metrics"] == first["metrics"]
    assert second["provenance"] == first["provenance"]
    assert (tmp_path / "first" / "evidence.npz").read_bytes() == (
        tmp_path / "second" / "evidence.npz"
    ).read_bytes()
    assert second["panels"]["recovery_scatter"]["render_status"] == "ok"
    assert set(second["arrays"]) >= {"components", "graphs", "uncertainty", "calibration"}

    # MLflow previews HTML from a blob URL, so relative figure paths cannot
    # resolve. Every displayed image must decode offline to its saved figure.
    class Images(HTMLParser):
        def __init__(self):
            super().__init__()
            self.figures = set()

        def handle_starttag(self, tag, attrs):
            if tag == "img":
                attrs = dict(attrs)
                assert attrs["src"].startswith("data:")
                with urlopen(attrs["src"]) as image:
                    actual = hashlib.sha256(image.read()).hexdigest()
                panel = second["panels"][attrs["alt"]]
                expected = next(
                    item["sha256"] for item in second["artifacts"] if item["path"] == panel["path"]
                )
                assert actual == expected
                self.figures.add(panel["path"])

    parsed = Images()
    parsed.feed((tmp_path / "second" / "report.html").read_text())
    assert parsed.figures == {
        item["path"] for item in second["artifacts"] if item["kind"] == "figure"
    }


def test_render_failure_is_persisted_and_strict_reports_raise(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from pymc_mfm.plotting import panels

    original = panels.plot_panel

    def fail_control_panel(name, summary):
        if name == "controls":
            raise RuntimeError("deliberate control-render failure")
        return original(name, summary)

    monkeypatch.setattr(panels, "plot_panel", fail_control_panel)
    with pytest.raises(DiagnosticError) as captured:
        build_report(make_result(), tmp_path / "strict")
    manifest = json.loads(captured.value.manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["panels"]["controls"]["status"] == "failed"
    assert manifest["panels"]["target_recovery"]["render_status"] == "ok"
    assert "deliberate control-render failure" in manifest["failures"][0]
    permissive = build_report(make_result(), tmp_path / "permissive", strict=False)
    assert permissive["status"] == "failed"
    assert permissive["statuses"]["panel/controls"]["status"] == "failed"


def test_report_rejects_tampered_evidence_and_preserves_existing_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "report"
    build_report(make_result(), output)
    with pytest.raises(FileExistsError):
        build_report(make_result(), output)
    evidence = output / "evidence.json"
    evidence.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        render_report(output, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_malformed_scoring_has_visible_status_even_without_strict_mode(tmp_path):
    report = build_report(
        make_result(), tmp_path / "bad_truth", truth=np.zeros((2, 8)), strict=False
    )
    assert report["status"] == "failed"
    assert report["metrics"] == {}
    assert report["statuses"]["scoring"]["status"] == "failed"
    assert all(panel["status"] == "failed" for panel in report["panels"].values())


def test_evaluator_captures_existing_paired_draws_in_natural_units(model, monkeypatch):
    from conftest import make_synthetic_corpus

    from pymc_mfm.training.evaluation import evaluate_rows

    corpus = make_synthetic_corpus()
    rows = [(0, 0)]
    actual = []
    original = model.sample_disentangled

    def retain(batch, n_samples, **kwargs):
        components, graphs = original(batch, n_samples, **kwargs)
        actual.append(
            (
                kwargs["window"],
                components.detach().cpu().numpy().copy(),
                graphs.detach().cpu().numpy().copy(),
                batch["sales_scale"].detach().cpu().numpy(),
            )
        )
        return components, graphs

    monkeypatch.setattr(model, "sample_disentangled", retain)
    captured = []
    record = evaluate_rows(
        model,
        [corpus],
        rows,
        n_samples=3,
        seed=9,
        device="cpu",
        leg="heldout",
        result_sink=lambda *values: captured.append(values),
    )
    assert len(actual) == 2
    assert record["result_capture"] == {"captured": 2, "failures": []}
    for result, truth, graph_truth, observed in captured:
        _, components, graphs, scale = next(item for item in actual if item[0] == result.window)
        np.testing.assert_allclose(
            result.components, components[:, 0].astype(np.float64) * float(scale[0])
        )
        np.testing.assert_array_equal(result.graphs, graphs[:, 0])
        span = (
            slice(0, result.context_length)
            if result.window == "context"
            else slice(result.context_length, None)
        )
        np.testing.assert_array_equal(observed, corpus["outcome_raw"][0][span])
        np.testing.assert_allclose(
            truth.sum(axis=-1),
            (corpus["outcome_raw"][0] - corpus["outcome_noise"][0])[span],
            rtol=1e-5,
            atol=1e-3,
        )
        assert graph_truth.shape == result.edge_mask.shape
        assert result.provenance["world_id"] == "corpus:0/row:0"


def test_explicit_graph_prior_support_is_independent_of_truth_and_padding():
    result = make_result()
    support = {"cardinality": {"cy": [1, 1], "zb": [1, 1]}, "forbidden_families": ["cc"]}
    report = graph_support_diagnostics(
        result.graphs,
        result.provenance["edge_families"],
        result.edge_mask,
        truth=[1, 0, 1],
        support=support,
    )
    assert report["metrics"]["wrong_cy_cardinality_mass"] == pytest.approx(0.4)
    assert report["metrics"]["wrong_zb_cardinality_mass"] == pytest.approx(0.6)
    assert report["metrics"]["off_prior_mass"] == pytest.approx(0.6)
    assert report["metrics"]["forbidden_family_mass"] == 0
    assert report["metrics"]["structure_exact_hit"] == 0
    assert report["metrics"]["family/cy/count_recovery"] == pytest.approx(0.6)
    assert report["metrics"]["family/zb/modal_count_hit"] == 0
    assert report["arrays"]["edge_count_histograms"]["cy"] == [2, 3]
    unknown = graph_support_diagnostics(
        result.graphs, result.provenance["edge_families"], result.edge_mask, truth=[1, 0, 1]
    )
    assert unknown["metrics"]["off_prior_mass"] is None
    assert unknown["statuses"]["off_prior_mass"]["status"] == "unavailable"
    forbidden = graph_support_diagnostics(
        result.graphs,
        result.provenance["edge_families"],
        result.edge_mask,
        support={"cardinality": {"cy": [0, 1]}, "forbidden_families": ["zb"]},
    )
    assert forbidden["metrics"]["forbidden_family_mass"] == pytest.approx(0.4)
    assert forbidden["metrics"]["off_prior_mass"] == pytest.approx(0.4)


def test_manifest_cannot_relabel_verified_result_identity(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "original"
    build_report(make_result(), output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["checkpoint_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        render_report(output, tmp_path / "relabeled")
    assert not (tmp_path / "relabeled").exists()


def test_missing_renderer_dependency_is_visible_not_an_empty_success(tmp_path, monkeypatch):
    from pymc_mfm.plotting import panels

    def absent(*args, **kwargs):
        raise ImportError("matplotlib renderer not installed")

    monkeypatch.setattr(panels, "plot_panel", absent)
    manifest = build_report(make_result(), tmp_path / "absent_renderer", strict=False)
    assert manifest["status"] == "failed"
    assert manifest["panels"]["target_recovery"]["render_status"] == "failed"
    assert manifest["panels"]["coverage"]["status"] == "unavailable"


def test_standalone_sbc_card_rerenders_saved_null_bands_without_resimulation(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from pymc_mfm.diagnostics import calibration

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
    first = write_calibration_report(card, tmp_path / "sbc")

    def forbidden(*args, **kwargs):
        raise AssertionError("rerender must not rerun rank/null simulation")

    monkeypatch.setattr(calibration, "rank_uniformity_stats", forbidden)
    second = render_report(tmp_path / "sbc", tmp_path / "sbc_replot")
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["calibration/total/ecdf_band_reject"] == 1
    assert set(second["panels"]) == {"ranks", "ecdf", "coverage"}
    assert all(panel["render_status"] == "ok" for panel in second["panels"].values())


def test_unstepped_evidence_uses_record_indices_not_fabricated_training_steps():
    report = compute_diagnostics(make_result(), evaluation_records=[{"score": 3}, {"score": 2}])
    series = report["arrays"]["evaluation"]
    assert series["coordinate"] == "record_index"
    assert series["steps"] is None
    assert series["series"]["score"] == [
        {"step": None, "record_index": 0, "value": 3},
        {"step": None, "record_index": 1, "value": 2},
    ]
    assert report["metrics"]["evaluation/record_index/1/score"] == 2
