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


def test_relative_diagnostics_use_the_recorded_context_ruler():
    result = make_result()
    truth = result.components.mean(axis=0)
    truth[:, 0] += 0.75
    truth[:, 3] += 0.25
    observed = truth.sum(axis=-1) + [0.5, -0.2, 0.1]
    reference = compute_diagnostics(result, truth=truth, observed=observed)["metrics"]
    for units in (1e-100, 1e100):
        converted = replace(
            result,
            components=result.components * units,
            provenance={**result.provenance, "sales_scale": 10 * units},
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            actual = compute_diagnostics(converted, truth=truth * units, observed=observed * units)[
                "metrics"
            ]
        for name in (
            "closure/latent_relative_error",
            "closure/draw_relative_error",
            "closure/draw_q90_relative_error",
            "share/indirect_share_mae",
            "share/control_share_mae",
            "share/active_control_share_mae",
            "latent_total/relative_mae",
            "latent_total/scaled_mae",
            "latent_total/var_ratio",
            "latent_total/r2",
        ):
            assert actual[name] == pytest.approx(reference[name], rel=1e-10, abs=1e-12), name


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
    assert second["panels"]["dag_recovery"]["render_status"] == "ok"
    graph_path = tmp_path / "second" / second["panels"]["dag_recovery"]["path"]
    assert graph_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
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
        assert result.provenance["head"] == "conditional"


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


EFFECT_NAMES = (
    "baseline",
    "channel:a",
    "channel:b",
    "control:w",
    "latent",
    "indirect:cc",
    "indirect:zc",
    "indirect:dc",
)


def effect_world(draws, *, scale=2.0):
    """Channels ``a`` and ``b`` (``cy`` edges) and control ``w`` (``zb``) in one query window.

    ``draws`` maps component names to ``[S, T]`` draws; the others are zero.
    Every component and edge is active and the sales scale is ``scale``.
    """
    shape = np.shape(next(iter(draws.values())))
    components = np.zeros((*shape, len(EFFECT_NAMES)))
    for name, values in draws.items():
        components[..., EFFECT_NAMES.index(name)] = values
    provenance = {
        "data_sha256": "c" * 64,
        "edge_endpoints": [
            ["channel:a", "sales"],
            ["channel:b", "sales"],
            ["control:w", "outcome"],
        ],
        "edge_families": ["cy", "cy", "zb"],
        "synthetic_training_evaluation": True,
    }
    if scale is not None:
        provenance["sales_scale"] = scale
    return DecompositionResult(
        components=components,
        graphs=np.ones((shape[0], 3), dtype=int),
        component_names=EFFECT_NAMES,
        component_families=tuple(name.split(":")[0] for name in EFFECT_NAMES),
        component_mask=np.ones(len(EFFECT_NAMES), dtype=bool),
        edge_names=("cy:channel:a->sales", "cy:channel:b->sales", "zb:control:w->outcome"),
        edge_mask=np.ones(3, dtype=bool),
        channel_names=("a", "b"),
        control_names=("w",),
        time=np.arange(shape[1]) + 10,
        window="query",
        mode="model_average",
        units="revenue",
        context_length=10,
        provenance=provenance,
    )


def effect_truth(truth, weeks):
    """``[T, 8]`` component truth from ``{name: [T]}``; the others are zero."""
    array = np.zeros((weeks, len(EFFECT_NAMES)))
    for name, values in truth.items():
        array[:, EFFECT_NAMES.index(name)] = values
    return array


ATTRIBUTION_DRAWS = {
    "baseline": [[10, 10], [12, 12]],
    "channel:a": [[2, 4], [4, 8]],
    "channel:b": [[0, 2], [0, -2]],
    "control:w": [[1, 1], [3, 3]],
}
ATTRIBUTION_TRUTH = {"baseline": [11, 11], "channel:a": [2, 6], "control:w": [2, 2]}


def test_live_dead_attribution_scores_each_effect_in_sales_scale_units():
    result = effect_world(ATTRIBUTION_DRAWS)
    truth = effect_truth(ATTRIBUTION_TRUTH, 2)
    report = compute_diagnostics(result, truth=truth, graph_truth=[1, 0, 1])
    # Sales scale 2. Live a: posterior-mean errors (0.5, 0) give bias 0.25 and
    # demeaned RMSE 0.25; weekly CRPS 0.25 and 0.5. Dead b: draws (0, 1) and
    # (0, -1) cancel in the posterior mean but not per draw. Live w: CRPS 0.25.
    expected = {
        "attribution/live_media/crps": 0.375,
        "attribution/live_media/mean_bias": 0.25,
        "attribution/live_media/mean_abs_error": 0.25,
        "attribution/live_media/demeaned_rmse": 0.25,
        "attribution/dead_media/leakage_scaled_mae": 0.0,
        "attribution/dead_media/draw_leakage_scaled_mae": 0.5,
        "attribution/live_control/crps": 0.25,
        "attribution/live_control/mean_bias": 0.0,
        "attribution/live_control/mean_abs_error": 0.0,
        "attribution/live_control/demeaned_rmse": 0.0,
    }
    assert {key: report["metrics"][key] for key in expected} == pytest.approx(expected, abs=1e-15)
    for key in ("leakage_scaled_mae", "draw_leakage_scaled_mae"):
        name = f"attribution/dead_control/{key}"
        assert report["metrics"][name] is None
        assert report["statuses"][f"metric/{name}"]["reason"] == "no_dead_effects"
    media = report["arrays"]["attribution"]["effects"]["media"]["effects"]
    assert [(entry["name"], entry["live"]) for entry in media] == [
        ("channel:a", True),
        ("channel:b", False),
    ]
    assert media[0]["crps"] == pytest.approx(0.375)
    assert media[1]["draw_leakage_scaled_mae"] == pytest.approx(0.5)
    # Swapped graph truth makes b the live channel and a the dead one.
    swapped = compute_diagnostics(result, truth=truth, graph_truth=[0, 1, 1])["metrics"]
    assert swapped["attribution/live_media/crps"] == pytest.approx(0.25)
    assert swapped["attribution/live_media/mean_bias"] == pytest.approx(0.0)
    assert swapped["attribution/dead_media/leakage_scaled_mae"] == pytest.approx(0.25)
    assert swapped["attribution/dead_media/draw_leakage_scaled_mae"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("evidence", "unavailable"),
    [
        pytest.param(
            {},
            {"attribution", "window_totals", "cumulative", "temporal", "reliability"},
            id="nothing",
        ),
        pytest.param({"truth": True}, {"attribution", "reliability"}, id="truth-only"),
        pytest.param(
            {"graph_truth": True},
            {"attribution", "window_totals", "cumulative", "temporal"},
            id="graph-truth-only",
        ),
    ],
)
def test_missing_evidence_marks_new_panels_unavailable(evidence, unavailable):
    result = effect_world(ATTRIBUTION_DRAWS)
    supplied = {
        "truth": effect_truth(ATTRIBUTION_TRUTH, 2),
        "graph_truth": [1, 0, 1],
    }
    report = compute_diagnostics(result, **{name: supplied[name] for name in evidence})
    for panel in ("attribution", "window_totals", "cumulative", "temporal", "reliability"):
        status = report["panels"][panel]["status"]
        assert status == ("unavailable" if panel in unavailable else "ok"), panel


def test_live_dead_scores_need_graph_truth_and_sales_scale_scores_need_the_scale():
    truth = effect_truth(ATTRIBUTION_TRUTH, 2)
    unlabeled = compute_diagnostics(effect_world(ATTRIBUTION_DRAWS), truth=truth)
    # Coverage of all channels needs no graph truth; its live/dead split does.
    assert unlabeled["metrics"]["timeavg/channel/cov80"] is not None
    for name in ("timeavg/channel/cov80_live", "attribution/live_media/crps"):
        assert unlabeled["metrics"][name] is None, name
    unscaled = compute_diagnostics(
        effect_world(ATTRIBUTION_DRAWS, scale=None), truth=truth, graph_truth=[1, 0, 1]
    )
    for name in ("attribution/live_media/crps", "cumulative/channel/energy_score"):
        assert unscaled["metrics"][name] is None, name
    assert unscaled["metrics"]["timeavg/channel/cov80_live"] == 1.0


# Channel a follows its truth in one draw and reverses it in the other; b is
# dead (zero truth) with one alternating and one constant draw.
TEMPORAL_DRAWS = {
    "baseline": [[10, 12, 14, 16], [10, 12, 14, 16]],
    "channel:a": [[1, 3, 2, 4], [4, 3, 2, 1]],
    "channel:b": [[0, 1, 0, 1], [1, 1, 1, 1]],
}
TEMPORAL_TRUTH = {"baseline": [10, 12, 14, 16], "channel:a": [1, 3, 2, 4]}


def test_temporal_coherence_uses_lag_one_pairs_within_each_family():
    report = compute_diagnostics(
        effect_world(TEMPORAL_DRAWS), truth=effect_truth(TEMPORAL_TRUTH, 4), graph_truth=[1, 0, 1]
    )
    metrics = report["metrics"]
    # corr((1, 3, 2), (3, 2, 4)) = -0.5; the reversed draw has lag-1 correlation 1,
    # b's alternating draw -1. Constant series (b's truth, its second draw, a's
    # zero residual) have no correlation and drop out of the slot means.
    expected = {
        "truth_lag1": -0.5,
        "draw_lag1": (0.25 + -1) / 2,
        "lag1_tracking_abs_error": 0.75,
        "truth_error_residual_lag1": (0.5 + -1) / 2,
        "posterior_centered_residual_lag1": (0.5 + -1) / 2,
    }
    for key, value in expected.items():
        assert metrics[f"temporal/channel/{key}"] == pytest.approx(value), key
    # With two draws every adjacent-week pair has a constant side.
    assert metrics["temporal/channel/adjacent_draw_correlation"] is None
    reason = report["statuses"]["metric/temporal/channel/adjacent_draw_correlation"]["reason"]
    assert reason == "constant_series"
    slots = {
        slot["name"]: slot["values"]
        for record in report["arrays"]["temporal"]["families"]
        if record["family"] == "channel"
        for slot in record["slots"]
    }
    assert slots["channel:a"]["draw_lag1"] == pytest.approx(0.25)
    assert slots["channel:b"]["truth_lag1"] is None
    # The baseline draws equal its linear truth: every lag-1 correlation is 1.
    assert metrics["temporal/baseline/truth_lag1"] == pytest.approx(1.0)
    assert metrics["temporal/baseline/lag1_tracking_abs_error"] == pytest.approx(0.0)
    assert {record["family"] for record in report["arrays"]["temporal"]["families"]} == {
        "baseline",
        "channel",
        "control",
        "latent_total",
        "indirect",
    }


def test_window_totals_cover_window_means_and_score_running_totals():
    observed = np.array([10.0, 20.0, 16.0, 20.0])
    report = compute_diagnostics(
        effect_world(TEMPORAL_DRAWS),
        truth=effect_truth(TEMPORAL_TRUTH, 4),
        graph_truth=[1, 0, 1],
        observed=observed,
    )
    metrics = report["metrics"]
    # a: both draws average 2.5, as does its truth (covered); b: draw means
    # 0.5 and 1 give the interval [0.55, 0.95], which misses its zero truth.
    assert metrics["timeavg/channel/cov80"] == 0.5
    assert metrics["timeavg/channel/cov80_live"] == 1.0
    assert metrics["timeavg/channel/cov80_dead"] == 0.0
    simultaneous = "window_totals/simultaneous_cov80"
    assert metrics[simultaneous] is None
    assert report["statuses"][f"metric/{simultaneous}"] == {
        "status": "unavailable",
        "reason": "fewer_than_two_live_effects",
    }
    channels = report["arrays"]["window_totals"]["channels"]
    covered = {entry["name"]: entry["covered"] for entry in channels}
    assert covered == {"channel:a": True, "channel:b": False}
    # Running-total errors of the two draws have squared norms 6 and 57 and
    # differ by 37; sales scale 2 and 8 coordinates.
    energy = (np.sqrt(6) + np.sqrt(57) - np.sqrt(37)) / (4 * np.sqrt(8))
    assert metrics["cumulative/channel/energy_score"] == pytest.approx(energy)
    running = report["arrays"]["cumulative"]["channels"][0]
    assert running["mean"] == pytest.approx([2.5, 5.5, 7.5, 10.0])
    assert running["truth"] == pytest.approx([1, 4, 6, 10])
    # Posterior-mean latent total (13, 16, 16.5, 19.5) against the structural
    # (11, 15, 16, 20): residual sum of squares 5.5, total 41.
    assert metrics["latent_total/r2"] == pytest.approx(1 - 5.5 / 41)
    # Draw totals (11, 16, 16, 21) and (15, 16, 17, 18) relative to the observed.
    per_draw = [np.mean([0, 1, 0, 1] / observed), np.mean([4, 1, 1, 2] / observed)]
    assert metrics["closure/draw_relative_error"] == pytest.approx(np.mean(per_draw))
    assert metrics["closure/draw_q90_relative_error"] == pytest.approx(
        per_draw[0] + 0.9 * (per_draw[1] - per_draw[0])
    )
    assert metrics["closure/latent_relative_error"] == pytest.approx(
        np.mean(np.array([2, 1, 0.5, 0.5]) / observed)
    )


JOINT_COVERAGE_DRAWS = {
    "baseline": [[10, 10]] * 4,
    "channel:a": [[-1, -1], [-1, -1], [1, 1], [1, 1]],
    "channel:b": [[-100, -100], [100, 100], [-100, -100], [100, 100]],
}
JOINT_COVERAGE_TRUTH = {"baseline": [10, 10], "channel:a": [2, 2], "channel:b": [1, 1]}


@pytest.mark.parametrize(
    ("a_truth", "joint", "pointwise_live"),
    [(2.0, 0.0, 0.5), (1.0, 1.0, 1.0)],
)
def test_simultaneous_window_coverage_is_the_studentized_live_family_event(
    a_truth, joint, pointwise_live
):
    from pymc_mfm.training.metrics import simultaneous_coverage

    result = effect_world(JOINT_COVERAGE_DRAWS, scale=None)
    truth = effect_truth({**JOINT_COVERAGE_TRUTH, "channel:a": [a_truth, a_truth]}, 2)
    report = compute_diagnostics(result, truth=truth, graph_truth=[1, 1, 1])
    name = "window_totals/simultaneous_cov80"
    # Draw time means are the four corners (+/-1, +/-100), so population SDs
    # are (1, 100) and every draw's standardized maximum is exactly 1.
    # Truth (2, 1) misses jointly despite the broad second-channel interval;
    # (1, 1) sits on the inclusive first-channel boundary and is covered.
    assert report["metrics"][name] == joint
    assert report["metrics"]["timeavg/channel/cov80_live"] == pointwise_live
    assert (
        report["metrics"][name],
        report["statuses"][f"metric/{name}"]["reason"],
    ) == simultaneous_coverage(result.components[..., [1, 2]], truth[:, [1, 2]])
    assert report["statuses"][f"metric/{name}"]["status"] == "ok"
    assert report["metric_units"][name] == "dimensionless"
    assert "timeavg/channel/simcov80_live" not in report["metrics"]


def test_simultaneous_window_coverage_excludes_dead_and_padded_channels():
    base = effect_world(JOINT_COVERAGE_DRAWS, scale=None)
    extra = np.zeros((*base.components.shape[:2], 2))
    extra[..., 0] = 10000
    result = replace(
        base,
        components=np.concatenate((base.components, extra), axis=-1),
        graphs=np.column_stack((base.graphs, np.zeros((base.n_samples, 2), dtype=int))),
        component_names=(*base.component_names, "channel:dead", "channel:padding"),
        component_families=(*base.component_families, "channel", "channel"),
        component_mask=np.r_[base.component_mask, True, False],
        edge_names=(*base.edge_names, "cy:channel:dead->sales", "cy:channel:padding->sales"),
        edge_mask=np.r_[base.edge_mask, True, False],
        channel_names=(*base.channel_names, "dead", "padding"),
        provenance={
            **base.provenance,
            "edge_endpoints": [
                *base.provenance["edge_endpoints"],
                ["channel:dead", "sales"],
                ["channel:padding", "sales"],
            ],
            "edge_families": [*base.provenance["edge_families"], "cy", "cy"],
        },
    )
    truth = np.column_stack(
        (
            effect_truth({**JOINT_COVERAGE_TRUTH, "channel:a": [1, 1], "channel:b": [0, 0]}, 2),
            np.zeros((2, 2)),
        )
    )
    report = compute_diagnostics(result, truth=truth, graph_truth=[1, 1, 1, 0, 0])
    name = "window_totals/simultaneous_cov80"
    # b is live even though its time-mean truth is zero. Dead and padded
    # channels have zero posterior SD and must not invalidate the live event.
    assert report["metrics"][name] == 1.0
    assert report["metrics"]["timeavg/channel/cov80_live"] == 1.0
    assert report["metrics"]["timeavg/channel/cov80_dead"] == 0.0
    assert report["metrics"]["timeavg/channel/cov80"] == pytest.approx(2 / 3)
    assert [entry["name"] for entry in report["arrays"]["window_totals"]["channels"]] == [
        "channel:a",
        "channel:b",
        "channel:dead",
    ]


@pytest.mark.parametrize(
    ("graph_truth", "truth_supplied", "constant_effect", "reason"),
    [
        pytest.param([0, 0, 1], True, False, "fewer_than_two_live_effects", id="no-live"),
        pytest.param([1, 1, 1], True, True, "zero_posterior_sd", id="constant-live"),
        pytest.param(None, True, False, "graph truth not supplied", id="no-graph-truth"),
        pytest.param([1, 1, 1], False, False, "simulation truth not supplied", id="no-truth"),
    ],
)
def test_simultaneous_window_coverage_preserves_unavailable_statuses(
    graph_truth, truth_supplied, constant_effect, reason
):
    draws = {
        **JOINT_COVERAGE_DRAWS,
        **({"channel:b": [[0, 0]] * 4} if constant_effect else {}),
    }
    report = compute_diagnostics(
        effect_world(draws, scale=None),
        truth=effect_truth(JOINT_COVERAGE_TRUTH, 2) if truth_supplied else None,
        graph_truth=graph_truth,
    )
    name = "window_totals/simultaneous_cov80"
    assert report["metrics"][name] is None
    assert report["metric_units"][name] == "dimensionless"
    assert report["statuses"][f"metric/{name}"] == {"status": "unavailable", "reason": reason}
    assert report["panels"]["window_totals"]["status"] == (
        "ok" if truth_supplied else "unavailable"
    )


def test_window_totals_panel_displays_joint_miss_beside_pointwise_share():
    pytest.importorskip("matplotlib")
    from pymc_mfm.plotting.panels import plot_window_totals

    report = compute_diagnostics(
        effect_world(JOINT_COVERAGE_DRAWS, scale=None),
        truth=effect_truth(JOINT_COVERAGE_TRUTH, 2),
        graph_truth=[1, 1, 1],
    )
    figure = plot_window_totals(report)
    figure.canvas.draw()
    rates = figure.axes[1]
    bars = {bar.get_gid(): bar.get_height() for bar in rates.patches}
    assert bars == {
        "timeavg/channel/cov80": 0.5,
        "timeavg/channel/cov80_live": 0.5,
        "window_totals/simultaneous_cov80": 0.0,
    }
    nominal = next(line for line in rates.lines if len(line.get_ydata()) == 2)
    np.testing.assert_array_equal(nominal.get_ydata(), [0.8, 0.8])


@pytest.mark.parametrize(
    ("graph_truth", "constant_effect", "reason"),
    [
        pytest.param([1, 0, 1], False, "fewer_than_two_live_effects", id="one-live"),
        pytest.param([1, 1, 1], True, "zero_posterior_sd", id="constant-live"),
        pytest.param(None, False, "graph truth not supplied", id="no-graph-truth"),
    ],
)
def test_window_totals_panel_marks_joint_unavailability_with_its_reason(
    graph_truth, constant_effect, reason
):
    pytest.importorskip("matplotlib")
    from pymc_mfm.plotting.panels import plot_window_totals

    draws = {
        **JOINT_COVERAGE_DRAWS,
        **({"channel:b": [[0, 0]] * 4} if constant_effect else {}),
    }
    report = compute_diagnostics(
        effect_world(draws, scale=None),
        truth=effect_truth(JOINT_COVERAGE_TRUTH, 2),
        graph_truth=graph_truth,
    )
    figure = plot_window_totals(report)
    figure.canvas.draw()
    rates = figure.axes[1]
    assert "window_totals/simultaneous_cov80" not in {bar.get_gid() for bar in rates.patches}
    joint_na = next(text for text in rates.texts if text.get_position()[0] == 3)
    assert joint_na.get_text() == "n/a"
    legend = "".join(text.get_text().replace("\n", "") for text in rates.get_legend().get_texts())
    assert reason in legend


def test_legacy_window_totals_summary_retains_pointwise_bars_without_rescoring(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    from pymc_mfm.diagnostics import core
    from pymc_mfm.plotting.panels import plot_window_totals
    from pymc_mfm.training import metrics as scoring

    summary = compute_diagnostics(
        effect_world(TEMPORAL_DRAWS),
        truth=effect_truth(TEMPORAL_TRUTH, 4),
        graph_truth=[1, 0, 1],
    )
    name = "window_totals/simultaneous_cov80"
    del summary["metrics"][name]
    del summary["metric_units"][name]
    del summary["statuses"][f"metric/{name}"]
    saved = tmp_path / "summary.json"
    saved.write_text(json.dumps(summary, allow_nan=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy summaries must not be rescored")

    monkeypatch.setattr(core, "compute_diagnostics", forbidden)
    monkeypatch.setattr(scoring, "time_average_scores", forbidden)
    monkeypatch.setattr(scoring, "simultaneous_coverage", forbidden)
    stored = json.loads(saved.read_text())
    figure = plot_window_totals(stored)
    figure.canvas.draw()
    rates = figure.axes[1]
    assert {bar.get_gid(): bar.get_height() for bar in rates.patches} == {
        "timeavg/channel/cov80": 0.5,
        "timeavg/channel/cov80_live": 1.0,
        "timeavg/channel/cov80_dead": 0.0,
    }
    joint_na = next(text for text in rates.texts if text.get_position()[0] == 3)
    assert joint_na.get_text() == "n/a"
    legend = " ".join(text.get_text() for text in rates.get_legend().get_texts())
    assert "not recorded" in " ".join(legend.split())
    assert name not in stored["metrics"]


def test_share_errors_divide_group_totals_by_the_outcome_magnitude():
    result = effect_world(
        {"indirect:cc": [[1, 1], [3, 3]], "control:w": [[3, 3], [3, 3]], "baseline": [[5, 5]] * 2}
    )
    truth = effect_truth({"indirect:cc": [1, 1], "control:w": [1, 1], "baseline": [5, 5]}, 2)
    metrics = compute_diagnostics(result, truth=truth, observed=[4.0, 8.0])["metrics"]
    # The indirect total is off by 1 each week, (1/4 + 1/8) / 2; the control
    # total by 2, (2/4 + 2/8) / 2, over its declared and its present slots alike.
    expected = {
        "indirect_share_mae": 0.1875,
        "control_share_mae": 0.375,
        "active_control_share_mae": 0.375,
    }
    for name, value in expected.items():
        assert metrics[f"share/{name}"] == pytest.approx(value), name
    # Without observations the structural outcome (7, 7) is the magnitude.
    unobserved = compute_diagnostics(result, truth=truth)["metrics"]
    assert unobserved["share/indirect_share_mae"] == pytest.approx(1 / 7)
    assert unobserved["share/control_share_mae"] == pytest.approx(2 / 7)


def test_graph_metrics_named_like_evaluator_metrics_carry_their_units():
    from pymc_mfm.training.evaluation import METRIC_UNITS

    result = make_result()
    units = compute_diagnostics(result, graph_truth=np.asarray(result.graphs[0]))["metric_units"]
    shared = {
        key: METRIC_UNITS[key.split("/", 1)[1]]
        for key in units
        if key.startswith("graph/") and key.split("/", 1)[1] in METRIC_UNITS
    }
    assert {"graph/log_score", "graph/brier", "graph/shd_mean", "graph/entropy"} <= set(shared)
    assert {key: units[key] for key in shared} == shared


def test_counts_carry_the_count_unit_with_and_without_truth():
    result = make_result()
    truth = result.components.mean(axis=0)
    counts = {
        "uncertainty/n_unique_graphs",
        "uncertainty/n_rare_groups",
        "uncertainty/n_singleton_groups",
        "graph/cardinality_mean",
        "graph/cardinality_sd",
        "graph/cardinality_absolute_error",
        *(
            f"graph_support/family/{family}/{key}"
            for family in ("cy", "zb")
            for key in ("mean_count", "modal_count", "true_count", "count_mae")
        ),
    }
    for evidence in ({}, {"truth": truth, "graph_truth": np.asarray(result.graphs[0])}):
        units = compute_diagnostics(result, **evidence)["metric_units"]
        # Rank counts exist only with truth, one per calibrated family.
        ranks = {
            key for key in units if key.startswith("calibration/") and key.endswith("/n_ranks")
        }
        assert bool(ranks) == ("truth" in evidence)
        assert {key: units[key] for key in counts | ranks} == dict.fromkeys(counts | ranks, "count")
        # Masses, rates and coverages next to those counts stay dimensionless.
        for key in ("uncertainty/rare_draw_mass", "graph_support/off_prior_mass"):
            assert units[key] == "dimensionless", key
        if "truth" in evidence:
            for key in (
                "graph_support/family/cy/count_recovery",
                "calibration/baseline/chi2_pvalue",
            ):
                assert units[key] == "dimensionless", key
