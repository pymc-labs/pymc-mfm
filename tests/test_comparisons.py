"""Scientific comparison contracts, identity boundaries and saved-only reporting."""

import hashlib
import json
import shutil
from dataclasses import replace

import numpy as np
import pytest

from pymc_mfm.diagnostics import comparisons
from pymc_mfm.diagnostics.comparisons import (
    ComparisonRecord,
    build_comparison_report,
    c2st_from_predictions,
    comparison_summary,
    input_support_summary,
    load_comparison_records,
    matched_omission,
    posterior_sbc_from_draws,
    reference_diagnostics,
    reference_fit_health,
    render_comparison_report,
    run_c2st,
    score_record,
)
from pymc_mfm.inference.result import DecompositionResult


def record(*, method="A", case="base", world="0", draws=None, names=None, families=None, **kwargs):
    names = ("baseline", "channel:a", "control:z") if names is None else tuple(names)
    families = ("baseline", "channel", "control") if families is None else tuple(families)
    draws = (
        np.tile([10.0, 2.0, 1.0], (4, 2, 1)) if draws is None else np.asarray(draws, dtype=float)
    )
    edges = tuple(
        f"{('cy' if family == 'channel' else 'zb')}:{name}"
        for name, family in zip(names, families)
        if family in ("channel", "control")
    )
    if not edges:
        edges = ("db:latent",)
    result = DecompositionResult(
        components=draws,
        graphs=np.ones((len(draws), len(edges))),
        component_names=names,
        component_families=families,
        component_mask=np.ones(len(names), dtype=bool),
        edge_names=edges,
        edge_mask=np.ones(len(edges), dtype=bool),
        channel_names=tuple(name for name, family in zip(names, families) if family == "channel"),
        control_names=tuple(name for name, family in zip(names, families) if family == "control"),
        time=np.arange(draws.shape[1]) + 2,
        window="query",
        mode="model_average",
        units="outcome",
        context_length=2,
        provenance={
            "data_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "inference_config": {"samples": len(draws)},
            "edge_families": [name.split(":")[0] for name in edges],
            "synthetic_training_evaluation": True,
            "source_step": 12,
            "data_split": "holdout",
        },
    )
    truth = kwargs.pop("truth", draws.mean(axis=0))
    live = kwargs.pop("live", np.ones(len(names), dtype=bool))
    return ComparisonRecord(
        case=case,
        world=world,
        method=method,
        result=result,
        kind=kwargs.pop("kind", "model"),
        truth=truth,
        live=live,
        **kwargs,
    )


def reference(result, **kwargs):
    return {
        "label": "Cached reference",
        "components": result.components,
        "component_names": result.component_names,
        "time": result.time,
        "window": result.window,
        "units": result.units,
        "provenance": {"data_sha256": result.provenance["data_sha256"]},
        **kwargs,
    }


def test_window_mean_preserves_within_draw_dependence_and_zero_path_definition():
    draws = np.array([[[0.0], [2.0]], [[2.0], [0.0]]])
    item = record(draws=draws, names=("control:z",), families=("control",), truth=np.ones((2, 1)))
    scored = score_record(item)
    row = scored["components"][0]
    assert row["window_mean"]["width"] == 0
    assert row["window_mean"]["crps"] == 0
    assert row["weekly"]["width"] == pytest.approx(1.6)
    cancellation = replace(
        item, result=replace(item.result, components=draws - 1), truth=np.zeros((2, 1))
    )
    scored = score_record(cancellation)
    assert scored["components"][0]["window_mean"]["zero_mass"] == 1
    assert scored["components"][0]["zero_path_mass"] == 0
    assert scored["families"]["control"]["false_exclusions"] == 0
    assert scored["recovered"] is True


def omission_pair():
    names = ("baseline", "channel:a", "channel:b")
    families = ("baseline", "channel", "channel")
    native_truth = np.tile([10.0, 2.0, 5.0], (4, 1))
    noise = np.array([0.1, -0.1, 0.5, -0.5])
    evidence = {
        "time": np.arange(4),
        "native_truth": native_truth,
        "noise": noise,
        "observed": native_truth.sum(axis=1) + noise,
        "native_inputs": np.tile([20.0, 40.0], (4, 1)),
    }
    shared = {
        "world_evidence": evidence,
        "input_names": names[1:],
        "truth_names": names,
        "observed": evidence["observed"][2:],
        "latent_truth": native_truth[2:].sum(axis=1),
    }
    before = record(
        draws=np.tile([10.0, 4.0, 105.0], (4, 2, 1)),
        names=names,
        families=families,
        truth=native_truth[2:],
        visible_inputs=names[1:],
        **shared,
    )
    after = record(
        case="omitted",
        draws=np.tile([10.0, 3.0], (4, 2, 1)),
        names=names[:2],
        families=families[:2],
        truth=native_truth[2:, :2],
        visible_inputs=("channel:a",),
        omitted_input="channel:b",
        **shared,
    )
    return before, after


def test_omission_scores_same_retained_ids_not_differently_sized_family_averages():
    before, after = omission_pair()
    scores = matched_omission(before, after)
    assert len(scores) == 1
    assert scores[0]["component_id"] == "channel:a"
    assert scores[0]["delta_crps"] == -1
    summary = comparison_summary(
        [before, after], omission_pairs=[("base", "omitted")], bootstrap_samples=20
    )
    assert summary["omission_summary"][0]["mean"] == -1
    assert summary["records"][1]["total"]["window_mean"]["truth"] == 17


def test_omission_rejects_retained_hidden_component_and_unbound_truth():
    before, after = omission_pair()
    retained_hidden = replace(
        before, case="omitted", visible_inputs=("channel:a",), omitted_input="channel:b"
    )
    with pytest.raises(ValueError, match="retained visible"):
        matched_omission(before, retained_hidden)
    changed_before = replace(before, truth=before.truth + 1)
    changed_after = replace(after, truth=after.truth + 1)
    with pytest.raises(ValueError, match="bound to original"):
        matched_omission(changed_before, changed_after)
    changed_observed = before.observed + 1
    with pytest.raises(ValueError, match="observations.*original"):
        matched_omission(
            replace(before, observed=changed_observed), replace(after, observed=changed_observed)
        )


@pytest.mark.parametrize(
    "change",
    ["raw_input", "scale", "checkpoint", "configuration", "missing_checkpoint", "aggregate_only"],
)
def test_omission_rejects_changes_to_original_world_or_inference(change):
    before, after = omission_pair()
    if change == "raw_input":
        evidence = dict(
            after.world_evidence, native_inputs=after.world_evidence["native_inputs"] + 1
        )
        after = replace(after, world_evidence=evidence)
    elif change == "scale":
        after = replace(after, scale=2)
    elif change == "aggregate_only":
        before, after = replace(before, world_evidence={}), replace(after, world_evidence={})
    else:
        provenance = dict(after.result.provenance)
        if change == "checkpoint":
            provenance["checkpoint_sha256"] = "c" * 64
        elif change == "configuration":
            provenance["inference_config"] = {"samples": 8}
        else:
            provenance.pop("checkpoint_sha256")
        after = replace(after, result=replace(after.result, provenance=provenance))
    with pytest.raises(ValueError):
        matched_omission(before, after)


def test_paired_worlds_cannot_disappear_when_truth_or_support_is_missing():
    a0, b0 = record(), record(method="B")
    a1, b1 = record(world="1"), record(method="B", world="1", truth=None)
    with pytest.raises(ValueError, match="asymmetric truth"):
        comparison_summary([a0, b0, a1, b1], method_pairs=[("A", "B")], bootstrap_samples=20)
    with pytest.raises(ValueError, match="asymmetric live"):
        comparison_summary(
            [a0, replace(b0, live=None)], method_pairs=[("A", "B")], bootstrap_samples=20
        )
    with pytest.raises(ValueError, match="exactly matched"):
        comparison_summary([a0, b0, a1], method_pairs=[("A", "B")], bootstrap_samples=20)


def test_full_graph_mode_recovery_is_not_draw_hit_or_total_edge_count():
    item = record()
    graphs = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])
    item = replace(item, result=replace(item.result, graphs=graphs), graph_truth=np.array([1, 0]))
    structure = score_record(item)["structure"]["metrics"]
    assert structure["structure_exact_hit"] == 0
    assert structure["family/cy/modal_count_hit"] == 0
    assert structure["family/zb/modal_count_hit"] == 0
    assert structure["family/cy/count_recovery"] == 0.25


def test_five_control_exclusions_require_five_live_and_more_than_half_zero_paths():
    names = tuple(f"control:{i}" for i in range(5))
    draws = np.ones((4, 2, 5))
    draws[:2, :, 0] = 0
    draws[:3, :, 1] = 0
    item = record(draws=draws, names=names, families=("control",) * 5, truth=np.ones((2, 5)))
    item = replace(item, control_edges=item.result.edge_names)
    scored = score_record(item)
    assert scored["five_control_false_exclusions"] == 1
    assert scored["five_control_modal_count_hit"] is True
    assert (
        score_record(replace(item, live=np.array([1, 1, 1, 1, 0], dtype=bool)))[
            "five_control_false_exclusions"
        ]
        is None
    )


def test_reference_health_and_zero_width_are_explicit_not_dropped():
    good = {
        "divergences": 0,
        "max_rhat": 1.0,
        "min_ess_bulk": 500,
        "min_ess_tail": 500,
        "tree_depth_hits": 0,
        "min_bfmi": 0.5,
    }
    assert reference_fit_health(good)["healthy"] is True
    assert reference_fit_health(dict(good, min_bfmi=0.1))["healthy"] is False
    assert reference_fit_health(dict(good, max_rhat=float("nan")))["healthy"] is None
    item = record(kind="reference")
    assert score_record(item)["reference_health"]["status"] == "unavailable"
    assert score_record(replace(item, kind="model"))["reference_health"] is None
    summary = reference_diagnostics(
        item.result, reference(item.result, health=dict(good, divergences=3))
    )
    assert summary["metrics"]["width_ratio"] is None
    assert summary["reference_health"]["healthy"] is False
    assert summary["arrays"]["reference"]["mean"][0] == [10, 2, 1]


def test_c2st_uses_paired_world_null_and_refuses_train_test_leakage():
    args = {
        "labels": [0, 0, 1, 1],
        "probabilities": [0.1, 0.1, 0.9, 0.9],
        "world_ids": ["a", "b", "a", "b"],
        "training_world_ids": ["train"],
        "classifier": "frozen",
    }
    scored = c2st_from_predictions(**args)
    assert scored["accuracy"] == 1
    assert scored["p_value"] == pytest.approx(0.5)
    assert scored["cross_entropy"] == pytest.approx(-np.log(0.9))
    with pytest.raises(ValueError, match="disjoint"):
        c2st_from_predictions(**dict(args, training_world_ids=["a"]))
    with pytest.raises(ValueError, match="one reference/approximate pair"):
        c2st_from_predictions(**dict(args, world_ids=["a", "b", "a", "c"]))


def test_c2st_detects_shift_and_retains_disjoint_world_split():
    pytest.importorskip("sklearn")
    result = run_c2st(
        np.full((32, 2), -10.0),
        np.full((32, 2), 10.0),
        world_ids=[str(i) for i in range(32)],
        classifier="logistic",
        seed=7,
    )
    assert result["accuracy"] == 1
    assert result["p_value"] < 0.01
    evidence = result["prediction_evidence"]
    assert not set(evidence["world_ids"]) & set(evidence["training_world_ids"])


def posterior_evidence(*, data_hash="a" * 64):
    return {
        "posterior_draws": np.zeros((256, 1)),
        "augmented_draws": np.zeros((256, 4, 1)),
        "query_indices": np.zeros(256, dtype=int),
        "quantity_names": ["control:z"],
        "provenance": {
            "observed_data_sha256": data_hash,
            "replicate_contract": "valid_sequential_continuation",
            "joint_draws": True,
        },
    }


def test_posterior_sbc_zero_atoms_include_both_tie_endpoints_and_bind_data():
    evidence = posterior_evidence()
    result = posterior_sbc_from_draws(**evidence, seed=4)
    assert set(result["ranks"]["control:z"]) == {0, 1, 2, 3, 4}
    assert min(result["pit"]["control:z"]) == 0.1
    assert max(result["pit"]["control:z"]) == 0.9
    item = record()
    with pytest.raises(ValueError, match="observed data identity"):
        reference_diagnostics(
            item.result,
            reference(item.result, posterior_sbc=posterior_evidence(data_hash="c" * 64)),
        )
    with pytest.raises(ValueError, match="replicate and joint-draw"):
        posterior_sbc_from_draws(
            **dict(evidence, provenance=dict(evidence["provenance"], joint_draws=False))
        )


def test_input_support_retains_raw_location_scale_and_constant_undefined():
    summary = input_support_summary(
        {"control:z": [100, 102], "channel:a": [2, 2]},
        {"control:z": [104, 108], "channel:a": [3, 3]},
    )
    assert summary["inputs"][0]["context"]["mean"] == 101
    assert summary["inputs"][0]["standardized_mean_shift"] == 5
    assert summary["inputs"][0]["scale_ratio"] == 2
    assert summary["inputs"][1]["standardized_mean_shift"] is None
    assert summary["inputs"][1]["scale_ratio"] is None


def test_public_decompose_result_enters_comparison_without_identity_translation(model):
    from pymc_mfm.inference.api import decompose

    observations = {
        "spend": np.ones((5, 2)),
        "controls": np.zeros((5, 2)),
        "context_outcome": np.array([10.0, 11.0]),
    }
    result = decompose(model, observations, n_samples=4)
    item = ComparisonRecord(
        case="observed", world="one", method="conditional", kind="model", result=result
    )
    summary = comparison_summary([item], bootstrap_samples=20)
    assert summary["records"][0]["provenance"]["data_sha256"] == result.provenance["data_sha256"]
    assert summary["records"][0]["total"]["window_mean"]["crps"] is None
    changed = decompose(
        model, dict(observations, context_outcome=np.array([10.0, 12.0])), n_samples=4
    )
    with pytest.raises(ValueError, match="data"):
        comparison_summary(
            [item, replace(item, method="different_data", result=changed)], bootstrap_samples=20
        )


@pytest.fixture(scope="module")
def saved_report(tmp_path_factory):
    pytest.importorskip("matplotlib")
    path = tmp_path_factory.mktemp("comparison") / "report"
    build_comparison_report([record()], path, bootstrap_samples=20)
    return path


def test_saved_report_round_trip_and_pure_replot(saved_report, tmp_path, monkeypatch):
    loaded = load_comparison_records(saved_report)
    np.testing.assert_array_equal(loaded[0].result.components, record().result.components)
    np.testing.assert_array_equal(loaded[0].truth, record().truth)

    def forbidden(*args, **kwargs):
        raise AssertionError("replot tried to recompute numerical evidence")

    monkeypatch.setattr(comparisons, "comparison_summary", forbidden)
    monkeypatch.setattr(comparisons, "score_record", forbidden)
    monkeypatch.setattr(comparisons, "run_c2st", forbidden)
    panels = render_comparison_report(saved_report, tmp_path / "replot")
    assert (
        next(panel for panel in panels if panel["name"] == "performance_query")["status"]
        == "available"
    )
    stored = json.loads((saved_report / "summary.json").read_text())
    import matplotlib.pyplot as plt

    from pymc_mfm.plotting.comparisons import plot_world_comparison

    figure = plot_world_comparison(stored, case="base", world="0")
    np.testing.assert_array_equal(
        figure.axes[0].lines[0].get_ydata(), stored["records"][0]["total"]["weekly"]["mean"]
    )
    plt.close(figure)


@pytest.mark.parametrize(
    "change", ["missing_result", "missing_arrays", "duplicate", "tampered_table"]
)
def test_bundle_rejects_partial_receipts_and_tampering(saved_report, tmp_path, change):
    root = tmp_path / "report"
    shutil.copytree(saved_report, root)
    manifest = json.loads((root / "manifest.json").read_text())
    if change.startswith("missing"):
        filename = "result.json" if change == "missing_result" else "arrays.npz"
        manifest["artifacts"] = [
            entry for entry in manifest["artifacts"] if entry["path"] != f"record_0000/{filename}"
        ]
    elif change == "duplicate":
        manifest["artifacts"].append(manifest["artifacts"][0])
    else:
        (root / "summary.json").write_text("{}")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_comparison_records(root)
    with pytest.raises(ValueError):
        render_comparison_report(root, tmp_path / "figures")


def test_missing_hidden_total_cannot_drop_a_paired_world():
    _, omitted = omission_pair()
    other = replace(omitted, method="B", latent_truth=None)
    with pytest.raises(ValueError, match="asymmetric latent total"):
        comparison_summary([omitted, other], method_pairs=[("A", "B")], bootstrap_samples=20)


def test_comparison_support_uses_explicit_prior_and_not_truth_as_prior():
    item = record()
    prior = {"cardinality": {"cy": [0, 0]}, "forbidden_families": ["zb"]}
    item = replace(
        item,
        result=replace(item.result, provenance=dict(item.result.provenance, graph_support=prior)),
    )
    scored = score_record(item)["structure"]
    assert scored["metrics"]["wrong_cy_cardinality_mass"] == 1
    assert scored["metrics"]["forbidden_family_mass"] == 1
    assert scored["metrics"]["off_prior_mass"] == 1
    assert scored["metrics"]["structure_exact_hit"] is None


def test_truth_and_reference_cannot_hide_nonzero_padded_contributions():
    item = record()
    draws = item.result.components.copy()
    draws[:, :, -1] = 0
    result = replace(item.result, components=draws, component_mask=np.array([1, 1, 0], dtype=bool))
    with pytest.raises(ValueError, match="truth must be zero"):
        replace(item, result=result, live=np.array([1, 1, 0], dtype=bool))
    with pytest.raises(ValueError, match="reference draws must be zero"):
        reference_diagnostics(result, reference(result, components=item.result.components))


def test_live_family_error_and_all_active_error_are_distinct():
    item = record(
        draws=np.tile([10.0, 3.0, 7.0], (4, 2, 1)),
        names=("baseline", "channel:a", "channel:inactive"),
        families=("baseline", "channel", "channel"),
        truth=np.tile([10.0, 2.0, 0.0], (2, 1)),
        live=np.array([1, 1, 0], dtype=bool),
        scale=2,
    )
    family = score_record(item)["families"]["channel"]
    assert family["total_signed_error"] == 0.5
    assert family["all_active_total_signed_error"] == 4
    assert family["inactive_mean_attribution"] == 3.5


def test_native_widths_cannot_mix_units_across_worlds():
    first, second = record(), record(world="1")
    second = replace(second, result=replace(second.result, units="EUR"))
    with pytest.raises(ValueError, match="share outcome units"):
        comparison_summary([first, second], bootstrap_samples=20)


def test_report_metrics_preserve_normalized_units_paired_deltas_and_uncertainty(tmp_path):
    pytest.importorskip("matplotlib")
    first = record(scale=10)
    first = replace(first, result=replace(first.result, units="USD"))
    second = replace(
        first, method="B", result=replace(first.result, components=first.result.components + 2)
    )
    manifest = build_comparison_report(
        [first, second], tmp_path / "comparison", method_pairs=[("A", "B")], bootstrap_samples=20
    )
    key = "aggregate/base/B/query/channel/crps/mean"
    assert manifest["metrics"][key] == pytest.approx(0.2)
    assert manifest["metric_dimensions"][key]["units"] == "fraction_of_outcome_scale"
    assert (
        manifest["metric_dimensions"]["aggregate/base/B/query/channel/width/mean"]["units"] == "USD"
    )
    delta = "paired/base/B minus A/query/channel/crps_difference"
    assert manifest["metrics"][f"{delta}/mean"] == pytest.approx(0.2)
    assert manifest["metrics"][f"{delta}/ci_low"] == pytest.approx(0.2)
    assert manifest["metrics"][f"{delta}/ci_high"] == pytest.approx(0.2)
    assert manifest["metrics"][f"{delta}/n_worlds"] == 1
    assert manifest["metric_dimensions"][f"{delta}/mean"]["source_step"] == 12


def test_report_persists_failed_panel_status_before_strict_error(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from pymc_mfm.plotting import comparisons as plots

    def fail(*args, **kwargs):
        raise RuntimeError("unrenderable test evidence")

    monkeypatch.setattr(plots, "plot_performance", fail)
    with pytest.raises(RuntimeError, match="saved in manifest"):
        build_comparison_report([record()], tmp_path / "failed", bootstrap_samples=20)
    manifest = json.loads((tmp_path / "failed" / "manifest.json").read_text())
    assert manifest["statuses"]["performance_query"]["status"] == "failed"
    assert manifest["statuses"]["coverage_query"]["status"] == "available"
    assert "unrenderable test evidence" in (tmp_path / "failed" / "index.html").read_text()


def test_posterior_sbc_and_c2st_figures_use_saved_numerical_evidence():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from pymc_mfm.plotting.comparisons import plot_c2st, plot_posterior_sbc

    summary = posterior_sbc_from_draws(**posterior_evidence(), seed=4)
    figure = plot_posterior_sbc(summary)
    np.testing.assert_array_equal(
        figure.axes[1].lines[0].get_xdata(), summary["ecdf"]["control:z"]["x"]
    )
    np.testing.assert_array_equal(
        figure.axes[2].lines[0].get_ydata(), list(summary["coverage"]["control:z"].values())
    )
    plt.close(figure)
    summary = c2st_from_predictions(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        ["a", "b", "a", "b"],
        training_world_ids=["train"],
        classifier="frozen",
    )
    figure = plot_c2st(summary)
    assert figure.axes[1].patches[0].get_height() == summary["accuracy"]
    assert figure.axes[2].patches[0].get_height() == summary["cross_entropy"]
    plt.close(figure)


def pinned_replication(tmp_path):
    data = tmp_path / "whitepaper" / "data"
    data.mkdir(parents=True)
    root = tmp_path / "results"
    receipts = {}
    for case in ("constant_intercept", "omitted_channel", "omitted_control", "sparse_inputs"):
        for directory in ("worlds", "networks/A", "networks/B", "references", "references_mmm"):
            path = root / directory / case / "w0000.npz"
            path.parent.mkdir(parents=True)
            np.savez(path, stored_window_means=np.array([[1.0], [2.0]]))
            receipts[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    metrics = data / "replication_metrics.csv"
    metrics.write_text(
        "case,world,method,window,media_crps,controls_crps,latent_total_crps,recovered,structure_exact_hit,live_channel_count_hit,live_control_count_hit,absorbed_share,baseline_error,media_total_error,controls_total_error\n"
        "constant_intercept,0,oracle,context,1,2,3,True,,,,,0,0,0\n"
        "constant_intercept,0,A_all,context,3,4,5,False,False,True,False,0.25,1,2,3\n"
        "constant_intercept,0,oracle,query,2,3,4,True,,,,,0,0,0\n"
        "constant_intercept,0,A_all,query,5,6,7,False,False,True,False,0.5,2,3,4\n"
    )
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"checkpoints": {"A": {"sha256": "b" * 64}}}))
    source = {
        "schema_version": 2,
        "source": {
            "raw_npz_sha256": receipts,
            "raw_npz_count": len(receipts),
            "metrics": "data/replication_metrics.csv",
            "manifest": "data/manifest.json",
            "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "population": {"worlds_per_case": 1},
        "verification": {
            "matched_world_pairs": 2,
            "false_effect_worlds": 1,
            "aggregate_scores_checked": 4,
            "max_absolute_crps_difference": 0,
            "score_atol": 1e-12,
        },
        "metrics": {},
        "failure_size": {},
        "recovery": {},
        "ab_verdict": {},
        "averaging": {},
        "omission": {},
        "five_controls": {},
        "false_effects": {"methods": {}, "threshold": 0.01},
        "bootstrap": {"seed": 7, "resamples": 20},
        "diagnostics": {},
        "limitations": ["Frozen fixture, no weekly draw bank."],
    }
    path = data / "figure_summary.json"
    path.write_text(json.dumps(source))
    return path, root


def test_frozen_adapter_recovers_context_structure_oracle_and_absorption_scalars(tmp_path):
    path, root = pinned_replication(tmp_path)
    summary = comparisons.load_replication_summary(
        path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), evidence_root=root
    )

    def value(metric, window, family="all"):
        return next(
            row["mean"]
            for row in summary["aggregate"]
            if row["method"] == "A_all"
            and row["metric"] == metric
            and row["window"] == window
            and row["family"] == family
        )

    assert value("recovered", "context") == 0
    assert value("structure_exact_hit", "query", "graph") == 0
    assert value("family/cy/modal_count_hit", "query", "graph") == 1
    assert value("absorbed_share", "query") == 0.5
    assert value("total_signed_error", "context", "channel") == 2
    assert (
        next(
            row["mean"]
            for row in summary["paired"]
            if row["window"] == "context" and row["family"] == "channel"
        )
        == 2
    )
    assert (
        next(
            row["mean"]
            for row in summary["paired"]
            if row["window"] == "query" and row["family"] == "channel"
        )
        == 3
    )
    assert summary["records"] == []
    assert summary["statuses"]["weekly_joint_draws"]["status"] == "unavailable"


@pytest.mark.parametrize("change", ["missing_raw", "altered_archive"])
def test_frozen_adapter_requires_complete_unmodified_raw_receipts(tmp_path, change):
    path, root = pinned_replication(tmp_path)
    source = json.loads(path.read_text())
    raw = "networks/A/omitted_channel/w0000.npz"
    if change == "missing_raw":
        source["source"]["raw_npz_sha256"].pop(raw)
        source["source"]["raw_npz_count"] -= 1
        path.write_text(json.dumps(source))
    else:
        (root / raw).write_bytes(b"altered")
    with pytest.raises(ValueError, match="paired raw evidence|raw evidence hash"):
        comparisons.load_replication_summary(
            path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), evidence_root=root
        )


@pytest.mark.parametrize("private", [False, True])
def test_real_comparison_report_delivers_scalars_or_withholds_every_private_artifact(
    tmp_path, monkeypatch, private
):
    pytest.importorskip("matplotlib")
    mlflow = pytest.importorskip("mlflow")
    from pymc_mfm.training.tracking import MlflowTrainingObserver, TrackingConfig

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    first = record()
    second = replace(
        first,
        method="B",
        result=replace(
            first.result,
            components=first.result.components + 2,
            provenance=dict(first.result.provenance, synthetic_training_evaluation=not private),
        ),
    )
    root = tmp_path / "comparison"
    manifest = build_comparison_report(
        [first, second], root, method_pairs=[("A", "B")], bootstrap_samples=20
    )
    observer = MlflowTrainingObserver(tmp_path / "run", TrackingConfig(diagnostics=False))
    observer.on_run_start({"run_id": "comparison-delivery", "status": "running"})
    observer.on_diagnostic(manifest, root)
    observer.on_run_end({"run_id": "comparison-delivery", "status": "finished", "final_step": 12})
    client = mlflow.MlflowClient(tracking_uri=observer.tracking_uri)
    events = [
        json.loads(path.read_text()) for path in sorted((observer.root / "events").glob("*.json"))
    ]
    event = next(event for event in events if event["kind"] == "diagnostic")
    assert event["payload"]["manifest"] == manifest
    paths = set()
    pending = [""]
    while pending:
        for artifact in client.list_artifacts(observer.run_id, pending.pop()):
            if artifact.is_dir:
                pending.append(artifact.path)
            else:
                paths.add(artifact.path)
    if private:
        assert client.get_run(observer.run_id).data.metrics == {}
        assert not any(
            path.endswith(("comparison.npz", "summary.json", ".png", "index.html"))
            for path in paths
        )
        receipt = next(event for event in events if event["kind"] == "diagnostic_receipt")
        assert receipt["payload"]["status"] == "withheld_sensitive"
    else:
        key = "paired/base/B minus A/query/channel/crps_difference/mean"
        scalar = next(
            scalar
            for scalar in event["scalars"]
            if scalar["original"].endswith(key.replace("~", "~0").replace("/", "~1"))
        )
        history = client.get_metric_history(observer.run_id, scalar["key"])
        assert [(point.step, point.value) for point in history] == [(12, 2.0)]
        assert any(path.endswith("/comparison.npz") for path in paths)
        assert any(path.endswith("/index.html") for path in paths)
    assert observer.summary["status"] == "finished"


def test_omission_bootstrap_weights_worlds_not_retained_component_counts():
    first_before, first_after = omission_pair()  # One retained component, delta -1.
    names = ("baseline", "channel:a", "channel:c", "channel:d", "channel:hidden")
    families = ("baseline",) + ("channel",) * 4
    truth = np.tile([10.0, 1.0, 1.0, 1.0, 5.0], (4, 1))
    evidence = {
        "time": np.arange(4),
        "native_truth": truth,
        "noise": np.zeros(4),
        "observed": truth.sum(axis=1),
        "native_inputs": np.ones((4, 4)),
    }
    common = {
        "world": "1",
        "world_evidence": evidence,
        "input_names": names[1:],
        "truth_names": names,
        "observed": evidence["observed"][2:],
        "latent_truth": evidence["observed"][2:],
    }
    second_before = record(
        draws=np.tile(truth[2:], (4, 1, 1)),
        names=names,
        families=families,
        truth=truth[2:],
        visible_inputs=names[1:],
        **common,
    )
    after_draws = np.tile([10.0, 4.0, 4.0, 4.0], (4, 2, 1))
    second_after = record(
        case="omitted",
        draws=after_draws,
        names=names[:-1],
        families=families[:-1],
        truth=truth[2:, :-1],
        visible_inputs=names[1:-1],
        omitted_input=names[-1],
        **common,
    )
    summary = comparison_summary(
        [first_before, first_after, second_before, second_after],
        omission_pairs=[("base", "omitted")],
        bootstrap_samples=200,
        seed=7,
    )
    delta = summary["omission_summary"][0]
    assert delta["mean"] == 1  # (-1 + mean(3, 3, 3)) / 2, not component-pooled 2.
    assert delta["n_worlds"] == 2
    assert delta["ci_low"] == -1
    assert delta["ci_high"] == 3


def test_corpus_support_preserves_world_level_location_scale_and_plot_values():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from pymc_mfm.plotting.comparisons import plot_corpus_support

    summary = comparisons.corpus_support_summary({"first": [98, 102], "second": [15, 25]})
    assert summary["worlds"] == [
        {"world": "first", "n_context": 2, "mean": 100, "std": 2},
        {"world": "second", "n_context": 2, "mean": 20, "std": 5},
    ]
    figure = plot_corpus_support(summary)
    np.testing.assert_array_equal(figure.axes[0].collections[0].get_offsets(), [[100, 2], [20, 5]])
    assert sum(patch.get_height() for patch in figure.axes[1].patches) == 2
    assert sum(patch.get_height() for patch in figure.axes[2].patches) == 2
    plt.close(figure)
