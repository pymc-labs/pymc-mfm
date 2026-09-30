"""Scoring rules against hand values and the in-loop evaluation record contract."""

import json
import math

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.data.corpus import corpus_batch
from pymc_mfm.model.graph import STRUCT_EDGE_KEYS, EdgeSlotLayout
from pymc_mfm.training import evaluation, metrics
from pymc_mfm.training.evaluation import (
    COUNT_METRICS,
    DECOMPOSITION_METRICS,
    EDGE_METRICS,
    GRAPH_METRICS,
    METRIC_BOUNDS,
    METRIC_UNITS,
    _score_decomposition,
    evaluate_rows,
    flatten_evaluation,
    metric_aggregation,
    selection_metrics,
    within_bounds,
)


def test_crps_of_a_point_mass_is_the_absolute_error():
    samples = np.full((5, 3), 2.0)
    truth = np.array([1.0, 2.0, 4.0])
    assert metrics.empirical_crps(samples, truth) == pytest.approx(1.0)


def test_energy_score_of_identical_draws_is_the_normalized_distance():
    x = np.array([1.0, 2.0, 2.0, 0.0])
    truth = np.array([0.0, 0.0, 0.0, 0.0])
    samples = np.tile(x, (3, 1))
    assert metrics.energy_score(samples, truth) == pytest.approx(3.0 / 2.0)
    assert metrics.energy_score(samples[:1], truth) is None


def test_energy_score_pair_term_matches_explicit_pairs():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(6, 4))
    truth = rng.normal(size=4)
    scale = 2.0
    truth_term = np.mean([np.linalg.norm(s - truth) for s in samples]) / scale
    pairs = [np.linalg.norm(samples[i] - samples[j]) for i in range(6) for j in range(6) if i < j]
    pair_term = sum(pairs) / (6 * 5) / scale
    assert metrics.energy_score(samples, truth) == pytest.approx(truth_term - pair_term)


def test_auroc_separated_ties_and_one_class():
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert metrics.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert metrics.auroc([0.1, 0.9], [1, 1]) is None
    rng = np.random.default_rng(1)
    p = np.round(rng.random(40), 1)
    y = rng.integers(0, 2, 40)
    pos, neg = p[y == 1], p[y == 0]
    pairwise = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert metrics.auroc(p, y) == pytest.approx(pairwise)


def test_ece_of_perfectly_calibrated_bins_is_zero():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert metrics.ece(p, y) == 0.0
    assert metrics.ece(np.array([0.9, 0.9]), np.array([0, 0])) == pytest.approx(0.9)


def test_coverage_counts_truth_inside_central_interval():
    samples = np.repeat(np.arange(11.0)[:, None], 4, axis=1)
    truth = np.array([5.0, 10.0, 0.0, 1.0])
    assert metrics.coverage(samples, truth) == 0.5


def test_log_score_and_brier_hand_values():
    p = np.array([0.5, 0.5])
    y = np.array([1.0, 0.0])
    assert metrics.log_score(p, y) == pytest.approx(math.log(2.0))
    assert metrics.brier(p, y) == pytest.approx(0.25)


@pytest.fixture(scope="module")
def corpus():
    return make_synthetic_corpus()


@pytest.fixture
def holdout(corpus):
    return [(0, int(row)) for row in np.flatnonzero(corpus["is_val"] == 1)]


@pytest.fixture
def record(model, corpus, holdout):
    return evaluate_rows(model, [corpus], holdout, n_samples=4, seed=1, device="cpu", leg="holdout")


def test_perfect_decomposition_scores_zero_and_closes_on_corpus_outcome(corpus, holdout):
    _, row = holdout[0]
    batch = corpus_batch(corpus, [row])
    truth = batch["targets_z"][0].double().numpy()
    n_context = batch["context_targets_z"].shape[1]
    world = _score_decomposition(
        np.repeat(truth[None], 3, axis=0),
        truth,
        batch["active_c_mask"][0].bool().numpy(),
        batch["active_m_mask"][0].bool().numpy(),
        batch["g_cy"][0].numpy(),
        batch["g_zb"][0].numpy(),
        np.asarray(corpus["outcome_raw"][row], dtype=np.float64)[n_context:],
        np.asarray(corpus["outcome_noise"][row], dtype=np.float64)[n_context:],
        float(batch["sales_scale"][0]),
        graph_c=np.repeat(batch["g_cy"][0].numpy()[None], 3, axis=0),
        graph_m=np.repeat(batch["g_zb"][0].numpy()[None], 3, axis=0),
    )
    # Identical draws leave residual and across-draw correlations without variation.
    constant = {
        "contribution_truth_error_residual_lag1",
        "contribution_posterior_centered_residual_lag1",
        "contribution_adjacent_draw_correlation",
    }
    assert set(world.values) | set(world.undefined) == set(DECOMPOSITION_METRICS)
    assert constant <= set(world.undefined)
    for key, reason in world.undefined.items():
        if key in constant:
            assert reason == "constant_series", key
        else:  # this world may lack live or dead channels/controls
            assert key.startswith(("attribution_", "timeavg_")), key
            assert reason in {
                "no_live_effects",
                "no_dead_effects",
                "fewer_than_two_live_effects",
                "zero_posterior_sd",
            }, key
    for key, value in world.values.items():
        if key.endswith("_cov80") or key.startswith("timeavg_"):
            assert value == 1.0, key
        elif key.startswith("closure_"):
            assert value == pytest.approx(0.0, abs=1e-5), key
        elif key == "latent_sales_r2":
            assert value == pytest.approx(1.0, abs=1e-6)
        elif key in ("contribution_truth_lag1", "contribution_draw_lag1"):
            assert value == pytest.approx(world.values["contribution_truth_lag1"], abs=1e-12)
        else:
            assert value == pytest.approx(0.0, abs=1e-12), key


def test_simultaneous_window_coverage_scores_live_media_and_controls_not_padding():
    draws = np.array([[-1, -100], [-1, 100], [1, -100], [1, 100]])[:, None, :]
    pred, truth = np.zeros((4, 2, 11)), np.zeros((2, 11))
    pred[..., 1:3] = draws
    pred[..., 4:6] = -draws
    truth[:, 1:3] = (2.0, 1.0)
    # Unused padded effects cannot contaminate either joint event.
    pred[..., [3, 6]] = np.nan
    scored = _score_decomposition(
        pred,
        truth,
        np.array([True, True, False]),
        np.array([True, True, False]),
        np.ones(3),
        np.ones(3),
        np.ones(2),
        np.zeros(2),
        1.0,
    )
    media, control = (
        "timeavg_contribution_simcov80_live",
        "timeavg_control_contribution_simcov80_live",
    )
    assert scored.values[media] == 0.0
    assert scored.values[control] == 1.0
    assert scored.weights[media] == scored.weights[control] == 1
    assert scored.values["timeavg_contribution_cov80_live"] == 0.5
    aggregation = metric_aggregation()
    for name in (media, control):
        assert aggregation[f"conditional/query/{name}"] == "world"
        assert METRIC_BOUNDS[name] == (0.0, 1.0)
        assert METRIC_UNITS[name] == "dimensionless"


def test_conditional_window_coverage_uses_only_aligned_gate_on_draws():
    pred = np.zeros((4, 2, 7))
    pred[:2, :, 1] = 2.0
    pred[2:, :, 2] = -2.0
    truth = np.zeros((2, 7))
    truth[:, 1] = 0.25
    truth[:, 2] = -0.25
    media_gates = np.array([[1], [1], [0], [0]])
    control_gates = np.array([[0], [0], [1], [1]])
    scored = _score_decomposition(
        pred,
        truth,
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1),
        np.ones(1),
        np.zeros(2),
        np.zeros(2),
        1.0,
        graph_c=media_gates,
        graph_m=control_gates,
    )
    # The full graph mixture covers both truths; the edge-on draws do not.
    assert scored.values["attribution_live_media_conditional_cov80"] == 0
    assert scored.values["attribution_live_control_conditional_cov80"] == 0
    assert scored.weights["attribution_live_control_conditional_cov80"] == 1
    short = _score_decomposition(
        pred,
        truth,
        np.ones(1, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(1),
        np.ones(1),
        np.zeros(2),
        np.zeros(2),
        1.0,
        graph_c=np.array([[1], [0], [0], [0]]),
        graph_m=control_gates,
    )
    assert short.undefined["attribution_live_media_conditional_cov80"] == (
        "needs_edge_on_draws_ge_2"
    )


def test_conditional_coverage_sums_window_and_weights_eligible_slots():
    pred = np.zeros((4, 2, 8))
    truth = np.zeros((2, 8))
    pred[0, :, 1] = [3, -2]
    pred[1, :, 1] = [-2, 3]
    truth[:, 1] = [3, -2]
    pred[2:, :, 2] = 3
    truth[:, 2] = [0.5, 0.5]
    scored = _score_decomposition(
        pred,
        truth,
        np.ones(2, dtype=bool),
        np.ones(1, dtype=bool),
        np.ones(2),
        np.zeros(1),
        truth.sum(axis=1),
        np.zeros(2),
        1.0,
        graph_c=np.array([[1, 0], [1, 0], [0, 1], [0, 1]]),
        graph_m=np.zeros((4, 1)),
    )
    assert scored.values["attribution_live_media_conditional_cov80"] == 0.5
    assert scored.weights["attribution_live_media_conditional_cov80"] == 2


def test_record_has_the_four_heads_with_fixed_keys(record, holdout):
    assert record["leg"] == "holdout"
    assert record["n_worlds"] == len(holdout)
    assert record["n_samples"] == 4
    heads = record["heads"]
    assert set(heads) == {"joint", "conditional", "structure_head", "joint_router"}
    assert set(record["labels"]) == set(heads)
    for head in ("joint", "conditional"):
        assert set(heads[head]) == {"query", "context"}
        for window in ("query", "context"):
            assert set(heads[head][window]) == set(DECOMPOSITION_METRICS)
    assert set(heads["structure_head"]) == set(STRUCT_EDGE_KEYS)
    assert set(heads["joint_router"]["edge"]) == set(STRUCT_EDGE_KEYS)
    for et in STRUCT_EDGE_KEYS:
        assert set(heads["structure_head"][et]) == set(EDGE_METRICS)
        assert set(heads["joint_router"]["edge"][et]) == set(EDGE_METRICS)
    assert set(heads["joint_router"]["graph"]) == set(GRAPH_METRICS)
    assert set(record["counts"]) == set(COUNT_METRICS)
    assert record["counts"]["worlds"] == len(holdout)
    per_world_keys = {
        key for key, kind in metric_aggregation().items() if kind != "pooled_edges"
    } - {"counts/worlds"}
    assert set(record["metric_worlds"]) == per_world_keys
    assert "per_world" not in record
    # Tables describe the same pooled active slots as the edge scores.
    for readout, scores in (
        ("structure_head", heads["structure_head"]),
        ("joint_router", heads["joint_router"]["edge"]),
    ):
        assert set(record["tables"][readout]) == set(STRUCT_EDGE_KEYS)
        for et, tables in record["tables"][readout].items():
            n, n_pos = scores[et]["n"], scores[et]["n_pos"]
            assert sum(row["count"] for row in tables["reliability"]) == n
            assert len(tables["thresholds"]) == (len(metrics.THRESHOLDS) if n else 0)
            for row in tables["thresholds"]:
                assert row["tp"] + row["fn"] == n_pos
                assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == n
            corp = tables["corp"]
            assert sum(row["count"] for row in corp) == (n if 0 < n_pos < n else 0)
            assert sum(row["n_pos"] for row in corp) == (n_pos if corp else 0)


def test_joint_and_conditional_are_distinct_and_populated(record):
    joint = record["heads"]["joint"]["query"]
    conditional = record["heads"]["conditional"]["query"]
    assert joint != conditional
    for values in (joint, conditional, record["heads"]["joint"]["context"]):
        assert math.isfinite(values["decomposition_energy_score"])
        assert math.isfinite(values["closure_rel_err"])
        assert values["contribution_padding_leakage_scaled_mae"] == 0.0
    cy = record["heads"]["structure_head"]["cy"]
    assert cy["n"] > 0 and cy["n_pos"] <= cy["n"]
    assert math.isfinite(cy["log_score"])
    graph = record["heads"]["joint_router"]["graph"]
    assert graph["shd_mean"] >= 0.0
    assert 0.0 <= graph["exact_recovery"] <= 1.0
    # Finite draws leave only data-dependent reasons, never a non-finite world.
    data_reasons = {
        "one_class",
        "no_active_slots",
        "no_negative_slots",
        "no_live_effects",
        "no_dead_effects",
        "fewer_than_two_live_effects",
        "zero_posterior_sd",
        "constant_series",
        "needs_2_active_nodes",
        "needs_2_active_slots",
    }
    for key, reason in record["undefined"].items():
        assert reason in data_reasons, (key, reason)


def test_single_draw_marks_energy_scores_undefined(model, corpus, holdout):
    single = evaluate_rows(
        model, [corpus], holdout[:2], n_samples=1, seed=3, device="cpu", leg="seen"
    )
    joint = single["heads"]["joint"]["query"]
    assert joint["decomposition_energy_score"] is None
    assert single["undefined"]["joint/query/decomposition_energy_score"] == "needs_n_samples_ge_2"
    assert math.isfinite(joint["baseline_intrinsic_crps"])


def test_seed_changes_draws_but_not_counts_and_is_reproducible(record, model, corpus, holdout):
    other = evaluate_rows(
        model, [corpus], holdout, n_samples=4, seed=2, device="cpu", leg="holdout"
    )
    assert other["n_worlds"] == record["n_worlds"]
    assert (
        other["heads"]["joint"]["query"]["decomposition_energy_score"]
        != record["heads"]["joint"]["query"]["decomposition_energy_score"]
    )
    again = evaluate_rows(
        model, [corpus], holdout, n_samples=4, seed=1, device="cpu", leg="holdout"
    )
    assert again == record


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("scoring_fails", [False, True], ids=["success", "scorer-error"])
def test_evaluation_leaves_rng_and_training_flag_untouched(
    model, corpus, holdout, monkeypatch, device, training, scoring_fails
):
    model.to(device).train(training)
    torch.manual_seed(123)
    # Snapshot every available accelerator, not just the evaluation device:
    # global manual_seed inside CPU evaluation would also perturb MPS/CUDA.
    cuda_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    mps_before = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    cpu_before = torch.get_rng_state().clone()

    if scoring_fails:
        error = RuntimeError("scoring failed")

        def fail_after_draws(*args, **kwargs):
            torch.rand(3)
            torch.rand(3, device=device)
            raise error

        monkeypatch.setattr(evaluation, "_score_batch", fail_after_draws)
        with pytest.raises(RuntimeError) as exc_info:
            evaluate_rows(model, [corpus], holdout[:2], n_samples=2, seed=7, device=device, leg="x")
        assert exc_info.value is error
    else:
        evaluate_rows(model, [corpus], holdout[:2], n_samples=2, seed=7, device=device, leg="x")

    assert torch.equal(cpu_before, torch.get_rng_state())
    if torch.cuda.is_available():
        for before, after in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True):
            assert torch.equal(before, after)
    if mps_before is not None:
        assert torch.equal(mps_before, torch.mps.get_rng_state())
    assert model.training is training


def test_flatten_uses_slashes_and_names_selection_keys(record):
    flat = flatten_evaluation(record)
    assert set(flat) == {f"holdout/{key}" for key in metric_aggregation()}
    for head in ("conditional", "joint"):
        keys = selection_metrics(head)
        assert math.isfinite(flat[keys.monitor]) and math.isfinite(flat[keys.context_gate])
        assert keys.structure_gate in flat and keys.leakage_gate in flat
    assert all(value is None or isinstance(value, float) for value in flat.values())
    n_cy = record["heads"]["structure_head"]["cy"]["n"]
    assert flat["holdout/structure_head/cy/n"] == float(n_cy)
    assert flat["holdout/counts/worlds"] == float(record["n_worlds"])
    for key, reason in record["undefined"].items():
        assert flat[f"holdout/{key}"] is None, (key, reason)


def test_metric_bounds_name_exactly_the_evaluated_metrics_and_hold(record):
    flat = flatten_evaluation(record)
    assert {key.rsplit("/", 1)[-1] for key in flat} == set(METRIC_BOUNDS)
    outside = {
        key: value
        for key, value in flat.items()
        if value is not None and not within_bounds(key, value)
    }
    assert outside == {}


def test_metric_units_name_every_metric_and_scale_like_their_scores(corpus, holdout):
    assert set(METRIC_UNITS) == set(METRIC_BOUNDS)
    assert set(METRIC_UNITS.values()) <= {"sales_scale", "nats", "count", "dimensionless"}
    # Scaling z-space draws and truth, with the natural outcome and noise, by 3
    # multiplies every sales_scale score by 3 and leaves the other scores unchanged.
    _, row = holdout[0]
    batch = corpus_batch(corpus, [row])
    truth = batch["targets_z"][0].double().numpy()
    draws = truth + np.random.default_rng(5).normal(scale=0.3, size=(6, *truth.shape))
    n_context = batch["context_targets_z"].shape[1]
    masks = [batch[key][0].numpy() for key in ("active_c_mask", "active_m_mask", "g_cy", "g_zb")]
    masks[:2] = [mask.astype(bool) for mask in masks[:2]]
    outcome, noise = (
        np.asarray(corpus[key][row], dtype=np.float64)[n_context:]
        for key in ("outcome_raw", "outcome_noise")
    )
    scale = float(batch["sales_scale"][0])
    base = _score_decomposition(draws, truth, *masks, outcome, noise, scale)
    scaled = _score_decomposition(3 * draws, 3 * truth, *masks, 3 * outcome, 3 * noise, scale)
    assert scaled.values.keys() == base.values.keys()
    assert {METRIC_UNITS[key] for key in base.values} == {"sales_scale", "dimensionless"}
    for key, value in base.values.items():
        factor = 3 if METRIC_UNITS[key] == "sales_scale" else 1
        assert scaled.values[key] == pytest.approx(factor * value, rel=1e-9, abs=1e-12), key
    # A unit conversion changes the raw observations AND their ruler, not the
    # canonical draws. Neither relative diagnostics nor canonical errors may move.
    for units in (1e-100, 1e200):
        converted = _score_decomposition(
            draws, truth, *masks, units * outcome, units * noise, units * scale
        )
        assert converted.undefined == base.undefined
        assert converted.values == pytest.approx(base.values, rel=1e-9, abs=1e-12)


def test_metric_units_refuse_a_metric_this_release_does_not_define():
    flat = "holdout/conditional/query/{}"
    assert evaluation.metric_units(flat.format("decomposition_energy_score")) == "sales_scale"
    with pytest.raises(ValueError, match="some_new_score"):
        evaluation.metric_units(flat.format("some_new_score"))


def test_record_is_json_serializable_without_nan(record):
    json.dumps(record, allow_nan=False)


def test_nonfinite_decomposition_coverage_is_undefined_not_zero(
    model, corpus, holdout, monkeypatch
):
    predict = model.predict

    def corrupt_baseline(*args, **kwargs):
        samples, graph = predict(*args, **kwargs)
        samples[..., 0] = float("nan")
        return samples, graph

    monkeypatch.setattr(model, "predict", corrupt_baseline)
    record = evaluate_rows(
        model, [corpus], holdout[:1], n_samples=2, seed=1, device="cpu", leg="holdout"
    )
    assert record["heads"]["joint"]["query"]["baseline_intrinsic_cov80"] is None
    assert record["undefined"]["joint/query/baseline_intrinsic_cov80"] == "nonfinite_in_1_worlds"
    assert math.isfinite(record["heads"]["conditional"]["query"]["baseline_intrinsic_cov80"])
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_auxiliary_probabilities_keep_truth_counts_but_no_scores(
    model, corpus, holdout, monkeypatch, invalid
):
    predictive_probs = model.struct_head.predictive_probs

    def corrupt_probability(logits):
        probs = predictive_probs(logits)
        probs["cy"][0, 0] = invalid  # First channel is active in the fixture.
        return probs

    monkeypatch.setattr(model.struct_head, "predictive_probs", corrupt_probability)
    record = evaluate_rows(
        model, [corpus], holdout[:1], n_samples=2, seed=1, device="cpu", leg="holdout"
    )
    edge = record["heads"]["structure_head"]["cy"]
    for metric in ("log_score", "brier", "auroc", "auprc", "ece", "mcb", "dsc", "null_fpr"):
        assert edge[metric] is None
        assert record["undefined"][f"structure_head/cy/{metric}"] == "nonfinite_in_1_slots"
    assert record["tables"]["structure_head"]["cy"] == {
        "reliability": [],
        "corp": [],
        "thresholds": [],
    }
    batch = corpus_batch(corpus, [holdout[0][1]])
    active = batch["active_c_mask"].bool()
    truth = batch["g_cy"][active]
    assert edge["n"] == truth.numel()
    assert edge["n_pos"] == int(truth.sum())
    assert edge["prevalence"] == pytest.approx(float(truth.mean()))
    # unc = prevalence (1 - prevalence) summarizes truth alone (one of two cy edges is live).
    assert edge["unc"] == pytest.approx(edge["prevalence"] * (1 - edge["prevalence"])) == 0.25
    assert "structure_head/cy/unc" not in record["undefined"]
    json.dumps(record, allow_nan=False)


def test_one_class_edge_families_evidence_no_corp_calibration():
    pool = evaluation._EdgePool()
    everything = np.ones(6, dtype=bool)
    cy = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2]), np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    dz = np.array([0.05, 0.1, 0.2, 0.0, 0.3, 0.1]), np.zeros(6)
    zb = np.array([0.6, 0.9, 0.2, 1.0, 0.7, 0.8]), np.ones(6)
    for et, (p, y) in (("cy", cy), ("dz", dz), ("zb", zb)):
        pool.add(et, p, y, everything)
    undefined = {}
    records, tables = pool.finalize("joint_router/edge", undefined)
    split = records["cy"]
    assert split["mcb"] - split["dsc"] + split["unc"] == pytest.approx(split["brier"])
    assert split["unc"] == 0.25 and "joint_router/edge/cy/mcb" not in undefined
    assert tables["cy"]["corp"] == metrics.isotonic_calibration(*cy)
    for et, (p, y) in (("dz", dz), ("zb", zb)):
        for key in ("mcb", "dsc", "unc"):
            assert records[et][key] is None
            assert undefined[f"joint_router/edge/{et}/{key}"] == "one_class"
        # ece keeps its recorded definition, so recorded one-class values stay comparable.
        assert records[et]["ece"] == metrics.ece(p, y)
        assert f"joint_router/edge/{et}/ece" not in undefined
        assert tables[et]["corp"] == [] and tables[et]["reliability"]
    for key in ("mcb", "dsc", "unc"):
        assert undefined[f"joint_router/edge/db/{key}"] == "no_active_slots"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_router_draws_are_not_scored_as_binary_graphs(
    model, corpus, holdout, monkeypatch, invalid
):
    predict = model.predict

    def corrupt_graph(*args, **kwargs):
        samples, graph = predict(*args, **kwargs)
        graph[..., 0] = invalid  # First packed cy slot is active.
        return samples, graph

    monkeypatch.setattr(model, "predict", corrupt_graph)
    record = evaluate_rows(
        model, [corpus], holdout[:1], n_samples=2, seed=1, device="cpu", leg="holdout"
    )
    for metric in GRAPH_METRICS:
        assert record["heads"]["joint_router"]["graph"][metric] is None
        assert record["undefined"][f"joint_router/graph/{metric}"] == "nonfinite_in_1_worlds"
    assert record["heads"]["joint_router"]["edge"]["cy"]["log_score"] is None
    assert record["undefined"]["joint_router/edge/cy/log_score"] == "nonfinite_in_1_slots"
    assert math.isfinite(record["heads"]["joint"]["query"]["decomposition_energy_score"])
    json.dumps(record, allow_nan=False)


def test_heads_selector_samples_only_the_selected_heads(
    model, corpus, holdout, record, monkeypatch
):
    calls = []
    for name in ("predict", "sample_context", "sample_disentangled"):
        original = getattr(model, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(model, name, spy)

    def run(heads, **kwargs):
        return evaluate_rows(
            model,
            [corpus],
            holdout,
            n_samples=4,
            seed=1,
            device="cpu",
            leg="holdout",
            heads=heads,
            **kwargs,
        )

    joint = run(("joint",))
    assert "sample_disentangled" not in calls
    assert set(joint["heads"]) == {"joint", "structure_head", "joint_router"}
    assert set(joint["labels"]) == set(joint["heads"])
    assert set(flatten_evaluation(joint)) == {
        f"holdout/{key}" for key in metric_aggregation(("joint",))
    }
    # The structure pass and the counts do not depend on the draws.
    assert joint["heads"]["structure_head"] == record["heads"]["structure_head"]
    assert joint["counts"] == record["counts"]

    calls.clear()
    conditional = run(("conditional",))
    assert "sample_context" not in calls and "predict" in calls  # the router needs joint draws
    assert set(conditional["heads"]) == {"conditional", "structure_head", "joint_router"}

    for heads in ((), ("joint", "joint"), ("router",), "joint"):
        with pytest.raises(ValueError):
            run(heads)
    with pytest.raises(ValueError):
        run(("joint",), result_sink=lambda *args: None)


@pytest.mark.parametrize("units", [1.0, 1e200])
def test_captured_results_describe_the_corpus_raw_inputs_of_each_window(
    model, corpus, holdout, units
):
    corpus = {
        **corpus,
        **{key: corpus[key].astype(np.float64) * units for key in ("treatment_raw", "covariates")},
    }
    captured = []
    evaluate_rows(
        model,
        [corpus],
        holdout[:1],
        n_samples=2,
        seed=1,
        device="cpu",
        leg="holdout",
        result_sink=lambda result, *evidence: captured.append(result),
    )
    _, row = holdout[0]
    context = int(corpus["support_mask"][row].sum())
    expected = [
        (f"{family}:{family}_{index}", np.asarray(values[:, index], dtype=np.float64))
        for family, values, mask in (
            ("channel", corpus["treatment_raw"][row], corpus["treatment_active_mask"][row]),
            ("control", corpus["covariates"][row], corpus["covariate_active_mask"][row]),
        )
        for index in np.flatnonzero(mask)
    ]
    assert sorted(result.window for result in captured) == ["context", "query"]
    for result in captured:
        summary = result.diagnostics().summary
        assert summary["panels"]["input_support"]["status"] == "ok", result.window
        rows = summary["arrays"]["input_support"]["inputs"]
        assert [entry["name"] for entry in rows] == [name for name, _ in expected]
        for entry, (_, series) in zip(rows, expected, strict=True):
            for window, values in (("context", series[:context]), ("query", series[context:])):
                assert entry[window] == pytest.approx(
                    {
                        "count": len(values),
                        "mean": (values / units).mean() * units,
                        "std": (values / units).std() * units,
                        "min": values.min(),
                        "max": values.max(),
                    }
                ), (entry["name"], window)


def test_per_world_values_reproduce_the_leg_aggregates(model, corpus, holdout, record):
    detailed = evaluate_rows(
        model, [corpus], holdout, n_samples=4, seed=1, device="cpu", leg="holdout", per_world=True
    )
    entries = detailed.pop("per_world")
    assert detailed == record  # opting in changes neither draws nor scores
    assert sorted((entry["corpus_index"], entry["row"]) for entry in entries) == sorted(holdout)
    flat = flatten_evaluation(record)
    for key, count in record["metric_worlds"].items():
        values = [entry["values"][key] for entry in entries if entry["values"][key] is not None]
        assert len(values) == count, key
        if values:
            assert np.mean(values) == pytest.approx(flat[f"holdout/{key}"], rel=1e-12), key
        else:
            assert flat[f"holdout/{key}"] is None, key
        for entry in entries:
            defined = entry["values"][key] is not None
            assert (entry["weights"][key] > 0) == defined, key
            assert (key in entry["undefined"]) == (not defined), key
    for entry in entries:
        assert entry["cell_id"] == int(corpus["cell_id"][entry["row"]])
        n_channels = int(corpus["treatment_active_mask"][entry["row"]].sum())
        assert entry["weights"]["joint/query/contribution_crps"] == n_channels
        assert entry["weights"]["joint/query/contribution_padding_leakage_scaled_mae"] == (
            corpus["treatment_active_mask"].shape[1] - n_channels
        )
        assert entry["weights"]["joint/query/decomposition_energy_score"] == 1


@pytest.mark.parametrize(
    "metadata",
    [
        "missing",
        "null",
        "nan",
        "wrong_shape",
        "boolean",
        "mixed_boolean",
        "wrapped_boolean",
        "masked",
    ],
)
def test_unusable_cell_metadata_preserves_evaluation_without_inventing_cells(
    model, corpus, holdout, metadata
):
    changed = dict(corpus)
    n_worlds = len(corpus["outcome_raw"])
    if metadata == "missing":
        changed.pop("cell_id")
    else:
        changed["cell_id"] = {
            "null": None,
            "nan": np.full(n_worlds, np.nan),
            "wrong_shape": np.zeros((n_worlds, 1), dtype=int),
            "boolean": np.zeros(n_worlds, dtype=bool),
            "mixed_boolean": [0, True, *range(2, n_worlds)],
            "wrapped_boolean": [0, np.array(True), *range(2, n_worlds)],
            "masked": np.ma.array(corpus["cell_id"], mask=np.arange(n_worlds) == 0),
        }[metadata]
    options = dict(
        n_samples=3, seed=17, device="cpu", leg="holdout", heads=("joint",), per_world=True
    )
    before = evaluate_rows(model, [corpus], holdout[:1], **options)
    after = evaluate_rows(model, [changed], holdout[:1], **options)
    assert after["heads"] == before["heads"]
    assert after["per_world"][0]["values"] == before["per_world"][0]["values"]
    assert after["per_world"][0]["cell_id"] is None
    json.dumps(after, allow_nan=False)


QUERY_SHIFTS = (1.0, 3.0)
CONTEXT_SHIFTS = (-1.0, 3.0)


def _live_dead_corpus():
    """Two held-out worlds: A with only live channels and dead controls, B the reverse."""
    corpus = make_synthetic_corpus(n_cells=4, draws_per_cell=1, context_lengths=(6,))
    layout = EdgeSlotLayout(3, 2, 2)
    graph = corpus["g"].copy()
    world_a, world_b = (int(row) for row in np.flatnonzero(corpus["is_val"] == 1))
    for row, media, control in ((world_a, 1, 0), (world_b, 0, 1)):
        graph[row, layout.slices["cy"]] = media * corpus["treatment_active_mask"][row]
        graph[row, layout.slices["zb"]] = control * corpus["covariate_active_mask"][row]
    corpus["g"] = graph
    return corpus, [(0, world_a), (0, world_b)]


def _stub_joint_draws(model, monkeypatch, *, zz_cycle=False):
    """Two joint draws per window: ``truth + shift_s + 0.5 (-1)^t`` in every column.

    Draws are float64, so the shifts are exact. Query router draws are the true
    graph and the true graph with every present channel's ``cy`` edge flipped;
    with ``zz_cycle`` the second draw also links every present control pair both ways.
    """

    def decomposition(truth, shifts):
        truth = truth.double()
        wiggle = 0.5 - (torch.arange(truth.shape[1]) % 2).double()
        shift = torch.tensor(shifts, dtype=torch.float64)
        return truth[None] + shift[:, None, None, None] + wiggle[None, None, :, None]

    def graph_flow(batch, n_weeks):
        layout = EdgeSlotLayout(3, 2, 2)
        truth = layout.pack({et: batch[f"g_{et}"] for et in STRUCT_EDGE_KEYS})
        flipped = truth.clone()
        cy = layout.slices["cy"]
        flipped[:, cy] = torch.where(batch["active_c_mask"] > 0, 1 - truth[:, cy], truth[:, cy])
        if zz_cycle:
            # Two controls have exactly two zz slots: z0 -> z1 and z1 -> z0.
            both = (batch["active_m_mask"] > 0).all(dim=-1, keepdim=True)
            zz = layout.slices["zz"]
            flipped[:, zz] = torch.where(both, torch.ones_like(truth[:, zz]), truth[:, zz])
        return (2 * torch.stack([truth, flipped]) - 1)[:, :, None].expand(-1, -1, n_weeks, -1)

    def predict(batch, n_samples, return_structure=False):
        truth = batch["targets_z"]
        return decomposition(truth, QUERY_SHIFTS), graph_flow(batch, truth.shape[1])

    def sample_context(batch, n_samples, return_structure=False):
        truth = batch["context_targets_z"]
        return decomposition(truth, CONTEXT_SHIFTS), graph_flow(batch, truth.shape[1])

    monkeypatch.setattr(model, "predict", predict)
    monkeypatch.setattr(model, "sample_context", sample_context)


def _cumulative_energy(shifts, n_weeks):
    # Running errors shift_s * t + cumsum(wiggle) in every present channel; the
    # channel count cancels from the dimension-normalized score.
    weeks = np.arange(1, n_weeks + 1)
    wiggle = np.cumsum(0.5 - np.arange(n_weeks) % 2)
    truth_term = np.mean([np.linalg.norm(shift * weeks + wiggle) for shift in shifts])
    pair_term = abs(shifts[1] - shifts[0]) * np.linalg.norm(weeks) / 2
    return (truth_term - pair_term) / np.sqrt(n_weeks)


def test_stub_draws_score_live_and_dead_effects_in_their_worlds(model, monkeypatch):
    corpus, rows = _live_dead_corpus()
    _stub_joint_draws(model, monkeypatch)
    record = evaluate_rows(
        model,
        [corpus],
        rows,
        n_samples=2,
        seed=0,
        device="cpu",
        leg="holdout",
        heads=("joint",),
        per_world=True,
    )
    query = record["heads"]["joint"]["query"]
    context = record["heads"]["joint"]["context"]
    # Query errors are 1 or 3 (+/- 0.5 by week); context errors -1 or 3.
    expected = {
        "query": {
            "crps": 1.5,
            "mean_bias": 2.0,
            "mean_abs_error": 2.0,
            "demeaned_rmse": 0.5,
            "leakage_scaled_mae": 2.0,
            "draw_leakage_scaled_mae": 2.0,
        },
        "context": {
            "crps": 1.0,
            "mean_bias": 1.0,
            "mean_abs_error": 1.0,
            "demeaned_rmse": 0.5,
            "leakage_scaled_mae": 1.0,
            "draw_leakage_scaled_mae": 2.0,
        },
    }
    for window, values in (("query", query), ("context", context)):
        for effect in ("media", "control"):
            for stat in metrics.LIVE_ATTRIBUTION_KEYS:
                key = f"attribution_live_{effect}_{stat}"
                assert values[key] == pytest.approx(expected[window][stat]), (window, key)
            for stat in metrics.DEAD_ATTRIBUTION_KEYS:
                key = f"attribution_dead_{effect}_{stat}"
                assert values[key] == pytest.approx(expected[window][stat]), (window, key)
            # Each group exists in exactly one of the two worlds.
            live = f"joint/{window}/attribution_live_{effect}_crps"
            dead = f"joint/{window}/attribution_dead_{effect}_leakage_scaled_mae"
            assert record["metric_worlds"][live] == record["metric_worlds"][dead] == 1
        # Draw time means sit at truth + shift: [1.2, 2.8] misses the truth, [-0.6, 2.6] covers it.
        for suffix in ("", "_live", "_dead"):
            covered = values[f"timeavg_contribution_cov80{suffix}"]
            assert covered == (0.0 if window == "query" else 1.0), (window, suffix)
        assert values["contribution_cumulative_energy_score"] == pytest.approx(
            _cumulative_energy(QUERY_SHIFTS if window == "query" else CONTEXT_SHIFTS, 6)
        )
        # Residuals from the truth alternate by week; across draws adjacent weeks move together.
        assert values["contribution_truth_error_residual_lag1"] == pytest.approx(-1.0)
        assert values["contribution_adjacent_draw_correlation"] == pytest.approx(1.0)
        key = f"joint/{window}/contribution_posterior_centered_residual_lag1"
        assert record["undefined"][key] == "constant_series"

    truth_lag1, draw_lag1, n_channels = [], [], []
    for _, row in rows:
        batch = corpus_batch(corpus, [row])
        present = batch["active_c_mask"][0].bool().numpy()
        series = batch["targets_z"][0].double().numpy()[:, 1:4][:, present]
        wiggle = 0.5 - np.arange(series.shape[0]) % 2
        lag1 = [np.corrcoef(z[:-1], z[1:])[0, 1] for z in series.T]
        shifted = [np.corrcoef(z[:-1] + wiggle[:-1], z[1:] + wiggle[1:])[0, 1] for z in series.T]
        truth_lag1.append(np.mean(lag1))
        draw_lag1.append(np.mean(shifted))
        n_channels.append(int(present.sum()))
    assert query["contribution_truth_lag1"] == pytest.approx(np.mean(truth_lag1))
    assert query["contribution_draw_lag1"] == pytest.approx(np.mean(draw_lag1), rel=1e-5)
    assert record["counts"]["active_channels"] == pytest.approx(np.mean(n_channels))

    world_a, world_b = record["per_world"]
    assert (world_a["row"], world_b["row"]) == (rows[0][1], rows[1][1])
    live = "joint/query/timeavg_contribution_cov80_live"
    dead = "joint/query/timeavg_contribution_cov80_dead"
    assert (world_a["weights"][live], world_a["weights"][dead]) == (n_channels[0], 0)
    assert (world_b["weights"][live], world_b["weights"][dead]) == (0, n_channels[1])
    assert world_b["undefined"][live] == "no_live_effects"
    assert world_a["undefined"]["joint/query/attribution_dead_media_leakage_scaled_mae"] == (
        "no_dead_effects"
    )
    # Controls are live through zb: A's controls are dead, B's are live.
    assert world_a["undefined"]["joint/query/attribution_live_control_crps"] == "no_live_effects"
    assert world_b["undefined"]["joint/query/attribution_dead_control_leakage_scaled_mae"] == (
        "no_dead_effects"
    )
    assert record["counts"]["active_controls"] == pytest.approx(
        np.mean([corpus["covariate_active_mask"][row].sum() for _, row in rows])
    )


def test_stub_router_graphs_score_whole_graph_statistics(model, monkeypatch):
    corpus, rows = _live_dead_corpus()
    _stub_joint_draws(model, monkeypatch)
    record = evaluate_rows(
        model, [corpus], rows, n_samples=2, seed=0, device="cpu", leg="holdout", heads=("joint",)
    )
    layout = EdgeSlotLayout(3, 2, 2)
    channels, slots = [], []
    for _, row in rows:
        masks = [
            corpus[key][row]
            for key in ("treatment_active_mask", "covariate_active_mask", "latent_active_mask")
        ]
        channels.append(int(masks[0].sum()))
        slots.append(int(layout.active_slot_mask(*masks).sum()))
    graph = record["heads"]["joint_router"]["graph"]
    # Draw 0 is the true graph; draw 1 differs on every present channel's cy edge.
    assert graph["shd_mean"] == pytest.approx(np.mean(channels) / 2)
    assert graph["shd_median"] == pytest.approx(np.mean(channels) / 2)
    assert graph["exact_recovery"] == 0.5
    assert graph["normalized_shd_mean"] == pytest.approx(
        np.mean([c / (2 * s) for c, s in zip(channels, slots, strict=True)])
    )
    assert graph["n_distinct"] == 2.0
    assert graph["entropy"] == pytest.approx(math.log(2.0))
    assert graph["categorical_brier_score"] == pytest.approx(0.0, abs=1e-12)
    # Every present cy marginal is 1/2: none exceeds 1/2 and the ranking is uninformative.
    cy = record["heads"]["joint_router"]["edge"]["cy"]
    assert cy["null_fpr"] == 0.0
    assert cy["auprc"] == pytest.approx(cy["prevalence"])
    assert cy["prevalence"] == pytest.approx(channels[0] / sum(channels))


def test_stub_router_control_two_cycles_are_counted(model, monkeypatch):
    corpus, rows = _live_dead_corpus()
    assert all(corpus["covariate_active_mask"][row].sum() == 2 for _, row in rows)
    _stub_joint_draws(model, monkeypatch, zz_cycle=True)
    record = evaluate_rows(
        model, [corpus], rows, n_samples=2, seed=0, device="cpu", leg="holdout", heads=("joint",)
    )
    graph = record["heads"]["joint_router"]["graph"]
    # Draw 1 links both present controls both ways; draw 0 is the acyclic truth.
    assert graph["zz_two_cycle_rate"] == graph["zz_cycle_rate"] == 0.5


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_evaluation_matches_cpu_on_draw_free_scores(model, corpus, holdout):
    def run(device):
        return evaluate_rows(
            model.to(device),
            [corpus],
            holdout,
            n_samples=2,
            seed=5,
            device=device,
            leg="holdout",
            heads=("joint",),
        )

    cpu, mps = run("cpu"), run("mps")
    assert mps["counts"] == cpu["counts"]
    for et in STRUCT_EDGE_KEYS:
        on_cpu, on_mps = cpu["heads"]["structure_head"][et], mps["heads"]["structure_head"][et]
        for key in ("n", "n_pos", "prevalence"):
            assert on_mps[key] == on_cpu[key], (et, key)
        for key in ("log_score", "brier", "auroc", "auprc", "ece", "null_fpr"):
            if on_cpu[key] is None:
                assert on_mps[key] is None, (et, key)
            else:
                assert on_mps[key] == pytest.approx(on_cpu[key], abs=1e-5), (et, key)
