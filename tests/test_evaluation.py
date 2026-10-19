"""Scoring rules against hand values and the in-loop evaluation record contract."""

import json
import math

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.data.corpus import corpus_batch
from pymc_mfm.model.graph import STRUCT_EDGE_KEYS
from pymc_mfm.training import evaluation, metrics
from pymc_mfm.training.evaluation import (
    DECOMPOSITION_METRICS,
    EDGE_METRICS,
    GATE_METRICS,
    GRAPH_METRICS,
    MONITOR_METRIC,
    _MeanAccumulator,
    _score_decomposition,
    evaluate_rows,
    flatten_evaluation,
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
    acc = _MeanAccumulator(DECOMPOSITION_METRICS)
    _score_decomposition(
        acc,
        np.repeat(truth[None], 3, axis=0),
        truth,
        batch["active_c_mask"][0].bool().numpy(),
        batch["active_m_mask"][0].bool().numpy(),
        np.asarray(corpus["outcome_raw"][row], dtype=np.float64)[n_context:],
        np.asarray(corpus["outcome_noise"][row], dtype=np.float64)[n_context:],
        float(batch["sales_scale"][0]),
    )
    undefined = {}
    values = acc.finalize("joint/query", undefined)
    assert undefined == {}
    for key, value in values.items():
        if key.endswith("_cov80"):
            assert value == 1.0, key
        elif key == "closure_rel_err":
            assert value == pytest.approx(0.0, abs=1e-5)
        else:
            assert value == pytest.approx(0.0, abs=1e-12), key


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
    for key, reason in record["undefined"].items():
        assert reason in {"one_class", "no_active_slots"}, (key, reason)


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


def test_flatten_uses_slashes_and_names_monitor_and_gates(record):
    flat = flatten_evaluation(record)
    assert MONITOR_METRIC in flat
    assert math.isfinite(flat[MONITOR_METRIC])
    for gate in GATE_METRICS:
        assert gate in flat
    assert all(key.startswith("holdout/") for key in flat)
    assert all(value is None or isinstance(value, float) for value in flat.values())
    n_cy = record["heads"]["structure_head"]["cy"]["n"]
    assert flat["holdout/structure_head/cy/n"] == float(n_cy)
    assert "holdout/joint_router/graph/shd_mean" in flat
    assert "holdout/joint_router/edge/cy/auroc" in flat
    for key, reason in record["undefined"].items():
        assert flat[f"holdout/{key}"] is None, (key, reason)


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
    for metric in ("log_score", "brier", "auroc", "ece"):
        assert edge[metric] is None
        assert record["undefined"][f"structure_head/cy/{metric}"] == "nonfinite_in_1_slots"
    batch = corpus_batch(corpus, [holdout[0][1]])
    active = batch["active_c_mask"].bool()
    truth = batch["g_cy"][active]
    assert edge["n"] == truth.numel()
    assert edge["n_pos"] == int(truth.sum())
    assert edge["prevalence"] == pytest.approx(float(truth.mean()))
    json.dumps(record, allow_nan=False)


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
