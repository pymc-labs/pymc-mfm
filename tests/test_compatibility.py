"""Recorded-evaluation compatibility: re-aggregation, tolerance rules, None semantics, inputs."""

import copy
import math

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.model.model import MFM
from pymc_mfm.training import check_recorded_evaluation, compatibility
from pymc_mfm.training.evaluation import evaluate_rows, flatten_evaluation, metric_aggregation

SHARDS = (530001, 530002)
MIXED_SHARD = 530003
N_SAMPLES = 3
EVAL_SEED = 40
AGGREGATION = metric_aggregation(("joint",))
HEADLINE = "joint/query/decomposition_energy_score"
MASKS = ("treatment_active_mask", "covariate_active_mask", "latent_active_mask")


def _worlds(corpora):
    """Three worlds from two shards with 1 and 2 present channels and 3 and 6 query weeks."""
    picked = []
    for seed, corpus in corpora.items():
        channels = corpus["treatment_active_mask"].sum(axis=1)
        weeks = corpus["support_mask"].shape[1] - corpus["support_mask"].sum(axis=1)
        for count, query in ((1, 3), (2, 6), (2, 3), (1, 6)):
            rows = np.flatnonzero((channels == count) & (weeks == query))
            if rows.size and (count, query) not in {(c, q) for _, _, c, q in picked}:
                picked.append((seed, int(rows[0]), count, query))
    assert len({count for *_, count, _ in picked}) == 2 and len(picked) >= 3
    return [{"shard_seed": seed, "row": row} for seed, row, _, _ in picked[:3]]


def _evaluate(model, corpora, rows, seed):
    return evaluate_rows(
        model,
        corpora,
        rows,
        n_samples=N_SAMPLES,
        seed=seed,
        device="cpu",
        leg="holdout",
        heads=("joint",),
        per_world=True,
    )


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(3)
    model = MFM(
        ModelConfig(
            d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=2
        )
    ).eval()
    corpora = {seed: make_synthetic_corpus(seed=index) for index, seed in enumerate(SHARDS)}
    worlds = _worlds(corpora)
    order = list(dict.fromkeys(world["shard_seed"] for world in worlds))
    rows = [(order.index(world["shard_seed"]), world["row"]) for world in worlds]
    records = [
        _evaluate(model, [corpora[seed] for seed in order], rows, EVAL_SEED + 1 + replicate)
        for replicate in range(compatibility.MIN_REPLICATES)
    ]
    # The recorded values are the evaluator's own leg values. A tiny model leaves some
    # temporal keys undefined in only some replicates; such keys fail by rule, so the
    # recorded evaluation keeps the consistent ones.
    flats = [flatten_evaluation(record) for record in records]
    means = {}
    for key in AGGREGATION:
        values = [flat[f"holdout/{key}"] for flat in flats]
        if all(value is None for value in values):
            means[key] = None
        elif None not in values:
            means[key] = float(np.mean(values))
    return model, corpora, worlds, records, means


@pytest.fixture(scope="module")
def mixed(setup):
    """One evaluation of three worlds that differ in weeks, channels and controls."""
    corpus = make_synthetic_corpus(k=5, n_cells=8, seed=33)
    geometry = np.stack(
        [
            corpus["treatment_active_mask"].sum(axis=1),
            corpus["support_mask"].shape[1] - corpus["support_mask"].sum(axis=1),
            corpus["covariate_active_mask"].sum(axis=1),
        ],
        axis=1,
    )
    rows = [  # (present channels, query weeks, present controls)
        int(np.flatnonzero((geometry == wanted).all(axis=1))[0])
        for wanted in ((1, 3, 2), (3, 6, 1), (4, 6, 2))
    ]
    evaluation = _evaluate(setup[0], [corpus], [(0, row) for row in rows], EVAL_SEED + 1)
    return corpus, [{"shard_seed": MIXED_SHARD, "row": row} for row in rows], evaluation


def recorded(worlds, metrics, *, pooled=(), **overrides):
    return {
        "status": "recorded",
        "source": "terminal in-training evaluation",
        "step": 10,
        "leg": "holdout",
        "head": "joint",
        "eval_seed": EVAL_SEED,
        "n_worlds": len(worlds),
        "n_samples": N_SAMPLES,
        "flow_steps": 16,
        "worlds": worlds,
        "metrics": dict(metrics),
        "aggregation": {
            key: "slot_pooled" if key in pooled else AGGREGATION[key] for key in metrics
        },
        **overrides,
    }


def compare(setup, metrics, records=None, *, pooled=()):
    pytest.importorskip("scipy")  # The comparison's t and binomial quantiles.
    _, corpora, worlds, fixture_records, _ = setup
    record = recorded(worlds, metrics, pooled=pooled)
    _, _, geometry = compatibility._worlds(record, corpora)
    return compatibility._compare_recorded(record, records or fixture_records, geometry)


def replay(mixed, metrics, pooled=()):
    """Compare with replicates that all equal the mixed-world evaluation: no spread."""
    pytest.importorskip("scipy")
    corpus, worlds, evaluation = mixed
    record = recorded(worlds, metrics, pooled=pooled)
    _, _, geometry = compatibility._worlds(record, {MIXED_SHARD: corpus})
    records = [evaluation] * compatibility.MIN_REPLICATES
    return compatibility._compare_recorded(record, records, geometry)


def _with_values(records, key, values):
    """Copies of evaluation records whose leg value of ``key`` is ``values[r]``."""
    *path, name = key.split("/")
    patched = copy.deepcopy(records)
    for record, value in zip(patched, values, strict=True):
        node = record["heads"]
        for part in path:
            node = node[part]
        node[name] = value
    return patched


def _off(result, key, tolerances):
    """The value ``tolerances`` of ``key``'s tolerances above its replicate mean in ``result``."""
    entry = result["keys"][key]
    return entry["mean"] + tolerances * entry["tolerance"]


def test_check_replays_the_recorded_worlds_with_fresh_seeds(setup):
    pytest.importorskip("scipy")
    model, corpora, worlds, records, means = setup
    # Leg values under metric_aggregation() labels, "slot" keys included.
    model.provenance = {"training_evaluation": recorded(worlds, means)}
    result = check_recorded_evaluation(model, corpora)
    assert result["passed"] is True and result["reason"] is None
    assert result["seeds"] == [EVAL_SEED + 1 + r for r in range(8)]
    assert result["exact_failures"] == result["stochastic_failures"] == []
    assert result["keys"][HEADLINE]["replicates"] == [
        flatten_evaluation(record)[f"holdout/{HEADLINE}"] for record in records
    ]


def test_slot_pooled_values_are_compared_with_slot_pooled_replicates(setup):
    pytest.importorskip("scipy")
    model, corpora, worlds, records, means = setup
    keys = [f"joint/{window}/contribution_truth_lag1" for window in ("query", "context")]
    pooled = {}
    for key in keys:  # draw-independent: every replicate pools to the same value
        entries = [entry for entry in records[0]["per_world"] if entry["values"][key] is not None]
        weights = [entry["weights"][key] for entry in entries]
        assert len(set(weights)) == 2  # worlds with one and two present channels
        total = sum(entry["values"][key] * w for entry, w in zip(entries, weights, strict=True))
        pooled[key] = total / sum(weights)
        assert abs(pooled[key] - means[key]) > 1e-3
    model.provenance = {"training_evaluation": recorded(worlds, {**means, **pooled}, pooled=keys)}
    assert check_recorded_evaluation(model, corpora)["passed"] is True
    # The same values labelled "slot" meet the equal-world leg values.
    assert compare(setup, {**means, **pooled})["exact_failures"] == keys


def test_one_event_where_it_weighs_most_is_within_tolerance(mixed):
    corpus, worlds, evaluation = mixed
    layout = EdgeSlotLayout(5, 2, 2)
    slots = [
        int(layout.active_slot_mask(*(corpus[mask][world["row"]] > 0 for mask in MASKS)).sum())
        for world in worlds
    ]
    cy = evaluation["heads"]["joint_router"]["edge"]["cy"]
    positives, negatives = cy["n_pos"], cy["n"] - cy["n_pos"]
    s = N_SAMPLES
    # By hand, the step of one event where it weighs most: a week of the shortest
    # window (3 query, 6 context weeks), a draw, a pair of the 3-channel world (cc) or
    # of a 2-control world (zz), a slot of the world with the fewest active slots, or
    # one of the negative cy slots, over the defined worlds; the equal-world slot means
    # weigh the 1 slot x 3 weeks of the 1-channel world, and the 2 controls x 3 weeks
    # of the first world tied with the 1 control x 6 weeks of the second, most.
    # One slot of one draw moves a draw's graph distance by one (the median of 3
    # distances by a whole one) and the count of distinct draws by one; co-moments
    # are multiples of 1/s**2 per slot pair; one draw moves a cy slot's frequency by
    # 1/s, so AUROC by up to 1/(2 min(n_pos, n_neg)) and AUPRC by less than 1/n_pos.
    equal_world = {
        "joint/query/baseline_intrinsic_cov80": 1 / (3 * 3),
        "joint/context/baseline_intrinsic_cov80": 1 / (3 * 6),
        "joint/query/contribution_cov80": 1 / (3 * (1 * 3)),
        "joint/query/control_contribution_cov80": 1 / (3 * 6),
        "joint_router/graph/exact_recovery": 1 / (3 * s),
        "joint_router/graph/cc_cycle_rate": 1 / (2 * s),
        "joint_router/graph/cc_two_cycle_rate": 1 / (2 * s * math.comb(3, 2)),
        "joint_router/graph/zz_two_cycle_rate": 1 / (2 * s * math.comb(2, 2)),
        "joint_router/graph/normalized_shd_mean": 1 / (3 * s * min(slots)),
        "joint_router/graph/shd_mean": 1 / (3 * s),
        "joint_router/graph/shd_median": 1 / 3,
        "joint_router/graph/n_distinct": 1 / 3,
        "joint_router/graph/pairwise_comoment_max_abs_err": 1 / (3 * s**2),
        "joint_router/graph/pairwise_comoment_abs_err": 1 / (3 * s**2 * math.comb(min(slots), 2)),
        "joint_router/edge/cy/null_fpr": 1 / negatives,
        "joint_router/edge/cy/auroc": 1 / (2 * min(positives, negatives)),
        "joint_router/edge/cy/auprc": 1 / positives,
    }
    # Pooled over the 1 + 3 + 4 channel slots of all worlds.
    slot_pooled = {
        "joint/query/contribution_cov80": 1 / (8 * 3),
        "joint/query/timeavg_contribution_cov80": 1 / 8,
    }
    for steps, pooled in ((equal_world, ()), (slot_pooled, tuple(slot_pooled))):
        base = {
            key: compatibility._replicate_value(
                key, "slot_pooled" if key in pooled else AGGREGATION[key], evaluation
            )
            for key in steps
        }
        for factor, passing in ((1.0, True), (1.01, False)):
            metrics = {key: base[key] + factor * step for key, step in steps.items()}
            result = replay(mixed, metrics, pooled)
            for key, step in steps.items():
                assert result["keys"][key]["passing"] is passing, key
                assert result["keys"][key]["tolerance"] == pytest.approx(step)


def test_the_median_distance_moves_half_a_slot_at_an_even_draw_count(setup, mixed):
    pytest.importorskip("scipy")
    corpus, worlds, _ = mixed
    evaluation = evaluate_rows(
        setup[0],
        [corpus],
        [(0, world["row"]) for world in worlds],
        n_samples=4,
        seed=EVAL_SEED + 1,
        device="cpu",
        leg="holdout",
        heads=("joint",),
        per_world=True,
    )
    key = "joint_router/graph/shd_median"
    # The median of 4 distances is the mean of the middle two, so one slot of one draw
    # moves a world's median by 1/2 and the equal-world mean of the 3 worlds by 1/6.
    base = compatibility._replicate_value(key, AGGREGATION[key], evaluation)
    for factor, passing in ((1.0, True), (1.01, False)):
        record = recorded(worlds, {key: base + factor / 6}, n_samples=4)
        _, _, geometry = compatibility._worlds(record, {MIXED_SHARD: corpus})
        entry = compatibility._compare_recorded(record, [evaluation] * 8, geometry)["keys"][key]
        assert entry["tolerance"] == pytest.approx(1 / 6) and entry["passing"] is passing


def test_a_key_is_graded_by_how_far_it_lies_from_its_replicates():
    rule = compatibility._compare
    step = 1 / 40
    # Without spread, or with a spread far below one event, only one event is allowed.
    for values in ([0.5] * 8, [0.5] * 7 + [0.5 + 1e-9]):
        for offset, passing in ((1.0, True), (1.01, False)):
            entry = rule("k", "stochastic", 0.5 + offset * step, values, False, step, 3.5, 5.0)
            assert entry["passing"] is passing
    # Without spread or event step, both intervals are EXACT_TOLERANCE wide.
    for offset, reason in (
        (0.9e-5, None),
        (2e-5, "outside_family_interval"),
        (3.1e-5, "gross_mismatch"),
    ):
        entry = rule("k", "stochastic", 0.5 + offset, [0.5] * 8, False, None, 3.5, 5.0)
        assert entry["reason"] == reason
        assert entry["tolerance"] == entry["family_tolerance"] == compatibility.EXACT_TOLERANCE
    # The floor applies only without spread: a varying key without a step keeps its
    # spread tolerance below EXACT_TOLERANCE, like A's co-moment mean (2.4e-7 +/- 1e-7).
    tiny = [2.4e-7 + 1e-7 * x for x in (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.25)]
    entry = rule("k", "stochastic", np.mean(tiny) + 1e-6, tiny, False, None, 3.5, 5.0)
    assert entry["tolerance"] == pytest.approx(3.5 * np.std(tiny, ddof=1) * math.sqrt(9 / 8))
    assert entry["passing"] is False
    spread = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    root = np.std(spread, ddof=1) * math.sqrt(1 + 1 / 8)
    tolerance, family = 3.5 * root, 5.0 * root
    # The 99% interval, the 99.9% interval of the family rule, and GROSS_FACTOR tolerances.
    for distance, reason in (
        (0.99 * tolerance, None),
        (1.01 * tolerance, "outside_replicate_interval"),
        (0.99 * family, "outside_replicate_interval"),
        (1.01 * family, "outside_family_interval"),
        (2.99 * tolerance, "outside_family_interval"),
        (3.01 * tolerance, "gross_mismatch"),
    ):
        entry = rule("k", "stochastic", np.mean(spread) + distance, spread, False, None, 3.5, 5.0)
        assert entry["reason"] == reason and entry["passing"] is (reason is None)
        assert entry["tolerance"] == pytest.approx(tolerance)
        assert entry["family_tolerance"] == pytest.approx(family)


def test_a_key_without_replicate_spread_fails_one_family_within_three_steps(setup):
    *_, records, means = setup
    # Every replicate draws 3 distinct graphs in each of the 3 worlds; one draw changes a
    # world's count by at most one, so the step is 1/3. Padding leakage is 0 in every
    # replicate and has no step: its floor is EXACT_TOLERANCE.
    discrete = "joint_router/graph/n_distinct"
    constant = "joint/query/contribution_padding_leakage_scaled_mae"
    for key in (discrete, constant):
        assert len({flatten_evaluation(record)[f"holdout/{key}"] for record in records}) == 1
    for key, step, family in (
        (discrete, 1 / 3, "joint_router/graph/spread"),
        (constant, compatibility.EXACT_TOLERANCE, constant),
    ):
        within = compare(setup, {**means, key: means[key] + 0.99 * step})
        assert within["passed"] is True and within["failing_families"] == []
        assert within["keys"][key]["tolerance"] == pytest.approx(step)
        # Outside the floor but within GROSS_FACTOR of it: one failing family.
        outside = compare(setup, {**means, key: means[key] + 2 * step})
        assert outside["passed"] is True and outside["failing_families"] == [family]
        far = compare(setup, {**means, key: means[key] + 3.5 * step})
        assert far["passed"] is False and far["reason"] == "gross_failures"
        assert far["gross_failures"] == [key]
    structural = compare(setup, {**means, constant: means[constant] + 1e-3})
    assert structural["reason"] == "gross_failures"


def test_undefined_values_pass_only_when_undefined_everywhere():
    def rule(kind, recorded_value, values, nonfinite=False):
        return compatibility._compare("k", kind, recorded_value, values, nonfinite, None, 3.5, 5.0)

    assert rule("stochastic", None, [None] * 8)["passing"] is True
    assert rule("stochastic", None, [1.0] * 8)["reason"] == "recorded_undefined"
    assert rule("exact", 1.0, [1.0] * 7 + [None])["reason"] == "replicate_undefined"
    assert rule("stochastic", 1.0, [1.0] * 8, nonfinite=True)["reason"] == "replicate_nonfinite"


def test_exact_keys_and_undefined_replicates_fail_the_verdict(setup):
    *_, records, means = setup
    assert compare(setup, means)["passed"] is True

    exact = "structure_head/cy/brier"
    close = compare(setup, {**means, exact: means[exact] + 0.9e-5})
    far = compare(setup, {**means, exact: means[exact] + 1.1e-5})
    assert close["passed"] is True
    assert far["passed"] is False and far["exact_failures"] == [exact]

    informational = "joint_router/edge/cy/log_score"
    ignored = compare(setup, {**means, informational: means[informational] + 100.0})
    assert ignored["passed"] is True and informational in ignored["informational"]

    broken = copy.deepcopy(records)
    broken[3]["undefined"]["joint/query/closure_rel_err"] = "nonfinite_in_1_worlds"
    flagged = compare(setup, means, broken)
    assert flagged["keys"]["joint/query/closure_rel_err"]["reason"] == "replicate_nonfinite"
    assert flagged["passed"] is False
    assert flagged["must_pass_failures"] == ["joint/query/closure_rel_err"]


def test_the_binomial_allowance_counts_failing_families(setup):
    *_, means = setup
    base = compare(setup, means)
    from scipy.stats import binom
    from scipy.stats import t as student_t

    compared = [
        key
        for key, entry in base["keys"].items()
        if entry["class"] == "stochastic" and entry["tolerance"] is not None
    ]
    families = {compatibility._family(key) for key in compared}
    # k counts the families with a compared stochastic key, not the keys.
    assert base["n_families"] == len(families) < len(compared)
    allowance = base["allowed_family_failures"]
    assert allowance == int(binom.ppf(0.99, len(families), 0.01)) >= 2
    # The t levels are 0.995 and 0.9995 on R - 1 = 7 dof.
    headline = base["keys"][HEADLINE]
    root = np.std(headline["replicates"], ddof=1) * math.sqrt(9 / 8)
    assert headline["tolerance"] == pytest.approx(student_t.ppf(0.995, 7) * root)
    assert headline["family_tolerance"] == pytest.approx(student_t.ppf(0.9995, 7) * root)

    # Outside their 99% and inside their 99.9% intervals; two keys of one family count once.
    pair = ["joint/query/contribution_crps", "joint/query/contribution_scaled_mae"]
    components = ("baseline_intrinsic", "control_contribution", "confounder_total", "indirect_cc")
    others = [f"joint/query/{component}_crps" for component in components][:allowance]
    for key in (*pair, *others):
        assert base["keys"][key]["family_tolerance"] > 1.2 * base["keys"][key]["tolerance"]

    def outside(*chosen):
        return compare(setup, {**means, **{key: _off(base, key, 1.2) for key in chosen}})

    within = outside(*pair, *others[:-1])
    assert within["passed"] is True and len(within["failing_families"]) == allowance
    worse = outside(*pair, *others)
    assert worse["passed"] is False and worse["reason"] == "allowance_exceeded"
    assert worse["failing_families"] == sorted(
        {compatibility._family(key) for key in (*pair, *others)}
    )


def test_near_duplicate_keys_inside_their_99_9_intervals_fail_one_family(setup):
    *_, records, means = setup
    # Two keys of one family whose replicates are identical, as near-duplicate keys
    # (replicate correlations of 0.98 to 1.00 in A and B) nearly are.
    first, twin = "joint/query/contribution_crps", "joint/query/contribution_scaled_mae"
    values = [flatten_evaluation(record)[f"holdout/{first}"] for record in records]
    twins = _with_values(records, twin, values)
    base = compare(setup, {**means, twin: means[first]}, twins)
    # Both between their 99% and 99.9% bounds: one failing family, within the allowance.
    shifted = {first: _off(base, first, 1.2), twin: _off(base, twin, 1.2)}
    result = compare(setup, {**means, **shifted}, twins)
    assert result["passed"] is True
    assert set(result["stochastic_failures"]) == {first, twin}
    assert result["failing_families"] == ["joint/query/contribution"]


FAMILY_PAIRS = {
    "joint/query/contribution": (
        "joint/query/contribution_crps",
        "joint/query/contribution_scaled_mae",
    ),
    "joint/context/temporal": (
        "joint/context/contribution_draw_lag1",
        "joint/context/contribution_lag1_tracking_abs_error",
    ),
    "joint/query/latent_sales": ("joint/query/closure_rel_err", "joint/query/latent_sales_r2"),
    "joint/query/attribution_live_media": (
        "joint/query/attribution_live_media_crps",
        "joint/query/attribution_live_media_mean_abs_error",
    ),
    "joint/query/timeavg_coverage": (
        "joint/query/timeavg_contribution_cov80",
        "joint/query/timeavg_contribution_cov80_dead",
    ),
    "joint_router/graph/distance": (
        "joint_router/graph/shd_mean",
        "joint_router/graph/energy_score",
    ),
    "joint_router/graph/spread": (
        "joint_router/graph/n_distinct",
        "joint_router/graph/pairwise_comoment_abs_err",
    ),
    "joint_router/graph/cc_cycles": (
        "joint_router/graph/cc_two_cycle_rate",
        "joint_router/graph/cc_cycle_rate",
    ),
    "joint_router/edge/cy": ("joint_router/edge/cy/brier", "joint_router/edge/cy/auroc"),
}


@pytest.mark.parametrize("family", FAMILY_PAIRS)
def test_two_keys_of_one_family_outside_their_99_9_intervals_fail(setup, family):
    *_, means = setup
    base = compare(setup, means)
    keys = FAMILY_PAIRS[family]
    # Twice the 99% tolerance lies outside the 99.9% interval and inside the gross limit.
    for key in keys:
        assert base["keys"][key]["family_tolerance"] < 2 * base["keys"][key]["tolerance"]
    one = compare(setup, {**means, keys[0]: _off(base, keys[0], 2.0)})
    assert one["passed"] is True and one["failing_families"] == [family]
    both = compare(setup, {**means, **{key: _off(base, key, 2.0) for key in keys}})
    assert both["passed"] is False and both["reason"] == "family_failures"
    assert {name: set(members) for name, members in both["family_failures"].items()} == {
        family: set(keys)
    }


def test_one_key_far_outside_its_interval_fails_the_check(setup):
    *_, means = setup
    base = compare(setup, means)
    # A family of its own counts once against the allowance up to GROSS_FACTOR
    # tolerances; beyond, the gross rule fails it and it is not counted again.
    key = "joint/query/contribution_cumulative_energy_score"
    for tolerances, reason, failing in ((2.9, None, [key]), (10.0, "gross_failures", [])):
        result = compare(setup, {**means, key: _off(base, key, tolerances)})
        assert result["reason"] == reason and result["failing_families"] == failing
    assert result["gross_failures"] == [key]
    # The headline must pass, even inside that limit, and a must-pass failure never
    # counts against the allowance, inside the gross limit or beyond it.
    for tolerances in (1.2, 10.0):
        headline = compare(setup, {**means, HEADLINE: _off(base, HEADLINE, tolerances)})
        assert headline["reason"] == "must_pass_failures"
        assert headline["must_pass_failures"] == [HEADLINE] and headline["failing_families"] == []


def test_the_first_failing_rule_names_the_reason(setup):
    *_, records, means = setup
    base = compare(setup, means)
    exact = "structure_head/cy/brier"
    pair = FAMILY_PAIRS["joint/query/contribution"]
    components = ("baseline_intrinsic", "control_contribution", "confounder_total")
    components += ("indirect_cc", "indirect_zc")
    others = [f"joint/query/{component}_crps" for component in components]
    others = others[: base["allowed_family_failures"] + 1]
    assert len(others) == base["allowed_family_failures"] + 1

    def reason(offsets, exact_offset=0.0):
        metrics = {**means, **{key: _off(base, key, value) for key, value in offsets.items()}}
        return compare(setup, {**metrics, exact: means[exact] + exact_offset})["reason"]

    # Each pair of adjacent rules failing together: the earlier one names the reason.
    assert reason({HEADLINE: 1.2}, exact_offset=1.1e-5) == "exact_failures"
    assert reason({HEADLINE: 10.0}) == "must_pass_failures"  # also a gross failure
    assert reason(dict.fromkeys(pair, 10.0)) == "gross_failures"  # also a family failure
    both = {**dict.fromkeys(pair, 2.0), **dict.fromkeys(others, 1.2)}
    assert reason(both) == "family_failures"  # and allowance + 1 failing families
    # An exact key undefined in one replicate fails, but alone it compares nothing.
    undefined = _with_values(records, exact, [means[exact]] * 7 + [None])
    lone = compare(setup, {exact: means[exact]}, undefined)
    assert lone["reason"] == "no_compared_keys" and lone["exact_failures"] == [exact]


def test_a_check_that_compares_no_key_fails(setup):
    model, corpora, worlds, _, means = setup
    informational = "joint_router/edge/cy/log_score"
    for metrics in ({}, {informational: means[informational]}):
        result = compare(setup, metrics)
        assert result["passed"] is False and result["reason"] == "no_compared_keys"

    # A single-channel world leaves its channel cycle rates undefined in every replicate.
    lone = next(
        world
        for world in worlds
        if corpora[world["shard_seed"]]["treatment_active_mask"][world["row"]].sum() == 1
    )
    evaluation = _evaluate(model, [corpora[lone["shard_seed"]]], [(0, lone["row"])], EVAL_SEED)
    rates = {f"joint_router/graph/cc_{rate}": None for rate in ("two_cycle_rate", "cycle_rate")}
    record = recorded([lone], rates)
    _, _, geometry = compatibility._worlds(record, corpora)
    records = [evaluation] * compatibility.MIN_REPLICATES
    result = compatibility._compare_recorded(record, records, geometry)
    assert [entry["reason"] for entry in result["keys"].values()] == ["undefined_as_recorded"] * 2
    assert result["passed"] is False and result["reason"] == "no_compared_keys"


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("structure_head/cy/auprc", "exact"),
        ("counts/active_channels", "exact"),
        ("joint_router/edge/zb/prevalence", "exact"),
        ("joint/context/contribution_truth_lag1", "exact"),
        ("joint_router/edge/cy/log_score", "informational"),
        ("joint_router/edge/cy/null_fpr", "stochastic"),
        ("joint/query/contribution_draw_lag1", "stochastic"),
    ],
)
def test_key_classes_follow_draw_dependence(key, kind):
    assert compatibility.key_class(key) == kind


def _record_key(record, key, value, label):
    record["metrics"][key] = value
    record["aggregation"][key] = label


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.update(flow_steps=64), id="flow-steps"),
        pytest.param(lambda r: r["worlds"].append({"shard_seed": 1, "row": 0}), id="n-worlds"),
        pytest.param(lambda r: r["worlds"][0].update(shard_seed=1), id="missing-shard"),
        pytest.param(
            lambda r: _record_key(r, "conditional/query/closure_rel_err", 0.1, "world"),
            id="unknown-key",
        ),
        pytest.param(
            lambda r: r["aggregation"].update({"joint/query/contribution_crps": "world"}),
            id="aggregation",
        ),
        pytest.param(
            lambda r: _record_key(r, "joint/query/closure_rel_err", 0.1, "slot_pooled"),
            id="slot-pooled-world-key",
        ),
        pytest.param(lambda r: r.pop("eval_seed"), id="missing-field"),
    ],
)
def test_invalid_recorded_evaluations_raise_before_evaluating(setup, mutate):
    model, corpora, worlds, _, _ = setup
    record = copy.deepcopy(recorded(worlds, {"joint/query/contribution_crps": 0.1}))
    mutate(record)
    model.provenance = {"training_evaluation": record}
    with pytest.raises(ValueError):
        check_recorded_evaluation(model, corpora)


def test_check_needs_a_recorded_evaluation_and_enough_replicates(setup):
    model, corpora, worlds, _, means = setup
    model.provenance = {}
    with pytest.raises(ValueError):
        check_recorded_evaluation(model, corpora)
    model.provenance = {"training_evaluation": recorded(worlds, means)}
    with pytest.raises(ValueError):
        check_recorded_evaluation(model, corpora, replicates=7)
