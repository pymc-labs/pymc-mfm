"""Evaluation scoring primitives and per-world family scorers against hand values."""

import math

import numpy as np
import pytest

from pymc_mktfm.training import metrics


def test_auprc_steps_once_per_distinct_score():
    # Sorted scores 0.9 | 0.8, 0.8 | 0.1 give (R, P) = (1/2, 1), (1, 2/3), (1, 1/2).
    for labels in ([1, 0, 1, 0], [1, 1, 0, 0]):
        value, reason = metrics.auprc([0.9, 0.8, 0.8, 0.1], labels)
        assert value == pytest.approx(5 / 6) and reason is None
    assert metrics.auprc([0.5] * 4, [1, 0, 1, 0]) == (pytest.approx(0.5), None)
    assert metrics.auprc([0.2, 0.9], [1, 1]) == (None, "one_class")
    assert metrics.auprc([0.2, 0.9], [0, 0]) == (None, "one_class")


def test_null_fpr_counts_negatives_strictly_above_one_half():
    value, reason = metrics.null_fpr([0.5, 0.51, 0.2, 0.9], [0, 0, 0, 1])
    assert value == pytest.approx(1 / 3) and reason is None
    assert metrics.null_fpr([0.7, 0.9], [1, 1]) == (None, "no_negative_slots")


def test_reliability_table_lists_only_occupied_bins():
    rows = metrics.reliability_table([0.05, 0.15, 0.15, 1.0], [0, 1, 0, 1])
    assert [row["bin"] for row in rows] == [0, 1, 9]
    assert [row["count"] for row in rows] == [1, 2, 1]
    assert [row["mean_probability"] for row in rows] == pytest.approx([0.05, 0.15, 1.0])
    assert [row["observed_frequency"] for row in rows] == [0.0, 0.5, 1.0]
    assert [(row["lower"], row["upper"]) for row in rows] == [
        pytest.approx((0.0, 0.1)),
        pytest.approx((0.1, 0.2)),
        pytest.approx((0.9, 1.0)),
    ]


def test_threshold_table_predicts_at_or_above_each_threshold():
    rows = metrics.threshold_table([0.05, 0.3, 0.6, 0.9], [0, 1, 0, 1], thresholds=(0.3, 0.95))
    assert rows[0] == {
        "threshold": 0.3,
        "tp": 2,
        "fp": 1,
        "tn": 1,
        "fn": 0,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "fpr": 0.5,
    }
    assert (rows[1]["tp"], rows[1]["fp"], rows[1]["tn"], rows[1]["fn"]) == (0, 0, 2, 2)
    assert rows[1]["precision"] is None and rows[1]["recall"] == 0.0 and rows[1]["fpr"] == 0.0


def _fitted(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    means, _, ends = metrics._pava(values * weights, weights)
    return np.repeat(means, np.diff(ends, prepend=0)), means


def test_pava_matches_scipy_isotonic_regression_with_ties_and_weights():
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(11)
    for trial in range(300):
        n = int(rng.integers(1, 80))
        # Half the trials draw from a coarse grid, so tied values and tied means abound.
        values = rng.integers(0, 4, n) / 3 if trial % 2 else rng.normal(size=n)
        weights = rng.uniform(0.05, 5.0, n)
        fitted, means = _fitted(values, weights)
        expected = optimize.isotonic_regression(values, weights=weights).x
        np.testing.assert_allclose(fitted, expected, rtol=0, atol=1e-12)
        assert np.all(np.diff(fitted) >= 0)


def test_isotonic_calibration_pools_violators_and_tied_probabilities():
    # Ties at 0.1 hold one positive in two; the 0.5 miss violates order and pools with them.
    rows = metrics.isotonic_calibration([0.9, 0.1, 0.5, 0.1], [1, 0, 0, 1])
    assert rows == [
        {
            "lower": 0.1,
            "upper": 0.5,
            "count": 3,
            "n_pos": 1,
            "mean_probability": pytest.approx(0.7 / 3),
            "calibrated_probability": pytest.approx(1 / 3),
        },
        {
            "lower": 0.9,
            "upper": 0.9,
            "count": 1,
            "n_pos": 1,
            "mean_probability": 0.9,
            "calibrated_probability": 1.0,
        },
    ]
    with pytest.raises(ValueError):
        metrics.isotonic_calibration([], [])
    # Pooling is exact on counts: the fit's two halves both hold half positives (6/12 and
    # 15/30), which float means (0.5 and 0.5000000000000001) would leave as two blocks.
    p = np.repeat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [6, 6, 7, 7, 11, 5])
    y = np.concatenate(
        [np.arange(count) < hits for count, hits in zip([6, 6, 7, 7, 11, 5], [3, 3, 5, 3, 6, 1])]
    )
    [row] = metrics.isotonic_calibration(p, y)
    assert (row["count"], row["n_pos"], row["calibrated_probability"]) == (42, 21, 0.5)


def test_isotonic_calibration_is_the_scipy_fit_over_distinct_probabilities():
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(12)
    for _ in range(100):
        p = np.round(rng.uniform(size=int(rng.integers(2, 120))), 1)
        y = (rng.uniform(size=p.size) < p).astype(float)
        values, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
        rates = np.bincount(inverse, weights=y) / counts
        expected = optimize.isotonic_regression(rates, weights=counts).x
        rows = metrics.isotonic_calibration(p, y)
        calibrated = np.empty(values.size)
        for row in rows:
            calibrated[(values >= row["lower"]) & (values <= row["upper"])] = row[
                "calibrated_probability"
            ]
        np.testing.assert_allclose(calibrated, expected, rtol=0, atol=1e-12)
        assert sum(row["count"] for row in rows) == p.size
        assert sum(row["n_pos"] for row in rows) == y.sum()
        assert all(a["upper"] < b["lower"] for a, b in zip(rows, rows[1:], strict=False))
        assert np.all(np.diff([row["calibrated_probability"] for row in rows]) > 0)


def test_brier_decomposition_is_exact_and_its_components_are_nonnegative():
    rng = np.random.default_rng(13)
    for _ in range(100):
        p = rng.uniform(size=int(rng.integers(2, 200)))
        y = (rng.uniform(size=p.size) < rng.uniform() * p + 0.3 * rng.uniform()).astype(float)
        split = metrics.brier_decomposition(p, y)
        assert split["mcb"] - split["dsc"] + split["unc"] == pytest.approx(
            metrics.brier(p, y), rel=0, abs=1e-12
        )
        assert split["unc"] == pytest.approx(y.mean() * (1 - y.mean()), rel=0, abs=1e-15)
        assert min(split["mcb"], split["dsc"]) >= -1e-12
    # One class: nothing to discriminate, so the whole Brier score is miscalibration.
    assert metrics.brier_decomposition([0.2, 0.3], [0, 0]) == {
        "mcb": pytest.approx(0.065),
        "dsc": 0.0,
        "unc": 0.0,
    }


def test_miscalibration_vanishes_for_calibrated_probabilities_only():
    rng = np.random.default_rng(14)
    p = rng.uniform(size=20_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    # The isotonic fit itself is calibrated exactly against y.
    rows = metrics.isotonic_calibration(p, y)
    recalibrated = np.empty(p.size)
    for row in rows:
        recalibrated[(p >= row["lower"]) & (p <= row["upper"])] = row["calibrated_probability"]
    assert metrics.brier_decomposition(recalibrated, y)["mcb"] == pytest.approx(0.0, abs=1e-12)
    # True probabilities are calibrated up to sampling noise; overconfident ones are not.
    calibrated = metrics.brier_decomposition(p, y)["mcb"]
    logit = np.log(p) - np.log1p(-p)
    overconfident = metrics.brier_decomposition(1 / (1 + np.exp(-3 * logit)), y)["mcb"]
    assert calibrated < 0.002
    assert overconfident > max(0.015, 10 * calibrated)


def test_one_class_notice_names_the_missing_class():
    assert metrics.one_class_notice(4, 2) is None
    assert metrics.one_class_notice(4, 0) == "no positive edges — calibration not evidenced"
    assert metrics.one_class_notice(4, 4) == "no negative edges — calibration not evidenced"


def test_correlations_are_undefined_for_constant_or_short_series():
    assert metrics.finite_correlation([1, 2, 3], [3, 2, 1]) == (pytest.approx(-1.0), None)
    assert metrics.finite_correlation([1, 1, 1], [1, 2, 3]) == (None, "constant_series")
    # Pairs (1,3), (3,2), (2,4): deviations (-1, 1, 0) and (0, -1, 1).
    assert metrics.lag1_correlation([1, 3, 2, 4]) == (pytest.approx(-0.5), None)
    assert metrics.lag1_correlation([2, 2, 2, 2]) == (None, "constant_series")
    assert metrics.lag1_correlation([1.0, 2.0]) == (None, "needs_t_ge_3")
    assert metrics.lag1_correlation([1.0]) == (None, "needs_t_ge_3")
    with pytest.raises(ValueError):
        metrics.finite_correlation([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])


def _temporal_example():
    # Slot 0: truth (1, 3, 2, 4) and draws (1, 2, 1, 2), (2, 4, 6, 8).
    # Slot 1: constant truth and draws (0, 1, 0, 1), (5, 5, 5, 5).
    truth = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    draws = np.array(
        [
            [[1.0, 0.0], [2.0, 1.0], [1.0, 0.0], [2.0, 1.0]],
            [[2.0, 5.0], [4.0, 5.0], [6.0, 5.0], [8.0, 5.0]],
        ]
    )
    return draws, truth


def test_temporal_scores_average_the_defined_slots():
    draws, truth = _temporal_example()
    world = metrics.temporal_scores(draws, truth, prefix="c_")
    expected = {
        "c_truth_lag1": (-0.5, 1),  # slot 1 truth is constant
        "c_draw_lag1": (np.mean([np.mean([-1.0, 1.0]), -1.0]), 2),
        "c_lag1_tracking_abs_error": (0.5, 1),
        "c_truth_error_residual_lag1": (np.mean([0.5, -1.0]), 2),
        "c_posterior_centered_residual_lag1": (np.mean([11 / 13, -1.0]), 2),
        "c_adjacent_draw_correlation": (1.0, 2),
    }
    assert world.undefined == {}
    for key, (value, weight) in expected.items():
        assert world.values[key] == pytest.approx(value), key
        assert world.weights[key] == weight, key


def test_temporal_scores_name_structurally_undefined_values():
    draws, truth = _temporal_example()
    lag1_keys = [key for key in metrics.TEMPORAL_KEYS if key != "adjacent_draw_correlation"]

    two_weeks = metrics.temporal_scores(draws[:, :2], truth[:2])
    assert all(two_weeks.undefined[key] == "needs_t_ge_3" for key in lag1_keys)
    assert two_weeks.values["adjacent_draw_correlation"] == pytest.approx(1.0)

    one_week = metrics.temporal_scores(draws[:, :1], truth[:1])
    assert all(one_week.undefined[key] == "needs_t_ge_3" for key in lag1_keys)
    assert one_week.undefined["adjacent_draw_correlation"] == "needs_t_ge_2"

    one_draw = metrics.temporal_scores(draws[:1], truth)
    assert one_draw.undefined == {
        "posterior_centered_residual_lag1": "needs_n_samples_ge_2",
        "adjacent_draw_correlation": "needs_n_samples_ge_2",
    }
    assert one_draw.values["draw_lag1"] == pytest.approx(-1.0)  # slot 1's only draw is constant

    constant = metrics.temporal_scores(np.zeros((2, 4, 2)), np.zeros((4, 2)))
    assert set(constant.undefined.values()) == {"constant_series"}

    broken = draws.copy()
    broken[1, 2, 1] = np.nan
    assert set(metrics.temporal_scores(broken, truth).undefined.values()) == {metrics.NONFINITE}


def test_attribution_scores_effects_before_averaging():
    truth = np.array([[0.0, 1.0], [6.0, 1.0]])
    draws = np.array([[[1.0, -1.0], [3.0, -1.0]], [[1.0, 1.0], [3.0, 1.0]]])
    # Errors of the posterior mean: effect 0 (1, -3), effect 1 (-1, -1). Per-effect
    # biases are -1 and -1, so |bias| averages to 1 while mean |error| would be 1.5.
    stats = metrics.attribution_stats(draws, truth)
    assert stats == pytest.approx(
        {"crps": 1.25, "mean_bias": -1.0, "mean_abs_error": 1.0, "demeaned_rmse": 1.0}
    )
    dead = metrics.leakage_stats(np.array([[[1.0], [1.0]], [[-3.0], [1.0]]]), np.zeros((2, 1)))
    assert dead == pytest.approx({"leakage_scaled_mae": 1.0, "draw_leakage_scaled_mae": 1.5})


def test_attribution_family_splits_live_and_dead_effects():
    truth = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    draws = np.array([[[1.0, -1.0, 1.0], [3.0, -1.0, 1.0]], [[1.0, 1.0, -3.0], [3.0, 1.0, 1.0]]])
    world = metrics.attribution_scores(
        draws, truth, np.array([1, 1, 0]), live_prefix="live_", dead_prefix="dead_"
    )
    assert world.undefined == {}
    assert world.values["live_mean_bias"] == pytest.approx(0.5)
    assert world.values["dead_leakage_scaled_mae"] == pytest.approx(1.0)
    assert world.values["dead_draw_leakage_scaled_mae"] == pytest.approx(1.5)

    all_live = metrics.attribution_scores(
        draws, truth, np.ones(3), live_prefix="live_", dead_prefix="dead_"
    )
    assert set(all_live.undefined) == {"dead_leakage_scaled_mae", "dead_draw_leakage_scaled_mae"}
    assert set(all_live.undefined.values()) == {"no_dead_effects"}
    all_dead = metrics.attribution_scores(
        draws, truth, np.zeros(3), live_prefix="live_", dead_prefix="dead_"
    )
    assert {all_dead.undefined[f"live_{key}"] for key in metrics.LIVE_ATTRIBUTION_KEYS} == {
        "no_live_effects"
    }
    no_graph = metrics.attribution_scores(draws, truth, None, live_prefix="l_", dead_prefix="d_")
    assert set(no_graph.undefined.values()) == {"missing_graph_truth"}

    draws[0, 0, 2] = np.inf  # a dead effect only
    partial = metrics.attribution_scores(
        draws, truth, np.array([1, 1, 0]), live_prefix="live_", dead_prefix="dead_"
    )
    assert set(partial.undefined) == {"dead_leakage_scaled_mae", "dead_draw_leakage_scaled_mae"}
    assert set(partial.undefined.values()) == {metrics.NONFINITE}
    assert partial.values["live_mean_bias"] == pytest.approx(0.5)


def test_latent_sales_scores_close_each_draw_against_structural_sales():
    totals = np.array([[11.0, 12.0], [9.0, 14.0]])
    structural = np.array([10.0, 12.0])
    observed = np.array([10.0, 20.0])
    # Per draw: (1/10 + 0/20) / 2 = 0.05 and (1/10 + 2/20) / 2 = 0.1.
    assert metrics.draw_closure(totals, structural, observed) == pytest.approx((0.075, 0.095))
    world = metrics.latent_sales_scores(totals, structural, observed)
    assert world.values == pytest.approx(
        {
            "closure_rel_err": 0.025,  # posterior mean (10, 13)
            "closure_draw_rel_err": 0.075,
            "closure_draw_q90_rel_err": 0.095,
            "latent_sales_r2": 0.5,
        }
    )
    flat = metrics.latent_sales_scores(totals, np.array([10.0, 10.0]), observed)
    assert flat.undefined == {"latent_sales_r2": "constant_series"}
    totals[0, 0] = np.nan
    assert set(metrics.latent_sales_scores(totals, structural, observed).undefined.values()) == {
        metrics.NONFINITE
    }


def test_r2_can_be_negative_and_is_undefined_for_constant_truth():
    assert metrics.r2([2.0, 0.0], [0.0, 1.0]) == (pytest.approx(-9.0), None)
    assert metrics.r2([1.0, 2.0], [3.0, 3.0]) == (None, "constant_series")


def test_time_average_coverage_uses_the_interval_of_draw_time_means():
    # Slot 0 draw means 0, 1, 2 cover the truth mean 1; slot 1 draws all average 0.
    draws = np.stack([np.stack([[s, 0.0], [s, 0.0]]) for s in (0.0, 1.0, 2.0)])
    truth = np.array([[0.5, 0.0], [1.5, 1.0]])
    assert metrics.time_average_coverage(draws, truth) == 0.5
    assert metrics.time_average_coverage(draws[..., 0], truth[:, 0]) == 1.0
    world = metrics.time_average_scores(draws, truth, np.array([1, 0]), prefix="t_")
    assert world.values == {"t_cov80": 0.5, "t_cov80_live": 1.0, "t_cov80_dead": 0.0}
    assert world.weights == {"t_cov80": 2, "t_cov80_live": 1, "t_cov80_dead": 1}
    assert world.undefined == {"t_simcov80_live": "fewer_than_two_live_effects"}
    only_live = metrics.time_average_scores(draws, truth, np.array([1, 1]))
    assert only_live.undefined == {
        "cov80_dead": "no_dead_effects",
        "simcov80_live": "zero_posterior_sd",
    }


def test_simultaneous_coverage_studentizes_each_effect_and_includes_the_boundary():
    draws = np.array([[-1, -100], [-1, 100], [1, -100], [1, 100]])[:, None, :]
    # A raw-deviation band of radius 100 would wrongly contain this truth.
    truth = np.array([[2.0, 1.0]])
    assert metrics.simultaneous_coverage(draws, truth) == (0.0, None)
    assert metrics.simultaneous_coverage(draws, [[0.0, 0.0]]) == (1.0, None)
    assert metrics.simultaneous_coverage(draws, [[1.0, 100.0]]) == (1.0, None)
    # One escaping effect must miss even when mean/L2 discrepancies are inside.
    assert metrics.simultaneous_coverage(draws, [[0.0, 120.0]]) == (0.0, None)
    assert metrics.simultaneous_coverage(draws[..., ::-1], [[120.0, 0.0]]) == (0.0, None)
    rescale = np.array([-3.0, 0.001])
    assert metrics.simultaneous_coverage(draws * rescale, truth * rescale) == (0.0, None)
    extreme_units = np.array([1e-200, -1e150])
    assert metrics.simultaneous_coverage(draws * extreme_units, truth * extreme_units) == (
        0.0,
        None,
    )
    # Draw-dependent temporal cancellation must happen before the joint event.
    temporal = np.repeat(draws, 2, axis=1)
    temporal[:, 0, 0] *= 11
    temporal[:, 1, 0] *= -9
    assert metrics.simultaneous_coverage(temporal, np.repeat(truth, 2, axis=0)) == (0.0, None)


def test_simultaneous_coverage_centers_draws_and_interpolates_the_requested_quantile():
    draws = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])[:, None, :]
    # Center 2.5; the linear .8 quantile gives a raw half-width of 4.5.
    assert metrics.simultaneous_coverage(draws, [[6.0, 6.0]]) == (1.0, None)
    assert metrics.simultaneous_coverage(draws, [[8.0, 8.0]]) == (0.0, None)
    assert metrics.simultaneous_coverage(draws, [[6.0, 6.0]], level=0.5) == (0.0, None)
    assert metrics.simultaneous_coverage(draws, [[8.0, 8.0]], level=0.9) == (1.0, None)


def test_simultaneous_coverage_filters_dead_effects_before_scoring():
    live_draws = np.array([[-1, -100], [-1, 100], [1, -100], [1, 100]])[:, None, :]
    draws = np.concatenate((live_draws, np.full((4, 1, 1), np.nan)), axis=-1)
    truth = np.array([[0.0, 0.0, np.nan]])
    world = metrics.time_average_scores(draws, truth, np.array([1, 1, 0]))
    assert world.values == {"simcov80_live": 1.0}
    assert world.weights == {"simcov80_live": 1}
    assert {world.undefined[key] for key in ("cov80", "cov80_live", "cov80_dead")} == {"nonfinite"}
    assert metrics.time_average_scores(live_draws, truth[:, :2], None).undefined == {
        "cov80_live": "missing_graph_truth",
        "cov80_dead": "missing_graph_truth",
        "simcov80_live": "missing_graph_truth",
    }


@pytest.mark.parametrize("effects", [0, 1])
def test_simultaneous_coverage_needs_two_live_effects(effects):
    assert metrics.simultaneous_coverage(np.ones((3, 2, effects)), np.zeros((2, effects))) == (
        None,
        "fewer_than_two_live_effects",
    )


def test_simultaneous_coverage_reports_degenerate_and_nonfinite_totals():
    draws = np.array([[-1.0, 0.0], [1.0, 0.0]])[:, None, :]
    assert metrics.simultaneous_coverage(draws, [[0.0, 0.0]]) == (None, "zero_posterior_sd")
    assert metrics.simultaneous_coverage(draws[:1], [[0.0, 0.0]]) == (None, "zero_posterior_sd")
    repeated = np.full((3, 1, 2), 0.1)
    assert metrics.simultaneous_coverage(repeated, [[0.1, 0.1]]) == (None, "zero_posterior_sd")
    narrow = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])[:, None, :] * 1e-100
    assert metrics.simultaneous_coverage(narrow, [[0.0, 0.0]]) == (1.0, None)
    assert metrics.simultaneous_coverage(narrow * 1e-100, [[0.0, 0.0]]) == (1.0, None)
    assert metrics.simultaneous_coverage(draws, [[np.inf, 0.0]]) == (None, "nonfinite")
    draws[0, 0, 1] = np.nan
    assert metrics.simultaneous_coverage(draws, [[0.0, 0.0]]) == (None, "nonfinite")
    assert metrics.simultaneous_coverage(np.full((2, 3, 2), 1e308), np.zeros((3, 2))) == (
        None,
        "nonfinite",
    )
    for level in (0.0, 1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="level"):
            metrics.simultaneous_coverage(np.zeros((2, 1, 2)), [[0.0, 0.0]], level)
    with pytest.raises(ValueError, match="nonempty time"):
        metrics.simultaneous_coverage(np.zeros((2, 0, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError, match="3-D draws"):
        metrics.simultaneous_coverage(np.zeros((2, 2)), np.zeros(2))


def test_exchangeable_truth_has_nominal_joint_coverage_unlike_all_pointwise_covered():
    rng = np.random.default_rng(27)
    n_worlds, n_draws = 1500, 401
    draws = rng.normal(size=(n_worlds, n_draws, 15))
    truth = rng.normal(size=(n_worlds, 15))
    low, high = np.quantile(draws, (0.1, 0.9), axis=1)
    pointwise = (low <= truth) & (truth <= high)
    all_covered = []
    for channels in range(2, 16):
        simultaneous = [
            metrics.simultaneous_coverage(x[:, None, :channels], y[None, :channels])[0]
            for x, y in zip(draws, truth, strict=True)
        ]
        # Monte Carlo uncertainty plus finite posterior quantile/SD estimation.
        assert np.mean(simultaneous) == pytest.approx(0.8, abs=0.045), channels
        # This is comparisons.score_record's all_covered event, not a joint 80% band.
        all_covered.append(float(pointwise[:, :channels].all(axis=1).mean()))
    assert all_covered[0] == pytest.approx(0.8**2, abs=0.05)
    assert all_covered[-1] < 0.08
    assert all(a >= b for a, b in zip(all_covered, all_covered[1:], strict=False))


def test_cumulative_energy_score_scores_running_totals():
    draws = np.array([[[1.0], [1.0]], [[0.0], [0.0]]])
    truth = np.array([[1.0], [0.0]])
    # Running totals (1, 2), (0, 0) against (1, 1), in d = 2 coordinates.
    expected = (1.0 + math.sqrt(2.0) - math.sqrt(5.0)) / (2.0 * math.sqrt(2.0))
    assert metrics.cumulative_energy_score(draws, truth) == (pytest.approx(expected), None)
    assert metrics.cumulative_energy_score(draws[:1], truth) == (None, "needs_n_samples_ge_2")


def test_categorical_brier_and_entropy_of_distinct_graphs():
    graphs = np.array([[1, 0], [1, 0], [0, 0], [1, 1]])
    truth = np.array([1, 0])
    # Truth frequency 1/2 and counts {2, 1, 1}: 1 - 1 + 2 / 12.
    assert metrics.categorical_brier(graphs, truth) == (pytest.approx(1 / 6), None)
    assert metrics.categorical_brier(graphs[:1], truth) == (None, "needs_n_samples_ge_2")
    assert metrics.categorical_brier(np.tile([[0, 1]], (3, 1)), truth) == (2.0, None)
    assert metrics.graph_entropy(graphs) == pytest.approx(1.5 * math.log(2.0))
    assert math.copysign(1.0, metrics.graph_entropy(graphs[:2])) == 1.0  # exactly +0.0


def test_pairwise_comoment_is_zero_for_independent_slots():
    dependent = np.array([[1, 1], [0, 0]])
    independent = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    assert metrics.pairwise_comoment(dependent) == pytest.approx([0.25])
    assert metrics.pairwise_comoment(independent) == pytest.approx([0.0])
    assert metrics.pairwise_comoment(np.array([[1], [0]])).size == 0


def test_cycle_rates_separate_two_cycles_from_longer_cycles():
    three_cycle = np.zeros((3, 3))
    three_cycle[0, 1] = three_cycle[1, 2] = three_cycle[2, 0] = 1
    chain = np.zeros((3, 3))
    chain[0, 1] = chain[1, 2] = 1
    two_cycle = np.zeros((3, 3))
    two_cycle[0, 1] = two_cycle[1, 0] = 1
    assert metrics.two_cycle_rate(three_cycle[None]) == (0.0, None)
    assert metrics.cycle_rate(three_cycle[None]) == (1.0, None)
    draws = np.stack([three_cycle, chain, two_cycle])
    # One mutual pair among 3 draws x 3 unordered pairs; two of three draws are cyclic.
    assert metrics.two_cycle_rate(draws) == (pytest.approx(1 / 9), None)
    assert metrics.cycle_rate(draws) == (pytest.approx(2 / 3), None)
    assert metrics.cycle_rate(np.zeros((2, 1, 1))) == (None, "needs_2_active_nodes")
    assert metrics.two_cycle_rate(np.zeros((2, 1, 1))) == (None, "needs_2_active_nodes")


def test_graph_scores_summarize_hamming_distances_and_distinct_draws():
    truth = np.array([1.0, 0.0, 0.0])
    graphs = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=float)
    mutual = np.zeros((4, 2, 2))
    mutual[0, 0, 1] = mutual[0, 1, 0] = 1
    world = metrics.graph_scores(graphs, truth, {"cc": np.zeros((4, 1, 1)), "zz": mutual})
    # Distances 0, 0, 1, 2 over 3 slots.
    expected = {
        "shd_mean": 0.75,
        "exact_recovery": 0.5,
        "shd_median": 0.5,
        "normalized_shd_mean": 0.25,
        "energy_score": (1 + math.sqrt(2) - math.sqrt(3)) / (12 * math.sqrt(3)),
        "categorical_brier_score": 1 / 6,
        "entropy": 1.5 * math.log(2.0),
        "n_distinct": 3.0,
        "pairwise_comoment_abs_err": (0.0625 + 0.0625 + 0.1875) / 3,
        "pairwise_comoment_max_abs_err": 0.1875,
        "zz_two_cycle_rate": 0.25,
        "zz_cycle_rate": 0.25,
    }
    assert world.values == pytest.approx(expected)
    assert world.undefined == {
        "cc_two_cycle_rate": "needs_2_active_nodes",
        "cc_cycle_rate": "needs_2_active_nodes",
    }
    empty = metrics.graph_scores(np.zeros((4, 0)), np.zeros(0), {})
    assert set(empty.undefined.values()) == {"no_active_slots"}
    graphs[2, 1] = np.nan
    assert set(metrics.graph_scores(graphs, truth, {}).undefined.values()) == {metrics.NONFINITE}


def test_scores_that_overflow_from_finite_inputs_are_nonfinite_not_nan():
    huge = np.full((2, 3, 1), 1e308)  # finite draws whose mean overflows
    with np.errstate(over="ignore", invalid="ignore"):
        assert metrics.r2([1e200, -1e200], [0.0, 1.0]) == (None, metrics.NONFINITE)
        world = metrics.attribution_scores(
            huge, np.zeros((3, 1)), np.array([1]), live_prefix="live_", dead_prefix="dead_"
        )
    assert world.values == {}
    assert {world.undefined[f"live_{key}"] for key in metrics.LIVE_ATTRIBUTION_KEYS} == {
        metrics.NONFINITE
    }
    with pytest.raises(ValueError):
        world.skip("live_crps", "no_live_effects")  # a key is scored once
