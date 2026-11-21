"""Coverage intervals preserve the recorded estimand and resample whole cells."""

import copy
import hashlib
import json

import numpy as np
import pytest

from pymc_mktfm.diagnostics import TrainingRun, build_report, render_report
from pymc_mktfm.diagnostics.comparisons import _estimate
from pymc_mktfm.diagnostics.coverage import coverage_intervals, evaluation_coverage_intervals

POINTWISE = "conditional/query/timeavg_contribution_cov80_live"
SIMULTANEOUS = "conditional/query/timeavg_contribution_simcov80_live"


def world(value, cell, *, row=0, corpus=0, slots=1, reason=None, key=POINTWISE):
    return {
        "corpus_index": corpus,
        "row": row,
        "cell_id": cell,
        "values": {key: value},
        "weights": {key: slots if value is not None else 0},
        "undefined": {} if reason is None else {key: reason},
    }


def legacy_bootstrap(values, *, seed, bootstrap_samples):
    """The original world bootstrap, with its bounded-memory RNG order."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 256):
        stop = min(start + 256, bootstrap_samples)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def test_duplicate_worlds_in_twenty_cells_do_not_supply_two_hundred_independent_units():
    values = np.repeat(np.linspace(0.05, 0.95, 20), 10)
    worlds = [world(value, row // 10, row=row) for row, value in enumerate(values)]
    interval = coverage_intervals(worlds, [POINTWISE], seed=31, bootstrap_samples=2000)[POINTWISE]
    _, naive_low, naive_high = legacy_bootstrap(values, seed=31, bootstrap_samples=2000)
    assert interval["status"] == "ok"
    assert (interval["n_worlds"], interval["n_cells"], interval["n_slots"]) == (200, 20, 200)
    assert interval["mean"] == pytest.approx(values.mean())
    assert interval["ci_high"] - interval["ci_low"] > 2 * (naive_high - naive_low)


def test_singleton_cells_match_legacy_bootstrap_exactly_beyond_one_rng_chunk():
    values = np.array([0.07, 0.91, 0.22, 0.64, 0.38, 0.83, 0.15, 0.52, 0.99])
    cells = [19, 2, 11, 0, 8, 1, 7, 5, 3]  # first occurrence, not sorted cell order
    worlds = [
        world(value, cell, row=row, slots=1 + row * row)
        for row, (value, cell) in enumerate(zip(values, cells, strict=True))
    ]
    interval = coverage_intervals(worlds, [POINTWISE], seed=14, bootstrap_samples=913)[POINTWISE]
    expected = legacy_bootstrap(values, seed=14, bootstrap_samples=913)
    assert (interval["mean"], interval["ci_low"], interval["ci_high"]) == expected
    assert (interval["n_worlds"], interval["n_cells"], interval["n_slots"]) == (9, 9, 213)
    unweighted = _estimate(values, seed=14, bootstrap_samples=913)
    assert (unweighted["mean"], unweighted["ci_low"], unweighted["ci_high"]) == expected


def test_uneven_cells_keep_world_weights_and_slot_counts_do_not_change_the_interval():
    worlds = [world(0, 0, slots=100), *(world(1, 1, row=row) for row in range(1, 4))]
    interval = coverage_intervals(worlds, [POINTWISE], seed=11, bootstrap_samples=1301)[POINTWISE]
    assert interval["mean"] == 0.75  # neither equal-cell .5 nor slot-pooled 3 / 103
    assert (interval["n_worlds"], interval["n_cells"], interval["n_slots"]) == (4, 2, 103)
    for entry in worlds:
        entry["weights"][POINTWISE] = 1
    rescaled = coverage_intervals(worlds, [POINTWISE], seed=11, bootstrap_samples=1301)[POINTWISE]
    assert (rescaled["mean"], rescaled["ci_low"], rescaled["ci_high"]) == (
        interval["mean"],
        interval["ci_low"],
        interval["ci_high"],
    )
    assert (rescaled["n_worlds"], rescaled["n_cells"], rescaled["n_slots"]) == (4, 2, 4)


def test_weighted_bootstrap_resamples_numerator_and_denominator_with_the_same_indices():
    values = np.array([0.0, 0.4, 0.7, 1.0])
    counts = np.array([1, 3, 2, 9])
    rng = np.random.default_rng(11)
    ratios = []
    for _ in range(1301):
        indices = rng.integers(4, size=4)
        ratios.append(sum(values[i] * counts[i] for i in indices) / sum(counts[indices]))
    expected = np.quantile(ratios, [0.025, 0.975])
    estimate = _estimate(values, weights=counts, seed=11, bootstrap_samples=1301)
    assert estimate["mean"] == pytest.approx(11.6 / 15)
    np.testing.assert_allclose(
        [estimate["ci_low"], estimate["ci_high"]], expected, rtol=0, atol=1e-15
    )
    groups = ([0], [0, 1, 1], [0, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1])
    totals = np.array([0.0, 2.0, 1.0, 5.0])
    rng = np.random.default_rng(11)
    ratios = []
    for _ in range(1301):
        indices = rng.integers(4, size=4)
        ratios.append(sum(totals[indices]) / sum(counts[indices]))
    expected = np.quantile(ratios, [0.025, 0.975])
    np.testing.assert_allclose(expected, [1 / 3, 11 / 18], rtol=0, atol=1e-15)
    worlds = []
    for cell, group in enumerate(groups):
        for position, value in enumerate(group):
            worlds.append(world(value, cell, row=len(worlds), slots=(cell + 1) * (position + 1)))
    recorded = event(1, 8 / 15)
    recorded["legs"]["holdout"] = {
        "n_worlds": 15,
        "rows": [[0, row] for row in range(15)],
        "metric_worlds": {POINTWISE: 15},
    }
    for slots in (None, 1):
        if slots is not None:
            for entry in worlds:
                entry["weights"][POINTWISE] = slots
        interval = evaluation_coverage_intervals(
            recorded,
            {"legs": {"holdout": {"coverage": worlds}}},
            seed=11,
            bootstrap_samples=1301,
        )["holdout"][POINTWISE]
        assert interval["mean"] == pytest.approx(8 / 15)
        np.testing.assert_allclose(
            [interval["ci_low"], interval["ci_high"]], expected, rtol=0, atol=1e-15
        )


def test_matching_cell_numbers_from_different_corpora_are_distinct_bootstrap_units():
    interval = coverage_intervals(
        [world(0, 7, corpus=0), world(1, 7, corpus=1)], [POINTWISE], bootstrap_samples=513
    )[POINTWISE]
    assert (interval["mean"], interval["n_worlds"], interval["n_cells"]) == (0.5, 2, 2)
    assert (interval["status"], interval["ci_low"], interval["ci_high"]) == ("ok", 0.0, 1.0)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"cell_id": None},
        {"cell_id": float("nan")},
        {"cell_id": float("inf")},
        {"cell_id": True},
        {"cell_id": 1.0},
        {"cell_id": "1"},
        {"cell_id": [1]},
        {"cell_id": np.array([[1]])},
        {"cell_id": np.array(1)},
        {"cell_id": 1, "corpus_index": None},
        {"cell_id": 1, "corpus_index": float("nan")},
        {"cell_id": 1, "corpus_index": [0]},
        {"cell_id": 1, "corpus_index": False},
    ],
)
def test_missing_or_unusable_identity_never_fabricates_independent_cells(metadata):
    first = world(0, 0, slots=10)
    first.pop("cell_id")
    first.update(metadata)
    interval = coverage_intervals([first, world(1, 2)], [POINTWISE])[POINTWISE]
    assert interval["mean"] == 0.5
    assert (interval["n_worlds"], interval["n_slots"], interval["n_cells"]) == (2, 11, None)
    assert interval["status"] == "unavailable" and interval["reason"] == "missing_cell_identity"
    assert interval["ci_low"] is None and interval["ci_high"] is None


def test_zero_or_one_scored_cell_keeps_mean_and_counts_but_has_no_interval():
    zero = coverage_intervals(
        [world(None, 0, reason="zero_posterior_sd", key=SIMULTANEOUS)], [SIMULTANEOUS]
    )[SIMULTANEOUS]
    assert zero["mean"] is None
    assert (zero["n_worlds"], zero["n_cells"], zero["n_slots"]) == (0, 0, 0)
    assert zero["status"] == "unavailable" and zero["reason"] == "no_scored_worlds"
    assert zero["undefined"] == {"zero_posterior_sd": 1}
    one = coverage_intervals(
        [world(0, 9, slots=2), world(1, 9, row=1, slots=3), world(None, 3, reason="nonfinite")],
        [POINTWISE],
    )[POINTWISE]
    assert (one["mean"], one["n_worlds"], one["n_cells"], one["n_slots"]) == (0.5, 2, 1, 5)
    assert one["status"] == "unavailable" and one["reason"] == "fewer_than_two_cells"
    assert one["undefined"] == {"nonfinite": 1}
    assert one["ci_low"] is None and one["ci_high"] is None


def test_nonfinite_worlds_are_dropped_explicitly_without_losing_the_finite_interval():
    worlds = [
        world(0, 1, slots=2),
        world(1, 2, slots=3),
        world(float("nan"), None, slots=100),
        world(float("inf"), None, slots=100),
        world(None, None, reason="nonfinite"),
        world(None, 4, reason="no_live_effects"),
    ]
    interval = coverage_intervals(worlds, [POINTWISE], seed=8, bootstrap_samples=777)[POINTWISE]
    expected = legacy_bootstrap([0, 1], seed=8, bootstrap_samples=777)
    assert (interval["mean"], interval["ci_low"], interval["ci_high"]) == expected
    assert (interval["n_worlds"], interval["n_cells"], interval["n_slots"]) == (2, 2, 5)
    assert interval["status"] == "partial" and interval["reason"] == "nonfinite_in_3_worlds"
    assert interval["undefined"] == {"nonfinite": 3, "no_live_effects": 1}
    entirely_nonfinite = coverage_intervals(worlds[2:5], [POINTWISE])[POINTWISE]
    assert entirely_nonfinite["mean"] is None and entirely_nonfinite["n_worlds"] == 0
    assert entirely_nonfinite["reason"] == "no_scored_worlds"
    assert entirely_nonfinite["undefined"] == {"nonfinite": 3}


def test_not_recorded_empty_and_entirely_undefined_evidence_are_not_numeric_zero():
    absent = coverage_intervals(None, [SIMULTANEOUS])[SIMULTANEOUS]
    missing_key = coverage_intervals([world(1, 0)], [SIMULTANEOUS])[SIMULTANEOUS]
    for interval in (absent, missing_key):
        assert interval["reason"] == "not_recorded" and interval["status"] == "unavailable"
        assert interval["mean"] is None and interval["ci_low"] is None
        assert interval["n_worlds"] is None and interval["n_cells"] is None
    empty = coverage_intervals([], [SIMULTANEOUS])[SIMULTANEOUS]
    assert empty["reason"] == "empty_worlds" and empty["mean"] is None
    assert (empty["n_worlds"], empty["n_cells"], empty["n_slots"]) == (0, 0, 0)
    partial = coverage_intervals(
        [world(0, 0), world(1, 1), world(1, 2, key=SIMULTANEOUS)], [POINTWISE]
    )[POINTWISE]
    assert partial["mean"] == 0.5 and partial["n_worlds"] == 2
    assert partial["status"] == "partial" and partial["reason"] == "incomplete_world_evidence"
    assert partial["undefined"] == {"not_recorded": 1}


def test_missing_slot_counts_do_not_change_the_world_estimand_or_the_cell_interval():
    worlds = [world(0.2, 0), world(0.8, 1)]
    worlds[0]["weights"] = {}
    interval = coverage_intervals(worlds, [POINTWISE], seed=3, bootstrap_samples=777)[POINTWISE]
    assert (interval["mean"], interval["ci_low"], interval["ci_high"]) == legacy_bootstrap(
        [0.2, 0.8], seed=3, bootstrap_samples=777
    )
    assert interval["n_slots"] is None and (interval["n_worlds"], interval["n_cells"]) == (2, 2)
    assert interval["status"] == "partial" and interval["reason"] == "missing_slot_counts"


@pytest.mark.parametrize("per_world", [{"cell_id": [1, 2]}, "worlds", np.zeros((2, 3))])
def test_wrong_shape_world_evidence_has_an_explicit_unavailable_reason(per_world):
    interval = coverage_intervals(per_world, [POINTWISE])[POINTWISE]
    assert interval["status"] == "unavailable" and interval["reason"] == "invalid_world_evidence"
    assert interval["mean"] is None and interval["n_cells"] is None


def event(step, mean):
    return {
        "step": step,
        "legs": {"holdout": {}, "seen": {}},
        "flat": {f"holdout/{POINTWISE}": mean, f"seen/{SIMULTANEOUS}": None},
    }


@pytest.mark.parametrize("damage", ["omit", "replace", "missing_score", "strip"])
def test_incomplete_saved_population_cannot_claim_bounds_for_recorded_coverage(damage):
    worlds = [world(value, row, row=row) for row, value in enumerate([0.0, 1.0, 1.0, 1.0])]
    recorded = event(2, 0.75)
    recorded["legs"]["holdout"] = {
        "n_worlds": 4,
        "rows": [[0, row] for row in range(4)],
        "metric_worlds": {POINTWISE: 4},
    }
    if damage == "omit":
        worlds = worlds[:2]
    elif damage == "replace":
        worlds[-1] = world(0.0, 0, row=0)
    elif damage == "strip":
        for entry in worlds:
            entry["values"].clear()
            entry["weights"].clear()
            entry["undefined"].clear()
    else:
        worlds[-1]["values"][POINTWISE] = None
        worlds[-1]["undefined"][POINTWISE] = "not_recorded"
    interval = evaluation_coverage_intervals(recorded, {"legs": {"holdout": {"coverage": worlds}}})[
        "holdout"
    ][POINTWISE]
    assert interval["status"] == "unavailable"
    assert interval["reason"] == "incomplete_world_evidence"
    assert interval["ci_low"] is None and interval["ci_high"] is None
    if damage == "strip":
        assert interval["mean"] is None
    else:
        assert interval["mean"] == pytest.approx(
            {"omit": 0.5, "replace": 0.5, "missing_score": 2 / 3}[damage]
        )


def test_saved_mean_must_match_recorded_coverage_allowing_roundoff_and_valid_exclusions():
    worlds = [world(value, row, row=row) for row, value in enumerate([0.0, 0.0, 1.0, 0.0])]
    recorded = event(2, 0.75)
    recorded["legs"]["holdout"] = {
        "n_worlds": 4,
        "rows": [[0, row] for row in range(4)],
        "metric_worlds": {POINTWISE: 4},
    }
    tables = {"legs": {"holdout": {"coverage": worlds}}}
    interval = evaluation_coverage_intervals(recorded, tables)["holdout"][POINTWISE]
    assert interval["mean"] == 0.25 and interval["n_worlds"] == 4
    assert (
        interval["status"] == "unavailable" and interval["reason"] == "inconsistent_recorded_mean"
    )
    assert interval["ci_low"] is None and interval["ci_high"] is None
    recorded["flat"][f"holdout/{POINTWISE}"] = 0.25 + 1e-15
    valid = evaluation_coverage_intervals(recorded, tables)["holdout"][POINTWISE]
    assert valid["status"] == "ok" and (valid["ci_low"], valid["ci_high"]) == (0.0, 0.75)
    worlds.append(world(None, 4, row=4, reason="nonfinite"))
    recorded["legs"]["holdout"]["rows"].append([0, 4])
    recorded["legs"]["holdout"]["n_worlds"] = 5
    partial = evaluation_coverage_intervals(recorded, tables)["holdout"][POINTWISE]
    assert partial["mean"] == 0.25 and partial["n_worlds"] == 4
    assert partial["status"] == "partial" and partial["reason"] == "nonfinite_in_1_worlds"
    assert (partial["ci_low"], partial["ci_high"]) == (0.0, 0.75)


def test_training_run_reads_each_saved_coverage_table_and_marks_legacy_events_not_recorded(
    tmp_path,
):
    (tmp_path / "run.json").write_text(json.dumps({"run_id": "coverage", "status": "finished"}))
    records = [event(2, 0.75), event(4, 0.25), event(6, 0.6), event(8, 0.6)]
    (tmp_path / "evaluations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    tables = tmp_path / "evaluation_tables"
    tables.mkdir()
    worlds = [world(0, 0, slots=100), *(world(1, 1, row=row) for row in range(1, 4))]
    for step in (2, 4):
        (tables / f"step_{step:09d}.json").write_text(
            json.dumps({"step": step, "legs": {"holdout": {"coverage": worlds}}})
        )
        for entry in worlds:
            entry["values"][POINTWISE] = 1 - entry["values"][POINTWISE]
    (tables / "step_000000008.json").write_text(json.dumps({"step": 8, "legs": {"holdout": {}}}))
    derived = TrainingRun.load(tmp_path).derived_metrics(seed=12, bootstrap_samples=913)
    for item, expected in zip(derived[:2], (0.75, 0.25), strict=True):
        interval = item["coverage_intervals"]["holdout"][POINTWISE]
        assert interval["mean"] == expected and interval["status"] == "ok"
        assert (interval["n_worlds"], interval["n_cells"], interval["n_slots"]) == (4, 2, 103)
        assert (interval["confidence_level"], interval["seed"], interval["bootstrap_samples"]) == (
            0.95,
            12,
            913,
        )
        missing = item["coverage_intervals"]["seen"][SIMULTANEOUS]
        assert missing["mean"] is None and missing["reason"] == "not_recorded"
    for item in derived[2:]:
        legacy = item["coverage_intervals"]["holdout"][POINTWISE]
        assert legacy["status"] == "unavailable" and legacy["reason"] == "not_recorded"
        assert legacy["mean"] is None and legacy["n_cells"] is None


def test_report_freezes_interval_evidence_and_rerenders_it_without_the_source_worlds(tmp_path):
    pytest.importorskip("matplotlib")
    from test_diagnostics import make_result

    worlds = [world(0, 0, slots=100), *(world(1, 1, row=row) for row in range(1, 4))]
    snapshot = [
        {
            "step": 2,
            "coverage_intervals": evaluation_coverage_intervals(
                event(2, 0.75),
                {"legs": {"holdout": {"coverage": worlds}}},
                seed=4,
                bootstrap_samples=913,
            ),
        }
    ]
    frozen = copy.deepcopy(snapshot)
    result = make_result()
    result.diagnostics()
    diagnostic = result.diagnostics(coverage_intervals=snapshot)
    snapshot[0]["coverage_intervals"]["holdout"][POINTWISE]["mean"] = 0.1
    worlds.clear()
    manifest = diagnostic.report(tmp_path / "accessor")
    built = build_report(result, tmp_path / "built", coverage_intervals=frozen)
    frozen[0]["coverage_intervals"]["holdout"][POINTWISE]["ci_low"] = 0.9
    for root, report in ((tmp_path / "accessor", manifest), (tmp_path / "built", built)):
        interval = report["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]
        assert (interval["mean"], interval["ci_low"], interval["ci_high"]) == (0.75, 0.0, 1.0)
        assert (interval["n_worlds"], interval["n_cells"], interval["n_slots"]) == (4, 2, 103)
        assert not any("coverage_intervals" in key for key in report["metrics"])
        saved = json.loads((root / "evidence.json").read_text())
        assert saved["evaluation_records"] == []
        assert (
            saved["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]["mean"]
            == 0.75
        )
        summary = json.loads((root / "diagnostics.json").read_text())
        assert (
            summary["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]["ci_low"]
            == 0.0
        )
    replayed = render_report(tmp_path / "accessor", tmp_path / "replayed")
    interval = replayed["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]
    assert (interval["mean"], interval["ci_low"], interval["ci_high"]) == (0.75, 0.0, 1.0)
    assert interval["seed"] == 4 and interval["bootstrap_samples"] == 913


def test_interval_only_diagnostics_do_not_contaminate_or_reuse_the_plain_cache():
    from test_diagnostics import make_result

    result = make_result()
    snapshots = []
    for values in ([0, 1, 1, 1], [0, 0, 0, 1]):
        worlds = [world(value, row, row=row) for row, value in enumerate(values)]
        snapshots.append(
            [
                {
                    "step": 2,
                    "coverage_intervals": {"holdout": coverage_intervals(worlds, [POINTWISE])},
                }
            ]
        )
    first = result.diagnostics(coverage_intervals=snapshots[0])
    assert result.diagnostics().summary["coverage_intervals"] == []
    second = result.diagnostics(coverage_intervals=snapshots[1])
    assert (
        first.summary["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]["mean"]
        == 0.75
    )
    assert (
        second.summary["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]["mean"]
        == 0.25
    )
    assert result.diagnostics().summary["coverage_intervals"] == []


def test_legacy_report_without_interval_evidence_rerenders_as_not_recorded(tmp_path):
    pytest.importorskip("matplotlib")
    from test_diagnostics import make_result

    root = tmp_path / "legacy"
    original = build_report(make_result(), root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("coverage_intervals")
    for name in ("evidence.json", "diagnostics.json"):
        path = root / name
        value = json.loads(path.read_text())
        value.pop("coverage_intervals")
        path.write_text(json.dumps(value))
        descriptor = next(item for item in manifest["artifacts"] if item["path"] == name)
        descriptor["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    replayed = render_report(root, tmp_path / "replayed")
    assert replayed["metrics"] == original["metrics"]
    assert replayed["coverage_intervals"] == []


def test_nonfinite_interval_snapshot_is_rejected_before_creating_a_report(tmp_path):
    from test_diagnostics import make_result

    with pytest.raises(ValueError, match="nonfinite"):
        build_report(make_result(), tmp_path / "invalid", coverage_intervals=[{"mean": np.nan}])
    assert not (tmp_path / "invalid").exists()


def test_rerender_rejects_relabeling_saved_interval_numbers(tmp_path):
    pytest.importorskip("matplotlib")
    from test_diagnostics import make_result

    snapshot = [{"step": 2, "coverage_intervals": {"holdout": {POINTWISE: {"mean": 0.75}}}}]
    root = tmp_path / "original"
    build_report(make_result(), root, coverage_intervals=snapshot)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["coverage_intervals"][0]["coverage_intervals"]["holdout"][POINTWISE]["mean"] = 0.25
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage intervals disagree"):
        render_report(root, tmp_path / "relabeled")
    assert not (tmp_path / "relabeled").exists()
