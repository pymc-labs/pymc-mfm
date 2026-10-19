"""Stratified per-effect readouts: rows, bins, unavailable strata, evaluator and figures."""

import json

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.diagnostics import (
    TrainingRun,
    effect_rows,
    misclassification_summary,
    record_series,
    result_effect_rows,
    row_strata,
    size_cuts,
    strata_summary,
    stratified_summary,
)
from pymc_mfm.diagnostics.strata import CELL_METRICS
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM
from pymc_mfm.training import metrics
from pymc_mfm.training.evaluation import evaluate_rows


def row(world, family, truth, mean, *, live=True, p=0.5, active=2, lives=1, spread=1.0):
    """One effect row with an interval of ``mean ± spread``."""
    return {
        "world": world,
        "effect": f"{family}:{world}",
        "family": family,
        "live": live,
        "active_channels": active,
        "live_channels": lives,
        "truth": truth,
        "mean": mean,
        "low": None if mean is None else mean - spread,
        "high": None if mean is None else mean + spread,
        "crps": None if mean is None or truth is None else abs(mean - truth),
        "edge_probability": p,
    }


def cells(summary, family="channel"):
    return {cell["stratum"]: cell for cell in summary[family]["cells"]}


def test_rows_hold_window_totals_their_interval_crps_and_edge_frequency():
    draws = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    truth = np.array([[1.0, 0.0, -2.0], [3.0, 0.0, 1.0]])
    draws[0, 1, 2] = np.nan
    edges = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
    rows = effect_rows(
        "w",
        ["channel:a", "channel:b", "control:z"],
        ["channel", "channel", "control"],
        draws,
        truth,
        live=[1, 0, None],
        edge_draws=edges,
    )
    totals = draws.sum(axis=1)
    first = rows[0]
    assert first["truth"] == 4.0
    assert first["mean"] == pytest.approx(totals[:, 0].mean())
    assert (first["low"], first["high"]) == pytest.approx(
        tuple(np.quantile(totals[:, 0], (0.1, 0.9)))
    )
    assert first["crps"] == pytest.approx(metrics.empirical_crps(totals[:, 0], 4.0))
    assert first["edge_probability"] == 0.8
    # Live channels count graph truth over channels only; a control's missing label does not matter.
    assert [(r["live"], r["active_channels"], r["live_channels"]) for r in rows] == [
        (True, 2, 1),
        (False, 2, 1),
        (None, 2, 1),
    ]
    assert rows[1]["truth"] == 0.0
    # A nonfinite draw leaves the truth but no prediction.
    assert rows[2]["truth"] == -1.0 and rows[2]["mean"] is None and rows[2]["crps"] is None
    json.dumps(rows, allow_nan=False)


def test_size_bins_are_exact_zero_then_right_closed_and_edge_bins_follow_the_threshold_rule():
    rows = [row("w", "channel", truth, truth) for truth in (0.0, 1.0, 1.5, -3.0, 3.5)]
    assert row_strata(rows, "true_size_bin", cuts=[1, 2, 3]) == ["zero", "q1", "q2", "q3", "q4"]
    probabilities = [0.0999, 0.1, 0.5, 0.8999, 1.0]
    rows = [row("w", "channel", 1.0, 1.0, p=p, live=False) for p in probabilities]
    assert row_strata(rows, "edge_probability_bin", cuts=[]) == [
        "[0, 0.1)",
        "[0.1, 0.25)",
        "[0.5, 0.75)",
        "[0.75, 0.9)",
        "[0.9, 1]",
    ]
    # Called live at p >= t, as in threshold_table: every bin from t up.
    summary = misclassification_summary(rows)["channel"]["thresholds"]
    table = metrics.threshold_table(np.array(probabilities), np.zeros(5))
    for level, expected in zip(summary, table, strict=True):
        counts = {cell["decision"]: cell["n_effects"] for cell in level["cells"]}
        assert counts == {key: expected[key] for key in ("tp", "fp", "tn", "fn")}
    with pytest.raises(ValueError):
        row_strata(rows, "true_size_bin", cuts=[2, 1])


def test_empty_strata_are_unavailable_with_a_reason_and_no_values():
    rows = [
        row("a", "channel", 1.0, 1.5, active=2),
        row("a", "channel", 2.0, 2.0, active=2),
        row("b", "channel", 3.0, 1.0, active=4),
    ]
    by_count = stratified_summary(rows, "active_channels", cuts=[])
    assert list(cells(by_count)) == ["2", "3", "4"]
    gap = cells(by_count)["3"]
    assert gap["status"] == "unavailable" and gap["reason"] == "no_worlds_in_stratum"
    assert gap["n_effects"] == 0 and all(gap[key] is None for key in CELL_METRICS)
    # Worlds exist at 2 and 4 channels, but none holds a control.
    for stratum, cell in cells(by_count, "control").items():
        expected = "no_worlds_in_stratum" if stratum == "3" else "no_effects_in_stratum"
        assert (cell["status"], cell["reason"]) == ("unavailable", expected)
    sizes = cells(stratified_summary(rows, "true_size_bin", cuts=[10.0]))
    assert sizes["zero"]["reason"] == "no_effects_in_stratum" and sizes["zero"]["coverage"] is None
    assert sizes["q2"]["reason"] == "no_effects_in_stratum"
    assert (sizes["q1"]["n_worlds"], sizes["q1"]["n_effects"]) == (2, 3)


def test_cells_pool_their_effects_and_count_the_nonfinite_ones():
    rows = [
        row("a", "channel", 1.0, 1.5, spread=1.0),
        row("a", "channel", 2.0, 0.5, spread=1.0),
        row("b", "channel", 3.0, 3.0, spread=0.5, live=False),
        row("c", "channel", 4.0, None),
    ]
    cell = cells(stratified_summary(rows, "true_size_bin", cuts=[]))["q1"]
    residual = np.array([0.5, -1.5, 0.0])
    assert (cell["n_worlds"], cell["n_effects"], cell["n_scored"], cell["n_live"]) == (3, 4, 3, 3)
    assert cell["status"] == "ok" and cell["reason"] == "nonfinite_in_1_effects"
    assert cell["coverage"] == pytest.approx(2 / 3)
    assert cell["signed_residual"] == pytest.approx(residual.mean())
    assert cell["abs_residual"] == pytest.approx(np.abs(residual).mean())
    assert cell["crps"] == pytest.approx(np.abs(residual).mean())
    assert [cell["residual_q10"], cell["residual_median"], cell["residual_q90"]] == pytest.approx(
        np.quantile(residual, (0.1, 0.5, 0.9)).tolist()
    )
    predicted = stratified_summary(rows, "predicted_size_bin", cuts=[])["channel"]
    assert predicted["excluded"] == [{"reason": "nonfinite_draws", "n_effects": 1}]
    alone = cells(stratified_summary(rows[3:], "true_size_bin", cuts=[]))["q1"]
    assert (alone["status"], alone["reason"]) == ("unavailable", "nonfinite_in_1_effects")


def test_summaries_are_strict_json_with_every_number_inside_a_list():
    rows = [row("a", "channel", 1.0, 1.5), row("a", "control", -2.0, -1.0, live=None)]
    summary = strata_summary(rows)
    assert summary["cuts"] == size_cuts(rows)
    json.dumps(summary, allow_nan=False)
    # Record series (run figures, report metrics) never turn strata into scalar series.
    assert record_series([{"step": 1, "strata": summary}], "evaluation")["series"] == {}
    control = misclassification_summary(rows)["control"]
    assert control["excluded"] == [{"reason": "missing_graph_truth", "n_effects": 1}]


def test_rows_without_a_world_or_a_finite_prediction_are_not_miscounted():
    from test_diagnostics import make_result

    result = make_result()  # records data_sha256 but no world_id, like decompose()
    rows = result_effect_rows(result, result.components.mean(axis=0), [1, 0, 1])
    assert {r["world"] for r in rows} == {result.provenance["data_sha256"]}
    # Distinct unnamed worlds cannot be counted, so they are not counted as one.
    unnamed = [row(None, "channel", 1.0, 1.0), row(None, "channel", 2.0, 2.0)]
    assert cells(stratified_summary(unnamed, "true_size_bin", cuts=[]))["q1"]["n_worlds"] is None
    # A CRPS that overflows voids the whole prediction instead of a partial row.
    huge = np.array([[[1.5e308]], [[-1.5e308]]])
    (overflow,) = effect_rows("w", ["channel:a"], ["channel"], huge, np.array([[1e308]]))
    assert [overflow[key] for key in ("mean", "low", "high", "crps")] == [None] * 4
    bins = cells(strata_summary([overflow])["by"]["true_size_bin"])
    assert (bins["q1"]["n_effects"], bins["q1"]["n_scored"], bins["q1"]["status"]) == (
        1,
        0,
        "unavailable",
    )


@pytest.fixture(scope="module")
def corpus():
    return make_synthetic_corpus()


@pytest.fixture
def holdout(corpus):
    return [(0, int(row)) for row in np.flatnonzero(corpus["is_val"] == 1)]


def test_evaluator_strata_pool_the_captured_rows_and_line_up_with_router_thresholds(
    model, corpus, holdout
):
    captured = []
    record = evaluate_rows(
        model,
        [corpus],
        holdout,
        n_samples=6,
        seed=3,
        device="cpu",
        leg="holdout",
        per_world=True,
        result_sink=lambda *args: captured.append(args),
    )
    by_world = {
        entry["effects"]["conditional"]["query"][0]["world"]: entry for entry in record["per_world"]
    }
    numeric = ("truth", "mean", "low", "high", "crps", "edge_probability")
    compared = 0
    for result, truth, graph_truth, _ in captured:
        expected = by_world[result.provenance["world_id"]]["effects"]["conditional"][result.window]
        rebuilt = result_effect_rows(result, truth, graph_truth)
        for mine, theirs in zip(rebuilt, expected, strict=True):
            assert {k: v for k, v in mine.items() if k not in numeric} == {
                k: v for k, v in theirs.items() if k not in numeric
            }
            for key in numeric:
                assert mine[key] == pytest.approx(theirs[key], rel=1e-9, abs=1e-12), key
            compared += 1
    assert compared
    for head, windows in record["strata"].items():
        for window, table in windows.items():
            pooled = [r for e in record["per_world"] for r in e["effects"][head][window]]
            assert table == strata_summary(pooled)
    # Joint query rows use the router draws behind the joint_router tables: same decisions.
    joint = record["strata"]["joint"]["query"]["misclassification"]
    for family, edge in (("channel", "cy"), ("control", "zb")):
        recorded = record["tables"]["joint_router"][edge]["thresholds"]
        for level, expected in zip(joint[family]["thresholds"], recorded, strict=True):
            counts = {cell["decision"]: cell["n_effects"] for cell in level["cells"]}
            assert counts == {key: expected[key] for key in ("tp", "fp", "tn", "fn")}


def test_size_bins_depend_only_on_the_evaluated_worlds(model, corpus, holdout):
    torch.manual_seed(99)
    other = MFM(
        ModelConfig(
            d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=2
        )
    ).eval()
    first, replay, second = (
        evaluate_rows(net, [corpus], holdout, n_samples=4, seed=seed, device="cpu", leg="holdout")
        for net, seed in ((model, 1), (model, 2), (other, 1))
    )
    for head in ("joint", "conditional"):
        for window in ("query", "context"):
            tables = [record["strata"][head][window] for record in (first, replay, second)]
            assert tables[0]["cuts"] and all(t["cuts"] == tables[0]["cuts"] for t in tables)
            true_bins = [
                [
                    (c["stratum"], c["n_effects"])
                    for c in t["by"]["true_size_bin"]["channel"]["cells"]
                ]
                for t in tables
            ]
            assert true_bins[0] == true_bins[1] == true_bins[2]
    assert (
        first["strata"]["joint"]["query"]["cuts"] == first["strata"]["conditional"]["query"]["cuts"]
    )


def strata_table():
    rows = [
        row("a", "channel", 0.0, 0.2, p=0.05, live=False, active=2, lives=1),
        row("a", "channel", 1.0, 1.4, p=0.95, active=2, lives=1),
        row("b", "channel", 3.0, 1.0, p=0.6, active=4, lives=2, spread=0.5),
        row("b", "control", -2.0, -1.0, p=0.3, active=4, lives=2),
    ]
    return strata_summary(rows), rows


def test_strata_figure_draws_cell_values_and_marks_unavailable_strata():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from pymc_mfm.plotting import plot_strata_table

    table, rows = strata_table()
    figure = Figure()
    target, untouched = figure.subplots(1, 2)
    returned = plot_strata_table(
        table, metric="coverage", by="active_channels", header="H", ax=target
    )
    assert returned is figure and figure._suptitle is None and "H" in target.get_title()
    assert not (untouched.lines or untouched.collections)
    drawn = {line.get_gid(): line for line in target.lines if line.get_gid()}
    for family in ("channel", "control"):
        values = [cell["coverage"] for cell in table["by"]["active_channels"][family]["cells"]]
        np.testing.assert_allclose(
            drawn[f"strata/{family}"].get_ydata(), [v for v in values if v is not None]
        )
    unavailable = sum(
        cell["coverage"] is None
        for record in table["by"]["active_channels"].values()
        for cell in record["cells"]
    )
    assert [text.get_text() for text in target.texts].count("n/a") == unavailable > 0
    residual = plot_strata_table(table, metric="signed_residual", by="true_size_bin", rows=rows)
    ax = residual.axes[0]
    points = sum(
        len(collection.get_offsets())
        for collection in ax.collections
        if (collection.get_gid() or "").startswith("strata/rows/")
    )
    assert points == len(rows)
    with pytest.raises(ValueError):
        plot_strata_table(table, metric="brier")


def test_run_strata_draw_a_recorded_evaluation_table(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    from test_training_run import write_run

    root = write_run(tmp_path / "run", evaluations=(2, 4))
    run = TrainingRun.load(root)
    with pytest.raises(ValueError, match="step_000000004.json not found"):
        run.plot.strata()
    table, _ = strata_table()
    (root / "evaluation_tables").mkdir()
    for step in (2, 4):
        legs = {"holdout": {"strata": {"conditional": {"query": table}}}}
        (root / "evaluation_tables" / f"step_{step:09d}.json").write_text(
            json.dumps({"step": step, "legs": legs})
        )
    figure = Figure()
    target = figure.subplots()
    assert run.plot.strata("crps", by="edge_probability_bin", step=2, ax=target) is figure
    assert "sentinel-run" in target.get_title() and "step 2" in target.get_title()
    values = [c["crps"] for c in table["by"]["edge_probability_bin"]["channel"]["cells"]]
    line = next(line for line in target.lines if line.get_gid() == "strata/channel")
    np.testing.assert_allclose(line.get_ydata(), [v for v in values if v is not None])
    with pytest.raises(ValueError, match="no joint query strata"):
        run.plot.strata(head="joint")
    with pytest.raises(ValueError, match="no evaluation at step 3"):
        run.plot.strata(step=3)
