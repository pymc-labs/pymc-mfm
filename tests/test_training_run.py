"""Training runs reload from their records alone; figures show the requested series."""

import json

import numpy as np
import pytest
from conftest import make_synthetic_corpus

from pymc_mfm.diagnostics import TrainingRun
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.training.evaluation import HEAD_LABELS, metric_units

pytest.importorskip("matplotlib")

LEGS = {"holdout": 100, "seen": 200}
HEADS = {"joint": 10, "conditional": 20}
WINDOWS = {"query": 1, "context": 2}


def sentinel(leg, head, window, step):
    """A value that identifies its leg, head, window and step."""
    return LEGS[leg] + HEADS[head] + WINDOWS[window] + step / 1000


def training_record(step):
    return {
        "step": step,
        "epoch": 0,
        "consumed_rows": 2 * step,
        "batch_rows": [[0, step]],
        "lr": 0.001 * step,
        "loss": 5.0 - step,
        "flow_loss": 2.0 - 0.1 * step,
        "struct_loss": 0.5,
        "struct_loss_raw": 0.25,
        "struct_loss_raw_status": "reconstructed_from_weighted_term",
        "objective_weights": {"struct_weight": 2.0},
        "grad_norm": 0.5 * step,
        "grad_clipped": step > 2,
    }


def evaluation_record(step):
    flat = {
        f"{leg}/{head}/{window}/decomposition_energy_score": sentinel(leg, head, window, step)
        for leg in LEGS
        for head in HEADS
        for window in WINDOWS
    }
    edge_scores = {
        "holdout": {"cy": (0.9, 0.1), "zb": (0.7, 0.3), "cc": (None, None)},
        "seen": {"cy": (0.95, 0.05), "zb": (0.85, 0.15), "cc": (0.5, 0.2)},
    }
    for leg, families in edge_scores.items():
        for family, (auroc, ece) in families.items():
            flat[f"{leg}/structure_head/{family}/auroc"] = auroc
            flat[f"{leg}/structure_head/{family}/ece"] = ece
            flat[f"{leg}/joint_router/edge/{family}/auroc"] = None if auroc is None else auroc / 2
        flat[f"{leg}/joint_router/graph/shd_mean"] = 1.5 if leg == "holdout" else 0.5
    # Each leg record labels its heads as the evaluator does.
    legs = {leg: {"labels": dict(HEAD_LABELS)} for leg in LEGS}
    return {
        "step": step,
        "epoch": 0,
        "consumed_rows": 2 * step,
        "n_samples": 7,
        "legs": legs,
        "flat": flat,
    }


def write_run(root, *, training=(1, 2, 3), evaluations=(2,)):
    root.mkdir()
    run = {"run_id": "r-1", "run_name": "sentinel-run", "status": "running", "final_step": 0}
    (root / "run.json").write_text(json.dumps(run))
    if training is not None:
        lines = "".join(json.dumps(training_record(step)) + "\n" for step in training)
        (root / "metrics.jsonl").write_text(lines)
    if evaluations is not None:
        lines = "".join(json.dumps(evaluation_record(step)) + "\n" for step in evaluations)
        (root / "evaluations.jsonl").write_text(lines)
    return root


def lines_of(figure):
    return {line.get_label(): line for ax in figure.axes for line in ax.lines}


def test_evaluation_figures_plot_the_requested_leg_head_and_window(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run", evaluations=(2, 4)))
    key = "flat/{}/{}/{}/decomposition_energy_score"
    for plotted, expected in (
        (
            run.plot.evaluation("decomposition_energy_score"),
            [("holdout", "conditional", "query"), ("seen", "conditional", "query")],
        ),
        (
            run.plot.evaluation(
                "decomposition_energy_score", head="joint", window="context", legs="seen"
            ),
            [("seen", "joint", "context")],
        ),
        (
            run.plot.heads("decomposition_energy_score", window="context", leg="holdout"),
            [("holdout", "joint", "context"), ("holdout", "conditional", "context")],
        ),
    ):
        lines = lines_of(plotted)
        assert set(lines) == {key.format(*cell) for cell in expected}
        for cell in expected:
            line = lines[key.format(*cell)]
            np.testing.assert_array_equal(line.get_xdata(), [2, 4])
            np.testing.assert_allclose(line.get_ydata(), [sentinel(*cell, 2), sentinel(*cell, 4)])
    structure = lines_of(run.plot.evaluation("auroc", head="structure_head", window="cy"))
    np.testing.assert_allclose(structure["flat/holdout/structure_head/cy/auroc"].get_ydata(), 0.9)
    np.testing.assert_allclose(structure["flat/seen/structure_head/cy/auroc"].get_ydata(), 0.95)
    with pytest.raises(ValueError, match="flat/holdout/conditional/query/absent"):
        run.plot.evaluation("absent")


def test_heads_overlay_every_head_that_records_the_metric(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run", evaluations=(2, 4)))
    edge = lines_of(run.plot.heads("auroc", window="cy"))
    # Decomposition heads record no edge scores: the structure head and the
    # joint router's edge marginals are the heads that score cy edges.
    assert set(edge) == {
        "flat/holdout/structure_head/cy/auroc",
        "flat/holdout/joint_router/edge/cy/auroc",
    }
    np.testing.assert_allclose(edge["flat/holdout/structure_head/cy/auroc"].get_ydata(), 0.9)
    np.testing.assert_allclose(edge["flat/holdout/joint_router/edge/cy/auroc"].get_ydata(), 0.45)
    seen = lines_of(run.plot.heads("auroc", window="cy", leg="seen"))
    np.testing.assert_allclose(seen["flat/seen/structure_head/cy/auroc"].get_ydata(), 0.95)
    with pytest.raises(ValueError, match="flat/holdout/structure_head/query/absent"):
        run.plot.heads("absent")


def test_evaluation_figures_state_each_heads_label_window_units_and_draws(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run", evaluations=(2, 4)))
    metric = "decomposition_energy_score"
    text = run.plot.evaluation(metric)._suptitle.get_text()
    # The recorded conditional label states the decoder's sampling mode.
    for stated in (HEAD_LABELS["conditional"], "query window", metric_units(metric), "7 draws"):
        assert stated in text, stated
    edges = run.plot.heads("auroc", window="cy")._suptitle.get_text()
    for stated in (HEAD_LABELS["structure_head"], HEAD_LABELS["joint_router"], "cy edges"):
        assert stated in edges, stated
    assert metric_units("auroc") in edges
    # No joint-head cy scores are recorded, so its label is not claimed.
    assert HEAD_LABELS["joint"] not in edges


def test_a_recorded_metric_this_release_does_not_define_draws_without_a_unit(tmp_path):
    from pymc_mfm.training.evaluation import METRIC_UNITS

    root = write_run(tmp_path / "run", evaluations=None)
    # A metric recorded by another release, which this one has no unit for.
    flat = {
        "holdout/conditional/query/some_new_score": 0.1,
        "seen/conditional/query/some_new_score": 0.2,
    }
    (root / "evaluations.jsonl").write_text(json.dumps({"step": 2, "flat": flat}) + "\n")
    run = TrainingRun.load(root)
    for figure in (run.plot.evaluation("some_new_score"), run.plot.heads("some_new_score")):
        line = lines_of(figure)["flat/holdout/conditional/query/some_new_score"]
        np.testing.assert_allclose(line.get_ydata(), [0.1])
        header = figure._suptitle.get_text()
        assert not any(f"units: {unit}" in header for unit in set(METRIC_UNITS.values()))


CORPUS_SUMMARY = {
    "corpora": 2,
    "train": {
        "worlds": 8,
        "cells": 4,
        "distinct_graphs": 3,
        "context_lengths": {"6": 6, "9": 2},
        "active_channels": {"2": 8},
        "active_controls": {"1": 2, "2": 6},
        "active_latents": {"1": 4, "2": 4},
        "edges": {"cy": {"1": 2, "2": 6}, "zb": {"0": 8}},
    },
    "holdout": {
        "worlds": 2,
        "cells": 1,
        "distinct_graphs": 1,
        "context_lengths": {"9": 2},
        "active_channels": {"2": 1, "3": 1},
        "active_controls": {"2": 2},
        "active_latents": {"2": 2},
        "edges": {"cy": {"3": 2}, "zb": {"1": 1, "2": 1}},
    },
    "shared_graphs": 1,
    "generator": {},
}


def bars_of(ax):
    """Bar heights keyed by ``(series label, category tick label)``."""
    from matplotlib.container import BarContainer

    ticks = [label.get_text() for label in ax.get_xticklabels()]
    return {
        (container.get_label(), ticks[round(bar.get_x() + bar.get_width() / 2)]): bar.get_height()
        for container in ax.containers
        if isinstance(container, BarContainer)
        for bar in container
    }


def by_split(train, holdout):
    """``bars_of`` keys and heights from per-split ``{category: height}`` mappings."""
    return {
        (split, category): height
        for split, heights in (("train", train), ("holdout", holdout))
        for category, height in heights.items()
    }


def test_corpus_figure_draws_the_recorded_corpus_summary(tmp_path):
    root = write_run(tmp_path / "run")
    with pytest.raises(ValueError, match="corpus_summary"):
        TrainingRun.load(root).plot.corpus()
    run = json.loads((root / "run.json").read_text())
    (root / "run.json").write_text(json.dumps({**run, "corpus_summary": CORPUS_SUMMARY}))
    sizes, context, channels, controls, latents, edges = TrainingRun.load(root).plot.corpus().axes
    assert bars_of(sizes) == by_split(
        {"Worlds": 8, "Cells": 4, "Distinct graphs": 3},
        {"Worlds": 2, "Cells": 1, "Distinct graphs": 1},
    )
    # Each split's share of its own worlds per recorded value.
    for ax, train, holdout in (
        (context, {"6": 6 / 8, "9": 2 / 8}, {"6": 0, "9": 1}),
        (channels, {"2": 1, "3": 0}, {"2": 0.5, "3": 0.5}),
        (controls, {"1": 2 / 8, "2": 6 / 8}, {"1": 0, "2": 1}),
        (latents, {"1": 0.5, "2": 0.5}, {"1": 0, "2": 1}),
    ):
        assert bars_of(ax) == pytest.approx(by_split(train, holdout))
    # Mean edges per world: train cy (2 x 1 + 6 x 2) / 8; held-out zb (1 + 2) / 2.
    expected = by_split({"cy": 1.75, "zb": 0}, {"cy": 3, "zb": 1.5})
    assert bars_of(edges) == pytest.approx(expected)


def test_single_axes_run_figures_draw_into_caller_axes_and_larger_ones_refuse(tmp_path):
    from matplotlib.figure import Figure

    root = write_run(tmp_path / "run", evaluations=(2, 4))
    run = TrainingRun.load(root)
    for series, draw in (
        ("lr", run.plot.learning_rate),
        ("grad_norm", run.plot.gradients),
        (
            "flat/holdout/conditional/query/decomposition_energy_score",
            lambda ax: run.plot.evaluation("decomposition_energy_score", ax=ax),
        ),
        ("flat/holdout/structure_head/cy/auroc", lambda ax: run.plot.heads("auroc", "cy", ax=ax)),
    ):
        figure = Figure()
        target, untouched = figure.subplots(1, 2)
        assert draw(ax=target) is figure, series
        assert figure._suptitle is None and "sentinel-run" in target.get_title(), series
        assert series in {line.get_label() for line in target.lines}, series
        assert not untouched.lines, series
    # Seven objective terms need two axes of at most six series each.
    extra = {f"extra_{index}_loss": 0.1 * index for index in range(3)}
    lines = [json.dumps({**training_record(step), **extra}) for step in (1, 2, 3)]
    (root / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    record = json.loads((root / "run.json").read_text())
    (root / "run.json").write_text(json.dumps({**record, "corpus_summary": CORPUS_SUMMARY}))
    grown = TrainingRun.load(root)
    for refuse in (grown.plot.losses, grown.plot.corpus):
        refused = Figure().subplots()
        with pytest.raises(ValueError):
            refuse(ax=refused)
        assert not (refused.lines or refused.patches or refused.get_title()), refuse


def test_training_figures_plot_recorded_objectives_schedule_and_gradients(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run"))
    losses = lines_of(run.plot.losses())
    assert set(losses) == {"loss", "flow_loss", "struct_loss", "struct_loss_raw"}
    np.testing.assert_allclose(losses["loss"].get_ydata(), [4.0, 3.0, 2.0])
    np.testing.assert_allclose(losses["struct_loss_raw"].get_ydata(), [0.25] * 3)
    learning_rate = lines_of(run.plot.learning_rate())["lr"]
    np.testing.assert_allclose(learning_rate.get_ydata(), [1e-3, 2e-3, 3e-3])
    figure = run.plot.gradients()
    gradients = lines_of(figure)
    np.testing.assert_array_equal(gradients["grad_norm"].get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(gradients["grad_norm"].get_ydata(), [0.5, 1.0, 1.5])
    np.testing.assert_array_equal(gradients["grad_clipped"].get_ydata(), [0, 0, 1])
    assert "sentinel-run" in figure._suptitle.get_text()


def test_derived_metrics_are_holdout_minus_seen_gaps_and_mean_edge_scores(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run", evaluations=(2, 4)))
    first, second = run.derived_metrics()
    assert (first["step"], second["step"]) == (2, 4)
    # Held-out minus seen: every decomposition sentinel differs by the leg codes, -100.
    assert first["holdout_minus_seen/conditional/query/decomposition_energy_score"] == (
        pytest.approx(-100.0)
    )
    assert first["holdout_minus_seen/structure_head/cy/auroc"] == pytest.approx(0.9 - 0.95)
    assert first["holdout_minus_seen/joint_router/graph/shd_mean"] == pytest.approx(1.0)
    assert first["holdout_minus_seen/structure_head/cc/auroc"] is None
    # Means over families with a defined score; the held-out cc score is undefined.
    assert first["holdout/structure_head/mean_auroc"] == pytest.approx((0.9 + 0.7) / 2)
    assert first["holdout/structure_head/mean_ece"] == pytest.approx((0.1 + 0.3) / 2)
    assert first["seen/structure_head/mean_auroc"] == pytest.approx((0.95 + 0.85 + 0.5) / 3)
    assert first["holdout/joint_router/edge/mean_auroc"] == pytest.approx((0.45 + 0.35) / 2)
    assert "holdout/joint_router/edge/mean_ece" not in first


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(json.dumps(training_record(4))[:25].encode(), id="cut-json"),
        # A write cut inside the two bytes of "é".
        pytest.param('{"step": 4, "note": "é"}'.encode()[:-3], id="cut-character"),
        # A write cut after a number too long for Python's integer conversion.
        pytest.param(b'{"step": 4, "loss": ' + b"1" * 5000, id="cut-long-integer"),
    ],
)
def test_an_unterminated_last_line_is_tolerated_and_reported(tmp_path, tail):
    root = write_run(tmp_path / "run")
    with (root / "metrics.jsonl").open("ab") as handle:
        handle.write(tail)
    run = TrainingRun.load(root)
    assert run.training["steps"] == [1, 2, 3]
    assert [(item["file"], item["line"]) for item in run.partial_lines] == [("metrics.jsonl", 4)]


@pytest.mark.parametrize(
    ("damage", "location"),
    [
        pytest.param(lambda lines: lines.insert(1, b"{broken"), "metrics.jsonl:2", id="middle"),
        pytest.param(lambda lines: lines.append(b"{broken"), "metrics.jsonl:4", id="terminated"),
        pytest.param(lambda lines: lines.insert(0, b"[1, 2]"), "metrics.jsonl:1", id="not-object"),
        pytest.param(
            lambda lines: lines.insert(1, b'{"note": "\xff\xfe"}'), "metrics.jsonl:2", id="not-utf8"
        ),
        pytest.param(
            lambda lines: lines.insert(1, b'{"step": 9, "loss": ' + b"1" * 5000 + b"}"),
            "metrics.jsonl:2",
            id="integer-past-digit-limit",
        ),
        pytest.param(
            lambda lines: lines.insert(1, b"[" * 100_000 + b"]" * 100_000),
            "metrics.jsonl:2",
            id="nesting-past-recursion-limit",
        ),
    ],
)
def test_corrupt_records_raise_with_file_and_line(tmp_path, damage, location):
    root = write_run(tmp_path / "run")
    lines = (root / "metrics.jsonl").read_bytes().splitlines()
    damage(lines)
    (root / "metrics.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ValueError, match=location):
        TrainingRun.load(root)
    # A partial last line of a run still writing never excuses earlier corruption.
    with (root / "metrics.jsonl").open("ab") as handle:
        handle.write(b'{"step": 5, "lr"')
    with pytest.raises(ValueError, match=location):
        TrainingRun.load(root)


def test_integers_past_int64_are_finite_record_values(tmp_path):
    root = write_run(tmp_path / "run")
    with (root / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": 2**64, "loss": 2**70}) + "\n")
    run = TrainingRun.load(root)
    assert run.training["steps"][-1] == 2**64
    assert run.training["series"]["loss"][-1]["value"] == 2**70


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b'{"status": "\xff"}', id="not-utf8"),
        pytest.param(b'{"final_step": ' + b"1" * 5000 + b"}", id="integer-past-digit-limit"),
    ],
)
def test_undecodable_run_records_name_their_file(tmp_path, content):
    root = write_run(tmp_path / "run")
    (root / "run.json").write_bytes(content)
    with pytest.raises(ValueError, match="run.json"):
        TrainingRun.load(root)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(b'{"qualification": "q"}', id="no-worlds"),
        pytest.param(b'{"worlds": [{"mean": 1.0}], "qualification": "q"}', id="no-std"),
        pytest.param(b'{"worlds": [{"mean": NaN, "std": 1.0}], "qualification": "q"}', id="nan"),
        # How statistics that overflowed were once written.
        pytest.param(
            b'{"worlds": [{"mean": {"status": "nonfinite"}, "std": 1.0}], "qualification": "q"}',
            id="status-object",
        ),
        pytest.param(b'{"worlds": "\xff"}', id="not-utf8"),
        pytest.param(
            b'{"worlds": [{"mean": 1.0, "std": 1.0}], "n_worlds": -1, "qualification": "q"}',
            id="negative-n-worlds",
        ),
        pytest.param(
            b'{"worlds": [{"mean": 1.0, "std": 1.0}], "n_worlds": 1, "qualification": 5}',
            id="qualification-not-text",
        ),
        pytest.param(
            b'{"worlds": [{"mean": 1.0, "std": 1.0}], "skipped": 5, "qualification": "q"}',
            id="skipped-not-an-object",
        ),
        pytest.param(
            b'{"worlds": [{"mean": 1.0, "std": 1.0}], "n_worlds": true, "qualification": "q"}',
            id="boolean-n-worlds",
        ),
    ],
)
def test_malformed_corpus_support_names_its_file(tmp_path, content):
    root = write_run(tmp_path / "run")
    (root / "corpus_support.json").write_bytes(content)
    with pytest.raises(ValueError, match="corpus_support.json"):
        TrainingRun.load(root).plot.corpus_support()


def test_missing_records_give_empty_series_and_figures_name_the_file(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run", evaluations=None))
    assert run.evaluation["series"] == {} and run.derived_metrics() == []
    with pytest.raises(ValueError, match="evaluations.jsonl"):
        run.plot.evaluation("decomposition_energy_score")
    with pytest.raises(ValueError, match="evaluations.jsonl"):
        run.plot.heads("decomposition_energy_score")
    empty = TrainingRun.load(write_run(tmp_path / "empty", training=None, evaluations=None))
    for method in ("losses", "learning_rate", "gradients"):
        with pytest.raises(ValueError, match="metrics.jsonl"):
            getattr(empty.plot, method)()
    with pytest.raises(FileNotFoundError, match="run.json"):
        TrainingRun.load(tmp_path / "absent")


def test_progress_comes_from_records_not_the_run_summary(tmp_path):
    run = TrainingRun.load(write_run(tmp_path / "run", training=(1, 2, 3), evaluations=(2,)))
    assert run.run["final_step"] == 0
    assert run.last_step == 3
    assert "step 3" in run.plot.learning_rate()._suptitle.get_text()


def test_reduced_train_records_reload_through_the_training_result(tmp_path):
    from test_training import TINY, reduced

    from pymc_mfm.training import TrackingConfig, train
    from pymc_mfm.training.trainer import LOSS_TERMS

    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=3)
    result = train(
        [corpus],
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(),
        device="cpu",
    )
    metrics, evaluations = (
        [json.loads(line) for line in (tmp_path / "run" / name).read_text().splitlines()]
        for name in ("metrics.jsonl", "evaluations.jsonl")
    )
    history = result.records()
    assert history.training["steps"] == [1, 2, 3, 4] and history.evaluation["steps"] == [2, 4]
    assert history.last_step == result.final_step == 4
    assert history.partial_lines == ()
    losses = lines_of(history.plot.losses())
    for name in LOSS_TERMS:
        np.testing.assert_allclose(losses[name].get_ydata(), [record[name] for record in metrics])
    figure = history.plot.evaluation("decomposition_energy_score")
    evaluation = lines_of(figure)
    for leg in ("holdout", "seen"):
        key = f"{leg}/conditional/query/decomposition_energy_score"
        np.testing.assert_allclose(
            evaluation[f"flat/{key}"].get_ydata(), [event["flat"][key] for event in evaluations]
        )
    # A trained run's leg records carry the head labels the header states.
    assert HEAD_LABELS["conditional"] in figure._suptitle.get_text()
    gap = "conditional/query/decomposition_energy_score"
    for derived, event in zip(history.derived_metrics(), evaluations, strict=True):
        assert derived[f"holdout_minus_seen/{gap}"] == pytest.approx(
            event["flat"][f"holdout/{gap}"] - event["flat"][f"seen/{gap}"]
        )
    # The corpus figure reads run.json; recount the corpus arrays per split.
    sizes, context, channels, _, _, edges = history.plot.corpus().axes
    rows = {"train": corpus["is_val"] == 0, "holdout": corpus["is_val"] == 1}
    graphs = corpus["g"]
    assert bars_of(sizes) == by_split(
        *(
            {
                "Worlds": keep.sum(),
                "Cells": np.unique(corpus["cell_id"][keep]).size,
                "Distinct graphs": np.unique(graphs[keep], axis=0).shape[0],
            }
            for keep in rows.values()
        )
    )
    for ax, per_world in (
        (context, corpus["support_mask"].sum(axis=1)),
        (channels, corpus["treatment_active_mask"].sum(axis=1)),
    ):
        values = sorted(set(per_world.tolist()))
        expected = by_split(
            *({str(v): np.mean(per_world[keep] == v) for v in values} for keep in rows.values())
        )
        assert bars_of(ax) == pytest.approx(expected)
    masks = ("treatment_active_mask", "covariate_active_mask", "latent_active_mask")
    layout = EdgeSlotLayout(*(corpus[key].shape[1] for key in masks))
    per_family = by_split(
        *(
            {
                family: graphs[keep][:, block].sum(axis=1).mean()
                for family, block in layout.slices.items()
            }
            for keep in rows.values()
        )
    )
    assert bars_of(edges) == pytest.approx(per_family)
