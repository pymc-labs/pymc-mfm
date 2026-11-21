"""Run-start corpus summary and per-module parameter counts, in run.json and MLflow."""

import json
import math

import numpy as np
import pytest
import torch
from conftest import GENERATOR_EDGE_TYPES, make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mktfm.model.config import ModelConfig
from pymc_mktfm.model.graph import EdgeSlotLayout
from pymc_mktfm.model.model import MFM
from pymc_mktfm.training import TrackingConfig, train
from pymc_mktfm.training.summary import corpus_summary, corpus_support, parameter_counts

LAYOUT = EdgeSlotLayout(3, 2, 1)
MODULES = ("encoder", "router", "structure_head", "graph_conditioner", "conditional_head")


def graph(**edges: list[int]) -> list[int]:
    """A flat ``g`` row with edges at the given within-family slot offsets."""
    row = np.zeros(LAYOUT.n_slots, dtype=np.uint8)
    for family, offsets in edges.items():
        row[[LAYOUT.slices[family].start + offset for offset in offsets]] = 1
    return row.tolist()


G0 = graph(cy=[0, 1], zb=[0])
G2 = graph(cy=[0, 1, 2], cc=[0])
G3 = graph(cy=[0], db=[0])


def corpus(*, is_val, cells, lengths, channels, controls, latents, graphs, diagnostics):
    support = np.arange(8)[None, :] < np.asarray(lengths)[:, None]
    return {
        "diagnostics": diagnostics,
        "is_val": np.asarray(is_val, dtype=np.uint8),
        "cell_id": np.asarray(cells, dtype=np.int32),
        "support_mask": support.astype(np.uint8),
        "treatment_active_mask": np.asarray(channels, dtype=np.uint8),
        "covariate_active_mask": np.asarray(controls, dtype=np.uint8),
        "latent_active_mask": np.asarray(latents, dtype=np.uint8),
        "g": np.asarray(graphs, dtype=np.uint8),
    }


def hand_built_corpora():
    first = corpus(
        is_val=[0, 0, 1, 1],
        cells=[0, 0, 1, 1],
        lengths=[5, 7, 5, 5],
        channels=[[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
        controls=[[1, 1], [1, 1], [1, 0], [1, 0]],
        latents=[[1], [1], [0], [1]],
        graphs=[G0, G0, G2, G3],
        diagnostics={
            "schema_version": 4,
            "edge_types": GENERATOR_EDGE_TYPES,
            "n_draw_failures": 2,
            "signal": {"quantiles": {"q10": 0.25, "q90": None}},
            "validated": True,
            "semantics": "text",
            "rate": math.nan,
        },
    )
    # The same graph recurs across corpora within a split and across splits.
    second = corpus(
        is_val=[0, 0, 1],
        cells=[0, 0, 1],
        lengths=[7, 5, 7],
        channels=[[1, 0, 0], [1, 1, 0], [1, 1, 0]],
        controls=[[1, 0], [1, 1], [1, 1]],
        latents=[[1], [1], [1]],
        graphs=[G3, G0, G0],
        diagnostics={
            "schema_version": 4,
            "n_draw_failures": 5,
            "signal": {"quantiles": {"q10": 0.75}},
            "elapsed": np.float32(1.5),
            "retries": np.int64(3),
            "accepted": np.bool_(True),
        },
    )
    return [first, second]


def test_corpus_summary_counts_each_split_across_corpora():
    summary = corpus_summary(hand_built_corpora())
    assert summary == {
        "corpora": 2,
        "train": {
            "worlds": 4,
            "cells": 2,
            "distinct_graphs": 2,
            "context_lengths": {"5": 2, "7": 2},
            "active_channels": {"1": 1, "2": 3},
            "active_controls": {"1": 1, "2": 3},
            "active_latents": {"1": 4},
            "edges": {
                "cy": {"1": 1, "2": 3},
                **dict.fromkeys(("dc", "dz", "zc", "cc", "zz"), {"0": 4}),
                "db": {"0": 3, "1": 1},
                "zb": {"0": 1, "1": 3},
            },
        },
        "holdout": {
            "worlds": 3,
            "cells": 2,
            "distinct_graphs": 3,
            "context_lengths": {"5": 2, "7": 1},
            "active_channels": {"1": 1, "2": 1, "3": 1},
            "active_controls": {"1": 2, "2": 1},
            "active_latents": {"0": 1, "1": 2},
            "edges": {
                "cy": {"1": 1, "2": 1, "3": 1},
                **dict.fromkeys(("dc", "dz", "zc", "zz"), {"0": 3}),
                "db": {"0": 2, "1": 1},
                "zb": {"0": 2, "1": 1},
                "cc": {"0": 2, "1": 1},
            },
        },
        "shared_graphs": 2,
        "generator": {
            "elapsed": {"min": 1.5, "max": 1.5},
            "n_draw_failures": {"min": 2, "max": 5},
            "retries": {"min": 3, "max": 3},
            "schema_version": {"min": 4, "max": 4},
            "signal.quantiles.q10": {"min": 0.25, "max": 0.75},
        },
    }
    json.dumps(summary, allow_nan=False)


def test_corpus_summary_rejects_corpora_of_different_graph_geometry():
    first, second = hand_built_corpora()
    wider = make_synthetic_corpus(k=4, m=2, j=1)
    with pytest.raises(ValueError):
        corpus_summary([first, wider])
    second["g"] = second["g"][:, :-1]
    with pytest.raises(ValueError):
        corpus_summary([first, second])


def test_parameter_counts_of_the_ab_geometry():
    with torch.device("meta"):
        model = MFM(ModelConfig(n_controls=6))
    assert parameter_counts(model) == {
        "encoder": 1_316_480,
        "router": 544_880,
        "structure_head": 50_568,
        "graph_conditioner": 182_784,
        "conditional_head": 348_871,
        "total": 2_443_583,
    }


def outcome_corpora():
    corpora = hand_built_corpora()
    rng = np.random.default_rng(3)
    for corpus in corpora:
        outcome = rng.uniform(1.0, 2.0, size=corpus["support_mask"].shape)
        # Query weeks would dominate any statistic that read past the context.
        corpus["outcome_raw"] = np.where(corpus["support_mask"], outcome, 1e6).astype(np.float32)
    return corpora


def test_corpus_support_describes_context_outcomes_and_subsamples_evenly():
    corpora = outcome_corpora()

    def check(summary, rows):
        assert summary["n_worlds"] == 7
        assert summary["skipped"] == {}
        assert [world["world"] for world in summary["worlds"]] == [f"{i}:{r}" for i, r in rows]
        for world, (index, row) in zip(summary["worlds"], rows, strict=True):
            corpus = corpora[index]
            values = corpus["outcome_raw"][row, corpus["support_mask"][row].astype(bool)]
            assert world["n_context"] == values.size
            assert world["mean"] == pytest.approx(values.astype(np.float64).mean())

    check(corpus_support(corpora), [(0, row) for row in range(4)] + [(1, row) for row in range(3)])
    # Seven rows in corpus order, three kept: rows 0, 3 and 6 of the concatenation.
    subset = corpus_support(corpora, max_worlds=3)
    check(subset, [(0, 0), (0, 3), (1, 2)])
    json.dumps(subset, allow_nan=False)


def test_corpus_support_skips_unusable_worlds_instead_of_failing():
    corpora = outcome_corpora()
    corpora[0]["outcome_raw"][1, 2] = np.nan
    corpora[0]["outcome_raw"][2, 6] = np.nan  # a query week: only the context is described
    corpora[1]["support_mask"][0] = 0
    # Finite float64 context outcomes whose statistics overflow: world 1:2's mean
    # and standard deviation, world 1:1's standard deviation only.
    corpora[1]["outcome_raw"] = corpora[1]["outcome_raw"].astype(np.float64)
    corpora[1]["outcome_raw"][2, :3] = 1e308
    corpora[1]["outcome_raw"][1, :3] = [1e308, -1e308, 1e308]
    summary = corpus_support(corpora)
    assert summary["skipped"] == {
        "0:1": "nonfinite_context_outcome",
        "1:0": "empty_context",
        "1:1": "nonfinite_context_statistics",
        "1:2": "nonfinite_context_statistics",
    }
    assert [world["world"] for world in summary["worlds"]] == ["0:0", "0:2", "0:3"]
    json.dumps(summary, allow_nan=False)
    for corpus in corpora:
        corpus["support_mask"][:] = 0
    unusable = corpus_support(corpora)
    assert unusable["worlds"] == []
    pytest.importorskip("matplotlib")
    from pymc_mktfm.plotting.comparisons import plot_corpus_support

    # The refusal states how many kept worlds were skipped, and why.
    with pytest.raises(ValueError) as refused:
        plot_corpus_support(unusable)
    assert "empty_context" in str(refused.value)
    assert str(len(unusable["skipped"])) in str(refused.value).split()


def test_training_records_the_summary_and_counts_in_run_json_and_mlflow(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    out = tmp_path / "run"
    result = train(
        [make_synthetic_corpus(n_cells=4, draws_per_cell=1, context_lengths=(6,), seed=8)],
        out_dir=out,
        model_config=TINY,
        training=reduced(
            steps=2,
            warmup_steps=1,
            eval_every=2,
            eval_worlds=1,
            eval_samples=2,
            milestone_steps=(),
            prefetch_batches=0,
        ),
        device="cpu",
        tracking=TrackingConfig(diagnostics=False),
    )
    run = json.loads((out / "run.json").read_text())
    summary, counts = run["corpus_summary"], run["parameter_counts"]
    budget = run["consumption_budget"]
    assert (summary["train"]["worlds"], summary["holdout"]["worlds"]) == (
        budget["train_rows"],
        budget["holdout_rows"],
    )
    assert counts["total"] == run["total_parameter_count"] == sum(counts[name] for name in MODULES)

    def scalars(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from scalars(item, (*path, key))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield "corpus/" + "/".join(path), float(value)

    expected = dict(scalars(summary))
    names = json.loads((out / "tracking" / "names.json").read_text())
    keys = {(entry["kind"], entry["original"]): key for key, entry in names.items()}
    client = mlflow.MlflowClient(tracking_uri=result.run["tracking"]["tracking_uri"])
    run_id = result.run["tracking"]["run_id"]
    logged = {
        original: [(item.step, item.value) for item in client.get_metric_history(run_id, key)]
        for (kind, original), key in keys.items()
        if kind == "metric" and original.startswith("corpus/")
    }
    assert logged == {original: [(0, value)] for original, value in expected.items()}
    params = client.get_run(run_id).data.params
    for name, count in counts.items():
        assert params[keys[("parameter", f"config/parameter_counts/{name}")]] == str(count)
    assert not [original for _, original in keys if original.startswith("config/corpus_summary")]
    support = json.loads((out / "corpus_support.json").read_text())
    assert support["n_worlds"] == budget["train_rows"] + budget["holdout_rows"]
    assert not [original for _, original in keys if "corpus_support" in original]
    pytest.importorskip("matplotlib")
    figure = result.records().plot.corpus_support()
    cloud = figure.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(
        cloud, [(world["mean"], world["std"]) for world in support["worlds"]], rtol=1e-6
    )
