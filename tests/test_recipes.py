"""Synthetic prior recipes and corpus loading identity."""

import dataclasses
import importlib.util
import json
import subprocess
import sys

import pytest
from conftest import make_synthetic_corpus

from pymc_mfm.data.profiles import PROFILE_A, PROFILE_B
from pymc_mfm.data.recipes import (
    PROFILE_A_PRIOR,
    load_corpora,
    prior_sidecar_path,
    profile_prior,
    provenance,
    recipe_for,
)

needs_generator = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)


def test_profile_a_prior_declares_supported_families_and_edge_budgets():
    assert set(PROFILE_A_PRIOR["edge_budget"]) == {"cy", "dy", "zy", "dc", "dz", "zc", "cc", "zz"}
    assert PROFILE_A_PRIOR["edge_budget"]["cy"] == (5, 5)
    assert PROFILE_A_PRIOR["edge_budget"]["zy"] == (3, 3)
    assert PROFILE_A_PRIOR["baseline_floor_scope"] == "non_treatment"
    assert set(PROFILE_A_PRIOR["carryover_family_probs"]) == {"none", "geometric", "weibull"}
    assert set(PROFILE_A_PRIOR["saturation_family_probs"]) == {
        "linear",
        "hill",
        "logistic",
        "michaelis_menten",
        "tanh",
        "root",
    }
    ranges = [value for value in PROFILE_A_PRIOR.values() if isinstance(value, (tuple, list))]
    assert ranges and all(isinstance(value, tuple) for value in ranges)


def test_profile_b_changes_only_live_channel_budget():
    a, b = recipe_for(PROFILE_A), recipe_for(PROFILE_B)
    assert b.payload["edge_budget"]["cy"] == PROFILE_B.live_channels == (4, 10)
    assert {k: v for k, v in b.payload.items() if k != "edge_budget"} == {
        k: v for k, v in a.payload.items() if k != "edge_budget"
    }
    assert {k: v for k, v in b.payload["edge_budget"].items() if k != "cy"} == {
        k: v for k, v in a.payload["edge_budget"].items() if k != "cy"
    }
    with pytest.raises(KeyError):
        recipe_for(dataclasses.replace(PROFILE_A, name="C"))


def test_import_does_not_load_the_generator():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pymc_mfm.data.recipes, sys; assert 'pymc_generator' not in sys.modules",
        ],
        check=True,
    )


@needs_generator
def test_profile_prior_constructs_exactly_the_payload():
    prior = profile_prior(PROFILE_A, n_cells=2, draws_per_cell=1, seed=1)
    assert dataclasses.asdict(prior) == {
        **PROFILE_A_PRIOR,
        "n_cells": 2,
        "draws_per_cell": 1,
        "seed": 1,
    }
    profile_b = profile_prior(PROFILE_B, n_cells=2, draws_per_cell=1, seed=1)
    assert dataclasses.asdict(profile_b)["edge_budget"]["cy"] == (4, 10)
    overridden = profile_prior(PROFILE_A, n_cells=2, draws_per_cell=1, seed=1, n_time_steps=52)
    assert overridden.n_time_steps == 52
    assert PROFILE_A_PRIOR["edge_budget"]["cy"] == (5, 5)


@needs_generator
def test_load_corpora_reports_identity_and_sidecar(tmp_path):
    from pymc_generator import save_corpus

    corpus = make_synthetic_corpus(n_cells=6, draws_per_cell=2, val_cells=2, context_lengths=(6, 9))
    path = tmp_path / "corpus.npz"
    save_corpus(corpus, path)

    (loaded,) = load_corpora([path])
    assert (loaded.n_worlds, loaded.n_train, loaded.n_holdout) == (12, 8, 4)
    assert loaded.context_lengths == (6, 9)
    assert loaded.corpus["diagnostics"]["schema_version"] == 4
    assert len(loaded.sha256) == 64
    assert provenance(loaded)["prior_sidecar"] is None

    prior_sidecar_path(path).write_text(json.dumps({"prior": {"seed": 7}}))
    record = provenance(loaded)
    assert record["prior_sidecar"] == {"prior": {"seed": 7}}
    assert record["context_lengths"] == [6, 9]
    json.dumps(record, allow_nan=False)


def test_synthetic_sharing_permission_requires_hash_bound_origin(tmp_path):
    from pymc_mfm.data.recipes import LoadedCorpus

    path = tmp_path / "corpus.npz"
    loaded = LoadedCorpus(str(path), "a" * 64, 2, 1, 1, (6,), {})
    assert provenance(loaded)["data_origin"] == "unknown"
    sidecar = prior_sidecar_path(path)
    sidecar.write_text(json.dumps({"data_origin": "synthetic", "corpus_sha256": "b" * 64}))
    assert provenance(loaded)["data_origin"] == "unknown"
    sidecar.write_text(json.dumps({"data_origin": "synthetic", "corpus_sha256": loaded.sha256}))
    assert provenance(loaded)["data_origin"] == "synthetic"


@needs_generator
def test_load_corpora_rejects_missing_split_arrays(tmp_path):
    from pymc_generator import save_corpus

    corpus = make_synthetic_corpus()
    del corpus["is_val"]
    path = tmp_path / "no_split.npz"
    save_corpus(corpus, path)
    with pytest.raises(ValueError, match="is_val"):
        load_corpora([path])


@needs_generator
def test_cli_overrides_reach_the_prior(monkeypatch, tmp_path):
    import pymc_mfm.data.__main__ as cli

    captured = {}

    def fake_generate(prior, path):
        captured["prior"] = prior
        captured["path"] = path
        return {"is_val": make_synthetic_corpus()["is_val"]}

    monkeypatch.setattr(cli, "generate_corpus", fake_generate)
    out = tmp_path / "b.npz"
    assert (
        cli.main(
            [
                "generate",
                "--profile",
                "B",
                "--n-cells",
                "3",
                "--draws-per-cell",
                "2",
                "--seed",
                "9",
                "--out",
                str(out),
                "--override",
                'edge_budget={"cy": [6, 6]}',
                "--override",
                "n_time_steps=52",
            ]
        )
        == 0
    )
    prior = captured["prior"]
    assert captured["path"] == str(out)
    assert (prior.n_cells, prior.draws_per_cell, prior.seed) == (3, 2, 9)
    assert prior.edge_budget == {"cy": (6, 6)}
    assert prior.n_time_steps == 52
