"""Synthetic prior recipes, shard plans, corpus generation and loading identity."""

import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_synthetic_corpus

import pymc_mfm.data.__main__ as cli
from pymc_mfm.data.profiles import PROFILE_A, PROFILE_B
from pymc_mfm.data.recipes import (
    PROFILE_A_PRIOR,
    SHARD_PLANS,
    ShardPlan,
    generate_corpus,
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


@pytest.mark.parametrize(
    ("name", "profile", "seeds"),
    [("A", PROFILE_A, range(530200, 530360)), ("B", PROFILE_B, range(530400, 530720))],
)
def test_profile_shard_plans_tile_the_profile_corpus(name, profile, seeds):
    plan = SHARD_PLANS[name]
    assert plan.name == name and plan.profile is profile
    assert (plan.n_cells, plan.draws_per_cell) == (125, 10)
    assert plan.n_shards * plan.n_cells * plan.draws_per_cell == plan.worlds == profile.worlds
    assert [plan.seed(shard) for shard in range(plan.n_shards)] == list(seeds)
    assert plan.shard_overrides(0) == {} and plan.shard_names is None
    for invalid in (-1, plan.n_shards, True, 1.0):
        with pytest.raises(ValueError):
            plan.seed(invalid)


def test_evaluation_plan_fixes_the_active_channel_count_of_each_shard():
    plan = SHARD_PLANS["evaluation"]
    assert plan.profile is PROFILE_A
    assert (plan.n_shards, plan.n_cells, plan.draws_per_cell, plan.worlds) == (10, 32, 4, 1280)
    assert [plan.seed(shard) for shard in range(10)] == list(range(540000, 549001, 1000))
    assert [plan.shard_overrides(shard) for shard in range(10)] == [
        {"n_treatments_active_range": (active, active)} for active in range(6, 16)
    ]
    assert plan.shard_names == tuple(f"eval_a{active:02d}" for active in range(6, 16))
    with pytest.raises(ValueError):
        plan.shard_overrides(10)


def test_shard_plan_rejects_per_shard_entries_that_do_not_match_its_shards():
    geometry = dict(
        name="p", profile=PROFILE_A, first_seed=1, seed_stride=1, n_cells=2, draws_per_cell=1
    )
    with pytest.raises(ValueError):
        ShardPlan(**geometry, n_shards=2, shard_names=("only",))
    with pytest.raises(ValueError):
        ShardPlan(**geometry, n_shards=2, overrides=({},) * 3)
    with pytest.raises(ValueError):
        ShardPlan(**{**geometry, "n_cells": 1}, n_shards=1)


def test_import_does_not_load_the_generator():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import pymc_mfm.data.recipes, sys; assert 'pymc_generator' not in sys.modules",
        ],
        check=True,
    )


def test_cli_generation_requires_the_repository_group(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pymc_generator", None)
    with pytest.raises(ImportError, match="uv sync --locked --group generation"):
        cli.main(["generate", "--evaluation-shard", "0", "--out", str(tmp_path)])


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
def test_plan_prior_keeps_the_shard_seed_draws_and_overrides_for_fewer_cells():
    prior = SHARD_PLANS["evaluation"].prior(9, n_cells=2)
    assert dataclasses.asdict(prior) == {
        **PROFILE_A_PRIOR,
        "n_cells": 2,
        "draws_per_cell": 4,
        "seed": 549000,
        "n_treatments_active_range": (15, 15),
    }


@pytest.mark.parametrize("version", [4, 5])
def test_load_corpora_reports_identity_and_sidecar(tmp_path, monkeypatch, version):
    monkeypatch.setitem(sys.modules, "pymc_generator", None)
    corpus = make_synthetic_corpus(n_cells=6, draws_per_cell=2, val_cells=2, context_lengths=(6, 9))
    diagnostics = dict(corpus.pop("diagnostics"), schema_version=version)
    if version == 5:
        corpus["trajectory_treatment_hf_sigma"] = np.zeros((12, 3), dtype=np.float64)
    labels = {"signal_metric_valid": np.ones((12, 3, 1), dtype=np.uint8)}
    path = tmp_path / "corpus.npz"
    np.savez_compressed(
        path,
        **corpus,
        diagnostics=np.array(json.dumps(diagnostics)),
        identifiability__signal_metric_valid=labels["signal_metric_valid"],
    )

    (loaded,) = load_corpora([path])
    assert (loaded.n_worlds, loaded.n_train, loaded.n_holdout) == (12, 8, 4)
    assert loaded.context_lengths == (6, 9)
    assert loaded.corpus["diagnostics"] == diagnostics
    assert set(loaded.corpus) == {*corpus, "diagnostics", "identifiability"}
    for key, value in corpus.items():
        np.testing.assert_array_equal(loaded.corpus[key], value, strict=True)
    assert loaded.corpus["identifiability"].keys() == labels.keys()
    for key, value in labels.items():
        np.testing.assert_array_equal(loaded.corpus["identifiability"][key], value, strict=True)
    assert loaded.path == str(path)
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
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


@pytest.mark.parametrize("missing", ["is_val", "cell_id", "support_mask"])
def test_load_corpora_rejects_missing_split_arrays(tmp_path, missing):
    corpus = make_synthetic_corpus()
    del corpus[missing]
    corpus["diagnostics"] = np.array(json.dumps(corpus["diagnostics"]))
    path = tmp_path / "no_split.npz"
    np.savez(path, **corpus)
    with pytest.raises(ValueError, match=rf"no_split\.npz: missing required arrays.*{missing}"):
        load_corpora([path])


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (np.array(["{}"]), "scalar diagnostics JSON"),
        (np.array("{not json"), "invalid diagnostics JSON"),
        (np.array("[]"), "diagnostics JSON must contain an object"),
        (np.array("{}"), "schema_version"),
        (np.array('{"schema_version": true}'), "schema_version"),
        (np.array('{"schema_version": 4.0}'), "schema_version"),
        (np.array('{"schema_version": 6}'), "schema_version"),
        (np.array('{"schema_version": 4, "edge_budget": {"db": 1}}'), "pre-v3 outcome-edge"),
        (np.array('{"schema_version": 5, "min_dead_channels": 0}'), "pre-v3 min_dead_channels"),
    ],
)
def test_load_corpora_rejects_invalid_diagnostics(tmp_path, diagnostics, message):
    path = tmp_path / "invalid.npz"
    np.savez(path, diagnostics=diagnostics)
    with pytest.raises(ValueError, match=rf"invalid\.npz: .*{message}"):
        load_corpora([path])


def test_load_corpora_rejects_legacy_dimensions_in_stamped_archive(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, diagnostics=np.array('{"schema_version": 4}'), K_active=np.array([1]))
    with pytest.raises(ValueError, match="legacy dimension keys.*K_active"):
        load_corpora([path])


def test_load_corpora_rejects_object_members(tmp_path):
    path = tmp_path / "object.npz"
    corpus = make_synthetic_corpus()
    corpus["diagnostics"] = np.array(json.dumps(corpus["diagnostics"]))
    np.savez(path, **corpus, identifiability__invalid=np.array([None], dtype=object))
    with pytest.raises(ValueError, match=r"object\.npz: .*Object arrays cannot be loaded"):
        load_corpora([path])


@pytest.mark.parametrize("content", [b"not an archive", b"PK\x03\x04truncated"])
def test_an_unreadable_corpus_names_its_file(tmp_path, content):
    good = tmp_path / "good.npz"
    corpus = make_synthetic_corpus()
    corpus["diagnostics"] = np.array(json.dumps(corpus["diagnostics"]))
    np.savez(good, **corpus)
    bad = tmp_path / "bad.npz"
    bad.write_bytes(content)
    with pytest.raises(ValueError, match=r"bad\.npz: not a readable NPZ archive"):
        load_corpora([good, bad])


def test_a_member_with_an_unsupported_compression_names_the_file(tmp_path):
    path = tmp_path / "packed.npz"
    corpus = make_synthetic_corpus()
    corpus["diagnostics"] = np.array(json.dumps(corpus["diagnostics"]))
    np.savez(path, **corpus)
    data = bytearray(path.read_bytes())
    # The compression method of the first central-directory record; zipfile supports no 99.
    central = data.index(b"PK\x01\x02")
    data[central + 10 : central + 12] = (99).to_bytes(2, "little")
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=r"packed\.npz: cannot read NPZ member"):
        load_corpora([path])


@needs_generator
def test_generate_corpus_replaces_a_corpus_only_with_a_complete_one(monkeypatch, tmp_path):
    import pymc_generator

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    generated = make_synthetic_corpus()
    monkeypatch.setattr(
        pymc_generator, "DataGenerator", lambda prior: SimpleNamespace(generate=lambda: generated)
    )
    prior = profile_prior(PROFILE_A, n_cells=2, draws_per_cell=1, seed=1)
    path = tmp_path / "corpus.npz"
    sidecar = prior_sidecar_path(path)
    generate_corpus(prior, path)
    assert sorted(tmp_path.iterdir()) == [path, sidecar]
    recorded = json.loads(sidecar.read_text())
    assert recorded["corpus_sha256"] == digest(path)
    assert recorded["prior"]["seed"] == 1
    before = (path.read_bytes(), sidecar.read_bytes())

    def interrupted(corpus, partial):
        partial.write_bytes(b"partial")
        raise RuntimeError("interrupted while saving")

    monkeypatch.setattr(pymc_generator, "save_corpus", interrupted)
    with pytest.raises(RuntimeError):
        generate_corpus(prior, path)
    assert (path.read_bytes(), sidecar.read_bytes()) == before
    assert sorted(tmp_path.iterdir()) == [path, sidecar]
    with pytest.raises(ValueError):
        generate_corpus(prior, tmp_path / "corpus.bin")


def _refuse_generation(prior, path):
    raise AssertionError("the CLI generated a corpus for invalid arguments")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--profile", "A", "--shard", "160"],
        ["--profile", "B", "--shard", "320"],
        ["--profile", "A", "--shard", "-1"],
        ["--evaluation-shard", "10"],
        ["--evaluation-shard", "-1"],
        ["--shard", "0"],
        ["--profile", "A", "--shard", "0", "--n-cells", "2"],
        ["--profile", "A", "--shard", "0", "--draws-per-cell", "10"],
        ["--profile", "A", "--shard", "0", "--seed", "530200"],
        ["--profile", "A", "--shard", "0", "--override", "n_time_steps=52"],
        ["--evaluation-shard", "0", "--profile", "A"],
        ["--evaluation-shard", "0", "--profile", "A", "--shard", "0"],
        ["--evaluation-shard", "0", "--override", "n_time_steps=52"],
        ["--evaluation-shard", "0", "--seed", "540000"],
        ["--profile", "A", "--n-cells", "2", "--draws-per-cell", "10"],
        ["--n-cells", "2", "--draws-per-cell", "10", "--seed", "1"],
    ],
)
def test_cli_rejects_invalid_shard_arguments_before_generating(arguments, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "generate_corpus", _refuse_generation)
    with pytest.raises(SystemExit) as raised:
        cli.main(["generate", *arguments, "--out", str(tmp_path)])
    assert raised.value.code == 2
    assert not list(tmp_path.iterdir())


def test_cli_refuses_an_unusable_or_existing_target_before_generating(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "generate_corpus", _refuse_generation)
    existing = tmp_path / "shard_540000.npz"
    existing.write_bytes(b"existing corpus")
    for out in (tmp_path, existing, tmp_path / "corpus"):
        with pytest.raises(SystemExit) as raised:
            cli.main(["generate", "--evaluation-shard", "0", "--out", str(out)])
        assert raised.value.code == 2
    assert existing.read_bytes() == b"existing corpus"
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory permissions")
def test_cli_refuses_an_unwritable_directory_before_generating(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "generate_corpus", _refuse_generation)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        for out in (locked, locked / "new" / "shard.npz"):
            with pytest.raises(SystemExit) as raised:
                cli.main(["generate", "--evaluation-shard", "0", "--out", str(out)])
            assert raised.value.code == 2
    finally:
        locked.chmod(0o755)
    assert not list(locked.iterdir())


@needs_generator
@pytest.mark.parametrize(
    "extra",
    [
        ["--n-cells", "0"],
        ["--n-cells", "1", "--override", "seed=5"],
        ["--n-cells", "1", "--override", "n_time_step=52"],
    ],
    ids=["invalid-value", "flag-field-override", "unknown-field-override"],
)
def test_cli_reports_an_invalid_custom_prior_as_a_usage_error(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(cli, "generate_corpus", _refuse_generation)
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "generate",
                "--profile",
                "A",
                *extra,
                "--draws-per-cell",
                "1",
                "--seed",
                "1",
                "--out",
                str(tmp_path / "custom.npz"),
            ]
        )
    assert raised.value.code == 2


@needs_generator
def test_cli_writes_plan_shards_to_a_directory_or_a_file(monkeypatch, tmp_path):
    written = []

    def fake_generate(prior, path):
        written.append((prior, path))
        return {"is_val": make_synthetic_corpus()["is_val"]}

    monkeypatch.setattr(cli, "generate_corpus", fake_generate)
    assert cli.main(["generate", "--profile", "B", "--shard", "319", "--out", str(tmp_path)]) == 0
    evaluation = tmp_path / "evaluation.npz"
    assert cli.main(["generate", "--evaluation-shard", "9", "--out", str(evaluation)]) == 0
    evaluation.write_bytes(b"previous corpus")
    arguments = ["generate", "--evaluation-shard", "9", "--out", str(evaluation), "--force"]
    assert cli.main(arguments) == 0

    (b_prior, b_path), (eval_prior, eval_path), (_, forced_path) = written
    assert b_path == tmp_path / "shard_530719.npz"
    assert (b_prior.seed, b_prior.n_cells, b_prior.draws_per_cell) == (530719, 125, 10)
    assert b_prior.edge_budget["cy"] == (4, 10)
    assert eval_path == forced_path == evaluation
    assert dataclasses.asdict(eval_prior) == {
        **PROFILE_A_PRIOR,
        "n_cells": 32,
        "draws_per_cell": 4,
        "seed": 549000,
        "n_treatments_active_range": (15, 15),
    }


@needs_generator
def test_cli_overrides_reach_the_prior(monkeypatch, tmp_path):
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
    assert captured["path"] == out
    assert (prior.n_cells, prior.draws_per_cell, prior.seed) == (3, 2, 9)
    assert prior.edge_budget == {"cy": (6, 6)}
    assert prior.n_time_steps == 52
