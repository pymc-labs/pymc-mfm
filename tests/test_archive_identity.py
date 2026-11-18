"""Archive identities derive from the evaluated weights, corpora and request alone."""

import copy
import json
import random
import re

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.inference import from_pretrained, save_pretrained
from pymc_mfm.inference._artifact_schema import SHA256_PATTERN
from pymc_mfm.inference._digest import digest_bytes, json_bytes
from pymc_mfm.training import archive_identity, evaluate_rows
from pymc_mfm.training._provenance import corpus_identity, export_lineage, verified_prior_sha256
from pymc_mfm.training.archive import _corpus_records, _model_identity, _sidecar_seed

#: The bundle formula's hash of ``_bound()``'s prior, pinned at the extraction.
GOLDEN_PRIOR = "5531545293ec5e915c18f60191b5a2c72ed170781b90aa6a5390c3cfb89497d3"
#: The bundle lineage ``config_hash`` of ``{"corpus_0": GOLDEN_PRIOR}``.
GOLDEN_CONFIG = "75cbd4ef1afba275c116346bfd4b1f51d70674b6d71d40a438ed155ed9a2b297"
LINEAGE = {"source_checkpoint_sha256": "c" * 64, "source_step": 7}
BUNDLE_FIELDS = ("checkpoint_sha256", "bundle_sha256", "profile", "source_step")
ORIGIN_FIELDS = ("shard_seed", "prior_sha256", "source_sha256", "data_origin")
#: Shuffled worlds of two corpora mixing both context lengths (6 and 9).
ROWS = [(1, 10), (0, 8), (1, 2), (0, 11), (0, 9), (1, 3), (0, 0), (1, 7)]


def _bound(digest="a" * 64, *, seed=777, bound_to=None):
    """A corpus record whose prior sidecar is bound to its file, or to ``bound_to``."""
    return {
        "data_origin": "synthetic",
        "sha256": digest,
        "prior_sidecar": {
            "prior": {"seed": seed},
            "pymc_generator_version": "test",
            "corpus_sha256": digest if bound_to is None else bound_to,
        },
    }


def _manifest(path):
    return json.loads((path / "manifest.json").read_text())


def _rng_states():
    return random.getstate(), np.random.get_state()[1].tolist(), torch.get_rng_state().tolist()


def test_model_identity_is_the_weight_digest_and_keeps_unchanged_bundle_lineage(model, tmp_path):
    path = save_pretrained(model, tmp_path / "bundle", provenance=LINEAGE, profile="A")
    loaded = from_pretrained(path)
    before = _rng_states()
    fresh = _model_identity(model)
    assert _rng_states() == before
    assert fresh["state_sha256"] == loaded.provenance["state_sha256"]
    assert all(fresh[key] is None for key in BUNDLE_FIELDS)
    # The load-time record names the bundle; its editable provenance does not.
    loaded.provenance = {"checkpoint_sha256": "f" * 64, "profile": "B", "source_step": 99}
    assert _model_identity(loaded) == {
        "state_sha256": fresh["state_sha256"],
        "config_sha256": fresh["config_sha256"],
        "checkpoint_sha256": "c" * 64,
        "bundle_sha256": _manifest(path)["bundle_sha256"],
        "profile": "A",
        "source_step": 7,
    }
    # Without a recorded source checkpoint, the bundle's weights file is its checkpoint.
    plain_path = save_pretrained(model, tmp_path / "plain")
    manifest = _manifest(plain_path)
    plain = _model_identity(from_pretrained(plain_path))
    assert plain["checkpoint_sha256"] == manifest["files"]["model.safetensors"]["sha256"]
    assert plain["bundle_sha256"] == manifest["bundle_sha256"]
    assert plain["profile"] is None and plain["source_step"] is None


def test_changed_weights_do_not_inherit_the_loaded_bundle_identity(model, tmp_path):
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "bundle", provenance=LINEAGE, profile="A")
    )
    unchanged = _model_identity(loaded)
    with torch.no_grad():
        next(loaded.parameters()).add_(1)
    changed = _model_identity(loaded)
    assert changed["state_sha256"] != unchanged["state_sha256"]
    assert all(changed[key] is None for key in BUNDLE_FIELDS)
    # The digest names the weights actually evaluated...
    resaved = from_pretrained(save_pretrained(loaded, tmp_path / "changed", provenance={}))
    assert changed["state_sha256"] == resaved.provenance["state_sha256"]
    # ...and stated lineage describes them, unconstrained by the loaded bundle.
    stated = _model_identity(loaded, checkpoint_sha256="d" * 64, source_step=9, profile="B")
    assert [stated[key] for key in ("checkpoint_sha256", "source_step", "profile")] == [
        "d" * 64,
        9,
        "B",
    ]


def test_explicit_checkpoint_identity_must_agree_with_a_loaded_bundle(model, tmp_path):
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "bundle", provenance=LINEAGE, profile="A")
    )
    recorded = _model_identity(loaded)
    for stated in ({"checkpoint_sha256": "d" * 64}, {"source_step": 8}, {"profile": "B"}):
        with pytest.raises(ValueError, match="conflicts with the loaded bundle's"):
            _model_identity(loaded, **stated)
    agreeing = _model_identity(loaded, checkpoint_sha256="c" * 64, source_step=7, profile="A")
    assert agreeing == recorded
    # Stated values fill what a bundle does not record.
    plain = from_pretrained(save_pretrained(model, tmp_path / "plain"))
    filled = _model_identity(plain, source_step=3, profile="A")
    assert (filled["source_step"], filled["profile"]) == (3, "A")
    with pytest.raises(ValueError, match="conflicts"):
        _model_identity(plain, checkpoint_sha256="d" * 64)
    fresh = _model_identity(model, checkpoint_sha256="e" * 64, source_step=0, profile="A-1")
    assert [fresh[key] for key in BUNDLE_FIELDS] == ["e" * 64, None, "A-1", 0]
    for malformed, message in (
        ({"checkpoint_sha256": "E" * 64}, "checkpoint_sha256 must be a full lowercase SHA-256"),
        ({"checkpoint_sha256": "e" * 63}, "checkpoint_sha256 must be a full lowercase SHA-256"),
        ({"source_step": -1}, "source_step must be a non-negative integer"),
        ({"source_step": True}, "source_step must be a non-negative integer"),
        ({"source_step": 1.0}, "source_step must be a non-negative integer"),
        ({"profile": "two words"}, "profile must be a simple"),
    ):
        with pytest.raises(ValueError, match=message):
            _model_identity(model, **malformed)


def test_worlds_and_calls_follow_the_requested_rows(model):
    corpora = [make_synthetic_corpus(seed=0), make_synthetic_corpus(seed=1)]
    settings = dict(
        n_samples=3, seed=11, leg="holdout", device="cpu", heads=("conditional", "joint")
    )
    before = _rng_states()
    identity = archive_identity(model, corpora, ROWS, **settings)
    assert _rng_states() == before
    assert json.loads(json_bytes(identity)) == identity
    evaluation = identity["evaluation"]
    worlds = evaluation["worlds"]
    assert [(world["corpus_index"], world["row"]) for world in worlds] == ROWS
    for world in worlds:
        corpus = corpora[world["corpus_index"]]
        assert world["cell_id"] == corpus["cell_id"][world["row"]]
        assert world["context_length"] == corpus["support_mask"][world["row"]].sum()
    assert evaluation["calls"] == [{"worlds": list(range(len(ROWS))), "seed": 11}]
    assert evaluation["heads"] == ["joint", "conditional"]
    assert (evaluation["n_samples"], evaluation["leg"]) == (3, "holdout")

    captures = []
    evaluate_rows(
        model,
        corpora,
        ROWS,
        n_samples=evaluation["n_samples"],
        seed=evaluation["calls"][0]["seed"],
        device="cpu",
        leg=evaluation["leg"],
        heads=tuple(evaluation["heads"]),
        result_sink=lambda result, *_: captures.append(result),
    )
    position = {(world["corpus_index"], world["row"]): i for i, world in enumerate(worlds)}
    captured = [
        (position[(result.provenance["corpus_index"], result.provenance["row"])], result.window)
        for result in captures
    ]
    # The declared context lengths give the evaluator's capture order...
    shortest_first = sorted(range(len(worlds)), key=lambda i: worlds[i]["context_length"])
    assert captured == [(i, window) for i in shortest_first for window in ("query", "context")]
    # ...and the declared inputs, settings and model configuration are those of every capture.
    for (i, _), result in zip(captured, captures, strict=True):
        assert result.provenance["data_sha256"] == worlds[i]["data_sha256"]
        assert result.context_length == worlds[i]["context_length"]
        assert (result.mode, result.units) == (evaluation["mode"], evaluation["units"])
        recorded = digest_bytes(json_bytes(result.provenance["inference_config"]["model"]))
        assert recorded == identity["model"]["config_sha256"]

    chunked = archive_identity(model, corpora, ROWS, worlds_per_call=3, **settings)
    assert chunked["evaluation"]["calls"] == [
        {"worlds": [0, 1, 2], "seed": 11},
        {"worlds": [3, 4, 5], "seed": 12},
        {"worlds": [6, 7], "seed": 13},
    ]
    assert {**chunked["evaluation"], "calls": evaluation["calls"]} == evaluation
    one_shot = zip([index for index, _ in ROWS], [row for _, row in ROWS], strict=True)
    assert archive_identity(model, corpora, one_shot, **settings) == identity


def test_shard_seeds_priors_and_origin_follow_verified_sidecars_and_the_pinned_bundle_formula():
    record = _bound()
    assert verified_prior_sha256(record) == GOLDEN_PRIOR
    lineage = export_lineage({"corpus": [record], "diagnostics": []}, "0" * 64)["prior"]
    assert lineage["config"]["corpus_prior_hashes"]["corpus_0"] == GOLDEN_PRIOR
    assert lineage["config_hash"] == GOLDEN_CONFIG
    assert lineage["status"] == "recorded_hash_references"
    run = {"corpus": [record, {"data_origin": "unknown"}], "diagnostics": []}
    partial = export_lineage(run, "0" * 64)["prior"]
    assert partial["config"]["corpus_prior_hashes"] == {"corpus_0": GOLDEN_PRIOR}
    assert partial["config_hash"] == GOLDEN_CONFIG
    assert partial["status"] == "partial_hash_references"

    corpus, other = make_synthetic_corpus(seed=0), make_synthetic_corpus(seed=1)

    def recorded(claim):
        (entry,) = _corpus_records([corpus], [claim])
        return tuple(entry[key] for key in ORIGIN_FIELDS)

    assert recorded(record) == (777, GOLDEN_PRIOR, "a" * 64, "synthetic")
    # A bare claim, or a sidecar of another file, is not a verified synthetic origin.
    for claim in ({"data_origin": "synthetic", "sha256": "a" * 64}, _bound(bound_to="b" * 64)):
        assert recorded(claim) == (None, None, "a" * 64, "unknown")
        assert _sidecar_seed(claim) is None
    (anonymous,) = _corpus_records([corpus])
    assert tuple(anonymous[key] for key in ORIGIN_FIELDS) == (None, None, None, "unknown")
    # Only a non-negative int seed of a verified sidecar names the shard.
    for seed in (-1, True, 777.0, "777"):
        assert verified_prior_sha256(_bound(seed=seed)) is not None
        assert _sidecar_seed(_bound(seed=seed)) is None
    provenance = [record, _bound("b" * 64, seed=778)]
    assert [entry["shard_seed"] for entry in _corpus_records([corpus, other], provenance)] == [
        777,
        778,
    ]
    explicit = _corpus_records([corpus, other], provenance, shard_seeds=(7, None))
    assert [entry["shard_seed"] for entry in explicit] == [7, None]
    assert [entry["data_origin"] for entry in explicit] == ["synthetic", "synthetic"]


def test_actual_inputs_identity_exists_when_content_sha256_is_unavailable():
    corpus = make_synthetic_corpus()
    corpus["identifiability"] = {"channel": np.ones(3)}
    corpus["treatment_shock_index"] = np.zeros((corpus["is_val"].shape[0], 0), dtype=np.int64)
    (entry,) = _corpus_records([corpus])
    assert entry["content_sha256"] is None
    assert SHA256_PATTERN.fullmatch(entry["actual_inputs_sha256"])
    assert _corpus_records([copy.deepcopy(corpus)]) == [entry]
    # The identities the trainer has already computed give the same record.
    assert _corpus_records([corpus], identities=[corpus_identity(corpus)]) == [entry]
    changed = {**corpus, "outcome_raw": corpus["outcome_raw"].copy()}
    changed["outcome_raw"][0, 0] += 1
    (other,) = _corpus_records([changed])
    assert other["actual_inputs_sha256"] != entry["actual_inputs_sha256"]
    # Without content digests, the two corpora are still told apart.
    both = _corpus_records([corpus, changed])
    assert [item["actual_inputs_sha256"] for item in both] == [
        entry["actual_inputs_sha256"],
        other["actual_inputs_sha256"],
    ]


def test_identity_refuses_equal_corpora_and_accepts_the_seed_range_ends(model):
    corpus, other = make_synthetic_corpus(seed=0), make_synthetic_corpus(seed=1)

    def identity(corpora=(corpus,), rows=((0, 8),), **request):
        settings = dict(n_samples=2, seed=0, leg="holdout", device="cpu")
        return archive_identity(model, list(corpora), list(rows), **{**settings, **request})

    sidecar_twins = (_bound("a" * 64, seed=5), _bound("b" * 64, seed=5))
    three = ((0, 8), (0, 9), (0, 10))
    # Refusals that need the corpus digests; test_evaluate_into_archive.py has those that
    # come before any work.
    for request, message in (
        (dict(corpora=(corpus, corpus)), "corpora 0 and 1 hold the same inputs"),
        (dict(corpora=(corpus, other), shard_seeds=(5, 5)), "corpora 0 and 1 share shard seed 5"),
        (dict(corpora=(corpus, other), corpus_provenance=sidecar_twins), "share shard seed 5"),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            identity(**request)
    assert identity(shard_seeds=(2**63,))["corpora"][0]["shard_seed"] == 2**63
    assert identity(seed=2**64 - 1)["evaluation"]["calls"] == [{"worlds": [0], "seed": 2**64 - 1}]
    lowest = identity(rows=three, worlds_per_call=2, seed=-(2**63))["evaluation"]["calls"]
    assert lowest == [{"worlds": [0, 1], "seed": -(2**63)}, {"worlds": [2], "seed": 1 - 2**63}]
