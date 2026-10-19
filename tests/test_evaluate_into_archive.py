"""Evaluating into an archive runs exactly the evaluation its identity declares."""

import json
import re
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus
from test_archive import TINY, assert_same_result, canonical
from test_archive_identity import _bound

from pymc_mfm.diagnostics import EvaluationArchive
from pymc_mfm.inference import from_pretrained, save_pretrained
from pymc_mfm.inference import result as result_module
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM
from pymc_mfm.training import archive as training_archive
from pymc_mfm.training import archive_identity, evaluate_into_archive, evaluate_rows
from pymc_mfm.training.archive import _corpus_records, _world_records
from pymc_mfm.training.evaluation import DECOMPOSITION_HEADS

#: Seven worlds of both corpora and both context lengths ((0, 8), (0, 9) and (0, 0) have
#: context 6, the others 9); calls of three worlds are [0, 1, 2], [3, 4, 5] and a ragged [6].
ROWS7 = [(1, 10), (0, 8), (1, 2), (0, 11), (0, 9), (1, 3), (0, 0)]
SEED = 31
ONE_CALL = [{"worlds": [0, 1, 2, 3, 4, 5, 6], "seed": SEED}]
THREE_CALLS = [
    {"worlds": [0, 1, 2], "seed": SEED},
    {"worlds": [3, 4, 5], "seed": SEED + 1},
    {"worlds": [6], "seed": SEED + 2},
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A loaded tiny model bundle and the ``bundle_sha256`` of its manifest."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        path = save_pretrained(
            MFM(ModelConfig(**TINY)).eval(), tmp_path_factory.mktemp("bundle") / "model"
        )
        model = from_pretrained(path)
    return model, json.loads((path / "manifest.json").read_text())["bundle_sha256"]


@pytest.fixture(scope="module")
def corpora():
    return [make_synthetic_corpus(seed=0), make_synthetic_corpus(seed=1)]


def run_calls(model, corpora, calls, **settings):
    """``evaluate_rows`` of each ``(rows, seed)`` of ``calls``: the sink tuples and the records."""
    captured = []
    records = [
        evaluate_rows(
            model,
            corpora,
            rows,
            seed=seed,
            device="cpu",
            result_sink=lambda *item: captured.append(item),
            **settings,
        )
        for rows, seed in calls
    ]
    return captured, records


def assert_archived(archive, captured):
    """The archive's records are field-equal to ``captured``, in archive order."""
    for (result, *evidence), (expected, *wanted) in zip(archive, captured, strict=True):
        assert_same_result(result, expected)
        for actual, value in zip(evidence, wanted, strict=True):
            np.testing.assert_array_equal(actual, value)
            assert actual.dtype == value.dtype


@pytest.mark.parametrize(
    ("heads", "worlds_per_call", "calls", "one_shot"),
    [
        pytest.param(DECOMPOSITION_HEADS, None, ONE_CALL, False, id="both-heads-one-call"),
        pytest.param(DECOMPOSITION_HEADS, 3, THREE_CALLS, False, id="both-heads-three-calls"),
        pytest.param(("conditional",), None, ONE_CALL, False, id="conditional-one-call"),
        pytest.param(("conditional",), 3, THREE_CALLS, True, id="conditional-one-shot-rows"),
    ],
)
def test_an_archive_replays_from_its_own_identity_and_records_the_request(
    bundle, corpora, tmp_path, heads, worlds_per_call, calls, one_shot
):
    model, bundle_sha256 = bundle
    path, records = evaluate_into_archive(
        model,
        corpora,
        iter(ROWS7) if one_shot else ROWS7,
        tmp_path / "archive",
        n_samples=4,
        seed=SEED,
        leg="evaluation",
        heads=heads,
        worlds_per_call=worlds_per_call,
    )
    archive = EvaluationArchive.load(path)
    identity = archive.identity
    evaluation = identity["evaluation"]
    worlds = evaluation["worlds"]
    assert [(world["corpus_index"], world["row"]) for world in worlds] == ROWS7
    assert evaluation["calls"] == calls
    assert evaluation["heads"] == [head for head in ("joint", "conditional") if head in heads]
    assert (evaluation["n_samples"], evaluation["leg"]) == (4, "evaluation")
    assert identity["model"]["bundle_sha256"] == bundle_sha256
    # The loaded identity alone replays the archived draws and the returned records.
    captured, replayed = run_calls(
        model,
        corpora,
        [
            ([(worlds[w]["corpus_index"], worlds[w]["row"]) for w in call["worlds"]], call["seed"])
            for call in evaluation["calls"]
        ],
        n_samples=evaluation["n_samples"],
        heads=tuple(evaluation["heads"]),
        leg=evaluation["leg"],
    )
    assert_archived(archive, captured)
    assert canonical(records) == canonical(replayed)


def test_request_settings_reach_evaluate_rows(bundle, corpora, tmp_path):
    model, _ = bundle
    settings = dict(n_samples=3, heads=("conditional",), leg="holdout")
    provenance = [_bound("a" * 64, seed=777), _bound("b" * 64, seed=778)]
    path, _ = evaluate_into_archive(
        model,
        corpora,
        ROWS7,
        tmp_path / "archive",
        seed=SEED,
        worlds_per_call=3,
        corpus_provenance=provenance,
        **settings,
    )
    # The calls named by the request alone, without reading the archive.
    captured, _ = run_calls(
        model,
        corpora,
        [(ROWS7[i : i + 3], SEED + j) for j, i in ((0, 0), (1, 3), (2, 6))],
        **settings,
    )
    archive = EvaluationArchive.load(path)
    assert_archived(archive, captured)
    # Verified sidecars name the shards and the priors, and mark the archive synthetic.
    assert set(archive.prior["corpus_prior_hashes"]) == {"shard:777", "shard:778"}
    assert all(
        entry.world_id == f"shard:{777 + entry.corpus_index}/row:{entry.row}"
        for entry in archive.entries
    )
    assert archive.synthetic is True


def test_capture_failures_abort_the_archive(bundle, corpora, tmp_path, monkeypatch):
    model, _ = bundle
    convert = result_module.result_from_batch

    def refuse_one_window(*args, **kwargs):
        provenance = kwargs["provenance"]
        world = provenance["corpus_index"], provenance["row"]
        if world == (0, 9) and kwargs["window"] == "context":
            raise ValueError("refused by the test")
        return convert(*args, **kwargs)

    monkeypatch.setattr(result_module, "result_from_batch", refuse_one_window)
    # (0, 9) is declared in the second call, after the first call was archived.
    failure = "first: corpus 0 row 9 context: ValueError: refused by the test"
    with pytest.raises(ValueError, match=re.escape(failure)):
        evaluate_into_archive(
            model,
            corpora,
            ROWS7,
            tmp_path / "archive",
            n_samples=2,
            seed=SEED,
            leg="evaluation",
            heads=("conditional",),
            worlds_per_call=3,
        )
    # Neither the target nor a staging directory remains.
    assert list(tmp_path.iterdir()) == []


def test_an_existing_target_is_refused_before_any_inference(bundle, corpora, tmp_path, monkeypatch):
    model, _ = bundle
    target = tmp_path / "taken"
    target.mkdir()
    (target / "kept.txt").write_text("not an archive")

    def never(*args, **kwargs):
        raise AssertionError("evaluate_rows ran before the target was checked")

    monkeypatch.setattr(training_archive, "evaluate_rows", never)
    with pytest.raises(FileExistsError, match="new or empty"):
        evaluate_into_archive(
            model, corpora, ROWS7, target, n_samples=2, seed=SEED, leg="evaluation"
        )
    assert [path.name for path in tmp_path.iterdir()] == ["taken"]
    assert [path.name for path in target.iterdir()] == ["kept.txt"]


class Reached(Exception):
    """Work that a cheap refusal must come before was reached."""


@pytest.fixture
def no_work(monkeypatch):
    """Hashing the weights or a corpus, or encoding a world, raises :class:`Reached`."""

    def reached(*args, **kwargs):
        raise Reached

    for name in ("_state_sha256", "corpus_identity", "batches_for_rows"):
        monkeypatch.setattr(training_archive, name, reached)


def test_limits_are_checked_before_any_work(bundle, corpora, no_work):
    model, _ = bundle
    corpus, large = corpora[0], make_synthetic_corpus(n_cells=2048)  # 4096 worlds
    rows = [(0, row) for row in range(4097)]
    request = dict(n_samples=2, seed=0, leg="evaluation", device="cpu")
    worlds = "an evaluation archive holds at most 4096 worlds; 4097 were selected"
    declared = "an evaluation archive declares at most 4096 corpora; 4097 were given"
    with pytest.raises(ValueError, match=re.escape(worlds)):
        archive_identity(model, [large], rows, **request)
    with pytest.raises(ValueError, match=re.escape(declared)):
        archive_identity(model, [corpus] * 4097, [(0, 8)], **request)
    with pytest.raises(ValueError, match=re.escape(worlds)):
        _world_records([large], rows)
    with pytest.raises(ValueError, match=re.escape(declared)):
        _corpus_records([corpus] * 4097)
    # The limits are inclusive: 4096 worlds or corpora go on to the work.
    with pytest.raises(Reached):
        archive_identity(model, [large], rows[:4096], **request)
    with pytest.raises(Reached):
        archive_identity(model, [corpus] * 4096, [(0, 8)], **request)
    with pytest.raises(Reached):
        _world_records([large], rows[:4096])
    with pytest.raises(Reached):
        _corpus_records([corpus] * 4096)


def _without_cell_ids(corpus):
    return [{key: value for key, value in corpus.items() if key != "cell_id"}]


#: Changes that make the valid three-world request of the test below malformed, each with a
#: fragment of its refusal; ``corpora`` maps the valid corpus to the corpora requested.
MALFORMED = [
    pytest.param({"rows": []}, "archive rows must name at least one world", id="no-rows"),
    pytest.param(
        {"rows": [(0, 8), (0, 9), (0, 8)]},
        "archive rows must be distinct; (0, 8) repeats",
        id="repeated-row",
    ),
    pytest.param(
        {"rows": [(0, 8), (0, 12)]},
        "world index 12 is outside [0, 12) of corpus 0",
        id="row-past-the-end",
    ),
    pytest.param({"rows": [(0, -1)]}, "world index -1 is outside [0, 12) of corpus 0", id="row-1"),
    pytest.param({"rows": [(1, 0)]}, "corpus index 1 is outside [0, 1)", id="corpus-past-the-end"),
    pytest.param({"rows": [(-1, 0)]}, "corpus index -1 is outside [0, 1)", id="corpus-1"),
    pytest.param({"corpora": _without_cell_ids}, "corpus 0 has no cell_id array", id="no-cells"),
    pytest.param(
        {"corpus_provenance": ({}, {})},
        "corpus_provenance needs one record per corpus",
        id="provenance-of-two-corpora",
    ),
    pytest.param(
        {"shard_seeds": (1, 2)}, "shard_seeds needs one seed per corpus", id="seeds-of-two-corpora"
    ),
    pytest.param(
        {"shard_seeds": (-1,)},
        "shard seed must be a non-negative integer or None, got -1",
        id="negative-shard-seed",
    ),
    pytest.param(
        {"shard_seeds": (True,)},
        "shard seed must be a non-negative integer or None, got True",
        id="bool-shard-seed",
    ),
    pytest.param(
        {"shard_seeds": (1.5,)},
        "shard seed must be a non-negative integer or None, got 1.5",
        id="fractional-shard-seed",
    ),
    pytest.param(
        {"checkpoint_sha256": "E" * 64},
        "checkpoint_sha256 must be a full lowercase SHA-256",
        id="uppercase-checkpoint",
    ),
    pytest.param({"source_step": -1}, "source_step must be a non-negative integer", id="step-1"),
    pytest.param({"profile": "two words"}, "profile must be a simple", id="spaced-profile"),
    pytest.param({"heads": ("joint",)}, "archives capture conditional draws", id="joint-head"),
    pytest.param({"n_samples": 0}, "n_samples must be a positive integer", id="no-draws"),
    pytest.param({"seed": 1.0}, "seed must be an int", id="float-seed"),
    pytest.param(
        {"seed": 2**64},
        "evaluation seed 18446744073709551616 of call 0 is outside",
        id="seed-above-range",
    ),
    pytest.param(
        {"seed": -(2**63) - 1},
        "of call 0 is outside the torch seed range [-2**63, 2**64)",
        id="seed-below-range",
    ),
    # Three calls of one world each: only the last call's seed leaves the range.
    pytest.param(
        {"worlds_per_call": 1, "seed": 2**64 - 2},
        "evaluation seed 18446744073709551616 of call 2",
        id="seed-of-a-later-call",
    ),
    pytest.param(
        {"worlds_per_call": 0},
        "worlds_per_call must be a positive integer or None",
        id="no-worlds-per-call",
    ),
    pytest.param(
        {"worlds_per_call": True},
        "worlds_per_call must be a positive integer or None",
        id="bool-worlds-per-call",
    ),
    pytest.param({"leg": ""}, "leg must be a nonempty string", id="empty-leg"),
    # What Python makes of a byte that is not UTF-8 in a command-line argument.
    pytest.param(
        {"leg": "holdout\udcff"},
        "leg is not valid Unicode text: surrogates not allowed at character 7",
        id="surrogate-leg",
    ),
    pytest.param({"device": "gpu"}, "device 'gpu' is not a torch device", id="unknown-device"),
    pytest.param({"device": "cuda0"}, "device 'cuda0' is not a torch device", id="device-typo"),
    pytest.param({"device": None}, "device None is not a torch device", id="no-device"),
]


@pytest.mark.parametrize(("change", "fragment"), MALFORMED)
def test_malformed_requests_are_refused_before_any_work(bundle, corpora, no_work, change, fragment):
    model, _ = bundle
    request = {
        "corpora": lambda corpus: [corpus],
        "rows": [(0, 8), (0, 9), (0, 10)],
        "n_samples": 2,
        "seed": 0,
        "leg": "evaluation",
        "device": "cpu",
        **change,
    }
    selected, rows = request.pop("corpora")(corpora[0]), request.pop("rows")
    with pytest.raises(ValueError, match=re.escape(fragment)):
        archive_identity(model, selected, rows, **request)


def test_importing_training_loads_no_diagnostics_code():
    code = (
        "import sys\n"
        "import pymc_mfm.training\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('pymc_mfm.diagnostics'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
