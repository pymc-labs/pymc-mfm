"""``python -m pymc_mktfm.training evaluate`` and ``--holdout-archives``: archives from the CLI."""

import importlib.util
import json

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mktfm.data.recipes import load_corpora, prior_sidecar_path, provenance
from pymc_mktfm.diagnostics import EvaluationArchive
from pymc_mktfm.inference import save_pretrained
from pymc_mktfm.model.model import MFM
from pymc_mktfm.training import TrackingConfig, TrainingConfig, load_checkpoint, train
from pymc_mktfm.training import archive as training_archive
from pymc_mktfm.training.__main__ import main
from pymc_mktfm.training._provenance import verified_prior_sha256
from pymc_mktfm.training.archive import _model_identity
from pymc_mktfm.training.checkpoint import file_sha256

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)

#: Generator seed recorded by the shard's verified prior sidecar.
SHARD_SEED = 777
#: Generator seed recorded by the second shard's verified prior sidecar.
SECOND_SEED = 778
#: The default ``--seed``: the trainer's held-out draw seed.
SEED = TrainingConfig().eval_seed + 1
#: The flat invocation of ``test_training``'s CLI test, after ``--corpus``/``--out-dir``.
FLAT = [
    "--no-tracking",
    "--steps",
    "2",
    "--batch-size",
    "2",
    "--warmup-steps",
    "1",
    "--eval-every",
    "2",
    "--eval-worlds",
    "2",
    "--eval-samples",
    "2",
    "--log-every",
    "1",
    "--device",
    "cpu",
    "--run-name",
    "cli",
]


def shard_corpus():
    """The shard's worlds: 8 cells of 2, the last two cells held out (``is_val == 1``)."""
    return make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9)


_IS_VAL = shard_corpus()["is_val"]
#: The shard's world count and held-out rows.
N_WORLDS = len(_IS_VAL)
HELD_OUT = np.flatnonzero(_IS_VAL == 1).tolist()


def second_corpus():
    """Another shard's worlds: other contents, and 6 cells of 2, so other held-out rows."""
    return make_synthetic_corpus(n_cells=6, draws_per_cell=2, seed=4)


def save(corpus, path, *, shard_seed=None):
    """Save ``corpus`` at ``path``, with a prior sidecar bound to the file if ``shard_seed``."""
    from pymc_generator import save_corpus

    save_corpus(corpus, path)
    if shard_seed is not None:
        sidecar = {
            "prior": {"seed": shard_seed},
            "pymc_generator_version": "test",
            "data_origin": "synthetic",
            "corpus_sha256": file_sha256(path),
        }
        prior_sidecar_path(path).write_text(json.dumps(sidecar))
    return path


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    """The saved shard, with a prior sidecar of generator seed ``SHARD_SEED`` bound to it."""
    return save(
        shard_corpus(), tmp_path_factory.mktemp("shard") / "corpus.npz", shard_seed=SHARD_SEED
    )


@pytest.fixture(scope="module")
def second_shard(tmp_path_factory):
    """The saved second shard, with a prior sidecar of generator seed ``SECOND_SEED``."""
    return save(
        second_corpus(), tmp_path_factory.mktemp("second") / "corpus.npz", shard_seed=SECOND_SEED
    )


@pytest.fixture(scope="module")
def no_holdout(tmp_path_factory):
    """A saved corpus without held-out rows."""
    return save(make_synthetic_corpus(val_cells=0), tmp_path_factory.mktemp("train") / "c.npz")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        return save_pretrained(MFM(TINY).eval(), tmp_path_factory.mktemp("bundle") / "model")


def evaluate(capsys, *arguments):
    """``evaluate`` with two draws per world: its exit code and its printed summary."""
    code = main(["evaluate", "--n-samples", "2", *map(str, arguments)])
    return code, json.loads(capsys.readouterr().out)


def rows(evaluation):
    return [(world["corpus_index"], world["row"]) for world in evaluation["worlds"]]


def calls(n_worlds, size):
    """Declared calls of ``n_worlds`` worlds: blocks of ``size`` with seeds ``SEED + k``."""
    return [
        {"worlds": list(range(start, min(start + size, n_worlds))), "seed": SEED + call}
        for call, start in enumerate(range(0, n_worlds, size))
    ]


def test_cli_evaluate_archives_a_bundle_on_held_out_rows(shard, bundle, tmp_path, capsys):
    code, summary = evaluate(
        capsys, "--model", bundle, "--corpus", shard, "--out-dir", tmp_path / "holdout"
    )
    assert code == 0
    archive = EvaluationArchive.load(tmp_path / "holdout")
    identity = archive.identity
    evaluation = identity["evaluation"]
    plan = calls(len(HELD_OUT), 4)
    assert summary == {
        "archive": str(archive.path),
        "archive_sha256": archive.archive_sha256,
        "worlds": len(HELD_OUT),
        "entries": 2 * len(HELD_OUT),
        "calls": len(plan),
        "leg": "holdout",
        "n_samples": 2,
        "seeds": [call["seed"] for call in plan],
        "heads": ["joint", "conditional"],
        "state_sha256": identity["model"]["state_sha256"],
    }
    # Every held-out row, ascending; four worlds per call, seeds up from the trainer's draw seed.
    assert rows(evaluation) == [(0, row) for row in HELD_OUT]
    assert evaluation["calls"] == plan
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert identity["model"]["bundle_sha256"] == manifest["bundle_sha256"]
    # The verified sidecar names the shard and its prior.
    assert identity["corpora"][0]["shard_seed"] == SHARD_SEED
    assert {entry.world_id for entry in archive.entries} == {
        f"shard:{SHARD_SEED}/row:{row}" for row in HELD_OUT
    }
    (loaded,) = load_corpora([shard])
    assert archive.prior == {
        "name": "verified_generation_priors",
        "corpus_prior_hashes": {f"shard:{SHARD_SEED}": verified_prior_sha256(provenance(loaded))},
    }
    assert archive.synthetic is True

    code, _ = evaluate(
        capsys,
        "--model",
        bundle,
        "--corpus",
        shard,
        "--out-dir",
        tmp_path / "all",
        "--split",
        "all",
    )
    evaluation = EvaluationArchive.load(tmp_path / "all").identity["evaluation"]
    assert (code, evaluation["leg"]) == (0, "all")
    assert rows(evaluation) == [(0, row) for row in range(N_WORLDS)]
    # Four worlds per call by default, with consecutive seeds.
    assert evaluation["calls"] == calls(N_WORLDS, 4)


def test_cli_evaluate_archives_selected_worlds_of_a_training_checkpoint(shard, tmp_path, capsys):
    train(
        [shard_corpus()],
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(),
        device="cpu",
        tracking=TrackingConfig(enabled=False, profile="cli-probe"),
    )
    checkpoint = tmp_path / "run" / "checkpoint_last.pt"
    model, payload = load_checkpoint(checkpoint)
    request = ["--model", checkpoint, "--corpus", shard, "--heads", "conditional"]
    # A held-out row, then a training row (the held-out cells are the last): the given order.
    held_out = HELD_OUT[0]
    training = next(row for row in range(N_WORLDS) if row not in HELD_OUT)
    code, _ = evaluate(
        capsys,
        *request,
        "--out-dir",
        tmp_path / "custom",
        "--world",
        f"{SHARD_SEED}:{held_out}",
        "--world",
        f"{SHARD_SEED}:{training}",
        "--worlds-per-call",
        "1",
        "--leg",
        "custom",
    )
    assert code == 0
    identity = EvaluationArchive.load(tmp_path / "custom").identity
    evaluation = identity["evaluation"]
    assert rows(evaluation) == [(0, held_out), (0, training)]
    assert (evaluation["leg"], evaluation["heads"]) == ("custom", ["conditional"])
    assert evaluation["calls"] == [{"worlds": [0], "seed": SEED}, {"worlds": [1], "seed": SEED + 1}]
    # The checkpoint's weights, named by its file digest, its step and its run's profile.
    assert identity["model"] == {
        **_model_identity(model),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_step": payload["step"],
        "profile": "cli-probe",
    }
    # Without --leg, selected worlds are archived as the "selected" leg.
    code, _ = evaluate(
        capsys, *request, "--out-dir", tmp_path / "selected", "--world", f"{SHARD_SEED}:{training}"
    )
    leg = EvaluationArchive.load(tmp_path / "selected").identity["evaluation"]["leg"]
    assert (code, leg) == (0, "selected")


def test_cli_evaluate_maps_world_seeds_to_their_corpora(
    shard, second_shard, bundle, tmp_path, capsys
):
    both = ["--model", bundle, "--corpus", shard, second_shard]
    # The second corpus's world first: each seed names its corpus, and the given order is kept.
    code, _ = evaluate(
        capsys,
        *both,
        "--out-dir",
        tmp_path / "selected",
        "--world",
        f"{SECOND_SEED}:3",
        "--world",
        f"{SHARD_SEED}:5",
    )
    assert code == 0
    archive = EvaluationArchive.load(tmp_path / "selected")
    identity = archive.identity
    assert rows(identity["evaluation"]) == [(1, 3), (0, 5)]
    assert {entry.world: entry.world_id for entry in archive.entries} == {
        0: f"shard:{SECOND_SEED}/row:3",
        1: f"shard:{SHARD_SEED}/row:5",
    }
    # Each corpus record names its own file and seed, in --corpus order.
    assert [(corpus["shard_seed"], corpus["source_sha256"]) for corpus in identity["corpora"]] == [
        (SHARD_SEED, file_sha256(shard)),
        (SECOND_SEED, file_sha256(second_shard)),
    ]
    # The default split takes the held-out rows of every corpus, in corpus order.
    code, _ = evaluate(capsys, *both, "--out-dir", tmp_path / "holdout")
    assert code == 0
    evaluation = EvaluationArchive.load(tmp_path / "holdout").identity["evaluation"]
    second_held_out = np.flatnonzero(second_corpus()["is_val"] == 1).tolist()
    assert rows(evaluation) == [(0, row) for row in HELD_OUT] + [
        (1, row) for row in second_held_out
    ]


#: Per case: the ``--corpus`` files, the other arguments and a fragment of the usage error.
USAGE_ERRORS = {
    "unknown-seed": ("shard", ["--world", "1:3"], "no --corpus has verified generator seed 1"),
    "malformed-world": ("shard", ["--world", "nope"], "argument --world: expected SEED:ROW"),
    "split-and-world": (
        "shard",
        ["--split", "all", "--world", f"{SHARD_SEED}:3"],
        "not allowed with argument --split",
    ),
    "no-conditional-head": ("shard", ["--heads", "joint"], "select the conditional head"),
    "no-draws": ("shard", ["--n-samples", "0"], "n_samples must be a positive integer"),
    "no-worlds-per-call": (
        "shard",
        ["--worlds-per-call", "0"],
        "worlds_per_call must be a positive integer",
    ),
    # The first row past the end.
    "row-out-of-range": (
        "shard",
        ["--world", f"{SHARD_SEED}:{N_WORLDS}"],
        f"seed {SHARD_SEED} has {N_WORLDS} worlds",
    ),
    "repeated-world": (
        "shard",
        ["--world", f"{SHARD_SEED}:3", "--world", f"{SHARD_SEED}:3"],
        f"{SHARD_SEED}:3 repeats",
    ),
    "seed-out-of-range": (
        "shard",
        ["--seed", "18446744073709551616"],
        "outside the torch seed range",
    ),
    "no-held-out-rows": ("no_holdout", [], "--split holdout selects no worlds"),
    "same-corpus-twice": ("twice", [], "hold the same inputs"),
    # argparse keeps the last --model.
    "missing-model": (
        "shard",
        ["--model", "/nonexistent/model"],
        "neither a model bundle directory nor a training checkpoint file",
    ),
    "unknown-device": ("shard", ["--device", "tpu"], "--device tpu"),
}


@pytest.mark.parametrize("case", USAGE_ERRORS)
def test_cli_evaluate_usage_errors_exit_with_status_2(
    shard, no_holdout, bundle, tmp_path, monkeypatch, capsys, case
):
    corpora, arguments, fragment = USAGE_ERRORS[case]
    paths = {"shard": [shard], "twice": [shard, shard], "no_holdout": [no_holdout]}[corpora]

    def inference(*args, **kwargs):
        raise AssertionError("usage errors must stop the CLI before any inference")

    monkeypatch.setattr(training_archive, "evaluate_rows", inference)
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "evaluate",
                "--model",
                str(bundle),
                "--corpus",
                *map(str, paths),
                "--out-dir",
                str(tmp_path / "archive"),
                *arguments,
            ]
        )
    assert stopped.value.code == 2
    assert fragment in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_training_writes_holdout_archives_when_asked(shard, tmp_path, capsys):
    run_dir = tmp_path / "run"
    flat = ["--corpus", str(shard), "--out-dir", str(run_dir), *FLAT, "--holdout-archives"]
    for refused in ("worst", "best,best"):
        with pytest.raises(SystemExit) as stopped:
            main([*flat, refused])
        assert stopped.value.code == 2
        assert "argument --holdout-archives" in capsys.readouterr().err
        assert not run_dir.exists()
    assert main([*flat, "terminal"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "finished"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["evaluation_policy"]["holdout_archives"]["checkpoints"] == ["terminal"]
    archive = EvaluationArchive.load(run_dir / "archives" / "step_000000002")
    assert archive.identity["evaluation"]["leg"] == "holdout"
    (entry,) = [
        item for item in run["diagnostics"] if item["recipe"] == "holdout_evaluation_archive"
    ]
    assert (entry["status"], entry["archive_sha256"]) == ("recorded", archive.archive_sha256)
