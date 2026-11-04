"""``python -m pymc_mfm.training score``: reproducible results rows, and ``cov95`` keys."""

import csv
import importlib.util
import json

import pytest
import torch
from conftest import make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mfm.data.recipes import load_corpora
from pymc_mfm.model.model import MFM
from pymc_mfm.training import (
    TrackingConfig,
    evaluate_rows,
    flatten_evaluation,
    load_checkpoint,
    train,
)
from pymc_mfm.training.__main__ import main
from pymc_mfm.training.score import (
    ALL_ROWS,
    REPLAY_ROWS,
    RESULT_COLUMNS,
    ScoreError,
    compare_with_record,
    eval_set_id,
    median_step_seconds,
    score_checkpoint,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)


def save(corpus, path):
    from pymc_generator import save_corpus

    save_corpus(corpus, path)
    return path


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory):
    return save(
        make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9),
        tmp_path_factory.mktemp("corpus") / "corpus.npz",
    )


@pytest.fixture(scope="module")
def run_dir(corpus_path, tmp_path_factory):
    """A short run that evaluates at its last step, on the corpus as ``score`` will load it."""
    out = tmp_path_factory.mktemp("run") / "run"
    train(
        [item.corpus for item in load_corpora([corpus_path])],
        out_dir=out,
        model_config=TINY,
        training=reduced(steps=4, eval_every=4, milestone_steps=()),
        device="cpu",
        tracking=TrackingConfig(enabled=False, profile="score-test"),
        run_name="score-test",
    )
    return out


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def score(capsys, *arguments):
    code = main(["score", *map(str, arguments)])
    return code, json.loads(capsys.readouterr().out)


def test_replay_reproduces_every_in_loop_holdout_key(run_dir, corpus_path, tmp_path, capsys):
    checkpoint = run_dir / "checkpoint_last.pt"
    _, payload = load_checkpoint(checkpoint)
    recorded = [key for key in payload["evaluation"]["flat"] if key.startswith("holdout/")]
    results = tmp_path / "results.csv"
    code, summary = score(
        capsys, "--checkpoint", checkpoint, "--corpus", corpus_path, "--replay",
        "--results", results,
    )  # fmt: skip
    (row,) = read_rows(results)
    assert code == 0
    assert tuple(row) == RESULT_COLUMNS
    assert len(recorded) > 0
    assert int(row["replay_keys_compared"]) == len(recorded)
    assert int(row["replay_keys_exact"]) == len(recorded)
    assert float(row["replay_max_abs_diff"]) == 0.0
    assert summary["replay_keys_compared"] == str(len(recorded))
    training = payload["training_config"]
    assert (row["mode"], row["run_name"], int(row["step"])) == ("replay", "score-test", 4)
    assert (int(row["seed"]), int(row["draws"])) == (
        training["eval_seed"] + 1,
        training["eval_samples"],
    )
    assert row["checkpoint_kind"] == "last"
    assert row["selection_monitor"] == "holdout/conditional/query/decomposition_energy_score"
    assert row["step_seconds_median"] != ""
    assert row["cov95"] != "" and float(row["cov95"]) >= float(row["cov80"])
    metrics = json.loads(row["metrics_json"])
    assert "conditional/query/contribution_cov95" in metrics
    assert all(not key.startswith(("holdout/", "evalset/")) for key in metrics)


def test_two_scorings_give_identical_rows(run_dir, corpus_path, tmp_path, capsys):
    checkpoint = run_dir / "checkpoint_last.pt"
    first = score_checkpoint(checkpoint, [corpus_path], mode="replay")
    assert score_checkpoint(checkpoint, [corpus_path], mode="replay") == first
    results = tmp_path / "repeat.csv"
    code, _ = score(
        capsys, "--checkpoint", checkpoint, "--corpus", corpus_path, "--eval-set",
        "--draws", 2, "--repeat", 2, "--results", results,
    )  # fmt: skip
    assert code == 0 and len(read_rows(results)) == 1


def test_changed_corpus_is_refused_before_scoring(run_dir, corpus_path, tmp_path, monkeypatch):
    changed = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9)
    changed["outcome_raw"] = changed["outcome_raw"] + 1.0
    other = save(changed, tmp_path / "changed.npz")
    checkpoint = run_dir / "checkpoint_last.pt"

    def fail(*args, **kwargs):
        raise AssertionError("scored a refused corpus")

    monkeypatch.setattr("pymc_mfm.training.score.evaluate_rows", fail)
    with pytest.raises(ScoreError, match="identity"):
        score_checkpoint(checkpoint, [other], mode="replay")
    results = tmp_path / "results.csv"
    with pytest.raises(SystemExit) as exit_info:
        main(["score", "--checkpoint", str(checkpoint), "--corpus", str(other), "--replay",
              "--results", str(results)])  # fmt: skip
    assert exit_info.value.code == 2
    assert not results.exists()


def test_results_csv_with_another_header_is_refused(run_dir, corpus_path, tmp_path):
    results = tmp_path / "results.csv"
    results.write_text("a,b\n1,2\n")
    arguments = ["--checkpoint", run_dir / "checkpoint_last.pt", "--corpus", corpus_path]
    arguments += ["--eval-set", "--draws", 2, "--results", results]
    with pytest.raises(SystemExit) as exit_info:
        main(["score", *map(str, arguments)])
    assert exit_info.value.code == 2
    assert results.read_text() == "a,b\n1,2\n"


def test_eval_set_row_and_stable_id(run_dir, corpus_path, tmp_path, capsys):
    checkpoint = run_dir / "checkpoint_last.pt"
    results = tmp_path / "results.csv"
    for _ in range(2):
        score(
            capsys, "--checkpoint", checkpoint, "--corpus", corpus_path, "--eval-set",
            "--seed", 7, "--draws", 2, "--results", results,
        )  # fmt: skip
    first, second = read_rows(results)
    assert first == second
    assert (first["mode"], first["row_rule"], first["seed"], first["draws"]) == (
        "eval-set",
        "all-rows",
        "7",
        "2",
    )
    assert first["worlds"] == "16"  # every row of the corpus, not only the held-out ones
    assert first["replay_keys_compared"] == ""
    assert first["eval_set_id"]


def test_eval_set_id_changes_with_corpus_content_seed_and_draws(
    run_dir, corpus_path, tmp_path, capsys
):
    other_corpus = save(
        make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=10), tmp_path / "other.npz"
    )
    results = tmp_path / "results.csv"
    for corpus, seed, draws in (
        (corpus_path, 7, 2),
        (corpus_path, 7, 2),
        (other_corpus, 7, 2),
        (corpus_path, 8, 2),
        (corpus_path, 7, 3),
    ):
        score(
            capsys, "--checkpoint", run_dir / "checkpoint_last.pt", "--corpus", corpus,
            "--eval-set", "--seed", seed, "--draws", draws, "--results", results,
        )  # fmt: skip
    ids = [row["eval_set_id"] for row in read_rows(results)]
    same, again, new_corpus, new_seed, new_draws = ids
    assert same == again
    assert len({same, new_corpus, new_seed, new_draws}) == 4


@pytest.mark.parametrize("rule", [ALL_ROWS, REPLAY_ROWS])
def test_eval_set_id_changes_with_selected_rows(rule):
    first = eval_set_id(["corpus"], rule, 7, 2, [(0, 1), (0, 2)])
    assert first == eval_set_id(["corpus"], rule, 7, 2, [(0, 1), (0, 2)])
    assert first != eval_set_id(["corpus"], rule, 7, 2, [(0, 1), (0, 3)])
    assert first != eval_set_id(["corpus"], rule, 7, 2, [(0, 2), (0, 1)])


def test_torn_final_metrics_line_is_ignored(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"step": 1, "step_seconds": 0.5}\n'
        '{"step": 2, "step_seconds": 1.5}\n'
        '{"step": 3, "step_seconds":'
    )
    assert median_step_seconds(tmp_path / "checkpoint_last.pt", 3) == 1.0
    metrics.write_text('{"step": 1, "step_seconds": 0.5}\n{broken}\n{"step": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        median_step_seconds(tmp_path / "checkpoint_last.pt", 3)


def test_missing_replay_key_differs_from_undefined_value():
    result = compare_with_record({"holdout/key": None}, {})
    assert result["exact"] == 0
    assert result["max_abs_diff"] == float("inf")


def test_main_propagates_unexpected_runtime_error(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr("pymc_mfm.training.score.score_checkpoint", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        main(
            [
                "score",
                "--checkpoint",
                "checkpoint.pt",
                "--corpus",
                "corpus.npz",
                "--eval-set",
                "--results",
                str(tmp_path / "results.csv"),
            ]
        )


def test_replay_rejects_seed_and_draws(run_dir, corpus_path):
    with pytest.raises(ScoreError, match="checkpoint"):
        score_checkpoint(run_dir / "checkpoint_last.pt", [corpus_path], mode="replay", seed=1)


# --- cov95 -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixed_flat():
    """Evaluation of a fixed tiny model: model seed 10, corpus seed 9, draw seed 5."""
    torch.manual_seed(10)
    model = MFM(TINY).eval()
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9)
    rows = [(0, row) for row, held in enumerate(corpus["is_val"]) if held == 1]
    record = evaluate_rows(
        model, [corpus], rows, n_samples=8, seed=5, device="cpu", leg="holdout", per_world=True
    )
    return flatten_evaluation(record)


def test_cov95_covers_at_least_as_much_as_cov80(fixed_flat):
    pairs = [
        (key, key.replace("cov80", "cov95")) for key in fixed_flat if "cov80" in key
        if "simcov80" not in key
    ]  # fmt: skip
    assert len(pairs) > 20
    for narrow, wide in pairs:
        assert wide in fixed_flat, wide
        assert (fixed_flat[narrow] is None) == (fixed_flat[wide] is None), wide
        if fixed_flat[narrow] is not None:
            assert fixed_flat[wide] >= fixed_flat[narrow], wide


def test_cov80_values_equal_those_computed_on_main(fixed_flat):
    # Values of main (36c3f22) for the same fixed evaluation, before cov95 existed.
    main_values = {
        "holdout/conditional/context/contribution_cov80": 0.7222222222222222,
        "holdout/conditional/context/control_contribution_cov80": 0.9722222222222222,
        "holdout/conditional/context/timeavg_contribution_cov80": 0.75,
        "holdout/conditional/query/contribution_cov80": 0.7916666666666666,
        "holdout/conditional/query/control_contribution_cov80": 0.9166666666666667,
        "holdout/joint/context/contribution_cov80": 0.8611111111111112,
        "holdout/joint/context/control_contribution_cov80": 0.9166666666666666,
        "holdout/joint/query/contribution_cov80": 1.0,
    }
    for key, value in main_values.items():
        assert fixed_flat[key] == pytest.approx(value, abs=1e-12), key


def test_replay_matches_generator_corpora_by_file_hash(tmp_path, capsys):
    """Generator corpora carry array-valued metadata, so they have no content SHA-256."""
    import numpy as np

    from pymc_mfm.data.recipes import provenance
    from pymc_mfm.training._provenance import corpus_identity

    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9)
    corpus["identifiability"] = {"signal_metrics": np.zeros((16, 2))}
    path = save(corpus, tmp_path / "generator_like.npz")
    loaded = load_corpora([path])
    assert corpus_identity(loaded[0].corpus)["content_sha256"] is None
    out = tmp_path / "run"
    train(
        [item.corpus for item in loaded],
        out_dir=out,
        model_config=TINY,
        training=reduced(steps=4, eval_every=4, milestone_steps=()),
        device="cpu",
        corpus_provenance=[provenance(item) for item in loaded],
        tracking=TrackingConfig(enabled=False, profile="score-test"),
    )
    results = tmp_path / "results.csv"
    code, summary = score(
        capsys, "--checkpoint", out / "checkpoint_last.pt", "--corpus", path, "--replay",
        "--results", results,
    )  # fmt: skip
    (row,) = read_rows(results)
    assert code == 0
    (key,) = json.loads(row["corpus_keys"])
    assert key.startswith(f"file:{loaded[0].sha256}+entries:")
    assert row["replay_max_abs_diff"] == "0.0"


def test_corpus_key_without_content_hash_still_sees_a_changed_array():
    from pymc_mfm.training._provenance import corpus_identity, corpus_key

    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9)
    corpus["identifiability"] = {"signal_metrics": torch.zeros(2).numpy()}
    changed = dict(corpus, outcome_raw=corpus["outcome_raw"] * 2)
    keys = [
        corpus_key({"sha256": "same-file", "actual_inputs": corpus_identity(item)})
        for item in (corpus, changed)
    ]
    assert corpus_identity(corpus)["content_sha256"] is None
    assert keys[0] is not None and keys[0] != keys[1]
