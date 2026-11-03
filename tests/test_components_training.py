"""Research components in training: checkpoint, resume, score, and refusal to export or load."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import research_plugin
import test_resume
import torch
from conftest import make_synthetic_corpus
from research_plugin import SAME_SHAPE, SMALL_FLOW
from test_resume import (
    NO_TRACKING,
    assert_resumed_equals_full,
    interrupted_run,
    run_train,
)
from test_training import TINY, reduced

from pymc_mfm.data.recipes import load_corpora
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.model.components import ResearchComponentError
from pymc_mfm.model.model import MFM
from pymc_mfm.training import load_checkpoint, train
from pymc_mfm.training.__main__ import main
from pymc_mfm.training.checkpoint import save_checkpoint

needs_generator = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)


def payload_of(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory):
    from pymc_generator import save_corpus

    path = tmp_path_factory.mktemp("corpus") / "corpus.npz"
    save_corpus(make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9), path)
    return path


@pytest.fixture(scope="module")
def research_run(corpus_path, tmp_path_factory):
    """50 updates of the small-flow research model with one evaluation, at step 50."""
    out = tmp_path_factory.mktemp("research") / "run"
    train(
        [item.corpus for item in load_corpora([corpus_path])],
        out_dir=out,
        model_config=TINY,
        components=SMALL_FLOW,
        training=reduced(steps=50, eval_every=50, milestone_steps=(), log_every=10),
        device="cpu",
        tracking=NO_TRACKING,
    )
    return out


@needs_generator
def test_research_run_finishes_without_an_exported_bundle(research_run):
    run = json.loads((research_run / "run.json").read_text())
    last = payload_of(research_run / "checkpoint_last.pt")
    assert run["status"] == "finished" and run["final_step"] == 50
    assert run["export"] == {
        "status": "skipped",
        "reason": "research components",
        "components": {"encoder": "bi_attention", **SMALL_FLOW},
        "source_checkpoint_sha256": last_sha(research_run),
        "source_step": 50,
    }
    assert not (research_run / "model").exists()
    assert last["schemas"]["components"] == run["export"]["components"]


def last_sha(run_dir):
    return next(c["sha256"] for c in json.loads((run_dir / "run.json").read_text())["checkpoints"]
                if c["kind"] == "last")  # fmt: skip


@needs_generator
def test_research_checkpoint_loads_only_with_opt_in_and_never_exports(research_run, tmp_path):
    checkpoint = research_run / "checkpoint_last.pt"
    with pytest.raises(ResearchComponentError, match="allow_research"):
        load_checkpoint(checkpoint)
    model, _ = load_checkpoint(checkpoint, allow_research=True)
    assert model.is_research and model.components["head"] == "p6b_small_flow"
    with pytest.raises(ResearchComponentError):
        save_pretrained(model, tmp_path / "bundle")


def test_default_run_checkpoints_carry_no_components(tmp_path):
    run_train(tmp_path / "run", training=reduced(steps=2, eval_every=2, milestone_steps=()))
    assert "components" not in payload_of(tmp_path / "run/checkpoint_last.pt")["schemas"]
    assert (tmp_path / "run/model").is_dir()


def test_same_shape_research_head_does_not_reload_as_the_default_model(tmp_path):
    # Its tensors fit the default model, so only the recorded components tell them apart.
    model = MFM(TINY, components=SAME_SHAPE)
    assert model.state_dict().keys() == MFM(TINY).state_dict().keys()
    path = tmp_path / "same_shape.pt"
    save_checkpoint(
        path, model=model, kind="best", step=1, training_config={}, run={}, consumed={},
        evaluation=None, selection={},
    )  # fmt: skip
    with pytest.raises(ResearchComponentError):
        load_checkpoint(path)
    reloaded, _ = load_checkpoint(path, allow_research=True)
    assert reloaded.components == model.components and reloaded.is_research


def test_unregistered_component_asks_for_the_plugin_module(tmp_path):
    path = tmp_path / "unknown.pt"
    save_checkpoint(
        path, model=MFM(TINY, components=SAME_SHAPE), kind="best", step=1, training_config={},
        run={}, consumed={}, evaluation=None, selection={},
    )  # fmt: skip
    payload = payload_of(path)
    payload["schemas"]["components"]["head"] = "not_registered"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="import the module that registers"):
        load_checkpoint(path, allow_research=True)


def test_resume_of_a_research_run_equals_the_uninterrupted_run(tmp_path, monkeypatch):
    def research_weights(path):
        model, _ = load_checkpoint(path, allow_research=True)
        return {key: value.clone() for key, value in model.state_dict().items()}

    monkeypatch.setattr(test_resume, "weights", research_weights)
    config = reduced(steps=20, eval_every=10, milestone_steps=(), prefetch_batches=0)
    kwargs = dict(training=config, components=SMALL_FLOW)
    run_train(tmp_path / "full", **kwargs)
    interrupted_run(tmp_path / "part", 10, **kwargs)
    result = run_train(
        tmp_path / "resumed", resume_from=tmp_path / "part/checkpoint_last.pt", **kwargs
    )
    assert result.status == "finished"
    assert_resumed_equals_full(tmp_path / "full", tmp_path / "resumed", 10)
    assert result.run["export"]["status"] == "skipped"


def test_resume_and_init_need_the_checkpoints_components(tmp_path):
    config = reduced(steps=4, eval_every=2, milestone_steps=(), prefetch_batches=0)
    run_train(tmp_path / "research", training=config, components=SMALL_FLOW)
    checkpoint = tmp_path / "research/checkpoint_last.pt"
    for start in ({"resume_from": checkpoint}, {"init_from": checkpoint}):
        with pytest.raises(ValueError, match="components differ"):
            run_train(tmp_path / f"default_{next(iter(start))}", training=config, **start)
    run_train(tmp_path / "default", training=config)
    with pytest.raises(ValueError, match="components differ"):
        run_train(
            tmp_path / "wrong", training=config, components=SMALL_FLOW,
            init_from=tmp_path / "default/checkpoint_last.pt",
        )  # fmt: skip
    warm = run_train(
        tmp_path / "warm", training=config, components=SMALL_FLOW, init_from=checkpoint
    )
    assert warm.run["initialized_from"]["step"] == 4


def test_research_run_refuses_holdout_archives_before_writing(tmp_path):
    with pytest.raises(ValueError, match="holdout_archives"):
        run_train(tmp_path / "run", components=SMALL_FLOW, holdout_archives=("terminal",))
    assert not (tmp_path / "run").exists()


@needs_generator
def test_score_replay_reproduces_a_research_checkpoint_exactly(
    research_run, corpus_path, tmp_path, capsys
):
    checkpoint = research_run / "checkpoint_last.pt"
    results = tmp_path / "results.csv"
    arguments = ["score", "--checkpoint", str(checkpoint), "--corpus", str(corpus_path),
                 "--replay", "--results", str(results)]  # fmt: skip
    with pytest.raises(SystemExit):
        main(arguments)
    assert "allow_research" in capsys.readouterr().err
    assert main([*arguments, "--allow-research", "--plugin", research_plugin.__name__]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["replay_keys_compared"] == summary["replay_keys_exact"] != "0"
    assert summary["replay_max_abs_diff"] == "0.0"
    (row,) = csv.DictReader(results.open())
    assert json.loads(row["components"]) == {"encoder": "bi_attention", **SMALL_FLOW}


@needs_generator
def test_training_cli_takes_components_and_plugins(corpus_path, tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(
        ["--corpus", str(corpus_path), "--out-dir", str(out), "--preset", "smoke",
         "--steps", "2", "--eval-every", "2", "--batch-size", "2", "--eval-worlds", "2",
         "--eval-samples", "2", "--no-tracking", "--device", "cpu", "--milestone-steps", "",
         "--plugin", research_plugin.__name__, "--component", "head=p6b_small_flow",
         "--component", "conditional_head=p6b_small_flow"]
    )  # fmt: skip
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "finished"
    assert payload_of(Path(out) / "checkpoint_last.pt")["schemas"]["components"]["head"] == (
        "p6b_small_flow"
    )
    assert not (out / "model").exists()
