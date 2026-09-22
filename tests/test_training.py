"""Training-runner contracts: schedule, records, checkpoints, determinism, failure."""

import importlib.util
import json
import math

import pytest
import torch
from conftest import make_synthetic_corpus

from pymc_mfm.inference import from_pretrained
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM
from pymc_mfm.training import (
    MONITOR_METRIC,
    TrackingConfig,
    TrainingConfig,
    TrainingError,
    TrainingObserver,
    load_checkpoint,
    train,
)
from pymc_mfm.training import trainer as trainer_module
from pymc_mfm.training.evaluation import GATE_METRICS
from pymc_mfm.training.stream import CorpusStream
from pymc_mfm.training.trainer import LOSS_TERMS, _select_checkpoint

TINY = ModelConfig(
    d_model=8,
    n_layers=1,
    n_heads=2,
    d_ff=16,
    flow_hidden=16,
    flow_n_freq=1,
    flow_steps_sample=2,
    flow_mc_samples=3,
    n_controls=2,
    n_demand=2,
)


def reduced(**overrides) -> TrainingConfig:
    values = dict(
        steps=4,
        batch_size=2,
        lr=1e-3,
        warmup_steps=2,
        eval_every=2,
        eval_worlds=2,
        eval_samples=3,
        log_every=1,
        milestone_steps=(2,),
        prefetch_batches=2,
        seed=5,
    )
    values.update(overrides)
    return TrainingConfig(**values)


class Recorder(TrainingObserver):
    def __init__(self):
        self.events = []

    def on_run_start(self, run):
        self.events.append(("start", run["status"]))

    def on_step(self, record):
        self.events.append(("step", record["step"]))

    def on_evaluation(self, record):
        self.events.append(("evaluation", record["step"]))

    def on_checkpoint(self, record):
        self.events.append(("checkpoint", record["kind"], record["step"]))

    def on_run_end(self, run):
        self.events.append(("end", run["status"]))


def test_learning_rate_follows_warmup_then_cosine_to_zero():
    config = TrainingConfig(steps=100, warmup_steps=10, lr=0.5, milestone_steps=())
    assert config.learning_rate(1) == pytest.approx(0.05)
    assert config.learning_rate(10) == pytest.approx(0.5)
    assert config.learning_rate(55) == pytest.approx(0.25)
    assert config.learning_rate(100) == pytest.approx(0.0, abs=1e-12)
    assert TrainingConfig(lr_schedule="constant", lr=0.5).learning_rate(62_500) == 0.5
    with pytest.raises(ValueError):
        config.learning_rate(101)

    truncated = TrainingConfig(steps=2, warmup_steps=5, lr=0.5, milestone_steps=())
    assert truncated.learning_rate(1) == pytest.approx(0.1)
    assert truncated.learning_rate(2) == pytest.approx(0.2)


def test_recorded_ab_contract_defines_canonical_budget_and_schedule():
    config = TrainingConfig()
    assert (config.steps, config.batch_size, config.lr) == (62_500, 64, 2.8e-3)
    assert (config.warmup_steps, config.eval_every, config.milestone_steps) == (
        3_125,
        6_250,
        (25_000, 50_000),
    )
    assert config.budget == "canonical"
    assert reduced().budget == "reduced"
    assert TrainingConfig.from_dict(config.to_dict()) == config
    assert config.learning_rate(config.warmup_steps) == config.lr
    assert config.learning_rate(config.steps) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"eval_samples": 1},
        {"steps": 10, "milestone_steps": (11,)},
        {"lr": True},
        {"grad_clip": True},
        {"weight_decay": True},
        {"weight_decay": False},
    ],
)
def test_config_rejects_invalid_training_values(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides)


@pytest.mark.parametrize(
    ("key", "value", "undefined_reason"),
    [
        pytest.param(MONITOR_METRIC, None, None, id="missing-monitor"),
        pytest.param(MONITOR_METRIC, float("nan"), None, id="nonfinite-monitor"),
        pytest.param(GATE_METRICS[0], float("inf"), None, id="nonfinite-structure"),
        pytest.param(GATE_METRICS[0], 0.2, "nonfinite_in_1_worlds", id="partial-structure"),
        pytest.param(GATE_METRICS[1], 0.0, "nonfinite_in_1_worlds", id="partial-leakage"),
        pytest.param(GATE_METRICS[1], 0.1, None, id="nonzero-leakage"),
    ],
)
def test_checkpoint_gate_failure_keeps_previous_best(key, value, undefined_reason):
    flat = {MONITOR_METRIC: 1.0, GATE_METRICS[0]: 0.2, GATE_METRICS[1]: 0.0}
    if value is None:
        del flat[key]
    else:
        flat[key] = value
    undefined = {} if undefined_reason is None else {key: undefined_reason}

    selection = _select_checkpoint(flat, undefined, best_value=2.0, best_step=4, step=8)

    assert selection["gates"][key]["passing"] is False
    assert selection["gates"][key]["reason"]
    assert all(gate["passing"] for name, gate in selection["gates"].items() if name != key)
    assert selection["gates_passing"] is False
    assert selection["promoted"] is False
    assert selection["best_value"] == 2.0 and selection["best_step"] == 4


def test_checkpoint_promotion_requires_clean_gates_and_strict_improvement():
    flat = {MONITOR_METRIC: 1.0, GATE_METRICS[0]: 0.2, GATE_METRICS[1]: 0.0}
    selection = _select_checkpoint(flat, {}, best_value=None, best_step=None, step=2)
    assert selection["promoted"] is True
    assert selection["best_value"] == 1.0 and selection["best_step"] == 2

    flat[MONITOR_METRIC] = 0.5
    improved = _select_checkpoint(flat, {}, best_value=1.0, best_step=2, step=4)
    assert improved["promoted"] is True
    assert improved["best_value"] == 0.5 and improved["best_step"] == 4

    for value in (0.5, 0.75):
        flat[MONITOR_METRIC] = value
        selection = _select_checkpoint(flat, {}, best_value=0.5, best_step=4, step=6)
        assert selection["gates_passing"] is True
        assert selection["promoted"] is False
        assert selection["best_value"] == 0.5 and selection["best_step"] == 4


def test_reduced_run_records_everything_and_reloads(tmp_path):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=3)
    recorder = Recorder()
    config = reduced()
    result = train(
        [corpus],
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=config,
        device="cpu",
        observers=[recorder],
    )
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "finished" and run["final_step"] == 4
    assert run["training_config"]["budget"] == "reduced"
    assert run["consumption_budget"]["train_rows"] == 12
    assert run["consumed"] == {
        **run["consumed"],
        "updates": 4,
        "rows": 8,
        "epoch": 0,
    }

    steps = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().split("\n")
        if line
    ]
    assert [record["step"] for record in steps] == [1, 2, 3, 4]
    for record in steps:
        assert all(math.isfinite(record[name]) for name in LOSS_TERMS)
        assert record["lr"] == config.learning_rate(record["step"])
        assert record["consumed_rows"] == record["step"] * config.batch_size
        assert math.isfinite(record["grad_norm"])
    trace = json.loads((tmp_path / "run" / "objective_trace.json").read_text())
    assert "per_family_structure_bce" in trace["executed_objectives"]

    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "evaluations.jsonl").read_text().split("\n")
        if line
    ]
    assert [event["step"] for event in events] == [2, 4]
    for event in events:
        assert set(event["legs"]) == {"holdout", "seen"}
        for leg in event["legs"].values():
            assert set(leg["heads"]) == {"joint", "conditional", "structure_head", "joint_router"}
        assert MONITOR_METRIC in event["flat"]
        assert event["selection"]["monitor"] == MONITOR_METRIC
    assert events[0]["selection"]["promoted"] is True

    files = {entry["file"]: entry for entry in run["checkpoints"]}
    assert set(files) >= {"checkpoint_last.pt", "checkpoint_best.pt", "checkpoint_step_2.pt"}
    assert files["checkpoint_last.pt"]["step"] == 4
    assert files["checkpoint_step_2.pt"]["kind"] == "milestone"
    assert not list((tmp_path / "run").glob(".*.tmp"))

    reloaded, payload = load_checkpoint(tmp_path / "run" / "checkpoint_last.pt")
    assert payload["step"] == 4 and payload["kind"] == "last"
    assert payload["evaluation"]["step"] == 4 and payload["run"]["run_id"] == run["run_id"]
    exported = from_pretrained(tmp_path / "run" / "model")
    for name, parameter in result.model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], parameter, rtol=0, atol=0)
        torch.testing.assert_close(exported.state_dict()[name], parameter, rtol=0, atol=0)

    milestone, milestone_payload = load_checkpoint(tmp_path / "run" / "checkpoint_step_2.pt")
    assert milestone_payload["evaluation"] is None
    assert any(
        not torch.equal(a, b)
        for a, b in zip(milestone.state_dict().values(), reloaded.state_dict().values())
    )

    from pymc_mfm.training.stream import batches_for_rows

    holdout = [(0, int(row)) for row in corpus["is_val"].nonzero()[0][:2]]
    batch, _ = batches_for_rows([corpus], holdout, device="cpu")[0]
    draws, graphs = reloaded.sample_disentangled(batch, 2)
    assert draws.shape[:2] == (2, len(holdout)) and torch.isfinite(draws).all()

    kinds = [event for event in recorder.events if event[0] != "step"]
    assert kinds[0] == ("start", "running") and kinds[-1] == ("end", "finished")
    assert ("evaluation", 2) in kinds and ("checkpoint", "milestone", 2) in kinds
    assert sum(event[0] == "step" for event in recorder.events) == 4


def test_seed_reproduces_weights_and_observers_or_evaluation_never_perturb_training(tmp_path):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=4)

    def run(name, observers=(), **overrides):
        result = train(
            [corpus],
            tracking=TrackingConfig(enabled=False),
            out_dir=tmp_path / name,
            model_config=TINY,
            training=reduced(**overrides),
            device="cpu",
            observers=observers,
        )
        losses = [
            json.loads(line)["loss"]
            for line in (tmp_path / name / "metrics.jsonl").read_text().split("\n")
            if line
        ]
        return result.model.state_dict(), losses

    base_state, base_losses = run("base")
    same_state, same_losses = run(
        "same_seed_other_eval",
        observers=[Recorder()],
        eval_samples=5,
        eval_worlds=3,
        prefetch_batches=0,
    )
    other_state, other_losses = run("other_seed", seed=6)
    assert base_losses == same_losses
    for name, parameter in base_state.items():
        torch.testing.assert_close(same_state[name], parameter, rtol=0, atol=0)
    assert base_losses != other_losses
    assert any(not torch.equal(other_state[name], value) for name, value in base_state.items())


def test_nonfinite_loss_fails_the_run_without_a_finished_record(tmp_path, monkeypatch):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=7)
    original = MFM.loss_terms
    calls = {"n": 0}

    def poisoned(self, batch, *, return_trace=False):
        calls["n"] += 1
        result = original(self, batch, return_trace=return_trace)
        if calls["n"] == 2:
            terms = result[0] if return_trace else result
            terms["loss"] = terms["loss"] * float("nan")
        return result

    monkeypatch.setattr(MFM, "loss_terms", poisoned)
    with pytest.raises(TrainingError):
        train(
            [corpus],
            tracking=TrackingConfig(enabled=False),
            out_dir=tmp_path / "run",
            model_config=TINY,
            training=reduced(prefetch_batches=0),
            device="cpu",
        )
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "failed" and run["final_step"] == 2
    assert run["consumed"]["updates"] == 2 and run["consumed"]["rows"] == 4
    assert run["error_type"] == "TrainingError"
    assert not (tmp_path / "run" / "model").exists()
    assert not (tmp_path / "run" / "checkpoint_last.pt").exists()


@pytest.mark.parametrize(
    ("exception_type", "status"),
    [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")],
)
def test_loss_failure_counts_only_completed_updates_with_prefetch(
    tmp_path, monkeypatch, exception_type, status
):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=7)
    recorder = Recorder()
    original = MFM.loss_terms
    failure = exception_type("loss computation stopped")
    calls = 0

    def fail_third_loss(self, batch, *, return_trace=False):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise failure
        return original(self, batch, return_trace=return_trace)

    monkeypatch.setattr(MFM, "loss_terms", fail_third_loss)
    with pytest.raises(exception_type) as exc_info:
        train(
            [corpus],
            tracking=TrackingConfig(enabled=False),
            out_dir=tmp_path / "run",
            model_config=TINY,
            training=reduced(eval_every=4, milestone_steps=(), prefetch_batches=2),
            device="cpu",
            observers=[recorder],
        )

    assert exc_info.value is failure
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == status
    assert run["final_step"] == run["consumed"]["updates"] == 2
    assert run["consumed"]["rows"] == run["consumed"]["distinct_rows"] == 4
    assert run["error"]
    steps = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in steps] == [1, 2]
    assert [record["consumed_rows"] for record in steps] == [2, 4]
    assert recorder.events[-1] == ("end", status)
    assert not (tmp_path / "run" / "model").exists()
    assert not (tmp_path / "run" / "checkpoint_last.pt").exists()


def test_prefetch_producer_failure_propagates_and_records_completed_updates(tmp_path, monkeypatch):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=7)
    original = CorpusStream.next_batch
    failure = RuntimeError("corpus producer failed")
    recorder = Recorder()

    def fail_third_batch(self):
        if self.batches == 2:
            raise failure
        return original(self)

    monkeypatch.setattr(CorpusStream, "next_batch", fail_third_batch)
    with pytest.raises(RuntimeError) as exc_info:
        train(
            [corpus],
            tracking=TrackingConfig(enabled=False),
            out_dir=tmp_path / "run",
            model_config=TINY,
            training=reduced(eval_every=4, milestone_steps=(), prefetch_batches=2),
            device="cpu",
            observers=[recorder],
        )

    assert exc_info.value is failure
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["final_step"] == run["consumed"]["updates"] == 2
    assert run["consumed"]["rows"] == 4
    assert run["error_type"] == "RuntimeError"
    assert recorder.events[-1] == ("end", "failed")
    assert not (tmp_path / "run" / "model").exists()


def test_startup_observer_failure_records_failed_run_without_consumption(tmp_path):
    failure = RuntimeError("observer startup failed")

    class FailingObserver(Recorder):
        def on_run_start(self, run):
            super().on_run_start(run)
            raise failure

    recorder = FailingObserver()
    with pytest.raises(RuntimeError) as exc_info:
        train(
            [make_synthetic_corpus()],
            tracking=TrackingConfig(enabled=False),
            out_dir=tmp_path / "run",
            model_config=TINY,
            training=reduced(prefetch_batches=2),
            device="cpu",
            observers=[recorder],
        )

    assert exc_info.value is failure
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["final_step"] == run["consumed"]["updates"] == 0
    assert run["consumed"]["rows"] == run["consumed"]["distinct_rows"] == 0
    assert run["error_type"] == "RuntimeError"
    assert recorder.events[-1] == ("end", "failed")
    assert not (tmp_path / "run" / "model").exists()


def test_partial_holdout_prediction_blocks_best_checkpoint_but_finishes_training(
    tmp_path, monkeypatch
):
    corpus = make_synthetic_corpus(n_cells=4, draws_per_cell=1, context_lengths=(6,))
    original = MFM.predict
    poisoned = False

    def partial_prediction(self, batch, n_samples=None, *, return_structure=False):
        nonlocal poisoned
        result = original(self, batch, n_samples, return_structure=return_structure)
        if not poisoned:
            draws = result[0] if return_structure else result
            draws[:, 0, :, 0] = float("nan")
            poisoned = True
        return result

    monkeypatch.setattr(MFM, "predict", partial_prediction)
    result = train(
        [corpus],
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(steps=1, eval_every=1, eval_samples=2, milestone_steps=()),
        device="cpu",
    )

    evaluation = json.loads((tmp_path / "run" / "evaluations.jsonl").read_text())
    holdout = evaluation["legs"]["holdout"]
    assert holdout["n_worlds"] == 2
    assert math.isfinite(evaluation["flat"][MONITOR_METRIC])
    metric = MONITOR_METRIC.removeprefix("holdout/")
    assert holdout["undefined"][metric] == "nonfinite_in_1_worlds"
    selection = evaluation["selection"]
    assert selection["gates"][MONITOR_METRIC]["passing"] is False
    assert all(selection["gates"][key]["passing"] for key in GATE_METRICS)
    assert selection["promoted"] is False
    assert result.status == "finished" and result.run["best"] is None
    assert (tmp_path / "run" / "checkpoint_last.pt").exists()
    assert (tmp_path / "run" / "model").is_dir()
    assert not (tmp_path / "run" / "checkpoint_best.pt").exists()


def test_multi_corpus_training_evaluates_all_available_holdout_and_seen_identities(
    tmp_path, monkeypatch
):
    corpora = [
        make_synthetic_corpus(
            n_cells=3, draws_per_cell=1, val_cells=1, context_lengths=(6,), seed=seed
        )
        for seed in (2, 3)
    ]
    evaluated_rows = {"holdout": [], "seen": []}
    original = trainer_module.evaluate_rows

    def record_evaluated_rows(model, corpora, rows, **kwargs):
        evaluated_rows[kwargs["leg"]].append(tuple(rows))
        return original(model, corpora, rows, **kwargs)

    monkeypatch.setattr(trainer_module, "evaluate_rows", record_evaluated_rows)
    result = train(
        corpora,
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(steps=2, eval_every=1, eval_worlds=10, eval_samples=2, milestone_steps=()),
        device="cpu",
    )

    evaluations = [
        json.loads(line)
        for line in (tmp_path / "run" / "evaluations.jsonl").read_text().splitlines()
    ]
    expected_holdout = {(0, 2), (1, 2)}
    expected_train = {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert result.status == "finished"
    assert result.run["consumed"]["rows"] == result.run["consumed"]["distinct_rows"] == 4
    for rows in evaluated_rows["holdout"]:
        assert len(rows) == 2 and set(rows) == expected_holdout
    assert evaluated_rows["holdout"][0] == evaluated_rows["holdout"][1]
    assert set(evaluated_rows["seen"][0]) <= expected_train
    assert set(evaluated_rows["seen"][1]) == expected_train
    assert evaluated_rows["seen"][1][:2] == evaluated_rows["seen"][0]
    assert [event["legs"]["holdout"]["n_worlds"] for event in evaluations] == [2, 2]
    assert [event["legs"]["seen"]["n_worlds"] for event in evaluations] == [2, 4]


def test_epoch_budget_excludes_rows_dropped_from_each_context_bucket(tmp_path):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=1)
    result = train(
        [corpus],
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(steps=3, eval_every=3, eval_samples=2, milestone_steps=()),
        device="cpu",
    )

    budget = result.run["consumption_budget"]
    assert budget["train_rows"] == 6
    assert budget["rows_per_epoch"] == 4
    assert budget["dropped_rows_per_epoch"] == 2
    assert budget["epochs"] == 1.5
    assert result.run["consumed"]["rows"] == 6
    steps = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["epoch"] for record in steps] == [0, 0, 1]


@pytest.mark.parametrize("hook", ["on_evaluation", "on_run_end", "on_checkpoint"])
def test_observer_failure_records_only_persisted_best_checkpoint(tmp_path, hook):
    failure = RuntimeError("observer delivery failed")

    class FailingObserver(TrainingObserver):
        pass

    def fail(self, record):
        # Exercise the callback after the best file was persisted, not the last file.
        if hook != "on_checkpoint" or record["kind"] == "best":
            raise failure

    setattr(FailingObserver, hook, fail)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError) as exc_info:
        train(
            [make_synthetic_corpus()],
            tracking=TrackingConfig(enabled=False),
            out_dir=out,
            model_config=TINY,
            training=reduced(steps=1, warmup_steps=1, milestone_steps=()),
            device="cpu",
            observers=[FailingObserver()],
        )
    assert exc_info.value is failure
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["error"] is not None
    assert run["final_step"] == run["consumed"]["updates"] == 1
    if hook == "on_evaluation":
        assert run["best"] is None
        assert not (out / "checkpoint_best.pt").exists()
    else:
        _, payload = load_checkpoint(out / "checkpoint_best.pt")
        assert run["best"] == {
            "step": payload["step"],
            "value": payload["selection"]["value"],
        }


def test_run_directory_and_geometry_are_guarded(tmp_path):
    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=8)
    train(
        [corpus],
        tracking=TrackingConfig(enabled=False),
        out_dir=tmp_path / "run",
        model_config=TINY,
        training=reduced(steps=1, eval_every=1, milestone_steps=()),
        device="cpu",
    )
    with pytest.raises(FileExistsError):
        train(
            [corpus],
            out_dir=tmp_path / "run",
            model_config=TINY,
            training=reduced(),
            tracking=TrackingConfig(enabled=False),
        )
    with pytest.raises(ValueError):
        train(
            [corpus],
            tracking=TrackingConfig(enabled=False),
            out_dir=tmp_path / "other",
            model_config=ModelConfig(n_controls=3, n_demand=2),
            training=reduced(),
        )


def test_checkpoint_loader_rejects_foreign_payloads(tmp_path):
    torch.save({"config": TINY.to_dict(), "state_dict": MFM(TINY).state_dict()}, tmp_path / "m.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "m.pt")


@pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
)
def test_cli_trains_from_saved_corpora_with_recorded_provenance(tmp_path, capsys):
    from pymc_generator import save_corpus

    from pymc_mfm.training.__main__ import main

    corpus = make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9)
    path = tmp_path / "corpus.npz"
    save_corpus(corpus, path)
    code = main(
        [
            "--no-tracking",
            "--corpus",
            str(path),
            "--out-dir",
            str(tmp_path / "run"),
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
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "finished" and summary["budget"] == "reduced"
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["run_name"] == "cli"
    assert run["training_config"]["milestone_steps"] == []
    assert run["model_config"]["n_controls"] == 2 and run["model_config"]["n_demand"] == 2
    assert run["corpus"][0]["path"] == str(path) and len(run["corpus"][0]["sha256"]) == 64
