"""Optional bf16 autocast and torch.compile: defaults, old records, and compiled-run fidelity."""

import json
import math
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import make_synthetic_corpus
from test_resume import NO_TRACKING, interrupted_run, run_train
from test_training import TINY, reduced

from pymc_mfm.inference import from_pretrained
from pymc_mfm.training import TrainingConfig, load_checkpoint, train


def metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def train_run(out: Path, config: TrainingConfig, corpus=None):
    return train(
        [make_synthetic_corpus() if corpus is None else corpus],
        out_dir=out,
        model_config=TINY,
        training=config,
        device="cpu",
        tracking=NO_TRACKING,
        measure_unadopted_acceleration=True,
    )


def test_a_default_config_is_fp32_and_uncompiled():
    config = TrainingConfig()
    assert (config.precision, config.compile) == ("fp32", False)
    assert config.to_dict()["precision"] == "fp32"
    assert config.to_dict()["compile"] is False


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="precision must be one of"):
        TrainingConfig(precision="fp16")


def test_a_record_written_before_these_fields_rebuilds_with_the_defaults():
    old_record = reduced().to_dict()
    del old_record["precision"], old_record["compile"]
    assert TrainingConfig.from_dict(old_record) == reduced()


def test_a_record_with_an_unknown_field_is_still_refused():
    with pytest.raises(ValueError, match="unknown=.*dtype"):
        TrainingConfig.from_dict({**reduced().to_dict(), "dtype": "fp32"})


def test_resume_accepts_a_checkpoint_written_before_these_fields(tmp_path):
    source = tmp_path / "source"
    config = interrupted_run(source, 3)
    path = source / "checkpoint_last.pt"
    payload = torch.load(path, weights_only=False)
    del payload["training_config"]["precision"], payload["training_config"]["compile"]
    torch.save(payload, path)

    resumed = run_train(tmp_path / "resumed", training=config, resume_from=path)

    assert resumed.run["status"] == "finished"
    assert resumed.run["training_config"]["precision"] == "fp32"


def test_resume_refuses_a_changed_precision(tmp_path):
    source = tmp_path / "source"
    config = interrupted_run(source, 3)
    with pytest.raises(ValueError, match=r"training_config differs.*precision"):
        run_train(
            tmp_path / "resumed",
            training=replace(config, precision="bf16"),
            resume_from=source / "checkpoint_last.pt",
            measure_unadopted_acceleration=True,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"precision": "bf16"}, "precision='bf16' are measured and not adopted"),
        ({"compile": True}, "compile=True are measured and not adopted"),
    ],
)
def test_unadopted_acceleration_is_refused_outside_measurement(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        train(
            [make_synthetic_corpus()],
            out_dir=tmp_path / "run",
            model_config=TINY,
            training=replace(reduced(steps=2, eval_every=2, milestone_steps=()), **change),
            device="cpu",
            tracking=NO_TRACKING,
        )
    assert not (tmp_path / "run").exists()


def test_the_training_cli_has_no_precision_or_compile_flag():
    from pymc_mfm.training.__main__ import build_parser

    flags = {option for action in build_parser()._actions for option in action.option_strings}
    assert not {"--precision", "--compile", "--no-compile"} & flags


def test_training_cli_fields_exclude_measurement_only_fields():
    from dataclasses import fields

    from pymc_mfm.training.config import training_cli_fields

    names = training_cli_fields()
    assert isinstance(names, frozenset)
    assert names == {field.name for field in fields(TrainingConfig)} - {"precision", "compile"}


def test_measured_cuda_step_clock_synchronizes(monkeypatch):
    from pymc_mfm.training import trainer

    calls = []
    device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda value: calls.append(value))
    monkeypatch.setattr(trainer.time, "perf_counter", lambda: 12.5)

    assert trainer._step_clock(device, measure=True) == 12.5
    assert trainer._step_clock(device, measure=True) == 12.5
    assert calls == [device, device]

    trainer._step_clock(device, measure=False)
    trainer._step_clock(torch.device("cpu"), measure=True)
    assert calls == [device, device]


def test_train_calls_step_clock_at_both_step_boundaries(tmp_path, monkeypatch):
    from pymc_mfm.training import trainer

    calls = []
    step_clock = trainer._step_clock

    def record_clock(device, *, measure):
        calls.append((device.type, measure))
        return step_clock(device, measure=measure)

    monkeypatch.setattr(trainer, "_step_clock", record_clock)
    train_run(tmp_path / "run", reduced(steps=2, eval_every=2, milestone_steps=()))

    assert calls == [("cpu", True)] * 4


def test_benchmark_summary_reports_wall_clock_between_post_warmup_logs(tmp_path):
    benchmark = Path(__file__).resolve().parents[1] / "benchmarks/precision_compile_check.py"
    summarize = runpy.run_path(str(benchmark))["summarize"]

    elapsed = [1, 2, 3, 4, 7, 9]
    records = [
        {"loss": 1.0, "step_seconds": 0.1, "elapsed_seconds": seconds} for seconds in elapsed
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(map(json.dumps, records)))
    (tmp_path / "evaluations.jsonl").write_text(json.dumps({"flat": {"monitor": 0.5}}))
    result = SimpleNamespace(
        run={
            "selection_policy": {"monitor": "monitor"},
            "compile": {"first_call_seconds": None, "graphs": None, "graph_breaks": None},
        }
    )

    summary = summarize(tmp_path, result, 100.0)

    assert summary["median_step_seconds"] == pytest.approx(0.1)
    assert summary["median_wall_seconds"] == pytest.approx(2.5)


def test_bf16_trains_with_finite_losses_on_fp32_master_weights(tmp_path):
    config = reduced(steps=6, eval_every=6, milestone_steps=())
    fp32 = train_run(tmp_path / "fp32", config)
    bf16 = train_run(tmp_path / "bf16", replace(config, precision="bf16"))

    losses = [record["loss"] for record in metrics(tmp_path / "bf16")]
    assert len(losses) == 6 and all(math.isfinite(loss) for loss in losses)
    assert {p.dtype for p in bf16.model.parameters()} == {torch.float32}
    # The step-1 loss sees the same fresh weights; only the forward rounding differs.
    first = [metrics(tmp_path / name)[0]["loss"] for name in ("fp32", "bf16")]
    assert first[1] == pytest.approx(first[0], rel=0.05)
    assert fp32.run["compile"]["enabled"] is False


def test_compiled_losses_match_eager_and_the_original_module_is_saved(tmp_path):
    # One active channel in every world keeps every parameter in every step's graph,
    # so eager and compiled AdamW updates see the same gradients.
    corpus = make_synthetic_corpus(k=2)
    config = reduced(steps=10, eval_every=10, milestone_steps=(), log_every=1)
    eager = train_run(tmp_path / "eager", config, corpus)
    compiled = train_run(tmp_path / "compiled", replace(config, compile=True), corpus)

    pairs = zip(metrics(tmp_path / "eager"), metrics(tmp_path / "compiled"), strict=True)
    assert max(abs(a["loss"] - b["loss"]) for a, b in pairs) < 1e-5

    record = compiled.run["compile"]
    assert record["enabled"] and record["fallback_random"] and record["backend"] == "inductor"
    assert record["first_call_seconds"] > 0 and record["graphs"] >= 1
    assert eager.run["compile"]["enabled"] is False

    model, _ = load_checkpoint(tmp_path / "compiled" / "checkpoint_last.pt")
    exported = from_pretrained(tmp_path / "compiled" / "model")
    for loaded in (model, exported):
        assert all(
            torch.equal(value, compiled.model.state_dict()[key])
            for key, value in loaded.state_dict().items()
        )
