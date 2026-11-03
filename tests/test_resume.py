"""Exact resume, same-directory resume and weights-only warm start."""

import json
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus
from test_training import TINY, reduced

from pymc_mfm.inference._artifact_schema import semantic_schema
from pymc_mfm.model.model import MFM
from pymc_mfm.training import (
    TrackingConfig,
    TrainingObserver,
    load_checkpoint,
    train,
)
from pymc_mfm.training import trainer as trainer_module
from pymc_mfm.training.checkpoint import CHECKPOINT_FORMAT_VERSION, file_sha256
from pymc_mfm.training.resume import (
    EMPTY_ROW_SEQUENCE,
    ROW_SEQUENCE_SCHEME,
    capture_rng_state,
    chain_row_sequence,
    check_rng_fits_device,
    restore_rng_state,
)
from pymc_mfm.training.stream import CorpusStream, PrefetchingStream, SeenRows
from pymc_mfm.training.trainer import LOSS_TERMS

NO_TRACKING = TrackingConfig(enabled=False)
COMPARED = (*LOSS_TERMS, "grad_norm", "lr", "row_sequence_sha256", "batch_rows", "epoch")


class InterruptAfterLast(TrainingObserver):
    """Raise KeyboardInterrupt once ``checkpoint_last.pt`` of ``step`` is on disk."""

    def __init__(self, step):
        self.step = step

    def on_checkpoint(self, record):
        if record["kind"] == "last" and record["step"] == self.step:
            raise KeyboardInterrupt


def run_train(out, *, observers=(), corpus=None, model_config=TINY, **kwargs):
    config = kwargs.pop("training", None) or reduced(steps=9, eval_every=3, milestone_steps=())
    return train(
        [make_synthetic_corpus() if corpus is None else corpus],
        out_dir=out,
        model_config=model_config,
        training=config,
        device="cpu",
        tracking=kwargs.pop("tracking", NO_TRACKING),
        observers=observers,
        **kwargs,
    )


def interrupted_run(out, step, **kwargs):
    config = kwargs.get("training") or reduced(steps=9, eval_every=step, milestone_steps=())
    kwargs["training"] = config
    with pytest.raises(KeyboardInterrupt):
        run_train(out, observers=[InterruptAfterLast(step)], **kwargs)
    return config


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def weights(path):
    model, _ = load_checkpoint(path)
    return {key: value.clone() for key, value in model.state_dict().items()}


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[key], b[key]) for key in a)


def assert_resumed_equals_full(full_dir, resumed_dir, after):
    full = {r["step"]: r for r in jsonl(full_dir / "metrics.jsonl")}
    resumed = {r["step"]: r for r in jsonl(resumed_dir / "metrics.jsonl")}
    later = [step for step in full if step > after]
    assert later and all(step in resumed for step in later)
    for step in later:
        for key in COMPARED:
            assert resumed[step][key] == full[step][key], (step, key)
    assert_same_weights(
        weights(full_dir / "checkpoint_last.pt"), weights(resumed_dir / "checkpoint_last.pt")
    )
    last_full = jsonl(full_dir / "evaluations.jsonl")[-1]
    last_resumed = jsonl(resumed_dir / "evaluations.jsonl")[-1]
    assert last_resumed["flat"] == last_full["flat"]
    assert last_resumed["legs"]["seen"]["rows"] == last_full["legs"]["seen"]["rows"]
    full_run = json.loads((full_dir / "run.json").read_text())
    resumed_run = json.loads((resumed_dir / "run.json").read_text())
    assert resumed_run["consumed"] == full_run["consumed"]
    assert resumed_run["status"] == "finished" and resumed_run["final_step"] == 9


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full") / "run"
    run_train(out)
    return out


@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize("step", [3, 4], ids=["mid-epoch", "epoch-boundary"])
def test_resume_in_a_new_directory_equals_the_uninterrupted_run(tmp_path, prefetch, step):
    # Eight training rows, batch 2, two context lengths: four batches per epoch,
    # so step 4 is an epoch boundary and step 3 is inside an epoch.
    base = reduced(steps=9, eval_every=3, milestone_steps=(), prefetch_batches=prefetch)
    config = replace(base, eval_every=step)
    full = tmp_path / "full_same_cadence"
    run_train(full, training=config)
    interrupted_run(tmp_path / "part", step, training=config)
    result = run_train(
        tmp_path / "resumed", training=config, resume_from=tmp_path / "part/checkpoint_last.pt"
    )
    assert result.status == "finished"
    assert_resumed_equals_full(full, tmp_path / "resumed", step)
    resumed_run = json.loads((tmp_path / "resumed/run.json").read_text())
    full_run = json.loads((full / "run.json").read_text())
    assert resumed_run["run_id"] != full_run["run_id"]
    source = json.loads((tmp_path / "part/run.json").read_text())
    assert resumed_run["resumed_from"] == {
        "path": str(tmp_path / "part/checkpoint_last.pt"),
        "step": step,
        "sha256": file_sha256(tmp_path / "part/checkpoint_last.pt"),
        "run_id": source["run_id"],
        "kind": "last",
        "history_copied": True,
    }
    # The new directory is a complete record: the source's prefix, then the new steps.
    steps = [r["step"] for r in jsonl(tmp_path / "resumed/metrics.jsonl")]
    assert steps == list(range(1, 10))


def test_prefetch_setting_may_differ_on_resume(tmp_path):
    config = reduced(steps=9, eval_every=4, milestone_steps=(), prefetch_batches=2)
    run_train(tmp_path / "full", training=config)
    interrupted_run(tmp_path / "part", 4, training=config)
    resumed = replace(config, prefetch_batches=0)
    run_train(
        tmp_path / "resumed", training=resumed, resume_from=tmp_path / "part/checkpoint_last.pt"
    )
    assert_resumed_equals_full(tmp_path / "full", tmp_path / "resumed", 4)


def test_resume_from_a_milestone_runs_the_evaluation_that_followed_it(tmp_path):
    config = reduced(steps=9, eval_every=3, milestone_steps=(3,))
    run_train(tmp_path / "full", training=config)
    run_train(
        tmp_path / "resumed", training=config, resume_from=tmp_path / "full/checkpoint_step_3.pt"
    )
    assert_resumed_equals_full(tmp_path / "full", tmp_path / "resumed", 3)
    assert [r["step"] for r in jsonl(tmp_path / "resumed/evaluations.jsonl")] == [3, 6, 9]
    full_eval = jsonl(tmp_path / "full/evaluations.jsonl")[0]
    assert jsonl(tmp_path / "resumed/evaluations.jsonl")[0]["flat"] == full_eval["flat"]


@pytest.mark.parametrize("skipped", ["stream", "rng"])
def test_negative_control_diverges_when_state_restoration_is_skipped(
    tmp_path, monkeypatch, full_run, skipped
):
    config = reduced(steps=9, eval_every=3, milestone_steps=())
    interrupted_run(tmp_path / "part", 3, training=config)
    if skipped == "stream":
        monkeypatch.setattr(CorpusStream, "restore", lambda self, state: None)
    else:
        monkeypatch.setattr(trainer_module, "restore_rng_state", lambda *args: None)
    run_train(
        tmp_path / "broken", training=config, resume_from=tmp_path / "part/checkpoint_last.pt"
    )
    full = {r["step"]: r for r in jsonl(full_run / "metrics.jsonl")}
    broken = {r["step"]: r for r in jsonl(tmp_path / "broken/metrics.jsonl")}
    differing = [
        step for step in range(4, 10) if any(broken[step][k] != full[step][k] for k in LOSS_TERMS)
    ]
    assert differing
    final = weights(tmp_path / "broken/checkpoint_last.pt")
    reference = weights(full_run / "checkpoint_last.pt")
    assert not all(torch.equal(final[key], reference[key]) for key in final)


# ---- same-directory resume ----


@pytest.mark.parametrize("hard_kill", [False, True], ids=["interrupt", "stale-run-json"])
def test_same_directory_resume_equals_the_uninterrupted_run(tmp_path, hard_kill):
    config = reduced(steps=9, eval_every=3, milestone_steps=(2, 5))
    run_train(tmp_path / "full", training=config)
    out = tmp_path / "run"
    interrupted_run(out, 6, training=config)
    original = json.loads((out / "run.json").read_text())
    checkpoint_sha256 = file_sha256(out / "checkpoint_last.pt")
    # What a run that outlived its checkpoint left: later steps, a torn line, a later
    # milestone and a later evaluation table.
    with (out / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({**jsonl(out / "metrics.jsonl")[-1], "step": 7, "loss": -1.0}))
        handle.write('\n{"step": 8, "lo')
    with (out / "evaluations.jsonl").open("a") as handle:
        handle.write(json.dumps({"step": 9, "flat": {}}) + "\n")
    (out / "evaluation_tables/step_000000009.json").write_text("{}")
    (out / "checkpoint_step_8.pt").write_bytes(b"stale")
    if hard_kill:
        stale = {**original, "status": "running", "checkpoints": [], "finished_at": None}
        (out / "run.json").write_text(json.dumps(stale))
    run_train(out, training=config, resume_from=out / "checkpoint_last.pt")

    assert_resumed_equals_full(tmp_path / "full", out, 6)
    assert [r["step"] for r in jsonl(out / "metrics.jsonl")] == list(range(1, 10))
    assert [r["step"] for r in jsonl(out / "evaluations.jsonl")] == [3, 6, 9]
    table = json.loads((out / "evaluation_tables/step_000000009.json").read_text())
    assert table["step"] == 9  # the stale placeholder was replaced by the real one
    assert not (out / "checkpoint_step_8.pt").exists()
    run = json.loads((out / "run.json").read_text())
    assert run["run_id"] == original["run_id"] and run["created_at"] == original["created_at"]
    assert "resumed_from" not in run
    assert [(r["from_step"], r["checkpoint_sha256"]) for r in run["resumes"]] == [
        (6, checkpoint_sha256)
    ]
    listed = [(e["kind"], e["step"], e["file"]) for e in run["checkpoints"]]
    assert [item for item in listed if item[0] != "best"] == [
        ("milestone", 2, "checkpoint_step_2.pt"),
        ("milestone", 5, "checkpoint_step_5.pt"),
        ("last", 9, "checkpoint_last.pt"),
    ]
    assert [item[2] for item in listed if item[0] == "best"] == ["checkpoint_best.pt"]
    for entry in run["checkpoints"]:
        assert file_sha256(out / entry["file"]) == entry["sha256"]
    best = next(e for e in run["checkpoints"] if e["kind"] == "best")
    assert run["best"]["step"] == best["step"]
    assert (out / "objective_trace.json").is_file()


def test_same_directory_resume_after_a_sigkill_equals_the_uninterrupted_run(tmp_path):
    config = reduced(steps=9, eval_every=3, milestone_steps=())
    run_train(tmp_path / "full", training=config)
    out = tmp_path / "run"
    script = Path(__file__).with_name("hard_kill_run.py")
    killed = subprocess.run([sys.executable, str(script), str(out), "6"], capture_output=True)
    assert killed.returncode == -signal.SIGKILL, killed.stderr.decode()
    assert json.loads((out / "run.json").read_text())["status"] == "running"

    run_train(out, training=config, resume_from=out / "checkpoint_last.pt")

    assert_resumed_equals_full(tmp_path / "full", out, 6)


def test_same_directory_resume_writes_the_best_checkpoint_the_interruption_preceded(tmp_path):
    out = tmp_path / "run"
    config = interrupted_run(out, 3)
    payload = torch.load(out / "checkpoint_last.pt", weights_only=True)
    assert payload["selection"]["promoted"]  # the first evaluation always promotes
    assert not (out / "checkpoint_best.pt").exists()  # the interrupt came before it
    run_train(out, training=config, resume_from=out / "checkpoint_last.pt")
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "finished"
    assert (out / "checkpoint_best.pt").is_file()


def test_same_directory_resume_only_starts_from_checkpoint_last(tmp_path):
    config = reduced(steps=9, eval_every=3, milestone_steps=(3,))
    out = tmp_path / "run"
    run_train(out, training=config)
    before = snapshot(out)
    with pytest.raises(ValueError, match="checkpoint_last.pt"):
        run_train(out, training=config, resume_from=out / "checkpoint_step_3.pt")
    assert snapshot(out) == before
    # A resume into another existing run is a FileExistsError, as for any new run.
    with pytest.raises(FileExistsError):
        run_train(out, training=config, init_from=out / "checkpoint_last.pt")
    assert snapshot(out) == before


# ---- refusals ----


def snapshot(directory):
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "tracking" not in path.parts
    }


def write_format2(path, *, training=None):
    """A format 2 checkpoint: the exact key set ``save_checkpoint`` wrote before ``resume``."""
    torch.manual_seed(1)
    model = MFM(TINY)
    config = reduced() if training is None else training
    payload = {
        "format_version": 2,
        "kind": "last",
        "step": 4,
        "config": model.config.to_dict(),
        "schemas": semantic_schema(model.config),
        "state_dict": model.state_dict(),
        "training_config": config.to_dict(),
        "run": {"run_id": "format2-run", "profile": "new-run", "corpus": []},
        "consumed": {},
        "evaluation": None,
        "selection": {},
    }
    torch.save(payload, path)
    return path


def write_dimensional(path):
    """A dimensional-preprocessing checkpoint: format 1 and no ``schemas``."""
    model = MFM(TINY)
    torch.save(
        {
            "format_version": 1,
            "kind": "last",
            "step": 4,
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_format_3_checkpoints_carry_resume_state_except_best(tmp_path):
    run_train(tmp_path / "run", training=reduced(steps=6, eval_every=3, milestone_steps=(2,)))
    assert CHECKPOINT_FORMAT_VERSION == 3
    for name, carries in (
        ("checkpoint_last.pt", True),
        ("checkpoint_step_2.pt", True),
        ("checkpoint_best.pt", False),
    ):
        _, payload = load_checkpoint(tmp_path / "run" / name)
        assert payload["format_version"] == 3
        assert (payload["resume"] is not None) is carries
    _, last = load_checkpoint(tmp_path / "run/checkpoint_last.pt")
    consumed = last["consumed"]
    assert consumed["row_sequence_scheme"] == ROW_SEQUENCE_SCHEME
    assert consumed["row_sequence_sha256"] == last["resume"]["consumption"]["row_sequence_sha256"]
    assert last["resume"]["step"] == 6


def test_format_2_checkpoint_loads_and_resume_refuses_it(tmp_path):
    fixture = write_format2(tmp_path / "format2.pt")
    model, payload = load_checkpoint(fixture)
    assert payload["format_version"] == 2 and "resume" not in payload
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="format 2.*init_from"):
        run_train(out, resume_from=fixture)
    assert not out.exists()


def test_init_from_format_2_starts_a_fresh_run_from_its_weights(tmp_path):
    fixture = write_format2(tmp_path / "format2.pt")
    reference = load_checkpoint(fixture)[0].state_dict()
    config = reduced(steps=3, eval_every=3, milestone_steps=())
    result = run_train(tmp_path / "run", training=config, init_from=fixture)
    run = result.run
    assert run["status"] == "finished" and run["final_step"] == 3
    assert run["initialized_from"] == {
        "path": str(fixture),
        "sha256": file_sha256(fixture),
        "step": 4,
        "format_version": 2,
    }
    assert [r["step"] for r in jsonl(tmp_path / "run/metrics.jsonl")] == [1, 2, 3]
    assert "resumed_from" not in run and run["run_id"] != "format2-run"
    # The weights moved off the initialization, which the first update started from.
    final = weights(tmp_path / "run/checkpoint_last.pt")
    assert any(not torch.equal(final[key], reference[key]) for key in final)
    # Step 1 of a warm start sees the warm weights: its loss differs from a cold start's.
    run_train(tmp_path / "cold", training=config)
    assert (
        jsonl(tmp_path / "cold/metrics.jsonl")[0]["loss"]
        != jsonl(tmp_path / "run/metrics.jsonl")[0]["loss"]
    )


def test_init_from_a_format_3_checkpoint_resets_the_optimizer_and_step(tmp_path, full_run):
    config = reduced(steps=2, eval_every=2, milestone_steps=())
    result = run_train(tmp_path / "run", training=config, init_from=full_run / "checkpoint_best.pt")
    assert result.run["initialized_from"]["format_version"] == 3
    assert result.run["initialized_from"]["step"] in (3, 6, 9)
    assert [r["step"] for r in jsonl(tmp_path / "run/metrics.jsonl")] == [1, 2]
    _, payload = load_checkpoint(tmp_path / "run/checkpoint_last.pt")
    assert payload["resume"]["step"] == 2


def test_init_from_refuses_a_different_model_config(tmp_path):
    fixture = write_format2(tmp_path / "format2.pt")
    other = TINY.__class__.from_dict({**TINY.to_dict(), "d_model": 16, "d_ff": 32})
    with pytest.raises(ValueError, match="model config differs.*d_model"):
        run_train(tmp_path / "run", model_config=other, init_from=fixture)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode", ["resume_from", "init_from"])
def test_dimensional_checkpoints_are_refused(tmp_path, mode):
    fixture = write_dimensional(tmp_path / "dimensional.pt")
    with pytest.raises(ValueError, match="incompatible input representation"):
        run_train(tmp_path / "run", **{mode: fixture})
    assert not (tmp_path / "run").exists()


def test_resume_refuses_a_best_checkpoint(tmp_path, full_run):
    with pytest.raises(ValueError, match="'best' checkpoint without resume state"):
        run_train(tmp_path / "run", resume_from=full_run / "checkpoint_best.pt")
    assert not (tmp_path / "run").exists()


def test_resume_refuses_a_changed_training_config_naming_the_fields(tmp_path, full_run):
    config = reduced(steps=9, eval_every=3, milestone_steps=())
    changed = replace(config, lr=2e-3, batch_size=2, eval_samples=4)
    with pytest.raises(ValueError, match=r"training_config differs.*eval_samples.*lr") as raised:
        run_train(tmp_path / "run", training=changed, resume_from=full_run / "checkpoint_last.pt")
    assert "batch_size" not in str(raised.value)
    assert not (tmp_path / "run").exists()


def test_resume_refuses_a_changed_model_config(tmp_path, full_run):
    other = TINY.__class__.from_dict({**TINY.to_dict(), "n_heads": 4})
    with pytest.raises(ValueError, match="model config differs.*n_heads"):
        run_train(tmp_path / "run", model_config=other, resume_from=full_run / "checkpoint_last.pt")
    assert not (tmp_path / "run").exists()


def test_resume_refuses_a_changed_corpus(tmp_path, full_run):
    with pytest.raises(ValueError, match="corpora differ"):
        run_train(
            tmp_path / "run",
            corpus=make_synthetic_corpus(seed=1),
            resume_from=full_run / "checkpoint_last.pt",
        )
    assert not (tmp_path / "run").exists()


def test_resume_matches_generator_corpora_by_file_hash(tmp_path):
    """Generator corpora carry array-valued metadata, so they have no content SHA-256."""
    save_corpus = pytest.importorskip("pymc_generator").save_corpus

    from pymc_mfm.data.recipes import load_corpora, provenance
    from pymc_mfm.training._provenance import corpus_identity

    corpus = make_synthetic_corpus()
    corpus["identifiability"] = {"signal_metrics": np.zeros((2, 2))}
    path = tmp_path / "generator_like.npz"
    save_corpus(corpus, path)
    (loaded,) = load_corpora([path])
    assert corpus_identity(loaded.corpus)["content_sha256"] is None
    files = {"corpus": loaded.corpus, "corpus_provenance": [provenance(loaded)]}
    full = run_train(tmp_path / "full", **files)
    interrupted_run(tmp_path / "run", 3, **files)
    run_train(tmp_path / "run", resume_from=tmp_path / "run" / "checkpoint_last.pt", **files)
    assert_resumed_equals_full(full.run_dir, tmp_path / "run", after=3)


def test_resume_refuses_a_changed_array_under_the_same_file_provenance(tmp_path):
    """Without a content hash, the file hash alone must not hide changed arrays."""
    save_corpus = pytest.importorskip("pymc_generator").save_corpus

    from pymc_mfm.data.recipes import load_corpora, provenance

    corpus = make_synthetic_corpus()
    corpus["identifiability"] = {"signal_metrics": np.zeros((2, 2))}
    path = tmp_path / "generator_like.npz"
    save_corpus(corpus, path)
    (loaded,) = load_corpora([path])
    record = provenance(loaded)
    interrupted_run(tmp_path / "run", 3, corpus=loaded.corpus, corpus_provenance=[record])
    changed = dict(loaded.corpus, outcome_raw=loaded.corpus["outcome_raw"] * 2)
    with pytest.raises(ValueError, match="corpora differ"):
        run_train(
            tmp_path / "other",
            corpus=changed,
            corpus_provenance=[record],
            resume_from=tmp_path / "run" / "checkpoint_last.pt",
        )
    assert not (tmp_path / "other").exists()


def test_a_refused_same_directory_resume_leaves_the_directory_untouched(tmp_path):
    out = tmp_path / "run"
    config = interrupted_run(out, 3)
    before = snapshot(out)
    with pytest.raises(ValueError, match="training_config differs.*lr"):
        run_train(out, training=replace(config, lr=3e-3), resume_from=out / "checkpoint_last.pt")
    assert snapshot(out) == before


def test_resume_and_init_are_exclusive(tmp_path, full_run):
    path = full_run / "checkpoint_last.pt"
    with pytest.raises(ValueError, match="exclusive"):
        run_train(tmp_path / "run", resume_from=path, init_from=path)


# ---- accelerator RNG ----


def resume_block(device, accelerator_rng):
    return {"device": device, "accelerator_rng": accelerator_rng}


def test_cpu_capture_holds_no_accelerator_state():
    captured = capture_rng_state(torch.device("cpu"))
    assert captured["device"] == "cpu" and captured["accelerator_rng"] is None
    assert torch.equal(captured["torch_rng"], torch.random.get_rng_state())


@pytest.mark.parametrize(
    ("saved", "device", "message"),
    [
        (resume_block("cuda:0", [torch.zeros(1)]), "cpu", "written on cuda:0.*on cpu"),
        (resume_block("mps", torch.zeros(1)), "cuda:0", "written on mps.*on cuda:0"),
        (resume_block("cpu", None), "mps", "written on cpu.*on mps"),
        (resume_block("cuda:0", [torch.zeros(1)] * 2), "cuda:1", "written on cuda:0.*on cuda:1"),
    ],
    ids=["cuda-to-cpu", "mps-to-cuda", "cpu-to-mps", "cuda-index"],
)
def test_resume_refuses_a_checkpoint_from_another_device(saved, device, message):
    with pytest.raises(ValueError, match=message):
        check_rng_fits_device(saved, torch.device(device))


def test_resume_refuses_a_cuda_checkpoint_when_the_device_count_differs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    saved = resume_block("cuda:0", [torch.zeros(1), torch.zeros(1)])
    with pytest.raises(ValueError, match="RNG states of 2 CUDA devices, but 1 are visible"):
        check_rng_fits_device(saved, torch.device("cuda:0"))
    check_rng_fits_device(resume_block("cuda:0", [torch.zeros(1)]), torch.device("cuda:0"))


def test_restore_sets_only_the_saved_devices_accelerator_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: calls.append(("cuda", states))
    )
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state: calls.append(("mps", state)))
    cpu_rng = torch.random.get_rng_state()
    for device, state in (("cuda:0", [torch.ones(1)]), ("mps", torch.ones(1)), ("cpu", None)):
        restore_rng_state({"torch_rng": cpu_rng, **resume_block(device, state)})
    assert [kind for kind, _ in calls] == ["cuda", "mps"]


def test_a_resume_from_another_device_is_refused_before_anything_is_written(tmp_path):
    part = tmp_path / "part"
    interrupted_run(part, 3)
    payload = torch.load(part / "checkpoint_last.pt", weights_only=True)
    payload["resume"].update(device="cuda:0", accelerator_rng=[torch.zeros(1)])
    foreign = tmp_path / "foreign.pt"
    torch.save(payload, foreign)
    with pytest.raises(ValueError, match="written on cuda:0"):
        run_train(tmp_path / "run", resume_from=foreign)
    assert not (tmp_path / "run").exists()


# ---- stream, digest and seen rows ----


@pytest.fixture
def corpus():
    return make_synthetic_corpus()


def drawn(stream, n):
    return [stream.next_batch()[1] for _ in range(n)]


def test_stream_restore_continues_the_exact_sequence_across_epochs(corpus):
    reference = CorpusStream([corpus], batch_size=2, seed=3)
    expected = drawn(reference, 14)
    for cut in (0, 1, 4, 5, 9):  # includes the epoch boundary after batch 4
        first = CorpusStream([corpus], batch_size=2, seed=3)
        drawn(first, cut)
        state = first.resume_state()
        restored = CorpusStream([corpus], batch_size=2, seed=3)
        restored.restore(state)
        assert drawn(restored, 14 - cut) == expected[cut:]
        assert restored.resume_state() == reference.resume_state()


def test_stream_snapshot_is_small_and_validated(corpus):
    stream = CorpusStream([corpus], batch_size=2, seed=3)
    drawn(stream, 5)
    state = stream.resume_state()
    assert len(json.dumps(state, default=int)) < 1500
    for name, other in (("seed", 4), ("batch_size", 4), ("split", "all"), ("n_rows", 99)):
        with pytest.raises(ValueError, match=name):
            CorpusStream([corpus], batch_size=2, seed=3).restore({**state, name: other})
    wrong = CorpusStream([corpus], batch_size=2, seed=3)
    with pytest.raises(ValueError, match="buckets"):
        wrong.restore({**state, "cursor": {1: 0}})


def test_prefetching_stream_snapshots_the_consumed_batch_not_the_producers(corpus):
    stream = CorpusStream([corpus], batch_size=2, seed=3)
    source = PrefetchingStream(stream, depth=3)
    try:
        for _ in range(3):
            source.next_batch()
        snapshot_state = source.consumed_state
    finally:
        source.stop()
    plain = CorpusStream([corpus], batch_size=2, seed=3)
    drawn(plain, 3)
    assert snapshot_state == plain.resume_state()


def test_seen_rows_round_trip_keeps_count_and_ordered_prefix():
    seen = SeenRows([5, 4], prefix_limit=3)
    seen.add([(0, 4), (1, 0), (0, 4), (0, 1), (1, 3)])
    state = seen.state()
    assert state["bitmaps"][0].dtype == torch.uint8
    restored = SeenRows([5, 4], prefix_limit=3)
    restored.restore(state)
    assert len(restored) == len(seen) == 4
    assert restored.prefix(3) == seen.prefix(3) == [(0, 4), (1, 0), (0, 1)]
    restored.add([(0, 1), (0, 2)])
    assert len(restored) == 5
    with pytest.raises(ValueError, match="does not match"):
        SeenRows([5, 5], prefix_limit=3).restore(state)


def test_chained_row_sequence_matches_its_definition():
    rows_1, rows_2 = [[0, 1], [0, 2]], [[1, 3]]
    import hashlib

    d0 = hashlib.sha256(b"").hexdigest()
    d1 = hashlib.sha256(bytes.fromhex(d0) + b"[[0,1],[0,2]]").hexdigest()
    d2 = hashlib.sha256(bytes.fromhex(d1) + b"[[1,3]]").hexdigest()
    assert EMPTY_ROW_SEQUENCE == d0
    assert chain_row_sequence(chain_row_sequence(d0, rows_1), rows_2) == d2


# ---- MLflow ----


def test_same_directory_resume_continues_the_mlflow_run_and_new_directory_starts_a_tagged_one(
    tmp_path,
):
    MlflowClient = pytest.importorskip("mlflow").MlflowClient

    tracking = TrackingConfig(diagnostics=False)
    config = reduced(steps=9, eval_every=3, milestone_steps=())
    out = tmp_path / "run"
    interrupted_run(out, 6, training=config, tracking=tracking)
    first = json.loads((out / "run.json").read_text())["tracking"]
    assert first["run_id"]
    events_before = len(list((out / "tracking/events").glob("*.json")))

    result = run_train(
        out, training=config, tracking=tracking, resume_from=out / "checkpoint_last.pt"
    )
    resumed = result.run["tracking"]
    assert resumed["run_id"] == first["run_id"]
    assert resumed["delivery_status"] == "healthy"
    client = MlflowClient(tracking_uri=resumed["tracking_uri"])
    run = client.get_run(first["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["mfm.resumed"] == "true"
    assert run.data.tags["mfm.lifecycle_status"] == "finished"
    # Local event numbering continues instead of overwriting the earlier events.
    numbers = sorted(int(p.name[:7]) for p in (out / "tracking/events").glob("*.json"))
    assert len(numbers) > events_before and numbers == list(range(1, len(numbers) + 1))

    other = tmp_path / "other"
    forked = run_train(
        other, training=config, tracking=tracking, resume_from=out / "checkpoint_last.pt"
    )
    assert forked.run["tracking"]["run_id"] != first["run_id"]
    tags = (
        MlflowClient(tracking_uri=forked.run["tracking"]["tracking_uri"])
        .get_run(forked.run["tracking"]["run_id"])
        .data.tags
    )
    assert tags["mfm.resume_from_local_run_id"] == result.run["run_id"]
    assert tags["mfm.resume_from_run_id"] == first["run_id"]
    assert tags["mfm.resume_from_step"] == "9"


def test_resume_into_new_directory_from_a_run_whose_history_is_gone(tmp_path):
    config = reduced(steps=9, eval_every=3, milestone_steps=())
    interrupted_run(tmp_path / "part", 3, training=config)
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "checkpoint_last.pt").write_bytes((tmp_path / "part/checkpoint_last.pt").read_bytes())
    result = run_train(
        tmp_path / "resumed", training=config, resume_from=moved / "checkpoint_last.pt"
    )
    assert result.run["resumed_from"]["history_copied"] is False
    assert [r["step"] for r in jsonl(tmp_path / "resumed/metrics.jsonl")] == list(range(4, 10))
    assert np.isfinite(jsonl(tmp_path / "resumed/metrics.jsonl")[0]["loss"])
