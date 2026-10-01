"""Opt-in held-out evaluation archives of training runs: what they hold, what they leave alone."""

import errno
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_synthetic_corpus
from test_archive import assert_same_result
from test_training import TINY, reduced

import pymc_mfm.diagnostics
from pymc_mfm.diagnostics import EvaluationArchive
from pymc_mfm.inference import from_pretrained
from pymc_mfm.inference import result as result_module
from pymc_mfm.training import (
    TrackingConfig,
    TrainingObserver,
    archive_identity,
    evaluate_rows,
    load_checkpoint,
    train,
)
from pymc_mfm.training import trainer as trainer_module
from pymc_mfm.training.archive import _model_identity
from pymc_mfm.training.checkpoint import file_sha256

#: Evaluates at steps 2 and 4 and promotes at both.
TRAINING = reduced(steps=4, eval_every=2, prefetch_batches=0)
RECIPE = "holdout_evaluation_archive"
REPORTS = "saved_conditional_evaluation_report"
#: metrics.jsonl fields that time the run rather than describe it.
TIMING = ("elapsed_seconds", "step_seconds", "batch_wait_seconds", "worlds_per_second")
NOT_GENERATED = [{"name": "terminal_checkpoint_diagnostics", "status": "not_generated"}]


@pytest.fixture(autouse=True)
def preserve_python_numpy_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class Checkpoints(TrainingObserver):
    """The SHA-256 of every checkpoint file as it was written, by ``(kind, step)``."""

    def __init__(self):
        self.sha256 = {}

    def on_checkpoint(self, record):
        self.sha256[(record["kind"], record["step"])] = record["sha256"]


def corpus():
    return make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=3)


def run_training(out_dir, corpora=None, *, tracking=None, **kwargs):
    return train(
        [corpus()] if corpora is None else corpora,
        out_dir=out_dir,
        model_config=TINY,
        training=TRAINING,
        device="cpu",
        tracking=TrackingConfig(enabled=False) if tracking is None else tracking,
        **kwargs,
    )


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def archive_entries(run):
    return [entry for entry in run["diagnostics"] if entry["recipe"] == RECIPE]


def digests(*arrays):
    """Each array's dtype, shape and SHA-256 of its bytes: equal only for bit-equal arrays."""
    return tuple(
        (array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        for array in map(np.ascontiguousarray, arrays)
    )


def replay(model, data, rows):
    """The holdout leg's captures, evaluated again post hoc as the trainer evaluates them."""
    captured = []
    evaluate_rows(
        model,
        [data],
        rows,
        n_samples=TRAINING.eval_samples,
        seed=TRAINING.eval_seed + 1,
        device="cpu",
        leg="holdout",
        result_sink=lambda *item: captured.append(item),
    )
    return captured


def assert_holds(archive, captured):
    """The archive's records equal ``captured`` bit for bit, in order."""
    for (result, *evidence), (expected, *expected_evidence) in zip(
        archive.records(), captured, strict=True
    ):
        assert_same_result(result, expected)
        for array, expected_array in zip(evidence, expected_evidence, strict=True):
            np.testing.assert_array_equal(array, expected_array)
            assert array.dtype == expected_array.dtype


def test_holdout_archives_leave_training_numerics_records_and_rng_unchanged(tmp_path):
    runs = {}
    for name, kwargs in (
        ("default", {}),
        ("terminal", {"holdout_archives": ("terminal",)}),
        ("both", {"holdout_archives": ("best", "terminal")}),
    ):
        random.seed(417)
        np.random.seed(718)
        result = run_training(tmp_path / name, **kwargs)
        states = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        runs[name] = result, states

    def records(result):
        evaluations = jsonl(result.run_dir / "evaluations.jsonl")
        captures = {
            (event["step"], leg): values.pop("result_capture")
            for event in evaluations
            for leg, values in event["legs"].items()
            if "result_capture" in values
        }
        metrics = [
            {key: value for key, value in record.items() if key not in TIMING}
            for record in jsonl(result.run_dir / "metrics.jsonl")
        ]
        tables = {
            path.name: path.read_bytes()
            for path in sorted((result.run_dir / "evaluation_tables").iterdir())
        }
        return evaluations, captures, metrics, tables

    default, default_states = runs["default"]
    evaluations, captures, metrics, tables = records(default)
    assert [event["step"] for event in evaluations] == [2, 4] and captures == {}
    assert not (default.run_dir / "archives").exists()
    assert "holdout_archives" not in default.run["evaluation_policy"]
    assert default.run["diagnostics"] == []
    assert from_pretrained(default.run_dir / "model").provenance["diagnostics"] == NOT_GENERATED
    n_worlds = len(evaluations[0]["legs"]["holdout"]["rows"])
    for name, archived in (("terminal", [4]), ("both", [2, 4])):
        result, states = runs[name]
        assert [(entry["path"], entry["status"]) for entry in archive_entries(result.run)] == [
            (f"archives/step_{step:09d}", "recorded") for step in archived
        ]
        other_evaluations, other_captures, other_metrics, other_tables = records(result)
        # Only the holdout leg of events that can be archived gains a capture record.
        assert other_captures == {
            (step, "holdout"): {"captured": 2 * n_worlds, "failures": []} for step in archived
        }
        assert other_evaluations == evaluations
        assert other_metrics == metrics
        assert other_tables == tables
        assert result.run["consumed"] == default.run["consumed"]
        for key, parameter in default.model.state_dict().items():
            torch.testing.assert_close(result.model.state_dict()[key], parameter, rtol=0, atol=0)
        assert states[0] == default_states[0]
        assert states[1][0] == default_states[1][0]
        np.testing.assert_array_equal(states[1][1], default_states[1][1])
        assert states[1][2:] == default_states[1][2:]
        torch.testing.assert_close(states[2], default_states[2], rtol=0, atol=0)


#: Per case: whether step 4's promotion is suppressed, the requested kinds, and for each
#: archived step its kinds and the kind of the checkpoint its identity names.
CASES = {
    "forced-best": (True, ("best",), {2: (["best"], "best")}),
    "forced-terminal": (True, ("terminal",), {4: (["terminal"], "last")}),
    "forced-both": (
        True,
        ("best", "terminal"),
        {2: (["best"], "best"), 4: (["terminal"], "last")},
    ),
    "both": (
        False,
        ("terminal", "best"),
        {2: (["best"], "best"), 4: (["best", "terminal"], "last")},
    ),
    "best": (False, ("best",), {2: (["best"], "best"), 4: (["best"], "last")}),
}


@pytest.mark.parametrize("case", CASES)
def test_archives_follow_the_selected_checkpoint_kinds(tmp_path, monkeypatch, case):
    forced, kinds, expected = CASES[case]
    if forced:
        select = trainer_module._select_checkpoint

        def promote_before_the_final_step_only(
            flat, undefined, best_value, best_step, step, *, head
        ):
            selection = select(flat, undefined, best_value, best_step, step, head=head)
            if step < TRAINING.steps:
                return selection
            return {
                **selection,
                "promoted": False,
                "best_value": best_value,
                "best_step": best_step,
            }

        monkeypatch.setattr(
            trainer_module, "_select_checkpoint", promote_before_the_final_step_only
        )
    data = corpus()
    checkpoints = Checkpoints()
    result = run_training(tmp_path / "run", [data], holdout_archives=kinds, observers=[checkpoints])
    out, run = result.run_dir, result.run
    assert result.status == "finished"
    events = {event["step"]: event for event in jsonl(out / "evaluations.jsonl")}
    assert [events[step]["selection"]["promoted"] for step in (2, 4)] == [True, not forced]
    assert run["evaluation_policy"]["holdout_archives"] == {
        "checkpoints": [kind for kind in ("best", "terminal") if kind in kinds],
        "leg": "holdout",
        "path": "archives/step_<N:09d>",
        "resampling": False,
    }
    entries = {entry["path"]: entry for entry in archive_entries(run)}
    names = [f"step_{step:09d}" for step in sorted(expected)]
    assert list(entries) == [f"archives/{name}" for name in names]
    assert sorted(path.name for path in (out / "archives").iterdir()) == names
    exported = from_pretrained(out / "model")

    for step, (step_kinds, bound) in expected.items():
        entry = entries[f"archives/step_{step:09d}"]
        archive = EvaluationArchive.load(out / entry["path"])
        identity = archive.identity
        rows = [tuple(row) for row in events[step]["legs"]["holdout"]["rows"]]
        evaluation = identity["evaluation"]
        assert [(world["corpus_index"], world["row"]) for world in evaluation["worlds"]] == rows
        assert evaluation["calls"] == [
            {"worlds": list(range(len(rows))), "seed": TRAINING.eval_seed + 1}
        ]
        assert (evaluation["leg"], evaluation["n_samples"], evaluation["heads"]) == (
            "holdout",
            TRAINING.eval_samples,
            ["joint", "conditional"],
        )
        for item, (record, *_) in zip(archive.entries, archive.records(), strict=True):
            assert item.world_id == record.provenance["world_id"]
            assert item.cell == ("corpus:0", int(data["cell_id"][item.row]))
        promoted = ("best", step) in checkpoints.sha256
        assert entry == {
            "name": f"step_{step}_holdout_archive",
            "sha256": file_sha256(archive.path / "manifest.json"),
            "checkpoint_sha256": checkpoints.sha256[("last", step)],
            "status": "recorded",
            "recipe": RECIPE,
            "path": f"archives/step_{step:09d}",
            "kinds": step_kinds,
            "archive_sha256": archive.archive_sha256,
            "state_sha256": identity["model"]["state_sha256"],
            **({"best_checkpoint_sha256": checkpoints.sha256[("best", step)]} if promoted else {}),
        }
        model = identity["model"]
        # A later promotion overwrites checkpoint_best.pt, so the binding is checked against
        # the digest of each file as it was written.
        assert model["checkpoint_sha256"] == checkpoints.sha256[(bound, step)]
        assert (model["source_step"], model["profile"]) == (step, run["profile"])

        if forced and bound == "best":
            best = out / "checkpoint_best.pt"
            assert model["checkpoint_sha256"] == entry["best_checkpoint_sha256"]
            assert model["checkpoint_sha256"] == file_sha256(best)
            reloaded, _ = load_checkpoint(best)
            assert model["state_sha256"] == _model_identity(reloaded)["state_sha256"]
            assert_holds(archive, replay(reloaded, data, rows))
        if step == TRAINING.steps:
            assert model["checkpoint_sha256"] == file_sha256(out / "checkpoint_last.pt")
            assert model["checkpoint_sha256"] == exported.provenance["checkpoint_sha256"]
            assert model["state_sha256"] == exported.provenance["state_sha256"]
            assert_holds(archive, replay(exported, data, rows))
            declared = archive_identity(
                exported,
                [data],
                rows,
                n_samples=TRAINING.eval_samples,
                seed=TRAINING.eval_seed + 1,
                leg="holdout",
                device="cpu",
            )
            assert identity["corpora"] == declared["corpora"]
            assert identity["evaluation"] == declared["evaluation"]

    final = entries.get(f"archives/step_{TRAINING.steps:09d}")
    assert exported.provenance["diagnostics"] == (
        NOT_GENERATED
        if final is None
        else [{key: final[key] for key in ("name", "sha256", "status", "recipe")}]
    )


@pytest.mark.filterwarnings("ignore:MLflow tracking is degraded:RuntimeWarning")
def test_archives_chain_with_diagnostics_without_changing_their_reports(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    delivered = []

    def build_report(result, out_dir, *, truth, graph_truth, observed, **records):
        manifest = {"status": "complete"}
        Path(out_dir).mkdir(parents=True)
        (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest))
        provenance = result.provenance
        key = (
            provenance["source_step"],
            provenance["data_split"],
            provenance["world_id"],
            result.window,
        )
        delivered.append((key, digests(result.components, truth, graph_truth, observed)))
        return manifest

    monkeypatch.setattr(pymc_mfm.diagnostics, "build_report", build_report)
    runs = {}
    for name, kwargs in (("plain", {}), ("archived", {"holdout_archives": ("terminal",)})):
        delivered.clear()
        result = run_training(tmp_path / name, tracking=TrackingConfig(), **kwargs)
        assert result.status == "finished"
        runs[name] = result, list(delivered)

    for result, reports in runs.values():
        captures = {
            (event["step"], leg, f"corpus:{corpus_index}/row:{row}", window)
            for event in jsonl(result.run_dir / "evaluations.jsonl")
            for leg in ("holdout", "seen")
            for corpus_index, row in event["legs"][leg]["rows"]
            for window in ("query", "context")
        }
        keys = [key for key, _ in reports]
        assert len(keys) == len(set(keys)) and set(keys) == captures

    def report_entries(result):
        return Counter(
            (entry["name"], entry["status"])
            for entry in result.run["diagnostics"]
            if entry["recipe"] == REPORTS
        )

    (plain, plain_reports), (archived, reports) = runs["plain"], runs["archived"]
    # Chained or not, every report receives the same draws and evidence.
    assert dict(reports) == dict(plain_reports)
    assert report_entries(archived) == report_entries(plain)
    assert archive_entries(plain.run) == []
    assert len(archived.run["diagnostics"]) == len(plain.run["diagnostics"]) + 1
    (entry,) = archive_entries(archived.run)
    assert (entry["status"], entry["kinds"]) == ("recorded", ["terminal"])
    final = {key: inputs for key, inputs in reports if key[:2] == (TRAINING.steps, "holdout")}
    archive = EvaluationArchive.load(archived.run_dir / entry["path"])
    # The archive holds exactly the final holdout leg's report inputs, in delivery order.
    stored = [
        ((TRAINING.steps, "holdout", item.world_id, item.window), digests(record.components, *rest))
        for item, (record, *rest) in zip(archive.entries, archive.records(), strict=True)
    ]
    assert stored == list(final.items())


def refuse_a_holdout_capture(monkeypatch):
    """The first held-out world's context draws cannot be converted: an incomplete archive."""
    capture = result_module.result_from_batch
    first = []

    def refuse_the_first_holdout_context(*args, **kwargs):
        provenance = kwargs["provenance"]
        if provenance["data_split"] == "holdout":
            if not first:
                first.append(provenance["world_id"])
            if kwargs["window"] == "context" and provenance["world_id"] == first[0]:
                raise ValueError("capture refused")
        return capture(*args, **kwargs)

    monkeypatch.setattr(result_module, "result_from_batch", refuse_the_first_holdout_context)
    return lambda: [(first[0], "context")]


def fill_the_disk(monkeypatch):
    """Writing the first truth.npz runs out of space; every capture itself succeeds."""
    savez = np.savez_compressed

    def full_disk(file, *args, **kwargs):
        if Path(file).name == "truth.npz":
            raise OSError(errno.ENOSPC, "No space left on device")
        return savez(file, *args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", full_disk)
    return lambda: []


@pytest.mark.parametrize(
    ("fail", "error_type"),
    [(refuse_a_holdout_capture, "ValueError"), (fill_the_disk, "OSError")],
    ids=["capture_refused", "disk_full"],
)
def test_a_failed_archive_is_recorded_and_training_finishes(
    tmp_path, monkeypatch, fail, error_type
):
    default = run_training(tmp_path / "default")
    capture_failures = fail(monkeypatch)
    result = run_training(tmp_path / "archived", holdout_archives=("terminal",))
    out = result.run_dir
    assert result.status == "finished"
    evaluations = jsonl(out / "evaluations.jsonl")
    failures = evaluations[-1]["legs"]["holdout"]["result_capture"]["failures"]
    assert [
        (f"corpus:{failure['corpus_index']}/row:{failure['row']}", failure["window"])
        for failure in failures
    ] == capture_failures()
    assert archive_entries(result.run) == [
        {
            "name": "step_4_holdout_archive",
            "checkpoint_sha256": file_sha256(out / "checkpoint_last.pt"),
            "status": "failed",
            "recipe": RECIPE,
            "path": "archives/step_000000004",
            "kinds": ["terminal"],
            "best_checkpoint_sha256": file_sha256(out / "checkpoint_best.pt"),
            "error_type": error_type,
        }
    ]
    # Neither the archive nor its hidden staging directory is left behind.
    assert list((out / "archives").iterdir()) == []
    assert [event["flat"] for event in evaluations] == [
        event["flat"] for event in jsonl(default.run_dir / "evaluations.jsonl")
    ]
    assert from_pretrained(out / "model").provenance["diagnostics"] == [
        {"name": "step_4_holdout_archive", "status": "failed", "recipe": RECIPE}
    ]


def test_an_archive_that_cannot_be_identified_fails_the_run_before_training(tmp_path):
    data = corpus()
    out = tmp_path / "archived"
    with pytest.raises(ValueError, match="same inputs"):
        run_training(out, [data, data], holdout_archives=("terminal",))
    run = json.loads((out / "run.json").read_text())
    assert (run["status"], run["error"]) == ("failed", "ValueError during initialization")
    assert run["checkpoints"] == [] and not list(out.glob("checkpoint_*.pt"))
    assert jsonl(out / "metrics.jsonl") == []
    assert not (out / "archives").exists()
    assert run_training(tmp_path / "plain", [data, data]).status == "finished"


def test_holdout_archive_kinds_are_checked_before_anything_is_written(tmp_path):
    for index, value in enumerate([("worst",), "best", ("best", "best"), True]):
        out = tmp_path / f"refused_{index}"
        message = (
            f"holdout_archives must be distinct names from ('best', 'terminal'), got {value!r}"
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            run_training(out, holdout_archives=value)
        assert not out.exists()
    result = run_training(tmp_path / "none", holdout_archives=None)
    assert result.status == "finished"
    assert not (result.run_dir / "archives").exists()
    assert "holdout_archives" not in result.run["evaluation_policy"]
    assert result.run["diagnostics"] == []


# save_pretrained expands "~" and exports model/ under HOME, so the tracker's local copy of
# the export reports degraded delivery; that predates archives and is not what is tested here.
@pytest.mark.filterwarnings("ignore:MLflow tracking is degraded:RuntimeWarning")
def test_archives_stay_in_a_run_directory_spelled_with_a_tilde(tmp_path, monkeypatch):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    result = run_training("~/run", holdout_archives=("terminal",))
    assert result.status == "finished"
    # train() keeps a literal "~": the run directory is ./~/run, and its archives belong there.
    run_dir = work / "~" / "run"
    assert json.loads((run_dir / "run.json").read_text())["run_id"] == result.run["run_id"]
    (entry,) = archive_entries(result.run)
    assert entry["status"] == "recorded"
    archive = EvaluationArchive.load(run_dir / entry["path"])
    assert entry["sha256"] == file_sha256(run_dir / entry["path"] / "manifest.json")
    assert entry["archive_sha256"] == archive.archive_sha256
    assert not (home / "run" / "archives").exists()
