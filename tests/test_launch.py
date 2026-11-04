"""``python -m pymc_mfm.training.launch``: bounded runs, limits, failure isolation and resume."""

import csv
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock

import pytest
from conftest import make_synthetic_corpus
from test_resume import InterruptAfterLast

from pymc_mfm.data.recipes import load_corpora
from pymc_mfm.training import TrackingConfig, launch_process, train
from pymc_mfm.training.config import PRESETS
from pymc_mfm.training.launch import Launcher, Limits, Manifest, main
from pymc_mfm.training.launch_process import read_progress, run_command
from pymc_mfm.training.launch_spec import (
    LaunchFileError,
    parse_launch_file,
    score_command,
    train_command,
)

#: Training fields that make a run finish in a few seconds on the tiny corpus.
FAST = {
    "steps": 4,
    "batch_size": 2,
    "warmup_steps": 2,
    "eval_every": 4,
    "eval_worlds": 2,
    "eval_samples": 3,
    "log_every": 1,
}
NO_MEMORY_LIMIT = ["--min-free-host-gib", "0"]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    pytest.importorskip("pymc_generator")
    from pymc_generator import save_corpus

    path = tmp_path_factory.mktemp("corpus") / "corpus.npz"
    save_corpus(make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9), path)
    return str(path)


def run_spec(corpus, name, *, training=None, timeout=120, score="replay"):
    return {
        "name": name,
        "preset": "smoke",
        "training": {**FAST, **(training or {})},
        "corpus": [corpus],
        "device": "cpu",
        "timeout_seconds": timeout,
        "score": {"mode": score} if score else None,
    }


def write_runs(tmp_path, runs):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs))
    return path


def launch(tmp_path, runs, *arguments, root="root"):
    path = write_runs(tmp_path, runs)
    code = main([str(path), "--root", str(tmp_path / root), *NO_MEMORY_LIMIT, *arguments])
    return code, json.loads((tmp_path / root / "launch.json").read_text())


def result_rows(root):
    with (root / "results.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def mixed(corpus, tmp_path_factory):
    """Two good runs (replay and eval-set scoring), one timed out, one with an unreadable corpus."""
    tmp_path = tmp_path_factory.mktemp("mixed")
    unreadable = tmp_path / "unreadable.npz"
    unreadable.write_text("not a corpus")
    runs = [
        run_spec(corpus, "slow", training={"steps": 100_000, "eval_every": 100_000}, timeout=1),
        run_spec(str(unreadable), "broken"),
        run_spec(corpus, "replayed"),
        run_spec(corpus, "frozen", score="eval-set"),
    ]
    code, manifest = launch(tmp_path, runs, "--max-parallel", "2")
    return code, manifest, tmp_path / "root"


def test_good_runs_finish_and_each_appends_one_result_row(mixed):
    _, manifest, root = mixed
    assert {name: manifest["runs"][name]["status"] for name in ("replayed", "frozen")} == {
        "replayed": "finished",
        "frozen": "finished",
    }
    rows = result_rows(root)
    assert sorted(row["run_name"] for row in rows) == ["frozen", "replayed"]
    assert sorted(row["mode"] for row in rows) == ["eval-set", "replay"]
    assert {manifest["runs"][name]["score_row"] for name in ("replayed", "frozen")} == {0, 1}


def test_manifest_records_what_a_finished_run_did(mixed):
    _, manifest, root = mixed
    record = manifest["runs"]["replayed"]
    assert record["returncode"] == 0 and record["wall_seconds"] > 0
    assert record["steps_done"] == 4 and record["worlds_per_second"] > 0
    assert "--preset smoke" in record["command"] and "--out-dir" in record["command"]
    assert (root / "runs" / "replayed" / "checkpoint_last.pt").is_file()


def test_timed_out_run_is_recorded_and_the_others_still_finish(mixed):
    _, manifest, root = mixed
    assert manifest["runs"]["slow"]["status"] == "timed_out"
    assert manifest["runs"]["replayed"]["status"] == "finished"
    assert len(result_rows(root)) == 2  # no score row for the timed-out run


def test_failing_run_keeps_its_stderr_and_does_not_stop_the_next(mixed):
    _, manifest, _ = mixed
    broken = manifest["runs"]["broken"]
    assert broken["status"] == "failed" and broken["returncode"] != 0
    assert broken["stderr_tail"]
    assert manifest["runs"]["frozen"]["status"] == "finished"


def test_exit_status_is_nonzero_when_a_run_did_not_finish(mixed):
    code, _, _ = mixed
    assert code == 1


def test_run_that_does_not_fit_the_remaining_budget_is_not_started(corpus, tmp_path):
    runs = [run_spec(corpus, "long", timeout=3600)]
    code, manifest = launch(tmp_path, runs, "--max-total-seconds", "60")
    assert code == 1
    assert manifest["runs"]["long"]["status"] == "skipped_budget"
    assert not (tmp_path / "root" / "runs" / "long").exists()


def test_training_timeout_uses_remaining_budget(corpus, tmp_path, monkeypatch):
    import pymc_mfm.training.launch as launch_module

    spec = parse_launch_file(write_runs(tmp_path, [run_spec(corpus, "a", timeout=120)]))[0]
    manifest = Manifest(tmp_path / "launch.json", {"runs": {"a": {"status": "pending"}}})
    launcher = Launcher(tmp_path, Limits(), manifest)
    monkeypatch.setattr(launcher, "remaining_seconds", lambda: 12.5)
    timeouts = []

    def fake_run(command, log, timeout, stop, on_start):
        timeouts.append(timeout)
        return launch_module.Outcome("finished", 0, 0.0)

    monkeypatch.setattr(launch_module, "run_command", fake_run)
    launcher.run_training(spec, tmp_path / "runs" / "a", "a")
    assert timeouts == [12.5]


def test_max_runs_limits_how_many_runs_start(corpus, tmp_path):
    runs = [run_spec(corpus, "a"), run_spec(corpus, "b")]
    _, manifest = launch(tmp_path, runs, "--max-runs", "0")
    assert {record["status"] for record in manifest["runs"].values()} == {"skipped_max_runs"}


def test_run_is_skipped_when_host_memory_stays_below_the_minimum(corpus, tmp_path):
    runs = [run_spec(corpus, "big")]
    arguments = ["--min-free-host-gib", "1e9", "--memory-wait-seconds", "0"]
    _, manifest = launch(tmp_path, runs, *arguments)
    assert manifest["runs"]["big"]["status"] == "skipped_memory"


def test_stop_during_memory_wait_is_interrupted(corpus, tmp_path, monkeypatch):
    import pymc_mfm.training.launch as launch_module

    spec = parse_launch_file(write_runs(tmp_path, [run_spec(corpus, "a")]))[0]
    manifest = Manifest(tmp_path / "launch.json", {"runs": {"a": {"status": "pending"}}})
    launcher = Launcher(tmp_path, Limits(min_free_host_gib=1, memory_wait_seconds=10), manifest)
    monkeypatch.setattr(launch_module, "host_free_gib", lambda: 0)
    timer = threading.Timer(0.1, launcher.stop.set)
    timer.start()
    try:
        launcher.run_one(spec, resume=False)
    finally:
        timer.join()
    assert manifest.status("a") == "interrupted"


def test_resume_continues_an_interrupted_run_in_its_own_directory(corpus, tmp_path):
    root = tmp_path / "root"
    training = replace(PRESETS["smoke"], **{**FAST, "steps": 6, "eval_every": 3})
    with pytest.raises(KeyboardInterrupt):
        train(
            [item.corpus for item in load_corpora([corpus])],
            out_dir=root / "runs" / "a",
            training=training,
            device="cpu",
            tracking=TrackingConfig(enabled=False),
            observers=[InterruptAfterLast(3)],
            run_name="a",
        )
    root.joinpath("launch.json").write_text(json.dumps({"runs": {"a": {"status": "timed_out"}}}))
    spec = run_spec(corpus, "a", training={"steps": 6, "eval_every": 3})

    code, manifest = launch(tmp_path, [spec], "--resume")

    assert code == 0
    assert "--resume-from" in manifest["runs"]["a"]["command"]
    assert [path.name for path in (root / "runs").iterdir()] == ["a"]
    steps = [
        json.loads(line)["step"]
        for line in (root / "runs/a/metrics.jsonl").read_text().splitlines()
    ]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert [row["run_name"] for row in result_rows(root)] == ["a"]


def test_resume_does_not_rerun_a_finished_run(corpus, tmp_path):
    runs = [run_spec(corpus, "a", score=None)]
    launch(tmp_path, runs)
    before = (tmp_path / "root" / "runs" / "a" / "run.json").stat().st_mtime_ns
    code, _ = launch(tmp_path, runs, "--resume")
    assert code == 0
    assert (tmp_path / "root" / "runs" / "a" / "run.json").stat().st_mtime_ns == before


def test_resume_without_checkpoint_moves_old_attempt(corpus, tmp_path):
    root = tmp_path / "root"
    run_dir = root / "runs" / "a"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("old attempt")
    root.joinpath("launch.json").write_text(json.dumps({"runs": {"a": {"status": "timed_out"}}}))

    code, manifest = launch(tmp_path, [run_spec(corpus, "a", score=None)], "--resume")

    record = manifest["runs"]["a"]
    assert code == 0 and record["status"] == "finished"
    assert record["restarted_from_scratch"] is True
    assert record["previous_run_dir"] == str(root / "runs" / "a.attempt-1")
    assert (root / "runs" / "a.attempt-1" / "run.json").read_text() == "old attempt"
    assert (run_dir / "checkpoint_last.pt").is_file()


def test_concurrency_runs_every_copy_in_its_own_directory(corpus, tmp_path):
    runs = [run_spec(corpus, "probe", score=None)]
    code, manifest = launch(tmp_path, runs, "--measure-concurrency", "1,2")
    points = manifest["concurrency"]["points"]
    assert code == 0 and [point["concurrent"] for point in points] == [1, 2]
    assert points[1]["statuses"] == ["finished", "finished"]
    assert len(list((tmp_path / "root" / "concurrency").iterdir())) == 3


def test_concurrency_ratio_is_the_aggregate_rate_over_the_solo_rate(corpus, tmp_path, monkeypatch):
    rates = {1: [10.0], 2: [8.0, 7.0], 4: [4.0, 4.0, 4.0, 4.0]}
    monkeypatch.setattr(
        Launcher, "run_copies", lambda self, spec, count: (rates[count], ["finished"] * count)
    )
    code, manifest = launch(tmp_path, [run_spec(corpus, "p")], "--measure-concurrency", "1,2,4")
    points = manifest["concurrency"]["points"]
    assert code == 0
    assert [point["aggregate_worlds_per_second"] for point in points] == [10.0, 15.0, 16.0]
    assert [point["ratio_to_solo"] for point in points] == [1.0, 1.5, 1.6]


def test_concurrency_copies_that_do_not_fit_the_budget_are_not_started(corpus, tmp_path):
    runs = [run_spec(corpus, "probe", timeout=3600, score=None)]
    arguments = ["--measure-concurrency", "1,2", "--max-total-seconds", "60"]
    code, manifest = launch(tmp_path, runs, *arguments)
    assert code == 1
    assert manifest["concurrency"]["points"][1]["statuses"] == ["skipped_budget"] * 2


def test_score_is_not_started_when_the_budget_is_spent(corpus, tmp_path):
    spec = parse_launch_file(write_runs(tmp_path, [run_spec(corpus, "a")]))[0]
    manifest = Manifest(tmp_path / "launch.json", {"runs": {"a": {"status": "trained"}}})
    launcher = Launcher(tmp_path, Limits(max_total_seconds=30), manifest)
    launcher.score(spec, tmp_path / "runs" / "a")
    record = manifest.data["runs"]["a"]
    assert record["status"] == "score_failed" and "no time left" in record["stderr_tail"][0]
    assert not (tmp_path / "results.csv").exists()


def test_a_queued_score_takes_its_timeout_after_the_previous_score(corpus, tmp_path, monkeypatch):
    """A score that waits for the lock gets the budget left when it starts, not when it queued."""
    import threading

    import pymc_mfm.training.launch as launch_module

    specs = parse_launch_file(write_runs(tmp_path, [run_spec(corpus, "a"), run_spec(corpus, "b")]))
    manifest = Manifest(
        tmp_path / "launch.json", {"runs": {"a": {"status": "trained"}, "b": {"status": "trained"}}}
    )
    launcher = Launcher(tmp_path, Limits(max_total_seconds=1000), manifest)
    clock = {"left": 500.0}
    monkeypatch.setattr(launcher, "remaining_seconds", lambda: clock["left"])
    first_running, release_first = threading.Event(), threading.Event()
    timeouts = {}

    def fake_run(command, log, timeout, stop, on_start):
        name = "a" if "a" not in timeouts else "b"
        timeouts[name] = timeout
        if name == "a":
            first_running.set()
            release_first.wait(10)
        return launch_module.Outcome(status="failed", returncode=1, seconds=0.0)

    monkeypatch.setattr(launch_module, "run_command", fake_run)
    first = threading.Thread(target=launcher.score, args=(specs[0], tmp_path / "runs" / "a"))
    first.start()
    assert first_running.wait(10)
    second = threading.Thread(target=launcher.score, args=(specs[1], tmp_path / "runs" / "b"))
    second.start()  # blocks on the score lock while the first score runs
    second.join(0.2)
    assert second.is_alive() and "b" not in timeouts
    clock["left"] = 120.0  # the first score used the budget down to 120 s
    release_first.set()
    first.join(10)
    second.join(10)
    assert timeouts == {"a": min(Limits().score_timeout_seconds, 500.0), "b": 120.0}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda runs: runs.append(dict(runs[0])), "used more than once"),
        (lambda runs: runs[0].update(preset="huge"), "unknown preset"),
        (lambda runs: runs[0]["training"].update(learning="x"), "unknown training fields"),
        (lambda runs: runs[0].update(corpus=["/missing.npz"]), "does not exist"),
        (lambda runs: runs[0].pop("timeout_seconds"), "timeout_seconds"),
        (lambda runs: runs[0].update(timeout_seconds=0), "timeout_seconds"),
        (lambda runs: runs[0].update(name="../escape"), "name must match"),
        (lambda runs: runs[0].update(score={"mode": "other"}), "score.mode"),
        (lambda runs: runs[0].update(score={"mode": "replay", "draws": 8}), "eval-set mode only"),
        (lambda runs: runs[0].update(score={"mode": "replay", "seed": 1}), "eval-set mode only"),
        (lambda runs: runs[0].update(score={"mode": "eval-set", "draws": 0}), "positive integer"),
        (lambda runs: runs[0].update(score={"mode": "eval-set", "seed": 1.5}), "positive integer"),
    ],
)
def test_invalid_launch_file_is_refused_before_anything_starts(
    corpus, tmp_path, capsys, change, message
):
    runs = [run_spec(corpus, "a")]
    change(runs)
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs))
    with pytest.raises(SystemExit) as raised:
        main([str(path), "--root", str(tmp_path / "root")])
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "root").exists()


def test_invalid_training_value_in_a_later_run_starts_nothing(corpus, tmp_path, capsys):
    runs = [run_spec(corpus, "good"), run_spec(corpus, "bad", training={"steps": 0})]
    path = write_runs(tmp_path, runs)
    with pytest.raises(SystemExit) as raised:
        main([str(path), "--root", str(tmp_path / "root"), *NO_MEMORY_LIMIT])
    assert raised.value.code == 2
    assert "run 'bad'" in capsys.readouterr().err
    assert not (tmp_path / "root").exists()


def test_milestones_beyond_the_steps_of_a_preset_run_are_dropped_like_the_cli_does(
    corpus, tmp_path
):
    runs = [run_spec(corpus, "a", training={"steps": 4})]
    runs[0]["preset"] = "canonical"  # its milestones are 25,000 and 50,000
    parse_launch_file(write_runs(tmp_path, runs))
    runs[0]["training"]["milestone_steps"] = [25_000]
    with pytest.raises(LaunchFileError, match="milestone_steps"):
        parse_launch_file(write_runs(tmp_path, runs))


def test_existing_manifest_needs_resume(corpus, tmp_path, capsys):
    runs = [run_spec(corpus, "a")]
    launch(tmp_path, runs, "--max-runs", "0")
    with pytest.raises(SystemExit):
        launch(tmp_path, runs)
    assert "use --resume" in capsys.readouterr().err


def test_train_command_passes_training_fields_as_cli_flags(corpus, tmp_path):
    spec = run_spec(corpus, "a", training={"milestone_steps": [], "lr": 0.5})
    (tmp_path / "runs.json").write_text(json.dumps([spec]))
    parsed = parse_launch_file(tmp_path / "runs.json")[0]
    command = train_command(parsed, tmp_path / "out", resume=True)
    assert command[command.index("--milestone-steps") + 1] == ""
    assert command[command.index("--lr") + 1] == "0.5"
    assert command[command.index("--resume-from") + 1] == str(tmp_path / "out/checkpoint_last.pt")


def test_launch_fields_match_training_cli(corpus, tmp_path):
    from pymc_mfm.training.__main__ import build_parser
    from pymc_mfm.training.config import training_cli_fields

    parser = build_parser()
    flags = {action.dest for action in parser._actions}
    assert training_cli_fields() <= flags
    runs = [run_spec(corpus, "a", training={"compile": True})]
    with pytest.raises(LaunchFileError, match="unknown training fields"):
        parse_launch_file(write_runs(tmp_path, runs))


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0])
def test_launch_rejects_non_finite_or_non_positive_timeout(corpus, tmp_path, timeout):
    with pytest.raises(LaunchFileError, match="finite positive"):
        parse_launch_file(write_runs(tmp_path, [run_spec(corpus, "a", timeout=timeout)]))


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-total-seconds", "nan"),
        ("--memory-wait-seconds", "inf"),
        ("--score-timeout-seconds", "-1"),
    ],
)
def test_launcher_rejects_invalid_durations(corpus, tmp_path, flag, value):
    path = write_runs(tmp_path, [run_spec(corpus, "a")])
    with pytest.raises(SystemExit) as raised:
        main([str(path), "--root", str(tmp_path / "root"), flag, value])
    assert raised.value.code == 2
    assert not (tmp_path / "root").exists()


def test_command_past_its_timeout_is_terminated(tmp_path):
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    outcome = run_command(sleeper, tmp_path / "logs" / "sleep", 0.5, threading.Event())
    assert outcome.status == "timed_out" and outcome.seconds < 30


def test_callback_failure_ends_child_group(tmp_path):
    pids = []

    def fail(pid):
        pids.append(pid)
        raise OSError(errno.ENOSPC, "disk full")

    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    with pytest.raises(OSError, match="disk full"):
        run_command(sleeper, tmp_path / "logs" / "failed", 30, threading.Event(), fail)
    assert not process_alive(pids[0])


def test_progress_of_a_run_without_metrics_is_zero_steps(tmp_path):
    assert read_progress(tmp_path) == {"steps_done": 0, "worlds_per_second": None}


MODAL_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "modal_devloop.py"


def load_modal_script(modal_module):
    """Import ``benchmarks/modal_devloop.py`` with ``modal_module`` standing in for modal."""
    spec = spec_from_file_location("modal_devloop", MODAL_SCRIPT)
    script = module_from_spec(spec)
    with (
        mock.patch.dict(sys.modules, {"modal": modal_module}),
        mock.patch.dict(os.environ, {"DEVLOOP_IMPORT_ONLY": "1"}),
    ):
        spec.loader.exec_module(script)
    return script


@pytest.mark.parametrize(
    "missing", ["MODAL_ENVIRONMENT", "DEVLOOP_REVISION", "DEVLOOP_INPUTS"], ids=str.lower
)
def test_a_modal_run_without_environment_or_inputs_stops_at_import(missing, monkeypatch):
    """`modal run` imports the file first: nothing can start, whichever function is named."""
    stub = mock.MagicMock()
    stub.is_local.return_value = True
    stub.config.config.get.return_value = None
    settings = {"MODAL_ENVIRONMENT": "any-env", "DEVLOOP_REVISION": "abc", "DEVLOOP_INPUTS": "x"}
    monkeypatch.delenv("DEVLOOP_IMPORT_ONLY", raising=False)
    for name, value in settings.items():
        if name == missing:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    spec = spec_from_file_location("modal_devloop_guard", MODAL_SCRIPT)
    with mock.patch.dict(sys.modules, {"modal": stub}), pytest.raises(Exception) as raised:
        spec.loader.exec_module(module_from_spec(spec))
    assert type(raised.value).__name__ == "RevisionError"
    stub.App.assert_not_called()  # no app, so no function can be registered or started


@pytest.fixture(scope="module")
def modal_script():
    return load_modal_script(mock.MagicMock())


def test_modal_script_imports_with_the_real_modal_when_it_is_installed():
    pytest.importorskip("modal")
    assert load_modal_script(sys.modules["modal"]).GPU == "L4"


def test_estimated_maximum_cost_is_timeout_times_the_hourly_rates(modal_script):
    hourly = 0.80 + 4 * 0.0473 + 16 * 0.008
    assert modal_script.estimate_max_cost_usd(3600) == pytest.approx(hourly)
    assert modal_script.estimate_max_cost_usd(1800) == pytest.approx(hourly / 2)


def test_container_timeout_is_capped_at_sixty_minutes(modal_script):
    assert modal_script.container_timeout_seconds(30) == 1800
    with pytest.raises(ValueError, match="max-minutes"):
        modal_script.container_timeout_seconds(61)
    assert modal_script.launcher_budget_seconds(1800) < 1800


def test_the_modal_environment_has_no_default(modal_script):
    assert modal_script.require_environment("any-env") == "any-env"
    for missing in (None, ""):
        with pytest.raises(modal_script.RevisionError, match="modal run -e"):
            modal_script.require_environment(missing)


def test_unpushed_or_dirty_revision_is_refused(modal_script):
    modal_script.check_revision("abc", remote_branches=["origin/main"], clone_status="")
    with pytest.raises(modal_script.RevisionError, match="push it first"):
        modal_script.check_revision("abc", remote_branches=[], clone_status="")
    with pytest.raises(modal_script.RevisionError, match="not clean"):
        modal_script.check_revision("abc", remote_branches=["origin/main"], clone_status="?? a.py")


def test_archive_drops_large_checkpoints_first_and_names_them(modal_script):
    mib = 2**20
    files = {"r/a.pt": 60 * mib, "r/b.pt": 40 * mib, "r/c.pt": 30 * mib, "r/m.jsonl": 90 * mib}
    kept, dropped = modal_script.select_archive_files(
        files, max_checkpoint_mib=50, max_archive_mib=200
    )
    assert dropped == ["r/a.pt"] and kept == ["r/b.pt", "r/c.pt", "r/m.jsonl"]
    kept, dropped = modal_script.select_archive_files(
        files, max_checkpoint_mib=50, max_archive_mib=125
    )
    assert dropped == ["r/a.pt", "r/b.pt"] and kept == ["r/c.pt", "r/m.jsonl"]


def test_archive_over_the_cap_without_checkpoints_is_refused(modal_script):
    files = {"r/m.jsonl": 201 * 2**20}
    with pytest.raises(ValueError, match="over 200 MiB"):
        modal_script.select_archive_files(files, max_checkpoint_mib=50)


def test_only_allowed_benchmarks_can_run_on_modal(modal_script):
    modal_script.check_bench_script("resume_acceptance.py")
    with pytest.raises(ValueError, match="not an allowed benchmark"):
        modal_script.check_bench_script("../etc/passwd")


def test_bench_timeout_ends_descendant_and_keeps_output(modal_script, tmp_path, monkeypatch):
    monkeypatch.setattr(modal_script, "REMOTE_SRC", str(tmp_path))
    pid_file = tmp_path / "descendant.pid"
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    parent = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); "
        "print('step 1 done',flush=True); time.sleep(60)"
    )
    stdout, exit_code = modal_script.run_benchmark([sys.executable, "-c", parent], 1)
    assert exit_code == -1
    assert stdout.startswith("step 1 done\n") and stdout.endswith("timed out\n")
    assert not process_alive(int(pid_file.read_text()))


def test_bench_that_finishes_returns_stdout_then_stderr_and_its_exit_code(
    modal_script, tmp_path, monkeypatch
):
    monkeypatch.setattr(modal_script, "REMOTE_SRC", str(tmp_path))
    command = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('err',file=sys.stderr); sys.exit(3)",
    ]
    assert modal_script.run_benchmark(command, 5) == ("out\nerr\n", 3)


def test_eval_set_seed_and_draws_are_passed_to_score(corpus, tmp_path):
    runs = [run_spec(corpus, "a")]
    runs[0]["score"] = {"mode": "eval-set", "seed": 7, "draws": 5}
    spec = parse_launch_file(write_runs(tmp_path, runs))[0]
    command = score_command(spec, tmp_path / "out", tmp_path / "results.csv")
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--draws") + 1] == "5"
    assert spec.score["corpus"] == (corpus,)


def process_alive(pid):
    """Whether ``pid`` runs; a zombie is not counted."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_for(condition, seconds=120):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.1)


#: Starts a descendant that ignores SIGTERM and records its pid, then runs as `parent`.
STUBBORN_DESCENDANT = """
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
{parent}
"""


def run_with_stubborn_descendant(tmp_path, parent, timeout):
    pid_file = tmp_path / "descendant.pid"
    script = STUBBORN_DESCENDANT.format(parent=parent)
    command = [sys.executable, "-c", script, str(pid_file)]
    outcome = run_command(command, tmp_path / "logs" / "x", timeout, threading.Event())
    return outcome, int(pid_file.read_text())


def test_descendant_that_ignores_sigterm_is_killed_when_the_command_times_out(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(launch_process, "TERMINATE_GRACE_SECONDS", 0.5)
    outcome, descendant = run_with_stubborn_descendant(tmp_path, "time.sleep(60)", timeout=2)
    assert outcome.status == "timed_out"
    assert not process_alive(descendant)


def test_descendant_left_behind_by_a_finished_command_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_process, "TERMINATE_GRACE_SECONDS", 0.5)
    outcome, descendant = run_with_stubborn_descendant(tmp_path, "time.sleep(1)", timeout=30)
    assert outcome.status == "finished"
    assert not process_alive(descendant)


def test_sigterm_ends_every_child_and_marks_the_runs_interrupted(corpus, tmp_path):
    """One run trains for ever; the other finishes training and then scores for ever."""
    runs = [
        run_spec(corpus, "training", training={"steps": 100_000, "eval_every": 100_000}),
        run_spec(corpus, "scoring", score="eval-set"),
    ]
    runs[1]["score"]["draws"] = 100_000
    root = tmp_path / "root"
    launcher = subprocess.Popen(
        [sys.executable, "-m", "pymc_mfm.training.launch", str(write_runs(tmp_path, runs))]
        + ["--root", str(root), "--max-parallel", "2", *NO_MEMORY_LIMIT]
    )

    def recorded():
        path = root / "launch.json"
        return json.loads(path.read_text())["runs"] if path.is_file() else {}

    wait_for(
        lambda: (
            "pid" in recorded().get("training", {}) and "score_pid" in recorded().get("scoring", {})
        )
    )
    pids = [recorded()["training"]["pid"], recorded()["scoring"]["score_pid"]]
    assert all(process_alive(pid) for pid in pids)

    launcher.send_signal(signal.SIGTERM)

    assert launcher.wait(timeout=60) == 1
    assert not any(process_alive(pid) for pid in pids)
    statuses = {name: record["status"] for name, record in recorded().items()}
    assert statuses == {"training": "interrupted", "scoring": "score_interrupted"}
