"""``python -m pymc_mfm.training.launch``: bounded runs, limits, failure isolation and resume."""

import csv
import importlib.util
import json
import sys
import threading
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest import mock

import pytest
from conftest import make_synthetic_corpus
from test_resume import InterruptAfterLast

from pymc_mfm.data.recipes import load_corpora
from pymc_mfm.training import TrackingConfig, train
from pymc_mfm.training.config import PRESETS
from pymc_mfm.training.launch import main
from pymc_mfm.training.launch_process import read_progress, run_command
from pymc_mfm.training.launch_spec import parse_launch_file, train_command

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymc_generator") is None, reason="pymc_generator not installed"
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


def launch(tmp_path, runs, *arguments, root="root"):
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs))
    code = main([str(path), "--root", str(tmp_path / root), *NO_MEMORY_LIMIT, *arguments])
    return code, json.loads((tmp_path / root / "launch.json").read_text())


def result_rows(root):
    with (root / "results.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def mixed(corpus, tmp_path_factory):
    """Two good runs (replay and eval-set scoring), one timed out, one with an invalid value."""
    tmp_path = tmp_path_factory.mktemp("mixed")
    runs = [
        run_spec(corpus, "slow", training={"steps": 100_000, "eval_every": 100_000}, timeout=1),
        run_spec(corpus, "broken", training={"steps": 0}),
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
    assert "steps" in "\n".join(broken["stderr_tail"])
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


def test_max_runs_limits_how_many_runs_start(corpus, tmp_path):
    runs = [run_spec(corpus, "a"), run_spec(corpus, "b")]
    _, manifest = launch(tmp_path, runs, "--max-runs", "0")
    assert {record["status"] for record in manifest["runs"].values()} == {"skipped_max_runs"}


def test_run_is_skipped_when_host_memory_stays_below_the_minimum(corpus, tmp_path):
    runs = [run_spec(corpus, "big")]
    arguments = ["--min-free-host-gib", "1e9", "--memory-wait-seconds", "0"]
    _, manifest = launch(tmp_path, runs, *arguments)
    assert manifest["runs"]["big"]["status"] == "skipped_memory"


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


def test_concurrency_measurement_records_aggregate_rate_and_ratio_to_solo(corpus, tmp_path):
    runs = [run_spec(corpus, "probe", score=None)]
    code, manifest = launch(tmp_path, runs, "--measure-concurrency", "1,2")
    points = manifest["concurrency"]["points"]
    assert code == 0 and [point["concurrent"] for point in points] == [1, 2]
    assert points[0]["ratio_to_solo"] == 1.0
    assert points[1]["statuses"] == ["finished", "finished"]
    assert (
        points[1]["ratio_to_solo"]
        == points[1]["aggregate_worlds_per_second"] / (points[0]["aggregate_worlds_per_second"])
    )
    assert len(list((tmp_path / "root" / "concurrency").iterdir())) == 3


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


def test_command_past_its_timeout_is_terminated(tmp_path):
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    outcome = run_command(sleeper, tmp_path / "logs" / "sleep", 0.5, threading.Event())
    assert outcome.status == "timed_out" and outcome.seconds < 30


def test_progress_of_a_run_without_metrics_is_zero_steps(tmp_path):
    assert read_progress(tmp_path) == {"steps_done": 0, "worlds_per_second": None}


MODAL_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "modal_devloop.py"


def load_modal_script(modal_module):
    """Import ``benchmarks/modal_devloop.py`` with ``modal_module`` standing in for modal."""
    spec = spec_from_file_location("modal_devloop", MODAL_SCRIPT)
    script = module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"modal": modal_module}):
        spec.loader.exec_module(script)
    return script


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
