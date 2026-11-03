"""``python -m pymc_mfm.training.launch``: run a bounded list of training configs and score them.

The launch file (see :mod:`pymc_mfm.training.launch_spec`) lists runs. Each run is a
subprocess of ``python -m pymc_mfm.training`` writing to ``<root>/runs/<name>``; a finished
run is then scored by ``python -m pymc_mfm.training score`` into ``<root>/results.csv``. A
failed, timed-out or skipped run is recorded and the others continue. The exit status is
nonzero unless every run finished (and was scored, when asked). ``<root>/launch.json``
records the state and child process id of every run and is rewritten after each change.

SIGINT and SIGTERM stop the launcher cleanly: every child process group is terminated, then
killed, the unfinished runs are marked ``interrupted`` and the manifest is written.

``--resume`` re-launches the runs of an existing root that did not finish, continuing a
run from its ``checkpoint_last.pt`` in the same directory when that file exists.
``--measure-concurrency 1,2,4`` instead runs the first run of the file N times at once
for each N and records the aggregate worlds per second and its ratio to the solo run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from pymc_mfm.training.launch_process import (
    gpu_free_gib,
    host_free_gib,
    read_progress,
    run_command,
    stderr_tail,
    uses_cuda,
)
from pymc_mfm.training.launch_spec import (
    LaunchFileError,
    RunSpec,
    parse_launch_file,
    score_command,
    train_command,
)
from pymc_mfm.training.resume import LAST_CHECKPOINT

MANIFEST = "launch.json"
RESULTS = "results.csv"
#: Run statuses whose training finished and whose score is still to do.
TRAINED = ("trained", "score_failed", "score_interrupted")
_MEMORY_POLL_SECONDS = 2.0
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@dataclass
class Limits:
    max_parallel: int = 1
    max_runs: int | None = None
    max_total_seconds: float | None = None
    min_free_host_gib: float = 4.0
    min_free_gpu_gib: float = 2.0
    memory_wait_seconds: float = 60.0
    score_timeout_seconds: float = 900.0


class Manifest:
    """``launch.json``: a dict rewritten atomically after each change, safe across threads."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data
        self._lock = threading.Lock()

    def save(self, **changes: Any) -> None:
        with self._lock:
            self.data.update(changes)
            partial = self.path.with_name(self.path.name + ".partial")
            partial.write_text(json.dumps(self.data, indent=2) + "\n")
            os.replace(partial, self.path)

    def update_run(self, name: str, **changes: Any) -> None:
        with self._lock:
            self.data["runs"][name].update(changes)
        self.save()

    def status(self, name: str) -> str:
        return self.data["runs"][name]["status"]


@dataclass
class Launcher:
    root: Path
    limits: Limits
    manifest: Manifest
    stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self._admission_lock = threading.Lock()
        self._score_lock = threading.Lock()  # one writer at a time for results.csv
        self._began = time.monotonic()
        self._started = 0

    def run_all(self, specs: Sequence[RunSpec], *, resume: bool) -> int:
        """Run every spec that has not finished; return the exit status."""
        runs = {} if not resume else self.manifest.data["runs"]
        for spec in specs:
            runs.setdefault(spec.name, {"status": "pending"})
        self.manifest.save(runs=runs)
        with ThreadPoolExecutor(self.limits.max_parallel) as pool:
            for future in [pool.submit(self.run_one, spec, resume) for spec in specs]:
                future.result()
        finished = all(record["status"] == "finished" for record in runs.values())
        return 0 if finished else 1

    def run_one(self, spec: RunSpec, resume: bool) -> None:
        """Train and score one run, recording every state change."""
        name, run_dir = spec.name, self.root / "runs" / spec.name
        if self.manifest.status(name) == "finished":
            return
        if self.manifest.status(name) not in TRAINED:
            skipped = self.admit(spec)
            if skipped:
                self.manifest.update_run(name, status=skipped)
                return
            self.train(spec, run_dir, resume=resume and (run_dir / LAST_CHECKPOINT).is_file())
        if self.manifest.status(name) in TRAINED:
            if spec.score:
                self.score(spec, run_dir)
            else:
                self.manifest.update_run(name, status="finished")

    def admit(self, spec: RunSpec) -> str | None:
        """Reserve a start for ``spec``; return the ``skipped_*`` status that refuses it, if any."""
        with self._admission_lock:
            limits = self.limits
            if self.stop.is_set():
                return "interrupted"
            if limits.max_runs is not None and self._started >= limits.max_runs:
                return "skipped_max_runs"
            if limits.max_total_seconds is not None:
                remaining = limits.max_total_seconds - (time.monotonic() - self._began)
                if spec.timeout_seconds > remaining:
                    return "skipped_budget"
            if not self.wait_for_memory(spec):
                return "skipped_memory"
            self._started += 1
            return None

    def wait_for_memory(self, spec: RunSpec) -> bool:
        """Wait up to ``memory_wait_seconds`` for the free memory a start needs."""
        deadline = time.monotonic() + self.limits.memory_wait_seconds
        while True:
            enough = host_free_gib() >= self.limits.min_free_host_gib
            if enough and uses_cuda(spec.device):
                enough = gpu_free_gib() >= self.limits.min_free_gpu_gib
            if enough:
                return True
            if time.monotonic() >= deadline or self.stop.wait(_MEMORY_POLL_SECONDS):
                return False

    def train(self, spec: RunSpec, run_dir: Path, *, resume: bool) -> None:
        command = train_command(spec, run_dir, resume=resume)
        self.manifest.update_run(spec.name, status="running", command=shlex.join(command))
        log = self.root / "logs" / f"{spec.name}.train"
        record_pid = self.record_pid(spec.name, "pid")
        outcome = run_command(command, log, spec.timeout_seconds, self.stop, record_pid)
        status = "trained" if outcome.status == "finished" else outcome.status
        self.manifest.update_run(
            spec.name,
            status=status,
            returncode=outcome.returncode,
            wall_seconds=outcome.seconds,
            stderr_tail=stderr_tail(log.with_suffix(".stderr")),
            **read_progress(run_dir),
        )

    def score(self, spec: RunSpec, run_dir: Path) -> None:
        results = self.root / RESULTS
        command = score_command(spec, run_dir, results)
        log = self.root / "logs" / f"{spec.name}.score"
        with self._score_lock:
            outcome = run_command(
                command,
                log,
                self.limits.score_timeout_seconds,
                self.stop,
                self.record_pid(spec.name, "score_pid"),
            )
            row = count_rows(results) - 1 if outcome.status == "finished" else None
        if outcome.status == "finished":
            self.manifest.update_run(spec.name, status="finished", score_row=row)
        else:
            failed = "score_interrupted" if outcome.status == "interrupted" else "score_failed"
            self.manifest.update_run(
                spec.name,
                status=failed,
                score_command=shlex.join(command),
                stderr_tail=stderr_tail(log.with_suffix(".stderr")),
            )

    def record_pid(self, name: str, key: str):
        """Callback that stores a child's process id in the manifest as soon as it starts."""
        return lambda pid: self.manifest.update_run(name, **{key: pid})

    def measure_concurrency(self, spec: RunSpec, points: Sequence[int]) -> int:
        """Run ``spec`` N times at once for each N; record aggregate worlds/s and the ratio."""
        measured: list[dict[str, Any]] = []
        self.manifest.save(runs={}, concurrency={"run": spec.name, "points": measured})
        solo = None
        for count in points:
            if self.stop.is_set():
                break
            rates, statuses = self.run_copies(spec, count)
            aggregate = sum(rates)
            if count == 1:
                solo = aggregate
            point = {
                "concurrent": count,
                "statuses": statuses,
                "aggregate_worlds_per_second": aggregate,
                "ratio_to_solo": aggregate / solo if solo else None,
            }
            measured.append(point)
            self.manifest.save()
        ok = all(status == "finished" for point in measured for status in point["statuses"])
        return 0 if ok and len(measured) == len(points) else 1

    def run_copies(self, spec: RunSpec, count: int) -> tuple[list[float], list[str]]:
        """Train ``count`` copies of ``spec`` at once in distinct directories, same seed."""

        def train_copy(index: int) -> tuple[str, float]:
            if not self.wait_for_memory(spec):
                return "skipped_memory", 0.0
            name = f"{spec.name}-x{count}-{index}"
            run_dir = self.root / "concurrency" / name
            command = train_command(spec, run_dir)
            log = self.root / "logs" / f"{name}.train"
            outcome = run_command(command, log, spec.timeout_seconds, self.stop)
            return outcome.status, read_progress(run_dir)["worlds_per_second"] or 0.0

        with ThreadPoolExecutor(count) as pool:
            results = list(pool.map(train_copy, range(count)))
        return [rate for _, rate in results], [status for status, _ in results]


def count_rows(path: Path) -> int:
    """Number of data rows of a CSV with a header row."""
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def build_parser() -> argparse.ArgumentParser:
    defaults = Limits()
    parser = argparse.ArgumentParser(
        prog="python -m pymc_mfm.training.launch", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("launch_file", type=Path, help="JSON list of runs")
    parser.add_argument("--root", type=Path, required=True, help="output root (runs/, results.csv)")
    parser.add_argument("--resume", action="store_true", help="re-launch runs that did not finish")
    parser.add_argument("--measure-concurrency", type=_counts, metavar="1,2,4")
    parser.add_argument("--max-parallel", type=int, default=defaults.max_parallel)
    parser.add_argument("--max-runs", type=int, default=None, help="start at most N runs")
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=None,
        help="refuse to start a run whose timeout exceeds the remaining budget",
    )
    parser.add_argument("--min-free-host-gib", type=float, default=defaults.min_free_host_gib)
    parser.add_argument("--min-free-gpu-gib", type=float, default=defaults.min_free_gpu_gib)
    parser.add_argument(
        "--memory-wait-seconds",
        type=float,
        default=defaults.memory_wait_seconds,
        help="wait this long for free memory before skipping a run",
    )
    parser.add_argument(
        "--score-timeout-seconds", type=float, default=defaults.score_timeout_seconds
    )
    return parser


def _counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(","))
    if counts[0] != 1 or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("expected positive counts starting with 1, e.g. 1,2,4")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    limits = Limits(**{f.name: getattr(args, f.name) for f in fields(Limits)})
    if limits.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    try:
        specs = parse_launch_file(args.launch_file)
    except LaunchFileError as exc:
        parser.error(str(exc))
    manifest = args.root / MANIFEST
    if manifest.is_file() and not args.resume:
        parser.error(f"{manifest} exists; use --resume or another --root")
    if args.resume and not manifest.is_file():
        parser.error(f"--resume needs an existing {manifest}")
    args.root.mkdir(parents=True, exist_ok=True)
    data = json.loads(manifest.read_text()) if args.resume else {}
    launcher = Launcher(args.root, limits, Manifest(manifest, data))
    launcher.manifest.save(launch_file=str(args.launch_file.resolve()), limits=asdict(limits))
    previous = {n: signal.signal(n, lambda *_: launcher.stop.set()) for n in STOP_SIGNALS}
    try:
        if args.measure_concurrency:
            return launcher.measure_concurrency(specs[0], args.measure_concurrency)
        return launcher.run_all(specs, resume=args.resume)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    sys.exit(main())
