"""Child processes of the launcher: memory checks, bounded commands and run progress."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

STDERR_TAIL_LINES = 40
#: Seconds a terminated run gets to exit before it is killed.
TERMINATE_GRACE_SECONDS = 10.0
_POLL_SECONDS = 0.2


def host_free_gib() -> float:
    """``MemAvailable`` of the host in GiB."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def gpu_free_gib() -> float:
    return torch.cuda.mem_get_info()[0] / 2**30


def uses_cuda(device: str) -> bool:
    return device.startswith("cuda") or (device == "auto" and torch.cuda.is_available())


def read_progress(run_dir: Path) -> dict[str, Any]:
    """Steps done and worlds per second from the last record of the run's ``metrics.jsonl``."""
    path = run_dir / "metrics.jsonl"
    lines = path.read_text().splitlines() if path.is_file() else []
    if not lines:
        return {"steps_done": 0, "worlds_per_second": None}
    last = json.loads(lines[-1])
    return {"steps_done": last["step"], "worlds_per_second": last["worlds_per_second"]}


def stderr_tail(path: Path) -> list[str]:
    lines = path.read_text(errors="replace").splitlines() if path.is_file() else []
    return lines[-STDERR_TAIL_LINES:]


@dataclass
class Outcome:
    #: ``finished``, ``failed``, ``timed_out`` or ``interrupted``.
    status: str
    returncode: int | None
    seconds: float


def run_command(
    command: Sequence[str],
    log_prefix: Path,
    timeout: float,
    stop: threading.Event,
    on_start: Callable[[int], None] = lambda pid: None,
) -> Outcome:
    """Run ``command`` in its own process group, logging to ``<prefix>.stdout/.stderr``.

    ``on_start`` receives the child's process id (also its group id) once it runs.

    At ``timeout`` seconds, or when ``stop`` is set, the group is terminated, then killed
    after :data:`TERMINATE_GRACE_SECONDS`.
    """
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (
        log_prefix.with_suffix(".stdout").open("ab") as out,
        log_prefix.with_suffix(".stderr").open("ab") as err,
    ):
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        on_start(process.pid)
        status = None
        while process.poll() is None and status is None:
            if stop.is_set():
                status = "interrupted"
            elif time.monotonic() - started > timeout:
                status = "timed_out"
            else:
                stop.wait(_POLL_SECONDS)
        if status is not None:
            _end_group(process)
    seconds = time.monotonic() - started
    if status is None:
        status = "finished" if process.returncode == 0 else "failed"
    return Outcome(status, process.returncode, seconds)


def _end_group(process: subprocess.Popen) -> None:
    """Terminate the process group the launcher created for ``process``, then kill it."""
    for sig, grace in ((signal.SIGTERM, TERMINATE_GRACE_SECONDS), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sig)
            process.wait(grace)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            process.wait()
            return
