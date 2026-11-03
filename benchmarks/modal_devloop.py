"""Run the dev-loop launcher and profiler on a Modal L4.

Run it with the ``modal`` CLI (its own tool environment); this file imports only the
standard library and ``modal``. The image holds a clean clone of a committed, pushed
revision, installed with ``uv sync --frozen``. Corpora and the launch file come from a
local directory baked into the image at ``/inputs`` (no Volume); paths in the launch file
are container paths such as ``/inputs/shard_550000.npz``.

    export DEVLOOP_REVISION=<commit>          # committed and pushed
    export DEVLOOP_INPUTS=/path/to/inputs     # corpora and launch.json
    modal run -e <environment> benchmarks/modal_devloop.py::probe
    modal run -e <environment> benchmarks/modal_devloop.py::launch --launch-file launch.json \\
        --max-minutes 30 --out-dir /tmp/devloop-modal
    modal run -e <environment> benchmarks/modal_devloop.py::profile --corpus shard_550000.npz

The Modal environment has no default here: pass it with ``-e`` (or ``MODAL_ENVIRONMENT``).
Without it, or without ``DEVLOOP_REVISION`` and ``DEVLOOP_INPUTS``, importing this file under
``modal run`` raises before any image is built or any function starts, whichever function
is named. ``DEVLOOP_IMPORT_ONLY=1`` imports it without those checks (tests only).

Every job costs money. Each function has an explicit ``timeout``; ``--max-minutes`` sets it
(at most 60) and the estimated maximum cost is printed before anything starts. The
launcher inside the container gets a budget a few minutes shorter than the timeout, so it
ends cleanly and the archive of the run root comes back even when runs are cut short.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

GPU = "L4"
CPU = 4.0
MEMORY_MIB = 16384
#: Modal rates (https://modal.com/pricing): the L4 and reserved CPU and memory, per hour.
L4_USD_PER_HOUR = 0.80
CPU_CORE_USD_PER_HOUR = 0.0473
MEMORY_GIB_USD_PER_HOUR = 0.008

HARD_CAP_MINUTES = 60
PROBE_TIMEOUT_SECONDS = 60
#: Seconds kept back from the container timeout so the launcher ends and the archive is built.
ARCHIVE_RESERVE_SECONDS = 180
#: Checkpoints above this size stay in the container (the archive is returned as bytes).
MAX_CHECKPOINT_MIB = 50
#: A returned archive over this total (uncompressed) size is refused.
MAX_ARCHIVE_MIB = 200
#: Allowed benchmarks and what each takes as ``--out``: a directory or a JSON report path.
BENCH_OUT = {
    "resume_acceptance.py": "dir",
    "precision_compile_check.py": "file",
    "train_profile.py": "file",
}
BENCH_SCRIPTS = tuple(BENCH_OUT)

REMOTE_SRC = "/root/pymc-mfm"
INPUTS = "/inputs"
VENV = "/opt/pymc-mfm"
PYTHON = f"{VENV}/bin/python"
WORK = Path("/root/work")
UV_VERSION = "0.9.10"
EXTRAS = ("generation", "training")

REPO = Path(__file__).resolve().parents[1]


class RevisionError(RuntimeError):
    """The revision cannot be sent to Modal."""


# ------------------------------------------------------------------ pure helpers


def estimate_max_cost_usd(
    timeout_seconds: float, *, cpu: float = CPU, memory_mib: int = MEMORY_MIB
) -> float:
    """Most a container can cost: its timeout times the L4, CPU and memory rates."""
    usd_per_hour = L4_USD_PER_HOUR + cpu * CPU_CORE_USD_PER_HOUR
    usd_per_hour += memory_mib / 1024 * MEMORY_GIB_USD_PER_HOUR
    return timeout_seconds / 3600 * usd_per_hour


def container_timeout_seconds(max_minutes: float) -> int:
    """The container timeout for ``--max-minutes``; at most :data:`HARD_CAP_MINUTES`."""
    if not 0 < max_minutes <= HARD_CAP_MINUTES:
        raise ValueError(f"--max-minutes must be in (0, {HARD_CAP_MINUTES}], got {max_minutes}")
    return int(max_minutes * 60)


def launcher_budget_seconds(timeout_seconds: int) -> int:
    """Launcher time budget inside a container of ``timeout_seconds``."""
    return max(timeout_seconds - ARCHIVE_RESERVE_SECONDS, 1)


def chosen_environment() -> str:
    """The environment ``modal run -e`` (or ``MODAL_ENVIRONMENT``) selected."""
    return require_environment(
        os.environ.get("MODAL_ENVIRONMENT") or modal.config.config.get("environment")
    )


def require_environment(value: str | None) -> str:
    """The Modal environment chosen for this run; refuse to guess one."""
    if not value:
        raise RevisionError("choose a Modal environment with modal run -e <environment>")
    return value


def check_revision(sha: str, *, remote_branches: list[str], clone_status: str) -> None:
    """Refuse an unpushed revision or a clone that is not clean."""
    if not remote_branches:
        raise RevisionError(f"{sha} is on no remote branch; push it first")
    if clone_status:
        raise RevisionError(f"the checkout of {sha} is not clean:\n{clone_status}")


def select_archive_files(
    files: dict[str, int], *, max_checkpoint_mib: int, max_archive_mib: int = MAX_ARCHIVE_MIB
) -> tuple[list[str], list[str]]:
    """Choose the files (name -> size in bytes) to archive; return ``(kept, dropped)`` names.

    Checkpoints (``.pt``) above ``max_checkpoint_mib`` go first, then the largest remaining
    checkpoints until the total fits ``max_archive_mib``. Raises when other files alone are
    over the cap.
    """
    cap = max_archive_mib * 2**20
    kept = dict(files)
    dropped = [
        name
        for name, size in files.items()
        if name.endswith(".pt") and size > max_checkpoint_mib * 2**20
    ]
    for name in dropped:
        del kept[name]
    checkpoints = sorted((n for n in kept if n.endswith(".pt")), key=kept.get, reverse=True)
    while sum(kept.values()) > cap and checkpoints:
        name = checkpoints.pop(0)
        dropped.append(name)
        del kept[name]
    if sum(kept.values()) > cap:
        raise ValueError(f"the archive stays over {max_archive_mib} MiB without checkpoints")
    return sorted(kept), sorted(dropped)


def check_bench_script(name: str) -> None:
    """Refuse a script that is not an allowed benchmark."""
    if name not in BENCH_SCRIPTS:
        raise ValueError(f"{name!r} is not an allowed benchmark; choose from {BENCH_SCRIPTS}")


# ------------------------------------------------------------------ local checkout


def _git(*args: str, cwd: Path = REPO) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def clean_checkout(revision: str) -> tuple[Path, str]:
    """A clean detached clone of the pushed ``revision``, cached by its full SHA."""
    sha = _git("rev-parse", "--verify", f"{revision}^{{commit}}")
    remote_branches = _git("branch", "--remotes", "--contains", sha).split()
    target = Path(tempfile.gettempdir()) / "pymc-mfm-modal" / sha
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{sha}.", dir=target.parent))
        common = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
        subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", "--no-hardlinks", common, str(staging)],
            check=True,
        )
        _git("checkout", "--quiet", "--detach", sha, cwd=staging)
        _git("remote", "remove", "origin", cwd=staging)
        staging.rename(target)
    status = _git("status", "--porcelain", "--untracked-files=normal", cwd=target)
    check_revision(sha, remote_branches=remote_branches, clone_status=status)
    subprocess.run(["uv", "lock", "--check"], cwd=target, check=True, capture_output=True)
    return target, sha


# ------------------------------------------------------------------ image and app

_extras = " ".join(f"--extra {name}" for name in EXTRAS)
_base = (
    modal.Image.debian_slim(python_version="3.13")
    # pytensor (pymc-generator) compiles C kernels; code provenance reads the revision with git.
    .apt_install("build-essential", "git")
    .run_commands("git config --system --add safe.directory '*'")
    .pip_install(f"uv=={UV_VERSION}")
    .env(
        {
            "UV_PROJECT_ENVIRONMENT": VENV,
            "UV_PYTHON": "/usr/local/bin/python",
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_LINK_MODE": "copy",
            "UV_COMPILE_BYTECODE": "1",
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTENSOR_FLAGS": "base_compiledir=/tmp/pytensor",
        }
    )
)

SOURCE_SHA = None
ENVIRONMENT = None
if modal.is_local() and not os.environ.get("DEVLOOP_IMPORT_ONLY"):
    # Every `modal run` imports this file first, so refuse here, before any image is built
    # or any function (an entrypoint or a remote function called directly) can start.
    ENVIRONMENT = chosen_environment()
    if not (os.environ.get("DEVLOOP_REVISION") and os.environ.get("DEVLOOP_INPUTS")):
        raise RevisionError("set DEVLOOP_REVISION and DEVLOOP_INPUTS before modal run")
    _tree, SOURCE_SHA = clean_checkout(os.environ["DEVLOOP_REVISION"])
    image = (
        _base.add_local_file(_tree / "pyproject.toml", f"{REMOTE_SRC}/pyproject.toml", copy=True)
        .add_local_file(_tree / "uv.lock", f"{REMOTE_SRC}/uv.lock", copy=True)
        .run_commands(f"cd {REMOTE_SRC} && uv sync --frozen --no-install-project {_extras}")
        .add_local_dir(_tree, REMOTE_SRC, copy=True)
        .run_commands(
            f"cd {REMOTE_SRC} && uv sync --frozen {_extras}",
            f'test -z "$(git -C {REMOTE_SRC} status --porcelain --untracked-files=normal)"',
            f'test "$(git -C {REMOTE_SRC} rev-parse HEAD)" = {SOURCE_SHA}',
        )
        .env({"MFM_SOURCE_REVISION": SOURCE_SHA})
        .add_local_dir(os.environ["DEVLOOP_INPUTS"], INPUTS, copy=True)
    )
else:
    image = _base

app = modal.App("pymc-mfm-devloop", image=image)
_container = {"gpu": GPU, "cpu": CPU, "memory": MEMORY_MIB}


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(argv), flush=True)
    return subprocess.run(argv, cwd=REMOTE_SRC, **kwargs)


# ------------------------------------------------------------------ remote functions

_PROBE = """
import json, torch
info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
if info["cuda_available"]:
    info["device"] = torch.cuda.get_device_name(0)
    info["free_total_bytes"] = torch.cuda.mem_get_info()
print(json.dumps(info))
"""


@app.function(timeout=PROBE_TIMEOUT_SECONDS, **_container)
def probe_remote() -> dict:
    """GPU, driver and torch facts of the container (one minute at most)."""
    smi = _run(["nvidia-smi"], capture_output=True, text=True)
    torch_info = _run([PYTHON, "-c", _PROBE], capture_output=True, text=True, check=True)
    return {
        "source_revision": os.environ["MFM_SOURCE_REVISION"],
        "torch": json.loads(torch_info.stdout),
        "nvidia_smi": smi.stdout or smi.stderr,
    }


def build_archive(root: Path, *, max_checkpoint_mib: int) -> bytes:
    """``root`` as tar.gz bytes; dropped checkpoints are named in ``DROPPED.txt`` inside it."""
    files = {
        str(path.relative_to(root.parent)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    kept, dropped = select_archive_files(files, max_checkpoint_mib=max_checkpoint_mib)
    if dropped:
        print("archive leaves out checkpoints:", *dropped, sep="\n  ", flush=True)
        (root / "DROPPED.txt").write_text("\n".join(dropped) + "\n")
        kept.append(f"{root.name}/DROPPED.txt")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in kept:
            archive.add(root.parent / name, arcname=name)
    return buffer.getvalue()


@app.function(timeout=container_timeout_seconds(HARD_CAP_MINUTES), **_container)
def run_launch_remote(
    launch_file: str, budget_seconds: int, extra_args: list[str], max_checkpoint_mib: int
) -> bytes:
    """Run the launcher on ``/inputs/<launch_file>``; return the run root as tar.gz bytes."""
    root = WORK / "launch"
    command = [
        PYTHON,
        "-m",
        "pymc_mfm.training.launch",
        f"{INPUTS}/{launch_file}",
        "--root",
        str(root),
        "--max-total-seconds",
        str(budget_seconds),
        *extra_args,
    ]
    returncode = _run(command).returncode
    (root / "exit_status.txt").write_text(f"{returncode}\n")
    return build_archive(root, max_checkpoint_mib=max_checkpoint_mib)


@app.function(timeout=container_timeout_seconds(HARD_CAP_MINUTES), **_container)
def profile_remote(corpus: str, preset: str, steps: int) -> dict:
    """Run ``benchmarks/train_profile.py`` on ``/inputs/<corpus>``; return its JSON report."""
    out = WORK / "profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "benchmarks/train_profile.py", "--corpus", f"{INPUTS}/{corpus}"]
    command += ["--preset", preset, "--steps", str(steps), "--device", "cuda", "--out", str(out)]
    _run(command, check=True)
    return json.loads(out.read_text())


def _text(output: str | bytes | None) -> str:
    """Captured output as text; ``TimeoutExpired`` carries bytes or None."""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output or ""


def run_benchmark(command: list[str], timeout_seconds: int) -> tuple[str, int]:
    """Run ``command``; return its stdout and stderr and exit code, or -1 after the timeout."""
    try:
        done = _run(command, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as expired:
        return _text(expired.stdout) + _text(expired.stderr) + "\ntimed out\n", -1
    return _text(done.stdout) + _text(done.stderr), done.returncode


@app.function(timeout=container_timeout_seconds(HARD_CAP_MINUTES), **_container)
def bench_remote(
    script: str, args: list[str], budget_seconds: int, max_checkpoint_mib: int
) -> dict:
    """Run ``benchmarks/<script> <args> --out DIR``; return stdout, exit code and DIR as tar.gz."""
    out = WORK / "bench-out"
    out.mkdir(parents=True, exist_ok=True)
    target = out if BENCH_OUT[script] == "dir" else out / "report.json"
    command = [PYTHON, f"benchmarks/{script}", *args, "--out", str(target)]
    stdout, exit_code = run_benchmark(command, budget_seconds)
    return {
        "stdout": stdout,
        "exit_code": exit_code,
        "archive": build_archive(out, max_checkpoint_mib=max_checkpoint_mib),
    }


# ------------------------------------------------------------------ local entrypoints


def _start(function, max_minutes: float):
    """Check the inputs, print the maximum cost and return ``function`` with its timeout set."""
    if SOURCE_SHA is None:
        raise RevisionError("set DEVLOOP_REVISION and DEVLOOP_INPUTS before modal run")
    seconds = container_timeout_seconds(max_minutes)
    print(
        f"environment {ENVIRONMENT}; revision {SOURCE_SHA}; timeout {seconds} s; "
        f"estimated maximum cost {estimate_max_cost_usd(seconds):.2f} USD"
    )
    return function.with_options(timeout=seconds), seconds


@app.local_entrypoint()
def probe():
    print(f"environment {ENVIRONMENT}")
    print(json.dumps(probe_remote.remote(), indent=2))


@app.local_entrypoint()
def launch(
    launch_file: str,
    max_minutes: float = 30,
    out_dir: str = "devloop-modal",
    extra_args: str = "",
    max_checkpoint_mib: int = MAX_CHECKPOINT_MIB,
):
    """Run the launch file in ``/inputs`` and save the returned archive under ``out_dir``."""
    function, seconds = _start(run_launch_remote, max_minutes)
    archive = function.remote(
        launch_file, launcher_budget_seconds(seconds), extra_args.split(), max_checkpoint_mib
    )
    target = Path(out_dir) / f"launch-{SOURCE_SHA[:10]}.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive)
    print(f"wrote {target} ({len(archive)} bytes)")


@app.local_entrypoint()
def profile(
    corpus: str,
    preset: str = "mini",
    steps: int = 10,
    max_minutes: float = 10,
    out_dir: str = "devloop-modal",
):
    """Profile training updates on the L4 and save the JSON report under ``out_dir``."""
    function, _ = _start(profile_remote, max_minutes)
    report = function.remote(corpus, preset, steps)
    target = Path(out_dir) / f"profile-{SOURCE_SHA[:10]}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {target}")


@app.local_entrypoint()
def bench(
    script: str,
    args: str = "",
    max_minutes: float = 30,
    out_dir: str = "devloop-modal",
    max_checkpoint_mib: int = MAX_CHECKPOINT_MIB,
):
    """Run an allowed ``benchmarks/<script>`` with ``--args``; it gets ``--out`` appended.

    ``--args`` is one string, for example ``"--device cuda --steps 200"``. Stdout and the
    archive of the ``--out`` directory are saved under ``out_dir``.
    """
    check_bench_script(script)
    function, seconds = _start(bench_remote, max_minutes)
    result = function.remote(
        script, shlex.split(args), launcher_budget_seconds(seconds), max_checkpoint_mib
    )
    stem = f"{Path(script).stem}-{SOURCE_SHA[:10]}"
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{stem}.stdout.txt").write_text(result["stdout"])
    (target / f"{stem}.tar.gz").write_bytes(result["archive"])
    print(f"exit code {result['exit_code']}; wrote {target / stem}.stdout.txt and .tar.gz")
