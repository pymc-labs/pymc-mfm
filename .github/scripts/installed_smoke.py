"""Exercise built wheel/sdist installations, never an editable checkout.

Run after `uv build`. Dependencies use the existing lockfile; environments and
all generated corpora/checkpoints/reports stay below the supplied work directory.
No remote tracking or model publishing is performed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def run(*command: str, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def check_archive(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = [item.name for item in archive.getmembers() if item.isfile()]
    for name in names:
        parts = Path(name).parts
        assert not Path(name).is_absolute() and ".." not in parts, name
        assert not set(parts) & {
            ".venv",
            ".conda",
            "__pycache__",
            "artifacts",
            "runs",
            "latest_model",
            "mlruns",
        }, name
        assert not any(part == ".env" or part.startswith(".env.") for part in parts), name
        assert Path(name).suffix not in {
            ".pt",
            ".pth",
            ".safetensors",
            ".nc",
            ".h5",
            ".npz",
            ".npy",
        }, name
        if path.suffix == ".whl":
            assert parts[0] == "pymc_mfm" or parts[0].endswith(".dist-info"), name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.work_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Build into a clean dist directory"
    for artifact in (*wheels, *sdists):
        check_archive(artifact)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("MLFLOW_TRACKING_URI", None)
    env.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    python_paths = {}
    for name, extras, artifact in (
        ("full", ("generation", "training", "hub"), sdists[0]),
        ("base", (), wheels[0]),
        ("diagnostics", ("diagnostics",), wheels[0]),
    ):
        prefix = out / name
        run("uv", "venv", "--python", sys.executable, str(prefix), cwd=out, env=env)
        python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python_paths[name] = python
        requirements = out / f"{name}-requirements.txt"
        command = [
            "uv",
            "export",
            "--quiet",
            "--locked",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ]
        for extra in extras:
            command.extend(("--extra", extra))
        run(*command, cwd=root, env=env)
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-r",
            str(requirements),
            cwd=out,
            env=env,
        )
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(artifact),
            cwd=out,
            env=env,
        )
        run("uv", "pip", "check", "--python", str(python), cwd=out, env=env)
        run(
            str(python),
            "-I",
            "-c",
            """
import pathlib
import sys
import pymc_mfm
from pymc_mfm.training._provenance import code_record
p = pathlib.Path(pymc_mfm.__file__).resolve()
assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), p
assert code_record()["revision_status"] == "not_available"
print(p)
""",
            cwd=out,
            env=env,
        )
    run(
        str(python_paths["full"]),
        "-I",
        str(root / "examples/workflow.py"),
        "--out",
        str(out / "workflow"),
        cwd=out,
        env=env,
    )
    forbidden = ("pymc_generator", "mlflow", "matplotlib", "scipy", "huggingface_hub")
    run(
        str(python_paths["base"]),
        "-I",
        "-c",
        f"import importlib.util\nfor name in {forbidden!r}:\n"
        "    assert importlib.util.find_spec(name) is None, name",
        cwd=out,
        env=env,
    )
    run(
        str(python_paths["base"]),
        "-I",
        str(root / "examples/observation_only.py"),
        "--model",
        str(out / "workflow/run/model"),
        "--observations",
        str(out / "workflow/observations.json"),
        "--out",
        str(out / "base-inference"),
        "--samples",
        "4",
        cwd=out,
        env=env,
    )
    run(
        str(python_paths["diagnostics"]),
        "-I",
        "-c",
        """
import importlib.util
from pymc_mfm import DecompositionResult
from pymc_mfm.diagnostics import build_report, render_report
assert all(
    importlib.util.find_spec(name) is None
    for name in ("pymc_generator", "mlflow", "huggingface_hub")
)
result = DecompositionResult.load("base-inference/query")
manifest = build_report(result, "diagnostics-report")
replay = render_report("diagnostics-report", "diagnostics-rerender")
assert replay["provenance"] == manifest["provenance"]
assert replay["panels"]["target_recovery"]["render_status"] == "ok"
assert all(
    panel.get("render_status") == "ok"
    for panel in replay["panels"].values()
    if panel["status"] not in ("unavailable", "not_applicable")
)
""",
        cwd=out,
        env=env,
    )
    print("Installed sdist workflow, base wheel inference and diagnostics-only replay passed.")


if __name__ == "__main__":
    main()
