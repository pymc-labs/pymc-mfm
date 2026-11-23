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
import tomllib
import zipfile
from pathlib import Path

#: The local wheel must produce a complete analysis without statistics or plotting.
PAIRED_INFERENCE = """
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pymc_mfm import from_pretrained

model = from_pretrained("workflow/run/model", device="cpu")
frame = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=7, freq="W-MON"),
    "outcome": [100., 110., 105., 115., 120., 125., 130.],
    "media": [1., 3., 2., 4., 3., 4., 5.],
    "control": [-1., 0., 1., 0., -1., 0., 1.],
})
context, query = frame.iloc[:5], frame.iloc[5:]
roles = dict(
    date_column="date", target_column="outcome",
    channel_columns=["media"], control_columns=["control"],
)
torch.manual_seed(31)
tree = model.inference(context, query, **roles, n_samples=4)
assert tree is model.idata
assert set(tree.children) == {"context", "query"}
assert tree["context/posterior"].sizes["date"] == len(context)
assert tree["query/posterior"].sizes["date"] == len(query)
np.testing.assert_array_equal(
    tree["context/observed_data"]["outcome"].values, context["outcome"].to_numpy()
)
np.testing.assert_array_equal(
    tree["query/observed_data"]["outcome"].values, query["outcome"].to_numpy()
)
after = torch.get_rng_state()
torch.manual_seed(31)
assert model.inference(context, query, **roles, n_samples=4, return_idata=False) is None
assert model.idata is not tree
assert torch.equal(torch.get_rng_state(), after)
for window in ("context", "query"):
    assert model.idata[f"{window}/posterior"].sizes["draw"] == 4
    xr.testing.assert_equal(tree[f"{window}/posterior"], model.idata[f"{window}/posterior"])
stored, result = model.idata, model.query_result
prediction = model.predict(query.assign(media=query["media"] * 1.2), n_samples=2)
assert prediction.components.shape[:2] == (2, len(query))
assert model.idata is stored and model.query_result is result
curves = model.response_curves(levels=[0, 1], weeks=1, n_samples=2)
assert "response_curves" not in stored.children
np.testing.assert_array_equal(
    model.idata["response_curves/posterior"]["contribution"].values,
    curves.draws.transpose(2, 0, 1)[None],
)
"""


def run(*command: str, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sdist_allowlist(root: Path) -> set[str]:
    """The top-level entries the sdist may contain, read from the build configuration."""
    config = tomllib.loads((root / "pyproject.toml").read_text())
    return set(config["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"])


def check_archive(path: Path, allowed_top_level: set[str]) -> None:
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
        else:
            assert len(parts) > 1 and parts[1] in allowed_top_level | {"PKG-INFO"}, name


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
    allowed_top_level = sdist_allowlist(root)
    for artifact in (*wheels, *sdists):
        check_archive(artifact, allowed_top_level)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("MLFLOW_TRACKING_URI", None)
    env.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    python_paths = {}
    for name, artifact in (("sdist", sdists[0]), ("wheel", wheels[0])):
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
        if name == "sdist":
            # The source workflow also produces the synthetic corpus used by the wheel check.
            command.extend(("--group", "generation"))
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
from importlib.metadata import metadata
from pathlib import Path
import sys

import pymc_mfm
from pymc_mfm.client import Client

assert "torch" not in sys.modules, "importing the client initialized PyTorch"
assert metadata("pymc-mfm").get_all("Provides-Extra") is None
assert Path(pymc_mfm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
from pymc_mfm.training._provenance import code_record
assert code_record()["revision_status"] == "not_available"
""",
            cwd=out,
            env=env,
        )
    run(
        str(python_paths["sdist"]),
        "-I",
        str(root / ".github/scripts/workflow.py"),
        "--out",
        str(out / "workflow"),
        cwd=out,
        env=env,
    )
    wheel_python = str(python_paths["wheel"])
    # The wheel environment trains from the saved corpus without the generator.
    run(
        wheel_python,
        "-I",
        "-c",
        "import importlib.util; assert importlib.util.find_spec('pymc_generator') is None",
        cwd=out,
        env=env,
    )
    run(
        wheel_python,
        "-I",
        "-m",
        "pymc_mfm.training",
        "--corpus",
        str(out / "workflow/corpus.npz"),
        "--out-dir",
        str(out / "wheel-training"),
        "--device",
        "cpu",
        "--no-tracking",
        "--steps",
        "2",
        "--batch-size",
        "2",
        "--warmup-steps",
        "1",
        "--eval-every",
        "2",
        "--eval-worlds",
        "1",
        "--eval-samples",
        "8",
        cwd=out,
        env=env,
    )
    assert (out / "wheel-training/checkpoint_last.pt").is_file()
    run(wheel_python, "-I", "-c", PAIRED_INFERENCE, cwd=out, env=env)
    run(
        wheel_python,
        "-I",
        str(root / ".github/scripts/observation_only.py"),
        "--model",
        str(out / "workflow/run/model"),
        "--observations",
        str(out / "workflow/observations.json"),
        "--out",
        str(out / "local-inference"),
        "--samples",
        "4",
        cwd=out,
        env=env,
    )
    run(
        wheel_python,
        "-I",
        "-c",
        """
import sys
from pymc_mfm.inference import DecompositionResult
for window in ("context", "query"):
    result = DecompositionResult.load(f"local-inference/{window}")
    tree = result.to_datatree()
    assert tree.attrs["window"] == window
    assert tree["posterior"]["components"].shape == (
        1, result.n_samples, len(result.time), int(result.component_mask.sum())
    )
assert "arviz_stats" not in sys.modules
""",
        cwd=out,
        env=env,
    )
    run(
        wheel_python,
        "-I",
        "-c",
        """
from pymc_mfm.inference import DecompositionResult
from pymc_mfm.diagnostics import build_report, render_report
result = DecompositionResult.load("local-inference/query")
manifest = build_report(result, "diagnostics-report")
replay = render_report("diagnostics-report", "diagnostics-rerender")
assert replay["provenance"] == manifest["provenance"]
assert replay["panels"]["target_recovery"]["render_status"] == "ok"
assert all(
    panel.get("render_status") == "ok"
    for panel in replay["panels"].values()
    if panel["status"] not in ("unavailable", "not_applicable")
)
tree = result.to_datatree()
assert tree.attrs["window"] == "query"
assert tree["posterior"]["components"].sizes["draw"] == result.n_samples
""",
        cwd=out,
        env=env,
    )
    run(
        wheel_python,
        "-I",
        "-c",
        """
import arviz_stats
from pymc_mfm.inference import DecompositionResult
trees = {}
for window in ("query", "context"):
    result = DecompositionResult.load(f"local-inference/{window}")
    trees[window] = tree = result.to_datatree()
    expected = result.components.shape[1] * result.component_mask.sum() + result.edge_mask.sum()
    assert len(arviz_stats.summary(tree, kind="stats")) == expected, window
assert trees["query"].attrs["window"] != trees["context"].attrs["window"]
""",
        cwd=out,
        env=env,
    )
    print(
        "Installed sdist workflow and standard wheel training, inference, and diagnostics passed."
    )


if __name__ == "__main__":
    main()
