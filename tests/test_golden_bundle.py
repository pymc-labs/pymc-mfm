"""A stored tiny bundle in the published format keeps loading.

``data/golden_bundle.json`` holds every file of one tiny profile bundle with a
recorded ``training_evaluation`` block: the JSON files as text lines and
``model.safetensors`` as base64 lines, so the repository holds no binary.
Loading exactly those bytes with ``from_pretrained`` fails when the bundle
schema, the provenance validator, the tensor catalog or ``ModelConfig`` drift,
which would also make published bundles unloadable. It records upper bounds
that loading does not enforce, ``torch<3`` and ``pymc-mfm<0.0.2``
(``RECORDED_COMPATIBILITY``, which the recording release must satisfy), so it
also proves that bundles keep loading under later releases, including one that
relaxes the supported ranges.

Record again only after an intentional format change::

    python -m tests.test_golden_bundle --write [--force]
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from unittest import mock

import pytest
import torch
from packaging.specifiers import SpecifierSet

import pymc_mfm.inference.checkpoint as checkpoint
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.model import MFM, ModelConfig

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_bundle.json"
PROFILE = "golden"
SEED = 20260923
MODEL = {
    "d_model": 4,
    "n_layers": 1,
    "n_heads": 1,
    "d_ff": 4,
    "flow_hidden": 4,
    "n_controls": 1,
    "n_demand": 1,
}
TRAINING_EVALUATION = {
    "status": "recorded",
    "source": "terminal in-training evaluation",
    "step": 40,
    "leg": "holdout",
    "head": "joint",
    "eval_seed": 616161,
    "n_worlds": 2,
    "n_samples": 8,
    "flow_steps": 16,
    "worlds": [{"shard_seed": 540000, "row": 5}, {"shard_seed": 541000, "row": 17}],
    "metrics": {
        "counts/worlds": 2.0,
        "joint/context/latent_sales_r2": -0.25,
        "joint/query/contribution_crps": 0.125,
        "joint/query/decomposition_energy_score": 0.5,
        "joint_router/edge/cy/auprc": None,
        "joint_router/graph/shd_mean": 1.5,
        "structure_head/zb/log_score": 0.75,
    },
    "aggregation": {
        "counts/worlds": "world",
        "joint/context/latent_sales_r2": "world",
        "joint/query/contribution_crps": "slot",
        "joint/query/decomposition_energy_score": "world",
        "joint_router/edge/cy/auprc": "pooled_edges",
        "joint_router/graph/shd_mean": "world",
        "structure_head/zb/log_score": "pooled_edges",
    },
}
PROVENANCE = {
    "source_checkpoint_sha256": "0123456789abcdef" * 4,
    "source_step": 40,
    "training_seed": 7,
    "training_evaluation": TRAINING_EVALUATION,
}
#: Recorded in place of this release's range: an upper bound later releases exceed.
RECORDED_COMPATIBILITY = {"pymc-mfm": ">=0.0.1.dev0,<0.0.2"}


def golden_files() -> dict[str, dict]:
    """Export the tiny bundle and encode its files for the JSON recording.

    The export records this release's compatibility constant updated with
    ``RECORDED_COMPATIBILITY``.
    """
    torch.manual_seed(SEED)
    model = MFM(ModelConfig(**MODEL))
    compatibility = {**checkpoint.COMPATIBILITY, **RECORDED_COMPATIBILITY}
    with (
        tempfile.TemporaryDirectory() as directory,
        mock.patch.object(checkpoint, "COMPATIBILITY", compatibility),
    ):
        bundle = save_pretrained(
            model, Path(directory) / "bundle", provenance=PROVENANCE, profile=PROFILE
        )
        files = {
            name: {"encoding": "utf-8", "lines": (bundle / name).read_text().split("\n")}
            for name in ("config.json", "manifest.json")
        }
        weights = base64.b64encode((bundle / "model.safetensors").read_bytes()).decode()
        files["model.safetensors"] = {
            "encoding": "base64",
            "lines": [weights[start : start + 76] for start in range(0, len(weights), 76)],
        }
    return files


def materialize(files: dict[str, dict], directory: Path) -> Path:
    """Write the recorded files back as the exact bundle bytes."""
    directory.mkdir(parents=True)
    for name, entry in files.items():
        if entry["encoding"] == "base64":
            data = base64.b64decode("".join(entry["lines"]), validate=True)
        else:
            data = "\n".join(entry["lines"]).encode("utf-8")
        (directory / name).write_bytes(data)
    return directory


@pytest.fixture
def collection(tmp_path):
    materialize(json.loads(GOLDEN.read_text())["files"], tmp_path / "profiles" / PROFILE)
    return tmp_path


def test_golden_bundle_loads_with_its_recorded_training_evaluation(collection):
    model = from_pretrained(collection, profile=PROFILE)
    assert model.config == ModelConfig(**MODEL)
    assert model.artifact_metadata["provenance"] == {"status": "recorded", **PROVENANCE}
    assert model.provenance["checkpoint_sha256"] == PROVENANCE["source_checkpoint_sha256"]


def test_golden_bundle_keeps_loading_beyond_its_recorded_upper_bounds(collection, monkeypatch):
    installed = {"pymc-mfm": "0.0.2", "torch": "3.0.0"}
    monkeypatch.setattr(checkpoint, "version", lambda name: installed.get(name) or version(name))
    # A later release that supports torch 3.
    monkeypatch.setattr(
        checkpoint, "COMPATIBILITY", {**checkpoint.COMPATIBILITY, "torch": ">=2.6,<4"}
    )
    model = from_pretrained(collection, profile=PROFILE)
    assert model.artifact_metadata["provenance"] == {"status": "recorded", **PROVENANCE}
    # Each installed version exceeds the upper bound that the golden records.
    recorded = model.artifact_metadata["compatibility"]
    assert not any(
        SpecifierSet(recorded[name]).contains(number, prereleases=True)
        for name, number in installed.items()
    )


def test_loading_a_recorded_training_evaluation_imports_no_training_code(collection):
    code = (
        "import sys\n"
        "from pymc_mfm import from_pretrained\n"
        f"from_pretrained(sys.argv[1], profile={PROFILE!r})\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('pymc_mfm.training'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(collection)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.test_golden_bundle",
        description="Record the golden bundle with the installed package.",
    )
    parser.add_argument("--write", action="store_true", required=True, help=f"write {GOLDEN}")
    parser.add_argument("--force", action="store_true", help="replace an existing recording")
    args = parser.parse_args(argv)
    if GOLDEN.exists() and not args.force:
        parser.error(f"{GOLDEN} exists; pass --force to replace the recorded bundle")
    payload = {"format_version": 1, "profile": PROFILE, "files": golden_files()}
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(GOLDEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
