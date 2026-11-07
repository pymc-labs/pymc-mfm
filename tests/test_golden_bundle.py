"""Reject a bundle of the retired semantic-schema version 1.

``data/golden_bundle_v1/`` holds the genuine text files of the original
dimensional bundle, byte for byte: ``config.json`` and ``manifest.json`` with
their old field layout. No checkpoint bytes are committed. The weights file is
generated at test time from a tiny model with a fixed seed, so its provenance
proves itself: it is an untrained initialisation made by code in this
repository, not a trained model.

A new weights file has other bytes, so the test rewrites exactly the manifest
fields that describe that file and nothing else:

* ``files["model.safetensors"]`` (``sha256`` and ``size``), and
* ``bundle_sha256``, the digest of the rest of the manifest, which those two
  values change.

``config.json`` stays untouched, so the ``config_sha256`` in the weights
header is the digest the manifest already holds. The loader keeps no digest
of the tensors in the files (it computes ``state_sha256`` after loading), so
nothing else needs rewriting. With integrity passing, the loader reaches the
representation check and must reject the bundle, locally and through the real
offline Hub cache, instead of re-recording it with the new schema over old
weights. The test asserts that the rewritten manifest differs from the
committed one only in those fields. Fresh bundles separately exercise
provenance round-tripping and inference-only imports.
"""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pymc_mfm.inference._artifact_schema import digest_bytes, file_record, json_bytes
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.model import MFM, ModelConfig

LEGACY = Path(__file__).resolve().parent / "data" / "golden_bundle_v1"
WEIGHTS = "model.safetensors"
PROFILE = "golden"
COMMIT = "a" * 40
REPO = "test-owner/dimensional-golden"
RETIRED_SCHEMA_VERSION = 1
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


def retired_manifest(legacy: dict, weights: Path) -> dict:
    """The legacy manifest with only the fields that describe ``weights`` rewritten."""
    manifest = copy.deepcopy(legacy)
    manifest["files"][WEIGHTS] = file_record(weights)
    manifest.pop("bundle_sha256")
    manifest["bundle_sha256"] = digest_bytes(json_bytes(manifest))
    return manifest


def write_retired_bundle(directory: Path) -> Path:
    """Copy the legacy text files and generate the seeded weights beside them."""
    directory.mkdir(parents=True)
    shutil.copyfile(LEGACY / "config.json", directory / "config.json")
    legacy_text = (LEGACY / "manifest.json").read_text()
    legacy = json.loads(legacy_text)
    assert json_bytes(legacy).decode() == legacy_text
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        state = MFM(ModelConfig(**MODEL)).state_dict()
    assert [entry["name"] for entry in legacy["tensors"]] == list(state)
    save_file(
        {name: tensor.detach().contiguous() for name, tensor in state.items()},
        str(directory / WEIGHTS),
        metadata={
            "format": "pt",
            "architecture": legacy["architecture"],
            "config_sha256": legacy["files"]["config.json"]["sha256"],
        },
    )
    manifest = retired_manifest(legacy, directory / WEIGHTS)
    (directory / "manifest.json").write_bytes(json_bytes(manifest))
    return directory


@pytest.fixture
def collection(tmp_path):
    write_retired_bundle(tmp_path / "profiles" / PROFILE)
    return tmp_path


@pytest.fixture
def current_bundle(model, tmp_path):
    return save_pretrained(model, tmp_path / "current", provenance=PROVENANCE, profile=PROFILE)


def test_retired_bundle_changes_only_the_weights_fields_of_the_legacy_text(tmp_path):
    bundle = write_retired_bundle(tmp_path / "bundle")
    assert (bundle / "config.json").read_bytes() == (LEGACY / "config.json").read_bytes()
    legacy = json.loads((LEGACY / "manifest.json").read_text())
    written = json.loads((bundle / "manifest.json").read_text())
    assert written.keys() == legacy.keys()
    assert written["files"].keys() == legacy["files"].keys()
    changed = {key for key in legacy if written[key] != legacy[key]}
    assert changed <= {"files", "bundle_sha256"}
    assert written["files"]["config.json"] == legacy["files"]["config.json"]
    assert written["files"][WEIGHTS].keys() == {"sha256", "size"}
    assert {key: value for key, value in written.items() if key not in changed} == {
        key: value for key, value in legacy.items() if key not in changed
    }
    for name, record in written["files"].items():
        assert file_record(bundle / name) == record


def test_dimensional_golden_bundle_requires_retraining(collection):
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        from_pretrained(collection, profile=PROFILE)


def test_offline_hub_cache_cannot_bypass_representation_rejection(tmp_path):
    pytest.importorskip("huggingface_hub")
    cache = tmp_path / "cache"
    repo = cache / "models--test-owner--dimensional-golden"
    snapshot = repo / "snapshots" / COMMIT
    write_retired_bundle(snapshot / "profiles" / PROFILE)
    (repo / "refs").mkdir()
    (repo / "refs" / "release").write_text(COMMIT)
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        from_pretrained(
            REPO,
            revision="release",
            profile=PROFILE,
            cache_dir=cache,
            local_files_only=True,
            token=False,
        )


def test_current_bundle_preserves_recorded_training_evaluation(current_bundle):
    model = from_pretrained(current_bundle)
    assert model.artifact_metadata["provenance"] == {"status": "recorded", **PROVENANCE}
    assert model.provenance["checkpoint_sha256"] == PROVENANCE["source_checkpoint_sha256"]


def test_loading_a_recorded_training_evaluation_imports_no_training_code(current_bundle):
    code = (
        "import sys\n"
        "from pymc_mfm import from_pretrained\n"
        f"from_pretrained(sys.argv[1], profile={PROFILE!r})\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('pymc_mfm.training'))\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(current_bundle)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
