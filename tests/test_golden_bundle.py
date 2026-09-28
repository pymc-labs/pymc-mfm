"""Keep the original dimensional bundle as an incompatible-representation fixture.

``data/golden_bundle.json`` retains every byte of a tiny semantic-version-1
bundle. It must be rejected locally and through the real offline Hub cache,
not re-recorded with a new schema over old weights. Fresh bundles separately
exercise provenance round-tripping and inference-only imports.
"""

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_bundle.json"
PROFILE = "golden"
COMMIT = "a" * 40
REPO = "test-owner/dimensional-golden"
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


@pytest.fixture
def current_bundle(model, tmp_path):
    return save_pretrained(model, tmp_path / "current", provenance=PROVENANCE, profile=PROFILE)


def test_dimensional_golden_bundle_requires_retraining(collection):
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        from_pretrained(collection, profile=PROFILE)


def test_offline_hub_cache_cannot_bypass_representation_rejection(tmp_path):
    pytest.importorskip("huggingface_hub")
    cache = tmp_path / "cache"
    repo = cache / "models--test-owner--dimensional-golden"
    snapshot = repo / "snapshots" / COMMIT
    materialize(json.loads(GOLDEN.read_text())["files"], snapshot / "profiles" / PROFILE)
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
