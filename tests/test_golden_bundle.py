"""Reject a bundle of the retired semantic-schema version 1.

The bundle is generated from code at test time, so no checkpoint bytes are
committed: a tiny model with a fixed seed is exported while the semantic
schema version reads 1, the retired dimensional representation. It must be
rejected locally and through the real offline Hub cache, not re-recorded with
the new schema over old weights. Fresh bundles separately exercise provenance
round-tripping and inference-only imports.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import pymc_mfm.inference._artifact_schema as artifact_schema
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.model import MFM, ModelConfig

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


def write_retired_bundle(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Export a seeded tiny model whose schema carries the retired version 1.

    The patch covers only the export, so the manifest hashes agree with the
    files and the loader reaches the schema check instead of failing integrity.
    """
    with torch.random.fork_rng(devices=[]), monkeypatch.context() as patch:
        torch.manual_seed(SEED)
        model = MFM(ModelConfig(**MODEL))
        patch.setattr(artifact_schema, "SEMANTIC_SCHEMA_VERSION", RETIRED_SCHEMA_VERSION)
        bundle = save_pretrained(model, directory, provenance=PROVENANCE, profile=PROFILE)
    recorded = json.loads((bundle / "config.json").read_text())["schemas"]["version"]
    assert recorded == RETIRED_SCHEMA_VERSION
    return bundle


@pytest.fixture
def collection(tmp_path, monkeypatch):
    write_retired_bundle(tmp_path / "profiles" / PROFILE, monkeypatch)
    return tmp_path


@pytest.fixture
def current_bundle(model, tmp_path):
    return save_pretrained(model, tmp_path / "current", provenance=PROVENANCE, profile=PROFILE)


def test_dimensional_golden_bundle_requires_retraining(collection):
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        from_pretrained(collection, profile=PROFILE)


def test_offline_hub_cache_cannot_bypass_representation_rejection(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    cache = tmp_path / "cache"
    repo = cache / "models--test-owner--dimensional-golden"
    snapshot = repo / "snapshots" / COMMIT
    write_retired_bundle(snapshot / "profiles" / PROFILE, monkeypatch)
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
