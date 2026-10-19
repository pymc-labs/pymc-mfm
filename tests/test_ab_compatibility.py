"""Opt-in: bundles A and B reproduce the training evaluation recorded in their provenance.

``MFM_AB_BUNDLES`` is a local bundle collection holding ``profiles/A`` and
``profiles/B``; ``MFM_AB_CORPORA`` is a directory with ``A/shard_<seed>.npz`` and
``B/shard_<seed>.npz`` for every recorded shard seed. The test skips only while
either variable is unset; a set but unusable location fails. Each profile runs
:data:`~pymc_mfm.training.compatibility.MIN_REPLICATES` joint-head evaluations of
its recorded worlds on the CPU, which takes a long time.
"""

import os
from pathlib import Path

import pytest

BUNDLES = os.environ.get("MFM_AB_BUNDLES")
CORPORA = os.environ.get("MFM_AB_CORPORA")

pytestmark = pytest.mark.skipif(
    BUNDLES is None or CORPORA is None,
    reason="set MFM_AB_BUNDLES and MFM_AB_CORPORA to check the A/B recorded evaluations",
)


@pytest.mark.parametrize("profile", ["A", "B"])
def test_bundle_reproduces_its_recorded_training_evaluation(profile):
    from pymc_mfm.data.recipes import load_corpora
    from pymc_mfm.inference import from_pretrained
    from pymc_mfm.training import check_recorded_evaluation

    assert Path(BUNDLES).is_dir(), f"MFM_AB_BUNDLES={BUNDLES!r} is not a directory"
    shards = Path(CORPORA) / profile
    assert shards.is_dir(), f"MFM_AB_CORPORA has no {profile}/ directory"
    model = from_pretrained(BUNDLES, profile=profile)
    recorded = model.provenance["training_evaluation"]
    seeds = sorted({world["shard_seed"] for world in recorded["worlds"]})
    loaded = load_corpora([shards / f"shard_{seed}.npz" for seed in seeds])
    result = check_recorded_evaluation(
        model, {seed: item.corpus for seed, item in zip(seeds, loaded, strict=True)}
    )
    assert result["passed"], {
        key: result[key]
        for key in (
            "reason",
            "exact_failures",
            "must_pass_failures",
            "gross_failures",
            "family_failures",
            "failing_families",
            "allowed_family_failures",
        )
    }
