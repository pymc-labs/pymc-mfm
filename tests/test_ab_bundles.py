"""Opt-in rejection of the original dimensional A/B bundle collection.

Set ``MFM_LEGACY_AB_BUNDLES`` to the collection holding ``profiles/A`` and
``profiles/B``. These weights were trained before canonical input scaling;
neither loading nor publishing compatibility cards may reinterpret them.
Fresh-model evaluation compatibility is covered in ``test_compatibility.py``.
"""

import os
from pathlib import Path

import pytest

from pymc_mfm.inference.hub import from_pretrained, write_model_cards


@pytest.fixture(scope="module", params=("A", "B"))
def legacy_bundle(request):
    value = os.environ.get("MFM_LEGACY_AB_BUNDLES")
    if value is None:
        pytest.skip("set MFM_LEGACY_AB_BUNDLES to check the original dimensional A/B bundles")
    root = Path(value)
    if not value or not root.is_dir():
        pytest.fail(f"MFM_LEGACY_AB_BUNDLES={value!r} is not a bundle collection directory")
    return root / "profiles" / request.param, request.param


def test_dimensional_ab_weights_require_retraining(legacy_bundle):
    path, profile = legacy_bundle
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        from_pretrained(path, profile=profile)


def test_dimensional_ab_weights_cannot_receive_current_compatibility_cards(legacy_bundle, tmp_path):
    source, profile = legacy_bundle
    copy = tmp_path / "profiles" / profile
    copy.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "manifest.json"):
        (copy / name).symlink_to((source / name).resolve())
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        write_model_cards(copy, license="not selected", corpus_plan=profile)
    assert not (copy / "README.md").exists()
