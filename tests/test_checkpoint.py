"""Checkpoint semantics must survive round-tripping, not just tensor shapes."""

import hashlib
import json
import math
import platform
import tomllib
from dataclasses import replace
from importlib.metadata import version as installed_version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from safetensors.torch import load_file, save_file

from pymc_mfm.inference._artifact_schema import COMPATIBILITY
from pymc_mfm.inference.api import decompose
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM
from pymc_mfm.training.checkpoint import load_checkpoint, save_checkpoint


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _rebind_bundle(path, *, metadata=None, state=None, change_catalog=None):
    """Re-sign a deliberately malformed bundle to exercise checks beyond hashes."""
    if metadata is not None:
        (path / "config.json").write_bytes(_json_bytes(metadata))
    config_hash = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    metadata = json.loads((path / "config.json").read_bytes())
    if state is None:
        state = load_file(path / "model.safetensors")
    save_file(
        state,
        path / "replacement.safetensors",
        metadata={
            "format": "pt",
            "architecture": metadata["architecture"],
            "config_sha256": config_hash,
        },
    )
    (path / "replacement.safetensors").replace(path / "model.safetensors")
    manifest = json.loads((path / "manifest.json").read_bytes())
    for name in manifest["files"]:
        data = (path / name).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    if change_catalog is not None:
        change_catalog(manifest["tensors"])
    del manifest["bundle_sha256"]
    manifest["bundle_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    (path / "manifest.json").write_bytes(_json_bytes(manifest))


def test_checkpoint_roundtrip_preserves_conditional_draws_and_dtype(model, batch, tmp_path):
    model = model.double()
    batch = {key: value.double() for key, value in batch.items()}
    path = tmp_path / "model"
    save_pretrained(model, path)
    loaded = from_pretrained(path)
    torch.manual_seed(20)
    expected, graphs = model.sample_disentangled(batch, 3)
    torch.manual_seed(20)
    actual, loaded_graphs = loaded.sample_disentangled(batch, 3)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(loaded_graphs, graphs, rtol=0, atol=0)
    torch.manual_seed(21)
    expected_loss = model.loss(batch)
    expected_loss.backward()
    torch.manual_seed(21)
    actual_loss = loaded.loss(batch)
    actual_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    # Multithreaded CPU backward kernels may reduce in a different order in each
    # pass, so gradients use float64 tolerance; draws, loss and tensors are exact.
    for original, restored in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(restored.grad, original.grad)
    assert loaded.config == model.config
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def _save_training_checkpoint(model, path):
    save_checkpoint(
        path,
        model=model,
        kind="last",
        step=1,
        training_config={},
        run={},
        consumed={},
        evaluation=None,
        selection={},
    )


def test_training_checkpoint_and_bundle_preserve_canonical_decomposition(
    model, synthetic_corpus, tmp_path
):
    context = int(synthetic_corpus["support_mask"][0].sum())
    observations = {
        "spend": synthetic_corpus["treatment_raw"][0],
        "controls": synthetic_corpus["covariates"][0],
        "context_outcome": synthetic_corpus["outcome_raw"][0, :context],
        "channel_mask": synthetic_corpus["treatment_active_mask"][0],
        "control_mask": synthetic_corpus["covariate_active_mask"][0],
        "latent_mask": synthetic_corpus["latent_active_mask"][0],
    }
    path = tmp_path / "checkpoint.pt"
    _save_training_checkpoint(model, path)
    restored, _ = load_checkpoint(path)
    exported = from_pretrained(save_pretrained(restored, tmp_path / "bundle"))
    torch.manual_seed(31)
    expected = decompose(model, observations, n_samples=2)
    for loaded in (restored, exported):
        torch.manual_seed(31)
        actual = decompose(loaded, observations, n_samples=2)
        np.testing.assert_allclose(actual.components, expected.components, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(actual.graphs, expected.graphs)
        assert actual.provenance["sales_scale"] == expected.provenance["sales_scale"]


@pytest.mark.parametrize("change", ["old_version", "different_outcome_ruler"])
def test_same_shaped_bundles_cannot_change_the_input_representation(model, tmp_path, change):
    path = save_pretrained(model, tmp_path / "model")
    metadata = json.loads((path / "config.json").read_bytes())
    if change == "old_version":
        metadata["schemas"]["version"] = 1
        metadata["model_config"]["format_version"] = 1
    else:
        metadata["schemas"]["normalization"]["sales_scale_quantile"] = 0.5
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        from_pretrained(path)


@pytest.mark.parametrize(
    "change", ["raw_state", "dimensional", "relabeled_format", "changed_schema"]
)
def test_training_weights_require_the_recorded_canonical_representation(model, tmp_path, change):
    path = tmp_path / "checkpoint.pt"
    _save_training_checkpoint(model, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if change == "raw_state":
        payload = {key: payload[key] for key in ("config", "state_dict")}
    elif change == "dimensional":
        payload["format_version"] = 1
        del payload["schemas"]
    elif change == "relabeled_format":
        payload["schemas"]["version"] = 1
        payload["config"]["format_version"] = 1
    else:
        payload["schemas"]["normalization"]["sales_scale_quantile"] = 0.5
    torch.save(payload, path)
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        load_checkpoint(path)


def test_export_cannot_relabel_an_incompatible_loaded_model(model, tmp_path):
    loaded = from_pretrained(save_pretrained(model, tmp_path / "current"))
    loaded.artifact_metadata["schemas"]["version"] = 1
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        save_pretrained(loaded, tmp_path / "relabeled_bundle", provenance={})
    with pytest.raises(ValueError, match="(?i)representation.*retrain"):
        _save_training_checkpoint(loaded, tmp_path / "relabeled_checkpoint.pt")
    assert not (tmp_path / "relabeled_bundle").exists()
    assert not (tmp_path / "relabeled_checkpoint.pt").exists()


def test_missing_semantic_field_never_uses_current_default(model, tmp_path):
    # n_demand equals its default here, so only strict field checking can reject it.
    model = MFM(replace(model.config, n_demand=ModelConfig().n_demand))
    path = save_pretrained(model, tmp_path / "model")
    metadata = json.loads((path / "config.json").read_bytes())
    del metadata["model_config"]["n_demand"]
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError):
        from_pretrained(path)


def test_incomplete_weights_do_not_leave_random_training_components(model, tmp_path):
    path = save_pretrained(model, tmp_path / "model")
    state = {
        key: value
        for key, value in load_file(path / "model.safetensors").items()
        if not key.startswith("struct_head.")
    }
    _rebind_bundle(path, state=state)
    with pytest.raises(ValueError):
        from_pretrained(path)


def test_unknown_semantics_are_rejected():
    payload = json.loads(json.dumps(ModelConfig().to_dict()))
    payload["arch"] = "pooled"
    with pytest.raises(ValueError):
        ModelConfig.from_dict(payload)
    del payload["arch"]
    # Version 1 carried architecture options and loss weights; its meaning differs.
    for version in (1, 99):
        payload["format_version"] = version
        with pytest.raises(ValueError):
            ModelConfig.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 7},
        {"n_heads": True},
        {"n_controls": 0},
        {"flow_hidden": 1.5},
    ],
)
def test_unusable_or_architecture_disabling_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    "head,version",
    [
        ("head", torch.tensor(4)),
        ("conditional_head", torch.tensor(4)),
        ("head", torch.tensor(5.0)),
    ],
)
def test_trajectory_versions_cannot_reinterpret_equal_shaped_weights(
    model, tmp_path, head, version
):
    path = save_pretrained(model, tmp_path / "model")
    state = dict(model.state_dict())
    state[f"{head}.trajectory_feature_version"] = version
    with pytest.raises(RuntimeError):
        model.load_state_dict(state)
    _rebind_bundle(path, state=state)
    with pytest.raises((ValueError, RuntimeError)):
        from_pretrained(path)


def test_mixed_numerical_dtype_and_frozen_parameters_survive(model, tmp_path):
    model = model.double()
    model.struct_head.float()
    original = next(model.parameters())
    original.requires_grad_(False)
    loaded = from_pretrained(save_pretrained(model, tmp_path / "model"))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    assert not next(loaded.parameters()).requires_grad
    assert loaded.head.trajectory_feature_version.dtype == torch.int64
    assert loaded.conditional_head.trajectory_feature_version.item() == 5


@pytest.mark.parametrize(
    "section,field",
    [
        ("representation", "feature_blocks"),
        ("targets", "ordered_blocks"),
        ("graph", "families"),
    ],
)
def test_equal_width_schema_reordering_cannot_reinterpret_weights(model, tmp_path, section, field):
    path = save_pretrained(model, tmp_path / "model")
    metadata = json.loads((path / "config.json").read_bytes())
    entries = metadata["schemas"][section][field]
    entries[0], entries[1] = entries[1], entries[0]
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError):
        from_pretrained(path)


@pytest.mark.parametrize(
    "field", ["schemas", "provenance", "compatibility", "inference", "producer"]
)
def test_missing_required_metadata_is_rejected(model, tmp_path, field):
    path = save_pretrained(model, tmp_path / "model")
    metadata = json.loads((path / "config.json").read_bytes())
    del metadata[field]
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError):
        from_pretrained(path)


@pytest.mark.parametrize("filename", ["config.json", "manifest.json", "model.safetensors"])
def test_incomplete_bundle_fails_closed(model, tmp_path, filename):
    path = save_pretrained(model, tmp_path / "model")
    (path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        from_pretrained(path)


@pytest.mark.parametrize("filename", ["config.json", "manifest.json", "model.safetensors"])
def test_corrupt_bundle_is_not_loaded(model, tmp_path, filename):
    path = save_pretrained(model, tmp_path / "model")
    data = bytearray((path / filename).read_bytes())
    data[-2] ^= 1
    (path / filename).write_bytes(data)
    with pytest.raises(ValueError):
        from_pretrained(path)


@pytest.mark.parametrize("field", ["architecture", "compatibility", "inference"])
def test_incompatible_metadata_cannot_reinterpret_valid_weights(model, tmp_path, field):
    path = save_pretrained(model, tmp_path / "model")
    metadata = json.loads((path / "config.json").read_bytes())
    if field == "architecture":
        metadata[field] = "not-the-canonical-architecture"
    elif field == "compatibility":
        metadata[field]["torch"] = ">=99"
    else:
        metadata[field]["ode_method"] = "heun"
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError):
        from_pretrained(path)


def _pretend_installed(monkeypatch, versions):
    """Make loading and export see ``versions`` as the installed package versions."""
    import pymc_mfm.inference.checkpoint as checkpoint

    monkeypatch.setattr(
        checkpoint, "version", lambda name: versions.get(name) or installed_version(name)
    )


#: Package versions the loader sees as installed, and the producer versions of a
#: bundle an earlier release exported.
INSTALLED = {"pymc-mfm": "0.0.2", "numpy": "2.4.6", "torch": "2.12.0", "safetensors": "0.8.0"}
PRODUCER = {"pymc-mfm": "0.0.1.dev0", "numpy": "2.4.6", "torch": "2.12.0", "safetensors": "0.8.0"}


def _rerecord(path, compatibility):
    """Re-sign ``path`` as a ``PRODUCER`` export that recorded ``compatibility``."""
    metadata = json.loads((path / "config.json").read_bytes())
    metadata["compatibility"].update(compatibility)
    metadata["producer"] = dict(PRODUCER)
    _rebind_bundle(path, metadata=metadata)
    return metadata


def test_recorded_upper_bounds_never_make_a_bundle_unloadable(model, tmp_path, monkeypatch):
    import pymc_mfm.inference.checkpoint as checkpoint

    path = save_pretrained(model, tmp_path / "model")
    # An upper bound the producing release recorded, which the installed version exceeds.
    _rerecord(path, {"pymc-mfm": ">=0.0.1.dev0,<0.0.2"})
    _pretend_installed(monkeypatch, INSTALLED)
    from_pretrained(path)
    # The installed release's own ranges still bound the environment...
    _pretend_installed(monkeypatch, {**INSTALLED, "torch": "3.0.0"})
    with pytest.raises(ValueError):
        from_pretrained(path)
    # ...until a later release relaxes them; the recorded torch<3 is not enforced.
    monkeypatch.setattr(
        checkpoint, "COMPATIBILITY", {**checkpoint.COMPATIBILITY, "torch": ">=2.6,<4"}
    )
    from_pretrained(path)


@pytest.mark.parametrize(
    "minimum,below",
    [(">=2.10,<3", "2.8.0"), (">2.10,<3", "2.10.0")],
    ids=["inclusive", "exclusive"],
)
def test_bundles_need_the_lower_bounds_they_record(model, tmp_path, monkeypatch, minimum, below):
    path = save_pretrained(model, tmp_path / "model")
    _rerecord(path, {"torch": minimum})
    _pretend_installed(monkeypatch, INSTALLED)
    from_pretrained(path)
    # Inside the installed release's range (>=2.6,<3) but not above the recorded minimum.
    _pretend_installed(monkeypatch, {**INSTALLED, "torch": below})
    with pytest.raises(ValueError):
        from_pretrained(path)


#: One edit per rule; each is rejected by that rule alone.
INVALID_COMPATIBILITY = {
    "comma_only": lambda metadata: metadata["compatibility"].update(torch=","),
    "blank": lambda metadata: metadata["compatibility"].update(torch=" "),
    "malformed": lambda metadata: metadata["compatibility"].update(torch="~=2"),
    "pinned": lambda metadata: metadata["compatibility"].update(torch="==2.12.0"),
    "not_text": lambda metadata: metadata["compatibility"].update(torch=2.6),
    "missing_package": lambda metadata: metadata["compatibility"].pop("numpy"),
    "extra_package": lambda metadata: metadata["compatibility"].update(scipy=">=1"),
    "noncanonical_producer": lambda metadata: metadata["producer"].update(torch="v2.12.0"),
    "unknown_build_producer": lambda metadata: metadata["producer"].update(
        torch="2.12.0+example.host"
    ),
    # A torch build label, which only torch may carry.
    "build_label_on_numpy": lambda metadata: metadata["producer"].update(numpy="2.4.6+cpu"),
    # The installed 0.0.2 meets the recorded minimum; the producer does not.
    "producer_below_its_range": lambda metadata: metadata["compatibility"].update(
        {"pymc-mfm": ">=0.0.2"}
    ),
    # Upper bounds are not enforced on the installed version, but bind the producer.
    "producer_above_its_range": lambda metadata: metadata["compatibility"].update(
        torch=">=2.6,<2.10"
    ),
}


@pytest.mark.parametrize("mutate", INVALID_COMPATIBILITY.values(), ids=INVALID_COMPATIBILITY.keys())
def test_recorded_compatibility_holds_bounds_that_its_producer_meets(
    model, tmp_path, monkeypatch, mutate
):
    path = save_pretrained(model, tmp_path / "model")
    metadata = _rerecord(path, {})
    _pretend_installed(monkeypatch, INSTALLED)
    from_pretrained(path)
    mutate(metadata)
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError):
        from_pretrained(path)


def test_export_refuses_an_environment_its_release_does_not_support(model, tmp_path, monkeypatch):
    _pretend_installed(monkeypatch, {"torch": "3.0.0"})
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "model")
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize(
    "installed",
    [{"torch": "2.12.0+example.host"}, {"numpy": "2.4.6+cpu"}],
    ids=["unknown_torch_label", "label_on_another_package"],
)
def test_versions_no_export_may_record_still_load_bundles(model, tmp_path, monkeypatch, installed):
    path = save_pretrained(model, tmp_path / "model")
    # Inside the supported range, but a local label no export may record: only torch
    # may carry one, and only a build label.
    _pretend_installed(monkeypatch, installed)
    from_pretrained(path)
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_the_python_version_comes_from_the_interpreter(model, tmp_path, monkeypatch):
    import pymc_mfm.inference._artifact_schema as schema

    # A CPython built from an untagged commit reports "3.13.9+", no PEP 440 version.
    monkeypatch.setattr(platform, "python_version", lambda: "3.13.9+")
    from_pretrained(save_pretrained(model, tmp_path / "model"))
    # A release candidate is a PEP 440 pre-release.
    candidate = SimpleNamespace(major=3, minor=14, micro=0, releaselevel="candidate", serial=1)
    monkeypatch.setattr(schema, "sys", SimpleNamespace(version_info=candidate))
    assert schema.python_version() == "3.14.0rc1"


def test_dependencies_admit_only_environments_the_package_supports():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    dependencies = {
        requirement.name: requirement.specifier
        for requirement in map(Requirement, project["dependencies"])
    }
    packages = ("numpy", "torch", "safetensors")
    assert {name: dependencies[name] for name in packages} == {
        name: SpecifierSet(COMPATIBILITY[name]) for name in packages
    }
    assert SpecifierSet(project["requires-python"]) == SpecifierSet(COMPATIBILITY["python"])


@pytest.mark.parametrize(
    "mutation", ["unexpected", "shape", "integer", "dtype_catalog", "nonfinite"]
)
def test_tensor_contract_is_stricter_than_load_state_dict(model, tmp_path, mutation):
    path = save_pretrained(model, tmp_path / "model")
    state = load_file(path / "model.safetensors")
    name = "encoder.input_proj.weight"
    if mutation == "unexpected":
        state["extra.weight"] = torch.ones(1)
    elif mutation == "shape":
        state[name] = state[name][:-1].contiguous()
    elif mutation == "integer":
        state[name] = state[name].to(torch.int64)
    elif mutation == "dtype_catalog":
        state[name] = state[name].double()
    else:
        state[name][0, 0] = float("nan")
    _rebind_bundle(path, state=state)
    with pytest.raises(ValueError):
        from_pretrained(path)


def test_export_never_discards_extra_modules_or_runtime_settings(model, tmp_path):
    model.extra_training_module = torch.nn.Identity()
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "model")
    del model.extra_training_module
    # A runtime demand width the recorded configuration does not describe.
    model.head.n_demand = model.config.n_demand + 1
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_source_identity_survives_reexport_with_actual_export_hashes(model, tmp_path):
    provenance = {
        "source_checkpoint_sha256": "a" * 64,
        "source_step": 62500,
        "tracking_run_id": "run-17",
    }
    original_path = save_pretrained(
        model, tmp_path / "original", provenance=provenance, profile="A"
    )
    original = from_pretrained(original_path)
    copy_path = save_pretrained(original, tmp_path / "copy")
    restored = from_pretrained(copy_path)
    assert restored.artifact_metadata["provenance"] == original.artifact_metadata["provenance"]
    assert (
        restored.provenance["checkpoint_sha256"]
        == original.provenance["checkpoint_sha256"]
        == "a" * 64
    )
    for instance, path in ((original, original_path), (restored, copy_path)):
        exported = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
        assert instance.provenance["exported_checkpoint_sha256"] == exported
        assert exported != "a" * 64
        manifest = json.loads((path / "manifest.json").read_bytes())
        identity = manifest.pop("bundle_sha256")
        assert (
            instance.provenance["bundle_sha256"]
            == identity
            == hashlib.sha256(_json_bytes(manifest)).hexdigest()
        )


def test_reexport_annotations_preserve_existing_source_and_run_identity(model, tmp_path):
    source = {"source_checkpoint_sha256": "a" * 64, "training_run_id": "training-17"}
    original = from_pretrained(
        save_pretrained(model, tmp_path / "original", provenance=source, profile="A")
    )
    copied = from_pretrained(
        save_pretrained(
            original,
            tmp_path / "copy",
            provenance={"training_run_id": "training-17", "tracking_run_id": "backend-17"},
            profile="A",
        )
    )
    assert copied.provenance["checkpoint_sha256"] == "a" * 64
    assert copied.provenance["training_run_id"] == "training-17"
    assert copied.provenance["tracking_run_id"] == "backend-17"
    assert copied.provenance["profile"] == "A"
    for name, value in original.state_dict().items():
        torch.testing.assert_close(copied.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("record_source", [False, True])
def test_reexport_cannot_replace_loaded_source_or_producing_run(model, tmp_path, record_source):
    source = {
        "training_run_id": "training-17",
        "tracking_local_run_id": "local-17",
        "tracking_run_id": "backend-17",
    }
    if record_source:
        source["source_checkpoint_sha256"] = "a" * 64
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "original", provenance=source, profile="A")
    )
    for provenance in (
        {"source_checkpoint_sha256": "f" * 64},
        {"training_run_id": "other-training"},
        {"tracking_local_run_id": "other-local"},
        {"tracking_run_id": "other-backend"},
    ):
        with pytest.raises(ValueError):
            save_pretrained(loaded, tmp_path / "conflict", provenance=provenance)
        assert not (tmp_path / "conflict").exists()
    with pytest.raises(ValueError):
        save_pretrained(loaded, tmp_path / "other-profile", profile="B")
    assert not (tmp_path / "other-profile").exists()
    # The loaded identity is kept with the loaded metadata, not the editable provenance.
    loaded.provenance = {}
    with pytest.raises(ValueError):
        save_pretrained(
            loaded, tmp_path / "conflict", provenance={"source_checkpoint_sha256": "f" * 64}
        )
    assert not (tmp_path / "conflict").exists()


@pytest.mark.parametrize("change", ["bfloat16", "scaled_tensor", "one_ulp_cleared_provenance"])
def test_changed_weights_never_inherit_the_loaded_identity(model, tmp_path, change):
    source = {"source_checkpoint_sha256": "a" * 64, "training_evaluation": _training_evaluation()}
    loaded = from_pretrained(
        save_pretrained(model, tmp_path / "original", provenance=source, profile="A")
    )
    if change == "bfloat16":
        loaded = loaded.to(torch.bfloat16)
    elif change == "scaled_tensor":
        with torch.no_grad():
            next(tensor for tensor in loaded.conditional_head.parameters() if tensor.any()).mul_(
                1.05
            )
    else:
        with torch.no_grad():
            weight = loaded.encoder.input_proj.weight
            weight[0, 0] = torch.nextafter(weight[0, 0], torch.tensor(math.inf))
        # Clearing the editable provenance does not hide the change.
        loaded.provenance = {}
    with pytest.raises(ValueError):
        save_pretrained(loaded, tmp_path / "inherited")
    assert not (tmp_path / "inherited").exists()
    # Stated provenance replaces, and need not agree with, the loaded identity.
    stated = {"source_checkpoint_sha256": "b" * 64}
    copy = from_pretrained(save_pretrained(loaded, tmp_path / "stated", provenance=stated))
    assert copy.artifact_metadata["provenance"] == {"status": "recorded", **stated}
    assert copy.provenance["checkpoint_sha256"] == "b" * 64
    # The loaded profile label is neither inherited nor enforced.
    assert copy.artifact_metadata["profile"] is None
    relabelled = from_pretrained(
        save_pretrained(loaded, tmp_path / "relabelled", provenance={}, profile="A2")
    )
    assert relabelled.artifact_metadata["profile"] == "A2"
    assert relabelled.artifact_metadata["provenance"] == {"status": "not_recorded"}


def test_recording_missing_provenance_survives_empty_reexport_annotations(model, tmp_path):
    original = from_pretrained(save_pretrained(model, tmp_path / "original"))
    assert original.artifact_metadata["provenance"]["status"] == "not_recorded"
    recorded = from_pretrained(
        save_pretrained(original, tmp_path / "recorded", provenance={"tracking_run_id": "run-17"})
    )
    assert recorded.artifact_metadata["provenance"]["status"] == "recorded"
    copied = from_pretrained(save_pretrained(recorded, tmp_path / "copied", provenance={}))
    assert copied.artifact_metadata["provenance"]["status"] == "recorded"
    assert copied.provenance["tracking_run_id"] == "run-17"


def test_export_refuses_to_modify_existing_bundle_or_neighbor_data(model, tmp_path):
    path = save_pretrained(model, tmp_path / "model")
    saved = {file.name: file.read_bytes() for file in path.iterdir()}
    (path / "observations.csv").write_text("private observations")
    with pytest.raises(FileExistsError):
        save_pretrained(model, path)
    for name, content in saved.items():
        assert (path / name).read_bytes() == content
    assert (path / "observations.csv").read_text() == "private observations"


@pytest.mark.parametrize(
    "provenance",
    [
        {"token": "do-not-store"},
        {"prior": {"config": {"api_key": "do-not-store"}}},
        {"data": [{"uri": "https://user:password@example.org/private"}]},
        {"diagnostics": [{"uri": "https://example.org/report?access_token=do-not-store"}]},
        {"diagnostics": [{"uri": "https://example.org/report#access_token=do-not-store"}]},
        {"observations": [[1.0, 2.0]]},
        {"tracking_run_id": "hf_doNotPersistThisCredential"},
    ],
)
def test_credentials_and_observation_payloads_are_never_exported(model, tmp_path, provenance):
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "model", provenance=provenance)
    assert not (tmp_path / "model").exists()


def test_duplicate_json_keys_and_pickle_files_are_never_accepted(model, tmp_path, monkeypatch):
    path = save_pretrained(model, tmp_path / "model")
    manifest = path / "manifest.json"
    manifest.write_text(
        manifest.read_text().replace(
            '"format_version": 1,',
            '"format_version": 1, "format_version": 1,',
            1,
        )
    )
    with pytest.raises(ValueError):
        from_pretrained(path)
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, checkpoint)

    def no_pickle(*args, **kwargs):
        pytest.fail("inference must never deserialize a pickle checkpoint")

    monkeypatch.setattr(torch, "load", no_pickle)
    with pytest.raises(FileNotFoundError):
        from_pretrained(checkpoint)


def test_loading_does_not_advance_the_sampling_rng(model, tmp_path):
    path = save_pretrained(model, tmp_path / "model")
    before = torch.get_rng_state()
    from_pretrained(path)
    assert torch.equal(torch.get_rng_state(), before)


def test_failed_serialization_never_exposes_partial_bundle(model, tmp_path, monkeypatch):
    import pymc_mfm.inference.checkpoint as checkpoint

    def fail_save(tensors, filename, **kwargs):
        from pathlib import Path

        Path(filename).write_bytes(b"incomplete weights")
        raise OSError("disk unavailable")

    monkeypatch.setattr(checkpoint, "save_file", fail_save)
    with pytest.raises(OSError):
        save_pretrained(model, tmp_path / "model")
    assert list(tmp_path.iterdir()) == []


def test_negative_training_seed_remains_reproducible_provenance(model, tmp_path):
    provenance = {"training_seed": -17, "source_step": 4}
    restored = from_pretrained(save_pretrained(model, tmp_path / "model", provenance=provenance))
    assert restored.provenance["training_seed"] == -17
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "invalid", provenance={"source_step": -1})


def _training_evaluation():
    metrics = {
        "joint/query/decomposition_energy_score": 0.25,
        "joint/query/contribution_crps": 0.125,
        "joint/query/contribution_scaled_mae": 0.0625,
        "conditional/context/latent_sales_r2": -0.5,
        "structure_head/cy/auroc": None,
        "joint_router/edge/zb/n_pos": 3.0,
        "joint_router/graph/cc_two_cycle_rate": 0.0,
        "counts/worlds": 2.0,
    }
    aggregation = dict.fromkeys(metrics, "world")
    aggregation["joint/query/contribution_crps"] = "slot"
    aggregation["joint/query/contribution_scaled_mae"] = "slot_pooled"
    aggregation["structure_head/cy/auroc"] = "pooled_edges"
    aggregation["joint_router/edge/zb/n_pos"] = "pooled_edges"
    return {
        "status": "recorded",
        "source": "terminal in-training evaluation",
        "step": 12,
        "leg": "holdout",
        "head": "joint",
        "eval_seed": 616161,
        "n_worlds": 2,
        "n_samples": 4,
        "flow_steps": 16,
        "worlds": [{"shard_seed": 540000, "row": 3}, {"shard_seed": 541000, "row": 0}],
        "metrics": metrics,
        "aggregation": aggregation,
    }


def _rename_metric(record, key):
    record["metrics"][key] = record["metrics"].pop("counts/worlds")
    record["aggregation"][key] = record["aggregation"].pop("counts/worlds")


INVALID_TRAINING_EVALUATIONS = {
    "missing_field": lambda record: record.pop("flow_steps"),
    "unknown_field": lambda record: record.update(notes="free text"),
    "unrecorded_status": lambda record: record.update(status="not_recorded"),
    "blank_source": lambda record: record.update(source=" "),
    "unknown_leg": lambda record: record.update(leg="prior_shift"),
    "unknown_head": lambda record: record.update(head="structure_head"),
    "boolean_seed": lambda record: record.update(eval_seed=True),
    "negative_step": lambda record: record.update(step=-1),
    "float_draw_count": lambda record: record.update(n_samples=4.0),
    "world_count": lambda record: record["worlds"].pop(),
    "world_fields": lambda record: record["worlds"][0].update(cell=1),
    "negative_row": lambda record: record["worlds"][0].update(row=-1),
    "no_metrics": lambda record: (record["metrics"].clear(), record["aggregation"].clear()),
    "unaggregated_metric": lambda record: record["aggregation"].pop("counts/worlds"),
    "integer_metric": lambda record: record["metrics"].update({"counts/worlds": 2}),
    "nonfinite_metric": lambda record: record["metrics"].update({"counts/worlds": float("nan")}),
    "unknown_aggregation": lambda record: record["aggregation"].update({"counts/worlds": "mean"}),
    "list_aggregation": lambda record: record["aggregation"].update({"counts/worlds": ["world"]}),
    "leg_prefixed_key": lambda record: _rename_metric(record, "holdout/counts/worlds"),
    "newline_terminated_key": lambda record: _rename_metric(record, "counts/worlds\n"),
    "unknown_head_key": lambda record: _rename_metric(record, "router/graph/shd_mean"),
}


@pytest.mark.parametrize(
    "mutate", INVALID_TRAINING_EVALUATIONS.values(), ids=INVALID_TRAINING_EVALUATIONS.keys()
)
def test_malformed_training_evaluation_is_never_exported(model, tmp_path, mutate):
    record = _training_evaluation()
    mutate(record)
    with pytest.raises(ValueError):
        save_pretrained(model, tmp_path / "model", provenance={"training_evaluation": record})
    assert not (tmp_path / "model").exists()


def test_training_evaluation_survives_reload_and_is_revalidated_on_load(model, tmp_path):
    record = _training_evaluation()
    path = save_pretrained(model, tmp_path / "model", provenance={"training_evaluation": record})
    assert from_pretrained(path).artifact_metadata["provenance"]["training_evaluation"] == record
    metadata = json.loads((path / "config.json").read_bytes())
    # Loading uses re.fullmatch: a trailing newline must not satisfy the key pattern's "$".
    _rename_metric(metadata["provenance"]["training_evaluation"], "counts/worlds\n")
    _rebind_bundle(path, metadata=metadata)
    with pytest.raises(ValueError):
        from_pretrained(path)
