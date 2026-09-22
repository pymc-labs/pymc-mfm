"""Checkpoint semantics must survive round-tripping, not just tensor shapes."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file

from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM


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
    model = MFM(
        replace(
            model.config,
            flow_ode_method="heun",
            flow_trajectory_correlation=0.8,
            struct_weight=0.7,
            struct_cy_weight=1.2,
            graph_conditional_weight=0.6,
            context_decomposition_weight=0.3,
            context_graph_weight=0.2,
        )
    ).double()
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
    for original, restored in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(restored.grad, original.grad, rtol=0, atol=0)
    assert loaded.config == model.config
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def test_missing_semantic_field_never_uses_current_default(model, tmp_path):
    path = save_pretrained(model, tmp_path / "model")
    metadata = json.loads((path / "config.json").read_bytes())
    del metadata["model_config"]["flow_trajectory_correlation"]
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
    payload["format_version"] = 2
    with pytest.raises(ValueError):
        ModelConfig.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 7},
        {"n_heads": True},
        {"n_controls": 0},
        {"flow_steps_sample": 1.5},
        {"flow_trajectory_correlation": 1.0},
        {"graph_conditional_weight": 0.0},
        {"context_graph_weight": float("nan")},
        {"flow_ode_method": "unknown"},
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
    model.head.ode_method = "heun"
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
