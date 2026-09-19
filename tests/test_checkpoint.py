"""Checkpoint semantics must survive round-tripping, not just tensor shapes."""

import json
from dataclasses import replace

import pytest
import torch

from pymc_mfm.inference.checkpoint import load_model, save_model
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM


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
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
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


def test_missing_semantic_field_never_uses_current_default(model, tmp_path):
    path = tmp_path / "model.pt"
    payload = {"config": model.config.to_dict(), "state_dict": model.state_dict()}
    del payload["config"]["flow_trajectory_correlation"]
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_model(path)


def test_incomplete_weights_do_not_leave_random_training_components(model, tmp_path):
    path = tmp_path / "model.pt"
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("struct_head.")
    }
    torch.save({"config": model.config.to_dict(), "state_dict": state}, path)
    with pytest.raises(RuntimeError):
        load_model(path)


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
    state = model.state_dict()
    state[f"{head}.trajectory_feature_version"] = version
    with pytest.raises(RuntimeError):
        model.load_state_dict(state)
    path = tmp_path / "incompatible.pt"
    torch.save({"config": model.config.to_dict(), "state_dict": state}, path)
    with pytest.raises(RuntimeError):
        load_model(path)
