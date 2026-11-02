"""Behavioral checks for the synthetic demo and its observation boundary."""

import numpy as np
from fastapi.testclient import TestClient

from demo import app as demo
from demo.simulator import N_CONTEXT, N_WEEKS, simulate


def test_world_responds_to_channel_strength_without_changing_inputs():
    first = simulate((3.0, 2.0), seed=7)
    second = simulate((1.5, 2.0), seed=7)
    np.testing.assert_array_equal(first.spend, second.spend)
    np.testing.assert_allclose(first.channels[:, 0], second.channels[:, 0] * 2)
    np.testing.assert_array_equal(first.channels[:, 1], second.channels[:, 1])
    assert np.all(first.outcome > 0)


def test_api_only_sends_context_outcomes_and_summarizes_draws(monkeypatch):
    class Result:
        component_names = ("baseline", "channel:TV", "channel:Search", "control:event")
        components = np.broadcast_to(
            np.array([1000.0, 300.0, 200.0, 0.0]), (16, N_WEEKS - N_CONTEXT, 4)
        )

    def decompose_stub(_model, observations, **kwargs):
        assert observations["spend"].shape == (N_WEEKS, 2)
        assert observations["context_outcome"].shape == (N_CONTEXT,)
        assert observations["controls"].shape == (N_WEEKS, 10)
        assert observations["control_mask"] == [1, 1] + [0] * 8
        assert kwargs["window"] == "query"
        return Result()

    monkeypatch.setattr(demo, "model", lambda: object())
    monkeypatch.setattr(demo, "decompose", decompose_stub)
    client = TestClient(demo.app)
    response = client.post("/api/simulate", json={"n_samples": 16, "betas": [3, 2]})
    assert response.status_code == 200
    data = response.json()
    assert data["n_ctx"] == N_CONTEXT
    assert data["fit"]["query_weeks"][0] == N_CONTEXT
    assert data["fit"]["baseline"]["mean"] == [1000.0] * (N_WEEKS - N_CONTEXT)
    assert data["fit"]["contributions"][0]["mean"] == [300.0] * (N_WEEKS - N_CONTEXT)
    assert data["fit"]["revenue"]["mean"] == [1500.0] * (N_WEEKS - N_CONTEXT)
    assert client.post("/api/simulate", json={"n_samples": 129}).status_code == 422
    assert client.post("/api/simulate", json={"betas": [5, 1]}).status_code == 422
