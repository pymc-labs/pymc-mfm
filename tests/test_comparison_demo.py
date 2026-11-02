"""Checks that the comparison uses the exact fitted dataset and aligned windows."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from demo import app as demo
from demo.comparison_world import N_CONTEXT


def test_fitted_world_and_river_observation_boundary(monkeypatch):
    fixture = json.loads((Path(__file__).parents[1] / "demo/fixtures/comparison.json").read_text())
    expected_world = fixture["world"]

    def fake_decompose(_model, observations, **kwargs):
        assert kwargs["window"] == "context"
        assert kwargs["n_samples"] == 128
        np.testing.assert_array_equal(
            observations["context_outcome"], expected_world["outcome"][:N_CONTEXT]
        )
        np.testing.assert_array_equal(
            observations["spend"], expected_world["spend"][: N_CONTEXT + 1]
        )
        assert observations["controls"].shape == (N_CONTEXT + 1, 10)
        np.testing.assert_array_equal(
            observations["controls"][:, 0], expected_world["promotion"][: N_CONTEXT + 1]
        )
        return SimpleNamespace(
            components=np.ones((128, N_CONTEXT, 3)),
            component_names=("baseline", "channel:TV", "channel:Search"),
        )

    monkeypatch.setattr(demo, "model", lambda: object())
    monkeypatch.setattr(demo, "decompose", fake_decompose)
    client = TestClient(demo.app)
    response = client.get("/api/comparison")
    assert response.status_code == 200
    body = response.json()
    assert body["weeks"] == N_CONTEXT
    assert body["river"]["channels"]["TV"]["mean"] == N_CONTEXT
    np.testing.assert_allclose(
        body["truth"]["channels"]["Search"],
        sum(row[1] for row in expected_world["channel_truth"][:N_CONTEXT]),
    )
    included = body["mmm"]["fits"]["with_promotion"]
    omitted = body["mmm"]["fits"]["without_promotion"]
    assert len(included["predicted_sales"]) == len(omitted["predicted_sales"]) == N_CONTEXT
    assert included["channels"]["Search"]["mean"] != omitted["channels"]["Search"]["mean"]
    assert included["diagnostics"]["divergences"] == omitted["diagnostics"]["divergences"] == 0
