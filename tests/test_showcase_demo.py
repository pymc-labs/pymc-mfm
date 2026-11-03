"""The whitepaper demonstration must use the pinned worlds without future sales."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from demo import app as demo


def test_showcase_uses_pinned_worlds_with_one_week_held_out(monkeypatch):
    folder = Path(__file__).parents[1] / "demo/fixtures/paper-worlds"
    seen = []

    def fake_decompose(_model, observations, **kwargs):
        assert kwargs["window"] == "query"
        assert kwargs["n_samples"] == 256
        assert observations["spend"].shape == (208, 5)
        assert observations["controls"].shape == (208, 10)
        assert observations["context_outcome"].shape == (207,)
        seen.append(observations)
        return SimpleNamespace(
            components=np.ones((256, 1, 6)),
            component_names=("baseline", *(f"channel:Channel {chr(65 + i)}" for i in range(5))),
        )

    monkeypatch.setattr(demo, "model", lambda: object())
    monkeypatch.setattr(demo, "decompose", fake_decompose)
    client = TestClient(demo.app)
    for case in ("91", "92"):
        response = client.get(f"/api/showcase?case={case}")
        assert response.status_code == 200
        result = response.json()
        with np.load(folder / f"world-{case}.npz", allow_pickle=False) as source:
            np.testing.assert_array_equal(
                seen[-1]["context_outcome"], source["outcome_raw"][0, :207]
            )
            np.testing.assert_array_equal(
                [channel["truth"] for channel in result["channels"]],
                source["treatment_contribution_raw"][0, -1, :5],
            )
        assert result["context_weeks"] == 207
        assert result["query_week"] == 208
        assert len(result["channels"]) == 5
    assert client.get("/api/showcase?case=other").status_code == 422


def test_whitepaper_and_comparison_pages_are_available():
    client = TestClient(demo.app)
    assert "Which channels drive sales?" in client.get("/showcase").text
    assert "Can you predict the next 42 weeks?" in client.get("/compare").text
    assert "Same sales. Different story?" in client.get("/stress-test").text
