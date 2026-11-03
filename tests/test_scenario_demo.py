"""Check the forecast boundary for each scenario."""

import numpy as np
from fastapi.testclient import TestClient

from demo import app as demo
from demo.scenario_world import N_CONTEXT, N_WEEKS, make_scenario


def test_scenario_forecast_does_not_provide_future_sales_to_river(monkeypatch):
    observed = []

    def decompose(_model, observations, **kwargs):
        observed.append(observations)
        assert kwargs["window"] == "query"

        class Result:
            components = np.ones((128, N_WEEKS - N_CONTEXT, 7))

        return Result()

    monkeypatch.setattr(demo, "model", lambda: object())
    monkeypatch.setattr(demo, "decompose", decompose)
    client = TestClient(demo.app)
    for name in (
        "baseline",
        "promotion",
        "halo",
        "baseline_trend",
        "promotion_trend",
        "halo_trend",
    ):
        result = client.get("/api/scenario", params={"case": name}).json()
        source = make_scenario(name)
        np.testing.assert_array_equal(observed[-1]["context_outcome"], source.outcome[:N_CONTEXT])
        assert len(result["river"]["forecast"]["mean"]) == N_WEEKS - N_CONTEXT
        assert result["context_weeks"] == N_CONTEXT
        assert len(result["channels"]) == 6
    assert client.get("/api/scenario?case=bad").status_code == 422


def test_trend_scenarios_only_change_baseline_and_preserve_inputs():
    for name in ("baseline", "promotion", "halo"):
        ordinary = make_scenario(name)
        trended = make_scenario(name + "_trend")
        np.testing.assert_array_equal(ordinary.spend, trended.spend)
        np.testing.assert_array_equal(ordinary.controls, trended.controls)
        np.testing.assert_allclose(trended.outcome - ordinary.outcome, 10 * np.arange(N_WEEKS))
