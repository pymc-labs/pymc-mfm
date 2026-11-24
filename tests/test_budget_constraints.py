"""Tests for direct-budget business constraints and JSON workflows."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from pymc_mfm.budget.constraints import resolve_bounds
from pymc_mfm.budget.controlled import ControlledBudgetModel
from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel


def load_example(name):
    path = Path(__file__).resolve().parents[1] / "examples" / "direct_budget" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_intersection_of_reference_and_budget_shares():
    request = {
        "current": [20, 30, 30, 20],
        "max_change": 10,
        "upper_share": [0.1, 1, 1, 1],
        "lower": [0, 25, 0, 0],
    }
    lo, hi = resolve_bounds(request, [100], 4, multi_period=False)
    np.testing.assert_allclose(lo, [[10, 25, 20, 10]])
    np.testing.assert_allclose(hi, [[10, 40, 40, 30]])


def test_relative_change_and_multi_period_broadcast():
    request = {
        "current": [20, 30, 30, 20],
        "max_change_fraction": 0.5,
        "upper_share": [0.2, 1, 1, 1],
    }
    lo, hi = resolve_bounds(request, [80, 100, 120], 4, multi_period=True)
    np.testing.assert_allclose(lo, [[10, 15, 15, 10]] * 3)
    np.testing.assert_allclose(hi[:, 0], [16, 20, 24])
    assert np.all(lo.sum(1) <= np.array([80, 100, 120]))


def test_infeasible_and_invalid_limits_rejected():
    bad = [
        {"current": [25] * 4, "max_change": 10, "upper_share": [0.1, 1, 1, 1]},
        {"current": [25] * 4, "max_change": 10, "max_change_fraction": 0.2},
        {"current": [25] * 4},
        {"max_change": 10},
        {"upper_share": [0.1, 0.1, 0.1, 0.1]},
        {"current": [True, 25, 25, 25], "max_change": 10},
        {"max_change": -1, "current": [25] * 4},
        {"upper_share": [0.1, 1, 1, float("nan")]},
        {"upper_shares": [0.1, 1, 1, 1]},
    ]
    for request in bad:
        with pytest.raises(ValueError):
            resolve_bounds(request, [100], 4, multi_period=False)


def test_actual_models_obey_caps_and_reference_limits():
    single = load_example("allocate")
    plan = load_example("allocate_plan")
    torch.manual_seed(29)
    spend = (torch.rand(12, 4) * 30).tolist()
    outcome = (torch.rand(12) * 50).tolist()
    controls = (torch.rand(12, 2)).tolist()
    shared = {
        "channels": ["TV", "Search", "Social", "Video"],
        "spend": spend,
        "outcome": outcome,
        "controls": controls,
        "current": [19, 30, 30, 21],
        "max_change": 10,
        "upper_share": [0.1, 1, 1, 1],
    }
    with torch.no_grad():
        one = single.allocate(
            ControlledBudgetModel(joint_features=True).eval(), {**shared, "budget": 100}
        )
        many = plan.allocate(
            ScheduleBudgetModel().eval(), {**shared, "totals": [90, 100, 110, 100]}
        )
    rows = [(one["allocation"], 100), *zip(many["allocations"], [90, 100, 110, 100], strict=True)]
    for row, total in rows:
        assert sum(row.values()) == pytest.approx(total, abs=3e-5)
        assert row["TV"] <= 0.1 * total + 1e-5
        for name, old in zip(shared["channels"], shared["current"], strict=True):
            assert abs(row[name] - old) <= 10 + 1e-5


def test_fixed_reference_is_not_a_week_to_week_ramp():
    request = {"current": [20, 30, 30, 20], "max_change": 10}
    lo, hi = resolve_bounds(request, [100, 110], 4, multi_period=True)
    np.testing.assert_allclose(lo[0], lo[1])
    np.testing.assert_allclose(hi[0], hi[1])
    with pytest.raises(ValueError, match="boolean"):
        resolve_bounds({}, [100, True], 4, multi_period=True)


def test_named_share_cap_on_actual_single_and_multi_period_models():
    single = load_example("allocate")
    plan = load_example("allocate_plan")
    torch.manual_seed(71)
    shared = {
        "channels": ["TV", "Search", "Social", "Video"],
        "spend": (torch.rand(12, 4) * 30).tolist(),
        "outcome": (torch.rand(12) * 50).tolist(),
        "controls": (torch.rand(12, 2)).tolist(),
        "current": [19, 30, 30, 21],
        "max_change": 10,
        "upper_share": {"TV": 0.1},
    }
    one = single.allocate(
        ControlledBudgetModel(joint_features=True).eval(), {**shared, "budget": 100}
    )
    many = plan.allocate(ScheduleBudgetModel().eval(), {**shared, "totals": [90, 100, 110, 100]})
    for row, total in [
        (one["allocation"], 100),
        *zip(many["allocations"], [90, 100, 110, 100], strict=True),
    ]:
        assert sum(row.values()) == pytest.approx(total, abs=3e-5)
        assert row["TV"] <= 0.1 * total + 1e-5
        for name, old in zip(shared["channels"], shared["current"], strict=True):
            assert abs(row[name] - old) <= 10 + 1e-5


def test_named_share_cap_rejects_unknown_and_invalid_channel_names():
    with pytest.raises(ValueError, match="unknown upper_share channels"):
        resolve_bounds(
            {"channels": ["TV", "Search"], "upper_share": {"tv": 0.1}}, [100], 2, multi_period=False
        )
    with pytest.raises(ValueError, match="unique channel names"):
        resolve_bounds(
            {"channels": ["TV", "TV"], "upper_share": {"TV": 0.1}}, [100], 2, multi_period=False
        )
    with pytest.raises(ValueError, match="booleans"):
        resolve_bounds(
            {"channels": ["TV", "Search"], "upper_share": {"TV": True}},
            [100],
            2,
            multi_period=False,
        )


def test_posthoc_projection_uses_unconstrained_model_and_obeys_limits():
    single = load_example("allocate")

    class TrackingModel(torch.nn.Module):
        def forward(self, spend, outcome, budget, lower, upper):
            self.lower, self.upper = lower.clone(), upper.clone()
            return budget[:, None] * torch.tensor([[0.7, 0.1, 0.1, 0.1]])

    model = TrackingModel()
    request = {
        "channels": ["TV", "Search", "Social", "Video"],
        "spend": [[20, 30, 30, 20], [19, 30, 30, 21]],
        "outcome": [80, 90],
        "budget": 100,
        "upper_share": {"TV": 0.1},
        "constraint_strategy": "project",
    }
    result = single.allocate(model, request)
    assert torch.equal(model.lower, torch.zeros(1, 4))
    assert torch.equal(model.upper, torch.full((1, 4), 100.0))
    assert result["allocation"] == pytest.approx(
        {"TV": 10, "Search": 30, "Social": 30, "Video": 30}, abs=1e-4
    )
    assert request["constraint_strategy"] == "project"


def test_invalid_posthoc_strategy_and_infeasible_bounds_are_rejected():
    single = load_example("allocate")
    request = {
        "channels": ["TV", "Search"],
        "spend": [[20, 30], [20, 30]],
        "outcome": [80, 90],
        "budget": 100,
    }
    with pytest.raises(ValueError, match="constraint_strategy"):
        single.allocate(None, {**request, "constraint_strategy": "unknown"})
    with pytest.raises(ValueError, match="infeasible"):
        single.allocate(
            None, {**request, "constraint_strategy": "project", "upper_share": [0.1, 0.1]}
        )
