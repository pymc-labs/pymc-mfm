"""Behavioral checks for direct residual budgets and safe checkpoints."""

import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.residual import ResidualBudgetModel
from pymc_mfm.budget.residual_io import load_residual_budget_model


def test_initial_budget_matches_projected_base_and_accepts_large_upper():
    torch.manual_seed(108)
    model = ResidualBudgetModel().eval()
    spend = torch.rand(4, 20, 3) * 10
    outcome = torch.randn(4, 20)
    controls = torch.randn(4, 20, 2)
    budget = torch.full((4,), 100.0)
    lower = torch.zeros(4, 3)
    upper = torch.tensor([[10.0, 200.0, 200.0]]).expand(4, -1)
    with torch.no_grad():
        actual = model(spend, outcome, controls, budget, lower, upper)
        unconstrained = model.base(
            spend, outcome, controls, budget, lower, budget[:, None].expand_as(upper)
        )
        expected = budget[:, None] * bounded_simplex(
            unconstrained / budget[:, None],
            lower,
            torch.minimum(upper, budget[:, None]) / budget[:, None],
        )
    assert torch.allclose(actual, expected, atol=2e-5)
    assert torch.allclose(actual.sum(1), budget, atol=2e-5)
    assert torch.all(actual[:, 0] <= 10.00002)
    with pytest.raises(ValueError, match="finite"):
        model(spend, outcome, controls, budget, lower, torch.full_like(upper, float("inf")))


def test_checkpoint_and_json_workflow_output_only_budgets(tmp_path):
    model = ResidualBudgetModel().eval()
    path = tmp_path / "residual.safetensors"
    save_file(
        model.state_dict(),
        str(path),
        metadata={"architecture": "residual-budget-v1", "hidden": "64"},
    )
    loaded = load_residual_budget_model(path)
    script = Path(__file__).resolve().parents[1] / "examples" / "direct_budget" / "allocate.py"
    spec = importlib.util.spec_from_file_location("allocate_residual_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = {
        "channels": ["TV", "Search"],
        "spend": [[10, 20], [12, 18]],
        "outcome": [20, 25],
        "controls": [[], []],
        "budget": 100,
        "upper_share": {"TV": 0.1},
    }
    assert isinstance(module.load_model(path), ResidualBudgetModel)
    result = module.allocate(loaded, request)
    assert set(result) == {"budget", "allocation"}
    assert set(result["allocation"]) == {"TV", "Search"}
    assert result["allocation"]["TV"] <= 10.00002
    assert sum(result["allocation"].values()) == pytest.approx(100, abs=2e-5)
    with pytest.raises(ValueError, match="controls"):
        module.allocate(loaded, {k: v for k, v in request.items() if k != "controls"})


def test_linear_replay_regret_respects_bounds(monkeypatch):
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "direct_budget"
    monkeypatch.syspath_prepend(str(example_dir))
    spec = importlib.util.spec_from_file_location(
        "train_residual_test", example_dir / "train_residual.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FixedBudget(torch.nn.Module):
        def forward(self, x, y, z, budget, lower, upper):
            return torch.tensor([[2.0, 8.0]])

    batch = (
        torch.ones(1, 2, 2),
        torch.ones(1, 2),
        torch.ones(1, 2, 1),
        torch.tensor([10.0]),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[8.0, 9.0]]),
        torch.tensor([[2.0, -1.0]]),
    )
    assert module.linear_regret(FixedBudget(), batch).item() == pytest.approx(18.0)


def test_interaction_training_scores_budget_not_predictions(monkeypatch):
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "direct_budget"
    monkeypatch.syspath_prepend(str(example_dir))
    spec = importlib.util.spec_from_file_location(
        "train_residual_interaction_test", example_dir / "train_residual.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FixedBudget(torch.nn.Module):
        def forward(self, x, y, z, budget, lower, upper):
            return torch.tensor([[2.0, 8.0]])

    batch = (
        torch.ones(1, 2, 2),
        torch.ones(1, 2),
        torch.ones(1, 2, 1),
        torch.tensor([10.0]),
        torch.zeros(1, 2),
        torch.full((1, 2), 10.0),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.zeros(1, 2),
        torch.tensor([100.0]),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    assert module.interaction_value(FixedBudget(), batch).item() == pytest.approx(16.0)
