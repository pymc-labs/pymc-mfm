"""Tests for direct budget allocation and its independent synthetic oracle."""

import itertools

import numpy as np
import pytest
import torch

from pymc_mfm.budget.model import DirectBudgetModel, bounded_simplex
from pymc_mfm.budget.synthetic import generate, oracle, utility


def test_oracle_beats_exhaustive_grid():
    strength = np.array([[3.0, 7.0, 5.0]])
    half = np.array([[0.4, 0.6, 0.8]])
    lower, upper = np.array([[0.1, 0.0, 0.2]]), np.array([[0.6, 0.7, 0.8]])
    result = oracle(strength, half, np.array([1.0]), lower, upper)
    assert np.isclose(result.sum(), 1)
    assert np.all((lower <= result) & (result <= upper))
    grid = np.array(
        [
            x
            for x in itertools.product(np.arange(0, 1.01, 0.05), repeat=3)
            if abs(sum(x) - 1) < 1e-8
            and np.all(np.array(x) >= lower[0] - 1e-8)
            and np.all(np.array(x) <= upper[0] + 1e-8)
        ]
    )
    assert utility(result, strength, half)[0] >= utility(grid, strength, half).max() - 1e-10


def test_constraints_and_gradient():
    scores = torch.tensor([[0.4, 0.3, 0.3], [0.1, 0.9, -0.1]], requires_grad=True)
    lo = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.1, 0.2]])
    hi = torch.tensor([[0.2, 0.8, 0.8], [0.5, 0.6, 0.4]])
    allocation = bounded_simplex(scores, lo, hi)
    assert torch.allclose(allocation.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.all(allocation >= lo - 1e-6) and torch.all(allocation <= hi + 1e-6)
    (allocation * torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0


def test_model_outputs_budget_and_respects_permutation():
    d = generate(6, seed=10)

    def tensor(name):
        return torch.tensor(d[name], dtype=torch.float32)

    s, y, b, lo, hi = [tensor(k) for k in ("spend", "outcome", "budget", "lower", "upper")]
    model = DirectBudgetModel().eval()
    with torch.no_grad():
        out = model(s, y, b, lo, hi)
        order = torch.tensor([2, 0, 3, 1])
        reordered = model(s[:, :, order], y, b, lo[:, order], hi[:, order])
    assert out.shape == (6, 4)
    assert torch.allclose(out.sum(-1), b, atol=2e-5)
    assert torch.all(out >= lo - 1e-5) and torch.all(out <= hi + 1e-5)
    assert torch.allclose(reordered, out[:, order], atol=1e-5)


def test_infeasible_constraints_rejected():
    with pytest.raises(ValueError, match="infeasible"):
        bounded_simplex(torch.zeros(1, 3), torch.zeros(1, 3), torch.full((1, 3), 0.2))
    with pytest.raises(ValueError, match="bounds"):
        bounded_simplex(torch.zeros(1, 3), torch.full((1, 3), -0.1), torch.ones(1, 3))


def test_saved_model_roundtrip_and_bound_changes(tmp_path):
    from safetensors.torch import save_file

    from pymc_mfm.budget.io import load_direct_budget_model

    model = DirectBudgetModel().eval()
    path = tmp_path / "weights.safetensors"
    save_file(
        model.state_dict(), str(path), metadata={"architecture": "direct-budget-v2", "hidden": "64"}
    )
    loaded = load_direct_budget_model(path)
    spend = torch.rand(1, 12, 3) * 30
    outcome = torch.rand(1, 12) * 50
    total = torch.tensor([100.0])
    lo = torch.zeros(1, 3)
    unconstrained_hi = torch.full((1, 3), 100.0)
    capped_hi = torch.tensor([[10.0, 100.0, 100.0]])
    with torch.no_grad():
        original = model(spend, outcome, total, lo, unconstrained_hi)
        restored = loaded(spend, outcome, total, lo, unconstrained_hi)
        capped = loaded(spend, outcome, total, lo, capped_hi)
    assert torch.allclose(original, restored)
    assert capped[0, 0] <= 10.00001
    assert torch.allclose(capped.sum(-1), total, atol=1e-5)


def test_projection_invariant_to_large_offset():
    logits = torch.tensor([[1200.4, 1200.3, 1200.3]])
    lo, hi = torch.zeros_like(logits), torch.ones_like(logits)
    result = bounded_simplex(logits, lo, hi)
    assert torch.allclose(result.sum(-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(result, bounded_simplex(logits - 1200, lo, hi), atol=1e-6)


def test_oracle_with_carryover_matches_grid():
    strength = np.array([[5.0, 3.0]])
    half = np.array([[0.4, 0.6]])
    carry = np.array([[0.2, 0.05]])
    factor = np.array([[0.4, 0.8]])
    lo, hi = np.zeros((1, 2)), np.ones((1, 2))
    result = oracle(strength, half, np.ones(1), lo, hi, carry, factor)
    grid = np.column_stack((np.linspace(0, 1, 10001), np.linspace(1, 0, 10001)))
    assert np.isclose(result.sum(), 1)
    assert utility(result, strength, half, carry, factor)[0] >= (
        utility(grid, strength, half, carry, factor).max() - 1e-9
    )


def test_currency_unit_invariance():
    data = generate(3, seed=48)
    s, y, b, lo, hi = (
        torch.tensor(data[key], dtype=torch.float32)
        for key in ("spend", "outcome", "budget", "lower", "upper")
    )
    model = DirectBudgetModel().eval()
    with torch.no_grad():
        result = model(s, y, b, lo, hi)
        scaled = model(s * 100, y, b * 100, lo * 100, hi * 100)
    assert torch.allclose(result, scaled / 100, atol=2e-5)


def test_json_cli_end_to_end(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    model_path = tmp_path / "weights.safetensors"
    input_path = tmp_path / "input.json"
    save_file(
        DirectBudgetModel().state_dict(),
        str(model_path),
        metadata={"architecture": "direct-budget-v2", "hidden": "64"},
    )
    input_path.write_text(
        json.dumps(
            {
                "channels": ["TV", "search", "social"],
                "spend": [[20, 30, 50], [30, 20, 50], [40, 40, 20]],
                "outcome": [100, 110, 95],
                "budget": 100,
                "upper": [10, 100, 100],
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/direct_budget/allocate.py"),
            "--model",
            str(model_path),
            "--input",
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    output = json.loads(result.stdout)
    assert set(output) == {"budget", "allocation"}
    assert sum(output["allocation"].values()) == pytest.approx(100, abs=2e-5)
    assert output["allocation"]["TV"] <= 10.00001


def test_checkpoint_requires_matching_feature_semantics(tmp_path):
    from safetensors.torch import save_file

    from pymc_mfm.budget.io import load_direct_budget_model

    path = tmp_path / "wrong.safetensors"
    save_file(
        DirectBudgetModel().state_dict(),
        str(path),
        metadata={"architecture": "direct-budget-v1", "hidden": "64"},
    )
    with pytest.raises(ValueError, match="checkpoint metadata"):
        load_direct_budget_model(path)


def test_redundant_upper_bound_is_allowed():
    model = DirectBudgetModel().eval()
    spend = torch.ones(1, 8, 2)
    outcome = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    budget = torch.tensor([100.0])
    with torch.no_grad():
        capped = model(spend, outcome, budget, torch.zeros(1, 2), torch.full((1, 2), 100))
        redundant = model(spend, outcome, budget, torch.zeros(1, 2), torch.full((1, 2), 1000))
    assert torch.allclose(capped, redundant)


def test_large_score_projection_obeys_tight_limits():
    scores = torch.tensor([[3140.0, -2600.0, 3000.0, -8500.0]], requires_grad=True)
    lo = torch.tensor([[0.234567, 0.123456, 0.07, 0.02]])
    hi = torch.tensor([[0.5, 0.24, 0.6, 0.5]])
    allocation = bounded_simplex(scores, lo, hi)
    assert torch.allclose(allocation.sum(-1), torch.ones(1), atol=1e-7)
    assert torch.all(allocation >= lo - 1e-7)
    assert torch.all(allocation <= hi + 1e-7)
