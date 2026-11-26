"""Behavioral checks for controlled direct allocation."""

import numpy as np
import pytest
import torch

from pymc_mfm.budget.controlled import ControlledBudgetModel, generate, oracle, response


def test_signed_oracle_beats_dense_grid_and_handles_flat_ties():
    a = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    h = np.full((2, 3), 0.5)
    carry = np.array([[0.2, 0.1, 0.3], [0.2, 0.3, 0.1]])
    factor = np.array([[0.5, 0.8, 0.7], [0.5, 0.8, 0.7]])
    slope = np.array([[0.0, -0.7, 0.0], [0.0, 0.0, 0.0]])
    lo = np.zeros((2, 3))
    hi = np.ones((2, 3))
    budget = np.ones(2)
    result = oracle(a, h, carry, factor, slope, budget, lo, hi)
    assert np.allclose(result.sum(-1), budget)
    grid = np.array(
        [(i / 100, j / 100, 1 - (i + j) / 100) for i in range(101) for j in range(101 - i)]
    )
    for row in range(2):
        value = response(result[row], a[row], h[row], carry[row], factor[row], slope[row])
        assert (
            value
            >= np.max(response(grid, a[row], h[row], carry[row], factor[row], slope[row])) - 1e-8
        )


def test_generated_worlds_have_feasible_oracle_and_signed_channels():
    d = generate(300, seed=18)
    assert set(np.unique(d["kind"])) == {0, 1, 2}
    assert np.allclose(d["target"].sum(1), d["budget"], atol=1e-8)
    assert np.all(d["target"] >= d["lower"] - 1e-8)
    assert np.all(d["target"] <= d["upper"] + 1e-8)


def test_output_respects_limits_with_or_without_controls_and_permutation():
    d = generate(5, seed=99)
    s, y, z, b, lo, hi = [
        torch.tensor(d[k], dtype=torch.float32)
        for k in ("spend", "outcome", "controls", "budget", "lower", "upper")
    ]
    model = ControlledBudgetModel().eval()
    with torch.no_grad():
        prediction = model(s, y, z, b, lo, hi)
        order = torch.tensor([2, 0, 3, 1])
        permuted = model(s[:, :, order], y, z, b, lo[:, order], hi[:, order])
        none = model(s, y, z[:, :, :0], b, lo, hi)
    assert torch.allclose(prediction.sum(-1), b, atol=2e-5)
    assert torch.all(prediction >= lo - 1e-5) and torch.all(prediction <= hi + 1e-5)
    assert torch.allclose(prediction[:, order], permuted, atol=2e-5)
    assert torch.allclose(none.sum(-1), b, atol=2e-5)
    with pytest.raises(ValueError, match="shape"):
        model(s, y, z[:, :-1], b, lo, hi)


def test_random_two_channel_oracles_match_grid():
    d = generate(100, seed=2324, channels=2)
    grid_fraction = np.linspace(0, 1, 20001)
    for i in range(len(d["budget"])):
        # Two bounds determine a closed feasible interval for channel one.
        left = max(d["lower"][i, 0], d["budget"][i] - d["upper"][i, 1])
        right = min(d["upper"][i, 0], d["budget"][i] - d["lower"][i, 1])
        x = left + (right - left) * grid_fraction
        grid = np.column_stack((x, d["budget"][i] - x))
        args = [d[key][i] for key in ("strength", "half", "carry", "factor", "slope")]
        actual = response(d["target"][i], *args)
        assert actual >= response(grid, *args).max() - 1e-7


def test_oracle_kkt_conditions_for_four_channels():
    d = generate(1000, seed=872)
    x = d["target"]
    marginal = (
        d["strength"] * d["half"] * d["factor"] / (d["half"] + d["carry"] + d["factor"] * x) ** 2
        + d["slope"] * d["factor"]
    )
    # Moving any small amount from channel i to j must not improve utility.
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            can_move = (x[:, i] > d["lower"][:, i] + 1e-7) & (x[:, j] < d["upper"][:, j] - 1e-7)
            assert np.all(marginal[can_move, j] <= marginal[can_move, i] + 1e-7)


def test_controlled_checkpoint_and_json_cli(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    path = tmp_path / "weights.safetensors"
    request = tmp_path / "request.json"
    save_file(
        ControlledBudgetModel().state_dict(),
        str(path),
        metadata={"architecture": "controlled-budget-v1", "hidden": "64"},
    )
    request.write_text(
        json.dumps(
            {
                "channels": ["TV", "search"],
                "spend": [[20, 80], [50, 50], [80, 20]],
                "outcome": [50, 70, 95],
                "controls": [[1], [2], [3]],
                "budget": 100,
                "upper": [10, 100],
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [
            sys.executable,
            str(root / "examples/direct_budget/allocate.py"),
            "--model",
            str(path),
            "--input",
            str(request),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    result = json.loads(run.stdout)
    assert set(result) == {"budget", "allocation"}
    assert result["allocation"]["TV"] <= 10.00001
    assert sum(result["allocation"].values()) == pytest.approx(100, abs=2e-5)


def test_positive_linear_prior_and_oracle_optimality():
    d = generate(1000, seed=2217, positive_linear=True)
    assert set(np.unique(d["kind"])) == {0, 1, 2, 3}
    x = d["target"]
    assert np.allclose(x.sum(-1), d["budget"], atol=1e-8)
    marginal = (
        d["strength"] * d["half"] * d["factor"] / (d["half"] + d["carry"] + d["factor"] * x) ** 2
        + d["slope"] * d["factor"]
    )
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            can_move = (x[:, i] > d["lower"][:, i] + 1e-7) & (x[:, j] < d["upper"][:, j] - 1e-7)
            assert np.all(marginal[can_move, j] <= marginal[can_move, i] + 1e-7)


def test_joint_model_permutation_and_checkpoint(tmp_path):
    from safetensors.torch import save_file

    from pymc_mfm.budget.io import load_budget_model

    d = generate(8, seed=914, positive_linear=True)
    s, y, z, b, lo, hi = [
        torch.tensor(d[k], dtype=torch.float32)
        for k in ("spend", "outcome", "controls", "budget", "lower", "upper")
    ]
    model = ControlledBudgetModel(joint_features=True).eval()
    path = tmp_path / "joint.safetensors"
    save_file(
        model.state_dict(),
        str(path),
        metadata={"architecture": "controlled-budget-v2", "hidden": "64"},
    )
    restored = load_budget_model(path)
    order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        first = model(s, y, z, b, lo, hi)
        second = restored(s, y, z, b, lo, hi)
        permuted = restored(s[:, :, order], y, z, b, lo[:, order], hi[:, order])
        controls_swapped = restored(s, y, z[:, :, [1, 0]], b, lo, hi)
    assert torch.allclose(first, second)
    assert torch.allclose(first[:, order], permuted, atol=2e-5)
    assert torch.allclose(first, controls_swapped, atol=2e-5)
    assert torch.allclose(first.sum(-1), b, atol=2e-5)


def test_joint_model_handles_constant_and_collinear_inputs():
    model = ControlledBudgetModel(joint_features=True).eval()
    spend = torch.ones(2, 8, 4) * 20
    outcome = torch.ones(2, 8) * 100
    controls = torch.ones(2, 8, 3)
    budget = torch.ones(2) * 100
    lower = torch.zeros(2, 4)
    upper = torch.ones(2, 4) * 100
    with torch.no_grad():
        result = model(spend, outcome, controls, budget, lower, upper)
    assert torch.isfinite(result).all()
    assert torch.allclose(result.sum(1), budget, atol=2e-5)


def test_controlled_evaluation_compares_feasible_policies():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/direct_budget/evaluate_controlled.py"
    spec = importlib.util.spec_from_file_location("budget_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = generate(25, seed=762, positive_linear=True)
    result = module.evaluate(ControlledBudgetModel(joint_features=True).eval(), data)
    for name in ("network", "equal", "ols"):
        assert result[name]["mean_regret"] >= -1e-4
    assert 0 <= result["paired_vs_ols"]["win_fraction"] <= 1


def test_out_of_prior_evaluator_matches_simple_oracle():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/direct_budget/evaluate_ood.py"
    spec = importlib.util.spec_from_file_location("budget_ood_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.generate_ood(50, seed=876, hidden_strength=0.7)
    allocation = np.stack((data["budget"], np.zeros(50)), axis=1)
    allocation2 = np.stack((np.zeros(50), data["budget"]), axis=1)
    max_at_vertices = np.maximum(
        (allocation * data["marginal"]).sum(1), (allocation2 * data["marginal"]).sum(1)
    )
    np.testing.assert_allclose(max_at_vertices, data["budget"] * data["marginal"].max(1))
    result = module.evaluate(ControlledBudgetModel(joint_features=True).eval(), data)
    assert all(result[key]["mean_regret"] >= -1e-4 for key in ("network", "ols", "equal"))
    assert result["max_budget_error"] < 1e-4


def test_structured_prior_retains_oracle_and_persistent_controls():
    d = generate(1000, seed=91223, structured=True, positive_linear=True)
    z = d["controls"]
    correlation = np.mean(z[:, 1:, :] * z[:, :-1, :])
    assert correlation > 0.2
    np.testing.assert_allclose(d["target"].sum(1), d["budget"], atol=1e-8)
    x = d["target"]
    marginal = (
        d["strength"] * d["half"] * d["factor"] / (d["half"] + d["carry"] + d["factor"] * x) ** 2
        + d["slope"] * d["factor"]
    )
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            valid = (x[:, i] > d["lower"][:, i] + 1e-7) & (x[:, j] < d["upper"][:, j] - 1e-7)
            assert np.all(marginal[valid, j] <= marginal[valid, i] + 1e-7)
