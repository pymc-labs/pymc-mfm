"""Validate broader synthetic worlds and their decision oracle."""

import numpy as np
import pytest

from pymc_mfm.budget.diverse import generate


@pytest.mark.parametrize("channels,weeks,controls", [(2, 52, 1), (4, 64, 2), (6, 104, 3)])
def test_broader_worlds_are_finite_and_oracle_is_optimal(channels, weeks, controls):
    data = generate(300, seed=3001 + channels, channels=channels, weeks=weeks, controls=controls)
    x = data["target"]
    lower = data["lower"]
    upper = data["upper"]
    assert data["spend"].shape == (300, weeks, channels)
    assert data["controls"].shape == (300, weeks, controls)
    assert np.isfinite(data["outcome"]).all()
    np.testing.assert_allclose(x.sum(1), data["budget"], atol=1e-8)
    assert np.all(x >= lower - 1e-8) and np.all(x <= upper + 1e-8)
    marginal = (
        data["strength"]
        * data["half"]
        * data["factor"]
        / (data["half"] + data["carry"] + data["factor"] * x) ** 2
        + data["slope"] * data["factor"]
    )
    for i in range(channels):
        for j in range(channels):
            valid = (x[:, i] > lower[:, i] + 1e-7) & (x[:, j] < upper[:, j] - 1e-7)
            assert np.all(marginal[valid, j] <= marginal[valid, i] + 1e-7)


def test_hidden_intensity_can_be_fixed_and_is_not_control():
    a = generate(20, seed=77, hidden=0.4)
    np.testing.assert_allclose(a["hidden_level"], 0.4)
    assert a["controls"].shape[-1] == 2


@pytest.mark.parametrize("prior", ["structured", "diverse", "linear_ood"])
def test_paired_evaluator_returns_budgets_and_finite_scores(tmp_path, prior):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    from pymc_mfm.budget.controlled import ControlledBudgetModel

    model = tmp_path / "model.safetensors"
    save_file(
        ControlledBudgetModel(joint_features=True).state_dict(),
        str(model),
        metadata={"architecture": "controlled-budget-v2", "hidden": "64"},
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/direct_budget/evaluate_diverse.py"),
            "--candidate",
            str(model),
            "--reference",
            str(model),
            "--prior",
            prior,
            "--seed",
            "300",
            "--worlds",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    output = json.loads(result.stdout)
    assert output["paired_candidate_vs_reference"]["mean_gain"] == pytest.approx(0, abs=1e-10)
    assert np.isfinite(output["candidate"]["network"]["mean_regret"])


def test_posthoc_comparison_respects_bounds_and_recomputes_constrained_optimum():
    import importlib.util
    import sys
    from pathlib import Path

    from pymc_mfm.budget.controlled import ControlledBudgetModel

    example_dir = Path(__file__).resolve().parents[1] / "examples" / "direct_budget"
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "evaluate_posthoc", example_dir / "evaluate_posthoc.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = generate(20, seed=357, hidden=0.0)
        model = ControlledBudgetModel(hidden=16, joint_features=True).eval()
        result = module.compare(model, data)
    finally:
        sys.path.pop(0)
    assert result["feasibility"]["max_budget_error"] < 1e-4
    assert result["feasibility"]["max_bound_violation"] < 1e-4
    assert result["network"]["mean_regret"] >= -1e-4
    assert result["posthoc_projection"]["mean_regret"] >= -1e-4
    assert np.isfinite(result["conditioned_vs_posthoc"]["mean_outcome_gain"])


def test_posthoc_linear_stress_has_valid_constrained_oracle():
    import importlib.util
    import sys
    from pathlib import Path

    from pymc_mfm.budget.controlled import ControlledBudgetModel

    example_dir = Path(__file__).resolve().parents[1] / "examples" / "direct_budget"
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "evaluate_posthoc_ood", example_dir / "evaluate_posthoc_ood.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.generate_ood(20, seed=358, hidden_strength=0.0)
        model = ControlledBudgetModel(hidden=16, joint_features=True).eval()
        result = module.compare(model, data)
    finally:
        sys.path.pop(0)
    assert result["feasibility"]["max_budget_error"] < 1e-4
    assert result["feasibility"]["max_bound_violation"] < 1e-4
    assert result["conditioned"]["mean_regret"] >= -1e-4
    assert result["posthoc"]["mean_regret"] >= -1e-4
    assert result["ols"]["mean_regret"] >= -1e-8
    assert np.isfinite(result["conditioned_vs_posthoc"]["mean_gain"])


def test_linear_mix_oracle_matches_greedy_linear_optimum():
    import importlib.util
    import sys
    from pathlib import Path

    example_dir = Path(__file__).resolve().parents[1] / "examples" / "direct_budget"
    sys.path.insert(0, str(example_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "train_linear_mix", example_dir / "train_linear_mix.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.linear_worlds(30, 359, (4, 64, 2), hidden=0.3)
    finally:
        sys.path.pop(0)
    for i in range(30):
        greedy = data["lower"][i].copy()
        remaining = data["budget"][i] - greedy.sum()
        for channel in np.argsort(-data["slope"][i]):
            increment = min(remaining, data["upper"][i, channel] - greedy[channel])
            greedy[channel] += increment
            remaining -= increment
        np.testing.assert_allclose(data["target"][i].sum(), data["budget"][i], atol=1e-6)
        np.testing.assert_allclose(
            data["slope"][i] @ data["target"][i], data["slope"][i] @ greedy, atol=1e-6
        )
