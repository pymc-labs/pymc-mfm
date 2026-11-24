"""Tests for direct multi-period budgets and synthetic planning utility."""

import numpy as np
import pytest
import torch

from pymc_mfm.budget.controlled import generate
from pymc_mfm.budget.planning import MultiPeriodBudgetModel, synthetic_plan_outcome


def inputs(n=3):
    d = generate(n, seed=123, positive_linear=True, structured=True)
    spend, outcome, controls, budget = [
        torch.tensor(d[k], dtype=torch.float32) for k in ("spend", "outcome", "controls", "budget")
    ]
    totals = budget[:, None] * torch.tensor([0.8, 1.0, 1.2])
    lo = torch.zeros(n, 3, 4)
    hi = totals[:, :, None].expand(-1, -1, 4).clone()
    hi[:, :, 0] = totals * 0.1
    return d, spend, outcome, controls, totals, lo, hi


def test_plan_budgets_and_channel_permutation():
    _, spend, outcome, controls, totals, lo, hi = inputs()
    model = MultiPeriodBudgetModel().eval()
    with torch.no_grad():
        allocation = model(spend, outcome, controls, totals, lo, hi)
        order = torch.tensor([2, 0, 3, 1])
        reordered = model(
            spend[:, :, order], outcome, controls, totals, lo[:, :, order], hi[:, :, order]
        )
    assert allocation.shape == (3, 3, 4)
    assert torch.allclose(allocation.sum(-1), totals, atol=2e-5)
    assert torch.all(allocation >= lo - 1e-5) and torch.all(allocation <= hi + 1e-5)
    assert torch.allclose(allocation[:, :, order], reordered, atol=2e-5)


def test_outcome_accumulates_carryover_and_gradients():
    alloc = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]], requires_grad=True)
    strength = torch.tensor([[3.0, 4.0]])
    half = torch.tensor([[1.0, 1.0]])
    state = torch.tensor([[0.5, 0.2]])
    factor = torch.tensor([[0.5, 0.25]])
    slope = torch.tensor([[0.0, -0.1]])
    result = synthetic_plan_outcome(alloc, strength, half, state, factor, slope)
    s = state.numpy()
    manual = 0
    for x in alloc.detach().numpy()[0]:
        s = (1 - factor.numpy()) * s + factor.numpy() * x
        manual += np.sum(strength.numpy() * s / (half.numpy() + s) + slope.numpy() * s)
    assert result.item() == pytest.approx(manual)
    result.sum().backward()
    assert torch.isfinite(alloc.grad).all() and alloc.grad.abs().sum() > 0


def test_plan_checkpoint_and_cli(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    from pymc_mfm.budget.planning_io import load_planning_model

    model = MultiPeriodBudgetModel().eval()
    path = tmp_path / "plan.safetensors"
    save_file(
        model.state_dict(),
        str(path),
        metadata={"architecture": "multi-period-budget-v1", "hidden": "64"},
    )
    loaded = load_planning_model(path)
    _, s, y, z, total, lo, hi = inputs(n=1)
    with torch.no_grad():
        assert torch.allclose(model(s, y, z, total, lo, hi), loaded(s, y, z, total, lo, hi))
    request = {
        "channels": ["TV", "Search", "Social", "Video"],
        "spend": s[0].tolist(),
        "outcome": y[0].tolist(),
        "controls": z[0].tolist(),
        "totals": total[0].tolist(),
        "lower": lo[0].tolist(),
        "upper": hi[0].tolist(),
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request))
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "examples/direct_budget/allocate_plan.py"),
        "--model",
        str(path),
        "--input",
        str(input_path),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    answer = json.loads(result.stdout)
    assert set(answer) == {"allocations"}
    for row, budget in zip(answer["allocations"], total[0].tolist(), strict=True):
        assert sum(row.values()) == pytest.approx(budget, abs=3e-5)
        assert row["TV"] <= 0.1 * budget + 1e-5


def test_planning_certificate_bounds_exhaustive_grid():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples/direct_budget"
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("eval_plan", root / "evaluate_planning.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        total = torch.ones(1, 2, dtype=torch.float64)
        lo = torch.zeros(1, 2, 2, dtype=torch.float64)
        hi = torch.ones_like(lo)
        initial = torch.full_like(lo, 0.5)
        params = tuple(
            torch.tensor(v, dtype=torch.float64)
            for v in ([[3.0, 4.0]], [[0.5, 1.0]], [[0.2, 0.3]], [[0.7, 0.4]], [[0.0, -0.1]])
        )
        solution, value, gap = module.solve(
            total, lo, hi, params, initial, iterations=400, step=0.5
        )
        grid = torch.linspace(0, 1, 201, dtype=torch.float64)
        a, b = torch.meshgrid(grid, grid, indexing="ij")
        plans = torch.stack((a, 1 - a, b, 1 - b), dim=-1).reshape(-1, 2, 2)
        truth = synthetic_plan_outcome(plans, *params).max()
        assert value.item() <= truth.item() + 1e-5
        assert value.item() + gap.item() >= truth.item() - 1e-8
        assert gap.item() < 0.01
        assert torch.allclose(solution.sum(-1), total, atol=1e-8)
    finally:
        sys.path.remove(str(root))


def test_ols_plan_obeys_each_period_limit():
    import importlib.util
    import sys
    from pathlib import Path

    from pymc_mfm.budget.controlled import generate

    root = Path(__file__).resolve().parents[1] / "examples/direct_budget"
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(
            "planning_ols", root / "evaluate_planning_ols.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = generate(30, seed=349, positive_linear=True, structured=True)
        ratios = np.array([0.8, 1.0, 1.2])
        data["totals"] = data["budget"][:, None] * ratios
        data["plan_lower"] = data["lower"][:, None, :] * ratios[None, :, None]
        data["plan_upper"] = data["upper"][:, None, :] * ratios[None, :, None]
        decision = module.ols_plan(data)
        np.testing.assert_allclose(decision.sum(-1), data["totals"], atol=1e-8)
        assert np.all(decision >= data["plan_lower"] - 1e-8)
        assert np.all(decision <= data["plan_upper"] + 1e-8)
    finally:
        sys.path.remove(str(root))


def test_linear_planning_ood_optimum_dominates_feasible_plans():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples/direct_budget"
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(
            "planning_ood", root / "evaluate_planning_ood.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.generate(30, seed=12345, hidden_strength=0.7)
        h = data["totals"].shape[1]
        weights = data["effect"][:, None, :] * (
            1 - data["decay"][:, None, :] ** np.arange(h, 0, -1)[None, :, None]
        )
        optimum = (data["totals"] * weights.max(-1)).sum(1)
        for share in np.linspace(0, 1, 11):
            plan = np.stack((data["totals"] * share, data["totals"] * (1 - share)), axis=-1)
            assert np.all(module.score(plan, data) <= optimum + 1e-8)
    finally:
        sys.path.remove(str(root))


def test_schedule_model_uses_future_totals_and_limits():
    from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel

    torch.manual_seed(7)
    _, s, y, z, _, _, _ = inputs(n=1)
    model = ScheduleBudgetModel().eval()
    totals = torch.tensor([[100.0, 80.0, 120.0]])
    other = torch.tensor([[100.0, 120.0, 80.0]])
    lo = torch.zeros(1, 3, 4)
    hi = totals[:, :, None].expand(-1, -1, 4).clone()
    hi_other = other[:, :, None].expand(-1, -1, 4).clone()
    with torch.no_grad():
        first = model(s, y, z, totals, lo, hi)
        second = model(s, y, z, other, lo, hi_other)
        altered = hi.clone()
        altered[:, 1, 0] = 10
        bounded = model(s, y, z, totals, lo, altered)
    assert torch.allclose(first.sum(-1), totals, atol=2e-5)
    assert torch.max(torch.abs(first[:, 0] - second[:, 0])) > 1e-6
    assert torch.max(torch.abs(first[:, 0] - bounded[:, 0])) > 1e-6
    assert bounded[0, 1, 0] <= 10.00001


def test_schedule_training_worlds_have_varying_feasible_bounds():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "examples/direct_budget"
    spec = importlib.util.spec_from_file_location("schedule_training", root / "train_schedule.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.make_data(200, seed=127, horizon=4, hidden=0.5)
    lo, hi, total = data["plan_lower"], data["plan_upper"], data["totals"]
    assert np.all(lo.sum(-1) <= total + 1e-10)
    assert np.all(hi.sum(-1) >= total - 1e-10)
    proportions = hi / total[:, :, None]
    assert np.any(abs(proportions[:, 0] - proportions[:, 1]) > 0.01)


def test_schedule_checkpoint_and_json_cli(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel
    from pymc_mfm.budget.schedule_planning_io import load_schedule_model

    model = ScheduleBudgetModel().eval()
    path = tmp_path / "schedule.safetensors"
    save_file(
        model.state_dict(),
        str(path),
        metadata={"architecture": "schedule-budget-v1", "hidden": "64"},
    )
    restored = load_schedule_model(path)
    _, s, y, z, total, lo, hi = inputs(n=1)
    with torch.no_grad():
        assert torch.allclose(model(s, y, z, total, lo, hi), restored(s, y, z, total, lo, hi))
    request = {
        "channels": ["TV", "Search", "Social", "Video"],
        "spend": s[0].tolist(),
        "outcome": y[0].tolist(),
        "controls": z[0].tolist(),
        "totals": total[0].tolist(),
        "upper": hi[0].tolist(),
    }
    path_in = tmp_path / "input.json"
    path_in.write_text(json.dumps(request))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/direct_budget/allocate_plan.py"),
            "--model",
            str(path),
            "--input",
            str(path_in),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    out = json.loads(result.stdout)
    assert set(out) == {"allocations"}
    for row, budget in zip(out["allocations"], total[0].tolist(), strict=True):
        assert sum(row.values()) == pytest.approx(budget, abs=3e-5)
        assert row["TV"] <= 0.1 * budget + 1e-5


def test_schedule_model_channel_permutation():
    from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel

    _, s, y, z, total, lo, hi = inputs(n=2)
    torch.manual_seed(88)
    model = ScheduleBudgetModel().eval()
    order = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        first = model(s, y, z, total, lo, hi)
        permuted = model(s[:, :, order], y, z, total, lo[:, :, order], hi[:, :, order])
    assert torch.allclose(first[:, :, order], permuted, atol=2e-5)


def test_schedule_ood_evaluator_cli(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from safetensors.torch import save_file

    from pymc_mfm.budget.controlled import ControlledBudgetModel
    from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel

    paths = [
        tmp_path / "schedule.safetensors",
        tmp_path / "previous.safetensors",
        tmp_path / "baseline.safetensors",
    ]
    values = [
        (ScheduleBudgetModel(), "schedule-budget-v1"),
        (MultiPeriodBudgetModel(), "multi-period-budget-v1"),
        (ControlledBudgetModel(joint_features=True), "controlled-budget-v2"),
    ]
    for path, (model, architecture) in zip(paths, values, strict=True):
        save_file(
            model.state_dict(), str(path), metadata={"architecture": architecture, "hidden": "64"}
        )
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "examples/direct_budget/evaluate_schedule_ood.py"),
        "--schedule",
        str(paths[0]),
        "--previous",
        str(paths[1]),
        "--baseline",
        str(paths[2]),
        "--seed",
        "101",
        "--worlds",
        "4",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    report = json.loads(result.stdout)
    assert set(report) == {"previous", "schedule", "paired_schedule_vs_previous"}
    assert report["schedule"]["max_budget_error"] < 1e-4
