"""Allocate a budget from a local JSON input using local research weights."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from pymc_mfm.budget.constraints import resolve_bounds
from pymc_mfm.budget.controlled import ControlledBudgetModel
from pymc_mfm.budget.io import load_budget_model
from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.residual import ResidualBudgetModel
from pymc_mfm.budget.residual_io import load_residual_budget_model
from pymc_mfm.budget.temporal_io import load_temporal_budget_model


def allocate(model, request):
    """Return per-channel budgets from an observation and constraint request."""
    request = dict(request)
    strategy = request.pop("constraint_strategy", "direct")
    if strategy not in ("direct", "project"):
        raise ValueError("constraint_strategy must be direct or project")
    channels = request["channels"]
    if (
        not isinstance(channels, list)
        or not channels
        or not all(isinstance(name, str) and name for name in channels)
    ):
        raise ValueError("channels must be a non-empty list of names")
    if len(set(channels)) != len(channels):
        raise ValueError("channel names must be unique")
    count = len(channels)
    spend = torch.tensor(request["spend"], dtype=torch.float32).unsqueeze(0)
    outcome = torch.tensor(request["outcome"], dtype=torch.float32).unsqueeze(0)
    budget = torch.tensor([request["budget"]], dtype=torch.float32)
    low, high = resolve_bounds(request, [request["budget"]], count, multi_period=False)
    lower = torch.tensor(low, dtype=torch.float32)
    upper = torch.tensor(high, dtype=torch.float32)
    if spend.ndim != 3 or spend.shape[-1] != count:
        raise ValueError("each spend row must have one entry per channel")
    with torch.no_grad():
        model_lower = lower if strategy == "direct" else torch.zeros_like(lower)
        model_upper = upper if strategy == "direct" else budget[:, None].expand_as(upper)
        if isinstance(model, (ControlledBudgetModel, ResidualBudgetModel)):
            if "controls" not in request:
                raise ValueError("controlled model requires an explicit controls array")
            controls = torch.tensor(request["controls"], dtype=torch.float32).unsqueeze(0)
            allocation = model(spend, outcome, controls, budget, model_lower, model_upper)
        else:
            allocation = model(spend, outcome, budget, model_lower, model_upper)
        if strategy == "project":
            allocation = budget[:, None] * bounded_simplex(
                allocation / budget[:, None], lower / budget[:, None], upper / budget[:, None]
            )
        values = allocation[0].tolist()
    return {"budget": float(budget[0]), "allocation": dict(zip(channels, values, strict=True))}


def load_model(path):
    """Load an explicitly tagged supported single-period checkpoint."""
    with safe_open(str(path), framework="pt", device="cpu") as file:
        architecture = (file.metadata() or {}).get("architecture")
    if architecture == "temporal-budget-v1":
        return load_temporal_budget_model(path)
    if architecture == "residual-budget-v1":
        return load_residual_budget_model(path)
    return load_budget_model(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.input.read_text())
    model = load_model(args.model)
    print(json.dumps(allocate(model, request), indent=2))


if __name__ == "__main__":
    main()
