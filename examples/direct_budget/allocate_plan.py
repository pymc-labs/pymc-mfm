"""Allocate a sequence of budgets from local data and local model weights."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from pymc_mfm.budget.constraints import resolve_bounds
from pymc_mfm.budget.planning_io import load_planning_model
from pymc_mfm.budget.schedule_planning_io import load_schedule_model


def allocate(model, request):
    names = request["channels"]
    if not isinstance(names, list) or not names or not all(isinstance(x, str) and x for x in names):
        raise ValueError("channels must be a nonempty list of names")
    if len(set(names)) != len(names):
        raise ValueError("channel names must be unique")
    x = torch.tensor(request["spend"], dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(request["outcome"], dtype=torch.float32).unsqueeze(0)
    z = torch.tensor(request["controls"], dtype=torch.float32).unsqueeze(0)
    totals = torch.tensor(request["totals"], dtype=torch.float32).unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != len(names) or totals.ndim != 2:
        raise ValueError("invalid channel history or period totals")
    low, high = resolve_bounds(request, request["totals"], len(names), multi_period=True)
    lo = torch.tensor(low, dtype=torch.float32).unsqueeze(0)
    hi = torch.tensor(high, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        plan = model(x, y, z, totals, lo, hi)[0].tolist()
    return {"allocations": [dict(zip(names, row, strict=True)) for row in plan]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    with safe_open(str(args.model), framework="pt", device="cpu") as f:
        architecture = (f.metadata() or {}).get("architecture")
    if architecture == "multi-period-budget-v1":
        model = load_planning_model(args.model)
    elif architecture == "schedule-budget-v1":
        model = load_schedule_model(args.model)
    else:
        raise ValueError("unknown planning checkpoint architecture")
    print(json.dumps(allocate(model, json.loads(args.input.read_text())), indent=2))


if __name__ == "__main__":
    main()
