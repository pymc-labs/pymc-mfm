"""Compare complete-schedule and previous planning models on matched worlds."""

import argparse
import json

import numpy as np
import torch
from evaluate_planning_ols import evaluate
from train_schedule import make_data, tensors

from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.planning import synthetic_plan_outcome
from pymc_mfm.budget.planning_io import load_planning_model
from pymc_mfm.budget.schedule_planning_io import load_schedule_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--worlds", type=int, default=2048)
    p.add_argument("--hidden", type=float, default=0)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--step", type=float, default=5)
    args = p.parse_args()
    if args.worlds < 2 or args.hidden < 0 or args.iterations < 1 or args.step <= 0:
        p.error("invalid configuration")
    torch.set_num_threads(4)
    data = make_data(args.worlds, args.seed, 4, hidden=args.hidden)
    models = {
        "previous": load_planning_model(args.previous),
        "schedule": load_schedule_model(args.schedule),
    }
    baseline = load_controlled_budget_model(args.baseline)
    report = {
        name: evaluate(model, baseline, data, args.iterations, args.step)
        for name, model in models.items()
    }
    s, y, z, total, lo, hi, strength, half, state, factor, slope = tensors(data)
    with torch.no_grad():
        values = {
            name: synthetic_plan_outcome(
                model(s, y, z, total, lo, hi), strength, half, state, factor, slope
            ).numpy()
            for name, model in models.items()
        }
    delta = values["schedule"] - values["previous"]
    report["paired_schedule_vs_previous"] = {
        "mean_gain": float(delta.mean()),
        "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "win_fraction": float((delta > 1e-5).mean()),
        "loss_fraction": float((delta < -1e-5).mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
