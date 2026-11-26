"""Compare constrained decisions with projecting an unconstrained decision later."""

import argparse
import json

import numpy as np
import torch
from evaluate_controlled import evaluate

from pymc_mfm.budget.controlled import oracle, response
from pymc_mfm.budget.diverse import generate
from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.model import bounded_simplex


def compare(model, data):
    """Score identical histories with limits known during or after inference."""
    total = data["budget"]
    anchor = total[:, None] * np.array([0.19, 0.30, 0.30, 0.21])
    lower = np.maximum(anchor - 0.1 * total[:, None], 0)
    upper = np.minimum(anchor + 0.1 * total[:, None], total[:, None])
    upper[:, 0] = np.minimum(upper[:, 0], 0.1 * total)
    constrained = dict(data, lower=lower, upper=upper)
    params = [data[k] for k in ("strength", "half", "carry", "factor", "slope")]
    constrained["target"] = oracle(*params, total, lower, upper)
    report = evaluate(model, constrained)
    history = [torch.tensor(data[k], dtype=torch.float32) for k in ("spend", "outcome", "controls")]
    b, lo, hi = [torch.tensor(x, dtype=torch.float32) for x in (total, lower, upper)]
    conditioned, projected = [], []
    dataset = torch.utils.data.TensorDataset(*history, b, lo, hi)
    with torch.no_grad():
        for x, y, z, budget, low, high in torch.utils.data.DataLoader(dataset, batch_size=256):
            decision = model(x, y, z, budget, low, high)
            free = model(x, y, z, budget, torch.zeros_like(low), budget[:, None].expand_as(high))
            later = (
                bounded_simplex(
                    free / budget[:, None], low / budget[:, None], high / budget[:, None]
                )
                * budget[:, None]
            )
            conditioned.append(decision)
            projected.append(later)
    conditioned, projected = (torch.cat(v).numpy() for v in (conditioned, projected))
    best = response(constrained["target"], *params)
    delta = response(conditioned, *params) - response(projected, *params)
    report["posthoc_projection"] = {
        "mean_regret": float((best - response(projected, *params)).mean())
    }
    report["conditioned_vs_posthoc"] = {
        "mean_outcome_gain": float(delta.mean()),
        "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "win_fraction": float((delta > 1e-6).mean()),
        "loss_fraction": float((delta < -1e-6).mean()),
    }
    report["feasibility"] = {
        "max_budget_error": float(
            max(np.abs(a.sum(1) - total).max() for a in (conditioned, projected))
        ),
        "max_bound_violation": float(
            max(np.maximum(lower - a, a - upper).max() for a in (conditioned, projected))
        ),
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--hidden", type=float, default=0.0)
    args = p.parse_args()
    if args.worlds < 2 or not np.isfinite(args.hidden) or args.hidden < 0:
        p.error("worlds must be at least 2 and hidden finite and nonnegative")
    torch.set_num_threads(4)
    data = generate(args.worlds, seed=args.seed, hidden=args.hidden)
    result = compare(load_controlled_budget_model(args.model), data)
    result["configuration"] = {
        "seed": args.seed,
        "worlds": args.worlds,
        "hidden": args.hidden,
        "tv_cap": 0.1,
        "max_change_fraction_of_total": 0.1,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
