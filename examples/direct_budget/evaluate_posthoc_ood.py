"""Stress-test preconditioned and post-projected budgets in a linear world."""

import argparse
import json

import numpy as np
import torch
from evaluate_ood import generate_ood

from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.model import bounded_simplex


def compare(model, data):
    """Compare identical histories under a feasible two-channel TV cap."""
    spend, outcome, controls, total, marginal = (
        data[k] for k in ("spend", "outcome", "controls", "budget", "marginal")
    )
    # Current allocation 20%/80%, change of at most 20% of the total,
    # and the first channel capped at 10%. The resulting feasible interval
    # for channel 1 is [0%, 10%]; channel 2 receives the remaining budget.
    low = np.stack((total * 0, total * 0.6), axis=1)
    high = np.stack((total * 0.1, total), axis=1)
    tensors = [
        torch.tensor(x, dtype=torch.float32) for x in (spend, outcome, controls, total, low, high)
    ]
    conditioned, projected = [], []
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=256)
    with torch.no_grad():
        for x, y, z, budget, lo, hi in loader:
            conditioned.append(model(x, y, z, budget, lo, hi))
            free = model(x, y, z, budget, torch.zeros_like(lo), budget[:, None].expand_as(hi))
            projected.append(
                bounded_simplex(free / budget[:, None], lo / budget[:, None], hi / budget[:, None])
                * budget[:, None]
            )
    conditioned, projected = (torch.cat(v).numpy() for v in (conditioned, projected))
    design = np.concatenate((np.ones((*outcome.shape, 1)), spend, controls), axis=2)
    coefficients = np.array(
        [np.linalg.lstsq(x, y, rcond=None)[0][1:3] for x, y in zip(design, outcome, strict=True)]
    )

    # Linear two-channel optimum: first channel gets its ceiling precisely when
    # its marginal effect exceeds the second channel's marginal effect.
    def allocation(score):
        first = np.where(score[:, 0] > score[:, 1], high[:, 0], low[:, 0])
        return np.stack((first, total - first), axis=1)

    optimum = (allocation(marginal) * marginal).sum(1)
    values = {
        "conditioned": (conditioned * marginal).sum(1),
        "posthoc": (projected * marginal).sum(1),
        "ols": (allocation(coefficients) * marginal).sum(1),
    }
    report = {k: {"mean_regret": float((optimum - v).mean())} for k, v in values.items()}
    for name in ("posthoc", "ols"):
        delta = values["conditioned"] - values[name]
        report["conditioned_vs_" + name] = {
            "mean_gain": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "win_fraction": float((delta > 1e-6).mean()),
            "loss_fraction": float((delta < -1e-6).mean()),
        }
    report["feasibility"] = {
        "max_budget_error": float(
            max(np.abs(a.sum(1) - total).max() for a in (conditioned, projected))
        ),
        "max_bound_violation": float(
            max(np.maximum(low - a, a - high).max() for a in (conditioned, projected))
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
    data = generate_ood(args.worlds, seed=args.seed, hidden_strength=args.hidden)
    result = compare(load_controlled_budget_model(args.model), data)
    result["configuration"] = {
        "seed": args.seed,
        "worlds": args.worlds,
        "hidden": args.hidden,
        "tv_cap": 0.1,
        "max_change_fraction_of_total": 0.2,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
