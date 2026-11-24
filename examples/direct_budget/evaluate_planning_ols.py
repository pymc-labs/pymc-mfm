"""Compare direct planning, one-period policy and an observed-control OLS rule."""

import argparse
import json

import numpy as np
import torch
from evaluate_planning import solve
from train_planning import make_data, myopic, tensors

from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.planning import synthetic_plan_outcome
from pymc_mfm.budget.planning_io import load_planning_model


def ols_plan(data):
    """Maximize a constant linear fitted effect with the same period limits."""
    x, y, z = (data[k] for k in ("spend", "outcome", "controls"))
    totals, lower, upper = (data[k] for k in ("totals", "plan_lower", "plan_upper"))
    design = np.concatenate((np.ones((*y.shape, 1)), x, z), axis=2)
    slopes = np.array(
        [
            np.linalg.lstsq(matrix, outcome, rcond=None)[0][1 : 1 + x.shape[-1]]
            for matrix, outcome in zip(design, y, strict=True)
        ]
    )
    plan = lower.copy()
    remaining = totals - plan.sum(-1)
    order = np.argsort(-slopes, axis=1)
    rows = np.arange(len(x))[:, None]
    periods = np.arange(totals.shape[1])[None, :]
    for j in range(x.shape[-1]):
        channel = order[:, j, None]
        room = upper[rows, periods, channel] - plan[rows, periods, channel]
        addition = np.minimum(remaining, room)
        plan[rows, periods, channel] += addition
        remaining -= addition
    return plan


def evaluate(model, baseline, data, iterations, step):
    s, y, z, total, lo, hi, strength, half, state, factor, slope = tensors(data)
    params = tuple(v.double() for v in (strength, half, state, factor, slope))
    total64, lo64, hi64 = total.double(), lo.double(), hi.double()
    with torch.no_grad():
        plan = model(s, y, z, total, lo, hi).double()
        single = myopic(baseline, s, y, z, total, lo, hi).double()
        # OLS is computed in original float64 inputs but rounded to the
        # same float32 constraints as the neural outputs before projection.
        linear = torch.tensor(ols_plan(data), dtype=torch.float32).double()
    from pymc_mfm.budget.model import bounded_simplex

    n, h, k = lo.shape

    def reproject(x):
        return (
            bounded_simplex(
                (x / total64[:, :, None]).reshape(-1, k),
                (lo64 / total64[:, :, None]).reshape(-1, k),
                (hi64 / total64[:, :, None]).reshape(-1, k),
            ).reshape(n, h, k)
            * total64[:, :, None]
        )

    plans = [reproject(v) for v in (plan, single, linear)]
    values = [synthetic_plan_outcome(v, *params) for v in plans]
    which = torch.stack(values, dim=1).argmax(1)
    initial = torch.stack(plans, dim=1)[torch.arange(n), which]
    _, candidate, gap = solve(total64, lo64, hi64, params, initial, iterations, step)
    result = {}
    for name, value in zip(("plan", "one_period", "ols"), values, strict=True):
        result[name] = {
            "mean_regret_lower": float((candidate - value).clamp_min(0).mean()),
            "mean_regret_upper": float((candidate + gap - value).clamp_min(0).mean()),
        }
    for name, value in zip(("one_period", "ols"), values[1:], strict=True):
        delta = (values[0] - value).numpy()
        result["paired_gain_vs_" + name] = {
            "mean": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "wins": float((delta > 1e-5).mean()),
            "losses": float((delta < -1e-5).mean()),
        }
    result["certificate"] = {
        "mean_gap": float(gap.mean()),
        "max_gap": float(gap.max()),
        "fraction_gap_below_0_01": float((gap < 0.01).double().mean()),
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
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
    print(
        json.dumps(
            evaluate(
                load_planning_model(args.model),
                load_controlled_budget_model(args.baseline),
                data,
                args.iterations,
                args.step,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
