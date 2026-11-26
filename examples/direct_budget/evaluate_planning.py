"""Evaluate multi-period budgets with a concavity-based optimality certificate."""

import argparse
import json

import numpy as np
import torch
from train_planning import make_data, myopic, tensors

from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.planning import synthetic_plan_outcome
from pymc_mfm.budget.planning_io import load_planning_model


def linear_maximizer(gradient, total, lower, upper):
    """Maximize a linear function on each bounded budget simplex."""
    order = gradient.argsort(dim=-1, descending=True)
    allocation = lower.clone()
    remainder = total - allocation.sum(-1)
    for j in range(gradient.shape[-1]):
        index = order[..., j : j + 1]
        current = allocation.gather(-1, index).squeeze(-1)
        cap = upper.gather(-1, index).squeeze(-1) - current
        addition = torch.minimum(remainder, cap).clamp_min(0)
        allocation.scatter_add_(-1, index, addition.unsqueeze(-1))
        remainder = remainder - addition
    return allocation


def certificate(allocation, total, lower, upper, params):
    """Upper-bound the optimum using a Frank-Wolfe gap for concave utility."""
    x = allocation.detach().requires_grad_(True)
    objective = synthetic_plan_outcome(x, *params)
    (grad,) = torch.autograd.grad(objective.sum(), x)
    extreme = linear_maximizer(grad, total, lower, upper)
    gap = (grad * (extreme - x)).sum((-1, -2)).clamp_min(0)
    return objective.detach(), gap.detach(), grad.detach()


def solve(total, lower, upper, params, initial, iterations, step):
    """Projected gradient search, with a bound even if the search stops early."""
    n, h, k = initial.shape
    x = initial.detach()
    for _ in range(iterations):
        _, _, grad = certificate(x, total, lower, upper, params)
        scores = ((x + step * grad) / total[:, :, None]).reshape(-1, k)
        x = (
            bounded_simplex(
                scores,
                (lower / total[:, :, None]).reshape(-1, k),
                (upper / total[:, :, None]).reshape(-1, k),
            ).reshape(n, h, k)
            * total[:, :, None]
        )
    final = synthetic_plan_outcome(x, *params)
    starting = synthetic_plan_outcome(initial, *params)
    x = torch.where((final >= starting)[:, None, None], x, initial)
    objective, gap, _ = certificate(x, total, lower, upper, params)
    return x, objective, gap


def evaluate(model, baseline, data, iterations, step):
    s, y, z, total, lo, hi, strength, half, state, factor, slope = tensors(data)
    # Float64 makes the certificate less sensitive to constraint rounding.
    params = tuple(x.double() for x in (strength, half, state, factor, slope))
    total64, lo64, hi64 = total.double(), lo.double(), hi.double()
    with torch.no_grad():
        plan = model(s, y, z, total, lo, hi).double()
        one = myopic(baseline, s, y, z, total, lo, hi).double()
        n, h, k = lo.shape

        def reproject(value):
            return (
                bounded_simplex(
                    (value / total64[:, :, None]).reshape(-1, k),
                    (lo64 / total64[:, :, None]).reshape(-1, k),
                    (hi64 / total64[:, :, None]).reshape(-1, k),
                ).reshape(n, h, k)
                * total64[:, :, None]
            )

        plan, one = reproject(plan), reproject(one)
        equal = (
            bounded_simplex(
                torch.full_like(lo64, 1 / k).reshape(-1, k),
                (lo64 / total64[:, :, None]).reshape(-1, k),
                (hi64 / total64[:, :, None]).reshape(-1, k),
            ).reshape(n, h, k)
            * total64[:, :, None]
        )
    # Start at the better feasible policy, so lower regret bounds are nonnegative.
    plan_value = synthetic_plan_outcome(plan, *params)
    one_value = synthetic_plan_outcome(one, *params)
    initial = torch.where((plan_value >= one_value)[:, None, None], plan, one)
    equal_value = synthetic_plan_outcome(equal, *params)
    initial_value = synthetic_plan_outcome(initial, *params)
    initial = torch.where((equal_value > initial_value)[:, None, None], equal, initial)
    _, candidate, gap = solve(total64, lo64, hi64, params, initial, iterations, step)
    plan_value = synthetic_plan_outcome(plan, *params)
    one_value = synthetic_plan_outcome(one, *params)
    equal_value = synthetic_plan_outcome(equal, *params)
    result = {}
    for name, value in (("plan", plan_value), ("one_period", one_value), ("equal", equal_value)):
        lower_regret = (candidate - value).clamp_min(0)
        upper_regret = (candidate + gap - value).clamp_min(0)
        result[name] = {
            "mean_regret_lower": float(lower_regret.mean()),
            "mean_regret_upper": float(upper_regret.mean()),
        }
    delta = (plan_value - one_value).numpy()
    result["paired_gain_vs_one_period"] = {
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
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--hidden", type=float, default=0)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--step", type=float, default=5)
    args = p.parse_args()
    if (
        args.worlds < 2
        or args.horizon < 1
        or args.iterations < 1
        or args.step <= 0
        or args.hidden < 0
    ):
        p.error("invalid evaluation configuration")
    torch.set_num_threads(4)
    data = make_data(args.worlds, args.seed, args.horizon, hidden=args.hidden)
    report = evaluate(
        load_planning_model(args.model),
        load_controlled_budget_model(args.baseline),
        data,
        args.iterations,
        args.step,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
