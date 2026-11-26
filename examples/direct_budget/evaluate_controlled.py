"""Compare a direct decision model with equal and linear-OLS policies."""

import argparse
import json

import numpy as np
import torch

from pymc_mfm.budget.controlled import generate, response
from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.model import bounded_simplex


def evaluate(model, data):
    """Evaluate decisions against each world's exact constrained optimum."""
    x, y, z = (data[k] for k in ("spend", "outcome", "controls"))
    total, lower, upper = (data[k] for k in ("budget", "lower", "upper"))
    inputs = [torch.tensor(v, dtype=torch.float32) for v in (x, y, z, total, lower, upper)]
    _, _, _, b, lo, hi = inputs
    with torch.no_grad():
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*inputs), batch_size=256
        )
        prediction = torch.cat([model(*batch) for batch in loader]).numpy()
        equal = (
            bounded_simplex(torch.full_like(lo, 1 / lo.shape[1]), lo / b[:, None], hi / b[:, None])
            * b[:, None]
        ).numpy()
    design = np.concatenate((np.ones((*y.shape, 1)), x, z), axis=2)
    estimates = np.array(
        [
            np.linalg.lstsq(matrix, outcome, rcond=None)[0][1 : 1 + x.shape[-1]]
            for matrix, outcome in zip(design, y, strict=True)
        ]
    )
    # The OLS policy maximizes its linear prediction under the same bounds.
    ols = lower.copy()
    remainder = total - ols.sum(1)
    order = np.argsort(-estimates, axis=1)
    rows = np.arange(len(total))
    for j in range(x.shape[-1]):
        index = order[:, j]
        addition = np.minimum(remainder, upper[rows, index] - ols[rows, index])
        ols[rows, index] += addition
        remainder -= addition
    params = [data[key] for key in ("strength", "half", "carry", "factor", "slope")]
    optimum = response(data["target"], *params)
    regret = {
        name: optimum - response(values, *params)
        for name, values in (("network", prediction), ("equal", equal), ("ols", ols))
    }
    report = {
        name: {"mean_regret": float(values.mean()), "median_regret": float(np.median(values))}
        for name, values in regret.items()
    }
    for name in ("equal", "ols"):
        delta = regret[name] - regret["network"]
        report["paired_vs_" + name] = {
            "mean_improvement": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "win_fraction": float((delta > 1e-8).mean()),
            "loss_fraction": float((delta < -1e-8).mean()),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=4000)
    parser.add_argument("--confounding", type=float, default=0.0)
    parser.add_argument("--positive-linear", action="store_true")
    parser.add_argument("--structured", action="store_true")
    args = parser.parse_args()
    if args.worlds < 2:
        parser.error("worlds must be at least two")
    torch.set_num_threads(4)
    data = generate(
        args.worlds,
        seed=args.seed,
        confounding=args.confounding,
        positive_linear=args.positive_linear,
        structured=args.structured,
    )
    print(json.dumps(evaluate(load_controlled_budget_model(args.model), data), indent=2))


if __name__ == "__main__":
    main()
