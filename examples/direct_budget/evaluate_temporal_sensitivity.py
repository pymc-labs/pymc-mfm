"""Check whether trained budgets respond to reordering past observations."""

import argparse
import json

import numpy as np
import torch

from pymc_mfm.budget.diverse import generate
from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.temporal_io import load_temporal_budget_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--worlds", type=int, default=1000)
    args = p.parse_args()
    if args.worlds < 2:
        p.error("worlds must be at least 2")
    torch.set_num_threads(4)
    data = generate(args.worlds, seed=args.seed, hidden=0)
    tensors = [
        torch.tensor(data[k], dtype=torch.float32)
        for k in ("spend", "outcome", "controls", "budget", "lower", "upper")
    ]
    permutation = torch.tensor(
        list(reversed(range(tensors[0].shape[1] - 1))) + [tensors[0].shape[1] - 1]
    )
    report = {}
    models = {
        "temporal": load_temporal_budget_model(args.candidate),
        "reference": load_controlled_budget_model(args.reference),
    }
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=256)
    with torch.no_grad():
        for name, model in models.items():
            differences = []
            for x, y, z, budget, lower, upper in loader:
                first = model(x, y, z, budget, lower, upper)
                second = model(
                    x[:, permutation], y[:, permutation], z[:, permutation], budget, lower, upper
                )
                differences.append((first - second).abs().sum(-1))
            change = torch.cat(differences).numpy()
            report[name] = {
                "mean_l1_change": float(change.mean()),
                "fraction_above_0_0001": float(np.mean(change > 1e-4)),
                "max_l1_change": float(change.max()),
            }
    report["configuration"] = {"seed": args.seed, "worlds": args.worlds}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
