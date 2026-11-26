"""Compare two direct budget policies on matched synthetic decisions."""

import argparse
import json

import numpy as np
import torch
from evaluate_controlled import evaluate as evaluate_prior
from evaluate_ood import evaluate as evaluate_ood
from evaluate_ood import generate_ood

from pymc_mfm.budget.controlled import generate as structured_generate
from pymc_mfm.budget.controlled import response
from pymc_mfm.budget.diverse import generate as diverse_generate
from pymc_mfm.budget.io import load_controlled_budget_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--prior", choices=("structured", "diverse", "linear_ood"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--worlds", type=int, default=4096)
    p.add_argument("--hidden", type=float, default=0)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--weeks", type=int, default=64)
    p.add_argument("--controls", type=int, default=2)
    args = p.parse_args()
    if (
        args.worlds < 2
        or args.hidden < 0
        or args.channels < 1
        or args.weeks < 2
        or args.controls < 0
    ):
        p.error("invalid evaluation configuration")
    torch.set_num_threads(4)
    if args.prior == "linear_ood":
        data = generate_ood(args.worlds, seed=args.seed, hidden_strength=args.hidden)
        fn = evaluate_ood
    else:
        options = dict(
            seed=args.seed, channels=args.channels, weeks=args.weeks, controls=args.controls
        )
        if args.prior == "structured":
            data = structured_generate(
                args.worlds,
                **options,
                confounding=args.hidden,
                positive_linear=True,
                structured=True,
            )
        else:
            data = diverse_generate(args.worlds, **options, hidden=args.hidden)
        fn = evaluate_prior
    models = {
        "candidate": load_controlled_budget_model(args.candidate),
        "reference": load_controlled_budget_model(args.reference),
    }
    report = {name: fn(model, data) for name, model in models.items()}
    s, y, z, b = [
        torch.tensor(data[key], dtype=torch.float32)
        for key in ("spend", "outcome", "controls", "budget")
    ]
    if args.prior == "linear_ood":
        lo = torch.zeros_like(s[:, 0])
        hi = b[:, None].expand_as(lo)
    else:
        lo = torch.tensor(data["lower"], dtype=torch.float32)
        hi = torch.tensor(data["upper"], dtype=torch.float32)
    predictions = {}
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(s, y, z, b, lo, hi), batch_size=256
    )
    with torch.no_grad():
        for name, model in models.items():
            predictions[name] = torch.cat([model(*batch) for batch in loader]).numpy()
    if args.prior == "linear_ood":
        values = {
            name: (allocation * data["marginal"]).sum(-1)
            for name, allocation in predictions.items()
        }
    else:
        params = [data[k] for k in ("strength", "half", "carry", "factor", "slope")]
        values = {name: response(allocation, *params) for name, allocation in predictions.items()}
    delta = values["candidate"] - values["reference"]
    report["paired_candidate_vs_reference"] = {
        "mean_gain": float(delta.mean()),
        "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "win_fraction": float((delta > 1e-6).mean()),
        "loss_fraction": float((delta < -1e-6).mean()),
    }
    report["configuration"] = {
        "prior": args.prior,
        "seed": args.seed,
        "worlds": args.worlds,
        "hidden": args.hidden,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
