"""Train and evaluate a direct allocation network on synthetic worlds.

Run with: python examples/direct_budget/train.py --output artifacts/direct-budget/model.safetensors
Weights are local research artifacts. The network is not trained on client data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from pymc_mfm.budget.model import DirectBudgetModel, bounded_simplex
from pymc_mfm.budget.synthetic import generate, utility


def tensors(data):
    return [
        torch.tensor(data[k], dtype=torch.float32)
        for k in ("spend", "outcome", "budget", "lower", "upper", "target")
    ]


def evaluate(model, data):
    s, y, b, lo, hi, target = tensors(data)
    model.eval()
    with torch.no_grad():
        pred = torch.cat(
            [model(*args) for args in DataLoader(TensorDataset(s, y, b, lo, hi), batch_size=512)]
        )
        equal = (
            bounded_simplex(torch.full_like(lo, 1 / lo.shape[-1]), lo / b[:, None], hi / b[:, None])
            * b[:, None]
        )
        # Recent mean spend, projected to the same constraints, is another
        # baseline; inputs are not chosen using the held-out outcomes.
        recent = s[:, -13:].mean(1)
        recent = (
            bounded_simplex(
                recent / recent.sum(-1, keepdim=True).clamp_min(1e-8),
                lo / b[:, None],
                hi / b[:, None],
            )
            * b[:, None]
        )
    true = utility(target.numpy(), data["strength"], data["half"], data["carry"], data["factor"])
    report = {}
    regrets = {}
    for name, alloc in (("network", pred), ("equal", equal), ("recent", recent)):
        arr = alloc.numpy()
        regret = true - utility(arr, data["strength"], data["half"], data["carry"], data["factor"])
        regrets[name] = regret
        report[name] = {
            "mean_regret": float(regret.mean()),
            "median_regret": float(np.median(regret)),
            "p90_regret": float(np.quantile(regret, 0.9)),
            "mean_allocation_mae": float(np.abs(arr - target.numpy()).mean()),
        }
    if "constraint_kind" in data:
        report["constraint_strata"] = {}
        for label, mask in (
            ("unconstrained", data["constraint_kind"] == 0),
            ("single_cap", data["constraint_kind"] == 1),
            ("box", data["constraint_kind"] >= 2),
        ):
            if mask.any():
                report["constraint_strata"][label] = {
                    "n": int(mask.sum()),
                    "network_mean_regret": float(regrets["network"][mask].mean()),
                    "equal_mean_regret": float(regrets["equal"][mask].mean()),
                }
    report["paired_vs_equal"] = {
        "mean_regret_reduction": float((regrets["equal"] - regrets["network"]).mean()),
        "standard_error": float(
            (regrets["equal"] - regrets["network"]).std(ddof=1) / np.sqrt(len(true))
        ),
        "fraction_lower_regret": float((regrets["network"] < regrets["equal"]).mean()),
    }
    report["feasibility"] = {
        "max_budget_error": float((pred.sum(-1) - b).abs().max()),
        "max_lower_violation": float((lo - pred).clamp_min(0).max()),
        "max_upper_violation": float((pred - hi).clamp_min(0).max()),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-worlds", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--objective", choices=("decision", "imitation"), default="decision")
    args = parser.parse_args()
    if args.epochs < 1 or args.train_worlds < 1 or args.threads < 1:
        parser.error("epochs, train-worlds and threads must be positive")
    if args.objective == "decision" and args.epochs < 4:
        parser.error("decision training requires at least four epochs")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train = generate(args.train_worlds, seed=args.seed)
    validation = generate(2000, seed=args.seed + 1)
    # Test seeds are not used for model selection.
    test = generate(4000, seed=args.seed + 2)
    stress = generate(4000, seed=args.seed + 3, confounding=0.5)
    loader = DataLoader(
        TensorDataset(
            *tensors(train),
            *(
                torch.tensor(train[key], dtype=torch.float32)
                for key in ("strength", "half", "carry", "factor")
            ),
        ),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = DirectBudgetModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    best, state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for s, y, b, lo, hi, target, strength, half, carry, factor in loader:
            prediction = model(s, y, b, lo, hi)
            if args.objective == "imitation" or epoch < 3:
                loss = (((prediction - target) / b[:, None]) ** 2).mean()
            else:
                effective = carry + factor * prediction
                loss = -(strength * effective / (half + effective)).sum(-1).mean() / 100
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = evaluate(model, validation)["network"]["mean_regret"]
        if score < best:
            best = score
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} val_mean_regret={score:.6f} best={best:.6f}", flush=True)
    model.load_state_dict(state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata={"architecture": "direct-budget-v2", "hidden": "64"})
    results = {
        "iid": evaluate(model, test),
        "confounding_stress": evaluate(model, stress),
        "test_worlds": len(test["budget"]),
        "test_seed": args.seed + 2,
        "stress_seed": args.seed + 3,
        "train_seed": args.seed,
        "train_worlds": args.train_worlds,
        "epochs": args.epochs,
        "objective": args.objective,
        "best_validation_regret": best,
    }
    output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
