"""Train a direct budget model with observed controls and signed effects."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from pymc_mfm.budget.controlled import ControlledBudgetModel, generate, response
from pymc_mfm.budget.model import bounded_simplex

KEYS = (
    "spend",
    "outcome",
    "controls",
    "budget",
    "lower",
    "upper",
    "target",
    "strength",
    "half",
    "carry",
    "factor",
    "slope",
)


def tensors(data):
    return [torch.tensor(data[k], dtype=torch.float32) for k in KEYS]


def evaluate(model, data):
    s, y, z, b, lo, hi, target, *_ = tensors(data)
    model.eval()
    with torch.no_grad():
        pred = torch.cat(
            [model(*args) for args in DataLoader(TensorDataset(s, y, z, b, lo, hi), batch_size=512)]
        )
        equal = (
            bounded_simplex(torch.full_like(lo, 1 / lo.shape[-1]), lo / b[:, None], hi / b[:, None])
            * b[:, None]
        )
    params = [data[k] for k in ("strength", "half", "carry", "factor", "slope")]
    best = response(data["target"], *params)
    results = {}
    regrets = {}
    for name, value in (("network", pred.numpy()), ("equal", equal.numpy())):
        regret = best - response(value, *params)
        regrets[name] = regret
        results[name] = {
            "mean_regret": float(regret.mean()),
            "median_regret": float(np.median(regret)),
            "p90_regret": float(np.quantile(regret, 0.9)),
        }
    delta = regrets["equal"] - regrets["network"]
    results["paired"] = {
        "mean_improvement": float(delta.mean()),
        "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "win_fraction": float((delta > 1e-8).mean()),
        "loss_fraction": float((delta < -1e-8).mean()),
    }
    results["feasibility"] = {
        "max_total_error": float((pred.sum(-1) - b).abs().max()),
        "max_lower_violation": float((lo - pred).clamp_min(0).max()),
        "max_upper_violation": float((pred - hi).clamp_min(0).max()),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-worlds", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=220)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--positive-linear", action="store_true")
    parser.add_argument("--joint-features", action="store_true")
    parser.add_argument("--structured", action="store_true")
    args = parser.parse_args()
    if args.epochs < 4 or args.train_worlds < 1 or args.threads < 1:
        parser.error(
            "decision training needs at least four epochs and positive world and thread counts"
        )
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train = generate(
        args.train_worlds,
        seed=args.seed,
        positive_linear=args.positive_linear,
        structured=args.structured,
    )
    val = generate(
        2000, seed=args.seed + 1, positive_linear=args.positive_linear, structured=args.structured
    )
    test = generate(
        4000, seed=args.seed + 2, positive_linear=args.positive_linear, structured=args.structured
    )
    stress = generate(
        4000,
        seed=args.seed + 3,
        confounding=0.5,
        positive_linear=args.positive_linear,
        structured=args.structured,
    )
    loader = DataLoader(
        TensorDataset(*tensors(train)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = ControlledBudgetModel(joint_features=args.joint_features)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    best = float("inf")
    state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        for s, y, z, b, lo, hi, target, strength, half, carry, factor, slope in loader:
            allocation = model(s, y, z, b, lo, hi)
            if epoch < 3:
                loss = (((allocation - target) / b[:, None]) ** 2).mean()
            else:
                exposure = carry + factor * allocation
                loss = (
                    -(strength * exposure / (half + exposure) + slope * exposure).sum(-1).mean()
                    / 100
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = evaluate(model, val)["network"]["mean_regret"]
        if score < best:
            best = score
            best_epoch = epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} validation={score:.6f} best={best:.6f}", flush=True)
    model.load_state_dict(state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(output),
        metadata={
            "architecture": "controlled-budget-v2"
            if args.joint_features
            else "controlled-budget-v1",
            "hidden": "64",
        },
    )
    result = {
        "structured": args.structured,
        "joint_features": args.joint_features,
        "positive_linear": args.positive_linear,
        "train_seed": args.seed,
        "train_worlds": args.train_worlds,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "validation_regret": best,
        "test_seed": args.seed + 2,
        "stress_seed": args.seed + 3,
        "iid": evaluate(model, test),
        "hidden_confounding": evaluate(model, stress),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
