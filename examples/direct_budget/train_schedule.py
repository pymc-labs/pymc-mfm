"""Train a direct multi-period allocation network on synthetic worlds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from pymc_mfm.budget.controlled import generate
from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.planning import synthetic_plan_outcome
from pymc_mfm.budget.planning_schedule import ScheduleBudgetModel


def make_data(n, seed, horizon, hidden=0):
    d = generate(n, seed=seed, positive_linear=True, structured=True, confounding=hidden)
    rng = np.random.default_rng(seed + 1000000)
    total = d["budget"][:, None] * rng.uniform(0.8, 1.2, (n, horizon))
    channels = d["spend"].shape[-1]
    anchor = rng.dirichlet(np.ones(channels) * 3, n * horizon).reshape(n, horizon, channels)
    lower = (
        np.maximum(0, anchor - rng.uniform(0.02, 0.3, (n, horizon, channels))) * total[:, :, None]
    )
    upper = (
        np.minimum(1, anchor + rng.uniform(0.02, 0.3, (n, horizon, channels))) * total[:, :, None]
    )
    kind = rng.integers(0, 4, (n, horizon))
    open_rows = kind < 2
    lower[open_rows] = 0
    upper[open_rows] = np.broadcast_to(total[:, :, None], upper.shape)[open_rows]
    row, period = np.where(kind == 1)
    channel = rng.integers(0, channels, len(row))
    upper[row, period, channel] = total[row, period] * rng.uniform(0.05, 0.25, len(row))
    d.update(
        totals=total, plan_lower=lower, plan_upper=upper, last_state=d["carry"] / (1 - d["factor"])
    )
    return d


KEYS = (
    "spend",
    "outcome",
    "controls",
    "totals",
    "plan_lower",
    "plan_upper",
    "strength",
    "half",
    "last_state",
    "factor",
    "slope",
)


def tensors(data):
    return [torch.tensor(data[k], dtype=torch.float32) for k in KEYS]


def myopic(single, s, y, z, total, lo, hi):
    n, horizon = total.shape
    k = s.shape[-1]
    return single(
        s.repeat_interleave(horizon, 0),
        y.repeat_interleave(horizon, 0),
        z.repeat_interleave(horizon, 0),
        total.reshape(-1),
        lo.reshape(-1, k),
        hi.reshape(-1, k),
    ).reshape(n, horizon, k)


def evaluate(model, baseline, data):
    all_values = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(*tensors(data)), batch_size=256):
            s, y, z, total, lo, hi, strength, half, state, factor, slope = batch
            plan = model(s, y, z, total, lo, hi)
            one = myopic(baseline, s, y, z, total, lo, hi)
            equal = bounded_simplex(
                torch.full_like(lo, 1 / lo.shape[-1]).flatten(0, 1),
                (lo / total[:, :, None]).flatten(0, 1),
                (hi / total[:, :, None]).flatten(0, 1),
            )
            equal = equal.reshape_as(lo) * total[:, :, None]
            args = (strength, half, state, factor, slope)
            vals = [synthetic_plan_outcome(a, *args) for a in (plan, one, equal)]
            all_values.append(torch.stack(vals, -1))
    values = torch.cat(all_values).numpy()
    report = {}
    for idx, name in ((1, "one_period_network"), (2, "equal")):
        delta = values[:, 0] - values[:, idx]
        report[name] = {
            "mean_outcome_gain": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "win_fraction": float((delta > 1e-5).mean()),
            "loss_fraction": float((delta < -1e-5).mean()),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-worlds", type=int, default=16000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.train_worlds < 4
        or args.train_worlds % 4
        or args.epochs < 1
        or args.horizon < 1
        or args.threads < 1
    ):
        parser.error("worlds, epochs, horizon and threads must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    # Mix four hidden-demand strengths. Hidden demand is not a model input.
    groups = [
        make_data(args.train_worlds // 4, args.seed + 100 * i, args.horizon, hidden=level)
        for i, level in enumerate((0, 0.25, 0.5, 0.75))
    ]
    train = {key: np.concatenate([group[key] for group in groups], axis=0) for key in groups[0]}
    val_groups = [
        make_data(500, args.seed + 100 * i + 1, args.horizon, hidden=level)
        for i, level in enumerate((0, 0.25, 0.5, 0.75))
    ]
    test = make_data(2000, args.seed + 2, args.horizon)
    hidden = make_data(2000, args.seed + 3, args.horizon, hidden=0.5)
    baseline = load_controlled_budget_model(args.baseline)
    model = ScheduleBudgetModel()
    model.single.load_state_dict(baseline.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
    loader = DataLoader(
        TensorDataset(*tensors(train)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best = -float("inf")
    chosen = None
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        for s, y, z, total, lo, hi, strength, half, state, factor, slope in loader:
            allocation = model(s, y, z, total, lo, hi)
            loss = (
                -synthetic_plan_outcome(allocation, strength, half, state, factor, slope).mean()
                / 100
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = np.mean(
            [
                evaluate(model, baseline, val)["one_period_network"]["mean_outcome_gain"]
                for val in val_groups
            ]
        )
        if score > best:
            best = score
            best_epoch = epoch + 1
            chosen = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} val_gain={score:.6f} best={best:.6f}", flush=True)
    model.load_state_dict(chosen)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(chosen, str(output), metadata={"architecture": "schedule-budget-v1", "hidden": "64"})
    result = {
        "seed": args.seed,
        "train_worlds": len(train["budget"]),
        "hidden_training_levels": [0, 0.25, 0.5, 0.75],
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "validation_gain": best,
        "horizon": args.horizon,
        "test": evaluate(model, baseline, test),
        "hidden_stress": evaluate(model, baseline, hidden),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
