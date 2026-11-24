"""Train a budget-only policy on mixed concave and linear carryover worlds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset
from train_controlled import evaluate, tensors
from train_diverse import GEOMETRIES, worlds

from pymc_mfm.budget.controlled import ControlledBudgetModel, oracle


def linear_worlds(n, seed, geometry, hidden=None):
    """Vary channel count, controls, carryover, effects, noise and limits."""
    k, weeks, controls = geometry
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, weeks, controls))
    latent = rng.normal(size=(n, weeks))
    persistence = rng.uniform(0, 0.8, (n, 1))
    for t in range(1, weeks):
        z[:, t] = persistence * z[:, t - 1] + np.sqrt(1 - persistence**2) * z[:, t]
        latent[:, t] = (
            persistence[:, 0] * latent[:, t - 1]
            + np.sqrt(1 - persistence[:, 0] ** 2) * latent[:, t]
        )
    hidden_level = rng.uniform(0, 0.8, n) if hidden is None else np.full(n, hidden)
    independent = rng.normal(size=(n, weeks, k))
    common = rng.normal(size=(n, weeks, 1))
    corr = rng.uniform(0, 0.85, (n, 1, 1))
    load = rng.normal(0, 0.25, (n, k, controls))
    variation = rng.uniform(0.15, 0.8, (n, 1, 1))
    log_spend = variation * (np.sqrt(1 - corr**2) * independent + corr * common)
    log_spend += (
        np.einsum("ntc,nkc->ntk", z, load) + 0.3 * hidden_level[:, None, None] * latent[:, :, None]
    )
    scale = rng.lognormal(3, 0.5, (n, 1, k))
    spend = scale * np.exp(log_spend)
    factor = rng.uniform(0.15, 0.95, (n, k))
    effect = rng.normal(0, 0.8, (n, k)) * rng.binomial(1, 0.75, (n, k))
    state = np.zeros((n, k))
    outcome = np.empty((n, weeks))
    for t in range(weeks):
        state = (1 - factor) * state + factor * spend[:, t]
        outcome[:, t] = (effect * state).sum(-1)
    outcome += 100 + np.einsum("ntc,nc->nt", z, rng.normal(0, 8, (n, controls)))
    outcome += hidden_level[:, None] * rng.uniform(0, 12, (n, 1)) * latent
    outcome += rng.normal(size=(n, weeks)) * rng.uniform(1, 6, (n, 1))
    budget = spend[:, -13:].sum(-1).mean(1) * rng.uniform(0.8, 1.2, n)
    anchor = rng.dirichlet(np.full(k, 3.0), n)
    lower = np.maximum(0, anchor - rng.uniform(0.02, 0.3, (n, k))) * budget[:, None]
    upper = np.minimum(1, anchor + rng.uniform(0.02, 0.3, (n, k))) * budget[:, None]
    cases = rng.integers(0, 3, n)
    lower[cases == 0], upper[cases == 0] = 0, budget[cases == 0, None]
    if k > 1:
        rows = np.flatnonzero(cases == 1)
        lower[rows] = 0
        upper[rows] = budget[rows, None]
        idx = rng.integers(0, k, len(rows))
        upper[rows, idx] = budget[rows] * rng.uniform(0.05, 0.25, len(rows))
    strength = np.zeros((n, k))
    half = np.ones((n, k))
    carry = np.zeros((n, k))
    slope = effect * factor
    ones = np.ones((n, k))
    target = oracle(strength, half, carry, ones, slope, budget, lower, upper)
    return dict(
        spend=spend,
        outcome=outcome,
        controls=z,
        budget=budget,
        lower=lower,
        upper=upper,
        target=target,
        strength=strength,
        half=half,
        carry=carry,
        factor=ones,
        slope=slope,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=27182)
    p.add_argument("--worlds-per-geometry", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.worlds_per_geometry < 2 or args.epochs < 4 or args.threads < 1:
        p.error("invalid training dimensions")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    geometries = (*GEOMETRIES, (2, 60, 3))
    training = []
    validation = []
    for i, geometry in enumerate(geometries):
        for generator in (worlds, linear_worlds):
            seed = args.seed + i * 1000 + (500 if generator is linear_worlds else 0)
            training.append(generator(args.worlds_per_geometry, seed, geometry))
            validation.append(generator(300, seed + 1, geometry))
    rng = torch.Generator().manual_seed(args.seed)
    loaders = [
        DataLoader(TensorDataset(*tensors(d)), batch_size=256, shuffle=True, generator=rng)
        for d in training
    ]
    model = ControlledBudgetModel(joint_features=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    best, state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        iters = [iter(loader) for loader in loaders]
        remaining = [len(loader) for loader in loaders]
        while sum(remaining):
            choices = [i for i, count in enumerate(remaining) if count]
            idx = choices[int(torch.randint(len(choices), (1,), generator=rng))]
            s, y, z, b, lo, hi, target, strength, half, carry, factor, slope = next(iters[idx])
            remaining[idx] -= 1
            prediction = model(s, y, z, b, lo, hi)
            if epoch < 3:
                loss = (((prediction - target) / b[:, None]) ** 2).mean()
            else:
                exposure = carry + factor * prediction
                loss = (
                    -(strength * exposure / (half + exposure) + slope * exposure).sum(-1).mean()
                    / 100
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = float(np.mean([evaluate(model, d)["network"]["mean_regret"] for d in validation]))
        if score < best:
            best, best_epoch = score, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} validation={score:.6f} best={best:.6f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata={"architecture": "controlled-budget-v2", "hidden": "64"})
    report = {
        "seed": args.seed,
        "worlds_per_generator_geometry": args.worlds_per_geometry,
        "geometries": geometries,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "validation_mean_regret": best,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
