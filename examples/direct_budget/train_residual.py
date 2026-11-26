"""Train a residual budget policy using only the repository's synthetic priors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evaluate_ood import generate_ood
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset
from train_controlled import evaluate, tensors
from train_temporal import GEOMETRIES, worlds

from pymc_mfm.budget.controlled import generate as generate_controlled
from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.residual import ResidualBudgetModel
from pymc_mfm.budget.temporal_io import load_temporal_budget_model


def linear_data(n, seed, hidden):
    data = generate_ood(n, seed=seed, hidden_strength=hidden)
    x, y, z, b, m = [
        torch.tensor(data[k], dtype=torch.float32)
        for k in ("spend", "outcome", "controls", "budget", "marginal")
    ]
    current = x[:, -1] * b[:, None] / x[:, -1].sum(1, keepdim=True)
    g = torch.Generator().manual_seed(seed + 700000)
    fraction = 0.1 + 0.7 * torch.rand((n, 1), generator=g)
    constrained = torch.rand((n, 1), generator=g) < 0.5
    lo = torch.where(constrained, (1 - fraction) * current, torch.zeros_like(current))
    hi = torch.where(constrained, torch.minimum((1 + fraction) * current, b[:, None]), b[:, None])
    return TensorDataset(x, y, z, b, lo, hi, m)


def linear_regret(model, batch):
    x, y, z, b, lo, hi, m = batch
    allocation = model(x, y, z, b, lo, hi)
    first = torch.argmax(m, dim=1, keepdim=True)
    optimum = lo.clone()
    add = torch.minimum(b - lo.sum(1), (hi - lo).gather(1, first)[:, 0])
    optimum.scatter_add_(1, first, add[:, None])
    second = 1 - first
    optimum.scatter_add_(1, second, (b - optimum.sum(1))[:, None])
    return ((optimum - allocation) * m).sum(1)


def forever(loader):
    while True:
        yield from loader


class ProjectedBaseline(torch.nn.Module):
    """Fixed temporal decision projected onto the supplied budget bounds."""

    def __init__(self, path):
        super().__init__()
        self.base = load_temporal_budget_model(path)

    def forward(self, spend, outcome, controls, budget, lower, upper):
        scale = budget[:, None]
        allocation = self.base(
            spend, outcome, controls, budget, torch.zeros_like(lower), scale.expand_as(upper)
        )
        return scale * bounded_simplex(allocation / scale, lower / scale, upper / scale)


def interaction_data(n, seed, hidden):
    data = generate_controlled(
        n,
        seed=seed,
        channels=4,
        weeks=64,
        controls=2,
        confounding=hidden,
        positive_linear=True,
        structured=True,
    )
    rng = np.random.default_rng(seed + 900000)
    pair_a = rng.integers(0, 4, n)
    pair_b = (pair_a + rng.integers(1, 4, n)) % 4
    beta = rng.uniform(50, 300, n) * rng.choice([-1, 1], n)
    state = np.zeros((n, 4))
    add = np.empty((n, 64))
    rows = np.arange(n)
    for t in range(64):
        state = (1 - data["factor"]) * state + data["factor"] * data["spend"][:, t]
        add[:, t] = beta * state[rows, pair_a] * state[rows, pair_b] / data["budget"] ** 2
    data["outcome"] = data["outcome"] + add
    existing = tensors(data)
    return TensorDataset(
        *existing,
        torch.tensor(beta, dtype=torch.float32),
        torch.tensor(pair_a),
        torch.tensor(pair_b),
    )


def interaction_value(model, batch):
    x, y, z, b, lo, hi, _, strength, half, carry, factor, slope, beta, pair_a, pair_b = batch
    allocation = model(x, y, z, b, lo, hi)
    exposure = carry + factor * allocation
    response = (strength * exposure / (half + exposure) + slope * exposure).sum(1)
    rows = torch.arange(len(b))
    return response + beta * exposure[rows, pair_a] * exposure[rows, pair_b] / b**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds-per-geometry", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--linear-replay", action="store_true")
    parser.add_argument("--interaction-replay", action="store_true")
    args = parser.parse_args()
    if args.worlds_per_geometry < 32 or args.epochs < 1 or args.threads < 1:
        parser.error("at least 32 worlds, one epoch and one thread are required")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output checkpoint or report already exists")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train = [
        worlds(args.worlds_per_geometry, args.seed + 100 * i, g) for i, g in enumerate(GEOMETRIES)
    ]
    validation = [worlds(128, args.seed + 100 * i + 1, g) for i, g in enumerate(GEOMETRIES)]
    test = [worlds(256, args.seed + 100 * i + 2, g) for i, g in enumerate(GEOMETRIES)]
    random = torch.Generator().manual_seed(args.seed)
    loaders = [
        DataLoader(TensorDataset(*tensors(d)), batch_size=64, shuffle=True, generator=random)
        for d in train
    ]
    linear_train, linear_valid = [], []
    if args.linear_replay:
        linear_train = [
            DataLoader(
                linear_data(2 * args.worlds_per_geometry, args.seed + 500000 + 1000 * i, h),
                batch_size=64,
                shuffle=True,
                generator=random,
            )
            for i, h in enumerate((0.0, 0.7))
        ]
        linear_valid = [
            DataLoader(linear_data(256, args.seed + 500001 + 1000 * i, h), batch_size=128)
            for i, h in enumerate((0.0, 0.7))
        ]
    interaction_train, interaction_valid = [], []
    teacher = None
    if args.interaction_replay:
        interaction_train = [
            DataLoader(
                interaction_data(2 * args.worlds_per_geometry, args.seed + 800000 + 1000 * i, h),
                batch_size=64,
                shuffle=True,
                generator=random,
            )
            for i, h in enumerate((0.0, 0.7))
        ]
        interaction_valid = [
            DataLoader(interaction_data(256, args.seed + 800001 + 1000 * i, h), batch_size=128)
            for i, h in enumerate((0.0, 0.7))
        ]
        teacher = ProjectedBaseline(args.base).eval()
    model = ResidualBudgetModel(load_temporal_budget_model(args.base))
    optimizer = torch.optim.AdamW([*model.encode.parameters(), *model.head.parameters()], lr=0.0005)
    best, selected, state = float("inf"), 0, None
    for epoch in range(args.epochs + 1):
        if epoch:
            model.train()
            linear_iters = [forever(loader) for loader in linear_train]
            interaction_iters = [forever(loader) for loader in interaction_train]
            for index, loader in enumerate(loaders):
                for (
                    x,
                    y,
                    z,
                    budget,
                    lower,
                    upper,
                    _,
                    strength,
                    half,
                    carry,
                    factor,
                    slope,
                ) in loader:
                    allocation = model(x, y, z, budget, lower, upper)
                    exposure = carry + factor * allocation
                    loss = (
                        -(strength * exposure / (half + exposure) + slope * exposure).sum(1).mean()
                        / 100
                    )
                    if args.linear_replay:
                        linear_loss = (
                            linear_regret(model, next(linear_iters[index % 2])).mean() / 10
                        )
                        loss = loss + linear_loss
                    if args.interaction_replay:
                        interaction_batch = next(interaction_iters[index % 2])
                        interaction_outcome = interaction_value(model, interaction_batch)
                        with torch.no_grad():
                            teacher_outcome = interaction_value(teacher, interaction_batch)
                        loss = loss + (teacher_outcome - interaction_outcome).mean() / 100
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        generic_score = float(
            np.mean([evaluate(model, d)["network"]["mean_regret"] for d in validation])
        )
        linear_score = 0.0
        if args.linear_replay:
            with torch.no_grad():
                linear_score = float(
                    np.mean(
                        [
                            linear_regret(model, batch).mean().item()
                            for loader in linear_valid
                            for batch in loader
                        ]
                    )
                )
        interaction_gain = 0.0
        if args.interaction_replay:
            with torch.no_grad():
                interaction_gain = float(
                    np.mean(
                        [
                            (interaction_value(model, batch) - interaction_value(teacher, batch))
                            .mean()
                            .item()
                            for loader in interaction_valid
                            for batch in loader
                        ]
                    )
                )
        score = generic_score / 100 + linear_score / 10 - interaction_gain / 100
        if score < best:
            best, selected = score, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch} validation_regret={score:.6f}", flush=True)
    model.load_state_dict(state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state, str(args.output), metadata={"architecture": "residual-budget-v1", "hidden": "64"}
    )
    report = {
        "seed": args.seed,
        "epochs": args.epochs,
        "worlds_per_geometry": args.worlds_per_geometry,
        "selected_epoch": selected,
        "validation_weighted_regret": best,
        "linear_replay": args.linear_replay,
        "interaction_replay": args.interaction_replay,
        "geometries": GEOMETRIES,
        "test": {},
    }
    for geometry, data in zip(GEOMETRIES, test, strict=True):
        report["test"][str(geometry)] = evaluate(model, data)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
