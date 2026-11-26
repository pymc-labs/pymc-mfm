"""Train a direct decision model across several channel and history geometries."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset
from train_controlled import evaluate, tensors

from pymc_mfm.budget.controlled import ControlledBudgetModel, generate

# These geometries describe training coverage, not supported real populations.
GEOMETRIES = ((2, 52, 1), (3, 64, 2), (4, 64, 2), (6, 104, 3))


def worlds(n, seed, geometry, confounding=0):
    channels, weeks, controls = geometry
    return generate(
        n,
        seed=seed,
        channels=channels,
        weeks=weeks,
        controls=controls,
        confounding=confounding,
        positive_linear=True,
        structured=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--worlds-per-geometry", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=620)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.worlds_per_geometry < 1 or args.epochs < 4 or args.threads < 1:
        parser.error("invalid training dimensions")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Disjoint seed blocks keep every training, selection and test world apart.
    train = [
        worlds(args.worlds_per_geometry, args.seed + 100 * i, g) for i, g in enumerate(GEOMETRIES)
    ]
    validation = [worlds(500, args.seed + 100 * i + 1, g) for i, g in enumerate(GEOMETRIES)]
    tests = [worlds(1000, args.seed + 100 * i + 2, g) for i, g in enumerate(GEOMETRIES)]
    stress = [
        worlds(1000, args.seed + 100 * i + 3, g, confounding=0.5) for i, g in enumerate(GEOMETRIES)
    ]
    rng = torch.Generator().manual_seed(args.seed)
    loaders = [
        DataLoader(TensorDataset(*tensors(d)), batch_size=256, shuffle=True, generator=rng)
        for d in train
    ]
    model = ControlledBudgetModel(joint_features=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    best = float("inf")
    state = None
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        iters = [iter(loader) for loader in loaders]
        # Interleave complete batches across geometries without padding.
        remaining = [len(loader) for loader in loaders]
        while sum(remaining):
            options = [i for i, count in enumerate(remaining) if count]
            idx = options[int(torch.randint(len(options), (1,), generator=rng))]
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
            opt.zero_grad()
            loss.backward()
            opt.step()
        values = [evaluate(model, d)["network"]["mean_regret"] for d in validation]
        score = float(np.mean(values))
        if score < best:
            best = score
            best_epoch = epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch + 1} validation={score:.6f} best={best:.6f}", flush=True)
    model.load_state_dict(state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata={"architecture": "controlled-budget-v2", "hidden": "64"})
    report = {
        "seed": args.seed,
        "worlds_per_geometry": args.worlds_per_geometry,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "validation_mean_regret": best,
        "geometries": GEOMETRIES,
        "test": {},
        "hidden_confounding": {},
    }
    for geometry, normal, hidden in zip(GEOMETRIES, tests, stress, strict=True):
        key = f"{geometry[0]}channels_{geometry[1]}weeks_{geometry[2]}controls"
        report["test"][key] = evaluate(model, normal)
        report["hidden_confounding"][key] = evaluate(model, hidden)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
