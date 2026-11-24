"""Evaluate allocation policies on an out-of-prior linear marketing world.

This simulator has correlated media, three observed controls, signed or null
media effects, carryover and optional hidden demand. It was designed after
model development, so it is a stress test rather than a blinded benchmark.
"""

import argparse
import json

import numpy as np
import torch

from pymc_mfm.budget.io import load_controlled_budget_model


def generate_ood(n, *, seed, hidden_strength):
    """Draw two-channel worlds with autoregressive demand and control inputs."""
    rng = np.random.default_rng(seed)
    weeks, channels, controls = 60, 2, 3
    noise = rng.normal(size=(n, weeks, controls + 1))
    latent = np.zeros_like(noise)
    latent[:, 0] = noise[:, 0]
    for t in range(1, weeks):
        latent[:, t] = 0.45 * latent[:, t - 1] + np.sqrt(1 - 0.45**2) * noise[:, t]
    z, hidden = latent[:, :, :controls], latent[:, :, controls]
    shared_media = rng.normal(size=(n, weeks, 1))
    independent_media = rng.normal(size=(n, weeks, channels))
    control_loading = rng.normal(0, 0.18, (n, channels, controls))
    log_spend = (
        0.3 * independent_media
        + 0.2 * shared_media
        + np.einsum("ntm,nkm->ntk", z, control_loading)
        + hidden_strength * 0.3 * hidden[:, :, None]
    )
    spend = 20 * np.exp(log_spend)
    decay = rng.uniform(0.1, 0.8, (n, channels))
    effect = rng.normal(0, 0.8, (n, channels)) * rng.binomial(1, 0.7, (n, channels))
    control_effect = rng.normal(0, 8, (n, controls))
    adstock = np.zeros((n, channels))
    outcome = np.empty((n, weeks))
    for t in range(weeks):
        adstock = decay * adstock + (1 - decay) * spend[:, t, :]
        outcome[:, t] = (effect * adstock).sum(1)
    outcome += 100 + np.einsum("ntm,nm->nt", z, control_effect)
    outcome += hidden_strength * 8 * hidden + rng.normal(0, 4, (n, weeks))
    budget = spend[:, -13:, :].sum(-1).mean(1)
    # A next-period spend increment changes outcome by this marginal effect.
    marginal = effect * (1 - decay)
    return dict(spend=spend, outcome=outcome, controls=z, budget=budget, marginal=marginal)


def evaluate(model, data):
    """Score next-period decisions against the exact linear optimum."""
    x, y, z, total, marginal = (
        data[k] for k in ("spend", "outcome", "controls", "budget", "marginal")
    )
    inputs = [torch.tensor(v, dtype=torch.float32) for v in (x, y, z, total)]
    s, o, c, b = inputs
    lo = torch.zeros_like(s[:, 0])
    hi = b[:, None].expand_as(lo)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(s, o, c, b, lo, hi), batch_size=256
    )
    with torch.no_grad():
        network = torch.cat([model(*batch) for batch in loader]).numpy()
    design = np.concatenate((np.ones((*y.shape, 1)), x, z), axis=2)
    coef = np.array(
        [np.linalg.lstsq(D, v, rcond=None)[0][1:3] for D, v in zip(design, y, strict=True)]
    )
    best = total * marginal.max(1)
    ols = total * marginal[np.arange(len(total)), coef.argmax(1)]
    regret = {
        "network": best - (network * marginal).sum(1),
        "ols": best - ols,
        "equal": best - total * marginal.mean(1),
    }
    report = {
        name: {"mean_regret": float(v.mean()), "median_regret": float(np.median(v))}
        for name, v in regret.items()
    }
    for name in ("ols", "equal"):
        delta = regret[name] - regret["network"]
        report["paired_vs_" + name] = {
            "mean_improvement": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "win_fraction": float((delta > 1e-6).mean()),
            "loss_fraction": float((delta < -1e-6).mean()),
        }
    report["max_budget_error"] = float(np.abs(network.sum(1) - total).max())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--hidden-strength", type=float, default=0)
    args = parser.parse_args()
    if args.worlds < 2 or args.hidden_strength < 0:
        parser.error("worlds must be at least two and hidden strength nonnegative")
    torch.set_num_threads(4)
    data = generate_ood(args.worlds, seed=args.seed, hidden_strength=args.hidden_strength)
    print(json.dumps(evaluate(load_controlled_budget_model(args.model), data), indent=2))


if __name__ == "__main__":
    main()
