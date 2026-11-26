"""Post-development linear carryover stress for direct budget planning."""

import argparse
import json

import numpy as np
import torch
from train_planning import myopic

from pymc_mfm.budget.io import load_controlled_budget_model
from pymc_mfm.budget.planning_io import load_planning_model


def generate(n, seed, hidden_strength):
    """Generate two channels, correlated spend and persistent observed controls."""
    rng = np.random.default_rng(seed)
    weeks, channels, controls, horizon = 60, 2, 3, 4
    noise = rng.normal(size=(n, weeks, controls + 1))
    latent = np.zeros_like(noise)
    latent[:, 0] = noise[:, 0]
    for t in range(1, weeks):
        latent[:, t] = 0.45 * latent[:, t - 1] + np.sqrt(1 - 0.45**2) * noise[:, t]
    z, hidden = latent[:, :, :controls], latent[:, :, controls]
    common = rng.normal(size=(n, weeks, 1))
    loading = rng.normal(0, 0.18, (n, channels, controls))
    log_spend = (
        0.3 * rng.normal(size=(n, weeks, channels))
        + 0.2 * common
        + np.einsum("ntm,nkm->ntk", z, loading)
        + hidden_strength * 0.3 * hidden[:, :, None]
    )
    spend = 20 * np.exp(log_spend)
    decay = rng.uniform(0.1, 0.8, (n, channels))
    effect = rng.normal(0, 0.8, (n, channels)) * rng.binomial(1, 0.7, (n, channels))
    control_effect = rng.normal(0, 8, (n, controls))
    state = np.zeros((n, channels))
    outcome = np.empty((n, weeks))
    for t in range(weeks):
        state = decay * state + (1 - decay) * spend[:, t]
        outcome[:, t] = (effect * state).sum(1)
    outcome += 100 + np.einsum("ntm,nm->nt", z, control_effect)
    outcome += hidden_strength * 8 * hidden + rng.normal(0, 4, (n, weeks))
    totals = spend[:, -13:].sum(-1).mean(1)[:, None] * np.array([0.8, 1, 1.2, 1])[None, :]
    return dict(
        spend=spend,
        outcome=outcome,
        controls=z,
        totals=totals,
        decay=decay,
        effect=effect,
        last_state=state,
        horizon=horizon,
    )


def score(plan, data):
    """Compute four-period incremental outcome relative to zero future spend."""
    horizon = plan.shape[1]
    weights = data["effect"][:, None, :] * (
        1 - data["decay"][:, None, :] ** np.arange(horizon, 0, -1)[None, :, None]
    )
    return (plan * weights).sum((1, 2))


def evaluate(model, baseline, data):
    x, y, z, total = (data[k] for k in ("spend", "outcome", "controls", "totals"))
    s, o, c, b = [torch.tensor(v, dtype=torch.float32) for v in (x, y, z, total)]
    lo = torch.zeros(len(x), total.shape[1], x.shape[-1])
    hi = b[:, :, None].expand_as(lo)
    with torch.no_grad():
        neural = model(s, o, c, b, lo, hi).numpy()
        single = myopic(baseline, s, o, c, b, lo, hi).numpy()
    design = np.concatenate((np.ones((*y.shape, 1)), x, z), 2)
    beta = np.array(
        [np.linalg.lstsq(D, v, rcond=None)[0][1:3] for D, v in zip(design, y, strict=True)]
    )
    choice = beta.argmax(1)
    ols = np.zeros_like(neural)
    rows = np.arange(len(x))[:, None]
    period = np.arange(total.shape[1])[None, :]
    ols[rows, period, choice[:, None]] = total
    equal = np.repeat(total[:, :, None] / x.shape[-1], x.shape[-1], axis=-1)
    h = total.shape[1]
    weight = data["effect"][:, None, :] * (
        1 - data["decay"][:, None, :] ** np.arange(h, 0, -1)[None, :, None]
    )
    optimum = (total * weight.max(-1)).sum(1)
    regrets = {
        name: optimum - score(p, data)
        for name, p in (("plan", neural), ("one_period", single), ("ols", ols), ("equal", equal))
    }
    report = {
        name: {"mean_regret": float(v.mean()), "median_regret": float(np.median(v))}
        for name, v in regrets.items()
    }
    for name in ("one_period", "ols", "equal"):
        delta = regrets[name] - regrets["plan"]
        report["paired_vs_" + name] = {
            "mean_gain": float(delta.mean()),
            "standard_error": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "wins": float((delta > 1e-6).mean()),
            "losses": float((delta < -1e-6).mean()),
        }
    report["max_budget_error"] = float(abs(neural.sum(-1) - total).max())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--hidden", type=float, default=0)
    args = parser.parse_args()
    if args.worlds < 2 or args.hidden < 0:
        parser.error("invalid input")
    torch.set_num_threads(4)
    data = generate(args.worlds, args.seed, args.hidden)
    print(
        json.dumps(
            evaluate(
                load_planning_model(args.model), load_controlled_budget_model(args.baseline), data
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
