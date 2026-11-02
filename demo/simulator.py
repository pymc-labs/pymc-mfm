"""A two-channel synthetic example adapted from the River-1 quickstart."""

from dataclasses import dataclass

import numpy as np

N_WEEKS = 208
N_CONTEXT = 166


@dataclass(frozen=True)
class World:
    spend: np.ndarray
    outcome: np.ndarray
    baseline: np.ndarray
    channels: np.ndarray
    events: np.ndarray
    controls: np.ndarray
    betas: np.ndarray


def simulate(betas=(3.0, 2.0), trend=0.0, seasonality=0.5, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N_WEEKS)
    x1 = rng.uniform(size=N_WEEKS)
    x1 = np.where(x1 > 0.9, x1, x1 / 2)
    x2 = rng.uniform(size=N_WEEKS)
    x2 = np.where(x2 > 0.8, x2, 0.0)
    spend = np.column_stack((x1, x2))
    adstock = np.zeros_like(spend)
    for j, alpha in enumerate((0.4, 0.2)):
        mass = sum(alpha**i for i in range(8))
        for lag in range(8):
            adstock[lag:, j] += (alpha**lag / mass) * spend[: N_WEEKS - lag, j]
    saturation = np.tanh(adstock * np.array((4.0, 3.0)) / 2)
    betas = np.asarray(betas, dtype=float)
    channels = 1000 * saturation * betas
    baseline = 1000 * (
        2
        + ((np.linspace(0, 50, N_WEEKS) + 10) ** 0.25 - 1)
        + trend * t
        + seasonality * np.sin(2 * np.pi * t / 52)
    )
    events = np.column_stack(((t == 60) | (t == 190), t == 130)).astype(float)
    controls = np.zeros((N_WEEKS, 10))
    controls[:, :2] = events
    outcome = baseline + channels.sum(axis=1) + events @ np.array([1500, 2500])
    outcome += 1000 * rng.normal(0, noise, N_WEEKS)
    if np.any(outcome <= 0):
        raise ValueError("Simulated outcomes must be positive")
    return World(spend * 1000, outcome, baseline, channels, events, controls, betas)


def response_curves(world):
    grid = np.linspace(0, 1100, 60)
    result = []
    for j, lam in enumerate((4.0, 3.0)):
        vals = 1000 * world.betas[j] * np.tanh(grid / 1000 * lam / 2)
        current = world.spend[:, j].mean()
        result.append(
            {
                "spend": grid.tolist(),
                "response": vals.tolist(),
                "current_spend": float(current),
                "current_response": float(
                    1000 * world.betas[j] * np.tanh(current / 1000 * lam / 2)
                ),
            }
        )
    return {"curves": result}
