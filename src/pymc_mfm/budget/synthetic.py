"""Synthetic training worlds and oracle, for research evaluation only.

The prior includes randomized spend, concave responses, carryover and seasonal
baseline. It is not a validated prior for real marketing systems.
"""

import numpy as np


def utility(allocation, strength, half, carry=0, factor=1):
    """Return the noise-free response for one-period synthetic allocation."""
    effective = carry + factor * allocation
    return (strength * effective / (half + effective)).sum(axis=-1)


def oracle(strength, half, budget, lower, upper, carry=0, factor=1):
    """Global maximizer for concave response subject to box and sum bounds."""
    left = np.zeros_like(budget)
    right = (strength * half * factor / (half + carry + factor * lower) ** 2).max(axis=-1) + 1
    for _ in range(80):
        level = (left + right) / 2
        x = np.clip(
            (np.sqrt(strength * half * factor / level[:, None]) - half - carry) / factor,
            lower,
            upper,
        )
        over = x.sum(axis=-1) > budget
        left = np.where(over, level, left)
        right = np.where(over, right, level)
    return np.clip(
        (np.sqrt(strength * half * factor / ((left + right)[:, None] / 2)) - half - carry) / factor,
        lower,
        upper,
    )


def generate(n, *, seed, channels=4, weeks=64, noise=0.15, confounding=0.0):
    """Create independent worlds; return observations, constraints and truth.

    ``confounding`` injects a shared unobserved demand factor into spend and
    outcomes. It is an out-of-prior stress test, not an identified correction.
    """
    rng = np.random.default_rng(seed)
    budget = rng.uniform(80, 120, n)
    strength = rng.lognormal(3.8, 0.5, (n, channels))
    half = rng.uniform(0.10, 0.45, (n, channels)) * budget[:, None]
    decay = rng.uniform(0, 0.7, (n, channels))
    spend = rng.uniform(0, 0.6, (n, weeks, channels)) * budget[:, None, None]
    demand = rng.normal(0, 1, (n, weeks))
    if confounding:
        spend *= np.exp(confounding * demand[:, :, None])
    state = np.zeros((n, channels))
    response = np.empty((n, weeks))
    for t in range(weeks):
        state = decay * state + (1 - decay) * spend[:, t, :]
        response[:, t] = utility(state, strength, half)
    carry, factor = decay * state, 1 - decay
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    amplitude = rng.uniform(0, 8, (n, 1))
    seasonal = amplitude * np.sin(2 * np.pi * np.arange(weeks)[None, :] / 52 + phase)
    outcome = 100 + response + seasonal + confounding * 25 * demand
    outcome += rng.normal(0, noise * 10, (n, weeks))
    anchor = rng.dirichlet(np.ones(channels) * 3, n)
    lower = np.maximum(0, anchor - rng.uniform(0.02, 0.3, (n, channels))) * budget[:, None]
    upper = np.minimum(1, anchor + rng.uniform(0.02, 0.3, (n, channels))) * budget[:, None]
    # Include unconstrained plans and single-channel caps in training, not
    # only tight boxes around a random feasible anchor.
    kind = rng.integers(0, 4, n)
    open_case = kind == 0
    lower[open_case], upper[open_case] = 0, budget[open_case, None]
    if channels > 1:
        cap_case = kind == 1
        lower[cap_case], upper[cap_case] = 0, budget[cap_case, None]
        rows = np.flatnonzero(cap_case)
        capped = rng.integers(0, channels, len(rows))
        upper[rows, capped] = budget[rows] * rng.uniform(0.05, 0.25, len(rows))
    best = oracle(strength, half, budget, lower, upper, carry, factor)
    return dict(
        spend=spend,
        outcome=outcome,
        budget=budget,
        lower=lower,
        upper=upper,
        target=best,
        strength=strength,
        half=half,
        carry=carry,
        factor=factor,
        constraint_kind=kind,
    )
