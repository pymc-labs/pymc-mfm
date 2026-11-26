"""A broader synthetic prior for direct budget decisions.

It mixes exposure scales, channel dependence, temporal persistence, noise and
unobserved demand. These are simulation assumptions, not causal evidence.
"""

import numpy as np

from pymc_mfm.budget.controlled import oracle, response


def generate(n, *, seed, channels=4, weeks=64, controls=2, hidden=None):
    """Return synthetic histories, fixed budgets, constraints and oracle truth."""
    rng = np.random.default_rng(seed)
    budget = rng.uniform(50, 150, n)
    half = budget[:, None] * rng.uniform(0.05, 0.65, (n, channels))
    factor = rng.uniform(0.15, 1, (n, channels))
    kind = rng.choice(4, size=(n, channels), p=[0.35, 0.2, 0.2, 0.25])
    strength = np.where(kind == 0, rng.lognormal(3.7, 0.65, (n, channels)), 0.0)
    magnitude = rng.lognormal(-0.7, 0.6, (n, channels))
    slope = np.where(kind == 2, -magnitude, np.where(kind == 3, magnitude, 0.0))
    shocks = rng.normal(size=(n, weeks, controls + 1))
    state = np.zeros_like(shocks)
    persistence = rng.uniform(0, 0.8, (n, 1))
    state[:, 0, :] = shocks[:, 0, :]
    for t in range(1, weeks):
        state[:, t, :] = (
            persistence * state[:, t - 1, :] + np.sqrt(1 - persistence**2) * shocks[:, t, :]
        )
    z, demand = state[:, :, :controls], state[:, :, controls]
    hidden_level = rng.uniform(0, 0.75, n) if hidden is None else np.full(n, hidden)
    independent = rng.normal(size=(n, weeks, channels))
    common = rng.normal(size=(n, weeks, 1))
    coupling = rng.uniform(0, 0.8, (n, 1, 1))
    latent = np.sqrt(1 - coupling**2) * independent + coupling * common
    controls_loading = rng.normal(0, 0.2, (n, channels, controls))
    variation = rng.uniform(0.15, 0.8, (n, 1, 1))
    exponent = (
        variation * latent
        + np.einsum("ntm,nkm->ntk", z, controls_loading)
        + hidden_level[:, None, None] * 0.3 * demand[:, :, None]
    )
    # A smooth lognormal family has a different support from the bounded
    # logistic generator. Per-world scale can also differ from future budget.
    typical = budget[:, None, None] / channels * rng.uniform(0.5, 1.6, (n, 1, channels))
    spend = typical * np.exp(exponent)
    adstock = np.zeros((n, channels))
    outcome = np.empty((n, weeks))
    for t in range(weeks):
        adstock = (1 - factor) * adstock + factor * spend[:, t, :]
        outcome[:, t] = response(adstock, strength, half, 0, 1, slope)
    control_effect = rng.normal(0, 6, (n, controls))
    outcome += 100 + np.einsum("ntm,nm->nt", z, control_effect)
    outcome += hidden_level[:, None] * rng.uniform(0, 12, (n, 1)) * demand
    amplitude = rng.uniform(0, 8, (n, 1))
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    outcome += amplitude * np.sin(2 * np.pi * np.arange(weeks)[None, :] / 52 + phase)
    outcome += rng.normal(size=(n, weeks)) * rng.uniform(0.5, 7, (n, 1))
    carry = (1 - factor) * adstock
    anchor = rng.dirichlet(np.ones(channels) * 3, n)
    lo = np.maximum(0, anchor - rng.uniform(0.02, 0.3, (n, channels))) * budget[:, None]
    hi = np.minimum(1, anchor + rng.uniform(0.02, 0.3, (n, channels))) * budget[:, None]
    cases = rng.integers(0, 4, n)
    open_cases = cases < 2
    lo[open_cases] = 0
    hi[open_cases] = budget[open_cases, None]
    if channels > 1:
        rows = np.flatnonzero(cases == 1)
        idx = rng.integers(0, channels, len(rows))
        hi[rows, idx] = budget[rows] * rng.uniform(0.05, 0.25, len(rows))
    target = oracle(strength, half, carry, factor, slope, budget, lo, hi)
    return dict(
        spend=spend,
        outcome=outcome,
        controls=z,
        budget=budget,
        lower=lo,
        upper=hi,
        target=target,
        strength=strength,
        half=half,
        carry=carry,
        factor=factor,
        slope=slope,
        kind=kind,
        hidden_level=hidden_level,
    )
