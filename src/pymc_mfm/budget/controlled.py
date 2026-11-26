"""Direct budget model and synthetic prior with controls and signed effects.

This separate research version keeps the v2 checkpoint semantics unchanged.
"""

import numpy as np
import torch
from torch import nn

from pymc_mfm.budget.model import bounded_simplex


def response(allocation, strength, half, carry, factor, slope):
    """One-period noise-free synthetic response, for training and scoring."""
    exposure = carry + factor * allocation
    return (strength * exposure / (half + exposure) + slope * exposure).sum(-1)


def oracle(strength, half, carry, factor, slope, budget, lower, upper):
    """Solve the concave synthetic allocation problem by marginal bisection."""

    def marginal(x):
        return strength * half * factor / (half + carry + factor * x) ** 2 + slope * factor

    left = marginal(upper).min(axis=1) - 1
    right = marginal(lower).max(axis=1) + 1
    for _ in range(80):
        price = (left + right) / 2
        above_linear = price[:, None] > slope * factor
        denom = np.maximum(price[:, None] - slope * factor, 1e-30)
        interior = (np.sqrt(strength * half * factor / denom) - half - carry) / factor
        allocation = np.where(above_linear, np.clip(interior, lower, upper), upper)
        over = allocation.sum(-1) > budget
        left = np.where(over, price, left)
        right = np.where(over, right, price)
    price = (left + right)[:, None] / 2
    above_linear = price > slope * factor
    denom = np.maximum(price - slope * factor, 1e-30)
    interior = (np.sqrt(strength * half * factor / denom) - half - carry) / factor
    result = np.where(above_linear, np.clip(interior, lower, upper), upper)
    # Flat null or linear channels can share a marginal value, creating a
    # jump in demand at the clearing price. Fill the residual across tied
    # channels; this selects one optimum from the set of optimal allocations.
    residual = budget - result.sum(-1)
    rows = np.arange(len(budget))
    flat_tie = (strength == 0) & (np.abs(slope * factor - price) < 1e-10)
    # Allocate discontinuity mass to flat channels before smooth ones.
    order = np.argsort(np.where(flat_tie, -1.0, np.abs(marginal(result) - price)), axis=1)
    for j in range(result.shape[1]):
        idx = order[:, j]
        room_up = upper[rows, idx] - result[rows, idx]
        room_down = result[rows, idx] - lower[rows, idx]
        change = np.clip(residual, -room_down, room_up)
        result[rows, idx] += change
        residual -= change
    return result


def generate(
    n,
    *,
    seed,
    channels=4,
    weeks=64,
    controls=2,
    confounding=0.0,
    positive_linear=False,
    structured=False,
):
    """Generate a bounded synthetic prior, including null and harmful channels.

    Controls influence spending and outcome, and are observed by the model.
    A separate hidden shock may affect both, but is never a model input.
    The optional structured prior adds correlated media and persistent controls.
    """
    rng = np.random.default_rng(seed)
    budget = rng.uniform(80, 120, n)
    half = rng.uniform(0.1, 0.45, (n, channels)) * budget[:, None]
    factor = rng.uniform(0.3, 1, (n, channels))
    probabilities = [0.45, 0.15, 0.20, 0.20] if positive_linear else [0.60, 0.20, 0.20]
    kind = rng.choice(len(probabilities), size=(n, channels), p=probabilities)
    strength = np.where(kind == 0, rng.lognormal(3.8, 0.5, (n, channels)), 0.0)
    slope = np.where(kind == 2, -rng.uniform(0.08, 0.6, (n, channels)), 0.0)
    if positive_linear:
        slope = np.where(kind == 3, rng.uniform(0.08, 0.6, (n, channels)), slope)
    z = rng.normal(size=(n, weeks, controls))
    if structured:
        persistence = rng.uniform(0, 0.75, (n, 1, 1))
        for t in range(1, weeks):
            z[:, t : t + 1] = (
                persistence * z[:, t - 1 : t] + np.sqrt(1 - persistence**2) * z[:, t : t + 1]
            )
    z_coef = rng.normal(0, 5, (n, controls))
    spend_loading = rng.normal(0, 0.35, (n, channels, controls))
    hidden = rng.normal(size=(n, weeks))
    latent = rng.normal(size=(n, weeks, channels))
    if structured:
        shared = rng.normal(size=(n, weeks, 1))
        coupling = rng.uniform(0, 0.85, (n, 1, 1))
        latent = coupling * shared + np.sqrt(1 - coupling**2) * latent
    covariates = np.einsum("ntm,nkm->ntk", z, spend_loading)
    spend = (
        budget[:, None, None]
        * 0.6
        / (1 + np.exp(-(latent + covariates + confounding * hidden[:, :, None])))
    )
    state = np.zeros((n, channels))
    outcome = np.empty((n, weeks))
    decay = 1 - factor
    for t in range(weeks):
        state = decay * state + factor * spend[:, t, :]
        outcome[:, t] = response(state, strength, half, 0, 1, slope)
    seasonal = rng.uniform(0, 8, (n, 1)) * np.sin(
        2 * np.pi * np.arange(weeks)[None, :] / 52 + rng.uniform(0, 2 * np.pi, (n, 1))
    )
    outcome += 100 + np.einsum("ntm,nm->nt", z, z_coef) + seasonal
    outcome += confounding * 12 * hidden + rng.normal(0, 1.5, (n, weeks))
    carry = decay * state
    anchor = rng.dirichlet(np.ones(channels) * 3, n)
    lower = np.maximum(0, anchor - rng.uniform(0.02, 0.3, (n, channels))) * budget[:, None]
    upper = np.minimum(1, anchor + rng.uniform(0.02, 0.3, (n, channels))) * budget[:, None]
    constraint = rng.integers(0, 4, n)
    mask = constraint < 2
    lower[mask], upper[mask] = 0, budget[mask, None]
    if channels > 1:
        rows = np.flatnonzero(constraint == 1)
        cap = rng.integers(0, channels, len(rows))
        upper[rows, cap] = budget[rows] * rng.uniform(0.05, 0.25, len(rows))
    target = oracle(strength, half, carry, factor, slope, budget, lower, upper)
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
        factor=factor,
        slope=slope,
        kind=kind,
        constraint_kind=constraint,
    )


class ControlledBudgetModel(nn.Module):
    """Map history, optional observed controls, budget and bounds to budgets only."""

    def __init__(self, hidden=64, *, joint_features=False):
        super().__init__()
        self.joint_features = joint_features
        self.observation = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.channel = nn.Sequential(
            nn.Linear(hidden + 5 + 2 * int(joint_features), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, spend, outcome, controls, budget, lower, upper):
        if (
            spend.ndim != 3
            or outcome.shape != spend.shape[:2]
            or controls.ndim != 3
            or controls.shape[:2] != spend.shape[:2]
        ):
            raise ValueError("invalid history or controls shape")
        b, t, k = spend.shape
        if t < 2 or k < 1 or budget.shape != (b,) or lower.shape != (b, k) or upper.shape != (b, k):
            raise ValueError("invalid budget, history or bounds shape")
        if not all(
            torch.isfinite(x).all().item() for x in (spend, outcome, controls, budget, lower, upper)
        ):
            raise ValueError("inputs must be finite")
        if (
            (spend < 0).any().item()
            or (budget <= 0).any().item()
            or ((lower < 0) | (upper < lower)).any().item()
        ):
            raise ValueError("invalid spend, budget or bounds")
        lo = lower / budget[:, None]
        hi = torch.minimum(upper, budget[:, None]) / budget[:, None]
        x = spend / budget[:, None, None]
        y = (outcome - outcome.mean(1, keepdim=True)) / outcome.std(1, keepdim=True).clamp_min(1e-6)
        # Residualize both observed spend and outcome on observed controls.
        # The ridge controls numerical instability. This is a feature, not
        # proof that the controls remove all confounding.
        z = controls
        if z.shape[-1]:
            z = z - z.mean(1, keepdim=True)
            z = z / z.std(1, keepdim=True).clamp_min(1e-6)
            eye = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)[None]
            gram = z.transpose(1, 2) @ z + eye * 0.01
            values = torch.cat((x, y.unsqueeze(-1)), dim=-1)
            residual = values - z @ torch.linalg.solve(gram, z.transpose(1, 2) @ values)
            rx, ry = residual[:, :, :k], residual[:, :, k]
        else:
            rx, ry = x, y
        ry = ry.unsqueeze(-1).expand(-1, -1, k)
        obs = torch.stack((x, x.square(), rx, ry, rx * ry), dim=-1)
        encoded = self.observation(obs).mean(1)
        extras = torch.stack((x.mean(1), x.std(1), lo, hi, x[:, -1, :]), dim=-1)
        if self.joint_features:
            centered = rx - rx.mean(1, keepdim=True)
            target = ry[:, :, 0] - ry[:, :, 0].mean(1, keepdim=True)
            eye = torch.eye(k, dtype=x.dtype, device=x.device)[None]
            gram = centered.transpose(1, 2) @ centered + eye * (t * 1e-4)
            inverse = torch.linalg.solve(gram, eye.expand(b, -1, -1))
            slope = (inverse @ centered.transpose(1, 2) @ target.unsqueeze(-1)).squeeze(-1)
            residual = target - (centered @ slope.unsqueeze(-1)).squeeze(-1)
            noise = residual.square().mean(1, keepdim=True).sqrt()
            uncertainty = noise * inverse.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
            features = torch.stack((slope, torch.log1p(uncertainty)), dim=-1)
            extras = torch.cat((extras, features), dim=-1)
        h = self.channel(torch.cat((encoded, extras), dim=-1))
        interaction, _ = self.attention(h, h, h, need_weights=False)
        return (
            bounded_simplex(self.head(torch.cat((h, interaction), dim=-1)).squeeze(-1), lo, hi)
            * budget[:, None]
        )
