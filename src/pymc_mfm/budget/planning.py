"""Experimental direct, multi-period budget allocation network."""

import torch
from torch import nn

from pymc_mfm.budget.controlled import ControlledBudgetModel
from pymc_mfm.budget.model import bounded_simplex


class MultiPeriodBudgetModel(nn.Module):
    """Output a channel budget for each period under per-period constraints.

    The network sees historical spend, outcome and controls, plus future
    totals and channel bounds. It returns budgets only. Future outcomes are
    not supplied to the model.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.single = ControlledBudgetModel(hidden=hidden, joint_features=True)
        self.time_head = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, spend, outcome, controls, totals, lower, upper):
        if totals.ndim != 2 or lower.ndim != 3 or upper.shape != lower.shape:
            raise ValueError("totals must be [batch, period] and limits [batch, period, channel]")
        b, horizon = totals.shape
        if horizon < 1 or lower.shape != (b, horizon, spend.shape[-1]):
            raise ValueError("invalid planning horizon or bound shapes")
        if spend.shape[0] != b or outcome.shape[0] != b or controls.shape[0] != b:
            raise ValueError("historical and planning batch sizes must match")
        k = spend.shape[-1]
        history = spend.repeat_interleave(horizon, dim=0)
        sales = outcome.repeat_interleave(horizon, dim=0)
        z = controls.repeat_interleave(horizon, dim=0)
        budget = totals.reshape(-1)
        lo = lower.reshape(-1, k)
        hi = upper.reshape(-1, k)
        initial = self.single(history, sales, z, budget, lo, hi) / budget[:, None]
        periods = torch.arange(horizon, device=spend.device, dtype=spend.dtype)
        time = ((periods + 1) / horizon).expand(b, -1).reshape(-1, 1).expand(-1, k)
        scale = (totals / totals.mean(1, keepdim=True)).reshape(-1, 1).expand(-1, k)
        last = history[:, -1, :] / budget[:, None]
        length = torch.full_like(time, float(horizon)).log1p()
        features = torch.stack((initial, time, scale, last, (1 - time) * last, length), dim=-1)
        scores = initial + self.time_head(features).squeeze(-1)
        allocation = bounded_simplex(
            scores, lo / budget[:, None], torch.minimum(hi, budget[:, None]) / budget[:, None]
        )
        return (allocation * budget[:, None]).reshape(b, horizon, k)


def synthetic_plan_outcome(allocation, strength, half, last_state, factor, slope):
    """Compute synthetic utility inside training and evaluation, not inference."""
    state = last_state
    value = torch.zeros_like(strength[:, 0])
    for period in range(allocation.shape[1]):
        state = (1 - factor) * state + factor * allocation[:, period, :]
        value = value + (strength * state / (half + state) + slope * state).sum(-1)
    return value
