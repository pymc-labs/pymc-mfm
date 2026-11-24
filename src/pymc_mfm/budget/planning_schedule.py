"""Direct planning network with an encoder for the complete budget schedule."""

import torch
from torch import nn

from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.planning import MultiPeriodBudgetModel


class ScheduleBudgetModel(MultiPeriodBudgetModel):
    """Map history, all future totals and per-period limits to budgets only."""

    def __init__(self, hidden=64):
        super().__init__(hidden=hidden)
        self.schedule = nn.GRU(3, 16, batch_first=True, bidirectional=True)
        self.time_head = nn.Sequential(
            nn.Linear(38, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, spend, outcome, controls, totals, lower, upper):
        if totals.ndim != 2 or lower.ndim != 3 or upper.shape != lower.shape:
            raise ValueError("totals must be [batch, period] and limits [batch, period, channel]")
        b, horizon = totals.shape
        if horizon < 1 or spend.ndim != 3 or lower.shape != (b, horizon, spend.shape[-1]):
            raise ValueError("invalid planning horizon or bound shapes")
        if spend.shape[0] != b or outcome.shape[0] != b or controls.shape[0] != b:
            raise ValueError("historical and planning batch sizes must match")
        if not torch.isfinite(totals).all().item() or (totals <= 0).any().item():
            raise ValueError("period totals must be finite and positive")
        k = spend.shape[-1]
        history = spend.repeat_interleave(horizon, dim=0)
        y = outcome.repeat_interleave(horizon, dim=0)
        z = controls.repeat_interleave(horizon, dim=0)
        budget = totals.reshape(-1)
        lo = lower.reshape(-1, k)
        hi = upper.reshape(-1, k)
        initial = self.single(history, y, z, budget, lo, hi) / budget[:, None]
        periods = torch.arange(horizon, device=spend.device, dtype=spend.dtype)
        time = ((periods + 1) / horizon).expand(b, -1).reshape(-1, 1).expand(-1, k)
        normalized = totals / totals.mean(1, keepdim=True)
        # Encode each channel's complete future total and bound schedule.
        schedule_inputs = torch.stack(
            (
                normalized[:, :, None].expand(-1, -1, k),
                lower / totals[:, :, None],
                torch.minimum(upper, totals[:, :, None]) / totals[:, :, None],
            ),
            dim=-1,
        )
        packed = schedule_inputs.permute(0, 2, 1, 3).reshape(b * k, horizon, 3)
        schedule, _ = self.schedule(packed)
        context = (
            schedule.reshape(b, k, horizon, -1).permute(0, 2, 1, 3).reshape(b * horizon, k, -1)
        )
        scale = normalized.reshape(-1, 1).expand(-1, k)
        last = history[:, -1, :] / budget[:, None]
        length = torch.full_like(time, float(horizon)).log1p()
        features = torch.stack((initial, time, scale, last, (1 - time) * last, length), dim=-1)
        scores = initial + self.time_head(torch.cat((features, context), dim=-1)).squeeze(-1)
        allocation = bounded_simplex(
            scores, lo / budget[:, None], torch.minimum(hi, budget[:, None]) / budget[:, None]
        )
        return (allocation * budget[:, None]).reshape(b, horizon, k)
