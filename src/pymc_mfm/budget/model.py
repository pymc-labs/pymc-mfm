"""A constraint-conditioned network that returns budget allocations only."""

import torch
from torch import nn


def bounded_simplex(logits: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Project scores onto a bounded unit simplex, with an implicit gradient.

    The active-set selection is piecewise constant. On each active set the
    returned expression has the exact gradient of Euclidean projection. All
    values are fractions of the total budget.
    """
    if logits.shape != lower.shape or logits.shape != upper.shape or logits.ndim != 2:
        raise ValueError("scores and bounds must have the same [batch, channel] shape")
    if not all(torch.isfinite(t).all().item() for t in (logits, lower, upper)):
        raise ValueError("scores and bounds must be finite")
    if ((lower < 0) | (upper > 1) | (lower > upper)).any().item():
        raise ValueError("bounds must satisfy 0 <= lower <= upper <= 1")
    if ((lower.sum(-1) > 1 + 1e-6) | (upper.sum(-1) < 1 - 1e-6)).any().item():
        raise ValueError("budget bounds are infeasible")
    # The scores can be much larger than budget fractions. Compute the
    # projection in float64 to avoid cancellation at active constraints.
    output_dtype = logits.dtype
    if logits.device.type != "mps":
        logits, lower, upper = logits.double(), lower.double(), upper.double()
    # Translation leaves the projection unchanged and reduces cancellation.
    logits = logits - logits.mean(dim=-1, keepdim=True)
    # Find the active set without retaining a bisection graph.
    with torch.no_grad():
        left = (logits - upper).amin(-1, keepdim=True) - 1
        right = (logits - lower).amax(-1, keepdim=True) + 1
        for _ in range(48):
            middle = (left + right) / 2
            mass = (logits - middle).clamp(min=lower, max=upper).sum(-1, keepdim=True)
            left = torch.where(mass > 1, middle, left)
            right = torch.where(mass > 1, right, middle)
        shifted = logits - (left + right) / 2
        free = (shifted > lower + 1e-7) & (shifted < upper - 1e-7)
        at_upper = shifted >= upper - 1e-7
        # Fixed bounds are accounted for as lower, not twice.
        at_upper = at_upper & ~free & (upper > lower + 1e-7)
        at_lower = ~free & ~at_upper
    count = free.sum(-1, keepdim=True).clamp(min=1)
    threshold = (
        (logits * free).sum(-1, keepdim=True)
        + (upper * at_upper).sum(-1, keepdim=True)
        + (lower * at_lower).sum(-1, keepdim=True)
        - 1
    ) / count
    return torch.where(free, logits - threshold, torch.where(at_upper, upper, lower)).to(
        output_dtype
    )


class DirectBudgetModel(nn.Module):
    """Map observed history and constraints directly to a budget allocation.

    Inputs are historical nonnegative spend [batch, time, channel], historical
    outcome [batch, time], total future budget [batch], and lower and upper
    bounds [batch, channel] in budget units. Output has the same units as the
    future budget. The model does not return sales or channel contributions.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.observation = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.channel = nn.Sequential(
            nn.Linear(hidden + 5, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, spend, outcome, budget, lower, upper):
        if spend.ndim != 3 or outcome.shape != spend.shape[:2]:
            raise ValueError("spend must be [batch,time,channel] and outcome [batch,time]")
        b, t, k = spend.shape
        if t < 2 or k < 1 or budget.shape != (b,) or lower.shape != (b, k) or upper.shape != (b, k):
            raise ValueError("invalid budget, history or bound shapes")
        if not all(torch.isfinite(x).all().item() for x in (spend, outcome, budget, lower, upper)):
            raise ValueError("inputs must be finite")
        if (spend < 0).any().item() or (budget <= 0).any().item():
            raise ValueError("historical spend must be nonnegative and budget positive")
        if ((lower < 0) | (upper < lower)).any().item():
            raise ValueError("bounds must satisfy 0 <= lower <= upper")
        # A per-channel ceiling above the total budget is redundant, not invalid.
        lo = lower / budget[:, None]
        hi = torch.minimum(upper, budget[:, None]) / budget[:, None]
        # Time series are expressed in units of the decision budget. Outcome
        # scaling is context-only and removes a nuisance scale and baseline.
        x = spend / budget[:, None, None]
        y = (outcome - outcome.mean(1, keepdim=True)) / outcome.std(1, keepdim=True).clamp_min(1e-6)
        y = y.unsqueeze(-1).expand(-1, -1, k)
        observations = torch.stack((x, x.square(), y, x * y), dim=-1)
        encoded = self.observation(observations).mean(dim=1)
        extras = torch.stack((x.mean(1), x.std(1), lo, hi, x[:, -1, :]), dim=-1)
        h = self.channel(torch.cat((encoded, extras), dim=-1))
        interaction, _ = self.attention(h, h, h, need_weights=False)
        scores = self.head(torch.cat((h, interaction), dim=-1)).squeeze(-1)
        return bounded_simplex(scores, lo, hi) * budget[:, None]
