"""Experimental direct budget policy with a constrained residual correction."""

import torch
from torch import nn

from pymc_mfm.budget.model import bounded_simplex
from pymc_mfm.budget.temporal import TemporalBudgetModel


class ResidualBudgetModel(nn.Module):
    """Return a feasible budget by correcting a frozen temporal budget target.

    The correction is initialized to zero. The base produces an unconstrained
    allocation; both its projection and the corrected allocation obey the same
    total and channel bounds. This is a decision model, not a response model.
    """

    def __init__(self, base: TemporalBudgetModel | None = None, *, hidden: int = 64):
        super().__init__()
        self.base = base if base is not None else TemporalBudgetModel(hidden=hidden)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.encode = nn.Sequential(nn.Linear(8, 48), nn.GELU(), nn.Linear(48, 48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, spend, outcome, controls, budget, lower, upper):
        # The base validates history, total, finiteness and feasibility.
        scale = budget[:, None]
        with torch.no_grad():
            # Validate the actual limits before using unconstrained limits.
            self.base(spend, outcome, controls, budget, lower, upper)
            free = self.base(
                spend, outcome, controls, budget, torch.zeros_like(lower), scale.expand_as(upper)
            )
            upper_fraction = torch.minimum(upper, scale) / scale
            projected = bounded_simplex(free / scale, lower / scale, upper_fraction)
        xs = spend / scale[:, None, :]
        yn = (outcome - outcome.mean(1, keepdim=True)) / outcome.std(1, keepdim=True).clamp_min(
            1e-6
        )
        covariance = ((xs - xs.mean(1, keepdim=True)) * yn[:, :, None]).mean(1)
        features = torch.stack(
            (
                free / scale,
                projected,
                lower / scale,
                upper_fraction,
                xs.mean(1),
                xs.std(1),
                xs[:, -1],
                covariance,
            ),
            dim=-1,
        )
        encoded = self.encode(features)
        pooled = encoded.mean(1, keepdim=True).expand_as(encoded)
        adjustment = self.head(torch.cat((encoded, pooled), dim=-1)).squeeze(-1)
        return scale * bounded_simplex(projected + adjustment, lower / scale, upper_fraction)
