"""Order-aware input encoder for direct budget decisions."""

import torch
from torch import nn

from pymc_mfm.budget.controlled import ControlledBudgetModel


class TemporalObservation(nn.Module):
    """Encode contemporaneous features and their associations at two lags."""

    def __init__(self, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )

    def forward(self, observations):
        x, y = observations[..., 2], observations[..., 3]
        one = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        two = torch.cat((torch.zeros_like(x[:, :2]), x[:, :-2]), dim=1)
        features = torch.cat(
            (
                observations,
                one.unsqueeze(-1),
                two.unsqueeze(-1),
                (one * y).unsqueeze(-1),
                (two * y).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.network(features)


class TemporalBudgetModel(ControlledBudgetModel):
    """Direct budget policy with order-sensitive historical features."""

    def __init__(self, hidden=64):
        super().__init__(hidden=hidden, joint_features=True)
        self.observation = TemporalObservation(hidden)
