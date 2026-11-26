"""Test order-sensitive budget policy and safe local checkpoints."""

import torch
from safetensors.torch import save_file

from pymc_mfm.budget.temporal import TemporalBudgetModel, TemporalObservation
from pymc_mfm.budget.temporal_io import load_temporal_budget_model


def test_temporal_encoder_can_distinguish_reordered_history():
    encoder = TemporalObservation(hidden=1)
    with torch.no_grad():
        encoder.network[0].weight.zero_()
        encoder.network[0].bias.zero_()
        encoder.network[0].weight[0, 7] = 1
        encoder.network[2].weight.fill_(1)
        encoder.network[2].bias.zero_()
    obs = torch.zeros(1, 4, 1, 5)
    obs[0, :, 0, 2] = torch.tensor([0.0, 1.0, 0.0, 2.0])
    obs[0, :, 0, 3] = torch.tensor([0.0, 0.0, 1.0, 1.0])
    permutation = torch.tensor([1, 0, 2, 3])
    assert not torch.allclose(encoder(obs).mean(1), encoder(obs[:, permutation]).mean(1))


def test_temporal_checkpoint_round_trip_returns_feasible_budget(tmp_path):
    torch.manual_seed(15)
    original = TemporalBudgetModel()
    path = tmp_path / "temporal.safetensors"
    save_file(
        original.state_dict(),
        str(path),
        metadata={"architecture": "temporal-budget-v1", "hidden": "64"},
    )
    model = load_temporal_budget_model(path)
    spend = torch.rand(2, 20, 3) * 10
    outcome = torch.randn(2, 20)
    controls = torch.randn(2, 20, 2)
    budget = torch.full((2,), 100.0)
    lo = torch.zeros(2, 3)
    hi = torch.tensor([[10.0, 100.0, 100.0], [10.0, 100.0, 100.0]])
    with torch.no_grad():
        allocation = model(spend, outcome, controls, budget, lo, hi)
    assert torch.allclose(allocation.sum(1), budget, atol=1e-4)
    assert (allocation >= lo - 1e-4).all() and (allocation <= hi + 1e-4).all()
