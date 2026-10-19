"""Sampling evaluates each flow only on coordinates live in some task.

A coordinate masked in every task is projected to zero after each Euler step, so
the sampler never evaluates its velocity. On every coordinate it does evaluate,
the velocity must equal the trained ``JointFlowHead.velocity`` up to float64
round-off; otherwise seeded draws would silently leave the trained sampler.
"""

import pytest
import torch

from pymc_mfm.model.flow import _LiveVelocity
from pymc_mfm.model.graph import EdgeSlotLayout

ROUND_OFF = {"rtol": 1e-10, "atol": 1e-12}


@pytest.fixture
def model64(model):
    return model.double()


@pytest.fixture
def batch64(batch):
    batch = {key: value.double() for key, value in batch.items()}
    # Channel 2 is padding in both tasks, so its coordinate and slots are dead;
    # control 1 and demand 1 are live in one task only.
    batch["active_c_mask"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float64)
    batch["active_m_mask"] = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    batch["active_j_mask"] = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    return batch


def _assert_live_velocity_matches(head, weekly, channels, controls, mask, *, n_samples=3):
    d = mask.shape[-1]
    generator = torch.Generator().manual_seed(5)
    shape = (n_samples, *weekly.shape[:-1], d)
    x = torch.randn(shape, generator=generator, dtype=torch.float64) * mask
    latent = torch.randn(shape, generator=generator, dtype=torch.float64) * mask
    live = _LiveVelocity(head, weekly, channels, controls, latent, mask)

    n_decomposition = 1 + channels.shape[-2] + controls.shape[-2] + head.n_fixed
    always = {0, *range(n_decomposition - head.n_fixed, n_decomposition)}
    live_somewhere = set(torch.nonzero((mask != 0).reshape(-1, d).any(dim=0)).flatten().tolist())
    evaluated = set(live.index.tolist())
    assert live_somewhere <= evaluated, "a coordinate live in some task was skipped"
    assert evaluated - live_somewhere <= always, "a coordinate dead in every task was evaluated"

    def expand(value):
        return value.unsqueeze(0).expand(n_samples, *value.shape)

    for t in (0.0, 0.3125, 0.9375):
        full = head.velocity(
            x,
            torch.full(shape[:-1], t, dtype=torch.float64),
            expand(weekly),
            expand(channels),
            expand(controls),
            latent,
            mask,
        )
        torch.testing.assert_close(
            live(x.index_select(-1, live.index), t),
            full.index_select(-1, live.index),
            **ROUND_OFF,
        )
    return live_somewhere


@pytest.mark.parametrize("window", ["query", "context"])
def test_router_live_velocity_matches_the_trained_velocity(model64, batch64, window):
    timeline = model64.encoder.forward_timeline(batch64)
    representations = timeline.query if window == "query" else timeline.context
    mask = model64._flow_state_mask(batch64)
    live = _assert_live_velocity_matches(
        model64.head,
        representations.weekly,
        representations.channels,
        representations.controls,
        mask,
    )
    assert len(live) < mask.shape[-1], "the fixture must leave coordinates dead in every task"


@pytest.mark.parametrize("graphs", ["active_and_empty", "both_empty"])
def test_decoder_live_velocity_matches_the_trained_velocity(model64, batch64, graphs):
    weekly, channels, controls = model64.encoder.forward_with_controls(batch64)
    layout = EdgeSlotLayout(3, 2, 2)
    state_mask = model64._flow_state_mask(batch64)
    possible = state_mask[:, 0, -layout.n_slots :]
    # An empty graph forbids every gated contribution: only the baseline stays stochastic.
    graph = possible * torch.tensor([[1.0], [0.0]] if graphs == "active_and_empty" else 0.0)
    n_dec = model64.conditional_head.state_dim(layout.K, layout.M)
    mask = model64._conditioned_state_mask(state_mask[..., :n_dec], graph, layout)
    _assert_live_velocity_matches(model64.conditional_head, weekly, channels, controls, mask)
