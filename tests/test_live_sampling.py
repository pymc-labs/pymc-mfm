"""Sampling evaluates each flow only on coordinates live in some task.

A coordinate masked in every task is projected to zero after each Euler step, so
the sampler never evaluates its velocity. On every coordinate it does evaluate,
the velocity must equal the trained ``JointFlowHead.velocity`` up to float64
round-off; otherwise seeded draws would silently leave the trained sampler.
Noise is drawn per chunk of draws; how draws are then batched for transport, and
whether batches run concurrently, must not change any draw.
"""

import pytest
import torch

import pymc_mfm.model.flow as flow
import pymc_mfm.model.model as model_module
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
    live = _LiveVelocity(head, weekly, channels, controls, mask)
    offset = live.trajectory_offset(latent)

    live_somewhere = set(torch.nonzero((mask != 0).reshape(-1, d).any(dim=0)).flatten().tolist())
    assert 0 in live_somewhere, "the baseline coordinate is never masked"
    assert set(live.index.tolist()) == live_somewhere, "evaluated coordinates must be the live ones"

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
            live.velocity(x.index_select(-1, live.index), t, offset),
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


def _draws(model, batch, monkeypatch, *, rows, workers, window):
    monkeypatch.setattr(flow, "_FLOW_SAMPLE_ROWS", rows)
    monkeypatch.setattr(model_module, "_FLOW_SAMPLE_ROWS", rows)
    monkeypatch.setattr(flow, "_SAMPLE_BATCH_WORKERS", workers)
    torch.manual_seed(41)
    # 19 draws cross the 16-draw noise chunk.
    return model.sample_disentangled(batch, 19, window=window)


@pytest.mark.parametrize("window", ["query", "context"])
def test_transport_batches_and_their_schedule_never_change_a_draw(
    model64, batch64, monkeypatch, window
):
    threads = torch.get_num_threads()
    torch.set_num_threads(max(threads, 4))
    try:
        whole, whole_graphs = _draws(
            model64, batch64, monkeypatch, rows=2**30, workers=1, window=window
        )
        # A one-row budget transports every draw in its own batch.
        serial, serial_graphs = _draws(
            model64, batch64, monkeypatch, rows=1, workers=1, window=window
        )
        concurrent, concurrent_graphs = _draws(
            model64, batch64, monkeypatch, rows=1, workers=4, window=window
        )
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(serial_graphs, whole_graphs)
    torch.testing.assert_close(serial, whole, **ROUND_OFF)
    assert torch.equal(concurrent_graphs, serial_graphs)
    assert torch.equal(concurrent, serial)
