"""Small complete-model fixtures; no external weights, corpus or service."""

import pytest
import torch

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.model.model import MFM


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture
def model():
    torch.manual_seed(10)
    return MFM(
        ModelConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            flow_hidden=16,
            flow_n_freq=1,
            flow_steps_sample=2,
            flow_mc_samples=3,
            n_controls=2,
            n_demand=2,
        )
    ).eval()


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(11)

    def normal(*shape):
        return torch.randn(*shape, generator=generator)

    b, c, m, j, context, query = 2, 3, 2, 2, 5, 3
    width = 1 + c + m + 4
    result = {
        "ctx_spend_z": normal(b, context, c),
        "qry_spend_z": normal(b, query, c),
        "ctx_controls_z": normal(b, context, m),
        "qry_controls_z": normal(b, query, m),
        "ctx_revenue_z": normal(b, context, 1),
        "spend_mean": normal(b, c),
        "spend_std": normal(b, c).abs() + 1,
        "ctx_week": normal(b, context, 1),
        "qry_week": normal(b, query, 1),
        "response_moments": normal(b, c, 4),
        "adstock_moments": normal(b, c, 9),
        "control_moments": normal(b, m, 3),
        "control_location_scale": normal(b, m, 2),
        "control_effect_moments": normal(b, m, 6),
        "active_c_mask": torch.ones(b, c),
        "active_m_mask": torch.ones(b, m),
        "active_j_mask": torch.ones(b, j),
        "targets_z": normal(b, query, width),
        "context_targets_z": normal(b, context, width),
    }
    result["active_c_mask"][0, -1] = 0
    result["active_m_mask"][1, -1] = 0
    result["active_j_mask"][0, -1] = 0
    layout = EdgeSlotLayout(c, m, j)
    graph = layout.active_slot_mask(
        result["active_c_mask"], result["active_m_mask"], result["active_j_mask"]
    )
    result.update({f"g_{family}": value for family, value in layout.unpack(graph).items()})
    return result
