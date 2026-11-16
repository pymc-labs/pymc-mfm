"""Small complete-model fixtures; no external weights, corpus or service."""

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import torch

from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.model.model import MFM

GENERATOR_EDGE_TYPES = ["cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"]
#: Skips a test unless the ``arviz`` extra (xarray, arviz-base, arviz-stats) is installed.
needs_arviz = pytest.mark.skipif(
    any(find_spec(name) is None for name in ("xarray", "arviz_base", "arviz_stats")),
    reason="needs the pymc-mfm[arviz] extra",
)


def make_synthetic_corpus(
    *,
    n_cells: int = 6,
    draws_per_cell: int = 2,
    n_weeks: int = 12,
    context_lengths: tuple[int, ...] = (6, 9),
    k: int = 3,
    m: int = 2,
    j: int = 2,
    val_cells: int = 2,
    seed: int = 0,
) -> dict:
    """A schema-4 shaped corpus satisfying every ``corpus_batch`` constraint.

    Worlds are grouped into cells; ``is_val`` is constant within a cell and the
    last ``val_cells`` cells are held out. Cells alternate context lengths so a
    corpus mixes geometries exactly like the generator's horizon splits. The
    first channel is always active and the last channel is always a padded
    slot, so padding-leakage metrics are defined for every world. The outcome
    is the exact sum of the decomposition components plus noise, and stays
    strictly positive through a large intrinsic baseline.
    """
    rng = np.random.default_rng(seed)
    n = n_cells * draws_per_cell
    cell_id = np.repeat(np.arange(n_cells, dtype=np.int32), draws_per_cell)
    is_val = (cell_id >= n_cells - val_cells).astype(np.uint8)
    lengths = np.asarray(context_lengths)[cell_id % len(context_lengths)]
    support = (np.arange(n_weeks)[None, :] < lengths[:, None]).astype(np.uint8)

    def active(width: int, padded: bool = False) -> np.ndarray:
        mask = rng.random((n_cells, width)) < 0.7
        mask[:, 0] = True
        if padded:
            mask[:, -1] = False
        return mask[cell_id].astype(np.uint8)

    active_c, active_m, active_j = active(k, padded=True), active(m), active(j)
    spend = np.exp(rng.normal(8.0, 0.5, (n, n_weeks, k))).astype(np.float32) * active_c[:, None]
    controls = rng.normal(0.0, 1.0, (n, n_weeks, m)).astype(np.float32) * active_m[:, None]
    baseline = 500.0 + rng.normal(0.0, 5.0, (n, n_weeks))
    direct = np.abs(rng.normal(20.0, 5.0, (n, n_weeks, k))) * active_c[:, None]
    covariate = rng.normal(0.0, 3.0, (n, n_weeks, m)) * active_m[:, None]
    latent = rng.normal(0.0, 2.0, (n, n_weeks, j)) * active_j[:, None]
    indirect = rng.normal(0.0, 1.0, (n, n_weeks, 3))
    noise = rng.normal(0.0, 1.0, (n, n_weeks))
    outcome = (
        baseline + direct.sum(-1) + covariate.sum(-1) + latent.sum(-1) + indirect.sum(-1) + noise
    )
    layout = EdgeSlotLayout(k, m, j)
    allowed = layout.active_slot_mask(active_c, active_m, active_j)
    graph = ((rng.random((n_cells, layout.n_slots)) < 0.5)[cell_id] & (allowed > 0)).astype(
        np.uint8
    )
    return {
        "diagnostics": {"schema_version": 4, "edge_types": list(GENERATOR_EDGE_TYPES)},
        "cell_id": cell_id,
        "is_val": is_val,
        "support_mask": support,
        "treatment_active_mask": active_c,
        "covariate_active_mask": active_m,
        "latent_active_mask": active_j,
        "treatment_raw": spend,
        "covariates": controls,
        "outcome_raw": outcome.astype(np.float32),
        "outcome_noise": noise.astype(np.float32),
        "baseline_intrinsic": baseline.astype(np.float32),
        "treatment_contribution_raw": direct.astype(np.float32),
        "covariate_contribution": covariate.astype(np.float32),
        "latent_unobserved_contribution": latent.astype(np.float32),
        "indirect_effects_by_source": indirect.astype(np.float32),
        "g": graph,
    }


@pytest.fixture
def synthetic_corpus():
    return make_synthetic_corpus()


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
            n_controls=2,
            n_demand=2,
        )
    ).eval()


#: Column roles of :func:`chart_frames`: two channels and one control.
CHART_ROLES = {
    "date_column": "date",
    "target_column": "sales",
    "channel_columns": ("search", "tv"),
    "control_columns": ("temperature",),
}


@pytest.fixture
def chart_frames():
    """24 history weeks and 6 forecast weeks with observed sales, for ``model.plot``."""
    rng = np.random.default_rng(5)
    weeks = 30
    search = rng.uniform(10.0, 60.0, weeks)
    tv = np.where(rng.random(weeks) > 0.4, rng.uniform(20.0, 90.0, weeks), 0.0)
    temperature = rng.normal(0.0, 3.0, weeks)
    sales = 200.0 + 0.8 * search + 0.5 * tv + 2.0 * temperature + rng.normal(0.0, 5.0, weeks)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=weeks, freq="W-MON"),
            "sales": sales,
            "search": search,
            "tv": tv,
            "temperature": temperature,
        }
    )
    return frame.iloc[:24].reset_index(drop=True), frame.iloc[24:].reset_index(drop=True)


@pytest.fixture
def predicted_model(model, chart_frames):
    """``model`` after ``add_context`` on the history and 40-draw ``inference`` on the rest."""
    history, future = chart_frames
    model.add_context(history, **CHART_ROLES)
    torch.manual_seed(3)
    model.inference(future, n_samples=40)
    return model


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
