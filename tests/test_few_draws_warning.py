"""Sampling with few draws warns once, never changes the draws, and nests quietly."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from pymc_mfm.inference.api import decompose
from pymc_mfm.model.flow import DEFAULT_N_SAMPLES, STABLE_N_SAMPLES, _no_few_draws_warning

COLUMNS = {
    "date_column": "date",
    "target_column": "sales",
    "channel_columns": ("search", "tv"),
    "control_columns": ("temperature",),
}


@pytest.fixture
def observations(synthetic_corpus):
    corpus = synthetic_corpus
    context = int(corpus["support_mask"][0].sum())
    return {
        "spend": corpus["treatment_raw"][0],
        "controls": corpus["covariates"][0],
        "context_outcome": corpus["outcome_raw"][0, :context],
        "channel_mask": corpus["treatment_active_mask"][0],
        "control_mask": corpus["covariate_active_mask"][0],
        "latent_mask": corpus["latent_active_mask"][0],
    }


@pytest.fixture
def fitted(model):
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "sales": [100.0, 125.0, 110.0, 140.0, 135.0],
            "search": [20.0, 35.0, 25.0, 45.0, 30.0],
            "tv": [80.0, 50.0, 65.0, 40.0, 55.0],
            "temperature": [-2.0, 1.0, 3.0, -1.0, 4.0],
        }
    )
    future = pd.DataFrame(
        {
            "date": pd.date_range("2024-02-05", periods=3, freq="W-MON"),
            "search": [60.0, 90.0, 15.0],
            "tv": [35.0, 75.0, 100.0],
            "temperature": [7.0, -3.0, 2.0],
        }
    )
    return model.add_context(history, **COLUMNS), future


def _few_draw_warnings(record):
    return [w for w in record if "few draws" in str(w.message)]


def test_threshold_and_default_are_documented_constants():
    assert STABLE_N_SAMPLES == 1_000
    assert DEFAULT_N_SAMPLES == 32


def test_decompose_warns_below_threshold_with_advice(model, observations):
    with pytest.warns(UserWarning, match="vary between runs") as record:
        decompose(model, observations, n_samples=4)
    (warning,) = _few_draw_warnings(record)
    message = str(warning.message)
    assert "n_samples=4" in message
    assert f"n_samples={STABLE_N_SAMPLES}" in message
    assert "torch.manual_seed" in message
    assert warning.filename == __file__


def test_default_draw_count_warns(model, observations):
    with pytest.warns(UserWarning, match=f"n_samples={DEFAULT_N_SAMPLES}:"):
        decompose(model, observations)


def test_at_and_above_threshold_do_not_warn(model, observations):
    # Stub the sampler so the draw count costs nothing; only the warning is under test.
    sampled = model.sample_disentangled
    model.sample_disentangled = lambda batch, n_samples, **kw: sampled(batch, 1, **kw)
    for n in (STABLE_N_SAMPLES, STABLE_N_SAMPLES + 1):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            decompose(model, observations, n_samples=n)
        assert not _few_draw_warnings(record)


def test_inference_warns_once_through_decompose(fitted):
    model, future = fitted
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.inference(future, n_samples=3)
    (warning,) = _few_draw_warnings(record)
    assert warning.filename == __file__


def test_tensor_level_entry_points_warn_once(model, batch):
    for call in (
        lambda: model.predict_batch(batch, 3),
        lambda: model.sample_context(batch, 3),
        lambda: model.sample_disentangled(batch, 3),
    ):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            call()
        assert len(_few_draw_warnings(record)) == 1


def test_library_internal_calls_stay_quiet(model, observations):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with _no_few_draws_warning():
            decompose(model, observations, n_samples=3)
    assert not _few_draw_warnings(record)


def test_warning_does_not_change_draws(model, observations):
    def run():
        torch.manual_seed(5)
        return decompose(model, observations, n_samples=4)

    with pytest.warns(UserWarning, match="few draws"):
        warned = run()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        silent = run()
    np.testing.assert_array_equal(warned.components, silent.components)
    np.testing.assert_array_equal(warned.graphs, silent.graphs)
    assert warned.n_samples == silent.n_samples == 4
