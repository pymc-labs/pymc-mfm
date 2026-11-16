"""``model.plot``: state checks, backends, shared context draws and the inputs chart."""

import sys

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES, needs_arviz

from pymc_mfm.plotting import set_backend
from pymc_mfm.plotting.charts import _backend
from pymc_mfm.plotting.charts._data import ModelSource
from pymc_mfm.plotting.charts._style import compact

pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


def _line(fig, row=0):
    return fig.axes[row].lines[0].get_ydata()


def test_charts_need_context_and_draws_need_inference(model, chart_frames):
    history, future = chart_frames
    with pytest.raises(RuntimeError, match="add_context"):
        model.plot.inputs()
    model.add_context(history, **CHART_ROLES)
    fig = model.plot.inputs()
    np.testing.assert_array_equal(_line(fig), history["sales"])
    with pytest.raises(RuntimeError, match=r"model\.inference"):
        ModelSource(model).window("query")


def test_inputs_show_history_forecast_weeks_and_observed_outcome(predicted_model, chart_frames):
    history, future = chart_frames
    fig = predicted_model.plot.inputs()
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    # The target column is called "sales"; charts call it the outcome.
    assert titles == [
        "outcome",
        "search (media covariate)",
        "tv (media covariate)",
        "temperature (control covariate)",
    ]
    both = [history, future]
    np.testing.assert_array_equal(_line(fig, 0), np.concatenate([f["sales"] for f in both]))
    np.testing.assert_array_equal(_line(fig, 2), np.concatenate([f["tv"] for f in both]))
    # Every row shades the forecast weeks, and only them.
    positions = fig.axes[0].lines[0].get_xdata()
    for ax in fig.axes:
        [span] = ax.patches
        left, right = span.get_x(), span.get_x() + span.get_width()
        assert positions[23] < left < positions[24] and right > positions[-1]


@pytest.mark.parametrize(
    ("sales", "expected"),
    [
        pytest.param(None, [np.nan] * 6, id="no-target-column"),
        pytest.param(["not observed", None, -np.inf] * 2, [np.nan] * 6, id="nothing-numeric"),
        pytest.param(["1.5", "x", 3, np.inf, 4.0, None], [1.5, np.nan, 3, np.nan, 4, np.nan]),
        pytest.param(
            [10**400, 2.0, "3", None, [1], 5],
            [np.nan, 2, 3, np.nan, np.nan, 5],
            id="int-past-float64",
        ),
    ],
)
def test_forecast_outcomes_are_kept_for_charts_without_changing_inference(
    model, chart_frames, sales, expected
):
    history, future = chart_frames
    if sales is None:
        future = future.drop(columns="sales")
    else:  # an object column, as a frame read from mixed sources would hold
        future = future.assign(sales=pd.Series(sales, index=future.index, dtype=object))
    model.add_context(history, **CHART_ROLES)
    torch.manual_seed(3)
    draws = model.inference(future, n_samples=5).components
    torch.manual_seed(3)
    np.testing.assert_array_equal(
        model.inference(future.drop(columns="sales", errors="ignore"), n_samples=5).components,
        draws,
    )
    model.inference(future, n_samples=5)
    np.testing.assert_array_equal(_line(model.plot.inputs(), 0)[24:], expected)


def test_forecast_outcomes_follow_the_sorted_dates(model, chart_frames):
    history, future = chart_frames
    model.add_context(history, **CHART_ROLES)
    model.inference(future.iloc[::-1], n_samples=5)
    np.testing.assert_array_equal(_line(model.plot.inputs(), 0)[24:], future["sales"])


def test_windows_name_components_and_align_inputs(predicted_model, chart_frames):
    history, future = chart_frames
    source = ModelSource(predicted_model)
    query = source.window("query")
    assert query.components[:3] == ("baseline", "search", "tv")
    assert query.components[3] == "temperature" and "__padded_control_1__" not in query.components
    assert query.draws.shape == (40, 6, len(query.components))
    np.testing.assert_array_equal(query.outcome, future["sales"])
    np.testing.assert_array_equal(query.media, future[["search", "tv"]])
    np.testing.assert_array_equal(query.dates, future["date"].to_numpy())
    np.testing.assert_allclose(
        query.expected, predicted_model._prediction_snapshot.query.components.sum(2)
    )
    context = source.window("context")
    np.testing.assert_array_equal(context.outcome, history["sales"])
    assert context.draws.shape[:2] == (40, 24)


def test_names_like_other_components_keep_full_names_and_their_own_draws(model, chart_frames):
    renamed = {"tv": "baseline", "temperature": "channel:baseline"}
    history, future = (frame.rename(columns=renamed) for frame in chart_frames)
    roles = {
        **CHART_ROLES,
        "channel_columns": ("search", "baseline"),
        "control_columns": ("channel:baseline",),
    }
    model.add_context(history, **roles)
    result = model.inference(future, n_samples=5)
    window = ModelSource(model).window("query")
    assert window.components[:4] == (
        "baseline",
        "search",
        "channel:baseline",
        "control:channel:baseline",
    )
    assert len(set(window.components)) == len(window.components)
    for name in ("channel:baseline", "control:channel:baseline"):
        index = result.component_names.index(name)
        np.testing.assert_array_equal(window.contribution(name), result.components[..., index])


@pytest.mark.parametrize(
    ("value", "label"),
    [(950, "950"), (999.5, "1k"), (12_500, "12.5k"), (999_999, "1M"), (-1_234.5, "-1.23k")],
)
def test_compact_labels_round_before_choosing_the_unit(value, label):
    assert compact(value) == label


def test_context_draws_are_sampled_once_and_shared_with_build_idata(predicted_model):
    source = ModelSource(predicted_model)
    first = source.window("context")
    state = torch.get_rng_state()
    second = source.window("context")
    assert torch.equal(torch.get_rng_state(), state)
    np.testing.assert_array_equal(first.draws, second.draws)


@needs_arviz
def test_build_idata_reuses_the_context_draws_charts_sampled(predicted_model):
    context = ModelSource(predicted_model).window("context")
    state = torch.get_rng_state()
    tree = predicted_model.build_idata()
    assert torch.equal(torch.get_rng_state(), state)
    exported = tree["context"].posterior["channel_contribution"].sel(channel="search")
    np.testing.assert_array_equal(exported.values[0], context.contribution("search"))


def test_backend_choice_and_its_install_hint(predicted_model, monkeypatch):
    with pytest.raises(ValueError, match="backend must be one of"):
        set_backend("bokeh")
    with pytest.raises(ValueError, match="backend must be one of"):
        predicted_model.plot.inputs(backend="bokeh")
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(ImportError, match=r'pip install "pymc-mfm\[interactive\]"'):
        predicted_model.plot.inputs(backend="plotly")
    set_backend("plotly")
    with pytest.raises(ImportError, match="interactive"):
        predicted_model.plot.inputs()


def test_plotly_inputs_match_matplotlib(predicted_model, chart_frames):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    history, future = chart_frames
    set_backend("plotly")
    fig = predicted_model.plot.inputs()
    assert isinstance(fig, go.Figure)
    assert [trace.name for trace in fig.data] == ["outcome", "search", "tv", "temperature"]
    np.testing.assert_array_equal(
        fig.data[0].y, np.concatenate([history["sales"], future["sales"]])
    )
    # Every row shades the forecast weeks, and only them.
    assert len(fig.layout.shapes) == len(fig.data)
    for span in fig.layout.shapes:
        assert history["date"].iloc[-1] < span.x0 < future["date"][0] < span.x1
        assert span.x1 > future["date"].iloc[-1]
