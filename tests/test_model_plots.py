"""``model.plot``: complete-inference readers, backends and the inputs chart."""

import sys

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES

from pymc_mfm.plotting import set_backend
from pymc_mfm.plotting.charts import (
    _backend,
    contributions,
    curves,
    fit,
    inputs,
    media,
    shares,
    totals,
)
from pymc_mfm.plotting.charts._data import IdataReader
from pymc_mfm.plotting.charts._style import compact


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


def _line(fig, row=0):
    return fig.axes[row].lines[0].get_ydata()


@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(inputs.prepare, id="inputs"),
        pytest.param(fit.prepare, id="fit"),
        pytest.param(shares.prepare, id="shares"),
        pytest.param(totals.prepare, id="totals"),
        pytest.param(media.prepare, id="media"),
        pytest.param(contributions.prepare, id="contributions"),
        pytest.param(curves.prepare, id="curves"),
    ],
)
def test_charts_need_complete_inference(model, prepare):
    state = torch.get_rng_state()
    with pytest.raises(RuntimeError, match=r"inference\(history, future, \.\.\.\)"):
        prepare(IdataReader(model.idata))
    assert torch.equal(torch.get_rng_state(), state)
    assert model.idata is None and model.query_result is None


@pytest.mark.usefixtures("matplotlib_backend")
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
    torch.manual_seed(3)
    model.inference(history, future, n_samples=5, **CHART_ROLES)
    draws = model.query_result.components
    torch.manual_seed(3)
    np.testing.assert_array_equal(
        model.predict(future.drop(columns="sales", errors="ignore"), n_samples=5).components,
        draws,
    )
    np.testing.assert_array_equal(
        inputs.prepare(IdataReader(model.idata)).series[0][2][24:], expected
    )


def test_forecast_outcomes_follow_the_sorted_dates(model, chart_frames):
    history, future = chart_frames
    model.inference(history, future.iloc[::-1], n_samples=5, **CHART_ROLES)
    np.testing.assert_array_equal(
        inputs.prepare(IdataReader(model.idata)).series[0][2][24:], future["sales"]
    )


def test_windows_name_components_and_align_inputs(predicted_model, chart_frames):
    history, future = chart_frames
    source = IdataReader(predicted_model.idata)
    query = source.window("query")
    assert query.components[:3] == ("baseline", "search", "tv")
    assert query.components[3] == "temperature" and "__padded_control_1__" not in query.components
    assert query.draws.shape == (40, 6, len(query.components))
    np.testing.assert_array_equal(query.outcome, future["sales"])
    np.testing.assert_array_equal(query.media, future[["search", "tv"]])
    np.testing.assert_array_equal(query.dates, future["date"].to_numpy())
    np.testing.assert_allclose(query.expected, predicted_model.query_result.components.sum(2))
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
    model.inference(history, future, n_samples=5, **roles)
    result = model.query_result
    window = IdataReader(model.idata).window("query")
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


def test_chart_readers_never_sample_and_match_the_complete_tree(predicted_model, monkeypatch):
    source = IdataReader(predicted_model.idata)
    tree = predicted_model.idata
    query = predicted_model.query_result
    state = torch.get_rng_state()

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("chart readers must not sample")

    monkeypatch.setattr(predicted_model, "sample_disentangled", unexpected_sampling)
    for name in ("context", "query"):
        first = source.window(name)
        second = source.window(name)
        np.testing.assert_array_equal(first.draws, second.draws)
        posterior = tree[f"{name}/posterior"]
        np.testing.assert_array_equal(posterior["date"], first.dates)
        np.testing.assert_array_equal(
            posterior["baseline"].values[0], first.contribution("baseline")
        )
        for channel in first.channels:
            exported = posterior["channel_contribution"].sel(channel=channel).values[0]
            np.testing.assert_array_equal(exported, first.contribution(channel))
        exported = posterior["control_contribution"].sel(control="temperature").values[0]
        np.testing.assert_array_equal(exported, first.contribution("temperature"))
    for prepare in (
        inputs.prepare,
        fit.prepare,
        shares.prepare,
        totals.prepare,
        media.prepare,
        contributions.prepare,
    ):
        prepare(source)
    assert torch.equal(torch.get_rng_state(), state)
    assert predicted_model.idata is tree and predicted_model.query_result is query


def test_chart_readers_stay_on_complete_inference_after_predict(predicted_model, chart_frames):
    _, future = chart_frames
    source = IdataReader(predicted_model.idata)
    tree = predicted_model.idata
    query = predicted_model.query_result
    before = source.windows(None)
    expected_inputs = inputs.prepare(source)
    independent = predicted_model.predict(
        future.iloc[:2].assign(search=1.0, sales=-1.0), n_samples=3, mode="map", units="USD"
    )
    assert independent is not query and independent.n_samples == 3
    assert independent.units == "USD"
    state = torch.get_rng_state()
    for previous, current in zip(before, source.windows(None), strict=True):
        np.testing.assert_array_equal(current.draws, previous.draws)
        np.testing.assert_array_equal(current.dates, previous.dates)
        np.testing.assert_array_equal(current.media, previous.media)
        np.testing.assert_array_equal(current.outcome, previous.outcome)
        assert current.units == previous.units == "outcome"
    current_inputs = inputs.prepare(source)
    np.testing.assert_array_equal(current_inputs.dates, expected_inputs.dates)
    for previous, current in zip(expected_inputs.series, current_inputs.series, strict=True):
        assert current[:2] == previous[:2]
        np.testing.assert_array_equal(current[2], previous[2])
    assert torch.equal(torch.get_rng_state(), state)
    assert predicted_model.idata is tree and predicted_model.query_result is query


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
