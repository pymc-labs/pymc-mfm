"""``model.plot.contribution_totals`` and ``model.plot.contribution_vs_media``."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES

from pymc_mfm.plotting.charts import _backend
from pymc_mfm.plotting.charts._data import ModelSource

pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


def _summary(values, interval=0.9):
    """Mean and central ``interval`` bounds over the draws (axis 0), by hand."""
    low, high = np.quantile(values, [(1 - interval) / 2, (1 + interval) / 2], axis=0)
    return values.mean(axis=0), low, high


def _panels(fig):
    """The drawn panels in order; a hidden axes fills an odd grid."""
    return [ax for ax in fig.axes if ax.get_visible()]


def _histogram(ax):
    from matplotlib.patches import StepPatch

    [steps] = [patch for patch in ax.patches if isinstance(patch, StepPatch)]
    return steps.get_data()


def _span(ax):
    from matplotlib.patches import StepPatch

    [span] = [patch for patch in ax.patches if not isinstance(patch, StepPatch)]
    return span.get_x(), span.get_x() + span.get_width()


def _lines(ax):
    """x of each vertical line by style: ``"-"`` is the mean, ``"--"`` the truth."""
    return {line.get_linestyle(): line.get_xdata()[0] for line in ax.lines}


def _points(ax, label):
    """``[point, (x, y)]`` of the scatter labelled ``label``; NaN where none is drawn."""
    [points] = [collection for collection in ax.collections if collection.get_label() == label]
    return np.ma.filled(np.ma.asarray(points.get_offsets(), dtype=float), np.nan)


def _bars(ax):
    """``(x, low, high)`` of each vertical interval bar."""
    from matplotlib.collections import LineCollection

    [bars] = [collection for collection in ax.collections if isinstance(collection, LineCollection)]
    segments = np.asarray(bars.get_segments())
    return segments[:, 0, 0], segments[:, 0, 1], segments[:, 1, 1]


def _history_truth(history, **columns):
    return pd.DataFrame(columns, index=pd.DatetimeIndex(history["date"]))


def test_totals_bin_each_channel_and_control_per_draw_total(predicted_model):
    fig = predicted_model.plot.contribution_totals()
    context = ModelSource(predicted_model).window("context")
    panels = _panels(fig)
    assert [ax.get_title(loc="left") for ax in panels] == ["search", "tv", "temperature"]
    for ax in panels:
        totals = context.contribution(ax.get_title(loc="left")).sum(axis=1)
        counts, edges = np.histogram(totals, 40)
        drawn = _histogram(ax)
        np.testing.assert_array_equal(drawn.values, counts)
        np.testing.assert_array_equal(drawn.edges, edges)
        assert drawn.values.sum() == context.n_samples == 40
        mean, low, high = _summary(totals)
        assert _lines(ax) == {"-": pytest.approx(mean)}
        np.testing.assert_allclose(_span(ax), (low, high))
        assert ax.get_xlabel() == "Total over 24 history weeks (outcome)"
    assert [ax.get_ylabel() for ax in panels] == ["Draws", "", "Draws"]
    heading = fig.get_suptitle()
    assert heading.startswith("Total contribution over the 24 history weeks, by component")
    assert "40 draws" in heading and "not causal lift" in heading


def test_forecast_totals_of_any_component_with_truth_never_resample(predicted_model, chart_frames):
    history, future = chart_frames
    rng = np.random.default_rng(0)
    weeks = pd.DatetimeIndex(pd.concat([history["date"], future["date"]]))
    truth = pd.DataFrame(
        {"baseline": rng.normal(200.0, 5.0, 30), "tv": rng.uniform(0.0, 50.0, 30)}, index=weeks
    ).iloc[::-1]  # any row order; weeks outside the window are ignored
    context = predicted_model._context_result()
    state = torch.get_rng_state()
    fig = predicted_model.plot.contribution_totals(
        window="query", components=["baseline", "tv"], interval=0.5, bins=7, truth=truth
    )
    assert predicted_model._context_result() is context
    assert torch.equal(torch.get_rng_state(), state)
    query = ModelSource(predicted_model).window("query")
    for ax, name in zip(fig.axes, ["baseline", "tv"], strict=True):
        assert ax.get_title(loc="left") == name
        totals = query.contribution(name).sum(axis=1)
        counts, edges = np.histogram(totals, 7)
        drawn = _histogram(ax)
        np.testing.assert_array_equal(drawn.values, counts)
        np.testing.assert_array_equal(drawn.edges, edges)
        mean, low, high = _summary(totals, 0.5)
        np.testing.assert_allclose(_span(ax), (low, high))
        true_total = truth.loc[future["date"], name].sum()
        assert _lines(ax) == {"-": pytest.approx(mean), "--": pytest.approx(true_total)}
        assert ax.get_xlabel() == "Total over 6 forecast weeks (outcome)"


def test_a_truth_total_needs_every_week_of_the_window(predicted_model, chart_frames):
    history, _ = chart_frames
    truth = _history_truth(history, search=np.ones(24))
    plot = predicted_model.plot
    with pytest.raises(ValueError, match=r"truth\['search'\] has no value for 2024-03-04;"):
        plot.contribution_totals(truth=truth.drop(truth.index[9]))
    with pytest.raises(ValueError, match="no value for 2024-01-01;"):
        plot.contribution_totals(truth=truth.assign(search=np.r_[np.nan, np.ones(23)]))
    with pytest.raises(ValueError, match=r"2024-03-04, .*2024-04-08 and 9 more; .*24 history"):
        plot.contribution_totals(truth=truth.iloc[:9])
    # Only drawn components need their weeks: "search" is not drawn here.
    [ax] = plot.contribution_totals(components=["tv"], truth=truth.drop(truth.index[9])).axes
    assert set(_lines(ax)) == {"-"}


def test_totals_check_their_arguments(predicted_model, chart_frames):
    history, _ = chart_frames
    plot = predicted_model.plot
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match="interval must be a fraction"):
        plot.contribution_totals(interval=1.5)
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(ValueError, match="window must be one of"):
        plot.contribution_totals(window="future")
    with pytest.raises(
        ValueError, match=r"unknown components \['radio'\]; available: \['baseline', 'search'"
    ):
        plot.contribution_totals(components=["search", "radio"])
    with pytest.raises(ValueError, match="not a string"):
        plot.contribution_totals(components="search")
    with pytest.raises(ValueError, match="at least one"):
        plot.contribution_totals(components=[])
    with pytest.raises(ValueError, match=r"truth columns \['sales'\] are not components"):
        plot.contribution_totals(truth=_history_truth(history, search=1.0, sales=1.0))
    with pytest.raises(ValueError, match="indexed by date"):
        plot.contribution_totals(truth=_history_truth(history, search=1.0).reset_index())
    assert torch.equal(torch.get_rng_state(), state)


def test_plotly_totals_draw_the_matplotlib_numbers(predicted_model, chart_frames):
    pytest.importorskip("plotly")
    history, _ = chart_frames
    truth = _history_truth(history, tv=np.linspace(1.0, 24.0, 24))
    drawn = _panels(predicted_model.plot.contribution_totals(truth=truth))
    fig = predicted_model.plot.contribution_totals(truth=truth, backend="plotly")
    assert [bar.name for bar in fig.data] == ["search", "tv", "temperature"]
    for ax, bar in zip(drawn, fig.data, strict=True):
        counts, edges, _ = _histogram(ax)
        np.testing.assert_array_equal(bar.y, counts)
        np.testing.assert_array_equal(bar.x, edges[:-1])
        np.testing.assert_array_equal(bar.width, np.diff(edges))
        assert bar.offset == 0
        shapes = [shape for shape in fig.layout.shapes if shape.xref == bar.xaxis]
        [span] = [shape for shape in shapes if shape.type == "rect"]
        np.testing.assert_allclose((span.x0, span.x1), _span(ax))
        lines = {
            "--" if shape.line.dash == "dash" else "-": shape.x0
            for shape in shapes
            if shape.type == "line"
        }
        assert lines == pytest.approx(_lines(ax))
        # Hovering a bar also reads out the panel's mean and interval.
        np.testing.assert_allclose(
            bar.customdata[0, 1:4], (_lines(ax)["-"], *_span(ax)), rtol=1e-12
        )
    assert "--" in _lines(drawn[1])
    assert "not causal lift" in fig.layout.title.text


def test_contribution_vs_media_plots_each_week_at_its_value(predicted_model, chart_frames):
    history, _ = chart_frames
    fig = predicted_model.plot.contribution_vs_media()
    context = ModelSource(predicted_model).window("context")
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["search", "tv"]
    for ax in fig.axes:
        name = ax.get_title(loc="left")
        mean, low, high = _summary(context.contribution(name))
        points = _points(ax, "Mean")
        np.testing.assert_array_equal(points[:, 0], history[name])
        np.testing.assert_allclose(points[:, 1], mean)
        x, bottom, top = _bars(ax)
        np.testing.assert_array_equal(x, history[name])
        np.testing.assert_allclose(bottom, low)
        np.testing.assert_allclose(top, high)
        assert ax.get_xlabel() == "Media covariate (weekly value)"
    search, tv = fig.axes
    assert search.get_shared_y_axes().joined(search, tv)
    assert (search.get_ylabel(), tv.get_ylabel()) == ("Weekly contribution (outcome)", "")
    assert "carryover" in fig.get_suptitle()


def test_truth_crosses_align_by_date_without_resampling(predicted_model, chart_frames):
    _, future = chart_frames
    truth = pd.DataFrame(
        {"tv": np.arange(6.0) * 10, "baseline": 1.0}, index=pd.DatetimeIndex(future["date"])
    )
    context = predicted_model._context_result()
    state = torch.get_rng_state()
    fig = predicted_model.plot.contribution_vs_media(
        window="query", channels=["tv"], truth=truth.iloc[[3, 0, 5, 1, 4]]
    )
    assert predicted_model._context_result() is context
    assert torch.equal(torch.get_rng_state(), state)
    [ax] = fig.axes
    assert ax.get_title(loc="left") == "tv"
    expected = truth["tv"].to_numpy().copy()
    expected[2] = np.nan  # the truth has no third week: no cross there
    crosses = _points(ax, "Truth")
    np.testing.assert_array_equal(crosses[:, 1], expected)
    np.testing.assert_array_equal(crosses[:, 0], np.where(np.isnan(expected), np.nan, future["tv"]))
    query = ModelSource(predicted_model).window("query")
    np.testing.assert_allclose(_points(ax, "Mean")[:, 1], query.contribution("tv").mean(axis=0))


def test_contribution_vs_media_checks_its_arguments(predicted_model, chart_frames):
    history, _ = chart_frames
    plot = predicted_model.plot
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match="interval must be a fraction"):
        plot.contribution_vs_media(interval=0)
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(
        ValueError, match=r"unknown channels \['baseline'\]; available: \['search', 'tv'\]"
    ):
        plot.contribution_vs_media(channels=["baseline"])
    with pytest.raises(ValueError, match="not a string"):
        plot.contribution_vs_media(channels="tv")
    with pytest.raises(ValueError, match=r"truth columns \['radio'\] are not components"):
        plot.contribution_vs_media(truth=_history_truth(history, radio=1.0))
    sundays = pd.DataFrame(
        {"tv": 1.0}, index=pd.DatetimeIndex(history["date"]) - pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError, match=r"truth\['tv'\] has no value in any of the 24 history"):
        plot.contribution_vs_media(truth=sundays)
    assert torch.equal(torch.get_rng_state(), state)


def test_a_channel_named_like_a_component_keeps_its_own_media_covariate(model, chart_frames):
    history, future = (frame.rename(columns={"tv": "baseline"}) for frame in chart_frames)
    model.inference(
        history,
        future,
        n_samples=5,
        **{**CHART_ROLES, "channel_columns": ("search", "baseline")},
    )
    fig = model.plot.contribution_vs_media(window="query", channels=["channel:baseline"])
    [ax] = fig.axes
    assert ax.get_title(loc="left") == "channel:baseline"
    points = _points(ax, "Mean")
    np.testing.assert_array_equal(points[:, 0], future["baseline"])
    query = ModelSource(model).window("query")
    np.testing.assert_allclose(points[:, 1], query.contribution("channel:baseline").mean(axis=0))


def test_plotly_contribution_vs_media_draws_the_matplotlib_numbers(predicted_model, chart_frames):
    pytest.importorskip("plotly")
    history, _ = chart_frames
    truth = _history_truth(history, search=np.linspace(5.0, 40.0, 24))
    drawn = predicted_model.plot.contribution_vs_media(truth=truth).axes
    fig = predicted_model.plot.contribution_vs_media(truth=truth, backend="plotly")
    means = [trace for trace in fig.data if trace.name == "Mean and 90% interval"]
    [crosses] = [trace for trace in fig.data if trace.name == "Truth"]
    for ax, trace in zip(drawn, means, strict=True):
        points = _points(ax, "Mean")
        np.testing.assert_array_equal(trace.x, points[:, 0])
        np.testing.assert_array_equal(trace.y, points[:, 1])
        _, low, high = _bars(ax)
        np.testing.assert_allclose(trace.y + trace.error_y.array, high)
        np.testing.assert_allclose(trace.y - trace.error_y.arrayminus, low)
        assert trace.text[0] == "2024-01-01" and len(trace.text) == 24
    np.testing.assert_array_equal(crosses.x, history["search"])
    np.testing.assert_array_equal(crosses.y, _points(drawn[0], "Truth")[:, 1])
    assert crosses.xaxis == means[0].xaxis
    assert fig.layout.yaxis2.matches == "y"  # one contribution axis
    assert "carryover" in fig.layout.title.text
