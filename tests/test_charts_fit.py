"""``model.plot.fit``: the expected outcome over the observed outcome, scored per window."""

import numpy as np
import pytest
import torch
from conftest import CHART_ROLES

from pymc_mfm.plotting.charts import _backend
from pymc_mfm.plotting.charts._data import ModelSource, Window, summarize
from pymc_mfm.plotting.charts.fit import fit, prepare

pytest.importorskip("matplotlib")

_LABELS = {"context": "History (context), 24 weeks", "query": "Forecast (query), 6 weeks"}


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


class _Source:
    """Hand-made windows in place of a model; ``prepare`` reads only ``windows``."""

    def __init__(self, *windows):
        self._windows = windows

    def windows(self, window):
        return tuple(part for part in self._windows if window in (None, part.name))


def _window(name, start, expected, outcome):
    """A one-component window whose expected-outcome draws are ``expected``."""
    expected = np.asarray(expected, dtype=float)
    weeks = expected.shape[1]
    return Window(
        name=name,
        dates=np.datetime64(start, "ns") + np.arange(weeks) * np.timedelta64(7, "D"),
        components=("baseline",),
        families=("baseline",),
        draws=expected[:, :, None],
        outcome=None if outcome is None else np.asarray(outcome, dtype=float),
        channels=(),
        spend=np.zeros((weeks, 0)),
        target="sales",
        units="outcome",
    )


def _band(collection, positions):
    """Lower and upper edge of a filled band at each of ``positions``."""
    vertices = collection.get_paths()[0].vertices
    edges = [vertices[vertices[:, 0] == position, 1] for position in positions]
    return np.array([edge.min() for edge in edges]), np.array([edge.max() for edge in edges])


def _cards(fig):
    """The text of each metric card below the chart, card by card."""
    return [[text.get_text() for text in card.texts] for card in fig.axes[1].child_axes]


def _traces(fig, **properties):
    """The plotly traces with ``properties``, in drawing order."""
    return list(fig.select_traces(selector=properties))


def _by_hand(y, draws, interval):
    """MAPE and R² of the mean, each with its sd over draws, and the weeks inside."""
    seen = np.isfinite(y)
    y, draws = y[seen], draws[:, seen]
    low, high = np.quantile(draws, [(1 - interval) / 2, (1 + interval) / 2], axis=0)

    def mape(fit):
        return np.mean([abs(value - fitted) / abs(value) for value, fitted in zip(y, fit)])

    def r2(fit):
        return 1 - sum((y - fit) ** 2) / sum((y - y.mean()) ** 2)

    mean = draws.mean(axis=0)
    inside = sum(bottom <= value <= top for value, bottom, top in zip(y, low, high))
    return (
        (mape(mean), np.std([mape(draw) for draw in draws])),
        (r2(mean), np.std([r2(draw) for draw in draws])),
        int(inside),
    )


def test_fit_draws_each_window_apart_over_the_observed_sales(predicted_model, chart_frames):
    history, future = chart_frames
    fig = predicted_model.plot.fit(interval=0.8)
    ax = fig.axes[0]
    *means, actual = ax.lines
    positions = actual.get_xdata()
    np.testing.assert_array_equal(
        actual.get_ydata(), np.concatenate([history["sales"], future["sales"]])
    )
    source = ModelSource(predicted_model)
    weeks = {"context": slice(0, 24), "query": slice(24, 30)}
    for line, band, name in zip(means, ax.collections, weeks, strict=True):
        mean, low, high = summarize(source.window(name).expected, 0.8)
        # Context and query are separate marginals: each is its own segment.
        np.testing.assert_array_equal(line.get_xdata(), positions[weeks[name]])
        np.testing.assert_allclose(line.get_ydata(), mean)
        np.testing.assert_allclose(_band(band, positions[weeks[name]]), (low, high))
    [span] = ax.patches
    left, right = span.get_x(), span.get_x() + span.get_width()
    assert positions[23] < left < positions[24] and right > positions[-1]
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend == ["80% interval", "Expected", "Observed sales", "Forecast weeks"]
    assert "80% pointwise intervals over 40 draws" in fig.get_suptitle()


def test_fit_scores_each_observed_window_on_its_card(predicted_model):
    source = ModelSource(predicted_model)
    data = prepare(source, interval=0.8)
    cards = _cards(predicted_model.plot.fit(interval=0.8))
    for score, card, name in zip(data.scores, cards, _LABELS, strict=True):
        window = source.window(name)
        (mape, mape_sd), (r2, r2_sd), inside = _by_hand(window.outcome, window.expected, 0.8)
        weeks = len(window.dates)
        assert (score.window, score.weeks, score.inside) == (name, weeks, inside)
        assert (score.mape, score.mape_sd) == pytest.approx((mape, mape_sd))
        assert (score.r2, score.r2_sd) == pytest.approx((r2, r2_sd))
        assert card == [
            _LABELS[name],
            "MAPE",
            f"{mape:.1%} ± {mape_sd:.1%}",
            "R²",
            f"{r2:.2f} ± {r2_sd:.2f}",
            "Coverage of the 80% interval",
            f"{inside / weeks:.0%} ({inside} of {weeks})",
        ]


def test_scores_leave_out_unobserved_weeks_and_undefined_metrics():
    context = _window(
        "context",
        "2024-01-01",
        [[10, 20, 30, 40], [12, 18, 34, 36], [8, 22, 26, 44], [10, 20, 30, 40]],
        [10, 0, np.nan, 41],
    )
    query = _window("query", "2024-01-29", [[4, 6], [5, 5], [6, 4], [5, 5]], [5, 5])
    history, forecast = prepare(_Source(context, query), interval=0.5).scores
    # The NaN week is unobserved; the zero week counts for R² and coverage, not MAPE.
    # Its 50% intervals are 9.5-10.5, 19.5-20.5 and 39-41: bounds count as inside.
    assert (history.weeks, history.inside) == (3, 2)
    mape = [(0 / 10 + 1 / 41) / 2, (2 / 10 + 5 / 41) / 2, (2 / 10 + 3 / 41) / 2]
    r2 = [1 - 401 / 914, 1 - 353 / 914, 1 - 497 / 914]
    assert (history.mape, history.mape_sd) == pytest.approx((mape[0], np.std([*mape, mape[0]])))
    assert (history.r2, history.r2_sd) == pytest.approx((r2[0], np.std([*r2, r2[0]])))
    # A constant outcome has no R², and its card leaves R² out.
    assert (forecast.weeks, forecast.inside, forecast.r2, forecast.r2_sd) == (2, 2, None, None)
    assert (forecast.mape, forecast.mape_sd) == pytest.approx((0, 0.1))
    assert _cards(fit(_Source(context, query), interval=0.5))[1] == [
        "Forecast (query), 2 weeks",
        "MAPE",
        "0.0% ± 10.0%",
        "Coverage of the 50% interval",
        "100% (2 of 2)",
    ]
    # One observed week is too few to score.
    assert prepare(_Source(_window("query", "2024-01-29", [[4, 6]], [5, np.nan]))).scores == ()


def test_forecast_weeks_without_sales_are_neither_drawn_nor_scored(model, chart_frames):
    history, future = chart_frames
    model.add_context(history, **CHART_ROLES)
    model.inference(future.drop(columns="sales"), n_samples=5)
    fig = model.plot.fit()
    np.testing.assert_array_equal(
        fig.axes[0].lines[-1].get_ydata(), np.concatenate([history["sales"], np.full(6, np.nan)])
    )
    assert [card[0] for card in _cards(fig)] == [_LABELS["context"]]
    query = model.plot.fit(window="query")
    # The expected line alone: no observed line and no row of cards.
    assert len(query.axes) == 1 and len(query.axes[0].lines) == 1


def test_query_fit_never_samples_the_context(predicted_model, chart_frames):
    _, future = chart_frames
    state = torch.get_rng_state()
    fig = predicted_model.plot.fit(window="query")
    assert torch.equal(torch.get_rng_state(), state)
    ax = fig.axes[0]
    line, actual = ax.lines
    mean, low, high = summarize(ModelSource(predicted_model).window("query").expected, 0.9)
    np.testing.assert_allclose(line.get_ydata(), mean)
    np.testing.assert_allclose(_band(ax.collections[0], actual.get_xdata()), (low, high))
    np.testing.assert_array_equal(actual.get_ydata(), future["sales"])
    assert not ax.patches  # one window: no forecast weeks to set apart
    assert [card[0] for card in _cards(fig)] == [_LABELS["query"]]
    predicted_model.plot.fit(window="context")
    assert not torch.equal(torch.get_rng_state(), state)  # the context draws are sampled now


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"window": "forecast"}, "window must be one of"),
        ({"interval": 1.0}, "interval must be a fraction"),
        ({"interval": 0}, "interval must be a fraction"),
    ],
)
def test_fit_rejects_unknown_windows_and_intervals_before_sampling(
    predicted_model, arguments, message
):
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=message):
        predicted_model.plot.fit(**arguments)
    assert torch.equal(torch.get_rng_state(), state)


def test_a_one_week_window_is_a_point_with_an_interval_bar():
    context = _window("context", "2024-01-01", [[4, 6], [5, 5], [6, 4], [5, 5]], [5, 5])
    query = _window("query", "2024-01-15", [[7], [9], [8], [8]], None)
    ax = fit(_Source(context, query), interval=0.5).axes[0]
    point = ax.lines[1]
    [[(_, low), (_, high)]] = ax.collections[1].get_segments()
    assert point.get_marker() == "o" and list(point.get_ydata()) == [8]
    assert (low, high) == (7.75, 8.25)


def test_plotly_fit_draws_the_numbers_matplotlib_draws(predicted_model, chart_frames):
    pytest.importorskip("plotly")
    history, future = chart_frames
    figure = predicted_model.plot.fit(interval=0.8)
    fig = predicted_model.plot.fit(interval=0.8, backend="plotly")
    *lines, actual = figure.axes[0].lines
    positions = actual.get_xdata()
    dates = np.concatenate([history["date"], future["date"]])
    means, bands = _traces(fig, legendgroup="expected"), _traces(fig, legendgroup="interval")
    segments = zip(lines, figure.axes[0].collections, (slice(0, 24), slice(24, 30)), strict=True)
    for index, (line, band, weeks) in enumerate(segments):
        trace, (upper, lower) = means[index], bands[2 * index : 2 * index + 2]
        np.testing.assert_array_equal(trace.x, dates[weeks])
        np.testing.assert_array_equal(trace.y, line.get_ydata())
        np.testing.assert_allclose((lower.y, upper.y), _band(band, positions[weeks]))
        # Hovering the mean shows its date, value and both interval bounds.
        np.testing.assert_array_equal(trace.customdata, np.column_stack((lower.y, upper.y)))
        assert trace.hovertemplate.startswith("%{x|%Y-%m-%d}<br>Expected %{y")
        assert "%{customdata[0]" in trace.hovertemplate
        assert "%{customdata[1]" in trace.hovertemplate
    [observed] = _traces(fig, legendgroup="actual")
    np.testing.assert_array_equal(observed.y, actual.get_ydata())
    [table] = _traces(fig, type="table")
    cards = _cards(figure)
    assert list(table.cells.values[0]) == [card[0] for card in cards]
    for column, name in enumerate(table.header.values[1:], start=1):
        assert list(table.cells.values[column]) == [card[card.index(name) + 1] for card in cards]
    [span] = fig.layout.shapes
    assert (span.name, span.showlegend) == ("Forecast weeks", True)
    assert history["date"].iloc[-1] < span.x0 < future["date"][0] < span.x1
    assert span.x1 > future["date"].iloc[-1]


def test_plotly_draws_a_one_week_window_and_leaves_undefined_scores_blank():
    pytest.importorskip("plotly")
    context = _window("context", "2024-01-01", [[4, 6], [5, 5], [6, 4], [5, 5]], [5, 5])
    query = _window("query", "2024-01-15", [[7], [9], [8], [8]], [8.5])
    fig = fit(_Source(context, query), interval=0.5, backend="plotly")
    *_, bar = _traces(fig, legendgroup="interval")
    *_, point = _traces(fig, legendgroup="expected")
    np.testing.assert_array_equal(bar.x, np.repeat(query.dates, 2))
    assert (list(bar.y), point.mode, list(point.y)) == ([7.75, 8.25], "lines+markers", [8])
    # One observed forecast week is too few to score; a constant outcome has no R².
    [table] = _traces(fig, type="table")
    assert [list(column) for column in table.cells.values] == [
        ["History (context), 2 weeks"],
        ["0.0% ± 10.0%"],
        [""],
        ["100% (2 of 2)"],
    ]
