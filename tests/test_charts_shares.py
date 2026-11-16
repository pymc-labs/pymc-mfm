"""``model.plot.shares``: each channel's and control's share of the expected outcome."""

from statistics import NormalDist

import numpy as np
import pytest
import torch

from pymc_mfm.plotting.charts import _backend
from pymc_mfm.plotting.charts._data import ModelSource, Window
from pymc_mfm.plotting.charts._style import CAVEAT, COLORS, caption, series_colors
from pymc_mfm.plotting.charts.shares import GRID_POINTS, prepare, prepare_window

pytest.importorskip("matplotlib")

COMPONENTS = ("baseline", "search", "tv", "temperature", "price")
FAMILIES = ("baseline", "channel", "channel", "control", "control")
#: 3 draws x 2 weeks of COMPONENTS; a week's expected sales are its row sum. Draw 1's
#: second week expects -100, so its two weeks total 0.
DRAWS = np.array(
    [
        [[50, 10, 30, 2, 8], [120, 20, 40, 4, 16]],  # expects 100, 200
        [[40, 20, 20, -4, 24], [-100, 10, 10, -20, 0]],  # expects 100, -100
        [[30, 10, 50, 2, 8], [250, 50, 150, 10, 40]],  # expects 100, 500
    ],
    dtype=float,
)
#: Per-draw shares of DRAWS in bar order: tv, search, price, temperature.
MEAN_SHARES = [  # contribution / expected, averaged over the weeks that expect more than 0
    [
        (30 / 100 + 40 / 200) / 2,
        (10 / 100 + 20 / 200) / 2,
        (8 / 100 + 16 / 200) / 2,
        (2 / 100 + 4 / 200) / 2,
    ],
    [20 / 100, 20 / 100, 24 / 100, -4 / 100],
    [
        (50 / 100 + 150 / 500) / 2,
        (10 / 100 + 50 / 500) / 2,
        (8 / 100 + 40 / 500) / 2,
        (2 / 100 + 10 / 500) / 2,
    ],
]
TOTAL_SHARES = [  # the weeks' contribution / their expected sales; draw 1 totals 0
    [70 / 300, 30 / 300, 24 / 300, 6 / 300],
    [200 / 600, 60 / 600, 48 / 600, 12 / 600],
]


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


def _window(draws, components=COMPONENTS, families=FAMILIES):
    """A hand-made history window of ``draws`` ``[sample, week, component]`` of sales."""
    weeks = draws.shape[1]
    channels = tuple(
        name for name, family in zip(components, families, strict=True) if family == "channel"
    )
    return Window(
        name="context",
        dates=np.datetime64("2024-01-01", "ns") + np.arange(weeks) * np.timedelta64(7, "D"),
        components=components,
        families=families,
        draws=draws,
        outcome=None,
        channels=channels,
        spend=np.zeros((weeks, len(channels))),
        target="sales",
        units="outcome",
    )


def _drivers(window):
    return [index for index, family in enumerate(window.families) if family in FAMILIES[1:]]


def _mean_shares(window):
    """Per-draw mean shares ``[kept draw, driver]``, by the definition, a draw at a time."""
    shares = []
    for draw in window.draws:
        expected = draw.sum(axis=1)
        kept = expected > 0
        if kept.any():
            shares.append([np.mean(draw[kept, i] / expected[kept]) for i in _drivers(window)])
    return np.array(shares)


def _total_shares(window):
    """Per-draw total shares ``[kept draw, driver]``, by the definition, a draw at a time."""
    return np.array(
        [
            [draw[:, i].sum() / draw.sum() for i in _drivers(window)]
            for draw in window.draws
            if draw.sum() > 0
        ]
    )


def _bars(ax, window, per_draw, interval):
    """Labels, widths and interval ends drawn on ``ax``, and the hand-computed ones."""
    labels = [label.get_text() for label in ax.get_yticklabels()]
    names = [window.components[index] for index in _drivers(window)]
    order = [names.index(label) for label in labels]
    bounds = np.quantile(per_draw, [(1 - interval) / 2, (1 + interval) / 2], axis=0)
    [errors] = ax.collections
    drawn = [bar.get_width() for bar in ax.patches], [line[:, 0] for line in errors.get_segments()]
    return labels, drawn, (per_draw.mean(axis=0)[order], bounds[:, order].T)


@pytest.mark.parametrize(
    ("share", "per_draw", "left_out", "note", "xlabel"),
    [
        pytest.param(
            "mean",
            MEAN_SHARES,
            (0, 1),
            "1 of 6 draw-weeks left out: expected sales ≤ 0",
            "Average of each week's share of expected sales, over 2 history weeks",
            id="mean",
        ),
        pytest.param(
            "total",
            TOTAL_SHARES,
            (1, 0),
            "1 of 3 draws left out: total expected sales ≤ 0",
            "Share of expected sales over the 2 history weeks",
            id="total",
        ),
    ],
)
def test_shares_of_a_hand_made_window(share, per_draw, left_out, note, xlabel):
    data = prepare_window(_window(DRAWS), share=share, view="bars", interval=0.8)
    # Channels, then controls, each largest mean first.
    assert data.names == ("tv", "search", "price", "temperature")
    assert data.colors == (COLORS["channel"],) * 2 + (COLORS["control"],) * 2
    np.testing.assert_allclose(data.shares, per_draw)
    np.testing.assert_allclose(data.mean, np.mean(per_draw, axis=0))
    np.testing.assert_allclose((data.low, data.high), np.quantile(per_draw, [0.1, 0.9], axis=0))
    assert (data.left_out_draws, data.left_out_draw_weeks) == left_out
    assert data.caption == (caption(0.8, len(per_draw), pointwise=False), CAVEAT, note)
    assert data.xlabel == xlabel


def test_a_draw_without_a_week_expecting_sales_is_left_out():
    draws = DRAWS.copy()
    draws[1, 0, 0] = -60  # draw 1's first week now expects 0
    data = prepare_window(_window(draws), share="mean", view="bars", interval=0.9)
    np.testing.assert_allclose(data.shares, [MEAN_SHARES[0], MEAN_SHARES[2]])
    assert (data.left_out_draws, data.left_out_draw_weeks) == (1, 2)
    assert data.caption[2:] == (
        "2 of 6 draw-weeks left out: expected sales ≤ 0",
        "1 of 3 draws left out: expected sales ≤ 0 in every week",
    )
    for share in ("mean", "total"):
        with pytest.raises(ValueError, match="no share is defined"):
            prepare_window(_window(-np.abs(DRAWS)), share=share, view="bars", interval=0.9)


def test_dist_densities_peak_at_one_where_normal_shares_center():
    # Normal quantiles as draws: their kernel density peaks at their mean.
    probabilities = (np.arange(2000) + 0.5) / 2000
    search = np.array([NormalDist(0.3, 0.05).inv_cdf(p) for p in probabilities])
    temperature = np.array([NormalDist(-0.05, 0.02).inv_cdf(p) for p in probabilities])
    price = np.zeros(2000)  # the same share in every draw
    draws = np.stack([1 - search - temperature, search, temperature, price], axis=-1)[:, None]
    window = _window(
        draws,
        components=("baseline", "search", "temperature", "price"),
        families=("baseline", "channel", "control", "control"),
    )
    data = prepare_window(window, share="total", view="dist", interval=0.9)
    assert data.names == ("search", "temperature", "price")  # component order
    assert data.colors == tuple(series_colors(data.names).values())
    np.testing.assert_allclose(data.shares, np.stack([search, temperature, price], axis=-1))
    np.testing.assert_array_equal(data.density.max(axis=1), 1.0)
    step = data.grid[1] - data.grid[0]
    modes = data.grid[data.density.argmax(axis=1)]
    np.testing.assert_allclose(modes, data.shares.mean(axis=0), atol=step)
    # One grid, three of the widest (Scott's rule) bandwidths past the extreme shares.
    reach = 3 * search.std(ddof=1) * 2000**-0.2
    assert data.grid.shape == (GRID_POINTS,)
    np.testing.assert_allclose(data.grid[[0, -1]], [temperature[0] - reach, search[-1] + reach])
    assert "Kernel densities, each scaled to a peak of 1" in data.caption


def test_default_bars_are_mean_shares_of_the_history_weeks(predicted_model):
    context = ModelSource(predicted_model).window("context")
    ax = predicted_model.plot.shares().axes[0]
    labels, drawn, expected = _bars(ax, context, _mean_shares(context), 0.9)
    assert sorted(labels[:2]) == ["search", "tv"] and labels[2] == "temperature"
    assert expected[0][0] >= expected[0][1]  # each family largest first
    np.testing.assert_allclose(drawn[0], expected[0])
    np.testing.assert_allclose(drawn[1], expected[1])
    assert (
        ax.get_xlabel() == "Average of each week's share of expected sales, over 24 history weeks"
    )


def test_query_total_shares_never_sample_the_context(predicted_model):
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.subplots()
    state = torch.get_rng_state()
    assert predicted_model.plot.shares(share="total", window="query", ax=ax) is fig
    assert torch.equal(torch.get_rng_state(), state)
    query = ModelSource(predicted_model).window("query")
    labels, drawn, expected = _bars(ax, query, _total_shares(query), 0.9)
    np.testing.assert_allclose(drawn[0], expected[0])
    np.testing.assert_allclose(drawn[1], expected[1])
    assert ax.get_xlabel() == "Share of expected sales over the 6 forecast weeks"
    assert ax.get_title().startswith("Share of expected sales by channel and control\n")


def test_plotly_bars_draw_the_matplotlib_numbers(predicted_model):
    pytest.importorskip("plotly")
    data = prepare(ModelSource(predicted_model), window="query", interval=0.8)
    ax = predicted_model.plot.shares(window="query", interval=0.8).axes[0]
    fig = predicted_model.plot.shares(window="query", interval=0.8, backend="plotly")
    assert [trace.name for trace in fig.data] == ["channels", "controls"]
    assert all(trace.orientation == "h" for trace in fig.data)
    assert list(fig.layout.yaxis.ticktext) == list(data.names)
    assert [label.get_text() for label in ax.get_yticklabels()] == list(data.names)
    np.testing.assert_allclose(np.concatenate([trace.x for trace in fig.data]), data.mean)
    np.testing.assert_allclose([bar.get_width() for bar in ax.patches], data.mean)
    errors = [
        np.concatenate([trace.error_x[end] for trace in fig.data])
        for end in ("arrayminus", "array")
    ]
    np.testing.assert_allclose(data.mean - errors[0], data.low)
    np.testing.assert_allclose(data.mean + errors[1], data.high)
    [segments] = ax.collections
    np.testing.assert_allclose(
        [line[:, 0] for line in segments.get_segments()], np.column_stack((data.low, data.high))
    )
    # A dashed line between the two channels and the control, in both backends.
    [separator] = [shape for shape in fig.layout.shapes if shape.line.dash == "dash"]
    assert separator.y0 == separator.y1 == 1.5
    [dashed] = [line for line in ax.lines if line.get_linestyle() == "--"]
    np.testing.assert_array_equal(dashed.get_ydata(), [1.5, 1.5])
    assert "80% interval" in fig.data[0].hovertemplate


def test_dist_draws_the_same_densities_in_both_backends(predicted_model):
    pytest.importorskip("plotly")
    data = prepare(ModelSource(predicted_model), window="query", view="dist")
    ax = predicted_model.plot.shares(window="query", view="dist").axes[0]
    fig = predicted_model.plot.shares(window="query", view="dist", backend="plotly")
    assert data.names == ("search", "tv", "temperature")
    assert [line.get_label() for line in ax.lines] == [trace.name for trace in fig.data]
    assert [trace.name for trace in fig.data] == list(data.names)
    assert len({trace.legendgroup for trace in fig.data}) == len(data.names)
    np.testing.assert_array_equal(data.density.max(axis=1), 1.0)
    for line, trace, density, color in zip(
        ax.lines, fig.data, data.density, data.colors, strict=True
    ):
        np.testing.assert_array_equal(line.get_xdata(), data.grid)
        np.testing.assert_array_equal(trace.x, data.grid)
        np.testing.assert_array_equal(line.get_ydata(), density)
        np.testing.assert_array_equal(trace.y, density)
        assert line.get_color() == trace.line.color == color
        assert trace.fill == "tozeroy"
    assert "Kernel densities, each scaled to a peak of 1" in fig.layout.title.text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"share": "median"}, "share must be one of"),
        ({"view": "violin"}, "view must be one of"),
        ({"window": "future"}, "window must be one of"),
        ({"interval": 1.0}, "interval must be a fraction"),
    ],
)
def test_invalid_arguments_raise_before_any_context_draw(predicted_model, arguments, message):
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=message):
        predicted_model.plot.shares(**arguments)
    assert torch.equal(torch.get_rng_state(), state)


def test_axes_are_for_matplotlib_only(predicted_model):
    pytest.importorskip("plotly")
    from matplotlib.figure import Figure

    with pytest.raises(ValueError, match="ax="):
        predicted_model.plot.shares(window="query", backend="plotly", ax=Figure().subplots())
