"""``model.plot.contributions_over_time``: stacked weekly means and per-component panels."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES

from pymc_mktfm.plotting.charts import _backend, contributions
from pymc_mktfm.plotting.charts._data import IdataReader, Window, summarize
from pymc_mktfm.plotting.charts._style import COLORS
from pymc_mktfm.plotting.charts.contributions import contributions_over_time

pytest.importorskip("matplotlib")

#: The fixture's components from the bottom of the stack up: channels on top.
ORDER = ("baseline", "latent", "indirect:cc", "indirect:zc", "indirect:dc", "temperature")
ORDER += ("search", "tv")
WEEK = np.timedelta64(7 * 24, "h")  # in hours, so half a week is exact


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


class _Source:
    """Serves hand-made windows where a chart reads ``IdataReader``."""

    def __init__(self, *windows: Window):
        self._windows = windows

    def windows(self, window):
        return self._windows

    def window(self, name):
        return self._windows[0]


def _window(means: dict[str, list[float]], families: tuple[str, ...]) -> Window:
    """Two draws per week whose mean is ``means``: one 1 below it, one 1 above."""
    mean = np.array(list(means.values()), dtype=float).T
    weeks = len(mean)
    return Window(
        name="context",
        dates=np.datetime64("2024-01-01", "ns") + WEEK * np.arange(weeks),
        components=tuple(means),
        families=families,
        draws=np.stack((mean - 1, mean + 1)),
        outcome=None,
        channels=("search",),
        media=np.zeros((weeks, 1)),
        units="outcome",
    )


def _stack(fig):
    """``(values, edges, baseline)`` of each drawn layer, window by window."""
    from matplotlib.patches import StepPatch

    return [patch.get_data() for patch in fig.axes[0].patches if isinstance(patch, StepPatch)]


def _expected_stack(window: Window, names) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(lower, upper)`` of each layer: positive means pile up from 0, negative ones down."""
    up = down = np.zeros(len(window.dates))
    layers = []
    for name in names:
        mean = window.contribution(name).mean(axis=0)
        layers.append((np.where(mean >= 0, up, down + mean), np.where(mean >= 0, up + mean, down)))
        up, down = up + np.maximum(mean, 0), down + np.minimum(mean, 0)
    return layers


def _edges(dates: np.ndarray) -> np.ndarray:
    """Weekly ``dates``' boundaries: half a week before each, and after the last."""
    return np.append(dates - WEEK / 2, dates[-1] + WEEK / 2)


def _band(collection, x):
    """``(low, high)`` of a ``fill_between`` band at each of ``x``."""
    vertices = collection.get_paths()[0].vertices
    at = [vertices[np.isclose(vertices[:, 0], value), 1] for value in x]
    return np.array([ys.min() for ys in at]), np.array([ys.max() for ys in at])


def test_stack_piles_positive_means_up_and_negative_means_down(predicted_model, chart_frames):
    from matplotlib.dates import date2num

    history, future = chart_frames
    fig = predicted_model.plot.contributions_over_time()
    ax = fig.axes[0]
    assert ax.get_legend_handles_labels()[1] == [*ORDER, "Observed outcome", "Forecast weeks"]
    drawn = _stack(fig)
    windows = IdataReader(predicted_model.idata).windows(None)
    assert len(drawn) == len(windows) * len(ORDER)
    for number, window in enumerate(windows):
        layers = drawn[number * len(ORDER) : (number + 1) * len(ORDER)]
        expected = _expected_stack(window, ORDER)
        for name, (values, edges, baseline), (lower, upper) in zip(
            ORDER, layers, expected, strict=True
        ):
            np.testing.assert_allclose(baseline, lower, rtol=1e-12, atol=1e-9, err_msg=name)
            np.testing.assert_allclose(values, upper, rtol=1e-12, atol=1e-9, err_msg=name)
            np.testing.assert_allclose(edges, date2num(_edges(window.dates)))
        # The stack's top is the sum of the positive weekly means, its bottom the negatives'.
        means = window.draws.mean(axis=0)
        np.testing.assert_allclose(
            np.max([values for values, _, _ in layers], axis=0), np.clip(means, 0, None).sum(1)
        )
        np.testing.assert_allclose(
            np.min([baseline for _, _, baseline in layers], axis=0), np.clip(means, None, 0).sum(1)
        )
    [actual] = ax.lines[1:]  # after the zero line
    np.testing.assert_array_equal(
        actual.get_ydata(), np.concatenate([history["sales"], future["sales"]])
    )
    # The forecast weeks are shaded from where the context's last week ends.
    [span] = [patch for patch in ax.patches if patch.get_label() == "Forecast weeks"]
    np.testing.assert_allclose(span.get_x(), date2num(future["date"].iloc[0] - WEEK / 2))


def test_a_negative_component_sits_below_zero_and_channels_on_top():
    window = _window(
        {
            "baseline": [10, 10, 10],
            "search": [6, 7, 8],
            "temperature": [-4, -1, 5],
            "latent": [2, -3, 0],
        },
        ("baseline", "channel", "control", "latent"),
    )
    fig = contributions_over_time(_Source(window))
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert labels == ["baseline", "latent", "temperature", "search"]
    expected = {
        "baseline": ([0, 0, 0], [10, 10, 10]),
        "latent": ([10, -3, 10], [12, 0, 10]),
        "temperature": ([-4, -4, 10], [0, -3, 15]),
        "search": ([12, 10, 15], [18, 17, 23]),
    }
    for name, (values, _, baseline) in zip(labels, _stack(fig), strict=True):
        np.testing.assert_array_equal(baseline, expected[name][0], err_msg=name)
        np.testing.assert_array_equal(values, expected[name][1], err_msg=name)


def test_components_pick_layers_keep_their_colors_and_never_resample(predicted_model, chart_frames):
    from matplotlib.colors import to_hex

    _, future = chart_frames
    plot = predicted_model.plot
    state = torch.get_rng_state()
    fig = plot.contributions_over_time(window="query", components=["tv", "search", "baseline"])
    assert torch.equal(torch.get_rng_state(), state)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ["baseline", "search", "tv", "Observed outcome"]
    query = IdataReader(predicted_model.idata).window("query")
    names = ("baseline", "search", "tv")
    for (values, _, baseline), (lower, upper) in zip(
        _stack(fig), _expected_stack(query, names), strict=True
    ):
        np.testing.assert_allclose(baseline, lower, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(values, upper, rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(ax.lines[-1].get_ydata(), future["sales"])
    assert not [patch for patch in ax.patches if patch.get_label() == "Forecast weeks"]
    # A component keeps the color it has among all of them; baseline is always grey.
    every = plot.contributions_over_time(window="query").axes[0].get_legend_handles_labels()
    colors = {
        label: to_hex(handle.get_facecolor())
        for handle, label in zip(*every, strict=True)
        if label in ORDER
    }
    assert colors["baseline"] == COLORS["baseline"]
    for handle, label in zip(handles[:3], names, strict=True):
        assert to_hex(handle.get_facecolor()) == colors[label]


def test_panels_draw_each_components_mean_and_interval_with_truth(predicted_model, chart_frames):
    history, future = chart_frames
    dates = pd.DatetimeIndex(np.concatenate([history["date"], future["date"]]))
    # Rows reversed, a date that is not drawn and two weeks missing: aligned by date.
    truth = pd.DataFrame(
        {"tv": np.arange(31.0), "baseline": -np.arange(31.0)},
        index=dates.append(pd.DatetimeIndex(["2030-01-07"])),
    )
    truth = truth.drop(index=dates[[3, 26]]).iloc[::-1]
    fig = predicted_model.plot.contributions_over_time(stacked=False, interval=0.8, truth=truth)
    panels = fig.axes[: len(ORDER)]
    assert [ax.get_title(loc="left") for ax in panels] == list(ORDER)
    assert not any(ax.get_visible() for ax in fig.axes[len(ORDER) :])
    windows = IdataReader(predicted_model.idata).windows(None)
    for ax, name in zip(panels, ORDER, strict=True):
        for number, window in enumerate(windows):
            mean, low, high = summarize(window.contribution(name), 0.8)
            line = ax.lines[number]
            np.testing.assert_allclose(line.get_ydata(), mean, rtol=1e-12, atol=1e-9)
            drawn_low, drawn_high = _band(ax.collections[number], line.get_xdata())
            np.testing.assert_allclose(drawn_low, low, rtol=1e-12, atol=1e-9)
            np.testing.assert_allclose(drawn_high, high, rtol=1e-12, atol=1e-9)
        dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
        if name not in truth:
            assert not dashed
            continue
        [dashed] = dashed
        expected = truth[name].reindex(dates).to_numpy()
        assert np.isnan(expected[[3, 26]]).all()
        np.testing.assert_array_equal(dashed.get_ydata(), expected)


def test_panels_follow_the_listed_components_while_the_stack_keeps_its_order(predicted_model):
    listed = ["tv", "baseline", "temperature"]
    panels = predicted_model.plot.contributions_over_time(
        window="query", stacked=False, components=listed
    )
    assert [ax.get_title(loc="left") for ax in panels.axes if ax.get_visible()] == listed
    stack = predicted_model.plot.contributions_over_time(window="query", components=listed)
    labels = stack.axes[0].get_legend_handles_labels()[1]
    assert [label for label in labels if label in listed] == ["baseline", "temperature", "tv"]


def test_a_one_week_window_still_draws_whole_weeks(model, chart_frames):
    from matplotlib.dates import date2num

    history, future = chart_frames
    model.inference(history, future.iloc[:1], n_samples=5, **CHART_ROLES)
    query = IdataReader(model.idata).window("query")
    fig = model.plot.contributions_over_time(window="query")
    for values, edges, _ in _stack(fig):
        assert values.shape == (1,)
        np.testing.assert_allclose(edges, date2num(_edges(query.dates)))
    panels = model.plot.contributions_over_time(window="query", stacked=False)
    mean, low, high = summarize(query.contribution("baseline"), 0.9)
    [bar] = panels.axes[0].collections
    np.testing.assert_allclose(bar.get_segments()[0][:, 1], [low[0], high[0]])
    np.testing.assert_allclose(panels.axes[0].lines[0].get_ydata(), mean)


def _in_seconds(tree):
    """``tree`` with every date coordinate held at one-second resolution."""
    from test_plots_idata import _rebuilt

    def seconds(data):
        return data.assign_coords(date=data["date"].to_numpy().astype("datetime64[s]"))

    paths = [node.path for node in tree.subtree if "date" in node.to_dataset(inherit=False).coords]
    return _rebuilt(tree, {path: seconds for path in paths})


@pytest.mark.parametrize("window", [None, "query"])
def test_a_stack_of_seconds_resolution_dates_still_draws_whole_weeks(model, chart_frames, window):
    history, future = chart_frames
    model.inference(history, future.iloc[:1], n_samples=5, **CHART_ROLES)
    tree = _in_seconds(model.idata)
    source = IdataReader(tree)
    assert source.window("query").dates.dtype == np.dtype("datetime64[s]")
    drawn = contributions.prepare(source, window=window)
    week = np.timedelta64(7, "D")
    for segment in drawn.segments:
        gaps = np.diff(segment.edges)
        assert (gaps == week).all()
    expected = contributions.prepare(IdataReader(model.idata), window=window)
    for got, want in zip(drawn.segments, expected.segments, strict=True):
        np.testing.assert_array_equal(got.edges, want.edges)


def _with_unread_context_draws(tree):
    """``tree`` whose context posterior variables are lazy and raise when read."""
    import xarray as xr
    from test_plots_idata import _rebuilt
    from xarray.core import indexing

    class Lazy(xr.backends.BackendArray):
        """A lazy array that raises when anything reads its values."""

        def __init__(self, like):
            self.shape, self.dtype = like.shape, like.dtype

        def __getitem__(self, key):
            raise AssertionError("a context posterior draw array was read")

    def stand_in(data):
        draws = {}
        for name, variable in data.data_vars.items():
            lazy = indexing.LazilyIndexedArray(Lazy(variable))
            draws[name] = xr.Variable(variable.dims, lazy)
        return data.assign(draws)

    return _rebuilt(tree, {"/context/posterior": stand_in})


def test_a_query_only_stack_reads_no_context_draw_array(predicted_model):
    tree = _with_unread_context_draws(predicted_model.idata)
    with pytest.raises(AssertionError, match="context posterior draw"):
        IdataReader(tree).window("context")
    drawn = contributions.prepare(IdataReader(tree), window="query")
    expected = contributions.prepare(IdataReader(predicted_model.idata), window="query")
    assert len(drawn.segments) == 1
    np.testing.assert_array_equal(drawn.segments[0].mean, expected.segments[0].mean)
    np.testing.assert_array_equal(drawn.segments[0].edges, expected.segments[0].edges)


def test_the_context_dates_come_from_the_dates_coordinate_alone(predicted_model):
    source = IdataReader(predicted_model.idata)
    np.testing.assert_array_equal(source.dates("context"), source.window("context").dates)
    with pytest.raises(ValueError, match="window must be one of"):
        source.dates("both")


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        pytest.param({"components": ["tv", "radio"]}, r"unknown components \['radio'\]; available"),
        pytest.param({"components": "tv"}, "not a string"),
        pytest.param({"components": []}, "at least one of"),
        pytest.param({"window": "future"}, "window must be one of"),
        pytest.param({"stacked": False, "interval": 1.5}, "interval must be a fraction"),
        pytest.param({"stacked": False, "truth": [1.0]}, "truth must be a pandas DataFrame"),
        pytest.param({"truth": "frame"}, "truth is drawn only with stacked=False"),
        pytest.param(
            {"stacked": False, "truth": "radio"}, r"truth columns \['radio'\] are not components"
        ),
        pytest.param(
            {"stacked": False, "truth": "sundays"},
            r"truth\['tv'\] has no value on any drawn date",
            id="truth-on-other-dates",
        ),
        pytest.param({"backend": "plotly", "ax": "axes"}, "does not apply to backend='plotly'"),
        pytest.param({"stacked": False, "ax": "axes"}, "draws one panel per component; ax="),
    ],
)
def test_arguments_are_checked(predicted_model, chart_frames, arguments, error):
    from matplotlib.figure import Figure

    dates = pd.DatetimeIndex(chart_frames[0]["date"])
    frames = {"frame": {"tv": 1.0}, "radio": {"tv": 1.0, "radio": 2.0}}
    if isinstance(arguments.get("truth"), str):
        index = dates - pd.Timedelta(days=1) if arguments["truth"] == "sundays" else dates
        arguments["truth"] = pd.DataFrame(frames.get(arguments["truth"], {"tv": 1.0}), index=index)
    if arguments.get("ax") == "axes":
        arguments["ax"] = Figure().subplots()
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=error):
        predicted_model.plot.contributions_over_time(**arguments)
    assert torch.equal(torch.get_rng_state(), state)


def test_plotly_draws_the_matplotlib_numbers(predicted_model, chart_frames):
    pytest.importorskip("plotly")
    from matplotlib.dates import date2num

    history, future = chart_frames
    plot = predicted_model.plot
    layers = _stack(plot.contributions_over_time())
    fig = plot.contributions_over_time(backend="plotly")
    tops = [index for index, trace in enumerate(fig.data) if trace.fill == "tonexty"]
    assert [fig.data[index].name for index in tops[: len(ORDER)]] == list(ORDER)
    assert not any(fig.data[index].showlegend for index in tops[len(ORDER) :])
    hovers = [trace for trace in fig.data if trace.mode == "markers"]
    windows = IdataReader(predicted_model.idata).windows(None)
    means = [window.contribution(name).mean(0) for window in windows for name in ORDER]
    for index, hover, mean, (values, edges, baseline) in zip(
        tops, hovers, means, layers, strict=True
    ):
        top, bottom = fig.data[index], fig.data[index - 1]
        # Each week holds flat to the next edge: the last edge repeats the last week.
        np.testing.assert_array_equal(top.y, np.append(values, values[-1]))
        np.testing.assert_array_equal(bottom.y, np.append(baseline, baseline[-1]))
        np.testing.assert_allclose(date2num(pd.to_datetime(top.x).to_numpy()), edges)
        np.testing.assert_allclose(hover.customdata, mean, rtol=1e-12, atol=1e-9)
    [actual] = [trace for trace in fig.data if trace.name == "Observed outcome"]
    np.testing.assert_array_equal(actual.y, np.concatenate([history["sales"], future["sales"]]))
    [span] = [shape for shape in fig.layout.shapes if shape.name == "Forecast weeks"]
    assert pd.Timestamp(span.x0) == future["date"].iloc[0] - WEEK / 2

    truth = pd.DataFrame({"search": np.arange(6.0)}, index=pd.DatetimeIndex(future["date"]))
    options = {"window": "query", "stacked": False, "interval": 0.7, "truth": truth}
    panels = plot.contributions_over_time(**options).axes
    fig = plot.contributions_over_time(backend="plotly", **options)
    for index, (ax, name) in enumerate(zip(panels, ORDER, strict=False)):
        axis = "x" if index == 0 else f"x{index + 1}"
        high, low, mean, *dashed = [trace for trace in fig.data if trace.xaxis == axis]
        drawn_low, drawn_high = _band(ax.collections[0], ax.lines[0].get_xdata())
        np.testing.assert_array_equal(mean.y, ax.lines[0].get_ydata())
        np.testing.assert_array_equal(low.y, drawn_low)
        np.testing.assert_array_equal(high.y, drawn_high)
        np.testing.assert_array_equal(mean.customdata, np.column_stack((low.y, high.y)))
        if name == "search":
            np.testing.assert_array_equal(dashed[0].y, ax.lines[1].get_ydata())
            np.testing.assert_array_equal(dashed[0].y, np.arange(6.0))
        else:
            assert not dashed
