"""A hosted curve summary: checked like local curves, saved, read and drawn with its warnings."""

import json

import numpy as np
import pytest
import torch
import xarray as xr
from conftest import CHART_ROLES
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from pymc_mfm import load_idata, save_idata, validate_idata
from pymc_mfm.client.plots import MAX_WARNING_LINES, WARNING_COLOR
from pymc_mfm.plotting.charts import Plots
from pymc_mfm.plotting.charts import curves as curve_charts
from pymc_mfm.plotting.charts._data import IdataReader

CHANNELS = CHART_ROLES["channel_columns"]
LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
WEEKS = 2
NAN = np.nan
#: search has every level; tv only 50% (alone between gaps), 150% and 200%.
MEAN = np.array([[0.0, 10.0, 20.0, 30.0, 40.0], [NAN, 15.0, NAN, 35.0, 45.0]])
LOW = MEAN - np.array([[1.0], [5.0]])
HIGH = MEAN + np.array([[2.0], [6.0]])
TOTALS = np.array([80.0, 60.0])
SPEND = "Spend is above the 95th percentile of the history."
SHORT = "History is shorter than 75 weeks."
WARNINGS = np.array([["", "", "", SPEND, ""], ["", f"{SPEND}\n{SHORT}", "", "", ""]])


@pytest.fixture
def inferred(model, chart_frames):
    """A complete local inference tree, without curves: 6 forecast weeks."""
    history, future = chart_frames
    torch.manual_seed(3)
    model.inference(history, future, n_samples=4, **CHART_ROLES)
    return model.idata


def _summarized(tree, *, warnings=WARNINGS):
    """``tree`` with a hosted summary as its curves, shaped as the hosted engine writes it."""
    nodes = {node.path: node.to_dataset(inherit=False) for node in tree.subtree}
    nodes["/response_curves"] = xr.Dataset(
        attrs={
            "weeks": WEEKS,
            "n_samples": 4,
            "units": "outcome",
            "random_seed": 7,
            "source_inference_revision": tree.attrs["inference_revision"],
            "source_provenance": tree["query"].attrs["provenance"],
            "provenance": json.dumps({"engine": "hosted", "analysis_id": "an_curves"}),
        }
    )
    dims = ("channel", "level")
    variables = {
        "mean": (dims, MEAN),
        "low": (dims, LOW),
        "high": (dims, HIGH),
        "media": (dims, TOTALS[:, None] * LEVELS),
        "total_media": ("channel", TOTALS),
    }
    if warnings is not None:
        variables["warning"] = (dims, warnings)
    nodes["/response_curves/summary"] = xr.Dataset(
        variables,
        coords={
            "channel": list(CHANNELS),
            "level": LEVELS,
            "date": tree["query/posterior"]["date"].to_numpy()[:WEEKS],
        },
        attrs={"interval": 0.9},
    )
    return xr.DataTree.from_dict(nodes, name=tree.name)


def _edit_summary(tree, change):
    node = tree["response_curves/summary"]
    node.dataset = change(node.to_dataset(inherit=False))


def _query_dates(tree):
    return tree["query/posterior"]["date"].to_numpy()


@pytest.mark.parametrize("warnings", [WARNINGS, None], ids=["warning", "no-warning"])
def test_a_summary_validates_and_round_trips_through_a_file(inferred, tmp_path, warnings):
    tree = _summarized(inferred, warnings=warnings)
    validate_idata(tree)
    save_idata(tree, tmp_path / "hosted.nc")
    loaded = load_idata(tmp_path / "hosted.nc")
    summary = IdataReader(loaded).curve_summary()
    assert summary.channels == CHANNELS
    np.testing.assert_array_equal(summary.levels, LEVELS)
    for read, stored in ((summary.mean, MEAN), (summary.low, LOW), (summary.high, HIGH)):
        np.testing.assert_array_equal(read, stored)  # NaN at the same missing points
    expected = np.full(MEAN.shape, "") if warnings is None else warnings
    assert summary.warnings.tolist() == expected.tolist()
    np.testing.assert_array_equal(
        loaded["response_curves/summary"]["date"].to_numpy(), _query_dates(inferred)[:WEEKS]
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            # The summary of an earlier run, copied onto the next inference of the model.
            lambda tree: tree.attrs.update(inference_revision=tree.attrs["inference_revision"] + 1),
            r"^/response_curves: source_inference_revision (\d+) differs from the tree's "
            r"inference_revision (?!\1)\d+",
            id="stale-revision",
        ),
        pytest.param(
            lambda tree: tree["response_curves"].attrs.update(source_provenance="{}"),
            r"^/response_curves: source_provenance differs from the provenance of /query",
            id="other-provenance",
        ),
        pytest.param(
            lambda tree: tree["response_curves"].attrs.update(weeks=99),
            r"^/response_curves/summary: date coordinate must hold the first 99 dates of "
            r"/query/posterior .*; /query/posterior has 6 dates",
            id="weeks-past-the-forecast",
        ),
        pytest.param(
            lambda tree: _edit_summary(
                tree, lambda data: data.assign_coords(date=_query_dates(tree)[1 : WEEKS + 1])
            ),
            r"^/response_curves/summary: date coordinate must hold the first 2 dates",
            id="later-dates",
        ),
        pytest.param(
            lambda tree: _edit_summary(tree, lambda data: data.drop_vars("date")),
            r"^/response_curves/summary: date coordinate is missing",
            id="no-dates",
        ),
        pytest.param(
            lambda tree: _edit_summary(tree, lambda data: data.assign_coords(date=[0, 1])),
            r"^/response_curves/summary: date coordinate must hold datetime64",
            id="numeric-dates",
        ),
        pytest.param(
            lambda tree: tree["response_curves"].attrs.update(n_samples=0),
            r"^/response_curves: attribute 'n_samples' must be positive",
            id="no-samples",
        ),
        pytest.param(
            lambda tree: tree["response_curves"].attrs.update(units=""),
            r"^/response_curves: attribute 'units' must be nonempty text",
            id="no-units",
        ),
        pytest.param(
            lambda tree: tree["response_curves"].attrs.update(random_seed=-1),
            r"^/response_curves: attribute 'random_seed' must be an integer from 0",
            id="negative-seed",
        ),
        pytest.param(
            lambda tree: _edit_summary(
                tree, lambda data: data.assign(warning=(("channel", "level"), np.zeros((2, 5))))
            ),
            r"^/response_curves/summary/warning: must hold text, got dtype float64",
            id="numeric-warning",
        ),
        pytest.param(
            lambda tree: _edit_summary(
                tree, lambda data: data.assign(warning=("channel", ["", ""]))
            ),
            r"^/response_curves/summary/warning: expected dims \('channel', 'level'\)",
            id="warning-per-channel",
        ),
    ],
)
def test_validate_rejects_a_summary_that_is_not_of_this_inference(inferred, change, message):
    tree = _summarized(inferred)
    validate_idata(tree)
    change(tree)
    with pytest.raises(ValueError, match=message):
        validate_idata(tree)


def _rings(ax):
    """The open red markers of ``ax``, as ``(level, mean)`` rows."""
    red = to_rgba(WARNING_COLOR)
    found = [
        collection
        for collection in ax.collections
        if len(collection.get_edgecolor()) and np.allclose(collection.get_edgecolor()[0], red)
    ]
    for collection in found:
        assert len(collection.get_facecolor()) == 0  # open: the mean shows through
    return np.concatenate([c.get_offsets() for c in found]) if found else np.empty((0, 2))


def test_matplotlib_rings_warned_levels_and_lists_each_warning_once(inferred):
    fig = Plots(_summarized(inferred)).response_curves(backend="matplotlib")
    search, tv = fig.axes
    np.testing.assert_array_equal(_rings(search), [[1.5, 30.0]])
    np.testing.assert_array_equal(_rings(tv), [[0.5, 15.0]])
    text = " ".join(fig.get_suptitle().split())
    assert f"Warning (search, tv): {SPEND}" in text
    assert f"Warning (tv): {SHORT}" in text
    assert text.count(SPEND) == 1
    [legend] = fig.legends
    assert "Level has a warning (see caption)" in [label.get_text() for label in legend.texts]

    plain = Plots(_summarized(inferred, warnings=None)).response_curves(backend="matplotlib")
    assert all(_rings(ax).size == 0 for ax in plain.axes)
    assert "Warning" not in plain.get_suptitle()


def test_plotly_rings_warned_levels_and_lists_each_warning_once(inferred):
    fig = Plots(_summarized(inferred)).response_curves(backend="plotly")
    rings = [
        trace
        for trace in fig.data
        if trace.marker.symbol == "circle-open" and trace.marker.color == WARNING_COLOR
    ]
    assert [ring.xaxis for ring in rings] == ["x", "x2"]
    for ring, level, mean in zip(rings, (1.5, 0.5), (30.0, 15.0), strict=True):
        np.testing.assert_array_equal(ring.x, [level])
        np.testing.assert_array_equal(ring.y, [mean])
    assert rings[1].hovertext == (f"{SPEND}<br>{SHORT}",)
    assert [ring.showlegend for ring in rings] == [True, False]
    assert f"Warning (search, tv): {SPEND}" in fig.layout.title.text
    assert f"Warning (tv): {SHORT}" in fig.layout.title.text

    plain = Plots(_summarized(inferred, warnings=None)).response_curves(backend="plotly")
    assert not [trace for trace in plain.data if trace.marker.symbol == "circle-open"]
    assert "Warning" not in plain.layout.title.text


def test_the_caption_lists_at_most_max_warning_lines(inferred):
    # 11 distinct warnings, only at levels with a mean: tv's levels hold two each.
    many = np.array(
        [
            [f"warning {i}" for i in range(5)],
            ["", "warning 5\nwarning 6", "", "warning 7\nwarning 8", "warning 9\nwarning 10"],
        ]
    )
    caption = curve_charts.prepare(IdataReader(_summarized(inferred, warnings=many))).caption
    listed = [line for line in caption if line.startswith("Warning (")]
    assert listed == [
        f"Warning ({'search' if i < 5 else 'tv'}): warning {i}" for i in range(MAX_WARNING_LINES)
    ]
    rest = 11 - MAX_WARNING_LINES
    assert caption[-1] == f"{rest} more warnings: see /response_curves/summary/warning."


def test_matplotlib_draws_a_bar_where_a_band_has_no_area(inferred):
    fig = Plots(_summarized(inferred)).response_curves(backend="matplotlib")
    search, tv = fig.axes
    assert not [c for c in search.collections if isinstance(c, LineCollection)]
    [bar] = [c for c in tv.collections if isinstance(c, LineCollection)]
    [segment] = bar.get_segments()
    np.testing.assert_array_equal(segment, [[0.5, 10.0], [0.5, 21.0]])
    assert bar.get_linewidth()[0] > 1 and bar.get_alpha() > 0
    # The band covers 150% to 200% only: nothing joins it to the bar across the gap.
    band = tv.collections[0]
    spans = [
        (path.vertices[:, 0].min(), path.vertices[:, 0].max())
        for path in band.get_paths()
        if np.ptp(path.vertices[:, 0]) > 0
    ]
    assert spans == [(1.5, 2.0)]


def test_plotly_draws_a_bar_where_a_band_has_no_area(inferred):
    fig = Plots(_summarized(inferred)).response_curves(backend="plotly")
    tv = [trace for trace in fig.data if trace.legendgroup == "interval" and trace.xaxis == "x2"]
    bands = [trace for trace in tv if trace.line.width == 0]
    assert [list(trace.x) for trace in bands] == [[1.5, 2.0], [1.5, 2.0]]
    [bar] = [trace for trace in tv if trace.line.width]
    np.testing.assert_array_equal(bar.x, [0.5, 0.5, NAN])
    np.testing.assert_array_equal(bar.y, [10.0, 21.0, NAN])
    assert bar.mode == "lines" and bar.line.width > 1 and not bar.connectgaps
    search = [trace for trace in fig.data if trace.legendgroup == "interval" and trace.xaxis == "x"]
    assert all(trace.line.width == 0 for trace in search)
