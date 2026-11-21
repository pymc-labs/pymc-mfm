"""``Plots(idata)`` and ``model.plot``: charts read the tree alone, live or from a file."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES

import pymc_mktfm.plotting as plotting
import pymc_mktfm.plotting.charts as charts
from pymc_mktfm import validate_idata
from pymc_mktfm.model.model import MFM
from pymc_mktfm.plotting import Plots
from pymc_mktfm.plotting.charts import _backend
from pymc_mktfm.plotting.charts._data import IdataReader

#: Every chart of the accessor, with options that exercise a second code path.
CHARTS = {
    "inputs": ("inputs", {}),
    "fit": ("fit", {}),
    "fit-query": ("fit", {"window": "query", "interval": 0.5}),
    "shares": ("shares", {}),
    "shares-dist": ("shares", {"view": "dist", "share": "total", "window": "query"}),
    "totals": ("contribution_totals", {}),
    "media": ("contribution_vs_media", {}),
    "curves": ("response_curves", {}),
    "contributions": ("contributions_over_time", {}),
    "contributions-panels": ("contributions_over_time", {"stacked": False, "window": "query"}),
}
WITH_LEVEL_ONE = [0, 0.5, 1, 2]
WITHOUT_LEVEL_ONE = [0, 0.5, 2]


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


@pytest.fixture(params=[WITH_LEVEL_ONE, WITHOUT_LEVEL_ONE], ids=["level-1", "no-level-1"])
def swept(request, model, chart_frames):
    """The model after cheap complete inference and a sweep, and the curves it returned."""
    history, future = chart_frames
    torch.manual_seed(3)
    model.inference(history, future, n_samples=6, **CHART_ROLES)
    return model, model.response_curves(levels=request.param, weeks=2, n_samples=4)


@pytest.fixture
def netcdf():
    """Skip a file test without the netCDF backend (``pymc-mktfm[draws]``)."""
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")


def _saved(tree, tmp_path):
    """``tree`` written to a netCDF file and opened again, loaded."""
    import xarray as xr

    tree.to_netcdf(tmp_path / "tree.nc", engine="h5netcdf")
    return xr.open_datatree(tmp_path / "tree.nc", engine="h5netcdf").load()


def _facts(fig):
    """Titles, labels, plotted lines, patches and collections of a matplotlib figure."""
    texts = [fig.get_suptitle()] + [text.get_text() for text in fig.texts]
    axes = []
    for ax in fig.axes:
        axes.append(
            {
                "labels": (
                    ax.get_title(),
                    ax.get_title(loc="left"),
                    ax.get_xlabel(),
                    ax.get_ylabel(),
                ),
                "lines": [(line.get_xdata(), line.get_ydata()) for line in ax.lines],
                "patches": [
                    p.get_patch_transform().transform(p.get_path().vertices) for p in ax.patches
                ],
                "paths": [
                    np.concatenate([path.vertices for path in c.get_paths()])
                    for c in ax.collections
                    if c.get_paths()
                ],
                "offsets": [np.asarray(c.get_offsets()) for c in ax.collections],
                "texts": [t.get_text() for t in ax.texts],
            }
        )
    return texts, axes


def _same(left, right):
    """Equal figures: the same text and the same plotted numbers to rtol 1e-5."""
    assert left[0] == right[0]
    assert len(left[1]) == len(right[1])
    for a, b in zip(left[1], right[1], strict=True):
        assert a["labels"] == b["labels"]
        assert a["texts"] == b["texts"]
        assert len(a["lines"]) == len(b["lines"])
        for (ax_, ay), (bx, by) in zip(a["lines"], b["lines"], strict=True):
            np.testing.assert_allclose(np.asarray(ax_, float), np.asarray(bx, float), rtol=1e-5)
            np.testing.assert_allclose(np.asarray(ay, float), np.asarray(by, float), rtol=1e-5)
        for key in ("patches", "paths", "offsets"):
            assert len(a[key]) == len(b[key])
            for x, y in zip(a[key], b[key], strict=True):
                np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def _draw_all(plots):
    import matplotlib.pyplot as plt

    drawn = {}
    for key, (method, options) in CHARTS.items():
        fig = getattr(plots, method)(**options)
        drawn[key] = _facts(fig)
        plt.close(fig)
    return drawn


def _check_curves(tree, returned):
    curves = IdataReader(tree).response_curves()
    assert curves.channels == returned.channels
    np.testing.assert_array_equal(curves.levels, returned.levels)
    np.testing.assert_allclose(curves.media, returned.media, rtol=1e-12)
    np.testing.assert_array_equal(curves.draws, returned.draws)


@pytest.mark.usefixtures("matplotlib_backend")
def test_charts_from_the_model_tree_equal_charts_from_the_model(swept):
    swept, returned = swept
    assert (returned.media > 0).all()
    _check_curves(swept.idata, returned)
    expected = _draw_all(Plots(swept.idata))
    for key, facts in _draw_all(swept.plot).items():
        _same(facts, expected[key])


@pytest.mark.usefixtures("matplotlib_backend", "netcdf")
def test_charts_from_a_file_equal_charts_from_the_model_tree(swept, tmp_path):
    swept, returned = swept
    expected = _draw_all(Plots(swept.idata))
    reread = _saved(swept.idata, tmp_path)
    _check_curves(reread, returned)
    for key, facts in _draw_all(Plots(reread)).items():
        _same(facts, expected[key])


@pytest.mark.usefixtures("matplotlib_backend")
def test_model_plot_reads_the_tree_not_the_private_state(swept, monkeypatch):
    swept, _ = swept
    expected = _draw_all(Plots(swept.idata))

    def refuse(name):
        def read(self):
            raise AssertionError(f"chart read model.{name}")

        return property(read)

    for name in ("_prediction_snapshot", "query_result"):
        monkeypatch.setattr(MFM, name, refuse(name), raising=False)
    for key, facts in _draw_all(swept.plot).items():
        _same(facts, expected[key])


@pytest.mark.usefixtures("matplotlib_backend")
def test_model_plot_follows_later_runs_and_plots_of_a_tree_do_not(model, chart_frames):
    history, future = chart_frames
    model.inference(history, future, n_samples=4, **CHART_ROLES)
    old_tree = model.idata
    plots, snapshot = model.plot, Plots(old_tree)
    first = _facts(snapshot.fit())
    _same(_facts(plots.fit()), first)
    torch.manual_seed(7)
    model.inference(history, future, n_samples=4, **CHART_ROLES)
    assert model.idata is not old_tree
    _same(_facts(plots.fit()), _facts(Plots(model.idata).fit()))
    _same(_facts(snapshot.fit()), first)
    assert not np.array_equal(_facts(plots.fit())[1][0]["lines"][0][1], first[1][0]["lines"][0][1])


@pytest.mark.usefixtures("matplotlib_backend")
def test_the_plot_accessor_returns_before_inference_and_the_chart_call_raises(model):
    plots = model.plot
    with pytest.raises(RuntimeError, match=r"inference\(history, future, \.\.\.\)"):
        plots.inputs()
    with pytest.raises(ValueError, match="backend must be one of"):
        plots.fit(backend="svg")


def _rebuilt(tree, reorder):
    """A copy of ``tree`` whose groups are rebuilt by the functions in ``reorder`` (by path)."""
    import xarray as xr

    groups = {}
    for node in tree.subtree:
        data = node.to_dataset(inherit=False)
        if node.path in reorder:
            data = reorder[node.path](data)
        groups[node.path] = data
    changed = xr.DataTree.from_dict(groups)
    changed.attrs = dict(tree.attrs)
    return changed


def _reversed(**dims):
    """Reorder a whole dataset, coordinates and values together, along ``dims``."""

    def reorder(data):
        return data.isel(
            {dim: np.arange(data.sizes[dim])[::-1] for dim in dims if dim in data.sizes}
        )

    return reorder


def _stored(tree, path, dim):
    return [str(label) for label in tree[path].to_dataset(inherit=False)[dim].to_numpy()]


@pytest.fixture
def reordered(predicted_model, chart_frames):
    """The model, its tree, and the tree with each window's groups stored in another order.

    The context stores reversed components, channels and dates; the query a rotated
    component order, reversed channels and dates; the curves reversed channels and levels.
    """
    returned = predicted_model.response_curves(levels=[0, 0.5, 2], weeks=2, n_samples=4)
    tree = predicted_model.idata
    n_query = tree["/query/constant_data"].sizes["component"]
    rotate = np.roll(np.arange(n_query), 1)
    flip = _reversed(component=1, channel=1, date=1, control=1)
    # Posterior dates stay in drawing order: they are the weeks the window draws.
    flip_posterior = _reversed(channel=1)
    changed = _rebuilt(
        tree,
        {
            "/context/constant_data": flip,
            "/context/observed_data": flip,
            "/context/posterior": flip_posterior,
            "/query/constant_data": lambda data: _reversed(channel=1, date=1)(
                data.isel(component=rotate)
            ),
            "/query/observed_data": flip,
            "/query/posterior": flip_posterior,
            "/response_curves/constant_data": _reversed(channel=1, level=1, date=1),
            "/response_curves/posterior": flip_posterior,
        },
    )
    for path, dim in (
        ("/context/constant_data", "component"),
        ("/context/constant_data", "channel"),
        ("/context/constant_data", "date"),
        ("/query/constant_data", "component"),
        ("/query/constant_data", "channel"),
        ("/query/constant_data", "date"),
        ("/context/posterior", "channel"),
        ("/response_curves/constant_data", "level"),
    ):
        assert _stored(changed, path, dim) != _stored(tree, path, dim), (path, dim)
    return predicted_model, returned, tree, changed


def test_charts_of_a_reordered_tree_draw_the_numbers_of_their_labels(reordered, chart_frames):
    """Each chart of a re-stored tree draws what the input frames and the sweep hold."""
    from pymc_mktfm.plotting.charts import contributions, fit, inputs, media

    _, returned, tree, changed = reordered
    history, future = chart_frames
    both = pd.concat([history, future])
    source = IdataReader(changed)
    original = IdataReader(tree)

    series = {name: values for name, _, values in inputs.prepare(source).series}
    np.testing.assert_array_equal(series["outcome"], both["sales"])
    for name in ("search", "tv", "temperature"):
        np.testing.assert_array_equal(series[name], both[name])
    np.testing.assert_array_equal(fit.prepare(source).observed, both["sales"])

    for window, frame in (("context", history), ("query", future)):
        panels = media.prepare(source, window=window).channels
        assert sorted(panel.name for panel in panels) == ["search", "tv"]
        reference = {p.name: p for p in media.prepare(original, window=window).channels}
        for panel in panels:
            np.testing.assert_array_equal(panel.media, frame[panel.name])
            np.testing.assert_array_equal(panel.mean, reference[panel.name].mean)
            np.testing.assert_array_equal(panel.low, reference[panel.name].low)

    # Layers sit in stored component order within a family: compare each by name.
    drawn, expected = contributions.prepare(source), contributions.prepare(original)
    assert sorted(drawn.components) == sorted(expected.components)
    for got, want in zip(drawn.segments, expected.segments, strict=True):
        for name in expected.components:
            a, b = drawn.components.index(name), expected.components.index(name)
            np.testing.assert_array_equal(got.mean[a], want.mean[b], err_msg=name)

    curves = source.response_curves()
    assert sorted(curves.channels) == sorted(returned.channels)
    order = [curves.channels.index(name) for name in returned.channels]
    np.testing.assert_array_equal(curves.levels, returned.levels)
    np.testing.assert_array_equal(curves.draws[order], returned.draws)
    swept_totals = future.iloc[:2][list(returned.channels)].sum().to_numpy()
    np.testing.assert_allclose(curves.media[order], swept_totals, rtol=1e-12)
    np.testing.assert_allclose(curves.media[order], returned.media, rtol=1e-12)
    # The stored amounts were reversed over channels and levels: read them back by label.
    amounts = source.curve_media()
    np.testing.assert_allclose(
        amounts[order], returned.media[:, None] * returned.levels, rtol=1e-12
    )


def test_a_saved_and_reopened_reordered_tree_draws_the_raw_arrays_by_label(
    reordered, netcdf, tmp_path
):
    """Plotted means equal the original tree's raw arrays, picked by label with xarray."""
    from pymc_mktfm.plotting.charts import contributions, media

    _, returned, tree, changed = reordered
    source = IdataReader(_saved(changed, tmp_path))

    def raw(window, family, label):
        group = tree[f"/{window}/posterior"].to_dataset(inherit=False)
        return group[f"{family}_contribution"].sel({family: label}).mean(("chain", "draw"))

    for window in ("context", "query"):
        for panel in media.prepare(source, window=window).channels:
            np.testing.assert_allclose(panel.mean, raw(window, "channel", panel.name), rtol=1e-12)
    drawn = contributions.prepare(source)
    for position, name in enumerate(drawn.components):
        family = {"search": "channel", "tv": "channel", "temperature": "control"}.get(name)
        if family is None:
            continue
        for segment, window in zip(drawn.segments, ("context", "query"), strict=True):
            np.testing.assert_allclose(
                segment.mean[position], raw(window, family, name), rtol=1e-12, err_msg=name
            )
    curves = source.response_curves()
    for row, channel in enumerate(curves.channels):
        want = returned.draws[returned.channels.index(channel)]
        np.testing.assert_array_equal(curves.draws[row], want)


@pytest.mark.usefixtures("matplotlib_backend")
def test_the_figures_of_a_reordered_tree_equal_the_figures_of_the_original(reordered):
    _, _, tree, changed = reordered
    for key in ("inputs", "fit", "shares"):
        method, options = CHARTS[key]
        _same(
            _facts(getattr(Plots(changed), method)(**options)),
            _facts(getattr(Plots(tree), method)(**options)),
        )


def _edit(tree, path, change):
    return _rebuilt(tree, {path: change})


@pytest.mark.parametrize(
    ("path", "change", "message"),
    [
        pytest.param(
            "/context/constant_data",
            lambda data: data.assign_coords(
                channel=[name.upper() for name in data["channel"].to_numpy()]
            ),
            r"/context/constant_data: channel labels \['SEARCH', 'TV'\] do not hold \['search'",
            id="media-channel-labels",
        ),
        pytest.param(
            "/query/constant_data",
            lambda data: data.assign_coords(date=data["date"].to_numpy() + np.timedelta64(1, "D")),
            r"/query/constant_data.*date",
            id="media-dates",
        ),
        pytest.param(
            "/query/observed_data",
            lambda data: data.assign_coords(date=data["date"].to_numpy() + np.timedelta64(1, "D")),
            r"/query/observed_data.*date",
            id="outcome-dates",
        ),
        pytest.param(
            "/context/constant_data",
            lambda data: data.assign(
                component_family=("component", ["nonsense"] * data.sizes["component"])
            ),
            r"/context/constant_data.*unknown component family 'nonsense'.*baseline",
            id="unknown-family",
        ),
        pytest.param(
            "/context/posterior",
            lambda data: data.assign_coords(channel=["x", "y"]),
            r"/context/posterior.*channel.*'search'",
            id="contribution-channel-labels",
        ),
    ],
)
def test_labels_that_do_not_match_raise_naming_the_group_and_labels(
    predicted_model, path, change, message
):
    changed = _edit(predicted_model.idata, path, change)
    reader = IdataReader(changed)
    with pytest.raises(ValueError, match=message):
        reader.covariates()
        reader.windows(None)


def test_windows_with_other_component_names_raise(predicted_model):
    from pymc_mktfm.plotting.charts import contributions

    tree = predicted_model.idata
    # Drop a component from the query window only: the stack cannot draw both windows.
    changed = _edit(
        tree,
        "/query/constant_data",
        lambda data: data.isel(component=slice(1, None)),
    )
    # The reader refuses the window before the stack compares component names.
    with pytest.raises(ValueError, match=r"/query/posterior.*no descriptor.*'baseline'"):
        contributions.prepare(IdataReader(changed))


def test_a_window_checks_its_columns_and_labels_without_stacking_draws(
    predicted_model, monkeypatch
):
    reader = IdataReader(predicted_model.idata)

    def refuse(*args, **kwargs):
        raise AssertionError("stacked the draws")

    monkeypatch.setattr(np, "stack", refuse)
    reader.check_window("context")
    changed = _edit(
        predicted_model.idata,
        "/context/constant_data",
        lambda data: data.assign_coords(channel=["x", "y"]),
    )
    with pytest.raises(ValueError, match="/context/constant_data"):
        IdataReader(changed).check_window("context")


@pytest.mark.parametrize("path", ["/context/posterior", "/response_curves/posterior"])
def test_a_tree_with_more_than_one_chain_raises(predicted_model, path):
    predicted_model.response_curves(levels=[0, 2], weeks=1, n_samples=3)

    def two_chains(data):
        import xarray as xr

        return xr.concat([data, data.assign_coords(chain=[1])], dim="chain")

    changed = _edit(predicted_model.idata, path, two_chains)
    reader = IdataReader(changed)
    with pytest.raises(ValueError, match=rf"{path}.*one chain.*got 2"):
        if "curves" in path:
            reader.response_curves()
        else:
            reader.window("context")


def test_repeated_component_names_raise_and_do_not_hang():
    from pymc_mktfm.plotting.charts._data import unique_names

    assert unique_names(["a", "a", "b"], ["x:a", "y:a", "b"]) == ("x:a", "y:a", "b")
    with pytest.raises(ValueError, match="component names must be unique"):
        unique_names(["a", "a"], ["x", "x"])


def test_a_tree_that_is_not_a_model_tree_is_named(predicted_model):
    with pytest.raises(ValueError, match="not a model tree"):
        IdataReader(predicted_model.idata["context"]).window("context")
    with pytest.raises(ValueError, match="window must be"):
        IdataReader(predicted_model.idata).window("both")


@pytest.mark.usefixtures("matplotlib_backend", "netcdf")
def test_a_file_draws_the_charts_with_torch_blocked(swept, tmp_path):
    swept, _ = swept
    swept.idata.to_netcdf(tmp_path / "tree.nc", engine="h5netcdf")
    script = textwrap.dedent(
        """
        import sys

        sys.modules["torch"] = None  # any import of torch now raises ImportError
        import xarray as xr

        from pymc_mktfm.plotting import Plots

        plots = Plots(xr.open_datatree(sys.argv[1], engine="h5netcdf").load())
        for name in (
            "inputs",
            "fit",
            "shares",
            "contribution_totals",
            "contribution_vs_media",
            "response_curves",
            "contributions_over_time",
        ):
            getattr(plots, name)()
        """
    )
    run = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "tree.nc")],
        env={"MPLBACKEND": "Agg", "PATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr


def _run(code):
    """Warnings (as ``category:message`` lines) that ``code`` raises in a fresh process."""
    script = "import warnings\nwarnings.simplefilter('always')\n" + textwrap.dedent(code)
    script += "\nfor w in caught:\n    print(w.category.__name__ + ':' + str(w.message))\n"
    run = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        env={"PATH": "", "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    return run.stdout.splitlines()


@pytest.mark.parametrize("module", ["pymc_mktfm.plotting", "pymc_mktfm.plotting.charts"])
def test_model_plots_warns_once_per_call_and_not_on_import(module):
    lines = _run(
        f"""
        class Model:
            idata = None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from {module} import ModelPlots
        assert not caught
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert type(ModelPlots(Model())).__name__ == "Plots"
        """
    )
    assert len(lines) == 1
    assert lines[0].startswith("DeprecationWarning:ModelPlots is deprecated")
    assert "Plots(idata)" in lines[0]


def test_backends_warns_once_per_process_on_import():
    lines = _run(
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from pymc_mktfm.plotting.charts import BACKENDS
            from pymc_mktfm.plotting.charts import BACKENDS as again
            import pymc_mktfm.plotting.charts as charts
            assert charts.BACKENDS == BACKENDS == again == ("matplotlib", "plotly")
        """
    )
    assert len(lines) == 1
    assert "BACKENDS is deprecated" in lines[0]


def test_unknown_module_attributes_still_raise():
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        charts.missing  # noqa: B018


def test_exported_names():
    assert sorted(charts.__all__) == ["Plots", "set_backend"]
    assert sorted(plotting.__all__) == sorted(
        [
            "GraphInspector",
            "GraphvizGraph",
            "Plots",
            "graph_to_dot",
            "plot_attribution",
            "plot_calibration",
            "plot_cardinality",
            "plot_channel_totals",
            "plot_component_trajectories",
            "plot_corpus_summary",
            "plot_cumulative",
            "plot_cycles",
            "plot_dag",
            "plot_family_totals",
            "plot_graph_inspector",
            "plot_graph_posterior",
            "plot_graph_statistics",
            "plot_panel",
            "plot_records",
            "plot_recovery_scatter",
            "plot_reliability",
            "plot_residual_by_magnitude",
            "plot_sbc_report",
            "plot_set_posterior",
            "plot_strata",
            "plot_strata_table",
            "plot_structure_heatmap",
            "plot_support",
            "plot_target_recovery",
            "plot_temporal",
            "plot_uncertainty",
            "plot_window_totals",
            "plot_zero_atoms",
            "set_backend",
        ]
    )


@pytest.mark.usefixtures("matplotlib_backend")
def test_a_plot_property_that_calls_model_plots_returns_a_live_accessor(chart_frames):
    import warnings

    from pymc_mktfm.model.config import ModelConfig
    from pymc_mktfm.plotting.charts import ModelPlots

    class Custom(MFM):
        plot = property(lambda self: ModelPlots(self))

    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=2
    )
    custom = Custom(config).eval()
    with pytest.warns(DeprecationWarning, match="ModelPlots is deprecated"):
        plots = custom.plot
    assert isinstance(plots, Plots)
    with pytest.raises(RuntimeError, match=r"inference\(history, future"):
        plots.inputs()
    history, future = chart_frames
    custom.inference(history, future, n_samples=3, **CHART_ROLES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The accessor from before the run is live: it reads the tree of the run.
        assert plots.inputs() is not None


def _swept_forecast(model, chart_frames, levels, weeks, search=None):
    history, future = chart_frames
    future = future.iloc[:weeks].copy()
    if search is not None:
        future["search"] = search
    torch.manual_seed(3)
    model.inference(history, future, n_samples=4, **CHART_ROLES)
    return model.response_curves(levels=levels, weeks=weeks, n_samples=3)


def test_the_curve_baseline_is_the_observed_total_when_no_level_is_one(model, chart_frames):
    """A tiny top level used to erase the total: ``total * level`` underflowed to zero."""
    returned = _swept_forecast(
        model, chart_frames, [0, np.nextafter(0.0, 1.0)], weeks=1, search=0.25
    )
    curves = IdataReader(model.idata).response_curves()
    assert returned.media[returned.channels.index("search")] == 0.25
    np.testing.assert_array_equal(curves.media, returned.media)
    from pymc_mktfm.plotting.charts import curves as curve_chart

    titles = curve_chart.prepare(IdataReader(model.idata)).panel_titles
    assert titles[curves.channels.index("search")].startswith("search · media covariate total 0.25")


def test_the_curve_baseline_is_finite_when_the_swept_total_overflows(model, chart_frames):
    """The stored ``total * level`` holds ``inf`` at the top level: dividing it gave ``inf``."""
    levels = [0, 0.5, 2]
    returned = _swept_forecast(model, chart_frames, levels, weeks=1)
    tree = model.idata
    channel = list(tree["/query/constant_data"]["channel"].to_numpy()).index("search")

    def huge_media(data):
        media = data["media"].copy()
        media[{"channel": channel}] = 1e308
        return data.assign(media=media)

    def huge_curves(data):
        stored = data["media"].copy()
        with np.errstate(over="ignore"):
            stored[{"channel": channel}] = 1e308 * np.asarray(levels)
        return data.assign(media=stored)

    changed = _rebuilt(
        tree,
        {"/query/constant_data": huge_media, "/response_curves/constant_data": huge_curves},
    )
    assert np.isinf(changed["/response_curves/constant_data"]["media"].to_numpy()).any()
    curves = IdataReader(changed).response_curves()
    assert curves.media[curves.channels.index("search")] == 1e308
    assert (
        curves.media[curves.channels.index("tv")] == returned.media[returned.channels.index("tv")]
    )


def _with_stored_amounts(tree, flip):
    """``tree`` whose curve media holds a distinct amount per channel and level.

    The amount of channel ``k`` at level ``l`` is ``1000 * (k + 1) + 7 * l + 0.5``, by the
    labels the tree stored them with. ``flip`` re-stores the group in reverse order.
    Returns the tree and ``{(channel, level): amount}``.
    """
    group = tree["/response_curves/constant_data"].to_dataset(inherit=False)
    channels, levels = group["channel"].to_numpy(), group["level"].to_numpy()
    grid = 1000.0 * (np.arange(len(channels))[:, None] + 1) + 7.0 * np.arange(len(levels)) + 0.5
    amounts = {
        (str(channel), float(level)): grid[row, column]
        for row, channel in enumerate(channels)
        for column, level in enumerate(levels)
    }

    def store(data):
        data = data.assign(media=(("channel", "level"), grid))
        return _reversed(channel=1, level=1)(data) if flip else data

    return _rebuilt(tree, {"/response_curves/constant_data": store}), amounts


@pytest.mark.parametrize("flip", [False, True], ids=["stored-order", "reversed-order"])
def test_the_curve_amounts_are_the_stored_media_by_channel_and_level(predicted_model, flip):
    """The chart data and the Plotly hover show the stored amounts, not total times level."""
    pytest.importorskip("plotly")
    from pymc_mktfm.plotting.charts import curves as curve_chart

    predicted_model.response_curves(levels=[0, 0.5, 2], weeks=2, n_samples=4)
    changed, amounts = _with_stored_amounts(predicted_model.idata, flip)
    data = curve_chart.prepare(IdataReader(changed))
    for row, channel in enumerate(data.channels):
        for column, level in enumerate(data.levels):
            assert data.media[row, column] == amounts[(channel, float(level))]
    fig = Plots(changed).response_curves(backend="plotly")
    hovers = [trace.customdata for trace in fig.data if trace.name == "Mean"]
    assert len(hovers) == len(data.channels)
    for row, hover in enumerate(hovers):
        np.testing.assert_array_equal(hover[:, 0], data.media[row])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            lambda data: data.assign_coords(
                channel=[name.upper() for name in data["channel"].to_numpy()]
            ),
            r"/response_curves/constant_data: channel labels \['SEARCH', 'TV'\] do not hold",
            id="channel-labels",
        ),
        pytest.param(
            lambda data: data.assign_coords(level=data["level"].to_numpy() + 0.25),
            r"/response_curves/constant_data: level labels .* do not hold",
            id="level-labels",
        ),
    ],
)
@pytest.mark.usefixtures("matplotlib_backend")
def test_curve_media_labels_that_do_not_match_the_curves_raise(predicted_model, change, message):
    predicted_model.response_curves(levels=[0, 0.5, 2], weeks=2, n_samples=4)
    changed = _edit(predicted_model.idata, "/response_curves/constant_data", change)
    with pytest.raises(ValueError, match=message):
        Plots(changed).response_curves()


def _drop_tv(data):
    keep = data["component"].to_numpy() != "channel:tv"
    assert not keep.all()
    return data.isel(component=keep)


@pytest.fixture
def without_tv(predicted_model):
    """The tree with the ``channel:tv`` descriptor deleted from both windows; draws stay."""

    changed = _rebuilt(
        predicted_model.idata,
        {"/context/constant_data": _drop_tv, "/query/constant_data": _drop_tv},
    )
    return predicted_model.idata, changed


@pytest.mark.usefixtures("matplotlib_backend")
def test_a_missing_component_descriptor_raises_instead_of_dropping_a_contribution(without_tv):
    tree, changed = without_tv
    # The channels contribute different amounts, so a dropped one would change the fit.
    context = IdataReader(tree).window("context")
    assert not np.allclose(context.contribution("tv"), context.contribution("search"))
    for name in ("context", "query"):
        with pytest.raises(ValueError, match=rf"/{name}/constant_data.*channel.*'tv'"):
            IdataReader(changed).check_window(name)
        with pytest.raises(ValueError, match=rf"/{name}/constant_data.*channel.*'tv'"):
            IdataReader(changed).window(name)
    # A window that alone lacks a descriptor fails in the reader, not later when stacking.
    for name in ("context", "query"):
        one = _rebuilt(tree, {f"/{name}/constant_data": _drop_tv})
        with pytest.raises(ValueError, match=rf"/{name}/constant_data.*channel.*'tv'"):
            Plots(one).contributions_over_time()
    for chart in ("fit", "shares", "contribution_totals", "contributions_over_time"):
        with pytest.raises(ValueError, match=r"/context/constant_data.*channel.*'tv'"):
            getattr(Plots(changed), chart)()


@pytest.mark.parametrize(("family", "label"), [("channel", "tv"), ("control", "temperature")])
@pytest.mark.usefixtures("matplotlib_backend")
def test_two_descriptors_for_one_draw_column_are_rejected(predicted_model, family, label):
    def add_alias(data):
        labels = data["component"].to_numpy().tolist()
        alias = labels.index(f"{family}:{label}")
        return data.isel(component=[*range(len(labels)), alias]).assign_coords(
            component=[*labels, label]
        )

    changed = _edit(predicted_model.idata, "/query/constant_data", add_alias)
    message = rf"^/query/constant_data: {family} descriptors must name each column once"
    with pytest.raises(ValueError, match=message):
        Plots(changed).contribution_totals(window="query")
    with pytest.raises(ValueError, match=message):
        validate_idata(changed)


def test_a_descriptor_for_a_channel_the_tree_does_not_name_raises(predicted_model):
    changed = _edit(
        predicted_model.idata,
        "/context/constant_data",
        lambda data: data.assign_coords(
            component=[
                "channel:ghost" if label == "channel:tv" else label
                for label in data["component"].to_numpy()
            ]
        ),
    )
    with pytest.raises(ValueError, match=r"/context/.*'ghost'"):
        IdataReader(changed).check_window("context")


@pytest.mark.parametrize("label", ["baseline", "indirect:cc"])
@pytest.mark.parametrize("window", ["context", "query"])
@pytest.mark.usefixtures("matplotlib_backend")
def test_a_posterior_contribution_without_a_descriptor_raises(predicted_model, label, window):
    """A baseline or indirect draw that no descriptor names would be left out of the sum."""
    variable = label.replace(":", "_")
    assert variable in predicted_model.idata[f"/{window}/posterior"].to_dataset(inherit=False)

    def drop(data):
        return data.isel(component=data["component"].to_numpy() != label)

    changed = _edit(predicted_model.idata, f"/{window}/constant_data", drop)
    pattern = rf"/{window}/posterior.*no descriptor.*'{variable}'"
    with pytest.raises(ValueError, match=pattern):
        IdataReader(changed).check_window(window)
    with pytest.raises(ValueError, match=pattern):
        Plots(changed).fit()


def test_a_descriptor_without_posterior_draws_raises(predicted_model):
    changed = _edit(
        predicted_model.idata,
        "/query/posterior",
        lambda data: data.drop_vars("baseline"),
    )
    with pytest.raises(ValueError, match=r"/query/posterior: no variable 'baseline'"):
        IdataReader(changed).check_window("query")
