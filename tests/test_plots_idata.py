"""``Plots(idata)`` and ``model.plot``: charts read the tree alone, live or from a file."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from conftest import CHART_ROLES

import pymc_mfm.plotting as plotting
import pymc_mfm.plotting.charts as charts
from pymc_mfm.model.model import MFM
from pymc_mfm.plotting import Plots
from pymc_mfm.plotting.charts import _backend
from pymc_mfm.plotting.charts._data import IdataReader

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


@pytest.mark.usefixtures("matplotlib_backend")
def test_charts_from_a_file_equal_charts_from_the_model_tree(swept, tmp_path):
    swept, returned = swept
    expected = _draw_all(Plots(swept.idata))
    reread = _saved(swept.idata, tmp_path)
    assert (returned.media > 0).all()
    for tree in (swept.idata, reread):
        curves = IdataReader(tree).response_curves()
        assert curves.channels == returned.channels
        np.testing.assert_array_equal(curves.levels, returned.levels)
        np.testing.assert_allclose(curves.media, returned.media, rtol=1e-12)
        np.testing.assert_array_equal(curves.draws, returned.draws)
    for key, facts in _draw_all(Plots(reread)).items():
        _same(facts, expected[key])
    for key, facts in _draw_all(swept.plot).items():
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


@pytest.mark.usefixtures("matplotlib_backend")
def test_stored_dimension_and_coordinate_order_do_not_change_the_charts(predicted_model):
    import xarray as xr

    tree = predicted_model.idata
    groups = {}
    for node in tree.subtree:
        data = node.to_dataset(inherit=False)
        if node.path in ("/context/constant_data", "/query/constant_data"):
            data = data.assign(
                media=data["media"]
                .transpose("channel", "date")
                .isel(channel=slice(None, None, -1)),
                controls=data["controls"].transpose("control", "date"),
            )
        groups[node.path] = data
    changed = xr.DataTree.from_dict(groups)
    changed.attrs = dict(tree.attrs)
    assert changed["/query/constant_data"]["media"].dims == ("channel", "date")
    expected = {
        key: _facts(getattr(Plots(tree), CHARTS[key][0])(**CHARTS[key][1]))
        for key in ("inputs", "media")
    }
    for key, facts in expected.items():
        method, options = CHARTS[key]
        _same(_facts(getattr(Plots(changed), method)(**options)), facts)


def test_repeated_component_names_raise_and_do_not_hang():
    from pymc_mfm.plotting.charts._data import unique_names

    assert unique_names(["a", "a", "b"], ["x:a", "y:a", "b"]) == ("x:a", "y:a", "b")
    with pytest.raises(ValueError, match="component names must be unique"):
        unique_names(["a", "a"], ["x", "x"])


def test_a_tree_that_is_not_a_model_tree_is_named(predicted_model):
    with pytest.raises(ValueError, match="not a model tree"):
        IdataReader(predicted_model.idata["context"]).window("context")
    with pytest.raises(ValueError, match="window must be"):
        IdataReader(predicted_model.idata).window("both")


@pytest.mark.usefixtures("matplotlib_backend")
def test_a_file_draws_the_charts_with_torch_blocked(swept, tmp_path):
    swept, _ = swept
    swept.idata.to_netcdf(tmp_path / "tree.nc", engine="h5netcdf")
    script = textwrap.dedent(
        """
        import sys

        sys.modules["torch"] = None  # any import of torch now raises ImportError
        import xarray as xr

        from pymc_mfm.plotting import Plots

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
        env={"PATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    return run.stdout.splitlines()


@pytest.mark.parametrize("module", ["pymc_mfm.plotting", "pymc_mfm.plotting.charts"])
def test_model_plots_warns_once_per_call_and_not_on_import(module):
    lines = _run(
        f"""
        class Model:
            plot = object()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from {module} import ModelPlots
        assert not caught
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert ModelPlots(Model()) is Model.plot
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
            from pymc_mfm.plotting.charts import BACKENDS
            from pymc_mfm.plotting.charts import BACKENDS as again
            import pymc_mfm.plotting.charts as charts
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
