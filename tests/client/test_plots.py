import pytest

from .client_fixtures import curve_result, status_record

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from pymc_mfm.client.plots import curve_points  # noqa: E402


@pytest.fixture
def done(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/result", (200, curve_result()))
    return client.get("an_1")


def test_curve_points_add_the_unrun_baseline_level(done):
    points = curve_points(done.table)
    tv = points[points["varied_channel"] == "tv"]
    assert tv["level"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert tv.loc[tv["level"] == 1.0, "mean"].item() == 100.0  # the baseline's tv contribution
    assert tv.loc[tv["level"] == 2.0, ["p05", "p95"]].iloc[0].tolist() == [190.0, 210.0]


def test_curve_points_use_the_channels_own_contribution(done):
    points = curve_points(done.table)
    search = points[points["varied_channel"] == "search"]
    assert search.loc[search["level"] == 2.0, "mean"].item() == 100.0


def test_plot_curves_draws_one_panel_per_channel(done):
    figure = done.plot_curves()
    assert isinstance(figure, Figure)
    assert [axis.get_title().split(" ")[0] for axis in figure.axes] == ["tv", "search"]
    line = figure.axes[0].lines[0]
    assert line.get_xdata().tolist() == [0.0, 50.0, 100.0, 150.0, 200.0]
    assert "Conditional contributions, not causal lift" in figure.texts[-1].get_text()


def test_plot_curves_needs_curve_rows(client, service):
    record = curve_result()
    record["table"] = [row for row in record["table"] if row["kind"] != "curve"]
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/result", (200, record))
    with pytest.raises(ValueError, match="curve"):
        client.get("an_1").plot_curves()


def test_plot_compare_shows_mean_differences_with_simulation_error(done):
    figure = done.plot_compare()
    axis = figure.axes[0]
    labels = [label.get_text() for label in axis.get_yticklabels()]
    assert labels == ["half"]  # the failed plan has no difference; curves are left out
    assert "simulation error, not model uncertainty" in figure.texts[-1].get_text()
    assert "interval" not in figure.texts[-1].get_text()


def test_plot_compare_can_include_curve_plans(done):
    figure = done.plot_compare(include_curves=True)
    assert len(figure.axes[0].get_yticklabels()) == 1 + 8


def test_plot_compare_rejects_an_unknown_component(done):
    with pytest.raises(ValueError, match="nope"):
        done.plot_compare(component="nope")


def test_plots_without_matplotlib_name_the_extra(done, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"pymc-mfm\[plot\]"):
        done.plot_curves()
