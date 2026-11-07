import pytest

from .client_fixtures import curve_result, result_rows, status_record

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


def _done_with_rows(client, service, rows_record):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/result", (200, rows_record))
    return client.get("an_1")


def test_every_level_of_every_channel_is_kept_without_scenarios(client, service):
    record = curve_result()
    record["table"] = [row for row in record["table"] if row["kind"] != "scenario"]
    points = curve_points(_done_with_rows(client, service, record).table)
    for channel in ("tv", "search"):
        levels = points.loc[points["varied_channel"] == channel, "level"].tolist()
        assert levels == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert points["mean"].notna().all()


def _bar_count(axis):
    return sum(len(segments.get_segments()) for segments in axis.containers[0].lines[2])


def _two_scenarios(se_of_second):
    record = curve_result()
    record["table"] += result_rows("double", "scenario", base="baseline", tv_mean=200.0)
    for row in record["table"]:
        if row["scenario_id"] == "double":
            row["simulation_se"] = se_of_second
    return record


def test_plot_compare_omits_bars_where_simulation_error_is_missing(client, service):
    done = _done_with_rows(client, service, _two_scenarios(None))
    axis = done.plot_compare().axes[0]
    labels = [label.get_text() for label in axis.get_yticklabels()]
    assert "double (simulation error unavailable)" in labels
    assert "half" in labels  # its own label is unchanged
    assert _bar_count(axis) == 1  # only "half" has a bar
    assert len(axis.lines) >= 2  # the open marker for "double" is still drawn


def test_plot_compare_draws_a_bar_for_every_row_with_simulation_error(client, service):
    done = _done_with_rows(client, service, _two_scenarios(0.25))
    axis = done.plot_compare().axes[0]
    assert _bar_count(axis) == 2
    assert not any("unavailable" in label.get_text() for label in axis.get_yticklabels())


WARNING = "spend above the history's 95th percentile"


def _warned_curve_result():
    record = curve_result()
    for row in record["table"]:
        if row["scenario_id"] == "curve:tv:baseline:2":
            row["warnings"] = [WARNING]
        if row["scenario_id"] == "half":
            row["warnings"] = ["half has a warning"]
    return record


def _footnote(figure):
    return figure._supxlabel.get_text()


def test_curve_points_carry_the_plan_warnings(client, service):
    points = curve_points(_done_with_rows(client, service, _warned_curve_result()).table)
    flagged = points[points["warnings"].map(len) > 0]
    assert flagged[["varied_channel", "level"]].values.tolist() == [["tv", 2.0]]


def test_plot_curves_marks_levels_with_warnings_and_prints_them(client, service):
    figure = _done_with_rows(client, service, _warned_curve_result()).plot_curves()
    tv_axis, search_axis = figure.axes[:2]
    assert len(tv_axis.collections) == 2  # the band and the warning markers
    assert tv_axis.collections[-1].get_offsets().tolist() == [[200.0, 200.0]]
    assert len(search_axis.collections) == 1  # no warning there
    assert WARNING in _footnote(figure)


def test_plot_curves_without_warnings_adds_no_warning_text(done):
    assert "Warning" not in _footnote(done.plot_curves())


def test_plot_compare_marks_a_plan_with_warnings(client, service):
    figure = _done_with_rows(client, service, _warned_curve_result()).plot_compare()
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert "half *" in labels
    assert "half has a warning" in _footnote(figure)
