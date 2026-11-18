"""``model.response_curves`` and ``model.plot.response_curves``: scenarios, draws and charts."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import CHART_ROLES

import pymc_mfm.inference as inference
from pymc_mfm.inference import api as api_module
from pymc_mfm.inference import curves as curves_module
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.plotting.charts import _backend
from pymc_mfm.plotting.charts import curves as curve_charts
from pymc_mfm.plotting.charts._data import IdataReader
from pymc_mfm.plotting.charts._style import CAVEAT

CHANNELS = list(CHART_ROLES["channel_columns"])


@pytest.fixture(autouse=True)
def matplotlib_default(monkeypatch):
    """Each test starts from, and leaves, the matplotlib default."""
    monkeypatch.setitem(_backend._DEFAULT, "backend", "matplotlib")


@pytest.fixture
def recorded(monkeypatch):
    """Each ``decompose`` call of ``response_curves``: its inputs, options and result."""
    calls = []
    decompose = api_module.decompose

    def record(model, observations, **options):
        result = decompose(model, observations, **options)
        inputs = {name: np.array(values) for name, values in observations.items()}
        calls.append({"observations": inputs, "options": options, "result": result})
        return result

    monkeypatch.setattr(api_module, "decompose", record)
    return calls


@pytest.fixture
def forecast_model(model, chart_frames):
    """``model`` after cheap complete 2-draw inference on the history/forecast pair."""
    history, future = chart_frames
    torch.manual_seed(3)
    model.inference(history, future, n_samples=2, **CHART_ROLES)
    return model


@pytest.fixture
def swept(forecast_model):
    """The model after a seeded sweep over two forecast weeks, and its curves."""
    torch.manual_seed(2)
    curves = forecast_model.response_curves(levels=[0, 0.5, 1, 2], weeks=2, n_samples=4)
    return forecast_model, curves


#: Known contributions of tv at three of the swept levels, as from a simulation.
TRUTH = pd.DataFrame({"tv": [40.0, 20.0, 30.0]}, index=[1.0, 0.0, 0.5])


def test_inference_exports_are_exactly_the_documented_names():
    assert sorted(inference.__all__) == [
        "DecompositionResult",
        "ResponseCurves",
        "WarmUp",
        "decompose",
        "from_pretrained",
        "list_profiles",
        "observation_batch",
        "push_to_hub",
        "save_pretrained",
        "warm_up_observations",
        "warm_up_sampling",
    ]
    for name in inference.__all__:
        assert getattr(inference, name).__name__ == name
    assert inference.ResponseCurves is curves_module.ResponseCurves


def _channel_totals(result, name):
    """Draws of channel ``name``'s contribution summed over the weeks of ``result``."""
    return result.components[:, :, result.component_names.index(f"channel:{name}")].sum(axis=1)


def _summary(frame, name):
    rows = frame[frame["channel"] == name]
    return rows["mean"].to_numpy(), rows["low"].to_numpy(), rows["high"].to_numpy()


def _points(collection):
    """The points a matplotlib scatter draws; points with a missing value are not drawn."""
    offsets = collection.get_offsets()
    return np.asarray(offsets)[~np.ma.getmaskarray(offsets).any(axis=1)]


def test_each_scenario_scales_one_channel_in_the_varied_weeks(
    predicted_model, chart_frames, recorded
):
    history, future = chart_frames
    torch.manual_seed(0)
    curves = predicted_model.response_curves(levels=[2, 0, 1, 0], weeks=2, n_samples=8)
    np.testing.assert_array_equal(curves.levels, [0, 1, 2])
    assert curves.channels == tuple(CHANNELS) and curves.draws.shape == (2, 3, 8)
    assert (curves.weeks, curves.n_samples) == (2, 8)
    assert curves.units == "outcome"
    np.testing.assert_array_equal(curves.media, future[CHANNELS].iloc[:2].sum())
    np.testing.assert_array_equal(curves.dates, future["date"].iloc[:2].to_numpy())
    frame = curves.to_frame()
    low, high = np.quantile(curves.draws, [0.05, 0.95], axis=2)
    np.testing.assert_allclose(frame["mean"], curves.draws.mean(axis=2).ravel())
    np.testing.assert_allclose(frame[["low", "high"]], np.column_stack((low.ravel(), high.ravel())))
    # Inputs stop after the varied weeks; history, controls and options stay as inferred.
    weeks = pd.concat([history, future.iloc[:2]], ignore_index=True)
    controls = np.zeros((26, 2))
    controls[:, 0] = weeks["temperature"]
    for call in recorded:
        np.testing.assert_array_equal(call["observations"]["context_outcome"], history["sales"])
        np.testing.assert_array_equal(call["observations"]["controls"], controls)
        np.testing.assert_array_equal(call["options"]["time"], weeks["date"].to_numpy())
        assert call["options"]["n_samples"] == 8 and call["options"]["window"] == "query"
        assert call["options"]["mode"] == "model_average" and call["options"]["graph"] is None
    # One unchanged run is level 1 of both channels; every other run scales one
    # media covariate in the varied weeks only.
    actual = weeks[CHANNELS].to_numpy()
    reference, *scenarios = recorded
    np.testing.assert_array_equal(reference["observations"]["spend"], actual)
    varied = [(0, 0.0), (0, 2.0), (1, 0.0), (1, 2.0)]
    assert len(scenarios) == len(varied)
    for call, (column, level) in zip(scenarios, varied, strict=True):
        expected = actual.copy()
        expected[24:, column] *= level
        np.testing.assert_array_equal(call["observations"]["spend"], expected)
    for row, name in enumerate(CHANNELS):
        np.testing.assert_array_equal(
            curves.draws[row, 1], _channel_totals(reference["result"], name)
        )
        for index, call in zip((0, 2), scenarios[2 * row : 2 * row + 2], strict=True):
            np.testing.assert_array_equal(
                curves.draws[row, index], _channel_totals(call["result"], name)
            )


def test_defaults_vary_every_channel_over_every_forecast_week(predicted_model, recorded):
    curves = predicted_model.response_curves(levels=[1, 2])
    assert len(recorded) == 3  # the shared level-1 run, then each channel at level 2
    assert curves.channels == tuple(CHANNELS) and curves.weeks == 6 and curves.n_samples == 40
    assert curves.draws.shape == (2, 2, 40)
    curves = predicted_model.response_curves(channels=["tv"], weeks=1, n_samples=2)
    np.testing.assert_array_equal(curves.levels, np.linspace(0, 2, 9))
    assert curves.channels == ("tv",) and curves.draws.shape == (1, 9, 2)


def test_a_media_covariate_at_zero_in_the_weeks_gets_the_unchanged_run(
    predicted_model, chart_frames, recorded
):
    # tv is zero in the first forecast week: scaling it changes no input.
    curves = predicted_model.response_curves(levels=[0, 0.5, 2], weeks=1)
    assert curves.media[1] == 0
    assert len(recorded) == 4  # search at each level, then one unchanged run for tv
    history, future = chart_frames
    actual = pd.concat([history, future.iloc[:1]], ignore_index=True)[CHANNELS].to_numpy()
    np.testing.assert_array_equal(recorded[-1]["observations"]["spend"], actual)
    unchanged = _channel_totals(recorded[-1]["result"], "tv")
    for level in range(3):
        np.testing.assert_array_equal(curves.draws[1, level], unchanged)


def test_to_frame_summarizes_the_draws_of_each_channel_and_level(forecast_model):
    torch.manual_seed(1)
    curves = forecast_model.response_curves(levels=[0.5, 1], weeks=3, n_samples=6)
    frame = curves.to_frame(0.8)
    assert list(frame.columns) == ["channel", "level", "media", "mean", "low", "high"]
    assert frame["channel"].tolist() == ["search", "search", "tv", "tv"]
    np.testing.assert_array_equal(frame["level"], [0.5, 1, 0.5, 1])
    np.testing.assert_allclose(frame["media"], np.outer(curves.media, [0.5, 1]).ravel())
    low, high = np.quantile(curves.draws, [0.1, 0.9], axis=2)
    np.testing.assert_allclose(frame["mean"], curves.draws.mean(axis=2).ravel())
    np.testing.assert_allclose(frame["low"], low.ravel())
    np.testing.assert_allclose(frame["high"], high.ravel())


def test_a_seed_repeats_the_draws(forecast_model):
    torch.manual_seed(5)
    first = forecast_model.response_curves(levels=[0, 1], weeks=1, n_samples=4)
    torch.manual_seed(5)
    second = forecast_model.response_curves(levels=[0, 1], weeks=1, n_samples=4)
    np.testing.assert_array_equal(first.draws, second.draws)


def test_random_seed_repeats_the_draws_and_is_recorded_in_the_curve_attrs(forecast_model):
    options = {"levels": [0, 1], "weeks": 1, "n_samples": 4}
    first = forecast_model.response_curves(**options, random_seed=5)
    held = forecast_model.idata
    second = forecast_model.response_curves(**options, random_seed=5)
    np.testing.assert_array_equal(first.draws, second.draws)
    assert held["response_curves"].attrs["random_seed"] == 5
    forecast_model.response_curves(**options, random_seed=9)
    assert forecast_model.idata["response_curves"].attrs["random_seed"] == 9
    assert held["response_curves"].attrs["random_seed"] == 5
    forecast_model.response_curves(**options)
    assert "random_seed" not in forecast_model.idata["response_curves"].attrs
    assert "random_seed" not in forecast_model.idata.attrs


def test_no_seed_keeps_the_callers_generator(forecast_model):
    options = {"levels": [0, 1], "weeks": 1, "n_samples": 4}
    torch.manual_seed(3)
    unseeded = forecast_model.response_curves(**options)
    seeded = forecast_model.response_curves(**options, random_seed=3)
    np.testing.assert_array_equal(unseeded.draws, seeded.draws)


@pytest.mark.parametrize("mode", ["model_average", "map", "fixed"])
def test_a_seed_equals_manual_seed_before_the_sweep(model, chart_frames, mode):
    history, future = chart_frames
    inference = {"n_samples": 3, "mode": mode}
    if mode == "fixed":
        layout = EdgeSlotLayout(2, model.config.n_controls, model.config.n_demand)
        graphs = np.zeros((3, 1, layout.n_slots))
        graphs[:, 0, layout.slices["cy"].start] = 1
        inference["graph"] = graphs
    elif mode == "map":
        inference["router_samples"] = 5
    model.inference(history, future, **CHART_ROLES, **inference)
    forecast_model = model
    options = {"levels": [0, 1, 2], "weeks": 1, "n_samples": 3}
    torch.manual_seed(0)
    expected = forecast_model.response_curves(**options)
    expected_state = torch.get_rng_state()
    torch.manual_seed(99)
    seeded = forecast_model.response_curves(**options, random_seed=0)
    assert torch.equal(torch.get_rng_state(), expected_state)
    np.testing.assert_array_equal(seeded.draws, expected.draws)


def test_a_failed_seeded_sweep_does_not_restore_the_generator(forecast_model, monkeypatch):
    decompose = api_module.decompose
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(kwargs["random_seed"])
        if len(calls) == 2:
            raise RuntimeError("sampling failed")
        return decompose(*args, **kwargs)

    monkeypatch.setattr(api_module, "decompose", fail_second)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="sampling failed"):
        forecast_model.response_curves(levels=[0, 1, 2], weeks=1, n_samples=4, random_seed=7)
    after = torch.get_rng_state()
    assert calls == [7, None]
    assert not torch.equal(after, before)
    torch.manual_seed(7)
    assert not torch.equal(torch.get_rng_state(), after)


@pytest.mark.parametrize("seed", [-1, 2**31, True, 1.5])
def test_invalid_seeds_fail_before_sampling(forecast_model, seed):
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="random_seed"):
        forecast_model.response_curves(levels=[0, 1], random_seed=seed)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("mode", ["fixed", "map"])
def test_scenarios_keep_the_mode_graphs_units_and_provenance_of_inference(
    model, chart_frames, recorded, mode
):
    history, future = chart_frames
    options = {"n_samples": 3, "mode": mode, "units": "USD", "provenance": {"scenario": "plan"}}
    if mode == "fixed":
        layout = EdgeSlotLayout(2, model.config.n_controls, model.config.n_demand)
        graphs = np.zeros((3, 1, layout.n_slots))
        graphs[0, 0, layout.slices["cy"].start] = 1
        graphs[1, 0, layout.slices["cy"].start + 1] = 1
        options["graph"] = graphs
    else:
        options["router_samples"] = 7
    model.inference(history, future, **CHART_ROLES, **options)
    recorded.clear()  # the patch also saw inference()
    query = model.query_result
    curves = model.response_curves(levels=[0, 1], weeks=1)
    # tv is zero in the first forecast week, so it reuses the unchanged run.
    assert curves.units == "USD" and curves.n_samples == 3 and len(recorded) == 2
    for call in recorded:
        used = call["options"]
        assert (used["mode"], used["units"], used["n_samples"]) == (mode, "USD", 3)
        assert call["result"].provenance["scenario"] == "plan"
        if mode == "fixed":
            np.testing.assert_array_equal(call["result"].graphs, query.graphs)
        else:
            assert call["result"].provenance["inference_config"]["router_samples"] == 7
    if mode == "fixed":
        with pytest.raises(ValueError, match="n_samples must be 3"):
            model.response_curves(n_samples=4)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"levels": [0, -0.5]}, "nonnegative"),
        ({"levels": [0, np.nan]}, "finite"),
        ({"levels": [1, np.inf]}, "finite"),
        ({"levels": []}, "nonempty"),
        ({"levels": [[0, 1]]}, "nonempty"),
        ({"levels": ["1"]}, "nonempty"),
        ({"levels": [1, 1.0]}, "at least two distinct levels"),
        ({"channels": ["radio"]}, r"unknown channels \['radio'\]"),
        ({"channels": []}, "at least one of"),
        ({"channels": "tv"}, "not a string"),
        ({"channels": ["tv", "tv"]}, "repeat"),
        ({"weeks": 0}, "from 1 to 6"),
        ({"weeks": 7}, "from 1 to 6"),
        ({"weeks": 1.5}, "from 1 to 6"),
        ({"weeks": True}, "from 1 to 6"),
        ({"n_samples": 0}, "positive integer"),
    ],
)
def test_invalid_arguments_fail_before_sampling(forecast_model, options, message):
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match=message):
        forecast_model.response_curves(**options)
    assert torch.equal(torch.get_rng_state(), state)


def test_curves_follow_the_latest_complete_inference(model, chart_frames):
    history, future = chart_frames
    with pytest.raises(RuntimeError, match="inference"):
        model.response_curves()
    with pytest.raises(RuntimeError, match=r"inference\(history, future, \.\.\.\)"):
        IdataReader(model.idata).response_curves()
    model.inference(history, future, n_samples=2, **CHART_ROLES)
    original_tree, query = model.idata, model.query_result
    with pytest.raises(RuntimeError, match=r"model\.response_curves\(\.\.\.\) first"):
        IdataReader(model.idata).response_curves()
    curves = model.response_curves(levels=[0, 1], weeks=1)
    tree = model.idata
    assert tree is not original_tree and "response_curves" not in original_tree.children
    with pytest.raises(ValueError, match="nonnegative"):
        model.response_curves(levels=[-1])
    with pytest.raises(ValueError, match="search"):
        model.inference(history, future.drop(columns="search"), n_samples=2, **CHART_ROLES)
    assert model.idata is tree and model.query_result is query
    np.testing.assert_array_equal(IdataReader(model.idata).response_curves().draws, curves.draws)
    model.predict(future.iloc[:2].assign(search=1.0), n_samples=3, mode="map", units="USD")
    assert model.idata is tree and model.query_result is query
    np.testing.assert_array_equal(IdataReader(model.idata).response_curves().draws, curves.draws)
    np.testing.assert_array_equal(
        curve_charts.prepare(IdataReader(model.idata)).mean, curves.draws.mean(axis=2)
    )
    model.inference(history, future.iloc[:1], n_samples=2, **CHART_ROLES)
    assert model.idata is not tree and "response_curves" not in model.idata.children
    assert "response_curves" in tree.children
    with pytest.raises(RuntimeError, match=r"model\.response_curves"):
        IdataReader(model.idata).response_curves()


def test_curves_and_the_complete_tree_share_the_exact_draw_bank(forecast_model):
    import xarray as xr

    base_tree = forecast_model.idata
    query = forecast_model.query_result
    curves = forecast_model.response_curves(levels=[0, 0.5, 1, 2], weeks=2, n_samples=5)
    tree = forecast_model.idata
    assert tree is not base_tree and "response_curves" not in base_tree.children
    assert forecast_model.query_result is query and forecast_model.idata is tree
    assert tree.attrs["inference_revision"] == base_tree.attrs["inference_revision"]
    assert (
        tree["response_curves"].attrs["source_inference_revision"]
        == base_tree.attrs["inference_revision"]
    )
    for name in ("context", "query"):
        xr.testing.assert_identical(tree[name], base_tree[name])
        assert tree[f"{name}/posterior"].sizes["draw"] == 2
    posterior = tree["response_curves/posterior"]
    assert posterior["contribution"].dims == ("chain", "draw", "channel", "level")
    np.testing.assert_array_equal(
        posterior["contribution"].values[0], curves.draws.transpose(2, 0, 1)
    )
    assert posterior.sizes["draw"] == 5
    np.testing.assert_array_equal(posterior["channel"], curves.channels)
    np.testing.assert_array_equal(posterior["level"], curves.levels)
    constants = tree["response_curves/constant_data"]
    np.testing.assert_array_equal(constants["media"], curves.media[:, None] * curves.levels)
    np.testing.assert_array_equal(constants["date"], curves.dates)
    state = torch.get_rng_state()
    displayed = curve_charts.prepare(IdataReader(forecast_model.idata), interval=0.8)
    low, high = np.quantile(curves.draws, [0.1, 0.9], axis=2)
    np.testing.assert_array_equal(displayed.mean, curves.draws.mean(axis=2))
    np.testing.assert_allclose(displayed.low, low)
    np.testing.assert_allclose(displayed.high, high)
    assert torch.equal(torch.get_rng_state(), state)
    assert forecast_model.idata is tree


def test_a_new_sweep_replaces_its_entire_subtree_without_rewriting_held_trees(forecast_model):
    import xarray as xr

    query = forecast_model.query_result
    first = forecast_model.response_curves(levels=[0, 0.5, 1, 2], weeks=3, n_samples=5)
    first_tree = forecast_model.idata
    snapshot = first_tree.copy(deep=True)
    second = forecast_model.response_curves(channels=["tv"], levels=[0, 1], weeks=1, n_samples=3)
    tree = forecast_model.idata
    assert tree is not first_tree
    np.testing.assert_array_equal(
        IdataReader(forecast_model.idata).response_curves().draws, second.draws
    )
    assert forecast_model.query_result is query
    posterior = tree["response_curves/posterior"]
    assert dict(posterior.sizes) == {"chain": 1, "draw": 3, "channel": 1, "level": 2}
    np.testing.assert_array_equal(posterior["channel"], ["tv"])
    np.testing.assert_array_equal(posterior["level"], [0, 1])
    np.testing.assert_array_equal(
        posterior["contribution"].values[0], second.draws.transpose(2, 0, 1)
    )
    np.testing.assert_array_equal(tree["response_curves/constant_data"]["date"], second.dates)
    assert first.draws.shape == (2, 4, 5)
    xr.testing.assert_identical(first_tree, snapshot)
    for name in ("context", "query"):
        xr.testing.assert_identical(tree[name], first_tree[name])


@pytest.mark.parametrize("keep_curves", [False, True], ids=["first-sweep", "replacement"])
@pytest.mark.parametrize("failure", ["sampling", "conversion"])
def test_failed_sweeps_preserve_the_entire_complete_inference(
    forecast_model, monkeypatch, keep_curves, failure
):
    import xarray as xr

    from pymc_mfm.inference import _datatree

    curves = forecast_model.response_curves(levels=[0, 1], weeks=1) if keep_curves else None
    tree = forecast_model.idata
    snapshot = tree.copy(deep=True)
    query = forecast_model.query_result
    if failure == "sampling":
        decompose = api_module.decompose
        calls = 0

        def fail_mid_sweep(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = decompose(*args, **kwargs)
            if calls == 2:
                raise RuntimeError("curve sampling failed")
            return result

        monkeypatch.setattr(api_module, "decompose", fail_mid_sweep)
    else:

        def fail_conversion(*args, **kwargs):
            raise RuntimeError("curve conversion failed")

        monkeypatch.setattr(_datatree, "attach_response_curves", fail_conversion)
    with pytest.raises(RuntimeError, match=f"curve {failure} failed"):
        forecast_model.response_curves(levels=[0, 0.5, 1, 2], weeks=2, n_samples=3)
    assert forecast_model.idata is tree and forecast_model.query_result is query
    xr.testing.assert_identical(tree, snapshot)
    if keep_curves:
        np.testing.assert_array_equal(
            IdataReader(forecast_model.idata).response_curves().draws, curves.draws
        )
    else:
        assert "response_curves" not in tree.children
        with pytest.raises(RuntimeError, match=r"model\.response_curves"):
            IdataReader(forecast_model.idata).response_curves()


def test_sweeps_ignore_independent_predictions(forecast_model, chart_frames, recorded):
    history, future = chart_frames
    query = forecast_model.query_result
    independent = forecast_model.predict(
        future.iloc[:1].assign(search=1.0), n_samples=3, mode="map", units="USD"
    )
    assert independent is not query
    recorded.clear()  # the patch also saw predict()
    curves = forecast_model.response_curves(levels=[1, 2], weeks=2)
    assert curves.weeks == 2 and curves.n_samples == 2 and curves.units == "outcome"
    np.testing.assert_array_equal(curves.media, future[CHANNELS].iloc[:2].sum())
    np.testing.assert_array_equal(
        recorded[0]["observations"]["spend"],
        pd.concat([history, future.iloc[:2]], ignore_index=True)[CHANNELS],
    )
    assert all(call["options"]["mode"] == "model_average" for call in recorded)
    assert forecast_model.query_result is query


@pytest.mark.usefixtures("matplotlib_backend")
def test_matplotlib_draws_the_frame_of_each_channel(swept):
    model, curves = swept
    tree, state = model.idata, torch.get_rng_state()
    fig = model.plot.response_curves(interval=0.8, truth=TRUTH)
    assert model.idata is tree and torch.equal(torch.get_rng_state(), state)
    frame = curves.to_frame(0.8)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == ["search · media covariate total 97.5", "tv · media covariate total 87.1"]
    levels = [0, 0.5, 1, 2]
    for ax, name in zip(fig.axes, CHANNELS, strict=True):
        mean, low, high = _summary(frame, name)
        mean_line, actual_line = ax.lines
        np.testing.assert_array_equal(mean_line.get_xdata(), levels)
        np.testing.assert_allclose(mean_line.get_ydata(), mean)
        # The band runs right along its low edge, then back along its high edge.
        vertices = ax.collections[0].get_paths()[0].vertices
        np.testing.assert_allclose(vertices[1:5], np.column_stack((levels, low)))
        np.testing.assert_allclose(vertices[6:10][::-1], np.column_stack((levels, high)))
        np.testing.assert_array_equal(actual_line.get_xdata(), [1, 1])
        assert actual_line.get_linestyle() == "--"
        np.testing.assert_allclose(_points(ax.collections[1]), [[1, mean[2]]])
    # Truth dots only where the truth has a value: none for search, none at level 2.
    assert _points(fig.axes[0].collections[2]).size == 0
    np.testing.assert_array_equal(
        _points(fig.axes[1].collections[2]), [[0, 20], [0.5, 30], [1, 40]]
    )
    text = " ".join(fig.get_suptitle().split())
    assert text.startswith("Response curves: contribution by media covariate level")
    assert "80% pointwise intervals over 4 draws" in text and CAVEAT in text
    assert "media covariate varied separately over a plan of the first 2 forecast weeks;" in text


def test_plotly_draws_the_same_numbers(swept):
    pytest.importorskip("plotly")
    model, curves = swept
    fig = model.plot.response_curves(truth=TRUTH, backend="plotly", title="Sweep")
    frame = curves.to_frame(0.9)
    means = [trace for trace in fig.data if trace.name == "Mean"]
    edges = [trace for trace in fig.data if trace.legendgroup == "interval"]
    dots = [trace for trace in fig.data if trace.legendgroup == "mean" and trace.mode == "markers"]
    truths = [trace for trace in fig.data if trace.name == "Truth"]
    for index, name in enumerate(CHANNELS):
        mean, low, high = _summary(frame, name)
        np.testing.assert_array_equal(means[index].x, [0, 0.5, 1, 2])
        np.testing.assert_allclose(means[index].y, mean)
        np.testing.assert_allclose(edges[2 * index].y, high)
        np.testing.assert_allclose(edges[2 * index + 1].y, low)
        np.testing.assert_allclose(dots[index].y, [mean[2]])
        rows = frame[frame["channel"] == name]
        np.testing.assert_allclose(means[index].customdata, rows[["media", "low", "high"]])
        assert f"{name} at %{{x:.1%}} of its observed level" in means[index].hovertemplate
    np.testing.assert_array_equal(truths[0].y, [np.nan] * 4)
    np.testing.assert_array_equal(truths[1].y, [20, 30, 40, np.nan])
    assert [shape.x0 for shape in fig.layout.shapes] == [1.0, 1.0]
    assert [shape.name for shape in fig.layout.shapes] == ["Observed level (100%)"] * 2
    assert fig.layout.title.text.startswith("Sweep<br>")
    assert "varied separately over a plan of the first 2 forecast weeks;" in fig.layout.title.text


@pytest.mark.usefixtures("matplotlib_backend")
def test_panels_say_when_a_covariate_is_zero_and_skip_the_missing_level_1(forecast_model):
    # tv is zero in the first forecast week; level 1 is not swept, so no dot.
    forecast_model.response_curves(levels=[0, 2], weeks=1)
    fig = forecast_model.plot.response_curves(channels=["tv"])
    [ax] = fig.axes
    assert ax.get_title(loc="left") == "tv · zero in these weeks"
    assert len(ax.collections) == 1  # the band only
    assert "over a one-week plan;" in " ".join(fig.get_suptitle().split())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"channels": ["radio"]}, r"unknown channels \['radio'\]; choose from \['search', 'tv'\]"),
        ({"interval": 1.5}, "interval"),
        ({"truth": {"tv": [1.0]}}, "DataFrame"),
        ({"truth": pd.DataFrame({"radio": [1.0]}, index=[1.0])}, "distinct channels"),
        ({"truth": pd.DataFrame({"tv": [1.0]}, index=[0.75])}, r"levels \[0\.75\]"),
        ({"truth": pd.DataFrame({"tv": [1.0, 2.0]}, index=[1.0, 1.0])}, "each level once"),
        ({"truth": pd.DataFrame({"tv": ["high"]}, index=[1.0])}, "numbers"),
    ],
)
def test_chart_arguments_are_checked(forecast_model, options, message):
    forecast_model.response_curves(levels=[0, 1], weeks=1)
    with pytest.raises(ValueError, match=message):
        curve_charts.prepare(IdataReader(forecast_model.idata), **options)
