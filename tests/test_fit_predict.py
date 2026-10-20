"""The dataframe facade retains context, not weights, and uses canonical inference."""

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file

from pymc_mfm.inference.api import decompose
from pymc_mfm.inference.checkpoint import save_pretrained
from pymc_mfm.inference.hub import from_pretrained
from pymc_mfm.inference.result import DecompositionResult
from pymc_mfm.model.graph import EdgeSlotLayout

CHANNELS = ("search", "tv")
CONTROLS = ("temperature",)
COLUMN_ROLES = {
    "date_column": "date",
    "target_column": "sales",
    "channel_columns": CHANNELS,
    "control_columns": CONTROLS,
}


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
            "sales": [100.0, 125.0, 110.0, 140.0, 135.0],
            "search": [20.0, 35.0, 25.0, 45.0, 30.0],
            "tv": [80.0, 50.0, 65.0, 40.0, 55.0],
            "temperature": [-2.0, 1.0, 3.0, -1.0, 4.0],
        }
    )


@pytest.fixture
def future():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-02-05", periods=3, freq="W-MON"),
            "search": [60.0, 90.0, 15.0],
            "tv": [35.0, 75.0, 100.0],
            "temperature": [7.0, -3.0, 2.0],
        }
    )


def _canonical_result(
    model,
    history,
    future,
    control_names,
    *,
    channel_columns=CHANNELS,
    control_columns=CONTROLS,
    **kwargs,
):
    """Build the reference independently of the dataframe adapter, including padding."""
    history, future = history.sort_values("date"), future.sort_values("date")
    combined = pd.concat((history, future), ignore_index=True)
    controls = np.zeros((len(combined), model.config.n_controls), dtype=np.float64)
    controls[:, : len(control_columns)] = combined[list(control_columns)].to_numpy(dtype=np.float64)
    return decompose(
        model,
        {
            "spend": combined[list(channel_columns)].to_numpy(dtype=np.float64),
            "controls": controls,
            "context_outcome": history["sales"].to_numpy(dtype=np.float64),
            "channel_mask": np.ones(len(channel_columns), dtype=bool),
            "control_mask": np.arange(model.config.n_controls) < len(control_columns),
            "latent_mask": np.ones(model.config.n_demand, dtype=bool),
        },
        channel_names=channel_columns,
        control_names=control_names,
        time=pd.to_datetime(combined["date"]).to_numpy(dtype="datetime64[ns]"),
        **kwargs,
    )


def _assert_same_result(actual, expected):
    for name in ("components", "graphs", "time", "component_mask", "edge_mask"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in (
        "component_names",
        "component_families",
        "edge_names",
        "channel_names",
        "control_names",
        "window",
        "mode",
        "units",
        "context_length",
    ):
        assert getattr(actual, name) == getattr(expected, name), name
    for name in ("data_sha256", "sales_scale"):
        assert actual.provenance[name] == expected.provenance[name], name


def test_fit_chains_and_predict_matches_canonical_decompose_with_sorted_alignment(
    model, history, future
):
    history_input = history.iloc[[3, 0, 4, 1, 2]][
        ["temperature", "date", "sales", "tv", "search"]
    ].copy()
    future_input = future.iloc[[2, 0, 1]][["tv", "temperature", "search", "date"]].copy()
    history_input["date"] = history_input["date"].dt.strftime("%Y-%m-%d")
    future_input["date"] = future_input["date"].dt.strftime("%Y-%m-%d")
    history_input["unrelated"] = "private note"
    future_input["unrelated"] = None
    original_history, original_future = history_input.copy(deep=True), future_input.copy(deep=True)
    before = torch.get_rng_state()
    assert model.fit(history_input, **COLUMN_ROLES) is model
    assert torch.equal(torch.get_rng_state(), before)

    torch.manual_seed(701)
    actual = model.predict(future_input, n_samples=4)
    torch.manual_seed(701)
    expected = _canonical_result(model, history, future, actual.control_names, n_samples=4)
    _assert_same_result(actual, expected)
    assert isinstance(actual, DecompositionResult)
    assert actual.channel_names == CHANNELS
    assert actual.control_names[: len(CONTROLS)] == CONTROLS
    assert len(actual.control_names) == model.config.n_controls
    assert len(set(actual.control_names)) == len(actual.control_names)
    control_columns = np.array([family == "control" for family in actual.component_families])
    np.testing.assert_array_equal(actual.component_mask[control_columns], [True, False])
    np.testing.assert_array_equal(actual.time, future["date"].to_numpy(dtype="datetime64[ns]"))
    assert actual.time.dtype == np.dtype("datetime64[ns]")
    assert actual.window == "query" and actual.provenance["head"] == "conditional"
    pd.testing.assert_frame_equal(history_input, original_history)
    pd.testing.assert_frame_equal(future_input, original_future)


@pytest.mark.parametrize("mode", ["fixed", "map"])
def test_predict_preserves_graph_options_units_and_provenance(model, history, future, mode):
    model.fit(history, **COLUMN_ROLES)
    options = {
        "n_samples": 3,
        "mode": mode,
        "units": "USD",
        "provenance": {"scenario": "board forecast"},
    }
    layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
    graph = np.zeros(layout.n_slots)
    graph[layout.slices["cy"].start] = 1
    if mode == "fixed":
        options["graph"] = graph
    else:
        options["router_samples"] = 7
    torch.manual_seed(702)
    actual = model.predict(future, **options)
    torch.manual_seed(702)
    expected = _canonical_result(model, history, future, actual.control_names, **options)
    _assert_same_result(actual, expected)
    assert actual.n_samples == 3 and actual.mode == mode and actual.units == "USD"
    assert actual.provenance["scenario"] == "board forecast"
    if mode == "fixed":
        np.testing.assert_array_equal(actual.graphs, np.broadcast_to(graph, actual.graphs.shape))
        assert actual.provenance["graph_source"] == "supplied_hypothesis"
    else:
        assert actual.provenance["inference_config"]["router_samples"] == 7


@pytest.mark.parametrize(
    "outcomes",
    [
        pytest.param(["not observed", None, -np.inf], id="nonnumeric"),
        pytest.param([1e100, 1e-100, 1e200], id="finite-numeric"),
    ],
)
def test_query_outcomes_are_ignored(model, history, future, outcomes):
    model.fit(history, **COLUMN_ROLES)
    torch.manual_seed(703)
    expected = model.predict(future, n_samples=3)
    poisoned = future.assign(sales=outcomes)
    torch.manual_seed(703)
    actual = model.predict(poisoned, n_samples=3)
    _assert_same_result(actual, expected)
    assert actual.provenance == expected.provenance
    assert actual.provenance["sales_scale"] == np.quantile(history["sales"], 0.95)


def test_history_and_role_lists_are_snapshots_and_refit_is_atomic(model, history, future):
    pristine = history.copy(deep=True)
    channels, controls = list(CHANNELS), list(CONTROLS)
    model.fit(
        history,
        date_column="date",
        target_column="sales",
        channel_columns=channels,
        control_columns=controls,
    )
    torch.manual_seed(704)
    expected = model.predict(future, n_samples=3)
    history["date"] = "not a date"
    history["sales"] = 0.0
    history.loc[:, [*CHANNELS, *CONTROLS]] = -999.0
    channels[:] = ["missing"]
    controls.clear()
    torch.manual_seed(704)
    _assert_same_result(model.predict(future, n_samples=3), expected)

    invalid_refit = pristine.rename(columns={"search": "new_search"}).assign(sales=0.0)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.fit(invalid_refit, **{**COLUMN_ROLES, "channel_columns": ("new_search", "tv")})
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(704)
    _assert_same_result(model.predict(future, n_samples=3), expected)

    replacement = pristine.assign(
        sales=pristine["sales"] * [1.0, 1.8, 0.7, 2.3, 1.2],
        search=pristine["search"] + [4.0, 15.0, 1.0, 20.0, 7.0],
        temperature=pristine["temperature"] * [-1.0, 2.0, 0.5, 3.0, -2.0],
    )
    assert model.fit(replacement, **COLUMN_ROLES) is model
    torch.manual_seed(704)
    actual = model.predict(future, n_samples=3)
    torch.manual_seed(704)
    reference = _canonical_result(model, replacement, future, actual.control_names, n_samples=3)
    _assert_same_result(actual, reference)
    assert actual.provenance["sales_scale"] == np.quantile(replacement["sales"], 0.95)
    assert not np.array_equal(actual.components, expected.components)


def test_queries_are_independent_and_rebuild_relative_time_for_each_horizon(model, history, future):
    model.fit(history, **COLUMN_ROLES)
    short = future.iloc[:1].copy()
    torch.manual_seed(705)
    expected_short = model.predict(short, n_samples=3)
    different_query = future.assign(search=future["search"] * 4)
    torch.manual_seed(706)
    actual_long = model.predict(different_query, n_samples=3)
    torch.manual_seed(706)
    expected_long = _canonical_result(
        model, history, different_query, actual_long.control_names, n_samples=3
    )
    _assert_same_result(actual_long, expected_long)
    torch.manual_seed(705)
    _assert_same_result(model.predict(short, n_samples=3), expected_short)


def test_padding_names_do_not_collide_with_real_controls(model, history, future):
    model.fit(history, **COLUMN_ROLES)
    torch.manual_seed(707)
    initial = model.predict(future, n_samples=2)
    real_name = initial.control_names[-1]
    renamed_history = history.rename(columns={"temperature": real_name})
    renamed_future = future.rename(columns={"temperature": real_name})
    model.fit(renamed_history, **{**COLUMN_ROLES, "control_columns": (real_name,)})
    torch.manual_seed(707)
    renamed = model.predict(renamed_future, n_samples=2)
    assert renamed.control_names[0] == real_name
    assert len(set(renamed.control_names)) == model.config.n_controls
    np.testing.assert_array_equal(renamed.components, initial.components)
    np.testing.assert_array_equal(renamed.graphs, initial.graphs)


def test_predict_adapts_channel_count_and_full_control_capacity(model, history, future):
    history = history.assign(price=[10.0, 11.0, 9.0, 12.0, 10.0])
    future = future.assign(price=[9.0, 10.0, 11.0])
    channels, controls = ("tv",), ("price", "temperature")
    model.fit(
        history,
        **{**COLUMN_ROLES, "channel_columns": channels, "control_columns": controls},
    )
    torch.manual_seed(710)
    actual = model.predict(future, n_samples=3)
    torch.manual_seed(710)
    expected = _canonical_result(
        model,
        history,
        future,
        controls,
        channel_columns=channels,
        control_columns=controls,
        n_samples=3,
    )
    _assert_same_result(actual, expected)
    assert actual.channel_names == channels and actual.control_names == controls
    control_axis = np.array([family == "control" for family in actual.component_families])
    np.testing.assert_array_equal(actual.component_mask[control_axis], [True, True])


def test_loaded_weights_state_and_mode_are_unchanged_and_bundles_exclude_context(
    model, history, future, tmp_path
):
    loaded = from_pretrained(save_pretrained(model, tmp_path / "source"))
    before_state = {name: value.clone() for name, value in loaded.state_dict().items()}
    before_trainability = [parameter.requires_grad for parameter in loaded.parameters()]
    before_bundle = save_pretrained(loaded, tmp_path / "before-fit")
    assert not loaded.training
    loaded.fit(history, **COLUMN_ROLES)
    assert not loaded.training
    loaded.train()
    assert loaded.fit(history, **COLUMN_ROLES) is loaded
    assert loaded.training
    torch.manual_seed(708)
    training_mode_result = loaded.predict(future, n_samples=3)
    assert loaded.training
    loaded.eval()
    torch.manual_seed(708)
    evaluation_mode_result = loaded.predict(future, n_samples=3)
    assert not loaded.training
    _assert_same_result(training_mode_result, evaluation_mode_result)
    assert loaded.state_dict().keys() == before_state.keys()
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in loaded.parameters()] == before_trainability
    assert all(parameter.grad is None for parameter in loaded.parameters())

    fitted_bundle = save_pretrained(loaded, tmp_path / "after-fit")
    assert {path.name for path in fitted_bundle.iterdir()} == {
        "config.json",
        "manifest.json",
        "model.safetensors",
    }
    assert (fitted_bundle / "config.json").read_bytes() == (
        before_bundle / "config.json"
    ).read_bytes()
    saved_weights = load_file(fitted_bundle / "model.safetensors")
    assert saved_weights.keys() == before_state.keys()
    for name, value in saved_weights.items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    restored = from_pretrained(fitted_bundle)
    with pytest.raises(RuntimeError, match="(?i)fit"):
        restored.predict(future, n_samples=2)


def test_predict_before_fit_is_clear_and_does_not_consume_rng(model, history, future):
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="(?i)fit"):
        model.predict(future, n_samples=2)
    with pytest.raises(ValueError):
        model.fit(history.assign(sales=0.0), **COLUMN_ROLES)
    with pytest.raises(RuntimeError, match="(?i)fit"):
        model.predict(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "roles",
    [
        pytest.param({"channel_columns": ()}, id="no-channels"),
        pytest.param({"control_columns": ()}, id="no-controls"),
        pytest.param({"date_column": ""}, id="empty-role-name"),
        pytest.param({"target_column": 5}, id="non-string-role-name"),
        pytest.param({"channel_columns": ("search", "search")}, id="duplicate-channels"),
        pytest.param({"control_columns": ("temperature", "temperature")}, id="duplicate-controls"),
        pytest.param({"control_columns": ("search",)}, id="overlapping-roles"),
        pytest.param({"channel_columns": "search"}, id="scalar-channel-list"),
        pytest.param(
            {"control_columns": ("temperature", "extra_control", "another_control")},
            id="too-many-controls",
        ),
    ],
)
def test_fit_rejects_invalid_role_schemas_before_rng_consumption(model, history, roles):
    history = history.assign(extra_control=1.0, another_control=2.0)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.fit(history, **{**COLUMN_ROLES, **roles})
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem", ["not_dataframe", "duplicate_columns", "missing_target", "one_row"]
)
def test_fit_rejects_invalid_table_shapes(model, history, problem):
    if problem == "not_dataframe":
        history = history.to_dict("list")
    elif problem == "duplicate_columns":
        history = pd.concat((history, history[["search"]]), axis=1)
    elif problem == "missing_target":
        history = history.drop(columns="sales")
    else:
        history = history.iloc[:1]
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.fit(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    ("column", "value"),
    [("search", -1e-50), ("sales", 0.0), ("temperature", np.inf), ("tv", 1 + 2j)],
)
def test_fit_validates_raw_numeric_domains_before_prediction(model, history, column, value):
    values = history[column].to_numpy(
        dtype=complex if isinstance(value, complex) else float, copy=True
    )
    values[0] = value
    history[column] = values
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.fit(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


def test_fit_validates_canonical_representation_without_waiting_for_a_query(model, history):
    history["sales"] = [1e-300, 1e300, 1e300, 1e300, 1e300]
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.fit(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem", ["missing", "invalid", "numeric", "aware", "duplicate", "irregular"]
)
def test_fit_rejects_invalid_or_irregular_dates(model, history, problem):
    if problem == "invalid":
        history["date"] = ["not a date", *history["date"].dt.strftime("%Y-%m-%d").tolist()[1:]]
    elif problem == "numeric":
        history["date"] = np.arange(len(history))
    elif problem == "aware":
        history["date"] = history["date"].dt.tz_localize("UTC")
    else:
        dates = history["date"].to_numpy(copy=True)
        if problem == "missing":
            dates[0] = np.datetime64("NaT")
        elif problem == "duplicate":
            dates[1] = dates[0]
        else:
            dates[2] += np.timedelta64(1, "D")
        history["date"] = dates
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.fit(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem",
    ["not_dataframe", "missing_channel", "empty", "negative_spend", "infinite", "text_control"],
)
def test_predict_validates_query_schema_and_domains_before_rng_consumption(
    model, history, future, problem
):
    model.fit(history, **COLUMN_ROLES)
    if problem == "not_dataframe":
        future = future.to_dict("list")
    elif problem == "missing_channel":
        future = future.drop(columns="search")
    elif problem == "empty":
        future = future.iloc[:0]
    elif problem == "negative_spend":
        future = future.assign(search=-1e-50)
    elif problem == "infinite":
        future = future.assign(tv=np.inf)
    else:
        future = future.assign(temperature="unknown")
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.predict(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem",
    ["missing", "aware", "duplicate", "overlap", "start_gap", "internal_gap", "single_off_grid"],
)
def test_predict_rejects_dates_outside_the_immediate_regular_horizon(
    model, history, future, problem
):
    model.fit(history, **COLUMN_ROLES)
    if problem == "aware":
        future["date"] = future["date"].dt.tz_localize("UTC")
    elif problem == "overlap":
        future["date"] = future["date"] - pd.Timedelta(days=7)
    elif problem == "start_gap":
        future["date"] = future["date"] + pd.Timedelta(days=7)
    elif problem == "single_off_grid":
        future = future.iloc[:1].copy()
        future["date"] = future["date"] + pd.Timedelta(days=1)
    else:
        dates = future["date"].to_numpy(copy=True)
        if problem == "missing":
            dates[0] = np.datetime64("NaT")
        elif problem == "duplicate":
            dates[1] = dates[0]
        else:
            dates[-1] += np.timedelta64(7, "D")
        future["date"] = dates
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.predict(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("column", ["search", "temperature"])
def test_nonzero_query_inputs_cannot_use_zero_context_scales(model, history, future, column):
    model.fit(history.assign(**{column: 0.0}), **COLUMN_ROLES)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="zero context"):
        model.predict(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    ("history_dates", "query_date"),
    [
        pytest.param(
            pd.date_range("2023-11-30", periods=5, freq="ME"),
            pd.Timestamp("2024-04-30"),
            id="calendar-month-end-across-leap-day",
        ),
        pytest.param(
            pd.to_datetime(["2024-01-01 06:00", "2024-01-01 11:00"]),
            pd.Timestamp("2024-01-01 16:00"),
            id="two-row-fixed-timedelta",
        ),
    ],
)
def test_calendar_and_two_row_cadences_accept_the_exact_next_single_row(
    model, history, future, history_dates, query_date
):
    history = history.iloc[: len(history_dates)].assign(date=history_dates)
    future = future.iloc[:1].assign(date=query_date)
    model.fit(history, **COLUMN_ROLES)
    torch.manual_seed(709)
    actual = model.predict(future, n_samples=2)
    torch.manual_seed(709)
    expected = _canonical_result(model, history, future, actual.control_names, n_samples=2)
    _assert_same_result(actual, expected)
    np.testing.assert_array_equal(actual.time, np.array([query_date], dtype="datetime64[ns]"))


@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_mixed_numeric_datetime_objects_cannot_be_coerced_to_date_labels(
    model, history, future, stage
):
    before = torch.get_rng_state()
    if stage == "fit":
        history = history.iloc[:2].copy()
        history["date"] = pd.Series([0.0, pd.Timestamp("1970-01-02")], dtype=object)
        with pytest.raises(ValueError):
            model.fit(history, **COLUMN_ROLES)
    else:
        model.fit(history, **COLUMN_ROLES)
        future["date"] = pd.Series(
            [float(future["date"].iloc[0].value), *future["date"].iloc[1:]], dtype=object
        )
        with pytest.raises(ValueError):
            model.predict(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)
