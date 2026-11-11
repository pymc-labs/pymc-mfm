"""The dataframe facade retains context, not weights, and uses canonical inference."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import needs_arviz
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
REQUEST_USAGE = {
    "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20},
    "query": {"steps": 3, "allocated_tokens": 15, "represented_tokens": 12},
    "total": {"allocated_tokens": 40, "represented_tokens": 32},
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
    for name in ("data_sha256", "sales_scale", "input_usage"):
        assert actual.provenance[name] == expected.provenance[name], name


def test_input_usage_inspects_context_and_horizons_without_a_prediction(model, history, future):
    model.add_context(history, **COLUMN_ROLES)
    before = torch.get_rng_state()
    context = model.input_usage()
    assert context == {"context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20}}
    usage = model.input_usage(future)
    assert usage == REQUEST_USAGE
    assert model.input_usage(future.iloc[:1]) == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20},
        "query": {"steps": 1, "allocated_tokens": 5, "represented_tokens": 4},
        "total": {"allocated_tokens": 30, "represented_tokens": 24},
    }
    ignored_columns = future.assign(sales=["unobserved", None, -np.inf], note="not an input")
    independent = model.input_usage(ignored_columns.iloc[::-1])
    assert independent == REQUEST_USAGE
    for section in independent.values():
        assert type(section) is dict
        assert all(type(value) is int for value in section.values())

    # Every section is caller-owned, including across context-only/full requests.
    for section in usage.values():
        for key in section:
            section[key] = -1
    assert context == {"context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20}}
    assert independent == REQUEST_USAGE
    context["context"]["steps"] = -1
    assert model.input_usage() == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20}
    }
    assert model.input_usage(future) == REQUEST_USAGE
    assert model.idata is None
    with pytest.raises(RuntimeError, match="inference"):
        model.build_idata()
    assert torch.equal(torch.get_rng_state(), before)


def test_add_context_chains_and_inference_matches_decompose_with_sorted_alignment(
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
    assert model.add_context(history_input, **COLUMN_ROLES) is model
    assert torch.equal(torch.get_rng_state(), before)

    torch.manual_seed(701)
    actual = model.inference(future_input, n_samples=4)
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
    assert actual.provenance["input_usage"] == model.input_usage(future_input) == REQUEST_USAGE
    pd.testing.assert_frame_equal(history_input, original_history)
    pd.testing.assert_frame_equal(future_input, original_future)


@pytest.mark.parametrize("mode", ["fixed", "map"])
def test_inference_preserves_graph_options_units_and_provenance(model, history, future, mode):
    model.add_context(history, **COLUMN_ROLES)
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
    actual = model.inference(future, **options)
    torch.manual_seed(702)
    expected = _canonical_result(model, history, future, actual.control_names, **options)
    _assert_same_result(actual, expected)
    assert actual.n_samples == 3 and actual.mode == mode and actual.units == "USD"
    assert actual.provenance["scenario"] == "board forecast"
    assert actual.provenance["input_usage"] == REQUEST_USAGE
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
    model.add_context(history, **COLUMN_ROLES)
    torch.manual_seed(703)
    expected = model.inference(future, n_samples=3)
    poisoned = future.assign(sales=outcomes)
    torch.manual_seed(703)
    actual = model.inference(poisoned, n_samples=3)
    _assert_same_result(actual, expected)
    assert actual.provenance == expected.provenance
    assert actual.provenance["sales_scale"] == np.quantile(history["sales"], 0.95)


def test_history_and_role_lists_are_snapshots_and_context_replacement_is_atomic(
    model, history, future
):
    pristine = history.copy(deep=True)
    channels, controls = list(CHANNELS), list(CONTROLS)
    model.add_context(
        history,
        date_column="date",
        target_column="sales",
        channel_columns=channels,
        control_columns=controls,
    )
    torch.manual_seed(704)
    expected = model.inference(future, n_samples=3)
    history["date"] = "not a date"
    history["sales"] = 0.0
    history.loc[:, [*CHANNELS, *CONTROLS]] = -999.0
    channels[:] = ["missing"]
    controls.clear()
    torch.manual_seed(704)
    _assert_same_result(model.inference(future, n_samples=3), expected)
    assert model.input_usage(future) == REQUEST_USAGE

    invalid_refit = pristine.rename(columns={"search": "new_search"}).assign(sales=0.0)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.add_context(
            invalid_refit, **{**COLUMN_ROLES, "channel_columns": ("new_search", "tv")}
        )
    assert torch.equal(torch.get_rng_state(), before)
    torch.manual_seed(704)
    _assert_same_result(model.inference(future, n_samples=3), expected)

    replacement = (
        pristine.assign(
            sales=pristine["sales"] * [1.0, 1.8, 0.7, 2.3, 1.2],
            search=pristine["search"] + [4.0, 15.0, 1.0, 20.0, 7.0],
            temperature=pristine["temperature"] * [-1.0, 2.0, 0.5, 3.0, -2.0],
        )
        .iloc[1:]
        .copy()
    )
    assert model.add_context(replacement, **COLUMN_ROLES) is model
    assert model.input_usage() == {
        "context": {"steps": 4, "allocated_tokens": 20, "represented_tokens": 16}
    }
    assert model.input_usage(future) == {
        "context": {"steps": 4, "allocated_tokens": 20, "represented_tokens": 16},
        "query": {"steps": 3, "allocated_tokens": 15, "represented_tokens": 12},
        "total": {"allocated_tokens": 35, "represented_tokens": 28},
    }
    torch.manual_seed(704)
    actual = model.inference(future, n_samples=3)
    torch.manual_seed(704)
    reference = _canonical_result(model, replacement, future, actual.control_names, n_samples=3)
    _assert_same_result(actual, reference)
    assert actual.provenance["sales_scale"] == np.quantile(replacement["sales"], 0.95)
    assert not np.array_equal(actual.components, expected.components)


def test_queries_are_independent_and_rebuild_relative_time_for_each_horizon(model, history, future):
    model.add_context(history, **COLUMN_ROLES)
    short = future.iloc[:1].copy()
    torch.manual_seed(705)
    expected_short = model.inference(short, n_samples=3)
    different_query = future.assign(search=future["search"] * 4)
    torch.manual_seed(706)
    actual_long = model.inference(different_query, n_samples=3)
    torch.manual_seed(706)
    expected_long = _canonical_result(
        model, history, different_query, actual_long.control_names, n_samples=3
    )
    _assert_same_result(actual_long, expected_long)
    torch.manual_seed(705)
    _assert_same_result(model.inference(short, n_samples=3), expected_short)


def test_padding_names_do_not_collide_with_real_controls(model, history, future):
    model.add_context(history, **COLUMN_ROLES)
    torch.manual_seed(707)
    initial = model.inference(future, n_samples=2)
    real_name = initial.control_names[-1]
    renamed_history = history.rename(columns={"temperature": real_name})
    renamed_future = future.rename(columns={"temperature": real_name})
    model.add_context(renamed_history, **{**COLUMN_ROLES, "control_columns": (real_name,)})
    torch.manual_seed(707)
    renamed = model.inference(renamed_future, n_samples=2)
    assert renamed.control_names[0] == real_name
    assert len(set(renamed.control_names)) == model.config.n_controls
    np.testing.assert_array_equal(renamed.components, initial.components)
    np.testing.assert_array_equal(renamed.graphs, initial.graphs)


def test_inference_adapts_channel_count_and_full_control_capacity(model, history, future):
    history = history.assign(price=[10.0, 11.0, 9.0, 12.0, 10.0])
    future = future.assign(price=[9.0, 10.0, 11.0])
    channels, controls = ("tv",), ("price", "temperature")
    model.add_context(
        history,
        **{**COLUMN_ROLES, "channel_columns": channels, "control_columns": controls},
    )
    torch.manual_seed(710)
    actual = model.inference(future, n_samples=3)
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
    assert (
        actual.provenance["input_usage"]
        == model.input_usage(future)
        == {
            "context": {"steps": 5, "allocated_tokens": 20, "represented_tokens": 20},
            "query": {"steps": 3, "allocated_tokens": 12, "represented_tokens": 12},
            "total": {"allocated_tokens": 32, "represented_tokens": 32},
        }
    )
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
    loaded.add_context(history, **COLUMN_ROLES)
    assert not loaded.training
    loaded.train()
    assert loaded.add_context(history, **COLUMN_ROLES) is loaded
    assert loaded.training
    assert loaded.input_usage(future) == REQUEST_USAGE
    torch.manual_seed(708)
    training_mode_result = loaded.inference(future, n_samples=3)
    assert loaded.training
    loaded.eval()
    torch.manual_seed(708)
    evaluation_mode_result = loaded.inference(future, n_samples=3)
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
    with pytest.raises(RuntimeError):
        restored.inference(future, n_samples=2)
    with pytest.raises(RuntimeError, match="add_context"):
        restored.input_usage()


def test_inference_without_context_does_not_consume_rng(model, history, future):
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="add_context"):
        model.input_usage()
    with pytest.raises(RuntimeError, match="add_context"):
        model.input_usage(future)
    with pytest.raises(RuntimeError):
        model.inference(future, n_samples=2)
    with pytest.raises(ValueError):
        model.add_context(history.assign(sales=0.0), **COLUMN_ROLES)
    with pytest.raises(RuntimeError):
        model.inference(future, n_samples=2)
    with pytest.raises(RuntimeError, match="add_context"):
        model.input_usage()
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
def test_add_context_rejects_invalid_role_schemas_before_rng_consumption(model, history, roles):
    history = history.assign(extra_control=1.0, another_control=2.0)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.add_context(history, **{**COLUMN_ROLES, **roles})
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem", ["not_dataframe", "duplicate_columns", "missing_target", "one_row"]
)
def test_add_context_rejects_invalid_table_shapes(model, history, problem):
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
        model.add_context(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    ("column", "value"),
    [("search", -1e-50), ("sales", 0.0), ("temperature", np.inf), ("tv", 1 + 2j)],
)
def test_add_context_validates_raw_numeric_domains_before_inference(model, history, column, value):
    values = history[column].to_numpy(
        dtype=complex if isinstance(value, complex) else float, copy=True
    )
    values[0] = value
    history[column] = values
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.add_context(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


def test_add_context_validates_canonical_representation_without_waiting_for_a_query(model, history):
    history["sales"] = [1e-300, 1e300, 1e300, 1e300, 1e300]
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.add_context(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem", ["missing", "invalid", "numeric", "aware", "duplicate", "irregular"]
)
def test_add_context_rejects_invalid_or_irregular_dates(model, history, problem):
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
        model.add_context(history, **COLUMN_ROLES)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem",
    ["not_dataframe", "missing_channel", "empty", "negative_spend", "infinite", "text_control"],
)
def test_inference_validates_query_schema_and_domains_before_rng_consumption(
    model, history, future, problem
):
    model.add_context(history, **COLUMN_ROLES)
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
        model.input_usage(future)
    with pytest.raises(ValueError):
        model.inference(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem",
    ["missing", "aware", "duplicate", "overlap", "start_gap", "internal_gap", "single_off_grid"],
)
def test_inference_rejects_dates_outside_the_immediate_regular_horizon(
    model, history, future, problem
):
    model.add_context(history, **COLUMN_ROLES)
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
        model.input_usage(future)
    with pytest.raises(ValueError):
        model.inference(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("column", ["search", "temperature"])
def test_nonzero_query_inputs_cannot_use_zero_context_scales(model, history, future, column):
    model.add_context(history.assign(**{column: 0.0}), **COLUMN_ROLES)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="zero context"):
        model.input_usage(future)
    with pytest.raises(ValueError, match="zero context"):
        model.inference(future, n_samples=2)
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
    model.add_context(history, **COLUMN_ROLES)
    torch.manual_seed(709)
    actual = model.inference(future, n_samples=2)
    torch.manual_seed(709)
    expected = _canonical_result(model, history, future, actual.control_names, n_samples=2)
    _assert_same_result(actual, expected)
    np.testing.assert_array_equal(actual.time, np.array([query_date], dtype="datetime64[ns]"))


@pytest.mark.parametrize("stage", ["add_context", "inference"])
def test_mixed_numeric_datetime_objects_cannot_be_coerced_to_date_labels(
    model, history, future, stage
):
    before = torch.get_rng_state()
    if stage == "add_context":
        history = history.iloc[:2].copy()
        history["date"] = pd.Series([0.0, pd.Timestamp("1970-01-02")], dtype=object)
        with pytest.raises(ValueError):
            model.add_context(history, **COLUMN_ROLES)
    else:
        model.add_context(history, **COLUMN_ROLES)
        future["date"] = pd.Series(
            [float(future["date"].iloc[0].value), *future["date"].iloc[1:]], dtype=object
        )
        with pytest.raises(ValueError):
            model.inference(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@needs_arviz
def test_input_usage_preserves_prediction_export_weights_mode_and_seeded_draws(
    model, history, future
):
    import xarray as xr

    # A constant represented channel also provides the known float16 overflow case.
    history = history.assign(search=1.0)
    future = future.assign(search=1.0)
    model.add_context(history, **COLUMN_ROLES)
    torch.manual_seed(711)
    query = model.inference(future, n_samples=3)
    stored = model.build_idata(store=True)
    original = stored.copy(deep=True)
    model.train()
    trainability = [parameter.requires_grad for parameter in model.parameters()]

    def assert_preserved(state, rng):
        assert model.training
        assert model.idata is stored
        assert torch.equal(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        assert [parameter.requires_grad for parameter in model.parameters()] == trainability
        assert all(parameter.grad is None for parameter in model.parameters())
        xr.testing.assert_identical(stored, original)
        rebuilt = model.build_idata()
        xr.testing.assert_equal(rebuilt, original)
        for window in ("context", "query"):
            assert (
                rebuilt[window].attrs["metadata_sha256"]
                == original[window].attrs["metadata_sha256"]
            )
        assert model.idata is stored
        assert torch.equal(torch.get_rng_state(), rng)

    state = {name: value.clone() for name, value in model.state_dict().items()}
    before = torch.get_rng_state()
    assert model.input_usage() == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20}
    }
    assert model.input_usage(future.iloc[:1]) == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20},
        "query": {"steps": 1, "allocated_tokens": 5, "represented_tokens": 4},
        "total": {"allocated_tokens": 30, "represented_tokens": 24},
    }
    with pytest.raises(ValueError):
        model.input_usage(future.assign(search=-1e-50))
    assert_preserved(state, before)
    torch.manual_seed(711)
    reference = _canonical_result(model, history, future, query.control_names, n_samples=3)
    _assert_same_result(query, reference)

    # The canonical float32 query is finite, but cannot be narrowed to model dtype.
    # Inspection must reject it before touching the completed prediction or export.
    model.half()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    before = torch.get_rng_state()
    assert model.input_usage(future) == REQUEST_USAGE
    with pytest.raises(ValueError, match="qry_spend_z.*finite model representation"):
        model.input_usage(future.assign(search=1e100))
    assert_preserved(state, before)


@needs_arviz
@pytest.mark.parametrize("mode", ["model_average", "map"])
def test_idata_named_values_match_both_marginals_and_work_with_arviz(model, history, future, mode):
    import arviz_base
    from arviz_stats import summary

    model.add_context(history.iloc[::-1], **COLUMN_ROLES)
    options = {"router_samples": 7} if mode == "map" else {}
    query = model.inference(future.iloc[::-1], n_samples=5, mode=mode, units="USD", **options)
    context_rng = torch.get_rng_state()
    tree = model.build_idata()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(context_rng)
        context = _canonical_result(
            model,
            history,
            future,
            query.control_names,
            n_samples=5,
            window="context",
            mode=mode,
            units="USD",
            **options,
        )

    assert model.idata is None
    assert set(tree.children) == {"context", "query"}
    assert "date" not in tree.coords
    for window, result in (("context", context), ("query", query)):
        subtree = tree[window]
        posterior = subtree["posterior"]
        assert set(posterior.data_vars) == {
            "baseline",
            "channel_contribution",
            "control_contribution",
            "latent",
            "indirect_cc",
            "indirect_zc",
            "indirect_dc",
            "graph",
        }
        np.testing.assert_array_equal(posterior["date"].values, result.time)
        assert posterior["channel"].values.tolist() == list(CHANNELS)
        assert posterior["control"].values.tolist() == list(CONTROLS)
        for variable, family, dimension in (
            ("channel_contribution", "channel", "channel"),
            ("control_contribution", "control", "control"),
        ):
            active = np.array(result.component_families) == family
            active &= result.component_mask
            assert posterior[variable].dims == ("chain", "draw", "date", dimension)
            np.testing.assert_array_equal(
                posterior[variable].values[0], result.components[..., active]
            )
        for component in ("baseline", "latent", "indirect:cc", "indirect:zc", "indirect:dc"):
            variable = component.replace(":", "_")
            assert posterior[variable].dims == ("chain", "draw", "date")
            np.testing.assert_array_equal(
                posterior[variable].values[0],
                result.components[..., result.component_names.index(component)],
            )
        assert posterior["graph"].dims == ("chain", "draw", "edge")
        np.testing.assert_array_equal(
            posterior["graph"].values[0], result.graphs[:, result.edge_mask]
        )
        constants = subtree["constant_data"]
        endpoints = np.array(result.provenance["edge_endpoints"])[result.edge_mask]
        np.testing.assert_array_equal(constants["edge_source"].values, endpoints[:, 0])
        np.testing.assert_array_equal(constants["edge_target"].values, endpoints[:, 1])
        for attrs in (subtree.attrs, posterior.attrs):
            assert attrs["window"] == window
            assert attrs["mode"] == mode
            assert attrs["units"] == "USD"
            assert attrs["data_sha256"] == result.provenance["data_sha256"]
            assert attrs["mcmc_diagnostics"] == "not_applicable"
        extracted = arviz_base.extract(subtree, var_names="channel_contribution")
        np.testing.assert_array_equal(
            extracted.sel(channel="search").values,
            posterior["channel_contribution"].sel(channel="search").values[0].T,
        )
        table = summary(subtree, var_names=["channel_contribution"], kind="stats", round_to="none")
        np.testing.assert_allclose(
            table["mean"].to_numpy(),
            posterior["channel_contribution"].values.mean(axis=(0, 1)).ravel(),
            rtol=1e-12,
            atol=1e-12,
        )
    assert set(tree["context"].children) == {"posterior", "constant_data", "observed_data"}
    assert set(tree["query"].children) == {"posterior", "constant_data"}
    assert tree["context"]["observed_data"]["outcome"].dims == ("date",)
    np.testing.assert_array_equal(
        tree["context"]["observed_data"]["outcome"].values, history["sales"].to_numpy()
    )


def test_idata_requires_successful_inference_for_the_current_context(model, history):
    assert model.idata is None
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError):
        model.build_idata()
    model.add_context(history, **COLUMN_ROLES)
    with pytest.raises(RuntimeError):
        model.build_idata(store=True)
    assert model.idata is None
    assert torch.equal(torch.get_rng_state(), before)


@needs_arviz
def test_idata_storage_modes_and_repeated_exports_preserve_the_draws(model, history, future):
    import xarray as xr

    model.add_context(history, **COLUMN_ROLES)
    query = model.inference(future, n_samples=4)
    query_values = query.components.copy()
    first = model.build_idata()
    assert model.idata is None
    torch.manual_seed(819)
    before = torch.get_rng_state()
    stored = model.build_idata(store=True)
    assert model.idata is stored
    assert model.build_idata(store=True) is stored
    returned = model.build_idata()
    assert returned is not stored
    assert model.idata is stored
    assert torch.equal(torch.get_rng_state(), before)
    for window in ("context", "query"):
        xr.testing.assert_equal(first[window]["posterior"], stored[window]["posterior"])
        xr.testing.assert_equal(returned[window]["posterior"], stored[window]["posterior"])
    np.testing.assert_array_equal(query.components, query_values)


@needs_arviz
def test_new_context_and_inference_invalidate_only_model_owned_idata(model, history, future):
    import xarray as xr

    model.add_context(history, **COLUMN_ROLES)
    first_query = model.inference(future, n_samples=3)
    first = model.build_idata(store=True)
    first_copy = first.copy(deep=True)
    second_query = model.inference(future.iloc[:1].assign(search=75.0), n_samples=3)
    assert model.idata is None
    second = model.build_idata(store=True)
    second_copy = second.copy(deep=True)
    assert first_query.provenance["data_sha256"] != second_query.provenance["data_sha256"]
    for window in ("context", "query"):
        assert second[window].attrs["data_sha256"] == second_query.provenance["data_sha256"]
    xr.testing.assert_identical(first, first_copy)

    model.add_context(history.assign(sales=history["sales"] * 1.5), **COLUMN_ROLES)
    assert model.idata is None
    with pytest.raises(RuntimeError):
        model.build_idata()
    model.inference(future, n_samples=3)
    third = model.build_idata(store=True)
    assert third["context"].attrs["data_sha256"] != second["context"].attrs["data_sha256"]
    xr.testing.assert_identical(first, first_copy)
    xr.testing.assert_identical(second, second_copy)


@needs_arviz
def test_failed_context_or_inference_preserves_last_successful_idata(model, history, future):
    import xarray as xr

    model.add_context(history, **COLUMN_ROLES)
    model.inference(future, n_samples=3)
    stored = model.build_idata(store=True)

    def assert_preserved():
        before = torch.get_rng_state()
        rebuilt = model.build_idata()
        assert model.idata is stored
        assert torch.equal(torch.get_rng_state(), before)
        xr.testing.assert_equal(rebuilt, stored)
        for window in ("context", "query"):
            assert (
                rebuilt[window].attrs["metadata_sha256"] == stored[window].attrs["metadata_sha256"]
            )

    with pytest.raises(ValueError):
        model.add_context(history.assign(sales=0.0), **COLUMN_ROLES)
    assert_preserved()
    with pytest.raises(ValueError):
        model.inference(future.assign(date=future["date"] + pd.Timedelta(days=7)), n_samples=3)
    assert_preserved()
    # Result construction can fail after the sampler has completed.
    with pytest.raises(TypeError):
        model.inference(future, n_samples=3, provenance={"scenario": object()})
    assert_preserved()
    assert model.build_idata(store=True) is stored


@needs_arviz
def test_idata_retains_prediction_inputs_provenance_and_each_fixed_graph(
    model, history, future, tmp_path
):
    model.add_context(history, **COLUMN_ROLES)
    layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
    graphs = np.zeros((3, 1, layout.n_slots))
    graphs[0, 0, layout.slices["cy"].start] = 1
    graphs[1, 0, layout.slices["cy"].start + 1] = 1
    original_graphs = graphs.copy()
    annotations = {"scenario": {"name": "original"}}
    query = model.inference(future, n_samples=3, mode="fixed", graph=graphs, provenance=annotations)
    np.testing.assert_array_equal(query.graphs, original_graphs[:, 0])
    with torch.random.fork_rng(devices=[]):
        expected_context = _canonical_result(
            model,
            history,
            future,
            query.control_names,
            n_samples=3,
            window="context",
            mode="fixed",
            graph=original_graphs,
            provenance={"scenario": {"name": "original"}},
        )
    context_metadata = json.loads(
        (expected_context.save(tmp_path / "context") / "result.json").read_text()
    )
    metadata = json.loads((query.save(tmp_path / "query") / "result.json").read_text())
    query_dates = query.time.copy()
    graphs[:] = 1
    annotations["scenario"]["name"] = object()
    future["date"] = "changed after inference"
    future.loc[:, [*CHANNELS, *CONTROLS]] = -999.0
    history["sales"] = 0.0
    tree = model.build_idata()
    assert tree["query"].attrs["metadata_sha256"] == metadata["metadata_sha256"]
    assert tree["context"].attrs["metadata_sha256"] == context_metadata["metadata_sha256"]
    np.testing.assert_array_equal(tree["query"]["posterior"]["date"].values, query_dates)
    for window in ("context", "query"):
        np.testing.assert_array_equal(
            tree[window]["posterior"]["graph"].values[0],
            original_graphs[:, 0, query.edge_mask],
        )
        assert tree[window].attrs["data_sha256"] == query.provenance["data_sha256"]
        assert tree[window].attrs["graph_source"] == "supplied_hypothesis"


@needs_arviz
def test_named_idata_preserves_odd_column_names_and_zero_fixed_graphs(model, history, future):
    channels = ("baseline", "paid:検索 / social")
    controls = ("__padded_control_1__",)
    rename = dict(zip((*CHANNELS, *CONTROLS), (*channels, *controls), strict=True))
    model.add_context(
        history.rename(columns=rename),
        **{**COLUMN_ROLES, "channel_columns": channels, "control_columns": controls},
    )
    layout = EdgeSlotLayout(len(channels), model.config.n_controls, model.config.n_demand)
    model.inference(
        future.rename(columns=rename), n_samples=3, mode="fixed", graph=np.zeros(layout.n_slots)
    )
    tree = model.build_idata()
    for window in ("context", "query"):
        posterior = tree[window]["posterior"]
        assert posterior["channel"].values.tolist() == list(channels)
        assert posterior["control"].values.tolist() == list(controls)
        np.testing.assert_array_equal(
            posterior["channel_contribution"].sel(channel=channels[1]).values, 0.0
        )
        np.testing.assert_array_equal(
            posterior["control_contribution"].sel(control=controls[0]).values, 0.0
        )
        np.testing.assert_array_equal(posterior["graph"].values, 0.0)


def test_missing_idata_extra_is_reported_before_context_sampling(
    model, history, future, monkeypatch
):
    model.add_context(history, **COLUMN_ROLES)
    model.inference(future, n_samples=3)
    for name in ("xarray", "arviz_base"):
        monkeypatch.setitem(sys.modules, name, None)
    before = torch.get_rng_state()
    with pytest.raises(ImportError, match=r"pymc-mfm\[arviz\]"):
        model.build_idata(store=True)
    assert model.idata is None
    assert torch.equal(torch.get_rng_state(), before)


@needs_arviz
def test_failed_conversion_does_not_redraw_a_completed_context(model, history, future):
    channel = "search\x00"
    model.add_context(
        history.rename(columns={"search": channel}),
        **{**COLUMN_ROLES, "channel_columns": (channel, "tv")},
    )
    model.inference(future.rename(columns={"search": channel}), n_samples=3)
    with pytest.raises(ValueError, match="NUL"):
        model.build_idata()
    after_context = torch.get_rng_state()
    with pytest.raises(ValueError, match="NUL"):
        model.build_idata()
    assert torch.equal(torch.get_rng_state(), after_context)
    assert model.idata is None


@needs_arviz
def test_weight_bundles_exclude_prediction_and_idata_state(model, history, future, tmp_path):
    baseline = save_pretrained(model, tmp_path / "baseline")
    model.add_context(history, **COLUMN_ROLES)
    model.inference(future, n_samples=3)
    model.build_idata(store=True)
    with_idata = save_pretrained(model, tmp_path / "with-idata")
    assert {path.name for path in with_idata.iterdir()} == {
        "config.json",
        "manifest.json",
        "model.safetensors",
    }
    assert (baseline / "config.json").read_bytes() == (with_idata / "config.json").read_bytes()
    before_weights = load_file(baseline / "model.safetensors")
    after_weights = load_file(with_idata / "model.safetensors")
    assert before_weights.keys() == after_weights.keys()
    for name, value in after_weights.items():
        torch.testing.assert_close(value, before_weights[name], rtol=0, atol=0)
    restored = from_pretrained(with_idata)
    assert restored.idata is None
    with pytest.raises(RuntimeError):
        restored.build_idata()


@needs_arviz
def test_loaded_recorded_model_builds_both_windows_without_replaying_derived_provenance(
    model, history, future, tmp_path
):
    from test_checkpoint import _training_evaluation

    recorded = _training_evaluation()
    loaded = from_pretrained(
        save_pretrained(
            model,
            tmp_path / "recorded",
            profile="A",
            provenance={"source_checkpoint_sha256": "a" * 64, "training_evaluation": recorded},
        )
    )
    loaded.add_context(history, **COLUMN_ROLES)
    query = loaded.inference(future, n_samples=3, provenance={"scenario": {"name": "loaded"}})
    returned = loaded.build_idata()
    assert loaded.idata is None
    stored = loaded.build_idata(store=True)
    assert loaded.idata is stored
    assert returned is not stored
    for tree in (returned, stored):
        for window in ("context", "query"):
            assert tree[window].attrs["state_sha256"] == loaded.provenance["state_sha256"]
            assert tree[window].attrs["data_sha256"] == query.provenance["data_sha256"]
    assert loaded.artifact_metadata["provenance"]["training_evaluation"] == recorded
