"""Paired dataframe inference publishes complete analysis snapshots, never weights."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from conftest import needs_arviz
from safetensors import safe_open
from safetensors.torch import load_file

from pymc_mktfm.inference.api import _decompose, decompose, observation_batch
from pymc_mktfm.inference.checkpoint import save_pretrained
from pymc_mktfm.inference.hub import from_pretrained
from pymc_mktfm.inference.result import DecompositionResult
from pymc_mktfm.model.graph import EdgeSlotLayout

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
    channel_names=None,
    latent_mask=None,
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
            "latent_mask": np.ones(model.config.n_demand, dtype=bool)
            if latent_mask is None
            else latent_mask,
        },
        channel_names=channel_columns if channel_names is None else channel_names,
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


def _canonical_pair(model, history, future, control_names, **options):
    """The public operation samples query first, then the independent context marginal."""
    query = _canonical_result(model, history, future, control_names, **options)
    context_options = {**options, "window": "context"}
    if query.mode == "fixed":
        context_options["graph"] = query.graphs[:, None, :]
    context = _canonical_result(model, history, future, control_names, **context_options)
    return query, context


def _assert_named_posterior(posterior, result):
    for variable, family, dimension in (
        ("channel_contribution", "channel", "channel"),
        ("control_contribution", "control", "control"),
    ):
        active = (np.array(result.component_families) == family) & result.component_mask
        assert posterior[variable].dims == ("chain", "draw", "date", dimension)
        np.testing.assert_array_equal(posterior[variable].values[0], result.components[..., active])
    for component in ("baseline", "latent", "indirect:cc", "indirect:zc", "indirect:dc"):
        variable = component.replace(":", "_")
        assert posterior[variable].dims == ("chain", "draw", "date")
        np.testing.assert_array_equal(
            posterior[variable].values[0],
            result.components[..., result.component_names.index(component)],
        )
    assert posterior["graph"].dims == ("chain", "draw", "edge")
    np.testing.assert_array_equal(posterior["graph"].values[0], result.graphs[:, result.edge_mask])


def _inject_inference_failure(monkeypatch, model, stage):
    """Fail after real sampling, or at conversion, without inventing candidate draws."""
    if stage == "conversion":
        from pymc_mktfm.inference import _datatree

        def fail(*args, **kwargs):
            raise RuntimeError("conversion failed")

        monkeypatch.setattr(_datatree, "prediction_datatree", fail)
    else:
        sample = model.sample_disentangled

        def fail(*args, **kwargs):
            result = sample(*args, **kwargs)
            if kwargs["window"] == stage:
                raise RuntimeError(f"{stage} sampling failed")
            return result

        monkeypatch.setattr(model, "sample_disentangled", fail)


def test_input_usage_inspects_explicit_context_and_horizons_without_inference(
    model, history, future
):
    before = torch.get_rng_state()
    context = model.input_usage(history, **COLUMN_ROLES)
    assert context == {"context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20}}
    usage = model.input_usage(history, future, **COLUMN_ROLES)
    assert usage == REQUEST_USAGE
    assert model.input_usage(history, future.iloc[:1], **COLUMN_ROLES) == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20},
        "query": {"steps": 1, "allocated_tokens": 5, "represented_tokens": 4},
        "total": {"allocated_tokens": 30, "represented_tokens": 24},
    }
    ignored_columns = future.assign(sales=["unobserved", None, -np.inf], note="not an input")
    independent = model.input_usage(history, ignored_columns.iloc[::-1], **COLUMN_ROLES)
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
    assert model.input_usage(history, **COLUMN_ROLES) == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20}
    }
    assert model.input_usage(history, future, **COLUMN_ROLES) == REQUEST_USAGE
    assert model.idata is None and model.query_result is None
    with pytest.raises(RuntimeError, match=r"inference\(context, query"):
        model.predict(future, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


def test_paired_inference_matches_decompose_with_sorted_alignment(model, history, future):
    history_input = history.iloc[[3, 0, 4, 1, 2]][
        ["temperature", "date", "sales", "tv", "search"]
    ].copy()
    future_input = future.iloc[[2, 0, 1]][["tv", "temperature", "search", "date"]].copy()
    history_input["date"] = history_input["date"].dt.strftime("%Y-%m-%d")
    future_input["date"] = future_input["date"].dt.strftime("%Y-%m-%d")
    history_input["unrelated"] = "private note"
    future_input["unrelated"] = None
    original_history, original_future = history_input.copy(deep=True), future_input.copy(deep=True)

    torch.manual_seed(701)
    tree = model.inference(history_input, future_input, **COLUMN_ROLES, n_samples=4)
    actual = model.query_result
    after = torch.get_rng_state()
    torch.manual_seed(701)
    expected, context = _canonical_pair(model, history, future, actual.control_names, n_samples=4)
    assert torch.equal(torch.get_rng_state(), after)
    _assert_same_result(actual, expected)
    assert tree is model.idata
    assert model.query_result is actual
    assert isinstance(actual, DecompositionResult)
    _assert_named_posterior(tree["query"]["posterior"], expected)
    _assert_named_posterior(tree["context"]["posterior"], context)
    assert actual.channel_names == CHANNELS
    assert actual.control_names[: len(CONTROLS)] == CONTROLS
    assert len(actual.control_names) == model.config.n_controls
    assert len(set(actual.control_names)) == len(actual.control_names)
    control_columns = np.array([family == "control" for family in actual.component_families])
    np.testing.assert_array_equal(actual.component_mask[control_columns], [True, False])
    np.testing.assert_array_equal(actual.time, future["date"].to_numpy(dtype="datetime64[ns]"))
    assert actual.time.dtype == np.dtype("datetime64[ns]")
    assert actual.window == "query" and actual.provenance["head"] == "conditional"
    assert (
        actual.provenance["input_usage"]
        == model.input_usage(history_input, future_input, **COLUMN_ROLES)
        == REQUEST_USAGE
    )
    pd.testing.assert_frame_equal(history_input, original_history)
    pd.testing.assert_frame_equal(future_input, original_future)


@pytest.mark.parametrize("mode", ["fixed", "map"])
def test_inference_preserves_graph_options_units_and_provenance(model, history, future, mode):
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
    tree = model.inference(history, future, **COLUMN_ROLES, **options)
    actual = model.query_result
    after = torch.get_rng_state()
    torch.manual_seed(702)
    expected, context = _canonical_pair(model, history, future, actual.control_names, **options)
    assert torch.equal(torch.get_rng_state(), after)
    _assert_same_result(actual, expected)
    _assert_named_posterior(tree["context"]["posterior"], context)
    assert actual.n_samples == 3 and actual.mode == mode and actual.units == "USD"
    assert actual.provenance["scenario"] == "board forecast"
    assert actual.provenance["input_usage"] == REQUEST_USAGE
    if mode == "fixed":
        np.testing.assert_array_equal(actual.graphs, np.broadcast_to(graph, actual.graphs.shape))
        assert actual.provenance["graph_source"] == "supplied_hypothesis"
    else:
        assert actual.provenance["inference_config"]["router_samples"] == 7


@pytest.mark.parametrize(
    ("outcomes", "observed"),
    [
        pytest.param(["not observed", None, -np.inf], None, id="all-unobserved"),
        pytest.param([1e100, 1e-100, 1e200], [1e100, 1e-100, 1e200], id="finite-numeric"),
        pytest.param(["123.5", "missing", np.inf], [123.5, np.nan, np.nan], id="partial"),
        pytest.param(
            pd.Series([10**400, 2.0, [1]], dtype=object),
            [np.nan, 2.0, np.nan],
            id="object-overflow",
        ),
        pytest.param([True, False, True], None, id="booleans"),
        pytest.param([1 + 2j, 2, 3], None, id="complex"),
        pytest.param(pd.date_range("2024-02-05", periods=3), None, id="datetime"),
    ],
)
def test_query_outcomes_are_observations_not_model_inputs(
    model, history, future, outcomes, observed
):
    torch.manual_seed(703)
    expected_tree = model.inference(history, future, **COLUMN_ROLES, n_samples=3)
    expected = model.query_result
    expected_rng = torch.get_rng_state()
    annotated = future.assign(sales=outcomes)
    torch.manual_seed(703)
    tree = model.inference(history, annotated.iloc[::-1], **COLUMN_ROLES, n_samples=3)
    actual = model.query_result
    assert torch.equal(torch.get_rng_state(), expected_rng)
    _assert_same_result(actual, expected)
    assert actual.provenance == expected.provenance
    assert actual.provenance["sales_scale"] == np.quantile(history["sales"], 0.95)
    for window in ("context", "query"):
        xr.testing.assert_equal(tree[window]["posterior"], expected_tree[window]["posterior"])
    if observed is None:
        assert "observed_data" not in tree["query"].children
    else:
        assert tree["query"]["observed_data"]["outcome"].dims == ("date",)
        np.testing.assert_array_equal(tree["query"]["observed_data"]["outcome"].values, observed)
        np.testing.assert_array_equal(
            tree["query"]["observed_data"]["date"].values,
            future["date"].to_numpy(dtype="datetime64[ns]"),
        )
    # Removing targets on a later run removes the group, not just its finite values.
    model.inference(history, future, **COLUMN_ROLES, n_samples=3)
    assert "observed_data" not in model.idata["query"].children


def test_history_and_role_lists_are_snapshots_and_replacement_is_atomic(model, history, future):
    pristine = history.copy(deep=True)
    channels, controls = list(CHANNELS), list(CONTROLS)
    torch.manual_seed(704)
    stored = model.inference(
        history,
        future,
        date_column="date",
        target_column="sales",
        channel_columns=channels,
        control_columns=controls,
        n_samples=3,
    )
    snapshot = stored.copy(deep=True)
    query = model.query_result
    torch.manual_seed(704)
    expected = _canonical_result(model, pristine, future, query.control_names, n_samples=3)
    history["date"] = "not a date"
    history["sales"] = 0.0
    history.loc[:, [*CHANNELS, *CONTROLS]] = -999.0
    channels[:] = ["missing"]
    controls.clear()
    torch.manual_seed(704)
    _assert_same_result(model.predict(future, n_samples=3), expected)
    xr.testing.assert_identical(stored, snapshot)
    assert model.idata is stored and model.query_result is query

    invalid_refit = pristine.rename(columns={"search": "new_search"}).assign(sales=0.0)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.inference(
            invalid_refit,
            future.rename(columns={"search": "new_search"}),
            **{**COLUMN_ROLES, "channel_columns": ("new_search", "tv")},
            n_samples=3,
        )
    assert torch.equal(torch.get_rng_state(), before)
    assert model.idata is stored and model.query_result is query
    torch.manual_seed(704)
    _assert_same_result(model.predict(future, n_samples=3), expected)

    replacement = (
        pristine.assign(
            sales=pristine["sales"] * [1.0, 1.8, 0.7, 2.3, 1.2],
            search=pristine["search"] + [4.0, 15.0, 1.0, 20.0, 7.0],
            temperature=pristine["temperature"] * [-1.0, 2.0, 0.5, 3.0, -2.0],
        )
        .iloc[1:]
        .copy()
    )
    assert model.input_usage(replacement, **COLUMN_ROLES) == {
        "context": {"steps": 4, "allocated_tokens": 20, "represented_tokens": 16}
    }
    assert model.input_usage(replacement, future, **COLUMN_ROLES) == {
        "context": {"steps": 4, "allocated_tokens": 20, "represented_tokens": 16},
        "query": {"steps": 3, "allocated_tokens": 15, "represented_tokens": 12},
        "total": {"allocated_tokens": 35, "represented_tokens": 28},
    }
    torch.manual_seed(704)
    replacement_tree = model.inference(replacement, future, **COLUMN_ROLES, n_samples=3)
    actual = model.query_result
    torch.manual_seed(704)
    reference, context = _canonical_pair(
        model, replacement, future, actual.control_names, n_samples=3
    )
    _assert_same_result(actual, reference)
    _assert_named_posterior(replacement_tree["context"]["posterior"], context)
    assert actual.provenance["sales_scale"] == np.quantile(replacement["sales"], 0.95)
    xr.testing.assert_identical(stored, snapshot)


@pytest.mark.parametrize("mode", ["model_average", "map", "fixed"])
def test_predict_is_query_only_and_preserves_the_complete_run_and_curves(
    model, history, future, mode
):
    layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
    graph = np.zeros(layout.n_slots)
    graph[layout.slices["cy"].start] = 1
    model.inference(
        history, future, **COLUMN_ROLES, n_samples=3, mode="fixed", graph=graph, units="USD"
    )
    query = model.query_result
    torch.manual_seed(705)
    curves = model.response_curves(levels=[0, 1], channels=["search"], weeks=2)
    stored = model.idata
    snapshot = stored.copy(deep=True)
    options = {"n_samples": 4}
    if mode == "fixed":
        options.update(mode=mode, graph=graph)
    elif mode == "map":
        options.update(mode=mode, router_samples=7)
    short = future.iloc[:1].assign(search=75.0)
    long = future.assign(search=future["search"] * 4)

    for plan, seed in ((short, 706), (long, 707), (short, 706)):
        torch.manual_seed(seed)
        actual = model.predict(plan, **options)
        after = torch.get_rng_state()
        torch.manual_seed(seed)
        expected = _canonical_result(model, history, plan, query.control_names, **options)
        _assert_same_result(actual, expected)
        assert torch.equal(torch.get_rng_state(), after)
        assert actual is not query
        assert actual.mode == mode and actual.units == "outcome" and actual.n_samples == 4
        assert model.query_result is query and model.idata is stored
        xr.testing.assert_identical(stored, snapshot)

    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.predict(short.assign(search=-1.0), **options)
    with pytest.raises(ValueError, match="graph"):
        model.predict(short, n_samples=4, mode="fixed")
    assert torch.equal(torch.get_rng_state(), before)
    assert model.query_result is query and model.idata is stored
    xr.testing.assert_identical(stored, snapshot)
    # The next curve sweep must still use the stored full plan and its fixed graphs.
    torch.manual_seed(705)
    repeated = model.response_curves(levels=[0, 1], channels=["search"], weeks=2)
    for name in ("draws", "media", "dates", "levels"):
        np.testing.assert_array_equal(getattr(repeated, name), getattr(curves, name))
    assert repeated.units == "USD" and model.query_result is query
    xr.testing.assert_identical(stored, snapshot)


def test_context_marginal_depends_on_the_current_query_plan_and_horizon(model, history, future):
    layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
    graph = np.zeros(layout.n_slots)
    graph[layout.slices["cy"].start] = 1
    for revision, plan in enumerate(
        (future, future.assign(search=future["search"] * 4), future.iloc[:1]), start=1
    ):
        torch.manual_seed(710)
        tree = model.inference(
            history, plan, **COLUMN_ROLES, n_samples=3, mode="fixed", graph=graph
        )
        query = model.query_result
        after = torch.get_rng_state()
        torch.manual_seed(710)
        expected_query, expected_context = _canonical_pair(
            model, history, plan, query.control_names, n_samples=3, mode="fixed", graph=graph
        )
        assert torch.equal(torch.get_rng_state(), after)
        assert tree.attrs["inference_revision"] == revision
        _assert_same_result(query, expected_query)
        _assert_named_posterior(tree["context"]["posterior"], expected_context)
        for window in ("context", "query"):
            provenance = json.loads(tree[window].attrs["provenance"])
            assert provenance["data_sha256"] == expected_context.provenance["data_sha256"]
            assert provenance["input_usage"]["query"]["steps"] == len(plan)


def test_padding_names_do_not_collide_with_real_controls(model, history, future):
    torch.manual_seed(707)
    model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    initial = model.query_result
    real_name = initial.control_names[-1]
    renamed_history = history.rename(columns={"temperature": real_name})
    renamed_future = future.rename(columns={"temperature": real_name})
    torch.manual_seed(707)
    tree = model.inference(
        renamed_history,
        renamed_future,
        **{**COLUMN_ROLES, "control_columns": (real_name,)},
        n_samples=2,
    )
    renamed = model.query_result
    assert renamed.control_names[0] == real_name
    assert len(set(renamed.control_names)) == model.config.n_controls
    np.testing.assert_array_equal(renamed.components, initial.components)
    np.testing.assert_array_equal(renamed.graphs, initial.graphs)
    assert tree["query"]["posterior"]["control"].values.tolist() == [real_name]


def test_inference_adapts_channel_count_and_full_control_capacity(model, history, future):
    history = history.assign(price=[10.0, 11.0, 9.0, 12.0, 10.0])
    future = future.assign(price=[9.0, 10.0, 11.0])
    channels, controls = ("tv",), ("price", "temperature")
    roles = {**COLUMN_ROLES, "channel_columns": channels, "control_columns": controls}
    torch.manual_seed(710)
    tree = model.inference(history, future, **roles, n_samples=3)
    actual = model.query_result
    torch.manual_seed(710)
    expected, context = _canonical_pair(
        model,
        history,
        future,
        controls,
        channel_columns=channels,
        control_columns=controls,
        n_samples=3,
    )
    _assert_same_result(actual, expected)
    _assert_named_posterior(tree["context"]["posterior"], context)
    assert (
        actual.provenance["input_usage"]
        == model.input_usage(history, future, **roles)
        == {
            "context": {"steps": 5, "allocated_tokens": 20, "represented_tokens": 20},
            "query": {"steps": 3, "allocated_tokens": 12, "represented_tokens": 12},
            "total": {"allocated_tokens": 32, "represented_tokens": 32},
        }
    )
    assert actual.channel_names == channels and actual.control_names == controls
    control_axis = np.array([family == "control" for family in actual.component_families])
    np.testing.assert_array_equal(actual.component_mask[control_axis], [True, True])
    for window in ("context", "query"):
        assert tree[window]["constant_data"]["control"].values.tolist() == list(controls)


def test_loaded_weights_trainability_and_mode_are_unchanged_by_inference(
    model, history, future, tmp_path
):
    loaded = from_pretrained(save_pretrained(model, tmp_path / "source"))
    before_state = {name: value.clone() for name, value in loaded.state_dict().items()}
    before_trainability = [parameter.requires_grad for parameter in loaded.parameters()]
    assert not loaded.training
    assert loaded.input_usage(history, future, **COLUMN_ROLES) == REQUEST_USAGE
    assert not loaded.training
    loaded.train()
    torch.manual_seed(708)
    training_tree = loaded.inference(history, future, **COLUMN_ROLES, n_samples=3)
    training_result = loaded.query_result
    training_rng = torch.get_rng_state()
    assert loaded.training
    loaded.eval()
    torch.manual_seed(708)
    evaluation_tree = loaded.inference(history, future, **COLUMN_ROLES, n_samples=3)
    assert not loaded.training
    assert torch.equal(torch.get_rng_state(), training_rng)
    _assert_same_result(training_result, loaded.query_result)
    xr.testing.assert_equal(training_tree, evaluation_tree)
    assert loaded.state_dict().keys() == before_state.keys()
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in loaded.parameters()] == before_trainability
    assert all(parameter.grad is None for parameter in loaded.parameters())


def test_inference_requires_both_frames_and_explicit_roles_without_consuming_rng(
    model, history, future
):
    before = torch.get_rng_state()
    with pytest.raises(TypeError):
        model.inference(future, n_samples=2)
    with pytest.raises(TypeError):
        model.inference(history, future, n_samples=2)
    assert model.idata is None and model.query_result is None
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
def test_inference_rejects_invalid_role_schemas_before_rng_consumption(
    model, history, future, roles
):
    history = history.assign(extra_control=1.0, another_control=2.0)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.inference(history, future, **{**COLUMN_ROLES, **roles}, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem", ["not_dataframe", "duplicate_columns", "missing_target", "one_row"]
)
def test_inference_rejects_invalid_context_table_shapes(model, history, future, problem):
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
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    ("column", "value"),
    [("search", -1e-50), ("sales", 0.0), ("temperature", np.inf), ("tv", 1 + 2j)],
)
def test_inference_validates_raw_context_numeric_domains(model, history, future, column, value):
    values = history[column].to_numpy(
        dtype=complex if isinstance(value, complex) else float, copy=True
    )
    values[0] = value
    history[column] = values
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


def test_preflight_and_inference_validate_canonical_context_representation(model, history, future):
    history["sales"] = [1e-300, 1e300, 1e300, 1e300, 1e300]
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.input_usage(history, **COLUMN_ROLES)
    with pytest.raises(ValueError):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem", ["missing", "invalid", "numeric", "aware", "duplicate", "irregular"]
)
def test_inference_rejects_invalid_or_irregular_context_dates(model, history, future, problem):
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
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem",
    ["not_dataframe", "missing_channel", "empty", "negative_spend", "infinite", "text_control"],
)
def test_inference_validates_query_schema_and_domains_before_rng_consumption(
    model, history, future, problem
):
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
        model.input_usage(history, future, **COLUMN_ROLES)
    with pytest.raises(ValueError):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "problem",
    ["missing", "aware", "duplicate", "overlap", "start_gap", "internal_gap", "single_off_grid"],
)
def test_inference_rejects_dates_outside_the_immediate_regular_horizon(
    model, history, future, problem
):
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
        model.input_usage(history, future, **COLUMN_ROLES)
    with pytest.raises(ValueError):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("column", ["search", "temperature"])
def test_nonzero_query_inputs_cannot_use_zero_context_scales(model, history, future, column):
    history = history.assign(**{column: 0.0})
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="zero context"):
        model.input_usage(history, future, **COLUMN_ROLES)
    with pytest.raises(ValueError, match="zero context"):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
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
    torch.manual_seed(709)
    tree = model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    actual = model.query_result
    torch.manual_seed(709)
    expected, context = _canonical_pair(model, history, future, actual.control_names, n_samples=2)
    _assert_same_result(actual, expected)
    _assert_named_posterior(tree["context"]["posterior"], context)
    np.testing.assert_array_equal(actual.time, np.array([query_date], dtype="datetime64[ns]"))


@pytest.mark.parametrize("window", ["context", "query"])
def test_mixed_numeric_datetime_objects_cannot_be_coerced_to_date_labels(
    model, history, future, window
):
    before = torch.get_rng_state()
    if window == "context":
        history = history.iloc[:2].copy()
        history["date"] = pd.Series([0.0, pd.Timestamp("1970-01-02")], dtype=object)
    else:
        future["date"] = pd.Series(
            [float(future["date"].iloc[0].value), *future["date"].iloc[1:]], dtype=object
        )
    with pytest.raises(ValueError):
        model.input_usage(history, future, **COLUMN_ROLES)
    with pytest.raises(ValueError):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    assert torch.equal(torch.get_rng_state(), before)


def test_input_usage_preserves_complete_inference_curves_weights_mode_and_seeded_draws(
    model, history, future
):
    # A constant represented channel also provides the known float16 overflow case.
    history = history.assign(search=1.0)
    future = future.assign(search=1.0)
    torch.manual_seed(711)
    model.inference(history, future, **COLUMN_ROLES, n_samples=3)
    query = model.query_result
    model.response_curves(levels=[0, 1], channels=["search"], weeks=1, n_samples=2)
    stored = model.idata
    original = stored.copy(deep=True)
    model.train()
    trainability = [parameter.requires_grad for parameter in model.parameters()]

    def assert_preserved(state, rng):
        assert model.training
        assert model.idata is stored and model.query_result is query
        assert torch.equal(torch.get_rng_state(), rng)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        assert [parameter.requires_grad for parameter in model.parameters()] == trainability
        assert all(parameter.grad is None for parameter in model.parameters())
        xr.testing.assert_identical(stored, original)

    state = {name: value.clone() for name, value in model.state_dict().items()}
    before = torch.get_rng_state()
    # Explicit preflight can inspect a different candidate without replacing the run.
    assert model.input_usage(history.iloc[1:], **COLUMN_ROLES) == {
        "context": {"steps": 4, "allocated_tokens": 20, "represented_tokens": 16}
    }
    assert model.input_usage(history, future.iloc[:1], **COLUMN_ROLES) == {
        "context": {"steps": 5, "allocated_tokens": 25, "represented_tokens": 20},
        "query": {"steps": 1, "allocated_tokens": 5, "represented_tokens": 4},
        "total": {"allocated_tokens": 30, "represented_tokens": 24},
    }
    with pytest.raises(ValueError):
        model.input_usage(history, future.assign(search=-1e-50), **COLUMN_ROLES)
    assert_preserved(state, before)
    torch.manual_seed(711)
    reference = _canonical_result(model, history, future, query.control_names, n_samples=3)
    _assert_same_result(query, reference)

    # Inspection rejects narrowing overflow without touching the completed run.
    model.half()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    before = torch.get_rng_state()
    assert model.input_usage(history, future, **COLUMN_ROLES) == REQUEST_USAGE
    with pytest.raises(ValueError, match="qry_spend_z.*finite model representation"):
        model.input_usage(history, future.assign(search=1e100), **COLUMN_ROLES)
    assert_preserved(state, before)


@pytest.mark.parametrize("mode", ["model_average", "map"])
def test_idata_named_values_inputs_and_provenance_match_both_marginals(
    model, history, future, mode
):
    import arviz_base

    annotations = {
        "scenario": {"name": "board forecast", "versions": [1, {"approved": True}]},
        "input_usage": {"context": {"steps": -1}},
    }
    options = {
        "n_samples": 5,
        "mode": mode,
        "units": "USD",
        "provenance": annotations,
    }
    if mode == "map":
        options["router_samples"] = 7
    torch.manual_seed(812)
    tree = model.inference(history.iloc[::-1], future.iloc[::-1], **COLUMN_ROLES, **options)
    query = model.query_result
    after = torch.get_rng_state()
    torch.manual_seed(812)
    expected_query, context = _canonical_pair(
        model, history, future, query.control_names, **options
    )
    assert torch.equal(torch.get_rng_state(), after)
    _assert_same_result(query, expected_query)
    assert model.idata is tree and model.query_result is query
    assert set(tree.children) == {"context", "query"}
    assert len(tree.coords) == 0
    assert tree.attrs["schema_version"] == 1
    assert tree.attrs["inference_revision"] == 1
    assert tree.attrs["date_column"] == "date" and tree.attrs["target_column"] == "sales"
    assert json.loads(tree.attrs["channel_columns"]) == list(CHANNELS)
    assert json.loads(tree.attrs["control_columns"]) == list(CONTROLS)
    for window, result, frame in (("context", context, history), ("query", query, future)):
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
        _assert_named_posterior(posterior, result)
        constants = subtree["constant_data"]
        assert constants["media"].dims == ("date", "channel")
        assert constants["controls"].dims == ("date", "control")
        np.testing.assert_array_equal(constants["date"].values, result.time)
        np.testing.assert_array_equal(constants["media"].values, frame[list(CHANNELS)].to_numpy())
        np.testing.assert_array_equal(
            constants["controls"].values, frame[list(CONTROLS)].to_numpy()
        )
        endpoints = np.array(result.provenance["edge_endpoints"])[result.edge_mask]
        np.testing.assert_array_equal(constants["edge_source"].values, endpoints[:, 0])
        np.testing.assert_array_equal(constants["edge_target"].values, endpoints[:, 1])
        for attrs in (subtree.attrs, posterior.attrs):
            assert attrs["window"] == window
            assert attrs["mode"] == mode
            assert attrs["units"] == "USD"
            assert attrs["data_sha256"] == result.provenance["data_sha256"]
            assert attrs["mcmc_diagnostics"] == "not_applicable"
            provenance = json.loads(attrs["provenance"])
            assert provenance == result.provenance
            assert provenance["scenario"] == annotations["scenario"]
            assert provenance["input_usage"] == REQUEST_USAGE
        extracted = arviz_base.extract(subtree, var_names="channel_contribution")
        np.testing.assert_array_equal(
            extracted.sel(channel="search").values,
            posterior["channel_contribution"].sel(channel="search").values[0].T,
        )
    assert set(tree["context"].children) == {"posterior", "constant_data", "observed_data"}
    assert set(tree["query"].children) == {"posterior", "constant_data"}
    assert tree["context"]["observed_data"]["outcome"].dims == ("date",)
    np.testing.assert_array_equal(
        tree["context"]["observed_data"]["outcome"].values, history["sales"].to_numpy()
    )


@needs_arviz
def test_idata_window_subtrees_support_optional_arviz_statistics(model, history, future):
    from arviz_stats import summary

    tree = model.inference(history, future, **COLUMN_ROLES, n_samples=5)
    for window in ("context", "query"):
        subtree = tree[window]
        table = summary(subtree, var_names=["channel_contribution"], kind="stats", round_to="none")
        np.testing.assert_allclose(
            table["mean"].to_numpy(),
            subtree["posterior"]["channel_contribution"].values.mean(axis=(0, 1)).ravel(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_return_idata_false_stores_the_same_complete_seeded_analysis(model, history, future):
    torch.manual_seed(819)
    first = model.inference(
        history,
        future.assign(sales=[150.0, np.nan, 160.0]),
        **COLUMN_ROLES,
        n_samples=4,
        return_idata=True,
    )
    first_query = model.query_result
    first_copy = first.copy(deep=True)
    first_rng = torch.get_rng_state()
    assert first is model.idata
    assert first.attrs["inference_revision"] == 1

    torch.manual_seed(819)
    returned = model.inference(
        history,
        future.assign(sales=[150.0, np.nan, 160.0]),
        **COLUMN_ROLES,
        n_samples=4,
        return_idata=False,
    )
    stored, query = model.idata, model.query_result
    assert returned is None
    assert stored is not None and stored is not first and query is not first_query
    assert stored.attrs["inference_revision"] == 2
    assert set(stored.children) == {"context", "query"}
    assert torch.equal(torch.get_rng_state(), first_rng)
    xr.testing.assert_equal(first, stored)
    _assert_same_result(query, first_query)
    assert query.provenance == first_query.provenance
    torch.manual_seed(819)
    expected_query, expected_context = _canonical_pair(
        model, history, future, query.control_names, n_samples=4
    )
    assert torch.equal(torch.get_rng_state(), first_rng)
    _assert_same_result(query, expected_query)
    _assert_named_posterior(stored["context"]["posterior"], expected_context)
    _assert_named_posterior(stored["query"]["posterior"], expected_query)
    before = torch.get_rng_state()
    assert model.idata is stored and model.idata is stored and model.query_result is query
    assert torch.equal(torch.get_rng_state(), before)
    xr.testing.assert_identical(first, first_copy)


@pytest.mark.parametrize("return_idata", [True, False])
def test_reruns_replace_complete_snapshots_and_curves_without_reusing_context(
    model, history, future, return_idata
):
    first = model.inference(history, future.assign(sales=155.0), **COLUMN_ROLES, n_samples=4)
    first_query = model.query_result
    model.response_curves(levels=[0, 1], weeks=2, n_samples=2)
    with_curves = model.idata
    snapshots = [(first, first.copy(deep=True)), (with_curves, with_curves.copy(deep=True))]
    replacement = history.iloc[1:].rename(columns={"temperature": "price"})
    replacement = replacement.assign(sales=replacement["sales"] * 1.5)
    short = future.iloc[:1].rename(columns={"temperature": "price"}).assign(search=75.0)
    roles = {
        **COLUMN_ROLES,
        "channel_columns": ("tv", "search"),
        "control_columns": ("price",),
    }
    previous_tree, previous_query = with_curves, first_query

    # Identical successive requests still sample both marginals from their own RNG state.
    for revision in (2, 3):
        start = torch.get_rng_state()
        returned = model.inference(
            replacement,
            short,
            **roles,
            n_samples=2,
            mode="map",
            router_samples=5,
            units="EUR",
            provenance={"scenario": {"name": "replacement"}},
            return_idata=return_idata,
        )
        tree, query = model.idata, model.query_result
        after = torch.get_rng_state()
        assert returned is tree if return_idata else returned is None
        assert tree is not previous_tree and query is not previous_query
        assert tree.attrs["inference_revision"] == revision
        assert set(tree.children) == {"context", "query"}
        assert set(tree["query"].children) == {"posterior", "constant_data"}
        assert tree["context"]["posterior"].sizes["date"] == 4
        assert tree["query"]["posterior"].sizes["date"] == 1
        assert tree["query"]["posterior"]["draw"].values.tolist() == [0, 1]
        assert tree["query"]["posterior"]["channel"].values.tolist() == ["tv", "search"]
        assert tree["query"]["posterior"]["control"].values.tolist() == ["price"]
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(start)
            expected_query, expected_context = _canonical_pair(
                model,
                replacement,
                short,
                query.control_names,
                channel_columns=("tv", "search"),
                control_columns=("price",),
                n_samples=2,
                mode="map",
                router_samples=5,
                units="EUR",
                provenance={"scenario": {"name": "replacement"}},
            )
            assert torch.equal(torch.get_rng_state(), after)
        _assert_same_result(query, expected_query)
        _assert_named_posterior(tree["context"]["posterior"], expected_context)
        _assert_named_posterior(tree["query"]["posterior"], expected_query)
        for old, snapshot in snapshots:
            xr.testing.assert_identical(old, snapshot)
        snapshots.append((tree, tree.copy(deep=True)))
        previous_tree, previous_query = tree, query


@pytest.mark.parametrize(
    "problem",
    [
        "context",
        "query",
        "roles",
        "count",
        "mode",
        "graph",
        "object_metadata",
        "nonfinite_metadata",
        "scalar_identity",
    ],
)
def test_failed_candidate_validation_preserves_complete_run_and_curves(
    model, history, future, problem
):
    model.inference(history, future, **COLUMN_ROLES, n_samples=3)
    query = model.query_result
    model.response_curves(levels=[0, 1], channels=["search"], weeks=1, n_samples=2)
    stored = model.idata
    snapshot = stored.copy(deep=True)
    context = history.assign(sales=history["sales"] * 1.5)
    plan = future.assign(search=75.0)
    roles, options = dict(COLUMN_ROLES), {"n_samples": 2, "return_idata": False}
    if problem == "context":
        context = context.assign(sales=0.0)
    elif problem == "query":
        plan = plan.assign(date=plan["date"] + pd.Timedelta(days=7))
    elif problem == "roles":
        roles["control_columns"] = ("search",)
    elif problem == "count":
        options["n_samples"] = 0
    elif problem == "mode":
        options["mode"] = "unknown"
    elif problem == "graph":
        layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
        options.update(mode="fixed", graph=np.full(layout.n_slots, 0.5))
    elif problem == "object_metadata":
        options["provenance"] = {"scenario": {"name": object()}}
    elif problem == "scalar_identity":
        options["provenance"] = {"state_sha256": 1}
    else:
        options["provenance"] = {"scenario": {"versions": [1, np.inf]}}
    before = torch.get_rng_state()
    with pytest.raises((TypeError, ValueError)):
        model.inference(context, plan, **roles, **options)
    assert torch.equal(torch.get_rng_state(), before)
    assert model.idata is stored and model.query_result is query
    xr.testing.assert_identical(stored, snapshot)
    torch.manual_seed(820)
    predicted = model.predict(future, n_samples=2)
    torch.manual_seed(820)
    expected = _canonical_result(model, history, future, query.control_names, n_samples=2)
    _assert_same_result(predicted, expected)
    assert model.idata is stored and model.query_result is query


def test_idata_snapshots_inputs_nested_provenance_and_each_fixed_graph(
    model, history, future, tmp_path
):
    layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
    graphs = np.zeros((3, 1, layout.n_slots))
    graphs[0, 0, layout.slices["cy"].start] = 1
    graphs[1, 0, layout.slices["cy"].start + 1] = 1
    original_graphs = graphs.copy()
    annotations = {"scenario": {"name": "original", "versions": [1, {"approved": True}]}}
    torch.manual_seed(821)
    tree = model.inference(
        history,
        future,
        **COLUMN_ROLES,
        n_samples=3,
        mode="fixed",
        graph=graphs,
        provenance=annotations,
    )
    query = model.query_result
    snapshot = tree.copy(deep=True)
    np.testing.assert_array_equal(query.graphs, original_graphs[:, 0])
    torch.manual_seed(821)
    expected_query, expected_context = _canonical_pair(
        model,
        history,
        future,
        query.control_names,
        n_samples=3,
        mode="fixed",
        graph=original_graphs,
        provenance=annotations,
    )
    _assert_same_result(query, expected_query)
    _assert_named_posterior(tree["context"]["posterior"], expected_context)
    context_metadata = json.loads(
        (expected_context.save(tmp_path / "context") / "result.json").read_text()
    )
    metadata = json.loads((query.save(tmp_path / "query") / "result.json").read_text())
    restored_query = DecompositionResult.load(tmp_path / "query")
    _assert_same_result(restored_query, query)
    assert restored_query.provenance == query.provenance
    query_dates = query.time.copy()
    graphs[:] = 1
    annotations["scenario"]["name"] = object()
    annotations["scenario"]["versions"][1]["approved"] = False
    future["date"] = "changed after inference"
    future.loc[:, [*CHANNELS, *CONTROLS]] = -999.0
    history["sales"] = 0.0
    assert tree is model.idata and query is model.query_result
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
        provenance = json.loads(tree[window].attrs["provenance"])
        assert provenance["scenario"] == {
            "name": "original",
            "versions": [1, {"approved": True}],
        }
        assert provenance["input_usage"] == REQUEST_USAGE
    xr.testing.assert_identical(tree, snapshot)


def test_named_idata_preserves_odd_column_names_and_zero_fixed_graphs(model, history, future):
    channels = ("baseline", "paid:検索 / social")
    controls = ("__padded_control_1__",)
    rename = dict(zip((*CHANNELS, *CONTROLS), (*channels, *controls), strict=True))
    layout = EdgeSlotLayout(len(channels), model.config.n_controls, model.config.n_demand)
    tree = model.inference(
        history.rename(columns=rename),
        future.rename(columns=rename),
        **{**COLUMN_ROLES, "channel_columns": channels, "control_columns": controls},
        n_samples=3,
        mode="fixed",
        graph=np.zeros(layout.n_slots),
    )
    assert json.loads(tree.attrs["channel_columns"]) == list(channels)
    assert json.loads(tree.attrs["control_columns"]) == list(controls)
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


@pytest.mark.parametrize("stage", ["query", "context", "conversion"])
@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("return_idata", [True, False])
def test_sampling_and_conversion_failures_publish_no_partial_run(
    model, history, future, monkeypatch, stage, populated, return_idata
):
    stored = query = snapshot = None
    if populated:
        model.inference(history, future, **COLUMN_ROLES, n_samples=3)
        query = model.query_result
        model.response_curves(levels=[0, 1], channels=["search"], weeks=1, n_samples=2)
        stored = model.idata
        snapshot = stored.copy(deep=True)
    context = history.assign(sales=history["sales"] * 1.5)
    plan = future.iloc[:1].assign(search=75.0)
    model.train()
    before = torch.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        if stage == "query":
            _canonical_result(model, context, plan, None, n_samples=2)
        else:
            _canonical_pair(model, context, plan, None, n_samples=2)
        expected_rng = torch.get_rng_state()
    with monkeypatch.context() as failure:
        _inject_inference_failure(failure, model, stage)
        with pytest.raises(RuntimeError, match=f"{stage} .*failed"):
            model.inference(context, plan, **COLUMN_ROLES, n_samples=2, return_idata=return_idata)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not torch.equal(before, expected_rng)
    assert model.idata is stored and model.query_result is query
    assert model.training
    if populated:
        xr.testing.assert_identical(stored, snapshot)
        torch.manual_seed(822)
        predicted = model.predict(future, n_samples=2)
        torch.manual_seed(822)
        expected = _canonical_result(model, history, future, query.control_names, n_samples=2)
        _assert_same_result(predicted, expected)
        assert model.idata is stored and model.query_result is query
    else:
        after_failure = torch.get_rng_state()
        with pytest.raises(RuntimeError, match=r"inference\(context, query"):
            model.predict(plan, n_samples=2)
        assert torch.equal(torch.get_rng_state(), after_failure)

    # A retry must start from scratch, not retain either unpublished candidate marginal.
    retry_start = torch.get_rng_state()
    returned = model.inference(
        context, plan, **COLUMN_ROLES, n_samples=2, return_idata=return_idata
    )
    tree, completed = model.idata, model.query_result
    retry_rng = torch.get_rng_state()
    assert returned is tree if return_idata else returned is None
    assert tree is not stored and completed is not query
    assert tree.attrs["inference_revision"] == (2 if populated else 1)
    assert set(tree.children) == {"context", "query"}
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(retry_start)
        expected_query, expected_context = _canonical_pair(
            model, context, plan, completed.control_names, n_samples=2
        )
        assert torch.equal(torch.get_rng_state(), retry_rng)
    _assert_same_result(completed, expected_query)
    _assert_named_posterior(tree["context"]["posterior"], expected_context)
    assert model.training
    if populated:
        xr.testing.assert_identical(stored, snapshot)


@pytest.mark.parametrize(
    "role", ["date_column", "target_column", "channel_columns", "control_columns"]
)
def test_nul_role_labels_are_rejected_before_sampling(model, history, future, role):
    roles = dict(COLUMN_ROLES)
    original = roles[role]
    name = original if isinstance(original, str) else original[0]
    invalid = name + "\x00"
    roles[role] = invalid if isinstance(original, str) else (invalid, *original[1:])
    context = history.rename(columns={name: invalid})
    plan = future.rename(columns={name: invalid})
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="NUL"):
        model.inference(context, plan, **roles, n_samples=2)
    assert model.idata is None and model.query_result is None
    assert torch.equal(torch.get_rng_state(), before)


def test_predict_failure_after_sampling_preserves_the_full_run_and_curves(
    model, history, future, monkeypatch
):
    model.inference(history, future, **COLUMN_ROLES, n_samples=3)
    query = model.query_result
    model.response_curves(levels=[0, 1], channels=["search"], weeks=1, n_samples=2)
    stored = model.idata
    snapshot = stored.copy(deep=True)
    plan = future.iloc[:1].assign(search=75.0)
    torch.manual_seed(823)
    expected = _canonical_result(model, history, plan, query.control_names, n_samples=2)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(823)
    with monkeypatch.context() as failure:
        _inject_inference_failure(failure, model, "query")
        with pytest.raises(RuntimeError, match="query sampling failed"):
            model.predict(plan, n_samples=2)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert model.query_result is query and model.idata is stored
    xr.testing.assert_identical(stored, snapshot)
    torch.manual_seed(823)
    _assert_same_result(model.predict(plan, n_samples=2), expected)
    assert model.query_result is query and model.idata is stored


def test_weight_bundles_exclude_complete_sessions_curves_and_predictions(
    model, history, future, tmp_path
):
    baseline = save_pretrained(model, tmp_path / "baseline")
    sentinel = "runtime-only-forecast-label"
    roles = {**COLUMN_ROLES, "channel_columns": (sentinel, "tv")}
    context = history.rename(columns={"search": sentinel})
    plan = future.rename(columns={"search": sentinel}).assign(sales=[157.0, np.nan, 163.0])
    model.inference(
        context,
        plan,
        **roles,
        n_samples=3,
        provenance={"scenario": {"name": sentinel}},
        return_idata=False,
    )
    query = model.query_result
    curves = model.response_curves(levels=[0, 1], channels=[sentinel], weeks=2, n_samples=2)
    stored = model.idata
    assert curves.channels == (sentinel,)
    assert stored["response_curves"]["posterior"].sizes["draw"] == 2
    assert sentinel in stored["query"]["constant_data"]["channel"].values
    independent = model.predict(plan.iloc[:1].assign(**{sentinel: 91.0}), n_samples=2)
    assert independent is not query and model.query_result is query and model.idata is stored
    with_session = save_pretrained(model, tmp_path / "with-session")
    assert {path.name for path in with_session.iterdir()} == {
        "config.json",
        "manifest.json",
        "model.safetensors",
    }
    assert (baseline / "config.json").read_bytes() == (with_session / "config.json").read_bytes()
    before_manifest = json.loads((baseline / "manifest.json").read_text())
    after_manifest = json.loads((with_session / "manifest.json").read_text())
    # Metadata key order in the safetensors container need not preserve its file digest.
    # Compare every semantic manifest field and the actual container metadata/weights.
    for manifest in (before_manifest, after_manifest):
        del manifest["bundle_sha256"]
        del manifest["files"]["model.safetensors"]["sha256"]
    assert before_manifest == after_manifest
    with (
        safe_open(baseline / "model.safetensors", framework="pt") as before_file,
        safe_open(with_session / "model.safetensors", framework="pt") as after_file,
    ):
        assert before_file.metadata() == after_file.metadata()
    for name in ("config.json", "manifest.json"):
        assert sentinel not in (with_session / name).read_text()
    before_weights = load_file(baseline / "model.safetensors")
    after_weights = load_file(with_session / "model.safetensors")
    assert before_weights.keys() == after_weights.keys()
    for name, value in after_weights.items():
        torch.testing.assert_close(value, before_weights[name], rtol=0, atol=0)
    restored = from_pretrained(with_session)
    assert restored.idata is None and restored.query_result is None
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match=r"inference\(context, query"):
        restored.predict(plan, n_samples=2)
    with pytest.raises(RuntimeError, match="inference"):
        restored.response_curves(levels=[0, 1])
    assert restored.input_usage(context, plan, **roles) == REQUEST_USAGE
    assert restored.idata is None and restored.query_result is None
    assert torch.equal(torch.get_rng_state(), before)


def test_independent_predictions_cannot_mutate_loaded_artifact_identity(
    model, history, future, tmp_path
):
    loaded = from_pretrained(save_pretrained(model, tmp_path / "model", profile="synthetic"))
    tree = loaded.inference(history, future, **COLUMN_ROLES, n_samples=2)
    query = loaded.query_result
    original_identity = deepcopy(loaded.artifact_metadata)
    original_provenance = deepcopy(query.provenance)
    original_tree = tree.copy(deep=True)

    prediction = loaded.predict(future, n_samples=2)
    prediction.provenance["artifact"]["profile"] = "independent edit"
    prediction.provenance["artifact"]["manifest"]["files"]["config.json"]["sha256"] = "f" * 64

    assert loaded.artifact_metadata == original_identity
    assert query.provenance == original_provenance
    assert loaded.idata is tree and loaded.query_result is query
    xr.testing.assert_identical(tree, original_tree)


def test_loaded_recorded_model_replays_both_windows_and_curves_without_derived_provenance(
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
    annotations = {"scenario": {"name": "loaded"}}
    torch.manual_seed(824)
    tree = loaded.inference(history, future, **COLUMN_ROLES, n_samples=3, provenance=annotations)
    query = loaded.query_result
    after = torch.get_rng_state()
    torch.manual_seed(824)
    expected_query, expected_context = _canonical_pair(
        loaded, history, future, query.control_names, n_samples=3, provenance=annotations
    )
    assert torch.equal(torch.get_rng_state(), after)
    _assert_same_result(query, expected_query)
    _assert_named_posterior(tree["context"]["posterior"], expected_context)
    assert loaded.idata is tree
    for window in ("context", "query"):
        assert tree[window].attrs["state_sha256"] == loaded.provenance["state_sha256"]
        assert tree[window].attrs["data_sha256"] == query.provenance["data_sha256"]
        assert json.loads(tree[window].attrs["provenance"])["scenario"] == {"name": "loaded"}
    # Neither caller nor public-result metadata is replayable sampling configuration.
    annotations["scenario"]["name"] = "caller edit"
    query.provenance["scenario"]["name"] = "result edit"
    query.provenance["inference_config"]["n_samples"] = 999
    torch.manual_seed(825)
    curves = loaded.response_curves(levels=[0, 1], channels=["search"], weeks=1, n_samples=2)
    torch.manual_seed(825)
    reference = _canonical_result(
        loaded,
        history,
        future.iloc[:1],
        query.control_names,
        n_samples=2,
        provenance={"scenario": {"name": "loaded"}},
    )
    component = reference.component_names.index("channel:search")
    np.testing.assert_array_equal(
        curves.draws[0, 1], reference.components[:, :, component].sum(axis=1)
    )
    assert loaded.query_result is query
    assert json.loads(loaded.idata["query"].attrs["provenance"])["scenario"] == {"name": "loaded"}
    curve_tree = loaded.idata["response_curves"]
    assert json.loads(curve_tree.attrs["source_provenance"]) == json.loads(
        tree["query"].attrs["provenance"]
    )
    assert curve_tree.attrs["source_inference_revision"] == tree.attrs["inference_revision"]
    assert curve_tree.attrs["mode"] == "model_average"
    assert curve_tree.attrs["units"] == "outcome"
    assert curve_tree.attrs["n_samples"] == 2 and curve_tree.attrs["weeks"] == 1
    assert loaded.artifact_metadata["provenance"]["training_evaluation"] == recorded


def _seeding_options(model, mode):
    options = {"n_samples": 3, "mode": mode}
    layout = EdgeSlotLayout(len(CHANNELS), model.config.n_controls, model.config.n_demand)
    graph = np.zeros(layout.n_slots)
    graph[layout.slices["cy"].start] = 1
    if mode == "fixed":
        options["graph"] = graph
    elif mode == "map":
        options["router_samples"] = 5
    return options


@pytest.mark.parametrize("mode", ["model_average", "map", "fixed"])
def test_a_seed_equals_manual_seed_before_the_call(model, history, future, mode):
    options = _seeding_options(model, mode)
    torch.manual_seed(0)
    expected = model.inference(history, future, **COLUMN_ROLES, **options)
    torch.manual_seed(99)
    caller_state = torch.get_rng_state()
    seeded = model.inference(history, future, **COLUMN_ROLES, **options, random_seed=0)
    # The seeded call draws the stream of manual_seed(0) from its own generator.
    assert torch.equal(torch.get_rng_state(), caller_state)
    xr.testing.assert_equal(seeded["context"], expected["context"])
    xr.testing.assert_equal(seeded["query"], expected["query"])
    predicted = model.predict(future, **options, random_seed=0)
    assert torch.equal(torch.get_rng_state(), caller_state)
    torch.manual_seed(0)
    _assert_same_result(model.predict(future, **options), predicted)


def test_a_seed_repeats_inference_and_predict_draws(model, history, future):
    first = model.inference(history, future, **COLUMN_ROLES, n_samples=4, random_seed=11)
    first_predicted = model.predict(future, n_samples=4, random_seed=12)
    second = model.inference(history, future, **COLUMN_ROLES, n_samples=4, random_seed=11)
    xr.testing.assert_equal(first, second)
    _assert_same_result(model.predict(future, n_samples=4, random_seed=12), first_predicted)
    other = model.inference(history, future, **COLUMN_ROLES, n_samples=4, random_seed=13)
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(other["query"], first["query"])


def test_no_seed_draws_from_and_advances_the_callers_generator(model, history, future):
    torch.manual_seed(3)
    before = torch.get_rng_state()
    unseeded = model.inference(history, future, **COLUMN_ROLES, n_samples=4)
    after = torch.get_rng_state()
    assert not torch.equal(after, before)
    seeded = model.inference(history, future, **COLUMN_ROLES, n_samples=4, random_seed=3)
    xr.testing.assert_equal(unseeded["query"], seeded["query"])
    assert torch.equal(torch.get_rng_state(), after)
    # Without a seed a second call continues the stream and draws new numbers.
    again = model.inference(history, future, **COLUMN_ROLES, n_samples=4)
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(again["query"], unseeded["query"])


def test_a_failed_seeded_call_leaves_the_generator_unchanged(model, history, future, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("context sampling failed")

    monkeypatch.setattr(model, "_sample_context", fail)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="context sampling failed"):
        model.inference(history, future, **COLUMN_ROLES, n_samples=4, random_seed=7)
    assert torch.equal(torch.get_rng_state(), before)


def _second_model():
    from pymc_mktfm.model.config import ModelConfig
    from pymc_mktfm.model.model import MFM

    torch.manual_seed(77)
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=2
    )
    return MFM(config).eval()


@pytest.mark.parametrize("mode", ["model_average", "map"])
def test_a_seeded_call_ignores_other_models_and_global_torch_use(
    model, history, future, monkeypatch, mode
):
    options = {"n_samples": 3, "mode": mode, "random_seed": 7}
    model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    isolated = model.predict(future, **options)
    other = _second_model()
    other.inference(history, future, **COLUMN_ROLES, n_samples=2)
    head = type(model.head)
    draw_noise = head._draw_noise
    interleaved = []

    def interleave(*args, **kwargs):
        # Everything a second caller can do to the global state, between seed and draws.
        if not interleaved:
            interleaved.append(True)
            other.predict(future, n_samples=2, random_seed=99)
            other.predict(future, n_samples=2)
            torch.manual_seed(5)
            torch.rand(3)
        return draw_noise(*args, **kwargs)

    monkeypatch.setattr(head, "_draw_noise", staticmethod(interleave))
    replayed = model.predict(future, **options)
    assert interleaved
    _assert_same_result(replayed, isolated)


def test_a_seeded_call_leaves_the_global_generator_state_unchanged(model, history, future):
    torch.manual_seed(4)
    before = torch.get_rng_state()
    model.inference(history, future, **COLUMN_ROLES, n_samples=3, random_seed=1)
    model.predict(future, n_samples=3, random_seed=1)
    model.response_curves(levels=[0, 2], weeks=1, n_samples=2, random_seed=1)
    assert torch.equal(torch.get_rng_state(), before)


def _inactive_node_graph():
    """A fixed graph with an edge on the second control slot, which the data leaves padded."""
    layout = EdgeSlotLayout(len(CHANNELS), 2, 2)
    graph = np.zeros(layout.n_slots)
    graph[layout.slices["zb"].start + 1] = 1
    return graph


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"mode": "bogus"}, id="mode"),
        pytest.param({"mode": "fixed"}, id="missing-graph"),
        pytest.param({"graph": np.zeros(3)}, id="graph-without-fixed"),
        pytest.param({"mode": "fixed", "graph": np.zeros(3)}, id="graph-wrong-shape"),
        pytest.param({"mode": "fixed", "graph": _inactive_node_graph()}, id="graph-inactive-node"),
        pytest.param({"n_samples": 0}, id="n_samples"),
        pytest.param({"units": ""}, id="units"),
        pytest.param({"provenance": "text"}, id="provenance"),
    ],
)
def test_an_invalid_argument_with_a_seed_leaves_the_generator_unchanged(
    model, history, future, options
):
    torch.manual_seed(1)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        model.inference(
            history, future, **COLUMN_ROLES, **{"n_samples": 2, **options}, random_seed=7
        )
    assert torch.equal(torch.get_rng_state(), before)
    assert model.idata is None


@pytest.mark.parametrize("seed", [None, 7])
@pytest.mark.parametrize("mode", ["model_average", "map", "fixed"])
def test_a_request_with_no_active_demand_entity_changes_no_random_state(
    model, history, future, mode, seed
):
    options = _seeding_options(model, mode)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="no active demand entities"):
        _canonical_result(
            model,
            history,
            future,
            ("temperature", "pad"),
            latent_mask=np.zeros(model.config.n_demand, dtype=bool),
            random_seed=seed,
            **options,
        )
    assert torch.equal(torch.get_rng_state(), before)


def test_the_public_decompose_signature_has_no_private_parameters():
    import inspect

    names = list(inspect.signature(decompose).parameters)
    assert names[-1] == "random_seed" and not [name for name in names if name.startswith("_")]


def _raw_world(model):
    steps = 8
    rng = np.random.default_rng(0)
    return {
        "spend": rng.uniform(1.0, 5.0, (steps, 2)),
        "controls": rng.uniform(0.0, 1.0, (steps, model.config.n_controls)),
        "context_outcome": np.arange(5.0) + 100.0,
    }


def _encoded_world(model, case):
    """An encoded batch that the model must reject, or the raw inputs when that is the way in."""
    raw = _raw_world(model)
    if case == "raw-no-media":
        # observation_batch refuses this itself (api.py), before any model code runs.
        return {**raw, "channel_mask": np.zeros(2, dtype=bool)}
    batch = observation_batch(**raw, model_config=model.config)
    if case == "encoded-no-latent":
        batch["active_j_mask"] = torch.zeros_like(batch["active_j_mask"])
    else:
        batch["active_c_mask"] = torch.zeros_like(batch["active_c_mask"])
    return batch


@pytest.mark.parametrize("mode", ["model_average", "map", "fixed"])
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("encoded-no-latent", "no active demand entities"),
        ("encoded-no-media", "no active media channels"),
        ("raw-no-media", "at least one represented entity"),
    ],
)
def test_a_rejected_encoded_or_media_request_draws_nothing(model, mode, case, message):
    options = _seeding_options(model, mode)
    observations = _encoded_world(model, case)
    torch.manual_seed(1)
    before = torch.get_rng_state()
    for seed in (None, 7):
        with pytest.raises(ValueError, match=message):
            decompose(model, observations, random_seed=seed, **options)
    assert torch.equal(torch.get_rng_state(), before)
    # A seeded call owns its generator, so a shared one shows whether anything drew.
    generator = torch.Generator().manual_seed(3)
    state = generator.get_state()
    with pytest.raises(ValueError, match=message):
        _decompose(model, observations, generator=generator, **options)
    assert torch.equal(generator.get_state(), state)


def test_the_seed_is_in_the_tree_and_not_in_saved_evidence_or_its_digest(
    model, history, future, tmp_path
):
    tree = model.inference(history, future, **COLUMN_ROLES, n_samples=3, random_seed=7)
    saved = {}
    for name, result in {
        "predict7": model.predict(future, n_samples=3, random_seed=7),
        "predict8": model.predict(future, n_samples=3, random_seed=8),
        "unseeded": model.predict(future, n_samples=3),
        "query": model.query_result,
    }.items():
        text = (result.save(tmp_path / name) / "result.json").read_text()
        saved[name] = json.loads(text)
        assert "random_seed" not in text
        assert DecompositionResult.load(tmp_path / name).provenance == result.provenance
    # The seed changes the draws and nothing in the metadata, so the digest is the same.
    assert saved["predict7"]["metadata_sha256"] == saved["predict8"]["metadata_sha256"]
    assert saved["predict7"]["metadata"] == saved["unseeded"]["metadata"]
    assert saved["predict7"]["arrays_sha256"] != saved["predict8"]["arrays_sha256"]
    # The tree records the seed; its window digests are those of the saved evidence.
    assert tree.attrs["random_seed"] == 7
    assert tree["query"].attrs["metadata_sha256"] == saved["query"]["metadata_sha256"]
    assert "random_seed" not in tree["query"].attrs and "random_seed" not in tree["context"].attrs


def test_a_rejected_request_starts_no_noise_prefetch(model, history, future, monkeypatch):
    # Force the GPU path's prefetch on the CPU: noise is "drawn elsewhere" than the device.
    from pymc_mktfm.model import flow
    from pymc_mktfm.model import model as model_module

    requests = []
    request = flow._NoisePrefetch.request
    monkeypatch.setattr(
        flow._NoisePrefetch,
        "request",
        lambda self, *args: requests.append(args) or request(self, *args),
    )
    monkeypatch.setattr(model_module, "_noise_device", lambda device: torch.device("meta"))
    for seed in (None, 7):
        with pytest.raises(ValueError, match="no active demand entities"):
            _canonical_result(
                model,
                history,
                future,
                ("temperature", "pad"),
                latent_mask=np.zeros(model.config.n_demand, dtype=bool),
                random_seed=seed,
            )
    assert requests == []


def test_an_invalid_seed_leaves_every_module_mode_alone(model, history, future):
    model.encoder.train()
    modes = [module.training for module in model.modules()]
    with pytest.raises(ValueError, match="random_seed"):
        _canonical_result(model, history, future, ("temperature", "pad"), random_seed=-1)
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize("seed", [-1, 2**31, True, 1.5])
def test_invalid_seeds_fail_before_sampling(model, history, future, seed):
    model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="random_seed"):
        model.inference(history, future, **COLUMN_ROLES, n_samples=2, random_seed=seed)
    with pytest.raises(ValueError, match="random_seed"):
        model.predict(future, n_samples=2, random_seed=seed)
    with pytest.raises(ValueError, match="random_seed"):
        model.response_curves(n_samples=2, random_seed=seed)
    assert torch.equal(torch.get_rng_state(), before)
    assert model.idata.attrs["inference_revision"] == 1


def test_the_root_records_the_seed_and_earlier_trees_keep_theirs(model, history, future):
    first = model.inference(history, future, **COLUMN_ROLES, n_samples=2, random_seed=5)
    second = model.inference(history, future, **COLUMN_ROLES, n_samples=2)
    third = model.inference(history, future, **COLUMN_ROLES, n_samples=2, random_seed=0)
    assert first.attrs["random_seed"] == 5
    assert "random_seed" not in second.attrs
    assert third.attrs["random_seed"] == 0
    assert first.attrs["random_seed"] == 5


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"units": ""}, id="empty-units"),
        pytest.param({"units": "\ud800"}, id="surrogate-units"),
        pytest.param({"provenance": {"bad": float("nan")}}, id="nan-provenance"),
        pytest.param({"provenance": {"bad": "\ud800"}}, id="surrogate-provenance"),
    ],
)
def test_decompose_checks_units_and_provenance_before_seeding(model, history, future, options):
    torch.manual_seed(1)
    before = torch.get_rng_state()
    with pytest.raises(ValueError):
        _canonical_result(
            model, history, future, ("temperature", "pad"), n_samples=2, random_seed=7, **options
        )
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "names",
    [
        pytest.param(("temperature",), id="too-few"),
        pytest.param(("pad", "pad"), id="repeated"),
        pytest.param((1, 2), id="not-strings"),
    ],
)
def test_decompose_checks_entity_names_before_seeding(model, history, future, names):
    torch.manual_seed(1)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="names"):
        _canonical_result(model, history, future, names, n_samples=2, random_seed=7)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("seed", [None, 7])
def test_decompose_accepts_one_shot_name_iterators(model, history, future, seed):
    names = ("temperature", "pad")
    torch.manual_seed(2)
    expected = _canonical_result(model, history, future, names, n_samples=2, random_seed=seed)
    torch.manual_seed(2)
    result = _canonical_result(
        model,
        history,
        future,
        iter(names),
        channel_names=(name for name in CHANNELS),
        n_samples=2,
        random_seed=seed,
    )
    _assert_same_result(result, expected)
    assert result.channel_names == CHANNELS and result.control_names == names


def test_decompose_accepts_any_provenance_mapping(model, history, future):
    from types import MappingProxyType

    result = _canonical_result(
        model,
        history,
        future,
        ("temperature", "pad"),
        n_samples=2,
        provenance=MappingProxyType({"scenario": "plan"}),
    )
    assert result.provenance["scenario"] == "plan"
