"""Exercise the same public model workflow with local, Hub, and River sources."""

import json
import re
import sys
import threading
import time
import traceback
import warnings

import httpx
import numpy as np
import pandas as pd
import pytest
from client.client_fixtures import (
    BASE_URL,
    FakeService,
    analysis_posts,
    curve_result,
    dataset_record,
    error_body,
    result_record,
    result_rows,
    river_draws,
    route_examples_to,
    started,
    status_record,
)

import pymc_mfm as mfm
from pymc_mfm.client import ApiError, Gone, Unauthorized

ROLES = {
    "date_column": "date",
    "target_column": "sales",
    "channel_columns": ("tv", "search"),
    "control_columns": ("price",),
}
DRAWS = 100
SEED = 17


@pytest.fixture
def frames():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-07-01", periods=6, freq="W-MON"),
            "sales": [510, 517, 509, 521, 1709, 1721],
            "tv": [10.0, 12.0, 10.0, 13.0, 11.0, 12.0],
            "search": [8.0, 10.0, 8.0, 11.0, 9.0, 10.0],
            "price": [2.0, 2.1, 2.2, 2.0, 2.1, 2.0],
        }
    )
    return data.iloc[:4], data.iloc[4:]


def run_model(model, history, future):
    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    assert result is model.result
    mfm.validate_idata(result.idata)
    for window, frame in (("context", history), ("query", future)):
        assert result.idata[f"/{window}/posterior"].sizes["draw"] == DRAWS
        np.testing.assert_array_equal(result.idata[f"/{window}/posterior"].date, frame.date)
        np.testing.assert_array_equal(result.idata[f"/{window}/observed_data"].outcome, frame.sales)
    # These native plot calls cannot require the engine or model weights.
    plt = pytest.importorskip("matplotlib.pyplot")
    for method in (result.plot.inputs, result.plot.fit, result.plot.contributions_over_time):
        figure = method(backend="matplotlib")
        assert figure.axes
        plt.close(figure)
    return result


def test_local_directory_and_hub_have_the_same_seeded_workflow(
    model, frames, tmp_path, monkeypatch
):
    xr = pytest.importorskip("xarray")
    hub = pytest.importorskip("huggingface_hub")
    from pymc_mfm.inference import save_pretrained

    commit = "a" * 40
    snapshot = tmp_path / "snapshots" / commit
    snapshot.parent.mkdir()
    save_pretrained(model, snapshot)
    requests = []

    def download(**options):
        requests.append(options)
        return str(snapshot)

    monkeypatch.setattr(hub, "snapshot_download", download)
    folder = mfm.load(snapshot, **ROLES)
    direct = run_model(folder, *frames)
    assert not requests
    repo = mfm.load("test-owner/example", revision=commit, token="example-hub-key", **ROLES)
    downloaded = run_model(repo, *frames)
    for name in ("context", "query"):
        xr.testing.assert_equal(
            direct.idata[f"/{name}/posterior"], downloaded.idata[f"/{name}/posterior"]
        )
    assert {request["repo_id"] for request in requests} == {"test-owner/example"}
    assert {request["token"] for request in requests} == {"example-hub-key"}
    curve_result_local = folder.response_curves(levels=(0, 1, 2), weeks=1, random_seed=SEED)
    assert set(direct.idata.children) == {"context", "query"}
    assert "posterior" in curve_result_local.idata["response_curves"].children


def draws_content(tmp_path, history, future, **options):
    """The baseline draws file River serves for ``history`` and ``future``: ``DRAWS`` draws."""
    return river_draws(tmp_path, history.date, future.date, draws=DRAWS, **options)


def test_hosted_workflow_uses_only_results_and_sweeps_on_the_service(frames, tmp_path, monkeypatch):
    pytest.importorskip("h5netcdf")
    history, future = frames
    content = draws_content(tmp_path, history, future)
    signed_url = "https://files.example.com/baseline.nc?sig=example"
    baseline = result_record(result_rows("baseline", "baseline", url=signed_url))
    baseline["meta"].update({"draws": DRAWS, "seed": SEED})
    service = FakeService()
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    service.on(
        "POST",
        "/v1/datasets",
        (410, error_body("gone", "This history expired")),
        (201, dataset_record()),
    )
    service.on(
        "POST", "/v1/analyses", (202, started(1)), (202, {**started(5), "analysis_id": "an_2"})
    )
    service.on("GET", "/v1/analyses/an_1", (200, status_record(runs_total=1)))
    service.on("GET", "/v1/analyses/an_2", (200, status_record(runs_total=5)))
    service.on("GET", "/v1/analyses/an_1/result", (200, baseline))
    service.on("GET", "/v1/analyses/an_2/result", (200, curve_result((0.0, 2.0))))
    downloads = []

    def serve(request):
        if request.url.host == "api.example.com":
            assert request.headers["Authorization"] == "Bearer example-river-key"
            return service(request)
        assert str(request.url) == signed_url
        assert "Authorization" not in request.headers
        downloads.append(request)
        return httpx.Response(200, content=content)

    monkeypatch.setenv("MFM_API_URL", BASE_URL)
    monkeypatch.setenv("MFM_API_KEY", "must-not-override-explicit-key")
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(serve)}),
    )

    with monkeypatch.context() as block_weights:
        for name in ("torch", "safetensors", "huggingface_hub"):
            block_weights.setitem(sys.modules, name, None)
        model = mfm.load("river-1", engine="hosted", token="example-river-key", **ROLES)
        assert not service.requests
        block_weights.setenv("MFM_API_URL", "https://other.example.com")
        with pytest.raises(mfm.NotSupported, match="draws per window"):
            model.inference(history, future, n_samples=99)
        assert not service.requests
        with pytest.raises(mfm.NotSupported, match="weights"):
            _ = model.network
        with pytest.warns(UserWarning, match="history is shorter than 75 weeks"):
            result = run_model(model, history, future)
        xr = pytest.importorskip("xarray")
        posterior = result.idata["/query/posterior"].ds
        assert posterior.channel_contribution.dtype == np.float64
        assert posterior.channel_contribution.sel(channel="tv").isel(chain=0, draw=0, date=0) == 23
        curves = model.response_curves(levels=(0, 1, 2), weeks=1, random_seed=SEED)
        assert model.result is curves and "response_curves" not in result.idata.children
        xr.testing.assert_identical(curves.idata["query"], result.idata["query"])
        mfm.validate_idata(curves.idata)
        figure = curves.plot.response_curves(backend="matplotlib")
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [0, 100, 200])
        pytest.importorskip("matplotlib.pyplot").close(figure)
        with pytest.raises(mfm.NotSupported, match=r"only support interval=0\.9"):
            curves.plot.response_curves(interval=0.5)

    inference, sweep = analysis_posts(service)
    assert set(inference["plan"]) == {"dates", "spend", "controls"}
    assert inference["explain_history"] is True and inference["seed"] == SEED
    assert sweep["plan"]["dates"] == inference["plan"]["dates"][:1]
    assert set(item["channel"] for item in sweep["curves"]) == {"tv", "search"}
    expired, recreated, next_upload = service.sent("POST", "/v1/datasets")
    assert expired.headers["Idempotency-Key"] == next_upload.headers["Idempotency-Key"]
    assert expired.headers["Idempotency-Key"] != recreated.headers["Idempotency-Key"]
    assert len(downloads) == 1  # curves are reported bands, not reconstructed draws


RIVER_KEY = "example-river-key"
SIGNED_URL = "https://files.example.com/baseline.nc?sig=example"


def baseline_result(notes=()):
    """The result of a hosted inference: its baseline draws are at ``SIGNED_URL``."""
    record = result_record(
        result_rows("baseline", "baseline", url=SIGNED_URL, warnings=list(notes))
    )
    record["meta"].update({"draws": DRAWS, "seed": SEED})
    return record


def river(*later, dataset_warnings=(), baseline_warnings=()):
    """A scripted River: one inference (an_1), then one analysis per result in ``later``."""
    dataset = dataset_record()
    dataset["warnings"] = list(dataset_warnings)
    service = FakeService()
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    service.on("POST", "/v1/datasets", (201, dataset))
    ids = [f"an_{number}" for number in range(1, len(later) + 2)]
    service.on("POST", "/v1/analyses", *((202, {**started(), "analysis_id": a}) for a in ids))
    for analysis_id, record in zip(ids, (baseline_result(baseline_warnings), *later), strict=True):
        service.on("GET", f"/v1/analyses/{analysis_id}", (200, status_record()))
        service.on("GET", f"/v1/analyses/{analysis_id}/result", (200, record))
    return service


def open_river(monkeypatch, service, draws, **roles):
    """A hosted model whose calls reach ``service``; the signed draws URL answers ``draws``."""
    real_client = httpx.Client

    def serve(request):
        if request.url.host == "api.example.com":
            return service(request)
        assert str(request.url) == SIGNED_URL
        return draws if isinstance(draws, httpx.Response) else httpx.Response(200, content=draws)

    monkeypatch.setenv("MFM_API_URL", BASE_URL)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(serve)}),
    )
    return mfm.load("river-1", engine="hosted", token=RIVER_KEY, **{**ROLES, **roles})


def assert_names_analysis(error, analysis_id):
    """A failure after River accepted ``analysis_id`` says how to inspect it."""
    notes = "\n".join(getattr(error, "__notes__", ()))
    assert "Calling the model again starts a new analysis" in notes
    assert f'pymc_mfm.client.Client(base_url=..., api_key=...).get("{analysis_id}")' in notes


def raised_here(record):
    """The messages of the UserWarnings in ``record`` that point at this file's calls.

    These are River's warnings, in order; a dependency's warnings point elsewhere.
    """
    return [
        str(item.message)
        for item in record
        if issubclass(item.category, UserWarning) and item.filename == __file__
    ]


def sweep(levels, *, failed=(), notes=None, base_notes=None):
    """A curve sweep result for tv and search, whose contribution is 100 times the level.

    ``failed`` points have no numbers. ``notes`` maps (channel, level) to the warnings of
    that plan, and ``base_notes`` maps a baseline component to the warnings of its row.
    """
    rows = result_rows("baseline", "baseline")
    for row in rows:
        row["warnings"] = list((base_notes or {}).get(row["component"], ()))
    for channel in ("tv", "search"):
        for level in levels:
            rows += result_rows(
                f"curve:{channel}:baseline:{level:g}",
                "curve",
                base="baseline",
                channel=channel,
                level=level,
                failed=(channel, level) in failed,
                warnings=(notes or {}).get((channel, level)),
                **{f"{channel}_mean": 100.0 * level},
            )
    return result_record(rows)


def numeric_labels(history, future):
    week = pd.Timedelta(weeks=1).value  # 1970 weeks if the labels were read as nanoseconds
    labels = np.arange(len(history) + len(future), dtype=object) * week
    return (
        history.assign(date=labels[: len(history)]),
        future.assign(date=labels[len(history) :]),
    )


HISTORY_DAYS = ("2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22")


def dated(history_days, plan_days):
    """An ``edit`` that gives the history and the plan these calendar days."""
    return lambda history, future: (
        history.assign(date=pd.to_datetime(list(history_days))),
        future.assign(date=pd.to_datetime(list(plan_days))),
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda history, future: (history, future.assign(tv=[np.nan, 12.0])), "missing or inf"),
        (numeric_labels, "not numeric row labels"),
        (
            lambda history, future: (
                history.assign(date=history.date.dt.tz_localize("UTC")),
                future.assign(date=future.date.dt.tz_localize("UTC")),
            ),
            "timezone-naive",
        ),
        (
            lambda history, future: (history.assign(sales=[510, 0, 509, 521]), future),
            "context outcomes must be positive; 'sales' has zero or negative values",
        ),
        (
            lambda history, future: (history.assign(tv=[10.0, -1.0, 10.0, 13.0]), future),
            r"context channel spend must be nonnegative; negative values in \['tv'\]",
        ),
        (
            lambda history, future: (history, future.assign(search=[9.0, -2.0])),
            r"query channel spend must be nonnegative; negative values in \['search'\]",
        ),
        (
            dated(
                ("2024-07-01", "2024-07-08", "2024-07-16", "2024-07-23"),
                ("2024-07-30", "2024-08-06"),
            ),
            re.escape(
                "Row 3 of your context has the date 2024-07-16, but row 2 has 2024-07-08. "
                "Dates must be 7 days apart."
            ),
        ),
        (
            dated(HISTORY_DAYS, ("2024-07-22", "2024-07-29")),
            re.escape(
                "Your history ends on 2024-07-22. Your plan starts on 2024-07-22. "
                "The next required date is 2024-07-29."
            ),
        ),
        (
            dated(HISTORY_DAYS, ("2024-08-05", "2024-08-12")),
            re.escape("Your plan starts on 2024-08-05. The next required date is 2024-07-29."),
        ),
        (
            dated(HISTORY_DAYS, ("2024-07-29", "2024-08-12")),
            re.escape(
                "Row 2 of your plan has the date 2024-08-12, but row 1 has 2024-07-29. "
                "The next required date is 2024-08-05."
            ),
        ),
        (
            lambda history, future: (history.assign(tv=0.0), future),
            re.escape("['tv'] are zero in every context week but not in the query"),
        ),
        (
            lambda history, future: (history.assign(price=0.0), future),
            re.escape("['price'] are zero in every context week but not in the query"),
        ),
        (
            lambda history, future: (history.assign(tv=history.tv.astype(str)), future),
            re.escape("'tv' must contain finite real numeric values"),
        ),
    ],
    ids=[
        "nan query spend",
        "numeric labels in an object column",
        "timezone-aware dates",
        "zero sales week",
        "negative history spend",
        "negative plan spend",
        "history with one 8-day step",
        "plan starting on the last history week",
        "plan a week after the history ends",
        "plan skipping a week",
        "channel with no history spend",
        "control zero throughout the history",
        "numbers stored as text",
    ],
)
def test_hosted_refuses_bad_inputs_before_any_request(frames, monkeypatch, edit, message):
    service = river()
    model = open_river(monkeypatch, service, b"")
    with pytest.raises(ValueError, match=message):
        model.inference(*edit(*frames), n_samples=DRAWS)
    assert service.requests == []  # no history list, upload or analysis
    assert model.result is None


@pytest.mark.parametrize(("column", "role"), [("tv", "spend"), ("price", "controls")])
def test_hosted_accepts_a_series_that_is_zero_in_history_and_plan(
    frames, tmp_path, monkeypatch, column, role
):
    history, future = (frame.assign(**{column: 0.0}) for frame in frames)
    service = river()
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    result = model.inference(history, future, n_samples=DRAWS)
    assert model.result is result
    (upload,) = [service.body(r) for r in service.sent("POST", "/v1/datasets")]
    assert upload[role][column] == [0.0] * len(history)


@pytest.mark.parametrize(
    ("token", "message"),
    [
        (None, "Pass token or set the MFM_API_KEY environment variable."),
        ("example river key", "Check token or MFM_API_KEY."),
    ],
    ids=["missing", "malformed"],
)
def test_hosted_credential_errors_name_the_token_parameter(monkeypatch, token, message):
    monkeypatch.setenv("MFM_API_URL", BASE_URL)
    monkeypatch.delenv("MFM_API_KEY", raising=False)
    with pytest.raises(ValueError, match=re.escape(message)) as caught:
        mfm.load("river-1", engine="hosted", token=token, **ROLES)
    assert "api_key" not in str(caught.value)  # load() has no api_key parameter


def test_hosted_rejected_key_keeps_guidance_for_the_accepted_analysis(frames, monkeypatch):
    service = river()
    service.on("GET", "/v1/analyses/an_1", (401, error_body("unauthorized", "Bad key")))
    model = open_river(monkeypatch, service, b"")

    with pytest.raises(Unauthorized) as caught:
        model.inference(*frames, n_samples=DRAWS)
    assert_names_analysis(caught.value, "an_1")
    rendered = "".join(traceback.format_exception(caught.value))
    assert "Check the River API key you supplied or set in MFM_API_KEY." in rendered
    assert "Connect page" in rendered
    assert RIVER_KEY not in rendered
    assert model.result is None


def test_local_load_refuses_the_river_key_as_a_hub_token(monkeypatch):
    hub_calls = []
    monkeypatch.setattr(mfm, "from_pretrained", lambda source, **options: hub_calls.append(options))
    monkeypatch.setenv("MFM_API_KEY", f" {RIVER_KEY}\n")
    with pytest.raises(ValueError, match="engine='hosted'") as caught:
        mfm.load("river-1", token=RIVER_KEY, **ROLES)
    assert hub_calls == []  # the River key never reaches the Hub client
    assert RIVER_KEY not in "".join(traceback.format_exception(caught.value))
    mfm.load("river-1", token="example-hub-key", **ROLES)
    assert [call["token"] for call in hub_calls] == ["example-hub-key"]


def test_local_river_1_without_hub_access_points_to_the_hosted_engine(monkeypatch):
    from pymc_mfm.inference.hub import HubArtifactError

    def denied(source, **options):
        raise HubArtifactError("access_denied_or_missing", "The Hub model is private.")

    monkeypatch.setattr(mfm, "from_pretrained", denied)
    with pytest.raises(HubArtifactError) as caught:
        mfm.load("river-1", **ROLES)
    assert 'mfm.load("river-1", engine="hosted"' in "\n".join(caught.value.__notes__)
    with pytest.raises(HubArtifactError) as other:
        mfm.load("example-org/other-model", **ROLES)
    assert not getattr(other.value, "__notes__", ())  # only the alias has a hosted twin


@pytest.mark.parametrize(
    ("source", "options", "error", "message"),
    [
        ("pymc-labs/mfm", {"engine": "hosted"}, mfm.NotSupported, "currently serves 'river-1'"),
        ("river-1", {"engine": "hosted", "profile": "river-1"}, mfm.NotSupported, "local engine"),
        ("river-1", {"engine": "hosted", "revision": "a" * 40}, mfm.NotSupported, "local engine"),
        ("river-1", {"engine": "hosted", "device": "cuda"}, mfm.NotSupported, "local engine"),
        ("river-1", {"engine": "hosted", "token": False}, TypeError, "API key as a string token"),
        ("river-1", {"base_url": BASE_URL}, ValueError, "base_url is only for the hosted engine"),
        ("river-1", {"engine": "Hosted"}, ValueError, "engine must be 'local' or 'hosted'"),
    ],
    ids=[
        "hosted source other than river-1",
        "hosted profile",
        "hosted revision",
        "hosted device",
        "hosted token=False",
        "local base_url",
        "engine typo",
    ],
)
def test_load_refuses_unsupported_arguments_before_any_request(
    monkeypatch, source, options, error, message
):
    service = FakeService()
    route_examples_to(service, monkeypatch)  # River credentials are set
    hub_calls = []
    monkeypatch.setattr(mfm, "from_pretrained", lambda source, **options: hub_calls.append(source))
    with pytest.raises(error, match=re.escape(message)):
        mfm.load(source, **options, **ROLES)
    assert service.requests == [] and hub_calls == []


@pytest.mark.parametrize(
    ("token", "variables"),
    [
        ("hf_example0hub0token", {}),
        (" hf_example0hub0token\n", {}),
        ("example-hub-key", {"HF_TOKEN": "example-hub-key"}),
        ("example-hub-key", {"HUGGING_FACE_HUB_TOKEN": " example-hub-key\n"}),
        (None, {"MFM_API_KEY": "hf_example0hub0token"}),
        (None, {"MFM_API_KEY": "example-hub-key", "HF_TOKEN": "example-hub-key"}),
    ],
    ids=[
        "hf_ form",
        "padded hf_ form",
        "equal to HF_TOKEN",
        "equal to HUGGING_FACE_HUB_TOKEN",
        "hf_ form in MFM_API_KEY",
        "MFM_API_KEY equal to HF_TOKEN",
    ],
)
def test_hosted_load_refuses_a_hugging_face_token(monkeypatch, token, variables):
    service = FakeService()
    route_examples_to(service, monkeypatch)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="is a Hugging Face token") as caught:
        mfm.load("river-1", engine="hosted", token=token, **ROLES)
    hub_token = (token or variables["MFM_API_KEY"]).strip()
    assert hub_token not in "".join(traceback.format_exception(caught.value))
    model = mfm.load("river-1", engine="hosted", token=RIVER_KEY, **ROLES)
    assert isinstance(model, mfm.Model)  # a River key still loads
    assert service.requests == []


def test_hosted_keeps_caller_timestamps_and_sends_calendar_days(frames, tmp_path, monkeypatch):
    history, future = frames
    content = draws_content(tmp_path, history, future)  # River labels calendar days
    noon = pd.Timedelta(hours=12)
    history = history.assign(date=history.date + noon)
    future = future.assign(date=future.date + noon)
    service = river(curve_result((0.0, 2.0)))
    model = open_river(monkeypatch, service, content)

    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    curves = model.response_curves(levels=(0, 1, 2), weeks=1)

    for window, frame in (("context", history), ("query", future)):
        for group in ("posterior", "constant_data", "observed_data"):
            dates = result.idata[f"/{window}/{group}"].date
            assert dates.dtype == np.dtype("datetime64[ns]")
            np.testing.assert_array_equal(dates, frame.date)
    np.testing.assert_array_equal(curves.idata["/response_curves/summary"].date, future.date[:1])
    mfm.validate_idata(curves.idata)
    days = ["2024-07-01", "2024-07-08", "2024-07-15", "2024-07-22"]
    assert [service.body(r)["dates"] for r in service.sent("POST", "/v1/datasets")] == [days] * 2
    inference, curve_sweep = analysis_posts(service)
    assert inference["plan"]["dates"] == ["2024-07-29", "2024-08-05"]
    assert curve_sweep["plan"]["dates"] == ["2024-07-29"]


def test_hosted_stores_numpy_role_names_as_builtin_str(frames, tmp_path, monkeypatch):
    history, future = frames
    roles = {
        "date_column": np.str_("date"),
        "target_column": np.str_("sales"),
        "channel_columns": [np.str_("tv"), np.str_("search")],
        "control_columns": np.array(["price"]),
    }
    model = open_river(monkeypatch, river(), draws_content(tmp_path, history, future), **roles)
    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    mfm.validate_idata(result.idata)  # save_idata runs the same check
    assert json.loads(result.idata.attrs["channel_columns"]) == ["tv", "search"]


def test_hosted_sweeps_send_the_inference_inputs_not_an_edited_tree(frames, tmp_path, monkeypatch):
    history, future = frames
    service = river(curve_result((0.0, 2.0)), curve_result((0.0, 2.0)))
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    observed = result.idata["/context/observed_data"]
    observed["outcome"] = observed["outcome"] * 2
    constants = result.idata["/query/constant_data"]
    constants["media"] = constants["media"] * 3

    first = model.response_curves(levels=(0, 1, 2), weeks=1)
    second = model.response_curves(levels=(0, 1, 2))  # sweeps the tree the model now holds

    assert model.result is second
    original, *swept = [service.body(r) for r in service.sent("POST", "/v1/datasets")]
    assert original["outcome"] == history.sales.tolist()
    assert swept == [original, original]
    inference, *sweeps = analysis_posts(service)
    plan = inference["plan"]
    assert sweeps[1]["plan"] == plan
    assert sweeps[0]["plan"] == {
        "dates": plan["dates"][:1],
        "spend": {name: values[:1] for name, values in plan["spend"].items()},
        "controls": {name: values[:1] for name, values in plan["controls"].items()},
    }
    np.testing.assert_array_equal(
        first.idata["/response_curves/summary"].total_media, future[["tv", "search"]].iloc[:1].sum()
    )


def test_hosted_sweep_keeps_gaps_and_the_previous_result_when_a_curve_fails(
    frames, tmp_path, monkeypatch
):
    history, future = frames
    partial = sweep((0.0, 0.5, 2.0), failed={("tv", 0.5)})
    no_search = sweep((0.0, 2.0), failed={("search", 0.0), ("search", 2.0)})
    service = river(partial, no_search)
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    model.inference(history, future, n_samples=DRAWS, random_seed=SEED)

    curves = model.response_curves(levels=(0, 0.5, 1, 2), weeks=1)
    summary = curves.idata["/response_curves/summary"]
    for name, values in (("mean", [0, np.nan, 100, 200]), ("low", [-10, np.nan, 90, 190])):
        np.testing.assert_array_equal(summary[name].sel(channel="tv"), values)
    np.testing.assert_array_equal(summary["mean"].sel(channel="search"), [0, 50, 50, 200])
    mfm.validate_idata(curves.idata)

    with pytest.raises(RuntimeError, match=r"an_3.*channels \['search'\]"):
        model.response_curves(levels=(0, 1, 2), weeks=1)
    assert model.result is curves


@pytest.mark.parametrize(
    "edit",
    [
        {"mean": None},
        {"p95": None},
        {"mean": None, "p05": None, "p95": None},
        {"p05": 220.0, "p95": 180.0},
    ],
    ids=["missing mean", "missing p95", "no statistics", "p05 above p95"],
)
def test_hosted_sweep_refuses_a_successful_point_without_a_band(
    frames, tmp_path, monkeypatch, edit
):
    history, future = frames
    record = sweep((0.0, 2.0))
    point = next(
        row
        for row in record["table"]
        if row["scenario_id"] == "curve:tv:baseline:2" and row["component"] == "tv"
    )
    assert point["status"] == "ok"
    point.update(edit)
    model = open_river(monkeypatch, river(record), draws_content(tmp_path, history, future))
    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    with pytest.raises(
        ValueError, match=r"'tv' at level 2 do not have a finite 90% band"
    ) as caught:
        model.response_curves(levels=(0, 1, 2), weeks=1)
    assert_names_analysis(caught.value, "an_2")
    assert model.result is result


@pytest.mark.parametrize("gap", ["no plan", "no channel row"])
def test_hosted_sweep_refuses_a_point_river_did_not_report(frames, tmp_path, monkeypatch, gap):
    history, future = frames
    record = sweep((0.0, 2.0))
    if gap == "no plan":
        levels, message = (0, 0.5, 1, 2), r"missing the 'tv' plan at level 0\.5"
    else:
        record["table"] = [
            row
            for row in record["table"]
            if (row["scenario_id"], row["component"]) != ("curve:tv:baseline:2", "tv")
        ]
        levels, message = (0, 1, 2), r"missing a successful 'tv' row at level 2"
    model = open_river(monkeypatch, river(record), draws_content(tmp_path, history, future))
    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    # Only a plan River reports failed may become a gap in the curve.
    with pytest.raises(ValueError, match=message) as caught:
        model.response_curves(levels=levels, weeks=1)
    assert_names_analysis(caught.value, "an_2")
    assert model.result is result


def test_hosted_results_keep_river_warnings(frames, tmp_path, monkeypatch):
    history, future = frames
    short = "history is shorter than 75 weeks"
    above = "tv spend is above the history's 95th percentile"
    curves_record = sweep(
        (0.0, 0.5, 2.0),
        failed={("tv", 0.5)},
        notes={
            ("tv", 0.5): ["a failed point keeps no warning"],
            ("tv", 2.0): [above, "tv spend is outside the history", above],
            ("search", 0.0): ["search spend is zero"],
        },
        base_notes={"tv": ["tv baseline warning"], "outcome": ["not a channel row"]},
    )
    service = river(curves_record, dataset_warnings=[short], baseline_warnings=[above, short])
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))

    with pytest.warns(UserWarning) as caught:
        result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    assert raised_here(caught) == [short, above]  # each points at the caller's line
    for window in ("context", "query"):
        provenance = json.loads(result.idata[window].attrs["provenance"])
        assert provenance["analysis_id"] == "an_1" and provenance["warnings"] == [short, above]

    with pytest.warns(UserWarning) as swept:
        curves = model.response_curves(levels=(0, 0.5, 1, 2), weeks=1)
    assert raised_here(swept) == [
        "tv baseline warning",
        above,
        "tv spend is outside the history",
        "search spend is zero",
    ]
    node = curves.idata["/response_curves"]
    assert json.loads(node.attrs["provenance"]) == {"engine": "hosted", "analysis_id": "an_2"}
    summary = node["summary"]
    assert summary["warning"].dtype.kind == "U"
    np.testing.assert_array_equal(
        summary["warning"].sel(channel=["tv", "search"]),
        [
            ["", "", "tv baseline warning", f"{above}\ntv spend is outside the history"],
            ["search spend is zero", "", "", ""],
        ],
    )
    np.testing.assert_array_equal(summary.date, future.date[:1])
    mfm.validate_idata(curves.idata)


def test_hosted_stores_and_emits_each_river_warning_on_one_line(frames, tmp_path, monkeypatch):
    history, future = frames
    wrapped = "tv spend is above\nthe history's  95th percentile\n"
    flat = "tv spend is above the history's 95th percentile"
    service = river(
        sweep((0.0, 2.0), notes={("tv", 2.0): [wrapped, flat]}), baseline_warnings=[wrapped, flat]
    )
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))

    with pytest.warns(UserWarning) as caught:
        result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    assert raised_here(caught) == [flat]
    assert json.loads(result.idata["query"].attrs["provenance"])["warnings"] == [flat]
    with pytest.warns(UserWarning) as swept:
        curves = model.response_curves(levels=(0, 1, 2), weeks=1)
    assert raised_here(swept) == [flat]
    summary = curves.idata["/response_curves/summary"]
    assert summary["warning"].sel(channel="tv", level=2.0).item() == flat
    mfm.validate_idata(curves.idata)


def test_hosted_warning_raised_as_an_error_keeps_a_result_that_can_be_swept(
    frames, tmp_path, monkeypatch
):
    history, future = frames
    short = "history is shorter than 75 weeks"
    service = river(
        baseline_result([short]),  # an_2: a second inference
        sweep((0.0, 2.0), notes={("tv", 2.0): [short]}),  # an_3
        sweep((0.0, 2.0)),  # an_4
        baseline_warnings=[short],
    )
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    with pytest.warns(UserWarning, match=short):
        kept = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=short) as inference_error:
            model.inference(history, future, n_samples=DRAWS)
        assert model.result is kept
        with pytest.raises(UserWarning, match=short) as sweep_error:
            model.response_curves(levels=(0, 1, 2), weeks=1)
        assert model.result is kept
    assert_names_analysis(inference_error.value, "an_2")
    assert_names_analysis(sweep_error.value, "an_3")

    curves = model.response_curves(levels=(0, 1, 2), weeks=1)
    source = json.loads(curves.idata["/response_curves"].attrs["source_provenance"])
    assert model.result is curves and source["analysis_id"] == "an_1"


@pytest.mark.parametrize(
    "failure", ["draws download", "draw count", "draws decomposition", "sweep status"]
)
def test_hosted_failure_after_acceptance_names_the_analysis(frames, tmp_path, monkeypatch, failure):
    history, future = frames
    content = draws_content(tmp_path, history, future)
    service = river(curve_result((0.0, 2.0)))
    accepted, kept = "an_1", None
    if failure == "draws download":
        model = open_river(monkeypatch, service, httpx.Response(404))
        with pytest.raises(ApiError) as caught:
            model.inference(history, future, n_samples=DRAWS)
    elif failure == "draw count":
        model = open_river(monkeypatch, service, content)  # the file holds DRAWS draws
        with pytest.raises(ValueError, match=f"expected {2 * DRAWS} draws, got {DRAWS}") as caught:
            model.inference(history, future, n_samples=2 * DRAWS)
    elif failure == "draws decomposition":
        # Each outcome draw is one more than the sum of its contributions.
        model = open_river(
            monkeypatch, service, draws_content(tmp_path, history, future, outcome_shift=1.0)
        )
        with pytest.raises(ValueError, match="contributions do not sum to its outcome") as caught:
            model.inference(history, future, n_samples=DRAWS)
    else:
        service.on("GET", "/v1/analyses/an_2", (410, error_body("gone", "Analysis expired")))
        model = open_river(monkeypatch, service, content)
        kept = model.inference(history, future, n_samples=DRAWS)
        with pytest.raises(Gone) as caught:
            model.response_curves(levels=(0, 1, 2), weeks=1)
        accepted = "an_2"
    assert_names_analysis(caught.value, accepted)
    shown = "".join(traceback.format_exception(caught.value))
    assert RIVER_KEY not in shown and "sig=" not in shown
    assert model.result is kept


@pytest.mark.parametrize(
    ("status", "error", "message"),
    [
        ("failed", RuntimeError, "Hosted analysis 'an_1' has no successful baseline run"),
        ("deleted", Gone, "Analysis an_1 was deleted"),
    ],
)
def test_hosted_inference_stops_at_once_when_river_ends_the_analysis(
    frames, tmp_path, monkeypatch, status, error, message
):
    history, future = frames
    service = river()
    # The baseline failed and two runs look active: this service sends no runs_cancelled.
    service.on("GET", "/v1/analyses/an_1", (200, status_record(status, runs_done=0, runs_failed=1)))
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("River had ended the analysis"))
    with pytest.raises(error, match=message) as caught:
        model.inference(history, future, n_samples=DRAWS)
    assert_names_analysis(caught.value, "an_1")
    assert len(service.sent("GET", "/v1/analyses/an_1")) == 1  # one read, no polling
    assert model.result is None


def test_hosted_inference_polls_a_running_analysis_until_it_ends(frames, tmp_path, monkeypatch):
    history, future = frames
    service = river()
    running = (200, status_record("running"))
    service.on("GET", "/v1/analyses/an_1", running, running, (200, status_record()))
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    result = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)

    assert len(service.sent("GET", "/v1/analyses/an_1")) == 3
    assert len(sleeps) == 2 and all(seconds > 0 for seconds in sleeps)
    assert service.sent("GET", "/v1/analyses/an_1/result")  # read only after River ended it
    assert model.result is result


def test_hosted_calls_during_a_running_inference_raise_model_busy(frames, tmp_path, monkeypatch):
    history, future = frames
    service = river()
    entered, release = threading.Event(), threading.Event()
    held = []

    def hold_the_first_request(request):
        if not held:
            held.append(request)
            entered.set()
            release.wait(timeout=10)
        return service(request)

    model = open_river(
        monkeypatch, hold_the_first_request, draws_content(tmp_path, history, future)
    )
    outcome = {}

    def infer():
        try:
            outcome["result"] = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
        except BaseException as error:  # checked below, in the test's thread
            outcome["error"] = error

    running = threading.Thread(target=infer)
    running.start()
    try:
        assert entered.wait(timeout=10)  # the inference now waits inside its first request
        with pytest.raises(mfm.ModelBusy):
            model.inference(history, future, n_samples=DRAWS)
        with pytest.raises(mfm.ModelBusy):
            model.response_curves(levels=(0, 1, 2), weeks=1)
        assert len(held) == 1 and service.requests == []  # neither call sent anything
    finally:
        release.set()
        running.join(timeout=10)
    assert not running.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert model.result is outcome["result"]
    assert len(analysis_posts(service)) == 1


def test_hosted_refuses_more_than_2000_draws_before_any_request(frames, tmp_path, monkeypatch):
    history, future = frames
    service = river()
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    with pytest.raises(mfm.NotSupported, match="draws per window"):
        model.inference(history, future, n_samples=2001)
    assert service.requests == [] and model.result is None

    result = model.inference(history, future, n_samples=DRAWS)
    sent = len(service.requests)
    with pytest.raises(mfm.NotSupported, match="draws per window"):
        model.response_curves(levels=(0, 1, 2), weeks=1, n_samples=2001)
    assert len(service.requests) == sent and model.result is result


def test_each_unseeded_hosted_inference_is_a_new_analysis_with_a_new_seed(
    frames, tmp_path, monkeypatch
):
    history, future = frames
    service = river(baseline_result())
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    results = [model.inference(history, future, n_samples=DRAWS) for _ in range(2)]

    seeds = [body["seed"] for body in analysis_posts(service)]
    assert all(type(seed) is int and 0 <= seed < 2**31 for seed in seeds)
    assert seeds[0] != seeds[1]  # River's default seed would repeat the same draws
    assert [result.idata.attrs["random_seed"] for result in results] == seeds
    keys = [r.headers["Idempotency-Key"] for r in service.sent("POST", "/v1/analyses")]
    assert all(key.startswith("mfm1-fresh-") for key in keys) and keys[0] != keys[1]


def test_a_failed_hosted_inference_keeps_the_previous_result(frames, tmp_path, monkeypatch):
    history, future = frames
    failed = result_record(result_rows("baseline", "baseline", failed=True))
    service = river(failed, sweep((0.0, 2.0)))  # an_2 fails, an_3 sweeps
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    first = model.inference(history, future, n_samples=DRAWS, random_seed=SEED)

    corrected = history.assign(sales=history.sales * 2)
    with pytest.raises(RuntimeError, match="'an_2' has no successful baseline run"):
        model.inference(corrected, future, n_samples=DRAWS)
    assert model.result is first and model.idata is first.idata

    curves = model.response_curves(levels=(0, 1, 2), weeks=1)  # sweeps the kept inference
    source = json.loads(curves.idata["/response_curves"].attrs["source_provenance"])
    assert source["analysis_id"] == "an_1"
    kept, refused, swept = [service.body(r) for r in service.sent("POST", "/v1/datasets")]
    assert swept == kept != refused


def observed_outcome(figure):
    """The observed outcome line of a matplotlib ``plot.fit`` figure."""
    (line,) = [line for line in figure.axes[0].lines if line.get_label() == "Observed outcome"]
    return line.get_ydata()


def test_model_plot_draws_the_latest_hosted_result(frames, tmp_path, monkeypatch):
    history, future = frames
    service = river(baseline_result())
    model = open_river(monkeypatch, service, draws_content(tmp_path, history, future))
    model.inference(history, future, n_samples=DRAWS, random_seed=SEED)
    plots = model.plot  # kept across inferences, as a notebook keeps a name
    before = plots.fit(backend="matplotlib")
    np.testing.assert_array_equal(observed_outcome(before), [*history.sales, *future.sales])

    corrected = history.assign(sales=history.sales * 2)
    model.inference(corrected, future, n_samples=DRAWS, random_seed=SEED)
    for charts in (plots, model.plot):
        figure = charts.fit(backend="matplotlib")
        np.testing.assert_array_equal(observed_outcome(figure), [*corrected.sales, *future.sales])
