"""Exercise the same public model workflow with local, Hub, and River sources."""

import json
import sys
import traceback

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
    started,
    status_record,
)

import pymc_mfm as mfm
from pymc_mfm.client import ApiError, Gone

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


def draws_content(tmp_path, history, future):
    """Synthetic named-variable `river-draws/1` file, including the history marginal."""
    xr = pytest.importorskip("xarray")
    draws = np.arange(DRAWS, dtype=np.float32)
    variables = {}
    coords = {"draw": np.arange(DRAWS), "channel": ["search", "tv"], "control": ["price"]}
    for prefix, time, frame in (("history_", "history_date", history), ("", "date", future)):
        coords[time] = frame.date.to_numpy()
        values = draws[:, None] + np.arange(len(frame), dtype=np.float32)[None, :]
        variables[prefix + "baseline"] = (("draw", time), values + 450)
        variables[prefix + "channel_contribution"] = (
            ("draw", time, "channel"),
            np.stack((values + 11, values + 23), axis=-1),
        )
        variables[prefix + "control_contribution"] = (("draw", time, "control"), values[..., None])
        variables[prefix + "latent"] = (("draw", time), values * np.float32(0.25))
        variables[prefix + "outcome"] = (("draw", time), 484 + values * np.float32(4.25))
    file = tmp_path / "draws.nc"
    xr.Dataset(variables, coords=coords, attrs={"schema": "river-draws/1"}).to_netcdf(
        file, engine="h5netcdf"
    )
    return file.read_bytes()


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


def river(*sweeps, dataset_warnings=(), baseline_warnings=()):
    """A scripted River: one inference (an_1), then one analysis per result in ``sweeps``."""
    dataset = dataset_record()
    dataset["warnings"] = list(dataset_warnings)
    baseline = result_record(
        result_rows("baseline", "baseline", url=SIGNED_URL, warnings=list(baseline_warnings))
    )
    baseline["meta"].update({"draws": DRAWS, "seed": SEED})
    service = FakeService()
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    service.on("POST", "/v1/datasets", (201, dataset))
    ids = [f"an_{number}" for number in range(1, len(sweeps) + 2)]
    service.on("POST", "/v1/analyses", *((202, {**started(), "analysis_id": a}) for a in ids))
    for analysis_id, record in zip(ids, (baseline, *sweeps), strict=True):
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
    ],
    ids=["nan query spend", "numeric labels in an object column", "timezone-aware dates"],
)
def test_hosted_refuses_bad_inputs_before_any_request(frames, monkeypatch, edit, message):
    service = river()
    model = open_river(monkeypatch, service, b"")
    with pytest.raises(ValueError, match=message):
        model.inference(*edit(*frames), n_samples=DRAWS)
    assert service.requests == []  # no history list, upload or analysis
    assert model.result is None


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
    [{"mean": None}, {"p95": None}, {"p05": 220.0, "p95": 180.0}],
    ids=["missing mean", "missing p95", "p05 above p95"],
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
    with pytest.raises(ValueError, match=r"'tv' at level 2 do not have a finite 90% band"):
        model.response_curves(levels=(0, 1, 2), weeks=1)
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
    assert [str(item.message) for item in caught] == [short, above]
    assert {item.filename for item in caught} == {__file__}  # points at the caller's line
    for window in ("context", "query"):
        provenance = json.loads(result.idata[window].attrs["provenance"])
        assert provenance["analysis_id"] == "an_1" and provenance["warnings"] == [short, above]

    curves = model.response_curves(levels=(0, 0.5, 1, 2), weeks=1)
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


@pytest.mark.parametrize("failure", ["draws download", "sweep status"])
def test_hosted_failure_after_acceptance_names_the_resume_call(
    frames, tmp_path, monkeypatch, failure
):
    history, future = frames
    content = draws_content(tmp_path, history, future)
    service = river(curve_result((0.0, 2.0)))
    if failure == "draws download":
        model = open_river(monkeypatch, service, httpx.Response(404))
        with pytest.raises(ApiError) as caught:
            model.inference(history, future, n_samples=DRAWS)
        accepted, kept = "an_1", None
    else:
        service.on("GET", "/v1/analyses/an_2", (410, error_body("gone", "Analysis expired")))
        model = open_river(monkeypatch, service, content)
        kept = model.inference(history, future, n_samples=DRAWS)
        with pytest.raises(Gone) as caught:
            model.response_curves(levels=(0, 1, 2), weeks=1)
        accepted = "an_2"
    notes = "\n".join(getattr(caught.value, "__notes__", ()))
    assert f'pymc_mfm.client.Client().get("{accepted}")' in notes
    shown = "".join(traceback.format_exception(caught.value))
    assert RIVER_KEY not in shown and "sig=" not in shown
    assert model.result is kept
