"""Exercise the same public model workflow with local, Hub, and River sources."""

import sys

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
