"""The hosted model: ``from_pretrained(api_key=)`` on a fake HTTP transport, no network."""

import inspect
import subprocess
import sys
import textwrap

import httpx
import numpy as np
import pandas as pd
import pytest

import pymc_mfm
from pymc_mfm import hosted
from pymc_mfm.client import ClientError, ValidationError

from .client_fixtures import (
    BASE_URL,
    FakeService,
    dataset_record,
    result_record,
    result_rows,
    started,
    status_record,
)

xr = pytest.importorskip("xarray")
pytest.importorskip("h5netcdf")

N_DRAWS = 100
CHANNELS = ("tv", "search")
ROLES = {
    "date_column": "week",
    "target_column": "sales",
    "channel_columns": CHANNELS,
    "control_columns": ["price"],
}
PLAN_WEEKS = pd.date_range("2024-07-29", periods=2, freq="W-MON")


def draws_file(*, graphs_extra=False) -> bytes:
    """A draws file in the layout of the service's ``river-draws/1`` files, made at test time."""
    rng = np.random.default_rng(0)

    def series(*shape):
        return rng.normal(size=shape).astype("float32")

    dims = ("draw", "date")
    variables = {
        "channel_contribution": (("draw", "date", "channel"), series(N_DRAWS, 2, 2)),
        "control_contribution": (("draw", "date", "control"), series(N_DRAWS, 2, 1)),
        "baseline": (dims, series(N_DRAWS, 2)),
        "latent": (dims, series(N_DRAWS, 2)),
        "indirect_cc": (dims, series(N_DRAWS, 2)),
        "indirect_zc": (dims, series(N_DRAWS, 2)),
        "indirect_dc": (dims, series(N_DRAWS, 2)),
    }
    variables["outcome"] = (
        dims,
        sum(v[1].sum(axis=2) if v[1].ndim == 3 else v[1] for v in variables.values()),
    )
    dataset = xr.Dataset(
        variables,
        coords={
            "draw": np.arange(N_DRAWS),
            "date": PLAN_WEEKS.to_numpy("M8[ns]"),
            "channel": np.array(["search", "tv"]),  # the file's order differs from the user's
            "control": np.array(["price"]),
        },
        attrs={
            "schema": "river-draws/1",
            "scenario_id": "baseline",
            "seed": 0,
            "mode": "model_average",
            "model_revision": "rev-test",
            "library_revision": "lib-test",
        },
    )
    return bytes(dataset.to_netcdf(engine="h5netcdf"))


class Fake:
    """The service and file host. ``service`` records every API request."""

    def __init__(self) -> None:
        self.service = FakeService()
        self.file_requests: list[httpx.Request] = []
        rows = result_rows("baseline", "baseline", url="https://files.example.com/b.nc?sig=1")
        self.service.on("GET", "/v1/datasets", (200, {"datasets": []}))
        self.service.on("POST", "/v1/datasets", (201, dataset_record(CHANNELS, ("price",))))
        self.service.on("POST", "/v1/analyses", (202, started()))
        self.service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
        self.service.on("GET", "/v1/analyses/an_1/result", (200, result_record(rows)))
        self.file = draws_file()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "files.example.com":
            self.file_requests.append(request)
            return httpx.Response(200, content=self.file)
        return self.service(request)


@pytest.fixture
def fake(monkeypatch) -> Fake:
    instance = Fake()
    real = pymc_mfm.client.Client

    def client(base_url, api_key, *, transport=None):
        return real(base_url, api_key, transport=httpx.MockTransport(instance))

    monkeypatch.setattr(hosted, "Client", client)
    return instance


def load():
    return pymc_mfm.from_pretrained(
        "pymc-labs/mfm", profile="river-1", api_key="test-key", base_url=BASE_URL
    )


def test_hosted_load_add_context_inference(fake, history, plan):
    model = load()
    assert model.add_context(history, **ROLES) is model
    result = model.inference(plan, n_samples=N_DRAWS)
    assert type(result).__name__ == "DecompositionResult"
    assert result.component_names == (
        "baseline",
        "channel:tv",
        "channel:search",
        "control:price",
        "latent",
        "indirect:cc",
        "indirect:zc",
        "indirect:dc",
    )
    assert result.components.shape == (N_DRAWS, 2, 8)
    assert result.channel_names == CHANNELS  # the user's order, not the file's
    assert result.context_length == 4 and result.window == "query"
    assert result.components.dtype == np.float32
    assert result.provenance["draws_dtype"] == "float32"
    assert result.provenance["seed"] == 0
    assert result.provenance["model_revision"] == "rev-test"
    assert not result.edge_mask.any()
    np.testing.assert_array_equal(result.time, PLAN_WEEKS.to_numpy("M8[ns]"))
    body = fake.service.body(fake.service.sent("POST", "/v1/analyses")[0])
    assert body["draws"] == N_DRAWS and "seed" not in body
    assert "Authorization" not in fake.file_requests[0].headers


def test_column_checks_run_before_any_request(fake, history):
    model = load()
    with pytest.raises(ValueError, match="nonnegative"):
        model.add_context(history.assign(tv=-1.0), **ROLES)
    weekly_break = history.drop(index=1)
    with pytest.raises(ValueError, match="week"):
        model.add_context(weekly_break, **ROLES)
    monthly = history.assign(week=pd.date_range("2024-01-01", periods=4, freq="MS"))
    with pytest.raises(ValueError, match="7 days"):
        model.add_context(monthly, **ROLES)
    assert fake.service.requests == []


def test_history_is_reused_when_the_service_already_has_it(fake, history):
    stored = dataset_record(CHANNELS, ("price",), dataset_id="ds_old", content_hash="x")
    from pymc_mfm.client import frame_content_hash

    stored["content_hash"] = frame_content_hash(
        history, date="week", outcome="sales", spend=list(CHANNELS), controls=["price"]
    )
    fake.service.on("GET", "/v1/datasets", (200, {"datasets": [stored]}))
    load().add_context(history, **ROLES)
    assert fake.service.sent("POST", "/v1/datasets") == []


def test_inference_refuses_few_draws_before_any_request(fake, history, plan):
    model = load().add_context(history, **ROLES)
    sent = len(fake.service.requests)
    with pytest.raises(ValueError, match="n_samples >= 100.*local default is 32"):
        model.inference(plan)  # the local default, 32
    with pytest.raises(ValueError, match="n_samples >= 100"):
        model.inference(plan, n_samples=99)
    assert len(fake.service.requests) == sent


def test_inference_needs_context_and_a_plan_that_follows_it(fake, history, plan):
    model = load()
    with pytest.raises(RuntimeError, match="add_context"):
        model.inference(plan, n_samples=N_DRAWS)
    model.add_context(history, **ROLES)
    late = plan.assign(week=plan["week"] + pd.Timedelta(weeks=1))
    with pytest.raises(ValueError, match="starts on"):
        model.inference(late, n_samples=N_DRAWS)
    assert fake.service.sent("POST", "/v1/analyses") == []


def test_service_errors_keep_the_client_classes(fake, history, plan):
    from .client_fixtures import error_body

    model = load().add_context(history, **ROLES)
    fake.service.on("POST", "/v1/analyses", (422, error_body("invalid_input", "draws too many")))
    with pytest.raises(ValidationError):
        model.inference(plan, n_samples=N_DRAWS)


def test_a_failed_analysis_is_a_client_error(fake, history, plan):
    model = load().add_context(history, **ROLES)
    fake.service.on("GET", "/v1/analyses/an_1", (200, status_record("failed")))
    with pytest.raises(ClientError, match="failed"):
        model.inference(plan, n_samples=N_DRAWS)


def test_arguments_of_a_local_load_are_refused_on_hosted():
    for options, text in (
        ({"revision": "abc"}, "revision"),
        ({"cache_dir": "x"}, "cache_dir"),
        ({"token": "hf"}, "token"),
        ({"device": "mps"}, "device"),
        ({"profile": "other"}, "river-1"),
    ):
        with pytest.raises(ValueError, match=text):
            pymc_mfm.from_pretrained("pymc-labs/mfm", api_key="k", base_url=BASE_URL, **options)
    with pytest.raises(ValueError, match="pymc-labs/mfm"):
        pymc_mfm.from_pretrained("models/river-1", api_key="k", base_url=BASE_URL)
    with pytest.raises(ValueError, match="api_key"):
        pymc_mfm.from_pretrained("pymc-labs/mfm", api_key="")
    with pytest.raises(ValueError, match="base_url applies only"):
        pymc_mfm.from_pretrained("models/river-1", base_url=BASE_URL)


def test_the_environment_key_does_not_select_the_hosted_backend(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("the hosted client was made")

    monkeypatch.setattr(hosted, "Client", forbidden)
    monkeypatch.setenv("MFM_API_KEY", "from-env")
    monkeypatch.setenv("MFM_API_URL", BASE_URL)
    with pytest.raises(Exception) as error:  # a missing folder, or no torch: any local error
        pymc_mfm.from_pretrained(str(tmp_path / "missing"))
    assert not isinstance(error.value, AssertionError)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.to("cpu"),
        lambda m: m.parameters(),
        lambda m: m.compile_sampling(),
        lambda m: m.decompose({}),
        lambda m: m.build_idata(),
        lambda m: m.input_usage(),
    ],
)
def test_refused_methods_name_the_method(fake, call):
    with pytest.raises(hosted.HostedUnsupported) as error:
        call(load())
    assert error.value.method in str(error.value)
    assert isinstance(error.value, ClientError)


@pytest.mark.parametrize(
    ("options", "name"),
    [({"mode": "map"}, "mode"), ({"graph": [1]}, "graph"), ({"router_samples": 3}, "router")],
)
def test_refused_inference_options(fake, history, plan, options, name):
    model = load().add_context(history, **ROLES)
    with pytest.raises(hosted.HostedUnsupported, match=name):
        model.inference(plan, n_samples=N_DRAWS, **options)


def test_graph_plots_of_a_hosted_result_are_refused(fake, history, plan):
    result = load().add_context(history, **ROLES).inference(plan, n_samples=N_DRAWS)
    for name in hosted.GRAPH_METHODS:
        with pytest.raises(hosted.HostedUnsupported, match=f"result.plot.{name}"):
            getattr(result.plot, name)()


def test_the_hosted_result_plots_a_reconstruction(fake, history, plan):
    pytest.importorskip("matplotlib")
    result = load().add_context(history, **ROLES).inference(plan, n_samples=N_DRAWS)
    figure = result.plot.reconstruction(observed=[1000.0, 1100.0])
    assert figure.axes
    assert result.plot.components().axes


def test_signatures_match_the_local_model():
    torch = pytest.importorskip("torch")  # noqa: F841
    from pymc_mfm.model import DEFAULT_N_SAMPLES, MFM

    assert hosted.LOCAL_DEFAULT_N_SAMPLES == DEFAULT_N_SAMPLES
    for name in ("add_context", "inference"):
        local = inspect.signature(getattr(MFM, name))
        remote = inspect.signature(getattr(hosted.HostedMFM, name))
        assert list(local.parameters) == list(remote.parameters), name
        for key, parameter in local.parameters.items():
            assert parameter.default == remote.parameters[key].default, (name, key)


def test_hosted_and_local_results_have_the_same_variables_coordinates_and_dims(fake, history, plan):
    torch = pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from pymc_mfm.model import MFM, ModelConfig

    torch.manual_seed(10)
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=1, n_demand=2
    )
    local_model = MFM(config).eval()
    local_model.add_context(history, **ROLES)
    local = local_model.inference(plan, n_samples=4)
    remote = load().add_context(history, **ROLES).inference(plan, n_samples=N_DRAWS)

    for name in (
        "component_names",
        "component_families",
        "channel_names",
        "control_names",
        "window",
        "mode",
        "units",
        "context_length",
    ):
        assert getattr(remote, name) == getattr(local, name), name
    np.testing.assert_array_equal(remote.time, local.time)
    assert remote.time.dtype == local.time.dtype
    np.testing.assert_array_equal(remote.component_mask, local.component_mask)
    assert remote.components.ndim == local.components.ndim == 3
    assert remote.components.shape[1:] == local.components.shape[1:]  # [time, component]
    observed = [1000.0, 1100.0]
    assert local.plot.reconstruction(observed=observed).axes
    assert remote.plot.reconstruction(observed=observed).axes


def test_the_three_load_lines(fake, history, plan, tmp_path):
    torch = pytest.importorskip("torch")
    from pymc_mfm.inference.checkpoint import save_pretrained
    from pymc_mfm.model import MFM, ModelConfig

    torch.manual_seed(10)
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=1, n_demand=2
    )
    save_pretrained(MFM(config).eval(), tmp_path / "river-1")
    models = [
        pymc_mfm.from_pretrained(
            "pymc-labs/mfm", profile="river-1", api_key="k", base_url=BASE_URL
        ),
        pymc_mfm.from_pretrained(str(tmp_path / "river-1")),
    ]
    assert isinstance(models[1], pymc_mfm.MFM)
    for model in models:  # downstream code is identical
        model.add_context(history, **ROLES)
        result = model.inference(plan, n_samples=N_DRAWS)
        assert result.n_samples in (N_DRAWS,)
        assert result.plot.reconstruction(observed=[1.0, 2.0]) is not None


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False
    )


def test_the_hosted_path_runs_without_torch():
    result = run(
        """
        import sys
        sys.modules["torch"] = None  # "import torch" fails, as in a light install
        sys.modules["safetensors"] = None
        import httpx, pandas as pd
        sys.path.insert(0, %r)
        from client.test_hosted import Fake, ROLES, load, hosted, pymc_mfm, BASE_URL
        import pymc_mfm.client
        fake = Fake()
        real = pymc_mfm.client.Client
        hosted.Client = lambda b, k, *, transport=None: real(
            b, k, transport=httpx.MockTransport(fake))
        history = pd.DataFrame({
            "week": pd.date_range("2024-07-01", periods=4, freq="W-MON"),
            "sales": [1000.0, 1100.0, 1050.0, 1200.0],
            "tv": [100.0, 0.0, 120.0, 110.0], "search": [50.0, 55.0, 60.0, 45.0],
            "price": [9.9, 9.9, 10.5, 10.5]})
        plan = pd.DataFrame({
            "week": pd.date_range("2024-07-29", periods=2, freq="W-MON"),
            "tv": [100.0, 100.0], "search": [50.0, 50.0], "price": [10.5, 10.5]})
        model = pymc_mfm.from_pretrained("pymc-labs/mfm", profile="river-1", api_key="k",
                                         base_url=BASE_URL)
        result = model.add_context(history, **ROLES).inference(plan, n_samples=100)
        assert result.components.shape == (100, 2, 8)
        figure = result.plot.reconstruction(observed=[1.0, 2.0])
        assert figure.axes
        assert "torch" not in sys.modules or sys.modules["torch"] is None
        try:
            pymc_mfm.from_pretrained("models/river-1")
        except ImportError as error:
            assert "pymc-mfm[local]" in str(error), error
        else:
            raise SystemExit("local load worked without torch")
        """
        % str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_the_torch_free_metric_tables_match_the_model_and_the_evaluator():
    pytest.importorskip("torch")
    from pymc_mfm.model.graph import SQUARE_EDGE_TYPES
    from pymc_mfm.training import _metric_tables, evaluation

    assert _metric_tables.SQUARE_EDGE_TYPES == SQUARE_EDGE_TYPES
    for name in ("METRIC_BOUNDS", "METRIC_UNITS", "GRAPH_METRICS", "metric_units"):
        assert getattr(evaluation, name) is getattr(_metric_tables, name)
