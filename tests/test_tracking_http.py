"""Bounded MLflow HTTP delivery against real local sockets.

Each backend listens on 127.0.0.1 and records every accepted connection, so the
retry settings, their precedence, the stop after a failed start and the
per-channel pauses after a later outage are observed from the network side.
MLflow's own getters report what its HTTP client reads.
"""

import json
import os
import random
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch
from test_tracking import events, tiny_run

from pymc_mktfm.model.model import MFM
from pymc_mktfm.training import MlflowTrainingObserver, TrackingConfig, TrainingObserver
from pymc_mktfm.training.checkpoint import file_sha256

mlflow_environment = pytest.importorskip("mlflow.environment_variables")

HTTP_VARIABLES = (
    "MLFLOW_HTTP_REQUEST_TIMEOUT",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR",
    "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER",
    "MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER",
)
# Seconds a busy backend asks a client to wait: far longer than MFM's 0 s / 2 s backoff.
RETRY_AFTER = 10


@pytest.fixture(autouse=True)
def isolated_http_environment(monkeypatch):
    for name in (*HTTP_VARIABLES, "MLFLOW_TRACKING_URI", "MLFLOW_ENABLE_ASYNC_LOGGING"):
        monkeypatch.delenv(name, raising=False)
    # No environment or system proxy may stand between MLflow and the local backends.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


class Backend(ThreadingHTTPServer):
    """An MLflow stand-in recording the time and mode of every accepted connection.

    ``close`` accepts and closes at once, ``hang`` accepts and never answers, and
    ``serve`` answers the REST calls of a run start until ``serve_until`` has been
    answered, then closes every later connection (the backend died mid-run). A
    test may switch ``mode`` itself, for example back to ``serve`` when an outage
    ends. A serving backend answers the first request to the ``reject_once`` path
    with HTTP 400, as MLflow does for a request it refuses, and every request to
    the ``busy`` path with HTTP 503 and ``Retry-After: RETRY_AFTER``, as a
    rate-limited or overloaded server does. It answers the first ``runs/create``
    only after ``create_delay`` seconds. With ``artifact_root="mlflow-artifacts:"``
    artifact uploads reach the backend too, as with ``mlflow server
    --artifacts-destination``; otherwise they go to that URI, such as a local
    directory or another backend. ``paths`` lists every answered request path in
    order. ``runs`` holds the ``tags`` and terminal ``status`` of every run, newest
    last; a test may add runs with ``add_run``. ``run_id`` is the run the last
    ``runs/create`` answered, and ``tags`` and ``status`` are its entries.
    ``runs/search`` selects runs with MLflow's own filter code, newest first and
    at most ``max_results``, and answers 400 to a filter MLflow refuses.
    """

    daemon_threads = True

    def __init__(
        self,
        mode,
        *,
        serve_until=None,
        artifact_root=None,
        reject_once=None,
        busy=None,
        create_delay=None,
    ):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mode, self.serve_until, self.artifact_root = mode, serve_until, artifact_root
        self.reject_once, self.busy, self.create_delay = reject_once, busy, create_delay
        self.paths = []
        self.runs, self.run_id = {}, None
        self.accepted = []
        self.held = []
        self.lock = threading.Lock()
        serving = threading.Thread(target=self.serve_forever, args=(0.05,), daemon=True)
        serving.start()

    @property
    def uri(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    @property
    def tags(self):
        return self.runs[self.run_id]["tags"]

    @property
    def status(self):
        return self.runs[self.run_id]["status"]

    def run(self, run_id):
        """A stored run as MLflow's REST API returns it."""
        entry = self.runs[run_id]
        info = {
            "run_id": run_id,
            "run_uuid": run_id,
            "experiment_id": "1",
            "status": entry["status"] or "RUNNING",
            "lifecycle_stage": "active",
            "artifact_uri": f"{self.artifact_root}/{run_id}",
        }
        tags = [{"key": key, "value": value} for key, value in entry["tags"].items()]
        return {"info": info, "data": {"tags": tags}}

    def add_run(self, tags, status=None):
        """Store a run, as the newest one, and return its id."""
        # A fresh id per run: MLflow caches artifact repositories per run id for the
        # whole process.
        run_id = uuid.uuid4().hex
        self.runs[run_id] = {"tags": dict(tags), "status": status}
        return run_id

    def search(self, query):
        """The runs a real server selects for a ``runs/search`` request.

        Every stored run is active in experiment ``"1"``, so a search of another
        experiment or of deleted runs finds none. MLflow's own
        ``SearchUtils.filter`` evaluates the filter (comparators included); a
        filter it refuses raises.
        """
        from mlflow.entities import Run, RunData, RunInfo, RunTag
        from mlflow.utils.search_utils import SearchUtils

        def entity(run_id):
            entry = self.runs[run_id]
            info = RunInfo(
                run_id=run_id,
                experiment_id="1",
                user_id="",
                status=entry["status"] or "RUNNING",
                start_time=0,
                end_time=None,
                lifecycle_stage="active",
            )
            tags = [RunTag(key, value) for key, value in entry["tags"].items()]
            return Run(info, RunData(tags=tags))

        if "1" not in query.get("experiment_ids", []) or query.get("run_view_type") == (
            "DELETED_ONLY"
        ):
            return []
        runs = [entity(run_id) for run_id in reversed(self.runs)]
        matching = SearchUtils.filter(runs, query["filter"])
        return [self.run(run.info.run_id) for run in matching][: int(query["max_results"])]

    def connections(self, mode=None):
        with self.lock:
            return [moment for moment, kind in self.accepted if mode in (None, kind)]

    def get_request(self):
        connection, address = super().get_request()
        with self.lock:
            mode = self.mode
            self.accepted.append((time.monotonic(), mode))
        if mode == "serve":
            return connection, address
        if mode == "hang":
            self.held.append(connection)
        else:
            connection.close()
        raise OSError(f"{mode} backend does not answer")

    def stop(self):
        self.shutdown()
        self.server_close()
        for connection in self.held:
            connection.close()


class _Handler(BaseHTTPRequestHandler):
    # One connection per request, so connection counts are request attempts.
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def reply(self, code, payload, headers=()):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def answer(self):
        server = self.server
        request = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = self.path.split("?")[0]
        with server.lock:
            server.paths.append(path)
            reject = server.reject_once is not None and path.endswith(server.reject_once)
            if reject:
                server.reject_once = None
            busy = server.busy is not None and path.endswith(server.busy)
        if reject:
            self.reply(400, {"error_code": "INVALID_PARAMETER_VALUE", "message": "rejected"})
            return
        if busy:
            error = {"error_code": "TEMPORARILY_UNAVAILABLE", "message": "busy"}
            self.reply(503, error, [("Retry-After", str(RETRY_AFTER))])
            return
        delay, payload = None, {}
        with server.lock:
            if path.endswith("/runs/create"):
                tags = {tag["key"]: tag["value"] for tag in json.loads(request)["tags"]}
                server.run_id = server.add_run(tags)
                delay, server.create_delay = server.create_delay, None
            elif path.endswith("/runs/set-tag"):
                fields = json.loads(request)
                server.runs[fields["run_id"]]["tags"][fields["key"]] = fields["value"]
            elif path.endswith("/runs/update"):
                fields = json.loads(request)
                server.runs[fields["run_id"]]["status"] = fields["status"]
            if path.endswith("/experiments/get-by-name"):
                experiment = {
                    "experiment_id": "1",
                    "name": "pymc-mktfm",
                    "lifecycle_stage": "active",
                }
                payload = {"experiment": experiment}
            elif path.endswith(("/runs/create", "/runs/get")):
                payload = {"run": server.run(server.run_id)}
            elif path.endswith("/runs/search"):
                try:
                    payload = {"runs": server.search(json.loads(request))}
                except Exception:  # noqa: BLE001 - a filter MLflow's parser refuses
                    payload = None
            if server.serve_until is not None and path.endswith(server.serve_until):
                # Before answering, so the client's next connection already finds it dead.
                server.mode = "close"
        if payload is None:
            self.reply(400, {"error_code": "INVALID_PARAMETER_VALUE", "message": "bad filter"})
            return
        if delay:
            time.sleep(delay)
        self.reply(200, payload)

    do_GET = do_POST = do_PUT = answer


@pytest.fixture
def backend():
    started = []

    def start(mode, **kwargs):
        started.append(Backend(mode, **kwargs))
        return started[-1]

    yield start
    for server in started:
        server.stop()


class HttpSettingsProbe(TrainingObserver):
    """What MLflow's HTTP client reads while the run trains."""

    def __init__(self):
        self.seen = []

    def on_step(self, record):
        self.seen.append({name: getattr(mlflow_environment, name).get() for name in HTTP_VARIABLES})


def failures(out_dir):
    """``(operation, delivery_stopped, transport)`` of every delivery receipt."""
    path = out_dir / "tracking" / "delivery_failures.jsonl"
    return [
        (entry["operation"], entry["delivery_stopped"], entry["transport"])
        for entry in map(json.loads, path.read_text().splitlines())
    ]


def state(out_dir):
    return json.loads((out_dir / "tracking" / "state.json").read_text())


def closed_pauses(out_dir):
    """The pause ledger: every pause that closed, in order."""
    path = out_dir / "tracking" / "delivery_pauses.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def step_sequences(out_dir):
    """The event sequence number of each step."""
    return {
        event["step"]: event["sequence"] for event in events(out_dir) if event["kind"] == "step"
    }


def answered(server, since):
    """REST endpoints of the requests the backend answered after the first ``since``."""
    return [path.rsplit("/mlflow/", 1)[-1] for path in server.paths[since:]]


def http_receipts(out_dir):
    return [event["payload"] for event in events(out_dir) if event["kind"] == "http_settings"]


def start_and_end(observer, run_id):
    try:
        with pytest.warns(RuntimeWarning):
            observer.on_run_start({"run_id": run_id, "status": "running"})
        observer.on_step({"step": 1, "loss": 0.5})
    finally:
        observer.on_run_end({"run_id": run_id, "status": "finished", "final_step": 1})


class Clock:
    """Monotonic seconds that a test advances, so cool-downs need no real waits."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def ctrl_c_once_hung(server):
    """Interrupt this thread 0.3 s after ``server`` accepts a connection it never answers."""
    interrupted = threading.get_ident()

    def ctrl_c():
        while not server.connections("hang"):
            time.sleep(0.01)
        time.sleep(0.3)
        signal.pthread_kill(interrupted, signal.SIGINT)

    threading.Thread(target=ctrl_c, daemon=True).start()


@pytest.fixture
def observe(tmp_path):
    """An observer of ``tmp_path`` on a test clock, started unless ``start=False``.

    Every observer's run ends at teardown, which restores the HTTP environment when
    a test fails first.
    """
    observers = []

    def observer_of(server, *, start=True, **config):
        observer = MlflowTrainingObserver(
            tmp_path, TrackingConfig(tracking_uri=server.uri, **config)
        )
        clock = Clock()
        observer._clock = clock
        observers.append(observer)
        if start:
            observer.on_run_start({"run_id": "observed", "status": "running"})
        return observer, clock

    yield observer_of
    for observer in observers:
        observer.on_run_end({"run_id": "observed", "status": "failed"})


@pytest.fixture
def paused(tmp_path, backend, observe):
    """A started observer whose backend refused the step-1 metrics at clock time 0."""
    server = backend("serve", artifact_root=(tmp_path / "remote").as_uri())
    observer, clock = observe(server, http_max_retries=1)
    server.mode = "close"
    with pytest.warns(RuntimeWarning):
        observer.on_step({"step": 1, "loss": 0.5})
    return server, observer, clock


def test_http_bounds_are_integers_mlflow_accepts():
    TrackingConfig(http_timeout=1, http_max_retries=0)
    TrackingConfig(http_max_retries=10)
    for bounds in (
        {"http_timeout": 0},
        {"http_timeout": True},
        {"http_timeout": 2.5},
        {"http_max_retries": -1},
        {"http_max_retries": 11},
        {"http_max_retries": False},
    ):
        with pytest.raises(ValueError):
            TrackingConfig(**bounds)


def test_unreachable_backend_costs_one_bounded_start_and_nothing_after(tmp_path, backend):
    server = backend("close")
    out = tmp_path / "run"
    with pytest.warns(RuntimeWarning):
        result = tiny_run(out, tracking=TrackingConfig(tracking_uri=server.uri, diagnostics=False))
    assert result.status == "finished"
    # MFM defaults: an immediate first retry, then backoff 1 s * 2 without jitter.
    accepted = server.connections()
    assert len(accepted) == 3
    assert accepted[1] - accepted[0] < 0.5
    assert accepted[2] - accepted[1] == pytest.approx(2.0, abs=0.5)
    assert failures(out) == [("start_run", True, True)]
    assert result.run["tracking"]["delivery_status"] == "degraded"
    assert [event["step"] for event in events(out) if event["kind"] == "step"] == [1, 2]
    assert not any(name in os.environ for name in HTTP_VARIABLES)


@pytest.mark.parametrize(
    ("environment", "field", "attempts"),
    [("1", None, 2), ("1", 0, 1)],
    ids=["environment_over_default", "field_over_environment"],
)
def test_retry_precedence_is_field_then_environment_then_default(
    tmp_path, backend, monkeypatch, environment, field, attempts
):
    server = backend("close")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", environment)
    config = TrackingConfig(tracking_uri=server.uri, http_max_retries=field)
    start_and_end(MlflowTrainingObserver(tmp_path, config), "precedence")
    assert len(server.connections()) == attempts
    assert failures(tmp_path) == [("start_run", True, True)]
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == environment


@pytest.mark.parametrize(
    ("environment", "field"),
    [(None, 1), ("1", None), ("30", 1)],
    ids=["field", "environment_over_default", "field_over_environment"],
)
def test_hung_backend_fails_start_after_one_bounded_timeout(
    tmp_path, backend, monkeypatch, environment, field
):
    server = backend("hang")
    if environment is not None:
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", environment)
    config = TrackingConfig(tracking_uri=server.uri, http_timeout=field, http_max_retries=0)
    started = time.perf_counter()
    start_and_end(MlflowTrainingObserver(tmp_path, config), "hung")
    # One 1 s read timeout: neither the 10 s default nor the 30 s environment value.
    assert 0.9 <= time.perf_counter() - started < 5
    assert len(server.connections()) == 1
    assert failures(tmp_path) == [("start_run", True, True)]


def test_retry_after_does_not_stretch_the_backoff(tmp_path, backend):
    # Each answer asks for RETRY_AFTER s, as a rate-limited or overloaded server does.
    server = backend("serve", busy="/experiments/get-by-name")
    start_and_end(MlflowTrainingObserver(tmp_path, TrackingConfig(tracking_uri=server.uri)), "busy")
    # MFM's backoff still applies: an immediate first retry, then 1 s * 2.
    accepted = server.connections()
    assert len(accepted) == 3
    assert accepted[1] - accepted[0] < 0.5
    assert accepted[2] - accepted[1] == pytest.approx(2.0, abs=0.5)
    assert failures(tmp_path) == [("start_run", True, True)]


def test_backend_dying_after_create_run_pauses_each_channel_until_its_run_end_attempt(
    tmp_path, backend
):
    remote = (tmp_path / "remote").as_uri()
    server = backend("serve", serve_until="/runs/create", artifact_root=remote)
    out = tmp_path / "run"
    config = TrackingConfig(tracking_uri=server.uri, http_max_retries=1, diagnostics=False)
    with pytest.warns(RuntimeWarning):
        result = tiny_run(out, tracking=config)
    assert result.status == "finished"
    assert result.run["tracking"]["run_id"] == server.run_id
    # The run_start upload, which asks the server for the artifact location, pauses
    # the uploads. The duplicate-run search fails too, which pauses nothing and is no
    # delivery receipt, and the configuration's metrics pause the tracking-server
    # writes. The cool-down skips every delivery in between, and each channel's
    # run-end attempt stops it. Each of the five makes 1 + 1 attempts.
    assert len(server.connections("close")) == 10
    assert failures(out) == [
        ("log_artifact", False, True),
        ("log_metrics", False, True),
        ("log_artifact", True, True),
        ("delivery_tag", True, True),
    ]
    tracking = result.run["tracking"]
    assert tracking["delivery_status"] == "degraded"
    # The run's tracking record holds both pauses, which never resumed.
    receipts = (out / "tracking" / "delivery_failures.jsonl").read_text().splitlines()
    opened = {
        channel: pauses["open"]["sequence"]
        for channel, pauses in tracking["delivery_pauses"].items()
    }
    assert opened == {
        "artifacts": json.loads(receipts[0])["sequence"],
        "tracking": json.loads(receipts[1])["sequence"],
    }
    records = events(out)
    assert [event["step"] for event in records if event["kind"] == "step"] == [1, 2]
    assert records[-1]["kind"] == "run_end"


def test_outage_after_start_pauses_delivery_until_a_probe_after_the_cool_down(paused):
    server, observer, clock = paused
    out = observer.out_dir
    # The backend is back, but nothing is sent before the 60 s cool-down ends.
    server.mode = "serve"
    clock.now = 59.0
    quiet = len(server.connections())
    observer.on_step({"step": 2, "loss": 0.4})
    observer.on_step({"step": 3, "loss": 0.3})
    assert len(server.connections()) == quiet
    clock.now = 60.0
    since = len(server.paths)
    observer.on_step({"step": 4, "loss": 0.2})
    observer.on_step({"step": 5, "loss": 0.1})
    # The probe marks the remote run degraded, then both steps' metrics arrive.
    assert answered(server, since) == ["runs/set-tag", "runs/log-batch", "runs/log-batch"]
    assert server.tags["mfm.delivery_status"] == "degraded"
    observer.on_run_end({"run_id": "outage", "status": "finished", "final_step": 5})
    assert server.tags["mfm.lifecycle_status"] == "finished"
    assert server.tags["mfm.delivery_status"] == "degraded"
    assert server.status == "FAILED"
    assert failures(out) == [("log_metrics", False, True)]
    steps = step_sequences(out)
    assert list(steps) == [1, 2, 3, 4, 5]
    # Steps 2 and 3 were a metric batch each, never replayed. Their event files
    # reached the artifact store, which this outage did not pause.
    assert closed_pauses(out) == [
        {
            "channel": "tracking",
            "sequence": steps[1],
            "resumed_sequence": steps[4],
            "skipped_deliveries": 2,
            "failed_probes": 0,
        }
    ]
    remote = out / "remote" / server.run_id
    uploaded = {path.name for path in (remote / "events").iterdir()}
    assert {path.name for path in (observer.root / "events").glob("*_step_step_*")} <= uploaded
    # The run end uploads the pause ledger too.
    assert closed_pauses(remote) == closed_pauses(out)


def test_dead_backend_is_probed_after_each_doubling_cool_down(paused):
    server, observer, clock = paused
    out = observer.out_dir
    # Cool-downs of 60, 120, 240 and 480 s, then the 600 s cap; a probe is 1 + 1 attempts.
    attempts = []
    moments = (59, 60, 179, 180, 419, 420, 899, 900, 1499, 1500, 2099, 2100)
    for step, moment in enumerate(moments, start=2):
        clock.now = moment
        before = len(server.connections())
        observer.on_step({"step": step, "loss": 0.1})
        attempts.append(len(server.connections()) - before)
    assert attempts == [0, 2] * 6
    assert failures(out) == [("log_metrics", False, True)]
    # Every step's metric batch was skipped, probed or not; the event files were
    # uploaded.
    step_1 = step_sequences(out)[1]
    assert state(out)["delivery_pauses"]["tracking"] == {
        "open": {
            "channel": "tracking",
            "sequence": step_1,
            "resumed_sequence": None,
            "skipped_deliveries": len(moments),
            "failed_probes": 6,
        },
        "pauses": 1,
        "skipped_deliveries": len(moments),
        "failed_probes": 6,
    }


def test_run_ending_paused_probes_at_once_and_delivers_the_tags_and_status(paused):
    server, observer, clock = paused
    out = observer.out_dir
    # The backend recovered within the cool-down, which has not elapsed.
    server.mode = "serve"
    (out / "model").mkdir()
    (out / "model" / "model.safetensors").write_bytes(b"exported weights")
    export = {"files": ["model.safetensors"]}
    observer.on_run_end(
        {"run_id": "outage", "status": "finished", "final_step": 1, "export": export}
    )
    assert server.tags["mfm.lifecycle_status"] == "finished"
    assert server.tags["mfm.delivery_status"] == "degraded"
    assert server.status == "FAILED"
    assert failures(out) == [("log_metrics", False, True)]
    local, remote = observer.root, out / "remote" / server.run_id
    run_end = next((local / "events").glob("*_run_end_*.json"))
    resumed = [pause["resumed_sequence"] for pause in closed_pauses(out)]
    assert resumed == [json.loads(run_end.read_text())["sequence"]]
    # The uploads, never paused, delivered the terminal evidence and the export.
    weights = next((local / "objects").glob("*/model_export/*/model.safetensors"))
    delivered = {path.relative_to(remote) for path in remote.rglob("*") if path.is_file()}
    assert {run_end.relative_to(local), weights.relative_to(local)} <= delivered
    # The tags closed the pause after those uploads; the uploaded records still match.
    assert closed_pauses(remote) == closed_pauses(out)
    assert state(remote) == state(out)


def test_run_ending_paused_on_a_dead_backend_makes_one_bounded_attempt(paused):
    server, observer, clock = paused
    out = observer.out_dir
    before = len(server.connections())
    observer.on_run_end({"run_id": "outage", "status": "finished", "final_step": 1})
    # The immediate probe is the tracking channel's one run-end attempt: 1 + 1
    # connections and a receipt that stops that channel.
    assert len(server.connections()) - before == 2
    assert failures(out) == [("log_metrics", False, True), ("delivery_tag", True, True)]
    # The uploads, which need no connection to this backend, still happen.
    assert list((out / "remote").rglob("*_run_end_*"))


def test_outage_during_the_run_end_costs_one_attempt_per_channel(tmp_path, backend):
    server = backend("serve", artifact_root="mlflow-artifacts:")
    config = TrackingConfig(tracking_uri=server.uri, http_max_retries=1)
    observer = MlflowTrainingObserver(tmp_path, config)
    observer.on_run_start({"run_id": "late", "status": "running"})
    server.mode = "close"
    with pytest.warns(RuntimeWarning):
        observer.on_run_end({"run_id": "late", "status": "finished", "final_step": 1})
    # Uploading the run_end event is the uploads' one run-end attempt and the
    # lifecycle tag the tracking channel's: 1 + 1 connections and one receipt each,
    # after which each channel skips its remaining terminal calls.
    assert len(server.connections("close")) == 4
    assert failures(tmp_path) == [("log_artifact", True, True), ("lifecycle_tag", True, True)]


def test_rejected_probe_resumes_delivery_with_its_own_receipt(paused):
    server, observer, clock = paused
    # Any answer, even a refusal of the probe, shows that the transport is back.
    server.mode = "serve"
    server.reject_once = "/runs/set-tag"
    clock.now = 60.0
    since = len(server.paths)
    observer.on_step({"step": 2, "loss": 0.4})
    assert answered(server, since) == ["runs/set-tag", "runs/log-batch"]
    assert failures(observer.out_dir) == [
        ("log_metrics", False, True),
        ("delivery_tag", False, False),
    ]


def test_a_rate_limited_channel_stays_paused_until_a_write_is_delivered(tmp_path, backend, observe):
    server = backend("serve", artifact_root=(tmp_path / "remote").as_uri())
    observer, clock = observe(server, http_max_retries=1)
    # The server stays up but answers every metric batch 503: one receipt, a pause.
    server.busy = "/runs/log-batch"
    with pytest.warns(RuntimeWarning):
        observer.on_step({"step": 1, "loss": 0.5})

    def step(number, moment):
        clock.now = moment
        since = len(server.paths)
        observer.on_step({"step": number, "loss": 0.1})
        return answered(server, since)

    refused = ["runs/log-batch"] * 2  # 1 + 1 attempts
    assert step(2, 59.0) == []
    # The tag probe gets through but the batch still fails: the same pause, now 120 s.
    assert step(3, 60.0) == ["runs/set-tag", *refused]
    assert step(4, 179.0) == []
    server.busy = None
    # A delivered batch closes the pause and resets the cool-down to 60 s.
    assert step(5, 180.0) == ["runs/set-tag", "runs/log-batch"]
    server.busy = "/runs/log-batch"
    assert step(6, 200.0) == refused
    assert step(7, 259.0) == []
    assert step(8, 260.0) == ["runs/set-tag", *refused]
    assert failures(tmp_path) == [("log_metrics", False, True)] * 2
    steps = step_sequences(tmp_path)
    assert closed_pauses(tmp_path) == [
        {
            "channel": "tracking",
            "sequence": steps[1],
            "resumed_sequence": steps[5],
            "skipped_deliveries": 3,
            "failed_probes": 1,
        }
    ]
    # state.json keeps the open pause and the totals, not the closed pauses.
    assert state(tmp_path)["delivery_pauses"]["tracking"] == {
        "open": {
            "channel": "tracking",
            "sequence": steps[6],
            "resumed_sequence": None,
            "skipped_deliveries": 2,
            "failed_probes": 1,
        },
        "pauses": 2,
        "skipped_deliveries": 5,
        "failed_probes": 2,
    }


def test_a_dead_artifact_store_pauses_only_the_uploads(tmp_path, backend, observe):
    store = backend("serve")
    server = backend("serve", artifact_root=f"{store.uri}/api/2.0/mlflow-artifacts/artifacts")
    observer, clock = observe(server, http_max_retries=1)
    store.mode = "close"
    since = len(server.paths)
    with pytest.warns(RuntimeWarning):
        observer.on_step({"step": 1, "loss": 0.5})
    clock.now = 59.0
    dead = len(store.connections())
    observer.on_step({"step": 2, "loss": 0.4})
    observer.on_step({"step": 3, "loss": 0.3})
    assert len(store.connections()) == dead
    store.mode = "serve"
    clock.now = 60.0
    observer.on_step({"step": 4, "loss": 0.2})
    observer.on_step({"step": 5, "loss": 0.1})
    # Every metric batch reached the tracking server, which never needed a probe.
    assert answered(server, since) == ["runs/log-batch"] * 5
    # Step 4's upload was the probe; steps 2 and 3 were skipped, never replayed.
    local = observer.root / "events"
    files = {step: next(local.glob(f"*_step_step_{step:09d}.json")).name for step in range(1, 6)}
    uploaded = {path.rsplit("/", 1)[-1] for path in store.paths}
    assert set(files.values()) & uploaded == {files[4], files[5]}
    steps = step_sequences(tmp_path)
    assert closed_pauses(tmp_path) == [
        {
            "channel": "artifacts",
            "sequence": steps[1],
            "resumed_sequence": steps[4],
            "skipped_deliveries": 2,
            "failed_probes": 0,
        }
    ]
    # The store dies again; the reset cool-down would end at 121 s.
    store.mode = "close"
    clock.now = 61.0
    observer.on_step({"step": 6, "loss": 0.1})
    clock.now = 70.0
    before = len(store.connections())
    observer.on_run_end({"run_id": "observed", "status": "finished", "final_step": 6})
    # The run end does not wait: one bounded upload attempt, then no more uploads,
    # while the tracking server still receives the tags and status.
    assert len(store.connections()) - before == 2
    assert failures(tmp_path) == [
        ("log_artifact", False, True),
        ("log_artifact", False, True),
        ("log_artifact", True, True),
    ]
    assert server.tags["mfm.delivery_status"] == "degraded"
    assert server.status == "FAILED"


def test_a_checkpoint_upload_that_ends_a_pause_is_credited_to_its_event(tmp_path, backend, observe):
    store = backend("serve")
    server = backend("serve", artifact_root=f"{store.uri}/api/2.0/mlflow-artifacts/artifacts")
    observer, clock = observe(server, http_max_retries=0)
    store.mode = "close"
    with pytest.warns(RuntimeWarning):
        observer.on_step({"step": 1, "loss": 0.5})
    clock.now = 30.0
    observer.on_step({"step": 2, "loss": 0.4})
    store.mode = "serve"
    clock.now = 60.0
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights at step 2")
    observer.on_checkpoint(
        {"path": str(weights), "kind": "last", "step": 2, "sha256": file_sha256(weights)}
    )
    # The pause closed with the checkpoint's delivery, not with step 2's skipped one.
    (checkpoint,) = [event for event in events(tmp_path) if event["kind"] == "checkpoint"]
    (pause,) = closed_pauses(tmp_path)
    assert pause["resumed_sequence"] == checkpoint["sequence"]


@pytest.mark.filterwarnings("ignore:MLflow tracking is degraded:RuntimeWarning")
def test_ctrl_c_during_a_hung_start_is_one_stopped_start_receipt(tmp_path, backend):
    # An interrupted start is a failed start: one stopped receipt, nothing later.
    server = backend("hang")
    ctrl_c_once_hung(server)
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        tiny_run(out, tracking=TrackingConfig(tracking_uri=server.uri, diagnostics=False))
    assert failures(out) == [("start_run", True, False)]
    assert len(server.connections()) == 1


@pytest.mark.filterwarnings("ignore:MLflow tracking is degraded:RuntimeWarning")
@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "wrapped"])
@pytest.mark.parametrize(
    ("interrupt", "status"),
    [(KeyboardInterrupt, "interrupted"), (SystemExit, "failed")],
    ids=["ctrl_c", "exit"],
)
def test_an_interrupted_start_names_and_raises_the_interrupt(
    tmp_path, monkeypatch, interrupt, status, wrapped
):
    # A database client can turn Ctrl-C into its own error while cleaning up, as
    # SQLAlchemy does during MLflow's migrations; the interrupt stays in the chain.
    client_type = pytest.importorskip("mlflow").MlflowClient

    def interrupted_create_run(self, *args, **kwargs):
        if not wrapped:
            raise interrupt
        try:
            raise interrupt
        except interrupt:
            # Implicitly chained: the interrupt survives only as __context__.
            raise RuntimeError("connection left unusable by the interrupt")  # noqa: B904

    monkeypatch.setattr(client_type, "create_run", interrupted_create_run)
    out = tmp_path / "run"
    with pytest.raises(interrupt):
        tiny_run(out, tracking=TrackingConfig(diagnostics=False))
    record = json.loads((out / "run.json").read_text())
    assert (record["status"], record["error_type"]) == (status, interrupt.__name__)
    assert failures(out) == [("start_run", True, False)]
    path = out / "tracking" / "delivery_failures.jsonl"
    assert [json.loads(line)["error_type"] for line in path.read_text().splitlines()] == [
        interrupt.__name__
    ]


@pytest.mark.filterwarnings("ignore:MLflow tracking is degraded:RuntimeWarning")
def test_an_interrupt_the_caller_already_handles_does_not_interrupt_a_failed_start(
    tmp_path, monkeypatch
):
    # Training started from a Ctrl-C handler: the failed start chains to that old
    # interrupt, which is not this run's; the run records the failure and trains on.
    client_type = pytest.importorskip("mlflow").MlflowClient

    def rejected_create_run(self, *args, **kwargs):
        raise RuntimeError("the backend rejected the run")

    monkeypatch.setattr(client_type, "create_run", rejected_create_run)
    out = tmp_path / "run"
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        try:
            tiny_run(out, tracking=TrackingConfig(diagnostics=False))
        except KeyboardInterrupt:
            # Escaping, it would abort the test session instead of failing the test.
            pytest.fail("the caller's handled interrupt interrupted the run")
    assert json.loads((out / "run.json").read_text())["status"] == "finished"
    assert failures(out) == [("start_run", True, False)]


@pytest.mark.filterwarnings("ignore:MLflow tracking is degraded:RuntimeWarning")
def test_ctrl_c_during_the_run_end_leaves_one_attempt_at_the_tags(tmp_path, backend, observe):
    server = backend("serve", artifact_root="mlflow-artifacts:")
    observer, _ = observe(server, http_timeout=1, http_max_retries=0)
    server.mode = "hang"
    # While the run_end upload waits on the hung backend.
    ctrl_c_once_hung(server)
    with pytest.raises(KeyboardInterrupt):
        observer.on_run_end({"run_id": "observed", "status": "finished", "final_step": 1})
    # Only the lifecycle tag follows the interrupted upload, once, and it stops the rest.
    assert len(server.connections("hang")) == 2
    assert failures(tmp_path) == [
        ("finalization_interrupted", False, False),
        ("lifecycle_tag", True, True),
    ]


def test_ctrl_c_during_the_run_end_uploads_sends_no_further_upload(tmp_path, backend, observe):
    store = backend("serve")
    server = backend("serve", artifact_root=f"{store.uri}/api/2.0/mlflow-artifacts/artifacts")
    observer, _ = observe(server, http_timeout=1, http_max_retries=0)
    # The tracking server drops step 1's metrics, then answers again.
    server.mode = "close"
    with pytest.warns(RuntimeWarning):
        observer.on_step({"step": 1, "loss": 0.5})
    server.mode = "serve"
    store.mode = "hang"
    ctrl_c_once_hung(store)
    with pytest.raises(KeyboardInterrupt):
        observer.on_run_end({"run_id": "observed", "status": "finished", "final_step": 1})
    # The tags and status reach the tracking server and close its pause, but the
    # interrupted upload is the last one.
    assert (server.status, [pause["channel"] for pause in closed_pauses(tmp_path)]) == (
        "KILLED",
        ["tracking"],
    )
    assert len(store.connections("hang")) == 1


def test_the_duplicate_search_ends_only_running_runs_of_this_start(tmp_path, backend, observe):
    # The first create_run outlasts the 1 s timeout, so MLflow's retry creates a second run.
    server = backend("serve", artifact_root=(tmp_path / "remote").as_uri(), create_delay=1.5)
    observer, _ = observe(server, start=False, http_timeout=1)
    # Up to retries + 1 = 3 running runs can carry one start's id. Beside the retried
    # create's copy there are an older running copy, an ended copy, and a running run
    # of another execution with the same run id.
    copy = {"mfm.local_run_id": "observed", "mfm.start_id": observer.summary["start_id"]}
    older = server.add_run(copy)
    ended = server.add_run(copy, status="FINISHED")
    other = server.add_run({"mfm.local_run_id": "observed", "mfm.start_id": "another start"})
    observer.on_run_start({"run_id": "observed", "status": "running"})
    observer.on_run_end({"run_id": "observed", "status": "finished", "final_step": 0})
    (retried,) = set(server.runs) - {observer.run_id, older, ended, other}
    for run_id in (retried, older):
        assert server.runs[run_id]["status"] == "FAILED"
        assert server.runs[run_id]["tags"]["mfm.duplicate_of"] == observer.run_id
    # Runs that are not running copies of this start are never touched.
    for run_id, status in ((ended, "FINISHED"), (other, None)):
        run = server.runs[run_id]
        assert (run["status"], "mfm.duplicate_of" in run["tags"]) == (status, False)
    assert server.status == "FINISHED"
    records = [event["payload"] for event in events(tmp_path) if event["kind"] == "duplicate_runs"]
    assert [record["count"] for record in records] == [2]
    assert set(records[0]["run_ids"]) == {retried, older}
    assert not (tmp_path / "tracking" / "delivery_failures.jsonl").exists()


def test_a_failed_duplicate_search_is_housekeeping_not_delivery(tmp_path, backend, observe):
    # The server answers every search 503, as an overloaded one does.
    server = backend("serve", artifact_root=(tmp_path / "remote").as_uri(), busy="/runs/search")
    observer, _ = observe(server, http_max_retries=1)
    since = len(server.paths)
    observer.on_run_config({"run_id": "observed", "seed": 3})
    observer.on_step({"step": 1, "loss": 0.5})
    # Nothing is paused: the parameters and the metrics reach the server.
    assert answered(server, since) == ["runs/log-batch", "runs/log-batch"]
    observer.on_run_end({"run_id": "observed", "status": "finished", "final_step": 1})
    # Every delivery succeeded, so the run ends healthy and FINISHED, without a receipt.
    assert state(tmp_path)["delivery_status"] == "healthy"
    assert (server.status, server.tags["mfm.delivery_status"]) == ("FINISHED", "healthy")
    assert not (tmp_path / "tracking" / "delivery_failures.jsonl").exists()
    # A local event records the failure: its operation and error type, no backend text.
    (record,) = [
        event["payload"] for event in events(tmp_path) if event["kind"] == "duplicate_search_failed"
    ]
    assert record == {"operation": "find_duplicate_runs", "error_type": record["error_type"]}


def test_a_failed_duplicate_end_is_housekeeping_not_delivery(tmp_path, backend, observe):
    # The duplicate's mfm.duplicate_of tag, the first set-tag request, is refused.
    server = backend(
        "serve", artifact_root=(tmp_path / "remote").as_uri(), reject_once="/runs/set-tag"
    )
    observer, _ = observe(server, start=False)
    server.add_run({"mfm.local_run_id": "observed", "mfm.start_id": observer.summary["start_id"]})
    observer.on_run_start({"run_id": "observed", "status": "running"})
    observer.on_run_end({"run_id": "observed", "status": "finished", "final_step": 0})
    assert state(tmp_path)["delivery_status"] == "healthy"
    assert (server.status, server.tags["mfm.delivery_status"]) == ("FINISHED", "healthy")
    assert not (tmp_path / "tracking" / "delivery_failures.jsonl").exists()
    (record,) = [
        event["payload"] for event in events(tmp_path) if event["kind"] == "duplicate_search_failed"
    ]
    assert record["operation"] == "end_duplicate_run"


def test_every_execution_gets_its_own_start_id_without_touching_the_rngs(tmp_path):
    # Reruns of one recipe are seeded alike and may reuse an output path; their
    # start ids must still differ, so neither's duplicate search can end the other's.
    ids = []
    for _ in range(2):
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
        observer = MlflowTrainingObserver(tmp_path, TrackingConfig(enabled=False))
        ids.append(observer.summary["start_id"])
        assert random.getstate() == before[0]
        np.testing.assert_array_equal(np.random.get_state()[1], before[1])
        assert torch.equal(torch.get_rng_state(), before[2])
    assert ids[0] != ids[1]


def test_only_a_tracking_server_is_searched_for_duplicate_runs(
    tmp_path, backend, observe, monkeypatch
):
    client_type = pytest.importorskip("mlflow").MlflowClient
    searches = []
    search_runs = client_type.search_runs

    def recorded(client, *args, **kwargs):
        searches.append(kwargs["filter_string"])
        return search_runs(client, *args, **kwargs)

    monkeypatch.setattr(client_type, "search_runs", recorded)
    # MLflow retries a create_run only over HTTP, so a local store holds no copies.
    local = MlflowTrainingObserver(tmp_path / "local")
    local.on_run_start({"run_id": "local", "status": "running"})
    local.on_run_end({"run_id": "local", "status": "finished", "final_step": 0})
    assert searches == []
    observe(backend("serve", artifact_root=(tmp_path / "remote").as_uri()))
    assert len(searches) == 1


@pytest.mark.parametrize(
    ("hooks", "receipts"),
    [
        (
            ("on_run_config", "on_step", "on_run_end"),
            [("log_parameters", False, True), ("delivery_tag", True, True)],
        ),
        (("on_step", "on_run_end"), [("log_metrics", False, True), ("delivery_tag", True, True)]),
        (("on_run_end",), [("lifecycle_tag", True, True)]),
    ],
    ids=["parameter_batch", "metric_batch", "tag"],
)
def test_async_logging_still_delivers_synchronously(
    tmp_path, backend, monkeypatch, hooks, receipts
):
    # A queued delivery would hide the failure and keep reconnecting from MLflow's
    # logging thread while delivery is paused.
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    remote = (tmp_path / "remote").as_uri()
    server = backend("serve", artifact_root=remote)
    config = TrackingConfig(tracking_uri=server.uri, http_max_retries=1)
    observer = MlflowTrainingObserver(tmp_path, config)
    records = {
        "on_run_config": {"run_id": "async", "seed": 3},
        "on_step": {"step": 1, "loss": 0.5},
        "on_run_end": {"run_id": "async", "status": "finished", "final_step": 1},
    }
    try:
        observer.on_run_start({"run_id": "async", "status": "running"})
        # The start resolved the artifact store; then the backend dies.
        server.mode = "close"
        with pytest.warns(RuntimeWarning):
            for hook in hooks:
                getattr(observer, hook)(records[hook])
    finally:
        observer.on_run_end(records["on_run_end"])
    # The failing write, then (while paused) the run end's immediate probe: 1 + 1
    # connections and one receipt each. In the tag case the write is the run end's
    # attempt.
    assert failures(tmp_path) == receipts
    assert len(server.connections("close")) == 2 * len(receipts)


def test_rejected_request_is_one_receipt_and_delivery_continues(tmp_path, backend):
    # A refused request (HTTP 4xx) says nothing about the transport; later
    # deliveries must still reach the backend.
    remote = (tmp_path / "remote").as_uri()
    server = backend("serve", artifact_root=remote, reject_once="/runs/log-batch")
    observer = MlflowTrainingObserver(tmp_path, TrackingConfig(tracking_uri=server.uri))
    observer.on_run_start({"run_id": "rejected", "status": "running"})
    with pytest.warns(RuntimeWarning):
        observer.on_run_config({"run_id": "rejected", "seed": 3})
    rejected_at = len(server.paths)
    observer.on_step({"step": 1, "loss": 0.5})
    observer.on_run_end({"run_id": "rejected", "status": "finished", "final_step": 1})
    assert failures(tmp_path) == [("log_parameters", False, False)]
    later = server.paths[rejected_at:]
    assert any(path.endswith("/runs/log-batch") for path in later)
    assert any(path.endswith("/runs/set-tag") for path in later)


def test_failing_training_restores_the_callers_http_environment_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "37")
    before = {name: os.environ.get(name) for name in HTTP_VARIABLES}
    original = MFM.loss_terms
    calls = 0

    def fail_second_update(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stopped update")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MFM, "loss_terms", fail_second_update)
    probe = HttpSettingsProbe()
    with pytest.raises(RuntimeError):
        tiny_run(
            tmp_path / "run",
            tracking=TrackingConfig(http_timeout=5, diagnostics=False),
            observers=[probe],
        )
    assert probe.seen == [
        {
            "MLFLOW_HTTP_REQUEST_TIMEOUT": 5,
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": 2,
            "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": 1,
            "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER": 0.0,
            "MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER": False,
        }
    ]
    assert {name: os.environ.get(name) for name in HTTP_VARIABLES} == before


@pytest.mark.parametrize(
    ("variable", "raw", "default"),
    [
        # MLflow parses the backoff factor as an int, so every request would fail.
        ("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "0.5", 1),
        # These parse, but a NaN or infinite jitter makes urllib3 sleep 120 s per retry.
        ("MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", "nan", 0.0),
        ("MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", "inf", 0.0),
        ("MLFLOW_HTTP_REQUEST_TIMEOUT", "-5", 10),
        # This parses too, but MLflow then fails every request.
        ("MLFLOW_HTTP_REQUEST_TIMEOUT", "0", 10),
    ],
)
def test_unusable_caller_value_gets_the_default_and_one_named_receipt(
    tmp_path, monkeypatch, variable, raw, default
):
    monkeypatch.setenv(variable, raw)
    probe = HttpSettingsProbe()
    out = tmp_path / "run"
    result = tiny_run(out, tracking=TrackingConfig(diagnostics=False), observers=[probe])
    assert result.status == "finished"
    assert result.run["tracking"]["delivery_status"] == "healthy"
    assert {seen[variable] for seen in probe.seen} == {default}
    receipts = http_receipts(out)
    assert [receipt["unparsable"] for receipt in receipts] == [[variable]]
    assert [receipt["conflicting"] for receipt in receipts] == [[]]
    assert os.environ[variable] == raw


def test_concurrent_runs_share_one_environment_until_the_last_one_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "37")
    first = MlflowTrainingObserver(tmp_path / "first", TrackingConfig(http_max_retries=1))
    second = MlflowTrainingObserver(tmp_path / "second", TrackingConfig(http_max_retries=3))
    try:
        first.on_run_start({"run_id": "first", "status": "running"})
        second.on_run_start({"run_id": "second", "status": "running"})
        second.on_run_end({"run_id": "second", "status": "finished"})
        # The earlier run's settings stay in force until it ends.
        retries = mlflow_environment.MLFLOW_HTTP_REQUEST_MAX_RETRIES.get()
    finally:
        second.on_run_end({"run_id": "second", "status": "finished"})
        first.on_run_end({"run_id": "first", "status": "finished"})
    assert retries == 1
    assert {name: os.environ.get(name) for name in HTTP_VARIABLES} == {
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "37",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": None,
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": None,
        "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER": None,
        "MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER": None,
    }
    assert http_receipts(tmp_path / "first") == []
    assert [receipt["conflicting"] for receipt in http_receipts(tmp_path / "second")] == [
        ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
    ]
    assert second.summary["delivery_status"] == "healthy"
