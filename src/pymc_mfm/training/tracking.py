"""Durable training evidence with optional, independently owned MLflow delivery.

No fluent MLflow API or autologging is used. Local JSON events are authoritative;
MLflow is a projection of their finite scalars and explicitly allowlisted files.
An unavailable dependency/backend never changes a numerical training operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import torch

from pymc_mfm.inference._artifact_schema import validate_profile
from pymc_mfm.training.checkpoint import CHECKPOINT_KINDS, file_sha256
from pymc_mfm.training.observer import TrainingObserver

_SECRET = re.compile(
    r"(?:^|_)(?:password|passwd|secret|token|authorization|api_key|access_key|credential)(?:$|_)",
    re.I,
)
_URI = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")
_KEY = re.compile(r"[^A-Za-z0-9_./ -]")
_SECRET_TEXT = re.compile(
    r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?|(?:api[_-]?key|password|passwd|secret|token)\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


def _secret_key(key: str) -> bool:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).replace("-", "_")
    return _SECRET.search(normalized) is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingConfig:
    """Delivery/report settings, deliberately separate from numerical training.

    ``tracking_uri`` takes precedence over ``MLFLOW_TRACKING_URI``. Otherwise
    the SQLite database and artifact store live below the run output directory.
    Diagnostics consume *existing* evaluation draws at every evaluation event;
    expensive reference fits, stress runs and SBC are never scheduled implicitly.

    ``http_timeout`` (seconds per request) and ``http_max_retries`` bound the
    MLflow HTTP client. Each takes precedence over the caller's
    ``MLFLOW_HTTP_REQUEST_TIMEOUT``/``MLFLOW_HTTP_REQUEST_MAX_RETRIES``, which
    take precedence over 10 s and 2 retries. Backoff factor 1 without jitter
    applies unless ``MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR``/``_JITTER`` are set,
    and a server's ``Retry-After`` answer does not stretch it unless
    ``MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER`` is true. A failed start stops
    delivery after one failure receipt. After a successful start,
    tracking-server writes and artifact uploads pause separately: a transport
    failure pauses its channel after one receipt, and the channel's later
    deliveries are skipped, never replayed, until a probe once per cool-down
    (60 s, doubling up to 600 s while the channel keeps failing) gets through.
    At the run end a paused channel probes at once, and the first transport
    failure on a channel is its one bounded run-end attempt. The local
    evidence stays complete.
    """

    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "pymc-mfm"
    profile: str = "new-run"
    diagnostics: bool = True
    http_timeout: int | None = None
    http_max_retries: int | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.diagnostics) is not bool:
            raise ValueError("enabled and diagnostics must be booleans")
        validate_profile(self.profile)
        if not self.experiment_name.strip() or not self.profile.strip():
            raise ValueError("experiment_name and profile must not be empty")
        if self.tracking_uri is not None and not self.tracking_uri.strip():
            raise ValueError("tracking_uri must not be empty")
        if self.http_timeout is not None and (
            type(self.http_timeout) is not int or self.http_timeout < 1
        ):
            raise ValueError("http_timeout must be None or an integer number of seconds >= 1")
        # MLflow rejects more than 10 retries (_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT).
        if self.http_max_retries is not None and (
            type(self.http_max_retries) is not int or not 0 <= self.http_max_retries <= 10
        ):
            raise ValueError("http_max_retries must be None or an integer from 0 to 10")


# MLflow reads these from the process environment on every request. Its own
# defaults (120 s timeout, 7 retries, exponential backoff with jitter) spend
# minutes on each call to an unreachable backend, and it waits as long as a
# 413, 429 or 503 answer's Retry-After header asks: up to 6 h per retry.
_HTTP_DEFAULTS: dict[str, bool | int | float] = {
    "MLFLOW_HTTP_REQUEST_TIMEOUT": 10,
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": 2,
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": 1,
    "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER": 0.0,
    "MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER": False,
}
_HTTP_FIELDS = {
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "http_timeout",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "http_max_retries",
}
# A parsed caller value below its minimum (0 unless listed) is unusable: MLflow
# fails every request with a zero timeout.
_HTTP_MINIMUMS = {"MLFLOW_HTTP_REQUEST_TIMEOUT": 1}
# Concurrently active observers share one process environment: the first one
# snapshots the caller's values and applies its settings, the last restores them.
_HTTP_LOCK = threading.Lock()
_HTTP_SHARED: dict[str, Any] = {
    "owners": 0,
    "previous": {},
    "caller": {},
    "unparsable": [],
    "applied": {},
}


def _apply_http_settings(config: TrackingConfig) -> dict[str, Any] | None:
    """Hold bounded MLflow HTTP settings for one run; describe any adjustment.

    Values resolve as field > caller environment > MFM default. Caller values
    are parsed with MLflow's own getters, so a value MLflow cannot parse falls
    back to the MFM default instead of failing every request. So does a parsed
    value that is non-finite, negative, or a timeout below 1 s: a NaN jitter,
    for example, would make urllib3 sleep its 120 s maximum between retries,
    and MLflow fails every request with a zero timeout. Upper bounds stay
    MLflow's to enforce, because a caller may have raised MLflow's own limit
    variables.
    The returned receipt (``None`` when nothing was adjusted) holds variable
    names and numeric settings only, never raw environment text.
    """
    from mlflow import environment_variables

    with _HTTP_LOCK:
        shared = _HTTP_SHARED
        if shared["owners"] == 0:
            caller, unparsable = {}, []
            for name in _HTTP_DEFAULTS:
                if name in os.environ:
                    try:
                        value = getattr(environment_variables, name).get()
                    except ValueError:
                        unparsable.append(name)
                        continue
                    if not math.isfinite(value) or value < _HTTP_MINIMUMS.get(name, 0):
                        unparsable.append(name)
                    else:
                        caller[name] = value
            shared.update(
                previous={name: os.environ.get(name) for name in _HTTP_DEFAULTS},
                caller=caller,
                unparsable=unparsable,
            )
        requested = {}
        for name, default in _HTTP_DEFAULTS.items():
            value = getattr(config, _HTTP_FIELDS[name]) if name in _HTTP_FIELDS else None
            requested[name] = shared["caller"].get(name, default) if value is None else value
        if shared["owners"] == 0:
            shared["applied"] = requested
            for name, value in requested.items():
                os.environ[name] = str(value)
        shared["owners"] += 1
        applied, unparsable = dict(shared["applied"]), list(shared["unparsable"])
    conflicting = [name for name, value in requested.items() if applied[name] != value]
    if not unparsable and not conflicting:
        return None
    return {
        "applied": applied,
        "requested": requested,
        "unparsable": unparsable,
        "conflicting": conflicting,
    }


def _restore_http_settings() -> None:
    with _HTTP_LOCK:
        _HTTP_SHARED["owners"] -= 1
        if _HTTP_SHARED["owners"] == 0:
            for name, value in _HTTP_SHARED["previous"].items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


# After a successful start, a transport failure pauses its delivery channel:
# the channel's deliveries are skipped for a cool-down, which doubles after each
# failed probe up to the cap and resets once a delivery on the channel succeeds.
_PAUSE_SECONDS = 60.0
_MAX_PAUSE_SECONDS = 600.0


@dataclass(slots=True)
class _Channel:
    """Delivery state of tracking-server writes or of artifact uploads.

    ``pause`` is the open pause record: its ``channel``, the ``sequence`` of the
    receipt that opened it, ``resumed_sequence`` (set when it closes),
    ``skipped_deliveries`` and ``failed_probes``. The next probe is due at
    ``probe_at``, which is None again once a probe got through; the pause closes
    when a delivery then succeeds. The totals count every pause of the run.
    """

    name: str
    pause: dict[str, Any] | None = None
    probe_at: float | None = None
    cooldown: float = _PAUSE_SECONDS
    final: bool = False
    stopped: bool = False
    pauses: int = 0
    skipped_deliveries: int = 0
    failed_probes: int = 0

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "open": None if self.pause is None else dict(self.pause),
            "pauses": self.pauses,
            "skipped_deliveries": self.skipped_deliveries,
            "failed_probes": self.failed_probes,
        }


def _transport_failure(exc: BaseException) -> bool:
    """Connection failure, timeout or exhausted retries, however MLflow wraps it.

    MLflow re-raises ``requests`` errors as ``MlflowException`` with the
    original as cause or context; retries exhausted on rate-limit or
    unavailable answers count too. After a successful start such a failure
    pauses its delivery channel. Other failures, such as a rejected request, a
    local file error or an S3/GCS/Azure client error that wraps no ``requests``
    error, affect one delivery and leave delivery running.
    """
    try:
        from requests import exceptions
    except ImportError:
        return False
    transport = (exceptions.ConnectionError, exceptions.Timeout, exceptions.RetryError)
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, transport):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _interrupt(exc: BaseException, outer: BaseException | None) -> BaseException | None:
    """The first non-``Exception`` (such as ``KeyboardInterrupt``) in ``exc``'s chain.

    A client can turn an interrupt into its own error while cleaning up (SQLAlchemy
    does, leaving the interrupt as ``__context__``); the interrupt must still win.
    The walk stops at ``outer``, the exception the caller was already handling:
    it and its chain predate the call.
    """
    seen = set()
    while exc is not None and exc is not outer and id(exc) not in seen:
        if not isinstance(exc, Exception):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def _safe_uri(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            parsed = urlsplit(match.group())

            def private(key):
                return (
                    _secret_key(key)
                    or key.lower() in {"auth", "sig", "signature"}
                    or key.lower().startswith(("x-amz-", "x-goog-"))
                )

            query = parse_qsl(parsed.query, keep_blank_values=True)
            safe_query = [(key, item) for key, item in query if not private(key)]
            private_fragment = any(private(key) for key, _ in parse_qsl(parsed.fragment))
            if "@" not in parsed.netloc and len(query) == len(safe_query) and not private_fragment:
                return match.group()
            host = parsed.netloc.rsplit("@", 1)[-1]
            return urlunsplit(
                (
                    parsed.scheme,
                    host,
                    parsed.path,
                    urlencode(safe_query) if len(query) != len(safe_query) else parsed.query,
                    "" if private_fragment else parsed.fragment,
                )
            )
        except ValueError:
            return "[redacted URI]"

    return _URI.sub(replace, value)


def evidence_value(value: Any) -> Any:
    """JSON evidence preserves undefined/nonfinite values without invalid JSON.

    Credential-like keys and URL credentials/query strings are never retained.
    Arrays remain structured evidence; they are not averaged into fake metrics.
    """
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if _secret_key(str(key)) else evidence_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [evidence_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return evidence_value(value.tolist())
    if isinstance(value, np.generic):
        return evidence_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"status": "nonfinite", "value": repr(value)}
    if isinstance(value, (str, Path)):
        return _SECRET_TEXT.sub(r"\1[redacted]", _safe_uri(str(value)))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(evidence_value(payload), indent=2, sort_keys=True, allow_nan=False)
    )
    temporary.replace(path)


def _leaves(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaves(item, (*path, str(key)))
    else:
        yield path, value


def _pointer(path: tuple[str, ...]) -> str:
    return "/".join(part.replace("~", "~0").replace("/", "~1") for part in path)


@contextmanager
def preserve_rng():
    """Keep delivery/rendering out of caller CPU and initialized accelerator RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    # MPS RNG queries initialize the device; only preserve an already used MPS
    # device, never initialize an accelerator for a CPU tracking operation.
    mps_state = None
    if torch.backends.mps.is_available() and torch.mps.current_allocated_memory() > 0:
        mps_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class MlflowTrainingObserver(TrainingObserver):
    """One owned client run plus complete local evidence and delivery receipts.

    ``tracking/state.json`` distinguishes numerical lifecycle from delivery
    health. Each event and every checkpoint version is retained even when
    delivery fails. Errors record operation/type, not backend exception strings
    (which can contain authentication headers, tokens, or observations).
    """

    def __init__(
        self,
        out_dir: str | Path,
        config: TrackingConfig | None = None,
        *,
        continue_run_id: str | None = None,
        continue_local: bool = False,
        tags: Mapping[str, str] | None = None,
    ):
        """``continue_run_id`` reopens an MLflow run. ``continue_local`` continues
        local events. ``tags`` are added to a newly created run.
        """
        self.out_dir = Path(out_dir).resolve()
        self.config = TrackingConfig() if config is None else config
        self.root = self.out_dir / "tracking"
        self.tracking_uri = (
            self.config.tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
            or f"sqlite:///{self.root / 'mlflow.db'}"
        )
        self.client = None
        self.run_id: str | None = None
        self.experiment_id: str | None = None
        self.local_run_id: str | None = None
        self.lifecycle_status = "running"
        self.delivery_status = "initializing" if self.config.enabled else "disabled"
        self.failure_count = 0
        self.delivery_seconds = 0.0
        self._sequence = 0
        self._warned = False
        self._names: dict[str, dict[str, Any]] = {}
        self._params: dict[str, str] = {}
        self._metric_type = None
        self._param_type = None
        self._started = False
        self._ended = False
        self._withheld_reports = False
        # Every attempt of this execution's create_run carries this random id as
        # the mfm.start_id tag, so its retried copies can be found (os.urandom
        # leaves the Python, NumPy and torch RNGs alone).
        self._start_id = uuid.uuid4().hex
        # Tracking-server writes and artifact uploads pause and stop separately
        # (see _deliver). Probes are due on _clock, a monotonic clock in seconds
        # that tests may replace.
        self._channels = {name: _Channel(name) for name in ("tracking", "artifacts")}
        self._clock = time.monotonic
        self._http_applied = False
        self._continue_run_id = continue_run_id
        self._extra_tags = {str(key): str(value) for key, value in (tags or {}).items()}
        # Parameters a reopened run already holds are immutable in MLflow: kept as recorded.
        self._inherited_params: set[str] = set()
        if continue_local:
            self._load_local_continuation()

    def _load_local_continuation(self) -> None:
        """Continue the event numbering and name table of an earlier execution of this run."""
        events = self.root / "events"
        numbers = [int(path.name[:7]) for path in events.glob("*.json") if path.name[:7].isdigit()]
        self._sequence = max(numbers, default=0)
        names = self.root / "names.json"
        if names.is_file():
            self._names = json.loads(names.read_text())

    @property
    def summary(self) -> dict[str, Any]:
        status = self.lifecycle_status
        if self.delivery_status in ("disabled", "degraded"):
            status = self.delivery_status
        return {
            "status": status,
            "lifecycle_status": self.lifecycle_status,
            "delivery_status": self.delivery_status,
            "enabled": self.config.enabled,
            "local_run_id": self.local_run_id,
            "start_id": self._start_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "tracking_uri": _safe_uri(self.tracking_uri),
            "experiment_name": self.config.experiment_name,
            "profile": self.config.profile,
            "failure_count": self.failure_count,
            "delivery_seconds": self.delivery_seconds,
            "delivery_pauses": {name: state.summary for name, state in self._channels.items()},
            "evidence": "tracking/events",
            "failures": "tracking/delivery_failures.jsonl",
            "closed_pauses": "tracking/delivery_pauses.jsonl",
            "name_table": "tracking/names.json",
        }

    def _state(self) -> None:
        _write_json(self.root / "state.json", self.summary)
        _write_json(self.root / "names.json", self._names)

    def _append(self, name: str, entry: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / name).open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
            handle.flush()

    def _failure(
        self,
        operation: str,
        exc: BaseException,
        *,
        artifact: str | None = None,
        stop: bool = False,
        transport: bool = False,
    ) -> None:
        """One receipt; a caller that pauses or stops a channel has done so already."""
        self.failure_count += 1
        self.delivery_status = "degraded"
        entry = {
            "sequence": self._sequence,
            "operation": operation,
            "error_type": type(exc).__name__,
            "artifact": artifact,
            "transport": transport,
            "delivery_stopped": stop,
            "timestamp_ms": int(time.time() * 1000),
            "reason": (
                "See retained local evidence; backend exception text is withheld for privacy."
            ),
        }
        try:
            self._append("delivery_failures.jsonl", entry)
            self._state()
        except Exception:
            # A broken output filesystem also cannot hold delivery-failure
            # evidence. The trainer's independent records remain authoritative.
            pass
        if not self._warned:
            self._warned = True
            try:
                warnings.warn(
                    "MLflow tracking is degraded; numerical training continues. "
                    "Inspect tracking/state.json and tracking/delivery_failures.jsonl. "
                    "Install pymc-mfm[training] for MLflow support, "
                    "or explicitly disable tracking.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            except Warning:
                pass

    def _guard(self, operation: str, function) -> Any:
        started = time.perf_counter()
        try:
            with preserve_rng():
                return function()
        except Exception as exc:
            self._failure(operation, exc)
            return None
        finally:
            self.delivery_seconds += time.perf_counter() - started

    def _deliver(
        self, operation: str, function, *, artifact: str | None = None, channel: str = "tracking"
    ) -> None:
        """One delivery on its channel, ``"tracking"`` or ``"artifacts"``.

        A failure is a receipt, never an exception, and a failure on one
        channel never skips a delivery on the other.
        """
        state = self._channels[channel]
        if not self.config.enabled or state.stopped:
            return
        if self.client is None or self.run_id is None:
            self._failure(operation, RuntimeError("backend unavailable"), artifact=artifact)
            return
        # Skipped deliveries are counted, not queued; the local events stay complete.
        if state.probe_at is not None and not self._probe(state):
            self._skip(state)
            return
        try:
            function()
        except Exception as exc:
            if not _transport_failure(exc):
                # The backend answered: one receipt, and delivery continues.
                self._failure(operation, exc, artifact=artifact)
            elif not self._lost(state, operation, exc, artifact=artifact):
                self._skip(state)
        else:
            if state.pause is not None:
                self._resume(state)

    def _probe(self, state: _Channel) -> bool:
        """Whether a paused channel attempts the delivery at hand.

        A paused channel is probed once per elapsed cool-down, and at once
        during the run end. An upload probes with itself. A tracking-server
        write is preceded by a rewrite of the run's delivery-status tag, so the
        remote run shows the degraded delivery as soon as the server answers
        again; the write at hand follows once the tag got through.
        """
        if not state.final and self._clock() < state.probe_at:
            return False
        if state.name == "tracking":
            try:
                self.client.set_tag(
                    self.run_id, "mfm.delivery_status", self.delivery_status, synchronous=True
                )
            except Exception as exc:
                if _transport_failure(exc):
                    self._lost(state, "delivery_tag", exc)
                    return False
                # The server answered; only this tag was refused.
                self._failure("delivery_tag", exc)
        state.probe_at = None
        return True

    def _lost(
        self, state: _Channel, operation: str, exc: BaseException, *, artifact: str | None = None
    ) -> bool:
        """A transport failure on a channel; whether it got a receipt of its own.

        Each later call to an unreachable backend would spend the whole
        timeout/retry schedule on the training thread. So a failure while the
        channel delivers opens a pause with one receipt. A failed probe, or a
        failed delivery right after a probe got through, continues the open
        pause and doubles its cool-down. During the run end the failure is the
        channel's one bounded attempt: one receipt, and the channel stops.
        """
        if state.final:
            state.stopped = True
        elif state.pause is None:
            state.pause = {
                "channel": state.name,
                "sequence": self._sequence,
                "resumed_sequence": None,
                "skipped_deliveries": 0,
                "failed_probes": 0,
            }
            state.pauses += 1
            state.probe_at = self._clock() + state.cooldown
        else:
            state.pause["failed_probes"] += 1
            state.failed_probes += 1
            state.cooldown = min(2 * state.cooldown, _MAX_PAUSE_SECONDS)
            state.probe_at = self._clock() + state.cooldown
            return False
        self._failure(operation, exc, artifact=artifact, stop=state.final, transport=True)
        return True

    def _skip(self, state: _Channel) -> None:
        state.pause["skipped_deliveries"] += 1
        state.skipped_deliveries += 1

    def _resume(self, state: _Channel) -> None:
        """A delivery succeeded on a paused channel: its pause closes into the ledger."""
        state.pause["resumed_sequence"] = self._sequence
        try:
            self._append("delivery_pauses.jsonl", state.pause)
        except OSError:
            # A broken output filesystem cannot hold the ledger; the totals remain.
            pass
        state.pause = None
        state.cooldown = _PAUSE_SECONDS

    def metric_name(
        self, original: str, *, path: tuple[str, ...] = (), kind: str = "metric"
    ) -> str:
        """Stable bounded key; the full original name/path remains in names.json.

        All names carry their complete SHA-256, including already legal names,
        so a raw name cannot impersonate a sanitized/truncated name. An actual
        digest collision is detected, never silently merged.
        """
        digest = hashlib.sha256(original.encode()).hexdigest()
        readable = _KEY.sub("_", original).strip(" /.") or "unnamed"
        key = f"{readable[:184]}--{digest}"
        entry = {"original": original, "path": list(path), "kind": kind}
        existing = self._names.get(key)
        if existing is not None and not path and existing["original"] == original:
            return key
        if existing is not None and existing != entry:
            raise ValueError("tracking name identity collision")
        self._names[key] = entry
        return key

    def _artifact(self, source: Path, destination: str) -> None:
        self._deliver(
            "log_artifact",
            lambda: self.client.log_artifact(self.run_id, str(source), artifact_path=destination),
            artifact=f"{destination}/{source.name}",
            channel="artifacts",
        )

    def _snapshot(
        self, source: Path, *, kind: str, step: int, deliver: bool = True
    ) -> dict[str, Any]:
        if not source.is_file() or source.is_symlink():
            raise ValueError("artifact must be a regular non-symlink file")
        digest = file_sha256(source)
        relative = Path("objects") / f"step_{step:09d}" / kind / digest / source.name
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        if deliver:
            self._artifact(target, relative.parent.as_posix())
        return {"path": relative.as_posix(), "sha256": digest, "kind": kind, "step": step}

    def _event(
        self,
        kind: str,
        record: Mapping[str, Any],
        *,
        metrics=None,
        metric_steps=None,
        deliver: bool = True,
    ) -> Path:
        self._sequence += 1
        payload = evidence_value(record)
        step = int(record.get("step", record.get("final_step", 0)))
        timestamp = int(time.time() * 1000)
        scalars = []
        for original, path, _value in metrics or ():
            value = payload
            for part in path:
                value = value.get(part) if isinstance(value, Mapping) else None
            if isinstance(value, Real) and math.isfinite(float(value)):
                key = self.metric_name(original, path=path)
                scalar_step = (
                    step if metric_steps is None else int(metric_steps.get(original, step))
                )
                scalars.append(
                    {
                        "key": key,
                        "original": original,
                        "path": list(path),
                        "value": float(value),
                        "step": scalar_step,
                    }
                )
        scalar_paths = {tuple(item["path"]) for item in scalars}
        fields = [
            {
                "path": list(path),
                "representation": "metric_and_json" if path in scalar_paths else "json",
                "status": "undefined" if value is None else "recorded",
            }
            for path, value in _leaves(payload)
        ]
        event = {
            "format_version": 1,
            "sequence": self._sequence,
            "kind": kind,
            "step": step,
            "timestamp_ms": timestamp,
            "payload": payload,
            "scalars": scalars,
            "fields": fields,
        }
        path = self.root / "events" / f"{self._sequence:07d}_{kind}_step_{step:09d}.json"
        _write_json(path, event)
        if deliver:
            self._artifact(path, "events")
        if scalars and deliver:
            for offset in range(0, len(scalars), 1000):
                batch = scalars[offset : offset + 1000]
                self._deliver(
                    "log_metrics",
                    lambda batch=batch: self.client.log_batch(
                        self.run_id,
                        metrics=[
                            self._metric_type(item["key"], item["value"], timestamp, item["step"])
                            for item in batch
                        ],
                        synchronous=True,
                    ),
                    artifact=str(path.relative_to(self.root)),
                )
        self._state()
        return path

    def _parameters(self, record: Mapping[str, Any]) -> None:
        pending = []
        for path, value in _leaves(evidence_value(record)):
            original = "config/" + _pointer(path)
            key = self.metric_name(original, path=path, kind="parameter")
            encoded = json.dumps(value, sort_keys=True, allow_nan=False)
            # Full values are in the configuration event, never silently lost.
            if len(encoded) > 500:
                encoded = (
                    "sha256:"
                    + hashlib.sha256(encoded.encode()).hexdigest()
                    + " (full value in configuration event)"
                )
            if key in self._inherited_params:
                continue
            if key in self._params:
                if self._params[key] != encoded:
                    raise ValueError("immutable tracking parameter changed")
                continue
            self._params[key] = encoded
            pending.append((key, encoded))
        for offset in range(0, len(pending), 100):
            batch = pending[offset : offset + 100]
            self._deliver(
                "log_parameters",
                lambda batch=batch: self.client.log_batch(
                    self.run_id,
                    params=[self._param_type(key, value) for key, value in batch],
                    synchronous=True,
                ),
            )

    def _end_duplicate_runs(self) -> None:
        """End the extra runs a retried ``create_run`` left behind: one bounded search.

        Only MLflow's HTTP client retries a request, so only a tracking server
        (``http``, ``https`` or ``databricks``) can hold them: up to retries + 1
        running runs carry this observer's ``mfm.start_id``. Each one that is
        not ours is tagged ``mfm.duplicate_of`` with our run id and terminated
        FAILED. This is housekeeping, not delivery: a failure is a local
        ``duplicate_search_failed`` event, pauses nothing and leaves the
        delivery status as it is.
        """
        scheme = urlsplit(self.tracking_uri).scheme
        if scheme not in ("http", "https", "databricks") and self.tracking_uri != "databricks":
            return
        from mlflow.entities import ViewType
        from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_MAX_RETRIES

        operation = "find_duplicate_runs"
        try:
            runs = self.client.search_runs(
                [self.experiment_id],
                filter_string=(
                    f"tags.`mfm.start_id` = '{self._start_id}' and attributes.status = 'RUNNING'"
                ),
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=MLFLOW_HTTP_REQUEST_MAX_RETRIES.get() + 1,
            )
            found = [run.info.run_id for run in runs if run.info.run_id != self.run_id]
            if found:
                self._event("duplicate_runs", {"count": len(found), "run_ids": found})
            operation = "end_duplicate_run"
            for run_id in found:
                self.client.set_tag(run_id, "mfm.duplicate_of", self.run_id, synchronous=True)
                self.client.set_terminated(run_id, status="FAILED")
        except Exception as exc:
            # Local only, and the error type without backend text, as in receipts.
            self._event(
                "duplicate_search_failed",
                {"operation": operation, "error_type": type(exc).__name__},
                deliver=False,
            )

    def _reopen_run(self):
        """The MLflow run of an interrupted execution, set RUNNING again, or ``None``.

        ``None`` (run unknown to this backend, or tracking pointed elsewhere) makes the
        caller create a new run, tagged with the run it could not continue. Metric
        history the interrupted execution logged for steps after the checkpoint stays
        in MLflow beside the resumed values; the local files are authoritative.
        """
        if self._continue_run_id is None:
            return None
        try:
            existing = self.client.get_run(self._continue_run_id)
        except Exception:
            self._extra_tags["mfm.continues_unavailable_run_id"] = self._continue_run_id
            return None
        self.experiment_id = existing.info.experiment_id
        self.client.update_run(self._continue_run_id, status="RUNNING")
        for key, value in (
            ("mfm.start_id", self._start_id),
            ("mfm.lifecycle_status", "running"),
            ("mfm.delivery_status", "healthy"),
            ("mfm.resumed", "true"),
        ):
            self.client.set_tag(self._continue_run_id, key, value, synchronous=True)
        self._inherited_params = set(existing.data.params)
        return existing

    def on_run_start(self, run: Mapping[str, Any]) -> None:
        def start():
            if self._started:
                raise RuntimeError("a tracking observer owns exactly one training execution")
            self._started = True
            self.local_run_id = str(run["run_id"])
            self._state()
            http_receipt = None
            if self.config.enabled:
                # Every exception raised below chains to what the caller is handling.
                outer = sys.exc_info()[1]
                try:
                    from mlflow import MlflowClient
                    from mlflow.entities import Metric, Param

                    self._metric_type, self._param_type = Metric, Param
                    # Before the trainer starts any thread; on_run_end restores them.
                    http_receipt = _apply_http_settings(self.config)
                    self._http_applied = True
                    self.client = MlflowClient(tracking_uri=self.tracking_uri)
                    experiment = self.client.get_experiment_by_name(self.config.experiment_name)
                    if experiment is None:
                        kwargs = {}
                        if not self.config.tracking_uri and not os.environ.get(
                            "MLFLOW_TRACKING_URI"
                        ):
                            kwargs["artifact_location"] = (self.root / "artifacts").as_uri()
                        self.experiment_id = self.client.create_experiment(
                            self.config.experiment_name, **kwargs
                        )
                    else:
                        self.experiment_id = experiment.experiment_id
                    created = self._reopen_run() or self.client.create_run(
                        self.experiment_id,
                        tags={
                            "mlflow.runName": evidence_value(
                                str(run.get("run_name") or self.local_run_id)
                            ),
                            "mfm.local_run_id": self.local_run_id,
                            "mfm.start_id": self._start_id,
                            "mfm.profile": self.config.profile,
                            "mfm.lifecycle_status": "running",
                            "mfm.delivery_status": "healthy",
                            **self._extra_tags,
                        },
                    )
                    self.run_id = created.info.run_id
                    self.delivery_status = "healthy"
                    self._state()
                except BaseException as exc:
                    # Without a backend run nothing can be delivered: one receipt, then
                    # stop. An interrupted start (Ctrl-C) is a failed start too, also
                    # when a client wrapped the interrupt in its own error.
                    interrupt = _interrupt(exc, outer)
                    stop = self.run_id is None
                    if stop:
                        for state in self._channels.values():
                            state.stopped = True
                    self._failure(
                        "start_run",
                        exc if interrupt is None else interrupt,
                        stop=stop,
                        transport=_transport_failure(exc),
                    )
                    if interrupt is exc:
                        raise
                    if interrupt is not None:
                        raise interrupt from exc
            self._event("run_start", run)
            self._snapshot(
                Path(__file__).with_name("tracking_catalog.json"), kind="catalog", step=0
            )
            self._event("tracking_config", asdict(self.config))
            if http_receipt is not None:
                self._event("http_settings", http_receipt)
            if self.run_id is not None:
                self._end_duplicate_runs()

        self._guard("run_start", start)

    def on_run_config(self, run: Mapping[str, Any]) -> None:
        def configure():
            # Corpus statistics are step-0 corpus/* metrics rather than parameters.
            corpus = run.get("corpus_summary") or {}
            self._event(
                "configuration",
                run,
                metrics=[
                    ("corpus/" + _pointer(path), ("corpus_summary", *path), value)
                    for path, value in _leaves(corpus)
                ],
            )
            mutable = {
                "status",
                "finished_at",
                "final_step",
                "error",
                "error_type",
                "checkpoints",
                "diagnostics",
                "best",
                "tracking",
                "consumed",
            }
            self._parameters(
                {
                    key: value
                    for key, value in run.items()
                    if key not in mutable and key != "corpus_summary"
                }
            )
            self._state()

        self._guard("run_config", configure)

    def on_step(self, record: Mapping[str, Any]) -> None:
        def log():
            self._event(
                "step",
                record,
                metrics=[
                    ("train/" + _pointer(path), path, value) for path, value in _leaves(record)
                ],
            )
            if record["step"] == 1 and (self.out_dir / "objective_trace.json").is_file():
                self._snapshot(
                    self.out_dir / "objective_trace.json", kind="objective_trace", step=1
                )

        self._guard("step", log)

    def on_evaluation(self, record: Mapping[str, Any]) -> None:
        def log():
            metrics = [
                ("evaluation/" + key, ("flat", key), value)
                for key, value in record.get("flat", {}).items()
            ]
            metrics.extend(
                ("evaluation/" + _pointer(path), path, value)
                for path, value in _leaves(record)
                if path and path[0] not in ("flat", "legs")
            )
            # Heads and counts reach MLflow through flat; metric_worlds, tables
            # and per_world stay JSON evidence in the event, never metrics.
            skipped = (
                "heads",
                "counts",
                "n_worlds",
                "n_samples",
                "seed",
                "metric_worlds",
                "tables",
                "per_world",
            )
            for leg, values in record.get("legs", {}).items():
                metrics.extend(
                    (f"evaluation/{leg}/{key}", ("legs", leg, key), values[key])
                    for key in ("n_worlds", "n_samples", "seed")
                    if key in values
                )
                metrics.extend(
                    (f"evaluation/{leg}/" + _pointer(path), ("legs", leg, *path), value)
                    for path, value in _leaves(values)
                    if path and path[0] not in skipped
                )
            self._event("evaluation", record, metrics=metrics)

        self._guard("evaluation", log)

    def on_checkpoint(self, record: Mapping[str, Any]) -> None:
        def log():
            try:
                source = Path(record["path"])
                if record["kind"] not in CHECKPOINT_KINDS:
                    raise ValueError("unknown checkpoint kind")
                if file_sha256(source) != record["sha256"]:
                    raise ValueError("checkpoint changed before delivery")
                artifact = self._snapshot(
                    source,
                    kind="checkpoint_" + str(record["kind"]),
                    step=int(record["step"]),
                    deliver=False,
                )
            except Exception as exc:
                self._event(
                    "checkpoint",
                    {
                        **record,
                        "artifact_status": "failed",
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            # The upload follows its event, so an upload that closes a pause
            # records this event as the one whose delivery resumed.
            self._event("checkpoint", {**record, "artifact": artifact})
            self._artifact(self.root / artifact["path"], Path(artifact["path"]).parent.as_posix())

        self._guard("checkpoint", log)

    def on_diagnostic(self, manifest: Mapping[str, Any], path: str | Path) -> None:
        """Deliver only declared-synthetic reports; keep private evidence local."""

        def log():
            root = Path(path).resolve()
            if root.is_file():
                root = root.parent
            provenance = manifest.get("provenance", {})
            dimensions = manifest.get("dimensions", {})
            source_step = provenance.get("source_step")
            step = 0 if source_step is None else int(source_step)
            checkpoint = provenance.get("checkpoint_sha256", "unbound")
            if not isinstance(checkpoint, str) or re.fullmatch(r"[a-f0-9]{64}", checkpoint) is None:
                checkpoint = "unbound"
            identity = hashlib.sha256(
                json.dumps(evidence_value(manifest), sort_keys=True).encode()
            ).hexdigest()
            prefix = f"diagnostics/step_{step:09d}/{checkpoint}/{identity}"
            sensitive = provenance.get("synthetic_training_evaluation") is not True or any(
                item.get("sensitive", False) for item in manifest.get("artifacts", [])
            )
            unsafe_metadata = json.dumps(evidence_value(manifest), sort_keys=True) != json.dumps(
                manifest, sort_keys=True
            )
            saved_manifest = self.root / "reports" / identity / "manifest.json"
            _write_json(saved_manifest, manifest)
            approved = []
            invalid_artifacts = False
            # JSON evidence can carry labels later rendered into PNG/HTML.
            # Inspect the whole allowlist before releasing any derivative.
            if not sensitive and not unsafe_metadata:
                for descriptor in manifest.get("artifacts", []):
                    relative = Path(descriptor["path"])
                    source = root / relative
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or source.is_symlink()
                        or not source.resolve().is_relative_to(root)
                    ):
                        invalid_artifacts = True
                        self._failure(
                            "diagnostic_allowlist",
                            ValueError("unsafe artifact path"),
                            artifact=str(relative),
                        )
                        continue
                    try:
                        if not source.is_file() or file_sha256(source) != descriptor["sha256"]:
                            raise ValueError("diagnostic artifact missing or changed")
                        if source.suffix == ".json":
                            contents = json.loads(source.read_text())
                            unsafe_metadata = (
                                unsafe_metadata or evidence_value(contents) != contents
                            )
                        approved.append((source, relative))
                    except Exception as exc:
                        invalid_artifacts = True
                        self._failure("diagnostic_artifact", exc, artifact=str(relative))
            if sensitive or unsafe_metadata or invalid_artifacts:
                self._withheld_reports = True
                withheld = (
                    "withheld_sensitive"
                    if sensitive
                    else "withheld_credentials"
                    if unsafe_metadata
                    else "failed_artifact_validation"
                )
                # Even one-week support summaries can equal raw observations.
                # No private provenance, scalar metrics, filenames or figures
                # leave the local evidence directory.
                self._event(
                    "diagnostic",
                    {
                        "step": step,
                        "manifest": manifest,
                        "delivery_status": withheld,
                    },
                    deliver=False,
                )
                self._event(
                    "diagnostic_receipt",
                    {
                        "step": step,
                        "checkpoint_sha256": checkpoint,
                        "report_sha256": identity,
                        "status": withheld,
                        "reason": (
                            "Full report upload requires explicitly synthetic, nonsensitive, "
                            "credential-free evidence."
                        ),
                    },
                )
                return
            report_dimensions = {
                "data_split": dimensions.get(
                    "data_split", provenance.get("data_split", "unspecified")
                ),
                "head": dimensions.get("head", "conditional"),
                "mode": dimensions.get("mode", "unspecified"),
                "window": dimensions.get("window", "unspecified"),
                "units": dimensions.get("units", "unspecified"),
                "world_id": provenance.get("world_id", provenance.get("row", "world")),
            }
            metrics = []
            metric_steps = {}
            representation = manifest.get("metric_representation", "scalars")
            if representation not in ("scalars", "per_world_table"):
                raise ValueError("unknown diagnostic metric representation")
            for key, value in manifest.get("metrics", {}).items():
                # Captured training worlds are a growing table, not a growing
                # set of run-level time series. Their full values remain in the
                # verified manifest and arrays. Canonical train/evaluation
                # histories already preserve their actual optimizer steps.
                if representation == "per_world_table":
                    continue
                specific = manifest.get("metric_dimensions", {}).get(key, {})
                labels = {
                    **report_dimensions,
                    "units": manifest.get("metric_units", {}).get(key, report_dimensions["units"]),
                    **specific,
                }
                dimensions_key = tuple(
                    str(labels.get(name, "unspecified"))
                    for name in (
                        "data_split",
                        "head",
                        "mode",
                        "window",
                        "units",
                        "world_id",
                        "profile",
                        "method",
                        "checkpoint_sha256",
                    )
                )
                original = "diagnostic/" + _pointer((*dimensions_key, key))
                metrics.append((original, ("manifest", "metrics", key), value))
                source_step = specific.get("source_step", step)
                metric_steps[original] = step if source_step is None else int(source_step)
            self._event(
                "diagnostic",
                {"step": step, "manifest": manifest, "artifact_path": prefix},
                metrics=metrics,
                metric_steps=metric_steps,
            )
            self._artifact(saved_manifest, prefix)
            for source, relative in approved:
                self._artifact(source, f"{prefix}/{relative.parent.as_posix()}".rstrip("/."))
            self._state()

        self._guard("diagnostic", log)

    def on_diagnostic_failure(self, record: Mapping[str, Any]) -> None:
        def log():
            self._event("diagnostic_failure", record, deliver=False)
            self._event(
                "diagnostic_failure_receipt",
                {
                    "step": record.get("step", 0),
                    "status": "failed",
                    "error_type": record.get("error_type", "DiagnosticError"),
                },
            )

        self._guard("diagnostic_failure", log)
        self._failure("build_report", RuntimeError("diagnostic report failed"))

    def on_run_end(self, run: Mapping[str, Any]) -> None:
        if self._ended:
            return
        self._ended = True
        try:
            self._finalize(run)
        finally:
            # The caller's MLflow HTTP environment is restored on every exit path.
            if self._http_applied:
                self._http_applied = False
                _restore_http_settings()

    def _finalize(self, run: Mapping[str, Any]) -> None:
        self.lifecycle_status = str(run["status"])
        # The run end waits for no cool-down: a paused channel probes at its
        # first terminal delivery, and the first transport failure on a channel
        # is that channel's one bounded run-end attempt.
        for state in self._channels.values():
            state.final = True
        tracking = self._channels["tracking"]
        finished = False

        def upload(*names: str) -> None:
            for name in names:
                source = self.root / name
                if source.is_file():
                    self._artifact(source, "tracking")

        def finish():
            nonlocal finished
            step = int(run.get("final_step", 0))
            self._event("run_end", {**run, "tracking": self.summary})
            for name in (
                "run.json",
                "metrics.jsonl",
                "evaluations.jsonl",
                "diagnostics.jsonl",
                "objective_trace.json",
            ):
                source = self.out_dir / name
                if source.is_file():
                    try:
                        public = name != "diagnostics.jsonl" or (
                            not self._withheld_reports
                            and bool(run.get("corpus"))
                            and all(
                                item.get("data_origin") == "synthetic" for item in run["corpus"]
                            )
                        )
                        self._snapshot(source, kind="terminal_evidence", step=step, deliver=public)
                    except Exception as exc:
                        self._failure("terminal_artifact", exc, artifact=name)
            # Export paths are explicitly recorded by the trainer, not a
            # recursive working-directory scan.
            for name in run.get("export", {}).get("files", []):
                relative = Path(name)
                source = self.out_dir / "model" / relative
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not source.resolve().is_relative_to(self.out_dir / "model")
                ):
                    self._failure(
                        "export_allowlist", ValueError("unsafe export path"), artifact=str(relative)
                    )
                    continue
                try:
                    self._snapshot(source, kind="model_export", step=step)
                except Exception as exc:
                    self._failure("export_artifact", exc, artifact=str(relative))
            self._state()
            upload("names.json", "state.json", "delivery_failures.jsonl", "delivery_pauses.jsonl")
            finished = True

        def terminate():
            terminal = {"finished": "FINISHED", "interrupted": "KILLED", "failed": "FAILED"}.get(
                self.lifecycle_status, "FAILED"
            )
            paused = tracking.pause is not None
            # A backend that answers again does not keep the remote run RUNNING:
            # the tags and status get the tracking channel's one run-end attempt,
            # after its probe if the channel is paused.
            self._deliver(
                "lifecycle_tag",
                lambda: self.client.set_tag(
                    self.run_id, "mfm.lifecycle_status", self.lifecycle_status, synchronous=True
                ),
            )
            self._deliver(
                "delivery_tag",
                lambda: self.client.set_tag(
                    self.run_id, "mfm.delivery_status", self.delivery_status, synchronous=True
                ),
            )
            # FINISHED is never asserted when delivery is incomplete.
            if self.delivery_status == "degraded" and terminal == "FINISHED":
                terminal = "FAILED"
            self._deliver(
                "terminate_run", lambda: self.client.set_terminated(self.run_id, status=terminal)
            )
            self._state()
            # finish() uploaded the state and the pause ledger before these calls
            # closed the tracking pause: upload both again. After an interrupted
            # finish(), the tags and status stay the run end's only network calls.
            if finished and paused and tracking.pause is None:
                upload("state.json", "delivery_pauses.jsonl")

        try:
            self._guard("run_end", finish)
        except BaseException as exc:
            if self.lifecycle_status in ("running", "finished"):
                self.lifecycle_status = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
            self._failure("finalization_interrupted", exc)
            error_type = type(exc).__name__
            # Local only: after an interruption, the terminate attempt below is
            # the run end's only network call.
            self._guard(
                "interruption_event",
                lambda: self._event(
                    "finalization_interrupted",
                    {
                        "status": self.lifecycle_status,
                        "error_type": error_type,
                        "final_step": run.get("final_step", 0),
                    },
                    deliver=False,
                ),
            )
            raise
        finally:
            # KeyboardInterrupt during artifact delivery must still attempt to
            # terminate our run, never an unrelated caller's active run.
            try:
                self._guard("terminate", terminate)
            except BaseException as exc:
                if self.lifecycle_status in ("running", "finished"):
                    self.lifecycle_status = (
                        "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    )
                self._failure("termination_interrupted", exc)
                raise
