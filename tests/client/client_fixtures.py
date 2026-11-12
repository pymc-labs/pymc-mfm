"""Helpers for the client tests: a scripted fake service and wire-shaped records."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .contract_check import check_response

BASE_URL = "https://api.example.com"
EXPIRES = "2030-01-08T00:00:00Z"

Reply = tuple[int, Any] | tuple[int, Any, dict[str, str]] | Exception


class FakeService:
    """A scripted service for ``httpx.MockTransport``.

    ``on(method, path, *replies)`` queues replies for a route; the last one repeats.
    A reply is ``(status, json_body[, headers])``, or an exception to raise.
    Every request is kept in ``requests``.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[Reply]] = {}
        self.requests: list[httpx.Request] = []

    def on(self, method: str, path: str, *replies: Reply) -> None:
        self.routes[(method, path)] = list(replies)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key not in self.routes:
            return httpx.Response(404, json=error_body("not_found", f"no route {key}"))
        replies = self.routes[key]
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        status, body, *headers = reply
        if not isinstance(body, bytes):
            check_response(request.method, request.url.path, status, body)
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, content=content, headers=headers[0] if headers else None)

    def sent(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if (r.method, r.url.path) == (method, path)]

    @staticmethod
    def body(request: httpx.Request) -> Any:
        return json.loads(request.content)


def error_body(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    detail = {"code": code, "message": message}
    if field:
        detail["field"] = field
    return {"error": detail}


def band(low: float, high: float, top: float) -> dict[str, float]:
    return {"p05": low, "p95": high, "max": top}


def dataset_record(
    channels: tuple[str, ...] = ("tv", "search"),
    controls: tuple[str, ...] = ("price",),
    *,
    dataset_id: str = "ds_123",
    content_hash: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """The 201 and 200 body of the dataset endpoints. Without ``content_hash``, an older
    server: the lifecycle fields are absent."""
    lifecycle = (
        {
            "content_hash": content_hash,
            "created_at": "2029-12-25T00:00:00Z",
            "last_used_at": "2029-12-26T00:00:00Z",
            "analyses": 2,
            "name": name,
        }
        if content_hash
        else {}
    )
    return {
        **lifecycle,
        "dataset_id": dataset_id,
        "expires_at": EXPIRES,
        "summary": {
            "weeks": 4,
            "start": "2024-07-01",
            "end": "2024-07-22",
            "channels": list(channels),
            "controls": list(controls),
        },
        "history_ranges": {
            name: {**band(1.0, 9.0, 10.0), "zero_share": 0.25, "all_zero": False}
            for name in (*channels, *controls)
        },
        "warnings": ["history is shorter than 75 weeks"],
    }


def started(runs_total: int = 3) -> dict[str, Any]:
    return {
        "analysis_id": "an_1",
        "runs_total": runs_total,
        "estimate_seconds": 42.0,
        "expires_at": EXPIRES,
    }


def preview_record(runs_total: int = 2) -> dict[str, Any]:
    """The 200 body of ``POST /v1/analyses`` with ``"preview": true``. Nothing is stored."""
    return {
        "plans": [],
        "runs_total": runs_total,
        "estimate_seconds": 42.0,
        "assumptions": {},
        "warnings": [],
    }


def status_record(
    status: str = "done", runs_total: int = 3, runs_failed: int = 0, **extra: Any
) -> dict[str, Any]:
    """The body of ``GET /v1/analyses/{id}``. ``extra`` adds fields such as ``dataset_id``,
    ``context``, ``seed`` and ``draws``."""
    return {
        "status": status,
        "runs_total": runs_total,
        "runs_done": runs_total if status in ("done", "failed") else 1,
        "runs_failed": runs_failed,
        "estimate_seconds": 42.0,
        "expires_at": EXPIRES,
        **extra,
    }


COMPONENTS = ("outcome", "baseline", "tv", "search", "price", "latent")


def result_rows(
    scenario_id: str,
    kind: str,
    *,
    base: str | None = None,
    channel: str | None = None,
    level: float | None = None,
    multipliers: dict[str, Any] | None = None,
    tv_mean: float = 100.0,
    search_mean: float = 50.0,
    failed: bool = False,
    spend_total: float = 300.0,
    url: str | None = None,
    url_expires: str = EXPIRES,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One plan's rows in the result table, shaped like the service's ``ResultRow``."""
    means = {
        "baseline": 500.0,
        "tv": tv_mean,
        "search": search_mean,
        "price": -5.0,
        "latent": 20.0,
    }
    means["outcome"] = sum(means.values())
    rows = []
    for component in COMPONENTS:
        mean = means[component]
        rows.append(
            {
                "scenario_id": scenario_id,
                "kind": kind,
                "base_scenario_id": base,
                "varied_channel": channel,
                "level": level,
                "multipliers": multipliers,
                "run_id": None if failed else f"run_{scenario_id}",
                "status": "failed" if failed else "ok",
                "spend_total": spend_total,
                "component": component,
                "mean": None if failed else mean,
                "p05": None if failed else mean - 10.0,
                "p95": None if failed else mean + 10.0,
                "window_mean": None if failed else mean / 2,
                "mean_difference": None if (failed or kind == "baseline") else mean * 0.1,
                "simulation_se": None if (failed or kind == "baseline") else 0.5,
                "sensitivity": False,
                "warnings": list(warnings or []),
                "draws_url": url,
                "draws_url_expires_at": url_expires if url else None,
            }
        )
    return rows


def result_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "table": rows,
        "history": None,
        "assumptions": {
            "baseline": {"price": {"values": [10.5, 10.5], "history": band(9.9, 10.5, 10.5)}}
        },
        "meta": {"seed": 7, "draws": 1000, "paired_on_one_host": True, "hosts": ["h1"]},
        "labels": {
            "intervals": "Pointwise 5th-95th percentiles of expected outcome.",
            "simulation_se": "Numerical precision of the mean difference only.",
            "difference_interval": "A difference interval is not supported.",
        },
    }


def curve_result(levels: tuple[float, ...] = (0.0, 0.5, 1.5, 2.0)) -> dict[str, Any]:
    """A baseline, one scenario, one failed scenario and curves for two channels.

    Level 1 has no curve row: the service does not run it again.
    """
    rows = result_rows("baseline", "baseline")
    rows += result_rows(
        "half",
        "scenario",
        base="baseline",
        multipliers={"tv": 0.5, "search": 0.5},
        tv_mean=50.0,
        search_mean=25.0,
    )
    rows += result_rows("broken", "scenario", base="baseline", failed=True)
    for channel in ("tv", "search"):
        for level in levels:
            gain = 100.0 * level
            means = {"tv_mean": gain} if channel == "tv" else {"search_mean": gain / 2}
            rows += result_rows(
                f"curve:{channel}:baseline:{level:g}",
                "curve",
                base="baseline",
                channel=channel,
                level=level,
                **means,
            )
    return result_record(rows)


def weeks_status(status: str = "done", **extra: Any) -> dict[str, Any]:
    """A status record of an analysis in the weeks (backtest) form."""
    return status_record(
        status,
        dataset_id="ds_123",
        context={
            "start": "2024-07-01",
            "end": "2024-07-15",
            "query_end": "2024-08-05",
            "channels": ["tv", "search"],
            "controls": ["price"],
        },
        seed=7,
        draws=500,
        **extra,
    )


def _share(name: str, group: str, total: float) -> dict[str, Any]:
    return {
        "name": name,
        "group": group,
        "total": {"value": total, "p05": total - 0.05, "p95": total + 0.05},
        "mean": {"value": total / 2, "p05": total / 2 - 0.05, "p95": total / 2 + 0.05},
    }


def fit_record() -> dict[str, Any]:
    """The ``fit`` object of a backtest result: scores and shares per window."""
    return {
        window: {
            "r2": {"value": r2, "sd": 0.02},
            "mape": {"value": 0.1, "sd": 0.01},
            "crps": 12.5,
            "shares": [_share("tv", "channel", 0.4), _share("price", "control", -0.1)],
        }
        for window, r2 in (("context", 0.95), ("query", 0.8))
    }


def response_curves_record() -> list[dict[str, Any]]:
    """The ``response_curves`` list of a backtest result: tv at three levels, one failed."""
    points = [
        {"level": 0.0, "value": 0.0, "p05": 0.0, "p95": 0.0},
        {"level": 0.5, "value": 40.0, "p05": 35.0, "p95": 45.0},
        {"level": 1.5, "value": None, "p05": None, "p95": None},
    ]
    return [{"channel": "tv", "spend": 220.0, "points": points}]


def weeks_result() -> dict[str, Any]:
    """A backtest result: a baseline, one scenario, fit and response curves."""
    rows = result_rows("baseline", "baseline")
    rows += result_rows("half", "scenario", base="baseline", multipliers={"tv": 0.5}, tv_mean=50.0)
    return {**result_record(rows), "fit": fit_record(), "response_curves": response_curves_record()}


def export_record(inputs: dict[str, Any] | None) -> dict[str, Any]:
    """An analysis export whose baseline scenario holds ``inputs`` (or none)."""
    return {
        "format": "mfm-analysis-export-v1",
        "analysis_id": "an_1",
        "scenarios": [
            {"scenario_id": "baseline", "kind": "baseline", "inputs": inputs},
            {"scenario_id": "half", "kind": "scenario", "inputs": None},
        ],
    }
