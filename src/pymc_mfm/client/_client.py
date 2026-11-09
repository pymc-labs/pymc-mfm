"""The client: datasets, analyses and the explore shortcut."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pandas as pd

from . import builders
from ._dates import check_plan_follows_history
from ._http import Http, new_idempotency_key
from ._wire import plan_body
from .analysis import Analysis, AnalysisStatus, Preview
from .datasets import Dataset, Datasets

#: Spend levels of ``explore``: 0 to 2 in steps of 0.25. Level 1 is the baseline.
EXPLORE_LEVELS = tuple(step * 0.25 for step in range(9))
#: Budget multiples of ``explore``. Each scales every channel.
EXPLORE_BUDGET_SCALES = (0.5, 1.5)


class Client:
    """Client for the River API v3.1.

    ``base_url`` and ``api_key`` default to the environment variables ``MFM_API_URL`` and
    ``MFM_API_KEY``. Requests retry on 429, 502, 503, 504 and connection errors, up to
    ``max_retries`` times, and wait as long as ``Retry-After`` asks, up to ``max_retry_after``
    seconds (default 60) so a huge value cannot stall the call. Each call that creates
    something sends one ``Idempotency-Key``, reused on its retries, so a retry never starts
    the work twice.

    ``transport`` replaces the network layer, for tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_retry_after: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = Http(
            base_url,
            api_key,
            timeout=timeout,
            max_retries=max_retries,
            max_retry_after=max_retry_after,
            transport=transport,
        )
        self.datasets = Datasets(self._http)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection pool."""
        self._http.close()

    def analyze(
        self,
        dataset: Dataset | str,
        plan: pd.DataFrame | Mapping[str, Any],
        *,
        scenarios: Sequence[Mapping[str, Any]] = (),
        curves: Sequence[Mapping[str, Any]] = (),
        decision_weeks: Sequence[int] | None = None,
        draws: int | None = None,
        seed: int | None = None,
        explain_history: bool = False,
        name: str | None = None,
        preview: bool = False,
        idempotency_key: str | None = None,
    ) -> Analysis | Preview:
        """Start an analysis, or with ``preview=True`` only check what it would run.

        ``dataset`` is a ``Dataset`` or its id. ``plan`` is the baseline: a DataFrame
        with a date column (or datetime index) and the dataset's channel and control
        columns, or a dict in the API's form. ``scenarios`` come from ``scenario`` and
        ``grid``, ``curves`` from ``curve``. ``decision_weeks=(i, j)`` (1-based, inclusive)
        limits where steps and curves may change spend. Omitted ``draws`` and ``seed`` take
        the service's defaults. ``name`` (1 to 120 characters) labels the analysis on the site
        and in its export. Pass the same ``idempotency_key`` to repeat a call safely.

        The plan's dates must be weekly, in order, and start the week after the dataset's
        last date; otherwise ``ValueError`` names the first bad row and nothing is sent.
        """
        record = self._resolve(dataset)
        wire_plan = plan_body(
            plan,
            channels=record.channels,
            controls=record.controls,
            date_column=record.date_column,
        )
        check_plan_follows_history(wire_plan, record.end)
        body: dict[str, Any] = {
            "dataset_id": record.dataset_id,
            "plan": wire_plan,
            "scenarios": list(scenarios),
            "curves": list(curves),
            "explain_history": explain_history,
        }
        optional = {"decision_weeks": decision_weeks, "draws": draws, "seed": seed}
        body.update({key: value for key, value in optional.items() if value is not None})
        if name is not None:
            body["name"] = name
        if preview:
            body["preview"] = True
        response = self._http.request(
            "POST",
            "/v1/analyses",
            body=body,
            idempotency_key=idempotency_key or new_idempotency_key(),
        )
        if preview:
            return Preview.from_wire(response)
        status = AnalysisStatus.from_wire(
            {
                "status": "queued",
                "runs_done": 0,
                "runs_failed": 0,
                **{key: response[key] for key in ("runs_total", "estimate_seconds", "expires_at")},
            }
        )
        return Analysis(
            self._http,
            response["analysis_id"],
            status,
            name=response.get("name", name),
            url=response.get("url"),
        )

    def explore(
        self,
        dataset: Dataset | str,
        plan: pd.DataFrame | Mapping[str, Any],
        **options: Any,
    ) -> Analysis | Preview:
        """A first look at a plan: a response curve for every channel, plus budget scenarios.

        Each channel is varied alone at levels 0 to 2 in steps of 0.25, where 1 is the
        baseline. Two scenarios scale the total budget by 0.5 and 1.5. For three channels the
        service runs 27 plans, because level 1 repeats the baseline. ``options`` are the other
        arguments of ``analyze``, except ``scenarios`` and ``curves``.
        """
        clashes = {"scenarios", "curves"} & set(options)
        if clashes:
            raise TypeError(
                f"explore builds its own {sorted(clashes)}; use analyze for custom ones"
            )
        record = self._resolve(dataset)
        curves = [builders.curve(channel, EXPLORE_LEVELS) for channel in record.channels]
        scenarios = [
            builders.scenario(f"budget_x{factor:g}", builders.scale(factor))
            for factor in EXPLORE_BUDGET_SCALES
        ]
        return self.analyze(record, plan, scenarios=scenarios, curves=curves, **options)

    def get(self, analysis_id: str) -> Analysis:
        """Resume an analysis by id. Raises ``NotFound`` or ``Gone`` for a bad or expired id."""
        record = self._http.request("GET", f"/v1/analyses/{analysis_id}")
        return Analysis(
            self._http,
            analysis_id,
            AnalysisStatus.from_wire(record),
            name=record.get("name"),
            url=record.get("url"),
        )

    def list_analyses(self, dataset: Dataset | str | None = None) -> pd.DataFrame:
        """Your analyses, one row each, optionally only those of one dataset."""
        dataset_id = dataset.dataset_id if isinstance(dataset, Dataset) else dataset
        params = {"dataset_id": dataset_id} if dataset_id else None
        record = self._http.request("GET", "/v1/analyses", params=params)
        columns = ["analysis_id", "status", "runs_total", "runs_done", "runs_failed", "expires_at"]
        frame = pd.DataFrame.from_records(record["analyses"], columns=columns)
        frame["expires_at"] = pd.to_datetime(frame["expires_at"], utc=True)
        return frame

    def _resolve(self, dataset: Dataset | str) -> Dataset:
        return dataset if isinstance(dataset, Dataset) else self.datasets.get(dataset)
