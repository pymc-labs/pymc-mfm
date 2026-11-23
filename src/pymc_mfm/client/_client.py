"""The client: datasets, analyses and the explore shortcut."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pandas as pd

from . import builders
from ._dates import check_plan_follows_history
from ._http import Http
from ._wire import as_names, plan_body, weeks_fields
from .analysis import Analysis, AnalysisStatus, Preview, start_analysis
from .datasets import Dataset, Datasets

#: Spend levels of ``explore``: 0 to 2 in steps of 0.25. Level 1 is the baseline.
EXPLORE_LEVELS = tuple(step * 0.25 for step in range(9))
#: Budget multipliers of ``explore``, the same presets as the site. Each multiplies every
#: channel; the scenario ids are ``spend_x1.2``, ``spend_x1.4`` and ``spend_x1.6``.
EXPLORE_BUDGET_MULTIPLIERS = (1.2, 1.4, 1.6)
#: Columns of ``Client.list_analyses``, in order.
LIST_COLUMNS = (
    "analysis_id",
    "dataset_id",
    "name",
    "url",
    "status",
    "runs_total",
    "runs_done",
    "runs_failed",
    "created_at",
    "expires_at",
)


class Client:
    """Client for the hosted River API.

    ``base_url`` and ``api_key`` default to the environment variables ``MFM_API_URL`` and
    ``MFM_API_KEY``. Requests retry on 429, 502, 503, 504 and connection errors, up to
    ``max_retries`` times, and wait as long as ``Retry-After`` asks, up to ``max_retry_after``
    seconds (default 60) so a huge value cannot stall the call. Each call that creates
    something sends an ``Idempotency-Key`` derived from the request content, reused on its
    retries, so a retry or a repeated call never starts the work twice.

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
        plan: pd.DataFrame | Mapping[str, Any] | None = None,
        *,
        context: Sequence[Any] | None = None,
        query_end: Any = None,
        channels: str | Sequence[str] | None = None,
        controls: str | Sequence[str] | None = None,
        scenarios: Sequence[Mapping[str, Any]] = (),
        curves: Sequence[Mapping[str, Any]] = (),
        decision_weeks: Sequence[int] | None = None,
        draws: int | None = None,
        seed: int | None = None,
        explain_history: bool | None = None,
        name: str | None = None,
        preview: bool = False,
        idempotency_key: str | None = None,
        fresh: bool = False,
    ) -> Analysis | Preview:
        """Start an analysis, or with ``preview=True`` only check what it would run.

        ``dataset`` is a ``Dataset`` or its id. Give the baseline in one of two forms:

        - A backtest of the dataset's own weeks: ``context=(start, end)`` are the first and
          last input weeks and ``query_end`` is the last predicted week; the predicted weeks
          follow the input weeks. The baseline is the dataset's own spend and controls in
          those weeks, and the result also has ``fit``, ``shares`` and ``curves``.
          ``channels`` and ``controls`` choose series of the dataset (default: all).
        - A ``plan`` for the weeks after the dataset: a DataFrame with a date column (or
          datetime index) and the dataset's channel and control columns, or a dict in the
          API's form. Its dates must be weekly, in order, and start the week after the
          dataset's last date; otherwise ``ValueError`` names the first bad row and nothing
          is sent.

        ``scenarios`` come from ``scenario`` and ``grid``, ``curves`` from ``curve``; the
        service runs none that you do not ask for. ``decision_weeks=(i, j)`` (1-based,
        inclusive) limits where scenarios and curves change spend. Omitted ``draws``,
        ``seed`` and ``explain_history`` take the service's defaults (``explain_history`` is
        on for a backtest and off for a plan). ``name`` (1 to 120 characters) labels the
        analysis on the site and in its export.

        By default the ``Idempotency-Key`` comes from the request content, so the same call
        repeated returns the same analysis instead of starting the work again, also from a new
        process. Pass ``fresh=True`` to start a new analysis (for example to retry a failed
        one), or your own ``idempotency_key``; not both.
        """
        weeks = _weeks_request(plan, context, query_end, channels, controls)
        if weeks is None:
            record = self._resolve(dataset)
            body = {"dataset_id": record.dataset_id, "plan": _plan_request(record, plan)}
        else:
            body = {"dataset_id": _dataset_id(dataset), **weeks}
        body.update({"scenarios": list(scenarios), "curves": list(curves)})
        optional = {
            "decision_weeks": decision_weeks,
            "draws": draws,
            "seed": seed,
            "explain_history": explain_history,
            "name": name,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        if preview:
            body["preview"] = True
            response = self._http.create(
                "/v1/analyses", body, idempotency_key=idempotency_key, fresh=fresh
            )
            return Preview.from_wire(response)
        return start_analysis(self._http, body, idempotency_key=idempotency_key, fresh=fresh)

    def explore(
        self,
        dataset: Dataset | str,
        plan: pd.DataFrame | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> Analysis | Preview:
        """A first look: a response curve for every channel, plus three budget scenarios.

        Each channel is varied alone at levels 0 to 2 in steps of 0.25, where 1 is the
        baseline. The scenarios ``spend_x1.2``, ``spend_x1.4`` and ``spend_x1.6`` multiply
        every channel's spend by 1.2, 1.4 and 1.6, as the site's presets do. For three
        channels the service runs 28 plans, because level 1 repeats the baseline.
        ``options`` are the other arguments of ``analyze``, such as ``context`` and
        ``query_end`` for a backtest or ``channels`` to explore some channels only, except
        ``scenarios`` and ``curves``.
        """
        clashes = {"scenarios", "curves"} & set(options)
        if clashes:
            raise TypeError(
                f"explore builds its own {sorted(clashes)}; use analyze for custom ones"
            )
        record = self._resolve(dataset)
        chosen = options.get("channels")
        channels = record.channels if chosen is None else as_names(chosen)
        curves = [builders.curve(channel, EXPLORE_LEVELS) for channel in channels]
        scenarios = [
            builders.scenario(f"spend_x{factor:g}", dict.fromkeys(channels, factor))
            for factor in EXPLORE_BUDGET_MULTIPLIERS
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
        """Your analyses, one row each, optionally only those of one dataset.

        Columns are ``LIST_COLUMNS`` in order, then any field the service adds later.
        """
        dataset_id = None if dataset is None else _dataset_id(dataset)
        params = {"dataset_id": dataset_id} if dataset_id else None
        record = self._http.request("GET", "/v1/analyses", params=params)
        frame = pd.DataFrame.from_records(record["analyses"])
        extra = [column for column in frame.columns if column not in LIST_COLUMNS]
        frame = frame.reindex(columns=[*LIST_COLUMNS, *extra])
        for column in ("created_at", "expires_at"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def _resolve(self, dataset: Dataset | str) -> Dataset:
        return dataset if isinstance(dataset, Dataset) else self.datasets.get(dataset)


def _dataset_id(dataset: Dataset | str) -> str:
    return dataset.dataset_id if isinstance(dataset, Dataset) else dataset


def _weeks_request(
    plan: Any,
    context: Sequence[Any] | None,
    query_end: Any,
    channels: str | Sequence[str] | None,
    controls: str | Sequence[str] | None,
) -> dict[str, Any] | None:
    """The weeks-form fields of a backtest, or ``None`` for an analysis of a plan."""
    if context is None and query_end is None:
        if plan is None:
            raise ValueError("Pass a plan, or context and query_end for a backtest")
        if channels is not None or controls is not None:
            raise ValueError(
                "channels and controls choose the series of a backtest; a plan uses them all"
            )
        return None
    if context is None or query_end is None:
        raise ValueError("Pass context and query_end together")
    if plan is not None:
        raise ValueError("Pass a plan, or context and query_end, not both")
    if isinstance(context, str) or len(context) != 2:
        raise ValueError("context must be a pair (start, end) of the first and last input weeks")
    start, end = context
    return weeks_fields(start, end, query_end, channels=channels, controls=controls)


def _plan_request(record: Dataset, plan: pd.DataFrame | Mapping[str, Any]) -> Any:
    """The wire ``plan``, checked to start the week after the dataset's last week."""
    wire_plan = plan_body(
        plan,
        channels=record.channels,
        controls=record.controls,
        date_column=record.date_column,
    )
    check_plan_follows_history(wire_plan, record.end)
    return wire_plan
