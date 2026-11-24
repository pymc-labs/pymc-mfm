"""Analyses: status, results, draws files and plots."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ._http import Http
from ._wire import weeks_fields
from .datasets import parse_datetime
from .errors import Forbidden, Gone

if TYPE_CHECKING:
    import xarray

#: Statuses that end polling, once every run has ended (see ``AnalysisStatus.finished``).
#: A deleted analysis answers 410 (``Gone``) or the status ``deleted``; ``wait`` raises ``Gone``.
TERMINAL_STATUSES = frozenset({"done", "failed"})
URL_RENEWAL_MARGIN = 30.0  # seconds; renew a signed URL this close to its expiry

#: Columns of the result table, in order. Fields the service adds later follow them.
TABLE_COLUMNS = (
    "scenario_id",
    "kind",
    "base_scenario_id",
    "varied_channel",
    "level",
    "multipliers",
    "run_id",
    "status",
    "spend_total",
    "component",
    "mean",
    "p05",
    "p95",
    "window_mean",
    "mean_difference",
    "simulation_se",
    "sensitivity",
    "warnings",
    "draws_url",
    "draws_url_expires_at",
)
#: Scores of ``Analysis.fit``, in order. ``crps`` has no ``sd``.
FIT_METRICS = ("r2", "mape", "crps")


@dataclass(frozen=True)
class AnalysisStatus:
    """Counts and state of an analysis.

    ``runs_cancelled`` and ``warning_details`` (structured warnings, next to the text
    warnings) are ``0`` and empty when an older service does not send them.

    ``finished`` is true when the result is ready: the status is ``done``, or it is
    ``failed`` and every run has ended (done, failed or cancelled). The service sets
    ``failed`` as soon as the baseline fails, while other runs can still be active.
    An older service that does not send ``runs_cancelled`` can leave a ``failed``
    analysis with cancelled runs never ``finished``; ``Analysis.wait`` then ends at its
    ``timeout``.
    """

    status: str
    runs_total: int
    runs_done: int
    runs_failed: int
    estimate_seconds: float
    expires_at: datetime
    runs_cancelled: int = 0
    warning_details: tuple[dict[str, Any], ...] = ()

    @property
    def finished(self) -> bool:
        if self.status == "done":
            return True
        ended = self.runs_done + self.runs_failed + self.runs_cancelled
        return self.status in TERMINAL_STATUSES and ended >= self.runs_total

    @classmethod
    def from_wire(cls, record: dict[str, Any]) -> AnalysisStatus:
        return cls(
            status=record["status"],
            runs_total=record["runs_total"],
            runs_done=record["runs_done"],
            runs_failed=record["runs_failed"],
            estimate_seconds=record["estimate_seconds"],
            expires_at=parse_datetime(record["expires_at"]),
            runs_cancelled=record.get("runs_cancelled", 0),
            warning_details=tuple(record.get("warning_details", ())),
        )


@dataclass(frozen=True, eq=False)
class Preview:
    """What an analysis would run. Nothing is stored or run."""

    plans: list[dict[str, Any]]
    runs_total: int
    estimate_seconds: float
    assumptions: dict[str, Any]
    warnings: list[str]

    @classmethod
    def from_wire(cls, record: dict[str, Any]) -> Preview:
        return cls(
            plans=record["plans"],
            runs_total=record["runs_total"],
            estimate_seconds=record["estimate_seconds"],
            assumptions=record.get("assumptions", {}),
            warnings=record.get("warnings", []),
        )

    def spend(self) -> pd.DataFrame:
        """Weekly spend of every expanded plan, indexed by plan and week number."""
        frames = {
            plan["scenario_id"]: pd.DataFrame(plan["spend"]).rename_axis("week")
            for plan in self.plans
        }
        return pd.concat(frames, names=["scenario_id"])


def result_table(rows: Iterable[dict[str, Any]], labels: dict[str, Any]) -> pd.DataFrame:
    """Turn result rows into a long table. A failed plan has missing ``mean``, ``p05``, ``p95``."""
    table = pd.DataFrame.from_records(list(rows))
    extra = [column for column in table.columns if column not in TABLE_COLUMNS]
    table = table.reindex(columns=[*TABLE_COLUMNS, *extra])
    table["draws_url_expires_at"] = pd.to_datetime(table["draws_url_expires_at"], utc=True)
    for column in ("level", "spend_total", "mean", "p05", "p95", "window_mean"):
        table[column] = pd.to_numeric(table[column])
    table.attrs["labels"] = labels
    return table


def start_analysis(
    http: Http, body: dict[str, Any], *, idempotency_key: str | None = None, fresh: bool = False
) -> Analysis:
    """Send one ``POST /v1/analyses`` body and return the analysis it started."""
    response = http.create("/v1/analyses", body, idempotency_key=idempotency_key, fresh=fresh)
    status = AnalysisStatus.from_wire(
        {
            "status": "queued",
            "runs_done": 0,
            "runs_failed": 0,
            **{key: response[key] for key in ("runs_total", "estimate_seconds", "expires_at")},
        }
    )
    return Analysis(
        http,
        response["analysis_id"],
        status,
        name=response.get("name", body.get("name")),
        url=response.get("url"),
    )


def _plan_fields(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The ``plan`` and ``decision_weeks`` of a request, from a scenario's exported inputs."""
    plan = {
        "dates": list(inputs["dates"]),
        "spend": inputs["spend"],
        "controls": inputs["controls"],
    }
    return {"plan": plan, **_decision_fields(inputs)}


def _decision_fields(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """``decision_weeks`` (1-based, inclusive) from a scenario's exported inputs, only when the
    window is narrower than the forecast weeks; empty otherwise, as the service's default."""
    dates = list(inputs["dates"])
    decision = list(inputs.get("decision_dates") or ())
    if not decision or decision == dates:
        return {}
    first = dates.index(decision[0]) + 1
    return {"decision_weeks": [first, first + len(decision) - 1]}


def _labelled(rows: list[dict[str, Any]], columns: Sequence[str], labels: Any) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=list(columns))
    for column in ("value", "sd", "p05", "p95", "spend", "level"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column])
    frame.attrs["labels"] = labels
    return frame


class Analysis:
    """One analysis: a baseline forecast, with scenarios and curves if asked.

    Get one from ``Client.analyze``, ``Client.get`` or ``again``. ``name`` is the optional
    name given to ``analyze``. ``url`` is the address of the result page on the service's site; give
    it to the person, who opens it to read the analysis. It is ``None`` when the service
    has no site address set or is an older version. Results follow the service's
    labels, kept in ``labels`` and in each table's ``attrs``: intervals are pointwise,
    exclude observation noise and are not causal lift, and ``simulation_se`` is numerical
    precision, not model uncertainty.

    ``table`` has one row per plan and component. Its columns, in order: ``scenario_id``
    (``baseline``, a scenario id or a curve plan), ``kind``, ``base_scenario_id`` (the plan the
    difference is measured from), ``varied_channel`` and ``level`` (of a curve plan),
    ``multipliers``, ``run_id``, ``status`` (``failed``: the plan has no numbers),
    ``spend_total``, ``component`` (``outcome``, ``baseline``, a channel, a control,
    ``latent`` or ``indirect:*``), ``mean``, ``p05`` and ``p95`` (totals over the forecast),
    ``window_mean`` (the total over the decision window), ``mean_difference`` (paired, from
    the base plan), ``simulation_se`` (the simulation error of ``mean_difference``),
    ``sensitivity``, ``warnings``, ``draws_url`` and ``draws_url_expires_at`` (a signed link
    to the plan's draws file and its expiry). Columns that a newer service adds come after
    these.
    """

    def __init__(
        self,
        http: Http,
        analysis_id: str,
        status: AnalysisStatus,
        *,
        name: str | None = None,
        url: str | None = None,
    ) -> None:
        self._http = http
        self.id = analysis_id
        self.name = name
        self.url = url
        self._status = status
        self._result: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Analysis(id={self.id!r}, status={self._status.status!r})"

    @property
    def expires_at(self) -> datetime:
        """The expiry of the analysis: the expiry of its dataset.

        The hosted River API keeps a dataset 7 days after upload and does not extend it.
        """
        return self._status.expires_at

    def status(self) -> AnalysisStatus:
        """Fetch the current status and run counts. Raises ``Gone`` once the analysis is
        deleted or expired."""
        self._read_status()
        return self._status

    def _read_status(self) -> dict[str, Any]:
        record = self._http.request("GET", f"/v1/analyses/{self.id}")
        self._status = AnalysisStatus.from_wire(record)
        self.name = record.get("name", self.name)
        self.url = record.get("url", self.url)
        return record

    @property
    def resume_call(self) -> str:
        """The call that picks this analysis up again, for example after a timeout."""
        return f"client.get({self.id!r}).wait()"

    def wait(
        self, timeout: float | None = None, poll: float = 2.0, *, announce: bool = True
    ) -> Analysis:
        """Poll until the analysis is ``done``, or ``failed`` with every run ended; return ``self``.

        Unless the analysis is already known to be finished, it first prints the analysis id and
        the call that resumes it to ``sys.stderr`` (``announce=False`` stays silent), before
        the first request, so the id survives a killed, timed-out or unreachable service.
        Raises ``TimeoutError`` after ``timeout`` seconds; the analysis keeps running, and the
        error text gives the call that resumes it. Raises ``Gone`` if the analysis is deleted or
        expires meanwhile. A ``failed`` analysis (its baseline failed) still has a result table with
        ``status == "failed"`` rows. An older service that does not send ``runs_cancelled`` can
        keep a ``failed`` analysis with cancelled runs from looking finished: pass ``timeout``.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        if announce and not self._status.finished:
            print(f"analysis id: {self.id}  (resume: {self.resume_call})", file=sys.stderr)
        while not self.status().finished:
            if self._status.status == "deleted":
                raise Gone(410, "deleted", f"Analysis {self.id} was deleted")
            if deadline is not None and time.monotonic() + poll > deadline:
                state = self._status
                ended = state.runs_done + state.runs_failed + state.runs_cancelled
                raise TimeoutError(
                    f"Analysis {self.id} is still {state.status!r} after waiting, with {ended} "
                    f"of {state.runs_total} runs ended. It keeps running. Do not start it again: "
                    f"resume with {self.resume_call}"
                )
            time.sleep(poll)
        return self

    def again(
        self,
        scenarios: Sequence[Mapping[str, Any]],
        *,
        curves: Sequence[Mapping[str, Any]] | None = None,
        explain_history: bool = False,
        idempotency_key: str | None = None,
        fresh: bool = False,
    ) -> Analysis:
        """Start a new analysis with this one's inputs and new ``scenarios``; return it.

        The new analysis repeats the dataset, the weeks, channels and controls of a backtest
        (or the baseline plan of a plan), the ``decision_weeks``, the seed and the draws.
        It runs no curves unless you pass ``curves``, and no history pass unless you set
        ``explain_history``, because this analysis already has them. It costs one more
        baseline run plus its scenarios.

        Repeating the same call returns the same analysis. Pass ``fresh=True`` (or a new
        ``idempotency_key``) to run it once more, for example after that analysis was deleted
        (``Gone``) or failed.

        Compare each scenario only with the baseline of its own analysis, not across
        analyses: equal draws across analyses are not guaranteed.

        The baseline and its decision window are read from the export, so every run must
        have ended: this raises ``NotReady`` until then. Call ``wait()`` first.
        """
        record = self._read_status()
        if record.get("dataset_id") is None:
            raise ValueError(f"Analysis {self.id} has no stored dataset to analyze again")
        inputs = self._baseline_inputs()
        context = record.get("context")
        if context is None:
            same = _plan_fields(inputs)
        else:
            same = {**_weeks_of(context), **_decision_fields(inputs)}
        body: dict[str, Any] = {
            "dataset_id": record["dataset_id"],
            **same,
            "scenarios": list(scenarios),
            "explain_history": explain_history,
        }
        if curves is not None:
            body["curves"] = list(curves)
        body.update({key: record[key] for key in ("seed", "draws") if record.get(key) is not None})
        return start_analysis(self._http, body, idempotency_key=idempotency_key, fresh=fresh)

    def _baseline_inputs(self) -> Mapping[str, Any]:
        """The exported inputs of the baseline. Raises ``NotReady`` until every run ends."""
        export = json.loads(self._http.download_authenticated(f"/v1/analyses/{self.id}/export"))
        baseline = next((item for item in export["scenarios"] if item["kind"] == "baseline"), None)
        if baseline is None or baseline.get("inputs") is None:
            raise ValueError(f"Analysis {self.id} has no stored baseline plan to repeat")
        return baseline["inputs"]

    # Results

    def _fetch_result(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._result is None or refresh:
            self._result = self._http.request("GET", f"/v1/analyses/{self.id}/result")
        return self._result

    @property
    def table(self) -> pd.DataFrame:
        """One row per plan and component. Raises ``NotReady`` until the analysis ends.

        The ``Analysis`` class description lists its columns.
        """
        result = self._fetch_result()
        return result_table(result["table"], result["labels"])

    @property
    def history(self) -> pd.DataFrame | None:
        """The baseline's history-window totals, if ``explain_history`` was set."""
        result = self._fetch_result()
        if result.get("history") is None:
            return None
        return result_table(result["history"], result["labels"])

    @property
    def fit(self) -> pd.DataFrame | None:
        """How well the baseline follows the actual outcome, for a backtest.

        One row per ``window`` (``context``: the input weeks; ``query``: the predicted weeks)
        and ``metric`` (``r2``, ``mape`` as a fraction, ``crps``), with ``value`` (the score
        of the mean line) and ``sd`` (its spread over draws; none for ``crps``). Returns
        ``None`` when the service has no fit, for example for an analysis of a plan.
        """
        result = self._fetch_result()
        fit = result.get("fit")
        if fit is None:
            return None
        rows = []
        for window, scores in fit.items():
            for metric in FIT_METRICS:
                score = (scores or {}).get(metric)
                if isinstance(score, Mapping):
                    value, sd = score.get("value"), score.get("sd")
                else:  # crps is one number; r2 is null when the outcome never changes
                    value, sd = score, None
                rows.append({"window": window, "metric": metric, "value": value, "sd": sd})
        return _labelled(rows, ["window", "metric", "value", "sd"], result["labels"])

    @property
    def shares(self) -> pd.DataFrame | None:
        """Each channel's and control's share of the expected outcome, for a backtest.

        One row per ``window``, series (``name``, ``group``: ``channel`` or ``control``) and
        ``share``: ``total`` is the series' total over the window divided by the total
        expected outcome, ``mean`` is the average of its weekly shares. ``value`` is the mean
        over draws, ``p05`` and ``p95`` the 5th and 95th percentiles. Returns ``None`` when
        the service has no fit, for example for an analysis of a plan.
        """
        result = self._fetch_result()
        fit = result.get("fit")
        if fit is None:
            return None
        rows = [
            {
                "window": window,
                "name": item["name"],
                "group": item["group"],
                "share": share,
                **{key: (item.get(share) or {}).get(key) for key in ("value", "p05", "p95")},
            }
            for window, scores in fit.items()
            for item in (scores or {}).get("shares") or ()
            for share in ("total", "mean")
        ]
        columns = ["window", "name", "group", "share", "value", "p05", "p95"]
        return _labelled(rows, columns, result["labels"])

    @property
    def curves(self) -> pd.DataFrame | None:
        """Response curves of a backtest: one row per channel and spend level.

        ``spend`` is the channel's baseline spend in the query weeks and ``level`` a
        multiple of it; ``value``, ``p05`` and ``p95`` are the mean, 5th and 95th percentile
        of the channel's own contribution over those weeks. They are missing where a level
        was not asked or its run failed. ``None`` for an analysis of a plan: read its curve
        rows in ``table`` or draw them with ``plot_curves()``.
        """
        result = self._fetch_result()
        curves = result.get("response_curves")
        if curves is None:
            return None
        rows = [
            {
                "channel": item["channel"],
                "spend": item.get("spend"),
                **{key: point.get(key) for key in ("level", "value", "p05", "p95")},
            }
            for item in curves
            for point in item["points"]
        ]
        columns = ["channel", "spend", "level", "value", "p05", "p95"]
        return _labelled(rows, columns, result["labels"])

    @property
    def assumptions(self) -> dict[str, Any]:
        """Control values used by each plan, beside their history ranges."""
        return self._fetch_result()["assumptions"]

    @property
    def meta(self) -> dict[str, Any]:
        """Seed, draws, mode and run hosts."""
        return self._fetch_result()["meta"]

    @property
    def labels(self) -> dict[str, str]:
        """How to read the numbers, as the service states it."""
        return self._fetch_result()["labels"]

    def weekly(self, scenario_id: str = "baseline") -> pd.DataFrame:
        """Weekly mean and 5th to 95th percentile of every component for one plan."""
        record = self._http.request(
            "GET", f"/v1/analyses/{self.id}/weekly", params={"scenario_id": scenario_id}
        )
        frame = pd.DataFrame.from_records(
            record["rows"], columns=["date", "component", "mean", "p05", "p95"]
        )
        frame["date"] = pd.to_datetime(frame["date"])
        frame.insert(0, "scenario_id", record["scenario_id"])
        frame.attrs["labels"] = self._fetch_result()["labels"]
        return frame

    # Draws

    def draws(self, scenario_ids: Sequence[str] | None = None) -> xarray.Dataset:
        """Per-draw results as one ``xarray.Dataset``, stacked along ``scenario``.

        Each plan's file is fetched when you call this. If a signed URL expires within 30
        seconds (``URL_RENEWAL_MARGIN``) or the file host returns HTTP 403, the client
        refreshes all links at most once per call; no run starts again. A renewed URL
        never outlives the analysis, so it is used until it expires, and this raises ``Gone``
        when it already has. Draws of different plans are separately sampled marginals, not
        a joint posterior; see the ``attrs`` of each file.
        """
        import xarray as xr

        urls, renewed = self._draws_urls(scenario_ids)
        parts = []
        for scenario_id in list(urls):
            content = self._download_draws(urls[scenario_id], refused_ok=not renewed)
            if content is None:
                # Renew every link once, outside the handler, so a recovered 403 cannot
                # appear in a later traceback; later files then use the renewed links.
                urls, renewed = self._draws_urls(list(urls), renew=True)
                content = self._http.download(urls[scenario_id])
            parts.append(xr.load_dataset(io.BytesIO(content), engine="h5netcdf"))
        return xr.concat(parts, dim=pd.Index(list(urls), name="scenario"), join="outer")

    def _draws_urls(
        self, scenario_ids: Sequence[str] | None, *, renew: bool = False
    ) -> tuple[dict[str, str], bool]:
        """The signed draws URL of each wanted plan, and whether the result was fetched again.

        ``renew`` fetches the result again first. Otherwise it is fetched again only when a
        URL expires within ``URL_RENEWAL_MARGIN``.
        """
        links = self._draws_links(scenario_ids, refresh=renew)
        soon = pd.Timestamp.now(tz=UTC) + pd.Timedelta(seconds=URL_RENEWAL_MARGIN)
        if not renew and (links["draws_url_expires_at"] <= soon).any():
            links = self._draws_links(scenario_ids, refresh=True)
            renew = True
        # The service caps a renewed URL at the analysis expiry: use it while it is valid.
        if renew and (links["draws_url_expires_at"] <= pd.Timestamp.now(tz=UTC)).any():
            raise Gone(
                410,
                "expired",
                f"The draws files of analysis {self.id} expired: their signed URLs cannot be "
                "renewed past the expiry of the analysis.",
            )
        return links["draws_url"].to_dict(), renew

    def _draws_links(self, scenario_ids: Sequence[str] | None, *, refresh: bool) -> pd.DataFrame:
        """``draws_url`` and ``draws_url_expires_at`` of each wanted plan, by ``scenario_id``."""
        if refresh:
            self._fetch_result(refresh=True)
        table = self.table.drop_duplicates("scenario_id")
        available = table.dropna(subset=["draws_url"]).set_index("scenario_id")
        if scenario_ids is None:
            wanted = list(available.index)
        else:
            wanted = list(scenario_ids)
            missing = [item for item in wanted if item not in available.index]
            if missing:
                raise ValueError(f"No draws file for scenarios {missing}: unknown or failed")
        if not wanted:
            raise ValueError("The analysis has no draws files")
        return available.loc[wanted, ["draws_url", "draws_url_expires_at"]]

    def _download_draws(self, url: str, *, refused_ok: bool) -> bytes | None:
        """The file at ``url``; ``None`` on HTTP 403 if a renewal is still allowed."""
        if not refused_ok:
            return self._http.download(url)
        try:
            return self._http.download(url)
        except Forbidden:
            return None

    def export(self, path: str | os.PathLike[str]) -> Path:
        """Download the analysis as one JSON file and return its path.

        The file holds the request, the scenarios with their inputs and warnings, the
        result table and the run settings. Raises ``NotReady`` until every run ends and
        ``Gone`` once the analysis expired or was deleted.
        """
        content = self._http.download_authenticated(f"/v1/analyses/{self.id}/export")
        target = Path(path)
        target.write_bytes(content)
        return target

    # Lifecycle

    def delete(self) -> None:
        """Cancel unstarted runs and delete the analysis and its data."""
        self._http.request("DELETE", f"/v1/analyses/{self.id}")

    # Plots

    def plot_curves(self, **kwargs: Any):
        """Response curves: per channel, the channel's contribution at each spend level.

        Draws the mean and the pointwise 5th to 95th percentile band. Returns a
        matplotlib ``Figure``.
        """
        from .plots import plot_curves

        return plot_curves(self.table, **kwargs)

    def plot_compare(self, **kwargs: Any):
        """Mean difference of each scenario against its base, with simulation error bars.

        The bars show simulation error, not model uncertainty. Returns a matplotlib
        ``Figure``.
        """
        from .plots import plot_compare

        return plot_compare(self.table, **kwargs)


def _weeks_of(context: Mapping[str, Any]) -> dict[str, Any]:
    """The weeks and series of a request, from an analysis's ``context``."""
    return weeks_fields(
        context["start"],
        context["end"],
        context["query_end"],
        channels=context["channels"],
        controls=context["controls"],
    )
