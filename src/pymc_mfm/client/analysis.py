"""Analyses: status, results, draws files and plots."""

from __future__ import annotations

import io
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from ._http import Http
from .datasets import parse_datetime
from .errors import Forbidden

if TYPE_CHECKING:
    import xarray

TERMINAL_STATUSES = frozenset({"done", "failed", "deleted"})
URL_RENEWAL_MARGIN = 30.0  # seconds; renew a signed URL this close to its expiry

#: Columns of the result table, in order. Fields the service adds later follow them.
TABLE_COLUMNS = (
    "scenario_id",
    "kind",
    "base_scenario_id",
    "varied_channel",
    "level",
    "steps",
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


@dataclass(frozen=True)
class AnalysisStatus:
    """Counts and state of an analysis."""

    status: str
    runs_total: int
    runs_done: int
    runs_failed: int
    estimate_seconds: float
    expires_at: datetime

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_wire(cls, record: dict[str, Any]) -> AnalysisStatus:
        return cls(
            status=record["status"],
            runs_total=record["runs_total"],
            runs_done=record["runs_done"],
            runs_failed=record["runs_failed"],
            estimate_seconds=record["estimate_seconds"],
            expires_at=parse_datetime(record["expires_at"]),
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


class Analysis:
    """One analysis: a baseline forecast, with scenarios and curves if asked.

    Get one from ``Client.analyze`` or ``Client.get``. Results follow the service's
    labels, kept in ``labels`` and in each table's ``attrs``: intervals are pointwise,
    exclude observation noise and are not causal lift, and ``simulation_se`` is numerical
    precision, not model uncertainty.
    """

    def __init__(self, http: Http, analysis_id: str, status: AnalysisStatus) -> None:
        self._http = http
        self.id = analysis_id
        self._status = status
        self._result: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Analysis(id={self.id!r}, status={self._status.status!r})"

    @property
    def expires_at(self) -> datetime:
        return self._status.expires_at

    def status(self) -> AnalysisStatus:
        """Fetch the current status and run counts."""
        self._status = AnalysisStatus.from_wire(
            self._http.request("GET", f"/v1/analyses/{self.id}")
        )
        return self._status

    def wait(self, timeout: float | None = None, poll: float = 2.0) -> Analysis:
        """Poll until the analysis is ``done``, ``failed`` or ``deleted``; return ``self``.

        Raises ``TimeoutError`` after ``timeout`` seconds. A ``failed`` analysis (its
        baseline failed) still has a result table with ``status == "failed"`` rows.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.status().finished:
            if deadline is not None and time.monotonic() + poll > deadline:
                raise TimeoutError(f"Analysis {self.id} is still {self._status.status!r}")
            time.sleep(poll)
        return self

    # Results

    def _fetch_result(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._result is None or refresh:
            self._result = self._http.request("GET", f"/v1/analyses/{self.id}/result")
        return self._result

    @property
    def table(self) -> pd.DataFrame:
        """One row per plan and component. Raises ``NotReady`` until the analysis ends."""
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

        Each plan's file is fetched when you call this. A signed URL that expired is
        replaced by fetching the result again; no run starts again. Needs the ``draws``
        extra. Draws of different plans are separately sampled marginals, not a joint
        posterior; see the ``attrs`` of each file.
        """
        xr = _require_xarray()
        urls = self._draws_urls(scenario_ids)
        parts = [self._open_draws(xr, scenario_id, urls) for scenario_id in urls]
        return xr.concat(parts, dim=pd.Index(list(urls), name="scenario"), join="outer")

    def _draws_urls(self, scenario_ids: Sequence[str] | None) -> dict[str, str]:
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
        now = pd.Timestamp.now(tz=UTC) + pd.Timedelta(seconds=URL_RENEWAL_MARGIN)
        if (available.loc[wanted, "draws_url_expires_at"] <= now).any():
            self._fetch_result(refresh=True)
            return self._draws_urls(scenario_ids)
        return {item: available.loc[item, "draws_url"] for item in wanted}

    def _open_draws(self, xr: Any, scenario_id: str, urls: dict[str, str]) -> xarray.Dataset:
        try:
            content = self._http.download(urls[scenario_id])
        except Forbidden:
            # A signed URL can expire between the result fetch and the download.
            self._fetch_result(refresh=True)
            content = self._http.download(self._draws_urls([scenario_id])[scenario_id])
        return xr.load_dataset(io.BytesIO(content), engine="h5netcdf")

    # Lifecycle

    def delete(self) -> None:
        """Cancel unstarted runs and delete the analysis and its data."""
        self._http.request("DELETE", f"/v1/analyses/{self.id}")

    # Plots

    def plot_curves(self, **kwargs: Any):
        """Response curves: per channel, the channel's contribution at each spend level.

        Draws the mean and the pointwise 5th to 95th percentile band. Returns a
        matplotlib ``Figure``. Needs the ``plot`` extra.
        """
        from .plots import plot_curves

        return plot_curves(self.table, **kwargs)

    def plot_compare(self, **kwargs: Any):
        """Mean difference of each scenario against its base, with simulation error bars.

        The bars show simulation error, not model uncertainty. Returns a matplotlib
        ``Figure``. Needs the ``plot`` extra.
        """
        from .plots import plot_compare

        return plot_compare(self.table, **kwargs)


def _require_xarray() -> Any:
    try:
        import h5netcdf  # noqa: F401
        import xarray
    except ImportError as error:
        raise ImportError(
            "Reading draws needs xarray and h5netcdf. "
            'Install them with: pip install "pymc-mfm[draws]"'
        ) from error
    return xarray
