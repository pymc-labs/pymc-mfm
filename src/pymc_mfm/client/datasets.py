"""Datasets: an uploaded weekly history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

import pandas as pd

from ._dates import check_weekly_dates
from ._hash import frame_content_hash
from ._http import Http
from ._wire import as_names, check_currency, dataset_body

MAX_OUTCOME_NAME = 60  # characters the service accepts in ``outcome_name``


def parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp, including a trailing ``Z``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_datetime(value: str | None) -> datetime | None:
    return None if value is None else parse_datetime(value)


@dataclass(frozen=True, eq=False)
class Dataset:
    """A stored history.

    ``history_ranges`` has one row per channel and control with the weekly 5th and 95th
    percentile, the weekly maximum, the share of zero weeks and whether the series is all
    zero. Use it to keep a plan inside what the history shows. The 5th and 95th percentiles
    are warning thresholds, not the edge of the observed data; ``max`` is the largest week.

    ``expires_at`` is the current deadline. The service moves it later each time you use the
    dataset, up to a fixed maximum after upload, so read it again with ``datasets.get``
    instead of keeping it. ``content_hash`` identifies the values (see
    ``dataset_content_hash``). ``content_hash``, ``created_at``, ``last_used_at`` and
    ``analyses`` (the number of live analyses), ``outcome_name`` and ``currency`` (labels from
    the summary) are ``None`` when an older service does not
    send them.
    """

    dataset_id: str
    expires_at: datetime
    weeks: int
    start: date
    end: date
    channels: tuple[str, ...]
    controls: tuple[str, ...]
    history_ranges: pd.DataFrame
    warnings: tuple[str, ...] = ()
    name: str | None = None
    date_column: str | None = None
    content_hash: str | None = None
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    analyses: int | None = None
    outcome_name: str | None = None
    currency: str | None = None

    @classmethod
    def from_wire(
        cls, record: dict[str, Any], *, name: str | None = None, date_column: str | None = None
    ) -> Dataset:
        summary = record["summary"]
        ranges = pd.DataFrame.from_dict(record.get("history_ranges", {}), orient="index")
        return cls(
            dataset_id=record["dataset_id"],
            expires_at=parse_datetime(record["expires_at"]),
            weeks=summary["weeks"],
            start=date.fromisoformat(summary["start"]),
            end=date.fromisoformat(summary["end"]),
            channels=tuple(summary["channels"]),
            controls=tuple(summary["controls"]),
            history_ranges=ranges,
            warnings=tuple(record.get("warnings", ())),
            name=name if name is not None else record.get("name"),
            date_column=date_column,
            content_hash=record.get("content_hash"),
            created_at=_optional_datetime(record.get("created_at")),
            last_used_at=_optional_datetime(record.get("last_used_at")),
            analyses=record.get("analyses"),
            outcome_name=summary.get("outcome"),
            currency=summary.get("currency"),
        )


class Datasets:
    """``client.datasets``: create, find, list, read and delete histories."""

    def __init__(self, http: Http) -> None:
        self._http = http

    def create(
        self,
        frame: pd.DataFrame,
        *,
        date: str,
        outcome: str,
        spend: Sequence[str] | str,
        controls: Sequence[str] | str = (),
        name: str | None = None,
        currency: str | None = None,
        reuse: bool = False,
        idempotency_key: str | None = None,
        fresh: bool = False,
    ) -> Dataset:
        """Upload a weekly history from a DataFrame.

        ``date``, ``outcome``, ``spend`` and ``controls`` name columns of ``frame``. Dates
        are sent as ISO dates. By default the ``Idempotency-Key`` comes from the request
        content, so the same upload repeated returns the stored dataset and stores no copy;
        after that dataset is deleted or expired, the call raises ``Gone``. Pass ``fresh=True``
        to store a new copy, or your own ``idempotency_key`` to choose the key; not both.

        The site labels results with ``outcome_name``, which is the ``outcome`` column name
        (sent only if it has 1 to 60 characters), and with ``currency``, an optional ISO 4217
        code of three uppercase letters. Neither is part of the content hash.

        With ``reuse=True`` the call first lists your datasets and returns the one whose
        ``content_hash`` equals the hash of this history, and uploads nothing. It never
        matches by name or by number of weeks. If no dataset matches, it uploads as usual.

        Before an upload, the history must have one row for each week, in date order, with
        no repeated date; otherwise ``ValueError`` names the first bad row and nothing is
        sent. A reused dataset is not checked, because the hash ignores row order. The
        client never sorts or fills your data.
        """
        if currency is not None:
            check_currency(currency)  # refuse a bad code before any request, even when reusing
        if reuse:
            found = self.find(frame, date=date, outcome=outcome, spend=spend, controls=controls)
            if found is not None:
                return found
        body = dataset_body(
            frame,
            date=date,
            outcome=outcome,
            spend=as_names(spend),
            controls=as_names(controls),
            name=name,
            outcome_name=outcome if 1 <= len(outcome) <= MAX_OUTCOME_NAME else None,
            currency=currency,
        )
        check_weekly_dates(body["dates"], "history")  # only here: reuse above ignores row order
        record = self._http.create(
            "/v1/datasets", body, idempotency_key=idempotency_key, fresh=fresh
        )
        return Dataset.from_wire(record, name=name, date_column=date)

    def find(
        self,
        frame: pd.DataFrame,
        *,
        date: str,
        outcome: str,
        spend: Sequence[str] | str,
        controls: Sequence[str] | str = (),
    ) -> Dataset | None:
        """Return your stored dataset with exactly this history, or ``None``.

        The match is the content hash only: the same dates, outcome, spend and controls.
        A dataset with the same name or the same number of weeks but one different value
        does not match. The returned dataset remembers ``date`` as its date column.
        """
        wanted = frame_content_hash(
            frame, date=date, outcome=outcome, spend=spend, controls=controls
        )
        for dataset in self.list():
            if dataset.content_hash == wanted:
                return replace(dataset, date_column=date)
        return None

    def list(self) -> list[Dataset]:
        """Your stored datasets, newest first as the service orders them."""
        record = self._http.request("GET", "/v1/datasets")
        return [Dataset.from_wire(item) for item in record["datasets"]]

    def get(self, dataset_id: str) -> Dataset:
        """Read the summary of a stored dataset."""
        return Dataset.from_wire(self._http.request("GET", f"/v1/datasets/{dataset_id}"))

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset and all its analyses. The service removes the data within minutes."""
        self._http.request("DELETE", f"/v1/datasets/{dataset_id}")
