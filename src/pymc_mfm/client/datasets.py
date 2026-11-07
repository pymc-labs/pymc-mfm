"""Datasets: an uploaded weekly history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any

import pandas as pd

from ._hash import frame_content_hash
from ._http import Http, new_idempotency_key
from ._wire import as_names, dataset_body


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
    ``analyses`` (the number of live analyses) are ``None`` when an older service does not
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
        reuse: bool = False,
        idempotency_key: str | None = None,
    ) -> Dataset:
        """Upload a weekly history from a DataFrame.

        ``date``, ``outcome``, ``spend`` and ``controls`` name columns of ``frame``. Dates
        are sent as ISO dates. Pass the same ``idempotency_key`` to repeat a call safely;
        by default each call gets a new key, reused on automatic retries.

        With ``reuse=True`` the call first lists your datasets and returns the one whose
        ``content_hash`` equals the hash of this history, and uploads nothing. It never
        matches by name or by number of weeks. If no dataset matches, it uploads as usual.
        """
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
        )
        record = self._http.request(
            "POST",
            "/v1/datasets",
            body=body,
            idempotency_key=idempotency_key or new_idempotency_key(),
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
