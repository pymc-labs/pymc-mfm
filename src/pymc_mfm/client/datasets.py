"""Datasets: an uploaded weekly history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from ._http import Http, new_idempotency_key
from ._wire import as_names, dataset_body


def parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp, including a trailing ``Z``."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True, eq=False)
class Dataset:
    """A stored history.

    ``history_ranges`` has one row per channel and control with the weekly 5th and 95th
    percentile, the weekly maximum, the share of zero weeks and whether the series is all
    zero. Use it to keep a plan inside what the history shows.
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
            name=name,
            date_column=date_column,
        )


class Datasets:
    """``client.datasets``: create, read and delete histories."""

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
        idempotency_key: str | None = None,
    ) -> Dataset:
        """Upload a weekly history from a DataFrame.

        ``date``, ``outcome``, ``spend`` and ``controls`` name columns of ``frame``. Dates
        are sent as ISO dates. Pass the same ``idempotency_key`` to repeat a call safely;
        by default each call gets a new key, reused on automatic retries.
        """
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

    def get(self, dataset_id: str) -> Dataset:
        """Read the summary of a stored dataset."""
        return Dataset.from_wire(self._http.request("GET", f"/v1/datasets/{dataset_id}"))

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset and all its analyses. The service removes the data within minutes."""
        self._http.request("DELETE", f"/v1/datasets/{dataset_id}")
