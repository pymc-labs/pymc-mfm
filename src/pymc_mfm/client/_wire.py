"""Conversion between pandas objects and the API's JSON."""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray | pd.Series | pd.Index):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    raise TypeError(f"Cannot send {type(value).__name__} as JSON")


def to_json(body: Any) -> bytes:
    """Encode a request body. Rejects NaN and infinity, which JSON cannot carry."""
    return json.dumps(body, default=_json_default, allow_nan=False).encode()


def canonical_json(body: Any) -> bytes:
    """Encode a request body with sorted keys and no spaces, so equal bodies give equal bytes."""
    return json.dumps(
        body, default=_json_default, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def iso_dates(values: Iterable[Any], *, name: str = "date") -> list[str]:
    """Convert a date column to ISO ``YYYY-MM-DD`` strings."""
    dates = pd.to_datetime(pd.Series(list(values)))
    if dates.isna().any():
        raise ValueError(f"Column {name!r} has missing dates")
    return dates.dt.strftime("%Y-%m-%d").tolist()


def iso_date(value: Any, *, name: str) -> str:
    """Convert one date (a string, ``date`` or ``Timestamp``) to an ISO ``YYYY-MM-DD`` string."""
    return iso_dates([value], name=name)[0]


def finite_floats(values: Iterable[Any], *, name: str) -> list[float]:
    """Convert a numeric column to floats and refuse missing or infinite values."""
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Column {name!r} must be numeric") from error
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"Column {name!r} has missing or infinite values")
    return numbers


def as_names(names: str | Sequence[str] | None) -> list[str]:
    if names is None:
        return []
    return [names] if isinstance(names, str) else list(names)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(frame.columns)}")


def series_frame(
    frame: pd.DataFrame, *, spend: Sequence[str], controls: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return the ``spend`` and ``controls`` mappings of a request body."""
    require_columns(frame, [*spend, *controls])
    return (
        {name: finite_floats(frame[name], name=name) for name in spend},
        {name: finite_floats(frame[name], name=name) for name in controls},
    )


def dataset_body(
    frame: pd.DataFrame,
    *,
    date: str,
    outcome: str,
    spend: Sequence[str],
    controls: Sequence[str],
    name: str | None,
    outcome_name: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Build the body of ``POST /v1/datasets`` from a DataFrame.

    ``outcome_name`` and ``currency`` are labels for the site; they are optional and are
    not part of the content hash.
    """
    require_columns(frame, [date, outcome])
    spend_series, control_series = series_frame(frame, spend=spend, controls=controls)
    body: dict[str, Any] = {
        "dates": iso_dates(frame[date], name=date),
        "outcome": finite_floats(frame[outcome], name=outcome),
        "spend": spend_series,
        "controls": control_series,
    }
    if name is not None:
        body["name"] = name
    if outcome_name is not None:
        body["outcome_name"] = outcome_name
    if currency is not None:
        body["currency"] = check_currency(currency)
    return body


def check_currency(currency: str) -> str:
    """Return ``currency`` if it is three uppercase letters (ISO 4217 shape)."""
    if not (
        isinstance(currency, str)
        and len(currency) == 3
        and currency.isascii()
        and currency.isalpha()
        and currency.isupper()
    ):
        raise ValueError(f"currency must be 3 uppercase letters such as 'USD', not {currency!r}")
    return currency


def weeks_fields(
    start: Any,
    end: Any,
    query_end: Any,
    *,
    channels: str | Sequence[str] | None = None,
    controls: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """The weeks-form fields of an analysis: ``context``, ``query`` and the chosen series.

    ``start`` and ``end`` are the first and last input weeks, ``query_end`` the last predicted
    week. ``channels`` and ``controls`` left as ``None`` are not sent: the service uses all.
    """
    fields: dict[str, Any] = {
        "context": {
            "start": iso_date(start, name="context start"),
            "end": iso_date(end, name="context end"),
        },
        "query": {"end": iso_date(query_end, name="query_end")},
    }
    if channels is not None:
        fields["channels"] = as_names(channels)
    if controls is not None:
        fields["controls"] = as_names(controls)
    return fields


def plan_body(
    plan: pd.DataFrame | Mapping[str, Any],
    *,
    channels: Sequence[str],
    controls: Sequence[str],
    date_column: str | None = None,
) -> Any:
    """Build the baseline ``plan`` of an analysis.

    A mapping is already in wire form and passes through. A DataFrame needs the
    dataset's channel and control columns, and dates in ``date_column``, a ``date``
    column, the one datetime column, or a datetime index.
    """
    if not isinstance(plan, pd.DataFrame):
        return plan
    dates = _plan_dates(plan, date_column, [*channels, *controls])
    spend_series, control_series = series_frame(plan, spend=channels, controls=controls)
    return {"dates": dates, "spend": spend_series, "controls": control_series}


def _plan_dates(plan: pd.DataFrame, date_column: str | None, used: Sequence[str]) -> list[str]:
    """Find the plan's dates: the named column, a ``date`` column, the only datetime
    column that is not a series, or a datetime index."""
    for column in (date_column, "date"):
        if column is not None and column in plan.columns:
            return iso_dates(plan[column], name=column)
    datetime_columns = [
        column
        for column in plan.columns
        if column not in used and pd.api.types.is_datetime64_any_dtype(plan[column])
    ]
    if len(datetime_columns) == 1:
        return iso_dates(plan[datetime_columns[0]], name=datetime_columns[0])
    if isinstance(plan.index, pd.DatetimeIndex):
        return iso_dates(plan.index)
    raise ValueError("The plan needs a 'date' column, one datetime column or a datetime index")
