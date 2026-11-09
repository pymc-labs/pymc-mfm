"""Local checks of weekly dates, run before a request is sent.

Rows are counted from one, in the order of the supplied data. Nothing here sorts, fills or
shifts dates: a problem is reported, and the caller fixes the input. The service keeps its own
generic messages; these messages help Python-client users only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import pandas as pd

WEEK = timedelta(days=7)


def check_weekly_dates(dates: Sequence[Any], label: str) -> None:
    """Raise ``ValueError`` at the first row that is not exactly one week after the row before.

    ``label`` names the data in the message, for example ``"history"`` or ``"plan"``. Each row
    is checked for a repeated date, a date out of order and a step that is not one week, in
    that order, so the message names the first row with any problem.
    """
    days = [_as_date(value) for value in dates]
    rows_of: dict[date, int] = {}
    for row, day in enumerate(days, start=1):
        rows_of.setdefault(day, row)
    for row, day in enumerate(days, start=1):
        if rows_of[day] < row:
            raise ValueError(
                f"Row {row} of your {label} repeats the date {day}, which is also in row "
                f"{rows_of[day]}. Each date can appear only once."
            )
        if row > 1:
            _check_step(days[row - 2], day, row, label, rows_of)


def check_plan_follows_history(plan: Any, history_end: date) -> None:
    """Raise ``ValueError`` unless the plan's weekly dates start the week after ``history_end``.

    A DataFrame plan is already converted to wire form. A plan mapping is checked only when
    it has a ``dates`` key, and then only when it has at least one date.
    """
    dates = plan.get("dates") if isinstance(plan, Mapping) else None
    if dates is None or len(dates) == 0:  # never truth-test: dates may be an array or Series
        return
    days = [_as_date(value) for value in dates]
    required = history_end + WEEK
    if days[0] != required:
        raise ValueError(
            f"Your history ends on {history_end}. Your plan starts on {days[0]}. "
            f"The next required date is {required}. "
            "Check whether a week is missing from your history or plan."
        )
    check_weekly_dates(days, "plan")


def _check_step(
    previous: date, day: date, row: int, label: str, rows_of: Mapping[date, int]
) -> None:
    gap = day - previous
    expected = previous + WEEK
    where = f"Row {row} of your {label} has the date {day}, but row {row - 1} has {previous}."
    unordered = "Dates must increase by one week for each row. The client does not sort your data."
    if day < previous:
        raise ValueError(f"{where} {unordered}")
    if gap != WEEK and expected in rows_of:
        raise ValueError(
            f"{where} The next required date is {expected}, which is in row "
            f"{rows_of[expected]}. {unordered}"
        )
    if gap % WEEK:
        raise ValueError(
            f"{where} Dates must be 7 days apart. The next required date is {expected}."
        )
    if gap != WEEK:
        raise ValueError(
            f"{where} The next required date is {expected}. "
            f"Check whether a week is missing from your {label}."
        )


def _as_date(value: Any) -> date:
    if isinstance(value, str):  # numpy.str_ too, which pandas does not parse
        value = str(value)
    return pd.Timestamp(value).date()
