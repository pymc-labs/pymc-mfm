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

    ``label`` names the data in the message, for example ``"history"`` or ``"plan"``. Repeated
    and out-of-order dates are reported before gaps, because a gap in unordered data is unclear.
    """
    days = [_as_date(value) for value in dates]
    _check_order(days, label)
    for row in range(2, len(days) + 1):
        _check_step(days[row - 2], days[row - 1], row, label)


def _check_order(days: Sequence[date], label: str) -> None:
    first_row: dict[date, int] = {}
    for row, day in enumerate(days, start=1):
        if day in first_row:
            raise ValueError(
                f"Row {row} of your {label} repeats the date {day}, which is also in row "
                f"{first_row[day]}. Each date can appear only once."
            )
        first_row[day] = row
        if row > 1 and day < days[row - 2]:
            raise ValueError(
                f"Row {row} of your {label} has the date {day}, but row {row - 1} has "
                f"{days[row - 2]}. Dates must increase by one week for each row. "
                "The client does not sort your data."
            )


def check_plan_follows_history(plan: Any, history_end: date) -> None:
    """Raise ``ValueError`` unless the plan's weekly dates start the week after ``history_end``.

    A DataFrame plan is already converted to wire form. A plan mapping is checked only when
    it has a ``dates`` key, and then only when it has at least one date.
    """
    if not isinstance(plan, Mapping) or not plan.get("dates"):
        return
    days = [_as_date(value) for value in plan["dates"]]
    required = history_end + WEEK
    if days[0] != required:
        raise ValueError(
            f"Your history ends on {history_end}. Your plan starts on {days[0]}. "
            f"The next required date is {required}. "
            "Check whether a week is missing from your history or plan."
        )
    check_weekly_dates(days, "plan")


def _check_step(previous: date, day: date, row: int, label: str) -> None:
    gap = day - previous
    expected = previous + WEEK
    where = f"Row {row} of your {label} has the date {day}, but row {row - 1} has {previous}."
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
    return pd.Timestamp(value).date()
