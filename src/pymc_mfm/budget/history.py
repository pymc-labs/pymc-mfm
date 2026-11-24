"""Read explicit regularly spaced marketing histories from CSV files."""

import csv
import math
from datetime import date
from pathlib import Path


def read_history(path: str | Path, *, channels, outcome, controls=(), date_column="date"):
    """Read aligned dates, channel spend, outcome and controls without filling gaps.

    Each row is a single period in chronological order, with an ISO date. All
    periods must be equally spaced in calendar days. Values are finite numbers,
    and spend is nonnegative. The caller controls the units and channel order.
    """
    channels, controls = list(channels), list(controls)
    names = [date_column, *channels, outcome, *controls]
    if not channels or not all(isinstance(x, str) and x for x in names):
        raise ValueError("date, outcome and at least one channel name are required")
    if len(set(names)) != len(names):
        raise ValueError("date, channel, outcome and control columns must be distinct")
    spend, outcomes, control_rows, dates = [], [], [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV must have unique column headers")
        missing = set(names) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"missing CSV columns: {sorted(missing)}")
        for row in reader:
            if None in row:
                raise ValueError(f"too many fields on CSV line {reader.line_num}")
            try:
                day = date.fromisoformat(row[date_column])
                values = [float(row[name]) for name in (*channels, outcome, *controls)]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid date or number on CSV line {reader.line_num}") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite number on CSV line {reader.line_num}")
            if any(value < 0 for value in values[: len(channels)]):
                raise ValueError(f"negative historical spend on CSV line {reader.line_num}")
            dates.append(day)
            spend.append(values[: len(channels)])
            outcomes.append(values[len(channels)])
            control_rows.append(values[len(channels) + 1 :])
    if len(dates) < 2:
        raise ValueError("at least two historical periods are required")
    intervals = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
    if intervals[0] <= 0 or any(step != intervals[0] for step in intervals):
        raise ValueError("dates must be strictly increasing and equally spaced")
    return {"channels": channels, "spend": spend, "outcome": outcomes, "controls": control_rows}
