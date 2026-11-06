"""Builders for the steps, scenarios and curves of an analysis request.

Each builder returns the plain dict that goes on the wire, so results can be
printed, edited and sent as they are. The service checks every value; builders
only refuse requests that cannot be sent at all.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from typing import Any

Step = dict[str, Any]
Scenario = dict[str, Any]

_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]")
MAX_ID_LENGTH = 64


def scale(k: float, channels: Sequence[str] | None = None) -> Step:
    """Multiply weekly spend by ``k`` in the decision window, for all or some channels."""
    step: Step = {"scale": k}
    if channels is not None:
        step["channels"] = list(channels)
    return step


def move(from_: str, to: str, share: float | None = None, amount: float | None = None) -> Step:
    """Move a share (0 to 1) or an absolute amount of spend from one channel to another."""
    if (share is None) == (amount is None):
        raise ValueError("Pass share or amount, not both and not neither")
    argument = {"share": share} if amount is None else {"amount": amount}
    return {"move": {"from": from_, "to": to, **argument}}


def shift(
    channel: str,
    into_weeks: Sequence[int] | None = None,
    pattern: Sequence[float] | None = None,
) -> Step:
    """Re-time one channel's spend.

    ``into_weeks=(i, j)`` spreads the total evenly over weeks ``i..j`` (1-based, inclusive).
    ``pattern`` spreads it by one weight per week of the decision window.
    """
    if (into_weeks is None) == (pattern is None):
        raise ValueError("Pass into_weeks or pattern, not both and not neither")
    argument = {"into_weeks": list(into_weeks)} if pattern is None else {"pattern": list(pattern)}
    return {"shift": {"channel": channel, **argument}}


def set_controls(**series: Sequence[float]) -> Step:
    """Replace control values in the decision window, one value per week."""
    if not series:
        raise ValueError("Pass at least one control series")
    return {"set_controls": {name: list(values) for name, values in series.items()}}


def curve(channel: str, levels: Sequence[float], base: str = "baseline") -> dict[str, Any]:
    """Ask for ``channel`` at each spend level, as a multiple of its spend in ``base``."""
    return {"channel": channel, "levels": list(levels), "base": base}


def scenario(id: str, *steps: Step, plan: Mapping[str, Any] | None = None) -> Scenario:
    """A named plan: either ordered ``steps`` on the baseline, or a partial ``plan``.

    ``plan`` holds only the series that change, for example
    ``{"spend": {"tv": [...]}}``. Dates come from the baseline.
    """
    if bool(steps) == (plan is not None):
        raise ValueError("Pass steps or plan, not both and not neither")
    if plan is not None:
        return {"id": id, "plan": dict(plan)}
    return {"id": id, "steps": list(steps)}


def grid(**axes: Sequence[float]) -> list[Scenario]:
    """Expand spend multiples per channel into one scenario for every combination.

    ``grid(tv=[0.5, 1, 1.5], search=[1, 2])`` gives scenarios ``tv_0.5__search_2``,
    ``tv_1.5__search_1`` and so on. Each name is a channel and each value scales that
    channel. A multiple of 1 changes nothing, so it adds no step, and the combination
    where every channel is 1 is the baseline and is left out. Ids follow the order of the
    axes and values, so the same call always gives the same ids.
    """
    if not axes:
        raise ValueError("Pass at least one channel, for example grid(tv=[0.5, 1, 1.5])")
    scenarios = []
    for combination in itertools.product(*axes.values()):
        changes = [
            (channel, multiple)
            for channel, multiple in zip(axes, combination, strict=True)
            if multiple != 1
        ]
        if not changes:
            continue
        identifier = "__".join(
            _ID_CHARACTERS.sub("_", f"{channel}_{multiple:g}") for channel, multiple in changes
        )
        steps = [scale(multiple, [channel]) for channel, multiple in changes]
        scenarios.append(scenario(identifier, *steps))
    _check_ids(scenarios)
    return scenarios


def _check_ids(scenarios: Sequence[Scenario]) -> None:
    identifiers = [item["id"] for item in scenarios]
    too_long = [item for item in identifiers if len(item) > MAX_ID_LENGTH]
    if too_long:
        raise ValueError(f"Scenario ids are limited to {MAX_ID_LENGTH} characters: {too_long[0]}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Axis values or channel names make duplicate scenario ids")
