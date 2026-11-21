"""Builders for the scenarios and curves of an analysis request.

Each builder returns the plain dict that goes on the wire, so results can be
printed, edited and sent as they are. The service checks every value; builders
only refuse requests that cannot be sent at all.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence
from typing import Any

#: A channel's change: a number multiplies its spend, ``amount(x)`` sets its total.
Multiplier = float | dict[str, float]
Scenario = dict[str, Any]

_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]")
MAX_ID_LENGTH = 64


def amount(x: float) -> dict[str, float]:
    """Set a channel's total spend in the window to ``x``, for example on a channel with no
    planned spend. The weekly pattern of the plan is kept; a channel with no spend in the
    window gets ``x`` split evenly over its weeks."""
    return {"amount": x}


def scenario(id: str, multipliers: Mapping[str, Multiplier]) -> Scenario:
    """A named budget change: one multiplier or ``amount`` per channel that changes.

    ``scenario("tv_to_social", {"tv": 0.8, "social": amount(24000)})`` multiplies TV spend in
    the window by 0.8 and sets social spend to 24,000 in total. A channel left out keeps its
    spend. The window is the query weeks, or ``decision_weeks`` of a plan.
    """
    if not isinstance(multipliers, Mapping):
        raise TypeError(
            f'Pass multipliers as a dict such as {{"tv": 1.2}}, not {type(multipliers).__name__}'
        )
    return {"id": id, "multipliers": {name: _copy(value) for name, value in multipliers.items()}}


def curve(channel: str, levels: Sequence[float], base: str = "baseline") -> dict[str, Any]:
    """Ask for ``channel`` at each spend level, as a multiple of its spend in ``base``."""
    return {"channel": channel, "levels": list(levels), "base": base}


def grid(axes: Mapping[str, Sequence[float]]) -> list[Scenario]:
    """Expand spend multipliers per channel into one scenario for every combination.

    ``grid({"tv": [0.5, 1, 1.5], "search": [1, 2]})`` gives scenarios ``tv_0.5__search_2``,
    ``tv_1.5__search_1`` and so on. Each key is a channel and each value multiplies that
    channel's spend. A multiplier of 1 changes nothing, so it is left out of the scenario,
    and the combination where every channel is 1 is the baseline and is left out. Ids follow
    the order of the axes and values, so the same call always gives the same ids.
    """
    if not isinstance(axes, Mapping) or not axes:
        raise ValueError('Pass at least one channel, for example grid({"tv": [0.5, 1, 1.5]})')
    scenarios = []
    for combination in itertools.product(*axes.values()):
        changes = {
            channel: multiple
            for channel, multiple in zip(axes, combination, strict=True)
            if multiple != 1
        }
        if not changes:
            continue
        identifier = "__".join(
            _ID_CHARACTERS.sub("_", f"{channel}_{multiple:g}")
            for channel, multiple in changes.items()
        )
        scenarios.append(scenario(identifier, changes))
    _check_ids(scenarios)
    return scenarios


def _copy(value: Any) -> Any:
    return dict(value) if isinstance(value, Mapping) else value


def _check_ids(scenarios: Sequence[Scenario]) -> None:
    identifiers = [item["id"] for item in scenarios]
    too_long = [item for item in identifiers if len(item) > MAX_ID_LENGTH]
    if too_long:
        raise ValueError(f"Scenario ids are limited to {MAX_ID_LENGTH} characters: {too_long[0]}")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Axis values or channel names make duplicate scenario ids")
