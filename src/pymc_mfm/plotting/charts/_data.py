"""The completed inference charts draw: history, forecast and named window draws.

:class:`IdataReader` is the one place the charts read a model tree (``MFM.idata``, or
one saved and opened again). It reads only the tree; it never samples.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymc_mfm.inference.curves import ResponseCurves

if TYPE_CHECKING:
    from xarray import Dataset, DataTree

WINDOWS = ("context", "query")


@dataclass(frozen=True, slots=True)
class Window:
    """Draws of one window, active components only, with what was observed there.

    ``draws`` is ``[sample, week, component]`` in outcome units; ``components`` are
    display names (``"tv"``, not ``"channel:tv"``) and ``families`` their component
    families. ``outcome`` is the observed outcome by week (NaN where unobserved), or
    ``None`` when none was supplied. ``media`` holds the raw weekly media covariates,
    one column per name in ``channels``.
    """

    name: str
    dates: np.ndarray
    components: tuple[str, ...]
    families: tuple[str, ...]
    draws: np.ndarray
    outcome: np.ndarray | None
    channels: tuple[str, ...]
    media: np.ndarray
    units: str

    @property
    def n_samples(self) -> int:
        return self.draws.shape[0]

    @property
    def expected(self) -> np.ndarray:
        """Expected outcome draws ``[sample, week]``: the sum of every component."""
        return self.draws.sum(axis=2)

    def contribution(self, name: str) -> np.ndarray:
        """Draws ``[sample, week]`` of the component named ``name``."""
        return self.draws[:, :, self.components.index(name)]


@dataclass(frozen=True, slots=True)
class Covariates:
    """Inputs by week over history then forecast: what the inputs chart draws.

    ``outcome`` is NaN where unobserved; ``media`` and ``controls`` have one column per
    name in ``channels`` and ``control_names``; ``first_query`` is the first forecast week.
    """

    dates: np.ndarray
    first_query: int
    outcome: np.ndarray
    channels: tuple[str, ...]
    media: np.ndarray
    control_names: tuple[str, ...]
    controls: np.ndarray


def unique_names(plain: Sequence[str], full: Sequence[str]) -> tuple[str, ...]:
    """Display names: ``plain`` names, with ``full`` where one repeats, until all are unique.

    ``full`` must hold no repeat, or no display names are unique: that raises ``ValueError``.
    """
    if len(set(full)) != len(full):
        raise ValueError(f"component names must be unique, got {list(full)}")
    names = dict(enumerate(plain))
    while True:
        counts = Counter(names.values())
        repeated = [index for index, name in names.items() if counts[name] > 1]
        if not repeated:
            return tuple(names.values())
        for index in repeated:
            names[index] = full[index]


def interval_quantiles(interval: float) -> tuple[float, float]:
    """Lower and upper quantiles of a central ``interval`` (a fraction)."""
    if isinstance(interval, bool) or not 0 < interval < 1:
        raise ValueError(f"interval must be a fraction between 0 and 1, got {interval!r}")
    return (1 - interval) / 2, (1 + interval) / 2


def summarize(
    values: np.ndarray, interval: float, axis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and central ``interval`` bounds of ``values`` over ``axis`` (the draws)."""
    low, high = interval_quantiles(interval)
    bounds = np.quantile(values, [low, high], axis=axis)
    return values.mean(axis=axis), bounds[0], bounds[1]


def window_weeks(window: Window) -> str:
    """``"24 history weeks"`` or ``"1 forecast week"``: the weeks ``window`` spans."""
    count = len(window.dates)
    kind = "history" if window.name == "context" else "forecast"
    return f"{count} {kind} week{'' if count == 1 else 's'}"


def checked_names(names: Sequence[str], available: Sequence[str], role: str) -> tuple[str, ...]:
    """``names`` as a tuple, each one of ``available`` once; ``role`` names them in errors."""
    if isinstance(names, str):
        raise ValueError(f"{role} must be a sequence of names, not a string")
    names = tuple(names)
    if not names:
        raise ValueError(f"{role} must name at least one of {list(available)}")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown {role} {unknown}; available: {list(available)}")
    repeated = [name for name, count in Counter(names).items() if count > 1]
    if repeated:
        raise ValueError(f"{role} repeat {repeated}; name each once")
    return names


def truth_by_week(truth: pd.DataFrame, window: Window) -> dict[str, np.ndarray]:
    """Each column of ``truth`` on the weeks of ``window``: NaN where it has no value.

    ``truth`` holds known contributions, from a simulation say: a DataFrame indexed
    by date with one column per component display name. Its other weeks are ignored.
    """
    if not isinstance(truth, pd.DataFrame) or not isinstance(truth.index, pd.DatetimeIndex):
        raise ValueError("truth must be a pandas DataFrame indexed by date (a DatetimeIndex)")
    if not truth.columns.is_unique:
        raise ValueError("truth columns must be unique")
    unknown = [column for column in truth.columns if column not in window.components]
    if unknown:
        raise ValueError(
            f"truth columns {unknown} are not components; components: {list(window.components)}"
        )
    weekly = truth.reindex(window.dates)
    return {
        column: weekly[column].to_numpy(dtype=np.float64, na_value=np.nan)
        for column in weekly.columns
    }


class IdataReader:
    """Read access to a model tree: ``/context``, ``/query`` and ``/response_curves``.

    ``idata`` is ``None`` before the first inference; every read then raises.
    """

    __slots__ = ("_idata",)

    def __init__(self, idata: DataTree | None):
        self._idata = idata

    def _tree(self) -> DataTree:
        """The tree, checked: raises before an inference or for a tree that is not a model's."""
        if self._idata is None:
            raise RuntimeError(
                "call model.inference(history, future, ...) successfully before plotting"
            )
        missing = [name for name in WINDOWS if name not in self._idata.children]
        if missing:
            raise ValueError(f"not a model tree: no {missing} group; expected /context and /query")
        return self._idata

    def _group(self, *path: str) -> Dataset:
        return self._tree()["/" + "/".join(path)].to_dataset(inherit=False)

    def _names(self, key: str) -> tuple[str, ...]:
        return tuple(json.loads(self._tree().attrs[key]))

    def _observed(self, name: str) -> np.ndarray | None:
        if "observed_data" not in self._tree()[name].children:
            return None
        return self._group(name, "observed_data")["outcome"].to_numpy()

    def _covariate(self, window: str, variable: str, dim: str, key: str) -> np.ndarray:
        """``variable`` of ``window`` as ``[date, name]``, columns in the order of ``key``."""
        stored = self._group(window, "constant_data")[variable]
        return stored.transpose("date", dim).sel({dim: list(self._names(key))}).to_numpy()

    def covariates(self) -> Covariates:
        """History then forecast inputs of the last complete inference."""
        parts = [self._group(name, "constant_data") for name in WINDOWS]
        outcomes = [
            np.full(len(part["date"]), np.nan) if observed is None else observed
            for part, observed in zip(parts, map(self._observed, WINDOWS), strict=True)
        ]
        return Covariates(
            dates=np.concatenate([part["date"].to_numpy() for part in parts]),
            first_query=len(parts[0]["date"]),
            outcome=np.concatenate(outcomes),
            channels=self._names("channel_columns"),
            media=np.concatenate(
                [self._covariate(name, "media", "channel", "channel_columns") for name in WINDOWS]
            ),
            control_names=self._names("control_columns"),
            controls=np.concatenate(
                [
                    self._covariate(name, "controls", "control", "control_columns")
                    for name in WINDOWS
                ]
            ),
        )

    def response_curves(self) -> ResponseCurves:
        """The latest curves of the last complete inference; raises before a sweep."""
        tree = self._tree()
        if "response_curves" not in tree.children:
            raise RuntimeError("call model.response_curves(...) first")
        posterior = self._group("response_curves", "posterior")["contribution"]
        constants = self._group("response_curves", "constant_data")
        attrs = tree["response_curves"].attrs
        levels = posterior["level"].to_numpy()
        channels = posterior["channel"].to_numpy()
        # The tree holds each channel's observed total times the level: undo it at level 1,
        # or at the top level when the sweep has no level 1.
        ones = np.flatnonzero(levels == 1.0)
        pick = ones[0] if ones.size else int(np.argmax(levels))
        return ResponseCurves(
            channels=tuple(str(name) for name in channels),
            levels=levels,
            media=constants["media"]
            .transpose("channel", "level")
            .sel(channel=channels)
            .isel(level=pick)
            .to_numpy()
            / levels[pick],
            draws=posterior.isel(chain=0).transpose("channel", "level", "draw").to_numpy(),
            weeks=int(attrs["weeks"]),
            dates=constants["date"].to_numpy(),
            units=str(attrs["units"]),
            n_samples=int(attrs["n_samples"]),
        )

    def window(self, name: str) -> Window:
        """Read the stored ``"context"`` or ``"query"`` marginal."""
        if name not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
        posterior = self._group(name, "posterior")
        constants = self._group(name, "constant_data")
        full = [str(label) for label in constants["component"].to_numpy()]
        families = tuple(str(label) for label in constants["component_family"].to_numpy())
        plain, columns = [], []
        for label, family in zip(full, families, strict=True):
            if family in ("channel", "control"):
                plain.append(label.removeprefix(f"{family}:"))
                variable = posterior[f"{family}_contribution"].isel(chain=0)
                columns.append(variable.sel({family: plain[-1]}))
            else:
                plain.append(label)
                columns.append(posterior[label.replace(":", "_")].isel(chain=0))
        return Window(
            name=name,
            dates=posterior["date"].to_numpy(),
            components=unique_names(plain, full),
            families=families,
            draws=np.stack([column.transpose("draw", "date").to_numpy() for column in columns], 2),
            outcome=self._observed(name),
            channels=self._names("channel_columns"),
            media=self._covariate(name, "media", "channel", "channel_columns"),
            units=str(posterior.attrs["units"]),
        )

    def windows(self, window: str | None) -> tuple[Window, ...]:
        """Both windows in time order when ``window`` is ``None``, else that one."""
        if window is None:
            return tuple(self.window(name) for name in WINDOWS)
        return (self.window(window),)
