"""The completed inference charts draw: history, forecast and named window draws.

:class:`IdataReader` is the one place the charts read a model tree (``MFM.idata``, or
one saved and opened again). It reads only the tree; it never samples.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymc_mfm.inference.curves import ResponseCurves

if TYPE_CHECKING:
    from xarray import DataArray, Dataset, DataTree

WINDOWS = ("context", "query")
#: Component families from the bottom of a stack up: channels end on top.
FAMILIES = ("baseline", "latent", "indirect", "control", "channel")
_CURVE_PATH = "/response_curves/summary"
_CURVE_INTERVAL = 0.9


@dataclass(frozen=True, slots=True)
class Window:
    """Draws of one window, active components only, with what was observed there.

    ``draws`` is ``[sample, week, component]`` in outcome units; ``components`` are
    display names (``"tv"``, not ``"channel:tv"``) and ``families`` their component
    families. ``outcome`` is the observed outcome by week (NaN where unobserved), or
    ``None`` when none was supplied. ``media`` holds the raw weekly media covariates,
    one column per name in ``channels``: the channel components, in component order.
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


@dataclass(frozen=True, slots=True)
class CurveSummary:
    """Stored curve statistics, each ``[channel, level]``, with no posterior draws.

    ``warnings`` holds each point's warning text: ``""`` where it has none, and several
    distinct messages joined by ``"\\n"``.
    """

    channels: tuple[str, ...]
    levels: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    media: np.ndarray
    total_media: np.ndarray
    weeks: int
    n_samples: int
    interval: float
    warnings: np.ndarray


def _curve_array(data: Dataset, name: str, *dims: str) -> DataArray:
    path = f"{_CURVE_PATH}/{name}"
    if name not in data.data_vars:
        raise ValueError(f"{path}: variable is missing")
    values = data[name]
    if len(values.dims) != len(dims) or set(values.dims) != set(dims):
        raise ValueError(f"{path}: expected dims {dims} in any order, got {values.dims}")
    return values.transpose(*dims)


def _curve_values(data: Dataset, name: str, *dims: str) -> np.ndarray:
    values = _curve_array(data, name, *dims)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{_CURVE_PATH}/{name}: must hold numbers, got dtype {values.dtype}")
    return values.to_numpy()


def _curve_warnings(data: Dataset, shape: tuple[int, ...]) -> np.ndarray:
    """The optional ``warning`` text ``[channel, level]``; a summary without one has none."""
    if "warning" not in data.data_vars:
        return np.full(shape, "")
    text = _curve_array(data, "warning", "channel", "level").to_numpy()
    if text.dtype.kind != "U" and not (
        text.dtype.kind == "O" and all(isinstance(item, str) for item in text.flat)
    ):
        raise ValueError(f"{_CURVE_PATH}/warning: must hold text, got dtype {text.dtype}")
    return text.astype(str)


def _curve_count(node: DataTree, name: str) -> int:
    value = node.attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(
            f"/response_curves: attribute {name!r} must be a positive integer, got {value!r}"
        )
    return int(value)


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


def _show(labels: Sequence) -> list[str]:
    return [str(label) for label in np.asarray(labels)]


def _select(path: str, data: DataArray, dim: str, wanted: Sequence) -> DataArray:
    """``data`` at the ``dim`` labels ``wanted``, in that order.

    Raises ``ValueError`` naming ``path`` and the labels if a label is absent or repeated.
    """
    stored = data[dim].to_numpy()
    if len(np.unique(stored)) != len(stored):
        raise ValueError(f"{path}: {dim} labels must be unique, got {_show(stored)}")
    missing = ~np.isin(np.asarray(wanted), stored)
    if missing.any():
        raise ValueError(
            f"{path}: {dim} labels {_show(stored)} do not hold "
            f"{_show(np.asarray(wanted)[missing])}; expected {_show(wanted)}"
        )
    return data.sel({dim: list(wanted)})


def _one_chain(path: str, data: Dataset | DataArray) -> None:
    """Raise ``ValueError`` unless ``data`` has a chain axis of size 1: the charts read chain 0."""
    chains = data.sizes.get("chain")
    if chains != 1:
        raise ValueError(f"{path}: charts read a tree with one chain, got {chains}")


@dataclass(frozen=True, slots=True)
class _Parts:
    """What :meth:`IdataReader.window` reads before it stacks the draws."""

    dates: np.ndarray
    components: tuple[str, ...]
    families: tuple[str, ...]
    columns: list[DataArray]
    outcome: np.ndarray | None
    channels: tuple[str, ...]
    media: np.ndarray
    units: str


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

    def _observed(self, name: str, dates: np.ndarray) -> np.ndarray | None:
        """The observed outcome at ``dates`` (by label), or ``None`` without an outcome."""
        if "observed_data" not in self._tree()[name].children:
            return None
        outcome = self._group(name, "observed_data")["outcome"]
        return _select(f"/{name}/observed_data", outcome, "date", dates).to_numpy()

    def _covariate(
        self, window: str, variable: str, dim: str, names: Sequence[str], dates: np.ndarray
    ) -> np.ndarray:
        """``variable`` of ``window`` as ``[date, name]`` at ``dates`` and ``names``, by label."""
        path = f"/{window}/constant_data"
        stored = self._group(window, "constant_data")[variable].transpose("date", dim)
        return _select(path, _select(path, stored, "date", dates), dim, names).to_numpy()

    def dates(self, name: str) -> np.ndarray:
        """The weeks of the ``"context"`` or ``"query"`` window; reads no draws."""
        if name not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
        return self._group(name, "posterior")["date"].to_numpy()

    def covariates(self) -> Covariates:
        """History then forecast inputs of the last complete inference."""
        channels, controls = self._names("channel_columns"), self._names("control_columns")
        weeks = [self.dates(name) for name in WINDOWS]
        outcomes = []
        for name, dates in zip(WINDOWS, weeks, strict=True):
            observed = self._observed(name, dates)
            outcomes.append(np.full(len(dates), np.nan) if observed is None else observed)
        return Covariates(
            dates=np.concatenate(weeks),
            first_query=len(weeks[0]),
            outcome=np.concatenate(outcomes),
            channels=channels,
            media=np.concatenate(
                [
                    self._covariate(name, "media", "channel", channels, dates)
                    for name, dates in zip(WINDOWS, weeks, strict=True)
                ]
            ),
            control_names=controls,
            controls=np.concatenate(
                [
                    self._covariate(name, "controls", "control", controls, dates)
                    for name, dates in zip(WINDOWS, weeks, strict=True)
                ]
            ),
        )

    def curve_summary(self, *, fallback: bool = False) -> CurveSummary | None:
        """Read and validate the stored summary; ``fallback`` prefers the local draws."""
        node = self._tree().children.get("response_curves")
        if node is None or "summary" not in node.children:
            return None
        if fallback and "posterior" in node.children:
            return None
        data = node.children["summary"].to_dataset(inherit=False)
        interval = data.attrs.get("interval")
        if not isinstance(interval, Real) or interval != _CURVE_INTERVAL:
            raise ValueError(f"{_CURVE_PATH}: attribute 'interval' must be {_CURVE_INTERVAL}")
        weeks, n_samples = _curve_count(node, "weeks"), _curve_count(node, "n_samples")
        units = node.attrs.get("units")
        if not isinstance(units, str) or not units:
            raise ValueError("/response_curves: attribute 'units' must be nonempty text")
        for dim in ("channel", "level"):
            if dim not in data.coords or data[dim].dims != (dim,):
                raise ValueError(f"{_CURVE_PATH}: expected a one-dimensional {dim} coordinate")
        channels = tuple(data["channel"].to_numpy().tolist())
        if not channels or not all(isinstance(name, str) for name in channels):
            raise ValueError(f"{_CURVE_PATH}: channel coordinate must hold one or more names")
        available = self._names("channel_columns")
        if len(set(channels)) != len(channels) or any(name not in available for name in channels):
            raise ValueError(
                f"{_CURVE_PATH}: channel coordinate must hold distinct names from {list(available)}"
            )
        levels = data["level"].to_numpy()
        if (
            levels.dtype.kind not in "iuf"
            or levels.size < 2
            or not np.isfinite(levels).all()
            or (levels < 0).any()
            or not (levels[1:] > levels[:-1]).all()
        ):
            raise ValueError(
                f"{_CURVE_PATH}: level coordinate must hold at least two finite, nonnegative, "
                "strictly increasing numbers"
            )
        mean, low, high, media = (
            _curve_values(data, name, "channel", "level")
            for name in ("mean", "low", "high", "media")
        )
        total_media = _curve_values(data, "total_media", "channel")
        for name, values in (("media", media), ("total_media", total_media)):
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{_CURVE_PATH}/{name}: must be finite and nonnegative")
        missing = np.isnan(mean)
        if (
            not np.array_equal(missing, np.isnan(low))
            or not np.array_equal(missing, np.isnan(high))
            or any(np.isinf(values).any() for values in (mean, low, high))
        ):
            raise ValueError(
                f"{_CURVE_PATH}: mean, low and high must be finite or all missing at each point"
            )
        # A central interval need not contain the mean of an asymmetric distribution.
        if (low > high).any():
            raise ValueError(f"{_CURVE_PATH}: low must be less than or equal to high at each point")
        return CurveSummary(
            channels=channels,
            levels=levels,
            mean=mean,
            low=low,
            high=high,
            media=media,
            total_media=total_media,
            weeks=weeks,
            n_samples=n_samples,
            interval=_CURVE_INTERVAL,
            warnings=_curve_warnings(data, mean.shape),
        )

    def response_curves(self) -> ResponseCurves:
        """The latest curves of the last complete inference; raises before a sweep."""
        tree = self._tree()
        if "response_curves" not in tree.children:
            raise RuntimeError("call model.response_curves(...) first")
        posterior = self._group("response_curves", "posterior")["contribution"]
        _one_chain("/response_curves/posterior", posterior)
        constants = self._group("response_curves", "constant_data")
        attrs = tree["response_curves"].attrs
        channels = posterior["channel"].to_numpy()
        dates = constants["date"].to_numpy()
        # Each channel's observed total over the swept weeks, from the media the forecast
        # window stores. The curves' own stored media is that total times each level.
        path = "/query/constant_data"
        media = self._group("query", "constant_data")["media"]
        media = _select(path, _select(path, media, "date", dates), "channel", channels)
        return ResponseCurves(
            channels=tuple(str(name) for name in channels),
            levels=posterior["level"].to_numpy(),
            media=media.sum("date").to_numpy(),
            draws=posterior.isel(chain=0).transpose("channel", "level", "draw").to_numpy(),
            weeks=int(attrs["weeks"]),
            dates=dates,
            units=str(attrs["units"]),
            n_samples=int(attrs["n_samples"]),
        )

    def curve_media(self) -> np.ndarray:
        """The stored spend ``[channel, level]`` of the curves, by label, in curve order."""
        posterior = self._group("response_curves", "posterior")["contribution"]
        path = "/response_curves/constant_data"
        stored = self._group("response_curves", "constant_data")["media"]
        stored = stored.transpose("channel", "level")
        stored = _select(
            path,
            _select(path, stored, "channel", posterior["channel"].to_numpy()),
            "level",
            posterior["level"].to_numpy(),
        )
        return stored.to_numpy()

    def _parts(self, name: str) -> _Parts:
        """A window's draw columns, labels and covariates, aligned by label and not stacked."""
        if name not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {name!r}")
        path = f"/{name}/posterior"
        posterior = self._group(name, "posterior")
        constants = self._group(name, "constant_data")
        _one_chain(path, posterior)
        dates = posterior["date"].to_numpy()
        full = [str(label) for label in constants["component"].to_numpy()]
        families = tuple(str(label) for label in constants["component_family"].to_numpy())
        unknown = [family for family in dict.fromkeys(families) if family not in FAMILIES]
        if unknown:
            raise ValueError(
                f"/{name}/constant_data: unknown component family {unknown[0]!r}; "
                f"expected one of {list(FAMILIES)}"
            )
        plain, columns = [], []
        for label, family in zip(full, families, strict=True):
            if family in ("channel", "control"):
                plain.append(label.removeprefix(f"{family}:"))
                variable = posterior[f"{family}_contribution"].isel(chain=0)
                columns.append(_select(path, variable, family, plain[-1:]).isel({family: 0}))
            else:
                plain.append(label)
                variable = label.replace(":", "_")
                if variable not in posterior:
                    raise ValueError(f"{path}: no variable {variable!r} for component {label!r}")
                columns.append(posterior[variable].isel(chain=0))
        # Scalar contributions (``baseline``, ``latent``, ``indirect_*``); the ``channel`` and
        # ``control`` variables are checked by role below, and ``graph`` is no contribution.
        described = {label.replace(":", "_") for label in full}
        undescribed = [
            str(variable)
            for variable in posterior.data_vars
            if str(variable).split("_")[0] in FAMILIES
            and str(variable).split("_")[0] not in ("channel", "control")
            and variable not in described
        ]
        if undescribed:
            raise ValueError(
                f"{path}: no descriptor in /{name}/constant_data for {undescribed}; "
                f"its descriptors are {full}"
            )
        columns = [column.transpose("draw", "date") for column in columns]
        if len({column.shape for column in columns}) > 1:
            raise ValueError(f"{path}: component draws differ in shape")
        roles = {
            "channel": self._names("channel_columns"),
            "control": self._names("control_columns"),
        }
        for family, named in roles.items():
            described = [n for n, f in zip(plain, families, strict=True) if f == family]
            if len(described) != len(set(described)):
                raise ValueError(
                    f"/{name}/constant_data: {family} descriptors must name each column once, "
                    f"got {described}"
                )
            absent = [label for label in named if label not in described]
            unknown = [label for label in described if label not in named]
            if absent or unknown:
                raise ValueError(
                    f"/{name}/constant_data: {family} descriptors {described} do not match "
                    f"the {family} columns {list(named)}: missing {absent}, unknown {unknown}"
                )
        channels = [n for n, family in zip(plain, families, strict=True) if family == "channel"]
        return _Parts(
            dates=dates,
            components=unique_names(plain, full),
            families=families,
            columns=columns,
            outcome=self._observed(name, dates),
            channels=tuple(channels),
            media=self._covariate(name, "media", "channel", channels, dates),
            units=str(posterior.attrs["units"]),
        )

    def check_window(self, name: str) -> None:
        """Raise ``ValueError`` unless ``name``'s labels and columns line up; stacks no draws."""
        self._parts(name)

    def window(self, name: str) -> Window:
        """Read the stored ``"context"`` or ``"query"`` marginal."""
        parts = self._parts(name)
        return Window(
            name=name,
            dates=parts.dates,
            components=parts.components,
            families=parts.families,
            draws=np.stack([column.to_numpy() for column in parts.columns], 2),
            outcome=parts.outcome,
            channels=parts.channels,
            media=parts.media,
            units=parts.units,
        )

    def windows(self, window: str | None) -> tuple[Window, ...]:
        """Both windows in time order when ``window`` is ``None``, else that one."""
        if window is None:
            return tuple(self.window(name) for name in WINDOWS)
        return (self.window(window),)
