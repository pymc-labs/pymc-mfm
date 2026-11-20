"""Immutable history and column roles for explicit context/query inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

from pymc_mfm.data.observations import encode_observations
from pymc_mfm.model.config import ModelConfig

if TYPE_CHECKING:
    from pymc_mfm.inference.result import DecompositionResult


def _column_names(value, role: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{role} must be a sequence of column names, not a string")
    try:
        names = tuple(value)
    except TypeError:
        raise ValueError(f"{role} must be a nonempty sequence of column names") from None
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{role} must contain nonempty string column names")
    if any("\x00" in name for name in names):
        raise ValueError(f"{role} must not contain NUL characters")
    for name in names:
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"{role} must contain valid Unicode column names") from None
    return tuple(str(name) for name in names)  # built-in str: np.str_ cannot be an HDF5 attribute


def _ordered_frame(data, required: tuple[str, ...], date_column: str):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one row")
    if not data.columns.is_unique:
        raise ValueError("data column names must be unique")
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    values = data[date_column]
    if is_numeric_dtype(values.dtype) or any(
        isinstance(value, (Number, np.bool_)) for value in values
    ):
        raise ValueError(f"{date_column!r} must contain dates, not numeric row labels")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
        if dates.tz is not None:
            raise ValueError("dates must be timezone-naive; convert them before inference")
        dates = dates.as_unit("ns")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid dates in {date_column!r}: {error}") from None
    if dates.hasnans or dates.has_duplicates:
        raise ValueError(f"{date_column!r} must contain valid, unique dates without NaT")
    order = np.argsort(dates.asi8, kind="stable")
    return data.iloc[order], dates[order]


def _numeric_columns(data: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        dtype = data[name].dtype
        if not is_numeric_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError(f"{name!r} must contain finite real numeric values")
    values = data.loc[:, list(names)].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{list(names)} must contain finite real numeric values")
    for index, name in enumerate(names):
        if (data[name].ne(0) & (values[:, index] == 0)).any():
            raise ValueError(f"{name!r} cannot be represented in float64 without underflow")
    return values


def _cadence(dates: pd.DatetimeIndex) -> pd.DateOffset:
    if len(dates) < 2:
        raise ValueError("context needs at least two dated rows to determine frequency")
    frequency = pd.infer_freq(dates) if len(dates) >= 3 else None
    if frequency is not None:
        return pd.tseries.frequencies.to_offset(frequency)
    intervals = dates[1:] - dates[:-1]
    if not (intervals == intervals[0]).all():
        raise ValueError("context dates must follow a regular frequency without missing periods")
    return pd.tseries.frequencies.to_offset(intervals[0])


@dataclass(frozen=True)
class _FittedContext:
    date_column: str
    target_column: str
    channel_columns: tuple[str, ...]
    control_columns: tuple[str, ...]
    control_names: tuple[str, ...]
    dates: np.ndarray
    spend: np.ndarray
    controls: np.ndarray
    outcome: np.ndarray
    control_mask: np.ndarray
    frequency: pd.DateOffset

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        *,
        model_config: ModelConfig,
        date_column: str,
        target_column: str,
        channel_columns: Sequence[str],
        control_columns: Sequence[str],
    ) -> _FittedContext:
        date_column, target_column = _column_names(
            (date_column, target_column), "date_column and target_column"
        )
        channels = _column_names(channel_columns, "channel_columns")
        controls = _column_names(control_columns, "control_columns")
        required = (date_column, target_column, *channels, *controls)
        if len(set(required)) != len(required):
            raise ValueError("date, target, channel and control column roles must be distinct")
        if len(controls) > model_config.n_controls:
            raise ValueError(
                f"model supports at most {model_config.n_controls} control columns; "
                f"received {len(controls)}"
            )
        frame, dates = _ordered_frame(data, required, date_column)
        frequency = _cadence(dates)
        spend = _numeric_columns(frame, channels)
        represented_controls = _numeric_columns(frame, controls)
        outcome = _numeric_columns(frame, (target_column,))[:, 0]
        if np.any(spend < 0) or np.any(outcome <= 0):
            raise ValueError("channel spend must be nonnegative and context outcomes positive")
        padded_controls = np.zeros((len(frame), model_config.n_controls), dtype=np.float64)
        padded_controls[:, : len(controls)] = represented_controls
        control_mask = np.arange(model_config.n_controls) < len(controls)
        control_names = list(controls)
        for index in range(len(controls), model_config.n_controls):
            name = f"__padded_control_{index}__"
            while name in control_names:
                name = "_" + name
            control_names.append(name)
        # Validate the same context-only representation as inference, without
        # manufacturing a query row or changing any model state.
        encoded = encode_observations(
            spend[None],
            padded_controls[None],
            outcome[None],
            np.ones((1, len(channels)), dtype=bool),
            control_mask[None],
            np.ones((1, model_config.n_demand), dtype=bool),
        )
        for name, value in encoded.items():
            with np.errstate(over="ignore", invalid="ignore"):
                narrowed = value if name == "sales_scale" else value.astype(np.float32)
            if not np.isfinite(narrowed).all():
                raise ValueError(f"{name} exceeds the finite model representation")
        date_values = dates.to_numpy(dtype="datetime64[ns]", copy=True)
        for array in (date_values, spend, padded_controls, outcome, control_mask):
            array.setflags(write=False)
        return cls(
            date_column,
            target_column,
            channels,
            controls,
            tuple(control_names),
            date_values,
            spend,
            padded_controls,
            outcome,
            control_mask,
            frequency,
        )

    def observations(self, data: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        required = (self.date_column, *self.channel_columns, *self.control_columns)
        frame, dates = _ordered_frame(data, required, self.date_column)
        expected = pd.date_range(
            start=pd.Timestamp(self.dates[-1]) + self.frequency,
            periods=len(dates),
            freq=self.frequency,
        )
        if not dates.equals(expected):
            raise ValueError(
                "query dates must start immediately after the last context date and "
                f"follow its {self.frequency.freqstr!r} frequency without gaps"
            )
        spend = _numeric_columns(frame, self.channel_columns)
        controls = _numeric_columns(frame, self.control_columns)
        padded_controls = np.zeros((len(frame), self.controls.shape[1]), dtype=np.float64)
        padded_controls[:, : len(self.control_columns)] = controls
        return {
            "spend": np.concatenate((self.spend, spend)),
            "controls": np.concatenate((self.controls, padded_controls)),
            "context_outcome": self.outcome,
            "control_mask": self.control_mask,
        }, np.concatenate((self.dates, dates.to_numpy(dtype="datetime64[ns]")))

    def query_outcome(self, data: pd.DataFrame) -> np.ndarray | None:
        """Observed query outcomes in date order, for figures only; never model input.

        ``None`` when ``data`` has no target column or none of its values is a finite
        number. Other entries that are not finite numbers become NaN (unobserved), so
        the column never fails or changes inference. Call after ``observations``,
        which validates the dates.
        """
        if self.target_column not in data.columns:
            return None
        frame, _ = _ordered_frame(data, (self.date_column,), self.date_column)
        column = frame[self.target_column]
        if is_datetime64_any_dtype(column.dtype) or is_timedelta64_dtype(column.dtype):
            return None
        if not is_numeric_dtype(column.dtype):
            try:
                column = pd.to_numeric(column, errors="coerce")
            except (TypeError, ValueError, OverflowError):
                # pandas refuses some object columns whole (an int past float64, a list):
                # convert entry by entry instead, each failure unobserved.
                column = column.map(_number).astype(np.float64)
        if (
            not is_numeric_dtype(column.dtype)
            or is_complex_dtype(column.dtype)
            or is_bool_dtype(column.dtype)
        ):
            return None
        values = column.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        values[~np.isfinite(values)] = np.nan
        if np.isnan(values).all():
            return None
        values.setflags(write=False)
        return values


def _number(value: object) -> float:
    """``value`` as a float, or NaN when it is not a number float64 can hold."""
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan


@dataclass(frozen=True, slots=True)
class _PredictionSnapshot:
    """Owned inputs and canonical sampling options for one successful query."""

    fitted_context: _FittedContext
    observations: dict[str, np.ndarray]
    time: np.ndarray
    query: DecompositionResult
    annotations: dict[str, object] | None
    n_samples: int
    mode: str
    router_samples: int | None
    units: str
    #: Observed query outcomes for figures (NaN where unobserved), or ``None``.
    query_outcome: np.ndarray | None = None

    def __post_init__(self):
        # Reuse the observation arrays already owned by the dataframe adapter,
        # including the fitted outcomes; never copy draws or re-encode here.
        for values in self.observations.values():
            values.setflags(write=False)
        self.time.setflags(write=False)
