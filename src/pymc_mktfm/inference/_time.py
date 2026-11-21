"""Private numeric/datetime coordinate handling for saved inference consumers."""

from __future__ import annotations

import numpy as np


def _validate_time(values, *, length=None, allow_bool=True, name="time"):
    """A nonempty increasing coordinate, without changing its values or datetime unit."""
    array = np.asarray(values)
    kinds = "biufM" if allow_bool else "iufM"
    if (
        array.ndim != 1
        or array.size == 0
        or (length is not None and array.shape != (length,))
        or array.dtype.kind not in kinds
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be a vector of finite numeric or valid datetime64 values"
            + (f" with length {length}" if length is not None else "")
        )
    if (array[1:] <= array[:-1]).any():
        raise ValueError(f"{name} labels must be strictly increasing")
    return array


def _time_to_json(values):
    """JSON-native coordinates: ISO dates, never datetime64.tolist()'s integer ticks."""
    array = np.asarray(values)
    if array.dtype.kind == "M":
        if np.isnat(array).any():
            raise ValueError("time cannot contain NaT")
        array = array.astype(array.dtype.newbyteorder("="), copy=False)
        return np.datetime_as_string(array).tolist()
    return array.tolist()


def _restore_time(values):
    """Restore ISO dates or numeric JSON coordinates without rounding unsigned labels."""
    array = np.asarray(values)
    if array.dtype.kind in "SU":
        return np.asarray(values, dtype="datetime64")
    if isinstance(values, (list, tuple)) and values and all(isinstance(v, int) for v in values):
        largest = max(values)
        if np.iinfo(np.int64).max < largest <= np.iinfo(np.uint64).max and min(values) >= 0:
            return np.asarray(values, dtype=np.uint64)
    return array


def _time_equal(first, second):
    """Compare coordinates without cross-unit datetime overflow or numeric coercion."""
    first, second = np.asarray(first), np.asarray(second)
    if (first.dtype.kind == "M") != (second.dtype.kind == "M"):
        return False
    if first.dtype.kind == "M" and first.dtype != second.dtype:
        dtype = np.result_type(first.dtype, second.dtype)
        left, right = first.astype(dtype, copy=False), second.astype(dtype, copy=False)
        if not np.array_equal(left.astype(first.dtype), first) or not np.array_equal(
            right.astype(second.dtype), second
        ):
            return False
        first, second = left, right
    if first.dtype != second.dtype and first.dtype.kind in "biuf" and second.dtype.kind in "biuf":
        return first.shape == second.shape and first.tolist() == second.tolist()
    return np.array_equal(first, second)
