"""Convert budget, share and distance limits into feasible channel bounds."""

import math

import numpy as np


def _numeric_array(value, shape, name):
    """Parse numeric scalar or exact-shape sequence, rejecting boolean values."""
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} must not contain booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain numbers")
    if raw.ndim == 0:
        raw = np.full(shape, raw.item(), dtype=np.float64)
    elif len(shape) == 2 and raw.shape == (shape[1],):
        raw = np.broadcast_to(raw, shape)
    elif raw.shape != shape:
        raise ValueError(f"{name} must be a scalar or have shape {shape}")
    return raw.astype(np.float64, copy=False)


def resolve_bounds(request, totals, channels, *, multi_period):
    """Resolve limits for each period, in spend units, before neural inference.

    ``current`` is a fixed reference per channel. ``max_change`` is an absolute
    allowed deviation, and ``max_change_fraction`` is relative to current.
    Exactly one may be supplied. It applies to every period independently;
    it is not a week-to-week ramp constraint. ``upper_share`` is a fraction of
    each period's total. A mapping can name channels with static upper-share
    limits; omitted channels default to one. Explicit bounds intersect with
    these derived bounds.
    """
    allowed = {
        "channels",
        "spend",
        "outcome",
        "controls",
        "lower",
        "upper",
        "upper_share",
        "current",
        "max_change",
        "max_change_fraction",
        "totals" if multi_period else "budget",
    }
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"unknown input fields: {sorted(unknown)}")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
        raise ValueError("channels must be a positive integer")
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(totals, dtype=object).flat):
        raise ValueError("totals must not contain booleans")
    total = np.asarray(totals)
    if total.dtype.kind not in "iuf" or total.ndim != 1 or not len(total):
        raise ValueError("totals must be a nonempty numeric vector")
    if not multi_period and len(total) != 1:
        raise ValueError("single-period input requires one total")
    total = total.astype(np.float64)
    if not np.isfinite(total).all() or np.any(total <= 0):
        raise ValueError("totals must be finite and positive")
    shape = (len(total), channels)
    external_shape = shape if multi_period else (channels,)
    lo = _numeric_array(request.get("lower", 0), external_shape, "lower").reshape(shape)
    hi = _numeric_array(
        request.get(
            "upper",
            np.broadcast_to(total[:, None], shape) if multi_period else np.full(channels, total[0]),
        ),
        external_shape,
        "upper",
    ).reshape(shape)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("bounds must be finite")
    if np.any(lo < 0) or np.any(hi < lo):
        raise ValueError("bounds must satisfy 0 <= lower <= upper")
    if "upper_share" in request:
        share_value = request["upper_share"]
        if isinstance(share_value, dict):
            names = request.get("channels")
            if (
                not isinstance(names, list)
                or len(names) != channels
                or not all(isinstance(name, str) and name for name in names)
                or len(set(names)) != channels
            ):
                raise ValueError("named upper_share requires unique channel names")
            unknown_channels = set(share_value) - set(names)
            if unknown_channels:
                raise ValueError(
                    f"unknown upper_share channels: {sorted(unknown_channels, key=str)}"
                )
            per_channel = _numeric_array(
                [share_value.get(name, 1.0) for name in names], (channels,), "upper_share"
            )
            share_value = np.broadcast_to(per_channel, shape) if multi_period else per_channel
        share = _numeric_array(share_value, external_shape, "upper_share").reshape(shape)
        if not np.isfinite(share).all() or np.any((share < 0) | (share > 1)):
            raise ValueError("upper_share must be finite and between zero and one")
        hi = np.minimum(hi, total[:, None] * share)
    abs_present = "max_change" in request
    rel_present = "max_change_fraction" in request
    if abs_present and rel_present:
        raise ValueError("specify only one of max_change and max_change_fraction")
    if ("current" in request) != (abs_present or rel_present):
        raise ValueError("current and exactly one maximum change must be supplied together")
    if abs_present or rel_present:
        current = _numeric_array(request["current"], (channels,), "current")
        change = _numeric_array(
            request["max_change" if abs_present else "max_change_fraction"],
            (channels,),
            "maximum change",
        )
        if not np.isfinite(current).all() or not np.isfinite(change).all():
            raise ValueError("current and maximum change must be finite")
        if np.any(current < 0) or np.any(change < 0):
            raise ValueError("current and maximum change must be nonnegative")
        distance = change if abs_present else current * change
        lo = np.maximum(lo, np.maximum(current - distance, 0)[None, :])
        hi = np.minimum(hi, (current + distance)[None, :])
    hi = np.minimum(hi, total[:, None])
    # Use a tight scale-aware tolerance to avoid rejecting normal decimal input.
    tolerance = np.maximum(total, 1) * 1e-10
    if (
        np.any(lo > hi + tolerance[:, None])
        or np.any(lo.sum(1) > total + tolerance)
        or np.any(hi.sum(1) < total - tolerance)
    ):
        raise ValueError("combined budget constraints are infeasible")
    # Avoid admitting a tiny inconsistent interval under the numerical tolerance.
    if np.any(lo > hi):
        raise ValueError("combined channel bounds are infeasible")
    if not math.isfinite(float(lo.sum())) or not math.isfinite(float(hi.sum())):
        raise ValueError("combined bounds must be finite")
    return lo, hi
