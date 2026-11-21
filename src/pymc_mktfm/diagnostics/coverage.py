"""Report-only coverage uncertainty from recorded, cell-identified worlds.

No function runs inference. Every interval targets the equal-world mean,
including metrics whose per-world weights count slots. Those slot counts are
metadata only; bootstrap weights count scored worlds within a corpus/cell pair.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


def is_coverage_key(key: str) -> bool:
    """Whether an evaluator key names pointwise or simultaneous 80% coverage."""
    return any(part in ("cov80", "simcov80") for part in key.rsplit("/", 1)[-1].split("_"))


def coverage_evidence(per_world: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only coverage scores, counts, reasons and recorded world/cell identity."""
    return [
        {
            **{name: world.get(name) for name in ("corpus_index", "row", "cell_id")},
            **{
                name: {key: value for key, value in world[name].items() if is_coverage_key(key)}
                for name in ("values", "weights", "undefined")
            },
        }
        for world in per_world
    ]


def _options(seed, bootstrap_samples):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, Integral)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def coverage_intervals(
    per_world: Sequence[Mapping[str, Any]] | None,
    keys: Sequence[str],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict[str, dict[str, Any]]:
    """95% cell-clustered percentile intervals for recorded coverage keys.

    ``per_world`` is :func:`evaluate_rows` evidence or its coverage-only saved
    subset. Cells are keyed by ``(corpus_index, cell_id)`` and kept in first
    occurrence order. Each cell contributes its mean and scored-world count to
    :func:`_estimate`; both are resampled together. Singleton cells therefore
    reproduce the world bootstrap exactly with the same seed and world order.

    ``n_worlds`` counts finite scores, ``n_cells`` their distinct cells and
    ``n_slots`` their recorded slot weights. Missing/bad cell identity makes the
    interval unavailable without discarding the finite mean or inventing
    independent worlds. One scored cell also has no interval. Undefined and
    nonfinite worlds are excluded and counted by reason. ``None`` evidence, or
    a key never recorded in that evidence, is ``not_recorded``, not zero.
    """
    # Imported here: ``comparisons`` loads training metrics, and archive readers
    # use ``pymc_mktfm.diagnostics`` without importing ``pymc_mktfm.training``.
    from .comparisons import _estimate

    _options(seed, bootstrap_samples)
    invalid_evidence = per_world is not None and (
        not isinstance(per_world, Sequence)
        or isinstance(per_world, (str, bytes))
        or any(not isinstance(world, Mapping) for world in per_world)
    )
    result = {}
    for key in dict.fromkeys(keys):
        evidence = {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "n_worlds": None,
            "n_cells": None,
            "n_slots": None,
            "status": "unavailable",
            "reason": "not_recorded",
            "confidence_level": 0.95,
            "seed": int(seed),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_unit": "cell",
            "estimand": "equal_world",
            "undefined": {},
        }
        result[key] = evidence
        if invalid_evidence:
            evidence["reason"] = "invalid_world_evidence"
            continue
        if per_world is None:
            continue
        if not per_world:
            evidence.update(n_worlds=0, n_cells=0, n_slots=0, reason="empty_worlds")
            continue
        values = []
        cells = {}
        undefined = Counter()
        n_slots = 0
        missing_identity = False
        missing_slots = False
        recorded = False
        for world in per_world:
            scores = world.get("values") or {}
            reasons = world.get("undefined") or {}
            if key not in scores and key not in reasons:
                undefined["not_recorded"] += 1
                continue
            recorded = True
            value = scores.get(key)
            if value is None:
                undefined[reasons.get(key) or "undefined"] += 1
                continue
            if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
                undefined["invalid_value"] += 1
                continue
            if not math.isfinite(value):
                undefined["nonfinite"] += 1
                continue
            value = float(value)
            values.append(value)
            weight = (world.get("weights") or {}).get(key)
            if _integer(weight) and weight > 0:
                n_slots += int(weight)
            else:
                missing_slots = True
            corpus_index, cell_id = world.get("corpus_index"), world.get("cell_id")
            if not _integer(corpus_index) or corpus_index < 0 or not _integer(cell_id):
                missing_identity = True
                continue
            cells.setdefault((int(corpus_index), int(cell_id)), []).append(value)
        if not recorded:
            continue
        evidence.update(
            n_worlds=len(values),
            n_cells=None if missing_identity else len(cells),
            n_slots=None if missing_slots else n_slots,
            undefined=dict(undefined),
        )
        if not values:
            evidence["reason"] = "no_scored_worlds"
            continue
        evidence["mean"] = float(np.mean(values))
        if missing_identity:
            evidence["reason"] = "missing_cell_identity"
            continue
        if len(cells) < 2:
            evidence["reason"] = "fewer_than_two_cells"
            continue
        estimate = _estimate(
            [np.mean(cell) for cell in cells.values()],
            weights=[len(cell) for cell in cells.values()],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        evidence.update(
            ci_low=estimate["ci_low"], ci_high=estimate["ci_high"], status="ok", reason=None
        )
        if undefined.get("nonfinite"):
            evidence.update(
                status="partial", reason=f"nonfinite_in_{undefined['nonfinite']}_worlds"
            )
        elif undefined.get("not_recorded"):
            evidence.update(status="partial", reason="incomplete_world_evidence")
        elif undefined.get("invalid_value"):
            evidence.update(status="partial", reason="invalid_world_values")
        elif missing_slots:
            evidence.update(status="partial", reason="missing_slot_counts")
    return result


def evaluation_coverage_intervals(
    record: Mapping[str, Any],
    tables: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Intervals by leg/key for an event and its already-written evaluation tables.

    Historical records without a coverage table expose ``not_recorded``
    evidence for their recorded coverage keys. Flat scores are never treated as
    per-world observations or used to infer missing cell identities.
    Available recorded row/count metadata must agree with the saved population;
    incomplete evidence retains its observed mean but has no coverage bounds.
    """
    _options(seed, bootstrap_samples)
    keys_by_leg = {leg: {} for leg in (record.get("legs") or {})}
    for key in record.get("flat") or {}:
        leg, _, name = key.partition("/")
        if is_coverage_key(name):
            keys_by_leg.setdefault(leg, {})[name] = None
    saved_legs = (tables or {}).get("legs") or {}
    result = {}
    for leg, keys in keys_by_leg.items():
        per_world = (saved_legs.get(leg) or {}).get("coverage")
        leg_record = (record.get("legs") or {}).get(leg) or {}
        incomplete = False
        if isinstance(per_world, (list, tuple)):
            expected_worlds = leg_record.get("n_worlds")
            incomplete = _integer(expected_worlds) and len(per_world) != expected_worlds
            expected_rows = leg_record.get("rows")
            if expected_rows is not None:
                try:
                    expected = Counter(tuple(row) for row in expected_rows)
                    actual = Counter(
                        (world.get("corpus_index"), world.get("row"))
                        for world in per_world
                        if isinstance(world, Mapping)
                    )
                    incomplete |= actual != expected
                except TypeError:
                    incomplete = True
            for world in per_world:
                if isinstance(world, Mapping):
                    for key in (*world.get("values", {}), *world.get("undefined", {})):
                        if is_coverage_key(key):
                            keys[key] = None
        result[leg] = coverage_intervals(
            per_world, tuple(keys), seed=seed, bootstrap_samples=bootstrap_samples
        )
        if per_world is not None:
            recorded_counts = leg_record.get("metric_worlds") or {}
            for key, interval in result[leg].items():
                count = recorded_counts.get(key)
                count_mismatch = _integer(count) and (
                    (interval["n_worlds"] is None and count > 0)
                    or (interval["n_worlds"] is not None and interval["n_worlds"] != count)
                )
                recorded_mean = (record.get("flat") or {}).get(f"{leg}/{key}")
                mean_mismatch = (
                    interval["mean"] is not None
                    and isinstance(recorded_mean, Real)
                    and math.isfinite(recorded_mean)
                    and not math.isclose(
                        interval["mean"], recorded_mean, rel_tol=1e-12, abs_tol=1e-12
                    )
                )
                if incomplete or count_mismatch or mean_mismatch:
                    interval.update(
                        ci_low=None,
                        ci_high=None,
                        status="unavailable",
                        reason="incomplete_world_evidence"
                        if incomplete or count_mismatch
                        else "inconsistent_recorded_mean",
                    )
    return result
