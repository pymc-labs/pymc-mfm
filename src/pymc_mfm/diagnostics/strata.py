"""Stratified per-effect readouts: where window-total coverage and residuals fail.

A row describes one active channel or control effect of one world: its
window-total truth, posterior mean, central 80% interval and CRPS in
``sales_scale`` units, the frequency of its direct ``cy``/``zb`` edge in the
same draws, its graph-truth live label, and the world's active and truly live
channel counts. :func:`strata_summary` groups rows per family by
:data:`STRATA_AXES` and joins each edge decision (TP/FP/TN/FN at every
:data:`~pymc_mfm.training.metrics.THRESHOLDS` value) with the size error.

Size bins are fixed by ``cuts``, stored with every summary: an exact ``zero``
bin, then right-closed bins of the absolute window total between the cuts.
:func:`size_cuts` takes them from quantiles of the rows' nonzero absolute true
totals, which do not depend on the model or its draws, so a replay or another
model on the same worlds bins alike. The same cuts bin the true and the
predicted size. Residuals are signed, ``mean - truth``, and never divided by
the truth. A stratum without effects is ``unavailable`` with its reason; no
cell is zero-filled. Everything is pure NumPy and strict JSON, and every
number of a summary sits inside a list, so record series never pick one up.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pymc_mfm.inference.result import DecompositionResult

#: Effect families of the rows, and the direct edge whose graph truth makes each live.
EFFECT_FAMILIES = ("channel", "control")
DIRECT_EDGES = {"channel": "cy", "control": "zb"}
#: Axes of :func:`stratified_summary`.
STRATA_AXES = (
    "true_size_bin",
    "predicted_size_bin",
    "edge_probability_bin",
    "active_channels",
    "live_channels",
)
_SIZE_AXES = STRATA_AXES[:2]
#: Axes over world-level counts; their strata run over every observed value.
_COUNT_AXES = STRATA_AXES[3:]
#: Quantiles of the nonzero absolute true window totals that :func:`size_cuts` returns.
SIZE_QUANTILES = (0.25, 0.5, 0.75)
#: Central interval of the window-total draws (NumPy linear quantiles, inclusive).
INTERVAL = (0.1, 0.9)
#: Edge decisions of :func:`misclassification_summary`, in ``threshold_table`` order.
DECISIONS = ("tp", "fp", "tn", "fn")
#: Values of every cell, computed over its scored effects (finite truth and draws).
CELL_METRICS = (
    "coverage",
    "crps",
    "signed_residual",
    "abs_residual",
    "residual_q10",
    "residual_median",
    "residual_q90",
)
#: Why a row has no place on an axis.
_MISSING = {
    "true_size_bin": "nonfinite_truth",
    "predicted_size_bin": "nonfinite_draws",
    "edge_probability_bin": "missing_edge_probability",
    "active_channels": "missing_active_channels",
    "live_channels": "missing_graph_truth",
}


@functools.cache
def _edge_bins() -> tuple[tuple[float, ...], tuple[tuple[str, float, float], ...]]:
    """Edge-frequency bin edges and ``(stratum, lower, upper)`` bins.

    The edges are ``0``, :data:`~pymc_mfm.training.metrics.THRESHOLDS` and ``1``;
    bins are left-closed (the last one closes at 1), so the effects called
    live at a threshold (``p >= t``) are the bins from ``t`` up. Read lazily:
    importing :mod:`pymc_mfm.training` loads the trainer.
    """
    from pymc_mfm.training.metrics import THRESHOLDS

    edges = (0.0, *(float(threshold) for threshold in THRESHOLDS), 1.0)
    bins = tuple(
        (f"[{lower:g}, {upper:g}{']' if upper == 1 else ')'}", lower, upper)
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    )
    return edges, bins


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def effect_rows(
    world: str | None,
    effects: Sequence[str],
    families: Sequence[str],
    draws,
    truth,
    *,
    live: Sequence[bool | None] | None = None,
    edge_draws=None,
) -> list[dict[str, Any]]:
    """Window-total rows of one world's active channel and control effects.

    ``draws`` is ``(S, T, E)`` and ``truth`` ``(T, E)``, both in ``sales_scale``
    units; ``effects`` names the ``E`` effects and ``families`` gives each one's
    :data:`EFFECT_FAMILIES` member. ``live`` is the graph truth of each effect's
    direct edge (``None`` for all or for one effect when unknown) and
    ``edge_draws`` ``(S, E)`` that edge's draws, a column of ``NaN`` when it has
    none. A total that is not finite gives ``None`` for its fields.
    """
    from pymc_mfm.training.metrics import empirical_crps

    draws = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    n = len(effects)
    if (
        draws.ndim != 3
        or draws.shape[0] < 1
        or draws.shape[2] != n
        or truth.shape != draws.shape[1:]
    ):
        raise ValueError(
            f"expected draws (S, T, {n}) and truth (T, {n}), got {draws.shape} and {truth.shape}"
        )
    families = [str(family) for family in families]
    if len(families) != n or set(families) - set(EFFECT_FAMILIES):
        raise ValueError(f"families must name one of {EFFECT_FAMILIES} per effect")
    labels = (
        [None] * n if live is None else [None if value is None else bool(value) for value in live]
    )
    if len(labels) != n:
        raise ValueError(f"live needs one label per effect ({n})")
    edges = None
    if edge_draws is not None:
        edges = np.asarray(edge_draws, dtype=np.float64)
        if edges.shape != (draws.shape[0], n):
            raise ValueError(f"edge_draws must have shape {(draws.shape[0], n)}, got {edges.shape}")
    channel = [family == "channel" for family in families]
    channel_labels = [label for label, flag in zip(labels, channel, strict=True) if flag]
    live_channels = None if None in channel_labels else sum(channel_labels)
    with np.errstate(over="ignore", invalid="ignore"):
        totals = draws.sum(axis=1)
        targets = truth.sum(axis=0)
    rows = []
    for e in range(n):
        target = _finite(targets[e])
        sample = totals[:, e]
        # Prediction fields are defined together: a row with truth and mean has all of them.
        prediction = dict.fromkeys(("mean", "low", "high", "crps"))
        if np.isfinite(sample).all():
            # Overflow from finite totals is caught below as a non-finite value.
            with np.errstate(over="ignore", invalid="ignore"):
                low, high = np.quantile(sample, INTERVAL)
                values = {"mean": sample.mean(), "low": low, "high": high}
                if target is not None:
                    values["crps"] = empirical_crps(sample, target)
            values = {key: _finite(value) for key, value in values.items()}
            if None not in values.values():
                prediction.update(values)
        edge = None if edges is None or not np.isfinite(edges[:, e]).all() else edges[:, e]
        rows.append(
            {
                "world": None if world is None else str(world),
                "effect": str(effects[e]),
                "family": families[e],
                "live": labels[e],
                "active_channels": sum(channel),
                "live_channels": live_channels,
                "truth": target,
                **prediction,
                "edge_probability": None if edge is None else float(edge.mean()),
            }
        )
    return rows


def _direct_slot(result: DecompositionResult, name: str, edge: str) -> int | None:
    """The unique active ``edge`` slot leaving component ``name``, or ``None``."""
    families = result.provenance.get("edge_families")
    endpoints = result.provenance.get("edge_endpoints")
    if families is None or endpoints is None:
        return None
    slots = [
        slot
        for slot, (family, pair) in enumerate(zip(families, endpoints, strict=True))
        if family == edge and pair[0] == name and result.edge_mask[slot]
    ]
    return slots[0] if len(slots) == 1 else None


def result_effect_rows(
    result: DecompositionResult, truth, graph_truth=None
) -> list[dict[str, Any]]:
    """:func:`effect_rows` of one result's active channels and controls.

    ``truth`` is natural-unit ``[time, component]`` aligned with
    ``result.components`` and ``graph_truth`` a binary ``[edge]`` vector. Both
    are divided into ``sales_scale`` units by the result's recorded
    ``sales_scale``, which is required. Edge frequencies come from
    ``result.graphs`` on each effect's unique active direct edge, found through
    the ``edge_families``/``edge_endpoints`` provenance; an effect without one
    has no edge frequency or live label. The world is the recorded
    ``world_id``, else the result's ``data_sha256``. The divisions round, so
    rebinning these rows with an evaluator's stored cuts can move an effect
    that sits exactly on a cut; the evaluator's own ``per_world`` rows rebin
    exactly.
    """
    scale = result.provenance.get("sales_scale")
    if scale is None:
        raise ValueError("stratified rows need the result's sales_scale provenance")
    families = np.asarray(result.component_families)
    keep = np.flatnonzero(
        np.asarray(result.component_mask, dtype=bool) & np.isin(families, EFFECT_FAMILIES)
    )
    names = [result.component_names[index] for index in keep]
    slots = [
        _direct_slot(result, name, DIRECT_EDGES[family])
        for name, family in zip(names, families[keep], strict=True)
    ]
    graphs = np.asarray(result.graphs, dtype=np.float64)
    edge_draws = np.full((graphs.shape[0], len(keep)), np.nan)
    for column, slot in enumerate(slots):
        if slot is not None:
            edge_draws[:, column] = graphs[:, slot]
    live = None
    if graph_truth is not None:
        graph_truth = np.asarray(graph_truth)
        live = [None if slot is None else bool(graph_truth[slot]) for slot in slots]
    world = result.provenance.get("world_id")
    if world is None:
        world = result.provenance.get("data_sha256")
    return effect_rows(
        world,
        names,
        families[keep].tolist(),
        np.asarray(result.components, dtype=np.float64)[..., keep] / scale,
        np.asarray(truth, dtype=np.float64)[:, keep] / scale,
        live=live,
        edge_draws=edge_draws,
    )


def size_cuts(rows: Iterable[Mapping[str, Any]], quantiles=SIZE_QUANTILES) -> list[float]:
    """Cut points of the size bins: ``quantiles`` of the nonzero absolute true totals.

    Empty when no row has a finite nonzero truth; the size axes then hold the
    ``zero`` bin and one bin above it.
    """
    sizes = [abs(row["truth"]) for row in rows if row["truth"] is not None and row["truth"] != 0]
    return [float(cut) for cut in np.quantile(sizes, quantiles)] if sizes else []


def _checked_cuts(cuts) -> list[float]:
    cuts = [float(cut) for cut in cuts]
    if any(not math.isfinite(cut) or cut <= 0 for cut in cuts) or cuts != sorted(cuts):
        raise ValueError("size cuts must be finite, positive and non-decreasing")
    return cuts


def _bins(by: str, cuts: Sequence[float], values: Sequence[int]):
    """``(stratum, lower, upper)`` of every stratum of ``by``, in order.

    ``values`` are the observed values of a count axis, whose strata run from
    the smallest to the largest.
    """
    if by in _SIZE_AXES:
        edges = [0.0, *cuts, None]
        return [("zero", 0.0, 0.0)] + [
            (f"q{index + 1}", edges[index], edges[index + 1]) for index in range(len(cuts) + 1)
        ]
    if by == "edge_probability_bin":
        return list(_edge_bins()[1])
    if not values:
        return []
    return [(str(value), value, value) for value in range(min(values), max(values) + 1)]


def _place(row: Mapping[str, Any], by: str, cuts: Sequence[float]) -> str | None:
    """The stratum of ``row`` on ``by``, or ``None`` when it has no place there."""
    if by in _SIZE_AXES:
        value = row["truth" if by == "true_size_bin" else "mean"]
        if value is None:
            return None
        size = abs(value)
        return "zero" if size == 0 else f"q{int(np.searchsorted(cuts, size, side='left')) + 1}"
    if by == "edge_probability_bin":
        value = row["edge_probability"]
        if value is None:
            return None
        edges, bins = _edge_bins()
        return bins[int(np.searchsorted(edges[1:-1], value, side="right"))][0]
    value = row[by]
    return None if value is None else str(value)


def row_strata(
    rows: Iterable[Mapping[str, Any]], by: str, *, cuts: Sequence[float]
) -> list[str | None]:
    """Each row's stratum on ``by`` (a cell's ``stratum``), ``None`` where it has none."""
    if by not in STRATA_AXES:
        raise ValueError(f"by must be one of {STRATA_AXES}, got {by!r}")
    cuts = _checked_cuts(cuts)
    return [_place(row, by, cuts) for row in rows]


#: Row fields that must all be present for a row to be scored.
_SCORED_FIELDS = ("truth", "mean", "low", "high", "crps")


def scored(row: Mapping[str, Any]) -> bool:
    """Whether a row has finite truth and prediction fields, so it enters cell values."""
    return all(row[key] is not None for key in _SCORED_FIELDS)


def _cell(rows: Sequence[Mapping[str, Any]], empty: str) -> dict[str, Any]:
    """Counts and :data:`CELL_METRICS` of one stratum's rows; ``empty`` is why none are.

    ``n_worlds`` is ``None`` when a row names no world, as distinct worlds cannot be told apart.
    """
    scored_rows = [row for row in rows if scored(row)]
    labels = [row["live"] for row in rows]
    worlds = [row["world"] for row in rows]
    cell: dict[str, Any] = {
        "status": "ok" if scored_rows else "unavailable",
        "reason": None,
        "n_worlds": None if None in worlds else len(set(worlds)),
        "n_effects": len(rows),
        "n_scored": len(scored_rows),
        "n_live": None if None in labels else sum(labels),
        **dict.fromkeys(CELL_METRICS),
    }
    if not rows:
        cell["reason"] = empty
    elif len(scored_rows) < len(rows):
        cell["reason"] = f"nonfinite_in_{len(rows) - len(scored_rows)}_effects"
    if scored_rows:
        truth = np.asarray([row["truth"] for row in scored_rows])
        residual = np.asarray([row["mean"] for row in scored_rows]) - truth
        covered = [row["low"] <= row["truth"] <= row["high"] for row in scored_rows]
        q10, median, q90 = np.quantile(residual, (0.1, 0.5, 0.9))
        values = {
            "coverage": np.mean(covered),
            "crps": np.mean([row["crps"] for row in scored_rows]),
            "signed_residual": residual.mean(),
            "abs_residual": np.abs(residual).mean(),
            "residual_q10": q10,
            "residual_median": median,
            "residual_q90": q90,
        }
        cell.update({key: _finite(value) for key, value in values.items()})
    return cell


def _excluded(counts: Mapping[str, int]) -> list[dict[str, Any]]:
    return [{"reason": reason, "n_effects": n} for reason, n in counts.items() if n]


def stratified_summary(
    rows: Iterable[Mapping[str, Any]], by: str, *, cuts: Sequence[float]
) -> dict[str, dict[str, list]]:
    """Cells of ``rows`` on one axis ``by``, per family.

    ``{family: {"cells": [...], "excluded": [...]}}``. Each cell has its
    ``stratum``, ``lower``/``upper`` bounds (``upper`` is ``None`` above the last
    size cut), ``status`` and ``reason``, the counts ``n_worlds``,
    ``n_effects``, ``n_scored`` and ``n_live``, and :data:`CELL_METRICS` over
    the :func:`scored` rows; ``n_worlds`` is ``None`` when a row names no world.
    Size and edge-frequency axes list every bin; count axes every value from
    the smallest to the largest observed in any family. A stratum without
    effects is ``unavailable`` (``no_worlds_in_stratum`` when no world of any
    family falls there, else ``no_effects_in_stratum``) with ``None`` metrics.
    ``excluded`` counts the rows the axis cannot place, with the reason.
    """
    rows = list(rows)
    cuts = _checked_cuts(cuts)
    places = row_strata(rows, by, cuts=cuts)
    observed = {place for place in places if place is not None}
    bins = _bins(by, cuts, sorted(int(place) for place in observed) if by in _COUNT_AXES else ())
    out: dict[str, dict[str, list]] = {}
    for family in EFFECT_FAMILIES:
        groups: dict[str, list] = {stratum: [] for stratum, _, _ in bins}
        missing = 0
        for row, place in zip(rows, places, strict=True):
            if row["family"] != family:
                continue
            if place is None:
                missing += 1
            else:
                groups[place].append(row)
        cells = []
        for stratum, lower, upper in bins:
            empty = (
                "no_worlds_in_stratum"
                if by in _COUNT_AXES and stratum not in observed
                else "no_effects_in_stratum"
            )
            cells.append(
                {
                    "stratum": stratum,
                    "lower": lower,
                    "upper": upper,
                    **_cell(groups[stratum], empty),
                }
            )
        out[family] = {"cells": cells, "excluded": _excluded({_MISSING[by]: missing})}
    return out


def misclassification_summary(
    rows: Iterable[Mapping[str, Any]], thresholds: Sequence[float] | None = None
) -> dict[str, dict[str, list]]:
    """Edge decisions joined with the size error, per family and threshold.

    An effect is called live when its direct-edge frequency is at least the
    threshold, the rule of :func:`~pymc_mfm.training.metrics.threshold_table`;
    with its graph-truth label that makes it a ``tp``, ``fp``, ``tn`` or ``fn``.
    ``{family: {"thresholds": [{"threshold", "cells"}], "excluded": [...]}}``:
    each decision's cell has the counts and :data:`CELL_METRICS` of
    :func:`stratified_summary` (its ``n_effects`` is the confusion count), and
    ``excluded`` counts effects without an edge frequency or a label.
    ``thresholds`` default to :data:`~pymc_mfm.training.metrics.THRESHOLDS`.
    """
    rows = list(rows)
    thresholds = _edge_bins()[0][1:-1] if thresholds is None else thresholds
    out: dict[str, dict[str, list]] = {}
    for family in EFFECT_FAMILIES:
        members = [row for row in rows if row["family"] == family]
        decided = [
            row
            for row in members
            if row["edge_probability"] is not None and row["live"] is not None
        ]
        levels = []
        for threshold in thresholds:
            groups: dict[str, list] = {decision: [] for decision in DECISIONS}
            for row in decided:
                called = row["edge_probability"] >= threshold
                if row["live"]:
                    groups["tp" if called else "fn"].append(row)
                else:
                    groups["fp" if called else "tn"].append(row)
            levels.append(
                {
                    "threshold": float(threshold),
                    "cells": [
                        {
                            "decision": decision,
                            **_cell(groups[decision], "no_effects_with_decision"),
                        }
                        for decision in DECISIONS
                    ],
                }
            )
        excluded = _excluded(
            {
                "missing_edge_probability": sum(row["edge_probability"] is None for row in members),
                "missing_graph_truth": sum(
                    row["edge_probability"] is not None and row["live"] is None for row in members
                ),
            }
        )
        out[family] = {"thresholds": levels, "excluded": excluded}
    return out


def strata_summary(
    rows: Iterable[Mapping[str, Any]], *, cuts: Sequence[float] | None = None
) -> dict[str, Any]:
    """Every :data:`STRATA_AXES` summary and the misclassification join of ``rows``.

    ``cuts`` defaults to :func:`size_cuts` of the rows; pass a stored
    summary's ``cuts`` to bin other rows alike. The result records the
    ``cuts``, edge-frequency bin edges and thresholds it used, under ``by`` the
    :func:`stratified_summary` of each axis and under ``misclassification`` the
    :func:`misclassification_summary`.
    """
    rows = list(rows)
    cuts = size_cuts(rows) if cuts is None else _checked_cuts(cuts)
    edges = _edge_bins()[0]
    return {
        "unit": "sales_scale",
        "size": "absolute window total: exactly zero, then right-closed bins between cuts",
        "interval": "central 80%: linear 0.1 and 0.9 quantiles of the window-total draws",
        "cuts": cuts,
        "edge_probability_edges": list(edges),
        "thresholds": list(edges[1:-1]),
        "by": {axis: stratified_summary(rows, axis, cuts=cuts) for axis in STRATA_AXES},
        "misclassification": misclassification_summary(rows),
    }
