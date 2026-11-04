"""In-loop evaluation for decomposition, structure, and graph readouts.

``evaluate_rows`` scores schema-4 corpus rows in normalized target space and
returns one JSON-safe record with separately labeled readouts:

* ``joint``: joint router marginals for query and context windows;
* ``conditional``: graph-conditional decoder draws;
* ``structure_head``: auxiliary edge probabilities over active slots;
* ``joint_router``: edge marginals and whole-graph statistics from joint draws.

Decomposition and graph scores are computed per world (slot families average
the world's slots) and then averaged with equal world weights; edge scores
pool the active slots of all worlds. Unavailable or non-finite values are
``None`` and named under ``undefined`` with a reason rather than coerced into
valid-looking metrics. :data:`METRIC_BOUNDS` states every metric's valid range,
:data:`METRIC_UNITS` its unit and :func:`selection_metrics` the keys that
select checkpoints.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from torch import Tensor

from pymc_mfm.data.observations import observation_identity
from pymc_mfm.model.graph import (
    SQUARE_EDGE_TYPES,
    STRUCT_EDGE_KEYS,
    EdgeSlotLayout,
    binary_graphs_from_flow,
    structure_active_masks,
)
from pymc_mfm.model.model import MFM
from pymc_mfm.training import metrics
from pymc_mfm.training.stream import batches_for_rows

if TYPE_CHECKING:
    from pymc_mfm.inference.result import DecompositionResult

ResultSink = Callable[["DecompositionResult", np.ndarray, np.ndarray, np.ndarray], None]

HEAD_LABELS = {
    "joint": "joint router marginal; not a conditional-decoder quality claim",
    "conditional": "graph-conditional decoder, model_average mode",
    "structure_head": "auxiliary BCE structure head; independent per-edge probabilities",
    "joint_router": (
        "edge marginals and whole-graph statistics recovered from joint-head graph draws"
    ),
}

DECOMPOSITION_HEADS = ("joint", "conditional")
WINDOWS = ("query", "context")
INDIRECT_SOURCES = ("cc", "zc", "dc")
_FAMILIES = (
    "baseline_intrinsic",
    "contribution",
    "control_contribution",
    "confounder_total",
    "indirect_cc",
    "indirect_zc",
    "indirect_dc",
    "indirect_total",
)
_FAMILY_STATS = ("scaled_mae", "crps", "cov80", "cov95")
_ENERGY_METRICS = (
    "decomposition_energy_score",
    "latent_sales_temporal_energy_score",
    "contribution_temporal_energy_score",
    "contribution_cumulative_energy_score",
)
#: Effect families scored live/dead: media (live by ``cy``) and controls (by ``zb``).
_ATTRIBUTION_EFFECTS = ("media", "control")
_ATTRIBUTION_METRICS = tuple(
    f"attribution_live_{effect}_{key}"
    for effect in _ATTRIBUTION_EFFECTS
    for key in metrics.LIVE_ATTRIBUTION_KEYS
) + tuple(
    f"attribution_dead_{effect}_{key}"
    for effect in _ATTRIBUTION_EFFECTS
    for key in metrics.DEAD_ATTRIBUTION_KEYS
)
_CONDITIONAL_COVERAGE_METRICS = tuple(
    f"attribution_live_{effect}_conditional_{level}"
    for effect in _ATTRIBUTION_EFFECTS
    for level in metrics.COVERAGE_BOUNDS
)
_TEMPORAL_METRICS = tuple(f"contribution_{key}" for key in metrics.TEMPORAL_KEYS)
_TIME_AVERAGE_METRICS = tuple(f"timeavg_contribution_{key}" for key in metrics.TIME_AVERAGE_KEYS)
_CONTROL_SIMULTANEOUS_METRIC = "timeavg_control_contribution_simcov80_live"
DECOMPOSITION_METRICS: tuple[str, ...] = (
    *(f"{family}_{stat}" for family in _FAMILIES for stat in _FAMILY_STATS),
    "contribution_padding_leakage_scaled_mae",
    *metrics.LATENT_SALES_KEYS,
    *_ENERGY_METRICS,
    *_ATTRIBUTION_METRICS,
    *_CONDITIONAL_COVERAGE_METRICS,
    *_TEMPORAL_METRICS,
    *_TIME_AVERAGE_METRICS,
    _CONTROL_SIMULTANEOUS_METRIC,
)
_EDGE_SCORES = (
    "log_score",
    "brier",
    "auroc",
    "auprc",
    "ece",
    *metrics.BRIER_DECOMPOSITION_KEYS,
    "null_fpr",
    "prevalence",
)
EDGE_METRICS = (*_EDGE_SCORES, "n", "n_pos")
GRAPH_METRICS = (
    *metrics.GRAPH_KEYS,
    *(f"{et}_{rate}" for et in SQUARE_EDGE_TYPES for rate in ("two_cycle_rate", "cycle_rate")),
)
#: Per-world means of active channels and controls, and the number of scored worlds.
COUNT_METRICS = ("active_channels", "active_controls", "worlds")
_WORLD_COUNTS = COUNT_METRICS[:2]

#: Decomposition metrics that average slots within a world; ``per_world``
#: weights count those slots, so ``sum(value * weight) / sum(weight)`` over
#: worlds pools the slots of every world.
SLOT_METRICS = frozenset(
    {
        *(
            f"{family}_{stat}"
            for family in ("contribution", "control_contribution")
            for stat in _FAMILY_STATS
        ),
        "contribution_padding_leakage_scaled_mae",
        *_CONDITIONAL_COVERAGE_METRICS,
        *_TEMPORAL_METRICS,
        *(name for name in _TIME_AVERAGE_METRICS if not name.endswith("_simcov80_live")),
    }
)

_NONNEGATIVE = (0.0, math.inf)
_PROBABILITY = (0.0, 1.0)
_CORRELATION = (-1.0, 1.0)

#: Valid ``(lower, upper)`` range of every metric, keyed by its exact name (the
#: last component of a flat key). No other names are valid.
METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    **{
        f"{family}_{stat}": _PROBABILITY if stat.startswith("cov") else _NONNEGATIVE
        for family in _FAMILIES
        for stat in _FAMILY_STATS
    },
    "contribution_padding_leakage_scaled_mae": _NONNEGATIVE,
    "closure_rel_err": _NONNEGATIVE,
    "closure_draw_rel_err": _NONNEGATIVE,
    "closure_draw_q90_rel_err": _NONNEGATIVE,
    "latent_sales_r2": (-math.inf, 1.0),
    **dict.fromkeys(_ENERGY_METRICS, _NONNEGATIVE),
    **{
        name: (-math.inf, math.inf) if name.endswith("_mean_bias") else _NONNEGATIVE
        for name in _ATTRIBUTION_METRICS
    },
    **dict.fromkeys(_CONDITIONAL_COVERAGE_METRICS, _PROBABILITY),
    "contribution_truth_lag1": _CORRELATION,
    "contribution_draw_lag1": _CORRELATION,
    "contribution_lag1_tracking_abs_error": (0.0, 2.0),
    "contribution_truth_error_residual_lag1": _CORRELATION,
    "contribution_posterior_centered_residual_lag1": _CORRELATION,
    "contribution_adjacent_draw_correlation": _CORRELATION,
    **dict.fromkeys(_TIME_AVERAGE_METRICS, _PROBABILITY),
    _CONTROL_SIMULTANEOUS_METRIC: _PROBABILITY,
    "log_score": _NONNEGATIVE,
    "brier": _PROBABILITY,
    "auroc": _PROBABILITY,
    "auprc": _PROBABILITY,
    "ece": _PROBABILITY,
    # CORP Brier components: mcb <= brier, and dsc <= unc = r (1 - r) <= 1/4.
    "mcb": _PROBABILITY,
    "dsc": (0.0, 0.25),
    "unc": (0.0, 0.25),
    "null_fpr": _PROBABILITY,
    "prevalence": _PROBABILITY,
    "n": _NONNEGATIVE,
    "n_pos": _NONNEGATIVE,
    "shd_mean": _NONNEGATIVE,
    "exact_recovery": _PROBABILITY,
    "shd_median": _NONNEGATIVE,
    "normalized_shd_mean": _PROBABILITY,
    "energy_score": _NONNEGATIVE,
    "categorical_brier_score": (0.0, 2.0),
    "entropy": _NONNEGATIVE,
    "n_distinct": (1.0, math.inf),
    "pairwise_comoment_abs_err": (0.0, 0.25),
    "pairwise_comoment_max_abs_err": (0.0, 0.25),
    **{
        f"{et}_{rate}": _PROBABILITY
        for et in SQUARE_EDGE_TYPES
        for rate in ("two_cycle_rate", "cycle_rate")
    },
    "active_channels": _NONNEGATIVE,
    "active_controls": _NONNEGATIVE,
    "worlds": (1.0, math.inf),
}
#: Round-off allowed beyond every bound: U-statistic energy scores, the
#: sorted-sample CRPS and the CORP Brier components (differences of mean
#: scores) can dip just below zero in floating point.
BOUNDS_SLACK = 1e-12
#: Unit of every metric, keyed like :data:`METRIC_BOUNDS`, as its scorer computes
#: it: ``sales_scale`` for errors and scores of z-space draws (natural units over
#: the world's ``sales_scale``), ``nats`` for natural-log losses and entropy,
#: ``count`` for counts and edge distances, and ``dimensionless`` for
#: probabilities, rates, coverage, correlations, relative errors and R².
METRIC_UNITS: dict[str, str] = {
    **{
        f"{family}_{stat}": "dimensionless" if stat.startswith("cov") else "sales_scale"
        for family in _FAMILIES
        for stat in _FAMILY_STATS
    },
    "contribution_padding_leakage_scaled_mae": "sales_scale",
    **dict.fromkeys(metrics.LATENT_SALES_KEYS, "dimensionless"),
    **dict.fromkeys(_ENERGY_METRICS, "sales_scale"),
    **dict.fromkeys(_ATTRIBUTION_METRICS, "sales_scale"),
    **dict.fromkeys(_CONDITIONAL_COVERAGE_METRICS, "dimensionless"),
    **dict.fromkeys(_TEMPORAL_METRICS, "dimensionless"),
    **dict.fromkeys(_TIME_AVERAGE_METRICS, "dimensionless"),
    _CONTROL_SIMULTANEOUS_METRIC: "dimensionless",
    "log_score": "nats",
    **dict.fromkeys(
        (
            "brier",
            "auroc",
            "auprc",
            "ece",
            *metrics.BRIER_DECOMPOSITION_KEYS,
            "null_fpr",
            "prevalence",
        ),
        "dimensionless",
    ),
    **dict.fromkeys(("n", "n_pos", "shd_mean", "shd_median", "n_distinct"), "count"),
    **dict.fromkeys(
        (
            "exact_recovery",
            "normalized_shd_mean",
            "energy_score",
            "categorical_brier_score",
            "pairwise_comoment_abs_err",
            "pairwise_comoment_max_abs_err",
        ),
        "dimensionless",
    ),
    "entropy": "nats",
    **{
        f"{et}_{rate}": "dimensionless"
        for et in SQUARE_EDGE_TYPES
        for rate in ("two_cycle_rate", "cycle_rate")
    },
    **dict.fromkeys(COUNT_METRICS, "count"),
}


def metric_bounds(key: str) -> tuple[float, float]:
    """``(lower, upper)`` for a flat key (with or without its leg) or a metric name."""
    return METRIC_BOUNDS[key.rsplit("/", 1)[-1]]


def metric_units(key: str) -> str:
    """:data:`METRIC_UNITS` unit of a flat key (with or without its leg) or a metric name.

    A metric this release does not define raises ``ValueError`` naming the key.
    """
    name = key.rsplit("/", 1)[-1]
    if name not in METRIC_UNITS:
        raise ValueError(f"{key!r}: no unit for {name!r} in this release")
    return METRIC_UNITS[name]


def within_bounds(key: str, value: float) -> bool:
    """Whether ``value`` lies in :func:`metric_bounds` of ``key`` up to :data:`BOUNDS_SLACK`."""
    lower, upper = metric_bounds(key)
    return lower - BOUNDS_SLACK <= value <= upper + BOUNDS_SLACK


class SelectionMetrics(NamedTuple):
    """Held-out flat keys that select checkpoints for one selection head."""

    monitor: str
    context_gate: str
    structure_gate: str
    leakage_gate: str


def selection_metrics(head: str) -> SelectionMetrics:
    """Checkpoint-selection keys for ``TrainingConfig.selection_head``.

    The minimized monitor is the head's held-out query decomposition energy
    score; the gates are the same head's context energy score, the auxiliary
    ``cy`` log score and the head's query padded-channel leakage.
    """
    if head not in DECOMPOSITION_HEADS:
        raise ValueError(f"selection head must be one of {DECOMPOSITION_HEADS}, got {head!r}")
    return SelectionMetrics(
        monitor=f"holdout/{head}/query/decomposition_energy_score",
        context_gate=f"holdout/{head}/context/decomposition_energy_score",
        structure_gate="holdout/structure_head/cy/log_score",
        leakage_gate=f"holdout/{head}/query/contribution_padding_leakage_scaled_mae",
    )


def _selected_heads(heads: Sequence[str]) -> tuple[str, ...]:
    """Validated decomposition heads in canonical (sampling) order."""
    if isinstance(heads, str):
        raise ValueError(f"heads must be a sequence of head names, got {heads!r}")
    chosen = tuple(heads)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(DECOMPOSITION_HEADS):
        raise ValueError(f"heads must be distinct names from {DECOMPOSITION_HEADS}, got {chosen}")
    return tuple(head for head in DECOMPOSITION_HEADS if head in chosen)


def metric_aggregation(heads: Sequence[str] = DECOMPOSITION_HEADS) -> dict[str, str]:
    """Every flat key without its leg, mapped to how worlds combine into its value.

    ``"world"``: equal-world mean of per-world values. ``"slot"``: equal-world
    mean of per-world slot means (see :data:`SLOT_METRICS`). ``"pooled_edges"``:
    one score over the active edge slots of all worlds. ``heads`` selects the
    decomposition heads as in :func:`evaluate_rows`.
    """
    out: dict[str, str] = {}
    for head in _selected_heads(heads):
        for window in WINDOWS:
            for name in DECOMPOSITION_METRICS:
                out[f"{head}/{window}/{name}"] = "slot" if name in SLOT_METRICS else "world"
    for prefix in ("structure_head", "joint_router/edge"):
        for et in STRUCT_EDGE_KEYS:
            for name in EDGE_METRICS:
                out[f"{prefix}/{et}/{name}"] = "pooled_edges"
    for name in GRAPH_METRICS:
        out[f"joint_router/graph/{name}"] = "world"
    for name in COUNT_METRICS:
        out[f"counts/{name}"] = "world"
    return out


@contextmanager
def isolated_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Seed the CPU generator and evaluation device, then restore both.

    This isolates evaluation draws from the caller's CPU, CUDA, or MPS random
    state, so changing evaluation settings cannot change training draws.
    """
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            elif device.type == "mps":
                torch.mps.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class _MeanAccumulator:
    """Equal-world means of one key family, with the worlds and reasons behind them."""

    def __init__(self, prefix: str, keys: Sequence[str]):
        self.prefix = prefix
        self.keys = tuple(keys)
        self._finite: dict[str, list[float]] = {key: [] for key in self.keys}
        self._nonfinite: dict[str, int] = dict.fromkeys(self.keys, 0)
        self._reasons: dict[str, str] = {}

    def add(self, world: metrics.WorldScores, entry: dict[str, Any] | None = None) -> None:
        """Take one world's scores; ``entry`` collects them for ``per_world``."""
        scored = world.values.keys() | world.undefined.keys()
        if scored != set(self.keys):
            raise ValueError(f"world scores disagree on {sorted(scored ^ set(self.keys))}")
        for key in self.keys:
            if key in world.values:
                self._finite[key].append(world.values[key])
            elif world.undefined[key] == metrics.NONFINITE:
                self._nonfinite[key] += 1
            else:
                self._reasons.setdefault(key, world.undefined[key])
        if entry is not None:
            for key in self.keys:
                name = f"{self.prefix}/{key}"
                entry["values"][name] = world.values.get(key)
                entry["weights"][name] = world.weights.get(key, 0)
                if key in world.undefined:
                    entry["undefined"][name] = world.undefined[key]

    def finalize(
        self, undefined: dict[str, str], metric_worlds: dict[str, int]
    ) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for key in self.keys:
            name = f"{self.prefix}/{key}"
            finite = self._finite[key]
            dropped = self._nonfinite[key]
            out[key] = float(np.mean(finite)) if finite else None
            metric_worlds[name] = len(finite)
            if dropped:
                undefined[name] = f"nonfinite_in_{dropped}_worlds"
            elif not finite:
                undefined[name] = self._reasons.get(key, "no_scored_worlds")
        return out


class _EdgePool:
    """Active-slot (probability, truth) pairs pooled over worlds per edge type."""

    def __init__(self):
        self._probs: dict[str, list[np.ndarray]] = {et: [] for et in STRUCT_EDGE_KEYS}
        self._truth: dict[str, list[np.ndarray]] = {et: [] for et in STRUCT_EDGE_KEYS}

    def add(self, et: str, probs: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> None:
        self._probs[et].append(probs[keep])
        self._truth[et].append(truth[keep])

    def finalize(
        self, prefix: str, undefined: dict[str, str]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list]]]:
        """Scores and reliability/CORP/threshold tables per edge type.

        A one-class edge type keeps its reliability and threshold tables but
        has no CORP table, like its undefined CORP scores.
        """
        records: dict[str, dict[str, Any]] = {}
        tables: dict[str, dict[str, list]] = {}
        for et in STRUCT_EDGE_KEYS:
            p = np.concatenate(self._probs[et]) if self._probs[et] else np.empty(0)
            y = np.concatenate(self._truth[et]) if self._truth[et] else np.empty(0)
            records[et] = _edge_record(p, y, f"{prefix}/{et}", undefined)
            scoreable = y.size > 0 and np.isfinite(p).all()
            two_class = scoreable and 0 < records[et]["n_pos"] < y.size
            tables[et] = {
                "reliability": metrics.reliability_table(p, y) if scoreable else [],
                "corp": metrics.isotonic_calibration(p, y) if two_class else [],
                "thresholds": metrics.threshold_table(p, y) if scoreable else [],
            }
        return records, tables


def _edge_record(
    p: np.ndarray, y: np.ndarray, prefix: str, undefined: dict[str, str]
) -> dict[str, Any]:
    n = int(y.size)
    n_pos = int(y.sum()) if n else 0
    if n == 0:
        for key in _EDGE_SCORES:
            undefined[f"{prefix}/{key}"] = "no_active_slots"
        return {**dict.fromkeys(_EDGE_SCORES), "n": 0, "n_pos": 0}
    # A one-class family evidences no calibration: its CORP components are
    # undefined (``ece`` keeps its historical, recorded definition).
    two_class = 0 < n_pos < n
    # ``unc = r (1 - r)`` summarizes truth alone, in brier_decomposition's form.
    unc = (n_pos * (n - n_pos) / n / n, None) if two_class else (None, "one_class")
    invalid = int((~np.isfinite(p)).sum())
    if invalid:
        for key in _EDGE_SCORES:
            if key not in ("prevalence", "unc"):
                undefined[f"{prefix}/{key}"] = f"nonfinite_in_{invalid}_slots"
        if unc[1] is not None:
            undefined[f"{prefix}/unc"] = unc[1]
        # Truth summaries remain defined even when predictions cannot be scored.
        return {
            **dict.fromkeys(_EDGE_SCORES),
            "prevalence": float(y.mean()),
            "unc": unc[0],
            "n": n,
            "n_pos": n_pos,
        }
    auroc = metrics.auroc(p, y)
    scored = {
        "auroc": (auroc, None if auroc is not None else "one_class"),
        "auprc": metrics.auprc(p, y),
        "null_fpr": metrics.null_fpr(p, y),
    }
    if two_class:
        split = metrics.brier_decomposition(p, y)
        scored.update({"mcb": (split["mcb"], None), "dsc": (split["dsc"], None), "unc": unc})
    else:
        scored.update(dict.fromkeys(metrics.BRIER_DECOMPOSITION_KEYS, (None, "one_class")))
    for key, (_, reason) in scored.items():
        if reason is not None:
            undefined[f"{prefix}/{key}"] = reason
    record = {
        "log_score": metrics.log_score(p, y),
        "brier": metrics.brier(p, y),
        **{key: value for key, (value, _) in scored.items()},
        "ece": metrics.ece(p, y),
        "prevalence": float(y.mean()),
        "n": n,
        "n_pos": n_pos,
    }
    return {key: record[key] for key in EDGE_METRICS}


def _numpy(value: Tensor) -> np.ndarray:
    # One combined device-and-dtype copy returns zeros for MPS tensors; copy to
    # the CPU first and convert there.
    return value.detach().cpu().to(torch.float64).numpy()


def _finite(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(array).all() for array in arrays)


def _score_decomposition(
    pred: np.ndarray,
    truth: np.ndarray,
    active_c: np.ndarray,
    active_m: np.ndarray,
    live_c: np.ndarray,
    live_m: np.ndarray,
    outcome: np.ndarray,
    noise: np.ndarray,
    scale: float,
    *,
    graph_c: np.ndarray | None = None,
    graph_m: np.ndarray | None = None,
) -> metrics.WorldScores:
    """One world's z-space decomposition metrics (:data:`DECOMPOSITION_METRICS`).

    ``pred`` is ``[S, T, D]``, ``truth`` ``[T, D]`` with
    ``D = 1 + K + M + 4 = [baseline | K direct | M covariate | confounder_total
    | indirect cc, zc, dc]``; ``live_c``/``live_m`` are the ``cy``/``zb`` graph
    truth that makes a present channel/control live. ``graph_c``/``graph_m``
    are the matching per-draw direct edges ``[S, K]``/``[S, M]``, if available;
    ``outcome``/``noise`` are natural ``[T]`` series. A family whose inputs are
    not all finite is ``nonfinite`` for every key.
    """
    n_samples = pred.shape[0]
    k, m = active_c.size, active_m.size
    if pred.shape[1:] != truth.shape or truth.shape[-1] != 1 + k + m + 4:
        raise ValueError(
            f"decomposition draws {pred.shape} must match truth {truth.shape} with "
            f"D = 1 + {k} + {m} + 4"
        )
    contrib = slice(1, 1 + k)
    controls = slice(1 + k, 1 + k + m)
    confounder = 1 + k + m
    indirect = slice(confounder + 1, confounder + 4)
    n_channels, n_controls = int(active_c.sum()), int(active_m.sum())
    world = metrics.WorldScores()

    def skip(names: Sequence[str], reason: str) -> None:
        for name in names:
            world.skip(name, reason)

    def family(name: str, p: np.ndarray, t: np.ndarray, weight: int = 1) -> None:
        names = [f"{name}_{stat}" for stat in _FAMILY_STATS]
        if not _finite(p, t):
            skip(names, metrics.NONFINITE)
            return
        world.put(f"{name}_scaled_mae", np.mean(np.abs(p.mean(axis=0) - t)), weight)
        world.put(f"{name}_crps", metrics.empirical_crps(p, t), weight)
        for stat, (low, high) in metrics.COVERAGE_BOUNDS.items():
            world.put(f"{name}_{stat}", metrics.coverage(p, t, low, high), weight)

    pred_contrib = pred[..., contrib][..., active_c]
    true_contrib = truth[:, contrib][:, active_c]
    pred_controls = pred[..., controls][..., active_m]
    true_controls = truth[:, controls][:, active_m]
    pred_indirect_total = pred[..., indirect].sum(axis=-1)
    true_indirect_total = truth[:, indirect].sum(axis=-1)

    family("baseline_intrinsic", pred[..., 0], truth[:, 0])
    if n_channels:
        family("contribution", pred_contrib, true_contrib, n_channels)
    else:
        skip([f"contribution_{stat}" for stat in _FAMILY_STATS], "no_active_channels")
    if n_controls:
        family("control_contribution", pred_controls, true_controls, n_controls)
    else:
        skip([f"control_contribution_{stat}" for stat in _FAMILY_STATS], "no_active_controls")
    family("confounder_total", pred[..., confounder], truth[:, confounder])
    for i, source in enumerate(INDIRECT_SOURCES):
        column = indirect.start + i
        family(f"indirect_{source}", pred[..., column], truth[:, column])
    family("indirect_total", pred_indirect_total, true_indirect_total)

    inactive_c = ~active_c
    leakage = "contribution_padding_leakage_scaled_mae"
    pred_padded = pred[..., contrib][..., inactive_c]
    true_padded = truth[:, contrib][:, inactive_c]
    if not inactive_c.any():
        world.skip(leakage, "no_inactive_channels")
    elif not _finite(pred_padded, true_padded):
        world.skip(leakage, metrics.NONFINITE)
    else:
        world.put(
            leakage,
            np.mean(np.abs(pred_padded.mean(axis=0) - true_padded)),
            int(inactive_c.sum()),
        )

    # Closure of the decomposition against STRUCTURAL outcome (observed net of
    # iid noise), with the observed magnitude as denominator.
    component_draws = np.concatenate(
        [
            pred[..., :1],
            pred_contrib,
            pred_controls,
            pred[..., confounder : confounder + 1],
            pred_indirect_total[..., None],
        ],
        axis=-1,
    )
    component_truth = np.concatenate(
        [
            truth[:, :1],
            true_contrib,
            true_controls,
            truth[:, confounder : confounder + 1],
            true_indirect_total[:, None],
        ],
        axis=-1,
    )
    draw_sales = component_draws.sum(axis=-1)
    # Keep dimensionless diagnostics in context units: raw-unit epsilon floors
    # and sums of squares would make checkpoint selection depend on sales units.
    observed_sales = outcome / scale
    world.update(
        metrics.latent_sales_scores(draw_sales, observed_sales - noise / scale, observed_sales)
    )

    def energy(name: str, samples: np.ndarray, target: np.ndarray, cumulative: bool) -> None:
        if not _finite(samples, target):
            world.skip(name, metrics.NONFINITE)
        elif n_samples < 2:
            world.skip(name, "needs_n_samples_ge_2")
        elif cumulative:
            world.score(name, metrics.cumulative_energy_score(samples, target))
        else:
            world.put(name, metrics.energy_score(samples, target))

    energy("decomposition_energy_score", component_draws, component_truth, False)
    # Latent (structural) sales in z-space: the summed component truth equals
    # (outcome - noise) / scale by the corpus identity, without float32 round-off.
    energy("latent_sales_temporal_energy_score", draw_sales, component_truth.sum(axis=-1), False)
    if n_channels:
        energy("contribution_temporal_energy_score", pred_contrib, true_contrib, False)
        energy("contribution_cumulative_energy_score", pred_contrib, true_contrib, True)
    else:
        skip(_ENERGY_METRICS[2:], "no_active_channels")

    live_channels = live_c[active_c] != 0
    for effect, samples, target, labels in (
        ("media", pred_contrib, true_contrib, live_channels),
        ("control", pred_controls, true_controls, live_m[active_m] != 0),
    ):
        world.update(
            metrics.attribution_scores(
                samples,
                target,
                labels,
                live_prefix=f"attribution_live_{effect}_",
                dead_prefix=f"attribution_dead_{effect}_",
            )
        )

    levels = metrics.COVERAGE_BOUNDS

    def conditional_coverage(
        effect: str,
        samples: np.ndarray,
        target: np.ndarray,
        live: np.ndarray,
        graph: np.ndarray | None,
    ) -> None:
        keys = {level: f"attribution_live_{effect}_conditional_{level}" for level in levels}
        if not live.any():
            skip(keys.values(), "no_live_effects")
            return
        if graph is None:
            skip(keys.values(), "missing_graph_draws")
            return
        if graph.shape != (n_samples, live.size):
            raise ValueError(f"{effect} graph draws must have shape {(n_samples, live.size)}")
        if not _finite(graph[:, live]):
            skip(keys.values(), metrics.NONFINITE)
            return
        covered: dict[str, list[bool]] = {level: [] for level in levels}
        for slot in np.flatnonzero(live):
            edge_on = graph[:, slot] == 1
            if edge_on.sum() < 2:
                continue
            total_draws = samples[edge_on, :, slot].sum(axis=1)
            total_truth = target[:, slot].sum()
            if not _finite(total_draws, np.asarray(total_truth)):
                skip(keys.values(), metrics.NONFINITE)
                return
            for level, (low_q, high_q) in levels.items():
                low, high = np.quantile(total_draws, (low_q, high_q))
                covered[level].append(low <= total_truth <= high)
        for level, key in keys.items():
            if covered[level]:
                world.put(key, np.mean(covered[level]), len(covered[level]))
            else:
                world.skip(key, "needs_edge_on_draws_ge_2")

    conditional_coverage(
        "media",
        pred_contrib,
        true_contrib,
        live_channels,
        None if graph_c is None else graph_c[:, active_c],
    )
    conditional_coverage(
        "control",
        pred_controls,
        true_controls,
        live_m[active_m] != 0,
        None if graph_m is None else graph_m[:, active_m],
    )

    live_controls = live_m[active_m] != 0
    world.score(
        _CONTROL_SIMULTANEOUS_METRIC,
        metrics.simultaneous_coverage(
            pred_controls[..., live_controls], true_controls[:, live_controls]
        ),
    )

    if n_channels:
        world.update(metrics.temporal_scores(pred_contrib, true_contrib, prefix="contribution_"))
        world.update(
            metrics.time_average_scores(
                pred_contrib, true_contrib, live_channels, prefix="timeavg_contribution_"
            )
        )
    else:
        skip((*_TEMPORAL_METRICS, *_TIME_AVERAGE_METRICS), "no_active_channels")
    return world


def _active_edge_masks(batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
    """Boolean per-edge activity with cc/zz diagonals excluded."""
    masks = structure_active_masks(batch)
    if masks is None:
        raise ValueError("evaluation batches must carry active_c/m/j masks")
    out = {et: _numpy(mask).astype(bool) for et, mask in masks.items()}
    for et in ("cc", "zz"):
        out[et] &= ~np.eye(out[et].shape[-1], dtype=bool)
    return out


def _effect_layout(
    active_c: np.ndarray, active_m: np.ndarray, layout: EdgeSlotLayout
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """One world's active channel and control effects for the strata rows.

    Their decomposition columns, direct ``cy``/``zb`` edge slots, names (as
    :func:`pymc_mfm.inference.result.result_from_batch` labels them) and families.
    """
    channels, controls = np.flatnonzero(active_c), np.flatnonzero(active_m)
    columns = np.r_[1 + channels, 1 + active_c.size + controls]
    slots = np.r_[layout.slices["cy"].start + channels, layout.slices["zb"].start + controls]
    names = [f"channel:channel_{i}" for i in channels] + [f"control:control_{i}" for i in controls]
    return columns, slots, names, ["channel"] * channels.size + ["control"] * controls.size


def _cell_identity(corpus: Mapping[str, Any], row: int) -> int | None:
    """A usable generator-cell ID, never an invented independence unit."""
    source = corpus.get("cell_id")
    if np.ma.is_masked(source):
        return None
    if isinstance(source, Sequence) and any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) for value in source
    ):
        return None
    try:
        ids = np.asarray(source)
    except (TypeError, ValueError):
        return None
    if ids.shape != (len(corpus["outcome_raw"]),) or ids.dtype.kind not in "iu":
        return None
    return int(ids[row])


def evaluate_rows(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    rows: Sequence[tuple[int, int]],
    *,
    n_samples: int,
    seed: int,
    device: str | torch.device,
    leg: str,
    heads: Sequence[str] = DECOMPOSITION_HEADS,
    per_world: bool = False,
    result_sink: ResultSink | None = None,
    max_batch_worlds: int | None = None,
) -> dict[str, Any]:
    """Score ``rows`` (``(corpus_index, row)`` pairs) on the decomposition ``heads``.

    ``heads`` is a non-empty subset of :data:`DECOMPOSITION_HEADS`. The
    structure head and joint router are always scored: they come from the
    structure pass and the joint query draws, which are made even when
    ``"joint"`` is not selected. Batches are built per context length with
    :func:`pymc_mfm.training.stream.batches_for_rows`, at most ``max_batch_worlds``
    worlds each when it is set, which bounds peak memory; sampling runs under a
    forked CPU/CUDA RNG seeded with ``seed`` and explicit MPS state restoration.
    Only the CPU and evaluation device are seeded; all caller RNG states and
    the model's ``training`` flag are untouched on return. Truth comes from the
    batch (``targets_z``, ``context_targets_z``, ``sales_scale``, masks,
    ``g_*``) and from the corpus (``outcome_raw``, ``outcome_noise``).

    The record is JSON-safe (floats, ints, ``None``, strings, lists):

    * ``leg``, ``n_worlds``, ``n_samples``, ``seed``, ``labels``;
    * ``heads``: ``{head: {window: metrics}}`` for the selected decomposition
      heads, ``"structure_head": {edge: metrics}`` and ``"joint_router":
      {"edge": {edge: metrics}, "graph": metrics}``;
    * ``counts``: equal-world means of active channels and controls, and the
      number of worlds;
    * ``undefined``: ``{"head/.../metric": reason}``. A metric with no finite
      world is ``None`` with its reason; a partially non-finite metric keeps the
      finite mean and reports ``nonfinite_in_<k>_worlds``;
    * ``metric_worlds``: ``{"head/.../metric": worlds with a finite value}`` for
      every per-world metric;
    * ``tables``: ``{"structure_head" | "joint_router": {edge: {"reliability",
      "corp", "thresholds"}}}`` from :func:`metrics.reliability_table`,
      :func:`metrics.isotonic_calibration` and :func:`metrics.threshold_table`
      (empty without scoreable slots; ``corp`` also empty for one truth class);
    * ``strata``: ``{head: {window: summary}}`` for the selected decomposition
      heads, the :func:`pymc_mfm.diagnostics.strata_summary` of every active
      channel and control effect's window total in ``sales_scale`` units: the
      head/window's own draws, their direct ``cy``/``zb`` edge frequency and
      its graph truth. Its size ``cuts`` are quartiles of the evaluated worlds'
      nonzero absolute true totals, the same for both heads and any model on
      these worlds;
    * ``per_world`` (only with ``per_world=True``), in evaluation order:
      ``[{corpus_index, row, cell_id, values, weights, undefined, effects}]`` with every
      per-world key's value (``None`` when undefined), its slot weight (``0``
      when undefined), its reason, and ``effects[head][window]``, the world's
      :func:`pymc_mfm.diagnostics.effect_rows` behind ``strata``. ``cell_id``
      is ``None`` when usable generator-cell metadata is absent.

    Metric keys are fixed (:data:`DECOMPOSITION_METRICS`, :data:`EDGE_METRICS`,
    :data:`GRAPH_METRICS`, :data:`COUNT_METRICS`); :func:`metric_aggregation`
    names how each combines worlds.

    Optional ``result_sink(result, truth, graph_truth, observed)`` captures each
    world's context/query conditional draws already computed here; it needs
    the conditional head. It never samples a second time. Capture conversion
    failures are explicit under ``result_capture`` and do not alter numerical
    scores; callback failures propagate. The callback receives natural-unit
    truth/observations separately.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not rows:
        raise ValueError("evaluate_rows needs at least one row")
    selected = _selected_heads(heads)
    if result_sink is not None and "conditional" not in selected:
        raise ValueError("result_sink captures conditional draws; select the conditional head")
    n_samples = int(n_samples)
    device = torch.device(device)

    decomposition = {
        (head, window): _MeanAccumulator(f"{head}/{window}", DECOMPOSITION_METRICS)
        for head in selected
        for window in WINDOWS
    }
    structure_pool = _EdgePool()
    router_pool = _EdgePool()
    graph_acc = _MeanAccumulator("joint_router/graph", GRAPH_METRICS)
    count_acc = _MeanAccumulator("counts", _WORLD_COUNTS)
    worlds: list[dict[str, Any]] | None = [] if per_world else None
    effects: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in decomposition}
    result_capture: dict[str, Any] = {"captured": 0, "failures": []}

    batches = batches_for_rows(corpora, rows, device=device, max_batch_worlds=max_batch_worlds)
    was_training = model.training
    model.eval()
    try:
        with isolated_rng(device, seed), torch.no_grad():
            for batch, batch_rows in batches:
                _score_batch(
                    model,
                    corpora,
                    batch,
                    batch_rows,
                    n_samples,
                    decomposition,
                    structure_pool,
                    router_pool,
                    graph_acc,
                    count_acc,
                    worlds=worlds,
                    effects=effects,
                    result_sink=result_sink,
                    result_capture=result_capture,
                    leg=leg,
                    seed=seed,
                )
    finally:
        model.train(was_training)

    undefined: dict[str, str] = {}
    metric_worlds: dict[str, int] = {}
    record_heads: dict[str, Any] = {
        head: {
            window: decomposition[(head, window)].finalize(undefined, metric_worlds)
            for window in WINDOWS
        }
        for head in selected
    }
    structure, structure_tables = structure_pool.finalize("structure_head", undefined)
    router_edges, router_tables = router_pool.finalize("joint_router/edge", undefined)
    record_heads["structure_head"] = structure
    record_heads["joint_router"] = {
        "edge": router_edges,
        "graph": graph_acc.finalize(undefined, metric_worlds),
    }
    counts = {**count_acc.finalize(undefined, metric_worlds), "worlds": len(rows)}
    from pymc_mfm.diagnostics.strata import strata_summary

    strata = {
        head: {window: strata_summary(effects[(head, window)]) for window in WINDOWS}
        for head in selected
    }
    return {
        "leg": str(leg),
        "n_worlds": len(rows),
        "n_samples": n_samples,
        "seed": int(seed),
        "heads": record_heads,
        "counts": counts,
        "undefined": undefined,
        "metric_worlds": metric_worlds,
        "tables": {"structure_head": structure_tables, "joint_router": router_tables},
        "strata": strata,
        "labels": {head: HEAD_LABELS[head] for head in record_heads},
        **({"per_world": worlds} if worlds is not None else {}),
        **({"result_capture": result_capture} if result_sink is not None else {}),
    }


def _score_batch(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    batch: dict[str, Tensor],
    batch_rows: Sequence[tuple[int, int]],
    n_samples: int,
    decomposition: dict[tuple[str, str], _MeanAccumulator],
    structure_pool: _EdgePool,
    router_pool: _EdgePool,
    graph_acc: _MeanAccumulator,
    count_acc: _MeanAccumulator,
    *,
    worlds: list[dict[str, Any]] | None = None,
    effects: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
    result_sink: ResultSink | None = None,
    result_capture: dict[str, Any] | None = None,
    leg: str = "",
    seed: int = 0,
) -> None:
    from pymc_mfm.diagnostics.strata import effect_rows

    active_c = _numpy(batch["active_c_mask"]).astype(bool)
    active_m = _numpy(batch["active_m_mask"]).astype(bool)
    active_j = _numpy(batch["active_j_mask"]).astype(bool)
    n_worlds = active_c.shape[0]
    if len(batch_rows) != n_worlds:
        raise ValueError(f"batch has {n_worlds} worlds but {len(batch_rows)} rows were named")
    layout = EdgeSlotLayout(active_c.shape[1], active_m.shape[1], active_j.shape[1])
    scale = _numpy(batch["sales_scale"])
    n_context = int(batch["context_targets_z"].shape[1])
    truth = {"query": _numpy(batch["targets_z"]), "context": _numpy(batch["context_targets_z"])}
    edge_truth = {et: _numpy(batch[f"g_{et}"]) for et in STRUCT_EDGE_KEYS}
    edge_active = _active_edge_masks(batch)
    heads = {head for head, _ in decomposition}

    weekly, channels, controls = model.encoder.forward_with_controls(batch)
    probs = model.struct_head.predictive_probs(model.structure_logits(weekly, channels, controls))
    probs = {et: _numpy(probs[et]) for et in STRUCT_EDGE_KEYS}

    # The joint query pass also feeds the router, so it always runs; the call
    # order below is the draw order of the seeded stream.
    joint_query, g_flow = model.predict_batch(batch, n_samples, return_structure=True)
    draws: dict[tuple[str, str], np.ndarray] = {}
    draw_graphs: dict[tuple[str, str], np.ndarray] = {}
    conditional_graphs = None
    if "joint" in heads:
        joint_context, context_flow = model.sample_context(batch, n_samples, return_structure=True)
        draws[("joint", "query")] = _numpy(joint_query)
        draws[("joint", "context")] = _numpy(joint_context)
    if "conditional" in heads:
        conditional_query, query_graphs = model.sample_disentangled(
            batch, n_samples, mode="model_average", window="query"
        )
        conditional_context, context_graphs = model.sample_disentangled(
            batch, n_samples, mode="model_average", window="context"
        )
        draws[("conditional", "query")] = _numpy(conditional_query)
        draws[("conditional", "context")] = _numpy(conditional_context)
        conditional_graphs = {"query": _numpy(query_graphs), "context": _numpy(context_graphs)}
        for window in WINDOWS:
            draw_graphs[("conditional", window)] = conditional_graphs[window]

    graphs = _numpy(binary_graphs_from_flow(g_flow))  # [S, B, L]
    # Thresholding NaN/Inf flow values would silently turn them into binary edges.
    finite_graph_slots = torch.isfinite(g_flow).all(dim=2).detach().cpu().numpy()
    graphs[~finite_graph_slots] = np.nan
    if graphs.shape[-1] != layout.n_slots:
        raise ValueError(
            f"joint router emitted {graphs.shape[-1]} slots but the geometry has {layout.n_slots}"
        )
    if "joint" in heads:
        draw_graphs[("joint", "query")] = graphs
        context_graphs = _numpy(binary_graphs_from_flow(context_flow))
        context_finite = torch.isfinite(context_flow).all(dim=2).detach().cpu().numpy()
        context_graphs[~context_finite] = np.nan
        if context_graphs.shape[-1] != layout.n_slots:
            raise ValueError(
                f"joint context router emitted {context_graphs.shape[-1]} slots but "
                f"the geometry has {layout.n_slots}"
            )
        draw_graphs[("joint", "context")] = context_graphs
    true_flat = np.asarray(layout.pack(edge_truth), dtype=np.float64)  # [B, L]
    slot_active = np.asarray(layout.active_slot_mask(active_c, active_m, active_j)).astype(bool)
    marginals = layout.unpack(graphs.mean(axis=0))  # blocks [B, ...]
    slot_families = np.asarray(layout.slot_edge_types)
    slot_endpoints = np.stack([layout.slot_src_index, layout.slot_dst_index], axis=1)  # [L, 2]

    window_slices = {"query": slice(n_context, None), "context": slice(None, n_context)}
    for b, (corpus_index, row) in enumerate(batch_rows):
        corpus = corpora[corpus_index]
        outcome = np.asarray(corpus["outcome_raw"][row], dtype=np.float64)
        noise = np.asarray(corpus["outcome_noise"][row], dtype=np.float64)
        world_id = f"corpus:{corpus_index}/row:{row}"
        columns, slots, names, families = _effect_layout(active_c[b], active_m[b], layout)
        live = true_flat[b, slots] != 0
        entry = (
            None
            if worlds is None
            else {
                "corpus_index": int(corpus_index),
                "row": int(row),
                "cell_id": _cell_identity(corpus, row),
                "values": {},
                "weights": {},
                "undefined": {},
                "effects": {},
            }
        )
        capture_provenance = None
        if result_sink is not None:
            from pymc_mfm.inference.result import _input_summary

            input_support = {}
            for key in ("ctx_spend_z", "qry_spend_z", "ctx_controls_z", "qry_controls_z"):
                values = _numpy(batch[key][b])
                input_support[key] = _input_summary(
                    values, "encoded context-normalized representation"
                )
            # The keys and layout ``decompose`` records, so diagnostics read both alike.
            prefixes = {"ctx": window_slices["context"], "qry": window_slices["query"]}
            for key, field in (("spend", "treatment_raw"), ("controls", "covariates")):
                raw = np.asarray(corpus[field][row], dtype=np.float64)
                for window, span in prefixes.items():
                    values = raw[span]
                    input_support[f"{window}_{key}_raw"] = _input_summary(
                        values, "native input units"
                    )
            context_observed = outcome[:n_context]
            input_support["ctx_outcome_raw"] = _input_summary(context_observed, "outcome")
            capture_provenance = {
                "data_split": str(leg),
                "corpus_index": int(corpus_index),
                "row": int(row),
                "world_id": world_id,
                "data_sha256": observation_identity(batch, world=b),
                "seed": int(seed),
                "input_support": input_support,
                "capture": "existing conditional evaluator samples; no additional inference",
            }
        for (head, window), samples in draws.items():
            span = window_slices[window]
            decomposition[(head, window)].add(
                _score_decomposition(
                    samples[:, b],
                    truth[window][b],
                    active_c[b],
                    active_m[b],
                    edge_truth["cy"][b],
                    edge_truth["zb"][b],
                    outcome[span],
                    noise[span],
                    float(scale[b]),
                    graph_c=draw_graphs[(head, window)][:, b, layout.slices["cy"]],
                    graph_m=draw_graphs[(head, window)][:, b, layout.slices["zb"]],
                ),
                entry,
            )
            world_rows = effect_rows(
                world_id,
                names,
                families,
                samples[:, b][..., columns],
                truth[window][b][:, columns],
                live=live,
                edge_draws=draw_graphs[(head, window)][:, b][:, slots],
            )
            if effects is not None:
                effects[(head, window)].extend(world_rows)
            if entry is not None:
                entry["effects"].setdefault(head, {})[window] = world_rows
            if result_sink is not None and head == "conditional":
                from pymc_mfm.inference.result import result_from_batch

                try:
                    captured = result_from_batch(
                        batch,
                        samples,
                        conditional_graphs[window],
                        model_config=model.config,
                        world=b,
                        window=window,
                        mode="model_average",
                        provenance=capture_provenance,
                    )
                except (ValueError, TypeError) as exc:
                    if result_capture is None:
                        raise
                    result_capture["failures"].append(
                        {
                            "corpus_index": int(corpus_index),
                            "row": int(row),
                            "window": window,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                else:
                    result_sink(
                        captured,
                        truth[window][b] * float(scale[b]),
                        true_flat[b].copy(),
                        outcome[span].copy(),
                    )
                    if result_capture is not None:
                        result_capture["captured"] += 1
        for et in STRUCT_EDGE_KEYS:
            keep = edge_active[et][b].ravel()
            y = edge_truth[et][b].ravel()
            structure_pool.add(et, probs[et][b].ravel(), y, keep)
            router_pool.add(et, np.asarray(marginals[et][b]).ravel(), y, keep)
        keep = slot_active[b]
        world_graphs = graphs[:, b, keep]
        squares = metrics.adjacency_blocks(
            world_graphs, slot_families[keep], slot_endpoints[keep], SQUARE_EDGE_TYPES
        )
        graph_acc.add(metrics.graph_scores(world_graphs, true_flat[b, keep], squares), entry)
        counts = metrics.WorldScores()
        counts.put("active_channels", active_c[b].sum())
        counts.put("active_controls", active_m[b].sum())
        count_acc.add(counts, entry)
        if entry is not None:
            worlds.append(entry)


def flatten_evaluation(record: Mapping[str, Any]) -> dict[str, float | None]:
    """Flatten a record to ``"{leg}/{head}/.../{metric}"`` scalar keys.

    Key shapes: ``{leg}/{joint|conditional}/{window}/{metric}``,
    ``{leg}/structure_head/{edge}/{metric}``,
    ``{leg}/joint_router/edge/{edge}/{metric}``,
    ``{leg}/joint_router/graph/{metric}`` and ``{leg}/counts/{metric}``.
    Integer counts become floats; undefined metrics stay ``None``. ``/`` is the
    only separator (MLflow-safe).
    """
    leg = record["leg"]
    heads = record["heads"]
    out: dict[str, float | None] = {}

    def put(prefix: str, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            out[f"{prefix}/{key}"] = None if value is None else float(value)

    for head in DECOMPOSITION_HEADS:
        for window, values in heads.get(head, {}).items():
            put(f"{leg}/{head}/{window}", values)
    for et, values in heads["structure_head"].items():
        put(f"{leg}/structure_head/{et}", values)
    for et, values in heads["joint_router"]["edge"].items():
        put(f"{leg}/joint_router/edge/{et}", values)
    put(f"{leg}/joint_router/graph", heads["joint_router"]["graph"])
    put(f"{leg}/counts", record["counts"])
    return out
