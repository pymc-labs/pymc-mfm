"""Metric names, valid ranges and units of the evaluator, without PyTorch.

Split from :mod:`pymc_mfm.training.evaluation`, which re-exports every name here, so that
the diagnostics catalog of a result can name metrics and units without importing the model.
"""

from __future__ import annotations

import math

from pymc_mfm.training import metrics

#: ``("cc", "zz")``, as ``pymc_mfm.model.graph.SQUARE_EDGE_TYPES``; a test keeps them equal.
SQUARE_EDGE_TYPES: tuple[str, ...] = ("cc", "zz")

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
_FAMILY_STATS = ("scaled_mae", "crps", "cov80")
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
    f"attribution_live_{effect}_conditional_cov80" for effect in _ATTRIBUTION_EFFECTS
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
        f"{family}_{stat}": _PROBABILITY if stat == "cov80" else _NONNEGATIVE
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
        f"{family}_{stat}": "dimensionless" if stat == "cov80" else "sales_scale"
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


def metric_units(key: str) -> str:
    """:data:`METRIC_UNITS` unit of a flat key (with or without its leg) or a metric name.

    A metric this release does not define raises ``ValueError`` naming the key.
    """
    name = key.rsplit("/", 1)[-1]
    if name not in METRIC_UNITS:
        raise ValueError(f"{key!r}: no unit for {name!r} in this release")
    return METRIC_UNITS[name]
