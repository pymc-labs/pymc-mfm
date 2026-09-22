"""Saved-draw calibration diagnostics with explicit independence qualifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
SCREENING_QUALIFICATION = (
    "Component/time ranks in one world are dependent screening diagnostics, "
    "not independent-world SBC evidence. No calibrated pass/fail is assigned."
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _ranks(ranks, n_post_samples):
    n_post_samples = _positive_int(n_post_samples, "n_post_samples")
    values = np.asarray(ranks, dtype=np.float64).ravel()
    if (
        not values.size
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or np.any((values < 0) | (values > n_post_samples))
    ):
        raise ValueError(f"ranks must be nonempty integers in [0,{n_post_samples}]")
    return values.astype(np.int64)


def randomized_ranks(samples, truth, *, seed=0, rng=None):
    """Rank each truth coordinate among aligned draws, randomizing exact ties.

    Draws are ``[S,...]``, truth exactly ``[...]``. The inclusive tie offset is
    uniform on ``0,...,ties``; zeros are atoms, not values to jitter or drop.
    An explicitly supplied generator is advanced; otherwise only a local RNG
    is used and ``seed`` is retained by the calling report.
    """
    values = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    if values.ndim < 1 or len(values) < 1 or values.shape[1:] != targets.shape:
        raise ValueError("truth shape must exactly match samples without their draw axis")
    if targets.size == 0 or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("samples and truth must be nonempty and finite")
    rng = np.random.default_rng(seed) if rng is None else rng
    below = np.sum(values < targets, axis=0)
    ties = np.sum(values == targets, axis=0)
    return below + rng.integers(0, ties + 1, size=targets.shape)


def compute_rank(true_value, posterior_samples, rng=None):
    """Randomized scalar rank, including the upper endpoint when all draws tie."""
    samples = np.asarray(posterior_samples, dtype=np.float64).ravel()
    return int(randomized_ranks(samples, np.asarray(true_value), rng=rng))


def coverage_from_ranks(ranks, n_post_samples, levels=COVERAGE_LEVELS):
    """Central coverage of discrete mid-PIT, distinct from sample quantile coverage."""
    values = _ranks(ranks, n_post_samples)
    levels = np.asarray(levels, dtype=np.float64)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all():
        raise ValueError("levels must be a nonempty finite vector")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must lie in (0,1)")
    pit = (values + 0.5) / (n_post_samples + 1)
    return {
        f"cov{100 * level:g}": float(np.mean(np.abs(pit - 0.5) <= level / 2)) for level in levels
    }


def _ks(counts, n, grid):
    cumulative = np.cumsum(counts, axis=-1)
    return np.maximum(
        np.max(cumulative / n - grid, axis=-1),
        np.max(grid - (cumulative - counts) / n, axis=-1),
    )


def rank_uniformity_stats(
    ranks,
    n_post_samples,
    *,
    independent_worlds=False,
    n_bins=20,
    simulations=10000,
    seed=0,
    alpha=0.05,
):
    """Rank histogram/PIT/ECDF; null tests require independent-world ranks.

    The KS null is simulated under the exact discrete rank law rather than a
    continuous asymptotic law. Simultaneous ECDF bands use binomial quantiles
    with Monte Carlo calibrated pointwise level. No inference is performed.
    """
    values = _ranks(ranks, n_post_samples)
    n_bins = min(_positive_int(n_bins, "n_bins"), n_post_samples + 1)
    simulations = _positive_int(simulations, "simulations")
    if not isinstance(independent_worlds, bool):
        raise ValueError("independent_worlds must be an explicit boolean")
    if not 0 < alpha < 1 or not np.isfinite(alpha):
        raise ValueError("alpha must lie in (0,1)")
    edges = np.linspace(0, n_post_samples + 1, n_bins + 1)
    histogram = np.histogram(values, bins=edges)[0]
    expected = len(values) * np.histogram(np.arange(n_post_samples + 1), bins=edges)[0]
    expected = expected / (n_post_samples + 1)
    counts = np.bincount(values, minlength=n_post_samples + 1)
    pit_grid = (np.arange(n_post_samples + 1) + 0.5) / (n_post_samples + 1)
    null_cdf = (np.arange(n_post_samples + 1) + 1) / (n_post_samples + 1)
    statistic = float(_ks(counts, len(values), pit_grid))
    report = {
        "ranks": values.tolist(),
        "pit": ((values + 0.5) / (n_post_samples + 1)).tolist(),
        "rank_histogram": histogram.tolist(),
        "rank_bin_edges": edges.tolist(),
        "expected_rank_histogram": expected.tolist(),
        "ecdf_grid": pit_grid.tolist(),
        "ecdf": (np.cumsum(counts) / len(values)).tolist(),
        "null_ecdf": null_cdf.tolist(),
        "ks": statistic,
        "chi2": float(np.sum((histogram - expected) ** 2 / expected)),
        "ks_pvalue": None,
        "chi2_pvalue": None,
        "ecdf_band_reject": None,
        "pot_c_pvalue": None,
        "pot_c_status": {"status": "unavailable", "reason": SCREENING_QUALIFICATION},
        "rank_coverage": coverage_from_ranks(values, n_post_samples),
        "n_ranks": len(values),
        "n_post_samples": n_post_samples,
        "seed": int(seed),
        "alpha": float(alpha),
        "independent_worlds": bool(independent_worlds),
        "qualification": SCREENING_QUALIFICATION,
        "status": "screening_only",
    }
    if not independent_worlds:
        return report
    if simulations < 100:
        raise ValueError("independent-world null tests need at least 100 simulations")
    from scipy.stats import binom, chi2

    rng = np.random.default_rng(seed)
    null_counts = rng.multinomial(
        len(values), np.full(n_post_samples + 1, 1 / (n_post_samples + 1)), size=simulations
    )
    null_statistic = _ks(null_counts, len(values), pit_grid)
    null_cumulative = np.cumsum(null_counts, axis=1)[:, :-1]
    probabilities = null_cdf[:-1]
    low_gamma, high_gamma = 1e-9, alpha
    for _ in range(40):
        gamma = np.sqrt(low_gamma * high_gamma)
        lower = binom.ppf(gamma / 2, len(values), probabilities)
        upper = binom.ppf(1 - gamma / 2, len(values), probabilities)
        cover = np.mean(np.all((null_cumulative >= lower) & (null_cumulative <= upper), axis=1))
        if cover < 1 - alpha:
            high_gamma = gamma
        else:
            low_gamma = gamma
    lower = binom.ppf(low_gamma / 2, len(values), probabilities) / len(values)
    upper = binom.ppf(1 - low_gamma / 2, len(values), probabilities) / len(values)
    actual = np.cumsum(counts)[:-1] / len(values)
    excursion = float(np.maximum(np.maximum(lower - actual, actual - upper), 0).max())
    report.update(
        ks_pvalue=float((1 + np.sum(null_statistic >= statistic)) / (1 + simulations)),
        chi2_pvalue=float(chi2.sf(report["chi2"], n_bins - 1)) if n_bins > 1 else None,
        ecdf_lower=lower.tolist() + [1.0],
        ecdf_upper=upper.tolist() + [1.0],
        ecdf_band_reject=float(excursion > 0),
        ecdf_max_excursion=excursion,
        ecdf_gamma=float(low_gamma),
        simulations=simulations,
        status="independent_worlds_declared",
        qualification=(
            "Independent worlds are a caller-supplied sampling contract, not inferred from array "
            "shape. Monte Carlo null tests have finite resolution; chi-square p-values are "
            "asymptotic and may be unreliable with small expected bin counts. A non-rejection "
            "does not prove calibration or out-of-distribution validity."
        ),
    )
    try:
        from arviz_stats.base import array_stats
    except ImportError:
        report["pot_c_status"] = {
            "status": "unavailable_dependency",
            "reason": "optional arviz_stats uniformity test not installed",
        }
    else:
        jitter = np.random.default_rng(20250211).random(len(values))
        randomized_pit = (values + jitter) / (n_post_samples + 1)
        value = float(array_stats.uniformity_test(randomized_pit)[0])
        if not np.isfinite(value):
            raise ValueError("optional pot_c uniformity test returned a nonfinite value")
        report["pot_c_pvalue"] = value
        report["pot_c_status"] = {
            "status": "graded_diagnostic",
            "reason": (
                "Order-statistic uniformity test is not a calibrated gate; "
                "finite-sample size can exceed nominal"
            ),
        }
    return report


def calibration_summary(samples, truth, *, seed=0, levels=COVERAGE_LEVELS):
    """One-world screening ranks and actual central-quantile interval coverage."""
    values = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    ranks = randomized_ranks(values, targets, seed=seed)
    report = rank_uniformity_stats(ranks, len(values), seed=seed)
    intervals = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("coverage levels must lie in (0,1)")
        low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2], axis=0)
        intervals.append(
            {
                "level": float(level),
                "coverage": float(np.mean((targets >= low) & (targets <= high))),
                "mean_width": float(np.mean(high - low)),
            }
        )
    report["intervals"] = intervals
    report["rank_shape"] = list(ranks.shape)
    report["interval_definition"] = (
        "central equal-tail, NumPy linear quantiles, inclusive endpoints"
    )
    return report


def sbc_report(
    samples,
    truth,
    *,
    world_ids: Sequence[str],
    quantity_names: Sequence[str],
    provenance: Mapping[str, object],
    seed=0,
    alpha=0.05,
    simulations=10000,
):
    """Report independent-world SBC from SAVED ``[W,S,Q]`` posterior draws.

    ``truth`` is ``[W,Q]``. Each quantity must supply exactly one statistic per
    independent simulated world (aggregate trajectories before calling).
    Explicit unique world IDs and prior/data provenance are required. Cross-
    quantity family-wise control uses Bonferroni, not an independence assumption.
    """
    values = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] < 1:
        raise ValueError("SBC samples must have shape [W>=2,S,Q>=1]")
    if targets.shape != (values.shape[0], values.shape[2]):
        raise ValueError("SBC truth must have shape [W,Q]")
    worlds = tuple(world_ids)
    quantities = tuple(quantity_names)
    if (
        len(worlds) != len(values)
        or not all(isinstance(name, str) and name for name in worlds)
        or len(set(worlds)) != len(worlds)
    ):
        raise ValueError("world_ids must uniquely identify every independent world")
    if (
        len(quantities) != values.shape[2]
        or not all(isinstance(name, str) and name for name in quantities)
        or len(set(quantities)) != len(quantities)
    ):
        raise ValueError("quantity_names must uniquely identify every quantity")
    if not provenance or provenance.get("independent_worlds") is not True:
        raise ValueError("provenance must explicitly declare independent_worlds=True")
    identity = provenance.get("data_sha256")
    if (
        not provenance.get("prior")
        or not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("SBC provenance requires a prior description and data_sha256 identity")
    rng = np.random.default_rng(seed)
    ranks = np.stack(
        [
            randomized_ranks(draws, target, rng=rng)
            for draws, target in zip(values, targets, strict=True)
        ]
    )
    per_quantity = {
        name: rank_uniformity_stats(
            ranks[:, index],
            values.shape[1],
            independent_worlds=True,
            seed=seed,
            alpha=alpha / len(quantities),
            simulations=simulations,
        )
        for index, name in enumerate(quantities)
    }
    return {
        "format_version": 1,
        "kind": "independent_world_sbc",
        "provenance": dict(provenance),
        "world_ids": list(worlds),
        "quantity_names": list(quantities),
        "seed": int(seed),
        "ranks": ranks.tolist(),
        "quantities": per_quantity,
        "family_wise": {
            "method": "Bonferroni",
            "alpha": float(alpha),
            "alpha_per_quantity": float(alpha / len(quantities)),
            "rejecting_quantities": [
                name for name, record in per_quantity.items() if record["ecdf_band_reject"]
            ],
        },
        "qualification": "A non-rejection is not proof of calibration; prior/data support applies.",
    }
