"""Pure-NumPy scoring rules for training evaluation.

The module provides empirical CRPS, central-interval coverage,
dimension-normalized energy score, and binary-probability diagnostics: log
score, Brier score, AUROC, and expected calibration error. ``energy_score``
returns ``None`` when fewer than two draws are available; callers record
unavailable or non-finite results explicitly.
"""

from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based ranks of a 1-D array with mean ranks for ties."""
    values = np.asarray(values, dtype=np.float64).ravel()
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        ranks_sorted[start:end] = 0.5 * (start + end + 1)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[sorter] = ranks_sorted
    return ranks


def empirical_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean empirical CRPS ``E|X - y| - 0.5 E|X - X'|`` over trailing dims.

    ``samples`` is ``(S, ...)`` with the draw axis first and ``truth`` matches the
    trailing shape. The pairwise term uses the sorted-sample identity
    ``E|X - X'| = 2 / S^2 * sum_i (2 i - S - 1) x_(i)`` in O(S log S).
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    if sample_arr.ndim < 1 or sample_arr.shape[0] < 1:
        raise ValueError("empirical_crps requires at least one posterior sample")
    if sample_arr.shape[1:] != truth_arr.shape:
        raise ValueError(
            "truth shape must match samples without the sample axis "
            f"(got samples {sample_arr.shape}, truth {truth_arr.shape})"
        )
    term1 = np.mean(np.abs(sample_arr - truth_arr[None, ...]), axis=0)
    sorted_samples = np.sort(sample_arr, axis=0)
    n = sorted_samples.shape[0]
    coeff = (2 * np.arange(1, n + 1, dtype=np.float64) - n - 1).reshape(
        (n,) + (1,) * (sorted_samples.ndim - 1)
    )
    half_pairwise = np.sum(coeff * sorted_samples, axis=0) / float(n * n)
    return float(np.mean(term1 - half_pairwise))


def coverage(samples: np.ndarray, truth: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """Return the fraction of truth entries inside central sample quantiles.

    ``samples`` is ``(S, ...)``; quantiles use NumPy's default linear method
    over the draw axis.
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    if sample_arr.shape[1:] != truth_arr.shape:
        raise ValueError(
            "truth shape must match samples without the sample axis "
            f"(got samples {sample_arr.shape}, truth {truth_arr.shape})"
        )
    # Comparisons against NaN quantiles would otherwise report false coverage.
    if not np.isfinite(sample_arr).all() or not np.isfinite(truth_arr).all():
        return float("nan")
    lo = np.quantile(sample_arr, low, axis=0)
    hi = np.quantile(sample_arr, high, axis=0)
    return float(np.mean((truth_arr >= lo) & (truth_arr <= hi)))


def energy_score(samples: np.ndarray, truth: np.ndarray) -> float | None:
    """Dimension-normalized energy-score U-statistic; ``None`` below two draws.

    ``samples`` is ``(S, ...)`` and ``truth`` matches the trailing shape; both are
    flattened to ``d`` coordinates. Returns
    ``mean_s ||X_s - y|| / sqrt(d) - sum_{s<t} ||X_s - X_t|| / (S (S - 1)) / sqrt(d)``,
    whose pair term is the unbiased off-diagonal mean.
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    if sample_arr.ndim < 2:
        raise ValueError(f"samples must be (S, ...), got {sample_arr.shape}")
    if truth_arr.shape != sample_arr.shape[1:]:
        raise ValueError(
            f"truth shape {truth_arr.shape} does not match samples {sample_arr.shape[1:]}"
        )
    n_draws = sample_arr.shape[0]
    if n_draws < 2:
        return None
    flat_samples = sample_arr.reshape(n_draws, -1)
    flat_truth = truth_arr.reshape(-1)
    dimension = flat_truth.size
    if dimension == 0:
        raise ValueError("energy score is undefined with no coordinates")
    scale = np.sqrt(float(dimension))
    truth_term = np.linalg.norm(flat_samples - flat_truth[None, :], axis=1).mean() / scale
    # Visit each unordered pair once without an S-by-S-by-d temporary. Reports
    # can contain hundreds of long trajectories, not just tiny minibatches.
    pair_sum = sum(
        np.linalg.norm(flat_samples[i + 1 :] - flat_samples[i], axis=1).sum()
        for i in range(n_draws - 1)
    )
    pair_term = pair_sum / (n_draws * (n_draws - 1) * scale)
    return float(truth_term - pair_term)


def log_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean binary negative log-likelihood ``-mean(y log p + (1 - y) log(1 - p))``.

    Probabilities are clipped to ``[eps, 1 - eps]`` in float64 before the log.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    eps = np.finfo(np.float64).eps
    p_clipped = np.clip(p_arr, eps, 1.0 - eps)
    return float(-np.mean(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log1p(-p_clipped)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared probability error ``mean((p - y)^2)``."""
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean((p_arr - y_arr) ** 2))


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    """AUROC as the Mann-Whitney U statistic with average ranks for ties.

    ``U = sum(ranks of positives) - n_pos (n_pos + 1) / 2``; the result is
    ``U / (n_pos n_neg)``. Returns ``None`` when only one class is present.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(p_arr)
    rank_sum_pos = float(ranks[y_arr == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over ``bins`` equal-width probability bins.

    ``sum_b (n_b / n) |mean(y in b) - mean(p in b)|``; empty bins contribute 0.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if bins < 1:
        raise ValueError("bins must be a positive integer")
    if y_arr.size == 0:
        raise ValueError("ece requires at least one probability/label pair")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.digitize(p_arr, edges[1:-1]), 0, bins - 1)
    n = y_arr.size
    total = 0.0
    for b in range(bins):
        in_bin = bin_idx == b
        n_b = int(in_bin.sum())
        if n_b == 0:
            continue
        total += (n_b / n) * abs(float(y_arr[in_bin].mean()) - float(p_arr[in_bin].mean()))
    return float(total)
