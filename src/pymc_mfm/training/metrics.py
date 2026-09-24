"""Pure-NumPy scoring rules for training evaluation.

Scalar rules: empirical CRPS, central-interval coverage, the
dimension-normalized energy score, and binary-probability diagnostics (log
score, Brier score, AUROC, AUPRC, null false-positive rate, expected
calibration error, reliability and threshold tables). ``energy_score`` and
``auroc`` return ``None`` when undefined.

The other scorers that can be undefined return a :data:`Scored` pair,
``(value, None)`` or ``(None, reason)``, or one world's :class:`WorldScores`
for a metric family: temporal coherence, live/dead attribution, per-draw
closure and latent-sales R², time-average coverage, the cumulative energy
score, and whole-graph posterior statistics. They never return NaN. Scorers of
single quantities reject non-finite inputs; family scorers record them as
``undefined[key] = "nonfinite"`` for every key of the family, and a score that
overflows from finite inputs is ``"nonfinite"`` as well.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

#: ``(value, None)`` when a score is defined, ``(None, reason)`` when it is not.
Scored = tuple[float | None, str | None]

#: Undefined reason of every key of a family whose inputs are not all finite.
NONFINITE = "nonfinite"
#: Equal-width probability bins of :func:`reliability_table`.
RELIABILITY_BINS = 10
#: Probability thresholds of :func:`threshold_table`.
THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9)
#: Denominator floor of relative closure errors.
CLOSURE_EPS = 1e-6

#: Keys of :func:`temporal_scores` (after its prefix).
TEMPORAL_KEYS = (
    "truth_lag1",
    "draw_lag1",
    "lag1_tracking_abs_error",
    "truth_error_residual_lag1",
    "posterior_centered_residual_lag1",
    "adjacent_draw_correlation",
)
#: Keys of :func:`attribution_stats` and :func:`leakage_stats`.
LIVE_ATTRIBUTION_KEYS = ("crps", "mean_bias", "mean_abs_error", "demeaned_rmse")
DEAD_ATTRIBUTION_KEYS = ("leakage_scaled_mae", "draw_leakage_scaled_mae")
#: Keys of :func:`latent_sales_scores`.
LATENT_SALES_KEYS = (
    "closure_rel_err",
    "closure_draw_rel_err",
    "closure_draw_q90_rel_err",
    "latent_sales_r2",
)
#: Keys of :func:`time_average_scores` (after its prefix).
TIME_AVERAGE_KEYS = ("cov80", "cov80_live", "cov80_dead")
#: Keys of :func:`graph_scores` that compare the draws with the true graph.
_TRUTH_GRAPH_KEYS = (
    "shd_mean",
    "exact_recovery",
    "shd_median",
    "normalized_shd_mean",
    "energy_score",
    "categorical_brier_score",
)
#: Keys of :func:`graph_scores` besides the per-block cycle rates.
GRAPH_KEYS = (
    *_TRUTH_GRAPH_KEYS,
    "entropy",
    "n_distinct",
    "pairwise_comoment_abs_err",
    "pairwise_comoment_max_abs_err",
)
_CYCLE_RATES = ("two_cycle_rate", "cycle_rate")


@dataclass(slots=True)
class WorldScores:
    """One world's values for a set of metric keys.

    Each key is either in ``values`` (finite), with ``weights[key]`` the number
    of slots averaged within the world (``1`` for a world-level quantity), or
    in ``undefined`` with the reason it has no value.
    """

    values: dict[str, float] = field(default_factory=dict)
    weights: dict[str, int] = field(default_factory=dict)
    undefined: dict[str, str] = field(default_factory=dict)

    def put(self, key: str, value: float, weight: int = 1) -> None:
        """Record a value; one that overflowed to a non-finite float is ``nonfinite``."""
        value = float(value)
        if weight < 1:
            raise ValueError(f"{key} needs a positive slot weight, got {weight}")
        if not math.isfinite(value):
            self.skip(key, NONFINITE)
            return
        self._claim(key)
        self.values[key] = value
        self.weights[key] = int(weight)

    def skip(self, key: str, reason: str) -> None:
        self._claim(key)
        self.undefined[key] = reason

    def score(self, key: str, scored: Scored, weight: int = 1) -> None:
        """Record a :data:`Scored` pair as a value or as its undefined reason."""
        value, reason = scored
        if value is None:
            self.skip(key, reason)
        else:
            self.put(key, value, weight)

    def update(self, other: WorldScores) -> None:
        for key, value in other.values.items():
            self.put(key, value, other.weights[key])
        for key, reason in other.undefined.items():
            self.skip(key, reason)

    def _claim(self, key: str) -> None:
        if key in self.values or key in self.undefined:
            raise ValueError(f"{key} is scored twice")


def _skip_all(world: WorldScores, keys: Sequence[str], reason: str) -> WorldScores:
    for key in keys:
        world.skip(key, reason)
    return world


def _all_finite(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(array).all() for array in arrays)


def _finite_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _checked(value: float) -> Scored:
    """A computed score, or ``nonfinite`` when finite inputs overflowed."""
    value = float(value)
    return (value, None) if math.isfinite(value) else (None, NONFINITE)


def _draws_and_truth(samples, truth, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """``(S, ...)`` draws with ``ndim`` axes and truth matching the trailing shape."""
    draws = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if draws.ndim != ndim or target.shape != draws.shape[1:] or draws.shape[0] < 1:
        raise ValueError(
            f"expected {ndim}-D draws (S, ...) and truth matching their trailing shape, "
            f"got {draws.shape} and {target.shape}"
        )
    return draws, target


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


def _binary_pairs(p, y) -> tuple[np.ndarray, np.ndarray]:
    p_arr = _finite_array(p, "probabilities").ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if p_arr.shape != y_arr.shape:
        raise ValueError(f"{p_arr.size} probabilities but {y_arr.size} labels")
    if not np.isin(y_arr, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return p_arr, y_arr


def auprc(p: np.ndarray, y: np.ndarray) -> Scored:
    """Tie-aware average precision ``sum_k (R_k - R_{k-1}) P_k``.

    Precision ``P_k`` and recall ``R_k`` are evaluated once per distinct score in
    decreasing order, so tied scores enter together and the value does not
    depend on input order (a constant score gives the prevalence).
    ``(None, "one_class")`` unless both labels occur.
    """
    p_arr, y_arr = _binary_pairs(p, y)
    n_pos = int(y_arr.sum())
    if n_pos in (0, y_arr.size):
        return None, "one_class"
    order = np.argsort(-p_arr, kind="mergesort")
    scores, hits = p_arr[order], y_arr[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    true_positives = np.cumsum(hits)[ends]
    precision = true_positives / (ends + 1)
    recall = true_positives / n_pos
    return float(np.sum(np.diff(recall, prepend=0.0) * precision)), None


def null_fpr(p: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Scored:
    """Share of negative slots scored strictly above ``threshold``.

    ``mean(p[y == 0] > threshold)``; ``(None, "no_negative_slots")`` without negatives.
    """
    p_arr, y_arr = _binary_pairs(p, y)
    negative = y_arr == 0
    if not negative.any():
        return None, "no_negative_slots"
    return float(np.mean(p_arr[negative] > threshold)), None


def reliability_table(
    p: np.ndarray, y: np.ndarray, bins: int = RELIABILITY_BINS
) -> list[dict[str, float | int]]:
    """Non-empty equal-width probability bins with their observed frequency.

    Rows ``{bin, lower, upper, count, mean_probability, observed_frequency}``.
    Bins are assigned as in :func:`ece`, so ``p = 1`` falls in the last bin.
    """
    p_arr, y_arr = _binary_pairs(p, y)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p_arr, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        in_bin = index == b
        count = int(in_bin.sum())
        if count:
            rows.append(
                {
                    "bin": b,
                    "lower": float(edges[b]),
                    "upper": float(edges[b + 1]),
                    "count": count,
                    "mean_probability": float(p_arr[in_bin].mean()),
                    "observed_frequency": float(y_arr[in_bin].mean()),
                }
            )
    return rows


def threshold_table(
    p: np.ndarray, y: np.ndarray, thresholds: Sequence[float] = THRESHOLDS
) -> list[dict[str, float | int | None]]:
    """Confusion counts per threshold, predicting positive when ``p >= threshold``.

    Rows ``{threshold, tp, fp, tn, fn, precision, recall, fpr}``; a ratio whose
    denominator is zero is ``None``.
    """
    p_arr, y_arr = _binary_pairs(p, y)
    positive = y_arr == 1
    rows = []
    for threshold in thresholds:
        predicted = p_arr >= threshold
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & ~positive))
        tn = int(np.sum(~predicted & ~positive))
        fn = int(np.sum(~predicted & positive))
        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "fpr": fp / (fp + tn) if fp + tn else None,
            }
        )
    return rows


def _row_correlations(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlations along the last axis, and where they are defined.

    A row is undefined when either side has zero range. Deviations are scaled
    by their largest magnitude first, so tiny or huge series neither underflow
    nor overflow; results are clipped to ``[-1, 1]``.
    """
    a = left - left.mean(axis=-1, keepdims=True)
    b = right - right.mean(axis=-1, keepdims=True)
    scale_a = np.abs(a).max(axis=-1)
    scale_b = np.abs(b).max(axis=-1)
    defined = (
        (np.ptp(left, axis=-1) > 0) & (np.ptp(right, axis=-1) > 0) & (scale_a > 0) & (scale_b > 0)
    )
    a = a / np.where(defined, scale_a, 1.0)[..., None]
    b = b / np.where(defined, scale_b, 1.0)[..., None]
    denominator = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    correlation = np.sum(a * b, axis=-1) / np.where(defined, denominator, 1.0)
    return np.where(defined, np.clip(correlation, -1.0, 1.0), 0.0), defined


def _mean_defined(values: np.ndarray, defined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over the last axis of the defined entries, and whether any is defined."""
    counts = defined.sum(axis=-1)
    total = np.where(defined, values, 0.0).sum(axis=-1)
    return total / np.maximum(counts, 1), counts > 0


def finite_correlation(left: np.ndarray, right: np.ndarray) -> Scored:
    """Pearson correlation of two finite vectors of equal length (at least two).

    ``(None, "constant_series")`` when either vector has zero range; the value
    is clipped to ``[-1, 1]``. Non-finite entries raise: pairs are never dropped.
    """
    a = _finite_array(left, "left")
    b = _finite_array(right, "right")
    if a.ndim != 1 or a.shape != b.shape or a.size < 2:
        raise ValueError(
            f"correlation needs two equal 1-D vectors of length >= 2, got {a.shape} and {b.shape}"
        )
    value, defined = _row_correlations(a, b)
    return _checked(value) if defined else (None, "constant_series")


def lag1_correlation(values: np.ndarray) -> Scored:
    """Correlation of adjacent entries, ``corr(x[:-1], x[1:])``, of a finite series.

    ``(None, "needs_t_ge_3")`` below three entries (two pairs) and ``(None,
    "constant_series")`` when either side has zero range.
    """
    series = _finite_array(values, "series")
    if series.ndim != 1:
        raise ValueError(f"lag-1 correlation needs a 1-D series, got {series.shape}")
    if series.size < 3:
        return None, "needs_t_ge_3"
    return finite_correlation(series[:-1], series[1:])


def temporal_scores(samples: np.ndarray, truth: np.ndarray, *, prefix: str = "") -> WorldScores:
    """Temporal coherence of draws ``(S, T, C)`` against truth ``(T, C)``, per slot.

    For slot ``c`` with truth ``z`` and draws ``Z_s``:

    * ``truth_lag1``: lag-1 correlation of ``z``;
    * ``draw_lag1``: mean over draws of the lag-1 correlation of ``Z_s``;
    * ``lag1_tracking_abs_error``: ``|draw_lag1 - truth_lag1|``;
    * ``truth_error_residual_lag1``: mean over draws of the lag-1 correlation
      of ``Z_s - z``;
    * ``posterior_centered_residual_lag1``: the same for ``Z_s - mean_s Z``;
    * ``adjacent_draw_correlation``: mean over weeks ``t`` of the correlation
      across draws of ``Z[:, t]`` and ``Z[:, t + 1]``.

    Means skip constant (undefined) correlations. A key's world value averages
    the slots where it is defined and its weight counts them; with no such slot
    it is ``constant_series``. The lag-1 family needs ``T >= 3``
    (``needs_t_ge_3``); the adjacent-draw correlation needs ``T >= 2``
    (``needs_t_ge_2``) and, like the posterior-centred residual, two draws
    (``needs_n_samples_ge_2``). Any non-finite input makes every key
    ``nonfinite``.
    """
    draws, target = _draws_and_truth(samples, truth, 3)
    if draws.shape[2] < 1:
        raise ValueError("temporal scores need at least one slot")
    world = WorldScores()
    if not _all_finite(draws, target):
        return _skip_all(world, [prefix + key for key in TEMPORAL_KEYS], NONFINITE)
    n_samples, n_weeks, _ = draws.shape
    z = target.T  # [C, T]
    paths = np.moveaxis(draws, 2, 0)  # [C, S, T]
    per_slot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if n_weeks >= 3:
        truth_lag1 = _row_correlations(z[:, :-1], z[:, 1:])
        draw_lag1 = _mean_defined(*_row_correlations(paths[..., :-1], paths[..., 1:]))
        residual = paths - z[:, None, :]
        per_slot["truth_lag1"] = truth_lag1
        per_slot["draw_lag1"] = draw_lag1
        per_slot["lag1_tracking_abs_error"] = (
            np.abs(draw_lag1[0] - truth_lag1[0]),
            draw_lag1[1] & truth_lag1[1],
        )
        per_slot["truth_error_residual_lag1"] = _mean_defined(
            *_row_correlations(residual[..., :-1], residual[..., 1:])
        )
        if n_samples >= 2:
            centered = paths - paths.mean(axis=1, keepdims=True)
            per_slot["posterior_centered_residual_lag1"] = _mean_defined(
                *_row_correlations(centered[..., :-1], centered[..., 1:])
            )
    if n_weeks >= 2 and n_samples >= 2:
        # [C, T - 1, S]: week pairs as rows, draws along the correlated axis.
        per_slot["adjacent_draw_correlation"] = _mean_defined(
            *_row_correlations(
                np.swapaxes(paths[..., :-1], 1, 2), np.swapaxes(paths[..., 1:], 1, 2)
            )
        )
    for key in TEMPORAL_KEYS:
        name = prefix + key
        if key in per_slot:
            values, defined = per_slot[key]
            count = int(defined.sum())
            if count:
                world.put(name, values[defined].mean(), count)
            else:
                world.skip(name, "constant_series")
        elif key == "adjacent_draw_correlation" and n_weeks < 2:
            world.skip(name, "needs_t_ge_2")
        elif key != "adjacent_draw_correlation" and n_weeks < 3:
            world.skip(name, "needs_t_ge_3")
        else:
            world.skip(name, "needs_n_samples_ge_2")
    return world


def attribution_stats(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Per-effect errors of draws ``(S, T, E)`` against truth ``(T, E)``.

    With the posterior-mean error ``err = mean_s X - y`` and per-effect bias
    ``b_e = mean_t err``: ``crps`` is the empirical CRPS over weeks and effects,
    ``mean_bias = mean_e b_e`` (signed), ``mean_abs_error = mean_e |b_e|`` and
    ``demeaned_rmse = mean_e sqrt(mean_t (err - b_e)^2)``. Effects are scored
    individually, never summed first.
    """
    draws, target = _draws_and_truth(samples, truth, 3)
    if target.shape[1] < 1 or not _all_finite(draws, target):
        raise ValueError("attribution needs at least one effect and finite inputs")
    error = draws.mean(axis=0) - target
    bias = error.mean(axis=0)
    return {
        "crps": empirical_crps(draws, target),
        "mean_bias": float(bias.mean()),
        "mean_abs_error": float(np.abs(bias).mean()),
        "demeaned_rmse": float(np.sqrt(np.mean((error - bias) ** 2, axis=0)).mean()),
    }


def leakage_stats(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Errors of effects whose truth is dead, on the posterior mean and per draw.

    ``leakage_scaled_mae = mean_{t,e} |mean_s X - y|`` and
    ``draw_leakage_scaled_mae = mean_{s,t,e} |X - y|`` for draws ``(S, T, E)``.
    """
    draws, target = _draws_and_truth(samples, truth, 3)
    if target.shape[1] < 1 or not _all_finite(draws, target):
        raise ValueError("leakage needs at least one effect and finite inputs")
    error = draws - target
    return {
        "leakage_scaled_mae": float(np.abs(error.mean(axis=0)).mean()),
        "draw_leakage_scaled_mae": float(np.abs(error).mean()),
    }


def _graph_labels(live, size: int) -> np.ndarray:
    labels = np.asarray(live)
    if labels.shape != (size,) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"live labels must be a binary vector of shape ({size},)")
    return labels.astype(bool)


def attribution_scores(
    samples: np.ndarray,
    truth: np.ndarray,
    live: np.ndarray | None,
    *,
    live_prefix: str,
    dead_prefix: str,
) -> WorldScores:
    """Live and dead attribution of one effect family in one world.

    ``samples`` ``(S, T, E)`` and ``truth`` ``(T, E)`` hold the present effects
    and ``live`` ``(E,)`` their graph labels (``None`` without graph truth).
    Live effects get :func:`attribution_stats` as ``live_prefix + key``, dead
    ones :func:`leakage_stats` as ``dead_prefix + key``. Reasons:
    ``missing_graph_truth``, ``no_live_effects``, ``no_dead_effects``, and
    ``nonfinite`` for a group whose inputs are not all finite.
    """
    draws, target = _draws_and_truth(samples, truth, 3)
    world = WorldScores()
    groups = (
        (live_prefix, LIVE_ATTRIBUTION_KEYS, attribution_stats, "no_live_effects"),
        (dead_prefix, DEAD_ATTRIBUTION_KEYS, leakage_stats, "no_dead_effects"),
    )
    if live is None:
        for prefix, keys, _, _ in groups:
            _skip_all(world, [prefix + key for key in keys], "missing_graph_truth")
        return world
    live_mask = _graph_labels(live, target.shape[1])
    for (prefix, keys, scorer, empty), mask in zip(groups, (live_mask, ~live_mask), strict=True):
        if not mask.any():
            _skip_all(world, [prefix + key for key in keys], empty)
        elif not _all_finite(draws[..., mask], target[:, mask]):
            _skip_all(world, [prefix + key for key in keys], NONFINITE)
        else:
            for key, value in scorer(draws[..., mask], target[:, mask]).items():
                world.put(prefix + key, value)
    return world


def draw_closure(
    draw_totals: np.ndarray,
    structural: np.ndarray,
    observed: np.ndarray,
    eps: float = CLOSURE_EPS,
) -> tuple[float, float]:
    """Per-draw relative closure error: its mean and 0.9 quantile over draws.

    Draw ``s`` of totals ``D`` ``(S, T)`` scores
    ``mean_t |D_s(t) - structural(t)| / max(|observed(t)|, eps)``; the quantile
    uses NumPy's default linear method.
    """
    draws, target = _draws_and_truth(draw_totals, structural, 2)
    scale = np.maximum(np.abs(_finite_array(observed, "observed")), eps)
    if scale.shape != target.shape or not _all_finite(draws, target):
        raise ValueError("closure needs finite draw totals, structural and observed series")
    per_draw = np.mean(np.abs(draws - target) / scale, axis=1)
    return float(per_draw.mean()), float(np.quantile(per_draw, 0.9))


def r2(pred: np.ndarray, truth: np.ndarray) -> Scored:
    """``1 - sum (pred - truth)^2 / sum (truth - mean(truth))^2`` of finite vectors.

    At most 1 and possibly negative; ``(None, "constant_series")`` when
    ``truth`` has zero range.
    """
    predicted = _finite_array(pred, "pred").ravel()
    target = _finite_array(truth, "truth").ravel()
    if predicted.shape != target.shape or target.size < 1:
        raise ValueError(f"r2 needs equal non-empty vectors, got {predicted.shape} {target.shape}")
    if np.ptp(target) == 0:
        return None, "constant_series"
    total = np.sum((target - target.mean()) ** 2)
    return _checked(1.0 - np.sum((predicted - target) ** 2) / total)


def latent_sales_scores(
    draw_totals: np.ndarray,
    structural: np.ndarray,
    observed: np.ndarray,
    *,
    eps: float = CLOSURE_EPS,
) -> WorldScores:
    """Closure and fit of draw totals ``(S, T)`` against structural sales ``(T,)``.

    ``closure_rel_err`` scores the posterior-mean total
    ``mean_t |mean_s D - structural| / max(|observed|, eps)``;
    ``closure_draw_rel_err`` and ``closure_draw_q90_rel_err`` summarize
    :func:`draw_closure`; ``latent_sales_r2`` is :func:`r2` of the
    posterior-mean total. Any non-finite input makes every key ``nonfinite``.
    """
    draws, target = _draws_and_truth(draw_totals, structural, 2)
    observed_arr = np.asarray(observed, dtype=np.float64)
    if observed_arr.shape != target.shape:
        raise ValueError(f"observed {observed_arr.shape} must match structural {target.shape}")
    world = WorldScores()
    if not _all_finite(draws, target, observed_arr):
        return _skip_all(world, LATENT_SALES_KEYS, NONFINITE)
    posterior_mean = draws.mean(axis=0)
    if not np.isfinite(posterior_mean).all():  # finite draws whose sum overflowed
        return _skip_all(world, LATENT_SALES_KEYS, NONFINITE)
    scale = np.maximum(np.abs(observed_arr), eps)
    mean_draw, q90_draw = draw_closure(draws, target, observed_arr, eps)
    world.put("closure_rel_err", np.mean(np.abs(posterior_mean - target) / scale))
    world.put("closure_draw_rel_err", mean_draw)
    world.put("closure_draw_q90_rel_err", q90_draw)
    world.score("latent_sales_r2", r2(posterior_mean, target))
    return world


def time_average_coverage(
    samples: np.ndarray, truth: np.ndarray, low: float = 0.1, high: float = 0.9
) -> float:
    """Share of series whose time-mean truth lies in the central interval of draw time means.

    ``samples`` is ``(S, T, ...)`` and ``truth`` ``(T, ...)``. Per trailing
    series, the per-draw time means ``mean_t X_s`` give NumPy-linear quantiles
    ``[low, high]``, and the series is covered when ``lo <= mean_t y <= hi``.
    A single series gives 0.0 or 1.0.
    """
    draws, target = _draws_and_truth(samples, truth, np.ndim(samples))
    if draws.ndim < 2 or not _all_finite(draws, target):
        raise ValueError("time-average coverage needs finite (S, T, ...) draws")
    lo, hi = np.quantile(draws.mean(axis=1), [low, high], axis=0)
    level = target.mean(axis=0)
    return float(np.mean((lo <= level) & (level <= hi)))


def time_average_scores(
    samples: np.ndarray, truth: np.ndarray, live: np.ndarray | None, *, prefix: str = ""
) -> WorldScores:
    """80% time-average coverage of draws ``(S, T, C)``: all, live and dead slots.

    ``cov80``, ``cov80_live`` and ``cov80_dead`` (after ``prefix``) are
    :func:`time_average_coverage` over the group, weighted by its slot count.
    ``live`` ``(C,)`` holds graph labels, ``None`` without graph truth
    (``missing_graph_truth``); empty groups are ``no_live_effects`` /
    ``no_dead_effects``; non-finite inputs make every key ``nonfinite``.
    """
    draws, target = _draws_and_truth(samples, truth, 3)
    if target.shape[1] < 1:
        raise ValueError("time-average coverage needs at least one slot")
    world = WorldScores()
    all_key, live_key, dead_key = (prefix + key for key in TIME_AVERAGE_KEYS)
    if not _all_finite(draws, target):
        return _skip_all(world, (all_key, live_key, dead_key), NONFINITE)
    world.put(all_key, time_average_coverage(draws, target), target.shape[1])
    if live is None:
        return _skip_all(world, (live_key, dead_key), "missing_graph_truth")
    live_mask = _graph_labels(live, target.shape[1])
    for key, mask, empty in (
        (live_key, live_mask, "no_live_effects"),
        (dead_key, ~live_mask, "no_dead_effects"),
    ):
        if mask.any():
            world.put(key, time_average_coverage(draws[..., mask], target[:, mask]), mask.sum())
        else:
            world.skip(key, empty)
    return world


def cumulative_energy_score(samples: np.ndarray, truth: np.ndarray) -> Scored:
    """Energy score of running totals ``cumsum_t`` of draws ``(S, T, ...)``.

    ``(None, "needs_n_samples_ge_2")`` below two draws.
    """
    draws = _finite_array(samples, "samples")
    target = _finite_array(truth, "truth")
    if draws.ndim < 2 or target.shape != draws.shape[1:]:
        raise ValueError(f"expected (S, T, ...) draws and (T, ...) truth, got {draws.shape}")
    value = energy_score(np.cumsum(draws, axis=1), np.cumsum(target, axis=0))
    return (None, "needs_n_samples_ge_2") if value is None else _checked(value)


def _binary(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isin(array, (0.0, 1.0)).all():
        raise ValueError(f"{name} must hold binary 0/1 entries")
    return array


def _graph_draws(graphs, truth=None) -> tuple[np.ndarray, np.ndarray | None]:
    draws = _binary(graphs, "graph draws")
    if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] < 1:
        raise ValueError(f"graph draws must be (S, L) with S, L >= 1, got {draws.shape}")
    if truth is None:
        return draws, None
    target = _binary(truth, "true graph")
    if target.shape != draws.shape[1:]:
        raise ValueError(f"true graph {target.shape} does not match draws {draws.shape}")
    return draws, target


def categorical_brier(graphs: np.ndarray, truth: np.ndarray) -> Scored:
    """Unbiased categorical Brier score of complete graph draws ``(S, L)``.

    ``1 - 2 f + sum_k c_k (c_k - 1) / (S (S - 1))``, with ``f`` the share of draws
    equal to the true graph ``(L,)`` and ``c_k`` the counts of the distinct
    draws; it lies in ``[0, 2]``. ``(None, "needs_n_samples_ge_2")`` below two draws.
    """
    draws, target = _graph_draws(graphs, truth)
    n = draws.shape[0]
    if n < 2:
        return None, "needs_n_samples_ge_2"
    exact = np.all(draws == target, axis=1).mean()
    _, counts = np.unique(draws, axis=0, return_counts=True)
    collision = np.sum(counts * (counts - 1)) / (n * (n - 1))
    return float(1.0 - 2.0 * exact + collision), None


def graph_entropy(graphs: np.ndarray) -> float:
    """Plug-in entropy in nats of the distinct complete graphs among draws ``(S, L)``.

    It saturates at ``log S`` once every draw is distinct.
    """
    draws, _ = _graph_draws(graphs)
    _, counts = np.unique(draws, axis=0, return_counts=True)
    n = draws.shape[0]
    return float(np.sum(counts / n * np.log(n / counts)))


def pairwise_comoment(graphs: np.ndarray) -> np.ndarray:
    """Dependence of binary draws ``(S, L)`` for every slot pair ``i < j``.

    ``|P(e_i = 1, e_j = 1) - P(e_i = 1) P(e_j = 1)|`` from draw frequencies:
    0 for independent slots and at most 0.25. Empty below two slots.
    """
    draws, _ = _graph_draws(graphs)
    marginal = draws.mean(axis=0)
    joint = draws.T @ draws / draws.shape[0]
    return np.abs(joint - np.outer(marginal, marginal))[np.triu_indices(draws.shape[1], k=1)]


def _adjacency(adjacency) -> np.ndarray:
    edges = _binary(adjacency, "adjacency")
    if edges.ndim != 3 or edges.shape[1] != edges.shape[2] or edges.shape[0] < 1:
        raise ValueError(f"adjacency draws must be (S, n, n), got {edges.shape}")
    return edges > 0.5


def two_cycle_rate(adjacency: np.ndarray) -> Scored:
    """Mean over draws and unordered node pairs of ``A_ij A_ji``, for draws ``(S, n, n)``.

    ``(None, "needs_2_active_nodes")`` below two nodes.
    """
    edges = _adjacency(adjacency)
    if edges.shape[1] < 2:
        return None, "needs_2_active_nodes"
    i, j = np.triu_indices(edges.shape[1], k=1)
    return float(np.mean(edges[:, i, j] & edges[:, j, i])), None


def cycle_rate(adjacency: np.ndarray) -> Scored:
    """Share of directed draws ``(S, n, n)`` that contain a cycle.

    Uses the boolean transitive closure, so cycles of any length count.
    ``(None, "needs_2_active_nodes")`` below two nodes.
    """
    reach = _adjacency(adjacency)
    n = reach.shape[1]
    if n < 2:
        return None, "needs_2_active_nodes"
    for k in range(n):
        reach = reach | (reach[:, :, k, None] & reach[:, None, k, :])
    return float(np.diagonal(reach, axis1=1, axis2=2).any(axis=1).mean()), None


def adjacency_draws(graphs: np.ndarray, endpoints: Sequence[Sequence]) -> np.ndarray:
    """Directed adjacency draws ``(S, n, n)`` of slot draws ``(S, L)``.

    ``endpoints`` holds each slot's ``(source, target)`` node labels (any
    hashable values). The ``n`` nodes are the distinct labels in order of first
    appearance, and entry ``[s, i, j]`` is draw ``s`` of the slot from node
    ``i`` to node ``j``: zero when no slot joins them, the largest draw when
    several do.
    """
    draws = np.asarray(graphs, dtype=np.float64)
    pairs = [tuple(pair) for pair in endpoints]
    if draws.ndim != 2 or len(pairs) != draws.shape[1] or any(len(pair) != 2 for pair in pairs):
        raise ValueError(
            f"expected (S, L) slot draws and L (source, target) pairs, got {draws.shape} "
            f"and {len(pairs)} pairs"
        )
    first_seen = dict.fromkeys(node for pair in pairs for node in pair)
    nodes = {node: index for index, node in enumerate(first_seen)}
    rows = np.asarray([nodes[source] for source, _ in pairs], dtype=np.intp)
    cols = np.asarray([nodes[target] for _, target in pairs], dtype=np.intp)
    adjacency = np.zeros((draws.shape[0], len(nodes), len(nodes)))
    np.maximum.at(adjacency, (slice(None), rows, cols), draws)
    return adjacency


def adjacency_blocks(
    graphs: np.ndarray,
    families: Sequence[str],
    endpoints: Sequence[Sequence],
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    """The ``blocks`` of :func:`graph_scores`: :func:`adjacency_draws` of each named family.

    ``graphs`` ``(S, L)`` are one world's active-slot draws, ``families``
    ``(L,)`` their edge families and ``endpoints`` their ``(source, target)``
    nodes. A family's nodes are the endpoints of its active slots; a family
    without one has an empty ``(S, 0, 0)`` block.
    """
    draws = np.asarray(graphs, dtype=np.float64)
    labels = np.asarray(families, dtype=str)
    if draws.ndim != 2 or labels.shape != draws.shape[1:]:
        raise ValueError(f"expected (S, L) slot draws and L edge families, got {draws.shape}")
    pairs = list(endpoints)
    return {
        name: adjacency_draws(
            draws[:, labels == name],
            [pair for pair, family in zip(pairs, labels, strict=True) if family == name],
        )
        for name in names
    }


def graph_scores(
    graphs: np.ndarray,
    truth: np.ndarray | None,
    blocks: Mapping[str, np.ndarray],
    *,
    prefix: str = "",
) -> WorldScores:
    """Whole-graph posterior statistics of binary draws over one world's active slots.

    ``graphs`` ``(S, L)`` and ``truth`` ``(L,)`` hold the active slots. With the
    Hamming distance ``D_s`` of draw ``s`` to ``truth``: ``shd_mean``/``shd_median``
    (mean and median of ``D_s``), ``normalized_shd_mean`` (mean of ``D_s / L``),
    ``exact_recovery`` (share of ``D_s = 0``), ``energy_score``
    (:func:`energy_score`) and ``categorical_brier_score``; without graph truth
    (``truth=None``) they are ``missing_graph_truth``. The other keys need no
    truth: ``entropy`` and ``n_distinct`` of the distinct draws, the mean and
    max of :func:`pairwise_comoment` (``needs_2_active_slots``), and
    ``{name}_two_cycle_rate`` and ``{name}_cycle_rate`` of each ``blocks``
    entry, a family name mapped to its active-node adjacency draws
    ``(S, n, n)`` (:func:`adjacency_blocks`). Keys carry ``prefix``. With no
    active slot every key is ``no_active_slots``; any non-finite input makes
    every key ``nonfinite``.
    """
    draws = np.asarray(graphs, dtype=np.float64)
    target = None if truth is None else np.asarray(truth, dtype=np.float64)
    if draws.ndim != 2 or (target is not None and target.shape != draws.shape[1:]):
        raise ValueError(f"expected (S, L) graph draws and (L,) truth, got {draws.shape}")
    world = WorldScores()
    names = [*GRAPH_KEYS, *(f"{name}_{rate}" for name in blocks for rate in _CYCLE_RATES)]
    squares = {name: np.asarray(block, dtype=np.float64) for name, block in blocks.items()}
    compared = () if target is None else (target,)
    if not _all_finite(draws, *compared, *squares.values()):
        return _skip_all(world, [prefix + name for name in names], NONFINITE)
    if draws.shape[1] == 0:
        return _skip_all(world, [prefix + name for name in names], "no_active_slots")
    if target is None:
        _skip_all(world, [prefix + key for key in _TRUTH_GRAPH_KEYS], "missing_graph_truth")
    else:
        distance = np.abs(draws - target).sum(axis=1)
        world.put(prefix + "shd_mean", distance.mean())
        world.put(prefix + "exact_recovery", np.mean(distance == 0))
        world.put(prefix + "shd_median", np.median(distance))
        world.put(prefix + "normalized_shd_mean", np.mean(distance / draws.shape[1]))
        energy = energy_score(draws, target)
        if energy is None:
            world.skip(prefix + "energy_score", "needs_n_samples_ge_2")
        else:
            world.put(prefix + "energy_score", energy)
        world.score(prefix + "categorical_brier_score", categorical_brier(draws, target))
    world.put(prefix + "entropy", graph_entropy(draws))
    world.put(prefix + "n_distinct", len(np.unique(draws, axis=0)))
    comoment = pairwise_comoment(draws)
    if comoment.size:
        world.put(prefix + "pairwise_comoment_abs_err", comoment.mean())
        world.put(prefix + "pairwise_comoment_max_abs_err", comoment.max())
    else:
        comoment_keys = ("pairwise_comoment_abs_err", "pairwise_comoment_max_abs_err")
        _skip_all(world, [prefix + key for key in comoment_keys], "needs_2_active_slots")
    for name, block in squares.items():
        world.score(f"{prefix}{name}_two_cycle_rate", two_cycle_rate(block))
        world.score(f"{prefix}{name}_cycle_rate", cycle_rate(block))
    return world
