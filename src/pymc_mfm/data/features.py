"""Context-only regression features for the canonical cell-grid encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache

import numpy as np

RESPONSE_MOMENT_DECAYS = (0.0, 0.3, 0.7)
RESPONSE_MOMENT_L_MAX = 4
N_RESPONSE_MOMENTS = 4
ADSTOCK_MOMENT_DECAYS = (0.3, 0.5, 0.7)
ADSTOCK_MOMENT_L_MAX = 8
ADSTOCK_MOMENTS_PER_DECAY = 3
N_ADSTOCK_MOMENTS = len(ADSTOCK_MOMENT_DECAYS) * ADSTOCK_MOMENTS_PER_DECAY
ADSTOCK_LEVEL_LIMIT = 50.0
N_CONTROL_MOMENTS = len(ADSTOCK_MOMENT_DECAYS)
CONTROL_MOMENT_T_DIVISOR = 4.0
N_CONTROL_LOCATION_SCALE = 2
CONTROL_EFFECT_MOMENTS_PER_DECAY = 2
N_CONTROL_EFFECT_MOMENTS = len(ADSTOCK_MOMENT_DECAYS) * CONTROL_EFFECT_MOMENTS_PER_DECAY

MEDIA_FEATURE_BLOCKS = (
    ("response_moments", N_RESPONSE_MOMENTS),
    ("adstock_moments", N_ADSTOCK_MOMENTS),
)
CONTROL_FEATURE_BLOCKS = (
    ("control_moments", N_CONTROL_MOMENTS),
    ("control_location_scale", N_CONTROL_LOCATION_SCALE),
    ("control_effect_moments", N_CONTROL_EFFECT_MOMENTS),
)
MOMENT_FEATURE_BLOCKS = MEDIA_FEATURE_BLOCKS + CONTROL_FEATURE_BLOCKS

# Exact fits carry no information about coefficient significance.
_EXACT_FIT_VARIANCE_RATIO = 1e-24


def _carryover_weights(decay: float, l_max: int) -> np.ndarray:
    weights = decay ** np.arange(l_max + 1, dtype=np.float64)
    return weights / weights.sum()


@lru_cache(maxsize=64)
def _carryover_operator(decays: tuple[float, ...], l_max: int, n_weeks: int) -> np.ndarray:
    """Stacked normalized geometric operators, shared by context geometry."""
    lag = np.arange(n_weeks)[:, None] - np.arange(n_weeks)[None, :]
    in_band = (lag >= 0) & (lag <= l_max)
    clipped = np.clip(lag, 0, l_max)
    operator = np.concatenate(
        [np.where(in_band, _carryover_weights(decay, l_max)[clipped], 0.0) for decay in decays],
        axis=0,
    )
    operator.setflags(write=False)
    return operator


def _carryover_stack(series: np.ndarray, decays: tuple[float, ...], l_max: int) -> np.ndarray:
    """Carryover of every column at every decay, shape ``(len(decays), T, C)``."""
    n_weeks, n_series = series.shape
    operator = _carryover_operator(decays, l_max, n_weeks)
    return (operator @ series).reshape(len(decays), n_weeks, n_series)


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _fit_and_t_statistics(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and t-statistics from one SVD, with separate rank cutoffs.

    The solve uses the ``lstsq(rcond=None)`` cutoff; coefficient variances use
    the Gram-matrix pseudoinverse cutoff. Exact fits have zero significance,
    including underdetermined designs and constant responses.
    """
    n_rows, n_columns = design.shape
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.0

    solve_keep = singular > np.finfo(np.float64).eps * max(n_rows, n_columns) * largest
    inverse_singular = np.where(solve_keep, 1.0 / np.where(solve_keep, singular, 1.0), 0.0)
    beta = right.T @ (inverse_singular * (left.T @ y))

    residual = y - design @ beta
    dof = max(n_rows - n_columns, 1)
    sigma2 = float(residual @ residual) / dof
    if sigma2 <= _EXACT_FIT_VARIANCE_RATIO * max(float(y @ y) / n_rows, 1.0):
        return beta, np.zeros_like(beta)

    gram_keep = singular * singular > np.finfo(np.float64).eps * n_columns * largest * largest
    scaled = np.where(gram_keep[:, None], right / np.where(gram_keep, singular, 1.0)[:, None], 0.0)
    variance = np.einsum("ki,ki->i", scaled, scaled) * sigma2
    standard_error = np.sqrt(np.maximum(variance, 1e-300))
    return beta, beta / standard_error


def _usable_spend_channels_from_std(spend_std: np.ndarray, n_channels: int | None) -> list[int]:
    width = spend_std.size if n_channels is None else min(n_channels, spend_std.size)
    return [channel for channel in range(width) if float(spend_std[channel]) >= 1e-12]


def moment_features(
    context: Mapping[str, np.ndarray], n_channels: int, n_controls: int
) -> dict[str, np.ndarray]:
    """Compute all five static feature blocks from observed context only.

    Required inputs are natural ``spend`` with shape ``(T, C)``, log natural
    sales in ``revenue`` with shape ``(T,)``, and raw ``controls`` with shape
    ``(T, M)``. Neither query observations nor decomposition or graph truth are
    read. Activity masks belong to the batch boundary and encoder; callers
    handle metadata masks rather than including them in this mapping.

    Output arrays are float32, in encoder column order:

    * ``response_moments`` (C, 4): control-residualized log-sales correlations
      at decays 0, 0.3 and 0.7, followed by a bounded standardized slope.
    * ``adstock_moments`` (C, 9): for decays 0.3, 0.5 and 0.7, the signed
      log1p carried mean, bounded partial slope and clipped implied level.
    * ``control_moments`` (M, 3): bounded absolute coefficient t-statistics
      from those same three joint natural-sales regressions.
    * ``control_location_scale`` (M, 2): raw mean and the normalization scale,
      with standard deviations at or below 1e-3 replaced by one.
    * ``control_effect_moments`` (M, 6): signed bounded standardized slope
      and clipped implied level at each of the three regression decays.

    Natural-sales regression features share the context p95 absolute-sales
    ruler, floored at 1e-6. Uninformative moment rows remain zero.
    """
    spend = np.asarray(context["spend"], dtype=np.float64)
    log_sales = np.asarray(context["revenue"], dtype=np.float64).reshape(-1)
    controls = np.asarray(context["controls"], dtype=np.float32)
    if controls.shape != (log_sales.size, n_controls):
        raise ValueError(
            "control representation requires context controls with shape "
            f"{(log_sales.size, n_controls)}, got {controls.shape}"
        )
    c_mean = controls.mean(axis=0)
    c_std = controls.std(axis=0)
    c_std = np.where(c_std > 1e-3, c_std, 1.0)

    response = np.zeros((n_channels, N_RESPONSE_MOMENTS), dtype=np.float32)
    adstock = np.zeros((n_channels, N_ADSTOCK_MOMENTS), dtype=np.float32)
    control = np.zeros((n_controls, N_CONTROL_MOMENTS), dtype=np.float32)
    control_effect = np.zeros((n_controls, N_CONTROL_EFFECT_MOMENTS), dtype=np.float32)
    features = {
        "response_moments": response,
        "adstock_moments": adstock,
        "control_moments": control,
        "control_location_scale": np.column_stack((c_mean, c_std)).astype("float32"),
        "control_effect_moments": control_effect,
    }
    if spend.ndim != 2 or spend.shape[0] != log_sales.size or log_sales.size < 3:
        return features

    spend_col_std = spend.std(axis=0)
    controls_mat = np.asarray(context["controls"], dtype=np.float64)
    control_col_std = controls_mat.std(axis=0)

    design: list[np.ndarray] = [np.ones(log_sales.size, dtype=np.float64), controls_mat]
    matrix = np.column_stack(design)
    coefficients, *_ = np.linalg.lstsq(matrix, log_sales, rcond=None)
    residual = log_sales - matrix @ coefficients
    residual_sd = float(residual.std())
    response_std = spend_col_std[: min(n_channels, spend.shape[1])]
    usable_response = np.flatnonzero(np.isfinite(response_std) & (response_std >= 1e-12))
    if math.isfinite(residual_sd) and residual_sd >= 1e-12 and usable_response.size:
        columns = spend[:, usable_response]
        carried = _carryover_stack(columns, RESPONSE_MOMENT_DECAYS, RESPONSE_MOMENT_L_MAX)
        carried_centred = carried - carried.mean(axis=1, keepdims=True)
        carried_ss = np.einsum("dtc,dtc->dc", carried_centred, carried_centred)
        residual_centred = residual - residual.mean()
        residual_ss = float(residual_centred @ residual_centred)
        denominator = np.sqrt(carried_ss * residual_ss)
        correlation = np.divide(
            np.einsum("dtc,t->dc", carried_centred, residual_centred),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator >= 1e-12,
        )
        correlation[np.sqrt(carried_ss / log_sales.size) < 1e-12] = 0.0

        columns_centred = columns - columns.mean(axis=0)
        slope_denominator = np.einsum("tc,tc->c", columns_centred, columns_centred)
        slope = np.divide(
            np.einsum("tc,t->c", columns_centred, residual),
            slope_denominator,
            out=np.zeros_like(slope_denominator),
            where=slope_denominator > 0.0,
        )
        rows = np.empty((usable_response.size, N_RESPONSE_MOMENTS), dtype=np.float64)
        rows[:, : len(RESPONSE_MOMENT_DECAYS)] = correlation.T
        rows[:, 3] = np.tanh(slope * spend_col_std[usable_response] / residual_sd)
        rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        rows[:, :3] = np.clip(rows[:, :3], -1.0, 1.0)
        response[usable_response] = rows.astype(np.float32)

    sales = np.exp(log_sales)
    if not np.isfinite(sales).all():
        return features
    sales_scale = max(float(np.quantile(np.abs(sales), 0.95)), 1e-6)
    n_weeks = sales.size
    usable_adstock = _usable_spend_channels_from_std(spend_col_std, n_channels)
    usable_control = _usable_spend_channels_from_std(spend_col_std, None)
    shared_usable = usable_adstock == usable_control

    adstock_stack: np.ndarray | None = None
    if shared_usable and usable_adstock:
        adstock_stack = _carryover_stack(
            spend[:, usable_adstock], ADSTOCK_MOMENT_DECAYS, ADSTOCK_MOMENT_L_MAX
        )
        control_stack = adstock_stack
    else:
        if usable_adstock:
            adstock_stack = _carryover_stack(
                spend[:, usable_adstock], ADSTOCK_MOMENT_DECAYS, ADSTOCK_MOMENT_L_MAX
            )
        control_stack = (
            _carryover_stack(spend[:, usable_control], ADSTOCK_MOMENT_DECAYS, ADSTOCK_MOMENT_L_MAX)
            if usable_control
            else np.zeros((len(ADSTOCK_MOMENT_DECAYS), n_weeks, 0))
        )
    adstock_series_sd = None if adstock_stack is None else adstock_stack.std(axis=1)
    adstock_series_mean = None if adstock_stack is None else adstock_stack.mean(axis=1)
    control_col_mean = controls_mat.mean(axis=0)

    nuisance_design: np.ndarray | None = None
    if adstock_stack is not None:
        nuisance_design = np.column_stack([np.ones(n_weeks, dtype=np.float64), controls_mat])
    n_nuisance = 0 if nuisance_design is None else nuisance_design.shape[1]

    for index in range(len(ADSTOCK_MOMENT_DECAYS)):
        carried_adstock = None if adstock_stack is None else adstock_stack[index]
        fit: np.ndarray | None = None
        statistics: np.ndarray | None = None
        if carried_adstock is not None and nuisance_design is not None:
            matrix = np.column_stack([nuisance_design, carried_adstock])
            if shared_usable:
                fit, statistics = _fit_and_t_statistics(sales, matrix)
            else:
                fit, *_ = np.linalg.lstsq(matrix, sales, rcond=None)

        if carried_adstock is not None and fit is not None:
            assert adstock_series_sd is not None and adstock_series_mean is not None
            betas = fit[n_nuisance:]
            base = index * ADSTOCK_MOMENTS_PER_DECAY
            for position, channel in enumerate(usable_adstock):
                series_sd = float(adstock_series_sd[index, position])
                if series_sd < 1e-12:
                    continue
                beta = float(betas[position])
                mean_over_scale = float(adstock_series_mean[index, position]) / sales_scale
                row = (
                    _signed_log1p(mean_over_scale),
                    math.tanh(beta * series_sd / sales_scale),
                    float(
                        np.clip(beta * mean_over_scale, -ADSTOCK_LEVEL_LIMIT, ADSTOCK_LEVEL_LIMIT)
                    ),
                )
                if all(math.isfinite(value) for value in row):
                    adstock[channel, base : base + ADSTOCK_MOMENTS_PER_DECAY] = row

        if statistics is None:
            matrix = np.column_stack([np.ones(n_weeks), controls_mat, control_stack[index]])
            fit, statistics = _fit_and_t_statistics(sales, matrix)
        assert fit is not None
        for control_idx in range(min(n_controls, controls_mat.shape[1])):
            if float(control_col_std[control_idx]) < 1e-12:
                continue
            value = math.tanh(abs(float(statistics[1 + control_idx])) / CONTROL_MOMENT_T_DIVISOR)
            if math.isfinite(value):
                control[control_idx, index] = value
            beta = float(fit[1 + control_idx])
            effect_row = (
                math.tanh(beta * float(control_col_std[control_idx]) / sales_scale),
                float(
                    np.clip(
                        beta * float(control_col_mean[control_idx]) / sales_scale,
                        -ADSTOCK_LEVEL_LIMIT,
                        ADSTOCK_LEVEL_LIMIT,
                    )
                ),
            )
            if all(math.isfinite(value) for value in effect_row):
                base = index * CONTROL_EFFECT_MOMENTS_PER_DECAY
                control_effect[control_idx, base : base + CONTROL_EFFECT_MOMENTS_PER_DECAY] = (
                    effect_row
                )

    return features
