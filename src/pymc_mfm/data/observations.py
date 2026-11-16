"""Context-only representation shared by corpus training and observed inference."""

from __future__ import annotations

import hashlib

import numpy as np

from pymc_mfm.data.features import moment_features


def observation_identity(batch, world=0) -> str:
    """Hash exact encoded input bytes, excluding simulation targets and graphs.

    Raw tensor bytes preserve dtypes without depending on NumPy's support for
    bfloat16. Inference and evaluator capture use the same single-world identity.
    """
    import torch  # imported on use: the encoders below need only NumPy

    digest = hashlib.sha256()
    for key in sorted(batch):
        if key.startswith("g_") or "targets" in key:
            continue
        value = batch[key][world : world + 1].detach().cpu().contiguous()
        # PyTorch considers length-one zero-stride axes contiguous, but a dtype
        # view still requires a unit final stride.
        if value.stride(-1) != 1:
            value = value.clone(memory_format=torch.contiguous_format)
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_columns(values: np.ndarray, n_context: int, *, name: str, rms: bool) -> np.ndarray:
    """Fit a positive context scale without summing or squaring raw magnitudes."""
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(values[:, :n_context]).max(axis=1)
    nonzero = magnitude > 0
    if np.any(~nonzero & np.any(values[:, n_context:] != 0, axis=1)):
        raise ValueError(f"{name} has a zero context scale with nonzero query values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        canonical = np.divide(
            values, magnitude[:, None], out=np.zeros_like(values), where=nonzero[:, None]
        )
        context = canonical[:, :n_context]
        relative_scale = (
            np.sqrt(np.mean(context * context, axis=1)) if rms else context.mean(axis=1)
        )
        np.divide(canonical, relative_scale[:, None], out=canonical, where=nonzero[:, None])
    if not np.isfinite(canonical).all() or np.any((values != 0) & (canonical == 0)):
        raise ValueError(f"{name} cannot be represented in float64 context units")
    return canonical


def encode_observations(
    spend: np.ndarray,
    controls: np.ndarray,
    context_outcome: np.ndarray,
    active_c: np.ndarray,
    active_m: np.ndarray,
    active_j: np.ndarray,
) -> dict[str, np.ndarray]:
    """Encode validated batched observations without targets or generating graphs.

    Shapes are ``[B,T,K]``, ``[B,T,M]``, ``[B,C]`` and node masks ``[B,D]``.
    The caller validates support and padding. Context p95 absolute outcome,
    per-channel mean spend and per-control RMS define the unit system before
    any features are computed. Query observations use those same scales.
    All-zero columns stay zero; zero context with nonzero query is unsupported.
    Both data entry points use this exact float64 path. The original outcome
    scale is returned separately as metadata; query outcomes never enter it.
    """
    b, n_weeks, k = spend.shape
    m = controls.shape[-1]
    n_context = context_outcome.shape[1]
    spend = _canonical_columns(spend, n_context, name="spend", rms=False)
    controls = _canonical_columns(controls, n_context, name="controls", rms=True)
    context_outcome = np.asarray(context_outcome, dtype=np.float64)
    scale = np.quantile(np.abs(context_outcome), 0.95, axis=1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        canonical_outcome = context_outcome / scale[:, None]
    if not np.isfinite(canonical_outcome).all() or np.any(canonical_outcome <= 0):
        raise ValueError("context_outcome cannot be represented in float64 context units")
    log_spend = np.log1p(spend)
    spend_mean = log_spend[:, :n_context].mean(axis=1)
    spend_std = log_spend[:, :n_context].std(axis=1)
    spend_z = (log_spend - spend_mean[:, None]) / np.maximum(spend_std[:, None], 1e-6)
    log_outcome = np.log(canonical_outcome)
    revenue_z = log_outcome - log_outcome.mean(axis=1, keepdims=True)
    controls_mean = controls[:, :n_context].mean(axis=1)
    controls_std = controls[:, :n_context].std(axis=1)
    controls_std = np.where(controls_std > 1e-3, controls_std, 1.0)
    controls_z = np.clip((controls - controls_mean[:, None]) / controls_std[:, None], -10, 10)
    week = np.arange(n_weeks, dtype=np.float32) / np.float32(n_weeks)
    weeks = np.broadcast_to(week[None, :, None], (b, n_weeks, 1)).copy()
    encoded = {
        "ctx_spend_z": spend_z[:, :n_context],
        "qry_spend_z": spend_z[:, n_context:],
        "ctx_controls_z": controls_z[:, :n_context],
        "qry_controls_z": controls_z[:, n_context:],
        "ctx_revenue_z": revenue_z[..., None],
        "spend_mean": spend_mean,
        "spend_std": spend_std,
        "ctx_week": weeks[:, :n_context],
        "qry_week": weeks[:, n_context:],
        "active_c_mask": active_c,
        "active_m_mask": active_m,
        "active_j_mask": active_j,
        "sales_scale": scale,
    }
    features = [
        moment_features(
            {
                "spend": spend[i, :n_context],
                "revenue": log_outcome[i],
                "controls": controls[i, :n_context],
            },
            k,
            m,
        )
        for i in range(b)
    ]
    encoded.update({key: np.stack([row[key] for row in features]) for key in features[0]})
    for key in ("control_moments", "control_location_scale", "control_effect_moments"):
        encoded[key] *= active_m[..., None]
    return encoded
