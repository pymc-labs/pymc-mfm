"""Adapt current ``pymc_generator`` corpora to the canonical model tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pymc_mfm.data.features import moment_features
from pymc_mfm.model.graph import EdgeSlotLayout

_GENERATOR_EDGE_TYPES = ("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz")


def corpus_batch(
    corpus: Mapping[str, Any],
    rows: Sequence[int] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Prepare selected schema-4 worlds without reading future outcome for features.

    Generate with ``pymc_generator.DataGenerator.generate`` or load with
    ``pymc_generator.load_corpus`` and validate with its ``validate_corpus``.
    This boundary checks the consumed arrays of selected rows, not the entire
    corpus's generation diagnostics or causal decomposition identities.

    Rows must share a nonempty support prefix and nonempty query suffix; group
    worlds by ``support_mask.sum(axis=1)`` before batching. Row order and repeats
    are preserved. No train/validation split or row filtering happens here.

    All outputs are float32 tensors. Decomposition targets exclude observation
    noise, keep signed components, and share the context p95 absolute-outcome
    ruler returned as ``sales_scale``. Multiply draws by
    ``batch["sales_scale"][None, :, None, None]`` to restore natural units.
    Statistics and all five moment blocks use only observed context. Treatments
    and direct treatment contributions must be nonnegative; observed outcomes
    must be strictly positive for the fixed log representation. Unsupported
    values are rejected, never clipped. Covariate zscores retain the canonical
    [-10, 10] clipping; other decomposition components may be signed.

    The generator's dy/zy slots address outcome Y. Their packed positions map to
    the checkpoint's internal db/zb names; this does not relabel their targets as
    effects on the intrinsic intercept. The model/checkpoint geometry is unchanged.
    No generator/PyMC import is needed to adapt an already loaded corpus.
    """
    diagnostics = corpus.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("corpus diagnostics must declare schema_version and edge_types")
    version = diagnostics.get("schema_version")
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)):
        raise ValueError("corpus schema_version must be integer 4")
    if version != 4:
        raise ValueError(
            "corpus_batch supports schema_version 4; use the generator's corpus reader"
        )
    edge_types = diagnostics.get("edge_types")
    if not isinstance(edge_types, (list, tuple)) or tuple(edge_types) != _GENERATOR_EDGE_TYPES:
        raise ValueError(f"corpus edge_types must be {_GENERATOR_EDGE_TYPES}")

    treatment_shape = np.shape(corpus["treatment_raw"])
    covariate_shape = np.shape(corpus["covariates"])
    latent_shape = np.shape(corpus["latent_active_mask"])
    if len(treatment_shape) != 3 or len(covariate_shape) != 3 or len(latent_shape) != 2:
        raise ValueError("treatments/covariates must have shape [N,T,D], latent mask [N,J]")
    n_worlds, n_weeks, k = treatment_shape
    if covariate_shape[:2] != (n_worlds, n_weeks) or latent_shape[0] != n_worlds:
        raise ValueError("corpus arrays must share world and time dimensions")
    m, j = covariate_shape[2], latent_shape[1]
    if n_worlds < 1 or n_weeks < 2 or min(k, m, j) < 1:
        raise ValueError(
            "MFM requires worlds, context/query weeks, and positive padded K/M/J sizes"
        )

    if rows is None:
        selected = np.arange(n_worlds)
    else:
        selected_rows = []
        for row in rows:
            if isinstance(row, (bool, np.bool_)):
                raise ValueError("rows must contain integer world indices, not booleans")
            try:
                row = index(row)
            except TypeError as error:
                raise ValueError("rows must contain integer world indices") from error
            if not 0 <= row < n_worlds:
                raise ValueError(f"world index {row} is outside [0, {n_worlds})")
            selected_rows.append(row)
        if not selected_rows:
            raise ValueError("rows must select at least one world")
        selected = np.asarray(selected_rows, dtype=np.intp)

    def values(key: str, shape: tuple[int, ...], *, binary: bool = False) -> np.ndarray:
        array = np.asarray(corpus[key])
        expected = (n_worlds, *shape)
        if array.shape != expected:
            raise ValueError(f"{key} must have shape {expected}, got {array.shape}")
        array = array[selected]
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"{key} must contain finite real values")
        if binary and not np.isin(array, (0, 1)).all():
            raise ValueError(f"{key} must be binary")
        return array

    support = values("support_mask", (n_weeks,), binary=True).astype(bool)
    context_lengths = support.sum(axis=1)
    if np.any((context_lengths == 0) | (context_lengths == n_weeks)):
        raise ValueError("support_mask needs nonempty context and query windows")
    if not np.array_equal(support, np.arange(n_weeks) < context_lengths[:, None]):
        raise ValueError("support_mask must be a contiguous context prefix followed by query")
    if not np.all(context_lengths == context_lengths[0]):
        raise ValueError(
            "batch rows must share a context length; group by support_mask.sum(axis=1)"
        )
    n_context = int(context_lengths[0])

    active_c = values("treatment_active_mask", (k,), binary=True)
    active_m = values("covariate_active_mask", (m,), binary=True)
    active_j = values("latent_active_mask", (j,), binary=True)
    if np.any(active_c.sum(axis=1) == 0) or np.any(active_m.sum(axis=1) == 0):
        raise ValueError("each world needs at least one active treatment and covariate")
    spend = values("treatment_raw", (n_weeks, k)).astype(np.float32)
    controls = values("covariates", (n_weeks, m)).astype(np.float32)
    outcome = values("outcome_raw", (n_weeks,)).astype(np.float64)
    if np.any(spend < 0):
        raise ValueError("treatment_raw must be nonnegative for the MFM representation")
    if np.any(outcome[:, :n_context] <= 0):
        raise ValueError("context outcome_raw must be strictly positive for log observations")

    baseline = values("baseline_intrinsic", (n_weeks,)).astype(np.float64)
    direct = values("treatment_contribution_raw", (n_weeks, k)).astype(np.float64)
    if np.any(direct < 0):
        raise ValueError("treatment_contribution_raw must be nonnegative for the MFM support")
    covariate = values("covariate_contribution", (n_weeks, m)).astype(np.float64)
    latent = values("latent_unobserved_contribution", (n_weeks, j)).astype(np.float64)
    indirect = values("indirect_effects_by_source", (n_weeks, 3)).astype(np.float64)
    for key, array, mask in (
        ("treatment_raw", spend, active_c),
        ("covariates", controls, active_m),
        ("treatment_contribution_raw", direct, active_c),
        ("covariate_contribution", covariate, active_m),
        ("latent_unobserved_contribution", latent, active_j),
    ):
        if np.any(np.any(array != 0, axis=1) & (mask == 0)):
            raise ValueError(f"{key} must be zero in inactive padded slots")

    layout = EdgeSlotLayout(k, m, j)
    graph = values("g", (layout.n_slots,), binary=True)
    if np.any(graph > layout.active_slot_mask(active_c, active_m, active_j)):
        raise ValueError("g has an edge touching an inactive padded node")

    log_spend = np.log1p(spend)
    spend_mean = log_spend[:, :n_context].mean(axis=1)
    spend_std = np.maximum(log_spend[:, :n_context].std(axis=1), 1e-6)
    spend_z = (log_spend - spend_mean[:, None]) / spend_std[:, None]
    log_outcome = np.log(outcome[:, :n_context]).astype(np.float32)
    revenue_z = log_outcome - log_outcome.mean(axis=1, keepdims=True)
    controls_mean = controls[:, :n_context].mean(axis=1)
    controls_std = controls[:, :n_context].std(axis=1)
    controls_std = np.where(controls_std > 1e-3, controls_std, 1.0)
    controls_z = np.clip((controls - controls_mean[:, None]) / controls_std[:, None], -10, 10)
    scale = np.maximum(np.quantile(np.abs(outcome[:, :n_context]), 0.95, axis=1), 1e-6)
    targets = (
        np.concatenate(
            (baseline[..., None], direct, covariate, latent.sum(axis=-1, keepdims=True), indirect),
            axis=-1,
        )
        / scale[:, None, None]
    )
    week = np.arange(n_weeks, dtype=np.float32) / np.float32(n_weeks)
    weeks = np.broadcast_to(week[None, :, None], (len(selected), n_weeks, 1)).copy()
    encoded = {
        "ctx_spend_z": spend_z[:, :n_context],
        "qry_spend_z": spend_z[:, n_context:],
        "ctx_controls_z": controls_z[:, :n_context],
        "qry_controls_z": controls_z[:, n_context:],
        "ctx_revenue_z": revenue_z[..., None],
        "spend_mean": (spend_mean - 9.5) / 1.5,
        "spend_std": (spend_std - 0.55) / 0.25,
        "ctx_week": weeks[:, :n_context],
        "qry_week": weeks[:, n_context:],
        "active_c_mask": active_c,
        "active_m_mask": active_m,
        "active_j_mask": active_j,
        "context_targets_z": targets[:, :n_context],
        "targets_z": targets[:, n_context:],
        "sales_scale": scale,
    }
    features = [
        moment_features(
            {
                "spend": spend[b, :n_context],
                "revenue": log_outcome[b],
                "controls": controls[b, :n_context],
            },
            k,
            m,
        )
        for b in range(len(selected))
    ]
    encoded.update({key: np.stack([row[key] for row in features]) for key in features[0]})
    encoded.update({f"g_{key}": value for key, value in layout.unpack(graph).items()})
    result = {}
    for key, array in encoded.items():
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{key} exceeds the finite float32 MFM representation")
        result[key] = torch.from_numpy(array).to(device)
    return result
