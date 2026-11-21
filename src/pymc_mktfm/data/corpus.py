"""Adapt current ``pymc_generator`` corpora to the canonical model tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pymc_mktfm.data.observations import encode_observations
from pymc_mktfm.model.graph import EdgeSlotLayout

#: Generator corpus schemas this library consumes. Schema 5 adds trajectory leaves and opt-in
#: truth arrays; every array read here is unchanged from schema 4.
SUPPORTED_CORPUS_SCHEMAS = (4, 5)

_GENERATOR_EDGE_TYPES = ("cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz")


def corpus_batch(
    corpus: Mapping[str, Any],
    rows: Sequence[int] | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Prepare selected schema-4 or schema-5 worlds without reading future outcome for features.

    Generate with ``pymc_generator.DataGenerator.generate`` or load with
    ``pymc_generator.load_corpus`` and validate with its ``validate_corpus``.
    This boundary checks the consumed arrays of selected rows, not the entire
    corpus's generation diagnostics or causal decomposition identities.

    Rows must share a nonempty support prefix and nonempty query suffix; group
    worlds by ``support_mask.sum(axis=1)`` before batching. Row order and repeats
    are preserved. No train/validation split or row filtering happens here.

    Model inputs and targets are float32 tensors on ``device``; ``sales_scale``
    is float64 CPU metadata, not a neural input. Decomposition targets exclude
    observation noise, keep signed components, and share the context p95
    absolute-outcome ruler. Decode in float64 with that ruler to restore natural
    units. Spend and controls are scaled by their context mean and RMS before
    feature construction. Statistics use only observed context. Treatments and
    direct contributions must be nonnegative; context outcomes must be positive.
    A column with zero context but nonzero query has no context-fitted scale
    and is rejected. Other decomposition components may be signed.

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
        raise ValueError("corpus schema_version must be an integer")
    if version not in SUPPORTED_CORPUS_SCHEMAS:
        raise ValueError(
            f"corpus_batch supports schema_version {SUPPORTED_CORPUS_SCHEMAS}; "
            "use the generator's corpus reader"
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
    spend = values("treatment_raw", (n_weeks, k))
    controls = values("covariates", (n_weeks, m))
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

    encoded = encode_observations(
        spend,
        controls,
        outcome[:, :n_context],
        active_c,
        active_m,
        active_j,
    )
    scale = encoded["sales_scale"][:, None, None]
    # Normalize each latent before summing: finite signed raw components can
    # otherwise overflow during reduction even when their total is representable.
    targets = np.concatenate(
        (
            baseline[..., None] / scale,
            direct / scale,
            covariate / scale,
            (latent / scale).sum(axis=-1, keepdims=True),
            indirect / scale,
        ),
        axis=-1,
    )
    encoded["context_targets_z"] = targets[:, :n_context]
    encoded["targets_z"] = targets[:, n_context:]
    encoded.update({f"g_{key}": value for key, value in layout.unpack(graph).items()})
    result = {}
    for key, array in encoded.items():
        dtype = np.float64 if key == "sales_scale" else np.float32
        array = np.ascontiguousarray(array, dtype=dtype)
        if not np.isfinite(array).all():
            raise ValueError(f"{key} exceeds the finite MFM representation")
        result[key] = torch.from_numpy(array).to("cpu" if key == "sales_scale" else device)
    return result
