"""Small observation-only boundary to the canonical conditional decoder."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import torch

from pymc_mfm.data.observations import encode_observations, observation_identity
from pymc_mfm.inference.result import DecompositionResult, _numpy, result_from_batch
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM


def observation_batch(
    spend,
    controls,
    context_outcome,
    *,
    model_config: ModelConfig,
    channel_mask=None,
    control_mask=None,
    latent_mask=None,
    device="cpu",
) -> dict[str, torch.Tensor]:
    """Prepare one observed world without a corpus, graph truth or dummy targets.

    Spend/controls span context and query: ``[T,K]`` / ``[T,M]``. Positive
    outcomes contain only the context prefix ``[C]``, with ``0 < C < T``.
    Controls preserve the artifact's padded width; supply explicit masks for
    padding. Omitted masks mean represented nodes, NOT known causal activity.
    Latent slots are architectural, not claimed observed confounder counts.
    Rows are equally spaced observations; time encoding is the trained relative
    horizon position. Outcomes must be positive and spend nonnegative.
    """

    def numeric(value, name, ndim, dtype):
        array = np.asarray(value)
        if array.ndim != ndim or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite real {ndim}-dimensional array")
        with np.errstate(over="ignore"):
            array = array.astype(dtype)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} exceeds the model representation")
        return array

    spend = numeric(spend, "spend", 2, np.float32)
    controls = numeric(controls, "controls", 2, np.float32)
    outcome = numeric(context_outcome, "context_outcome", 1, np.float64)
    t, k = spend.shape
    if t < 2 or k < 1 or controls.shape != (t, model_config.n_controls):
        raise ValueError("spend/controls must share time and the artifact's control width")
    if not 0 < len(outcome) < t:
        raise ValueError("context outcomes need a nonempty prefix and nonempty query suffix")
    if np.any(spend < 0) or np.any(outcome <= 0):
        raise ValueError("spend must be nonnegative and context outcomes strictly positive")

    def mask(value, name, size, require_live):
        array = np.ones(size, dtype=bool) if value is None else np.asarray(value)
        if array.shape != (size,) or not np.isin(array, (0, 1)).all():
            raise ValueError(f"{name} must be a binary vector of length {size}")
        array = array.astype(bool)
        if require_live and not array.any():
            raise ValueError(f"{name} needs at least one represented entity")
        return array

    active_c = mask(channel_mask, "channel_mask", k, True)
    active_m = mask(control_mask, "control_mask", model_config.n_controls, True)
    active_j = mask(latent_mask, "latent_mask", model_config.n_demand, False)
    if np.any(spend[:, ~active_c]) or np.any(controls[:, ~active_m]):
        raise ValueError("padded observation columns must be zero")
    encoded = encode_observations(
        spend[None], controls[None], outcome[None], active_c[None], active_m[None], active_j[None]
    )
    tensors = {}
    for key, value in encoded.items():
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.ascontiguousarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{key} exceeds the finite model representation")
        tensors[key] = torch.from_numpy(value).to(device)
    return tensors


def decompose(
    model: MFM,
    observations: Mapping[str, object],
    *,
    n_samples: int | None = None,
    window: str = "query",
    mode: str = "model_average",
    graph=None,
    router_samples: int | None = None,
    channel_names=None,
    control_names=None,
    time=None,
    units: str = "outcome",
    provenance=None,
) -> DecompositionResult:
    """Sample aligned conditional evidence from observed inputs or an encoded batch.

    ``observations`` is either an observation_batch result for one world, or
    keyword inputs ``spend``, ``controls``, ``context_outcome`` and optional masks.
    Fixed graphs are supplied hypotheses; modal graphs are empirical router modes,
    not thresholded auxiliary-head probabilities. This call does not promise
    calibrated business attribution, predictive observation noise, or a paired
    context/query posterior. The caller controls the PyTorch random seed.
    """
    parameter = next(model.parameters())
    if "ctx_spend_z" in observations:
        batch = dict(observations)
        if batch["ctx_spend_z"].shape[0] != 1:
            raise ValueError("decompose accepts one observed world at a time")
        if any(not isinstance(value, torch.Tensor) for value in batch.values()):
            raise ValueError("encoded observation values must be tensors")
        batch = {
            key: value.to(device=parameter.device, dtype=parameter.dtype)
            for key, value in batch.items()
        }
    else:
        batch = observation_batch(
            **observations, model_config=model.config, device=parameter.device
        )
        batch = {key: value.to(dtype=parameter.dtype) for key, value in batch.items()}
    if graph is not None:
        # Validate the caller's precision before the model converts graph dtype.
        supplied_graph = graph if isinstance(graph, torch.Tensor) else np.asarray(graph)
        graph = torch.as_tensor(supplied_graph)
        if not ((graph == 0) | (graph == 1)).all():
            raise ValueError("supplied graph must be binary before dtype conversion")
        graph = graph.to(device=parameter.device, dtype=parameter.dtype)
        if graph.ndim == 1:
            graph = graph[None]
    identity = dict(getattr(model, "provenance", {}))
    artifact = getattr(model, "artifact_metadata", None)
    if artifact:
        identity["artifact"] = artifact
    identity.update(provenance or {})
    # Hash only the representation read by inference; never write observations.
    identity["data_sha256"] = observation_identity(batch)
    identity["head"] = "conditional"
    support_arrays = {
        key: (_numpy(batch[key][0]), "encoded context-normalized representation")
        for key in ("ctx_spend_z", "qry_spend_z", "ctx_controls_z", "qry_controls_z")
    }
    if "spend" in observations:
        context_length = batch["ctx_spend_z"].shape[1]
        for name in ("spend", "controls"):
            values = np.asarray(observations[name])
            support_arrays[f"ctx_{name}_raw"] = (values[:context_length], "native input units")
            support_arrays[f"qry_{name}_raw"] = (values[context_length:], "native input units")
        support_arrays["ctx_outcome_raw"] = (np.asarray(observations["context_outcome"]), units)
    identity["input_support"] = {}
    for key, (values, input_units) in support_arrays.items():
        values = values.astype(np.float64)
        identity["input_support"][key] = {
            "units": input_units,
            "n": int(values.shape[0]),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
        }
    if time is not None:
        time = np.asarray(time)
        full_length = batch["ctx_spend_z"].shape[1] + batch["qry_spend_z"].shape[1]
        if (
            time.shape != (full_length,)
            or time.dtype.kind not in "iuf"
            or not np.isfinite(time).all()
        ):
            raise ValueError("time must label every context/query row with finite numeric values")
        if (time[1:] <= time[:-1]).any():
            raise ValueError("time labels must be strictly increasing")
    was_training = model.training
    try:
        model.eval()
        components, graphs = model.sample_disentangled(
            batch,
            n_samples=n_samples,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
            window=window,
        )
    finally:
        model.train(was_training)
    result = result_from_batch(
        batch,
        components,
        graphs,
        model_config=model.config,
        window=window,
        mode=mode,
        provenance=identity,
        channel_names=channel_names,
        control_names=control_names,
        units=units,
        router_samples=(components.shape[0] if router_samples is None else router_samples)
        if mode == "map"
        else None,
    )
    if time is not None:
        selected = (
            time[: result.context_length] if window == "context" else time[result.context_length :]
        )
        result = replace(result, time=selected)
    return result
