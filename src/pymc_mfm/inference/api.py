"""Small observation-only boundary to the canonical conditional decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch

from pymc_mfm.data.features import MOMENT_FEATURE_BLOCKS
from pymc_mfm.data.observations import encode_observations, observation_identity
from pymc_mfm.inference._time import _validate_time
from pymc_mfm.inference.result import (
    DecompositionResult,
    _input_summary,
    _numpy,
    result_from_batch,
)
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.flow import DEFAULT_N_SAMPLES
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
    Each world's context fits the sales p95, per-channel mean spend and per-control
    RMS scales, before any features or precision narrowing. Query uses the same
    scales; zero-context/nonzero-query columns are unsupported. ``sales_scale``
    stays float64 CPU metadata for restoring units, even with a low-precision model.
    """

    def numeric(value, name, ndim):
        array = np.asarray(value)
        if array.ndim != ndim or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite real {ndim}-dimensional array")
        return array

    spend = numeric(spend, "spend", 2)
    controls = numeric(controls, "controls", 2)
    outcome = numeric(context_outcome, "context_outcome", 1)
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
            dtype = np.float64 if key == "sales_scale" else np.float32
            value = np.ascontiguousarray(value, dtype=dtype)
        if not np.isfinite(value).all():
            raise ValueError(f"{key} exceeds the finite model representation")
        tensors[key] = torch.from_numpy(value).to("cpu" if key == "sales_scale" else device)
    return tensors


def _model_batch(
    batch: dict[str, torch.Tensor], *, device: str | torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Narrow model inputs while retaining float64 CPU sales-scale metadata."""
    batch = {
        key: (
            value.to(device="cpu", dtype=torch.float64)
            if key == "sales_scale"
            else value.to(device=device, dtype=dtype)
        )
        for key, value in batch.items()
    }
    for key, value in batch.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} exceeds the finite model representation")
    if (batch["sales_scale"] <= 0).any():
        raise ValueError("sales_scale must remain positive")
    return batch


def decompose(
    model: MFM,
    observations: Mapping[str, object],
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
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
    context/query posterior. The caller controls the PyTorch random seed;
    ``n_samples`` defaults to ``DEFAULT_N_SAMPLES`` conditional draws.
    Caller provenance may supplement, but not replace, loaded artifact identities.
    """
    if not isinstance(observations, Mapping):
        raise ValueError("observations must be a mapping")
    parameter = next(model.parameters())
    if "ctx_spend_z" in observations or "spend" not in observations:
        if not all(isinstance(key, str) for key in observations):
            raise ValueError("encoded observation keys must be strings")
        batch = {
            key: value
            for key, value in observations.items()
            if not key.startswith("g_") and "targets" not in key
        }
        if any(not isinstance(value, torch.Tensor) for value in batch.values()):
            raise ValueError("encoded observation values must be tensors")
        for key in ("ctx_spend_z", "qry_spend_z"):
            if key not in batch or batch[key].ndim != 3 or min(batch[key].shape) < 1:
                raise ValueError(f"{key} must have nonempty shape [world,time,channel]")
        b, c, k = batch["ctx_spend_z"].shape
        if b != 1:
            raise ValueError("decompose accepts one observed world at a time")
        q = batch["qry_spend_z"].shape[1]
        m, j = model.config.n_controls, model.config.n_demand
        shapes = {
            "qry_spend_z": (b, q, k),
            "ctx_controls_z": (b, c, m),
            "qry_controls_z": (b, q, m),
            "ctx_revenue_z": (b, c, 1),
            "spend_mean": (b, k),
            "spend_std": (b, k),
            "ctx_week": (b, c, 1),
            "qry_week": (b, q, 1),
            "active_c_mask": (b, k),
            "active_m_mask": (b, m),
            "active_j_mask": (b, j),
            "sales_scale": (b,),
        }
        allowed = shapes.keys() | {"ctx_spend_z"} | {name for name, _ in MOMENT_FEATURE_BLOCKS}
        if batch.keys() - allowed:
            raise ValueError("encoded observations contain unknown fields")
        for key, shape in shapes.items():
            if key not in batch or batch[key].shape != shape:
                raise ValueError(f"{key} must have shape {shape}")
        model.encoder.assert_batch_alignment(batch)
        for key, value in batch.items():
            if value.is_complex() or not torch.isfinite(value).all():
                raise ValueError(f"{key} must contain finite real values")
        for key in ("active_c_mask", "active_m_mask", "active_j_mask"):
            if ((batch[key] != 0) & (batch[key] != 1)).any():
                raise ValueError(f"{key} must be binary before dtype conversion")
        if (batch["sales_scale"] <= 0).any():
            raise ValueError("sales_scale must be positive")
    else:
        batch = observation_batch(**observations, model_config=model.config, device="cpu")
    # Checks, hashes and summaries read a CPU copy in the model's dtype, so a GPU model
    # waits for no device round trip; only sampling reads the device copy.
    batch = _model_batch(batch, device="cpu", dtype=parameter.dtype)
    device_batch = _to_device(batch, parameter.device)
    if graph is not None:
        # Validate the caller's precision before the model converts graph dtype. A
        # copy keeps a caller's (possibly read-only) array out of the tensor.
        supplied_graph = graph if isinstance(graph, torch.Tensor) else np.array(graph)
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
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        for key, value in provenance.items():
            if key in identity and identity[key] is not None and identity[key] != value:
                raise ValueError(f"provenance {key!r} conflicts with the loaded artifact")
        identity.update(provenance)
    _drop_recorded_evaluation(identity, getattr(model, "provenance", {}).get("training_evaluation"))
    if time is not None:
        full_length = batch["ctx_spend_z"].shape[1] + batch["qry_spend_z"].shape[1]
        time = _validate_time(time, length=full_length, allow_bool=False)
    was_training = model.training
    # Switching modes walks every submodule; skip it when every module already evaluates.
    switch_modes = any(module.training for module in model.modules())
    try:
        if switch_modes:
            model.eval()
        components, graphs = model.sample_disentangled(
            device_batch,
            n_samples=n_samples,
            mode=mode,
            graph=graph,
            router_samples=router_samples,
            window=window,
        )
    finally:
        if switch_modes:
            model.train(was_training)
    # Hash and summarize while an asynchronous device may still be sampling.
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
        identity["input_support"][key] = _input_summary(values, input_units)
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


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Sampling copy of ``batch`` on ``device``; ``sales_scale`` stays on the CPU.

    The other tensors move in one transfer, as views of one buffer: on a GPU each
    separate copy would wait for the device.
    """
    if device.type == "cpu":
        return dict(batch)
    moved = [key for key in batch if key != "sales_scale"]
    flat = torch.cat([batch[key].reshape(-1) for key in moved]).to(device)
    views = dict(zip(moved, flat.split([batch[key].numel() for key in moved]), strict=True))
    return {
        key: value if key == "sales_scale" else views[key].view(value.shape)
        for key, value in batch.items()
    }


def _drop_recorded_evaluation(identity: dict, recorded) -> None:
    """Keep a bundle's recorded training evaluation out of each result it produces.

    The record (held-out worlds plus every recorded metric) stays in the bundle's
    ``config.json``; results already name that bundle by ``bundle_sha256`` and
    ``source_checkpoint_sha256``. Only copies equal to the bundle's own record are
    dropped, so caller annotations are never silently removed.
    """
    if recorded is None:
        return
    if identity.get("training_evaluation") == recorded:
        del identity["training_evaluation"]
    artifact = identity.get("artifact")
    if isinstance(artifact, Mapping) and isinstance(artifact.get("provenance"), Mapping):
        provenance = artifact["provenance"]
        if provenance.get("training_evaluation") == recorded:
            identity["artifact"] = {
                **artifact,
                "provenance": {
                    key: value for key, value in provenance.items() if key != "training_evaluation"
                },
            }
