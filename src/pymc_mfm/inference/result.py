"""Aligned conditional evidence, independent of rendering and tracking services."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ARRAY_FIELDS = ("components", "graphs", "component_mask", "edge_mask", "time")
_TUPLE_FIELDS = (
    "component_names",
    "component_families",
    "edge_names",
    "channel_names",
    "control_names",
)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # Widen exactly: NumPy has no native bfloat16 representation.
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """One world's conditional component/graph draws for one marginal window.

    Components ``[sample,time,component]`` and binary graphs ``[sample,edge]``
    share sample identity. Context and query results are separate marginals:
    concatenating their sample axes does not create full-horizon joint draws.
    Padding is explicit, not an estimate of causal activity. No simulation
    truth or observation noise is manufactured by this contract.
    """

    components: np.ndarray
    graphs: np.ndarray
    component_names: tuple[str, ...]
    component_families: tuple[str, ...]
    component_mask: np.ndarray
    edge_names: tuple[str, ...]
    edge_mask: np.ndarray
    channel_names: tuple[str, ...]
    control_names: tuple[str, ...]
    time: np.ndarray
    window: str
    mode: str
    units: str
    provenance: dict[str, Any]
    context_length: int

    def __post_init__(self):
        for name in _ARRAY_FIELDS:
            array = np.asarray(getattr(self, name))
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain finite real values")
            if name.endswith("mask"):
                if not np.isin(array, (0, 1)).all():
                    raise ValueError(f"{name} must be binary")
                array = array.astype(bool, copy=False)
            object.__setattr__(self, name, array)
        for name in _TUPLE_FIELDS:
            values = tuple(getattr(self, name))
            if not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{name} must contain nonempty strings")
            if name != "component_families" and len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, values)
        if self.components.ndim != 3 or min(self.components.shape) < 1:
            raise ValueError("components must have nonempty shape [sample,time,component]")
        s, t, d = self.components.shape
        if self.graphs.ndim != 2 or self.graphs.shape[0] != s or self.graphs.shape[1] < 1:
            raise ValueError("graphs must have shape [sample,edge], aligned with components")
        if not np.isin(self.graphs, (0, 1)).all():
            raise ValueError("graphs must be binary complete-graph draws")
        if len(self.component_names) != d or len(self.component_families) != d:
            raise ValueError("component labels must match the component axis")
        if self.component_mask.shape != (d,) or self.edge_mask.shape != (self.graphs.shape[1],):
            raise ValueError("masks must match their component/edge axes")
        if len(self.edge_names) != self.graphs.shape[1]:
            raise ValueError("edge names must match the graph axis")
        if self.time.shape != (t,) or (self.time[1:] <= self.time[:-1]).any():
            raise ValueError("time must be a strictly increasing numeric vector")
        if self.window not in ("context", "query"):
            raise ValueError("window must be context or query")
        if self.mode not in ("model_average", "map", "fixed"):
            raise ValueError("mode must be model_average, map or fixed")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a nonempty label")
        if type(self.context_length) is not int or self.context_length < 1:
            raise ValueError("context_length must be a positive integer")
        if self.window == "context" and t != self.context_length:
            raise ValueError("context results must span context_length rows")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a JSON object")
        json.dumps(self.provenance, allow_nan=False)
        if "inference_config" in self.provenance:
            inference = self.provenance["inference_config"]
            if not isinstance(inference, dict):
                raise ValueError("inference_config must be a JSON object")
            for key, expected in (("mode", self.mode), ("window", self.window), ("n_samples", s)):
                if key in inference and (
                    type(inference[key]) is not type(expected) or inference[key] != expected
                ):
                    raise ValueError(f"result {key} conflicts with recorded inference_config")
        graph_source = "supplied_hypothesis" if self.mode == "fixed" else "joint_router"
        if "graph_source" in self.provenance and self.provenance["graph_source"] != graph_source:
            raise ValueError("result mode conflicts with recorded graph_source")
        allowed = {"baseline", "channel", "control", "latent", "indirect"}
        if not set(self.component_families) <= allowed:
            raise ValueError("unknown component family")
        if self.component_families.count("channel") != len(self.channel_names):
            raise ValueError("channel names must preserve the component channel axis")
        if self.component_families.count("control") != len(self.control_names):
            raise ValueError("control names must preserve the component control axis")
        if np.any(self.graphs[:, ~self.edge_mask]):
            raise ValueError("graphs contain edges touching padded nodes")
        if np.any(self.components[..., ~self.component_mask]):
            raise ValueError("components contain contributions from padded entities")

    @property
    def n_samples(self) -> int:
        return self.components.shape[0]

    def save(self, path: str | Path) -> Path:
        """Atomically create a new evidence directory, never mutate cached evidence."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in _ARRAY_FIELDS
        }
        with tempfile.TemporaryDirectory(prefix=".result-", dir=path.parent) as temporary:
            staged = Path(temporary) / "result"
            staged.mkdir()
            arrays = staged / "arrays.npz"
            np.savez_compressed(arrays, **{name: getattr(self, name) for name in _ARRAY_FIELDS})
            with arrays.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            payload = {"format_version": 1, "arrays_sha256": digest, "metadata": metadata}
            (staged / "result.json").write_text(json.dumps(payload, allow_nan=False, indent=2))
            staged.rename(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> DecompositionResult:
        """Load verified, non-executable saved evidence without inference."""
        path = Path(path)
        payload = json.loads((path / "result.json").read_text())
        if set(payload) != {"format_version", "arrays_sha256", "metadata"}:
            raise ValueError("invalid result metadata fields")
        if type(payload["format_version"]) is not int or payload["format_version"] != 1:
            raise ValueError("unsupported result format_version")
        arrays_path = path / "arrays.npz"
        with arrays_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != payload["arrays_sha256"]:
            raise ValueError("result array checksum mismatch")
        metadata = payload["metadata"]
        expected = {item.name for item in fields(cls)} - set(_ARRAY_FIELDS)
        if not isinstance(metadata, dict) or set(metadata) != expected:
            raise ValueError("incomplete or unknown result metadata")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(_ARRAY_FIELDS):
                raise ValueError("incomplete or unknown result arrays")
            return cls(**metadata, **{key: arrays[key] for key in _ARRAY_FIELDS})


def result_from_batch(
    batch,
    components,
    graphs,
    *,
    model_config,
    world=0,
    window="query",
    mode="model_average",
    provenance=None,
    channel_names=None,
    control_names=None,
    units="outcome",
    router_samples=None,
) -> DecompositionResult:
    """Label and decode existing conditional samples; never sample the network."""
    from pymc_mfm.model.graph import EdgeSlotLayout

    if window not in ("query", "context"):
        raise ValueError("window must be context or query")
    components, graphs = _numpy(components), _numpy(graphs)
    if components.ndim != 4 or graphs.ndim != 3:
        raise ValueError("expected components [S,B,T,D] and graphs [S,B,L]")
    if type(world) is not int or not 0 <= world < components.shape[1]:
        raise ValueError("world must index the batch axis")
    if graphs.shape[:2] != components.shape[:2]:
        raise ValueError("component and graph sample/world axes must align")
    active_c = _numpy(batch["active_c_mask"])[world].astype(bool)
    active_m = _numpy(batch["active_m_mask"])[world].astype(bool)
    active_j = _numpy(batch["active_j_mask"])[world].astype(bool)
    k, m, j = len(active_c), len(active_m), len(active_j)
    if (m, j) != (model_config.n_controls, model_config.n_demand):
        raise ValueError("result geometry disagrees with model configuration")
    channels = (
        tuple(f"channel_{i}" for i in range(k)) if channel_names is None else tuple(channel_names)
    )
    controls = (
        tuple(f"control_{i}" for i in range(m)) if control_names is None else tuple(control_names)
    )
    if len(channels) != k or len(controls) != m:
        raise ValueError("entity names must preserve padded entity axes")
    names = (
        "baseline",
        *(f"channel:{name}" for name in channels),
        *(f"control:{name}" for name in controls),
        "latent",
        "indirect:cc",
        "indirect:zc",
        "indirect:dc",
    )
    families = (
        "baseline",
        *("channel" for _ in channels),
        *("control" for _ in controls),
        "latent",
        "indirect",
        "indirect",
        "indirect",
    )
    layout = EdgeSlotLayout(k, m, j)
    nodes = [
        *(f"channel:{name}" for name in channels),
        *(f"control:{name}" for name in controls),
        *(f"latent:{i}" for i in range(j)),
        "outcome",
        "outcome",
    ]
    edges = tuple(
        f"{family}:{nodes[src]}->{nodes[dst]}"
        for family, src, dst in zip(
            layout.slot_edge_types, layout.slot_src_index, layout.slot_dst_index, strict=True
        )
    )
    context_length = batch["ctx_spend_z"].shape[1]
    length = batch["qry_spend_z" if window == "query" else "ctx_spend_z"].shape[1]
    if components.shape[2] != length:
        raise ValueError("component time axis does not match the selected window")
    scale = float(_numpy(batch["sales_scale"])[world])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("sales_scale must be finite and positive")
    metadata = dict(provenance or {})
    metadata["edge_endpoints"] = [
        [nodes[src], nodes[dst]]
        for src, dst in zip(layout.slot_src_index, layout.slot_dst_index, strict=True)
    ]
    metadata["edge_families"] = list(layout.slot_edge_types)
    # Internal db/zb and cy slots all target the same observed outcome.
    metadata["node_names"] = list(dict.fromkeys(nodes))
    metadata["inference_config"] = {
        "model": model_config.to_dict(),
        "mode": mode,
        "window": window,
        "n_samples": int(components.shape[0]),
        "router_samples": (
            int(components.shape[0])
            if mode == "model_average"
            else (None if router_samples is None else int(router_samples))
        ),
    }
    metadata["sales_scale"] = scale
    metadata["representation"] = "natural; latent total excludes observation noise"
    metadata["graph_source"] = "supplied_hypothesis" if mode == "fixed" else "joint_router"
    metadata["marginal_windows"] = "separately sampled; not aligned across windows"
    return DecompositionResult(
        components=components[:, world].astype(np.float64) * scale,
        graphs=graphs[:, world],
        component_names=names,
        component_families=families,
        component_mask=np.r_[True, active_c, active_m, active_j.any(), True, True, True],
        edge_names=edges,
        edge_mask=layout.active_slot_mask(active_c, active_m, active_j).astype(bool),
        channel_names=channels,
        control_names=controls,
        time=np.arange(length) + (context_length if window == "query" else 0),
        window=window,
        mode=mode,
        units=units,
        provenance=metadata,
        context_length=context_length,
    )
