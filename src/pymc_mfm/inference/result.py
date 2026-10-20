"""Aligned conditional evidence; diagnostics and figures load only when requested."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import torch

from ._artifact_schema import json_bytes
from ._time import _validate_time

if TYPE_CHECKING:
    from xarray import DataTree

    from pymc_mfm.diagnostics.accessors import Diagnostics, ResultPlots

#: ``result.json`` version: metadata is hash-bound like the arrays since version 2.
RESULT_FORMAT_VERSION = 2
_PAYLOAD_KEYS = frozenset({"format_version", "arrays_sha256", "metadata_sha256", "metadata"})

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


def _input_summary(values, units):
    """Describe finite input columns without overflowing raw sums or squares."""
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.max(np.abs(values), axis=0)
    relative = values / np.where(magnitude > 0, magnitude, 1.0)
    return {
        "units": units,
        "n": len(values),
        "mean": (relative.mean(axis=0) * magnitude).tolist(),
        "std": (relative.std(axis=0) * magnitude).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def _require_text(text, path):
    """Text that cannot encode as UTF-8 (a lone surrogate) breaks every figure and report."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"{path} is not valid Unicode text: {exc.reason} at character {exc.start}"
        ) from None


def _require_json_text(value, path):
    """String keys and UTF-8-encodable strings at every depth.

    JSON silently stringifies other keys, so saved identity would drift on reload.
    An explicit stack walks any depth ``json.dumps`` accepts; the caller runs it
    after ``json.dumps``, which rejects circular references.
    """
    stack = [(value, path)]
    while stack:
        value, path = stack.pop()
        if isinstance(value, str):
            _require_text(value, path)
        elif isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{path} keys must be strings, got {key!r}")
                _require_text(key, f"{path} key")
                stack.append((item, f"{path}/{key}"))
        elif isinstance(value, (list, tuple)):
            stack.extend((item, f"{path}/{index}") for index, item in enumerate(value))


def _checked_evidence(
    result: DecompositionResult,
    truth,
    graph_truth,
    observed,
    *,
    stored: bool,
    partial: bool = False,
) -> dict[str, np.ndarray]:
    """Checked float64 evidence of one result, by the evidence rules of ``compute_diagnostics``.

    ``truth`` ``[T,D]``, ``graph_truth`` ``[L]`` and ``observed`` ``[T]`` align with
    ``result``. Supplied arrays must be real and are copied to float64; ``stored`` ones
    (decoded from an archive's ``truth.npz``) must already be exactly float64 and are kept
    as they are. With ``partial`` a ``None`` piece is left out of the returned mapping and
    only the supplied pieces are checked.
    """
    _, times, width = result.components.shape
    shapes = {
        "truth": (times, width),
        "graph_truth": (result.graphs.shape[1],),
        "observed": (times,),
    }
    evidence = {}
    for name, value in zip(shapes, (truth, graph_truth, observed), strict=True):
        if partial and value is None:
            continue
        if stored:
            array = value
            if array.dtype != np.float64:
                raise ValueError(f"{name} must be float64, got {array.dtype}")
        else:
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise ValueError(f"{name} must be a real array")
            array = np.array(array, dtype=np.float64, copy=True)
        if array.shape != shapes[name]:
            raise ValueError(f"{name} has shape {array.shape}; the result needs {shapes[name]}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        evidence[name] = array
    if "graph_truth" in evidence and not np.isin(evidence["graph_truth"], (0, 1)).all():
        raise ValueError("graph_truth must be binary")
    if "truth" in evidence and np.any(evidence["truth"][:, ~result.component_mask]):
        raise ValueError("truth is nonzero on padded components")
    if "graph_truth" in evidence and np.any(evidence["graph_truth"][~result.edge_mask]):
        raise ValueError("graph_truth has edges outside the active support")
    return evidence


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """One world's conditional component/graph draws for one marginal window.

    Components ``[sample,time,component]`` and binary graphs ``[sample,edge]``
    share sample identity. Context and query results are separate marginals:
    concatenating their sample axes does not create full-horizon joint draws.
    Padding is explicit, not an estimate of causal activity. No simulation
    truth or observation noise is manufactured by this contract. The array
    fields are read-only copies of the supplied arrays.
    Time labels are finite numeric values or strictly increasing, non-NaT
    ``datetime64`` values, retained in their original unit and dtype.
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
    # Evidence-free diagnostics shared by ``plot`` and ``diagnostics()``. Never
    # compared, shown, saved, pickled or copied; ``dataclasses.replace`` starts
    # with an empty cache.
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __getstate__(self):
        return [
            None if item.name == "_cache" else getattr(self, item.name) for item in fields(self)
        ]

    def __setstate__(self, state):
        for item, value in zip(fields(self), state, strict=True):
            if item.name in _ARRAY_FIELDS:
                value.setflags(write=False)  # unpickled and deep-copied arrays are writable
            object.__setattr__(self, item.name, {} if item.name == "_cache" else value)

    def __post_init__(self):
        for name in _ARRAY_FIELDS:
            # A read-only copy: the caller's buffer, reused after construction,
            # cannot change these draws or the diagnostics computed from them.
            array = np.array(getattr(self, name), copy=True)
            if name == "time":
                _validate_time(array)
            elif array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain finite real values")
            if name.endswith("mask"):
                if not np.isin(array, (0, 1)).all():
                    raise ValueError(f"{name} must be binary")
                array = array.astype(bool, copy=False)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in _TUPLE_FIELDS:
            values = tuple(getattr(self, name))
            if not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{name} must contain nonempty strings")
            for item in values:
                _require_text(item, name)
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
        if self.time.shape != (t,):
            raise ValueError("time must label every component row")
        if self.window not in ("context", "query"):
            raise ValueError("window must be context or query")
        if self.mode not in ("model_average", "map", "fixed"):
            raise ValueError("mode must be model_average, map or fixed")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a nonempty label")
        _require_text(self.units, "units")
        if type(self.context_length) is not int or self.context_length < 1:
            raise ValueError("context_length must be a positive integer")
        if self.window == "context" and t != self.context_length:
            raise ValueError("context results must span context_length rows")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a JSON object")
        json.dumps(self.provenance, allow_nan=False)
        _require_json_text(self.provenance, "provenance")
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

    @property
    def plot(self) -> ResultPlots:
        """Figures of these draws; matplotlib is imported only when a figure is drawn."""
        from pymc_mfm.diagnostics.accessors import ResultPlots

        return ResultPlots(self)

    def diagnostics(
        self,
        *,
        truth=None,
        graph_truth=None,
        observed=None,
        reference=None,
        training_records=(),
        evaluation_records=(),
        coverage_intervals=(),
    ) -> Diagnostics:
        """Score these draws once against the supplied evidence; never samples.

        The returned object holds the computed catalog, so its metrics,
        statuses, figures and report reuse one computation. Without evidence
        the object is also cached on this result and shared with ``plot``.
        ``coverage_intervals`` supplies report-only saved evaluation uncertainty;
        it is frozen with the other evidence and never added to scalar metrics.
        """
        from pymc_mfm.diagnostics.accessors import Diagnostics

        training_records, evaluation_records = list(training_records), list(evaluation_records)
        coverage_intervals = list(coverage_intervals)
        evidence_free = (
            truth is None
            and graph_truth is None
            and observed is None
            and reference is None
            and not training_records
            and not evaluation_records
            and not coverage_intervals
        )
        if evidence_free and "diagnostics" in self._cache:
            return self._cache["diagnostics"]
        diagnostics = Diagnostics(
            self,
            truth=truth,
            graph_truth=graph_truth,
            observed=observed,
            reference=reference,
            training_records=training_records,
            evaluation_records=evaluation_records,
            coverage_intervals=coverage_intervals,
        )
        if evidence_free:
            self._cache["diagnostics"] = diagnostics
        return diagnostics

    def to_datatree(self, *, observed=None, truth=None, graph_truth=None) -> DataTree:
        """These draws as an ArviZ ``xarray.DataTree``; needs the ``pymc-mfm[arviz]`` extra.

        Export only: ``save`` stays the evidence format and MFM never reads a tree back.
        Groups: ``posterior`` (``components[chain=1, draw, time, component]`` and
        ``graph[chain=1, draw, edge]``, active slots only), ``constant_data`` (component
        and edge families, edge endpoints), ``observed_data`` (``outcome[time]``, from
        ``observed``) and ``truth`` (``components``/``graph`` from synthetic ``truth`` /
        ``graph_truth``), never mixed with observations. Evidence follows the rules of
        :meth:`diagnostics`; the padded columns it must leave at zero are dropped. Draws
        are one chain of independent amortized samples: R-hat is undefined; ESS and MCSE
        describe only Monte Carlo variability, not convergence or amortization accuracy.
        One tree holds one window: never concatenate context and query trees.
        This call imports xarray and arviz-base lazily.
        """
        from ._datatree import result_datatree

        return result_datatree(self, observed=observed, truth=truth, graph_truth=graph_truth)

    def _metadata(self) -> dict[str, Any]:
        """The non-array fields ``save`` writes and its ``metadata_sha256`` digests."""
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.init and item.name not in _ARRAY_FIELDS
        }

    def save(self, path: str | Path) -> Path:
        """Atomically create a new evidence directory, never mutate cached evidence."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._metadata()
        with tempfile.TemporaryDirectory(prefix=".result-", dir=path.parent) as temporary:
            staged = Path(temporary) / "result"
            staged.mkdir()
            arrays = staged / "arrays.npz"
            np.savez_compressed(arrays, **{name: getattr(self, name) for name in _ARRAY_FIELDS})
            with arrays.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            payload = {
                "format_version": RESULT_FORMAT_VERSION,
                "arrays_sha256": digest,
                "metadata_sha256": hashlib.sha256(json_bytes(metadata)).hexdigest(),
                "metadata": metadata,
            }
            (staged / "result.json").write_text(json.dumps(payload, allow_nan=False, indent=2))
            staged.rename(path)
        return path

    @staticmethod
    def _checked_payload(payload: object) -> dict:
        """The ``result.json`` checks of :meth:`load`, with its messages.

        An object, ``format_version`` 2 (checked before the key set), the exact
        key set. Returns ``payload`` itself.
        """
        if not isinstance(payload, dict):
            raise ValueError("invalid result metadata fields")
        version = payload.get("format_version")
        if type(version) is not int or version != RESULT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported result format_version {version!r}; "
                f"this release reads version {RESULT_FORMAT_VERSION}"
            )
        if set(payload) != _PAYLOAD_KEYS:
            raise ValueError("invalid result metadata fields")
        return payload

    @classmethod
    def _from_saved(cls, payload: dict, arrays: bytes | BinaryIO) -> DecompositionResult:
        """Build a result from a checked payload and its ``arrays.npz``, as :meth:`load` does.

        ``arrays`` is the file's bytes or the file itself, open at its start. In
        :meth:`load`'s order and with its messages: the checksum of exactly that
        content, the metadata checksum and fields, then the array members, decoded
        from the same content. A file is streamed, never held in memory whole.
        """
        if isinstance(arrays, bytes):
            digest, stream = hashlib.sha256(arrays).hexdigest(), io.BytesIO(arrays)
        else:
            digest, stream = hashlib.file_digest(arrays, "sha256").hexdigest(), arrays
            stream.seek(0)
        if digest != payload["arrays_sha256"]:
            raise ValueError("result array checksum mismatch")
        metadata = payload["metadata"]
        if (
            not isinstance(metadata, dict)
            or hashlib.sha256(json_bytes(metadata)).hexdigest() != payload["metadata_sha256"]
        ):
            raise ValueError("result metadata checksum mismatch")
        expected = {item.name for item in fields(cls) if item.init} - set(_ARRAY_FIELDS)
        if set(metadata) != expected:
            raise ValueError("incomplete or unknown result metadata")
        with np.load(stream, allow_pickle=False) as stored:
            if set(stored.files) != set(_ARRAY_FIELDS):
                raise ValueError("incomplete or unknown result arrays")
            return cls(**metadata, **{key: stored[key] for key in _ARRAY_FIELDS})

    @classmethod
    def load(cls, path: str | Path) -> DecompositionResult:
        """Load verified, non-executable saved evidence without inference.

        Arrays and metadata (labels, units, provenance) are both hash-bound, so
        an edited ``result.json`` is rejected instead of relabeling the draws.
        """
        path = Path(path)
        payload = cls._checked_payload(json.loads((path / "result.json").read_text()))
        with (path / "arrays.npz").open("rb") as handle:
            return cls._from_saved(payload, handle)


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
    # Every result decodes graph-conditional decoder draws; figures label this head.
    metadata["head"] = "conditional"
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
