"""ArviZ ``DataTree`` export of conditional draws; xarray and arviz-base load only when called.

Export only. Evidence stays :meth:`DecompositionResult.save` and evaluation archives; MFM
never reads a tree back or scores one. A tree holds one marginal window: context and query
are separately sampled, so their trees are never concatenated or stacked along ``draw``.

Every group is a plain ``xarray.Dataset`` with explicit dimensions and coordinates (no
ArviZ ``rcParams`` are consulted) and the standard ArviZ attributes of
``arviz_base.make_attrs``. MFM attributes are flat text and numbers, never ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._artifact_schema import digest_bytes, json_bytes
from .result import DecompositionResult, _checked_evidence

if TYPE_CHECKING:
    from xarray import DataTree

    from pymc_mfm.diagnostics.archive import EvaluationArchive

_EXTRA = "DataTree export requires the pymc-mfm[arviz] extra: pip install 'pymc-mfm[arviz]'"
_SAMPLE_DIMS = ("chain", "draw")
#: Component axis order of an archive tree: the slot order of ``result_from_batch``.
_FAMILY_RANK = {
    family: rank
    for rank, family in enumerate(("baseline", "channel", "control", "latent", "indirect"))
}
_FLOAT64_MAX = float(np.finfo(np.float64).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_GRAPH_DRAWS = {
    "model_average": "joint-router graph draws, one per component draw",
    "map": "the selected modal hypothesis repeated in every draw; not posterior samples",
    "fixed": "the supplied hypothesis repeated in every draw; not posterior samples",
}
_NOTES = {
    "sampling": "one chain of independent amortized conditional draws, not a Markov chain",
    "mcmc_diagnostics": "not_applicable",
    "marginal_windows": (
        "context and query are separate marginals: never concatenate their trees' time axes "
        "or stack them along draw"
    ),
}


def _require_arviz():
    """``(xarray, arviz_base)``, or an ``ImportError`` naming the extra."""
    try:
        import arviz_base
        import xarray
    except ImportError as exc:
        raise ImportError(_EXTRA) from exc
    return xarray, arviz_base


def _dataset(xr, arviz_base, variables, coords, *, sample_dims=(), attrs=None):
    """One group: ``variables`` ``{name: (dims, values)}`` with the coordinates of their dims."""
    import pymc_mfm

    used = {dim for dims, _ in variables.values() for dim in dims}
    return xr.Dataset(
        {name: (dims, values) for name, (dims, values) in variables.items()},
        coords={dim: values for dim, values in coords.items() if dim in used},
        attrs=arviz_base.make_attrs(
            attrs=attrs, inference_library=pymc_mfm, sample_dims=list(sample_dims)
        ),
    )


def _tree(xr, arviz_base, groups, attrs):
    import pymc_mfm

    tree = xr.DataTree.from_dict(groups)
    tree.attrs = arviz_base.make_attrs(attrs=attrs, inference_library=pymc_mfm)
    return tree


def _sales_scale(value: object) -> float:
    """A recorded ``sales_scale`` checked as an archive capture's is."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= _FLOAT64_MAX
    ):
        raise ValueError(f"provenance sales_scale must be a finite positive number, got {value!r}")
    return float(value)


def _edge_constants(provenance: dict[str, Any], n_edges: int) -> dict[str, np.ndarray]:
    """Recorded per-slot edge ``family``, ``source`` and ``target``; ``None`` means not recorded."""
    constants = {}
    families = provenance.get("edge_families")
    if families is not None:
        if (
            not isinstance(families, (list, tuple))
            or len(families) != n_edges
            or not all(isinstance(family, str) and family for family in families)
        ):
            raise ValueError(
                f"provenance edge_families must name one family per edge slot ({n_edges})"
            )
        constants["family"] = _labels(families, "edge_families")
    endpoints = provenance.get("edge_endpoints")
    if endpoints is not None:
        if (
            not isinstance(endpoints, (list, tuple))
            or len(endpoints) != n_edges
            or not all(
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(isinstance(node, str) and node for node in pair)
                for pair in endpoints
            )
        ):
            raise ValueError(
                "provenance edge_endpoints must hold one [source, target] pair of node names "
                f"per edge slot ({n_edges})"
            )
        constants["source"] = _labels([pair[0] for pair in endpoints], "edge_endpoints")
        constants["target"] = _labels([pair[1] for pair in endpoints], "edge_endpoints")
    return constants


def _metadata_sha256(result: DecompositionResult) -> str:
    """The ``metadata_sha256`` that :meth:`DecompositionResult.save` records for ``result``."""
    return digest_bytes(json_bytes(result._metadata()))


def _result_attrs(result: DecompositionResult) -> dict[str, Any]:
    provenance = result.provenance
    attrs = {
        "window": result.window,
        "mode": result.mode,
        "units": result.units,
        "n_samples": result.n_samples,
        "context_length": result.context_length,
        "metadata_sha256": _metadata_sha256(result),
    }
    if provenance.get("sales_scale") is not None:
        attrs["sales_scale"] = _sales_scale(provenance["sales_scale"])
    for key in ("state_sha256", "data_sha256", "graph_source"):
        value = provenance.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"provenance {key} must be text, got {value!r}")
            attrs[key] = value
    return {**attrs, "graph_draws": _GRAPH_DRAWS[result.mode], **_NOTES}


def result_datatree(
    result: DecompositionResult, *, observed=None, truth=None, graph_truth=None
) -> DataTree:
    """See :meth:`DecompositionResult.to_datatree`."""
    xr, arviz_base = _require_arviz()
    active = result.component_mask
    if not active.any():
        raise ValueError("DataTree export needs at least one active component")
    edges = result.edge_mask
    has_edges = bool(edges.any())
    evidence = _checked_evidence(result, truth, graph_truth, observed, stored=False, partial=True)
    edge_constants = _edge_constants(result.provenance, result.graphs.shape[1])
    attrs = _result_attrs(result)
    coords = {
        "chain": np.array([0]),
        "draw": np.arange(result.n_samples),
        "time": _time_coordinate(np.array(result.time)),
        "component": _labels(result.component_names, "component")[active],
        "edge": _labels(result.edge_names, "edge")[edges],
    }
    posterior = {
        "components": (
            ("chain", "draw", "time", "component"),
            _numeric(result.components[None][..., active], widen_float32=True),
        )
    }
    constants = {
        "component_family": (("component",), _labels(result.component_families, "family")[active])
    }
    truth_group = {}
    if "truth" in evidence:
        truth_group["components"] = (("time", "component"), evidence["truth"][:, active])
    if has_edges:
        posterior["graph"] = (("chain", "draw", "edge"), _numeric(result.graphs[None][..., edges]))
        for key, values in edge_constants.items():
            constants[f"edge_{key}"] = (("edge",), values[edges])
        if "graph_truth" in evidence:
            truth_group["graph"] = (("edge",), evidence["graph_truth"][edges])
    groups = {
        "posterior": _dataset(
            xr, arviz_base, posterior, coords, sample_dims=_SAMPLE_DIMS, attrs=attrs
        ),
        "constant_data": _dataset(xr, arviz_base, constants, coords),
    }
    if "observed" in evidence:
        groups["observed_data"] = _dataset(
            xr, arviz_base, {"outcome": (("time",), evidence["observed"])}, coords
        )
    if truth_group:
        groups["truth"] = _dataset(xr, arviz_base, truth_group, coords)
    return _tree(xr, arviz_base, groups, attrs)


def _labels(values, name: str) -> np.ndarray:
    """Labels as a NumPy text array, which would drop trailing NULs: NULs are refused."""
    values = list(values)
    if any("\x00" in value for value in values):
        raise ValueError(f"{name} labels cannot contain NUL characters")
    return np.array(values, dtype=str)


def _time_coordinate(values: np.ndarray) -> np.ndarray:
    """Pandas-compatible numeric coordinates, preserving Boolean label selection."""
    values = values.astype(values.dtype.newbyteorder("="), copy=False)
    return values.astype(np.float64) if values.dtype == np.float16 else values


def _numeric(values: np.ndarray, *, widen_float32: bool = False) -> np.ndarray:
    """Native byte order and exact widening of dtypes unsafe for ArviZ summaries.

    Booleans cannot be subtracted; float16 and float32 components can under/overflow in
    reductions. Binary graphs need no float32 widening. Saved results can retain
    non-native byte order, which downstream reductions may not support.
    """
    values = values.astype(values.dtype.newbyteorder("="), copy=False)
    if (
        values.dtype.kind == "b"
        or values.dtype == np.float16
        or (widen_float32 and values.dtype == np.float32)
    ):
        return values.astype(np.float64)
    return values


def _integers(name: str, values: list[int]) -> np.ndarray:
    """``int64``, else ``uint64`` (call seeds reach ``2**64 - 1``); never an inferred dtype."""
    if min(values) >= np.iinfo(np.int64).min and max(values) <= _INT64_MAX:
        return np.array(values, dtype=np.int64)
    if min(values) >= 0 and max(values) <= np.iinfo(np.uint64).max:
        return np.array(values, dtype=np.uint64)
    raise ValueError(f"archive {name} values do not fit one 64-bit integer type")


def _archive_attrs(archive: EvaluationArchive, window: str) -> dict[str, Any]:
    identity = archive.identity
    model, evaluation = identity["model"], identity["evaluation"]
    attrs = {
        "window": window,
        "mode": evaluation["mode"],
        "units": evaluation["units"],
        "n_samples": evaluation["n_samples"],
        "leg": evaluation["leg"],
        "archive_sha256": archive.archive_sha256,
        "dataset_sha256": archive.dataset_sha256,
        "state_sha256": model["state_sha256"],
        "config_sha256": model["config_sha256"],
    }
    for key in ("checkpoint_sha256", "bundle_sha256", "profile", "source_step"):
        if model[key] is not None:
            attrs[key] = model[key]
    return {**attrs, "graph_draws": _GRAPH_DRAWS[evaluation["mode"]], **_NOTES}


def archive_datatree(archive: EvaluationArchive, window: str) -> DataTree:
    """See :meth:`EvaluationArchive.to_datatree`; ``window`` is already checked."""
    xr, arviz_base = _require_arviz()
    attrs = _archive_attrs(archive, window)
    n_samples = archive.identity["evaluation"]["n_samples"]
    independent = {entry.index for entry in archive.independent_entries(window)}
    worlds = []
    families: dict[str, str] = {}
    active_names: set[str] = set()
    for entry in archive.entries:
        if entry.window != window:
            continue
        result, truth, graph_truth, observed = archive.record(entry.index)
        active, edges = result.component_mask, result.edge_mask
        labels = _labels(result.component_names, "component").tolist()
        for name, family in zip(labels, result.component_families, strict=True):
            if families.setdefault(name, family) != family:
                raise ValueError(
                    f"component {name!r} is {families[name]!r} in one world and {family!r} in "
                    f"{entry.world_id}"
                )
        names = np.array(labels, dtype=str)[active]
        active_names.update(names.tolist())
        edge_names = _labels(result.edge_names, "edge")[edges]
        edge_constants = _edge_constants(result.provenance, result.graphs.shape[1])
        worlds.append(
            {
                "entry": entry,
                "time": _time_coordinate(np.array(result.time)),
                "names": names,
                "components": result.components[..., active],
                "truth": truth[:, active],
                "observed": np.array(observed),
                "pairs": [f"{entry.world_id}|{edge}" for edge in edge_names],
                "edge_name": edge_names,
                "graph": result.graphs[:, edges],
                "graph_truth": graph_truth[edges],
                "edge_constants": {key: values[edges] for key, values in edge_constants.items()},
                "sales_scale": _sales_scale(result.provenance["sales_scale"]),
                "metadata_sha256": _metadata_sha256(result),
            }
        )
    # First appearance over every world's full slot labels, stably in family order.
    component_axis = sorted(
        (name for name in families if name in active_names),
        key=lambda name: _FAMILY_RANK[families[name]],
    )
    if not component_axis:
        raise ValueError("DataTree export needs at least one active component")
    position = {name: index for index, name in enumerate(component_axis)}
    # NumPy promotion can round large integer labels when worlds use different dtypes.
    time_representatives = {}
    for world in worlds:
        for label in world["time"].tolist():
            # Object indexes match numeric vectors to 0/1, but not to False/True.
            if label not in time_representatives or isinstance(time_representatives[label], bool):
                time_representatives[label] = label
    time_labels = sorted(time_representatives.values())
    time_axis = np.asarray(time_labels)
    time_kinds = {world["time"].dtype.kind for world in worlds}
    mixed_boolean_time = "b" in time_kinds and len(time_kinds) > 1
    if time_axis.tolist() != time_labels or mixed_boolean_time:
        time_axis = np.asarray(time_labels, dtype=object)
    time_position = {label: index for index, label in enumerate(time_labels)}
    shape = (len(worlds), len(time_axis), len(component_axis))
    components = np.full((1, n_samples, *shape), np.nan)
    truth_values = np.full(shape, np.nan)
    outcome = np.full(shape[:2], np.nan)
    for index, world in enumerate(worlds):
        rows = np.array([time_position[label] for label in world["time"].tolist()], dtype=np.intp)
        columns = np.array([position[name] for name in world["names"]], dtype=np.intp)
        components[0, :, index][:, rows[:, None], columns] = world["components"]
        truth_values[index][np.ix_(rows, columns)] = world["truth"]
        outcome[index, rows] = world["observed"]
    entries = [world["entry"] for world in worlds]
    coords = {
        "chain": np.array([0]),
        "draw": np.arange(n_samples),
        "world": np.array([entry.world_id for entry in entries], dtype=str),
        "time": time_axis,
        "component": np.array(component_axis, dtype=str),
        "world_edge": np.array([pair for world in worlds for pair in world["pairs"]], dtype=str),
    }
    world_constants = {
        name: _integers(name, [getattr(entry, name) for entry in entries])
        for name in ("corpus_index", "row", "cell_id", "context_length", "call", "seed")
    }
    world_constants.update(
        cell=np.array([f"{entry.cell[0]}/cell:{entry.cell[1]}" for entry in entries], dtype=str),
        independent=np.array([entry.index in independent for entry in entries]),
        data_sha256=np.array([entry.data_sha256 for entry in entries], dtype=str),
        sales_scale=np.array([world["sales_scale"] for world in worlds], dtype=np.float64),
        metadata_sha256=np.array([world["metadata_sha256"] for world in worlds], dtype=str),
    )
    constants = {name: (("world",), values) for name, values in world_constants.items()}
    constants["component_family"] = (
        ("component",),
        np.array([families[name] for name in component_axis], dtype=str),
    )
    posterior = {"components": (("chain", "draw", "world", "time", "component"), components)}
    truth_group = {"components": (("world", "time", "component"), truth_values)}
    if len(coords["world_edge"]):  # empty only for hand-edited captures without a live edge
        posterior["graph"] = (
            ("chain", "draw", "world_edge"),
            _numeric(np.concatenate([world["graph"] for world in worlds], axis=1))[None],
        )
        truth_group["graph"] = (
            ("world_edge",),
            np.concatenate([world["graph_truth"] for world in worlds]),
        )
        pair_constants = {
            "edge_world": np.repeat(coords["world"], [len(world["pairs"]) for world in worlds]),
            "edge_name": np.concatenate([world["edge_name"] for world in worlds]),
            **{
                f"edge_{key}": np.concatenate([world["edge_constants"][key] for world in worlds])
                for key in ("family", "source", "target")
            },
        }
        constants.update(
            {name: (("world_edge",), values) for name, values in pair_constants.items()}
        )
    groups = {
        "posterior": _dataset(
            xr, arviz_base, posterior, coords, sample_dims=_SAMPLE_DIMS, attrs=attrs
        ),
        "constant_data": _dataset(xr, arviz_base, constants, coords),
        "observed_data": _dataset(
            xr, arviz_base, {"outcome": (("world", "time"), outcome)}, coords
        ),
        "truth": _dataset(xr, arviz_base, truth_group, coords),
    }
    return _tree(xr, arviz_base, groups, attrs)
