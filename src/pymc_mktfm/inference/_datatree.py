"""ArviZ ``DataTree`` export of conditional draws; xarray and arviz-base load only when called.

Export only. Evidence stays :meth:`DecompositionResult.save` and evaluation archives; MFM
never reads a tree back or scores one. Each window is a separate marginal: grouping context
and query trees never pairs their draws or joins their time axes.

Every group is a plain ``xarray.Dataset`` with explicit dimensions and coordinates (no
ArviZ ``rcParams`` are consulted) and the standard ArviZ attributes of
``arviz_base.make_attrs``. MFM attributes are flat text and numbers, never ``None``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ._digest import digest_bytes, json_bytes
from .result import DecompositionResult, _checked_evidence

if TYPE_CHECKING:
    from xarray import DataTree

    from pymc_mktfm.diagnostics.archive import EvaluationArchive
    from pymc_mktfm.inference.curves import ResponseCurves
    from pymc_mktfm.inference.dataframe import _PredictionSnapshot

_EXTRA = "DataTree export requires the pymc-mktfm[arviz] extra: pip install 'pymc-mktfm[arviz]'"
_SAMPLE_DIMS = ("chain", "draw")
_MODEL_TREE_SCHEMA_VERSION = 1
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
    "fixed": "caller-supplied graph hypotheses; not learned graph-posterior samples",
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
    import pymc_mktfm

    used = {dim for dims, _ in variables.values() for dim in dims}
    return xr.Dataset(
        {name: (dims, values) for name, (dims, values) in variables.items()},
        coords={dim: values for dim, values in coords.items() if dim in used},
        attrs=arviz_base.make_attrs(
            attrs=attrs, inference_library=pymc_mktfm, sample_dims=list(sample_dims)
        ),
    )


def _tree(xr, arviz_base, groups, attrs):
    import pymc_mktfm

    tree = xr.DataTree.from_dict(groups)
    tree.attrs = arviz_base.make_attrs(attrs=attrs, inference_library=pymc_mktfm)
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
            attrs[key] = str(value)  # a str subclass, such as np.str_, cannot be an HDF5 attribute
    return {**attrs, "graph_draws": _GRAPH_DRAWS[result.mode], **_NOTES}


def _named_posterior(result: DecompositionResult, families: np.ndarray, coords):
    """Named contributions, using entity axes rather than parsing prefixed slot labels."""
    posterior = {}
    for family, names in (("channel", result.channel_names), ("control", result.control_names)):
        members = families == family
        active = members & result.component_mask
        coords[family] = _labels(names, family)[result.component_mask[members]]
        posterior[f"{family}_contribution"] = (
            ("chain", "draw", "date", family),
            _numeric(result.components[None][..., active], widen_float32=True),
        )
    scalar_names = {
        "baseline": "baseline",
        "latent": "latent",
        "indirect:cc": "indirect_cc",
        "indirect:zc": "indirect_zc",
        "indirect:dc": "indirect_dc",
    }
    for index, name in enumerate(result.component_names):
        if result.component_mask[index] and families[index] not in ("channel", "control"):
            posterior[scalar_names[name]] = (
                ("chain", "draw", "date"),
                _numeric(result.components[None, ..., index], widen_float32=True),
            )
    return posterior


def _result_groups(
    xr,
    arviz_base,
    result: DecompositionResult,
    *,
    observed=None,
    truth=None,
    graph_truth=None,
    named: bool = False,
):
    """One window's checked groups and identity, in the generic or named schema."""
    active = result.component_mask
    if not active.any():
        raise ValueError("DataTree export needs at least one active component")
    edges = result.edge_mask
    has_edges = bool(edges.any())
    evidence = _checked_evidence(result, truth, graph_truth, observed, stored=False, partial=True)
    edge_constants = _edge_constants(result.provenance, result.graphs.shape[1])
    attrs = _result_attrs(result)
    time_dim = "date" if named else "time"
    families = _labels(result.component_families, "family")
    coords = {
        "chain": np.array([0]),
        "draw": np.arange(result.n_samples),
        time_dim: _time_coordinate(np.array(result.time)),
        "component": _labels(result.component_names, "component")[active],
        "edge": _labels(result.edge_names, "edge")[edges],
    }
    posterior = (
        _named_posterior(result, families, coords)
        if named
        else {
            "components": (
                ("chain", "draw", time_dim, "component"),
                _numeric(result.components[None][..., active], widen_float32=True),
            )
        }
    )
    constants = {"component_family": (("component",), families[active])}
    truth_group = {}
    if "truth" in evidence:
        truth_group["components"] = ((time_dim, "component"), evidence["truth"][:, active])
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
            xr, arviz_base, {"outcome": ((time_dim,), evidence["observed"])}, coords
        )
    if truth_group:
        groups["truth"] = _dataset(xr, arviz_base, truth_group, coords)
    return groups, attrs


def result_datatree(
    result: DecompositionResult, *, observed=None, truth=None, graph_truth=None
) -> DataTree:
    """See :meth:`DecompositionResult.to_datatree`."""
    xr, arviz_base = _require_arviz()
    groups, attrs = _result_groups(
        xr, arviz_base, result, observed=observed, truth=truth, graph_truth=graph_truth
    )
    return _tree(xr, arviz_base, groups, attrs)


def prediction_datatree(
    context_result: DecompositionResult,
    prediction_snapshot: _PredictionSnapshot,
    *,
    revision: int,
    random_seed: int | None = None,
) -> DataTree:
    """Build a complete model tree from the two completed, separately sampled marginals.

    Covariates and observations retain their original units. Query observations are
    already sanitized by the dataframe adapter and never participate in inference.
    Every window owns its axes; matching draw indices do not imply paired draws.
    The root records ``random_seed`` when it is not ``None``.
    """
    xr, arviz_base = _require_arviz()
    history = prediction_snapshot.fitted_context
    start = len(history.dates)
    groups = {}
    for window, result, rows in (
        ("context", context_result, slice(0, start)),
        ("query", prediction_snapshot.query, slice(start, None)),
    ):
        window_groups, attrs = _result_groups(
            xr,
            arviz_base,
            result,
            observed=history.outcome if window == "context" else None,
            named=True,
        )
        attrs["provenance"] = json_bytes(result.provenance).decode("utf-8")
        window_groups["posterior"].attrs["provenance"] = attrs["provenance"]
        coords = {
            "date": _time_coordinate(result.time),
            "channel": _labels(history.channel_columns, "channel"),
            "control": _labels(history.control_columns, "control"),
        }
        constants = window_groups["constant_data"].assign_coords(coords)
        constants["media"] = (
            ("date", "channel"),
            _numeric(prediction_snapshot.observations["spend"][rows]),
        )
        constants["controls"] = (
            ("date", "control"),
            _numeric(
                prediction_snapshot.observations["controls"][rows, : len(history.control_columns)]
            ),
        )
        window_groups["constant_data"] = constants
        if window == "query":
            observed = prediction_snapshot.query_outcome
            if observed is not None and np.isfinite(observed).any():
                window_groups["observed_data"] = _dataset(
                    xr, arviz_base, {"outcome": (("date",), _numeric(observed))}, coords
                )
        for dataset in window_groups.values():
            for variable in dataset.variables.values():
                variable.data.setflags(write=False)
        groups[window] = _tree(xr, arviz_base, window_groups, attrs)
    return _tree(
        xr,
        arviz_base,
        groups,
        {
            "schema_version": _MODEL_TREE_SCHEMA_VERSION,
            "inference_revision": revision,
            "date_column": history.date_column,
            "target_column": history.target_column,
            "channel_columns": json_bytes(history.channel_columns).decode("utf-8"),
            "control_columns": json_bytes(history.control_columns).decode("utf-8"),
            **({} if random_seed is None else {"random_seed": random_seed}),
            **_NOTES,
        },
    )


def attach_response_curves(
    tree: DataTree,
    curves: ResponseCurves,
    prediction: _PredictionSnapshot,
    *,
    random_seed: int | None = None,
) -> DataTree:
    """Replace the entire curve subtree in a new root, retaining immutable draw buffers.

    Reconstruct nodes from datasets, never from live tree nodes. Shallow dataset copies
    retain the numeric buffers, while recursive metadata copies isolate mutable attrs
    and encodings on every dataset, coordinate and data variable. The curve attrs
    record ``random_seed`` when it is not ``None``.
    """
    xr, arviz_base = _require_arviz()
    attrs = {
        "source_inference_revision": tree.attrs["inference_revision"],
        "source_provenance": tree["query"].attrs["provenance"],
        "mode": prediction.mode,
        "units": curves.units,
        "n_samples": curves.n_samples,
        "weeks": curves.weeks,
        "graph_draws": _GRAPH_DRAWS[prediction.mode],
        "level_draws": (
            "levels are separate marginals: draws are not paired across levels; "
            "the unchanged reference may be reused across channels and zero-media levels"
        ),
        **({} if random_seed is None else {"random_seed": random_seed}),
        **_NOTES,
    }
    if prediction.mode == "model_average":
        attrs["router_samples"] = curves.n_samples
    elif prediction.mode == "map":
        attrs["router_samples"] = (
            curves.n_samples if prediction.router_samples is None else prediction.router_samples
        )
    coords = {
        "chain": np.array([0]),
        "draw": np.arange(curves.n_samples),
        "channel": _labels(curves.channels, "channel"),
        "level": _numeric(curves.levels),
    }
    curve_groups = {
        "/response_curves": _dataset(xr, arviz_base, {}, {}, attrs=attrs),
        "/response_curves/posterior": _dataset(
            xr,
            arviz_base,
            {
                "contribution": (
                    ("chain", "draw", "channel", "level"),
                    _numeric(curves.draws, widen_float32=True).transpose(2, 0, 1)[None],
                )
            },
            coords,
            sample_dims=_SAMPLE_DIMS,
            attrs=attrs,
        ),
        "/response_curves/constant_data": _dataset(
            xr,
            arviz_base,
            {"media": (("channel", "level"), curves.media[:, None] * curves.levels)},
            coords,
        ).assign_coords(date=("date", _time_coordinate(curves.dates))),
    }
    for dataset in curve_groups.values():
        for variable in dataset.variables.values():
            variable.data.setflags(write=False)
    groups = {}
    for node in tree.subtree:
        if node.path == "/response_curves" or node.path.startswith("/response_curves/"):
            continue
        dataset = node.to_dataset(inherit=False).copy(deep=False)
        dataset.attrs = deepcopy(dataset.attrs)
        dataset.encoding = deepcopy(dataset.encoding)
        for variable in dataset.variables.values():
            variable.attrs = deepcopy(variable.attrs)
            variable.encoding = deepcopy(variable.encoding)
        groups[node.path] = dataset
    return xr.DataTree.from_dict({**groups, **curve_groups}, name=tree.name)


def _labels(values, name: str) -> np.ndarray:
    """Labels as a NumPy text array, which would drop trailing NULs: NULs are refused."""
    values = list(values)
    if any("\x00" in value for value in values):
        raise ValueError(f"{name} labels cannot contain NUL characters")
    return np.array(values, dtype=str)


def _datetime_coordinates(arrays):
    """Choose one pandas/xarray resolution without changing any world's date labels."""
    for unit in ("ns", "us", "ms", "s"):
        converted = [values.astype(f"datetime64[{unit}]", copy=False) for values in arrays]
        if all(
            not np.isnat(dates).any()
            and np.array_equal(dates.astype(values.dtype, copy=False), values)
            for values, dates in zip(arrays, converted, strict=True)
        ):
            return converted
    raise ValueError("datetime64 coordinates need a lossless common time unit")


def _time_coordinate(values: np.ndarray) -> np.ndarray:
    """Pandas-compatible numeric or datetime coordinates, retaining native date values."""
    values = values.astype(values.dtype.newbyteorder("="), copy=False)
    if values.dtype.kind == "M":
        return _datetime_coordinates((values,))[0]
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
    time_kinds = {world["time"].dtype.kind for world in worlds}
    datetime_time = "M" in time_kinds
    if datetime_time:
        if len(time_kinds) != 1:
            raise ValueError("archive time coordinates cannot mix numeric and datetime64 labels")
        coordinates = _datetime_coordinates([world["time"] for world in worlds])
        for world, dates in zip(worlds, coordinates, strict=True):
            world["time"] = dates
        time_axis = np.unique(np.concatenate([world["time"] for world in worlds]))
    else:
        # NumPy promotion can round large integer labels when worlds use different dtypes.
        time_representatives = {}
        for world in worlds:
            for label in world["time"].tolist():
                # Object indexes match numeric vectors to 0/1, but not to False/True.
                if label not in time_representatives or isinstance(
                    time_representatives[label], bool
                ):
                    time_representatives[label] = label
        time_labels = sorted(time_representatives.values())
        time_axis = np.asarray(time_labels)
        mixed_boolean_time = "b" in time_kinds and len(time_kinds) > 1
        if time_axis.tolist() != time_labels or mixed_boolean_time:
            time_axis = np.asarray(time_labels, dtype=object)
        time_position = {label: index for index, label in enumerate(time_labels)}
    shape = (len(worlds), len(time_axis), len(component_axis))
    components = np.full((1, n_samples, *shape), np.nan)
    truth_values = np.full(shape, np.nan)
    outcome = np.full(shape[:2], np.nan)
    for index, world in enumerate(worlds):
        rows = (
            np.searchsorted(time_axis, world["time"])
            if datetime_time
            else np.array([time_position[label] for label in world["time"].tolist()], dtype=np.intp)
        )
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
