"""Check a model tree, write it to one netCDF file and read it back, all without PyTorch.

A model tree is ``MFM.idata`` or a file made by :func:`save_idata`. Files need the
``[draws]`` extra: ``pip install "pymc-mktfm[draws]"``.
"""

from __future__ import annotations

import json
import numbers
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mktfm.plotting.charts._data import FAMILIES, WINDOWS, IdataReader

if TYPE_CHECKING:
    from xarray import Dataset, DataTree

_EXTRA = (
    "Saving and loading a model tree needs the pymc-mktfm[draws] extra: pip install "
    '"pymc-mktfm[draws]"'
)
_SCHEMA_VERSION = 1
_SAMPLE_DIMS = ("chain", "draw")
_ROLES = ("channel", "control")
#: Dimensions the readers select by coordinate, and those that must hold at least one value.
_INDEXED_DIMS = ("chain", "draw", "date", "channel", "control", "level", "component")
_NONEMPTY_DIMS = ("chain", "draw", "date")


def _fail(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _attr(node: DataTree, path: str, name: str) -> Any:
    if name not in node.attrs:
        _fail(path, f"attribute {name!r} is missing")
    return node.attrs[name]


def _integer_attr(node: DataTree, path: str, name: str) -> int:
    value = _attr(node, path, name)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail(path, f"attribute {name!r} must be an integer, got {value!r}")
    return int(value)


_MAX_SEED = 2**31 - 1


def _seed_attr(node: DataTree, path: str) -> int | None:
    """The recorded ``random_seed``, or ``None`` when absent, as in an older file.

    A file gives it back as a NumPy integer, which is accepted; a bool, float or text is not.
    """
    if "random_seed" not in node.attrs:
        return None
    value = node.attrs["random_seed"]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail(path, f"attribute 'random_seed' must be an integer, got {value!r}")
    if not 0 <= value <= _MAX_SEED:
        _fail(
            path,
            f"attribute 'random_seed' must be an integer from 0 to {_MAX_SEED}, got {value!r}",
        )
    return int(value)


def _join(path: str, name: str) -> str:
    return f"{path.rstrip('/')}/{name}"


def _group(tree: DataTree, *names: str) -> Dataset:
    path = "/" + "/".join(names)
    node = tree
    for name in names:
        if name not in node.children:
            _fail(path, "group is missing")
        node = node[name]
    return node.to_dataset(inherit=False)


def _variable(data: Dataset, path: str, name: str, dims: tuple[str, ...], *, numeric: bool = True):
    """The schema variable ``name`` of group ``path``, checked; coordinates are checked here.

    Only the dimensions this variable uses need a coordinate, so a dimension that an
    extension variable alone uses is left alone.
    """
    if name not in data:
        _fail(_join(path, name), "variable is missing")
    variable = data[name]
    # Readers select by dim name, so any order of the same dims is valid.
    if sorted(variable.dims) != sorted(dims):
        _fail(_join(path, name), f"expected dims {dims} in any order, got {variable.dims}")
    for dim in dims:
        if dim in _INDEXED_DIMS and dim not in data.coords:
            _fail(path, f"{dim} coordinate is missing (variable {name!r} uses it)")
        if dim in _NONEMPTY_DIMS and variable.sizes[dim] == 0:
            _fail(path, f"{dim} has no values")
        if dim == "date" and not np.issubdtype(data["date"].dtype, np.datetime64):
            _fail(path, f"date coordinate must hold datetime64 values, got {data['date'].dtype}")
    if numeric and not _is_number(variable.dtype):
        _fail(_join(path, name), f"must hold numbers, got dtype {variable.dtype}")
    return variable


def _is_number(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer)


def _one_chain(path: str, data: Dataset) -> None:
    """The charts read chain 0, so any other chain would be dropped without a word."""
    if data.sizes["chain"] != 1:
        _fail(path, f"charts read a tree with one chain, got {data.sizes['chain']}")


def _same_coordinates(path: str, groups: dict[str, Dataset], dims: tuple[str, ...]) -> None:
    """Each of ``dims`` holds the same labels, in any order, in every group that has it."""
    for dim in dims:
        found = [
            (name, np.sort(data[dim].to_numpy()))
            for name, data in groups.items()
            if dim in data.coords
        ]
        for name, values in found[1:]:
            if not np.array_equal(values, found[0][1]):
                _fail(f"{path}/{name}", f"{dim} coordinate differs from {path}/{found[0][0]}")


def _dates(path: str, dates: np.ndarray, *, least: int) -> None:
    if len(dates) < least:
        _fail(path, f"date coordinate needs at least {least} dates, got {len(dates)}")
    if len(dates) > 1 and not (np.diff(dates) > np.timedelta64(0, "ns")).all():
        _fail(path, "date coordinate must be strictly increasing")


def _roles(tree: DataTree) -> dict[str, tuple[str, ...]]:
    """The channel and control names the root records."""
    roles = {}
    for role in _ROLES:
        name = f"{role}_columns"
        text = _attr(tree, "/", name)
        try:
            names = json.loads(text) if isinstance(text, str) else None
        except json.JSONDecodeError:
            names = None
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            _fail("/", f"attribute {name!r} must be JSON text holding a list of names")
        roles[role] = tuple(names)
    return roles


def _text_attrs(path: str, attrs: dict[str, Any]) -> None:
    for name, value in attrs.items():
        if not isinstance(value, str):
            continue
        if type(value) is not str:
            _fail(path, f"attribute {name!r} must be a built-in str, got {type(value).__name__}")
        if "\x00" in value:
            _fail(path, f"attribute {name!r} must not contain a NUL character")


def _check_text(tree: DataTree) -> None:
    """Text that HDF5 cannot store: attribute text that is not ``str``, and NUL in any text."""
    for node in tree.subtree:
        _text_attrs(node.path, node.attrs)
        for name, variable in node.to_dataset(inherit=False).variables.items():
            where = _join(node.path, str(name))
            _text_attrs(where, variable.attrs)
            if variable.dtype.kind not in "OUS":
                continue  # numbers hold no text, and reading them loads every draw
            values = variable.to_numpy()
            if values.dtype.kind == "O" and not all(isinstance(item, str) for item in values.flat):
                _fail(where, "labels must be str")
            if values.dtype.kind in "OU" and any("\x00" in str(item) for item in values.flat):
                _fail(where, "labels must not contain a NUL character")


def _provenance(node: DataTree, path: str) -> str:
    """The recorded provenance text; it must be a JSON object."""
    text = _attr(node, path, "provenance")
    try:
        value = json.loads(text) if isinstance(text, str) else None
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        _fail(path, "attribute 'provenance' must be JSON text holding an object")
    return text


def _check_window(tree: DataTree, window: str, roles: dict[str, tuple[str, ...]]) -> None:
    root = f"/{window}"
    _provenance(tree[window], root)
    posterior = _group(tree, window, "posterior")
    constants = _group(tree, window, "constant_data")
    groups = {"posterior": posterior, "constant_data": constants}
    children = tree[window].children
    if window == "context" and "observed_data" not in children:
        _fail(f"{root}/observed_data", "group is missing; the history needs its outcome")
    if "observed_data" in children:
        observed = _group(tree, window, "observed_data")
        _variable(observed, f"{root}/observed_data", "outcome", ("date",))
        groups["observed_data"] = observed
    families = _variable(
        constants, f"{root}/constant_data", "component_family", ("component",), numeric=False
    )
    labels = families["component"].to_numpy()
    if len(set(labels)) != len(labels):
        _fail(f"{root}/constant_data", f"component labels must be unique, got {list(labels)}")
    component_families = families.to_numpy()
    described_roles = tuple(
        dict.fromkeys(family for family in component_families if family in roles)
    )
    for family in described_roles:
        _variable(
            posterior,
            f"{root}/posterior",
            f"{family}_contribution",
            (*_SAMPLE_DIMS, "date", family),
        )
    for label, family in zip(labels, component_families, strict=True):
        if family not in roles and family in FAMILIES:
            _variable(
                posterior,
                f"{root}/posterior",
                str(label).replace(":", "_"),
                (*_SAMPLE_DIMS, "date"),
            )
    if "graph" in posterior:
        graph = _variable(posterior, f"{root}/posterior", "graph", (*_SAMPLE_DIMS, "edge"))
        if not np.isin(graph.to_numpy(), (0, 1)).all():
            _fail(_join(f"{root}/posterior", "graph"), "must be binary: 0 or 1")
    _variable(constants, f"{root}/constant_data", "media", ("date", "channel"))
    _variable(constants, f"{root}/constant_data", "controls", ("date", "control"))
    _one_chain(f"{root}/posterior", posterior)
    _dates(f"{root}/posterior", posterior["date"].to_numpy(), least=2 if window == "context" else 1)
    _same_coordinates(root, groups, ("date",))
    _same_coordinates(root, {"posterior": posterior, "constant_data": constants}, described_roles)
    for role, names in roles.items():
        recorded = tuple(str(name) for name in constants[role].to_numpy())
        if sorted(recorded) != sorted(names):
            _fail(
                f"{root}/constant_data",
                f"{role} coordinate {recorded} differs from "
                f"root attribute '{role}_columns' {names}",
            )


def _check_curves(tree: DataTree, roles: dict[str, tuple[str, ...]]) -> None:
    root = "/response_curves"
    node = tree["response_curves"]
    weeks = _integer_attr(node, root, "weeks")
    n_samples = _integer_attr(node, root, "n_samples")
    for name, count in (("weeks", weeks), ("n_samples", n_samples)):
        if count < 1:
            _fail(root, f"attribute {name!r} must be positive, got {count}")
    units = _attr(node, root, "units")
    if not isinstance(units, str) or not units:
        _fail(root, f"attribute 'units' must be nonempty text, got {units!r}")
    source = _integer_attr(node, root, "source_inference_revision")
    revision = _integer_attr(tree, "/", "inference_revision")
    if source != revision:
        _fail(
            root,
            f"source_inference_revision {source} differs from the tree's "
            f"inference_revision {revision}; run response_curves again",
        )
    # The revision counts one model's runs, so the provenance tells apart two models.
    if _attr(node, root, "source_provenance") != tree["query"].attrs["provenance"]:
        _fail(
            root,
            "source_provenance differs from the provenance of /query; run response_curves again",
        )
    posterior = _group(tree, "response_curves", "posterior")
    constants = _group(tree, "response_curves", "constant_data")
    path = f"{root}/posterior"
    # One sweep writes one seed to both groups.
    seed = _seed_attr(node, root)
    posterior_seed = _seed_attr(tree["response_curves/posterior"], path)
    if seed != posterior_seed:
        _fail(
            path,
            f"attribute 'random_seed' is {posterior_seed!r} but {root} records {seed!r}; "
            "run response_curves again",
        )
    _variable(posterior, path, "contribution", (*_SAMPLE_DIMS, "channel", "level"))
    _variable(constants, f"{root}/constant_data", "media", ("channel", "level"))
    _one_chain(path, posterior)
    if posterior.sizes["draw"] != n_samples:
        _fail(
            path,
            f"attribute n_samples {n_samples} differs from the {posterior.sizes['draw']} draws",
        )
    channels = [str(name) for name in posterior["channel"].to_numpy()]
    if not channels:
        _fail(
            path,
            f"channel coordinate must hold at least one of {list(roles['channel'])}; "
            f"keep at least one channel",
        )
    unknown = [name for name in channels if name not in roles["channel"]]
    if len(set(channels)) != len(channels) or unknown:
        _fail(
            path,
            f"channel coordinate {channels} must hold each of {list(roles['channel'])} at most "
            f"once, and no other name",
        )
    levels = posterior["level"].to_numpy()
    if not _is_number(levels.dtype):
        _fail(path, f"level coordinate must hold numbers, got dtype {levels.dtype}")
    if not (np.isfinite(levels).all() and (levels >= 0).all()):
        _fail(path, "level coordinate must hold finite values of at least 0")
    if not (levels > 0).any():
        _fail(path, "level coordinate needs at least one positive level")
    if len(levels) < 2 or not (levels[1:] > levels[:-1]).all():
        _fail(path, f"level coordinate needs at least 2 values, strictly increasing, got {levels}")
    if "date" not in constants.coords:
        _fail(f"{root}/constant_data", "date coordinate is missing")
    dates = constants["date"].to_numpy()
    if not np.issubdtype(dates.dtype, np.datetime64):
        _fail(f"{root}/constant_data", f"date coordinate must hold datetime64, got {dates.dtype}")
    query_dates = tree["query/posterior"].to_dataset(inherit=False)["date"].to_numpy()
    if len(dates) != weeks or not np.array_equal(np.sort(dates), query_dates[:weeks]):
        _fail(
            f"{root}/constant_data",
            f"date coordinate must hold the first {weeks} dates of /query/posterior "
            f"(attribute 'weeks'), got {len(dates)} dates",
        )
    _same_coordinates(
        root, {"posterior": posterior, "constant_data": constants}, ("channel", "level")
    )


def _is_tree(value: object) -> bool:
    try:
        from xarray import DataTree
    except ImportError:  # without xarray nothing can be a DataTree
        return False
    return isinstance(value, DataTree)


def _read(path: str, read: Callable[[], Any]) -> None:
    try:
        read()
    except ValueError as error:
        if str(error).startswith("/"):  # the reader already names the path
            raise
        raise ValueError(f"{path}: cannot be read: ValueError: {error}") from error
    except Exception as error:
        raise ValueError(f"{path}: cannot be read: {type(error).__name__}: {error}") from error


def validate_idata(idata: DataTree) -> None:
    """Raise ``ValueError`` unless ``idata`` is a schema 1 model tree that saves and charts.

    A tree that passes can be written by :func:`save_idata` and drawn by ``Plots``. Each
    error names the path that fails and what is wrong. The check covers the root
    attributes, text that HDF5 cannot store, the groups, dimensions, coordinates and
    number types of both windows, the provenance, and the curves when present. The
    history needs its observed outcome and at least two increasing dates, and each
    posterior has one chain. Components and channels are paired by label, so their stored
    order does not matter. The check then runs the chart reads on both windows and on the
    curves, without stacking the draws. Unknown variables, groups and attributes need
    no schema entry, but their text must still be storable, so ``MFM.idata`` as built is valid.
    """
    if not _is_tree(idata):
        _fail(
            "/",
            f"expected an xarray DataTree, got {type(idata).__name__}; "
            "model.idata is None until the first complete inference",
        )
    version = _integer_attr(idata, "/", "schema_version")
    if version > _SCHEMA_VERSION:
        _fail(
            "/",
            f"schema_version {version} is newer than this pymc-mktfm reads ({_SCHEMA_VERSION}); "
            "upgrade pymc-mktfm",
        )
    if version != _SCHEMA_VERSION:
        _fail("/", f"schema_version must be {_SCHEMA_VERSION}, got {version}")
    _integer_attr(idata, "/", "inference_revision")
    _seed_attr(idata, "/")
    roles = _roles(idata)
    _check_text(idata)
    for window in WINDOWS:
        _check_window(idata, window, roles)
    curved = "response_curves" in idata.children
    if curved:
        _check_curves(idata, roles)
    reader = IdataReader(idata)
    for window in WINDOWS:
        _read(f"/{window}", lambda window=window: reader.check_window(window))
    _read("/context/constant_data and /query/constant_data", reader.covariates)
    if curved:
        _read("/response_curves", reader.response_curves)


def _require_draws():
    """``xarray``, or an ``ImportError`` naming the extra."""
    try:
        import h5netcdf  # noqa: F401
        import h5py  # noqa: F401
        import xarray
    except ImportError as error:
        raise ImportError(_EXTRA) from error
    return xarray


def save_idata(idata: DataTree, path: str | os.PathLike[str]) -> None:
    """Write the model tree ``idata`` to the netCDF file ``path``; never overwrite.

    The tree is checked by :func:`validate_idata` first. The file is written beside
    ``path`` and then hard-linked to it, so ``path`` appears whole or not at all, and
    ``FileExistsError`` leaves an existing file as it was. A file system without hard
    links gets an exclusive create and a copy; a failed copy or close removes the file it
    made, but a crash there can leave a partial file.
    """
    _require_draws()
    validate_idata(idata)
    target = Path(path)
    # A short name, so any legal target name leaves room for it.
    temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        idata.to_netcdf(temporary, engine="h5netcdf")
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise
        except OSError:
            # A failed exclusive create made no file, so it needs no cleanup.
            out = target.open("xb")
            try:
                with out, temporary.open("rb") as written:
                    shutil.copyfileobj(written, out)
            except BaseException:
                target.unlink(missing_ok=True)  # the file this call created
                raise
    finally:
        temporary.unlink(missing_ok=True)


def load_idata(path: str | os.PathLike[str]) -> DataTree:
    """Read the model tree in the netCDF file ``path`` into memory and validate it.

    Every variable is loaded before the file closes, which is when this returns.
    """
    xarray = _require_draws()
    with xarray.open_datatree(path, engine="h5netcdf") as tree:
        # Load node by node: DataTree.load() leaves a variable lazy when its name
        # matches the path of another group.
        nodes = {node.path: node.to_dataset(inherit=False).load() for node in tree.subtree}
        idata = xarray.DataTree.from_dict(nodes, name=tree.name)
    validate_idata(idata)
    return idata
