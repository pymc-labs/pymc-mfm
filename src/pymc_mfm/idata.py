"""Check a model tree, write it to one netCDF file and read it back, all without PyTorch.

A model tree is ``MFM.idata`` or a file made by :func:`save_idata`. Files need the
``[draws]`` extra: ``pip install "pymc-mfm[draws]"``.
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

from pymc_mfm.plotting.charts._data import WINDOWS, IdataReader

if TYPE_CHECKING:
    from xarray import Dataset, DataTree

_EXTRA = (
    'Saving and loading a model tree needs the pymc-mfm[draws] extra: pip install "pymc-mfm[draws]"'
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


def _group(tree: DataTree, *names: str) -> Dataset:
    path = "/" + "/".join(names)
    node = tree
    for name in names:
        if name not in node.children:
            _fail(path, "group is missing")
        node = node[name]
    data = node.to_dataset(inherit=False)
    for dim, size in data.sizes.items():
        if dim in _INDEXED_DIMS and dim not in data.coords:
            _fail(path, f"{dim} coordinate is missing")
        if dim in _NONEMPTY_DIMS and size == 0:
            _fail(path, f"{dim} has no values")
    if "date" in data.coords and not np.issubdtype(data["date"].dtype, np.datetime64):
        _fail(path, f"date coordinate must hold datetime64 values, got {data['date'].dtype}")
    return data


def _variable(data: Dataset, path: str, name: str, dims: tuple[str, ...]):
    if name not in data:
        _fail(f"{path}/{name}", "variable is missing")
    # Readers select by dim name, so any order of the same dims is valid.
    if sorted(data[name].dims) != sorted(dims):
        _fail(f"{path}/{name}", f"expected dims {dims} in any order, got {data[name].dims}")
    return data[name]


def _same_coordinates(path: str, groups: dict[str, Dataset], dims: tuple[str, ...]) -> None:
    """Each of ``dims`` has one coordinate in every group of ``groups`` that has it."""
    for dim in dims:
        found = [
            (name, data[dim].to_numpy()) for name, data in groups.items() if dim in data.coords
        ]
        for name, values in found[1:]:
            if not np.array_equal(values, found[0][1]):
                _fail(f"{path}/{name}", f"{dim} coordinate differs from {path}/{found[0][0]}")


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


def _check_window(tree: DataTree, window: str, roles: dict[str, tuple[str, ...]]) -> None:
    root = f"/{window}"
    posterior = _group(tree, window, "posterior")
    constants = _group(tree, window, "constant_data")
    groups = {"posterior": posterior, "constant_data": constants}
    if "observed_data" in tree[window].children:
        observed = _group(tree, window, "observed_data")
        _variable(observed, f"{root}/observed_data", "outcome", ("date",))
        groups["observed_data"] = observed
    families = _variable(constants, f"{root}/constant_data", "component_family", ("component",))
    labels = families["component"].to_numpy()
    if len(set(labels)) != len(labels):
        _fail(f"{root}/constant_data", f"component labels must be unique, got {list(labels)}")
    for label, family in zip(labels, families.to_numpy(), strict=True):
        if family in roles:
            _variable(
                posterior,
                f"{root}/posterior",
                f"{family}_contribution",
                (*_SAMPLE_DIMS, "date", family),
            )
        else:
            _variable(
                posterior,
                f"{root}/posterior",
                str(label).replace(":", "_"),
                (*_SAMPLE_DIMS, "date"),
            )
    _variable(constants, f"{root}/constant_data", "media", ("date", "channel"))
    _variable(constants, f"{root}/constant_data", "controls", ("date", "control"))
    _same_coordinates(root, groups, ("date", *_ROLES))
    for role, names in roles.items():
        recorded = tuple(str(name) for name in constants[role].to_numpy())
        if recorded != names:
            _fail(
                f"{root}/constant_data",
                f"{role} coordinate {recorded} differs from "
                f"root attribute '{role}_columns' {names}",
            )


def _check_curves(tree: DataTree) -> None:
    root = "/response_curves"
    node = tree["response_curves"]
    for name in ("weeks", "units", "n_samples"):
        _attr(node, root, name)
    source = _integer_attr(node, root, "source_inference_revision")
    revision = _integer_attr(tree, "/", "inference_revision")
    if source != revision:
        _fail(
            root,
            f"source_inference_revision {source} differs from the tree's "
            f"inference_revision {revision}; run response_curves again",
        )
    posterior = _group(tree, "response_curves", "posterior")
    constants = _group(tree, "response_curves", "constant_data")
    _variable(posterior, f"{root}/posterior", "contribution", (*_SAMPLE_DIMS, "channel", "level"))
    _variable(constants, f"{root}/constant_data", "media", ("channel", "level"))
    # The reader recovers observed media at level 1, or else at the top level.
    if not (posterior["level"].to_numpy() > 0).any():
        _fail(f"{root}/posterior", "level coordinate needs at least one positive level")
    if "date" not in constants.coords:
        _fail(f"{root}/constant_data", "date coordinate is missing")
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
    except Exception as error:
        raise ValueError(f"{path}: cannot be read: {type(error).__name__}: {error}") from error


def validate_idata(idata: DataTree) -> None:
    """Raise ``ValueError`` unless ``idata`` is a schema 1 model tree.

    Each error names the path that fails. The check covers the root attributes, the
    groups, dimensions and coordinates of both windows, and the curves when present.
    It then runs every chart read on both windows and on the curves. Unknown variables,
    groups and attributes are ignored, so ``MFM.idata`` as built is valid.
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
            f"schema_version {version} is newer than this pymc-mfm reads ({_SCHEMA_VERSION}); "
            "upgrade pymc-mfm",
        )
    if version != _SCHEMA_VERSION:
        _fail("/", f"schema_version must be {_SCHEMA_VERSION}, got {version}")
    _integer_attr(idata, "/", "inference_revision")
    roles = _roles(idata)
    for window in WINDOWS:
        _check_window(idata, window, roles)
    curved = "response_curves" in idata.children
    if curved:
        _check_curves(idata)
    reader = IdataReader(idata)
    for window in WINDOWS:
        _read(f"/{window}", lambda window=window: reader.window(window))
    _read("/context/constant_data and /query/constant_data", reader.covariates)
    if curved:
        _read("/response_curves", reader.response_curves)


def _require_draws():
    """``xarray``, or an ``ImportError`` naming the extra."""
    try:
        import h5netcdf  # noqa: F401
        import xarray
    except ImportError as error:
        raise ImportError(_EXTRA) from error
    return xarray


def save_idata(idata: DataTree, path: str | os.PathLike[str]) -> None:
    """Write the model tree ``idata`` to the netCDF file ``path``; never overwrite.

    The tree is checked by :func:`validate_idata` first. The file is written beside
    ``path`` and then hard-linked to it, so ``path`` appears whole or not at all, and
    ``FileExistsError`` leaves an existing file as it was. A file system without hard
    links gets an exclusive create and a copy; a failed copy removes the file it made,
    but a crash there can leave a partial file.
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
            with target.open("xb") as out:
                try:
                    with temporary.open("rb") as written:
                        shutil.copyfileobj(written, out)
                except BaseException:
                    out.close()
                    target.unlink(missing_ok=True)  # the file this call created
                    raise
    finally:
        temporary.unlink(missing_ok=True)


def load_idata(path: str | os.PathLike[str]) -> DataTree:
    """Read the model tree in the netCDF file ``path`` into memory and validate it.

    The file is closed when this returns.
    """
    xarray = _require_draws()
    with xarray.open_datatree(path, engine="h5netcdf") as tree:
        idata = tree.load()
    validate_idata(idata)
    return idata
