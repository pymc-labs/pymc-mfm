"""``validate_idata``, ``save_idata`` and ``load_idata``: model tree files, no overwrite."""

import errno
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import xarray as xr
from conftest import CHART_ROLES

import pymc_mfm
from pymc_mfm import load_idata, save_idata, validate_idata
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.plotting.charts._data import unique_names


@pytest.fixture
def tree(model, chart_frames):
    """A model tree with both windows and response curves."""
    history, future = chart_frames
    torch.manual_seed(3)
    model.inference(history, future, n_samples=4, **CHART_ROLES)
    model.response_curves(levels=[0, 1, 2], weeks=2, n_samples=3)
    return model.idata


@pytest.fixture
def files():
    """Require the optional h5netcdf, h5py and xarray only for tests that write files."""
    for name in ("h5netcdf", "h5py"):
        pytest.importorskip(name)


def _without(tree, group, name):
    """A copy of ``tree`` whose ``group`` lacks the variable or attribute ``name``."""
    copy = tree.copy(deep=True)
    node = copy[group] if group != "/" else copy
    if name in node.attrs:
        del node.attrs[name]
    else:
        node.dataset = node.to_dataset(inherit=False).drop_vars(name)
    return copy


def test_validate_accepts_a_raw_tree_and_unknown_additions(tree):
    validate_idata(tree)
    extended = tree.copy(deep=True)
    extended.attrs["note"] = "kept"
    extended["context"].attrs["extra"] = 1
    extended["context/posterior"].dataset = (
        extended["context/posterior"].to_dataset(inherit=False).assign(extra=("date", np.zeros(24)))
    )
    extended["notes"] = xr.DataTree(xr.Dataset({"x": ("i", [1.0])}))
    validate_idata(extended)


@pytest.mark.parametrize("mode", ["map", "fixed"])
@pytest.mark.usefixtures("files")
def test_validate_accepts_raw_trees_from_every_mode(model, chart_frames, tmp_path, mode):
    history, future = chart_frames
    options = {"n_samples": 3, "mode": mode, "units": "USD", "provenance": {"scenario": "plan"}}
    if mode == "fixed":
        layout = EdgeSlotLayout(2, model.config.n_controls, model.config.n_demand)
        graphs = np.zeros((3, 1, layout.n_slots))
        graphs[0, 0, layout.slices["cy"].start] = 1
        options["graph"] = graphs
    model.inference(history, future, **CHART_ROLES, **options)
    model.response_curves(levels=[0, 1], weeks=1)
    validate_idata(model.idata)
    save_idata(model.idata, tmp_path / "tree.nc")
    load_idata(tmp_path / "tree.nc")


def test_validate_accepts_a_tree_without_curves(tree):
    plain = tree.copy(deep=True)
    del plain["response_curves"]
    validate_idata(plain)


def test_validate_accepts_variables_stored_in_another_dim_order(tree):
    for window in ("context", "query"):
        node = tree[f"{window}/constant_data"]
        data = node.to_dataset(inherit=False)
        node.dataset = data.assign(
            media=data["media"].transpose("channel", "date"),
            controls=data["controls"].transpose("control", "date"),
        )
    curves = tree["response_curves/constant_data"]
    data = curves.to_dataset(inherit=False)
    curves.dataset = data.assign(media=data["media"].transpose("level", "channel"))
    validate_idata(tree)


def test_validate_rejects_a_newer_schema_with_an_upgrade_hint(tree):
    tree.attrs["schema_version"] = 2
    with pytest.raises(ValueError, match=r"^/: schema_version 2 is newer.*upgrade pymc-mfm"):
        validate_idata(tree)


def test_validate_rejects_a_missing_role_attribute(tree):
    with pytest.raises(ValueError, match=r"^/: attribute 'control_columns' is missing"):
        validate_idata(_without(tree, "/", "control_columns"))


def test_validate_rejects_role_attributes_that_disagree_with_the_variables(tree):
    tree.attrs["channel_columns"] = '["search", "radio"]'
    with pytest.raises(ValueError, match=r"^/context/constant_data: channel coordinate"):
        validate_idata(tree)


def test_validate_rejects_covariate_dates_that_differ_from_posterior_dates(tree):
    constants = tree["context/constant_data"]
    data = constants.to_dataset(inherit=False)
    constants.dataset = data.assign_coords(date=data["date"] + np.timedelta64(1, "D"))
    with pytest.raises(
        ValueError,
        match=r"^/context/constant_data: date coordinate differs from /context/posterior",
    ):
        validate_idata(tree)


def test_validate_names_the_missing_variable_and_group(tree):
    with pytest.raises(ValueError, match=r"^/query/posterior/channel_contribution: variable"):
        validate_idata(_without(tree, "query/posterior", "channel_contribution"))
    with pytest.raises(ValueError, match=r"^/response_curves/constant_data: group is missing"):
        copy = tree.copy(deep=True)
        del copy["response_curves"]["constant_data"]
        validate_idata(copy)


def test_validate_names_the_path_of_a_failing_read(tree):
    with pytest.raises(ValueError, match=r"^/query: cannot be read"):
        validate_idata(_without(tree, "query/posterior", "units"))


def test_validate_rejects_curves_from_another_inference(tree):
    tree["response_curves"].attrs["source_inference_revision"] = 7
    with pytest.raises(ValueError, match=r"^/response_curves: source_inference_revision 7"):
        validate_idata(tree)


@pytest.mark.usefixtures("files")
def test_save_then_load_returns_an_equal_tree_with_the_file_closed(tree, tmp_path):
    import h5py

    path = tmp_path / "tree.nc"
    save_idata(tree, path)
    loaded = load_idata(path)
    for node in loaded.subtree:
        for name, variable in node.to_dataset(inherit=False).data_vars.items():
            assert isinstance(variable.variable._data, np.ndarray), f"{node.path}/{name} is lazy"
    with h5py.File(path, "r+"):  # fails while the loader still holds the file
        pass
    assert [node.path for node in loaded.subtree] == [node.path for node in tree.subtree]
    for node in tree.subtree:
        xr.testing.assert_equal(
            node.to_dataset(inherit=False), loaded[node.path].to_dataset(inherit=False)
        )
    path.unlink()
    # The data stay readable after the file is gone.
    np.testing.assert_array_equal(
        loaded["query/posterior/channel_contribution"].to_numpy(),
        tree["query/posterior/channel_contribution"].to_numpy(),
    )
    assert list(tmp_path.iterdir()) == []


def test_validate_rejects_repeated_component_labels_without_hanging(tree):
    constants = tree["query/constant_data"]
    data = constants.to_dataset(inherit=False)
    labels = data["component"].to_numpy().copy()
    labels[1] = labels[0]
    constants.dataset = data.assign_coords(component=labels)
    with pytest.raises(ValueError, match=r"^/query/constant_data: component labels must be unique"):
        validate_idata(tree)
    with pytest.raises(ValueError, match="component names must be unique"):
        unique_names(["a", "a"], ["x", "x"])


def test_validate_rejects_a_missing_reader_coordinate(tree):
    posterior = tree["query/posterior"]
    posterior.dataset = posterior.to_dataset(inherit=False).drop_vars("date")
    with pytest.raises(ValueError, match=r"^/query/posterior: date coordinate is missing"):
        validate_idata(tree)


@pytest.mark.parametrize(
    ("group", "dim", "path"),
    [
        ("query/posterior", "draw", "/query/posterior"),
        ("context/posterior", "chain", "/context/posterior"),
        ("context/constant_data", "date", "/context/constant_data"),
        ("response_curves/posterior", "draw", "/response_curves/posterior"),
    ],
)
def test_validate_rejects_empty_sample_or_date_dimensions(tree, group, dim, path):
    node = tree[group]
    node.dataset = node.to_dataset(inherit=False).isel({dim: slice(0, 0)})
    with pytest.raises(ValueError, match=rf"^{path}: {dim} has no values"):
        validate_idata(tree)


def test_validate_rejects_a_value_that_is_not_a_tree():
    with pytest.raises(ValueError, match=r"^/: expected an xarray DataTree, got NoneType"):
        validate_idata(None)


def test_validate_rejects_dates_that_are_not_datetimes(tree):
    node = tree["query/posterior"]
    data = node.to_dataset(inherit=False)
    node.dataset = data.assign_coords(date=np.arange(data.sizes["date"]))
    with pytest.raises(
        ValueError, match=r"^/query/posterior: date coordinate must hold datetime64"
    ):
        validate_idata(tree)


def test_validate_rejects_curves_without_a_positive_level(tree):
    for group in ("response_curves/posterior", "response_curves/constant_data"):
        node = tree[group]
        data = node.to_dataset(inherit=False)
        node.dataset = data.assign_coords(level=np.zeros(data.sizes["level"]))
    with pytest.raises(ValueError, match=r"^/response_curves/posterior: level coordinate needs"):
        validate_idata(tree)


@pytest.mark.parametrize(
    ("group", "name", "value", "path"),
    [
        ("/", "schema_version", 1.5, "/"),
        ("/", "schema_version", True, "/"),
        ("/", "schema_version", "bad", "/"),
        ("/", "schema_version", None, "/"),
        ("/", "inference_revision", "1", "/"),
        ("/", "channel_columns", ["search", "tv"], "/"),
        ("/", "control_columns", "not json", "/"),
        ("response_curves", "source_inference_revision", 1.0, "/response_curves"),
    ],
)
def test_validate_rejects_malformed_attributes(tree, group, name, value, path):
    (tree if group == "/" else tree[group]).attrs[name] = value
    with pytest.raises(ValueError, match=rf"^{path}: .*{name}"):
        validate_idata(tree)


def test_validate_accepts_numpy_integer_attributes(tree):
    tree.attrs["schema_version"] = np.int64(1)
    tree.attrs["inference_revision"] = np.int64(tree.attrs["inference_revision"])
    tree["response_curves"].attrs["source_inference_revision"] = np.int64(
        tree.attrs["inference_revision"]
    )
    validate_idata(tree)


@pytest.mark.usefixtures("files")
def test_save_never_overwrites_and_leaves_no_temporary_file(tree, tmp_path):
    path = tmp_path / "tree.nc"
    path.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        save_idata(tree, path)
    assert path.read_bytes() == b"keep me"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.usefixtures("files")
def test_save_without_hard_links_still_refuses_to_overwrite(tree, tmp_path, monkeypatch):
    def no_links(source, target):
        raise OSError(errno.EPERM, "hard links are not supported")

    monkeypatch.setattr(os, "link", no_links)
    path = tmp_path / "tree.nc"
    save_idata(tree, path)
    validate_idata(load_idata(path))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_idata(tree, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    # A copy that fails part way removes its partial file, so a retry can succeed.
    other = tmp_path / "other.nc"

    def fail_mid_copy(source, target):
        target.write(b"x" * 100)
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(shutil, "copyfileobj", fail_mid_copy)
    with pytest.raises(OSError, match="disk full"):
        save_idata(tree, other)
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.undo()
    save_idata(tree, other)
    validate_idata(load_idata(other))


@pytest.mark.usefixtures("files")
def test_a_long_legal_name_saves(tree, tmp_path):
    path = tmp_path / ("n" * 240 + ".nc")
    save_idata(tree, path)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.usefixtures("files")
def test_a_failed_write_leaves_no_temporary_file(tree, tmp_path, monkeypatch):
    def broken(self, path, **kwargs):
        open(path, "wb").close()
        raise RuntimeError("disk full")

    monkeypatch.setattr(xr.DataTree, "to_netcdf", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        save_idata(tree, tmp_path / "tree.nc")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("files")
def test_save_validates_before_it_writes(tree, tmp_path):
    with pytest.raises(ValueError, match="control_columns"):
        save_idata(_without(tree, "/", "control_columns"), tmp_path / "tree.nc")
    assert list(tmp_path.iterdir()) == []


def test_the_names_are_top_level_exports():
    assert {"save_idata", "load_idata", "validate_idata"} <= set(pymc_mfm.__all__)


def _run(script, *args):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        env={"MPLBACKEND": "Agg", "PATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.usefixtures("files")
def test_a_file_loads_and_draws_with_torch_blocked(tree, tmp_path):
    save_idata(tree, tmp_path / "tree.nc")
    run = _run(
        """
        import sys

        sys.modules["torch"] = None  # any import of torch now raises ImportError
        from pymc_mfm import Plots, load_idata

        Plots(load_idata(sys.argv[1])).fit()
        """,
        str(tmp_path / "tree.nc"),
    )
    assert run.returncode == 0, run.stderr


def test_without_h5netcdf_the_error_names_the_draws_extra(tree, tmp_path):
    run = _run(
        """
        import sys

        sys.modules["h5netcdf"] = None
        from pymc_mfm import load_idata

        try:
            load_idata(sys.argv[1])
        except ImportError as error:
            assert 'pip install "pymc-mfm[draws]"' in str(error), error
        else:
            raise SystemExit("load_idata ran without h5netcdf")
        """,
        str(tmp_path / "tree.nc"),
    )
    assert run.returncode == 0, run.stderr
