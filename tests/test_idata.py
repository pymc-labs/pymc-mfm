"""``validate_idata``, ``save_idata`` and ``load_idata``: model tree files, no overwrite."""

import errno
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import torch
import xarray as xr
from conftest import CHART_ROLES
from xarray.backends.common import BackendArray

import pymc_mktfm
from pymc_mktfm import load_idata, save_idata, validate_idata
from pymc_mktfm.model.graph import EdgeSlotLayout
from pymc_mktfm.plotting.charts._data import IdataReader, unique_names


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


def _edit(tree, group, change):
    """Replace the dataset of ``group`` in ``tree`` with ``change(dataset)``."""
    node = tree[group]
    node.dataset = change(node.to_dataset(inherit=False))


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
    with pytest.raises(ValueError, match=r"^/: schema_version 2 is newer.*upgrade pymc-mktfm"):
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


def test_validate_reports_the_group_of_a_missing_posterior_draw_coordinate(tree):
    _edit(tree, "query/posterior", lambda data: data.drop_vars("draw"))
    with pytest.raises(ValueError, match=r"^/query/posterior: draw coordinate is missing"):
        validate_idata(tree)


@pytest.mark.usefixtures("files")
def test_an_extension_variable_may_use_an_axis_without_a_coordinate(tree, tmp_path):
    _edit(tree, "query/constant_data", lambda data: data.assign(external_draws=("draw", [1.0, 2])))
    validate_idata(tree)
    save_idata(tree, tmp_path / "tree.nc")
    loaded = load_idata(tmp_path / "tree.nc")
    np.testing.assert_array_equal(loaded["query/constant_data/external_draws"].to_numpy(), [1, 2])


@pytest.mark.usefixtures("files")
def test_observations_can_carry_an_unrelated_channel_coordinate(tree, tmp_path):
    path = "context/observed_data"
    _edit(
        tree,
        path,
        lambda data: data.assign_coords(channel=["external"]).assign(sensor=("channel", [2.5])),
    )
    target = tmp_path / "tree.nc"
    save_idata(tree, target)
    sensor = load_idata(target)[f"{path}/sensor"]
    np.testing.assert_array_equal(sensor["channel"], ["external"])
    np.testing.assert_array_equal(sensor, [2.5])


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


_SEED_PLACES = ("/", "response_curves", "response_curves/posterior")
_MAX_SEED = 2**31 - 1


@pytest.fixture
def seeded(model, chart_frames):
    """A tree from a seeded inference (root seed 11) and seeded curves (seed 12)."""
    history, future = chart_frames
    model.inference(history, future, n_samples=4, random_seed=11, **CHART_ROLES)
    model.response_curves(levels=[0, 1, 2], weeks=2, n_samples=3, random_seed=12)
    return model.idata


def _node(tree, place):
    return tree if place == "/" else tree[place]


def test_validate_accepts_a_tree_that_records_no_seed(tree):
    for place in _SEED_PLACES:
        assert "random_seed" not in _node(tree, place).attrs
    validate_idata(tree)


@pytest.mark.parametrize("place", _SEED_PLACES)
@pytest.mark.parametrize(
    "value", ["not-an-integer", "7", 7.0, True, None, -1, _MAX_SEED + 1, np.float64(7.0)]
)
def test_validate_rejects_a_malformed_recorded_seed_naming_the_path(seeded, place, value):
    _node(seeded, place).attrs["random_seed"] = value
    with pytest.raises(ValueError, match=rf"^/{place.strip('/')}: .*'random_seed'.*integer"):
        validate_idata(seeded)


@pytest.mark.parametrize("value", [0, 1, _MAX_SEED, np.int64(5)])
def test_validate_accepts_a_seed_in_range_including_the_ends(seeded, value):
    for place in _SEED_PLACES:
        _node(seeded, place).attrs["random_seed"] = value
    validate_idata(seeded)


def test_validate_rejects_curve_seeds_that_disagree(seeded):
    seeded["response_curves/posterior"].attrs["random_seed"] = 13
    with pytest.raises(
        ValueError, match=r"^/response_curves/posterior: .*random_seed.* is 13 .*records 12"
    ):
        validate_idata(seeded)
    del seeded["response_curves/posterior"].attrs["random_seed"]
    with pytest.raises(ValueError, match=r"^/response_curves/posterior: .*random_seed"):
        validate_idata(seeded)


def test_validate_accepts_curves_that_record_no_seed_under_a_seeded_root(seeded):
    for place in ("response_curves", "response_curves/posterior"):
        del seeded[place].attrs["random_seed"]
    validate_idata(seeded)


@pytest.mark.usefixtures("files")
def test_a_seeded_tree_saves_loads_and_validates_with_its_seeds(seeded, tmp_path):
    validate_idata(seeded)
    assert seeded.attrs["random_seed"] == 11
    assert seeded["response_curves"].attrs["random_seed"] == 12
    save_idata(seeded, tmp_path / "seeded.nc")
    loaded = load_idata(tmp_path / "seeded.nc")
    validate_idata(loaded)
    assert loaded.attrs["random_seed"] == 11
    for place in ("response_curves", "response_curves/posterior"):
        assert loaded[place].attrs["random_seed"] == 12


@pytest.mark.usefixtures("files")
def test_save_refuses_a_malformed_seed(seeded, tmp_path):
    seeded.attrs["random_seed"] = "not-an-integer"
    with pytest.raises(ValueError, match=r"^/: .*'random_seed'"):
        save_idata(seeded, tmp_path / "bad.nc")
    assert not (tmp_path / "bad.nc").exists()
    seeded.attrs["random_seed"] = 11
    seeded["response_curves"].attrs["random_seed"] = -1  # the probe: curve posterior says 12
    with pytest.raises(ValueError, match=r"^/response_curves: .*'random_seed'"):
        save_idata(seeded, tmp_path / "bad.nc")
    assert not (tmp_path / "bad.nc").exists()


@pytest.mark.parametrize(
    ("group", "value", "message"),
    [
        ("/", "not-an-integer", r"^/: .*'random_seed'"),
        ("/response_curves", -1, r"^/response_curves: .*'random_seed'"),
        ("/response_curves/posterior", 13, r"^/response_curves/posterior: .*random_seed"),
    ],
)
@pytest.mark.usefixtures("files")
def test_load_refuses_a_file_that_holds_a_malformed_seed_and_closes_it(
    seeded, tmp_path, group, value, message
):
    import h5py

    path = tmp_path / "seeded.nc"
    save_idata(seeded, path)  # a valid file; the seed is changed behind save's check
    with h5py.File(path, "r+") as file:
        file[group].attrs["random_seed"] = value
    with pytest.raises(ValueError, match=message):
        load_idata(path)
    with h5py.File(path, "r+"):  # fails while the loader still holds the file
        pass


@pytest.mark.usefixtures("files")
def test_a_seeded_file_validates_with_torch_blocked(seeded, tmp_path):
    save_idata(seeded, tmp_path / "seeded.nc")
    # The ends of the seed range; the curve group and its posterior must agree.
    seeded.attrs["random_seed"] = 0
    for place in ("response_curves", "response_curves/posterior"):
        seeded[place].attrs["random_seed"] = _MAX_SEED
    save_idata(seeded, tmp_path / "ends.nc")
    run = _run(
        """
        import sys

        sys.modules["torch"] = None
        from pymc_mktfm import load_idata, validate_idata

        for path, root, curves in ((sys.argv[1], 11, 12), (sys.argv[2], 0, 2**31 - 1)):
            tree = load_idata(path)
            validate_idata(tree)
            assert tree.attrs["random_seed"] == root
            for place in ("response_curves", "response_curves/posterior"):
                assert tree[place].attrs["random_seed"] == curves
        """,
        str(tmp_path / "seeded.nc"),
        str(tmp_path / "ends.nc"),
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.usefixtures("files")
def test_save_never_overwrites_and_leaves_no_temporary_file(tree, tmp_path):
    path = tmp_path / "tree.nc"
    path.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        save_idata(tree, path)
    assert path.read_bytes() == b"keep me"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("hard_links", [True, False])
@pytest.mark.usefixtures("files")
def test_competing_saves_leave_only_the_winners_file(tree, tmp_path, monkeypatch, hard_links):
    target = tmp_path / "tree.nc"
    ready = Barrier(2, timeout=30)
    link = os.link

    def publish_together(source, destination):
        ready.wait()
        if not hard_links:
            raise OSError(errno.EPERM, "hard links are not supported")
        link(source, destination)

    def save(candidate):
        try:
            save_idata(candidate, target)
        except FileExistsError:
            return None
        return candidate.attrs["writer"]

    candidates = [tree.copy(), tree.copy()]
    for index, candidate in enumerate(candidates):
        candidate.attrs["writer"] = index
    monkeypatch.setattr(os, "link", publish_together)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(save, candidates))
    [winner] = [result for result in results if result is not None]
    assert load_idata(target).attrs["writer"] == winner
    assert list(tmp_path.iterdir()) == [target]


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


class _FailsAtClose:
    """The exclusive-create file, which writes its bytes and then fails to close."""

    def __init__(self, real):
        self._real = real

    def write(self, data):
        return self._real.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._real.close()
        raise OSError(errno.ENOSPC, "disk full at close")


@pytest.mark.usefixtures("files")
def test_a_failure_at_close_of_the_copy_leaves_no_file_and_a_retry_works(
    tree, tmp_path, monkeypatch
):
    def no_links(source, target):
        raise OSError(errno.EPERM, "hard links are not supported")

    monkeypatch.setattr(os, "link", no_links)
    real_open = Path.open

    def fails_at_close(self, mode="r", *args, **kwargs):
        file = real_open(self, mode, *args, **kwargs)
        return _FailsAtClose(file) if mode == "xb" else file

    path = tmp_path / "tree.nc"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fails_at_close)
        with pytest.raises(OSError, match="disk full at close"):
            save_idata(tree, path)
    assert list(tmp_path.iterdir()) == []
    save_idata(tree, path)
    validate_idata(load_idata(path))


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


@pytest.mark.parametrize("extra", ["graph", "media"])
@pytest.mark.usefixtures("files")
def test_a_group_named_like_a_variable_does_not_keep_the_file_open(tree, tmp_path, extra):
    import h5py

    tree[extra] = xr.DataTree(xr.Dataset({"x": ("i", [1.0, 2.0])}))
    path = tmp_path / "tree.nc"
    save_idata(tree, path)
    loaded = load_idata(path)
    for node in loaded.subtree:
        for name, variable in node.to_dataset(inherit=False).data_vars.items():
            assert isinstance(variable.variable._data, np.ndarray), f"{node.path}/{name} is lazy"
    with h5py.File(path, "r+"):
        pass
    path.unlink()
    np.testing.assert_array_equal(loaded[f"{extra}/x"].to_numpy(), [1, 2])
    for node in tree.subtree:
        for name, variable in node.to_dataset(inherit=False).data_vars.items():
            np.testing.assert_array_equal(
                loaded[node.path].to_dataset(inherit=False)[name].to_numpy(), variable.to_numpy()
            )


@pytest.mark.usefixtures("files")
def test_numpy_text_names_and_units_save_as_text(model, chart_frames, tmp_path):
    history, future = chart_frames
    roles = {
        "date_column": np.str_("date"),
        "target_column": np.str_("sales"),
        "channel_columns": (np.str_("search"), np.str_("tv")),
        "control_columns": (np.str_("temperature"),),
    }
    model.inference(history, future, n_samples=2, units=np.str_("USD"), **roles)
    save_idata(model.idata, tmp_path / "tree.nc")
    loaded = load_idata(tmp_path / "tree.nc")
    assert loaded.attrs["date_column"] == "date"
    assert json.loads(loaded.attrs["channel_columns"]) == ["search", "tv"]
    assert loaded["query/posterior"].attrs["units"] == "USD"


@pytest.mark.usefixtures("files")
def test_numpy_text_identity_saves_as_text(model, chart_frames, tmp_path):
    history, future = chart_frames
    identity = {"state_sha256": "a" * 64, "data_sha256": "b" * 64, "graph_source": "joint_router"}
    model.inference(
        history,
        future,
        n_samples=2,
        provenance={key: np.str_(text) for key, text in identity.items()},
        **CHART_ROLES,
    )
    validate_idata(model.idata)
    save_idata(model.idata, tmp_path / "tree.nc")
    for window in ("context", "query"):
        attrs = model.idata[f"{window}/posterior"].attrs
        assert {key: type(attrs[key]) for key in identity} == dict.fromkeys(identity, str)
        assert attrs["state_sha256"] == identity["state_sha256"]


@pytest.mark.parametrize("key", ["state_sha256", "data_sha256", "graph_source"])
def test_a_nul_in_a_recorded_identity_is_refused_before_sampling(
    predicted_model, chart_frames, key
):
    history, future = chart_frames
    before = predicted_model.idata
    with pytest.raises(ValueError, match=f"provenance {key} must not contain NUL"):
        predicted_model.inference(history, future, provenance={key: "ab\x00c"}, **CHART_ROLES)
    assert predicted_model.idata is before


def _set_coords(node, **coords):
    node.dataset = node.to_dataset(inherit=False).assign_coords(**coords)


@pytest.mark.parametrize(
    ("group", "change", "message"),
    [
        (
            "/",
            lambda node: node.attrs.update(date_column=np.str_("date")),
            r"^/: attribute 'date_column' must be a built-in str, got str_",
        ),
        (
            "query",
            lambda node: node.attrs.update(note="a\x00b"),
            r"^/query: attribute 'note' must not contain a NUL",
        ),
        (
            "query/constant_data",
            lambda node: _set_coords(node, channel=np.array(["sea\x00rch", "tv"])),
            r"^/query/constant_data/channel: labels must not contain a NUL",
        ),
        (
            "query/constant_data",
            lambda node: _set_coords(node, edge=np.array([1] * node.sizes["edge"], dtype=object)),
            r"^/query/constant_data/edge: labels must be str",
        ),
    ],
)
def test_validate_rejects_text_that_hdf5_cannot_store(tree, group, change, message):
    change(tree if group == "/" else tree[group])
    with pytest.raises(ValueError, match=message):
        validate_idata(tree)


def _reverse(tree, group, **dims):
    """Store ``group`` in another order along ``dims``, coordinates and values together."""
    _edit(
        tree,
        group,
        lambda data: data.isel({dim: slice(None, None, -1) for dim in dims if dim in data.sizes}),
    )


def _chart_numbers(tree):
    """What the charts draw, keyed by label, so stored order cannot change it."""
    reader = IdataReader(tree)
    numbers = {}
    for name in ("context", "query"):
        window = reader.window(name)
        numbers[name] = {
            "dates": window.dates,
            "outcome": window.outcome,
            "draws": {c: window.contribution(c) for c in window.components},
            "media": dict(zip(window.channels, window.media.T, strict=True)),
        }
    curves = reader.response_curves()
    numbers["curves"] = {
        "draws": dict(zip(curves.channels, curves.draws, strict=True)),
        "media": dict(zip(curves.channels, curves.media, strict=True)),
        "levels": curves.levels,
    }
    covariates = reader.covariates()
    numbers["covariates"] = {
        "media": dict(zip(covariates.channels, covariates.media.T, strict=True)),
        "outcome": covariates.outcome,
    }
    return numbers


def _same_numbers(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, dict):
            _same_numbers(value, right[key])
        else:
            np.testing.assert_array_equal(value, right[key], err_msg=key)


@pytest.mark.usefixtures("files")
def test_a_tree_stored_in_another_label_order_is_valid_and_charts_the_same(tree, tmp_path):
    expected = _chart_numbers(tree)
    for window in ("context", "query"):
        _reverse(tree, f"{window}/constant_data", component=1, channel=1, control=1, date=1)
        _reverse(tree, f"{window}/posterior", channel=1)
    _reverse(tree, "context/observed_data", date=1)
    _reverse(tree, "response_curves/constant_data", channel=1, level=1, date=1)
    _reverse(tree, "response_curves/posterior", channel=1)
    validate_idata(tree)
    _same_numbers(_chart_numbers(tree), expected)
    save_idata(tree, tmp_path / "tree.nc")
    _same_numbers(_chart_numbers(load_idata(tmp_path / "tree.nc")), expected)


def test_validate_gives_the_reader_message_for_an_unknown_family(tree):
    def rename(data):
        families = np.where(data["component_family"] == "latent", "bogus", data["component_family"])
        return data.assign(component_family=("component", families))

    _edit(tree, "query/constant_data", rename)
    with pytest.raises(
        ValueError, match=r"^/query/constant_data: unknown component family 'bogus'"
    ):
        validate_idata(tree)


def test_validate_gives_the_reader_message_for_a_label_the_draws_lack(tree):
    stored = tree["query/constant_data/component"].to_numpy()
    labels = [str(label).replace("channel:tv", "channel:radio") for label in stored]
    _edit(tree, "query/constant_data", lambda data: data.assign_coords(component=labels))
    with pytest.raises(
        ValueError, match=r"^/query/posterior: channel labels .* do not hold \['radio'\]"
    ):
        validate_idata(tree)


def test_validate_refuses_a_window_that_lacks_a_channel_component(tree):
    def drop(data):
        keep = [str(label) != "channel:tv" for label in data["component"].to_numpy()]
        return data.isel(component=keep)

    _edit(tree, "query/constant_data", drop)
    with pytest.raises(
        ValueError, match=r"^/query/constant_data: channel descriptors .* missing \['tv'\]"
    ):
        validate_idata(tree)


def _all_groups(tree, window, change):
    for group in ("posterior", "constant_data", "observed_data"):
        _edit(tree, f"{window}/{group}", change)


def _as_text(name):
    return lambda data: data.assign({name: data[name].astype(str)})


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda tree: tree["context"].__delitem__("observed_data"),
            r"^/context/observed_data: group is missing",
        ),
        (
            lambda tree: _edit(tree, "context/observed_data", _as_text("outcome")),
            r"^/context/observed_data/outcome: must hold numbers",
        ),
        (
            lambda tree: _edit(tree, "context/posterior", _as_text("baseline")),
            r"^/context/posterior/baseline: must hold numbers",
        ),
        (
            lambda tree: _edit(tree, "context/constant_data", _as_text("media")),
            r"^/context/constant_data/media: must hold numbers",
        ),
        (
            lambda tree: _all_groups(tree, "context", lambda data: data.isel(date=slice(0, 1))),
            r"^/context/posterior: date coordinate needs at least 2 dates, got 1",
        ),
        (
            lambda tree: _all_groups(
                tree, "context", lambda data: data.isel(date=slice(None, None, -1))
            ),
            r"^/context/posterior: date coordinate must be strictly increasing",
        ),
    ],
)
def test_validate_requires_a_complete_numeric_history(tree, edit, message):
    edit(tree)
    with pytest.raises(ValueError, match=message):
        validate_idata(tree)


@pytest.mark.usefixtures("files")
def test_a_query_without_observations_or_with_gaps_is_valid_and_saves(tree, tmp_path):
    first = tree.copy(deep=True)
    del first["query"]["observed_data"]
    validate_idata(first)
    save_idata(first, tmp_path / "none.nc")
    loaded = load_idata(tmp_path / "none.nc")
    assert IdataReader(loaded).window("query").outcome is None
    assert IdataReader(loaded).window("context").outcome is not None
    gaps = tree.copy(deep=True)
    _edit(
        gaps,
        "query/observed_data",
        lambda data: data.assign(outcome=data["outcome"].where(data["date"] != data["date"][0])),
    )
    save_idata(gaps, tmp_path / "gaps.nc")
    outcome = IdataReader(load_idata(tmp_path / "gaps.nc")).window("query").outcome
    assert np.isnan(outcome[0]) and np.isfinite(outcome[1:]).all()


def _second_chain(data):
    return xr.concat([data, data.assign_coords(chain=[1])], "chain", data_vars="minimal")


@pytest.mark.parametrize("group", ["context", "query", "response_curves"])
def test_validate_rejects_a_second_chain_naming_the_group(tree, group):
    _edit(tree, f"{group}/posterior", _second_chain)
    with pytest.raises(
        ValueError, match=rf"^/{group}/posterior: charts read a tree with one chain, got 2"
    ):
        validate_idata(tree)


def test_validate_rejects_curves_of_another_run_with_the_same_revision(tree):
    tree["query"].attrs["provenance"] = json.dumps({"other": "run"})
    with pytest.raises(ValueError, match=r"^/response_curves: source_provenance differs"):
        validate_idata(tree)


def _curve_levels(levels):
    def change(tree):
        for group in ("posterior", "constant_data"):
            _edit(
                tree,
                f"response_curves/{group}",
                lambda data: data.isel(level=slice(0, len(levels))).assign_coords(level=levels),
            )

    return change


def _curve_channels(tree):
    for group in ("posterior", "constant_data"):
        _edit(
            tree,
            f"response_curves/{group}",
            lambda data: data.assign_coords(channel=["search", "radio"]),
        )


def _curve_no_channels(tree):
    for group in ("posterior", "constant_data"):
        _edit(tree, f"response_curves/{group}", lambda data: data.isel(channel=slice(0, 0)))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda tree: tree["response_curves"].attrs.update(weeks=8),
            r"^/response_curves/constant_data: date coordinate must hold the first 8 dates",
        ),
        (
            lambda tree: tree["response_curves"].attrs.update(n_samples=999),
            r"^/response_curves/posterior: attribute n_samples 999",
        ),
        (
            lambda tree: tree["response_curves"].attrs.update(units=""),
            r"^/response_curves: attribute 'units' must be nonempty",
        ),
        (
            lambda tree: tree["response_curves"].attrs.update(weeks=0),
            r"^/response_curves: attribute 'weeks' must be positive",
        ),
        (_curve_channels, r"^/response_curves/posterior: channel coordinate"),
        (_curve_no_channels, r"^/response_curves/posterior: channel coordinate must hold at least"),
        (
            lambda tree: _edit(
                tree,
                "response_curves/constant_data",
                lambda data: data.assign_coords(date=data["date"] + np.timedelta64(1, "D")),
            ),
            r"^/response_curves/constant_data: date coordinate must hold the first 2 dates",
        ),
        (
            _curve_levels(np.array(["a", "b", "c"])),
            r"^/response_curves/posterior: level coordinate must hold numbers",
        ),
        (
            _curve_levels(np.array([-1.0, 0.0, 1.0])),
            r"^/response_curves/posterior: level coordinate must hold finite values of at least 0",
        ),
        (
            _curve_levels(np.array([0.0, np.nan, 1.0])),
            r"^/response_curves/posterior: level coordinate must hold finite",
        ),
        (
            _curve_levels(np.array([0.0, 1.0, 1.0])),
            r"^/response_curves/posterior: level coordinate needs at least 2 values, strictly",
        ),
        (
            _curve_levels(np.array([2, 1], dtype=np.uint64)),
            r"^/response_curves/posterior: level coordinate needs at least 2 values, strictly",
        ),
        (
            _curve_levels(np.array([1.0])),
            r"^/response_curves/posterior: level coordinate needs at least 2 values",
        ),
    ],
)
def test_validate_rejects_curves_that_do_not_match_their_axes_or_the_query(tree, edit, message):
    edit(tree)
    with pytest.raises(ValueError, match=message):
        validate_idata(tree)


@pytest.mark.parametrize("keep", [[1], [1, 0]])
@pytest.mark.usefixtures("files")
def test_curves_of_a_reordered_channel_subset_are_valid(tree, tmp_path, keep):
    for group in ("posterior", "constant_data"):
        _edit(tree, f"response_curves/{group}", lambda data: data.isel(channel=keep))
    validate_idata(tree)
    save_idata(tree, tmp_path / "tree.nc")
    curves = IdataReader(load_idata(tmp_path / "tree.nc")).response_curves()
    assert curves.channels == tuple(["search", "tv"][i] for i in keep)


_PROVENANCE = r"^/query: attribute 'provenance' must be JSON text holding an object"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda node: node.attrs.__delitem__("provenance"), r"^/query: attribute 'provenance' is"),
        (lambda node: node.attrs.update(provenance="not json"), _PROVENANCE),
        (lambda node: node.attrs.update(provenance="[1, 2]"), _PROVENANCE),
    ],
)
def test_validate_requires_provenance_that_is_a_json_object(tree, change, message):
    change(tree["query"])
    with pytest.raises(ValueError, match=message):
        validate_idata(tree)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.assign(graph=d["graph"].astype(str)), r"^/query/posterior/graph: must hold"),
        (lambda d: d.assign(graph=d["graph"] * 2), r"^/query/posterior/graph: must be binary"),
        (
            lambda d: d.assign(graph=("date", np.zeros(d.sizes["date"]))),
            r"^/query/posterior/graph: expected dims",
        ),
    ],
)
def test_validate_rejects_a_graph_that_is_not_binary_on_chain_draw_edge(tree, change, message):
    _edit(tree, "query/posterior", change)
    with pytest.raises(ValueError, match=message):
        validate_idata(tree)


@pytest.mark.usefixtures("files")
def test_a_tree_without_graph_draws_is_valid_and_saves(tree, tmp_path):
    for window in ("context", "query"):
        _edit(tree, f"{window}/posterior", lambda data: data.drop_vars("graph"))
    save_idata(tree, tmp_path / "tree.nc")
    validate_idata(load_idata(tmp_path / "tree.nc"))


def test_validate_does_not_stack_the_draws(tree, monkeypatch):
    def stacked(*args, **kwargs):
        raise AssertionError("validate_idata stacked draws")

    monkeypatch.setattr(IdataReader, "window", stacked)
    monkeypatch.setattr(np, "stack", stacked)
    validate_idata(tree)


class _Unreadable(BackendArray):
    """Lazy file-style array that raises when anything reads its values."""

    def __init__(self, like):
        self.shape, self.dtype = like.shape, like.dtype

    def __getitem__(self, key):
        raise AssertionError("validate_idata read the draws")


def test_validate_does_not_read_numeric_draws(tree):
    from xarray.core.indexing import LazilyIndexedArray

    for group in ("context/posterior", "query/posterior"):  # the curve reader loads its own
        node = tree[group]
        data = node.to_dataset(inherit=False)
        lazy = {
            name: (var.dims, LazilyIndexedArray(_Unreadable(var.variable)))
            for name, var in data.data_vars.items()
            if var.dtype.kind in "fi" and name != "graph"  # the binary check reads graph on purpose
        }
        node.dataset = data.assign(lazy)
    validate_idata(tree)


def test_the_names_are_top_level_exports():
    assert {"save_idata", "load_idata", "validate_idata"} <= set(pymc_mktfm.__all__)


def _run(script, *args):
    src = str(Path(pymc_mktfm.__file__).parents[1])  # the code under test, not another install
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        env={"MPLBACKEND": "Agg", "PATH": "", "PYTHONPATH": src},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.usefixtures("files")
def test_a_file_loads_with_torch_and_matplotlib_blocked(tree, tmp_path):
    save_idata(tree, tmp_path / "tree.nc")
    run = _run(
        """
        import sys

        for name in ("torch", "matplotlib"):
            sys.modules[name] = None  # any import now raises ImportError
        from pymc_mktfm import load_idata, validate_idata

        tree = load_idata(sys.argv[1])
        validate_idata(tree)
        assert tree["query/posterior/channel_contribution"].sizes["date"] == 6
        """,
        str(tmp_path / "tree.nc"),
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.usefixtures("files", "matplotlib_backend")
def test_a_file_draws_a_chart_with_torch_blocked(tree, tmp_path):
    save_idata(tree, tmp_path / "tree.nc")
    run = _run(
        """
        import sys

        sys.modules["torch"] = None
        from pymc_mktfm import Plots, load_idata

        Plots(load_idata(sys.argv[1])).fit()
        """,
        str(tmp_path / "tree.nc"),
    )
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("blocked", ["h5netcdf", "h5py"])
def test_without_a_file_backend_the_error_names_the_draws_extra(tree, tmp_path, blocked):
    with (tmp_path / "tree.pickle").open("wb") as out:
        pickle.dump(tree, out)
    run = _run(
        """
        import pickle
        import sys

        sys.modules[sys.argv[1]] = None
        from pymc_mktfm import load_idata, save_idata, validate_idata

        tree = pickle.load(open(sys.argv[2], "rb"))
        validate_idata(tree)  # memory only: needs no file backend
        for call in (lambda: load_idata(sys.argv[2]), lambda: save_idata(tree, sys.argv[2] + "x")):
            try:
                call()
            except ImportError as error:
                assert 'pip install "pymc-mktfm[draws]"' in str(error), error
            else:
                raise SystemExit("a file call ran without " + sys.argv[1])
        """,
        blocked,
        str(tmp_path / "tree.pickle"),
    )
    assert run.returncode == 0, run.stderr
