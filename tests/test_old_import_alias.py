"""The old import path gives the same objects as the new one."""

import importlib
import subprocess
import sys

import pytest

import pymc_mfm
import pymc_mktfm


def test_top_level_name_is_the_same_module():
    assert pymc_mfm is pymc_mktfm
    assert sys.modules["pymc_mfm"] is pymc_mktfm


@pytest.mark.parametrize("name", ["client", "client._http", "client.errors"])
def test_submodules_are_the_same_objects(name):
    old = importlib.import_module(f"pymc_mfm.{name}")
    new = importlib.import_module(f"pymc_mktfm.{name}")
    assert old is new
    assert sys.modules[f"pymc_mfm.{name}"] is sys.modules[f"pymc_mktfm.{name}"]
    assert old.__name__ == f"pymc_mktfm.{name}"
    assert old.__spec__.name == f"pymc_mktfm.{name}"


def test_from_imports_give_identical_classes():
    old_client = importlib.import_module("pymc_mfm.client")
    old_errors = importlib.import_module("pymc_mfm.client.errors")
    from pymc_mktfm.client import Client
    from pymc_mktfm.client.errors import NotFound

    assert old_client.Client is Client
    assert old_errors.NotFound is NotFound
    namespace: dict[str, object] = {}
    exec("from pymc_mfm.client import Client as OldClient", namespace)
    assert namespace["OldClient"] is Client


@pytest.mark.parametrize("first", ["pymc_mfm", "pymc_mktfm"])
def test_fresh_interpreter_import_order_reload_and_pickle(first):
    """Either name may be imported first; reload keeps identity; old-path pickles load."""
    code = f"""
import importlib, pickle, sys
import {first}.client.errors
import pymc_mfm.client.errors as old
import pymc_mktfm.client.errors as new
assert old is new
module = importlib.reload(sys.modules["pymc_mfm.client.errors"])
assert sys.modules["pymc_mfm.client.errors"] is sys.modules["pymc_mktfm.client.errors"] is module
data = b"cpymc_mfm.client.errors\\nNotFound\\n."
assert pickle.loads(data) is module.NotFound
"""
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr


def test_unknown_old_submodule_is_not_found():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pymc_mfm.no_such_module")
