"""Repeated ``analyze`` calls start one analysis, against a local stub with a held worker."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import pymc_mktfm
from pymc_mktfm.client import Client, Gone, Preview

from . import local_service

SCRIPT = Path(__file__).with_name("analyze_once.py")
WEEKS = {"context": ("2024-07-01", "2024-07-15"), "query_end": "2024-08-05"}


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    with local_service.serve() as (base_url, state):
        yield base_url, state


@pytest.fixture
def local_client(stub):
    with Client(stub[0], local_service.API_KEY, max_retries=0) as client:
        yield client


def run_script(base_url: str, *args: str) -> subprocess.Popen[str]:
    source = str(Path(pymc_mktfm.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": source, "NO_PROXY": "127.0.0.1"}
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), base_url, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def analysis_ids(client: Client) -> list[str]:
    return list(client.list_analyses()["analysis_id"])


def test_three_processes_with_equal_inputs_store_one_analysis(stub, local_client):
    base_url, _ = stub
    processes = [run_script(base_url) for _ in range(3)]
    outputs = [process.communicate(timeout=60) for process in processes]
    assert [process.returncode for process in processes] == [0, 0, 0], outputs
    printed = {out.splitlines()[0] for out, _ in outputs}
    assert len(printed) == 1
    assert analysis_ids(local_client) == list(printed)


def test_a_changed_seed_in_another_process_makes_a_new_analysis(stub, local_client):
    base_url, _ = stub
    processes = [run_script(base_url, "7"), run_script(base_url, "8")]
    for process in processes:
        out, err = process.communicate(timeout=60)
        assert process.returncode == 0, err
    assert len(analysis_ids(local_client)) == 2


def test_fresh_and_a_changed_seed_each_make_a_new_analysis(local_client):
    first = local_client.analyze("ds_1", seed=7, **WEEKS)
    assert local_client.analyze("ds_1", seed=7, **WEEKS).id == first.id
    assert local_client.analyze("ds_1", seed=7, fresh=True, **WEEKS).id != first.id
    assert local_client.analyze("ds_1", seed=8, **WEEKS).id != first.id
    assert len(analysis_ids(local_client)) == 3


def test_delete_then_rerun_makes_a_new_analysis_with_fresh(local_client):
    first = local_client.analyze("ds_1", seed=7, **WEEKS)
    first.delete()
    with pytest.raises(Gone, match="fresh=True"):
        local_client.analyze("ds_1", seed=7, **WEEKS)
    second = local_client.analyze("ds_1", seed=7, fresh=True, **WEEKS)
    assert second.id != first.id
    assert analysis_ids(local_client) == [second.id]


def test_preview_makes_no_analysis(local_client):
    for _ in range(2):
        assert isinstance(local_client.analyze("ds_1", seed=7, preview=True, **WEEKS), Preview)
    assert analysis_ids(local_client) == []
