"""The first example on the PyPI page of the client package runs against a fake service.

A client change that breaks the example on ``packaging/client/README.template.md`` fails here. The
example makes its own data and writes first_analysis.csv to the working directory.
"""

import re
from pathlib import Path

import httpx
import pandas as pd
import pytest

from .client_fixtures import (
    BASE_URL,
    FakeService,
    dataset_record,
    result_record,
    result_rows,
    started,
    status_record,
)

TEMPLATE = Path(__file__).resolve().parents[2] / "packaging" / "client" / "README.template.md"

pytestmark = pytest.mark.skipif(not TEMPLATE.is_file(), reason="needs a Git checkout")


def test_the_page_example_runs_as_pasted(tmp_path, monkeypatch, capsys):
    page = TEMPLATE.read_text().split("## First example")[1]
    code = re.search(r"```python\n(.*?)```", page, re.DOTALL).group(1)
    record = dataset_record(controls=("price",))
    record["summary"] = {
        **record["summary"],
        "weeks": 104,
        "start": "2024-11-11",
        "end": "2026-11-02",
    }
    rows = result_rows("baseline", "baseline")
    for scenario_id in ("half", "tv_to_search"):
        rows += result_rows(scenario_id, "scenario", base="baseline")
    service = FakeService()
    service.on("GET", "/v1/datasets", (200, {"datasets": []}))
    service.on("POST", "/v1/datasets", (201, record))
    service.on("POST", "/v1/analyses", (202, started()))
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/result", (200, result_record(rows)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MFM_API_URL", BASE_URL)
    monkeypatch.setenv("MFM_API_KEY", "test-key")
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(service)}),
    )
    exec(compile(code, str(TEMPLATE), "exec"), {"__name__": "page_example"})
    assert "tv_to_search" in capsys.readouterr().out
    assert len(pd.read_csv(tmp_path / "first_analysis.csv")) == 3
    sent = [service.body(r) for r in service.sent("POST", "/v1/analyses")]
    assert [s["id"] for s in sent[-1]["scenarios"]] == ["half", "tv_to_search"]
