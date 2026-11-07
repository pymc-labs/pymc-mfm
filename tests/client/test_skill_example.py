"""The main example in skills/mfm/SKILL.md runs against a fake service.

A client change that breaks the skill's example fails here. The example reads
history.csv and plan.csv from the working directory and writes two PNG files.
"""

import re
import shutil
from pathlib import Path

import httpx
import pandas as pd
import pytest

from .client_fixtures import (
    BASE_URL,
    FakeService,
    curve_result,
    dataset_record,
    result_record,
    result_rows,
    started,
    status_record,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "mfm"
SKILL = SKILL_DIR / "SKILL.md"
EXAMPLE = r"<!-- skill-example: {name} -->\n```python\n(.*?)```"
ALLOWED_URLS = ("https://github.com/pymc-labs/pymc-mfm.git",)


def skill_example(name: str = "main") -> str:
    match = re.search(EXAMPLE.format(name=name), SKILL.read_text(), re.DOTALL)
    assert match, f"SKILL.md must mark the example with '<!-- skill-example: {name} -->'"
    return match.group(1)


def compare_rows() -> list[dict]:
    rows = result_rows("baseline", "baseline")
    for scenario_id in ("reduce_20", "tv_to_search", "tv_front_loaded"):
        rows += result_rows(scenario_id, "scenario", base="baseline")
    return rows


def analysis_id(number: int) -> dict:
    return {**started(), "analysis_id": f"an_{number}"}


@pytest.fixture
def service() -> FakeService:
    """The routes the example calls, in order: forecast, preview, compare, curves."""
    fake = FakeService()
    fake.on("POST", "/v1/datasets", (201, dataset_record(controls=("holiday",))))
    plans = [{"scenario_id": "baseline", "spend": {"tv": [100.0, 100.0], "search": [50.0, 50.0]}}]
    preview = {"plans": plans, "runs_total": 8, "estimate_seconds": 40.0, "warnings": []}
    fake.on(
        "POST",
        "/v1/analyses",
        (202, analysis_id(1)),
        (200, preview),
        (202, analysis_id(2)),
        (202, analysis_id(3)),
    )
    for number in (1, 2, 3):
        fake.on("GET", f"/v1/analyses/an_{number}", (200, status_record("done")))
    baseline = result_record(result_rows("baseline", "baseline"))
    fake.on("GET", "/v1/analyses/an_1/result", (200, baseline))
    fake.on("GET", "/v1/analyses/an_2/result", (200, result_record(compare_rows())))
    fake.on("GET", "/v1/analyses/an_3/result", (200, curve_result()))
    weekly = [{"date": "2024-07-29", "component": "outcome", "mean": 1.0, "p05": 0.5, "p95": 1.5}]
    weekly_reply = {"scenario_id": "tv_to_search", "rows": weekly}
    fake.on("GET", "/v1/analyses/an_2/weekly", (200, weekly_reply))
    fake.on("DELETE", "/v1/datasets/ds_123", (202, {"deletion": "in_progress"}))
    return fake


@pytest.fixture
def workdir(tmp_path, monkeypatch, service):
    """A work directory with the skill's example CSVs, a fake network and the two variables."""
    for name in ("history.csv", "plan.csv"):
        shutil.copy(SKILL_DIR / "examples" / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MFM_API_URL", BASE_URL)
    monkeypatch.setenv("MFM_API_KEY", "test-key")
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(service)}),
    )
    return tmp_path


def test_skill_example_runs_and_follows_the_planned_calls(workdir, service):
    namespace = {"__name__": "skill_example"}
    exec(compile(skill_example(), str(SKILL), "exec"), namespace)
    # The main example keeps the dataset, so the user can open the analysis on the site.
    assert not service.sent("DELETE", "/v1/datasets/ds_123")
    exec(compile(skill_example("delete"), str(SKILL), "exec"), namespace)

    assert (workdir / "compare.png").stat().st_size > 0
    assert (workdir / "curves.png").stat().st_size > 0
    posts = [service.body(r) for r in service.sent("POST", "/v1/analyses")]
    assert [bool(body.get("preview")) for body in posts] == [False, True, False, False]
    assert [s["id"] for s in posts[2]["scenarios"]][:3] == [
        "reduce_20",
        "tv_to_search",
        "tv_front_loaded",
    ]
    assert [c["channel"] for c in posts[3]["curves"]] == ["tv", "search"]
    assert service.sent("DELETE", "/v1/datasets/ds_123")
    assert all(r.headers["Authorization"] == "Bearer test-key" for r in service.requests)


def test_skill_files_name_no_service_address():
    text = "".join(path.read_text() for path in SKILL_DIR.rglob("*.md"))
    urls = set(re.findall(r"https?://[^\s)\"'`>]+", text))
    assert urls <= set(ALLOWED_URLS)
    assert "MFM_API_URL" in text and "MFM_API_KEY" in text


def test_example_data_fits_the_example():
    history = pd.read_csv(SKILL_DIR / "examples" / "history.csv", parse_dates=["date"])
    plan = pd.read_csv(SKILL_DIR / "examples" / "plan.csv", parse_dates=["date"])
    assert list(history.columns) == ["date", "sales", "tv", "search", "holiday"]
    assert list(plan.columns) == ["date", "tv", "search", "holiday"]
    assert len(history) == 104 and len(plan) == 8
    weeks = pd.concat([history["date"], plan["date"]])
    assert (weeks.diff().dropna() == pd.Timedelta(days=7)).all()
