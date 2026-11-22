"""The Python examples of the guide and the README run against a fake service.

A tester copies them with only the install line and the two environment variables, so each
must run as pasted: the Get started script (docs/guide/index.qmd, marked blocks), every block of
docs/guide/http-api.qmd in page order, and the client example of README.md.
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
    export_record,
    preview_record,
    result_record,
    result_rows,
    started,
    status_record,
    weeks_result,
    weeks_status,
)

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide" / "index.qmd"
CLIENT_PAGE = ROOT / "docs" / "guide" / "http-api.qmd"
README = ROOT / "README.md"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
URL = "https://site.example/analyses/an_1"
# The synthetic history of the examples: 104 weeks from 2024-11-11, so it ends on 2026-11-02.
HISTORY_START, HISTORY_END, PLAN_START = "2024-11-11", "2026-11-02", "2026-11-09"


def guide_example(name: str) -> str:
    pattern = rf"<!-- guide-example: {name} -->\n```python\n(.*?)```"
    match = re.search(pattern, GUIDE.read_text(), re.DOTALL)
    assert match, f"index.qmd must mark the example with '<!-- guide-example: {name} -->'"
    return match.group(1)


def section(path: Path, heading: str) -> str:
    """The text of one ``## `` section of a Markdown file, up to the next one."""
    text = path.read_text()
    start = text.index(f"\n{heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else None]


def history_record() -> dict:
    """The fake summary of the examples' history, so the client's plan check passes."""
    record = dataset_record(controls=("price",))
    record["summary"] = {**record["summary"], "weeks": 104}
    record["summary"]["start"], record["summary"]["end"] = HISTORY_START, HISTORY_END
    return record


def compare_rows(*scenario_ids: str) -> list[dict]:
    rows = result_rows("baseline", "baseline")
    for scenario_id in scenario_ids:
        rows += result_rows(scenario_id, "scenario", base="baseline")
    return rows


@pytest.fixture
def service() -> FakeService:
    """Analyses an_1 (the plan) and an_2 (the backtest), both done, then an_3 (again)."""
    fake = FakeService()
    fake.on("GET", "/v1/datasets", (200, {"datasets": []}))
    fake.on("POST", "/v1/datasets", (201, history_record()))
    fake.on(
        "POST",
        "/v1/analyses",
        *[(202, {**started(), "analysis_id": f"an_{n}", "url": URL}) for n in (1, 2, 3)],
        (200, preview_record()),
    )
    fake.on("GET", "/v1/analyses/an_1", (200, status_record("done", url=URL)))
    fake.on("GET", "/v1/analyses/an_1/result", (200, result_record(compare_rows("half"))))
    fake.on("GET", "/v1/analyses/an_1/export", (200, export_record(None)))
    fake.on("GET", "/v1/analyses/an_2", (200, weeks_status()))
    fake.on("GET", "/v1/analyses/an_2/result", (200, weeks_result()))
    weeks = ["2026-08-10", "2026-08-17"]
    inputs = {"dates": weeks, "spend": {}, "controls": {}, "decision_dates": weeks}
    fake.on("GET", "/v1/analyses/an_2/export", (200, export_record(inputs)))
    fake.on("GET", "/v1/analyses/an_3", (200, status_record("done")))
    return fake


@pytest.fixture
def workdir(tmp_path, monkeypatch, service):
    """An empty work directory, a fake network and the two variables, as a tester has."""
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


def run(code: str, source: Path) -> dict:
    namespace = {"__name__": "guide_example"}
    exec(compile(code, str(source), "exec"), namespace)
    return namespace


def analysis_posts(service: FakeService) -> list[dict]:
    return [service.body(r) for r in service.sent("POST", "/v1/analyses")]


def check_upload(service: FakeService) -> None:
    (upload,) = [service.body(r) for r in service.sent("POST", "/v1/datasets")]
    assert len(upload["dates"]) == 104
    assert (upload["dates"][0], upload["dates"][-1]) == (HISTORY_START, HISTORY_END)
    assert min(upload["outcome"]) > 0
    assert min(upload["spend"]["tv"] + upload["spend"]["search"]) > 0
    assert list(upload["controls"]) == ["price"]


def test_get_started_script_and_backtest_run_as_pasted(workdir, service, capsys):
    run(guide_example("quickstart") + guide_example("backtest"), GUIDE)

    check_upload(service)
    first, backtest = analysis_posts(service)
    assert first["plan"]["dates"][0] == PLAN_START and len(first["plan"]["dates"]) == 8
    assert [s["id"] for s in first["scenarios"]] == ["half", "tv_to_search"]
    assert first["scenarios"][1]["multipliers"] == {"tv": 0.8, "search": {"amount": 60_800}}
    assert (first["seed"], first["name"]) == (7, "first-scenarios")
    assert backtest["context"] == {"start": HISTORY_START, "end": "2026-08-03"}
    assert backtest["query"] == {"end": HISTORY_END}
    captured = capsys.readouterr()
    assert "scenario_id" in captured.out and "mean_difference" in captured.out
    assert f"result page: {URL}" in captured.out
    assert "r2" in captured.out and "crps" in captured.out  # the backtest's fit table
    assert "client.get('an_1').wait()" in captured.err  # the resume hint the text describes
    saved = pd.read_csv(workdir / "first_analysis.csv")
    assert list(saved["scenario_id"]) == ["baseline", "half"]


def test_own_data_block_replaces_step_1_of_the_script(workdir, service, capsys):
    script = guide_example("quickstart")
    synthetic = run(script.split("# 2.")[0], GUIDE)["history"]
    own_file = workdir / "path" / "to" / "your_history.csv"
    own_file.parent.mkdir(parents=True)
    synthetic.sample(frac=1, random_state=0).to_csv(own_file, index=False)  # unsorted on disk
    step_1 = re.search(r"# 1\..*?(?=# 2\.)", script, re.DOTALL).group(0)

    run(script.replace(step_1, guide_example("own-data") + "\n"), GUIDE)

    check_upload(service)
    (first,) = analysis_posts(service)
    assert first["plan"]["dates"][0] == PLAN_START
    assert f"result page: {URL}" in capsys.readouterr().out


def test_client_page_blocks_run_in_page_order(workdir, service):
    blocks = PYTHON_BLOCK.findall(CLIENT_PAGE.read_text())
    assert len(blocks) >= 5
    run("\n".join(blocks), CLIENT_PAGE)

    check_upload(service)
    plan, backtest, again, preview, explore = analysis_posts(service)
    assert len(plan["scenarios"]) == 6 and plan["decision_weeks"] == [1, 8]
    assert backtest["context"] == {"start": HISTORY_START, "end": "2026-08-03"}
    assert again["scenarios"] == [{"id": "search_x1.5", "multipliers": {"search": 1.5}}]
    assert preview["preview"] is True and explore["preview"] is True
    assert [c["channel"] for c in explore["curves"]] == ["tv", "search"]


def test_readme_client_example_runs_as_pasted(workdir, service):
    (block,) = PYTHON_BLOCK.findall(section(README, "## Call the River API"))
    run(block, README)

    check_upload(service)
    assert len(analysis_posts(service)) == 3  # the plan, the backtest and again
    assert (workdir / "analysis.json").stat().st_size > 0
