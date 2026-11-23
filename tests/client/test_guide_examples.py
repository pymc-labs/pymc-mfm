"""The Python examples of the guide and the README run against a fake service.

A tester copies them with only the install line and the two environment variables, so each
must run as pasted: the Get started script (docs/guide/index.qmd, marked blocks), every block of
docs/guide/python-client.qmd in page order, and the client example of README.md.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .client_fixtures import (
    FakeService,
    analysis_posts,
    dataset_record,
    export_record,
    fit_record,
    marked_example,
    preview_record,
    result_record,
    result_rows,
    route_examples_to,
    started,
    status_record,
    weeks_result,
    weeks_status,
)

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide" / "index.qmd"
CLIENT_PAGE = ROOT / "docs" / "guide" / "python-client.qmd"
README = ROOT / "README.md"
PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
URL = "https://site.example/analyses/an_1"
# The synthetic history of the examples: 104 weeks from 2024-11-11, so it ends on 2026-11-02.
HISTORY_START, HISTORY_END, PLAN_START = "2024-11-11", "2026-11-02", "2026-11-09"
PLAN_DATES = list(pd.date_range(PLAN_START, periods=8, freq="7D").strftime("%Y-%m-%d"))
SCENARIOS = ("half", "tv_to_search")
# The columns that step 4 of the Get started script prints and saves.
OUTCOME_COLUMNS = [
    "scenario_id",
    "status",
    "mean",
    "p05",
    "p95",
    "mean_difference",
    "simulation_se",
]


def guide_example(name: str) -> str:
    return marked_example(GUIDE, "guide-example", name)


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


def compare_rows() -> list[dict]:
    rows = result_rows("baseline", "baseline")
    for scenario_id in SCENARIOS:
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
    fake.on("GET", "/v1/analyses/an_1/result", (200, result_record(compare_rows())))
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
    route_examples_to(service, monkeypatch)
    return tmp_path


def run(code: str, source: Path) -> dict:
    namespace = {"__name__": "guide_example"}
    exec(compile(code, str(source), "exec"), namespace)
    return namespace


def check_upload(service: FakeService, history: pd.DataFrame) -> None:
    """The one upload holds ``history`` in date order, with its column roles."""
    (upload,) = [service.body(r) for r in service.sent("POST", "/v1/datasets")]
    assert upload["dates"] == [day.date().isoformat() for day in history["week"]]
    assert upload["outcome"] == pytest.approx(list(history["sales"]))
    assert upload["spend"]["tv"] == pytest.approx(list(history["tv"]))
    assert upload["spend"]["search"] == pytest.approx(list(history["search"]))
    assert upload["controls"] == {"price": pytest.approx(list(history["price"]))}


def synthetic_history() -> pd.DataFrame:
    """Step 1 of the Get started script, run on its own."""
    return run(guide_example("quickstart").split("# 2.")[0], GUIDE)["history"]


def test_get_started_script_and_backtest_run_as_pasted(workdir, service, capsys):
    run(guide_example("quickstart") + guide_example("backtest"), GUIDE)

    history = synthetic_history()
    assert (history["week"].iloc[0], history["week"].iloc[-1]) == (
        pd.Timestamp(HISTORY_START),
        pd.Timestamp(HISTORY_END),
    )
    check_upload(service, history)
    first, backtest = analysis_posts(service)
    assert first["plan"]["dates"] == PLAN_DATES
    assert first["plan"]["spend"] == {"tv": [8_000.0] * 8, "search": [6_000.0] * 8}
    assert first["plan"]["controls"] == {"price": [10.0] * 8}
    # The page's arithmetic: 20% of the tv budget (64,000) moves to search (48,000).
    assert first["scenarios"] == [
        {"id": "half", "multipliers": {"tv": 0.5, "search": 0.5}},
        {"id": "tv_to_search", "multipliers": {"tv": 0.8, "search": {"amount": 60_800}}},
    ]
    assert 0.8 * 8 * 8_000 + 60_800 == 8 * (8_000 + 6_000)
    assert (first["seed"], first["name"]) == (7, "first-scenarios")
    assert backtest["context"] == {"start": HISTORY_START, "end": "2026-08-03"}
    assert backtest["query"] == {"end": HISTORY_END}
    assert (backtest["seed"], backtest["name"]) == (7, "first-backtest")

    # Step 4 saves the outcome rows of the service's table: the baseline and each scenario.
    expected = pd.DataFrame(compare_rows())
    expected = expected[expected["component"] == "outcome"][OUTCOME_COLUMNS]
    saved = pd.read_csv(workdir / "first_analysis.csv")
    pd.testing.assert_frame_equal(saved, expected.reset_index(drop=True), check_dtype=False)
    assert list(saved["scenario_id"]) == ["baseline", *SCENARIOS]

    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[0] == "dataset ds_123 expires 2030-01-08 00:00:00+00:00"
    assert lines[1] == "dataset warning: history is shorter than 75 weeks"
    assert lines[2].split() == OUTCOME_COLUMNS
    assert [line.split()[0] for line in lines[3:6]] == ["baseline", *SCENARIOS]
    assert lines[6] == f"result page: {URL}"
    # The backtest's fit table: two windows times three scores, then its result page.
    assert lines[7].split() == ["window", "metric", "value", "sd"]
    fit = [line.split() for line in lines[8:14]]
    assert [row[:2] for row in fit] == [
        [window, metric] for window in ("context", "query") for metric in ("r2", "mape", "crps")
    ]
    assert float(fit[0][2]) == fit_record()["context"]["r2"]["value"]
    assert lines[14:] == [f"backtest result page: {URL}"]
    # The resume hint that the page describes, once for each analysis.
    assert err.splitlines() == [
        "analysis id: an_1  (resume: client.get('an_1').wait())",
        "analysis id: an_2  (resume: client.get('an_2').wait())",
    ]


@pytest.mark.parametrize("weeks", [104, 230])
def test_own_data_block_replaces_step_1_of_the_script(workdir, service, capsys, weeks):
    # Own data ending on the same week as the synthetic data, unsorted on disk. With 230 weeks
    # the block keeps the last 200, so history plus the 8-week plan stays within 208 weeks.
    dates = pd.date_range(end=HISTORY_END, periods=weeks, freq="7D")
    rng = np.random.default_rng(1)
    own = pd.DataFrame(
        {
            "week": dates,
            "sales": rng.uniform(30_000, 50_000, weeks),
            "tv": rng.uniform(0, 9_000, weeks),
            "search": rng.uniform(0, 7_000, weeks),
            "price": rng.uniform(9, 11, weeks),
        }
    )
    own_file = workdir / "path" / "to" / "your_history.csv"
    own_file.parent.mkdir(parents=True)
    own.sample(frac=1, random_state=0).to_csv(own_file, index=False)
    script = guide_example("quickstart")
    step_1 = re.search(r"# 1\..*?(?=# 2\.)", script, re.DOTALL).group(0)

    run(script.replace(step_1, guide_example("own-data") + "\n"), GUIDE)

    kept = own.tail(min(weeks, 200))
    check_upload(service, kept)
    (first,) = analysis_posts(service)
    assert first["plan"]["dates"] == PLAN_DATES
    assert len(kept) + len(first["plan"]["dates"]) <= 208
    assert f"result page: {URL}" in capsys.readouterr().out.splitlines()


def test_client_page_blocks_run_in_page_order(workdir, service):
    blocks = PYTHON_BLOCK.findall(CLIENT_PAGE.read_text())
    assert len(blocks) == 5  # create a client, analyze, backtest, again, preview and explore
    run("\n".join(blocks), CLIENT_PAGE)

    check_upload(service, synthetic_history())
    plan, backtest, again, preview, explore = analysis_posts(service)
    assert [s["id"] for s in plan["scenarios"]] == [
        *SCENARIOS,
        "tv_0.5__search_0.5",
        "tv_0.5__search_1.5",
        "tv_1.5__search_0.5",
        "tv_1.5__search_1.5",
    ]
    assert plan["scenarios"][1]["multipliers"] == {"tv": 0.8, "search": {"amount": 60_800}}
    assert plan["decision_weeks"] == [1, 8]
    assert backtest["context"] == {"start": HISTORY_START, "end": "2026-08-03"}
    assert backtest["query"] == {"end": HISTORY_END}
    assert again["scenarios"] == [{"id": "search_x1.5", "multipliers": {"search": 1.5}}]
    assert preview["preview"] is True and explore["preview"] is True
    assert [c["channel"] for c in explore["curves"]] == ["tv", "search"]


def test_readme_client_example_runs_as_pasted(workdir, service):
    (block,) = PYTHON_BLOCK.findall(section(README, "## Call the River API"))
    run(block, README)

    check_upload(service, synthetic_history())
    plan, backtest, again = analysis_posts(service)
    assert [s["id"] for s in plan["scenarios"]][:2] == list(SCENARIOS)
    assert plan["scenarios"][1]["multipliers"] == {"tv": 0.8, "search": {"amount": 60_800}}
    assert (plan["seed"], plan["name"]) == (7, "Spend scenarios")
    assert backtest["context"] == {"start": HISTORY_START, "end": "2026-08-03"}
    assert backtest["scenarios"] == [
        {"id": "budget_x1.2", "multipliers": {"tv": 1.2, "search": 1.2}}
    ]
    assert again["scenarios"] == [{"id": "search_x1.5", "multipliers": {"search": 1.5}}]
    # analysis.export writes the service's export bytes unchanged.
    assert json.loads((workdir / "analysis.json").read_text()) == export_record(None)
