"""The main example in skills/mfm/SKILL.md runs against a fake service.

A client change that breaks the skill's example fails here. The example reads
history.csv and plan.csv from the working directory and writes compare.png.
"""

import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from .client_fixtures import (
    FakeService,
    analysis_posts,
    curve_result,
    dataset_record,
    export_record,
    marked_example,
    result_record,
    result_rows,
    route_examples_to,
    started,
    status_record,
    weeks_result,
    weeks_status,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "mfm"
SKILL = SKILL_DIR / "SKILL.md"
GRID_IDS = (
    "tv_0.5__search_0.5",
    "tv_0.5__search_1.5",
    "tv_1.5__search_0.5",
    "tv_1.5__search_1.5",
)


def compare_rows() -> list[dict]:
    rows = result_rows("baseline", "baseline")
    for scenario_id in ("reduce_20", "tv_to_search"):
        rows += result_rows(scenario_id, "scenario", base="baseline")
    return rows


def grid_rows() -> list[dict]:
    rows = result_rows("baseline", "baseline")
    for scenario_id in GRID_IDS:
        rows += result_rows(scenario_id, "scenario", base="baseline")
    return rows


def example_dataset_record() -> dict:
    """The fake service's summary of the skill's example history, so the client's
    check that the plan starts the week after the history passes as it does live."""
    history = pd.read_csv(SKILL_DIR / "examples" / "history.csv", parse_dates=["date"])
    record = dataset_record(controls=("holiday",))
    record["summary"] = {
        **record["summary"],
        "weeks": len(history),
        "start": history["date"].min().date().isoformat(),
        "end": history["date"].max().date().isoformat(),
    }
    return record


def analysis_id(number: int) -> dict:
    return {**started(), "analysis_id": f"an_{number}"}


@pytest.fixture
def service() -> FakeService:
    """The routes the examples call: preview, comparison, then optional examples."""
    fake = FakeService()
    fake.on("GET", "/v1/datasets", (200, {"datasets": []}))
    fake.on("POST", "/v1/datasets", (201, example_dataset_record()))
    plans = [{"scenario_id": "baseline", "spend": {"tv": [100.0, 100.0], "search": [50.0, 50.0]}}]
    preview = {"plans": plans, "runs_total": 3, "estimate_seconds": 40.0, "warnings": []}
    fake.on(
        "POST",
        "/v1/analyses",
        (200, preview),
        (202, analysis_id(1)),
        (202, analysis_id(2)),
        (202, analysis_id(3)),
    )
    for number in (1, 2, 3):
        fake.on("GET", f"/v1/analyses/an_{number}", (200, status_record("done")))
    fake.on("GET", "/v1/analyses/an_1/result", (200, result_record(compare_rows())))
    fake.on("GET", "/v1/analyses/an_2/result", (200, curve_result()))  # the optional example
    fake.on("GET", "/v1/analyses/an_3/result", (200, result_record(compare_rows())))
    weekly = [{"date": "2024-07-29", "component": "outcome", "mean": 1.0, "p05": 0.5, "p95": 1.5}]
    weekly_reply = {"scenario_id": "tv_to_search", "rows": weekly}
    fake.on("GET", "/v1/analyses/an_1/weekly", (200, weekly_reply))
    fake.on("DELETE", "/v1/datasets/ds_123", (202, {"deletion": "in_progress"}))
    return fake


@pytest.fixture
def workdir(tmp_path, monkeypatch, service):
    """A work directory with the skill's example CSVs, a fake network and the two variables."""
    for name in ("history.csv", "plan.csv"):
        shutil.copy(SKILL_DIR / "examples" / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    route_examples_to(service, monkeypatch)
    return tmp_path


def run_example(name: str, namespace: dict) -> None:
    code = marked_example(SKILL, "skill-example", name)
    exec(compile(code, str(SKILL), "exec"), namespace)


def test_main_example_compares_three_plans_in_one_analysis(workdir, service, capsys):
    namespace = {"__name__": "skill_example"}
    run_example("main", namespace)
    # The main example keeps the dataset, so the user can open the analysis on the site.
    assert not service.sent("DELETE", "/v1/datasets/ds_123")
    run_example("delete", namespace)

    assert [p.name for p in workdir.glob("*.png")] == ["compare.png"]
    assert (workdir / "compare.png").stat().st_size > 0
    preview, analysis = analysis_posts(service)  # exactly one preview and one analysis
    assert preview["preview"] is True and not analysis.get("preview")
    plan = pd.read_csv(workdir / "plan.csv")
    moved = plan["search"].sum() + 0.2 * plan["tv"].sum()
    for body in (preview, analysis):
        assert [s["id"] for s in body["scenarios"]] == ["reduce_20", "tv_to_search"]
        reduce, move = (s["multipliers"] for s in body["scenarios"])
        assert reduce == {"tv": 0.8, "search": 0.8}
        assert move["tv"] == 0.8 and move["search"] == {"amount": pytest.approx(moved)}
        assert not body.get("curves")
        assert all("steps" not in s for s in body["scenarios"])
    assert "result link:" in capsys.readouterr().out
    assert service.sent("DELETE", "/v1/datasets/ds_123")
    assert all(r.headers["Authorization"] == "Bearer test-key" for r in service.requests)


def test_backtest_example_sends_the_weeks_form_and_reads_fit_shares_and_curves(
    workdir, service, capsys
):
    service.on("GET", "/v1/analyses/an_2", (200, weeks_status()))
    service.on("GET", "/v1/analyses/an_2/result", (200, weeks_result()))
    namespace = {"__name__": "skill_example"}
    run_example("main", namespace)
    run_example("backtest", namespace)

    body = analysis_posts(service)[2]
    assert body["context"] == {"start": "2024-11-11", "end": "2026-08-03"}
    assert body["query"] == {"end": "2026-11-02"}
    assert "plan" not in body
    assert len(body["scenarios"]) == 1  # one budget scenario
    assert body["scenarios"][0]["multipliers"] == {"tv": 1.2, "search": 1.2}
    assert [c["channel"] for c in body["curves"]] == ["tv", "search"]
    assert (body["seed"], body["name"]) == (7, "Backtest")
    output = capsys.readouterr().out
    assert "r2" in output and "crps" in output  # the fit table was printed
    assert "channel" in output  # the shares and the curves were printed


def test_backtest_example_without_fit_keeps_the_result_link(workdir, service, capsys):
    url = "https://site.example/analyses/an_2"
    service.on(
        "GET",
        "/v1/analyses/an_2",
        (200, weeks_status("failed", runs_done=0, runs_failed=1, runs_total=1, url=url)),
    )
    failed = result_rows("baseline", "baseline", failed=True)
    service.on("GET", "/v1/analyses/an_2/result", (200, {**result_record(failed), "fit": None}))

    namespace = {"__name__": "skill_example"}
    run_example("main", namespace)
    run_example("backtest", namespace)

    output = capsys.readouterr().out.splitlines()
    assert f"result link: {url}" in output
    assert "No fit scores are available for this backtest." in output


def test_again_example_starts_a_second_analysis_with_the_same_weeks(workdir, service):
    service.on("GET", "/v1/analyses/an_2", (200, weeks_status()))
    service.on("GET", "/v1/analyses/an_2/result", (200, weeks_result()))
    weeks = ["2024-07-22", "2024-07-29", "2024-08-05"]
    inputs = {"dates": weeks, "spend": {}, "controls": {}, "decision_dates": weeks}
    service.on("GET", "/v1/analyses/an_2/export", (200, export_record(inputs)))
    namespace = {"__name__": "skill_example"}
    run_example("main", namespace)
    run_example("backtest", namespace)
    run_example("again", namespace)

    body = analysis_posts(service)[3]
    assert body["dataset_id"] == "ds_123"
    assert body["context"] == {"start": "2024-07-01", "end": "2024-07-15"}
    assert body["query"] == {"end": "2024-08-05"}
    assert body["scenarios"] == [{"id": "search_x1.5", "multipliers": {"search": 1.5}}]
    assert not body.get("curves") and body["explain_history"] is False
    assert (body["seed"], body["draws"]) == (7, 500)
    assert "decision_weeks" not in body  # the window was all the predicted weeks
    assert service.sent("GET", "/v1/analyses/an_3/result")


def test_curves_example_requests_one_curve_per_channel(workdir, service):
    namespace = {"__name__": "skill_example"}
    run_example("main", namespace)
    run_example("curves", namespace)

    body = analysis_posts(service)[2]
    assert [c["channel"] for c in body["curves"]] == ["tv", "search"]
    assert not body.get("scenarios")
    assert (workdir / "curves.png").stat().st_size > 0


def test_grid_example_requests_one_scenario_per_combination(workdir, service):
    service.on("GET", "/v1/analyses/an_2/result", (200, result_record(grid_rows())))
    namespace = {"__name__": "skill_example"}
    run_example("main", namespace)
    run_example("grid", namespace)

    body = analysis_posts(service)[2]
    assert [s["id"] for s in body["scenarios"]] == list(GRID_IDS)  # a 2x2 grid
    assert body["scenarios"][0]["multipliers"] == {"tv": 0.5, "search": 0.5}
    assert not body.get("curves")
    assert (workdir / "grid.png").stat().st_size > 0


def test_skill_files_name_no_service_address():
    text = "".join(path.read_text() for path in SKILL_DIR.rglob("*.md"))
    assert re.findall(r"https?://[^\s)\"'`>]+", text) == []
    assert "MFM_API_URL" in text and "MFM_API_KEY" in text


def test_example_data_fits_the_example():
    history = pd.read_csv(SKILL_DIR / "examples" / "history.csv", parse_dates=["date"])
    plan = pd.read_csv(SKILL_DIR / "examples" / "plan.csv", parse_dates=["date"])
    assert list(history.columns) == ["date", "sales", "tv", "search", "holiday"]
    assert list(plan.columns) == ["date", "tv", "search", "holiday"]
    assert len(history) == 104 and len(plan) == 8
    weeks = pd.concat([history["date"], plan["date"]])
    assert (weeks.diff().dropna() == pd.Timedelta(days=7)).all()
