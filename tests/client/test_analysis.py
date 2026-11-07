import json

import numpy as np
import pytest

from pymc_mfm.client import (
    Analysis,
    Gone,
    NotReady,
    Preview,
    curve,
    grid,
    scale,
    scenario,
    shift,
)

from .client_fixtures import (
    FakeService,
    curve_result,
    dataset_record,
    error_body,
    started,
    status_record,
)


@pytest.fixture
def dataset(client, service, history):
    service.on("POST", "/v1/datasets", (201, dataset_record()))
    return client.datasets.create(
        history,
        date="week",
        outcome="sales",
        spend=["tv", "search"],
        controls=["price"],
    )


def body_of(service: FakeService, path: str = "/v1/analyses"):
    return json.loads(service.sent("POST", path)[-1].content)


def test_analyze_sends_the_plan_scenarios_and_curves(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started()))
    scenarios = [
        scenario("half", scale(0.5)),
        scenario("half_frontload", scale(0.5), shift("tv", into_weeks=(1, 2))),
    ]
    analysis = client.analyze(
        dataset,
        plan,
        scenarios=scenarios,
        curves=[curve("tv", [0, 2])],
        decision_weeks=(1, 2),
        draws=500,
        seed=7,
        explain_history=True,
    )
    assert body_of(service) == {
        "dataset_id": "ds_123",
        "plan": {
            "dates": ["2024-07-29", "2024-08-05"],
            "spend": {"tv": [100.0, 100.0], "search": [50.0, 50.0]},
            "controls": {"price": [10.5, 10.5]},
        },
        "scenarios": scenarios,
        "curves": [{"channel": "tv", "levels": [0, 2], "base": "baseline"}],
        "decision_weeks": [1, 2],
        "draws": 500,
        "seed": 7,
        "explain_history": True,
    }
    assert isinstance(analysis, Analysis) and analysis.id == "an_1"
    assert analysis.expires_at.year == 2030


def test_analyze_omits_unset_options_and_sends_an_idempotency_key(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(dataset, plan)
    body = body_of(service)
    assert {"decision_weeks", "draws", "seed", "preview"}.isdisjoint(body)
    assert body["scenarios"] == [] and body["curves"] == []
    assert service.sent("POST", "/v1/analyses")[0].headers["Idempotency-Key"]


def test_analyze_retries_with_one_idempotency_key(client, service, dataset, plan, sleeps):
    service.on("POST", "/v1/analyses", (503, error_body("unavailable", "x")), (202, started(1)))
    client.analyze(dataset, plan)
    first, second = service.sent("POST", "/v1/analyses")
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]


def test_analyze_takes_a_dataset_id_and_looks_it_up(client, service, plan):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze("ds_123", plan)
    assert body_of(service)["plan"]["spend"].keys() == {"tv", "search"}


def test_plan_may_be_a_dict_in_wire_form(client, service, dataset):
    service.on("POST", "/v1/analyses", (202, started(1)))
    plan = {
        "dates": ["2024-07-29"],
        "spend": {"tv": np.array([1.0]), "search": [2.0]},
        "controls": {"price": [1.0]},
    }
    client.analyze(dataset, plan)
    assert body_of(service)["plan"]["spend"] == {"tv": [1.0], "search": [2.0]}


def test_plan_dataframe_may_use_a_datetime_index(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(dataset, plan.set_index("week"))
    assert body_of(service)["plan"]["dates"] == ["2024-07-29", "2024-08-05"]


def test_plan_without_dates_is_refused(client, service, dataset, plan):
    with pytest.raises(ValueError, match="date"):
        client.analyze(dataset, plan.reset_index(drop=True).drop(columns="week"))
    with pytest.raises(ValueError, match="date"):
        client.analyze(dataset, plan.assign(other=plan["week"]).rename(columns={"week": "w"}))


def test_plan_missing_a_channel_is_refused(client, service, dataset, plan):
    with pytest.raises(ValueError, match="search"):
        client.analyze(dataset, plan.drop(columns="search"))


def test_grid_scenarios_go_out_as_given(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(4)))
    client.analyze(dataset, plan, scenarios=grid(tv=[0.5, 1.5], search=[0.5, 1.5]))
    ids = [item["id"] for item in body_of(service)["scenarios"]]
    assert ids == [
        "tv_0.5__search_0.5",
        "tv_0.5__search_1.5",
        "tv_1.5__search_0.5",
        "tv_1.5__search_1.5",
    ]


def test_explore_builds_nine_level_curves_and_two_budget_scenarios(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(27)))
    client.explore(dataset, plan, seed=3)
    body = body_of(service)
    assert body["curves"] == [
        {
            "channel": channel,
            "levels": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            "base": "baseline",
        }
        for channel in ("tv", "search")
    ]
    assert body["scenarios"] == [
        {"id": "budget_x0.5", "steps": [{"scale": 0.5}]},
        {"id": "budget_x1.5", "steps": [{"scale": 1.5}]},
    ]
    assert body["seed"] == 3


def test_explore_refuses_its_own_arguments(client, dataset, plan):
    with pytest.raises(TypeError, match="scenarios"):
        client.explore(dataset, plan, scenarios=[])


def test_preview_returns_the_expansion_and_stores_nothing(client, service, dataset, plan):
    service.on(
        "POST",
        "/v1/analyses",
        (
            200,
            {
                "plans": [
                    {"scenario_id": "baseline", "steps": [], "spend": {"tv": [1.0, 1.0]}},
                    {"scenario_id": "half", "steps": [{"scale": 0.5}], "spend": {"tv": [0.5, 0.5]}},
                ],
                "runs_total": 2,
                "estimate_seconds": 30.0,
                "assumptions": {},
                "warnings": ["spend above history"],
            },
        ),
    )
    preview = client.analyze(dataset, plan, scenarios=[scenario("half", scale(0.5))], preview=True)
    assert body_of(service)["preview"] is True
    assert isinstance(preview, Preview)
    assert preview.runs_total == 2 and preview.warnings == ["spend above history"]
    assert preview.spend().loc[("half", 0), "tv"] == 0.5


def test_get_resumes_an_analysis(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("running")))
    analysis = client.get("an_1")
    assert analysis.id == "an_1" and analysis.expires_at.year == 2030


def test_get_of_an_expired_analysis_raises_gone(client, service):
    service.on("GET", "/v1/analyses/an_1", (410, error_body("expired", "expired")))
    with pytest.raises(Gone):
        client.get("an_1")


def test_wait_polls_until_done(client, service):
    service.on(
        "GET",
        "/v1/analyses/an_1",
        (200, status_record("queued")),
        (200, status_record("running")),
        (200, status_record("done")),
    )
    analysis = client.get("an_1")
    assert analysis.wait(poll=0) is analysis
    assert analysis.status().status == "done"
    assert analysis.status().finished


def test_wait_times_out(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("running")))
    analysis = client.get("an_1")
    with pytest.raises(TimeoutError, match="running"):
        analysis.wait(timeout=0, poll=0.01)


def test_wait_returns_for_a_failed_analysis(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("failed", runs_failed=3)))
    assert client.get("an_1").wait(poll=0).status().runs_failed == 3


def test_table_before_the_end_raises_not_ready(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("running")))
    service.on("GET", "/v1/analyses/an_1/result", (409, error_body("not_ready", "still running")))
    with pytest.raises(NotReady):
        client.get("an_1").table


@pytest.fixture
def done(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/result", (200, curve_result()))
    return client.get("an_1")


def test_table_is_a_long_pandas_table(done):
    table = done.table
    assert list(table.columns[:5]) == [
        "scenario_id",
        "kind",
        "base_scenario_id",
        "varied_channel",
        "level",
    ]
    assert {"window_mean", "mean_difference", "simulation_se", "draws_url"} <= set(table.columns)
    assert set(table["kind"]) == {"baseline", "scenario", "curve"}
    assert len(table) == 6 * (1 + 2 + 8)  # 6 components; baseline, 2 scenarios, 8 curve levels
    row = table[(table["scenario_id"] == "half") & (table["component"] == "tv")].iloc[0]
    assert (row["mean"], row["p05"], row["p95"], row["window_mean"]) == (50.0, 40.0, 60.0, 25.0)
    assert row["mean_difference"] == 5.0 and row["simulation_se"] == 0.5


def test_failed_plan_rows_have_missing_numbers(done):
    broken = done.table.query("scenario_id == 'broken'")
    assert set(broken["status"]) == {"failed"}
    assert broken[["mean", "p05", "p95", "window_mean"]].isna().all().all()
    assert broken["mean"].dtype == float


def test_labels_stay_on_the_objects(done):
    assert "difference_interval" in done.labels
    assert done.table.attrs["labels"] == done.labels
    assert done.meta["seed"] == 7
    assert done.assumptions["baseline"]["price"]["values"] == [10.5, 10.5]
    assert done.history is None


def test_the_result_is_fetched_once(done, service):
    done.table, done.table, done.labels
    assert len(service.sent("GET", "/v1/analyses/an_1/result")) == 1


def test_history_table_when_explained(client, service):
    record = curve_result()
    record["history"] = record["table"][:6]
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/result", (200, record))
    history = client.get("an_1").history
    assert len(history) == 6 and history.attrs["labels"]["intervals"]


def test_weekly_rows(client, service, done):
    service.on(
        "GET",
        "/v1/analyses/an_1/weekly",
        (
            200,
            {
                "scenario_id": "half",
                "rows": [
                    {"date": "2024-07-29", "component": "tv", "mean": 1.0, "p05": 0.5, "p95": 1.5},
                    {"date": "2024-08-05", "component": "tv", "mean": 2.0, "p05": 1.5, "p95": 2.5},
                ],
            },
        ),
    )
    weekly = done.weekly("half")
    request = service.sent("GET", "/v1/analyses/an_1/weekly")[0]
    assert request.url.params["scenario_id"] == "half"
    assert list(weekly.columns) == ["scenario_id", "date", "component", "mean", "p05", "p95"]
    assert weekly["date"].dtype.kind == "M" and weekly["mean"].tolist() == [1.0, 2.0]
    assert "intervals" in weekly.attrs["labels"]


def test_delete_an_analysis(client, service, done):
    service.on("DELETE", "/v1/analyses/an_1", (202, {"deletion": "in_progress"}))
    assert done.delete() is None
    assert service.sent("DELETE", "/v1/analyses/an_1")


def test_list_analyses_filters_by_dataset(client, service, dataset):
    item = {
        "analysis_id": "an_1",
        "status": "done",
        "runs_total": 3,
        "runs_done": 3,
        "runs_failed": 0,
        "expires_at": "2030-01-08T00:00:00Z",
    }
    service.on("GET", "/v1/analyses", (200, {"analyses": [item]}))
    frame = client.list_analyses(dataset)
    assert service.requests[-1].url.params["dataset_id"] == "ds_123"
    assert frame["analysis_id"].tolist() == ["an_1"]
