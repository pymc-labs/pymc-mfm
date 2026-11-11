import json

import numpy as np
import pandas as pd
import pytest

from pymc_mfm.client import (
    Analysis,
    Gone,
    NotReady,
    PayloadTooLarge,
    Preview,
    amount,
    curve,
    grid,
    scenario,
)

from .client_fixtures import (
    FakeService,
    curve_result,
    dataset_record,
    error_body,
    export_record,
    started,
    status_record,
    weeks_result,
    weeks_status,
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
        scenario("half", {"tv": 0.5, "search": 0.5}),
        scenario("tv_to_search", {"tv": 0.8, "search": amount(120.0)}),
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
    client.analyze(dataset, plan, scenarios=grid({"tv": [0.5, 1.5], "search": [0.5, 1.5]}))
    ids = [item["id"] for item in body_of(service)["scenarios"]]
    assert ids == [
        "tv_0.5__search_0.5",
        "tv_0.5__search_1.5",
        "tv_1.5__search_0.5",
        "tv_1.5__search_1.5",
    ]


def test_explore_builds_nine_level_curves_and_three_budget_scenarios(
    client, service, dataset, plan
):
    service.on("POST", "/v1/analyses", (202, started(28)))
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
        {"id": f"spend_x{factor}", "multipliers": {"tv": factor, "search": factor}}
        for factor in (1.2, 1.4, 1.6)
    ]
    assert body["seed"] == 3


def test_explore_of_some_channels_varies_only_those(client, service, dataset):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.explore(
        dataset, context=("2024-07-01", "2024-07-15"), query_end="2024-08-05", channels=["tv"]
    )
    body = body_of(service)
    assert [item["channel"] for item in body["curves"]] == ["tv"]
    assert body["scenarios"][0] == {"id": "spend_x1.2", "multipliers": {"tv": 1.2}}
    assert body["channels"] == ["tv"] and "plan" not in body


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
                    {"scenario_id": "baseline", "multipliers": None, "spend": {"tv": [1.0, 1.0]}},
                    {
                        "scenario_id": "half",
                        "multipliers": {"tv": 0.5},
                        "spend": {"tv": [0.5, 0.5]},
                    },
                ],
                "runs_total": 2,
                "estimate_seconds": 30.0,
                "assumptions": {},
                "warnings": ["spend above history"],
            },
        ),
    )
    preview = client.analyze(dataset, plan, scenarios=[scenario("half", {"tv": 0.5})], preview=True)
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
    assert list(table.columns[:6]) == [
        "scenario_id",
        "kind",
        "base_scenario_id",
        "varied_channel",
        "level",
        "multipliers",
    ]
    assert {"window_mean", "mean_difference", "simulation_se", "draws_url"} <= set(table.columns)
    assert set(table["kind"]) == {"baseline", "scenario", "curve"}
    assert len(table) == 6 * (1 + 2 + 8)  # 6 components; baseline, 2 scenarios, 8 curve levels
    row = table[(table["scenario_id"] == "half") & (table["component"] == "tv")].iloc[0]
    assert (row["mean"], row["p05"], row["p95"], row["window_mean"]) == (50.0, 40.0, 60.0, 25.0)
    assert row["mean_difference"] == 5.0 and row["simulation_se"] == 0.5


def test_multipliers_column_holds_each_scenarios_change(done):
    table = done.table
    half = table.query("scenario_id == 'half'")["multipliers"]
    assert half.tolist() == [{"tv": 0.5, "search": 0.5}] * 6  # one row per component
    assert table.query("scenario_id == 'baseline'")["multipliers"].isna().all()


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


def test_analyze_sends_an_optional_name_and_keeps_name_and_url(client, service, dataset, plan):
    reply = {**started(), "name": "Spend scenarios", "url": "https://site.example/analyses/an_1"}
    service.on("POST", "/v1/analyses", (202, reply))
    analysis = client.analyze(dataset, plan, name="Spend scenarios")
    assert body_of(service)["name"] == "Spend scenarios"
    assert analysis.name == "Spend scenarios"
    assert analysis.url == "https://site.example/analyses/an_1"


def test_analyze_omits_the_name_and_an_older_server_has_no_url(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started()))
    analysis = client.analyze(dataset, plan)
    assert "name" not in body_of(service)
    assert analysis.name is None and analysis.url is None


def test_get_reads_name_and_url_and_the_expiry_can_change(client, service):
    first = {
        **status_record("running"),
        "name": "Mine",
        "url": "https://site.example/analyses/an_1",
    }
    later = {**status_record("done"), "expires_at": "2030-01-20T00:00:00Z"}
    service.on("GET", "/v1/analyses/an_1", (200, first), (200, later))
    analysis = client.get("an_1")
    assert (analysis.name, analysis.url) == ("Mine", "https://site.example/analyses/an_1")
    assert analysis.expires_at.day == 8
    analysis.status()
    assert analysis.expires_at.day == 20
    assert analysis.url == "https://site.example/analyses/an_1"  # kept when a reply omits it


def test_export_downloads_the_json_file(client, service, tmp_path):
    document = b'{"format": "mfm-analysis-export-v1", "analysis_id": "an_1"}'
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done")))
    service.on("GET", "/v1/analyses/an_1/export", (200, document))
    path = client.get("an_1").export(tmp_path / "out.json")
    assert path.read_bytes() == document
    request = service.sent("GET", "/v1/analyses/an_1/export")[0]
    assert request.headers["Authorization"] == "Bearer test-key"


def test_export_of_an_unfinished_analysis_raises_not_ready(client, service, tmp_path):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("running")))
    service.on("GET", "/v1/analyses/an_1/export", (409, error_body("not_ready", "still running")))
    with pytest.raises(NotReady):
        client.get("an_1").export(tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


PLAN_START = "Your history ends on 2024-07-22. Your plan starts on {start}."
NEXT_DATE = "The next required date is 2024-07-29."
CHECK = "Check whether a week is missing from your history or plan."


def wire_plan(dates):
    count = len(dates)
    return {
        "dates": dates,
        "spend": {"tv": [1.0] * count, "search": [2.0] * count},
        "controls": {"price": [1.0] * count},
    }


@pytest.mark.parametrize("start", ["2024-08-05", "2024-07-22", "2024-07-15"])
def test_a_plan_not_starting_the_week_after_the_history_is_refused(
    client, service, dataset, plan, start
):
    late = plan.assign(week=[pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=7)])
    message = f"{PLAN_START.format(start=start)} {NEXT_DATE} {CHECK}"
    for bad in (late, wire_plan([start, "2099-01-01"])):
        with pytest.raises(ValueError) as error:
            client.analyze(dataset, bad)
        assert str(error.value) == message
    assert not service.sent("POST", "/v1/analyses")


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (
            ["2024-07-29", "2024-08-05", "2024-08-19"],
            "Row 3 of your plan has the date 2024-08-19, but row 2 has 2024-08-05.",
        ),
        (
            ["2024-07-29", "2024-08-05", "2024-08-05"],
            "Row 3 of your plan repeats the date 2024-08-05, which is also in row 2.",
        ),
        (
            ["2024-07-29", "2024-08-12", "2024-08-05"],
            "Row 2 of your plan has the date 2024-08-12, but row 1 has 2024-07-29. "
            "The next required date is 2024-08-05, which is in row 3.",
        ),
    ],
)
def test_a_plan_with_a_gap_duplicate_or_reversed_dates_names_the_row(
    client, service, dataset, plan, dates, message
):
    frame = pd.DataFrame({"week": pd.to_datetime(dates), "tv": 1.0, "search": 2.0, "price": 3.0})
    frame.index = [7, 5, 6]  # index labels are not row positions
    for bad in (frame, wire_plan(dates)):
        with pytest.raises(ValueError, match=message):
            client.analyze(dataset, bad)
    assert not service.sent("POST", "/v1/analyses")


@pytest.mark.parametrize("call", ["analyze", "explore"])
def test_a_late_plan_is_refused_for_a_dataset_id_before_any_submission(client, service, plan, call):
    service.on("GET", "/v1/datasets/ds_123", (200, dataset_record()))
    late = plan.assign(week=pd.date_range("2024-08-05", periods=2, freq="W-MON"))
    with pytest.raises(ValueError, match="Your plan starts on 2024-08-05"):
        getattr(client, call)("ds_123", late)
    assert not service.sent("POST", "/v1/analyses")


def test_a_corrected_plan_is_sent_unchanged(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(dataset, plan)
    client.analyze(dataset, wire_plan(["2024-07-29", "2024-08-05"]))
    first, second = (json.loads(r.content)["plan"]["dates"] for r in service.requests[-2:])
    assert first == second == ["2024-07-29", "2024-08-05"]


def test_a_wire_plan_without_dates_is_left_to_the_service(client, service, dataset):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(dataset, {"spend": {"tv": [1.0]}})
    assert len(service.sent("POST", "/v1/analyses")) == 1


@pytest.mark.parametrize("container", [np.array, pd.Series, pd.Index])
def test_a_wire_plan_with_array_dates_is_sent_as_before(client, service, dataset, container):
    service.on("POST", "/v1/analyses", (202, started(1)))
    dates = ["2024-07-29", "2024-08-05"]
    client.analyze(dataset, wire_plan(dates))
    client.analyze(dataset, wire_plan(container(dates)))
    first, second = (r.content for r in service.requests[-2:])
    assert json.loads(first)["plan"] == json.loads(second)["plan"]


# The weeks (backtest) form

CONTEXT = ("2024-07-01", "2024-07-15")
QUERY_END = "2024-08-05"


def test_analyze_sends_the_weeks_form_without_a_plan(client, service, dataset):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(dataset, context=CONTEXT, query_end=QUERY_END, seed=7)
    assert body_of(service) == {
        "dataset_id": "ds_123",
        "context": {"start": "2024-07-01", "end": "2024-07-15"},
        "query": {"end": "2024-08-05"},
        "scenarios": [],
        "curves": [],
        "seed": 7,
    }


def test_weeks_form_sends_chosen_series_and_accepts_timestamps(client, service, dataset):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(
        "ds_123",
        context=(pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-15").date()),
        query_end=pd.Timestamp(QUERY_END),
        channels="tv",
        controls=[],
    )
    body = body_of(service)
    assert body["context"] == {"start": "2024-07-01", "end": "2024-07-15"}
    assert body["query"] == {"end": "2024-08-05"}
    assert body["channels"] == ["tv"] and body["controls"] == []
    assert not service.sent("GET", "/v1/datasets/ds_123")  # the weeks form needs no lookup


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "plan, or context and query_end"),
        ({"context": CONTEXT}, "context and query_end together"),
        ({"query_end": QUERY_END}, "context and query_end together"),
        ({"context": CONTEXT, "query_end": QUERY_END, "plan": {}}, "not both"),
        ({"context": ("2024-07-01",), "query_end": QUERY_END}, "pair"),
        ({"context": "2024-07-01", "query_end": QUERY_END}, "pair"),
    ],
)
def test_bad_weeks_arguments_are_refused_before_any_request(
    client, service, dataset, arguments, message
):
    with pytest.raises(ValueError, match=message):
        client.analyze(dataset, **arguments)
    assert not service.sent("POST", "/v1/analyses")


@pytest.mark.parametrize("option", ["channels", "controls"])
def test_series_choice_needs_the_weeks_form(client, service, dataset, plan, option):
    with pytest.raises(ValueError, match="channels and controls"):
        client.analyze(dataset, plan, **{option: ["tv"]})
    assert not service.sent("POST", "/v1/analyses")


def test_weeks_form_preview_sends_the_flag(client, service, dataset):
    service.on(
        "POST",
        "/v1/analyses",
        (200, {"plans": [], "runs_total": 1, "estimate_seconds": 5.0, "warnings": []}),
    )
    preview = client.analyze(dataset, context=CONTEXT, query_end=QUERY_END, preview=True)
    assert isinstance(preview, Preview)
    assert body_of(service)["preview"] is True


# fit, shares and curves


@pytest.fixture
def backtest(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, weeks_status()))
    service.on("GET", "/v1/analyses/an_1/result", (200, weeks_result()))
    return client.get("an_1")


def test_fit_has_one_row_per_window_and_metric(backtest):
    fit = backtest.fit
    assert list(fit.columns) == ["window", "metric", "value", "sd"]
    assert fit["window"].tolist() == ["context"] * 3 + ["query"] * 3
    assert fit["metric"].tolist() == ["r2", "mape", "crps"] * 2
    query_r2 = fit.query("window == 'query' and metric == 'r2'").iloc[0]
    assert (query_r2["value"], query_r2["sd"]) == (0.8, 0.02)
    crps = fit.query("metric == 'crps'")
    assert crps["value"].tolist() == [12.5, 12.5] and crps["sd"].isna().all()
    assert fit.attrs["labels"] == backtest.labels


def test_shares_have_a_total_and_a_mean_per_series_and_window(backtest):
    shares = backtest.shares
    assert list(shares.columns) == ["window", "name", "group", "share", "value", "p05", "p95"]
    assert len(shares) == 2 * 2 * 2  # windows, series, total and mean
    row = shares.query("window == 'query' and name == 'tv' and share == 'total'").iloc[0]
    assert row["group"] == "channel"
    assert [row["value"], row["p05"], row["p95"]] == pytest.approx([0.4, 0.35, 0.45])
    control = shares.query("name == 'price' and share == 'mean'").iloc[0]
    assert control["group"] == "control" and control["value"] == pytest.approx(-0.05)


def test_curves_have_one_row_per_channel_and_level(backtest):
    curves = backtest.curves
    assert list(curves.columns) == ["channel", "spend", "level", "value", "p05", "p95"]
    assert curves["level"].tolist() == [0.0, 0.5, 1.5]
    assert curves["spend"].tolist() == [220.0] * 3
    assert curves.query("level == 0.5").iloc[0]["value"] == 40.0
    assert curves.query("level == 1.5")[["value", "p05", "p95"]].isna().all().all()


def test_fit_shares_and_curves_are_none_for_a_plan(done):
    assert done.fit is None and done.shares is None and done.curves is None


def test_a_backtest_table_has_multipliers(backtest):
    assert backtest.table.query("scenario_id == 'half'")["multipliers"].iloc[0] == {"tv": 0.5}


# Analysis.again


def baseline_inputs(**extra):
    return {
        "dates": ["2024-07-29", "2024-08-05", "2024-08-12"],
        "spend": {"tv": [1.0, 2.0, 3.0], "search": [4.0, 5.0, 6.0]},
        "controls": {"price": [10.0, 10.0, 10.0]},
        **extra,
    }


def test_again_repeats_a_backtest(client, service, backtest):
    service.on("POST", "/v1/analyses", (202, {**started(2), "analysis_id": "an_2"}))
    more = backtest.again([scenario("search_x1.5", {"search": 1.5})])
    assert body_of(service) == {
        "dataset_id": "ds_123",
        "context": {"start": "2024-07-01", "end": "2024-07-15"},
        "query": {"end": "2024-08-05"},
        "channels": ["tv", "search"],
        "controls": ["price"],
        "scenarios": [{"id": "search_x1.5", "multipliers": {"search": 1.5}}],
        "explain_history": False,
        "seed": 7,
        "draws": 500,
    }
    assert more.id == "an_2"
    assert not service.sent("GET", "/v1/analyses/an_1/export")  # the weeks need no export


def test_again_passes_curves_and_explain_history(client, service, backtest):
    service.on("POST", "/v1/analyses", (202, started(1)))
    backtest.again([], curves=[curve("tv", [0, 2])], explain_history=True)
    body = body_of(service)
    assert body["curves"] == [{"channel": "tv", "levels": [0, 2], "base": "baseline"}]
    assert body["explain_history"] is True


def test_again_repeated_sends_the_same_idempotency_key(client, service, backtest):
    service.on("POST", "/v1/analyses", (202, started(1)))
    scenarios = [scenario("search_x1.5", {"search": 1.5})]
    backtest.again(scenarios)
    backtest.again(scenarios)
    first, second = service.sent("POST", "/v1/analyses")
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]


def classic_status(**extra):
    return status_record("done", dataset_id="ds_123", seed=3, draws=200, **extra)


def test_again_repeats_a_plan_from_the_exports_baseline(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, classic_status()))
    service.on("GET", "/v1/analyses/an_1/export", (200, export_record(baseline_inputs())))
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.get("an_1").again([scenario("tv_up", {"tv": 1.2})])
    body = body_of(service)
    assert body["plan"] == {
        "dates": ["2024-07-29", "2024-08-05", "2024-08-12"],
        "spend": {"tv": [1.0, 2.0, 3.0], "search": [4.0, 5.0, 6.0]},
        "controls": {"price": [10.0, 10.0, 10.0]},
    }
    assert "decision_weeks" not in body and "context" not in body
    assert (body["dataset_id"], body["seed"], body["draws"]) == ("ds_123", 3, 200)


def test_again_turns_decision_dates_into_decision_weeks(client, service):
    inputs = baseline_inputs(decision_dates=["2024-08-05", "2024-08-12"])
    service.on("GET", "/v1/analyses/an_1", (200, classic_status()))
    service.on("GET", "/v1/analyses/an_1/export", (200, export_record(inputs)))
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.get("an_1").again([scenario("tv_up", {"tv": 1.2})])
    assert body_of(service)["decision_weeks"] == [2, 3]


def test_again_ignores_decision_dates_that_cover_the_plan(client, service):
    inputs = baseline_inputs(decision_dates=baseline_inputs()["dates"])
    service.on("GET", "/v1/analyses/an_1", (200, classic_status()))
    service.on("GET", "/v1/analyses/an_1/export", (200, export_record(inputs)))
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.get("an_1").again([])
    assert "decision_weeks" not in body_of(service)


def test_again_needs_a_stored_dataset(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, status_record("done", dataset_id=None)))
    with pytest.raises(ValueError, match="no stored dataset"):
        client.get("an_1").again([])
    assert not service.sent("POST", "/v1/analyses")


def test_again_needs_a_stored_baseline_plan(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, classic_status()))
    service.on("GET", "/v1/analyses/an_1/export", (200, export_record(None)))
    with pytest.raises(ValueError, match="no stored baseline plan"):
        client.get("an_1").again([])
    assert not service.sent("POST", "/v1/analyses")


def test_again_of_a_plan_waits_for_the_runs(client, service):
    service.on("GET", "/v1/analyses/an_1", (200, classic_status()))
    service.on("GET", "/v1/analyses/an_1/export", (409, error_body("not_ready", "running")))
    with pytest.raises(NotReady):
        client.get("an_1").again([])


# Errors and keys


def test_a_413_raises_payload_too_large_without_retries(client, service, dataset, plan, sleeps):
    service.on("POST", "/v1/analyses", (413, error_body("payload_too_large", "Too big.")))
    with pytest.raises(PayloadTooLarge) as caught:
        client.analyze(dataset, plan)
    assert caught.value.status == 413
    assert len(service.sent("POST", "/v1/analyses")) == 1 and sleeps == []


def test_wait_raises_gone_when_the_analysis_is_deleted_meanwhile(client, service):
    service.on(
        "GET",
        "/v1/analyses/an_1",
        (200, status_record("running")),
        (410, error_body("expired", "The analysis was deleted.")),
    )
    with pytest.raises(Gone):
        client.get("an_1").wait(poll=0)


def test_an_equal_analyze_call_sends_the_same_content_key(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(1)))
    scenarios = [scenario("half", {"tv": 0.5})]
    client.analyze(dataset, plan, scenarios=scenarios, seed=1)
    client.analyze(dataset, plan, scenarios=scenarios, seed=1)
    client.analyze(dataset, plan, scenarios=scenarios, seed=2)
    keys = [r.headers["Idempotency-Key"] for r in service.sent("POST", "/v1/analyses")]
    same, again, other = keys
    assert same == again != other


def test_analyze_uses_a_caller_key_as_given(client, service, dataset, plan):
    service.on("POST", "/v1/analyses", (202, started(1)))
    client.analyze(dataset, plan, idempotency_key="mine-1")
    assert service.sent("POST", "/v1/analyses")[0].headers["Idempotency-Key"] == "mine-1"
