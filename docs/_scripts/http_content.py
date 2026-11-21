# ruff: noqa: E501  (page copy: long prose lines are easier to review than split strings)
"""Page copy for the HTTP API reference.

``http_pages.py`` reads the routes, the headers, the fields and the errors of each page from the
exported spec (``docs/api/openapi-v1.json``) and fails if this module has no page for a route of
the spec, or a page for a route it lacks. This module keeps only the prose, the samples and the
hand-written notes of each page. Response values in the samples are illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    """One header, path, query or body field."""

    name: str
    text: str
    kind: str = ""
    required: bool = False


@dataclass(frozen=True)
class Detail:
    """The copy of a route page. The spec supplies its headers, fields and errors. Empty fields "
    "leave their section out."""

    slug: str
    title: str
    summary: str
    lead: str
    notes: tuple[tuple[str, str], ...] = ()
    response: str = ""
    response_text: str = ""
    curl: str = ""
    python: str = ""
    extra_blocks: tuple[tuple[str, str, str], ...] = field(default=())


AUTH = Param(
    "Authorization",
    "`Bearer $MKTFM_API_KEY`. See [Authentication](authentication.qmd).",
    required=True,
)
JSON_TYPE = Param("Content-Type", "`application/json`. The body is JSON only.", required=True)
ID_PARAM = Param("id", "The id from the create call.", "path")

DATASET_RECORD = """{
  "dataset_id": "ds_123",
  "content_hash": "mfm-dataset-v1:<sha256 hex digest>",
  "created_at": "2026-11-09T09:30:00Z",
  "last_used_at": "2026-11-09T09:30:00Z",
  "analyses": 0,
  "name": "weekly-history",
  "expires_at": "2026-11-16T09:30:00Z",
  "summary": {
    "weeks": 3, "start": "2026-10-19", "end": "2026-11-02",
    "channels": ["tv", "search"], "controls": ["holiday"],
    "outcome": "sales", "currency": "USD"
  },
  "history_ranges": {
    "tv": {"p05": 4000.0, "p95": 4400.0, "max": 4400.0, "zero_share": 0.0, "all_zero": false}
  },
  "warnings": [],
  "warning_details": []
}"""

DATASET_BODY = """{
  "name": "weekly-history",
  "dates": ["2026-10-19", "2026-10-26", "2026-11-02"],
  "outcome": [1012.5, 1048.0, 1031.2],
  "spend": {"tv": [4000, 4200, 4400], "search": [2500, 2600, 2700]},
  "controls": {"holiday": [0, 0, 0]}
}"""

PLAN_BODY = """{
  "dataset_id": "ds_123",
  "name": "Spend scenarios",
  "plan": {
    "dates": ["2026-11-09", "2026-11-16"],
    "spend": {"tv": [9000, 9000], "search": [6000, 6000]},
    "controls": {"holiday": [0, 0]}
  },
  "decision_weeks": [1, 2],
  "scenarios": [
    {"id": "reduce_20", "multipliers": {"tv": 0.8, "search": 0.8}}
  ],
  "curves": [{"channel": "tv", "levels": [0, 0.5, 1.5, 2]}],
  "draws": 1000,
  "seed": 7
}"""

BACKTEST_BODY = """{
  "dataset_id": "ds_123",
  "name": "Q4 backtest",
  "context": {"start": "2026-10-19", "end": "2026-10-26"},
  "query": {"end": "2026-11-02"},
  "channels": ["tv", "search"],
  "controls": ["holiday"],
  "scenarios": [
    {"id": "budget_x1.2", "multipliers": {"tv": 1.2, "search": 1.2}}
  ],
  "curves": [{"channel": "tv", "levels": [0, 0.5, 1.5, 2]}],
  "seed": 7
}"""


def curl_data(body: str) -> str:
    """A JSON body as the last line of a curl command, indented under ``--data``."""
    return "  --data '" + body.replace("\n", "\n  ") + "'"


STATUS_RECORD = """{
  "analysis_id": "an_123", "dataset_id": "ds_123",
  "status": "running",
  "runs_total": 6, "runs_done": 3, "runs_failed": 0, "runs_cancelled": 0,
  "estimate_seconds": 40,
  "name": "Spend scenarios", "url": null,
  "expires_at": "2026-11-16T09:30:00Z",
  "seed": 7, "draws": 1000, "explain_history": false, "context": null,
  "warnings": [], "warning_details": []
}"""

RESULT_RECORD = """{
  "table": [
    {"scenario_id": "baseline", "kind": "baseline", "component": "outcome", "status": "ok",
     "mean": 51234.5, "p05": 48800.1, "p95": 53710.9, "draws_url": "<signed link>"}
  ],
  "labels": {"intervals": "...", "simulation_se": "...", "difference_interval": "..."},
  "history": "...", "assumptions": "...", "meta": "...", "context": null,
  "fit": null, "response_curves": null, "input_usage": "..."
}"""

EXPORT_RECORD = """{
  "format": "mfm-analysis-export-v1",
  "analysis_id": "an_123", "name": "Spend scenarios",
  "created_at": "2026-11-09T09:40:00Z", "expires_at": "2026-11-16T09:30:00Z",
  "dataset": "...", "request": "...", "units": "...", "results": "...",
  "seed": 7, "draws": 1000,
  "model_revision": "...", "library_revision": "...", "exported_at": "2026-11-09T10:00:00Z",
  "scenarios": [
    {"scenario_id": "baseline", "kind": "baseline", "base_scenario_id": null,
     "multipliers": {}, "status": "ok", "failure_code": null,
     "inputs": "...", "warnings": [], "warning_details": []}
  ]
}"""

LIST_ANALYSES_RECORD = """{
  "analyses": [
    {"analysis_id": "an_123", "dataset_id": "ds_123", "name": "Spend scenarios", "url": null,
     "status": "done", "runs_total": 6, "runs_done": 6, "runs_failed": 0,
     "created_at": "2026-11-09T09:40:00Z", "expires_at": "2026-11-16T09:30:00Z"}
  ]
}"""

CURL_GET = 'curl -sS "$MKTFM_API_URL{path}" \\\n  -H "Authorization: Bearer $MKTFM_API_KEY"'
CURL_DELETE = (
    'curl -sS -X DELETE "$MKTFM_API_URL{path}" \\\n  -H "Authorization: Bearer $MKTFM_API_KEY"'
)
CLIENT = (
    "from pymc_mktfm.client import Client\n\nclient = Client()  # reads MKTFM_API_URL and "
    "MKTFM_API_KEY\n"
)

DETAILS: dict[tuple[str, str], Detail] = {
    ("POST", "/v1/datasets"): Detail(
        slug="datasets-create",
        title="Upload a dataset",
        summary="Store one weekly history.",
        lead="Store a weekly history of the outcome, the spend of each channel and the controls. "
        "Later calls refer to it by `dataset_id`.",
        notes=(
            (
                "Limits",
                "The body is at most 5 MB. `outcome_name` and `currency` are not part of the "
                "content hash. "
                "The summary returns them as `summary.outcome` and `summary.currency`. "
                "The samples show three weeks to stay short.",
            ),
            (
                "Compare spend with the history",
                "`history_ranges` gives, for each channel and control, the weekly `p05`, `p95` and "
                "`max`, "
                "`zero_share` and `all_zero`. Spend outside the 5th to 95th percentile of the "
                "history gets a "
                "warning, and the model has little data there. The percentiles are warning "
                "thresholds, "
                "not the edge of the observed data.",
            ),
        ),
        response=DATASET_RECORD,
        curl=(
            "KEY=$(uuidgen)  # reuse on retries\n"
            'curl -sS -X POST "$MKTFM_API_URL/v1/datasets" \\\n'
            '  -H "Authorization: Bearer $MKTFM_API_KEY" \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -H "Idempotency-Key: $KEY" \\\n' + curl_data(DATASET_BODY)
        ),
        python=CLIENT + "\n# history: a pandas DataFrame with one row for each week\n"
        "dataset = client.datasets.create(\n"
        '    history, date="week", outcome="sales",\n'
        '    spend=["tv", "search"], controls=["holiday"], name="weekly-history",\n'
        ")",
    ),
    ("GET", "/v1/datasets"): Detail(
        slug="datasets-list",
        title="List datasets",
        summary="List your stored datasets.",
        lead="List your datasets. Call this before an upload, and reuse a stored dataset when its "
        "`content_hash` equals the hash of your data.",
        notes=(
            (
                "Reuse a stored dataset",
                "Match on `content_hash` only. Never match by name or by number of weeks: one "
                "changed value "
                "gives a different hash. To compute the hash, take `dates`, `outcome`, `spend` and "
                "`controls` "
                "(not `name`). Write each date as `YYYY-MM-DD`. Convert every number to a float: "
                "`-0.0` becomes "
                "`0.0`, and NaN and infinity are errors. Sort the rows by date and reorder every "
                "series the "
                "same way. Build "
                '`{"v":1,"dates":[...],"outcome":[...],"spend":{...},"controls":{...}}` '
                'and encode it with `json.dumps(obj, sort_keys=True, separators=(",", ":"), '
                'allow_nan=False, ensure_ascii=False)` as UTF-8. The hash is `"mfm-dataset-v1:"` '
                "plus the "
                "SHA-256 hex digest. The Python client does this in `pymc_mktfm.client.dataset_content_hash`.",
            ),
        ),
        response='{\n  "datasets": [\n'
        + "    ...one dataset record each, as in Upload a dataset...\n  ]\n}",
        curl=CURL_GET.format(path="/v1/datasets"),
        python=CLIENT + "\nclient.datasets.list()",
    ),
    ("GET", "/v1/datasets/{id}"): Detail(
        slug="datasets-get",
        title="Get a dataset",
        summary="Read one dataset summary.",
        lead="Read the summary of one dataset. Read it again when you need `expires_at`: using a dataset moves it later.",
        notes=(
            (
                "Retention",
                "The service keeps a dataset 7 days after last use, and at most 30 days after "
                "upload. "
                "Using a dataset, for example by starting an analysis on it, moves `expires_at` "
                "later. "
                "Do not treat it as a fixed deadline. An analysis has the `expires_at` of its dataset.",
            ),
        ),
        response=DATASET_RECORD,
        curl=CURL_GET.format(path="/v1/datasets/ds_123"),
        python=CLIENT + '\nclient.datasets.get("ds_123")',
    ),
    ("DELETE", "/v1/datasets/{id}"): Detail(
        slug="datasets-delete",
        title="Delete a dataset",
        summary="Delete a dataset and all its analyses.",
        lead="Delete a dataset. This also deletes all its analyses.",
        response_text="The service accepted the delete. Later calls with the id answer `410`.",
        curl=CURL_DELETE.format(path="/v1/datasets/ds_123"),
        python=CLIENT + '\nclient.datasets.delete("ds_123")',
    ),
    ("POST", "/v1/analyses"): Detail(
        slug="analyses-create",
        title="Create an analysis",
        summary="Start an analysis, or preview what it would run.",
        lead="Start an analysis of a stored dataset: one forecast, with optional budget scenarios and "
        "response curves. The call answers `202`. Poll the analysis, then read the result. "
        'With `"preview": true` it answers `200` and stores and runs nothing.',
        notes=(
            (
                "Plan and backtest",
                "A **plan** gives the weeks after the history. It is the baseline, with id "
                "`baseline` (reserved). "
                "It holds spend for every channel and a value for every control, for every "
                "forecast week. "
                "Its dates start the week after the history.\n\n"
                "A **backtest** uses weeks the dataset already holds. The model reads the input "
                "weeks "
                "`context.start` to `context.end` and predicts the weeks after them, up to "
                "`query.end`. "
                "It does not see the outcome of the predicted weeks, so the result can score the "
                "forecast "
                "against it. The baseline is the dataset's own spend and controls in the predicted "
                "weeks. "
                "Every date is a week of the dataset, and `query.end` is after `context.end`. "
                "Do not send `plan` with them: the answer is `422 plan_with_context`.\n\n"
                "The window is the only part of the forecast that scenarios and curves change: the "
                "predicted "
                "weeks of a backtest, or `decision_weeks` of a plan.",
            ),
            (
                "Scenarios",
                'A scenario is `{"id", "multipliers"}`. The `id` is unique, 1 to 64 characters '
                "from "
                "`A-Za-z0-9_.-`. `baseline` and the `curve:` prefix are reserved. `multipliers` "
                "maps a channel "
                "to a number from 0 to 5 (rounded to 4 decimals), which multiplies its spend in "
                "the window, or "
                'to `{"amount": X}` (X is 0 or more, rounded to cents), which sets its total spend '
                "in the "
                "window to X. An amount keeps the weekly pattern. On a channel with no spend in "
                "the window, it "
                "puts X / W in each of the W window weeks. A channel left out keeps its spend. A "
                "scenario "
                "keeps every control.",
            ),
            (
                "Curves",
                'A curve is `{"channel", "levels", "base"?}`. Each level multiplies the spend of '
                "that channel "
                "in the window. Level 1 equals the base and does not run again. Level ids are "
                "`curve:<channel>:<base>:<level>`. For a first look, use levels 0 to 2 in steps of "
                "0.25. "
                "An analysis has at most 200 different plans, and equal plans run once.",
            ),
        ),
        response='{\n  "analysis_id": "an_123",\n  "dataset_id": "ds_123",\n  "name": "Spend scenarios",\n  "url": null,\n'
        '  "runs_total": 6,\n  "estimate_seconds": 40,\n  "expires_at": "2026-11-16T09:30:00Z",\n'
        '  "warnings": [],\n  "warning_details": []\n}',
        response_text='`url` is `null` when the service has no site address. A repeat with the same `Idempotency-Key` and body answers `202` again with the header `Idempotent-Replayed: true`. With `"preview": true` the answer is `200` with the expanded plans, and nothing is stored.',
        curl=(
            "KEY=$(uuidgen)  # reuse on retries\n"
            'curl -sS -X POST "$MKTFM_API_URL/v1/analyses" \\\n'
            '  -H "Authorization: Bearer $MKTFM_API_KEY" \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -H "Idempotency-Key: $KEY" \\\n' + curl_data(PLAN_BODY)
        ),
        python=CLIENT + "\nfrom pymc_mktfm.client import curve, scenario\n\n"
        "# dataset: the Dataset from client.datasets.create (see Upload a dataset)\n"
        "# plan: a DataFrame for the weeks after the history, from 2026-11-09:\n"
        "#       a week column, spend for tv and search, and the holiday control\n"
        "analysis = client.analyze(\n"
        "    dataset, plan,\n"
        '    scenarios=[scenario("reduce_20", {"tv": 0.8, "search": 0.8})],\n'
        '    curves=[curve("tv", [0, 0.5, 1.5, 2])],\n'
        '    decision_weeks=(1, 2), name="Spend scenarios", seed=7,\n'
        ")",
        extra_blocks=(("Backtest body", "json", BACKTEST_BODY),),
    ),
    ("GET", "/v1/analyses"): Detail(
        slug="analyses-list",
        title="List analyses",
        summary="List your analyses.",
        lead="List your analyses, or only those of one dataset.",
        response=LIST_ANALYSES_RECORD,
        curl='curl -sS "$MKTFM_API_URL/v1/analyses?dataset_id=ds_123" \\\n  -H "Authorization: Bearer $MKTFM_API_KEY"',
        python=CLIENT + '\nclient.list_analyses("ds_123")',
    ),
    ("GET", "/v1/analyses/{id}"): Detail(
        slug="analyses-get",
        title="Get an analysis",
        summary="Read the status and run counts.",
        lead="Read the status of an analysis. Poll this route every 3 seconds until `status` is `done`, "
        "`failed` or `deleted`.",
        notes=(
            (
                "Status",
                "`status` is one of `queued`, `running`, `done`, `failed` and `deleted`. The first "
                "run after idle "
                "can take about 20 seconds more. The result is ready only when every run has "
                "ended. A `done` "
                "analysis can still have failed scenario or curve plans: their rows have `status` "
                "`failed` and "
                "no numbers. `failed` means the baseline failed. `runs_cancelled` counts runs that "
                "never "
                "started because the baseline run failed. A scenario added on the site sets a "
                "`done` analysis "
                "back to `running` until it ends.",
            ),
        ),
        response=STATUS_RECORD,
        response_text="`url` is `null` when the service has no site address. `runs_cancelled` counts the runs that never started because the baseline run failed.",
        curl=CURL_GET.format(path="/v1/analyses/an_123"),
        python=CLIENT + '\nclient.get("an_123").status()',
    ),
    ("GET", "/v1/analyses/{id}/result"): Detail(
        slug="analyses-result",
        title="Get the result",
        summary="Read the table of plans and components.",
        lead="Read the result of an analysis. Until every run ends, the call answers `409 not_ready`: keep polling.",
        notes=(
            (
                "The table",
                "`table` has one row for each plan and component. Columns: `scenario_id`, `kind` "
                "(`baseline`, "
                "`scenario`, `curve`), `base_scenario_id`, `varied_channel`, `level`, "
                "`multipliers`, `run_id`, "
                "`status` (`ok`, `failed`), `spend_total`, `component`, `mean`, `p05`, `p95`, "
                "`window_mean`, "
                "`mean_difference`, `simulation_se`, `sensitivity`, `warnings`, `draws_url` and "
                "`draws_url_expires_at`. Components are `outcome`, `baseline`, each channel, each "
                "control, "
                "`latent` and active `indirect:*` rows.\n\n"
                "`mean`, `p05` and `p95` are totals over the whole forecast. `window_mean` is also "
                "a total, not a "
                "weekly mean: the mean over draws of the total over the window. `mean_difference` "
                "is a paired "
                "mean difference from the base plan. There is no interval for a difference between "
                "scenarios. "
                "Quote `labels` when you share a result.",
            ),
            (
                "Backtest fields",
                "A backtest's result also has `context` (`start`, `end`, `query_end`, `channels`, "
                "`controls`), "
                '`fit` and `response_curves`. `fit` is `{"context": ..., "query": ...}`, one for '
                "the input weeks "
                "and one for the predicted weeks. Each has `dates`, `actual`, `mean`, `p05`, "
                "`p95`, `r2` and "
                '`mape` (`{"value", "sd"}`; `r2` is `null` when the outcome never changes), `crps` '
                "and "
                '`shares`. `response_curves` is `[{"channel", "spend", "points": [{"level", '
                '"value", '
                '"p05", "p95"}]}]`, where `spend` is the baseline spend of the channel in the '
                "predicted weeks. "
                "Numbers are `null` where a level was not asked or its run failed. A plan's result "
                "has `null` in "
                "`context`, `fit` and `response_curves`. Every result has `input_usage`: the input "
                "token counts "
                "of the model for the baseline run, or `null` before it is done.",
            ),
            (
                "Draws files",
                "Each plan's draws file is netCDF4 (float32, schema `river-draws/1`). Open it with "
                '`xarray.open_dataset(path, engine="h5netcdf")`. A signed `draws_url` lasts about '
                "an hour. "
                "If its download answers `403`, fetch the result again for new links. Never send "
                "the API key "
                "to a `draws_url`. With the same seed, draw `i` of two plans is numerically "
                "coupled, so "
                "`sd(d_i) / sqrt(draws)` is the standard error of the mean difference. It is still "
                "not a joint "
                "posterior sample.",
            ),
        ),
        response=RESULT_RECORD,
        response_text="The sample is shortened. Request `?compact=true` for the compact table: one row of numbers for each plan and component, and `plans` with what the rows of each plan share. The default stays the full table.",
        curl=CURL_GET.format(path="/v1/analyses/an_123/result"),
        python=CLIENT + '\nanalysis = client.get("an_123")\nanalysis.table',
    ),
    ("GET", "/v1/analyses/{id}/weekly"): Detail(
        slug="analyses-weekly",
        title="Get weekly rows",
        summary="Read weekly rows for one plan.",
        lead="Read the weekly rows of one plan of an analysis.",
        response='{\n  "scenario_id": "baseline",\n  "rows": [\n'
        '    {"date": "2026-11-09", "component": "outcome", "mean": 1043.2, "p05": 981.4, "p95": 1107.9}\n  ]\n}',
        curl='curl -sS "$MKTFM_API_URL/v1/analyses/an_123/weekly?scenario_id=baseline" \\\n'
        '  -H "Authorization: Bearer $MKTFM_API_KEY"',
        python=CLIENT + '\nclient.get("an_123").weekly("baseline")',
    ),
    ("GET", "/v1/analyses/{id}/export"): Detail(
        slug="analyses-export",
        title="Export an analysis",
        summary="Save the whole analysis as one JSON file.",
        lead="Save the whole analysis as one JSON file. Until every run ends, the call answers `409 not_ready`.",
        response=EXPORT_RECORD,
        response_text="The sample is shortened. `?compact=true` works as for the result. `request` is the request as the service stored it; its keys are not a stable contract.",
        curl=CURL_GET.format(path="/v1/analyses/an_123/export") + " \\\n  -o analysis.json",
        python=CLIENT + '\nclient.get("an_123").export("analysis.json")',
    ),
    ("DELETE", "/v1/analyses/{id}"): Detail(
        slug="analyses-delete",
        title="Delete an analysis",
        summary="Cancel unstarted runs and delete the analysis.",
        lead="Cancel the runs that have not started and delete the analysis.",
        response_text="The service accepted the delete. A `410` after the delete is expected.",
        curl=CURL_DELETE.format(path="/v1/analyses/an_123"),
        python=CLIENT + '\nclient.get("an_123").delete()',
    ),
}


# The Errors page: one row for each status the service answers, as in http.md "Errors and retries".
STATUS_ROWS = (
    ("401", "The key is bad. Check the key."),
    ("404", "Wrong id, or the id of another user."),
    (
        "400 `bad_request`, 408 `timeout`",
        "The body did not arrive in full, or too slowly. Repeat the call with the same key.",
    ),
    ("409 `idempotency_conflict`", "The key was used with another body. Use a new key."),
    ("409 `plan_failed`", "That plan has no result."),
    (
        "409 `not_ready`",
        "The analysis has not ended. Keep polling. This body also has a top-level `status`, the status of the analysis.",
    ),
    (
        "410",
        "The resource was deleted or expired. Do not retry. A repeat of a create call whose "
        "earlier analysis "
        "was deleted also answers `410`. Only a new `Idempotency-Key` starts new work.",
    ),
    ("413", "The body is too large. Send fewer weeks, scenarios or curves in one call."),
    (
        "422",
        "The service refused the request, and nothing started. Read `error.code`, `error.field` "
        "and "
        "`error.message`, then fix the request. See Refusals. Do not loop on a `422`.",
    ),
    (
        "429, 502, 503, 504",
        "Wait `Retry-After` seconds, or back off from 2 seconds with a cap of 60, and repeat with "
        "the same key. "
        "`rate_limited` waits 60 seconds. `overloaded_user_analyses` means 20 unfinished analyses "
        "and "
        "`overloaded_user_runs` means 200 unfinished runs: wait, or delete one. `503 unavailable` "
        "and `503 busy` "
        "mean the service is busy. `503 maintenance` means the service takes no new analysis for "
        "now; a preview "
        "and a repeat of an earlier create call still work.",
    ),
)

ERROR_BODY = (
    '{\n  "error": {\n    "code": "...",\n    "message": "...",\n    "field": "...",\n    '
    '"request_id": "req_..."\n  }\n}'
)
REFUSAL_BODY = (
    '{\n  "error": {\n    "code": "steps_removed",\n    "message": "...",\n    "field": '
    '"scenarios[0].steps",\n    "request_id": "req_..."\n  }\n}'
)
ENV_BLOCK = 'export MKTFM_API_URL="<your service URL>"\nexport MKTFM_API_KEY="<your key>"'

#: The ``failure_code`` of a plan that has no result, and what to do.
FAILURE_ROWS = (
    ("worker_stopped", "The service stopped the run. Send the analysis again."),
    ("plan_outside_limits", "The plan is outside what the model accepts. Change the plan."),
    ("data_unusable", "The model refused the input values. Check the history and plan values."),
    ("cancelled", "The run was cancelled. Start it again if you still want it."),
    (
        "unknown",
        "The service gives no reason. Send it again, and quote the `request_id` if it fails again.",
    ),
)
