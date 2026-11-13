# ruff: noqa: E501  (page copy: long prose lines are easier to review than split strings)
"""Page copy for the HTTP API reference.

``http_pages.py`` reads the route list from ``skills/mfm/reference/http.md`` and fails if
this module and that file disagree about which routes exist. The wording here restates
facts of that file for a developer reader. Response values in the samples are illustrative.
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
    """Everything a route page shows. Empty fields leave their section out."""

    slug: str
    title: str
    summary: str
    lead: str
    headers: tuple[Param, ...] = ()
    inputs: tuple[Param, ...] = ()
    inputs_title: str = "Parameters"
    notes: tuple[tuple[str, str], ...] = ()
    status: str = ""
    response: str = ""
    response_text: str = ""
    errors: tuple[str, ...] = ()
    curl: str = ""
    python: str = ""
    extra_blocks: tuple[tuple[str, str, str], ...] = field(default=())


AUTH = Param(
    "Authorization",
    "`Bearer $MFM_API_KEY`. See [Authentication](authentication.qmd).",
    required=True,
)
JSON_TYPE = Param("Content-Type", "`application/json`. The body is JSON only.", required=True)
IDEMPOTENCY = Param(
    "Idempotency-Key",
    "A fresh UUID, 1 to 200 characters from `A-Z a-z 0-9 . _ : -`. Repeat a call with the same key "
    "and body and the work starts once.",
    required=True,
)
ANALYSIS_IDEMPOTENCY = Param(
    "Idempotency-Key",
    'A fresh UUID, as for a dataset. A preview (`"preview": true`) does not need it.',
)

ID_PARAM = Param("id", "The id from the create call.", "path")

# Error rows shared by many routes: (status and code, meaning). Keys are used in Detail.errors.
ERRORS = {
    "401": ("401", "The key is bad."),
    "404": ("404", "Wrong id, or the id of another user."),
    "409_conflict": (
        "409 `idempotency_conflict`",
        "The key was used with another body. Use a new key.",
    ),
    "409_not_ready": ("409 `not_ready`", "The analysis has not ended. Keep polling."),
    "410": ("410", "The resource was deleted or expired. Do not retry."),
    "413": ("413", "The body is too large. Send fewer weeks, scenarios or curves in one call."),
    "422": (
        "422",
        "The service refused the request before any run, and nothing started. "
        "Read `error.code`, `error.field` and `error.message`. See [Errors](errors.qmd).",
    ),
    "429": ("429", "Wait `Retry-After` seconds, then repeat with the same key."),
}

DATASET_FIELDS = (
    "`dataset_id`, `content_hash`, `created_at`, `last_used_at`, `analyses` (the number of live analyses), "
    "`name`, `expires_at`, `summary`, `history_ranges`, `warnings` and `warning_details`."
)

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
  "status": "running",
  "runs_total": 7, "runs_done": 3, "runs_failed": 0, "runs_cancelled": 0,
  "estimate_seconds": 40,
  "name": "Spend scenarios", "url": null,
  "expires_at": "2026-11-16T09:30:00Z",
  "seed": 7, "draws": 1000, "explain_history": false, "context": null,
  "dataset_id": "ds_123", "created_at": "2026-11-09T09:40:00Z",
  "warning_details": []
}"""

RESULT_RECORD = """{
  "table": [
    {"scenario_id": "baseline", "kind": "baseline", "component": "outcome", "status": "ok",
     "mean": 51234.5, "p05": 48800.1, "p95": 53710.9, "draws_url": "<signed link>"}
  ],
  "labels": {"outcome": "sales"},
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
     "multipliers": null, "status": "ok", "failure_code": null,
     "inputs": "...", "warnings": [], "warning_details": []}
  ]
}"""

LIST_ANALYSES_RECORD = """{
  "analyses": [
    {"analysis_id": "an_123", "dataset_id": "ds_123", "name": "Spend scenarios", "url": null,
     "status": "done", "runs_total": 7, "runs_done": 7, "runs_failed": 0,
     "created_at": "2026-11-09T09:40:00Z", "expires_at": "2026-11-16T09:30:00Z"}
  ]
}"""

CURL_GET = 'curl -sS "$MFM_API_URL{path}" \\\n  -H "Authorization: Bearer $MFM_API_KEY"'
CURL_DELETE = (
    'curl -sS -X DELETE "$MFM_API_URL{path}" \\\n  -H "Authorization: Bearer $MFM_API_KEY"'
)
CLIENT = (
    "from pymc_mfm.client import Client\n\nclient = Client()  # reads MFM_API_URL and MFM_API_KEY\n"
)

DETAILS: dict[tuple[str, str], Detail] = {
    ("POST", "/v1/datasets"): Detail(
        slug="datasets-create",
        title="Upload a dataset",
        summary="Store one weekly history.",
        lead="Store a weekly history of the outcome, the spend of each channel and the controls. "
        "Later calls refer to it by `dataset_id`.",
        headers=(AUTH, JSON_TYPE, IDEMPOTENCY),
        inputs_title="Body",
        inputs=(
            Param("name", "A label for the dataset.", "string"),
            Param("dates", "Weekly, contiguous ISO dates.", "array of dates", True),
            Param("outcome", "One value for each date.", "array of numbers", True),
            Param("spend", "One array of spend for each channel.", "object", True),
            Param("controls", "One array for each control.", "object"),
            Param("outcome_name", "A label for the site, 1 to 60 characters.", "string"),
            Param("currency", "Three uppercase letters, ISO 4217 shape, such as `USD`.", "string"),
        ),
        notes=(
            (
                "Limits",
                "The body is at most 5 MB. `outcome_name` and `currency` are not part of the content hash. "
                "The summary returns them as `summary.outcome` and `summary.currency`. "
                "The samples show three weeks to stay short.",
            ),
            (
                "Compare spend with the history",
                "`history_ranges` gives, for each channel and control, the weekly `p05`, `p95` and `max`, "
                "`zero_share` and `all_zero`. Spend outside the 5th to 95th percentile of the history gets a "
                "warning, and the model has little data there. The percentiles are warning thresholds, "
                "not the edge of the observed data.",
            ),
        ),
        status="201 Created",
        response=DATASET_RECORD,
        response_text="The dataset record, with " + DATASET_FIELDS,
        errors=("401", "409_conflict", "413", "422", "429"),
        curl=(
            "KEY=$(uuidgen)  # keep this key for every retry of this upload\n"
            'curl -sS -X POST "$MFM_API_URL/v1/datasets" \\\n'
            '  -H "Authorization: Bearer $MFM_API_KEY" \\\n'
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
        headers=(AUTH,),
        notes=(
            (
                "Reuse a stored dataset",
                "Match on `content_hash` only. Never match by name or by number of weeks: one changed value "
                "gives a different hash. To compute the hash, take `dates`, `outcome`, `spend` and `controls` "
                "(not `name`). Write each date as `YYYY-MM-DD`. Convert every number to a float: `-0.0` becomes "
                "`0.0`, and NaN and infinity are errors. Sort the rows by date and reorder every series the "
                'same way. Build `{"v":1,"dates":[...],"outcome":[...],"spend":{...},"controls":{...}}` '
                'and encode it with `json.dumps(obj, sort_keys=True, separators=(",", ":"), '
                'allow_nan=False, ensure_ascii=False)` as UTF-8. The hash is `"mfm-dataset-v1:"` plus the '
                "SHA-256 hex digest. The Python client does this in `pymc_mfm.client.dataset_content_hash`.",
            ),
        ),
        status="200 OK",
        response='{\n  "datasets": [\n'
        + "    ...one dataset record each, as in Upload a dataset...\n  ]\n}",
        response_text='`{"datasets": [...]}`. Each record has ' + DATASET_FIELDS,
        errors=("401", "429"),
        curl=CURL_GET.format(path="/v1/datasets"),
        python=CLIENT + "\nclient.datasets.list()",
    ),
    ("GET", "/v1/datasets/{id}"): Detail(
        slug="datasets-get",
        title="Get a dataset",
        summary="Read one dataset summary.",
        lead="Read the summary of one dataset. Read it again when you need `expires_at`: using a dataset moves it later.",
        headers=(AUTH,),
        inputs=(ID_PARAM,),
        notes=(
            (
                "Retention",
                "The service keeps a dataset 7 days after last use, and at most 30 days after upload. "
                "Using a dataset, for example by starting an analysis on it, moves `expires_at` later. "
                "Do not treat it as a fixed deadline. An analysis has the `expires_at` of its dataset.",
            ),
        ),
        status="200 OK",
        response=DATASET_RECORD,
        response_text="The dataset record, with " + DATASET_FIELDS,
        errors=("401", "404", "410"),
        curl=CURL_GET.format(path="/v1/datasets/ds_123"),
        python=CLIENT + '\nclient.datasets.get("ds_123")',
    ),
    ("DELETE", "/v1/datasets/{id}"): Detail(
        slug="datasets-delete",
        title="Delete a dataset",
        summary="Delete a dataset and all its analyses.",
        lead="Delete a dataset. This also deletes all its analyses.",
        headers=(AUTH,),
        inputs=(ID_PARAM,),
        status="202 Accepted",
        response_text="The service accepted the delete. Later calls with the id answer `410`.",
        errors=("401", "404", "410"),
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
        headers=(AUTH, JSON_TYPE, ANALYSIS_IDEMPOTENCY),
        inputs_title="Body",
        inputs=(
            Param("dataset_id", "The dataset to analyse.", "string", True),
            Param(
                "plan",
                "The baseline for the weeks after the history. Use `plan` or `context`, not both.",
                "object",
            ),
            Param("context", "`start` and `end` of the input weeks of a backtest.", "object"),
            Param(
                "query",
                "`end`, the last predicted week of a backtest. It comes with `context`.",
                "object",
            ),
            Param(
                "channels",
                "Channels of the dataset for a backtest. Default: all.",
                "array of strings",
            ),
            Param(
                "controls",
                "Controls of the dataset for a backtest. Default: all.",
                "array of strings",
            ),
            Param(
                "scenarios",
                "Spend changes to compare with the baseline. There are no default scenarios.",
                "array",
            ),
            Param("curves", "Response curves, one object for each channel.", "array"),
            Param(
                "decision_weeks",
                "`[i, j]`, 1-based and inclusive. Default: the whole plan.",
                "array of integers",
            ),
            Param(
                "explain_history",
                "Default `true` for a backtest and `false` for a plan.",
                "boolean",
            ),
            Param("draws", "1 to 2,000. Default 1,000. Use 200 to 500 for exploration.", "integer"),
            Param("seed", "0 to 2147483647. The server picks one if you leave it out.", "integer"),
            Param("name", "1 to 120 characters, trimmed. Other values answer `422`.", "string"),
            Param(
                "preview",
                "`true` to return the expanded plans without storing or running anything.",
                "boolean",
            ),
        ),
        notes=(
            (
                "Plan and backtest",
                "A **plan** gives the weeks after the history. It is the baseline, with id `baseline` (reserved). "
                "It holds spend for every channel and a value for every control, for every forecast week. "
                "Its dates start the week after the history.\n\n"
                "A **backtest** uses weeks the dataset already holds. The model reads the input weeks "
                "`context.start` to `context.end` and predicts the weeks after them, up to `query.end`. "
                "It does not see the outcome of the predicted weeks, so the result can score the forecast "
                "against it. The baseline is the dataset's own spend and controls in the predicted weeks. "
                "Every date is a week of the dataset, and `query.end` is after `context.end`. "
                "Do not send `plan` with them: the answer is `422 plan_with_context`.\n\n"
                "The window is the only part of the forecast that scenarios and curves change: the predicted "
                "weeks of a backtest, or `decision_weeks` of a plan.",
            ),
            (
                "Scenarios",
                'A scenario is `{"id", "multipliers"}`. The `id` is unique, 1 to 64 characters from '
                "`A-Za-z0-9_.-`. `baseline` and the `curve:` prefix are reserved. `multipliers` maps a channel "
                "to a number from 0 to 5 (rounded to 4 decimals), which multiplies its spend in the window, or "
                'to `{"amount": X}` (X is 0 or more, rounded to cents), which sets its total spend in the '
                "window to X. An amount keeps the weekly pattern. On a channel with no spend in the window, it "
                "puts X / W in each of the W window weeks. A channel left out keeps its spend. A scenario "
                "keeps every control.",
            ),
            (
                "Curves",
                'A curve is `{"channel", "levels", "base"?}`. Each level multiplies the spend of that channel '
                "in the window. Level 1 equals the base and does not run again. Level ids are "
                "`curve:<channel>:<base>:<level>`. For a first look, use levels 0 to 2 in steps of 0.25. "
                "An analysis has at most 200 different plans, and equal plans run once.",
            ),
        ),
        status="202 Accepted",
        response='{\n  "analysis_id": "an_123",\n  "name": "Spend scenarios",\n  "url": null,\n'
        '  "runs_total": 7,\n  "estimate_seconds": 40,\n  "expires_at": "2026-11-16T09:30:00Z"\n}',
        response_text="`analysis_id`, `name`, `url`, `runs_total`, `estimate_seconds` and `expires_at`. "
        '`url` is `null` when the service has no site address. With `"preview": true` the answer is `200` '
        "with the expanded `plans`, `runs_total`, `estimate_seconds`, `assumptions` and `warnings`.",
        errors=("401", "409_conflict", "413", "422", "429"),
        curl=(
            "KEY=$(uuidgen)  # keep this key for every retry of this analysis\n"
            'curl -sS -X POST "$MFM_API_URL/v1/analyses" \\\n'
            '  -H "Authorization: Bearer $MFM_API_KEY" \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -H "Idempotency-Key: $KEY" \\\n' + curl_data(PLAN_BODY)
        ),
        python=CLIENT + "\nfrom pymc_mfm.client import curve, scenario\n\n"
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
        headers=(AUTH,),
        inputs_title="Query",
        inputs=(Param("dataset_id", "Return only the analyses of this dataset.", "string"),),
        status="200 OK",
        response=LIST_ANALYSES_RECORD,
        response_text='`{"analyses": [...]}`. Each item has `analysis_id`, `dataset_id`, `name`, `url`, `status`, '
        "`runs_total`, `runs_done`, `runs_failed`, `created_at` and `expires_at`.",
        errors=("401", "429"),
        curl='curl -sS "$MFM_API_URL/v1/analyses?dataset_id=ds_123" \\\n  -H "Authorization: Bearer $MFM_API_KEY"',
        python=CLIENT + '\nclient.list_analyses("ds_123")',
    ),
    ("GET", "/v1/analyses/{id}"): Detail(
        slug="analyses-get",
        title="Get an analysis",
        summary="Read the status and run counts.",
        lead="Read the status of an analysis. Poll this route every 3 seconds until `status` is `done`, "
        "`failed` or `deleted`.",
        headers=(AUTH,),
        inputs=(ID_PARAM,),
        notes=(
            (
                "Status",
                "`status` is one of `queued`, `running`, `done`, `failed` and `deleted`. The first run after idle "
                "can take about 20 seconds more. The result is ready only when every run has ended. A `done` "
                "analysis can still have failed scenario or curve plans: their rows have `status` `failed` and "
                "no numbers. `failed` means the baseline failed. `runs_cancelled` counts runs that never "
                "started because the baseline run failed. A scenario added on the site sets a `done` analysis "
                "back to `running` until it ends.",
            ),
        ),
        status="200 OK",
        response=STATUS_RECORD,
        response_text="`status`, the run counts `runs_total`, `runs_done`, `runs_failed` and `runs_cancelled`, "
        "`estimate_seconds`, and `name`, `url`, `expires_at`, `seed`, `draws`, `explain_history`, `context` "
        "(`null` for a plan), `dataset_id`, `created_at` and `warning_details`.",
        errors=("401", "404", "410"),
        curl=CURL_GET.format(path="/v1/analyses/an_123"),
        python=CLIENT + '\nclient.get("an_123").status()',
    ),
    ("GET", "/v1/analyses/{id}/result"): Detail(
        slug="analyses-result",
        title="Get the result",
        summary="Read the table of plans and components.",
        lead="Read the result of an analysis. Until every run ends, the call answers `409 not_ready`: keep polling.",
        headers=(AUTH,),
        inputs=(ID_PARAM,),
        notes=(
            (
                "The table",
                "`table` has one row for each plan and component. Columns: `scenario_id`, `kind` (`baseline`, "
                "`scenario`, `curve`), `base_scenario_id`, `varied_channel`, `level`, `multipliers`, `run_id`, "
                "`status` (`ok`, `failed`), `spend_total`, `component`, `mean`, `p05`, `p95`, `window_mean`, "
                "`mean_difference`, `simulation_se`, `sensitivity`, `warnings`, `draws_url` and "
                "`draws_url_expires_at`. Components are `outcome`, `baseline`, each channel, each control, "
                "`latent` and active `indirect:*` rows.\n\n"
                "`mean`, `p05` and `p95` are totals over the whole forecast. `window_mean` is also a total, not a "
                "weekly mean: the mean over draws of the total over the window. `mean_difference` is a paired "
                "mean difference from the base plan. There is no interval for a difference between scenarios. "
                "Quote `labels` when you share a result.",
            ),
            (
                "Backtest fields",
                "A backtest's result also has `context` (`start`, `end`, `query_end`, `channels`, `controls`), "
                '`fit` and `response_curves`. `fit` is `{"context": ..., "query": ...}`, one for the input weeks '
                "and one for the predicted weeks. Each has `dates`, `actual`, `mean`, `p05`, `p95`, `r2` and "
                '`mape` (`{"value", "sd"}`; `r2` is `null` when the outcome never changes), `crps` and '
                '`shares`. `response_curves` is `[{"channel", "spend", "points": [{"level", "value", '
                '"p05", "p95"}]}]`, where `spend` is the baseline spend of the channel in the predicted weeks. '
                "Numbers are `null` where a level was not asked or its run failed. A plan's result has `null` in "
                "`context`, `fit` and `response_curves`. Every result has `input_usage`: the input token counts "
                "of the model for the baseline run, or `null` before it is done.",
            ),
            (
                "Draws files",
                "Each plan's draws file is netCDF4 (float32, schema `river-draws/1`). Open it with "
                '`xarray.open_dataset(path, engine="h5netcdf")`. A signed `draws_url` lasts about an hour. '
                "If its download answers `403`, fetch the result again for new links. Never send the API key "
                "to a `draws_url`. With the same seed, draw `i` of two plans is numerically coupled, so "
                "`sd(d_i) / sqrt(draws)` is the standard error of the mean difference. It is still not a joint "
                "posterior sample.",
            ),
        ),
        status="200 OK",
        response=RESULT_RECORD,
        response_text="`table`, `labels`, `history`, `assumptions`, `meta`, `context`, `fit`, `response_curves` and "
        "`input_usage`. The sample is shortened.",
        errors=("401", "404", "409_not_ready", "410"),
        curl=CURL_GET.format(path="/v1/analyses/an_123/result"),
        python=CLIENT + '\nanalysis = client.get("an_123")\nanalysis.table',
    ),
    ("GET", "/v1/analyses/{id}/weekly"): Detail(
        slug="analyses-weekly",
        title="Get weekly rows",
        summary="Read weekly rows for one plan.",
        lead="Read the weekly rows of one plan of an analysis.",
        headers=(AUTH,),
        inputs_title="Parameters",
        inputs=(
            ID_PARAM,
            Param("scenario_id", "The plan to read.", "query"),
        ),
        status="200 OK",
        response='{\n  "scenario_id": "baseline",\n  "rows": [\n'
        '    {"date": "2026-11-09", "component": "outcome", "mean": 1043.2, "p05": 981.4, "p95": 1107.9}\n  ]\n}',
        response_text="`scenario_id` and `rows`, each with `date`, `component`, `mean`, `p05` and `p95`.",
        errors=("401", "404", "410"),
        curl='curl -sS "$MFM_API_URL/v1/analyses/an_123/weekly?scenario_id=baseline" \\\n'
        '  -H "Authorization: Bearer $MFM_API_KEY"',
        python=CLIENT + '\nclient.get("an_123").weekly("baseline")',
    ),
    ("GET", "/v1/analyses/{id}/export"): Detail(
        slug="analyses-export",
        title="Export an analysis",
        summary="Save the whole analysis as one JSON file.",
        lead="Save the whole analysis as one JSON file. Until every run ends, the call answers `409 not_ready`.",
        headers=(AUTH,),
        inputs=(ID_PARAM,),
        status="200 OK",
        response=EXPORT_RECORD,
        response_text="`format` (`mfm-analysis-export-v1`), `analysis_id`, `name`, `created_at`, `expires_at`, "
        "`dataset`, `request`, `units`, `results`, `seed`, `draws`, `model_revision`, `library_revision`, "
        "`exported_at` and `scenarios`. Each scenario has `scenario_id`, `kind`, `base_scenario_id`, "
        "`multipliers`, `status`, `failure_code`, `inputs`, `warnings` and `warning_details`. The sample is shortened.",
        errors=("401", "404", "409_not_ready", "410"),
        curl=CURL_GET.format(path="/v1/analyses/an_123/export") + " \\\n  -o analysis.json",
        python=CLIENT + '\nclient.get("an_123").export("analysis.json")',
    ),
    ("DELETE", "/v1/analyses/{id}"): Detail(
        slug="analyses-delete",
        title="Delete an analysis",
        summary="Cancel unstarted runs and delete the analysis.",
        lead="Cancel the runs that have not started and delete the analysis.",
        headers=(AUTH,),
        inputs=(ID_PARAM,),
        status="202 Accepted",
        response_text="The service accepted the delete. A `410` after the delete is expected.",
        errors=("401", "404", "410"),
        curl=CURL_DELETE.format(path="/v1/analyses/an_123"),
        python=CLIENT + '\nclient.get("an_123").delete()',
    ),
}


# The Errors page: one row for each status the service answers, as in http.md "Errors and retries".
STATUS_ROWS = (
    ("401", "The key is bad. Check the key."),
    ("404", "Wrong id, or the id of another user."),
    ("409 `idempotency_conflict`", "The key was used with another body. Use a new key."),
    ("409 `plan_failed`", "That plan has no result."),
    ("409 `not_ready`", "The analysis has not ended. Keep polling."),
    (
        "410",
        "The resource was deleted or expired. Do not retry. A repeat of a create call whose earlier analysis "
        "was deleted also answers `410`. Only a new `Idempotency-Key` starts new work.",
    ),
    ("413", "The body is too large. Send fewer weeks, scenarios or curves in one call."),
    (
        "422",
        "The service refused the request, and nothing started. Read `error.code`, `error.field` and "
        "`error.message`, then fix the request. See Refusals. Do not loop on a `422`.",
    ),
    (
        "429, 502, 503, 504",
        "Wait `Retry-After` seconds, or back off from 2 seconds with a cap of 60, and repeat with the same key. "
        "`rate_limited` waits 60 seconds. `overloaded_user_analyses` means 20 unfinished analyses and "
        "`overloaded_user_runs` means 200 unfinished runs: wait, or delete one.",
    ),
)

ERROR_BODY = (
    '{\n  "error": {\n    "code": "...",\n    "message": "...",\n    "field": "..."\n  }\n}'
)
REFUSAL_BODY = '{\n  "error": {\n    "code": "steps_removed",\n    "message": "...",\n    "field": "scenarios[0].steps"\n  }\n}'
ENV_BLOCK = 'export MFM_API_URL="<from the Console>"\nexport MFM_API_KEY="<your key>"'
