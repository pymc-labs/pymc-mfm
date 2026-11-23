# MFM API over HTTP

Use these calls when you are not using the Python client. The reading rules, limits and
deletion rules in `SKILL.md` apply here too. To use the Python client instead, follow the
setup in `SKILL.md` on Python 3.13 or later.

The base URL is in `MFM_API_URL` and the key is in `MFM_API_KEY`. If one is missing, stop and ask the
user. Never print the key and never write the URL or the key into a file.

Every request carries `Authorization: Bearer $MFM_API_KEY`. Create calls send
`Content-Type: application/json`. The body is JSON only. Other users' ids return `404`. Expired or
deleted resources return `410`.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/datasets` | Upload a history. `201` with `dataset_id`, `content_hash`, `created_at`, `last_used_at`, `analyses`, `name`, `expires_at`, `summary`, `history_ranges`, `warnings`, `warning_details`. |
| `GET /v1/datasets` | List your datasets, newest first: an object with `datasets` and, when you page, `next_cursor`. Each dataset has `dataset_id`, `content_hash`, `created_at`, `last_used_at`, `analyses`, `name`, `expires_at`, `summary`, `history_ranges`, `warnings` and `warning_details`. |
| `GET /v1/datasets/{id}` | The summary, with `dataset_id`, `content_hash`, `created_at`, `last_used_at`, `analyses`, `name`, `expires_at`, `summary`, `history_ranges`, `warnings` and `warning_details`. |
| `DELETE /v1/datasets/{id}` | Delete the dataset and all its analyses. `202` with `deletion`. |
| `POST /v1/analyses` | Start an analysis. `202` with `analysis_id`, `dataset_id`, `name`, `url`, `runs_total`, `estimate_seconds`, `expires_at`, `warnings` and `warning_details`. With `"preview": true`: `200` with the expanded `plans`, `runs_total`, `estimate_seconds`, `assumptions`, `warnings` and `warning_details`. Nothing is stored or run. |
| `GET /v1/analyses/{id}` | `analysis_id`, `dataset_id`, `status`, the run counts `runs_total`, `runs_done`, `runs_failed` and `runs_cancelled`, `estimate_seconds`, and `name`, `url`, `expires_at`, `seed`, `draws`, `explain_history`, `context` (`null` for a plan), `warnings` and `warning_details`. |
| `GET /v1/analyses/{id}/result` | The result: `table`, `labels`, `history`, `assumptions`, `meta`, `context`, `fit`, `response_curves` and `input_usage`; with `?compact=true` also `plans`. `409 not_ready` until the analysis ends. |
| `GET /v1/analyses/{id}/weekly?scenario_id=` | Weekly rows for one plan: `scenario_id` and `rows`, each with `date`, `component`, `mean`, `p05` and `p95`. `409 not_ready` until the analysis ends. `409 plan_failed` when that plan has no result. `422 missing_field` without `scenario_id`. |
| `GET /v1/analyses/{id}/export` | The whole analysis as one JSON file (`mfm-analysis-export-v1`) with `format`, `analysis_id`, `name`, `created_at`, `expires_at`, `dataset`, `request`, `units`, `results`, `seed`, `draws`, `model_revision`, `library_revision`, `exported_at` and `scenarios`, each with `scenario_id`, `kind`, `base_scenario_id`, `multipliers`, `status`, `failure_code`, `inputs`, `warnings` and `warning_details`. `409 not_ready` until every run ends. |
| `DELETE /v1/analyses/{id}` | Cancel unstarted runs and delete the analysis. `202` with `deletion`. |
| `GET /v1/analyses?dataset_id=` | List your analyses, newest first: an object with `analyses` and, when you page, `next_cursor`. Each analysis has `analysis_id`, `dataset_id`, `name`, `url`, `status`, `runs_total`, `runs_done`, `runs_failed`, `created_at` and `expires_at`. |

Do not call other endpoints.

Optional query parameters change what a call returns. Without them the answer is the same as before.

- `limit` (1 to 200) and `cursor` on `GET /v1/datasets` and `GET /v1/analyses`. Without both, the list is whole. With
  `limit`, the list has at most that many items and the answer adds `next_cursor`: send it as `cursor` for the next
  page. `next_cursor` is `null` on the last page. A bad `limit` is `422 invalid_limit` and a bad `cursor` is `422 invalid_cursor`.
- `compact=true` (or `1`) on `GET /v1/analyses/{id}/result` and `GET /v1/analyses/{id}/export`. The `table` has one
  row of numbers for each plan and component, and `plans` holds what the rows of each plan share (its `kind`,
  `multipliers`, `warnings`, `draws_url` and the other plan keys); a row repeats such a key only where it differs.
  The full table stays the default. Any value other than `true`, `false`, `1` or `0` is `422 invalid_input`.

Older servers may leave out `content_hash`, `created_at`, `last_used_at`, `analyses`, `name`,
`url`, `runs_cancelled`, `warnings` and `warning_details`. Treat a missing field as unknown. `url` is `null`
when the service has no site address.
`runs_cancelled` counts runs that never started because the baseline run failed.
`warning_details` lists the warnings as structured objects, next to the plain-text `warnings`.

## Retention

The service keeps a dataset 7 days after upload, then deletes it and its analyses. Using the
dataset does not extend this. Read the deadline from `expires_at`. An analysis has the `expires_at`
of its dataset. Deleting a dataset deletes
all its analyses. Report the results first, then ask before deleting.

## Reusing a dataset

Before you upload, call `GET /v1/datasets`. Reuse a stored dataset only when its `content_hash`
equals the hash of your data. Never match by name or by number of weeks: one changed value gives a
different hash. Compute the hash like this: take `dates`, `outcome`, `spend` and `controls` (not
`name`); write each date as `YYYY-MM-DD`; convert every number to a float (`-0.0` becomes `0.0`;
NaN and infinity are errors); sort the rows by date and reorder every series the same way; build
`{"v":1,"dates":[...],"outcome":[...],"spend":{...},"controls":{...}}`; encode it with
`json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)` as
UTF-8; the hash is `"mfm-dataset-v1:"` plus the SHA-256 hex digest. The Python client does this in
`pymc_mfm.client.dataset_content_hash`.

## Call order

1. `POST /v1/datasets`. Send an `Idempotency-Key` header (a fresh UUID; 1 to 200 characters from
   `A-Z a-z 0-9 . _ : -`) so a repeat after an error starts no second upload. The service does not require
   the header. Body:

   ```json
   {"name": "weekly-history", "dates": ["2024-07-01", "..."], "outcome": [1012.5, "..."],
    "spend": {"tv": ["..."], "search": ["..."]}, "controls": {"holiday": ["..."]}}
   ```

   Optional labels for the site: `outcome_name` (1 to 60 characters) and `currency` (3 uppercase
   letters, ISO 4217 shape, such as `USD`). Neither is part of the content hash; the summary
   returns them as `summary.outcome` and `summary.currency`. The Python client sends the outcome
   column name as `outcome_name`.
   Dates are weekly, contiguous ISO dates, exactly 7 days apart, and may fall on any weekday (use the same weekday every week). The body is at most 5 MB. `history_ranges` gives, per
   channel and control, the weekly `p05`, `p95` and `max`, `zero_share` and `all_zero`.
2. Check `history_ranges` against the plan you will send. The weekly `p05` and `p95` are warning
   thresholds, not the edge of the observed data. Spend outside the history's 5th to 95th percentile
   gets a warning, and the model has little data there.
3. `POST /v1/analyses` with `"preview": true` first. It stores nothing and needs no `Idempotency-Key`. Show the user
   the run count and the warnings. If `runs_total` is large, give the count and the estimate before
   you start.
4. `POST /v1/analyses` without `preview`, with a fresh `Idempotency-Key` (optional, but a repeat needs it to start no second run). The baseline comes in one
   of two forms. A **plan** gives the weeks after the history:

   ```json
   {"dataset_id": "ds_123",
    "plan": {"dates": ["2026-11-09", "... 8 weeks"],
             "spend": {"tv": [9000, "..."], "search": [6000, "..."]},
             "controls": {"holiday": ["..."]}},
    "name": "Spend scenarios", "decision_weeks": [1, 8],
    "scenarios": [
      {"id": "reduce_20", "multipliers": {"tv": 0.8, "search": 0.8}},
      {"id": "tv_to_search", "multipliers": {"tv": 0.8, "search": {"amount": 62400}}}],
    "curves": [{"channel": "tv", "levels": [0, 0.5, 1.5, 2]}],
    "draws": 1000, "seed": 7}
   ```

   A **backtest** uses weeks the dataset already holds. The model reads the input weeks
   `context.start` to `context.end` and predicts the weeks after them up to `query.end`. It does not
   see the outcome of the predicted weeks, so the result can score the forecast against it:

   ```json
   {"dataset_id": "ds_123", "context": {"start": "2024-11-11", "end": "2026-08-03"},
    "query": {"end": "2026-11-02"}, "channels": ["tv", "search"], "controls": ["holiday"],
    "scenarios": [{"id": "budget_x1.2", "multipliers": {"tv": 1.2, "search": 1.2}}],
    "curves": [{"channel": "tv", "levels": [0, 0.5, 1.5, 2]}],
    "seed": 7, "name": "Q4 backtest"}
   ```

   - `plan` is the baseline (id `baseline`, reserved): spend for every channel and a value for every
     control, for every forecast week. Its dates start the week after the history.
   - In a backtest the baseline is the dataset's own spend and controls in the predicted weeks.
     `context` and `query` come together; every date is a week of the dataset, and `query.end` is
     after `context.end`. Do not send `plan` with them (`422 plan_with_context`). `channels` and
     `controls` choose series of the dataset (default: all). `explain_history` defaults to `true` for a
     backtest and `false` for a plan.
   - The window is the only part of the forecast that scenarios and curves change: the predicted
     weeks of a backtest, or `decision_weeks` `[i, j]` of a plan (1-based, inclusive; default the whole
     plan).
   - A scenario is `{"id", "multipliers"}`. The `id` is unique (1 to 64 characters from
     `A-Za-z0-9_.-`; `baseline` and the `curve:` prefix are reserved). `multipliers` maps a channel to a
     number from 0 to 5 (rounded to 4 decimals), which multiplies its spend in the window, or to
     `{"amount": X}` (X 0 or more, rounded to cents), which sets its total spend in the window to X. An
     amount keeps the weekly pattern; on a channel with no spend in the window it puts X / W in each of
     the W window weeks. A channel left out keeps its spend. A scenario keeps every control. There are
     no default scenarios.
   - A curve is `{"channel", "levels", "base"?}`. Each level multiplies that channel's spend in the
     window. Level 1 equals the base and does not run again. Level ids are
     `curve:<channel>:<base>:<level>`. For a first look use levels 0 to 2 in steps of 0.25.
   - `name` is optional: 1 to 120 characters, trimmed; `422` otherwise.
   - `draws` is 100 to 2,000 (default 1,000). Use 200 to 500 for exploration. `seed` is 0 to 2147483647;
     a request without one uses seed 0. The same request gives the same numbers only with the same
     model revision, library revision, device class, dtype and execution settings, and results can
     differ across devices. More draws reduce sampling error. A different seed does not.
   - At most 200 different plans per analysis. Equal plans run once.
5. Poll `GET /v1/analyses/{id}` every 3 seconds until `status` is `done` or `deleted`, or `failed`
   with `runs_done + runs_failed + runs_cancelled` equal to `runs_total`.
   `status` is one of `queued`, `running`, `done`, `failed`, `deleted`. The first run after idle can take
   about 20 seconds more. The result is ready only when every run has ended. A `done` analysis can
   still have failed scenario or curve plans: their rows have `status` `failed` and no numbers.
   `failed` means the baseline failed. `409 not_ready` from the result call means: keep polling.
   A scenario added to the analysis on the site sets a `done` analysis back to `running` until it ends.
6. `GET /v1/analyses/{id}/result` for the table, and `GET /v1/analyses/{id}/weekly?scenario_id=` for one
   plan's weekly rows (`scenario_id`, `rows` with `date`, `component`, `mean`, `p05`, `p95`). Each
   `table[*].draws_url` is a signed link to that plan's draws file.
7. Give the user the numbers and the result link (`url` from the analysis record; give the
   analysis id if `url` is `null`). Then ask before deleting. `GET /v1/analyses/{id}/export` saves the
   whole analysis as JSON. `DELETE /v1/analyses/{id}` or `DELETE /v1/datasets/{id}` (this also
   deletes all its analyses) when the user agrees. A `410` after the delete is expected.

## More scenarios

There is no endpoint that adds scenarios to an analysis. To try more, send a new `POST /v1/analyses`
with the same baseline, window, `seed` and `draws` and the new scenarios. Read them from
`GET /v1/analyses/{id}`: a backtest repeats its `context` (`start`, `end`, `query_end`, `channels`,
`controls`). Then read the export (`409 not_ready` until the analysis has ended): in its `scenarios`
list, the entry with `kind` `baseline` has `inputs` with `dates` (the forecast weeks), `spend` and
`controls` (for a plan, the plan) and `decision_dates` (the window). When `decision_dates` are fewer
than `dates`, send `decision_weeks` `[i, j]`: the 1-based positions of the first and last decision
date in `dates`. This applies to both forms. Set `explain_history` to `false` and leave out curves you
already have. Compare each scenario only with the baseline of its own analysis, never with a number
from another analysis: their draws are not paired.

## Refusals

The service rejects these before any run, with `422`. Warnings are never refusals.

| Code | Cause |
| --- | --- |
| `steps_removed` (field `scenarios[i].steps`) | Scenario steps were removed. Send `multipliers`. |
| `scenario_plan_removed` | Scenario plans were removed. Send `multipliers`. |
| `invalid_scenario` | The scenario changes no spend, for example a multiplier on a channel with no spend in the window (send an amount there). |
| `unknown_name` | A scenario or curve names a channel the dataset does not have. |
| `value_out_of_range` | A multiplier outside 0 to 5, or a negative amount. |
| `plan_with_context` | A body with both `plan` and `context`. |
| `invalid_window`, `context_too_short`, `too_long` | A backtest date that is not a week of the dataset, a query that does not follow the context, a context under 2 weeks, or too many weeks in total. |
| `history_all_zero` | Spend or control values that are not zero where the history (or the context) series is all zero. |

The service also refuses negative spend and unknown series names.

## Refusal codes

Every code of a `422`, by where it arises. Read `error.field` for the part of the request at fault.

- Any create call: `invalid_json`, `nonfinite`, `body_too_complex`, `invalid_idempotency_key`, `unknown_field`, `missing_field`, `invalid_type`, `invalid_value`, `value_out_of_range`, `too_long`, `too_short`, `storage_quota`.
- Dataset upload: `invalid_date`, `dates_not_weekly`, `length_mismatch`, `nonpositive_outcome`, `negative_spend`, `channel_count`, `control_count`, `controls_required`, `duplicate_name`, `reserved_name`, `invalid_name`, `invalid_outcome_name`, `invalid_currency`.
- Analysis request: `invalid_observations` (from the dry run of the model), `dataset_required`, `history_with_context`, `plan_with_context`, `history_all_zero`, `dates_not_contiguous`, `name_mismatch`, `draws_out_of_range`, `seed_out_of_range`, `invalid_weeks`, `too_many_plans`, `too_many_requested_plans`.
- Scenarios and curves: `invalid_scenario`, `steps_removed`, `scenario_plan_removed`, `invalid_id`, `reserved_id`, `duplicate_id`, `unknown_name`.
- Backtest weeks and series: `invalid_window`, `context_too_short`, `series_required`, `unknown_series`, `duplicate_series`.
- Lists and result options: `invalid_limit`, `invalid_cursor`, `invalid_input`.

## The result

`table` has one row per plan and component. Columns: `scenario_id`, `kind` (`baseline`, `scenario`,
`curve`), `base_scenario_id`, `varied_channel`, `level`, `multipliers`, `run_id`, `status` (`ok`, `failed`),
`spend_total`, `component`, `mean`, `p05`, `p95`, `window_mean`, `mean_difference`, `simulation_se`,
`sensitivity`, `warnings`, `draws_url`, `draws_url_expires_at`. Components are `outcome`, `baseline`,
each channel, each control, `latent` and active `indirect:*` rows. `mean`, `p05` and `p95` are totals
over the whole forecast. `window_mean` is also a total, not a weekly mean: the mean over draws of the
total over the window. `mean_difference` is a paired mean difference from the base plan; there is no
interval for a difference between scenarios. The response also has `assumptions`, `meta` and `labels`.
Quote `labels` when you share a result. Read the numbers as `SKILL.md` says.

A backtest's result also has:

- `context`: `start`, `end`, `query_end`, `channels`, `controls`.
- `fit`: `{"context": ..., "query": ...}`, one for the input and one for the predicted weeks. Each has
  `dates`, `actual`, `mean`, `p05`, `p95`, `r2` and `mape` (`{"value", "sd"}`; `value` scores the mean
  line, `sd` is the spread over draws; `r2` is `null` when the outcome never changes), `crps` and
  `shares` (per series: `name`, `group`, and `total` and `mean` shares as `{"value", "p05", "p95"}`).
- `response_curves`: `[{"channel", "spend", "points": [{"level", "value", "p05", "p95"}]}]`, where
  `spend` is the channel's baseline spend in the predicted weeks. Numbers are `null` where a level was
  not asked or its run failed.

A plan's result has `null` in these three fields. Every result also has `input_usage`: the model's
input token counts for the baseline run, or `null` before it is done.

## Errors and retries

Error bodies look like `{"error": {"code", "message", "request_id", "field"?}}`. `request_id` is also the
`X-Request-Id` header. A `409 not_ready` body also has a top-level `status`, the status of the analysis.

- After a timeout or a connection error, repeat the same create call with the same key and body. This
  never starts the work twice.
- `408 timeout` and `400 bad_request`: the body did not arrive in full or in time. Repeat the call with the same key.
- `429 rate_limited`, `429 overloaded_user_analyses`, `429 overloaded_user_runs` and `502/503/504`: wait
  `Retry-After` seconds (or back off from 2 s, cap 60 s) and repeat with the same key. `rate_limited` waits 60 s.
  `overloaded_user_analyses` means 20 unfinished analyses and `overloaded_user_runs` 200 unfinished runs: wait or
  delete one. `503 unavailable` and `503 busy`: the service is busy; wait and repeat. `503 maintenance`: the service
  takes no new analysis for now; a preview and a repeat of an earlier create call still work.
- `413 body_too_large`: the body is too large. Send fewer weeks, scenarios or curves in one call.
- `409 idempotency_conflict`: the key was used with another body. Use a new key. `409 plan_failed`: that
  plan has no result. `409 not_ready`: the analysis has not ended; keep polling.
- `410 gone`: the resource was deleted or expired. Do not retry. A repeat of a create call whose earlier
  analysis was deleted also gets `410`; only a new `Idempotency-Key` starts new work.
- `404 not_found`: wrong id or another user's id. `404 unknown_scenario`: the analysis has no such plan.
- `401 unauthorized` on a River API request: check the River key. A signed file request sends no
  River key; report a 401 there as a file host rejection.
- `422`: read `error.code`, `error.field` and `error.message`. Fix the request (see "Refusals"). Nothing
  started. Do not loop on a `422`.
- A failed plan is not an HTTP error: the status call answers `200`, and the export carries `failure_code` for that plan; its result rows have `status` `failed` and no numbers.
  To send a failed analysis again, use a new `Idempotency-Key` (Python client: `fresh=True`): the same key returns the failed analysis.
  `worker_stopped`: the service stopped the run; send the analysis again. `plan_outside_limits`: change the plan.
  `data_unusable`: the model refused the input values; check the history and plan values. `cancelled`: the run was cancelled;
  start it again if you still want it. `unknown`: no reason is given; send it again, and quote `request_id` if it fails again.
- `500 internal_error`: the service failed. Repeat later; if it persists, quote `request_id`.

## Draws files

Each plan's draws file is netCDF4 (float32, schema `river-draws/1`). Open it with
`xarray.open_dataset(path, engine="h5netcdf")`. A signed `draws_url` lasts about an hour. If its
download answers `403`, fetch the result again for new links. Never send the API key to a `draws_url`.
With the same seed, draw `i` of two plans is numerically coupled, so `sd(d_i) / sqrt(draws)` is the
standard error of the mean difference. It is still not a joint posterior sample.
