# MFM API over HTTP

For agents without Python. The Python client in `SKILL.md` makes the same calls. The reading
rules, limits and deletion rules in `SKILL.md` apply here too.

To use the Python client instead, install it with `uv` (Python 3.13 or later, GitHub read access):
`uv venv -p 3.13` and
`uv pip install "pymc-mfm[client] @ git+https://github.com/pymc-labs/pymc-mfm.git"`.

The base URL is in `MFM_API_URL` and the key is in `MFM_API_KEY`. If one is missing, stop and ask the
user. Never print the key and never write the URL or the key into a file.

Every request carries `Authorization: Bearer $MFM_API_KEY`. Create calls send
`Content-Type: application/json`. The body is JSON only. Other users' ids return `404`. Expired or
deleted resources return `410`.

## Endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /v1/datasets` | Upload a history. `201` with `dataset_id`, `content_hash`, `created_at`, `last_used_at`, `analyses`, `expires_at`, `summary`, `history_ranges`, `warnings`. |
| `GET /v1/datasets` | List your datasets: `{"datasets": [...]}`, each with the same fields. |
| `GET /v1/datasets/{id}` | The summary, with the same fields. |
| `DELETE /v1/datasets/{id}` | Delete the dataset and all its analyses. `202`. |
| `POST /v1/analyses` | Start an analysis. `202` with `analysis_id`, `name`, `url`, `runs_total`, `estimate_seconds`, `expires_at`. With `"preview": true`: `200` with the expanded plans, `runs_total`, `estimate_seconds`, `assumptions` and `warnings`. Nothing is stored or run. |
| `GET /v1/analyses/{id}` | Status and counts, with `name`, `url` and `expires_at`. |
| `GET /v1/analyses/{id}/result` | The result table. `409 not_ready` until the analysis ends. |
| `GET /v1/analyses/{id}/weekly?scenario_id=` | Weekly rows for one plan. |
| `GET /v1/analyses/{id}/export` | The whole analysis as one JSON file (`mfm-analysis-export-v1`). `409 not_ready` until every run ends. |
| `DELETE /v1/analyses/{id}` | Cancel unstarted runs and delete the analysis. `202`. |
| `GET /v1/analyses?dataset_id=` | List your analyses. |

Do not call other endpoints.

Older servers may leave out `content_hash`, `created_at`, `last_used_at`, `analyses`, `name` and
`url`. Treat a missing field as unknown. `url` is `null` when the service has no site address.

## Retention

The service keeps a dataset 7 days after last use, at most 30 days after upload. Using a dataset
(for example, starting an analysis on it) moves `expires_at` later, so read it again; do not treat
it as a fixed deadline. An analysis has the `expires_at` of its dataset. Deleting a dataset deletes
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
   `A-Z a-z 0-9 . _ : -`). Body:

   ```json
   {"name": "weekly-history", "dates": ["2024-07-01", "..."], "outcome": [1012.5, "..."],
    "spend": {"tv": ["..."], "search": ["..."]}, "controls": {"holiday": ["..."]}}
   ```

   Dates are weekly, contiguous, ISO dates. The body is at most 5 MB. `history_ranges` gives, per
   channel and control, the weekly `p05`, `p95` and `max`, `zero_share` and `all_zero`.
2. Check `history_ranges` against the plan you will send. The weekly `p05` and `p95` are warning
   thresholds, not the edge of the observed data. Spend outside the history's 5th to 95th percentile
   gets a warning, and the model has little data there.
3. `POST /v1/analyses` with `"preview": true` first. It needs no `Idempotency-Key`. Show the user
   the run count and the warnings. If `runs_total` is large, give the count and the estimate before
   you start.
4. `POST /v1/analyses` without `preview`, with a fresh `Idempotency-Key`. Body:

   ```json
   {"dataset_id": "ds_123",
    "plan": {"dates": ["2026-07-06", "... 8 weeks"],
             "spend": {"tv": [100, "..."], "search": [50, "..."]},
             "controls": {"holiday": ["..."]}},
    "name": "Spend scenarios", "decision_weeks": [1, 8],
    "scenarios": [
      {"id": "half", "steps": [{"scale": 0.5}]},
      {"id": "tv_to_search", "steps": [{"move": {"from": "tv", "to": "search", "share": 0.2}}]},
      {"id": "mine", "plan": {"spend": {"tv": ["..."]}}}],
    "curves": [{"channel": "tv", "levels": [0, 0.5, 1.5, 2]}],
    "draws": 1000, "seed": 7}
   ```

   - `plan` is the baseline (id `baseline`, reserved): spend for every channel and a value for every
     control, for every forecast week. Its dates start the week after the history.
   - `decision_weeks` `[i, j]` (1-based, inclusive; default the whole plan) is the only window that
     steps and curves may change.
   - A scenario has a unique `id` (1 to 64 characters from `A-Za-z0-9_.-`; `baseline` and the `curve:`
     prefix are reserved) and exactly one of `steps` or `plan`. A scenario `plan` gives only the series
     it changes.
   - A curve is `{"channel", "levels", "base"?}`. Each level multiplies that channel's spend in the
     decision window. Level 1 equals the base and does not run again. Level ids are
     `curve:<channel>:<base>:<level>`. For a first look use levels 0 to 2 in steps of 0.25.
   - `name` is optional: 1 to 120 characters, trimmed; `422` otherwise.
   - `draws` is 1 to 2,000 (default 1,000). Use 200 to 500 for exploration. `seed` is 0 to 2147483647;
     the server picks one if you leave it out.
   - At most 200 different plans per analysis. Equal plans run once.
5. Poll `GET /v1/analyses/{id}` every 3 seconds until `status` is `done`, `failed` or `deleted`.
   `status` is one of `queued`, `running`, `done`, `failed`, `deleted`. The first run after idle can take
   about 20 seconds more. There are no partial results. `409 not_ready` from the result call means:
   keep polling.
6. `GET /v1/analyses/{id}/result` for the table, and `GET /v1/analyses/{id}/weekly?scenario_id=` for one
   plan's weekly rows (`scenario_id`, `rows` with `date`, `component`, `mean`, `p05`, `p95`). Each
   `table[*].draws_url` is a signed link to that plan's draws file.
7. Give the user the numbers and the result link (`url` from the analysis record; give the
   analysis id if `url` is `null`). Then ask before deleting. `GET /v1/analyses/{id}/export` saves the
   whole analysis as JSON. `DELETE /v1/analyses/{id}` or `DELETE /v1/datasets/{id}` (this also
   deletes all its analyses) when the user agrees. A `410` after the delete is expected.

## Steps

Steps apply in the listed order, inside `decision_weeks`.

| Step | Effect |
| --- | --- |
| `{"scale": k}` | Multiply every channel's spend by `k` (0 or more). With `"channels": [...]`, only those. |
| `{"move": {"from": a, "to": b, "share": s}}` | Move share `s` (0 to 1) of `a`'s total to `b`. Use `amount` instead of `share` for an absolute amount. Total spend stays the same. |
| `{"shift": {"channel": c, "into_weeks": [i, j]}}` | Put `c`'s total evenly into weeks `i` to `j`. Or give `pattern`, one non-negative weight per window week. |
| `{"set_controls": {"holiday": [...]}}` | Replace control values. One value per window week. |

The service rejects, before any run, spend or control values that are not zero where the history
series is all zero, negative spend, and unknown names. Warnings are never refusals.

## The result

`table` has one row per plan and component. Columns: `scenario_id`, `kind` (`baseline`, `scenario`,
`curve`), `base_scenario_id`, `varied_channel`, `level`, `steps`, `run_id`, `status` (`ok`, `failed`),
`spend_total`, `component`, `mean`, `p05`, `p95`, `window_mean`, `mean_difference`, `simulation_se`,
`sensitivity`, `warnings`, `draws_url`, `draws_url_expires_at`. Components are `outcome`, `baseline`,
each channel, each control, `latent` and active `indirect:*` rows. The response also has `assumptions`,
`meta` and `labels`. Quote `labels` when you share a result. Read the numbers as `SKILL.md` says.

## Errors and retries

Error bodies look like `{"error": {"code", "message", "field"?}}`.

- After a timeout or a connection error, repeat the same create call with the same key and body. This
  never starts the work twice.
- `429` and `502/503/504`: wait `Retry-After` seconds (or back off from 2 s, cap 60 s) and repeat with
  the same key. `rate_limited` waits 60 s. `overloaded_user_analyses` means 20 unfinished analyses: wait
  or delete one.
- `409 idempotency_conflict`: the key was used with another body. Use a new key. `409 plan_failed`: that
  plan has no result.
- `410`: the resource was deleted or expired. Do not retry.
- `404`: wrong id or another user's id. `401`: bad key; stop and tell the user.
- `422`: read `error.code`, `error.field` and `error.message`. Fix the request. Nothing started. Do not
  loop on a `422`.

## Draws files

Each plan's draws file is netCDF4 (float32, schema `river-draws/1`). Open it with
`xarray.open_dataset(path, engine="h5netcdf")`. A signed `draws_url` lasts about an hour. If its
download answers `403`, fetch the result again for new links. Never send the API key to a `draws_url`.
With the same seed, draw `i` of two plans is numerically coupled, so `sd(d_i) / sqrt(draws)` is the
standard error of the mean difference. It is still not a joint posterior sample.
