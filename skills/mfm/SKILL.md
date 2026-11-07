---
name: mfm
description: Use the MFM API (River-1) to forecast a weekly outcome and compare spend scenarios from a weekly marketing history. Use when the user asks to run MFM or River-1, compare spend plans, sweep scenarios, or draw response curves from weekly spend, controls and an outcome. Covers install, setup, limits, the Python client, showing charts, reading the numbers, errors and deletion.
---

# MFM API (River-1)

The MFM API (Marketing Foundational Models) runs River-1 on a weekly history of spend, controls and
an outcome. One resource, an **analysis**, answers every question: a forecast of one plan,
scenarios (spend plans to compare) or response curves. This skill uses the Python client,
`pymc_mfm.client`. Agents without Python read `reference/http.md`.

Research prototype. Synthetic data. Not for budget decisions on its own.

## Set up

1. Python 3.13 or later is required. Git must be signed in to GitHub with read access to the
   `pymc-labs/pymc-mfm` repository (the repository is private). Install with `uv`, which can fetch
   Python 3.13:

   ```bash
   uv venv -p 3.13
   uv pip install "pymc-mfm[client] @ git+https://github.com/pymc-labs/pymc-mfm.git"
   ```

   The `[client]` extra adds the plots and the draws reader. It does not install PyTorch.
2. Check that `MFM_API_URL` and `MFM_API_KEY` are set in the environment. The client reads both.
   If one is missing, stop and ask the user. Do not look for them in files. Never print or log the
   key. Never write the URL or the key into a file, a commit or a message.
3. Check the data. It is weekly aggregates: contiguous weekly dates, 1 to 10 spend channels
   (0 or more each), 1 to 10 controls, a positive outcome, and unique column names that are not
   `baseline`, `latent`, `outcome` or `indirect:*`. 75 to 207 weeks of history is the supported
   range (a warning outside it). A dataset body is at most 5 MB.
4. Tell the user once: the service deletes the data 7 days after you create it. You can delete it
   earlier when the user no longer needs it.

## The four steps

Run this example with `history.csv` (columns `date`, `sales`, `tv`, `search`, `holiday`) and
`plan.csv` (the forecast weeks, same columns without `sales`). The plan starts the week after the
history ends. If the user gives no data, copy the synthetic files in `examples/` next to this skill
(104 weeks of history, 8 plan weeks). With the user's data, change the column names and channels to
match.

<!-- skill-example: main -->
```python
import pandas as pd

from pymc_mfm.client import Client, curve, grid, move, scale, scenario, shift

client = Client()  # reads MFM_API_URL and MFM_API_KEY

# 1. Upload the weekly history. The service keeps it for 7 days.
history = pd.read_csv("history.csv", parse_dates=["date"])
dataset = client.datasets.create(
    history,
    date="date",
    outcome="sales",
    spend=["tv", "search"],
    controls=["holiday"],
    name="weekly-history",
)
print(dataset.dataset_id, dataset.weeks, dataset.start, dataset.end)
print(dataset.history_ranges)  # weekly p05 and p95 of each series: what the history shows

# 2. Forecast the baseline plan.
plan = pd.read_csv("plan.csv", parse_dates=["date"])
first = client.analyze(dataset, plan, seed=7).wait(timeout=600)
print(first.table[["scenario_id", "component", "mean", "p05", "p95"]])

# 3. Compare scenarios. Preview first: it runs nothing.
scenarios = [
    scenario("reduce_20", scale(0.8)),
    scenario("tv_to_search", move("tv", "search", share=1 / 3)),
    scenario("tv_front_loaded", shift("tv", into_weeks=(1, 4))),
    *grid(tv=[0.5, 1.5], search=[0.5, 1.5]),
]
preview = client.analyze(dataset, plan, scenarios=scenarios, preview=True)
print(preview.runs_total, preview.estimate_seconds, preview.warnings)
print(preview.spend().groupby(level="scenario_id").sum())

analysis = client.analyze(dataset, plan, scenarios=scenarios, seed=7)
analysis.wait(timeout=600)
print("analysis id:", analysis.id)  # give this to the user; resume with client.get(analysis.id)

# 4. Read the results.
outcome = analysis.table.query("component == 'outcome'")
print(outcome[["scenario_id", "spend_total", "mean", "p05", "p95", "mean_difference"]])
print(
    analysis.weekly("tv_to_search").query("component == 'outcome'")[["date", "mean", "p05", "p95"]]
)
print(analysis.labels)  # how the service says to read these numbers

# Response curves: each channel alone at spend levels 0 to 2 times the baseline.
curves = client.analyze(
    dataset, plan, curves=[curve("tv", [0, 0.5, 1.5, 2]), curve("search", [0, 0.5, 1.5, 2])], seed=7
).wait(timeout=600)
print("curves analysis id:", curves.id)

# Show the results: save both charts as PNG, then show the files to the user.
analysis.plot_compare().savefig("compare.png", dpi=150)
curves.plot_curves().savefig("curves.png", dpi=150)
```

Do not delete the dataset in the same script: deleting it also deletes the analyses the user opens
on the site. Delete it later, as "Delete the data" says.

`client.explore(dataset, plan)` is a first look: a curve for every channel at nine spend levels and
two budget scenarios. Pass `preview=True` to `analyze` or `explore` to see the run count first.

### Builders

| Builder | Effect inside the decision window |
| --- | --- |
| `scale(k, channels=None)` | Multiply spend by `k`, for all channels or the listed ones. |
| `move(from_, to, share=)` or `move(from_, to, amount=)` | Move a share (0 to 1) or an amount of one channel's total to another. Total spend stays the same. |
| `shift(channel, into_weeks=(i, j))` or `shift(channel, pattern=[...])` | Put a channel's total evenly into weeks `i` to `j` (1-based), or spread it by weights. |
| `set_controls(name=[...])` | Replace control values. The plan gets `sensitivity: true`. |
| `scenario(id, *steps)` or `scenario(id, plan={...})` | A named plan: ordered steps, or a partial plan. Ids are unique. |
| `grid(tv=[0.5, 1.5], search=[...])` | One scenario for every combination of spend multiples. |
| `curve(channel, levels, base="baseline")` | One channel at each spend level, as a multiple of its baseline spend. |

`analyze` also takes `decision_weeks=(i, j)` (the only weeks that steps and curves may change),
`draws` and `seed`. Use `draws` of 200 to 500 for exploration and 1,000 (the default) for numbers
you report. At most 2,000. Set `seed` when the user needs a repeatable result. At most 200 different
plans run per analysis. A plan needs spend for every channel and a value for every control, for every
forecast week.

Look at `dataset.history_ranges` and the preview warnings before you start. Spend outside the
history's 5th to 95th percentile gets a warning, and the model has no data there. If
`preview.runs_total` is large, give the user the run count and `estimate_seconds` before you start.

## Show the results to the user

Do not only print tables. Save `plot_compare()` and `plot_curves()` as PNG (the example does this)
and show both files to the user. Give the analysis id, so the person can open the analysis on the
MFM API site (the team gives the address). Say what each chart is:

- `plot_compare()` draws each scenario's mean difference from its base plan. The bars are simulation
  error, not model uncertainty.
- `plot_curves()` draws each channel's contribution at each spend level, with the mean and the
  pointwise 5th to 95th percentile band.

If the machine has no display, set `MPLBACKEND=Agg` before you run the script.

## How to read and report numbers

Always say what a number is. Quote `analysis.labels` when you share a result.

- `mean`, `p05`, `p95`: mean and pointwise 5th to 95th percentile of the **expected outcome without
  observation noise**. They are not an interval for actual weekly sales.
- These are **conditional model responses** to the inputs, **not calibrated causal lift**. Do not say a plan
  "causes" or "will deliver" a change. Say the model expects it.
- `mean_difference` is the plan's mean minus its base plan's mean. `simulation_se` is the numerical precision
  of that difference only. It is **not model uncertainty**.
- **There is no interval or probability for a difference between plans.** Never build one from `p05` and `p95`,
  and never say a difference is "significant". Draws of two plans are not one joint posterior.
- A difference smaller than about 2 `simulation_se` is not resolved: say it is below numerical precision at this
  `draws` setting, or use more draws.
- The table has one row per plan and component. Quote the `outcome` rows for totals and the channel rows for
  contributions.
- `outcome` is the sum of all component rows within each draw. Totals are over the whole forecast. `window_mean` is
  the mean over `decision_weeks` only.
- Report plan warnings next to any number from that plan, especially "outside the history's 5th-95th percentile"
  (low spend counts too) and "above the largest rolling total". The history gives no support for extrapolation.
- `sensitivity: true` means the plan's controls differ from the baseline's.
- A `failed` row has no numbers. Say so. If the baseline failed, the analysis is `failed`.

## Errors and limits

The client retries 429, 502, 503, 504 and connection errors, up to 3 times, waits as long as
`Retry-After` asks (at most 60 s), and reuses one idempotency key per call. A retry never starts the
work twice. It raises these errors, all in `pymc_mfm.client`:

| Error | Meaning and action |
| --- | --- |
| `Unauthorized` (401) | Bad key. Stop and tell the user. |
| `ValidationError` (400, 422) | Read `error.code`, `error.field` and `error.message`. Fix the request. Nothing started. Do not loop. |
| `NotFound` (404) | Wrong id, or another user's id. |
| `Gone` (410) | The resource was deleted or expired. Do not retry. Create it again if the user needs it. |
| `NotReady` (409 `not_ready`) | The analysis is not finished. Call `wait()`. |
| `Conflict` (other 409) | `idempotency_conflict`: the key was used with another body; use a new key. `plan_failed`: that plan has no result. |
| `RateLimited` (429) | Too many requests, or `overloaded_user_analyses`: 20 unfinished analyses. Wait `error.retry_after` seconds, or delete one. |
| `ServerError` (5xx), `ApiConnectionError` | The retries ran out. Wait and try again later. |

`wait(timeout=600)` raises `TimeoutError` when the analysis is still running. The first run after an
idle period can take about 20 seconds more (cold start). There are no partial results. Do not create
analyses in a loop without checking `runs_total`. Per user, at most 20 unfinished analyses and 50
admitted runs run at once.

A signed `draws_url` lasts about an hour. `analysis.draws()` reads the per-draw files and fetches new
links when they expired. It needs the `draws` extra (included in `[client]`). Never send the API key
to a `draws_url`.

## Delete the data

Deleting a dataset deletes all its analyses, including the ones the user opens on the site. So give
the user the numbers, the charts and the analysis id first, then ask whether to delete now or let the
data expire after 7 days. When the user says delete:

<!-- skill-example: delete -->
```python
client.datasets.delete(dataset.dataset_id)  # the dataset and all its analyses
```

`analysis.delete()` deletes one analysis. The service removes the data within minutes. A `Gone`
error after the delete is expected.

## Do not

- Do not put secrets, the base URL or other users' ids into files, commits or logs.
- Do not keep data longer than the user needs it. Offer to delete it when the work is done.
- Do not call endpoints that this page and `reference/http.md` do not list. There is no `/v1/jobs`, no
  server-side grid, no optimizer and no difference interval.
