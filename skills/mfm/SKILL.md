---
name: mfm
description: Use the MFM API (River-1) to forecast a weekly outcome, backtest past weeks and compare budget scenarios from a weekly marketing history. Use when the user asks to run MFM or River-1, compare spend plans, sweep scenarios, check the fit on past weeks, or draw response curves from weekly spend, controls and an outcome. Covers install, setup, limits, the Python client, showing charts, reading the numbers, errors and deletion.
---

# MFM API (River-1)

The MFM API (Marketing Foundational Models) runs River-1 on a weekly history of spend, controls and
an outcome. One resource, an **analysis**, answers every question: a forecast of a baseline,
scenarios (budget changes to compare with it) and response curves. The baseline is either a
**plan** for the weeks after the history, or a **backtest**: weeks the dataset already holds,
so the forecast can be compared with what happened. This skill uses the Python client,
`pymc_mfm.client`. Agents without Python read `reference/http.md`.

Research prototype. Synthetic or internal data only. Not for budget decisions on its own.

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
4. Tell the user once: the service keeps the data 7 days after last use, at most 30 days after
   upload. Each analysis you run on a dataset counts as use and moves its expiry later, up to the
   30-day limit. Read the current deadline from `dataset.expires_at`; do not store it. You can
   delete the data earlier when the user no longer needs it.

## The four steps

Run this example with `history.csv` (columns `date`, `sales`, `tv`, `search`, `holiday`) and
`plan.csv` (the forecast weeks, same columns without `sales`). The plan starts the week after the
history ends. If the user gives no data, copy the synthetic files in `examples/` next to this skill
(104 weeks of history, 8 plan weeks). With the user's data, change the column names and channels to
match.

<!-- skill-example: main -->
```python
import pandas as pd

from pymc_mfm.client import Client, amount, scenario

client = Client()  # reads MFM_API_URL and MFM_API_KEY

# 1. Upload the weekly history, unless the service already holds exactly this data.
#    The service keeps it 7 days after last use, at most 30 days after upload.
for stored in client.datasets.list():  # your datasets, with content_hash and expires_at
    print(stored.dataset_id, stored.name, stored.weeks, stored.analyses, stored.expires_at)
history = pd.read_csv("history.csv", parse_dates=["date"])
dataset = client.datasets.create(
    history,
    date="date",
    outcome="sales",
    spend=["tv", "search"],
    controls=["holiday"],
    name="weekly-history",
    # currency="USD",  # optional: three uppercase letters, if the outcome is money
    reuse=True,  # reuse a dataset only when its content_hash matches exactly
)
print(dataset.dataset_id, dataset.weeks, dataset.start, dataset.end)
print(dataset.history_ranges)  # weekly p05 and p95 of each series: warning thresholds

# 2. Compare the current plan with two alternatives. The plan is the baseline, so two
#    scenarios make three plans in one analysis. Preview first: it runs nothing.
#    A number multiplies a channel's spend; amount(x) sets the channel's total spend.
plan = pd.read_csv("plan.csv", parse_dates=["date"])
tv_total, search_total = plan["tv"].sum(), plan["search"].sum()
scenarios = [
    scenario("reduce_20", {"tv": 0.8, "search": 0.8}),
    # Move a fifth of TV's money to search; total spend stays the same.
    scenario("tv_to_search", {"tv": 0.8, "search": amount(search_total + 0.2 * tv_total)}),
]
preview = client.analyze(dataset, plan, scenarios=scenarios, preview=True)
print(preview.runs_total, preview.estimate_seconds, preview.warnings)
print(preview.spend().groupby(level="scenario_id").sum())

analysis = client.analyze(dataset, plan, scenarios=scenarios, seed=7, name="Plan comparison")
print("analysis id:", analysis.id)  # print first: after a timeout, resume with client.get(id)
print("result link:", analysis.url)  # give this link to the user
analysis.wait(timeout=600)

# 3. Read the results.
outcome = analysis.table.query("component == 'outcome'")
print(outcome[["scenario_id", "spend_total", "mean", "p05", "p95", "mean_difference"]])
print(
    analysis.weekly("tv_to_search").query("component == 'outcome'")[["date", "mean", "p05", "p95"]]
)
print(analysis.labels)  # how the service says to read these numbers

# 4. Show the comparison: save the chart as PNG, then show the file to the user.
analysis.plot_compare().savefig("compare.png", dpi=150)
```

Use the user's own alternatives in place of `reduce_20` and `tv_to_search`, with ids that name them.
Keep the preview step. Add the optional examples below only when the user asks for them.

A backtest, when the user asks how well the model fits, or for scenarios on past weeks. The
model reads the input weeks (`context`) and predicts the weeks after them up to `query_end`. It does
not see the outcome of the predicted weeks, so `fit` compares the forecast with what happened. The
baseline is the dataset's own spend and controls in the predicted weeks; no plan is sent.

<!-- skill-example: backtest -->
```python
from pymc_mfm.client import curve

backtest = client.analyze(
    dataset,
    context=("2024-11-11", "2026-08-03"),  # first and last input week
    query_end="2026-11-02",  # last predicted week; prediction starts the week after the context
    scenarios=[scenario("budget_x1.2", {"tv": 1.2, "search": 1.2})],
    curves=[curve("tv", [0, 0.5, 1.5, 2]), curve("search", [0, 0.5, 1.5, 2])],
    seed=7,
    name="Backtest",
).wait(timeout=600)
print("result link:", backtest.url)
print(backtest.fit)  # r2, mape (a fraction) and crps, for the input and the predicted weeks
print(backtest.shares.query("window == 'query' and share == 'total'"))  # each series' share
print(backtest.curves)  # each channel's contribution at each spend level
```

`channels=[...]` and `controls=[...]` use only some series of the dataset (default: all). Every
date must be a week of the dataset. `fit`, `shares` and `curves` are `None` for a plan.

More scenarios on an analysis that has ended, when the user asks for more after seeing the results:

<!-- skill-example: again -->
```python
more = backtest.again([scenario("search_x1.5", {"search": 1.5})]).wait(timeout=600)
print(more.table.query("component == 'outcome'")[["scenario_id", "mean", "mean_difference"]])
```

`again` starts a new analysis with the same dataset, weeks (or plan), series, `decision_weeks`, seed
and draws. It reads them from the export, so it raises `NotReady` until the analysis has ended: call
`wait()` first. It runs one more baseline and the new scenarios, but no curves unless you pass `curves=`. Compare each
scenario only with the baseline of its own analysis, never with a number from another analysis.

Response curves, only when the user asks for curves: each channel alone at spend levels 0 to 2
times the baseline.

<!-- skill-example: curves -->
```python
from pymc_mfm.client import curve

curves = client.analyze(
    dataset, plan, curves=[curve("tv", [0, 0.5, 1.5, 2]), curve("search", [0, 0.5, 1.5, 2])], seed=7
).wait(timeout=600)
print("curves analysis id:", curves.id)
curves.plot_curves().savefig("curves.png", dpi=150)
```

A grid, only when the user asks for a grid of spend levels:

<!-- skill-example: grid -->
```python
from pymc_mfm.client import grid

grid_scenarios = grid({"tv": [0.5, 1.5], "search": [0.5, 1.5]})  # four scenarios
grid_analysis = client.analyze(dataset, plan, scenarios=grid_scenarios, seed=7, name="Spend grid")
grid_analysis.wait(timeout=600)
print("result link:", grid_analysis.url)
grid_analysis.plot_compare().savefig("grid.png", dpi=150)
```

Do not delete the dataset in the same script: deleting it also deletes the analyses the user opens
on the site. Delete it later, as "Delete the data" says.

`client.explore(dataset, plan)` is a first look: a curve for every channel at spend levels 0 to 2
in steps of 0.25, and the scenarios `spend_x1.2`, `spend_x1.4` and `spend_x1.6` (every channel times
1.2, 1.4 and 1.6). For a backtest, pass `context=` and `query_end=` in place of `plan`. Pass
`preview=True` to `analyze` or `explore` to see the run count first.

### Builders

A scenario changes spend only inside the window: the predicted weeks of a backtest, or
`decision_weeks` of a plan (default: the whole plan).

| Builder | Effect inside the window |
| --- | --- |
| `scenario(id, {"tv": 0.8, "search": 1.3})` | A named budget change. A number (0 to 5) multiplies that channel's spend. A channel left out keeps its spend. Ids are unique. |
| `amount(x)` | A channel's value in a scenario: set its total spend in the window to `x` (0 or more). The weekly pattern stays; a channel with no spend in the window gets `x` split evenly over its weeks. |
| `grid({"tv": [0.5, 1.5], "search": [...]})` | One scenario for every combination of multipliers. |
| `curve(channel, levels, base="baseline")` | One channel at each spend level, as a multiple of its spend in `base`. |

A scenario keeps each channel's weekly pattern and every control. There is no scenario for spend
timing or control values. A different plan can be its own analysis, but do not report a difference
between two analyses: their draws are not paired.

`analyze` also takes `name` (optional, 1 to 120 characters: the label the user sees on the site and
in the export), `decision_weeks=(i, j)` (1-based, inclusive: the only weeks that scenarios and
curves may change), `draws` and `seed`. Use `draws` of 200 to 500 for exploration and the default of 1,000 (or more, up
to the maximum of 2,000) for numbers you report. A request without a `seed` uses seed 0. The same request gives the
same numbers only with the same model revision, library revision, device class, dtype and execution
settings; results can differ across devices. More draws reduce sampling error. A different `seed` does
not. At most 200 different plans run per analysis. A plan needs spend for every channel and a value for every control, for every
forecast week.

Look at `dataset.history_ranges` and the preview warnings before you start. The weekly `p05` and
`p95` are warning thresholds, not the edge of the observed data (`max` is the largest week). Spend
outside the history's 5th to 95th percentile gets a warning, and the model has little data there. If
`preview.runs_total` is large, give the user the run count and `estimate_seconds` before you start.

## Show the results to the user

Do not only print tables. Save `plot_compare()` as PNG (the example does this) and show the file to
the user. Save and show `plot_curves()` only when the user asked for curves. Give the person the
result link, `analysis.url`, to open the analysis on the MFM API site. If `url` is `None`, give the
analysis id instead. Say what each chart is:

- `plot_compare()` draws each scenario's mean difference from its base plan. The bars are simulation
  error, not model uncertainty.
- `plot_curves()` (only for curves) draws each channel's contribution at each spend level, with the
  mean and the pointwise 5th to 95th percentile band. A red open marker means that plan has a
  warning, and the warnings are printed under the chart.

If the machine has no display, set `MPLBACKEND=Agg` before you run the script.
`analysis.export("analysis.json")` saves the whole analysis (request, scenario inputs, warnings,
results) as one JSON file, once every run has ended.

## How to read and report numbers

Always say what a number is. Quote `analysis.labels` when you share a result.

- `mean`, `p05`, `p95`: mean and pointwise 5th to 95th percentile of the **expected outcome without
  observation noise**. They are not an interval for actual weekly sales.
- These are **conditional model responses** to the inputs, **not calibrated causal lift**. Do not say a plan
  "causes" or "will deliver" a change. Say the model expects it.
- `mean_difference` is a paired mean difference: the plan's mean minus its base plan's mean, from draws
  with the same seed in the same analysis. `simulation_se` is the numerical precision of that difference
  only. It is **not model uncertainty**.
- **There is no interval or probability for a difference between scenarios.** Report the mean difference
  only. Never build an interval from `p05` and `p95`, and never say a difference is "significant". Draws of
  two plans are not one joint posterior. Never subtract numbers of two different analyses.
- A difference smaller than about 2 `simulation_se` is not resolved: say it is below numerical precision at this
  `draws` setting, or use more draws.
- The table has one row per plan and component. Quote the `outcome` rows for totals and the channel rows for
  contributions.
- `outcome` is the sum of all component rows within each draw. `mean`, `p05` and `p95` are totals over the
  whole forecast. `window_mean` is also a total, not a weekly mean: the mean over draws of the total over the
  window (`decision_weeks`, or the predicted weeks of a backtest).
- Report plan warnings next to any number from that plan, especially "outside the history's 5th-95th percentile"
  (low spend counts too) and "above the largest rolling total". The history gives no support for extrapolation.
- A `failed` row has no numbers. Say so. A `done` analysis can still have failed scenario or curve rows.
  If the baseline failed, the analysis is `failed`.
- `fit` (backtest only) scores the baseline against the actual outcome, for the input weeks (`context`)
  and the predicted weeks (`query`). `r2` and `mape` (a fraction) score the mean line; their `sd` is the
  spread of the score over draws. `crps` scores the draws (lower is better). The predicted weeks are the
  real test: the model did not see their outcome.

## Errors and limits

The client retries 429, 502, 503, 504 and connection errors, up to 3 times, waits as long as
`Retry-After` asks (at most 60 s), and reuses one idempotency key per call. A retry never starts the
work twice. It raises these errors, all in `pymc_mfm.client`:

| Error | Meaning and action |
| --- | --- |
| `Unauthorized` (401) | Bad key. Stop and tell the user. |
| `ValidationError` (400, 422) | Read `error.code`, `error.field` and `error.message`. Fix the request. Nothing started. Do not loop. `steps_removed`: scenario steps were removed; send multipliers. |
| `NotFound` (404) | Wrong id, or another user's id. |
| `Gone` (410) | The resource was deleted or expired. Do not retry the same call. On a create call (`create`, `analyze`, `again`), an earlier equal request was deleted: if the user needs it again, pass `fresh=True`. |
| `NotReady` (409 `not_ready`) | The analysis is not finished. Call `wait()`. |
| `Conflict` (other 409) | `idempotency_conflict`: the key was used with another body; use a new key. `plan_failed`: that plan has no result. |
| `PayloadTooLarge` (413) | The request is too large. Send fewer weeks, scenarios or curves in one call. |
| `RateLimited` (429) | Too many requests, `overloaded_user_analyses` (20 unfinished analyses) or `overloaded_user_runs` (200 unfinished runs). Wait `error.retry_after` seconds, or delete an analysis. |
| `ServerError` (5xx), `ApiConnectionError` | The retries ran out. Wait and try again later. |

Before it uploads a history or submits an analysis, the client raises `ValueError` when the dates
have a missing week, a duplicate, the wrong order, or a plan that does not start the week after
the history. Read the message, which names the row and dates, and fix the input. Never sort, fill
or shift data to make it pass.

`wait(timeout=600)` raises `TimeoutError` when the analysis is still running. The analysis keeps
running. Do not run the script again. Run `client.get("<analysis id>").wait(timeout=600)`. An identical
`analyze` call returns the same analysis, because the client sends a key derived from the request. Use
`fresh=True` only if the user wants a new run, for example after `failed`. The first run after an
idle period can take about 20 seconds more (cold start). The result is ready only when every run has
ended; then each plan has numbers or a `failed` row. Do not create analyses in a loop without checking
`runs_total`. Per user, at most 20 unfinished analyses and 200 unfinished runs, and 50 runs run at once.

A signed `draws_url` lasts about an hour. `analysis.draws()` reads the per-draw files and fetches new
links when they expired. It needs the `draws` extra (included in `[client]`). Never send the API key
to a `draws_url`.

## Delete the data

Deleting a dataset deletes all its analyses, including the ones the user opens on the site. So report
the results first: the numbers, the charts and the result link. Then ask before deleting, and say that
deleting the dataset removes its analyses. Otherwise the data expires 7 days after last use, at most
30 days after upload. When the user says delete:

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
