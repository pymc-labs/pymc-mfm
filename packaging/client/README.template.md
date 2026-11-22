# pymc-mfm

Python client for PyMC MFM (marketing foundational models), the hosted service from PyMC Labs.
It uploads a weekly marketing history, runs a forecast or a backtest, compares budget scenarios and
returns the results as pandas tables. The models run on the service, not on your machine.

The service is in a closed pilot. You need an API key from the PyMC Labs pilot: without one the
client cannot do anything. Create your API key on the Connect page of the console.

## Install

```bash
pip install pymc-mfm
```

It needs Python {requires_python}. To read all draws or draw plots, install the extra:
`pip install "pymc-mfm[client]"`.

Set the address of the service and your key:

```bash
export MFM_API_URL="<the address from your pilot invitation>"
export MFM_API_KEY="<your API key>"
```

## First example

This makes its own weekly data, uploads it, analyzes a plan for the 8 weeks after it with two
budget scenarios, and prints the outcome rows. It needs no file.

```python
import numpy as np
import pandas as pd

from pymc_mfm.client import Client, amount, scenario

# 1. Synthetic data: a weekly history of 104 weeks and a plan for the 8 weeks after it.
rng = np.random.default_rng(0)
weeks = pd.date_range("2024-11-11", periods=104, freq="7D")
tv = rng.gamma(4.0, 2_000.0, size=104)
search = rng.gamma(4.0, 1_500.0, size=104)
price = rng.normal(10.0, 0.3, size=104)
noise = rng.normal(0, 500, size=104)
sales = 40_000 + 0.8 * tv + 1.2 * search - 1_000 * (price - 10.0) + noise
history = pd.DataFrame({"week": weeks, "sales": sales, "tv": tv, "search": search, "price": price})
first_plan_week = history["week"].iloc[-1] + pd.Timedelta(weeks=1)
plan = pd.DataFrame(
    {
        "week": pd.date_range(first_plan_week, periods=8, freq="7D"),
        "tv": 8_000.0,  # planned spend per week
        "search": 6_000.0,
        "price": 10.0,  # the control value per week
    }
)

# 2. Upload the history. With reuse=True, the same data again reuses the stored dataset.
client = Client()  # reads MFM_API_URL and MFM_API_KEY
dataset = client.datasets.create(
    history,
    date="week",
    outcome="sales",
    spend=["tv", "search"],
    controls=["price"],
    name="first-look",
    reuse=True,
)
print("dataset", dataset.dataset_id, "expires", dataset.expires_at)
for warning in dataset.warnings:
    print("dataset warning:", warning)

# 3. Analyze the plan and two budget scenarios, and wait until the analysis ends.
analysis = client.analyze(
    dataset,
    plan,
    scenarios=[
        scenario("half", {"tv": 0.5, "search": 0.5}),
        scenario("tv_to_search", {"tv": 0.8, "search": amount(60_800)}),
    ],
    seed=7,
    name="first-scenarios",
)
analysis.wait()

# 4. Print the outcome rows, save them, and print the address of the result page.
table = analysis.table
outcome = table[table["component"] == "outcome"]
columns = ["scenario_id", "status", "mean", "p05", "p95", "mean_difference", "simulation_se"]
print(outcome[columns].to_string(index=False))
outcome[columns].to_csv("first_analysis.csv", index=False)
print("result page:", analysis.url)
```

## Help

If something fails, reply to your invitation. Include the full error text, the time, and the dataset or analysis id if you have one.

Results are conditional model responses to your inputs. They are not calibrated causal lift.
Use synthetic or internal data only.

## Licence

The client code and documentation distributed in pymc-mfm {version} are licensed under Apache-2.0. This grant excludes the hosted service, model weights, training and inference code. Future releases may use different licences. Use of the hosted service is under its own terms.

Copyright 2026 The PyMC Labs Developers
