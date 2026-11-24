# PyMC Marketing Foundational Models

PyMC MFM (marketing foundational models) trains and runs amortized neural networks for
marketing mix decomposition. A trained network reads one weekly time series (media spend,
controls and the observed outcome) and returns posterior draws of each component's
contribution: baseline, each channel, each control and unobserved factors. The network is
trained on many simulated datasets, so a new dataset needs no model fitting.

This repository is the library. Trained weights are separate. A **bundle** is a folder
with one trained network's weights. **River-1** is PyMC Labs' trained network: load its
bundle from the Hugging Face Hub, or run it on the hosted River API.

> **Status: alpha.** The API can change between releases.

**Documentation:** see the [Guide](https://github.com/pymc-labs/pymc-mfm/blob/main/docs/guide/index.qmd).

## Installation

Python 3.13 or newer:

```bash
pip install pymc-mfm
uv add pymc-mfm   # in a uv project
```

No trained weights are included.

- On Linux, pip and uv install the CUDA build of PyTorch, about 3 GB. On a machine
  without a GPU, install the CPU build first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- PyPI has no PyTorch wheels for Intel Macs. There, install PyTorch from conda-forge
  first.
- `pymc-mfm` is not on conda-forge yet. In a conda environment, install it with pip.
- In a uv project, set `requires-python = ">=3.13"` in `pyproject.toml`, because uv
  resolves for every Python version the project allows.

## Quickstart

`mfm.load` opens a model and `model.inference(context, query)` runs it. `context` is the
history: one row per week with spend, controls and the observed outcome. `query` holds the
weeks to explain or forecast, starting the week after the history ends.

The example loads a bundle from `./models/my-model`. If you have no bundle, replace that
line with one from the table below.

<!-- docs-example: first-analysis -->
```python
import numpy as np
import pandas as pd

import pymc_mfm as mfm

# Synthetic data: 104 weeks of history and an 8-week plan after it.
rng = np.random.default_rng(0)
weeks = pd.date_range("2024-11-11", periods=104, freq="7D")
tv = rng.gamma(4.0, 2_000.0, size=104)
search = rng.gamma(4.0, 1_500.0, size=104)
price = rng.normal(10.0, 0.3, size=104)
noise = rng.normal(0, 500, size=104)
sales = 40_000 + 0.8 * tv + 1.2 * search - 1_000 * (price - 10.0) + noise
history = pd.DataFrame({"week": weeks, "sales": sales, "tv": tv, "search": search, "price": price})
future = pd.DataFrame(
    {
        "week": pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=8, freq="7D"),
        "tv": 8_000.0,
        "search": 6_000.0,
        "price": 10.0,
    }
)

roles = {
    "date_column": "week",
    "target_column": "sales",
    "channel_columns": ["tv", "search"],
    "control_columns": ["price"],
}
model = mfm.load("./models/my-model", **roles)  # a bundle folder; see the table below
result = model.inference(history, future, n_samples=100, random_seed=0)

result.plot.fit()  # expected vs observed sales
result.plot.contributions_over_time()  # weekly contribution of every component
result.idata  # context and query draws in an xarray.DataTree
```

Change only the `load` call to choose the model and where it runs:

| Call | Runs | Needs |
| --- | --- | --- |
| `mfm.load("./models/my-model", **roles)` | On your machine, from a bundle folder | The bundle |
| `mfm.load("river-1", **roles)` | On your machine, with River-1 weights from Hugging Face | Access to the PyMC Labs Hugging Face organization |
| `mfm.load("river-1", engine="hosted", **roles)` | On the hosted River API | An invitation: `MFM_API_URL` and `MFM_API_KEY` |

The [River-1 quickstart notebook](https://github.com/pymc-labs/pymc-mfm/blob/main/examples/river_1_quickstart.ipynb) walks through
this workflow. To call the River API directly for budget scenarios and backtests, see
[the River API guide](https://github.com/pymc-labs/pymc-mfm/blob/main/docs/guide/http-api.qmd).

## Development

```bash
uv sync --locked
uv run --no-sync pytest -q -n 2 -m "not slow"
uv run ruff check . && uv run ruff format --check .
uv run --no-sync python scripts/check_public_boundary.py
```

This repository is public, history included: never commit trained weights, credentials,
user data or internal deployment names. Open pull requests against `main`;
[AGENTS.md](https://github.com/pymc-labs/pymc-mfm/blob/main/AGENTS.md) has the contributor rules.

**Documentation site.** `docs/` is a [Quarto](https://quarto.org) project. With Quarto
CLI 1.11.9, run `uv sync --locked --group docs`, then `uv run --no-sync quarto preview docs`.

## License

The code is licensed under [Apache-2.0](https://github.com/pymc-labs/pymc-mfm/blob/main/LICENSE). Model weights are separate
artifacts with their own licenses.
