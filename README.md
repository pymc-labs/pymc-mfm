# PyMC MFM

PyMC MFM trains and runs amortized neural networks for marketing mix
decomposition. A trained network reads one time series (media spend, controls
and the observed outcome) and returns posterior draws of each component's
contribution: baseline, each channel, each control and unobserved factors.
The network is trained once on many simulated datasets. For a new dataset,
`fit` loads observed context; it does not train the network's weights.

This repository is the library. Simulators, training corpora and trained weights
are separate artifacts: models load from a local folder or from the Hugging Face Hub.

## Installation

Python 3.13 or newer.

```bash
pip install "pymc-mfm[hub] @ git+https://github.com/pymc-labs/pymc-mfm.git"
```

The base install (NumPy, pandas, PyTorch, safetensors and packaging) loads local
models and runs inference. Optional extras:

| Extra | Adds |
| --- | --- |
| `hub` | Loading from and publishing to the Hugging Face Hub |
| `training` | MLflow tracking and the diagnostics recorded during training |
| `diagnostics` | Diagnostics, plots and HTML reports for results |
| `arviz` | Export of results as an ArviZ `DataTree` |
| `generation` | `pymc-generator`, to simulate corpora and read `.npz` corpora |
| `examples` | `pymc-marketing`, for the example notebook |

## Decompose a time series

```python
import numpy as np
import pandas as pd
import torch

from pymc_mfm import from_pretrained

model = from_pretrained("models/my-model")  # a model folder; see "Load a trained model"

rng = np.random.default_rng(0)
n_weeks, n_context = 104, 80
spend = rng.gamma(2.0, 50.0, size=(n_weeks, 2))  # 2 channels
data = pd.DataFrame(
    {
        "date": pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON"),
        "search": spend[:, 0],
        "social": spend[:, 1],
        "seasonality": np.sin(2 * np.pi * np.arange(n_weeks) / 52),
        "sales": 1_000 + spend @ [2.0, 1.0] + rng.normal(0, 20, n_weeks),
    }
)

model.fit(
    data.iloc[:n_context],
    date_column="date",
    target_column="sales",
    channel_columns=["search", "social"],
    control_columns=["seasonality"],
)
future = data.iloc[n_context:].drop(columns="sales")
torch.manual_seed(0)
query = model.predict(future, n_samples=500)

query.components.shape  # (500, 24, D): [draw, week, component]
query.component_names  # ('baseline', 'channel:search', 'channel:social', ..., 'indirect:dc')
```

`fit` snapshots the history and column roles, validates the inputs, and returns
the model. It needs at least two rows, strictly positive outcomes, nonnegative
spend, and at least one channel and control. Control padding and masks are
automatic; excess controls are rejected and channel count is not fixed.

`predict` requires only the stored date, channel and control columns. Physical
column order does not matter; extra columns and any query targets are ignored.
Dates must be timezone-naive, unique, and regularly spaced. Rows are sorted
without modifying the dataframe. Queries must start at the next context period
and follow its cadence without gaps. Calendar frequency is inferred when
possible; two-row histories use their fixed interval.

Repeated predictions reuse the same context without appending predictions.
A successful second `fit` replaces context; a failed fit preserves it.
`predict` before `fit` raises an error. Weight bundles exclude private history,
so call `fit` again after loading. The illustrative data above exercises the API,
not attribution calibration.

### Context and query snapshots

With the `[arviz]` extra, request both decomposition windows after a successful
`fit` and `predict`:

```python
idata = model.build_idata()  # return-only; does not assign or replace model.idata
assert model.idata is None

stored = model.build_idata(store=True)
assert model.idata is stored
search_draws = idata["query"]["posterior"]["channel_contribution"].sel(channel="search")

# Retain separate scenario snapshots without accumulating history on the model.
scenario = future.assign(search=lambda frame: frame["search"] * 1.2)
model.predict(scenario, n_samples=500)
assert model.idata is None
idata_scenario = model.build_idata()
# idata and stored still contain the original scenario, unchanged.
```

The returned `xarray.DataTree` has `context` and `query` subtrees, each with its
own `posterior` and `constant_data`. Known context outcomes appear separately as
`context/observed_data/outcome(date)`; query outcomes are neither read nor invented.
ArviZ operations take a window subtree, for example
`arviz_stats.summary(idata["query"], var_names=["channel_contribution"], kind="stats")`,
not the convenience root.

| Posterior variable | Dimensions |
| --- | --- |
| `channel_contribution` | `chain, draw, date, channel` |
| `control_contribution` | `chain, draw, date, control` |
| `baseline`, `latent` | `chain, draw, date` |
| `indirect_cc`, `indirect_zc`, `indirect_dc` | `chain, draw, date` |
| `graph` | `chain, draw, edge` |

`date` contains each window's original sorted dates. `channel` and `control`
contain the original column names, without internal prefixes or padded slots.
Contributions are in outcome units, not raw spend/control values; baseline and
indirect totals do not acquire a channel-by-control grid. Padding is absent, not
estimated causal inactivity, and inactive scalar slots are omitted. `chain=1`
and `draw` retain the uncertainty in the amortized samples. **Context and query
are separate marginals**: equal draw indices do not pair them into full-horizon
trajectories. Do not concatenate their date axes to manufacture joint draws.

`model.idata` is a passive, read-only property, initially `None`; reading it never
samples or converts. `store=True` returns the same tree exposed by that property
and reuses it while current. A return-only build creates a separate container
and leaves any stored tree unchanged. A successful `fit` or `predict` clears the
stored reference and context-draw cache; failures preserve the last successful
state. Earlier caller-held trees are never rewritten by later model calls.
Weight bundles exclude history, prediction snapshots and exports.

The first build samples the missing context marginal once, using the retained
history/query inputs and inference options; it never resamples query. Subsequent
builds reuse both sets of draws, even in return-only mode. Thus `store=False`
means no retained **DataTree**, not no retained inference state. Prediction
snapshots consume memory, and fresh exports may allocate arrays. Context
sampling and optional xarray/ArviZ imports are deferred until requested; a
missing extra is reported before context sampling. Building after `fit` alone
raises an error: the current sampler needs a successful prediction with a
nonempty query window.

Lower-level `result.to_datatree()` and archive exports retain their generic
`time`/`component` layouts; the model builder supplies the named dataframe-facing
schema above.

### Lower-level array inputs

`decompose(model, observations, window="query", ...)` remains available for
array/tensor callers; `window="context"` selects the observed window instead.
It requires the explicit padded observation mapping below.
`model.predict_batch(batch, ...)` is the separate tensor-level joint sampler.

`T` time steps, `K` channels, `M = model.config.n_controls` control slots,
`J = model.config.n_demand` latent slots. Rows are equally spaced steps (for
example weeks). The first `C` rows are the **context**, where the outcome is
known; the other `T - C` rows are the **query**.

| Key | Shape | Rules |
| --- | --- | --- |
| `spend` | `[T, K]` | All `T` rows. Nonnegative. Any `K >= 1`. |
| `controls` | `[T, M]` | All `T` rows. Width must equal `M`; unused columns are zero. |
| `context_outcome` | `[C]` | Outcome of the first `C` rows, `0 < C < T`. Strictly positive. |
| `channel_mask` | `[K]` | Optional, binary. `0` marks a padding column (must be zero). Default all `1`. |
| `control_mask` | `[M]` | Optional, binary. At least one active control. Default all `1`. |
| `latent_mask` | `[J]` | Optional, binary. Default all `1`. |

Every series is rescaled with statistics of its context rows, so any units work.
A column that is zero in the context but nonzero in the query is rejected.

### Output

`predict` returns a query `DecompositionResult`; lower-level `decompose` can return
either window:

| Field | Shape | Content |
| --- | --- | --- |
| `components` | `[S, T_window, D]` | Draws in outcome units. `T_window` is `C` (context) or `T - C` (query). |
| `component_names` | `[D]` | Baseline, `K` channels, `M` controls, latent total, indirect `cc`, `zc`, `dc` (`D = K + M + 5`) |
| `graphs` | `[S, L]` | Causal graph sampled with each draw (edge layout below) |
| `time` | `[T_window]` | Query dates (`predict`); row indices or supplied labels (`decompose`) |

- `predict` derives names from column roles; `decompose` accepts `channel_names=`
  and `control_names=`.
- Components exclude observation noise. Context and query calls are separate
  draws, not one joint posterior.
- Sampling skips padded channel, control and latent slots, their graph edges
  and contributions the graph rules out, so padding adds little cost. Skipping
  them does not change the draws.
- `mode="model_average"` (default) samples graphs; `mode="map"` uses the most
  frequent graph; `mode="fixed", graph=...` conditions on a given graph.
- `result.save(path)` and `DecompositionResult.load(path)` store and reload a
  result without the model.
- With `[diagnostics]`: `result.plot.reconstruction(observed=...)` and
  `result.diagnostics(truth=...)`. With `[arviz]`: `result.to_datatree()`.
- Dates survive result save/load, plots and export. Saved results retain their
  original datetime precision; DataTree export requires a lossless supported
  second-through-nanosecond resolution and rejects rounding or collapsed labels.

## Train an amortized network

### Training data

A corpus is a dict of NumPy arrays that holds `N` simulated datasets ("worlds")
with their true components and causal graph. Any simulator can produce it. All
worlds share `T`; `K`, `M` and `J` are padded widths, and masks mark the real slots.

| Key | Shape | Content |
| --- | --- | --- |
| `treatment_raw` | `[N, T, K]` | Media spend, `>= 0` |
| `covariates` | `[N, T, M]` | Controls |
| `outcome_raw` | `[N, T]` | Outcome, `> 0` in the context |
| `support_mask` | `[N, T]` | `1` for context rows, then `0` for query rows; both parts nonempty |
| `treatment_active_mask` | `[N, K]` | Real channels; at least one per world |
| `covariate_active_mask` | `[N, M]` | Real controls; at least one per world |
| `latent_active_mask` | `[N, J]` | Real latent factors |
| `baseline_intrinsic` | `[N, T]` | True baseline |
| `treatment_contribution_raw` | `[N, T, K]` | True channel contributions, `>= 0` |
| `covariate_contribution` | `[N, T, M]` | True control contributions |
| `latent_unobserved_contribution` | `[N, T, J]` | True latent contributions |
| `indirect_effects_by_source` | `[N, T, 3]` | True indirect effects, sources `cc`, `zc`, `dc` |
| `outcome_noise` | `[N, T]` | Observation noise |
| `g` | `[N, L]` | True causal graph, binary |
| `is_val` | `[N]` | `1` for held-out worlds, used only for evaluation; at least one |
| `cell_id` | `[N]` | Integer group of related worlds; keep a group in one split |
| `diagnostics` | `dict` | `{"schema_version": 4, "edge_types": ["cy", "dc", "dz", "dy", "zy", "zc", "cc", "zz"]}` |

The components plus `outcome_noise` add up to `outcome_raw`. Padded slots are
zero in every array.

The graph `g` concatenates these edge blocks, each ordered source-major:

| Block | Edges | Size |
| --- | --- | --- |
| `cy` | channel → outcome | `K` |
| `dc` | latent → channel | `J·K` |
| `dz` | latent → control | `J·M` |
| `dy` | latent → outcome | `J` |
| `zy` | control → outcome | `M` |
| `zc` | control → channel | `M·K` |
| `cc` | channel → channel, no self-loops | `K·(K-1)` |
| `zz` | control → control, no self-loops | `M·(M-1)` |

`pymc_mfm.model.graph.EdgeSlotLayout(K, M, J).n_slots` gives `L`.

### Run training

```bash
pip install "pymc-mfm[training] @ git+https://github.com/pymc-labs/pymc-mfm.git"
```

```python
from pymc_mfm import ModelConfig
from pymc_mfm.training import TrainingConfig, train

result = train(
    [corpus],  # one or more corpora with the same K, M, J and T
    out_dir="runs/my-model",  # a new directory
    model_config=ModelConfig(n_controls=6, n_demand=1),  # M and J of the corpus
    training=TrainingConfig(
        steps=5_000,
        batch_size=64,
        warmup_steps=250,
        eval_every=1_000,
        milestone_steps=(),
    ),
    device="auto",  # cuda, mps or cpu
)
model = result.model
```

- `ModelConfig` sets the network size (`d_model=128`, `n_layers=4`, `n_heads=8`,
  `d_ff=512`, `flow_hidden=256`) and the corpus widths `n_controls` (`M`) and
  `n_demand` (`J`). The number of channels is not fixed.
- `TrainingConfig()` defaults to the full run: 62,500 AdamW updates of 64
  worlds, peak learning rate 2.8e-3 with 3,125 warmup updates and cosine decay,
  evaluation every 6,250 updates on 16 held-out worlds. For shorter runs, also
  reduce `warmup_steps`, `eval_every` and `milestone_steps`.
- Worlds with `is_val == 0` are trained on, `is_val == 1` are evaluated.
- MLflow tracking is on by default and writes to a local SQLite store in
  `out_dir` (or `MLFLOW_TRACKING_URI`). Disable it with
  `train(..., tracking=TrackingConfig(enabled=False))`.
- `python -m pymc_mfm.training --help` lists the same options as a CLI. The CLI
  reads `.npz` corpora and needs the `generation` extra.
- For bitwise-repeatable CPU runs, set `OMP_NUM_THREADS=1`.

`out_dir` contains:

| Path | Content |
| --- | --- |
| `model/` | Final model folder |
| `checkpoint_best.pt` | Checkpoint with the best held-out score |
| `checkpoint_last.pt` | Latest checkpoint |
| `run.json` | Configuration, corpus summary and run status |
| `metrics.jsonl`, `evaluations.jsonl` | Training losses and held-out evaluations |

Checkpoints hold no optimizer state, so a run cannot be resumed.

## Load a trained model

A model folder holds `config.json`, `model.safetensors` and `manifest.json`.

```python
from pymc_mfm import from_pretrained, save_pretrained
from pymc_mfm.training import load_checkpoint

model = from_pretrained("runs/my-model/model", device="cpu")  # final model

best, payload = load_checkpoint("runs/my-model/checkpoint_best.pt")
save_pretrained(best, "models/my-model")  # export a checkpoint as a model folder
```

Loaded models are in evaluation mode. `save_pretrained` writes only into a new or
empty directory. Loading checks file hashes, configuration and package versions,
and never unpickles code.

## Load models from Hugging Face

Needs the `hub` extra.

```python
from pymc_mfm import from_pretrained, list_profiles

list_profiles("pymc-labs/mfm")  # models in the repository
model = from_pretrained(
    "pymc-labs/mfm",
    profile="river-1",
    revision="970e6095f14824d7e25ecace7e20a47bebde35b8",  # pin a full commit
)
```

- A repository holds one model at its root (omit `profile`) or several models
  under `profiles/<name>/`.
- Private repositories use `hf auth login`, `HF_TOKEN` or `token=`.
  `pymc-labs/mfm` is private to the PyMC Labs organization.
- `cache_dir=` sets the download folder; `local_files_only=True` loads from the
  cache without network access.
- `model.provenance["hub_revision"]` records the commit that was loaded.
- `push_to_hub(model, "org/repo", profile="my-model", license="...")` publishes
  a model and returns the commit id. New repositories are private.

[`examples/river_1_quickstart.ipynb`](examples/river_1_quickstart.ipynb) loads a
model from the Hub, decomposes a simulated dataset and plots the result.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

`benchmarks/inference_speed.py` times `decompose` with 1,000 draws of River-1 on
the quickstart data, read from the local Hugging Face cache (load River-1 once
while logged in). `--write-reference` saves the seeded draws of one version and
`--check-reference` compares another version's draws against them.

With Conda or Mamba:

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --group dev -e .
```

## Status

Version `0.0.1.dev0`, not a published release. Code and model licenses are not
yet selected. See [RELEASE.md](RELEASE.md) for changes and known limits.
