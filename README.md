# PyMC MFM

PyMC MFM trains and runs amortized neural networks for marketing mix
decomposition. A trained network reads one time series (media spend, controls
and the observed outcome) and returns posterior draws of each component's
contribution: baseline, each channel, each control and unobserved factors.
The network is trained once on many simulated datasets. For a new dataset,
`add_context` loads observed context; it does not train the network's weights.

This repository is the library. Simulators, training corpora and trained weights
are separate artifacts: models load from a local folder or from the Hugging Face Hub.

## Installation

Python 3.13 or newer.

```bash
pip install "pymc-mfm[local] @ git+https://github.com/pymc-labs/pymc-mfm.git"
```

The default install, without `[local]`, is a light client for the River API (NumPy,
pandas, httpx and packaging; about 130 MB, no PyTorch). `pymc_mfm.client` is the only
part that runs there; see "Call the River API". The `[local]` extra adds PyTorch and
safetensors, which load local models and run inference in your process. Names such as
`from_pretrained` and `MFM` load on first use and raise an `ImportError` that names
`pymc-mfm[local]` when PyTorch is missing. Optional extras:

| Extra | Adds |
| --- | --- |
| `local` | PyTorch and safetensors, to load local models and run inference |
| `draws` | `xarray`, `h5netcdf` and `h5py`, to read the API's per-draw files |
| `plot` | `matplotlib`, for the client's plots |
| `client` | `draws` and `plot` |
| `hub` | `local`, and loading from and publishing to the Hugging Face Hub |
| `training` | `local`, MLflow tracking and the diagnostics recorded during training |
| `diagnostics` | `local`, and diagnostics, plots and HTML reports for results |
| `arviz` | `local`, and export of results as an ArviZ `DataTree` |
| `generation` | `local`, and `pymc-generator`, to simulate corpora and read `.npz` corpora |
| `examples` | `local`, and `pymc-marketing`, for the example notebook |

Every extra that runs the model includes `local`, so installing `pymc-mfm[hub]` still
installs PyTorch.

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

model.add_context(
    data.iloc[:n_context],
    date_column="date",
    target_column="sales",
    channel_columns=["search", "social"],
    control_columns=["seasonality"],
)
future = data.iloc[n_context:].drop(columns="sales")
torch.manual_seed(0)
query = model.inference(future, n_samples=500)

query.components.shape  # (500, 24, D): [draw, week, component]
query.component_names  # ('baseline', 'channel:search', 'channel:social', ..., 'indirect:dc')
```

`add_context` snapshots the history and column roles, validates the inputs, and returns
the model. It needs at least two rows, strictly positive outcomes, nonnegative
spend, and at least one channel and control. Control padding and masks are
automatic; excess controls are rejected and channel count is not fixed.

`inference` requires only the stored date, channel and control columns. Physical
column order does not matter; extra columns and any query targets are ignored.
Dates must be timezone-naive, unique, and regularly spaced. Rows are sorted
without modifying the dataframe. Queries must start at the next context period
and follow its cadence without gaps. Calendar frequency is inferred when
possible; two-row histories use their fixed interval.

Repeated predictions reuse the same context without appending predictions.
A successful second `add_context` replaces context; a failed call preserves it.
`inference` before `add_context` raises an error. Weight bundles exclude private
history, so call `add_context` again after loading. The illustrative data above
exercises the API, not attribution calibration.

### Context and query snapshots

With the `[arviz]` extra, request both decomposition windows after a successful
`add_context` and `inference`:

```python
idata = model.build_idata()  # return-only; does not assign or replace model.idata
assert model.idata is None

stored = model.build_idata(store=True)
assert model.idata is stored
search_draws = idata["query"]["posterior"]["channel_contribution"].sel(channel="search")

# Retain separate scenario snapshots without accumulating history on the model.
scenario = future.assign(search=lambda frame: frame["search"] * 1.2)
model.inference(scenario, n_samples=500)
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
and leaves any stored tree unchanged. A successful `add_context` or `inference` clears the
stored reference and context-draw cache; failures preserve the last successful
state. Earlier caller-held trees are never rewritten by later model calls.
Weight bundles exclude history, prediction snapshots and exports.

The first build samples the missing context marginal once, using the retained
history/query inputs and inference options; it never resamples query. Subsequent
builds reuse both sets of draws, even in return-only mode. Thus `store=False`
means no retained **DataTree**, not no retained inference state. Prediction
snapshots consume memory, and fresh exports may allocate arrays. Context
sampling and optional xarray/ArviZ imports are deferred until requested; a
missing extra is reported before context sampling. Building after `add_context` alone
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

`inference` returns a query `DecompositionResult`; lower-level `decompose` can return
either window:

| Field | Shape | Content |
| --- | --- | --- |
| `components` | `[S, T_window, D]` | Draws in outcome units. `T_window` is `C` (context) or `T - C` (query). |
| `component_names` | `[D]` | Baseline, `K` channels, `M` controls, latent total, indirect `cc`, `zc`, `dc` (`D = K + M + 5`) |
| `graphs` | `[S, L]` | Causal graph sampled with each draw (edge layout below) |
| `time` | `[T_window]` | Query dates (`inference`); row indices or supplied labels (`decompose`) |

- `inference` derives names from column roles; `decompose` accepts `channel_names=`
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

### Sampling on a GPU (Apple or NVIDIA)

Load the model on the GPU. The same calls then sample with fused kernels and need no
training, conversion, warm-up or compiler:

```python
model = from_pretrained("pymc-labs/mfm", profile="river-1", device="mps")  # Apple silicon
model = from_pretrained("pymc-labs/mfm", profile="river-1", device="cuda")  # NVIDIA GPU
result = decompose(model, observations, n_samples=1000)  # or model.inference(...)
```

River-1, 1,000 draws, the quickstart world with a 166-week context, PyTorch 2.14, median
of warm calls (`benchmarks/inference_speed.py --query-weeks 10,50,150,205 --device mps`
or `--device cuda`):

| Query weeks | CPU, Apple M3 Max (s) | Apple M3 Max GPU (s) | NVIDIA L4 (s) | NVIDIA A100 (s) | NVIDIA H100 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.66 | 0.075 | 0.10 | 0.084 | 0.063 |
| 50 | 2.51 | 0.30 | 0.35 | 0.23 | 0.13 |
| 150 | 7.25 | 0.86 | 1.00 | 0.60 | 0.32 |
| 205 | 10.17 | 1.17 | 1.35 | 0.80 | 0.41 |

- GPU sampling draws its base noise with PyTorch's CPU generator, so a seeded call draws
  the same samples on CPU, Apple and NVIDIA GPUs up to float32 round-off (identical graphs
  on these inputs on an Apple M3 Max and NVIDIA T4, A10G, L4, A100 and H100). The noise is
  drawn on a worker thread while the GPU runs; do not draw random numbers from other
  threads during a call. CUDA sampling used to draw with the CUDA generator.
- Each Euler step runs one kernel per coordinate network that computes its first layer,
  its hidden product and its output without writing activations to memory: Metal on Apple
  GPUs; CUDA C++ on NVIDIA GPUs, compiled on first use by NVRTC, which PyTorch's CUDA
  wheels include (no CUDA toolkit or compiler). Against PyTorch's GPU operations on the
  same GPU this is 2.7–7.2x faster at 50–205 query weeks and 2.5–3.3x at 10, where the CPU
  sets more of the pace (NVIDIA T4 to H100).
- The NVIDIA kernels run on the current CUDA device. Other devices, ROCm, float64 models,
  compiled sampling and builds whose NVRTC cannot compile for the GPU (sampling warns once)
  use PyTorch's operations, with the same draws.
- Sampling on an Apple GPU needs PyTorch 2.9 or newer: older MPS releases miscompute
  sampling (2.8) or lack operations it uses, and raise a clear error.
- On the CPU, sampling transports up to eight batches at once, never more than PyTorch's
  thread count. Off Apple silicon each batch runs on its own share of the threads, and
  each network's static and context terms fill its first layer before the scalar product
  accumulates into it; on x86 its hidden layer also runs on row chunks that stay in cache.
  On Linux x86 capped at 8 CPUs the calls above ran 1.6–2.4x faster on AMD EPYC (Zen 3
  and 4) and 1.0–1.35x faster on Intel Xeon (Ice Lake, Skylake or Cascade Lake, Sapphire
  and Emerald Rapids), whose multi-threaded matrix products already used the cores well. On
  GitHub's 2-vCPU runners (random weights of River-1's architecture, against main in the
  same environment) Windows x86 sampled 1.25–1.36x faster, Linux x86 (one PyTorch thread)
  1.14–1.16x and Linux ARM 1.04–1.06x, plus 1–2% from the first-layer fill there (measured
  against the previous revision). Off Apple silicon draws agree with main up to
  float32 round-off (no graph flipped; components within 2e-7 of the largest on the
  runners, 9e-7 against the Mac reference on Linux x86). Apple silicon already ran eight
  batches sharing every thread, and each batch still conditions its own graphs, so its CPU
  draws are bitwise unchanged.
- A call at 205 query weeks used about 1 GiB on the Apple GPU (Metal driver allocation)
  and 0.3 GiB on NVIDIA GPUs (PyTorch allocation). GPU sampling conditions each distinct
  graph once per call when they fit one batch (River-1 draws 3–4 on this world), else per
  batch: random weights of River-1's architecture, which draw a distinct graph almost every
  draw, peaked at 4.2 GiB on the Apple GPU and 1.2 GiB on NVIDIA GPUs at 205 weeks (main:
  1.2 and 0.4 GiB) and sampled 3.4x (M3 Max) and 2.5x (T4) faster than main.
- In a macOS virtual machine (GPU "Apple Paravirtual device", as on GitHub's macOS
  runners), PyTorch 2.14's MPS computes linear layers with a bias wrongly (46% relative
  error on a 4,096 × 256 layer), so GPU draws there are wrong; sample on the CPU in a VM.

### Faster repeated sampling (optional)

A long-running process that samples many times, such as a service, can compile the
sampler once with `torch.compile`. On an NVIDIA L4 with River-1 and 1,000 draws, warm
compiled calls ran about 1.4 times as fast as eager calls on four test inputs; the first
compilation took 20–30 s.

```python
from pymc_mfm.inference import warm_up_sampling

model = from_pretrained("pymc-labs/mfm", profile="river-1").to("cuda")
model.compile_sampling()  # opt-in; sampling stays eager until this call
warm_up_sampling(model, channels=range(1, 11))  # compile now, not on the first request
result = decompose(model, observations, n_samples=1000)
model.sampling_status  # SamplingStatus(compiled=True, mode='default', compiles=..., fallback=None)
```

- The first call at a new kind of input compiles, which can take tens of seconds. Warm
  up with worlds, draw counts and modes like the ones you serve. `compiles` counts
  every compilation, so a later increase shows an input the warm-up did not cover.
- Compiled draws follow the eager distribution; they agree with eager draws up to
  float round-off but are not bit-identical. With a fixed seed, serialized calls on one
  device repeat their draws, including a call that compiles. Sampling uses PyTorch's
  process-wide random generator, so run warm-up before other threads sample.
- PyTorch's compile cache (`TORCHINDUCTOR_CACHE_DIR`) lets a restarted process reuse
  earlier compilations. A compiler failure warns, returns the model to eager sampling and
  shows in `sampling_status.fallback`. `model.disable_compiled_sampling()` switches it off.
- On CPU, compiling usually does not speed sampling up.

## Call the River API

`pymc_mfm.client` is an HTTP client for the River API v3.1. It needs no PyTorch.
Give it the service URL and an API key as arguments or in `MFM_API_URL` and
`MFM_API_KEY`. Install `pymc-mfm[client]` to read draws and draw plots.

```python
import pandas as pd

from pymc_mfm.client import Client, grid, move, scale, scenario, shift

client = Client("https://api.example.com")  # key from MFM_API_KEY

# `history` is a weekly DataFrame; `plan` has the forecast weeks of the same columns.
dataset = client.datasets.create(
    history,
    date="week",
    outcome="sales",
    spend=["tv", "search"],
    controls=["price"],
    name="brand-a",
)
print(dataset.history_ranges)  # what the history shows, per series

analysis = client.analyze(
    dataset,
    plan,  # the baseline plan, a DataFrame or a dict
    scenarios=[
        scenario("half", scale(0.5)),
        scenario("tv_to_search", move("tv", "search", share=0.2)),
        scenario("front_loaded", shift("tv", into_weeks=(1, 4))),
        *grid(tv=[0.5, 1.5], search=[0.5, 1.5]),
    ],
    decision_weeks=(1, 8),
    seed=7,
)
analysis.wait()
analysis.table  # one row per plan and component; mean, p05, p95, window_mean, ...
```

`client.explore(dataset, plan)` is a first look: a response curve for every channel at
nine spend levels (0 to 2 times the baseline in steps of 0.25) and two budget scenarios.
Pass `preview=True` to `analyze` or `explore` to see the expanded plans, run count and
warnings without starting anything. `client.get(analysis_id)` resumes an analysis.

An `Analysis` has `status()`, `wait(timeout=, poll=)`, `table`, `weekly(scenario_id)`,
`draws(scenario_ids=None)`, `plot_curves()`, `plot_compare()` and `delete()`. `draws()` reads
each plan's per-draw file into one `xarray.Dataset` stacked along `scenario`; if a signed
URL has expired it fetches the result again for new URLs and never reruns anything.

Read the numbers as the result's `labels` say, which the client keeps on the objects
(`analysis.labels`, and `attrs["labels"]` on tables). Intervals are pointwise 5th to 95th
percentiles of expected outcome and are not causal lift. The service gives no interval for
the difference between two plans: `simulation_se` measures numerical precision of the mean
difference only, and `plot_compare()` draws it as "simulation error, not model uncertainty".

Every request that creates something carries an `Idempotency-Key`, reused when the client
retries, so a retry never starts the work twice. The client retries HTTP 429, 502, 503 and 504
and connection errors, and waits as long as `Retry-After` asks, up to `Client(max_retry_after=...)` seconds (default 60). Errors raise `NotFound`,
`Gone`, `Conflict`, `NotReady`, `RateLimited`, `ValidationError` and others, each with
`code`, `message` and `field` from the service.

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
model from the Hub, decomposes a simulated dataset and plots the result. Its
context scatter overlays River-1's weekly posterior mean channel contributions and
the simulated true contributions against raw spend, with one panel per channel.
Its scenario sweep varies each channel from 0% to 200% of its actual spend path
over eight query weeks, keeping history and other inputs fixed. It plots the
eight-week channel contribution with pointwise 90% approximate posterior intervals
and generator-computed truth at the same spend levels, highlighting the original
100% spend path. These are conditional scenario responses, not identified causal
lift curves.

## Development

```bash
uv sync --locked --extra local --extra client
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The client tests need only the light install. To run them without PyTorch, as CI does:

```bash
uv sync --locked
uv run --no-sync pytest -q tests/client --confcutdir=tests/client
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

### What may be committed

This repository will be public, history included. Commit library code, tests,
synthetic fixtures and docs. Never commit trained weights, credentials, user data,
internal deployment names or private results. [AGENTS.md](AGENTS.md) has the full
table and the boundary check, which CI runs on every pull request:

```bash
python3 scripts/check_public_boundary.py
```

Maintainers can pair a private companion checkout, linked at an ignored `.private`
path. The library never needs it:

```text
<parent>/
├── pymc-mfm/            this repository: install, test and build here
│   ├── .private ──┐     ignored link, maintainers only
│   └── .venv/     │     one environment for both checkouts
└── <companion>/  ◄┘     private repository with its own setup.sh
```

## Status

Version `0.0.1.dev0`, not a published release. Code and model licenses are not
yet selected. See [RELEASE.md](RELEASE.md) for changes and known limits.
