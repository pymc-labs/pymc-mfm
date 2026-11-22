# PyMC MFM

PyMC MFM (marketing foundational models) trains and runs amortized neural
networks for marketing mix decomposition. A trained network reads one time
series (media spend, controls and the observed outcome) and returns posterior
draws of each component's contribution: baseline, each channel, each control
and unobserved factors.
The network is trained once on many simulated datasets. For a new dataset,
`inference(context, query, ...)` conditions on observed history without training its weights.

This repository is the library. Simulators, training corpora and trained weights
are separate artifacts: models load from a local folder or from the Hugging Face Hub.

## Installation

Python 3.13 or newer.

```bash
pip install "pymc-mfm[local] @ git+https://github.com/pymc-labs/pymc-mfm.git"
```

The default install, without `[local]`, is a light client for the River API (NumPy,
pandas, httpx and packaging; about 130 MB, no PyTorch). `pymc_mfm.client` is the only
part that runs there; see "Call the River API". The `[local]` extra adds PyTorch,
safetensors, xarray and ArviZ Base for local models and complete in-memory inference trees.
Names such as `from_pretrained` and `MFM` load on first use and raise an `ImportError` that names
`pymc-mfm[local]` when PyTorch is missing. Optional extras:

| Extra | Adds |
| --- | --- |
| `local` | PyTorch, safetensors, xarray and ArviZ Base for local models and inference trees |
| `draws` | `xarray`, `h5netcdf` and `h5py`, to read the API's per-draw files |
| `plot` | `matplotlib`, for `model.plot` charts and the client's plots |
| `interactive` | `plotly`, for interactive `model.plot` charts |
| `client` | `draws` and `plot` |
| `hub` | `local`, and loading from and publishing to the Hugging Face Hub |
| `training` | `local`, MLflow tracking and the diagnostics recorded during training |
| `diagnostics` | `local`, and diagnostics, plots and HTML reports for results |
| `arviz` | `local`, and ArviZ statistics for result analysis |
| `generation` | `local`, and `pymc-generator`, to simulate corpora and read `.npz` corpora |
| `examples` | `local`, and `pymc-marketing`, for the example notebook |

Every extra that runs the model includes `local`, so installing `pymc-mfm[hub]` still
installs PyTorch.

## Decompose a time series

```python
import numpy as np
import pandas as pd

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

history, future = data.iloc[:n_context], data.iloc[n_context:]
roles = {
    "date_column": "date",
    "target_column": "sales",
    "channel_columns": ["search", "social"],
    "control_columns": ["seasonality"],
}
# Preflight explicit inputs before sampling; no previous inference is required.
usage = model.input_usage(history, future, **roles)
idata = model.inference(history, future, **roles, n_samples=500, random_seed=0)
assert idata is model.idata
query = model.query_result  # exact query draw bank for components, diagnostics and graphs

query.components.shape  # (500, 24, D): [draw, week, component]
query.component_names  # ('baseline', 'channel:search', 'channel:social', ..., 'indirect:dc')
```

`random_seed` replays the same random numbers: the same seed and inputs give the same
draws. The outputs agree up to floating-point round-off. They can differ slightly
across devices, thread counts, and compiled or eager runs. On a CPU with the
same thread count, a repeat gives the same result. A seeded call uses its own generator
and does not draw from or seed PyTorch's global random state, so other models and `torch`
calls cannot disturb it. After `compile_sampling()`, PyTorch itself saves and restores the global state while it compiles, which can undo global draws that another thread makes in that time. In `inference` the context window
continues the query window's stream. The model tree (`idata`) records the seed; the result
of `predict` or `decompose` does not, so keep the seed yourself if you save one. Without
`random_seed`, a call draws from PyTorch's global generator and advances it, so
`torch.manual_seed` before the call still sets the draws and each call gives new ones.
`predict` and `response_curves` take the same keyword.

`inference` snapshots the explicit history, query and column roles and samples both
windows before returning their complete `xarray.DataTree`. It stores that same tree
as `model.idata` by default. History needs at least two rows, strictly positive
outcomes, nonnegative spend, and at least one channel and control. Control padding
and masks are automatic; excess controls are rejected and channel count is not fixed.

The query requires the specified date, channel and control columns. Physical
column order does not matter and extra columns are ignored. The model never reads
query targets: a target column in the query frame is kept only so `model.plot` can
compare the forecast with what happened (values that are not finite numbers count as
unobserved).
Dates must be timezone-naive, unique, and regularly spaced. Rows are sorted
without modifying the dataframe. Queries must start at the next context period
and follow its cadence without gaps. Calendar frequency is inferred when
possible; two-row histories use their fixed interval.

River-1 was trained on weekly data with 75 to 207 context weeks, 1 to 10 channels
and 1 to 10 controls. The library does not warn outside this range.

Each successful complete call atomically replaces the stored run and clears its
response curves; failures preserve the previous complete run. Changing a future
plan or its horizon can change context draws too, so every full call samples both
marginals afresh. Weight bundles exclude runtime history, results, curves and idata.
After loading weights, run a complete inference again. The illustrative data above
exercises the API, not attribution calibration.

For an independent query-only calculation after complete inference, use
`prediction = model.predict(future, n_samples=200)`. It reuses the last successful
history and roles without appending predictions, sampling context, or replacing
`model.idata`, `model.query_result`, model charts or curves. Its sampling options
have their normal defaults, not the previous run's settings; fixed mode requires
an explicit graph. Before complete inference, `predict` raises a prerequisite error.

### Charts

`model.plot` draws the latest complete `inference` with matplotlib (the `[plot]`
extra) or plotly (the `[interactive]` extra), never an intervening `predict`:

```python
model.plot.inputs()  # outcome, media and control covariates by week, forecast weeks shaded
model.plot.fit()  # expected vs observed outcome, with MAPE, R² and interval coverage
model.plot.contributions_over_time()  # stacked weekly means of every component
model.plot.shares()  # share of the expected outcome by media and control covariate
model.plot.contribution_totals()  # each component's total over the window, over draws
model.plot.contribution_vs_media()  # each week's media covariate against its contribution

model.response_curves(levels=[0, 0.5, 1, 1.5, 2], weeks=8)  # samples sweep scenarios
model.plot.response_curves()

model.plot.fit(backend="plotly")  # interactive: hover, zoom, click a legend entry to hide it
```

| Method | Shows | Main options |
| --- | --- | --- |
| `inputs()` | Outcome, media covariates and control covariates by week | |
| `fit()` | Mean and interval of the expected outcome over the observed outcome; MAPE and R² of the mean line (± their standard deviation over draws) and the empirical coverage of the interval | `window` |
| `contributions_over_time()` | Stacked weekly means of every component; `stacked=False` gives one panel per component with its interval | `window`, `components`, `stacked`, `truth` |
| `shares()` | Each media and control covariate's share of the expected outcome: bars with intervals, or `view="dist"` kernel densities | `share="mean"` or `"total"`, `window`, `view` |
| `contribution_totals()` | Histogram of each component's total over the window | `window`, `components`, `bins`, `truth` |
| `contribution_vs_media()` | Each week's raw media covariate against its mean contribution | `window`, `channels`, `truth` |
| `response_curves()` | The latest `model.response_curves(...)`: contribution by media covariate level, observed level marked | `channels`, `truth` |

Every chart takes `backend=` (`"matplotlib"` or `"plotly"`; set a session default with
`pymc_mfm.set_backend`), `title=` and, where it summarizes draws, `interval=`
(default 0.9). Charts with one axes also take matplotlib `ax=`. Matplotlib charts return a
`Figure` that notebooks show without pyplot; plotly charts return a
`plotly.graph_objects.Figure`. `window` is `"context"` (history weeks), `"query"`
(forecast weeks) or, for the charts over time, `None` for both. `truth`, for simulated
data, is a DataFrame of known contributions indexed by date (by level for response
curves) with one column per media or control covariate name.

Charts never sample: both marginals are ready when complete inference succeeds.
`response_curves` samples separate media-covariate/level scenarios (one shared run
for level 1), then returns the curves and attaches them to a new `model.idata` root.
It keeps the result until the next successful complete `inference`; `predict`
does not replace it. Failed inference or curve calculation leaves the previous
complete analysis intact. Intervals are central intervals of the draws (pointwise
over weeks or levels) and exclude observation noise; contributions are conditional
on the inputs, not causal lift.

### Number of draws

`n_samples` defaults to 1,000 draws for each marginal in complete inference and for
query-only prediction. Summaries from few draws change between runs: on the
River-1 quickstart world, the posterior-mean contribution of one channel moved by
up to about 36% across 20 seeds at 32 draws, by 9% at 1,000 draws and by 3% at
4,000 draws. Cost grows about linearly with draws. Historical **query-only**
CPU measurements for River-1 (8 threads, shared Linux x86 machine,
`benchmarks/inference_speed.py --repeats 3`, median of 3 runs) were
2.6, 12.3 and 34.8 s at 1,000 draws and 4.4, 22.0 and 69.8 s at 2,000 draws, for 10, 50
and 150 query weeks. A full paired inference additionally samples context and
constructs/stores its tree; these timings do not measure that complete operation.
Memory also grows with draws: CPU sampling holds every draw's noise
for every task at once, so split a large `predict_batch` into smaller batches or lower
`n_samples`. For numbers you report, keep the default (1,000 draws) or raise it, and seed
the call. For a quick look, pass a smaller count:

```python
# For numbers you report: default draws, seeded
prediction = model.predict(future, random_seed=0)

# Exploration only: fewer draws, so estimates are rougher
quick = model.predict(future, n_samples=200)  # independent; stored analysis unchanged
```

### Context and query snapshots

The `[local]` extra includes complete in-memory trees; no export call or optional
statistics package is required:

```python
idata = model.idata  # current complete snapshot, including any attached curves
assert idata is model.idata
search_draws = idata["query"]["posterior"]["channel_contribution"].sel(channel="search")
assert query.provenance["input_usage"] == usage

# A new complete scenario replaces the model's run, not caller-held snapshots.
scenario = future.assign(search=lambda frame: frame["search"] * 1.2)
idata_scenario = model.inference(history, scenario, **roles, n_samples=500)
assert idata_scenario is model.idata

# Suppress only the return value, never sampling, conversion or storage.
returned = model.inference(history, future, **roles, n_samples=500, return_idata=False)
assert returned is None
assert model.idata is not None  # the complete replacement, with both marginals
# idata and idata_scenario still describe their original runs.
```

`return_idata=True` is the default and returns the exact current `model.idata`.
`False` returns `None` but completes and stores the same analysis; it is not a
lower-cost or no-storage mode. `input_usage(history, **roles)` inspects context
alone; supplying `future` inspects the complete request without sampling or
changing stored state or RNG.

The tree has `context` and `query` subtrees, each with its own `posterior` and
`constant_data`. Covariates are original-unit `media(date, channel)` and
`controls(date, control)`, without padding. Context observations appear as
`context/observed_data/outcome(date)`. If any query outcomes are finite,
`query/observed_data/outcome(date)` stores them with NaN for unobserved dates;
these outcomes are never neural inputs. Complete window provenance is JSON text
in each window's `attrs["provenance"]`, or a dictionary on `model.query_result.provenance`.
ArviZ statistics (the `[arviz]` extra) take a window subtree, for example
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

`model.idata` and `model.query_result` are passive properties, initially `None`;
reading them never samples or converts. Successful complete inference replaces
the root and increments `inference_revision` only after all work succeeds.
Successful curves publish another root with `/response_curves/posterior`
`contribution(chain, draw, channel, level)` and `/response_curves/constant_data`
media totals and actual sweep dates. Curves may use a different draw count from
either window and never replace `query_result`. Earlier caller-held trees and
results are snapshots; do not mutate them to change model inputs.

Both marginals and their analysis representation consume memory even with
`return_idata=False`. Query-only `predict` avoids context sampling and tree
conversion while leaving the stored analysis unchanged. There is no history-only
model tree or chart state.

File export is explicit: install `[draws]`, write with
`pymc_mfm.save_idata(model.idata, "analysis.nc")` (it never overwrites) and read
with `pymc_mfm.load_idata("analysis.nc")`. Result evidence
`query.save(...)` / `DecompositionResult.load(...)` is separate from tree export;
neither restores a runnable dataframe session. Trees can contain sensitive
outcomes, covariates and labels; review them before sharing.

Lower-level `result.to_datatree()` and archive exports retain their generic
`time`/`component` layouts; complete model inference supplies the named schema above.

### Lower-level array inputs

`pymc_mfm.inference.decompose(model, observations, window="query", ...)` remains available for
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

`model.query_result`, query-only `predict` and lower-level `decompose` expose a
`DecompositionResult`; the latter can return either window:

| Field | Shape | Content |
| --- | --- | --- |
| `components` | `[S, T_window, D]` | Draws in outcome units. `T_window` is `C` (context) or `T - C` (query). |
| `component_names` | `[D]` | Baseline, `K` channels, `M` controls, latent total, indirect `cc`, `zc`, `dc` (`D = K + M + 5`) |
| `graphs` | `[S, L]` | Causal graph sampled with each draw (edge layout below) |
| `time` | `[T_window]` | Query dates (`query_result` / `predict`); row indices or supplied labels (`decompose`) |

- Dataframe inference/prediction derives names from column roles; `decompose` accepts
  `channel_names=` and `control_names=`.
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
  `result.diagnostics(truth=...)`. `result.to_datatree()` exports the generic result schema.
- Dates survive result save/load, plots and export. Saved results retain their
  original datetime precision; DataTree export requires a lossless supported
  second-through-nanosecond resolution and rejects rounding or collapsed labels.

### Inspect sampled graphs

With `[diagnostics]`, switch between the marginal-average graph and individual
complete graphs without rerunning inference:

```python
result = model.query_result  # exact completed query draws; no resampling for helpers
result.plot.dag()  # Average: numeric labels show marginal edge frequencies
result.plot.dag(graph=0)  # Most frequent complete graph: binary edge indicators
result.plot.dag(graph=1).savefig("graph-rank-1.svg")

from pathlib import Path

Path("graph-average.dot").write_text(result.plot.dot(), encoding="utf-8", newline="\n")
Path("graph-rank-0.dot").write_text(result.plot.dot(graph=0), encoding="utf-8", newline="\n")

inspector = result.plot.graph_inspector()
inspector  # Interactive display in a trusted notebook
inspector.save("graphs.html")  # Open this local file in a browser; no server required
```

Ranks are **zero-based**, ordered by descending draw count, then ascending binary
graph vector for ties. The inspector includes every observed complete graph, a
count/frequency distribution with 50-row pages, an Average option, and download
buttons for the currently displayed SVG and Graphviz DOT source. Empty-edge graphs
retain their named active nodes; padded nodes are omitted. Configured channel/control
column names appear with their role prefixes, and the target is labeled `outcome`.

The same methods are available on `result.diagnostics(...).plot`; the standalone
`pymc_mfm.plotting.plot_dag(summary, graph=...)`, `graph_to_dot(summary, graph=...)`
and `plot_graph_inspector(summary)` also accept saved diagnostic summaries.
The inspector and DOT exports show prediction graphs only, even when diagnostics
include graph truth; `dag()` retains its separate truth comparison.

Diagrams use the system Graphviz `dot` engine with its defaults unchanged:
ellipses, natural edge routing and native spacing. Average edges carry numeric
frequency labels; individual graphs use plain binary arrows. Install Graphviz
with `brew install graphviz` on macOS or `sudo apt-get install graphviz` on
Debian/Ubuntu, and ensure `dot` is on `PATH`. No Python Graphviz package is needed.

`dag()` returns a `GraphvizGraph`, not a Matplotlib figure. Its `savefig` method
writes native PNG or vector SVG without Matplotlib-only options such as
`bbox_inches`. Creating the optional inspector also requires `dot`: it renders
all discovered graphs in one batch and embeds their SVGs. Opening the resulting
HTML requires only a browser, not Graphviz.

Writing DOT source does **not** require Graphviz. Saved DOT can also be rendered
directly with its command-line tools:

```bash
dot -Tsvg graph-rank-0.dot -o graph-rank-0.svg
```

The Average may combine edges never sampled together and may contain cycles.
Graph frequencies are not effect strengths, goodness-of-fit scores or evidence of
causal identification. MAP/fixed results show only the selected/supplied hypothesis,
not the pre-selection graph distribution. `graph_posterior()` remains the compact
top-15 frequency chart; use the inspector to select graphs beyond that cutoff.

The inspector is self-contained and uses no external assets or network service.
Notebook output runs in an isolated frame; notebook trust/JavaScript permission is
required for interaction. Exported HTML embeds graph topology, column labels,
counts/frequencies and display metadata, but not observations or contribution
draws. Review it as a result artifact before sharing.

Node labels preserve literal Unicode, quotes, backslashes and line breaks.
Characters that cannot appear in valid XML 1.0 SVG labels raise `ValueError`
when exporting DOT, creating a diagram or building the inspector, rather than
silently replacing the labels.

### Sampling on a GPU (Apple or NVIDIA)

Load the model on the GPU. The same calls then sample with fused kernels and need no
training, conversion or installed compiler; the first CUDA call in a process compiles the
kernels (see below):

```python
from pymc_mfm import from_pretrained
from pymc_mfm.inference import decompose

model = from_pretrained("pymc-labs/mfm", profile="river-1", device="mps")  # Apple silicon
model = from_pretrained("pymc-labs/mfm", profile="river-1", device="cuda")  # NVIDIA GPU
result = decompose(model, observations, n_samples=1000)  # independent array-level query draws
```

Historical query-only timings for River-1, 1,000 draws, a 166-week context, PyTorch 2.14, median
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
  drawn on a worker thread while the GPU runs; without `random_seed`, do not draw random
  numbers from other threads during a call. A seeded call draws from its own generator.
  CUDA sampling used to draw with the CUDA generator.
- Each Euler step runs one kernel per coordinate network that computes its first layer,
  its hidden product and its output without writing activations to memory: Metal on Apple
  GPUs; CUDA C++ on NVIDIA GPUs, compiled by NVRTC, which PyTorch's CUDA wheels include
  (no CUDA toolkit or compiler; found on every supported PyTorch, 2.6 and newer, in its
  CUDA 11.8 to 13 builds). Against PyTorch's GPU operations on the same GPU
  this is 2.7–7.2x faster at 50–205 query weeks and 2.3–3.3x at 10, where the CPU sets
  more of the pace (NVIDIA T4 to H100 and B200).
- The NVIDIA kernels run on the current CUDA device. Other devices, ROCm, float64 models,
  compiled sampling and GPUs on which any of the kernels fails to build (sampling warns
  once) use PyTorch's operations, in the smaller batches those need, with the same draws.
- A process's first CUDA sampling call compiles the kernels: about 1 s of NVRTC on a T4 or
  L4 (1.4 s with CUDA 11.8 builds), in background threads from the start of the call, so
  it overlaps the call's other first-use work. PyTorch 2.14's CUDA 13 build leaves the
  compiled kernels in the NVIDIA driver's cache (`~/.nv/ComputeCache`), so later processes
  on the machine skip it; with PyTorch 2.6's CUDA 12.4 and 2.7's CUDA 11.8 builds every
  process compiles.
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
  runners), PyTorch 2.14's MPS computes every linear layer with a bias wrongly (30–80%
  relative error), so sampling there raises an error instead of returning wrong draws;
  sample on the CPU in a VM.

### Faster repeated sampling (optional)

A long-running process that samples many times, such as a service, can compile the
sampler once with `torch.compile`. On an NVIDIA L4 with River-1 and 1,000 draws, warm
compiled calls ran about 1.4 times as fast as eager calls on four test inputs; the first
compilation took 20–30 s.

```python
from pymc_mfm import from_pretrained
from pymc_mfm.inference import decompose, warm_up_sampling

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
  float round-off but are not bit-identical. With a fixed seed, a call replays the same
  random numbers, including a call that compiles. Without a seed, sampling uses PyTorch's
  process-wide random generator, so run warm-up before other threads sample. While it
  compiles, PyTorch saves and restores that generator's state, so global draws that another
  thread makes in that time can repeat.
- PyTorch's compile cache (`TORCHINDUCTOR_CACHE_DIR`) lets a restarted process reuse
  earlier compilations. A compiler failure warns, returns the model to eager sampling and
  shows in `sampling_status.fallback`. `model.disable_compiled_sampling()` switches it off.
- On CPU, compiling usually does not speed sampling up.

## Call the River API

`pymc_mfm.client` is the Python client for the hosted River API (routes under `/v1`). It needs
no PyTorch. Access to the hosted service is by invitation. Create your API key on the Connect
page of the River API console, which also shows your service URL. Give the client the service URL
and the key as arguments or in `MFM_API_URL` and `MFM_API_KEY`. The client needs Python 3.11 or newer. Install it from PyPI:

```bash
pip install pymc-mfm
```

To read every draw (`.nc` files) or draw plots, run `pip install "pymc-mfm[client]"` instead.

[Get started](docs/guide/index.qmd) has a first script to copy and run.

The hosted River API keeps a dataset 7 days after upload, then deletes it and its analyses. Using
the dataset does not extend this.

An agent skill for the client is in [`skills/mfm/SKILL.md`](skills/mfm/SKILL.md).

```python
import numpy as np
import pandas as pd

from pymc_mfm.client import Client, amount, curve, grid, scenario

client = Client()  # reads MFM_API_URL and MFM_API_KEY

# Synthetic data: a weekly history of 104 weeks and a plan for the 8 weeks after it.
# Replace `history` and `plan` with your own DataFrames of the same form.
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
        "tv": 8_000.0,
        "search": 6_000.0,
        "price": 10.0,
    }
)

dataset = client.datasets.create(
    history,
    date="week",
    outcome="sales",
    spend=["tv", "search"],
    controls=["price"],
    name="brand-a",
    reuse=True,  # reuse a stored dataset only when its content hash matches exactly
)
print(dataset.history_ranges)  # per series; p05 and p95 are warning thresholds, not the data's edge

analysis = client.analyze(
    dataset,
    plan,  # the baseline plan for the weeks after the history, a DataFrame or a dict
    scenarios=[
        scenario("half", {"tv": 0.5, "search": 0.5}),  # a number multiplies a channel's spend
        scenario("tv_to_search", {"tv": 0.8, "search": amount(60_800)}),  # sets a total
        *grid({"tv": [0.5, 1.5], "search": [0.5, 1.5]}),
    ],
    decision_weeks=(1, 8),
    seed=7,
    name="Spend scenarios",  # optional label, shown on the site and in the export
)
analysis.wait()
analysis.url  # the result page for the person; None when the service has no site address
analysis.export("analysis.json")  # the whole analysis as one JSON file
analysis.table  # one row per plan and component; mean, p05, p95, window_mean, ...

# A backtest: read the input weeks, predict the weeks after them and score the forecast.
backtest = client.analyze(
    dataset,
    context=("2024-11-11", "2026-08-03"),  # first and last input week
    query_end="2026-11-02",  # last predicted week
    scenarios=[scenario("budget_x1.2", {"tv": 1.2, "search": 1.2})],
    curves=[curve("tv", [0, 0.5, 1.5, 2])],
).wait()
backtest.fit  # r2, mape and crps against the actual outcome, input and predicted weeks
backtest.shares  # each channel's and control's share of the expected outcome
backtest.curves  # each channel's contribution at each spend level

# More scenarios: a new analysis with the same inputs, seed and draws.
more = backtest.again([scenario("search_x1.5", {"search": 1.5})]).wait()
```

A scenario changes spend in a window: the predicted weeks of a backtest, or
`decision_weeks` of a plan. In `scenario(id, multipliers)` a number from 0 to 5 multiplies
a channel's spend there, and `amount(x)` sets its total there to `x`, keeping the weekly
pattern (a channel with no spend gets `x` split evenly). Channels left out keep their
spend. Scenarios do not change spend timing or controls. A backtest's `channels=` and
`controls=` choose series of the dataset (default: all); `fit`, `shares` and `curves` are
`None` for a plan. `again(scenarios, curves=None, explain_history=False, idempotency_key=None)`
repeats the dataset, weeks or plan, series, `decision_weeks`, seed and draws, and runs one
more baseline. It reads the baseline from the export, so it raises `NotReady` until the
analysis has ended: call `wait()` first. Compare each scenario only with the baseline of
its own analysis.

`client.explore(dataset, plan)` is a first look: a response curve for every channel at
nine spend levels (0 to 2 times the baseline in steps of 0.25) and the budget scenarios
`spend_x1.2`, `spend_x1.4` and `spend_x1.6`. It also takes `context=` and `query_end=` for
a backtest. Pass `preview=True` to `analyze` or `explore` to see the expanded plans, run
count and warnings without starting anything. `client.get(analysis_id)` resumes an analysis.

`client.datasets.list()` returns your stored datasets with `content_hash`, `created_at`,
`last_used_at`, `analyses` (the live analyses) and `expires_at`. Those fields are `None` when an
older service does not send them. `expires_at` is the deadline the service sets, so read it
instead of computing one. The content hash covers the dates, outcome, spend and controls,
not the name. `client.datasets.find(frame, ...)` and `create(..., reuse=True)` reuse a stored
dataset only when its hash equals that of the frame: never by name or by number of weeks.
`dataset_content_hash(body)` and `frame_content_hash(frame, ...)` compute the hash locally.

An `Analysis` has `status()`, `wait(timeout=, poll=)`, `table`, `history`, `fit`, `shares`,
`curves`, `weekly(scenario_id)`, `draws(scenario_ids=None)`, `plot_curves()`,
`plot_compare()`, `again(scenarios)`, `export(path)` and `delete()`.
It also has `name` and `url`. The plots mark plans that have warnings and print the warnings
under the chart. `draws()` reads
each plan's per-draw file into one `xarray.Dataset` stacked along `scenario`; if a signed
URL has expired it fetches the result again for new URLs and never reruns anything.

Read the numbers as the result's `labels` say, which the client keeps on the objects
(`analysis.labels`, and `attrs["labels"]` on tables). Intervals are pointwise 5th to 95th
percentiles of expected outcome and are not causal lift. `mean`, `p05`, `p95` and
`window_mean` are totals: `window_mean` is the expected total over the window, not a weekly
mean. The service gives no interval for a difference between scenarios, only the paired mean
difference `mean_difference`: `simulation_se` measures its numerical precision only, and
`plot_compare()` draws it as "simulation error, not model uncertainty".

Every request that creates something carries an `Idempotency-Key` derived from the request
content and reused when the client retries, so a retry or a repeated call never starts the
work twice; pass `fresh=True` to run an equal request again. If the earlier equal
request was deleted or expired, the service answers 410 and the client raises `Gone`: pass
`fresh=True` (or a new `idempotency_key=`) to run it again. `fresh=True` and `idempotency_key=`
cannot be combined, and `datasets.create(reuse=True)` takes neither. The client never picks
a new key on its own. `AnalysisStatus` also has `runs_cancelled`, and `Dataset` and
`AnalysisStatus` have `warning_details` (structured warnings); both are `0` or empty on an
older service. The client retries HTTP 429, 502, 503 and 504
and connection errors, and waits as long as `Retry-After` asks, up to `Client(max_retry_after=...)` seconds (default 60). Errors raise `NotFound`,
`Gone`, `Conflict`, `NotReady`, `PayloadTooLarge`, `RateLimited`, `ValidationError` and
others, each with `code`, `message` and `field` from the service. The code `steps_removed`
means that a scenario in the request has `steps`, which the service no longer accepts;
`scenario` never sends them.

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
from pymc_mfm.model import ModelConfig
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
from pymc_mfm import from_pretrained
from pymc_mfm.inference import save_pretrained
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
- `pymc_mfm.inference.push_to_hub(model, "org/repo", profile="my-model", license="...")` publishes
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
uv run --no-sync pytest -q -n 2 -m "not slow"
uv run ruff check .
uv run ruff format --check .
```

The fast command excludes tests marked `slow`. Plain `pytest` applies no marker filter,
so file and node selections retain their slow cases. CI runs both lanes. Use at most
`-n 2` on a small machine.

Install all extras before running the slow integration tests, which need the generation,
training and examples dependencies. Run these tests serially:

```bash
uv sync --locked --all-extras
uv run --no-sync pytest -q -m slow
```

After that sync, `uv run --no-sync pytest -q` runs the full suite, including slow tests,
serially.

The client tests need only the light install. To run them without PyTorch, as CI does:

```bash
uv sync --locked
uv run --no-sync pytest -q tests/client --confcutdir=tests/client
```

`benchmarks/inference_speed.py` times `decompose` with 1,000 draws of River-1 on
the quickstart data, read from the local Hugging Face cache (load River-1 once
while logged in). `--write-reference` saves the seeded draws of one version and
`--check-reference` compares another version's draws against them; a check fails on any
flipped graph by default. Write the reference with this script while the other version is
the one imported (for example `PYTHONPATH=<main checkout>/src`) and the same
`--query-weeks` or `--windows`.

With Conda or Mamba:

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --group dev -e .
```

### Documentation site

The website in `docs/` is a [Quarto](https://quarto.org) project: a landing page, the
Guide, the API reference, Models and the Blog. It needs the Quarto CLI (1.7 or newer;
CI pins 1.7.31) and the `docs` dependency group:

```bash
uv sync --locked --group docs
uv run --no-sync quarto preview docs   # live preview
uv run --no-sync quarto render docs    # writes docs/_site/
uv run --no-sync quarto render docs --profile pilot   # writes docs/_site-pilot/
```

`docs/_quarto.yml` holds the settings both sites share. The full site (the default
profile) takes its pages, menus and sidebars from `docs/_quarto-full.yml`. The pilot site
carries only the path of an invited tester (landing page, Get started, the client guide and
the client reference); `docs/_quarto-pilot.yml` lists it. Text for one site only sits in
`::: {.content-hidden when-profile="pilot"}` or `::: {.content-visible when-profile="pilot"}`
blocks.

Before each render, `docs/_scripts/prepare.py` copies `examples/*.ipynb` into the Guide
and writes the API reference from the docstrings (quartodoc). Both are generated and
ignored by Git.

Guide code samples scroll inside the code body, keeping the language label and Copy
button fixed. HTTP API samples wrap within their narrower panels instead.

| Section | Source |
| --- | --- |
| Guide | `docs/guide/*.qmd`; listed in `website.sidebar` in `docs/_quarto-full.yml` |
| Guide examples | every notebook in `examples/`, with its stored outputs |
| API reference | docstrings; pages and order in the `quartodoc` section of `docs/_quarto.yml` |
| Models | one page per released model: `docs/models/<name>/index.qmd`, with `name` (the model without its version), `order`, `badge`, `specs` and `image` in its front matter for the Models page card; announced models (cards that say "Coming soon" and do not link) in `docs/models/upcoming.yml`. Card art lives in `docs/assets/img/models/`. List each model in the Models menu in `docs/_quarto-full.yml` (a text-only `"<Name> SOON"` item for an announced one) |
| Blog | one folder per post: `docs/blog/posts/<date>-<slug>/index.qmd` or `index.ipynb` |

Nothing executes at build time: notebooks and posts render the outputs they were saved
with. Prefer SVG figures (text files). A PNG or JPEG is binary and needs an entry in
`.github/public-boundary-allow.toml`, pinned with `sha256`, like a notebook with image
outputs. The Docs workflow builds the site on every pull request; publishing to GitHub
Pages is a manual run of that workflow on `main`.

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
