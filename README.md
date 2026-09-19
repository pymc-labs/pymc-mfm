# PyMC MFM

PyMC MFM is being developed as a standalone library for training and using
**graph-conditional Marketing Foundation Models**.

**Current status:** `0.0.1.dev0` includes the complete trainable graph-conditional
model, its five observation feature blocks, one model configuration, local
model save/load, and an adapter for public `pymc-generator` corpora. A training
runner (#10), raw-observation inference (#11), diagnostics/plots (#14), Hugging
Face transport (#15), and MLflow tracking (#16) remain separate
[implementation stages](https://github.com/pymc-labs/pymc-mfm/issues/6).

## Structure

```text
src/pymc_mfm/
  model/         Canonical architecture and its neural-network components
  data/          Observation preprocessing, corpus generation and loading
  training/      Training objectives, optimization and checkpoint writing
  inference/     Checkpoint loading and observation-based inference APIs
  diagnostics/   Numerical diagnostics, calibration and evaluation metrics
  plotting/      Figures and reports, separate from numerical diagnostics
tests/           Focused behavioral and equivalence tests
examples/        Small examples using the public API
pyproject.toml   Package metadata, dependencies and development-tool settings
uv.lock          Locked dependencies for the uv workflow
environment.yml  Conda Python environment
```

The model contains a causal bi-attention encoder, a joint contribution/graph
router, a graph conditioner, a support-projected conditional contribution flow,
and the trainable extended structure head. Both flow heads retain coherent
trajectory/context-state modules. The full objective scores query and context
decomposition, context graph structure, graph-conditioned query/context
decomposition, and auxiliary structure prediction.

Base runtime dependencies are NumPy and PyTorch only. Base imports do not load
PFN, plotting, tracking, generation, or network-authentication packages. The
`generation` extra installs public `pymc-generator` and its released modeling
dependencies; the `dev` group provides pytest and Ruff.

Model weights, corpora, and generated experiment outputs do **not** belong in the
library. They remain external, on Hugging Face or
[Google Drive](https://drive.google.com/drive/folders/1G3jcruRo3PVq_mE_6Lvoiobfz2smc8Mg).

## Model configuration and local artifacts

```python
from pymc_mfm import MFM, ModelConfig, load_model, save_model

model = MFM(ModelConfig())  # Fresh, untrained weights; the A/B architecture.
save_model(model, "model.pt")
restored = load_model("model.pt", device="cpu")  # Evaluation mode.
```

`ModelConfig` is the direct construction input, not a preset/override resolver.
Its defaults are the shared A/B dimensions and active objective weights.
`to_dict()` emits every field and a format version; `from_dict()` refuses missing,
unknown, or incompatible fields instead of silently filling defaults. Save/load
uses that same configuration and strictly restores all trained module groups,
including the auxiliary head. Tensor dtype is preserved. Optimizer state and
run bookkeeping are not included in these model artifacts.

Architecture semantics are fixed: all five feature blocks, causal attention,
the eight graph families in `cy, dc, dz, db, zb, zc, cc, zz` order, zero atoms,
support projection, and the complete active training path. No historical
architecture selectors or optional-head flags are exposed. Paths, device, seed,
optimizer, corpus selection, and run budget stay outside `ModelConfig`.

`pymc_mfm.data.profiles.PROFILE_A` and `PROFILE_B` identify external reference
checkpoints and corpora only. A used `fw53_d100_r1` (200,000 worlds, five live
channels); B used `fw53cy410_d200_r0` (400,000 worlds, four to ten live channels).
Both use `ModelConfig()`. Profile checksums identify the **original** checkpoints,
not files rewritten by `save_model`. The loader accepts the canonical MFM format
only; there is no automatic PFN-format fallback.

### Tensor-level training and inference

The implemented model consumes an already prepared tensor dictionary. Numeric
inputs, including padded slots, must be finite; NaN/Inf are not padding sentinels.

| Batch fields | Shape |
| --- | --- |
| `ctx_spend_z`, `qry_spend_z` | `[B,Tcontext,K]`, `[B,Tquery,K]` |
| `ctx_controls_z`, `qry_controls_z` | `[B,Tcontext,M]`, `[B,Tquery,M]` |
| `ctx_revenue_z`, `ctx_week`, `qry_week` | `[B,Tcontext,1]`, `[B,Tcontext,1]`, `[B,Tquery,1]` |
| `spend_mean`, `spend_std` | `[B,K]` |
| `response_moments`, `adstock_moments` | `[B,K,4]`, `[B,K,9]` |
| `control_moments`, `control_location_scale`, `control_effect_moments` | `[B,M,3]`, `[B,M,2]`, `[B,M,6]` |
| `active_c_mask`, `active_m_mask`, `active_j_mask` | Binary `[B,K]`, `[B,M]`, `[B,J]` padding masks |
| `targets_z`, `context_targets_z` (training only) | `[B,Tquery,D]`, `[B,Tcontext,D]` |
| `g_cy`, `g_dc`, `g_dz`, `g_db`, `g_zb`, `g_zc`, `g_cc`, `g_zz` (training only) | Unpacked binary edge-family arrays |
| `sales_scale` (corpus adapter metadata) | `[B]`, for restoring natural target units |

Here `D = 1 + K + M + 4`: intrinsic baseline, K media contributions, M control
contributions, total confounder contribution, then indirect `cc`, `zc`, `dc`.
Every decomposition coordinate is natural value divided by
`max(p95(abs(context sales)), 1e-6)`; intrinsic baseline excludes observation
noise. Week coordinates are zero-based `t / T` over the complete series.
The corpus adapter below produces this representation. A raw-observation
inference adapter remains separate work.

`pymc_mfm.data.features.moment_features(context, K, M)` computes the five
feature blocks from natural `spend`, **log** natural-sales `revenue`, and raw
`controls` arrays in the observed context only. Its outputs are unbatched;
callers retain the padding masks. It is not a complete observation adapter.

`model.loss(batch)` returns the full differentiable training objective;
`model.loss_terms(batch, return_trace=True)` also exposes executed objective
values and structure-family reductions. `model.sample_disentangled(batch, S)`
returns decomposition draws `[S,B,Tquery,D]` and their conditioning graphs
`[S,B,L]`. Modes are `model_average` (router-drawn graphs), `map` (empirical
**complete-graph** mode), and `fixed` (explicit binary graphs). These inference
paths never use the `g_*` labels or decomposition targets. `window="context"`
selects the trained context marginal; separate context/query calls are **not**
paired full-horizon draws. `model.predict` and `model.sample_context` expose the
joint router marginals rather than the conditional decoder.

### Generating and preparing a corpus

Install the optional public generator from its `main` branch:

```bash
uv sync --locked --extra generation
uv run --locked --extra generation python
```

Or install `.[generation]` with pip in your isolated environment. The Git URL
is declared in package metadata, so it works without uv-specific overrides.
`pyproject.toml` tracks `main`; `uv.lock` records the exact tested commit,
[`ce068d5`](https://github.com/pymc-labs/pymc-generator/commit/ce068d598fc0438b511c265ef09d7ba33354cb48).
Update deliberately with `uv lock --upgrade-package pymc-generator`, then rerun
the generation checks. Installation requires Git/network access; adapting an
already loaded corpus or running model inference does not.

```python
import numpy as np
import pymc_generator as pg

from pymc_mfm import MFM, ModelConfig
from pymc_mfm.data.corpus import corpus_batch

prior = pg.make_scm_prior(
    n_treatments=3,
    n_covariates=2,
    n_latent=1,
    n_treatments_active_range=(2, 3),
    n_covariates_active_range=(1, 2),
    n_time_steps=16,
    l_max=4,
    carryover_burn_in=0,
    query_frac=0.5,
    n_cells=2,
    draws_per_cell=2,
    seed=42,
)
corpus = pg.DataGenerator(prior).generate()  # Includes generator validation.
train_rows = np.flatnonzero(corpus["is_val"] == 0)
batch = corpus_batch(corpus, rows=train_rows)
model = MFM(ModelConfig(n_controls=2, n_demand=1))
loss = model.loss(batch)
loss.backward()
```

Use `pg.save_corpus` / `pg.load_corpus` for storage and the generator's
`DataGenerator.validate_corpus` for full corpus validation after loading.
`corpus_batch` checks the consumed arrays of selected rows only. It does not
choose a train/validation split, resample worlds, or silently truncate time.
For corpora with different support lengths, group selected rows by
`corpus["support_mask"].sum(axis=1)` before batching. Preserve the generator's
cell-level `is_val` split. Match the padded latent count to `ModelConfig.n_demand`.

The adapter consumes schema **4** and its domain-neutral array names. Upstream
`dy`/`zy` graph slots retain their positions as the model's internal `db`/`zb`
checkpoint labels; they represent effects on the outcome, not on the intrinsic
intercept. Decomposition targets exclude `outcome_noise`; their ordered
`cc, zc, dc` indirect split is the generator's attribution convention, not an
identified causal decomposition. Draws return to natural units with
`draws * batch["sales_scale"][None, :, None, None]`.

All normalization statistics and moment features use observed context only.
The A/B log representation requires positive observed outcomes, nonnegative
treatments/direct contributions, and at least one active treatment and
covariate. Unsupported worlds are rejected, not clipped into that domain.
Other decomposition terms can be signed. The canonical covariate zscore
clipping at `[-10, 10]` is retained. Do not substitute generator-normalized
arrays or `outcome_scale` for the model's context-based representation.

### Reference and reproducibility boundaries

The extraction source is PFN PR #113 at
[`52687cf25cca9be13094a397fabdf016367c0a52`](https://github.com/pymc-labs/pymc-pfn/tree/52687cf25cca9be13094a397fabdf016367c0a52).
The original A/B checkpoint state dictionaries load without key remapping.
Local verification against both originals covered strict loading, encoder
outputs, active losses, every parameter gradient, all three conditional graph
modes in both windows, and canonical artifact round-tripping. These checks do
not establish full-corpus evaluation or retraining equivalence.

The port also rejects incompatible trajectory-feature markers when loading
weights. A reproduced source edge case in single-week/constant trajectory
normalization was corrected: the zero-variance branch now has finite gradients
without changing ordinary-input outputs.

The optional generation stack resolves **published**
[`pymc-marketing==1.1.0`](https://pypi.org/project/pymc-marketing/1.1.0/),
`pymc==6.2.0`, `pytensor==3.2.4`, and `numpy==2.4.6`. PyMC-Marketing is not
a Git dependency. Its named-dimension adstock/saturation API and coexistence
with MFM have been exercised; do not restore the old PyMC 6.0.1/PyTensor 3.0.7
pins. Reference-fitting tools remain outside base inference.

Public `pymc-generator` is integrated directly from `main`; no PyPI publication
is required. Verification generated four worlds with varying active dimensions,
validated and round-tripped their corpus, reconstructed their natural outcome
from targets plus observation noise, and checked graph-slot mapping. It also
exercised gradients through all five trained module groups, an optimizer step,
all three conditional modes in both windows, and exact model-artifact replay.
This proves the software/data boundary, **not** equivalence of the new generator
distribution to the original A/B corpora, recovery calibration, or retraining.

The complete diagnostic/report/plot suite remains required release work in #14,
not an optional omission. Its eventual dependencies must stay out of base
imports, with lazy plotting imports, headless rendering, optional external
renderers, and separate reference-fitting tools. HF will transport the local
model format; MLflow will consume existing training/diagnostic records. The
training integration must provide default-on tracking, an explicit opt-out and
a local backend. Neither integration nor a hosted service is needed here.

## Development environments

Python **3.13 or newer** is the support baseline. Both setup paths below default
to Python 3.13 and keep their environments inside the checkout. Run commands from
the repository root and choose one environment workflow.

### uv

[Install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked
uv run python -c "import pymc_mfm"
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

`uv sync` installs the package in editable mode and includes the `dev` dependency
group in `.venv`. When deliberately changing dependencies, edit `pyproject.toml`,
run `uv lock`, and commit the updated lockfile.

### Conda / Mamba

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m pip install --group dev -e .
python -c "import pymc_mfm"
ruff check .
ruff format --check .
```

Mamba can replace Conda in the environment-creation command. Pip 25.1+ supports
`--group dev`, so package and development dependencies remain declared only in
`pyproject.toml`. This workflow resolves those declarations with pip; it does not
consume `uv.lock`. Do not install project dependencies into Conda's `base`.

The regression suite covers complete-objective gradients, truth isolation,
graph support and draw alignment, complete-graph mode, strict configuration and
weight loading, and dtype-preserving artifact round trips. It requires no
external checkpoints, corpora or hosted services.
