# PyMC MFM

PyMC MFM is a standalone library for training and using
**graph-conditional Marketing Foundation Models**.

**Current status:** `0.0.1.dev0` includes the complete trainable
graph-conditional model, shared context-only observation preprocessing,
canonical corpus recipes and training, conditional decomposition results,
diagnostics and saved-evidence reports, strict model bundles, optional
Hugging Face transport, and default-enabled optional MLflow tracking.

## Structure

```text
src/pymc_mfm/
  model/         Canonical architecture and its neural-network components
  data/          Observation preprocessing, corpus generation and loading
  training/      Training objectives, optimization and checkpoint writing
  inference/     Checkpoint loading and observation-based inference APIs
  diagnostics/   Numerical diagnostics, calibration and evaluation metrics
  plotting/      Figures and reports, separate from numerical diagnostics
tests/           Focused behavioral tests
examples/        Small examples using the public API
pyproject.toml   Package metadata, dependencies and development-tool settings
uv.lock          Locked dependencies for the uv workflow
environment.yml  Conda Python environment
```

The model contains a causal bi-attention encoder, a joint contribution/graph
router, a graph conditioner, a support-projected conditional contribution flow,
and the trainable extended structure head. Both flow heads include coherent
trajectory/context-state modules. The full objective scores query and context
decomposition, context graph structure, graph-conditioned query/context
decomposition, and auxiliary structure prediction.

Base runtime dependencies are NumPy, PyTorch, safetensors and packaging.
Base inference does not import generation, plotting, tracking, or Hub clients.
Install `[diagnostics]` for the complete numerical/plotting suite, `[hub]` for
remote artifact transport, `[training]` for diagnostics and MLflow, and
`[generation]` for public `pymc-generator`. The `dev` group supplies pytest and Ruff.

Model weights, corpora, and generated experiment outputs are external artifacts,
not bundled with the library. Supply a known local model directory or an explicit
Hub repository and revision; no pretrained model is selected automatically.

## Model configuration and local artifacts

```python
from pymc_mfm import MFM, ModelConfig, from_pretrained, save_pretrained

model = MFM(ModelConfig())  # Fresh, untrained weights.
save_pretrained(model, "model")  # New directory; existing bundles are immutable.
restored = from_pretrained("model", device="cpu")  # Offline, evaluation mode.
```

`ModelConfig` is the direct construction input, not a preset/override resolver.
It holds only model size (`d_model`, `n_layers`, `n_heads`, `d_ff`, `flow_hidden`)
and training geometry (`n_controls`, `n_demand`); its defaults are the A/B
network. `to_dict()` emits every field and `format_version` 2; `from_dict()`
refuses missing, unknown, or incompatible fields instead of silently filling
defaults, and version-1 payloads (which carried flow settings and loss weights)
are rejected. Save/load uses that same configuration and strictly restores all
trained module groups, including the auxiliary head. Tensor dtype and parameter
trainability are preserved.
The directory contains `config.json`, `model.safetensors`, and `manifest.json`,
with ordered schemas, full configuration, provenance, compatibility requirements,
tensor descriptions and file/bundle checksums. Missing, reordered, incompatible
or corrupt state is rejected. There is no pickle or remote-code loading fallback.
A bundle loads where the installed packages satisfy the installed release's own
compatibility constant (`numpy<3`, `torch<3`, `safetensors<1`, the package's
dependency ranges; a later release may relax them) and every lower bound the
bundle records. The upper bounds it records state the producing release's tested
range and are not enforced, so a published bundle never becomes permanently
unloadable. Its recorded ranges may use only `>=`, `>`, `<` and `<=`, the
producer versions it records must be release versions within them, and schemas
and inference semantics must match exactly. A release version is canonical
PEP 440 text, pre-, post- and dev-releases included, without a local label. Only
torch may carry one, and only a build label: `cpu`, `cuN`, `rocmN[.N...]` or `xpu`,
optionally followed by `.cxx11.abi`, `.git<commit>` or `.lw.git<commit>` as in the
CPU cxx11-ABI and AMD ROCm wheels (`2.7.1+cpu.cxx11.abi`,
`2.7.1+rocm7.0.0.git56b79e13`), or the commit of a source or NVIDIA NGC build,
optionally with a `git` prefix or an `.nvYY.MM` suffix (`2.9.0a0+git5228986`,
`2.8.0a0+5228986c39.nv25.5`); a commit is 7-40 hex digits. Exports record
versions as `packaging` normalizes them (`nv25.05` as `nv25.5`), and the
installed Python version comes from `sys.version_info`. `save_pretrained`
refuses to export from an environment outside the constant or whose versions
are not release versions, and `train` runs the same check before it writes
anything, rather than at its terminal export. An environment inside the
constant loads bundles either way.
Optimizer/RNG state is absent: model exports are not training-resume archives.

Architecture semantics are fixed: all five feature blocks, causal attention,
the eight graph families in `cy, dc, dz, db, zb, zc, cc, zz` order, zero atoms,
support projection, the trajectory-coherent flow, and the complete active training
path. The flow numerics are module constants in `pymc_mfm.model`, not fields:
`FLOW_N_FREQ = 2` Fourier time frequencies, `FLOW_STEPS_SAMPLE = 16` Euler steps
and `FLOW_TRAJECTORY_CORRELATION = 0.98`. No architecture selectors or
optional-head flags are exposed. Paths, device, seed, optimizer, loss weights,
checkpoint selection, corpus selection, and run budget stay outside `ModelConfig`
and belong to the training recipe (`TrainingConfig`).

`pymc_mfm.data.profiles.PROFILE_A` and `PROFILE_B` are named corpus
profiles. They describe data metadata only and never modify `ModelConfig`.
`from_pretrained` accepts strict MFM directory bundles locally or retrieves
that same format through the Hub extra.

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
The corpus and observation-only adapters share this exact numerical representation.
`observation_batch` requires no generating graph, simulation truth or dummy targets.

`pymc_mfm.data.features.moment_features(context, K, M)` computes the five
feature blocks from natural `spend`, **log** natural-sales `revenue`, and raw
`controls` arrays in the observed context only. Its outputs are unbatched;
callers retain the padding masks. It is not a complete observation adapter.

`model.loss(batch, weights=LossWeights())` returns the full differentiable training
objective; `model.loss_terms(batch, weights=..., return_trace=True)` also exposes
executed objective values and structure-family reductions. `LossWeights` (exported
from `pymc_mfm.model`) defaults to the A/B weights: `struct_weight=1`,
`struct_cy_weight=1`, `graph_conditional_weight=1`,
`context_decomposition_weight=0.5`, `context_graph_weight=0.1`.
`model.sample_disentangled(batch, S)` returns decomposition draws `[S,B,Tquery,D]`
and their conditioning graphs `[S,B,L]`. `S` defaults to `DEFAULT_N_SAMPLES = 32`
here and in `predict`, `sample_context`, `sample` and `decompose`; it must be a
positive integer (booleans and fractions are rejected). Modes are
`model_average` (router-drawn graphs), `map` (empirical
**complete-graph** mode), and `fixed` (explicit binary graphs). These inference
paths never use the `g_*` labels or decomposition targets. `window="context"`
selects the trained context marginal; separate context/query calls are **not**
paired full-horizon draws. `model.predict` and `model.sample_context` expose the
joint router marginals rather than the conditional decoder.

### Generating and preparing a corpus

Install the optional public generator with the project's locked dependencies:

```bash
uv sync --locked --extra generation
uv run --locked --extra generation python
```

Or install `.[generation]` with pip in your isolated environment. Package metadata
declares `pymc-generator` from
`git+https://github.com/pymc-labs/pymc-generator.git@main`, so no uv-specific
override is needed. The dependency follows `@main`; `uv.lock` records the
resolved commit used by `uv sync --locked`. Pip does not consume `uv.lock` and
resolves the branch when installing. Refresh the locked revision deliberately
with `uv lock --upgrade-package pymc-generator` and rerun the generation/workflow
checks before accepting the updated lockfile. `tests/test_generator_drift.py`
regenerates small A, B and evaluation probes from the shard plans and compares
every array against stored hashes, so a refreshed generator that changes the A/B
priors fails the suite; a failure names the generator commit the hashes were
recorded with and the installed one. The weekly `Generator drift` workflow runs
the same test against the branch itself, before anyone refreshes the lock. After
an intended generator change, re-record with
`python -m tests.test_generator_drift --write --force`. Installation requires
Git/network access; adapting an already loaded corpus or running local model
inference does not.

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

The adapter consumes schema **4** and its domain-neutral array names. The
generator's `dy`/`zy` graph slots retain their positions as the model's internal
`db`/`zb` checkpoint labels; they represent effects on the outcome, not on the
intrinsic intercept. Decomposition targets exclude `outcome_noise`; their
ordered `cc, zc, dc` indirect split is the generator's attribution convention,
not an identified causal decomposition. Draws return to natural units with
`draws * batch["sales_scale"][None, :, None, None]`.

All normalization statistics and moment features use observed context only.
The model's log representation requires positive observed outcomes, nonnegative
treatments/direct contributions, and at least one active treatment and
covariate. Unsupported worlds are rejected, not clipped into that domain.
Other decomposition terms can be signed. The canonical covariate zscore
clipping at `[-10, 10]` is retained. Do not substitute generator-normalized
arrays or `outcome_scale` for the model's context-based representation.

### Canonical corpus recipes

`pymc_mfm.data.recipes.PROFILE_A_PRIOR` is profile A's canonical public
`pymc-generator` configuration. It sets the direct-effect edge budget to
`edge_budget["cy"] = (5, 5)`; `recipe_for(PROFILE_B)` uses `(4, 10)`.
These are synthetic training-prior settings, not counts inferred from observed
columns.

`pymc_mfm.data.recipes.SHARD_PLANS` stores the shard plans as data. Every shard
`i` has seed `plan.seed(i)`, and its world count is `n_cells × draws_per_cell`:

| Plan | Profile | Shards | Seeds | Cells × draws per shard | Worlds |
| --- | --- | --- | --- | --- | --- |
| `A` | A | 160 | 530200–530359 | 125 × 10 | 200,000 |
| `B` | B | 320 | 530400–530719 | 125 × 10 | 400,000 |
| `evaluation` | A | 10 (`eval_a06`–`eval_a15`) | 540000–549000, step 1000 | 32 × 4 | 1,280 |

Evaluation shard `i` fixes exactly `6 + i` active channels
(`n_treatments_active_range = (6 + i, 6 + i)`); everything else is profile A's
prior. Generate one planned shard with:

```bash
mkdir -p corpora/A corpora/evaluation
uv run --locked --extra generation python -m pymc_mfm.data generate \
  --profile A --shard 0 --out corpora/A/
uv run --locked --extra generation python -m pymc_mfm.data generate \
  --evaluation-shard 3 --out corpora/evaluation/
```

`--out` is either an existing directory, which receives `shard_<seed>.npz`, or a
`.npz` path (missing parent directories are created). An existing target is
refused unless `--force` is given. Arguments, prior settings and write access
are all checked before generation starts, and the corpus and its sidecar are
written atomically. `--shard` excludes `--n-cells`,
`--draws-per-cell`, `--seed` and `--override`; `--evaluation-shard` also excludes
`--profile`. A custom corpus still takes explicit settings:

```bash
uv run --locked --extra generation python -m pymc_mfm.data generate \
  --profile A --n-cells 60 --draws-per-cell 4 --seed 20260923 \
  --out corpora/profile_A_small.npz
```

Each write adds a `<path>.prior.json` sidecar with the complete `SCMPrior`,
generator version, and timestamp. `load_corpora` accepts compatible schema-4
files; `provenance` records their path, SHA-256, split counts, context lengths,
and any sidecar in the run manifest.

## Training

`pymc_mfm.training.train` is the single entry point; `python -m pymc_mfm.training`
wraps it. `TrainingConfig()` defines the canonical full-run configuration; corpus choice
is supplied separately:

| Setting | Canonical value |
| --- | --- |
| Updates / batch | 62,500 AdamW updates of 64 worlds (`budget == "canonical"`) |
| Learning rate | peak 2.8e-3, linear warmup over 3,125 updates, cosine decay to exactly 0 |
| Optimizer | AdamW, weight decay 0.01, betas (0.9, 0.999), eps 1e-8, global grad-norm clip 1.0 |
| Seed | 31 for weight init and corpus order; 616161 for evaluation |
| Evaluation | every 6,250 updates and at the final update, 16 held-out worlds, 128 draws |
| Checkpoints | `checkpoint_last.pt` at every evaluation, milestones at 25,000 and 50,000 |
| Loss weights | `struct_weight=1`, `struct_cy_weight=1`, `graph_conditional_weight=1`, `context_decomposition_weight=0.5`, `context_graph_weight=0.1` (`TrainingConfig.loss_weights`) |
| Checkpoint selection | `selection_head="conditional"` (or `"joint"`) |

Every `TrainingConfig` field, including the five loss weights and
`selection_head`, is adjustable and has a matching `python -m pymc_mfm.training`
flag. Model size and geometry live in `ModelConfig`; nothing else about the
architecture is configurable.

Any other `steps`/`batch_size` is a **reduced** budget on the same code path
and is recorded as such. A reduced run should also shrink `warmup_steps`,
`eval_every` and `milestone_steps`:

```bash
uv run --locked --extra generation --extra training python -m pymc_mfm.training \
  --corpus corpora/profile_A_small.npz --out-dir runs/reduced_A \
  --steps 200 --warmup-steps 10 --eval-every 100 --milestone-steps 150 \
  --eval-samples 32 --log-every 10 --device auto --tracking-uri sqlite:///runs/mlflow.db
```

```python
from pymc_mfm.data.recipes import load_corpora, provenance
from pymc_mfm.training import TrackingConfig, TrainingConfig, load_checkpoint, train

loaded = load_corpora(["corpora/profile_A_small.npz"])
result = train(
    [item.corpus for item in loaded],
    out_dir="runs/reduced_A",
    training=TrainingConfig(steps=200, warmup_steps=10, eval_every=100, milestone_steps=(150,)),
    corpus_provenance=[provenance(item) for item in loaded],
    tracking=TrackingConfig(tracking_uri="sqlite:///runs/mlflow.db"),
)
model, payload = load_checkpoint("runs/reduced_A/checkpoint_last.pt")
```

The architecture defaults to `ModelConfig()` sized to the corpus controls and
latents; a supplied `ModelConfig` must match the corpus geometry. Corpora are
consumed in epochs: each epoch permutes all `is_val == 0` worlds, buckets them
by context length, and serves whole batches from one bucket at a time, chosen
in proportion to the rows it still holds; leftover rows that cannot fill a
batch are dropped at the reshuffle. `is_val == 1` worlds are never trained on
and supply the held-out evaluation leg. Batches are encoded by one background
thread (`prefetch_batches`) and moved to the device by the training thread. The
epoch budget divides by the rows that fit into whole batches; `rows_per_epoch`
and `dropped_rows_per_epoch` expose the difference from the total training
rows. Consumed counters include completed optimizer updates only, not prefetched
or interrupted batches.

### Run outputs

| File | Content |
| --- | --- |
| `run.json` | run id, status (`running` -> `finished`/`failed`/`interrupted`), environment, resolved model/training/optimizer configuration and objective weights, per-module `parameter_counts` (`encoder`, `router`, `structure_head`, `graph_conditioner`, `conditional_head`, `total`), corpus provenance and `corpus_summary` (per-split worlds, cells, distinct graphs, context-length and active-channel/control/latent histograms, per-family edge counts, graphs shared between splits, generator diagnostics as min/max), consumption budget in epochs, evaluation and selection policy, checkpoint list with SHA-256, persisted best step, exact consumed updates/rows/distinct worlds |
| `objective_trace.json` | the first update's executed objective trace (all terms, weights, structure-family normalization) |
| `metrics.jsonl` | one record at update 1 and every `log_every` updates: `loss`, `flow_loss`, `context_flow_loss`, `graph_conditional_flow_loss`, `graph_conditional_context_flow_loss`, `task_loss`, `struct_loss`, `lr`, pre-clip `grad_norm`, `grad_clipped`, epoch, consumed rows, batch context length, timings and throughput |
| `evaluations.jsonl` | one record per evaluation event with the `holdout` and `seen` legs, a flat `{leg}/{head}/{window}/{metric}` view and the selection decision |
| `evaluation_tables/step_<N:09d>.json` (e.g. `step_000006250.json`) | per-edge-family reliability tables (the non-empty rows of 10 equal-width probability bins) and threshold tables (0.1, 0.25, 0.5, 0.75, 0.9) for the structure head and the joint router, written before the matching `evaluations.jsonl` line |
| `corpus_support.json` | context-outcome mean and standard deviation of each world (`"<corpus>:<row>"`, both splits), at most 5,000 evenly spaced worlds, with the total in `n_worlds` and kept worlds with an empty or non-finite context, or with overflowing statistics, in `skipped`; written at the start and never logged to MLflow |
| `checkpoint_last.pt`, `checkpoint_best.pt`, `checkpoint_step_<N>.pt` | atomic training checkpoints (`load_checkpoint`): `config`, `state_dict`, training configuration, run identity and provenance, consumption, the accompanying evaluation record and selection |
| `model/` | complete terminal safetensors/JSON model export; training checkpoint and exported tensor hashes remain distinct |

A nonfinite loss or gradient norm at a logging update fails the run with the
reason in `run.json`; interrupted runs are recorded as `interrupted`. Nothing
is written that claims completion for an unfinished run. Optimizer state is not
stored and no resume path exists.

### Evaluation readouts and checkpoint selection

Each evaluation event scores the same fixed held-out worlds (and the first
worlds the optimizer has seen) under an isolated CPU and evaluation-device RNG.
Evaluation does not advance the training RNG; accelerator kernel determinism
remains a PyTorch/device constraint. Four readouts are recorded separately:

| Head | Model call | What it is |
| --- | --- | --- |
| `joint` | `predict` / `sample_context` | the joint router marginal |
| `conditional` | `sample_disentangled(mode="model_average")` | the graph-conditional decoder used by inference |
| `structure_head` | `structure_logits` | auxiliary BCE edge probabilities, pooled over active slots |
| `joint_router` | graph draws of the joint head | edge marginals plus whole-graph posterior statistics |

Decomposition metrics are computed in the training z-space for both heads and
both windows. `latent_sales_r2` and closure compare natural units against
`outcome - noise`:

| Family | Metrics |
| --- | --- |
| Component recovery | `*_scaled_mae`, `*_crps`, `*_cov80` per component family, `closure_rel_err`, decomposition / latent-sales / contribution energy scores |
| Live/dead attribution | media live when `cy = 1`, controls live when `zb = 1`: `attribution_live_{media,control}_{crps,mean_bias,mean_abs_error,demeaned_rmse}`; dead-effect leakage on the posterior mean and per draw, `attribution_dead_{media,control}_{leakage,draw_leakage}_scaled_mae` |
| Per-draw closure | `closure_draw_rel_err` (mean over draws), `closure_draw_q90_rel_err` |
| Temporal coherence | `contribution_truth_lag1`, `contribution_draw_lag1`, `contribution_lag1_tracking_abs_error`, `contribution_truth_error_residual_lag1`, `contribution_posterior_centered_residual_lag1`, `contribution_adjacent_draw_correlation` |
| Window totals | `timeavg_contribution_cov80` (all, `_live`, `_dead` channels), `contribution_cumulative_energy_score`, `latent_sales_r2` |
| Padding | `contribution_padding_leakage_scaled_mae` (padded slots only; distinct from dead-effect leakage) |

`structure_head` and `joint_router/edge` report `log_score`, `brier`, `auroc`,
`auprc`, `ece`, `null_fpr` (share of negative edges with p > 0.5), `prevalence`,
`n` and `n_pos` per edge family, pooled over active slots. `joint_router/graph`
reports `shd_mean`, `shd_median`, `normalized_shd_mean`, `exact_recovery`, the
graph `energy_score`, `categorical_brier_score`, `entropy`, `n_distinct`,
`cc`/`zz` two-cycle and cycle rates (the prior forbids cycles, so non-zero rates
flag invalid graphs), and `pairwise_comoment_abs_err` /
`pairwise_comoment_max_abs_err` (dependence between edge draws). `{leg}/counts/*`
records mean active channels and controls and the number of worlds.
`result.diagnostics(...)` reports the same whole-graph statistics for one result
as `graph/<key>`.

Each metric is a mean over worlds; slot families (per-channel contributions,
temporal coherence and time-average coverage) average a world's defined slots
first, and edge scores pool active slots across worlds.
`pymc_mfm.training.metric_aggregation()` names the class of every key: `world`
(mean over worlds), `slot` (per-world slot means averaged over worlds) or
`pooled_edges`. A recorded training evaluation may also label a slot key
`slot_pooled` (slots pooled across worlds, as in the A/B records). Metrics
undefined for a world are dropped and named in `undefined` with their reason; a
metric undefined everywhere is `null`, never zero. Each leg also records
`metric_worlds` (worlds behind every metric) and `tables`, and
`evaluate_rows(..., per_world=True)` adds per-world values and slot weights.
These stay in the JSON records and are not logged as MLflow metrics. Nonfinite
active edge probabilities make the pooled predictive scores `null` with the
affected-slot count; truth counts and prevalence remain available. Nonfinite
graph draws are marked unavailable rather than thresholded into apparently
valid binary graphs. Held-out − seen gaps, share MAEs and mean AUROC/ECE are
report-only (`TrainingRun.derived_metrics()` and the diagnostic reports).

Checkpoint selection follows `TrainingConfig.selection_head` (the conditional
decoder by default). It minimizes `holdout/{head}/query/decomposition_energy_score`.
Promotion also requires the same head's context energy score
(`holdout/{head}/context/decomposition_energy_score`), the auxiliary `cy` log
score and the head's padded-channel leakage (which must be zero) to be finite,
fully defined and within `METRIC_BOUNDS`, with a 1e-12 slack for rounding; the
leakage gate passes vacuously when no held-out world has padded channels, and
says so. An out-of-bounds gate blocks promotion as `metric_outside_bounds`. Any
held-out metric of the selected head that dropped worlds with non-finite inputs
(draws or truth, `nonfinite_in_<n>_worlds`) also blocks promotion, gated or
not, and is listed in `selection.nonfinite_keys`; out-of-bounds values of other
metrics are listed in `selection.bounds_violations`. `run.json`'s
`selection_policy` records the gates, bounds and slack it applied.
`checkpoint_best.pt` is written only on improvement with all gates passing, and
the terminal checkpoint remains available.

`check_recorded_evaluation(model, corpora_by_seed)` re-evaluates a bundle on the
held-out worlds recorded in its `training_evaluation` provenance (joint head,
draw seeds `eval_seed + 1 + r` for at least 8 replicates) and compares every
recorded metric with the same aggregate of the replicates: `slot_pooled` values
pool the slots of all worlds, and `world`, `slot` and `pooled_edges` values are
the evaluator's leg values. Draw-independent keys must agree within 1e-5 or one
event. A draw-dependent key's tolerance is the half-width of a 99% prediction
interval of the replicates, never less than the key's resolution (the change one
event makes where it moves the key most, one indicator term or one slot of one
router draw, or for the co-moments the spacing of their value grid), and never
less than 1e-5 when the replicates do not vary. The check fails, and its
`reason` names the first rule that failed, when it compares no key; when a
draw-independent key fails; when the headline
`joint/query/decomposition_energy_score` lies outside its tolerance, a key's
definedness differs between the recording and any replicate, or a replicate is
non-finite; when a key lies more than three tolerances off; when two keys of one
metric family lie outside their 99.9% intervals; or when more metric families
have a key outside its tolerance than a 99% binomial allowance (1% per family)
over the families it compares. In a simulation that draws correct replays from
the A and B replicates (each metric family from its replicate mean and
covariance, 10,000 trials per bundle), 2.2-2.3% of correct replays fail, about
half of them on the headline alone; drawing all keys jointly from the
8-replicate covariance, which overstates how families move together, gives
5.6-5.8% (also 10,000 trials). Leaving out one of the 8 real replicates as the
recording fails 0 of 16 times, and the real A and B recordings lie at most 0.801
and 0.689 of a tolerance from their replicates. A failure is therefore evidence
to investigate, not proof of a broken model. Router edge log scores are
reported but not counted. The opt-in `tests/test_ab_compatibility.py` runs it
for A and B when `MFM_AB_BUNDLES` and `MFM_AB_CORPORA` are set.

### Record consumers

`pymc_mfm.training.TrainingObserver` receives lifecycle, configuration, step,
evaluation, diagnostic and checkpoint records. MLflow delivery is enabled by
default when available, with an explicit opt-out below. Local evidence is always
retained; delivery failure does not alter numerical training. An arbitrary
user-supplied observer that raises still fails the run rather than hiding the error.

### Reproducibility

`run.json`, training checkpoints, and corpus prior sidecars preserve the
resolved model/training configuration, corpus identity, objective weights,
random seeds, evaluation and selection records, and checkpoint SHA-256 values.
They allow a run to be inspected from local artifacts without external
services. The records describe this package's inputs and outputs; performance
claims remain limited to the stored evaluation results.

Fixed seeds alone do not make multithreaded CPU runs bitwise repeatable. PyTorch's
multithreaded CPU kernels can reduce in a different order from run to run, so two
runs with identical seeds can drift apart in the last float32 bits, starting with
the first backward pass. For bitwise-repeatable CPU runs, pin PyTorch to one
intra-op thread before any tensor work, either with `OMP_NUM_THREADS=1` in the
environment or with `torch.set_num_threads(1)` at the top of the script:

```bash
OMP_NUM_THREADS=1 python -m pymc_mfm.training ...
```

Results are repeatable only for the same thread count, hardware and library
versions; a single-threaded run and a multithreaded run differ from each other.

## Observation-only conditional inference

Supply a known local MFM bundle directory. The example below uses `model` from
the save/load example above, which has untrained weights; substitute your own
trained bundle for meaningful evaluation. The library does not choose a
pretrained checkpoint for you.

```python
import numpy as np
from pymc_mfm import DecompositionResult, decompose, from_pretrained

model = from_pretrained("model")
week = np.arange(12)
controls = np.zeros((12, model.config.n_controls))
controls[:, 0] = np.sin(week)
result = decompose(
    model,
    {
        "spend": np.column_stack([100 + 5 * week, 80 + 2 * week]),
        "controls": controls,
        "context_outcome": 500 + 4 * week[:8],
        "control_mask": [1] + [0] * (model.config.n_controls - 1),
    },
    n_samples=32,
    channel_names=("Search", "Social"),
    window="query",
    mode="model_average",
    units="sales units",
)
result.save("decomposition")
saved = DecompositionResult.load("decomposition")  # No inference or network access.
```

These illustrative inputs exercise the interface, not a calibrated attribution
claim. Spend/controls include the whole horizon; positive outcomes contain only
the observed context prefix. Inputs are equally spaced. Padding is explicit:
inactive columns must be zero and control width must match the artifact.
Omitted masks mean represented nodes, not known causal activity.
Optional numeric `time` labels preserve original row identities without changing
the trained relative-time representation.

`components` has shape `[sample, time, component]` in declared natural units;
`graphs` has shape `[sample, edge]`, aligned to those same draws. Names, masks,
input identity, actual sampling settings, and checkpoint provenance travel with
the result. `mode="map"` uses the empirical complete-graph mode, with optional
`router_samples`; `mode="fixed", graph=...` describes a supplied complete-graph
hypothesis, not its posterior probability. It does not implement a lift-test
likelihood. Auxiliary structure probabilities never route conditional inference.
Context and query calls are **separate marginal evaluations**, not a joint
full-horizon posterior. Component totals exclude observation noise.

`examples/observation_only.py` is an executable offline example. `--model` is
required and must name an existing local bundle directory. Its input JSON contains
only the `spend`, `controls`, `context_outcome` and optional mask fields shown
above. It saves the two marginal windows separately and reports CPU load time,
sampling time and peak process RSS (including imports and loading):

```bash
uv run --no-sync python examples/observation_only.py \
  --model model --observations observations.json --out runs/observed --samples 32
```

Validation precedes dtype narrowing: tiny negative spend, nonzero padding and
near-binary encoded masks cannot become valid through rounding. Caller annotations
cannot replace a loaded checkpoint's identity. Saved results reject
contradictory mode/window/sample-count metadata instead of turning a fixed-graph
hypothesis into graph-averaged posterior evidence.

## Diagnostics, plots and reusable evidence

```bash
uv sync --locked --extra diagnostics
```

```python
from pymc_mfm.diagnostics import build_report, render_report

manifest = build_report(saved, "reports/observed")
replayed = render_report("reports/observed", "reports/observed-replayed")
```

Open `reports/observed/report.html`. `build_report` accepts optional aligned
`truth[time, component]`, binary `graph_truth[edge]`, noisy `observed[time]`,
training/evaluation records, and reference evidence. It computes from the
supplied draws, never resamples the model or fits a reference posterior.
Report HTML embeds its PNG panels, so figures also render in notebook and MLflow
blob-URL previews without a sibling-file server. Separate PNGs and machine-readable
evidence remain versioned artifacts; download the report directory to follow its
relative evidence links.
The original result and diagnostic evidence, scalar/table/array values, figures,
definitions, statuses, lineage, and hashes are retained in the report directory.
`render_report` verifies saved identities and renders those values again without
model execution, training, MCMC, or rescoring.

The catalog includes all component families and indirect-source splits, credible
bands and recovery, graph edge/complete-graph views and DAGs, the evaluator's
whole-graph statistics as `graph/<key>` (entropy, distinct graphs, co-moments
and cc/zz cycle rates also without graph truth), support/padding and
zero atoms, training/optimization/evaluation trajectories, calibration screening,
and within-/between-graph empirical uncertainty. Router and auxiliary-head
readouts remain separately labeled.

Missing truth, prior support, training histories, or usable references is an
explicit unavailable/not-applicable status, not zero or a passing calibration
claim. Nonfinite/undefined results retain their reason. `strict=True` raises
`DiagnosticError` for scoring/rendering failures **after saving failure evidence**.
Fixed and modal-graph summaries describe that selected hypothesis; empirical
graph frequencies come only from model-averaged router draws.

One-world component/week ranks are dependent screening diagnostics, not
independent-world SBC proof. `sbc_report` accepts saved independent-world draws
and explicit prior/data identities; `write_calibration_report` persists its
report card. `comparisons` supplies saved-draw posterior-SBC and reference C2ST,
world-separated classifier fitting, and reference convergence/qualification flags.
Finite graph-mixture variance uses population moments and reports closure and
rare/singleton counts; it is not a calibration guarantee. Simultaneous SBC
ECDF bounds use a labeled Bonferroni correction across dependent quantities.

`ComparisonRecord`, `comparison_summary`, `build_comparison_report`, and
`render_comparison_report` in `pymc_mfm.diagnostics.comparisons` preserve A/B,
oracle, graph-average/modal/fixed, coverage, matched-omission, and stress
comparisons. They require common data and entity identities. Matched omissions
need paired raw evidence; aggregate-only rows cannot invent the missing pairing.
Saved marginal quantiles are never reconstructed into fictitious joint trajectory
draws.

Rendering is headless by default and works in notebooks without paper fonts or
hosted services; library figures carry a PNG representation, so Jupyter shows
them without pyplot. `examples/diagnostics.ipynb` trains a tiny model for a few
updates, plots its records with `TrainingRun`, decomposes a synthetic world, and
walks through `result.plot`, `result.diagnostics(...)` with generating truth, and
`diag.report(...)`. Its weights are explicitly **untrained** for any real use;
`tests/test_examples.py` executes it headless. Graphviz is optional; the
standard DAG renderer uses matplotlib and declares alternate-renderer
availability rather than silently skipping a panel.

### Plot accessors

Plotting hangs off the objects you already have. Accessing `.plot` imports
matplotlib lazily (`import pymc_mfm` and `decompose` never do), every method
returns a `matplotlib.figure.Figure` without registering it with pyplot, and no
method runs the model, training or a reference fit. Single-panel result and
diagnostics methods (for example `graph_posterior`, `reliability`,
`calibration("coverage")`) and every `run.plot` method take `ax=`: a figure of
one axes is drawn into the caller's axes and the root figure is returned, also
for subfigure axes; a figure that needs more axes (`calibration("pit")`/`"ecdf"`,
`run.plot.corpus()`, `run.plot.losses()` over several pages) raises `ValueError`
stating how many. An `ax=` heading goes in the caller's axes title, so give the
caller's figure a layout engine (`layout="constrained"` or `"tight"`) or save it
with `bbox_inches="tight"` to make room for it; its words longer than 24
characters (long metric names) break after `_`, `/` or `-`, and an unnamed run
is labelled by the first 25 characters of its `run_id` (start time and 8 hex
digits; `run.json` keeps the full id). Result, diagnostics and comparison
figures state head, window, mode, units and draw count. Run evaluation figures
state each metric's units, window or edge scope, legs, draws per world and each
head's label (the conditional label names its sampling mode); record, corpus
and input-support figures name the run or the data instead of a head.
`pymc_mfm.training.evaluation.metric_units(key)` gives each evaluation metric's
unit (`sales_scale`, `nats`, `count` or `dimensionless`) and raises `ValueError`
for a metric this release does not define, which run figures state as
`units: unknown to this release`. Figures
are a `Figure` subclass that renders in notebooks, pickles as a plain
matplotlib `Figure` (loadable without `pymc_mfm`), and keeps notebook display
through `copy.deepcopy`; test with `isinstance`, not `type(...) is Figure`.

```python
result = decompose(model, observations, n_samples=256)
result.plot.components()  # family trajectories, 80% bands, boundary
result.plot.channels()
result.plot.channel_totals()
result.plot.controls()
result.plot.indirect()
result.plot.reconstruction(observed=y)
result.plot.graph()  # edge probabilities grouped by family
result.plot.dag()
result.plot.graph_posterior()
result.plot.cardinality()
result.plot.zero_atoms()
result.plot.uncertainty()
result.plot.support()
result.plot.graph_statistics()  # distinct graphs, entropy, co-moments; SHD with truth; needs an active edge
result.plot.cycles()  # complete-graph and cc/zz cycle and two-cycle rates
result.plot.input_support()

diag = result.diagnostics(truth=truth, graph_truth=graph_truth, observed=y)
diag.metrics, diag.statuses  # computed once and cached on `diag`
diag.plot.recovery()
diag.plot.calibration("pit")  # or "ecdf", "coverage"
diag.plot.attribution()  # live/dead effects: CRPS, bias, MAE, leakage
diag.plot.window_totals()  # window-mean 80% coverage (live/dead), latent-total fit
diag.plot.cumulative()  # running channel totals vs truth, cumulative energy score
diag.plot.temporal()  # lag-1 and residual autocorrelation per family
diag.plot.reliability()  # router edge probabilities vs graph truth
diag.report("reports/world_0")  # same files as build_report, no rescoring

from pymc_mfm.diagnostics import Comparison, TrainingRun

run = TrainingRun.load("runs/A")  # or train(...).records(); reads records only
run.plot.losses()
run.plot.learning_rate()
run.plot.gradients()
run.plot.evaluation("decomposition_energy_score", head="conditional", window="query")
run.plot.heads("decomposition_energy_score")  # joint vs conditional, held-out
run.plot.heads("auroc", window="cy")  # structure head vs joint router edges
run.plot.corpus()  # run.json corpus summary
run.plot.corpus_support()  # corpus_support.json world cloud: context mean vs std
run.derived_metrics()  # held-out minus seen gaps, mean AUROC/ECE

comparison = Comparison({"A": result_a, "B": result_b}, truth=truth, live=live)
comparison.plot.performance()
comparison.plot.coverage()
```

Accessors delegate to the same `pymc_mfm.plotting` functions that `build_report`
and `render_report` use, and plotted values come from the same diagnostic
summary as the metric tables. Methods that need truth, graph truth, reference
evidence or records raise `ValueError` naming the missing input; reports mark
those panels unavailable. `TrainingRun.load` decodes each line separately and
tolerates one unterminated last line (listed in `partial_lines`, also when a
write stopped inside a UTF-8 character), raises with `file:line` on corruption,
undecodable bytes or a value past the JSON decoder's limits elsewhere, gives
empty series for missing files, and takes progress from the records rather than
from `run.json`. A `run.json` or `corpus_support.json` that does not decode, or
a `corpus_support.json` without a finite mean and std per world, raises
`ValueError` naming the file. `Comparison` requires one window, world and
`data_sha256` across its inputs, and every result must record its `sales_scale`
as a finite positive number. It exposes every comparison figure (A/B, oracle via
`oracle_pairs`, graph averaging, matched omission, stress summaries and reference
health). Its default `method_pairs` skip same-checkpoint `model_average` vs
`map`/`fixed` pairs, which `plot.graph_averaging()` draws unless `oracle_pairs`
declares that pair (in either order) an oracle contrast; `plot.ab()` needs
method-comparison rows. `CalibrationReport(sbc_report(...))` exposes the SBC report
card through `.plot.card(kind)` and `.report(out_dir)`. The input-support panel
draws the raw spend/control statistics that `decompose`, and the evaluator's
result captures in training reports, record in `input_support`.

Saved results bind their labels and provenance with `metadata_sha256` in
`result.json` (`format_version` 2): an edited unit, mode or provenance value is
rejected on load, version-1 results must be regenerated, and provenance keys
must be strings. Units, names and provenance text must encode as UTF-8;
provenance may nest as deep as `json` accepts, and a circular one raises
`ValueError`. A result's arrays are read-only copies of the supplied arrays, so
a caller reusing its buffer cannot change the result, its cached diagnostics or
a report written from them. Pickled or copied results leave the cached
diagnostics behind.

## Hugging Face artifact transport

Install `[hub]`; local directories need neither that extra nor authentication.
`from_pretrained` uses the same strict local reconstruction after official Hub
snapshot retrieval. An omitted revision selects the Hub repository's default
branch. Branches/tags resolve once; every selected file is then loaded at the same
immutable full commit, retained in model provenance. Supply a full commit for
reproducible loading and release instructions.

Root bundles use no `profile` argument. Collections store separate complete
bundles under `profiles/A`, `profiles/B`, or another explicit profile name.
`list_profiles(repository, revision=...)` discovers manifests without downloading
both weight sets; reuse its resolved revision when selecting a profile.
A/B labels identify artifact/corpus provenance, not architecture alternatives.
There is no built-in personal namespace or unpinned official-release default.

```python
from pymc_mfm import from_pretrained, list_profiles

# Set these to an authorized repository and its immutable release commit.
profiles = list_profiles(repository, revision=release_commit)
model = from_pretrained(
    repository,
    revision=profiles["A"]["revision"],
    profile="A",
    cache_dir="model-cache",
    local_files_only=False,
)
offline = from_pretrained(
    repository,
    revision=model.provenance["hub_revision"],
    profile="A",
    cache_dir="model-cache",
    local_files_only=True,
)
```

Normal `hf auth login` / `HF_TOKEN` / explicit `token` credentials apply; none are
stored in model metadata. Offline cache misses, missing revisions/access and
transport failures remain distinguishable without exposing backend credentials.
Cached snapshots are never modified. Standard `hf download` directories load
through `from_pretrained(local_directory, profile="A")`.
With `hf` 1.32, offline multi-file/pattern downloads can attempt remote tree
listing when tree metadata is not cached. Use the SDK offline call above or
single-file CLI calls for each selected
`config.json`, `model.safetensors`, and `manifest.json`.

Publishing is a separate, explicitly authorized maintainer action:
`push_to_hub(model, repository, profile="A", license=approved_license)` returns
the coherent commit's `oid`. Pass it as the next load's revision. New repositories
default private; existing visibility is unchanged. Only generated bundle/card
files are uploaded, not observations, corpora, or the working directory.
Publication requires a maintainer-approved artifact license and refuses a
conflicting/unknown existing collection license. No license approval or
production repository is inferred by this package. Each profile has its own card
at `profiles/<P>/README.md` (architecture, the recorded training evaluation,
units, context/query semantics, limitations and license); the root `README.md`
indexes every published profile, so pushing one profile never rewrites another's
card. A card describes the profile's `SHARD_PLANS` training plan only when
`corpus_plan="A"` or `"B"` is passed, and otherwise only the provenance
references; it never infers the plan from the profile label. A plan whose shards
do not hold every world of the bundle's recorded `training_evaluation` (its
`shard_seed` among the plan's seeds and its row below `n_cells * draws_per_cell`)
is refused. A repository or directory holds either profiles or one root bundle,
never both.
`write_model_cards(path, license=..., corpus_plan=None)` in
`pymc_mfm.inference.hub` renders the same cards locally for review with any
explicit license text; `corpus_plan` applies when `path` holds one bundle, so
pass each `profiles/<P>` directory. Cards are written to a temporary file and
renamed, so a symlinked card in a cache snapshot is replaced and shared blobs are
never rewritten. Cards do not claim Transformers/AutoModel or hosted inference
support.

`push_to_hub(..., recorded_outputs=path)` uploads a recorded-output sidecar,
`profiles/<P>/recorded_outputs.json`, in the same commit. The sidecar holds
fixed raw observations and the expected `decompose` outputs per mode and window.
It is bound to one profile, one source checkpoint SHA-256 and the weights'
`state_sha256` (which `from_pretrained` records in `model.provenance`).
`push_to_hub` validates it strictly: rows of a `SHARD_PLANS` shard with the
bundle's control and latent widths and at least one active channel and control,
shape-consistent arrays of finite float64 numbers, int64 seeds, and a producer
limited to release versions, platform system/machine and the torch thread count.
It uploads the canonical serialization. Republishing a profile without a sidecar
removes the old one.

Bundle provenance can carry `source_checkpoint_sha256` (the original training
checkpoint, separate from exported hashes) and `training_evaluation`: the
recorded held-out evaluation, including its leg, head, evaluation seed, draw
count, flow steps, the ordered `{shard_seed, row}` worlds, and per-metric values
and aggregation classes. Loading validates its structure only, so later metric
renames never make a published bundle unloadable. Membership in the live metric
set is checked by the conversion and by `check_recorded_evaluation`. A loaded
model whose weights have changed (a dtype cast or an edit) inherits neither the
loaded provenance nor the loaded profile when saved; pass `provenance=`
explicitly (`{}` records none) and `profile=` to label it. The load-time weight
digest is kept in `model.artifact_metadata`, so editing `model.provenance` does
not hide the change.

Preserve training checkpoint hashes separately from exported-file and bundle
hashes; serialization can change bytes without changing state (safetensors does
not write its header metadata in a fixed order). A real push → pinned download →
strict load → conditional inference check requires an authorized test
repository. The default tests never publish or require secrets.

### A and B bundles

A and B are one network (`ModelConfig(n_controls=6)`) trained on different
corpora (plans `A` and `B` above). Their MFM bundles are converted from the
original checkpoints and are distributed through the Hugging Face Hub only, not
from this repository. Until publication is approved, a local collection with
`profiles/A` and `profiles/B` loads with
`from_pretrained(collection, profile="A")`. `tests/test_ab_bundles.py` checks each
bundle's source checkpoint SHA-256, recorded evaluation and recorded outputs
when `MFM_AB_BUNDLES` points at such a collection, or when the pinned published
revision is cached; otherwise it skips. After publication, set its `PUBLISHED`
constant to the repository and full commit.

## MLflow lifecycle and evidence

```bash
uv sync --locked --extra training --extra generation
```

Numerical `TrainingConfig` is unchanged by tracking. `TrackingConfig` controls
delivery independently; the default enables MLflow when installed and collects
diagnostics from existing evaluation draws. Its default SQLite database and
artifact store live below the selected run output directory, not inside the
installed package. No hosted server is required. For comparisons across runs,
configure one shared backend and experiment:

```python
from pymc_mfm.training import TrackingConfig, train

tracking = TrackingConfig(
    tracking_uri="sqlite:///runs/comparison-mlflow.db",
    experiment_name="canonical-comparison",
    profile="A",
)
result = train(corpora, out_dir="runs/A-seed31", tracking=tracking)
# Explicit opt-out: TrackingConfig(enabled=False).
# Independently skip scheduled diagnostics: TrackingConfig(diagnostics=False).
```

CLI equivalents are `--tracking-uri`, `--experiment-name`, `--profile`,
`--no-tracking` and `--no-diagnostics`. An explicit URI takes precedence over
`MLFLOW_TRACKING_URI`. For an authorized remote backend, set
`MLFLOW_TRACKING_URI=https://your-tracking-server` and configure credentials
normally outside the training configuration. Local server mode is:

```bash
uv run --no-sync mlflow server \
  --backend-store-uri sqlite:///runs/comparison-mlflow.db \
  --artifacts-destination runs/mlflow-artifacts --host 127.0.0.1 --port 5000
```

`examples/tracking.py` reads complete histories and versioned report artifacts
through `MlflowClient`. `tracking_catalog.json` under `pymc_mfm.training`
documents producer → field → representation → cadence. The generated name
table preserves original dimensions even when safe metric names require encoding.
The canonical loop logs every active joint/conditional/context/auxiliary loss,
actual objective weights, learning rate, gradient health, timing and consumption
counters at `log_every`; evaluation, checkpoints and saved diagnostic suites use
the existing evaluation cadence. Optimizer update is the primary metric step.
At step 0 the run logs every numeric leaf of `run.json`'s `corpus_summary` as a
`corpus/<path>` metric, and the per-module parameter counts as
`config/parameter_counts/*` parameters.
Expensive stress/SBC/reference fits are explicit saved-evidence operations, not
hidden per-batch jobs. Structured ranks, matrices and all captured per-world
diagnostic tables remain versioned artifacts, not misleading scalar averages
or thousands of new metric names per evaluation. Their manifests explicitly
declare `metric_representation="per_world_table"`. Canonical train/evaluation
histories keep their original optimizer steps. Saved comparison and calibration
reports are delivered by the caller: pass the report manifest to
`MlflowTrainingObserver.on_diagnostic(manifest, report_dir)` to log its
dimensioned scalar metrics and files. No library path schedules that delivery.

HTTP delivery is bounded. `TrackingConfig(http_timeout=..., http_max_retries=...)`
sets MLflow's per-request timeout (seconds, at least 1) and retry count (0 to 10).
Precedence is the explicit field, then the caller's `MLFLOW_HTTP_REQUEST_TIMEOUT`
/ `MLFLOW_HTTP_REQUEST_MAX_RETRIES` / `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR` /
`MLFLOW_HTTP_REQUEST_BACKOFF_JITTER` / `MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER`
environment, then MFM's defaults: a 10 s timeout, 2 retries, backoff factor 1, no
jitter, and no waiting on `Retry-After`. On its own, MLflow waits as long as a
rate-limited or overloaded server's `Retry-After` header asks, up to 6 hours per
retry; with MFM's default, 429 and 503 answers are retried on the same backoff as
a refused connection. The values apply process-wide for the duration of the run,
so any other MLflow client in the process uses them too, and the previous
environment is restored afterwards, including on failure. CLI users set the
environment variables. A caller value that MLflow cannot parse, or that is
non-finite, negative or a zero timeout, falls back to MFM's default. Such a value,
or a conflict with another concurrently tracked run, is recorded as one local
`http_settings` event that names the variable. Every MLflow write is synchronous,
so these bounds also hold when `MLFLOW_ENABLE_ASYNC_LOGGING` is set. Uploads to an
`s3://`, `gs://` or Azure artifact store go through that store's own client: MFM
does not bound its timeouts and retries, and treats its failures as transport
failures only when they wrap a `requests` connection error or timeout.
If the run cannot be started on the backend, the observer records one delivery
receipt with `delivery_stopped: true` and makes no further network calls, not even
at the run end; all later evidence stays local. With the defaults, a refused
connection costs a 2-update run about 3 s instead of minutes of client retries. A
host that accepts connections but never answers costs about 30 s: three 10 s
timeouts plus backoff. Lower `http_timeout` / `http_max_retries` to shorten this.

After a successful start, delivery runs on two channels that pause separately:
tracking-server writes (metrics, parameters, tags and the run status) and artifact
uploads. A broken artifact store does not hold back the metrics, and a
tracking-server outage does not hold back the uploads. A transport failure on a
channel (a connection error, a timeout, or retries exhausted, including on
rate-limit or unavailable answers) pauses that channel: one receipt with
`transport: true` and `delivery_stopped: false`, and the delivery status stays
`degraded` for the rest of the run. For 60 s every delivery on the channel is
skipped without a network call. The channel's next delivery after that probes it:
an upload probes with itself, and a tracking-server write is preceded by a write
of the run's `mfm.delivery_status` tag. If the probe, or the delivery right after
it, fails, the pause continues and its cool-down doubles, up to 10 minutes,
without another receipt. The pause closes when a delivery on the channel
succeeds, and only then does the cool-down return to 60 s. Skipped deliveries are
not replayed; `tracking/events` still holds every event. `delivery_pauses` in
`tracking/state.json` and in the run's `tracking` record holds, for each channel
(`tracking`, `artifacts`), the `open` pause or null and the channel's totals of
`pauses`, `skipped_deliveries` and `failed_probes`. Each closed pause is appended
to `tracking/delivery_pauses.jsonl`: its `channel`, its receipt's `sequence`, the
`resumed_sequence` of the event whose delivery closed it, `skipped_deliveries`
and `failed_probes`. The run end waits for no cool-down: a paused channel probes
at its first terminal delivery, and the first transport failure on a channel is
that channel's one bounded run-end attempt, with a receipt that has
`delivery_stopped: true`; the channel then sends nothing more. So a backend that
answers again receives the `run_end` event, `run.json` and the other terminal
records, the model export, the tags and the terminal status, and does not keep
the remote run `RUNNING`: after a pause, a finished run ends `FAILED` with
`mfm.lifecycle_status=finished` and `mfm.delivery_status=degraded`, because
`FINISHED` is never asserted after incomplete delivery. When the tags close a
tracking pause after the terminal uploads, `tracking/state.json` and the pause
ledger are uploaded again so the remote copies match the local ones. A dead server
costs each channel that reaches it one bounded run-end attempt; with
`mlflow-artifacts:`, uploads reach the tracking server too. After Ctrl-C during the
run end, the only network call left is one attempt at the tags and status. Other
failures, such as a rejected request or a local file error, get one receipt each
(`transport: false`) and delivery continues.

One client-owned MLflow run corresponds to one training execution. MLflow's HTTP
client retries every request, writes included, so a request slower than
`http_timeout` can reach a tracking server up to `http_max_retries + 1` times: a
slow `create_run` can leave extra runs, and a slow upload can be stored more than
once and still get a failure receipt and a pause. Every attempt of the
`create_run` carries the same random `mfm.start_id` tag. After a successful start
against a tracking server (an `http`, `https` or `databricks` URI) the observer
therefore searches once for other running runs with its `mfm.start_id`, tags each
one `mfm.duplicate_of` with its own run id, ends it `FAILED`, and records one local
`duplicate_runs` event with their `count` and `run_ids`. This is housekeeping, not
delivery: a failed search or end is a local `duplicate_search_failed` event with its
`operation` and `error_type`, pauses nothing and leaves the delivery status as it
is. A failed start sends nothing more, so it leaves such runs running; `start_id`
in `tracking/state.json` holds the tag value that finds them. Large uploads through
a proxying server (`mlflow server --artifacts-destination`) need an `http_timeout`
above the time the server takes to store them. The observer does not attach to or
terminate an unrelated fluent active run. Local
`tracking/state.json`, event records, delivery receipts and name tables distinguish
disabled, missing-support, degraded and successful delivery from the numerical
run's lifecycle. Backend failures warn and retain local evidence without changing
updates, RNG, data consumption or checkpoint selection. Interruption/failure is
not advertised as success; abrupt process death leaves running state rather
than inventing finalization. Checkpoint/report paths retain step, kind, exact
checkpoint identity and per-mode lineage instead of overwriting earlier evidence.

Full synthetic report upload requires explicitly synthetic corpus provenance.
The generation helper writes a corpus-hash-bound synthetic-origin sidecar; an
unmatched/missing sidecar is unknown origin. Unknown/private reports remain
local, with only safe withheld-status/identity receipts sent to MLflow. Raw
observations, credential-bearing provenance and derived private artifacts are
not automatically uploaded. Tracking a report only reads already-saved outputs.
Producing run IDs can be included in model provenance; Hub loading never requires
MLflow and training never requires Hub publication.

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

### Small public workflow

```bash
uv sync --locked --extra generation --extra training
uv run --no-sync python examples/workflow.py --out runs/workflow
```

This generates and reloads twelve synthetic worlds from an explicitly reduced
profile-A recipe (16 time points, context 8), runs two real AdamW updates on the
full canonical architecture, reloads both the training checkpoint and local
safetensors export, and checks aligned conditional outputs. It writes
`observations.json`, reusable query draws, observation-only and truth-enabled
reports, and an inference-free rerender. Open `runs/workflow/synthetic-report/report.html`.
This is an integration exercise, **not** usable pretrained weights or a calibrated
attribution model.

Tracking is enabled with an explicitly local SQLite backend under the output
directory, so an ambient remote tracking URI cannot receive this example.
The example checks actual MLflow metric histories, downloaded checkpoint/report
hashes, healthy delivery and run→checkpoint→result→report lineage. The metric
name table maps producer names to stored MLflow keys; use it rather than guessing
the encoded keys. Outputs require a new directory and are never published.

CI runs the existing base/full behavioral suites, then builds wheel and sdist.
The same installed-artifact checks can be run locally:

```bash
uv build
uv run --no-sync python .github/scripts/installed_smoke.py --work-dir artifacts/installed
```

The script uses locked dependencies in three fresh environments, executes the
sdist workflow with isolated imports, loads its model with only the base wheel,
and renders the saved output with only the diagnostics extra. Import origins and
absence of training/Hub/plotting dependencies in the relevant lanes are checked.
The checks use generated synthetic data and local artifacts; they require no
external checkpoints, service credentials, MCMC, long training run or registry writes.

## Release status

This remains `0.0.1.dev0`, not a published or approved public release.
See [release preparation and prerequisites](RELEASE.md) for supported behavior,
resource and scientific limits, and outstanding licensing/distribution decisions.
Maintainers must choose and approve code and model licenses before distribution;
those licenses are not yet declared.
Repository visibility changes and package/model publication require explicit
maintainer approval; a successful local smoke test does not grant that approval.
