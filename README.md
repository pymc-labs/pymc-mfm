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

Base runtime dependencies are NumPy, PyTorch, safetensors and packaging.
Base inference does not import generation, plotting, tracking, or Hub clients.
Install `[diagnostics]` for the complete numerical/plotting suite, `[hub]` for
remote artifact transport, `[training]` for diagnostics and MLflow, and
`[generation]` for public `pymc-generator`. The `dev` group supplies pytest and Ruff.

Model weights, corpora, and generated experiment outputs do **not** belong in the
library. They remain external, on Hugging Face or
[Google Drive](https://drive.google.com/drive/folders/1G3jcruRo3PVq_mE_6Lvoiobfz2smc8Mg).

## Model configuration and local artifacts

```python
from pymc_mfm import MFM, ModelConfig, from_pretrained, save_pretrained

model = MFM(ModelConfig())  # Fresh, untrained weights.
save_pretrained(model, "model")  # New directory; existing bundles are immutable.
restored = from_pretrained("model", device="cpu")  # Offline, evaluation mode.
```

`ModelConfig` is the direct construction input, not a preset/override resolver.
Its defaults define the shared dimensions and active objective weights.
`to_dict()` emits every field and a format version; `from_dict()` refuses missing,
unknown, or incompatible fields instead of silently filling defaults. Save/load
uses that same configuration and strictly restores all trained module groups,
including the auxiliary head. Tensor dtype and parameter trainability are preserved.
The directory contains `config.json`, `model.safetensors`, and `manifest.json`,
with ordered schemas, full configuration, provenance, compatibility requirements,
tensor descriptions and file/bundle checksums. Missing, reordered, incompatible
or corrupt state is rejected. There is no pickle or remote-code loading fallback.
Optimizer/RNG state is absent: model exports are not training-resume archives.

Architecture semantics are fixed: all five feature blocks, causal attention,
the eight graph families in `cy, dc, dz, db, zb, zc, cc, zz` order, zero atoms,
support projection, and the complete active training path. No architecture
selectors or optional-head flags are exposed. Paths, device, seed, optimizer,
corpus selection, and run budget stay outside `ModelConfig`.

`pymc_mfm.data.profiles.PROFILE_A` and `PROFILE_B` are named corpus/checkpoint
profiles. They describe data metadata only and never modify `ModelConfig`.
Checksums identify designated reference weights. `from_pretrained` accepts strict
MFM directory bundles locally or retrieves that same format through the Hub extra.

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

The adapter consumes schema **4** and its domain-neutral array names. The
generator's `dy`/`zy` graph slots retain their positions as the model's internal
`db`/`zb` checkpoint labels; they represent effects on the outcome, not on the
intrinsic intercept. Decomposition targets exclude `outcome_noise`; their
ordered `cc, zc, dc` indirect split is the generator's attribution convention,
not an identified causal decomposition. Draws return to natural units with
`draws * batch["sales_scale"][None, :, None, None]`.

All normalization statistics and moment features use observed context only.
The A/B log representation requires positive observed outcomes, nonnegative
treatments/direct contributions, and at least one active treatment and
covariate. Unsupported worlds are rejected, not clipped into that domain.
Other decomposition terms can be signed. The canonical covariate zscore
clipping at `[-10, 10]` is retained. Do not substitute generator-normalized
arrays or `outcome_scale` for the model's context-based representation.

### Canonical corpus recipes

`pymc_mfm.data.recipes.FW53_PRIOR` is profile A's canonical public
`pymc-generator` configuration. `recipe_for(PROFILE_B)` exposes the documented
profile-B live-channel range through `edge_budget["cy"] = (4, 10)` and marks
the result `reconstructed=True`; use it as an explicit configuration, not as a
claim that a separately produced corpus has been recreated.

```bash
uv run --locked --extra generation python -m pymc_mfm.data generate \
  --profile A --n-cells 60 --draws-per-cell 4 --seed 530000 --out corpora/fw53_A_small.npz
```

This writes the corpus and a `<path>.prior.json` sidecar with the complete
`SCMPrior`, generator version, and timestamp. `load_corpora` accepts compatible
schema-4 files; `provenance` records their path, SHA-256, split counts, context
lengths, and any sidecar in the run manifest.

## Training

`pymc_mfm.training.train` is the single entry point; `python -m pymc_mfm.training`
wraps it. `TrainingConfig()` is the recorded full-run contract; corpus choice
is supplied separately:

| Setting | Canonical value |
| --- | --- |
| Updates / batch | 62,500 AdamW updates of 64 worlds (`budget == "canonical"`) |
| Learning rate | peak 2.8e-3, linear warmup over 3,125 updates, cosine decay to exactly 0 |
| Optimizer | AdamW, weight decay 0.01, betas (0.9, 0.999), eps 1e-8, global grad-norm clip 1.0 |
| Seed | 31 for weight init and corpus order; 616161 for evaluation |
| Evaluation | every 6,250 updates and at the final update, 16 held-out worlds, 128 draws |
| Checkpoints | `checkpoint_last.pt` at every evaluation, milestones at 25,000 and 50,000 |

Any other `steps`/`batch_size` is a **reduced** budget on the same code path
and is recorded as such. A reduced run should also shrink `warmup_steps`,
`eval_every` and `milestone_steps`:

```bash
uv run --locked --extra generation --extra training python -m pymc_mfm.training \
  --corpus corpora/fw53_A_small.npz --out-dir runs/reduced_A \
  --steps 200 --warmup-steps 10 --eval-every 100 --milestone-steps 150 \
  --eval-samples 32 --log-every 10 --device auto
```

```python
from pymc_mfm.data.recipes import load_corpora, provenance
from pymc_mfm.training import TrainingConfig, load_checkpoint, train

loaded = load_corpora(["corpora/fw53_A_small.npz"])
result = train(
    [item.corpus for item in loaded],
    out_dir="runs/reduced_A",
    training=TrainingConfig(steps=200, warmup_steps=10, eval_every=100, milestone_steps=(150,)),
    corpus_provenance=[provenance(item) for item in loaded],
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
| `run.json` | run id, status (`running` -> `finished`/`failed`/`interrupted`), environment, resolved model/training/optimizer configuration and objective weights, corpus provenance, consumption budget in epochs, evaluation and selection policy, checkpoint list with SHA-256, persisted best step, exact consumed updates/rows/distinct worlds |
| `objective_trace.json` | the first update's executed objective trace (all terms, weights, structure-family normalization) |
| `metrics.jsonl` | one record at update 1 and every `log_every` updates: `loss`, `flow_loss`, `context_flow_loss`, `graph_conditional_flow_loss`, `graph_conditional_context_flow_loss`, `task_loss`, `struct_loss`, `lr`, pre-clip `grad_norm`, `grad_clipped`, epoch, consumed rows, batch context length, timings and throughput |
| `evaluations.jsonl` | one record per evaluation event with the `holdout` and `seen` legs, a flat `{leg}/{head}/{window}/{metric}` view and the selection decision |
| `checkpoint_last.pt`, `checkpoint_best.pt`, `checkpoint_step_<N>.pt` | atomic training checkpoints (`load_checkpoint`): `config`, `state_dict`, training configuration, run identity and provenance, consumption, the accompanying evaluation record and selection |
| `model/` | complete terminal safetensors/JSON model export; original training checkpoint and exported tensor hashes remain distinct |

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
| `joint_router` | graph draws of the joint head | edge marginals plus `shd_mean`/`exact_recovery` of whole graphs |

Decomposition metrics (`*_scaled_mae`, `*_crps`, `*_cov80` per component
family, `contribution_padding_leakage_scaled_mae`, `closure_rel_err`, energy
scores) are computed in the training z-space for both the query and context
windows. Metrics undefined for a world are dropped and named in `undefined`
with their reason; a metric undefined everywhere is `null`, never zero.
Nonfinite active edge probabilities make the pooled predictive scores `null`
with the affected-slot count; truth counts and prevalence remain available.
Nonfinite graph draws are marked unavailable rather than thresholded into
apparently valid binary graphs.

Checkpoint selection minimizes
`holdout/joint/query/decomposition_energy_score`. Promotion also requires the
auxiliary `cy` log score to be finite and padded-channel leakage to be zero;
`checkpoint_best.pt` is written only on improvement with all gates passing.
Missing, nonfinite, or partially defined gate metrics block promotion with a
recorded reason, while the terminal checkpoint remains available. Selection
reads the **joint head**, not the conditional decoder; every selection record
states this scope.

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

## Observation-only conditional inference

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
bands and recovery, graph edge/complete-graph views and DAGs, support/padding and
zero atoms, training/optimization/evaluation trajectories, calibration screening,
and within-/between-graph empirical uncertainty. Router and auxiliary-head
readouts remain separately labeled. `source_inventory.json` under
`pymc_mfm.diagnostics` records capability destinations and applicability.

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
graph-average/modal/fixed, coverage, matched-omission, and stress comparisons.
They require common data and entity identities. Matched omissions need paired
raw evidence or verified derived evidence; aggregate-only CSV rows cannot invent
the missing pairing. `load_replication_summary` accepts a pinned summary SHA-256
and verifies the referenced raw inputs before replotting saved aggregate evidence.
Compressed historical quantiles are never reconstructed into fictitious joint
trajectory draws.

Rendering is headless by default and works in notebooks without paper fonts or
hosted services. `examples/diagnostics.ipynb` demonstrates observation-only
reporting with explicitly **untrained** small-model weights. Graphviz is optional;
the standard DAG renderer uses matplotlib and declares alternate-renderer
availability rather than silently skipping a panel.

## Hugging Face artifact transport

Install `[hub]`; local directories need neither that extra nor authentication.
`from_pretrained` uses the same strict local reconstruction after official Hub
snapshot retrieval. An explicit revision is required for remote loading.
Branches/tags resolve once; every selected file is then loaded at the same
immutable full commit. The resolved revision is retained in model provenance.

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
With `hf` 1.32, offline multi-file/pattern downloads attempt remote tree listing;
use the SDK offline call above or single-file CLI calls for each selected
`config.json`, `model.safetensors`, and `manifest.json`.

Publishing is a separate, explicitly authorized maintainer action:
`push_to_hub(model, repository, profile="A", license=approved_license)` returns
the coherent commit's `oid`. Pass it as the next load's revision. New repositories
default private; existing visibility is unchanged. Only generated bundle/card
files are uploaded, not observations, corpora, or the working directory.
Publication requires a maintainer-approved artifact license and refuses a
conflicting/unknown existing collection license. No license approval or
production repository is inferred by this package. Model cards describe the
canonical architecture, provenance, domain, units and diagnostic limitations;
they do not claim Transformers/AutoModel or hosted inference support.

Preserve original checkpoint hashes separately from exported-file and bundle
hashes; serialization can change bytes without changing state. A real
push → pinned download → strict load → conditional inference check requires an
authorized test repository. The default tests never publish or require secrets.

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
Expensive stress/SBC/reference fits are explicit saved-evidence operations, not
hidden per-batch jobs. Structured ranks, matrices and all captured per-world
diagnostic tables remain versioned artifacts, not misleading scalar averages
or thousands of new metric names per evaluation. Their manifests explicitly
declare `metric_representation="per_world_table"`. Canonical train/evaluation
histories keep their original optimizer steps; standalone scalar comparison
and calibration reports retain dimensioned MLflow metrics.

One client-owned MLflow run corresponds to one training execution. The observer
does not attach to or terminate an unrelated fluent active run. Local
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
