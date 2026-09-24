# Release preparation

PyMC MFM remains at **`0.0.1.dev0`**. These notes describe the standalone
library's development capabilities and distribution prerequisites, not an
approved or published release.

## Changes

- Observation-only inference checks numeric domains, padding, masks and tensor
  structure before precision narrowing. Training targets never enter inference;
  all padded control feature blocks are zero.
- Model bundles use strict configuration and safetensors loading. Checkpoint,
  training-run and data identities remain attached to exports and results;
  caller annotations cannot replace a loaded checkpoint's identity.
- Saved decomposition evidence rejects contradictory graph mode, marginal
  window and sample-count metadata. Reports can be rendered again from saved
  evidence without inference, training or reference fitting.
- Public entrypoints cover corpus generation, training, checkpoint/export
  loading, conditional inference and diagnostics. The small workflow includes
  actual local MLflow metric and artifact readback.
- Tracking readback defaults to the producing run's saved backend, not an
  unrelated ambient URI. Custom diagnostic-observer exceptions fail the run
  after local evidence is saved; optional report-build failures remain explicit.
- The optional generator dependency follows
  `git+https://github.com/pymc-labs/pymc-generator.git@main`; `uv.lock` records the
  resolved commit for locked uv installations.
- **Fixed architecture.** `ModelConfig` holds only size and geometry.
  - The five loss weights and checkpoint selection (`selection_head`, conditional by default) belong to `TrainingConfig` / `LossWeights`.
  - The flow numerics are constants: 2 Fourier frequencies, 16 Euler steps, trajectory correlation 0.98.
  - `DEFAULT_N_SAMPLES = 32` is the draw count when `n_samples` is omitted.
  - A float64 architecture lock test pins losses, samples and every parameter's default-loss gradient of a two-layer model to outputs recorded before the refactor.
- **A/B bundles.**
  - Bundle provenance can record the original `source_checkpoint_sha256` and the recorded training evaluation (`training_evaluation`, aggregation classes `world`, `slot`, `slot_pooled`, `pooled_edges`).
  - `push_to_hub` uploads per-profile model cards, a collection index and an optional recorded-output sidecar; `write_model_cards` renders the cards locally. Cards show a training plan only for an explicit `corpus_plan` whose shards hold every recorded evaluation world, and a location never mixes a root bundle with profiles.
  - The sidecar is bound to the weights' `state_sha256` and the bundle's control and latent widths, validated strictly (at least one active channel and control, finite float64 numbers, int64 seeds) and uploaded in canonical form. A model whose weights changed after loading inherits neither the loaded provenance nor the loaded profile; the load-time weight digest is kept in `model.artifact_metadata`.
  - Bundles load where the installed versions satisfy the installed release's compatibility constant and every lower bound the bundle records; recorded upper bounds are not enforced, so a later release can relax the constant for published bundles. `numpy` and `torch` are capped below 3 in the dependencies, matching the constant. `save_pretrained` refuses an environment outside it or whose package versions are not release versions (canonical PEP 440 without a local label; torch alone may carry the build label of a CPU, CUDA, ROCm or XPU wheel, including the CPU cxx11-ABI and AMD ROCm `.git<commit>` wheels, or the commit of a source or NVIDIA NGC build), and `train` runs the same check before writing anything; the refusal names the installed package and version. Exports record versions as `packaging` normalizes them; the installed Python version comes from `sys.version_info`.
  - A and B are distributed through the Hub only, and are not yet published.
- **Corpora.**
  - `SHARD_PLANS` stores the A, B and evaluation shard plans; `generate --profile A --shard i` and `generate --evaluation-shard i` reproduce them.
  - A drift test compares generator probes against stored hashes and names the generator commit they were recorded with. A weekly `Generator drift` workflow runs it against the generator's tracked branch; regular CI uses the locked commit.
  - `run.json` records a corpus summary and per-module parameter counts, also logged to MLflow at step 0. `corpus_support.json` records per-world context-outcome location and scale (at most 5,000 evenly spaced worlds).
- **Evaluation.**
  - New metric families: live/dead attribution, per-draw closure, temporal coherence, window totals, router graph posterior statistics, edge AUPRC and `null_fpr`, and counts.
  - Also recorded: reliability/threshold tables, per-metric world counts and optional per-world values.
  - Checkpoint gates validate `METRIC_BOUNDS` and include the selected head's context energy score; a held-out metric of that head with non-finite worlds blocks promotion (`selection.nonfinite_keys`).
  - `check_recorded_evaluation` reproduces a bundle's recorded training evaluation within stated tolerances.
  - MPS evaluation no longer converts tensors to zeros.
- **Plot suite.**
  - Figures no longer use pyplot and carry a PNG representation for notebooks. Result, diagnostics and comparison figures state head, window, mode, units and draw count; run evaluation figures state units, scope, legs, draws and head labels.
  - New report panels: attribution, window totals, cumulative contribution, temporal coherence, edge reliability, graph statistics (every `joint_router/graph` statistic as `graph/<key>`, the truth-free ones also without graph truth), cycle check (complete graph plus cc/zz cycle and two-cycle rates) and input support. `run.plot.corpus()` plots the corpus summary and `run.plot.corpus_support()` the world-level context support.
  - `result.plot`, `result.diagnostics(...)`, `TrainingRun.load(...).plot`, `Comparison(...).plot` and `CalibrationReport(...).plot`, all on the same plotting functions as the reports.
  - Posterior SBC includes the 90% level.
  - Single-panel methods and every `run.plot` method take `ax=` (also on subfigure axes); larger figures refuse it with the axes count. Headers wrap inside the figure, and library figure titles rewrap at draw time (wide glyphs such as CJK units). `ax=` headings break words longer than 24 characters and need the caller's layout engine or `bbox_inches="tight"`; an unnamed run is labelled by the first 25 characters of its `run_id`, and run evaluation headings no longer repeat the metric or the generic title.
  - `metric_units(key)` names each evaluation metric's unit, and run evaluation figures state head label, scope, units and draws; a recorded metric this release does not define is drawn with its units stated as unknown.
  - Calibration coverage draws nested hollow markers, so families with equal coverage stay visible, and keeps markers at 0 and 1 whole. Graph statistics mark the cumulative-mass curve and keep legends above the bars.
  - `corpus_support` skips worlds whose context statistics overflow (`nonfinite_context_statistics`); `plot_corpus_support` without a usable world states how many kept worlds were skipped and why.
  - Figures pickle as plain matplotlib figures; `copy.deepcopy` keeps notebook display.
  - Training-report result captures record `input_support` under `decompose`'s keys, so their input-support panel draws.
  - `Comparison`'s default method pairs skip same-checkpoint graph-averaging pairs, a declared oracle pair replaces the automatic pair of the same two methods, and every mapped result must record a finite positive `sales_scale`.
- **Review fixes.**
  - `count_mae` no longer wraps for unsigned graph arrays.
  - MLflow HTTP delivery has bounded timeouts and retries, and a server's `Retry-After` no longer stretches them. It stops after a failed start. After a successful start, tracking-server writes and artifact uploads pause separately: a transport failure pauses its channel with one receipt, and a probe after a 60 s cool-down, doubling up to 10 min while the channel keeps failing, resumes it. At the run end a paused channel probes at once, and the first transport failure on a channel is its one bounded attempt, so a backend that answers again receives the terminal evidence, model export, tags and status and does not keep the run `RUNNING`. After Ctrl-C during the run end, only that attempt at the tags and status remains. `tracking/state.json` keeps the open pause and totals per channel; closed pauses go to `tracking/delivery_pauses.jsonl`. On a tracking server, runs left by a retried slow `create_run`, found by the random `mfm.start_id` tag every attempt carries, are ended `FAILED` with an `mfm.duplicate_of` tag; a failed search is a local event that pauses nothing and does not degrade delivery.
  - `result.json` hashes its metadata.
  - Pickled or copied `DecompositionResult`s leave the cached diagnostics behind; units, names and provenance text must encode as UTF-8. Provenance nests as deep as `json` accepts again, and a circular one raises `ValueError`.
  - `TrainingRun.load` decodes records line by line, so a write cut inside a UTF-8 character is tolerated in the last line. Undecodable `run.json` or `corpus_support.json`, over-long integers and over-deep nesting raise `ValueError` naming the file (and line); integers of any size are finite values.
  - The README documents the CPU thread setting for bitwise-repeatable runs.
  - `load_replication_summary` was removed; it parsed a summary format that MFM never produces.
  - CI plants ignored local files (`*.pt`, `*.npz`, `.env.*`) before building, so a build that stops honouring `.gitignore` fails the installed-distribution check.

### Breaking changes

- `ModelConfig` `format_version` 2: version-1 bundles and training checkpoints are rejected.
- The loss-weight and flow fields are gone from `ModelConfig`; `flow_ode_method`/Heun is removed and `model.ode.integrate_flow` no longer takes `method=` (Euler only).
- `MFM.loss`/`loss_terms` take `weights=`; `inference_metadata()` takes no argument.
- Draw counts must be positive integers; `n_samples=None` no longer selects the default (omit the argument instead).
- Checkpoint selection defaults to the conditional head. `MONITOR_METRIC`/`GATE_METRICS` are replaced by `selection_metrics(head)`.
- `result.json` `format_version` 2: version-1 saved results must be regenerated, and provenance keys must be strings.
- `plot_records` takes a record summary and an explicit header.
- `PROFILE_A_TRAIN_CELLS`/`PROFILE_A_SHARD_BASE_SEED` are removed; use `SHARD_PLANS`. Plan-A shard seeds start at 530200 (`SHARD_PLANS["A"].seed(0)`), not the removed base seed 530000, so corpora generated as `530000 + i` are not plan-A shards.
- `CLOSURE_EPS` moved from `pymc_mfm.training.evaluation` to `pymc_mfm.training.metrics`.
- `generate` refuses an existing target without `--force`.
- `load_replication_summary` is removed.
- After a failed MLflow start, delivery stops with one receipt instead of one receipt per delivery.
- After a later MLflow transport failure, that delivery channel (tracking-server writes or artifact uploads) pauses with one receipt; deliveries skipped during the pause are not replayed.
- A tracked run sets `MLFLOW_HTTP_REQUEST_TIMEOUT`, `MLFLOW_HTTP_REQUEST_MAX_RETRIES`, `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`, `MLFLOW_HTTP_REQUEST_BACKOFF_JITTER` and `MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER` process-wide for its duration, so any other MLflow client in the process uses MFM's bounds meanwhile (caller values of these variables still take precedence over MFM's defaults).
- Diagnostic report keys `attribution/{channel,control}/{live,dead}/*` are replaced by `attribution/{live,dead}_{media,control}/*`. `PANEL_NAMES` gains eight entries.
- Library figures are a `Figure` subclass (`type(fig) is Figure` no longer holds); `plot_c2st`/`plot_posterior_sbc` take `metadata=`.
- `Comparison` raises when a mapped result records no `sales_scale`, or one that is not a finite positive number, instead of assuming 1; `ComparisonRecord` rejects a boolean or non-numeric `scale`.
- `DecompositionResult` rejects text that cannot encode as UTF-8. Its arrays are read-only copies of the supplied arrays: writing into `result.components` (or `graphs`, the masks or `time`) raises.
- Diagnostic `graph/empirical_entropy` is renamed `graph/entropy`; with no active edges, `graph/shd_mean`, `graph/exact_recovery` and `graph/entropy` are undefined (`no_active_slots`) instead of 0, 1 and 0, the new `graph/n_distinct` is undefined too, and the `graph_statistics` panel is unavailable. `graph/log_score` is in nats; `graph/cardinality_*`, `uncertainty/n_*`, `graph_support/family/*/{mean,modal,true}_count`, `count_mae` and `calibration/*/n_ranks` are counts.
- `metric_units` raises `ValueError` instead of `KeyError` for a metric it does not define, and `plot_calibration` raises `ValueError` for a `kind` other than `coverage`, `ranks` or `ecdf` instead of drawing the ECDF.
- `numpy` and `torch` dependencies are capped `<3`. `save_pretrained`, and `train` before it writes anything, refuse an environment outside the compatibility constant or whose package versions are not release versions.
- Loading refuses a bundle whose recorded compatibility is not a nonempty range of `>=`, `>`, `<`, `<=` bounds, or whose producer versions are not release versions within it; recorded upper bounds no longer block loading.
- `push_to_hub` refuses a recorded-output sidecar whose control or latent width differs from the bundle's, and `push_to_hub`/`write_model_cards` refuse a `corpus_plan` that the recorded evaluation worlds contradict. A repository or directory can no longer hold both a root bundle and `profiles/<name>/` bundles.
- A model whose weights changed after loading no longer inherits or enforces the loaded provenance or profile: `save_pretrained` raises unless `provenance=` is passed (`{}` records none).

## Public API and evidence limits

The offline observation example requires `--model` to name a known local MFM
bundle. No pretrained weights are bundled or selected automatically. Base local
inference does not require generation, plotting, MLflow or Hub clients.
`[generation]` supplies the generator, `[diagnostics]` adds numerical diagnostics
and plotting, `[training]` adds diagnostics and MLflow, and `[hub]` enables remote
artifact transport. Installing dependencies can require network access; local
inference and saved-evidence replay do not. Remote loading resolves the selected
revision, or the default branch when omitted, to one recorded immutable commit.
Use a full commit for reproducibility; offline Hub loading needs a populated cache.

Context and query inference calls are independent marginal evaluations, not
paired full-horizon draws. `model_average` samples router graphs, `map` uses
the empirical complete-graph mode, and `fixed` conditions on a supplied graph
hypothesis. A supplied graph is not posterior graph evidence or a lift-test
likelihood; auxiliary structure probabilities do not route these calls.

Truth-based diagnostics require aligned truth, training curves require recorded
histories, and reference diagnostics require usable, qualified reference
evidence. Independent-world calibration requires independent-world draws and
explicit prior/data identities. Missing inputs are unavailable or not applicable,
not zero scores or successful calibration checks. One-world rank screening and
finite graph-mixture variance are not calibration guarantees.

## Training resources and small workflow

The public canonical configuration specifies 62,500 AdamW updates of 64 worlds,
seed 31, a 128-dimensional encoder with four layers and eight heads, and flow
width 256. Profile A specifies 200,000 synthetic worlds with five direct-effect
channels; profile B specifies 400,000 worlds with four to ten. These are
training-prior settings, not causal activity inferred from observation columns.
They specify the workload, not a GPU-memory requirement or a wall-clock
performance guarantee.

```bash
uv sync --locked --extra generation --extra training
uv run --no-sync python examples/workflow.py --out runs/workflow
```

This example generates twelve synthetic worlds with 16 rows and an eight-row
context, then trains the full architecture for two optimizer updates. It
exercises checkpoint and safetensors loading, observation-only inference,
observation-only and synthetic-truth reports, and inference-free report replay.
It is an integration exercise, **not usable pretrained weights** or evidence of
full-budget training quality.

The workflow explicitly selects a local SQLite MLflow backend. It reads actual
backend metric histories, downloads checkpoint/report artifacts and checks their
hashes against the producing run's records. The saved metric-name table resolves
backend keys. Outputs require a new directory and are not published. See the
[README](README.md) for the inference, diagnostics and installed-distribution
entrypoints.

Fresh-install verification passed the full workflow from the built sdist,
offline inference from a base-only wheel environment, and saved-result
render/replay from a diagnostics-only wheel environment. Each ran with Python
`-I`, verified noneditable installed imports, and checked its dependency set.

One base-wheel CPU smoke on an Apple M3 Max (macOS arm64, Python 3.13.9,
PyTorch 2.14.0, two CPU threads) measured 3.83 seconds to load the full model,
0.22 seconds for four context draws and 0.17 seconds for four query draws.
Each window had eight rows and 26 components; peak **whole-process** RSS was
379.75 MiB. These are single-run observations using the two-update example,
not a benchmark distribution, hardware minimum or trained-model quality claim.

None of these workflows establishes business-domain calibration, causal
identification, out-of-distribution validity, lift-test evidence or safe
unattended budget optimization.

## Before distribution

- Maintainers must choose and approve licenses for the MFM code and any model
  weights before distributing them; those licenses are not yet declared.
  Approve any accompanying data or diagnostic evidence for redistribution too.
- Any distributed pretrained model needs an immutable bundle reference,
  checksums, a model card and a clearly stated evaluation scope. The two-update
  example is not a pretrained model release.
- Hub publication needs an authorized destination and an approved artifact
  license. A real upload, pinned download and offline reload must be verified
  for the intended artifacts before advertising that distribution path.
- Version/tag selection, repository visibility, package-registry publication
  and model publication each require explicit maintainer authorization.
- The generator intentionally follows a Git branch. Source and directly
  installed wheel/sdist distributions support that dependency; PyPI
  [does not accept direct-URL dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies).
  Any future PyPI release needs a separate distribution-policy decision.
