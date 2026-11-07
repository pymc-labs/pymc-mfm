# Release preparation

PyMC MFM remains at **`0.0.1.dev0`**. These notes describe the standalone
library's development capabilities and distribution prerequisites, not an
approved or published release.

## Changes

- **Light default install and an HTTP client for the River API.** PyTorch and safetensors
  moved from the base dependencies to a new `local` extra. A plain `pip install pymc-mfm`
  now installs NumPy, pandas, httpx and packaging only, and `import pymc_mfm` no longer
  imports PyTorch. Model names (`MFM`, `ModelConfig`, `decompose`, `from_pretrained` and
  the rest) load on first use and raise an `ImportError` naming `pymc-mfm[local]` when
  PyTorch is missing. This is a breaking change for users who relied on the default
  install running models: install `pymc-mfm[local]`. The extras `hub`, `training`,
  `diagnostics`, `arviz`, `generation` and `examples` include `local`, so they behave as
  before. New `pymc_mfm.client` is a torch-free client for the River API v3.1: datasets,
  analyses with scenarios and curves, `explore`, pandas results, per-draw xarray files and
  two plots. New extras `draws` (`xarray`, `h5netcdf`, `h5py`), `plot` (`matplotlib`) and
  `client` (both). CI adds a `light` job that tests the client without PyTorch.
- **Opt-in compiled sampling.** `model.compile_sampling()` compiles the per-step flow
  velocity with `torch.compile`; `warm_up_sampling` compiles representative inputs
  ahead of use, and `model.sampling_status` reports compilations and any eager
  fallback. Sampling stays eager unless a caller opts in, and eager draws are unchanged.
  Compiled draws agree with eager draws up to float round-off. Seeded draws do not depend on whether a call
  compiled, and compilation adds nothing to checkpoints.
- **Quickstart actual-spend scenario sweep.** Each channel is varied separately
  from 0% to 200% of its actual eight-week query spend path, with unchanged history
  and other inputs. The notebook plots eight-week channel totals with pointwise
  90% approximate posterior intervals, including graph uncertainty but excluding
  observation noise. Generator-computed adstock/saturation truth is overlaid in the
  same units, with the original 100% spend path marked separately. These conditional
  responses are not identified causal lift.
- **Quickstart context spend-contribution scatter plots.** One panel per channel
  overlays River-1's weekly posterior mean and simulated true contributions against
  raw spend, using only context weeks; no adstocked-spend axis or query points.
- **Context/query DataTree snapshots.** After dataframe `add_context` and `inference`,
  `model.build_idata()` exports separate context/query marginals with named
  `date`, `channel` and `control` axes; `store=True` also exposes the returned
  tree as passive `model.idata`. Context sampling is deferred and cached per
  prediction; query is never resampled. Successful context loads/inference calls invalidate
  model-held exports without rewriting earlier caller-held trees; failures
  preserve the previous state. The optional `[arviz]` extra remains lazy, and
  runtime prediction/export state stays out of weight bundles. Existing generic
  single-result and archive export schemas are unchanged.
- **Dataframe `add_context` / `inference`.** Load weights with `from_pretrained`, then call
  `model.add_context(history, date_column=..., target_column=..., channel_columns=...,
  control_columns=...)` to load observed context without training the weights.
  `model.inference(future_data)` returns graph-conditional decomposition draws in
  original outcome units. It handles control padding, masks and context-only
  scaling internally, ignores query outcomes, and retains the same history for
  independent future scenarios. Dates are sorted and must follow the context's
  regular cadence without gaps. Loading new context replaces it atomically; weight
  bundles never contain it. At least one channel and one control remain required.
  The old encoded-tensor `model.predict` is now `model.predict_batch`; all internal
  callers have migrated, without an overloaded input contract or compatibility alias.
- **Native inference dates.** Results retain `datetime64` coordinates through
  save/load, plots, report replay and DataTree export; diagnostic JSON uses ISO
  strings rather than integer timestamps. Existing numeric labels and result
  format version 2 remain supported.
- **Faster sampling, same draws.** The router and conditional flows evaluate
  only state coordinates live in some task (padding and graph-forbidden
  contributions are skipped) and project each network's step-invariant inputs
  once per call. Base noise is still drawn per chunk of 16 draws in the same
  order, but many chunks are integrated together (bounded by a state-row
  budget), and each distinct conditioning graph is passed through the graph
  conditioner once. Conditional sampling draws router graphs with
  `MFM.sample(..., structure_only=True)`, which skips the confounder and
  indirect totals that no graph coordinate depends on. On Apple silicon CPUs up
  to eight balanced batches run at once, so Accelerate's matrix unit overlaps
  elementwise work; this never exceeds `torch.get_num_threads()` and gives
  bitwise the same draws as running them in turn. Seeded draws match the
  previous sampler up to float round-off; weights, bundles and the training
  objective are unchanged. River-1 on the quickstart world, 1,000 draws (Apple
  M3 Max): CPU query 50.2 s → 2.0 s and context 65.6 s → 3.2 s; MPS query
  16.4 s → 1.4 s and context 20.6 s → 2.3 s. Peak memory is lower than before.
  `benchmarks/inference_speed.py` reproduces the CPU timing.
- **Joint coverage and coverage uncertainty.** A studentized 80% simultaneous
  band scores each world's truth-live media and control window totals as
  `timeavg_{contribution,control_contribution}_simcov80_live`; the existing
  pointwise metrics and checkpoint selection are unchanged. One-result window
  totals show the channel joint event beside pointwise coverage. Coverage-only
  world/cell evidence is retained in evaluation tables, not scalar events or
  checkpoints. `TrainingRun.derived_metrics()` and diagnostic reports reuse the
  comparison bootstrap for report-only 95% intervals, resampling generator
  cells while preserving equal-world means. Missing cell identity or fewer
  than two cells gives explicit unavailable bounds; saved report snapshots
  rerender without inference and old reports mark missing quantities not recorded.
- Shared context-fitted input scaling: per-channel mean spend, per-control RMS,
  and outcome p95 before all features. Original outcome units are restored in
  float64 for every decomposition draw. Both training and inference use this path.
  Positive unit changes do not require broader raw-magnitude priors; this does
  not establish calibration for new causal structures or relative effects.
- Relative evaluation and diagnostic scores remain in context units, and latent
  targets are normalized before reduction to avoid raw-unit overflow.
- Observation-only inference checks numeric domains, padding, masks and tensor
  structure before precision narrowing. Training targets never enter inference;
  all padded control feature blocks are zero.
- Model bundles use strict configuration and safetensors loading. Checkpoint,
  training-run and data identities remain attached to exports and results;
  caller annotations cannot replace a loaded checkpoint's identity. Loaded weights
  are copied out of the file buffer into aligned allocations, so on CPU a bundle
  reproduces its source model's outputs bit for bit with the same thread count,
  libraries and hardware (unaligned weights changed x86 BLAS rounding).
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
  - Original dimensional A/B weights require retraining for semantic representation 2; matching tensor shapes are not compatibility.
  - Bundle provenance can record the original `source_checkpoint_sha256` and the recorded training evaluation (`training_evaluation`, aggregation classes `world`, `slot`, `slot_pooled`, `pooled_edges`).
  - `push_to_hub` uploads per-profile model cards, a collection index and an optional recorded-output sidecar; `write_model_cards` renders the cards locally. Cards show a training plan only for an explicit `corpus_plan` whose shards hold every recorded evaluation world, and a location never mixes a root bundle with profiles.
  - The sidecar is bound to the weights' `state_sha256` and the bundle's control and latent widths, validated strictly (at least one active channel and control, finite float64 numbers, int64 seeds) and uploaded in canonical form. A model whose weights changed after loading inherits neither the loaded provenance nor the loaded profile; the load-time weight digest is kept in `model.artifact_metadata`.
  - `push_to_hub` and `write_model_cards` accept a maintainer-written `description` (Markdown), rendered as the card's `Model description` section ahead of the generated sections. It applies to one bundle and is refused before any network operation if it holds a Hub token, a bearer credential or a credential-bearing link; links are checked one at a time, so headings and variable names in prose are not misread as URL parts.
  - PyMC Labs' first published model, `river-1` (2,443,583 parameters, the final checkpoint of run `profile-c-9add72ea6d35`), is profile `river-1` of the private collection `pymc-labs/mfm`; its training corpus is in the private dataset `pymc-labs/mfm`. `examples/river_1_quickstart.ipynb` simulates PyMC-Marketing-style data with PyMC-Marketing's adstock and saturation transformers (new `[examples]` extra), decomposes it with River-1 (80% context, 20% query, 1,000 draws) and plots each channel's and event's weekly contribution and the posterior of its window total against the truth.
  - Bundles load where the installed versions satisfy the installed release's compatibility constant and every lower bound the bundle records; recorded upper bounds are not enforced, so a later release can relax the constant for published bundles. `numpy` and `torch` are capped below 3 in the dependencies, matching the constant. `save_pretrained` refuses an environment outside it or whose package versions are not release versions (canonical PEP 440 without a local label; torch alone may carry the build label of a CPU, CUDA, ROCm or XPU wheel, including the CPU cxx11-ABI and AMD ROCm `.git<commit>` wheels, or the commit of a source or NVIDIA NGC build), and `train` runs the same check before writing anything; the refusal names the installed package and version. Exports record versions as `packaging` normalizes them; the installed Python version comes from `sys.version_info`.
  - A and B are distributed through the Hub only, and are not yet published.
- **Corpora.**
  - `SHARD_PLANS` stores the A, B and evaluation shard plans; `generate --profile A --shard i` and `generate --evaluation-shard i` reproduce them.
  - A drift test compares generator probes against stored hashes and names the generator commit they were recorded with. A weekly `Generator drift` workflow runs it against the generator's tracked branch; regular CI uses the locked commit.
  - `run.json` records a corpus summary and per-module parameter counts, also logged to MLflow at step 0. `corpus_support.json` records per-world context-outcome location and scale (at most 5,000 evenly spaced worlds).
- **Evaluation.**
  - New metric families: live/dead attribution, per-draw closure, temporal coherence, window totals, router graph posterior statistics, edge AUPRC and `null_fpr`, and counts.
  - `attribution_live_{media,control}_conditional_cov80` measures central 80% coverage of signed window-summed true-live effects using each head/window's own graph-aligned edge-on draws. A slot needs at least two edge-on draws; only eligible slots enter its per-world mean and weight, and no eligible slot is undefined rather than zero.
  - Small edge-on draw counts make sample-quantile interval coverage non-nominal even under a calibrated posterior; the independent-cell randomized PIT is the formal check.
  - `conditional_sbc_report` records separate conditional and model-average rank/ECDF checks on the same independent generator cells (at most one world per cell per corpus), using only edge-on draws for truly live conditional effects, including signed controls. Pooled continuous PIT and eligible full-draw discrete tests share Bonferroni family-wise control. Unavailable or low-rank checks are not passes; this is a prior/data simulation check, not out-of-distribution validation. The evaluation shards use the A prior, so B bundle checks on those cells include a training-prior shift rather than isolating its magnitude head.
  - Also recorded: reliability/CORP/threshold tables, per-metric world counts and optional per-world values.
  - CORP edge calibration: `isotonic_calibration` (a pure-NumPy weighted pool-adjacent-violators fit that pools tied probabilities and compares blocks exactly on counts) and `brier_decomposition` in `pymc_mfm.training.metrics`. Edge families record `mcb`, `dsc` and `unc` (`brier = mcb - dsc + unc`, `pooled_edges`, dimensionless) and a `corp` table beside `reliability` and `thresholds`. A family with one truth class records them `null` with reason `one_class` and an empty `corp` table; its existing `ece` is unchanged, so recorded evaluations stay comparable. `unc` is a truth summary like `prevalence`: it stays defined when predictions are non-finite, and `check_recorded_evaluation` compares `joint_router/edge/*/unc` exactly and reports the router's `mcb`/`dsc` as informational.
  - Model cards note "no positive edges — calibration not evidenced" (or "no negative edges — …") beside the calibration scores (`ece`, `mcb`, `dsc`, `unc`) of an edge family recorded with one truth class, and `TrainingRun.derived_metrics()` leaves such families out of `mean_ece`.
  - Stratified readouts (`pymc_mfm.diagnostics.strata`): one row per active channel and control effect of each world (`effect_rows`, `result_effect_rows`) with its window-total truth, posterior mean, central 80% interval and CRPS in `sales_scale` units, its direct `cy`/`zb` edge frequency in the same draws and its graph-truth label. `strata_summary` groups them per family by true and predicted size bin (an exact `zero` bin, then right-closed bins at stored `cuts`: quartiles of the worlds' nonzero absolute true totals, so replays and other models on the same worlds bin identically), edge-frequency bin, active channels and live channels, with counts, coverage, CRPS and signed/absolute residuals (never divided by the truth). `misclassification` joins each threshold's TP/FP/TN/FN edge call (`p >= t`) with the same values. Empty strata are `unavailable` with a reason, never zero.
  - `evaluate_rows` records `strata[head][window]` for the selected heads, and `per_world=True` adds each world's rows. Training writes them only to `evaluation_tables/step_<N>.json`; `evaluations.jsonl`, checkpoints and MLflow do not carry them, and no selection gate reads them.
  - Checkpoint gates validate `METRIC_BOUNDS` and include the selected head's context energy score; a legacy held-out metric of that head with non-finite worlds blocks promotion (`selection.nonfinite_keys`). The two additive simultaneous-coverage families are excluded from this guard, preserving checkpoint selection while retaining their unavailable reasons.
  - `check_recorded_evaluation` reproduces a bundle's recorded training evaluation within stated tolerances.
  - MPS evaluation no longer converts tensors to zeros.
  - Evaluation archives (`pymc_mfm.diagnostics.EvaluationArchive`) keep the evaluator's conditional captures for cross-world reports. `EvaluationArchive.create(out_dir, identity)` returns a writer whose `add` is a `result_sink` and whose `close` publishes atomically. The layout (`manifest.json` plus one saved `DecompositionResult` directory and `truth.npz` per world window) and its strict manifest declare the model, the corpora and the evaluation, including the call plan with consecutive seeds, and fix the archive order. `EvaluationArchive.load` reads every file once through a bounded reader that refuses symlinked archive files and entry directories, size- and hash-checks every byte it decodes before use, and names the offending file in `ArchiveIntegrityError`. Reads yield `(result, truth, graph_truth, observed)` 4-tuples in capture order, and converters feed `strata_summary` (every world), `sbc_report` (family window totals) and `conditional_sbc_report` (both with one world per generator cell), all without a model call. `archive_identity`, `evaluate_into_archive(worlds_per_call=...)` and `python -m pymc_mfm.training evaluate` (4 worlds per call by default) build and write archives, and `train(holdout_archives=...)` / `--holdout-archives` archives a run's held-out evaluations. Only results whose inputs equal a declared corpus world are accepted, and identities from `archive_identity` are marked synthetic only with a hash-bound sidecar (hand-built identities are trusted as written). `DecompositionResult.load` is split into private steps so that archives decode the bytes they hashed; `load`'s behaviour is unchanged (no user action).
- **ArviZ export.**
  - New optional `[arviz]` extra (`arviz-stats[xarray]`; the lockfile resolves it with no new packages). `DecompositionResult.to_datatree(observed=None, truth=None, graph_truth=None)` and `EvaluationArchive.to_datatree(window)` export draws as an ArviZ `xarray.DataTree` and raise an `ImportError` naming `pymc-mfm[arviz]` without it. The export imports xarray and arviz-base when called; `import pymc_mfm`, `decompose` and `import pymc_mfm.diagnostics` never do.
  - A result tree holds `posterior` `components[chain=1, draw, time, component]` and `graph[chain=1, draw, edge]` on the active slots only (padded slots are absent, not zero), `constant_data` component and edge families and edge endpoints, the caller's `observed` as `observed_data` and synthetic `truth`/`graph_truth` in a separate `truth` group. An archive tree adds a `world` dimension over the union of the worlds' time labels and active components (`NaN` where a world has none; sum with `min_count=1`), compacts active edges to `world_edge` pairs (`<world_id>|<edge>`) and keeps per-world identities, cells and independence in `constant_data`.
  - Root and `posterior` attributes record window, mode, units, draw count and digests (a result's `metadata_sha256` is the one `save` writes to `result.json`) and state that the draws are one chain of independent amortized draws (`mcmc_diagnostics="not_applicable"`): R-hat is undefined; ESS and MCSE describe only Monte Carlo variability, not convergence or amortization accuracy. One tree holds one window: context and query stay separate trees. Export only: saved results and archives stay the evidence, MFM never reads a tree back, and writing netCDF needs a backend the extra does not install.
  - Archives and exports share one private evidence check, and `save` and the export one metadata digest; saved bytes, rules and messages are unchanged (no user action).
  - The installed-distribution check adds an arviz-only wheel lane; its base and diagnostics lanes check that xarray and arviz-base are absent and that the export names the missing extra.
- **Plot suite.**
  - Figures no longer use pyplot and carry a PNG representation for notebooks. Result, diagnostics and comparison figures state head, window, mode, units and draw count; run evaluation figures state units, scope, legs, draws and head labels.
  - New report panels: attribution, window totals, cumulative contribution, temporal coherence, edge reliability, graph statistics (every `joint_router/graph` statistic as `graph/<key>`, the truth-free ones also without graph truth), cycle check (complete graph plus cc/zz cycle and two-cycle rates) and input support. `run.plot.corpus()` plots the corpus summary and `run.plot.corpus_support()` the world-level context support.
  - Edge reliability draws the CORP isotonic curve by default (`diag.plot.reliability()`, with the pooled `graph/{mcb,dsc,unc}` decomposition); `diag.plot.reliability("binned")` keeps the equal-width bins. An edge set with one truth class draws no curve and its legend states that calibration is not evidenced; when every active edge has one truth class, `graph/ece` and the reliability panel are `not_evidenced` in every graph mode. Reports of evidence saved before CORP fits rerender their bins.
  - Residual and strata figures: `plot_strata_table` (recorded or computed strata tables), `plot_strata` and `plot_residual_by_magnitude` (one result's effects). Reports add the `residual_true_size`, `residual_predicted_size` and `strata` panels; `diag.plot.residuals("true" | "predicted")`, `diag.plot.strata(metric, by=...)` and `run.plot.strata(metric, by=..., head=, window=, leg=, step=)` take `ax=`. Unavailable strata are marked `n/a`, not drawn as zero.
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
- **Examples.** `examples/` holds only the River-1 quickstart; the diagnostics notebook moved to `examples/dev/`. The two-update workflow and offline inference scripts that the installed-distribution check runs moved to `.github/scripts/`, and `examples/tracking.py` was removed.
- **README.** Rewritten as a short usage guide: installation and extras, inference input and output shapes, the training corpus format, training, loading local models and loading from the Hugging Face Hub. The evaluation-archive size, memory and time guidance moved to the `pymc_mfm.diagnostics.archive` docstring.

### Breaking changes

- `MFM.add_context` replaces dataframe `MFM.fit`, and `MFM.inference` replaces
  dataframe `MFM.predict`. Arguments and results are unchanged; `predict_batch`
  remains the separate tensor-level sampler.
- Semantic schema **2** and training-checkpoint format **2** require context-fitted
  unit-invariant preprocessing. Old A/B bundles/checkpoints are deliberately
  rejected with retraining guidance; no legacy mode or schema-only conversion.
  Existing raw corpora remain usable for fresh training.
- `sales_scale` is float64 CPU metadata, not a model input. Tensor-level callers
  restore units after moving draws to CPU/float64; `decompose` does this automatically.
- Zero-context/nonzero-query columns are rejected rather than assigned an
  arbitrary native-unit scale. Nonzero constants and all-zero columns remain valid.
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
- Diagnostic report keys `attribution/{channel,control}/{live,dead}/*` are replaced by `attribution/{live,dead}_{media,control}/*`. `PANEL_NAMES` gains eleven entries.
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
and plotting, `[training]` adds diagnostics and MLflow, `[hub]` enables remote
artifact transport, `[arviz]` adds ArviZ `DataTree` export (xarray), a view
of draws rather than evidence, and `[examples]` adds `pymc-marketing` for the
River-1 quickstart. Installing dependencies can require network
access; local inference and saved-evidence replay do not. Remote loading
resolves the selected revision, or the default branch when omitted, to one
recorded immutable commit. Use a full commit for reproducibility; offline Hub
loading needs a populated cache.

Context and query inference calls are independent marginal evaluations, not
paired full-horizon draws. `model_average` samples router graphs, `map` uses
the empirical complete-graph mode, and `fixed` conditions on a supplied graph
hypothesis. A supplied graph is not posterior graph evidence or a lift-test
likelihood; auxiliary structure probabilities do not route these calls.

Truth-based diagnostics require aligned truth, training curves require recorded
histories, and reference diagnostics require usable, qualified reference
evidence. Independent-world calibration requires independent-world draws and
explicit prior/data identities; conditional SBC additionally requires independent
generator cells, not repeated worlds or posterior draws counted as new rank
replicates. Missing inputs or insufficient eligible ranks are unavailable or not
applicable, not zero scores or successful calibration checks. One-world rank
screening, conditional-SBC non-rejection and finite graph-mixture variance are
not calibration guarantees or out-of-distribution claims. Archive SBC declares
independence for the first world of each generator cell of distinct corpora,
and archive hashes detect changes, not who made them.

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
uv run --no-sync python .github/scripts/workflow.py --out runs/workflow
```

This workflow generates twelve synthetic worlds with 16 rows and an eight-row
context, then trains the full architecture for two optimizer updates. It
exercises checkpoint and safetensors loading, observation-only inference,
observation-only and synthetic-truth reports, and inference-free report replay.
It also checks graph equality and restoration of all contribution draws in both
windows under Case-10-style, tiny, and independently rescaled channel/control
units. `summary.json` records each comparison. It is an integration exercise,
**not usable pretrained weights** or evidence of full-budget training quality.

The workflow explicitly selects a local SQLite MLflow backend. It reads actual
backend metric histories, downloads checkpoint/report artifacts and checks their
hashes against the producing run's records. The saved metric-name table resolves
backend keys. Outputs require a new directory and are not published. The
[README](README.md) covers installation, training and inference. The
installed-distribution check runs after `uv build` with
`uv run --no-sync python .github/scripts/installed_smoke.py --work-dir artifacts/installed`.

Fresh-install verification passed the full workflow from the built sdist,
offline inference from a base-only wheel environment, saved-result
render/replay from a diagnostics-only wheel environment, and `DataTree` export
with ArviZ summaries of both saved windows from an arviz-only wheel environment;
the base and diagnostics environments have no xarray and refuse the export with
an `ImportError` naming `pymc-mfm[arviz]`. Each ran with Python `-I`, verified
noneditable installed imports, and checked its dependency set.

One base-wheel CPU smoke on an Apple M3 Max (macOS arm64, Python 3.13.9,
PyTorch 2.14.0, two CPU threads) measured 3.83 seconds to load the full model,
0.22 seconds for four context draws and 0.17 seconds for four query draws.
Each window had eight rows and 26 components; peak **whole-process** RSS was
379.75 MiB. These are single-run observations using the two-update workflow,
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
  workflow is not a pretrained model release.
- Hub publication needs an authorized destination and an approved artifact
  license. A real upload, pinned download and offline reload must be verified
  for the intended artifacts before advertising that distribution path.
- Version/tag selection, repository visibility, package-registry publication
  and model publication each require explicit maintainer authorization.
- The generator intentionally follows a Git branch. Source and directly
  installed wheel/sdist distributions support that dependency; PyPI
  [does not accept direct-URL dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies).
  Any future PyPI release needs a separate distribution-policy decision.
