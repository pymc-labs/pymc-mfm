# Design: one-call inference and model-owned idata

Status: proposal, not implemented. Stacked on [PR #115](https://github.com/pymc-labs/pymc-mfm/pull/115), head `c9c6c231516217961ded867ed2ad1fe265df7d70`.

## Recommended API

Make **`model.inference(context, query, ...)` the complete operation**: validate both frames, sample both marginals, and return the same populated tree stored as `model.idata`. There is no preceding context-attachment call or subsequent export call.

| API | Responsibility | Return and stored-state effect |
| --- | --- | --- |
| `model.inference(context, query, ...)` | Complete inference for an explicit history/future pair | Returns and stores a new complete `xarray.DataTree`; atomically replaces the previous run |
| `model.predict(query, ...)` | Query-only decomposition using the last successful inference's history and column roles | Returns a fresh `DecompositionResult`; does not modify the stored inference, idata or curves |
| `model.response_curves(...)` | Sweep scenarios around the last complete inference | Returns `ResponseCurves` and attaches/replaces its subtree in a new `model.idata` root |
| `model.idata` | Read the last complete inference, with any attached curve output | Passive property; `None` before the first successful inference |
| `model.query_result` | Access the exact query `DecompositionResult` from the last complete inference | Passive property; `None` before the first successful inference |

`inference` requires both dataframes and keyword-only column roles: `date_column`, `target_column`, `channel_columns`, `control_columns`. Keep the existing sampling options: `n_samples`, `mode`, `graph`, `router_samples`, `units`, `provenance`. Roles are explicit on each complete call; do not guess them from column names or a previous run.

`predict` inherits only the stored history and role schema. It accepts the current query-only inference's sampling options/defaults; it does not silently inherit a previous draw count or fixed graph bank. `mode="fixed"` still requires explicit `graph`. Calling it before a successful complete inference raises a clear error. It never appends predictions to history or samples context.

**State decision:** prediction is a calculation, not a replacement of the stored complete analysis. `model.plot` and `response_curves` continue to describe the last successful `inference`, not an intervening `predict`. To make another plan the stored/plotted analysis, call `inference(context, new_query, ...)`. This avoids retaining context draws from one plan alongside query draws from another. Do not add an `update_idata` flag or a partially populated query-only model tree.

### What this simplifies

Remove public `MFM.add_context` and `MFM.build_idata`; keep context validation/snapshotting private. The earlier proposal's data-only idata state is no longer necessary: before inference there is no tree, and after inference both marginals are ready. Rename the old query-only operation to `predict` rather than overloading `inference` to mean two different things. Preserve array-level `decompose`, `DecompositionResult`, and standalone result persistence/export.

The `query_result` reader is needed because existing diagnostics, graph inspection and evidence-saving APIs consume a `DecompositionResult`. It exposes the already-computed query result without reconstruction, copying or sampling. A result returned by `predict` is independent and is not installed as `model.query_result`.

## Why context and query belong in one call

At the PR #115 base:

- [`MFM.add_context`](../../src/pymc_mfm/model/model.py) validates/stores history; `MFM.inference` samples query only and retains `_PredictionSnapshot`.
- `MFM.build_idata` samples missing context draws and converts the windows; only `store=True` populates idata.
- [`ModelSource.window`](../../src/pymc_mfm/plotting/charts/_data.py) also triggers that context sampling on the first context chart.
- [`prediction_datatree`](../../src/pymc_mfm/inference/_datatree.py) already separates the two posterior marginals, but omits input covariates and query observations.
- [`observation_batch`](../../src/pymc_mfm/inference/api.py) requires a nonempty query. Context draws depend on the actual context-plus-query encoder input, including future covariates and relative time positions.

Changing a future plan or its horizon can therefore change context draws even when history is unchanged. Accepting both inputs at the sampling boundary makes this dependency explicit. Never manufacture future rows, silently hold out history, or use dummy covariates to obtain context draws.

One call does **not** change the statistical model: context and query remain separate marginals, not paired full-horizon trajectories. Matching draw indices do not pair their samples. Contributions exclude observation noise and are not automatically causal lift.

### Upstream precedent

[PyMC `sample`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) returns inference data by default. [PyMC-Marketing's inspected `ModelBuilder`](https://github.com/pymc-labs/pymc-marketing/blob/6f88b343a9837e4b68577f30c754819cb90bb36c/pymc_marketing/model_builder.py) stores and returns idata from `fit`, and removes posterior-derived outputs when replacing the posterior. Its prediction helpers illustrate that a convenience result return need not match the main inference return.

Keep this repository's `xarray.DataTree`; do not introduce legacy `arviz.InferenceData` alongside it. Here, “idata” names that tree. PyMC's stable sampling documentation names `InferenceData`; those Python types are not interchangeable. MFM draws are amortized conditional draws, not MCMC samples.

## Lifecycle, overwrite and ownership

Construction and loading a weight bundle leave idata, query result, context and curves unset. A successful `inference` installs the context snapshot, complete prediction, both marginals and tree together. Repeating it replaces that entire run and removes curves, whether history, plan, sampling settings or only provenance changed. Identical inputs still sample afresh.

A shorter plan replaces a longer plan: no leftover dates, coordinates, observations or variables. Every complete inference regenerates both marginals; do not cache context draws merely because history is unchanged. Failed validation, sampling or conversion preserves the entire last successful run and its curves.

A successful `response_curves` call replaces its entire previous subtree, including channels/levels omitted on the rerun. Internal scenarios do not change the base inference. A failed sweep or conversion leaves the last curve result/tree intact. `predict`, `input_usage`, property reads and charts never publish changes to the stored run; sampling in `predict` still consumes random numbers.

Each successful publishing operation assigns a newly assembled root to `_idata`. `model.inference(...) is model.idata` holds immediately after the call; repeated property reads return that same current root. Earlier caller-held trees and results remain snapshots and must not be rewritten by later calls. Curve attachment does not replace `query_result`.

Copy tree structure and mutable attribute mappings, but reuse unchanged owned/read-only numeric buffers where safe. Do not reparent live DataTree nodes, share mutable tree structure between published snapshots, or deep-copy all posterior arrays merely to attach curves. Required named-slot/dtype conversion copies are acceptable. Verify xarray aliasing behavior rather than assuming a shallow copy isolates everything.

The public tree is read-only by convention. Users copy before editing; editing a tree is not a supported way to change neural inputs or model state. Keep the workflow sequential: concurrent mutating calls on one model are unsupported, with no new locks, background tasks, results registry or export history.

## Tree schema and provenance

Preserve existing named posterior variables, active-slot filtering and separate context/query paths. Each subtree owns its axes; the convenience root has no date, draw, channel, level or edge coordinates.

| Path | Contents | Published by |
| --- | --- | --- |
| `/context/observed_data` | Sorted original-unit `outcome(date)` | `inference` |
| `/context/constant_data` | `media(date, channel)`, `controls(date, control)` and available component/edge descriptors | `inference` |
| `/context/posterior` | Existing named decomposition variables and graph draws | `inference` |
| `/query/constant_data` | This plan's original-unit covariates and descriptors | `inference` |
| `/query/observed_data` | Optional `outcome(date)` with NaN for unobserved dates | `inference`, only when any finite outcome is supplied |
| `/query/posterior` | Existing named decomposition variables and graph draws | `inference` |
| `/response_curves/posterior` | `contribution(chain, draw, channel, level)` | `response_curves` |
| `/response_curves/constant_data` | `media(channel, level)` totals and actual sweep dates on a local date axis | `response_curves` |

Window posteriors retain `chain=1`; contributions have `chain, draw, date` plus channel/control where appropriate, and graph has `chain, draw, edge`. Preserve omission of padded controls/inactive graph slots and the converter's zero-edge behavior. Configured role names/order and target/date column names remain serializable metadata; role labels are coordinate values, not group paths.

Query outcomes follow PR #115's `query_outcome` sanitization and remain display/evaluation-only. They never enter encoder inputs, scaling, graph selection or inference input identity. A plan with no finite observed outcome has no query observation group. Changing only query outcomes cannot change seeded draws.

Keep units and existing scalar identity attributes on each inference window. Store its complete canonical result provenance as JSON text in `attrs["provenance"]`, including authoritative `input_usage`, sampling configuration and nested caller annotations. Readers JSON-decode that attribute, or use `model.query_result.provenance`. Reuse existing JSON validation/canonicalization, rejecting unsupported/nonfinite metadata before sampling rather than stringifying arbitrary objects. Generic standalone result exports keep their existing contract.

Add a model-tree schema version and one model-local `inference_revision`, advanced only on successful complete inference. Curves record the source revision, actual sweep sampling settings and units; levels/dates remain coordinates/data. Base-run provenance is separately labeled JSON `source_provenance`, not a claim that every truncated/scaled scenario has the original full-plan input identity or usage. No separate context revision is needed because context is installed only as part of a complete run.

Preserve the notes about independent amortized draws, inapplicable MCMC convergence diagnostics and separate marginals. Curve level draws are not paired; the unchanged reference may be reused across channels/zero-input levels. Do not add synthetic observation noise, posterior-predictive/likelihood/sample-stat groups or invented evidence. ArviZ summaries target a window subtree such as `model.idata['query']`, not the root; curve draws have their own axes and may use a different count.

## Execution boundary

### Complete `inference(context, query, ...)`

1. Validate/snapshot candidate history with `_FittedContext`, then validate the query against those candidate roles, cadence and context-only scales. Validate sampling options, exportable labels and provenance; capture query observations separately. Never attach candidate context to the live model as an intermediate step.
2. Compute query first, preserving the current seeded query-sampling order. Compute context once from the same actual pair and canonical sampling settings. Fixed mode replays the exact per-draw query graphs, as the current `_context_result` does.
3. Build candidate `_PredictionSnapshot`, completed context result and complete DataTree without changing published fields.
4. After all work succeeds, publish candidate context, prediction, context result, idata and revision together, removing old curves.

A first-call failure leaves an uninitialized model; a later failure leaves the prior complete run. Validation errors occur before sampling. Random numbers consumed by failed sampling are not rolled back; unpublished query/context candidates are discarded rather than hidden in a partial cache. Preserve existing restoration of training/eval mode and unchanged weights/gradients.

The full call deliberately pays for both marginals and their analysis representation. Measure latency and retained/peak memory; do not promise current query-only costs. This is why `predict` remains a separate, cheaper query-only operation rather than a flag that changes inference's meaning.

### Query-only `predict(query, ...)`

Use the context/roles from the last successful complete run, validate the new query/options and call the shared query computation. Return its `DecompositionResult` without publishing any context, prediction, curve, idata or revision changes. Do not call complete `inference` internally, convert a tree, or generate context draws. Successful and failed predictions leave the stored run unchanged; only sampling may advance RNG. New query dates start immediately after the stored history, not after previous query predictions.

Factor the current query calculation into a private helper accepting an explicit candidate context, so both methods reuse it without temporarily installing model state. Stateless `decompose` remains available for independent array-level query inference without establishing a complete dataframe session.

### Readers, curves and preflight

Remove sampling from `_context_result`; make it a checked reader or remove it. Every model chart reads the last complete inference and its attached curves, never an intervening prediction result. Before complete inference, charts raise a clear prerequisite error; there is no history-only public chart state. Independent prediction-result diagnostics can use the returned result directly.

Build curves and their tree off to the side, then publish `_response_curves` and a new root together. Internal scenarios continue to use stateless `decompose`, never state-publishing inference; no context marginal is needed per sweep level. Charts and tree values share completed numerical results, not independently regenerated draws.

Preserve no-sampling preflight through `input_usage`: accept explicit context/query and role inputs for a new run, sharing the candidate validation path without a prior successful inference or state publication. Preserve context-only usage inspection without inventing a zero-length query; migrate existing stored-context examples to explicit inputs. Do not require expensive inference merely to inspect request size.

## Dependencies and persistence

Add explicit `xarray>=2024.10` and `arviz-base>=1.3,<2` dependencies to `local` and update the lockfile; the lock already contains ArviZ Base 1.3.0. Normal local inference must construct idata without an optional export extra. Do not pull in PyMC, PyMC-Marketing, statistics or plotting just for tree construction. Base HTTP-client imports remain lightweight; the `arviz` extra continues to supply optional statistics.

Core lifecycle, prediction isolation, curve conversion, rollback and bundle privacy acceptance runs with `local` plus the test runner, without statistics/plotting. Split the current `needs_arviz` gate, whole-file matplotlib skip in `test_response_curves.py` and combined lazy-import check in `test_datatree.py`: mandatory dataframe behavior cannot be skipped because optional rendering/statistics are absent, while standalone array/result operations retain lazy tree imports.

Analysis-tree file export remains explicit and requires the existing `draws` extra with `engine="h5netcdf"` for netCDF roundtrips. `local` guarantees an in-memory tree, not a file backend. `query_result.save`/`DecompositionResult.load` remain evidence persistence, distinct from analysis-tree export. Neither restores an attached runnable session.

Trees may contain sensitive outcomes, covariates, labels and results. `save_pretrained` and Hub weight bundles must never capture runtime context, predictions, curves or idata. Loading weights leaves all runtime analysis state unset. No automatic file writes or persistence redesign.

## Migration and implementation sequence

This PR contains only the proposal. These are future work units, not completed changes.

| Order | Files / surface | Work and acceptance |
| --- | --- | --- |
| 1 | `inference/dataframe.py`, `inference/_datatree.py` | Share candidate context/query validation and complete-tree/curve assembly; keep generic result/archive exports stable. Prove sorted original-unit inputs, role order and isolated axes. |
| 2 | `model/model.py`, `pyproject.toml`, `uv.lock` | Introduce complete paired-input inference, nonpublishing query-only `predict`, passive `query_result` and transactional publication. Remove public `add_context`, `build_idata` and lazy context sampling. Make preflight accept explicit inputs. Prove first-call completeness and prediction isolation. |
| 3 | `inference/curves.py`, `plotting/charts/_data.py`, model integration | Attach curves transactionally; make charts sampling-free and keep internal scenarios nonpublishing. Prove exact numerical agreement and independence from `predict`. |
| 4 | `tests/conftest.py`, lifecycle/DataTree/curve/chart/example tests | Migrate changed contracts, separate core/optional gates, retain standalone import/export coverage, and exercise the acceptance cases below. Remove obsolete two-step/lazy-storage tests rather than preserving shims. |
| 5 | `README.md`, `RELEASE.md`, `docs/guide/decompose.qmd`, `docs/_quarto.yml`, `examples/river_1_quickstart.ipynb` | Make one complete inference call the primary example. Explain `predict` isolation, paired-input dependence, costs and API breaks. Re-execute synthetic examples, review/pin notebook bytes and preserve every graph analysis/export. |

Schema/validation precedes model integration; curve/chart migration and end-to-end examples follow. Use one integration owner for publication state. Before implementing exported-symbol changes, use reference navigation when available and inventory every repository caller, including input-usage examples. Clean cutover: no old `inference(query)` overload or aliases for removed public methods.

Most existing `add_context(history)` plus `inference(query)` callers become one `inference(history, query, roles...)`. Repeated plans that need stored trees/charts remain complete paired calls. Deliberate query-only computations use `predict(query)` after a completed run; their returned result is analyzed independently.

In the quickstart, keep the returned tree for analysis and bind the exact `model.query_result` for components, diagnostics, graph catalogs/ranks, DOT/SVG exports and evidence saving. Neither curves nor `predict` replaces that result. Do not re-run `decompose` merely to recover a result for plotting, or reconstruct masks/padded metadata from the active-only tree.

## Verification plan for implementation

Use existing small synthetic model/dataframe fixtures; assert observable values and state, not source text or incidental wiring.

- **One-call completeness:** on a fresh model, one explicit history/query call returns the stored tree with both marginals, covariates and valid observations. No context setter, export builder or chart is required. Match the canonical query-then-context sequence for model-average, MAP and per-draw fixed graphs, including padding, role order, zero-edge cases, units and provenance. Caller frame/list mutations cannot alter snapshots; weights/gradients and model mode remain intact.
- **Replacement and identical reruns:** change history, roles, horizon, draw count/mode or metadata; the complete new run replaces all old coordinates/observations and drops curves. For identical successive calls without reseeding, capture each starting RNG state and compare its canonical sampling sequence; assert fresh root/revision publication and unchanged older trees, not probabilistic “arrays differ.”
- **Prediction isolation:** predict a different plan/horizon/settings after full inference and curves. Match canonical query-only draws; assert the full tree, query result, context marginal, curve state and revision are unchanged. Cover identical repeated predictions, no-inference prerequisite, fixed graphs, invalid queries and failures after sampling. A subsequent full inference still replaces everything correctly. Prediction RNG advances as query sampling requires, not as a hidden context run.
- **Observations/provenance:** partial/missing/invalid query targets follow existing sanitization. Target-only changes cannot affect seeded draws, scales or inference input identity. Reruns without observations remove the old observation group. Nested provenance and authoritative input usage survive the documented JSON representation.
- **Atomic failures:** invalid candidate history/query/options/metadata; context sampling failure after successful query sampling; posterior conversion failure; mid-sweep and curve conversion failures. Test both fresh and previously populated models; no candidate context or partial run becomes visible and no revision advances. Do not require RNG rollback after sampling.
- **Curves and readers:** compare attached curve draws/totals/dates with `ResponseCurves`, including a different sample count, smaller channel/level subsets, fixed graphs, shorter plans and zero-input reference reuse. Charts/property reads never sample; curve attachment preserves base results. Readers remain on the last full inference after `predict`, with errors before the first full run. Check old-tree buffer/metadata isolation across publishing calls.
- **Exact-result migration:** query-result diagnostics use the original query draw bank. Extend quickstart acceptance beyond finite shapes: compare graph counts/ranks/frequencies, selected endpoints and DOT/SVG topology to that bank; preserve every existing graph analysis/export. Check result evidence save/load separately from analysis-tree export.
- **Preflight and packaging:** explicit-pair/context-only `input_usage` works before inference, validates without sampling and preserves stored state/RNG. Core inference/prediction/curve/rollback tests run in a genuinely local-only environment; optional statistics/rendering cannot hide them. Retain standalone lazy imports and lightweight base-client imports.
- **Privacy:** extend the bundle test with sentinel-bearing context/query/curves and independent predictions; compare configuration, manifest and weights with a runtime-free baseline. Loading restores no context, prediction, curves, query result or tree. Existing tests do not yet establish populated-curve/manifest coverage.
- **Interop:** with `draws` and explicit `h5netcdf`, roundtrip a complete tree with partial observations, nested provenance, authoritative usage and curves having a different draw count. Compare axes/dates, JSON-decoded metadata, missingness and values. Summarize actual window subtrees when statistics are installed.

Run the narrow computational tests after integration, then optional rendering/statistics and the migrated quickstart. Its native SVG exports require system Graphviz (`dot`), not just the Python extras; use the existing synthetic stand-in model so acceptance needs no Hub access. Inspect context/query and curve charts in supported backends. Measure full inference and query-only prediction latency/retained/peak memory on the same synthetic workload; report costs without fabricated targets.

## Alternatives rejected

- **Two mandatory calls, context setter plus inference:** hides a dependency that needs both inputs and forces a data-only intermediate state. One paired call is the primary workflow.
- **One overloaded inference method or a `window`/`store` mode switch:** gives the main method conditional completeness, return types or publication rules. Separate complete `inference` and nonpublishing `predict` keep those contracts fixed.
- **Prediction replaces only query in the stored tree:** would leave context posterior tied to an older future plan. Removing that posterior instead would reintroduce partial model trees. Keep the complete run unchanged.
- **Retain `add_context` as an alias or advanced public setter:** adds another public state transition without helping the one-call workflow. Keep its useful validation private and retain stateless `decompose` for lower-level work.
- **Lazy context sampling in a property or chart:** keeps surprise computation/errors after the main operation. Complete inference finishes both marginals before publishing.
- **One forever-mutated tree or flattened/merged marginals:** risks rewriting caller-held results, stale coordinates and false draw pairing. Publish replacement roots with independent subtree axes.

The API choice is intentionally small: **`inference(context, query)` completes and stores an analysis; `predict(query)` computes an independent query result.** Role/sampling keywords make inputs explicit. This replaces the previous proposal's context-only public lifecycle without changing the underlying probabilistic model.
