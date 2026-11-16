# Design: model-owned inference data

Status: proposal, not implemented. Stacked on [PR #115](https://github.com/pymc-labs/pymc-mfm/pull/115), head `c9c6c231516217961ded867ed2ad1fe265df7d70`.

## Goal and recommended contract

Make inference data part of the dataframe workflow, not a separate export step:

- `model.add_context(history, ...)` creates and stores a populated `model.idata` immediately, without sampling or training.
- `idata = model.inference(plan, ...)` automatically adds the context and query draws and returns the exact object stored as `model.idata`.
- `model.response_curves(...)` adds its result to that tree while retaining its existing `ResponseCurves` return type.
- A successful rerun replaces its previous output and removes dependent outputs. Nothing appends draws, dates, scenarios or run history implicitly.
- Reading `model.idata` and drawing charts never runs inference, converts arrays or consumes random numbers.
- A failed operation preserves all previously published model state. A successful operation publishes a new tree; previously returned trees are not rewritten.

This is a deliberate public API change: remove `MFM.build_idata` and change the dataframe `MFM.inference` return from `DecompositionResult` to `xarray.DataTree`. Keep array-level `decompose`, `DecompositionResult`, and its standalone export/persistence APIs. Do not add a return-type switch, `store` flag, lazy property, or compatibility alias.

Expose a passive `model.query_result` property for the exact `DecompositionResult` already retained by `_PredictionSnapshot`. It is `None` before inference and after successful context replacement; successful inference replaces it, failed calls preserve it, and curve calculations do not change it. It performs no reconstruction, copying or sampling. This small companion accessor preserves result-specific diagnostics, graph inspection and evidence saving without a second inference run or a public DataTree-to-result adapter.

## Evidence and the important modeling constraint

At the PR #115 base:

- [`MFM.add_context`](../../src/pymc_mfm/model/model.py) stores an immutable `_FittedContext`, clears predictions and curves, and sets `_idata` to `None`.
- `MFM.inference` returns a `DecompositionResult`, retains `_PredictionSnapshot`, and clears the context-draw cache, curves and idata.
- `MFM.build_idata` is an explicit, optional-dependency export. Its first call samples context; only `store=True` populates the passive idata property.
- `MFM._context_result` is also called by [`ModelSource.window`](../../src/pymc_mfm/plotting/charts/_data.py), so a context chart may currently sample.
- [`prediction_datatree`](../../src/pymc_mfm/inference/_datatree.py) already separates context/query marginals and uses named components. It does not include input covariates or query observations.
- [`observation_batch`](../../src/pymc_mfm/inference/api.py) requires a nonempty query. Context decomposition uses the actual context-plus-query encoder input, including its relative time positions. A different plan or horizon can change context draws even when history is unchanged.

Consequently, **creating idata at `add_context` does not mean producing a context posterior there**. Populate real history and metadata, but omit posterior groups until an actual plan is inferred. Do not manufacture future rows, silently hold out history, or use dummy covariates to make the sampler run. Context-only decomposition would require a separate modeling/API proposal.

### Upstream precedent, not a schema to copy blindly

[PyMC `sample`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) returns inference data by default. [PyMC-Marketing's `ModelBuilder` at the inspected revision](https://github.com/pymc-labs/pymc-marketing/blob/6f88b343a9837e4b68577f30c754819cb90bb36c/pymc_marketing/model_builder.py) stores and returns idata from `fit`; its fit path also removes posterior-derived groups before updating. Its posterior-predictive method can add groups while returning extracted predictions. These establish ownership and lifecycle precedents, not identical return contracts for every method.

Retain this repository's `xarray.DataTree` representation rather than introducing legacy `arviz.InferenceData` alongside it. Here, “idata” names that tree. PyMC-Marketing's current source also uses DataTree; PyMC's documented stable sampling API names `InferenceData`. Do not claim those Python types are interchangeable.

MFM draws are amortized conditional decomposition draws, not MCMC samples or noisy posterior-predictive observations. Preserve those distinctions despite familiar ArviZ group names.

## Lifecycle and overwrite semantics

`model.idata` is `None` on construction and after loading a weight bundle. It becomes non-`None` after the first successful `add_context` and stays populated through subsequent successful operations.

| Operation | Published data | Replaced or invalidated | Return |
| --- | --- | --- | --- |
| `add_context(history, ...)` | New context observations, covariates and role metadata | All earlier context, both posterior marginals, query data and response curves | `self`, preserving chaining |
| `inference(plan, ...)` | Current history plus this plan's covariates, available query observations, and both posterior marginals | Entire previous query subtree, context posterior and its inference metadata, response curves | New `model.idata` |
| `response_curves(...)` | Current inference plus the completed sweep | Entire previous response-curve subtree, including omitted channels/levels | Existing `ResponseCurves` result; new tree available through `model.idata` |
| `idata`, `input_usage`, charts | No new model state | Nothing | Existing property/report/figure contract |
| Any failed operation | No new model state | Nothing | Raise the original error |

Every inference rerun regenerates **both** marginals, even if only future inputs, horizon, draw count, graph mode, units or provenance changed. Never reuse context draws solely because history is unchanged. Query draws and context draws are separate marginals; equal draw indices do not pair them across time windows.

A shorter plan replaces a longer plan: no leftover dates, coordinates, observations or variables. A curve rerun with fewer channels or levels removes old ones. Repeating identical inputs still performs fresh sampling and overwrites; it is not an inference cache. A new context removes query and curve groups entirely rather than publishing empty placeholders.

### Object identity and caller-held results

“Inference adds to idata” describes the logical dataset, not permanent Python object identity. Each successful state-changing operation assigns a newly assembled root to `_idata`. Repeated property reads return the same current root. `model.inference(...) is model.idata` holds immediately after the call; an earlier local reference remains the earlier snapshot after another call.

Copy tree structure and mutable attribute mappings, but reuse unchanged, owned, read-only numeric buffers where safe. Never deep-copy the entire history and posterior on every curve update. Conversely, do not reparent live DataTree nodes or share mutable tree structure with older snapshots. Numeric conversion copies required for named active slots or safe dtypes are acceptable; redundant draw copies are not.

The public tree is read-only by convention, not a custom immutable wrapper. Users must copy before editing it; manual mutation is not a supported way to change context, inference or curves. Model updates must not mutate older published trees. Validate buffer ownership/read-only behavior against the actual xarray APIs during implementation rather than assuming a shallow tree copy isolates everything.

## Tree schema

Preserve existing context/query posterior paths, names, active-slot filtering and dimensions. Extend their data groups instead of flattening both windows into a global posterior. No date, draw, channel, level or edge coordinates on the convenience root: each subtree owns its axes, so differently shaped runs cannot align accidentally.

| Path | Contents and dimensions | Available after |
| --- | --- | --- |
| `/context/observed_data` | `outcome(date)` with sorted, original-unit history outcomes | `add_context` |
| `/context/constant_data` | `media(date, channel)`, `controls(date, control)` for supplied, unpadded roles; existing component/edge descriptors added when available | `add_context`; descriptors completed by inference |
| `/context/posterior` | Existing named decomposition variables and `graph` | `inference` |
| `/query/constant_data` | This plan's media/control covariates, with the same named-role convention | `inference` |
| `/query/observed_data` | Optional `outcome(date)`; finite supplied values, NaN for unobserved dates | `inference`, only when any observation is available |
| `/query/posterior` | Existing named decomposition variables and `graph` | `inference` |
| `/response_curves/posterior` | `contribution(chain, draw, channel, level)` from the completed sweep | `response_curves` |
| `/response_curves/constant_data` | `media(channel, level)` totals at each multiplier and the actual sweep dates on a local date axis | `response_curves` |

Both window posterior groups retain `chain=1`; contributions have `chain, draw, date` plus `channel` or `control` where appropriate, and graph has `chain, draw, edge`. Padded controls and inactive graph slots remain absent. Preserve zero-edge behavior: do not invent a graph variable where the existing converter omits it. Covariates are original-unit values, not model-normalized inputs.

Retain configured role names/order and the target/date column names as serializable metadata, while observed values use the stable variable name `outcome`. Treat role labels as coordinate values, never group paths or parsed identifiers. Detect labels that cannot be represented by the supported tree/storage contract during context validation, before any sampling.

Query outcomes follow PR #115's `query_outcome` sanitization. They remain evaluation/display-only: they never enter encoder inputs, scaling, graph selection or inference input identity. Omit `/query/observed_data` entirely when there are no finite query observations. Changing only those observations must not change seeded decomposition draws.

Keep units and the existing selected scalar identity attributes on each inference window. Additionally, store the complete canonical result provenance as a documented JSON string in each window's `attrs["provenance"]`, including authoritative `input_usage`, inference configuration and nested caller annotations. The replacement for `query.provenance["input_usage"]` is the `input_usage` entry obtained by JSON-decoding `model.idata["query"].attrs["provenance"]`; `model.query_result.provenance` also remains available. Reuse the existing JSON validation/canonicalization rules, reject unsupported or nonfinite values before sampling, and never stringify arbitrary objects as a fallback. This changes the model-tree schema, not generic standalone result exports.

Add a versioned model-tree schema attribute and model-local context/prediction revision numbers, committed only on success. Curve metadata records source revisions, actual sweep sampling settings and units; levels/dates live in coordinates/data rather than nonportable nested attributes. Store base-prediction provenance separately as JSON `source_provenance`, not as a claim that every truncated/scaled scenario has the original full-plan input identity or usage. Revisions describe local dependency lineage, not globally unique identities or replacements for existing data/model digests. Do not claim inference provenance exists on the data-only tree before sampling.

Preserve the existing warnings about independent amortized draws, inapplicable MCMC convergence diagnostics, and separate context/query marginals. Add a curve-specific note: level draws are not paired; the shared unchanged-input reference can be reused across channels/zero-input levels. Curve distributions describe conditional contributions, not causal lift. Do not label decomposition draws as `posterior_predictive`, add observation noise, or invent likelihood/sample-stat groups.

ArviZ summaries target a window subtree such as `model.idata['query']`, not the convenience root. The response-curve subtree has its own sample axes and may have a different draw count from the base prediction.

## Sampling and publication boundary

### `add_context`

Validate/snapshot the dataframe using `_FittedContext`; construct the context-only data groups from that candidate. Publish the context and tree together, then discard all previous prediction-derived state. It performs no encoder/router/decoder work and does not touch the random generator. A tree-construction failure is a failed context update, not a half-installed context.

### `inference`

1. Require context, validate the plan/options/exportable labels/provenance and capture display-only query observations.
2. Compute the query marginal first, preserving the existing seeded query-sampling order. Compute the context marginal once from the same actual plan and canonical sampling settings. For fixed mode, replay the exact per-draw query graphs as the current `_context_result` does; do not independently select replacement graphs.
3. Construct candidate snapshots, both converted posterior groups and the complete current tree without mutating published fields.
4. Only after all work succeeds, replace prediction/context-result/idata state together and remove old curves.

Eagerly computing context is intentional: inference completion means both marginals are ready, and plotting cannot surprise users with sampling or errors. It moves the existing first context-chart/export cost into every dataframe inference call, including query-only workflows. It also retains both marginals and their tree representation in memory. Measure that cost; do not promise the present query-only latency or memory footprint. Query-only/high-throughput consumers can use the existing stateless `decompose` API, not a second dataframe return/storage mode.

A sampling/conversion failure leaves the last successful state intact. Random numbers already consumed by failed sampling are not rolled back, and retries may redraw candidate results. Input/metadata validation failures must occur before sampling. Successful query draws from a failed unpublished transaction are not retained in a hidden partial cache. Model training/eval mode restoration and unchanged weights/gradients remain existing requirements.

### Response curves and charts

Build the entire curve result and its subtree off to the side, then publish both `_response_curves` and the new idata root together. Internal scenario evaluations continue to call stateless `decompose`, never public `MFM.inference`: a sweep must not overwrite the base prediction or recursively compute/publish context marginals for each level. An exception at any level or during tree conversion leaves the previous curves and base inference untouched.

Remove sampling from `_context_result`; make it a checked reader of completed inference state or remove it if callers no longer need it. Charts reuse the exact published computational draws/curve buffers, not independently regenerated results. `plot.inputs` remains useful after `add_context`; charts that require posterior draws still raise a clear “call inference first” error. Chart validation/import failures do not change idata or random state.

### Internal ownership

Keep the validated dataframe context and `_PredictionSnapshot` for model inputs, sampling configuration and provenance replay. `query_result` exposes the snapshot's existing result; context results remain private computational state. Those objects and idata derive from the same completed draws. The tree is the primary public analysis representation, not input to the neural model. Do not introduce a second sampler path, a generic results registry, an event system, or synchronized histories of exports. Share existing conversion helpers where their contracts fit; leave generic single-result and evaluation-archive exports unchanged.

The model remains a sequential mutable workflow, not a concurrent session manager. Concurrent mutating calls on one instance are unsupported. Do not add locks, background tasks or a new cache invalidation framework.

## Dependencies and persistence

Automatic idata must work with the normal local-model install, not only when an optional export extra happens to be installed. Add explicit `xarray>=2024.10` and `arviz-base>=1.3,<2` dependencies to `local` and update the lockfile. The current lock already includes ArviZ Base 1.3.0. Do not add PyMC, PyMC-Marketing, plotting backends or ArviZ statistics to the local inference requirement just to construct trees. Keep base HTTP-client imports lightweight; `pymc-mfm[arviz]` can continue to supply optional statistics capabilities, with its description updated accordingly.

Keep dependency gates honest: core context/inference lifecycle, curve conversion, rollback and bundle privacy tests must run with `local` plus the test runner, without `arviz_stats`, matplotlib or plotly. The current `needs_arviz` marker also requires statistics, and `test_response_curves.py` currently skips the whole file without matplotlib; separate computational acceptance from statistics/rendering gates. Split the combined lazy-import test in `test_datatree.py`: the standalone array API and saved-result operations retain their lazy tree-import boundary, while dataframe operations intentionally import the now-required tree packages.

Do not expand this PR into a persistence redesign. Existing xarray tree export is explicit analysis-data export; it may contain sensitive history, inputs, observed query outcomes, labels and inference results. `save_pretrained` and model Hub bundles remain strictly weights/configuration/provenance only and must never capture runtime idata, contexts, predictions or curves. Loading a weight bundle still yields `idata is None`. Reading a saved analysis tree does not restore a runnable model session.

File export still requires the existing `draws` extra: specify `engine="h5netcdf"` for DataTree netCDF roundtrips. The `local` dependency expansion guarantees in-memory idata, not a netCDF backend. Keep `model.query_result.save`/`DecompositionResult.load` as result-evidence persistence, distinct from xarray analysis-tree export; neither restores an attached model session.

## Migration and implementation sequence

This PR contains only this proposal. The following are implementation work units, not completed changes.

| Order | Files / surface | Work and acceptance |
| --- | --- | --- |
| 1 | `inference/_datatree.py`, `inference/dataframe.py` | Define context-only data groups, complete window assembly and curve conversion. Reuse named posterior conversion. Validate metadata early. Keep standalone result/archive exports stable. Prove sorted original-unit inputs, role ordering and isolated axes. |
| 2 | `pyproject.toml`, `uv.lock`, `model/model.py` | Make tree dependencies part of `local`; implement atomic automatic creation/publication; return and store the same tree from inference; compute both marginals; expose the already-retained `query_result`; remove `build_idata` and lazy sampling. Prove fresh data-only context and replacement semantics. |
| 3 | `inference/curves.py`, `model/model.py`, `plotting/charts/_data.py` | Attach complete curve subtrees transactionally; preserve stateless sweep calls and existing typed curve return. Make all charts sampling-free. Prove that plots and idata share numerical results. |
| 4 | `tests/conftest.py`, `tests/test_add_context_inference.py`, `tests/test_datatree.py`, `tests/test_model_plots.py`, `tests/test_response_curves.py`, affected chart tests | Migrate consumer contracts and cover the failure/overwrite cases below. Split required computational acceptance from optional stats/rendering gates and preserve standalone lazy-import/export tests. Remove tests that pin optional storage, lazy sampling or old return-type contracts instead of retaining obsolete behavior. |
| 5 | `README.md`, `RELEASE.md`, `docs/guide/decompose.qmd`, `docs/_quarto.yml`, `examples/river_1_quickstart.ipynb`, `tests/test_examples.py` | Migrate every dataframe consumer and API reference; replace explicit build/store examples with automatic property/return use. Explain the data-only initial state and eager-context cost. Re-execute synthetic/example material and review/pin changed notebook bytes under existing publication rules. |

Dependencies: schema first; model lifecycle next; curves/charts next; end-to-end consumer migration last. Converter/schema coverage and dependency metadata can be prepared independently, but one owner integrates the model/curve publication boundary. Before implementing exported-symbol changes, use reference navigation when available and inventory all repository callers, including tests, examples and documentation. No hidden compatibility shim for `build_idata` or the old dataframe inference return.

Migrate dataframe analysis to the named posterior tree. Existing result-specific work uses `model.query_result`: `components`, diagnostics, DAG/inspector helpers and `save` all operate on the exact query draw bank that populated idata. It is not an independently sampled approximation. In the quickstart, retain the returned tree for high-level analysis and bind a separate result reference from `model.query_result` immediately after inference; the graph catalog, ranked topologies, DOT/SVG exports and result-evidence persistence use that reference. The response-curve sweep must not replace it. Preserve configured names, masks, dates and padded-layout metadata through the existing result object rather than reconstructing them from active-only DataTree arrays.

The stateless `decompose` API remains for callers who intentionally choose independent, result-centric inference. It is not the notebook migration mechanism and must not be presented as a no-resampling reader of current idata. Name the dataframe return-type break and new result accessor prominently in release notes.

## Verification plan for implementation

Use the existing small synthetic model/dataframe fixtures. Test observable values, state transitions and numerical agreement, not implementation source text or incidental call wiring.

- **Initial context:** real sorted outcomes/covariates and names exist immediately, with no fabricated query or posterior. Caller dataframe/list mutations cannot change stored values. Context attachment, idata reads and `input_usage` leave RNG/weights/mode unchanged.
- **Completed inference:** returned tree is the stored tree; both marginals numerically match the canonical lower-level sampling sequence. Cover model-average, MAP and fixed per-draw graphs, supplied role ordering, padded controls, no active edges and nondefault units/provenance. Inference still restores model mode and does not change weights or gradients.
- **Replacement:** rerun with shorter dates, a different draw count/mode and changed inputs; old coordinates/observations/curve groups disappear, context draws match the new plan, and earlier caller-held trees remain identical. Repeat with changed role names/width and new history; no old query/posterior survives.
- **Identical reruns and metadata-only changes:** make consecutive identical calls without reseeding; capture each call's starting RNG state and compare to its canonical query-then-context sequence. Assert new root publication and prediction-revision advancement while old trees remain unchanged. Repeat with only provenance changes. This detects accidental caching deterministically; do not assert that stochastic arrays merely “differ.”
- **Curve attachment:** compare stored raw curve draws/totals/dates with the returned `ResponseCurves`; allow a different sweep sample count. A second sweep with fewer levels/channels replaces rather than merges. Base window arrays remain unchanged. Fixed-graph restrictions, shorter-plan semantics and zero-input reference reuse remain correct.
- **Query observations:** partial/missing/invalid target values preserve PR #115 sanitization, dates and missingness. Changing only query outcomes cannot change seeded posterior values, input scales or input identity. A later plan without observations removes the old observation group.
- **Atomic failures:** inject context conversion failure; invalid inference input/provenance; failure during context sampling after successful query sampling; posterior conversion failure; failure midway through a sweep; curve conversion failure. In each case assert the entire previous public tree, private context/prediction/curve state and revisions remain current. Do not assert RNG rollback after sampling.
- **Readers and aliasing:** repeated property reads and context/query/curve charts leave RNG and trees unchanged; figure data agrees with stored posterior/curve numbers. Hold an old tree across every successful operation and assert it is not rewritten. Check that candidate node attachment/attribute updates cannot leak into prior trees.
- **Exact-result consumers:** `query_result` is the original query draw bank used for the tree, remains unchanged by curves/readers and becomes unavailable on context replacement. Extend the quickstart acceptance beyond finite shapes: derive complete-graph counts/ranks/frequencies from that bank, verify selected edge endpoints and compare DOT/SVG topology to the selected graph. Preserve every existing graph analysis/export and verify result save/load independently of analysis-tree export.
- **Packaging/privacy:** an isolated `local`-only environment, without statistics/plotting packages, exercises core lifecycle/curves/rollback/privacy acceptance; base client imports remain lightweight. Extend the existing bundle test, which does not currently populate curves or compare the manifest: build a synthetic sentinel-bearing context/query/curve session, compare all bundle artifacts (configuration, manifest and weights) against a runtime-free baseline, and confirm no sentinel/runtime data is captured. Loading must restore no context, prediction, curves, `query_result` or idata.
- **Interop:** summarize context/query subtrees when `arviz_stats` is installed. With the `draws` extra and explicit `h5netcdf` engine, export/reload a context-only tree and a populated tree bearing partial query observations, nested caller annotations, authoritative input usage and curves with a different draw count. Compare dimensions, dates, JSON-decoded provenance, missingness, attributes and values. No automatic file writes or model-session rehydration.

Run the narrow lifecycle/DataTree/curve tests in the local-only environment after integration; run rendering/statistics checks separately with their extras. Run the migrated quickstart using its declared Python extras and a Graphviz-equipped environment (`dot` is a system executable, not installed by the `examples` extra), with the existing small synthetic stand-in model so no Hub access is required. Inspect at least one context/query chart and one curve chart in the supported backends, since the sampling boundary and data ownership changed. Measure query-only-before versus complete-inference-after latency and retained/peak memory on the same synthetic workload; report the tradeoff without a fabricated performance target. No live model, cloud account, private data or token is required for acceptance.

## Alternatives considered

- **Keep `build_idata(store=True)` or make `idata` lazy:** still requires a second action or hides sampling/conversion in a read; fails the requested lifecycle.
- **Create an empty tree at `add_context`:** satisfies object existence but provides no usable history; populate actual observations and covariates instead.
- **Sample context at `add_context`:** the existing inference contract requires a real future plan. A fabricated one changes the question being answered.
- **Keep the current dataframe result return and only set the property:** less migration, but misses the requested PyMC-style primary outcome. Recommend a clean idata return, with the already-existing lower-level API for result-centric callers.
- **Compute query eagerly but context on the first chart/property:** faster query-only inference, but retains the hidden sampling/error boundary. Prefer explicit eager completion.
- **Mutate one tree forever:** easier stale-reference surprises and partial updates on failure. Prefer replacement snapshots with reusable numerical buffers.
- **Flatten context and query or merge on rerun:** risks implying paired draws and retaining stale labels/dates. Keep independent subtrees and whole-output replacement.

The principal review choices are the deliberate inference return-type break and eager-context cost. The recommendation above accepts both to make the high-level lifecycle predictable; it does not alter the underlying probabilistic model.
