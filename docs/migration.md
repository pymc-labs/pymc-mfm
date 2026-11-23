# Migrating from earlier versions

Each deprecated name keeps working as a call and warns with a `DeprecationWarning` until it is removed.

The six `pymc_mfm.<name>` rows in the table warn on first access in a process. Overlapping first accesses may each warn. A name that loaded stays silent after that, even if you enable warnings later. Python hides `DeprecationWarning` by default, except for code in `__main__`: run with `python -W default` or `pytest` to see it from a library. A failed load is not cached, so the next access warns and retries.

| Old | New |
| --- | --- |
| `pymc_mfm.plotting.ModelPlots(model)`, `pymc_mfm.plotting.charts.ModelPlots(model)` | `model.plot`, or `pymc_mfm.Plots(model.idata)` for a snapshot of one run. `ModelPlots(model)` still works and returns the live accessor of `model.idata`. It is now a function, so `isinstance(x, ModelPlots)` and subclassing no longer work; test for `pymc_mfm.Plots`. |
| `pymc_mfm.plotting.charts.BACKENDS` | No replacement. The values are `"matplotlib"` and `"plotly"`. |
| `pymc_mfm.ModelConfig` | `pymc_mfm.model.ModelConfig` |
| `pymc_mfm.DecompositionResult` | `pymc_mfm.inference.DecompositionResult` |
| `pymc_mfm.decompose` | `pymc_mfm.inference.decompose` |
| `pymc_mfm.observation_batch` | `pymc_mfm.inference.observation_batch` |
| `pymc_mfm.save_pretrained` | `pymc_mfm.inference.save_pretrained` |
| `pymc_mfm.push_to_hub` | `pymc_mfm.inference.push_to_hub` |
| `model.add_context(history, ...)` | Removed, with no warning. Pass the history to `model.inference(context, query, date_column=..., target_column=..., channel_columns=..., control_columns=...)`. |
| `model.build_idata()` | Removed, with no warning. `model.inference(...)` builds the tree and stores it as `model.idata`. It also returns the tree unless you pass `return_idata=False`. |
| `model.inference(query)` | Removed, with no warning. Use `model.inference(context, query, date_column=..., target_column=..., channel_columns=..., control_columns=...)`. It returns the stored `DataTree`, or `None` with `return_idata=False`, not a `DecompositionResult`. Read the old return value from `model.query_result` after the call (example below). |

Code that used `.components` or `.diagnostics()` on the old return value reads the same query draws from `model.query_result`, without resampling:

```python
idata = model.inference(
    history,
    future,
    date_column="date",
    target_column="sales",
    channel_columns=["search", "social"],
    control_columns=["seasonality"],
)
query = model.query_result  # the DecompositionResult that inference(query) returned
query.components
query.diagnostics()
```

## What else changes

- On Python 3.13 or newer, use the [standard install](../README.md#installation) without an extras suffix. It includes local inference and training, the hosted SDK with draws and plots, diagnostics, and Hugging Face Hub support. Existing supported `.npz` corpora work with the training CLI without `pymc-generator`; the repository's `generation` group is for producing new synthetic corpora.
- `from pymc_mfm import *` and `dir(pymc_mfm)` no longer list the six research names above. A script that relies on a star import for them now gets `NameError`, with no warning. `pymc_mfm.decompose` and `from pymc_mfm import decompose` still work and warn. Import the names from their subpackages. A star import loads `MFM` and its dependencies, unlike a plain `import pymc_mfm`.
- Charts read the model tree (`model.idata`), not private model state. Figures match earlier versions. The same charts draw from a tree saved to a file and opened again with xarray.
- `model.plot` is live: it follows the latest `inference`. `Plots(idata)` draws the tree it was given and does not change.
- Local sampling defaults to 1,000 draws instead of 2,000. To reproduce a `model.inference` run that used the previous default, pass `n_samples=2_000` with the original seed and the same execution settings. Response curves inherit the source inference's draw count by default. The hosted client's `draws` default is set by the service and is unchanged.
