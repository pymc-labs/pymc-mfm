# Migrating from earlier versions

Each deprecated name keeps working as a call and warns with a `DeprecationWarning` until it is removed.

The six `pymc_mfm.<name>` rows in the table warn on first access in a process. Overlapping first accesses may each warn. A name that loaded stays silent after that, even if you enable warnings later. Python hides `DeprecationWarning` by default, except for code in `__main__`: run with `python -W default` or `pytest` to see it from a library. A load that fails, for example without PyTorch, is not cached, so the next access warns again and raises the same `ImportError` with the `pymc-mfm[local]` hint.

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
| `model.inference(query)` | Removed, with no warning. Use `model.inference(context, query, date_column=..., target_column=..., channel_columns=..., control_columns=...)`. |

## What else changes

- `from pymc_mfm import *` and `dir(pymc_mfm)` no longer list the six research names above. A script that relies on a star import for them now gets `NameError`, with no warning. `pymc_mfm.decompose` and `from pymc_mfm import decompose` still work and warn. Import the names from their subpackages. A star import needs the `[local]` extra, because `__all__` includes `MFM`.
- Charts read the model tree (`model.idata`), not private model state. Figures match earlier versions. The same charts draw from a tree saved to a file and opened again with xarray.
- `model.plot` is live: it follows the latest `inference`. `Plots(idata)` draws the tree it was given and does not change.
