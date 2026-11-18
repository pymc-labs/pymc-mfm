# Migrating from earlier versions

Each deprecated name keeps working and warns with a `DeprecationWarning` until it is removed.

| Old | New |
| --- | --- |
| `pymc_mfm.plotting.ModelPlots(model)`, `pymc_mfm.plotting.charts.ModelPlots(model)` | `model.plot`, or `pymc_mfm.Plots(model.idata)` for a snapshot of one run. `ModelPlots` is now a function, so `isinstance(x, ModelPlots)` no longer works; test for `pymc_mfm.Plots`. |
| `pymc_mfm.plotting.charts.BACKENDS` | No replacement. The values are `"matplotlib"` and `"plotly"`. |

## What else changes

- Charts read the model tree (`model.idata`), not private model state. The same charts draw from a tree saved to a file and opened again with xarray.
- `model.plot` is live: it follows the latest `inference`. `Plots(idata)` draws the tree it was given and does not change.
- Chart numbers come from the tree's float64 values, not float32 draws, so figures differ from earlier versions by rounding only.
