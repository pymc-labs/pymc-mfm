# Migrating from earlier versions

Each deprecated name keeps working as a call and warns with a `DeprecationWarning` until it is removed. `ModelPlots` is now a function: calls to `ModelPlots(model)` keep working and return the live accessor of `model.idata`, but `isinstance` checks and subclassing do not.

| Old | New |
| --- | --- |
| `pymc_mfm.plotting.ModelPlots(model)`, `pymc_mfm.plotting.charts.ModelPlots(model)` | `model.plot`, or `pymc_mfm.Plots(model.idata)` for a snapshot of one run. `ModelPlots` is now a function, so `isinstance(x, ModelPlots)` no longer works; test for `pymc_mfm.Plots`. |
| `pymc_mfm.plotting.charts.BACKENDS` | No replacement. The values are `"matplotlib"` and `"plotly"`. |
| `pymc_mfm.ModelConfig` | `pymc_mfm.model.ModelConfig` |
| `pymc_mfm.DecompositionResult` | `pymc_mfm.inference.DecompositionResult` |
| `pymc_mfm.decompose` | `pymc_mfm.inference.decompose` |
| `pymc_mfm.observation_batch` | `pymc_mfm.inference.observation_batch` |
| `pymc_mfm.save_pretrained` | `pymc_mfm.inference.save_pretrained` |
| `pymc_mfm.push_to_hub` | `pymc_mfm.inference.push_to_hub` |

## What else changes

- `from pymc_mfm import *` no longer brings the six research names above. Import them from their subpackages.
- Charts read the model tree (`model.idata`), not private model state. The same charts draw from a tree saved to a file and opened again with xarray.
- `model.plot` is live: it follows the latest `inference`. `Plots(idata)` draws the tree it was given and does not change.
- Charts read the model tree instead of the model's stored results. Figures match earlier versions.
