# Migrating from earlier versions

Each deprecated name keeps working as a call and warns with a `DeprecationWarning` until it is removed.

The six `pymc_mktfm.<name>` rows in the table warn on first access in a process. Overlapping first accesses may each warn. A name that loaded stays silent after that, even if you enable warnings later. Python hides `DeprecationWarning` by default, except for code in `__main__`: run with `python -W default` or `pytest` to see it from a library. A load that fails, for example without PyTorch, is not cached, so the next access warns again and raises the same `ImportError` with the `pymc-mktfm[local]` hint.

| Old | New |
| --- | --- |
| `pymc_mktfm.plotting.ModelPlots(model)`, `pymc_mktfm.plotting.charts.ModelPlots(model)` | `model.plot`, or `pymc_mktfm.Plots(model.idata)` for a snapshot of one run. `ModelPlots(model)` still works and returns the live accessor of `model.idata`. It is now a function, so `isinstance(x, ModelPlots)` and subclassing no longer work; test for `pymc_mktfm.Plots`. |
| `pymc_mktfm.plotting.charts.BACKENDS` | No replacement. The values are `"matplotlib"` and `"plotly"`. |
| `pymc_mktfm.ModelConfig` | `pymc_mktfm.model.ModelConfig` |
| `pymc_mktfm.DecompositionResult` | `pymc_mktfm.inference.DecompositionResult` |
| `pymc_mktfm.decompose` | `pymc_mktfm.inference.decompose` |
| `pymc_mktfm.observation_batch` | `pymc_mktfm.inference.observation_batch` |
| `pymc_mktfm.save_pretrained` | `pymc_mktfm.inference.save_pretrained` |
| `pymc_mktfm.push_to_hub` | `pymc_mktfm.inference.push_to_hub` |

## What else changes

- `from pymc_mktfm import *` and `dir(pymc_mktfm)` no longer list the six research names above. A script that relies on a star import for them now gets `NameError`, with no warning. `pymc_mktfm.decompose` and `from pymc_mktfm import decompose` still work and warn. Import the names from their subpackages. A star import needs the `[local]` extra, because `__all__` includes `MFM`.
- Charts read the model tree (`model.idata`), not private model state. Figures match earlier versions. The same charts draw from a tree saved to a file and opened again with xarray.
- `model.plot` is live: it follows the latest `inference`. `Plots(idata)` draws the tree it was given and does not change.
