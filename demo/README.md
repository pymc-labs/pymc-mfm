# River-1 synthetic simulator

This prototype adapts the original PyMC-PFN simulator to River-1. It generates a
synthetic world based on the River-1 quickstart and asks the pinned River-1 model
to decompose its held-out query period. True responses come from the simulator,
not the model. Contribution / spend ratios are descriptive, not incremental ROAS.
The latent-sales interval excludes observation noise. This is a research demo,
not a validated real-data application. Sliders adjust the generating world; they
do not change the model weights. It is not a benchmark or an accuracy guarantee.

Run locally with Python 3.13, the `demo` dependency group and the `hub` extra:

```sh
uv sync --locked --group demo --extra hub
uv run --locked --group demo --extra hub uvicorn demo.app:app --port 8743
```

The default loader uses the private `pymc-labs/mfm` Hugging Face repository,
profile `river-1`, revision `970e6095f14824d7e25ecace7e20a47bebde35b8`.
Authenticate through the normal Hugging Face mechanism. Alternatively,
`RIVER_MODEL_PATH` may point to an already downloaded three-file model folder.
Do not commit model weights or Hugging Face credentials.

`demo/Dockerfile` expects a **private** build context holding `pyproject.toml`,
`README.md`, `src/`, `demo/` and `model/` (config, manifest, safetensors).
Thomas requested a public demonstration. Keep replicas scaled to zero when idle,
bound concurrency and instances, and do not expose model bundle files or secrets
through HTTP. The prototype has not been independently validated for production.

## Model-check comparison

`/compare` shows two real, precomputed PyMC-Marketing 0.11.1 MMM fits on a fixed
synthetic dataset, one including promotion as a control and one omitting it.
Search spend is deliberately correlated with promotion in this single example.
Both fits used four NUTS chains with 500 warmup and 500 posterior draws per
chain, target acceptance 0.92, and passed the recorded divergence, R-hat and
bulk-ESS checks. See `fixtures/comparison.json` for diagnostics, version
information, fitted-sales summaries, the exact dataset and its SHA-256.

On each page load River-1 decomposes the **same 166 observed weeks** using that
saved dataset and the promotion control. Its context inference API additionally
requires one query week of spend and promotion, without that week's sales.
Intervals are 10th–90th percentiles of posterior total contributions. The
classic model's line and RMSE are in-sample, not forecast performance. Model
agreement does not establish causality; this is a deliberately constructed
example, not a real-world benchmark or a general performance claim.

`fit_comparison.py` contains the dataset-generation and fitting code. To
reproduce, use a separate Python 3.10 environment with PyMC 5.25.1,
PyMC-Marketing 0.11.1, NumPy 2.2.6 and ArviZ 0.23.4, then run from the
repository root with `python -m demo.fit_comparison`. The script locally filters
PyTensor's obsolete `-ld64` flag on newer macOS; it does not alter system files.
Sampler outputs can vary across platforms; the committed fixture preserves the
exact dataset and draws' summaries used by the demo.
