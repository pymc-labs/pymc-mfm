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
