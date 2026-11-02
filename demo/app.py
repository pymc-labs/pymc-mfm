"""Interactive River-1 synthetic decomposition demo."""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from demo.simulator import N_CONTEXT, N_WEEKS, response_curves, simulate
from pymc_mfm import decompose, from_pretrained

PIN = "970e6095f14824d7e25ecace7e20a47bebde35b8"
app = FastAPI(title="River-1 Simulator")
lock = threading.Lock()


@lru_cache(maxsize=1)
def model():
    bundle = os.environ.get("RIVER_MODEL_PATH")
    if bundle:
        return from_pretrained(bundle, device="cpu")
    return from_pretrained("pymc-labs/mfm", profile="river-1", revision=PIN, device="cpu")


class Request(BaseModel):
    betas: list[float] = Field(default=[3.0, 2.0], min_length=2, max_length=2)
    trend_slope: float = Field(default=0, ge=-0.002, le=0.002)
    seasonality_amp: float = Field(default=0.5, ge=0, le=0.8)
    noise_std: float = Field(default=0.25, ge=0.02, le=0.35)
    seed: int = Field(default=0, ge=0, le=1000000)
    n_samples: int = Field(default=32, ge=16, le=128)
    n_channels: int = 2


def band(values):
    q = np.quantile(values, [0.1, 0.9], axis=0)
    return {"mean": values.mean(axis=0).tolist(), "lo": q[0].tolist(), "hi": q[1].tolist()}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/api/info")
def info():
    m = model()
    return {
        "model": "River-1",
        "revision": PIN,
        "n_params": sum(p.numel() for p in m.parameters()),
        "device": "cpu",
        "default_samples": 32,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/simulate")
def run(req: Request):
    if not all(0.5 <= x <= 4.0 for x in req.betas) or req.n_channels != 2:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422, detail="Two channel strengths between 0.5 and 4 required"
        )
    started = time.perf_counter()
    world = simulate(req.betas, req.trend_slope, req.seasonality_amp, req.noise_std, req.seed)
    sim_ms = (time.perf_counter() - started) * 1000
    observations = {
        "spend": world.spend,
        "controls": world.controls,
        "context_outcome": world.outcome[:N_CONTEXT],
        "control_mask": [1, 1] + [0] * 8,
    }
    with lock:
        m = model()
        torch.manual_seed(0)
        started = time.perf_counter()
        result = decompose(
            m, observations, n_samples=req.n_samples, window="query", channel_names=("TV", "Search")
        )
        elapsed = (time.perf_counter() - started) * 1000
    draws = result.components
    channel_draws = [
        draws[:, :, result.component_names.index(f"channel:{name}")] for name in ("TV", "Search")
    ]
    base_draws = draws[:, :, result.component_names.index("baseline")]
    q = slice(N_CONTEXT, N_WEEKS)
    spend_totals = world.spend[q].sum(axis=0)
    true_roas = world.channels[q].sum(axis=0) / spend_totals
    roas = [band(d.sum(axis=1) / spend_totals[i]) for i, d in enumerate(channel_draws)]
    return {
        "params": {"n_channels": 2},
        "sim_ms": sim_ms,
        "n_ctx": N_CONTEXT,
        "truth": {
            "revenue": world.outcome.tolist(),
            "baseline": world.baseline.tolist(),
            "contributions": [world.channels[q, i].tolist() for i in range(2)],
            "roas": true_roas.tolist(),
            "beta": world.betas.tolist(),
        },
        "fit": {
            "query_weeks": list(range(N_CONTEXT, N_WEEKS)),
            "latency_ms": elapsed,
            "n_samples": req.n_samples,
            "revenue": band(draws.sum(axis=2)),
            "baseline": band(base_draws),
            "contributions": [band(x) for x in channel_draws],
            "roas": roas,
        },
        "response_curves": response_curves(world),
    }
