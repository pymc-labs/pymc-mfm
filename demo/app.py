"""Interactive River-1 synthetic decomposition demo."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from demo.comparison_world import CHANNELS, ComparisonWorld
from demo.comparison_world import N_CONTEXT as COMPARE_CONTEXT
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


@app.get("/stress-test")
def comparison_page():
    return FileResponse(Path(__file__).parent / "static/compare.html")


def comparison_result():
    """Run River live for the exact synthetic dataset used by the frozen MMM fits."""
    fixture = json.loads((Path(__file__).parent / "fixtures/comparison.json").read_text())
    world = ComparisonWorld(
        **{key: np.asarray(value, dtype=np.float64) for key, value in fixture["world"].items()}
    )
    data = np.column_stack((world.spend, world.promotion, world.outcome)).astype("<f8")
    if hashlib.sha256(data.tobytes()).hexdigest() != fixture["data_sha256"]:
        raise RuntimeError("Comparison fixture does not match the synthetic world")
    horizon = COMPARE_CONTEXT + 1
    controls = np.zeros((horizon, 10))
    controls[:, 0] = world.promotion[:horizon]
    observations = {
        "spend": world.spend[:horizon],
        "controls": controls,
        "context_outcome": world.outcome[:COMPARE_CONTEXT],
        "control_mask": [1] + [0] * 9,
    }
    with lock:
        torch.manual_seed(0)
        started = time.perf_counter()
        result = decompose(
            model(), observations, n_samples=128, window="context", channel_names=CHANNELS
        )
        elapsed = time.perf_counter() - started
    contributions = {
        channel: band(
            result.components[:, :, result.component_names.index(f"channel:{channel}")].sum(axis=1)
        )
        for channel in CHANNELS
    }
    context = slice(0, COMPARE_CONTEXT)
    return {
        "weeks": COMPARE_CONTEXT,
        "truth": {
            "sales": world.outcome[context].tolist(),
            "promotion": world.promotion[context].tolist(),
            "spend": {
                channel: world.spend[context, j].tolist() for j, channel in enumerate(CHANNELS)
            },
            "channels": {
                channel: float(world.channel_truth[context, j].sum())
                for j, channel in enumerate(CHANNELS)
            },
            "promotion_total": float(world.promotion_truth[context].sum()),
        },
        "river": {"channels": contributions, "n_samples": 128, "run_seconds": elapsed},
        "mmm": fixture,
    }


@app.get("/api/comparison")
def comparison():
    return comparison_result()


@app.get("/showcase")
def showcase_page():
    return FileResponse(Path(__file__).parent / "static/showcase.html")


@app.get("/api/showcase")
def showcase(case: Literal["91", "92"] = "91"):
    """Run River-1 on one explicitly selected whitepaper world, one week ahead."""
    folder = Path(__file__).parent / "fixtures/paper-worlds"
    meta = json.loads((folder / "sources.json").read_text())
    path = folder / f"world-{case}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["cases"][case]["fixture_sha256"]:
        raise RuntimeError("Whitepaper example fixture does not match its source record")
    with np.load(path, allow_pickle=False) as world:
        channel_mask = world["treatment_active_mask"][0].astype(bool)
        spend = world["treatment_raw"][0][:, channel_mask].copy()
        controls = world["covariates"][0].copy()
        sales = world["outcome_raw"][0].copy()
        control_mask = world["covariate_active_mask"][0].copy()
        latent_mask = world["latent_active_mask"][0].copy()
        truth = world["treatment_contribution_raw"][0, -1, channel_mask].copy()
    names = [f"Channel {chr(65 + i)}" for i in range(len(truth))]
    observations = {
        "spend": spend,
        "controls": controls,
        "context_outcome": sales[:207],
        "control_mask": control_mask,
        "latent_mask": latent_mask,
    }
    with lock:
        torch.manual_seed(0)
        started = time.perf_counter()
        result = decompose(
            model(), observations, n_samples=256, window="query", channel_names=names
        )
        seconds = time.perf_counter() - started
    channels = []
    for i, name in enumerate(names):
        draws = result.components[:, 0, result.component_names.index(f"channel:{name}")]
        quantiles = np.quantile(draws, [0.05, 0.95])
        channels.append(
            {
                "name": name,
                "mean": float(np.mean(draws)),
                "lo": float(quantiles[0]),
                "hi": float(quantiles[1]),
                "truth": float(truth[i]),
                "spend": spend[:, i].tolist(),
            }
        )
    return {
        "case": case,
        "context_weeks": 207,
        "query_week": 208,
        "run_seconds": seconds,
        "n_samples": 256,
        "channels": channels,
        "history_sales": sales[:207].tolist(),
        "provenance": {
            "source_sha256": meta["cases"][case]["source_sha256"],
            "model_revision": PIN,
        },
    }


@app.get("/api/scenario")
def scenario_forecast(
    case: Literal[
        "baseline", "promotion", "halo", "baseline_trend", "promotion_trend", "halo_trend"
    ] = "baseline",
):
    """Estimate direct channel contributions in the observed context."""
    from demo.scenario_world import CHANNELS as SCENARIO_CHANNELS
    from demo.scenario_world import N_CONTEXT as SCENARIO_CONTEXT
    from demo.scenario_world import make_scenario

    world = make_scenario(case)
    controls = np.zeros((len(world.outcome), 10))
    controls[:, 0] = world.controls[:, 0]
    observations = {
        "spend": world.spend[: SCENARIO_CONTEXT + 1],
        "controls": controls[: SCENARIO_CONTEXT + 1],
        "context_outcome": world.outcome[:SCENARIO_CONTEXT],
        "control_mask": [1] + [0] * 9,
    }
    with lock:
        torch.manual_seed(0)
        started = time.perf_counter()
        result = decompose(
            model(), observations, n_samples=128, window="context", channel_names=SCENARIO_CHANNELS
        )
        elapsed = time.perf_counter() - started
    from demo.scoring import fair_crps

    channel_draws = np.stack(
        [
            result.components[:, :, result.component_names.index(f"channel:{name}")].mean(axis=1)
            for name in SCENARIO_CHANNELS
        ],
        axis=1,
    )
    truth = world.contributions[:SCENARIO_CONTEXT].mean(axis=0)
    return {
        "case": case,
        "channels": SCENARIO_CHANNELS,
        "context_weeks": SCENARIO_CONTEXT,
        "spend": world.spend[:SCENARIO_CONTEXT].tolist(),
        "controls": world.controls[:SCENARIO_CONTEXT].tolist(),
        "outcome": world.outcome[:SCENARIO_CONTEXT].tolist(),
        "contribution_truth": truth.tolist(),
        "river": {
            "contributions": band(channel_draws),
            "contribution_crps": fair_crps(channel_draws, truth),
            "draws": int(channel_draws.shape[0]),
            "seconds": elapsed,
        },
    }


@app.get("/compare")
def scenario_page():
    return FileResponse(Path(__file__).parent / "static/scenarios.html")


@app.get("/tymc-worker.js")
def tymc_worker():
    return FileResponse(
        Path(__file__).parent / "static/tymc-worker.js", media_type="application/javascript"
    )
