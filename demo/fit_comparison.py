"""Rebuild the two genuine PyMC-Marketing fits for the fixed comparison world.

Run in a Python environment with pymc-marketing 0.11.1, PyMC 5.25.1 and ArviZ.
Only summary statistics are saved; the original synthetic generator is versioned.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pymc_marketing

# PyTensor 2.31 requests Apple's retired -ld64 compatibility linker on recent
# macOS. Filter only that flag for this process; use the system linker instead.
if sys.platform == "darwin":
    from pytensor.link.c.cmodule import GCC_compiler

    _compile_args = GCC_compiler.compile_args
    GCC_compiler.compile_args = staticmethod(
        lambda *args, **kwargs: [arg for arg in _compile_args(*args, **kwargs) if arg != "-ld64"]
    )

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from demo.comparison_world import CHANNELS, N_CONTEXT, SEED, make_world

OUT = Path(__file__).parent / "fixtures" / "comparison.json"


def summarise(x):
    q = np.quantile(x, [0.1, 0.9])
    return {"mean": float(np.mean(x)), "lo": float(q[0]), "hi": float(q[1])}


def main():
    world = make_world()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-06", periods=N_CONTEXT, freq="W-MON"),
            "TV": world.spend[:N_CONTEXT, 0],
            "Search": world.spend[:N_CONTEXT, 1],
            "promotion": world.promotion[:N_CONTEXT],
        }
    )
    data_hash = hashlib.sha256(
        np.column_stack((world.spend, world.promotion, world.outcome)).astype("<f8").tobytes()
    ).hexdigest()
    result = {
        "description": "Fixed synthetic world; Search spend is associated with promotion.",
        "seed": SEED,
        "numpy_version": np.__version__,
        "world": {
            key: getattr(world, key).tolist()
            for key in (
                "spend",
                "promotion",
                "outcome",
                "baseline",
                "channel_truth",
                "promotion_truth",
            )
        },
        "context_weeks": N_CONTEXT,
        "data_sha256": data_hash,
        "pymc_version": pm.__version__,
        "pymc_marketing_version": pymc_marketing.__version__,
        "mcmc": {"chains": 4, "draws_per_chain": 500, "tune_per_chain": 500, "target_accept": 0.92},
        "fits": {},
    }
    for include_promo in (True, False):
        name = "with_promotion" if include_promo else "without_promotion"
        mmm = MMM(
            date_column="date",
            channel_columns=list(CHANNELS),
            control_columns=["promotion"] if include_promo else None,
            adstock=GeometricAdstock(l_max=8, normalize=True),
            saturation=LogisticSaturation(),
            yearly_seasonality=1,
        )
        start = time.monotonic()
        idata = mmm.fit(
            frame,
            world.outcome[:N_CONTEXT],
            chains=4,
            cores=2,
            draws=500,
            tune=500,
            target_accept=0.92,
            random_seed=SEED,
            progressbar=False,
        )
        elapsed = time.monotonic() - start
        div = int(idata.sample_stats.diverging.sum())
        names = [
            n
            for n in idata.posterior.data_vars
            if n
            not in {
                "mu",
                "channel_contributions",
                "control_contributions",
                "fourier_contributions",
                "yearly_seasonality_contribution",
                "total_contributions",
            }
        ]
        rhats = az.rhat(idata, var_names=names).to_array().values
        ess = az.ess(idata, var_names=names, method="bulk").to_array().values
        max_rhat = float(np.nanmax(rhats))
        min_ess = float(np.nanmin(ess))
        if div or max_rhat > 1.01 or min_ess < 400:
            raise RuntimeError(
                f"{name}: sampler diagnostics failed: divergences={div}, "
                f"rhat={max_rhat}, ess={min_ess}"
            )
        contribution = mmm.compute_channel_contribution_original_scale().sum(dim="date")
        summaries = {ch: summarise(contribution.sel(channel=ch).values) for ch in CHANNELS}
        mu_scaled = idata.posterior["mu"].mean(dim=("chain", "draw")).values
        mu = mmm.get_target_transformer().inverse_transform(mu_scaled[:, None]).ravel()
        rmse = float(np.sqrt(np.mean((mu - world.outcome[:N_CONTEXT]) ** 2)))
        result["fits"][name] = {
            "channels": summaries,
            "predicted_sales": mu.tolist(),
            "rmse": rmse,
            "diagnostics": {"divergences": div, "max_rhat": max_rhat, "min_bulk_ess": min_ess},
            "fit_seconds": round(elapsed, 2),
        }
        print(name, summaries, "rmse", rmse, "rhat", max_rhat, "ess", min_ess, flush=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
