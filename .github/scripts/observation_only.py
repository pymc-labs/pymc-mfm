"""Run an offline bundle on observations, retaining separate marginal windows.

Input JSON contains spend, controls, context_outcome, and optional channel_mask,
control_mask and latent_mask. No query outcomes, truth, corpus or service is read.
The output can be rendered later with the diagnostics extra, without resampling.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import torch

from pymc_mktfm import from_pretrained
from pymc_mktfm.inference import decompose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local MFM bundle directory")
    parser.add_argument("--observations", type=Path, required=True, help="Observation-only JSON")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    observations = json.loads(args.observations.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model = from_pretrained(args.model, device="cpu")
    summary = {"load_seconds": time.perf_counter() - started, "windows": {}}
    for window in ("context", "query"):
        # Each call is a separate marginal, not a paired full-horizon posterior.
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        result = decompose(model, observations, n_samples=args.samples, window=window)
        elapsed = time.perf_counter() - started
        result.save(args.out / window)
        summary["windows"][window] = {
            "seconds": elapsed,
            "shape": list(result.components.shape),
            "mode": result.mode,
            "units": result.units,
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary.update(
        checkpoint_sha256=model.provenance["checkpoint_sha256"],
        bundle_sha256=model.provenance["bundle_sha256"],
        peak_process_rss_mib=rss / (1024**2 if platform.system() == "Darwin" else 1024),
        platform=platform.platform(),
        torch_threads=torch.get_num_threads(),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
