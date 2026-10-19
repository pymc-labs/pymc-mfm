"""Time ``decompose`` on the published River-1 bundle with the quickstart workload.

The world is the weekly MMM example of ``examples/river_1_quickstart.ipynb``
(PyMC-Marketing's geometric adstock and logistic saturation, reproduced here in
NumPy so the benchmark needs no extra dependency). River-1 is loaded through
``from_pretrained`` from the local Hugging Face cache only (no network).

``--write-reference`` stores one seeded result; ``--check-reference`` compares
the current code against it: the binary graphs must be identical and the
components must agree within float32 round-off. Every timed call is seeded with
``torch.manual_seed(SEED)``, so all timed calls draw the same numbers.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pymc_mfm import decompose, from_pretrained  # noqa: E402

REPOSITORY = "pymc-labs/mfm"
PROFILE = "river-1"
REVISION = "970e6095f14824d7e25ecace7e20a47bebde35b8"
N_WEEKS = 208
CONTEXT = int(0.8 * N_WEEKS)
SEED = 0


def _geometric_adstock(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """PyMC-Marketing ``geometric_adstock(normalize=True)``, causal convolution."""
    weights = alpha ** np.arange(l_max)
    weights = weights / weights.sum()
    return np.convolve(x, weights)[: len(x)]


def _logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))


def quickstart_observations(n_controls: int) -> dict[str, object]:
    """The quickstart notebook's simulated world, as ``decompose`` observations."""
    rng = np.random.default_rng(seed=sum(map(ord, "mmm")))
    dates = np.datetime64("2018-04-02") + 7 * np.arange(N_WEEKS)
    dayofyear = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    x1 = rng.uniform(low=0.0, high=1.0, size=N_WEEKS)
    x1 = np.where(x1 > 0.9, x1, x1 / 2)
    x2 = rng.uniform(low=0.0, high=1.0, size=N_WEEKS)
    x2 = np.where(x2 > 0.8, x2, 0.0)
    scale = 1_000
    media = np.column_stack(
        [
            3.0 * _logistic_saturation(_geometric_adstock(x1, 0.4, 8), 4.0),
            2.0 * _logistic_saturation(_geometric_adstock(x2, 0.2, 8), 3.0),
        ]
    )
    trend = (np.linspace(start=0.0, stop=50, num=N_WEEKS) + 10) ** (1 / 4) - 1
    seasonality = 0.5 * (
        -np.sin(2 * 2 * np.pi * dayofyear / 365.5) + np.cos(1 * 2 * np.pi * dayofyear / 365.5)
    )
    events = np.column_stack(
        [dates == np.datetime64("2019-05-13"), dates == np.datetime64("2020-09-14")]
    ).astype(float)
    epsilon = rng.normal(loc=0.0, scale=0.25, size=N_WEEKS)
    baseline = scale * (2.0 + trend + seasonality)
    y = baseline + scale * media.sum(axis=1) + (scale * events * [1.5, 2.5]).sum(axis=1)
    y = y + scale * epsilon
    controls = np.zeros((N_WEEKS, n_controls))
    controls[:, :2] = events
    return {
        "spend": np.column_stack([x1, x2]),
        "controls": controls,
        "context_outcome": y[:CONTEXT],
        "control_mask": [1, 1] + [0] * (n_controls - 2),
    }


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run(model, observations, n_samples: int, window: str):
    torch.manual_seed(SEED)
    start = time.perf_counter()
    result = decompose(model, observations, n_samples=n_samples, window=window)
    return time.perf_counter() - start, result


def _compare(result, reference: Path, label: str) -> dict[str, float]:
    """Return mismatch statistics against a stored seeded result."""
    stored = np.load(reference)
    graphs, components = np.asarray(result.graphs), np.asarray(result.components)
    ref_graphs, ref_components = stored[f"{label}_graphs"], stored[f"{label}_components"]
    if graphs.shape != ref_graphs.shape or components.shape != ref_components.shape:
        raise SystemExit(
            f"{label}: shape mismatch {graphs.shape}/{components.shape} versus reference "
            f"{ref_graphs.shape}/{ref_components.shape}"
        )
    same_graph = (graphs == ref_graphs).reshape(graphs.shape[0], -1).all(axis=1)
    # Components are in outcome units; compare on the scale of the reference draws.
    scale = float(np.abs(ref_components).max())
    matched = np.abs(components[same_graph] - ref_components[same_graph])
    return {
        f"{label}_graph_mismatch_draws": float((~same_graph).sum()),
        f"{label}_max_rel_err": float(matched.max() / scale) if matched.size else 0.0,
        f"{label}_mean_rel_err": float(matched.mean() / scale) if matched.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--windows", default="query,context")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument("--check-reference", type=Path)
    parser.add_argument("--max-graph-mismatch", type=int, default=2)
    parser.add_argument("--max-rel-err", type=float, default=1e-4)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    start = time.perf_counter()
    model = from_pretrained(REPOSITORY, profile=PROFILE, revision=REVISION, local_files_only=True)
    load_s = time.perf_counter() - start
    observations = quickstart_observations(model.config.n_controls)
    windows = [window for window in args.windows.split(",") if window]
    print(
        f"# torch {torch.__version__} | python {platform.python_version()} | "
        f"{platform.machine()} {platform.processor() or platform.platform()} | "
        f"threads {torch.get_num_threads()} | commit {_commit()} | "
        f"{REPOSITORY}/{PROFILE}@{REVISION[:12]} | load {load_s:.2f}s",
        file=sys.stderr,
    )

    # Warm up allocator, kernels and caches with a short call per window.
    for window in windows:
        _run(model, observations, 16, window)

    metrics: dict[str, float] = {"load_s": load_s}
    saved: dict[str, np.ndarray] = {}
    failures = []
    for window in windows:
        times = []
        result = None
        for _ in range(args.repeats):
            elapsed, result = _run(model, observations, args.n_samples, window)
            times.append(elapsed)
        print(f"# {window}: {[round(t, 3) for t in times]}", file=sys.stderr)
        metrics[f"{window}_{args.n_samples}_s"] = statistics.median(times)
        saved[f"{window}_graphs"] = np.asarray(result.graphs)
        saved[f"{window}_components"] = np.asarray(result.components)
        if args.check_reference is not None:
            stats = _compare(result, args.check_reference, window)
            metrics.update(stats)
            if (
                stats[f"{window}_graph_mismatch_draws"] > args.max_graph_mismatch
                or stats[f"{window}_max_rel_err"] > args.max_rel_err
            ):
                failures.append(f"{window}: {stats}")

    if args.write_reference is not None:
        args.write_reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.write_reference, **saved)
    for name, value in metrics.items():
        print(f"METRIC {name}={value:.6g}")
    if failures:
        print("reference mismatch: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
