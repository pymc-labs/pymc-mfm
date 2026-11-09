"""Time ``decompose`` on the published River-1 bundle with the quickstart workload.

The world is the weekly MMM example of ``examples/river_1_quickstart.ipynb``
(PyMC-Marketing's geometric adstock and logistic saturation, reproduced here in
NumPy so the benchmark needs no extra dependency). River-1 is loaded through
``from_pretrained`` from the local Hugging Face cache only (no network).

By default the benchmark times the query and context windows of the 208-week
quickstart world. ``--query-weeks 10,50,150,205`` instead times the query window at
each forecast horizon: the same generator runs for ``CONTEXT + 205`` weeks, the
first ``CONTEXT`` (166) weeks are the context and the next ``N`` weeks the query, so
every level shares one history and differs only in its number of query weeks.

``--write-reference`` stores one seeded result per case together with the median
times; ``--check-reference`` compares the current code against it: a case fails if any
binary graph differs (``--max-graph-mismatch``, default 0) or, where the graphs agree, a
component differs by more than ``--max-rel-err`` (default 1e-4) of the largest reference
component. It also reports each case's speedup over the stored times. To compare two
versions, write the reference with this script while the other version's package is the
one imported, with the same ``--query-weeks`` or ``--windows``, for example
``PYTHONPATH=<main checkout>/src python benchmarks/inference_speed.py --query-weeks
10,50,150,205 --write-reference ref.npz`` (the header on stderr names the imported
package); ``main``'s own copy of this script has no ``--query-weeks``. Every timed call is
seeded with ``torch.manual_seed(SEED)``, so all timed calls draw the same numbers.
``--device`` moves the loaded model before sampling (for example ``mps``).
"""

from __future__ import annotations

import argparse
import math
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

import pymc_mfm  # noqa: E402
from pymc_mfm import decompose, from_pretrained  # noqa: E402

REPOSITORY = "pymc-labs/mfm"
PROFILE = "river-1"
REVISION = "970e6095f14824d7e25ecace7e20a47bebde35b8"
N_WEEKS = 208
CONTEXT = int(0.8 * N_WEEKS)
#: Longest query horizon of ``--query-weeks``; it fixes the simulated world's length.
MAX_QUERY_WEEKS = 205
SEED = 0


def _geometric_adstock(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """PyMC-Marketing ``geometric_adstock(normalize=True)``, causal convolution."""
    weights = alpha ** np.arange(l_max)
    weights = weights / weights.sum()
    return np.convolve(x, weights)[: len(x)]


def _logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - np.exp(-lam * x)) / (1.0 + np.exp(-lam * x))


def quickstart_observations(
    n_controls: int,
    *,
    n_weeks: int = N_WEEKS,
    context: int = CONTEXT,
    query_weeks: int | None = None,
) -> dict[str, object]:
    """The quickstart notebook's simulated world, as ``decompose`` observations.

    The world runs for ``n_weeks``; the first ``context`` rows are the context and the
    next ``query_weeks`` rows (default: all remaining weeks) the query.
    """
    rng = np.random.default_rng(seed=sum(map(ord, "mmm")))
    dates = np.datetime64("2018-04-02") + 7 * np.arange(n_weeks)
    dayofyear = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    x1 = rng.uniform(low=0.0, high=1.0, size=n_weeks)
    x1 = np.where(x1 > 0.9, x1, x1 / 2)
    x2 = rng.uniform(low=0.0, high=1.0, size=n_weeks)
    x2 = np.where(x2 > 0.8, x2, 0.0)
    scale = 1_000
    media = np.column_stack(
        [
            3.0 * _logistic_saturation(_geometric_adstock(x1, 0.4, 8), 4.0),
            2.0 * _logistic_saturation(_geometric_adstock(x2, 0.2, 8), 3.0),
        ]
    )
    trend = (np.linspace(start=0.0, stop=50, num=n_weeks) + 10) ** (1 / 4) - 1
    seasonality = 0.5 * (
        -np.sin(2 * 2 * np.pi * dayofyear / 365.5) + np.cos(1 * 2 * np.pi * dayofyear / 365.5)
    )
    events = np.column_stack(
        [dates == np.datetime64("2019-05-13"), dates == np.datetime64("2020-09-14")]
    ).astype(float)
    epsilon = rng.normal(loc=0.0, scale=0.25, size=n_weeks)
    baseline = scale * (2.0 + trend + seasonality)
    y = baseline + scale * media.sum(axis=1) + (scale * events * [1.5, 2.5]).sum(axis=1)
    y = y + scale * epsilon
    controls = np.zeros((n_weeks, n_controls))
    controls[:, :2] = events
    rows = n_weeks if query_weeks is None else context + query_weeks
    if not context < rows <= n_weeks:
        raise ValueError(f"{query_weeks} query weeks do not fit a {n_weeks}-week world")
    return {
        "spend": np.column_stack([x1, x2])[:rows],
        "controls": controls[:rows],
        "context_outcome": y[:context],
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _run(model, observations, n_samples: int, window: str):
    device = next(model.parameters()).device
    torch.manual_seed(SEED)
    _synchronize(device)
    start = time.perf_counter()
    result = decompose(model, observations, n_samples=n_samples, window=window)
    _synchronize(device)
    return time.perf_counter() - start, result


def _reference(path: Path, labels: list[str]):
    """The stored results of ``path``, which must hold a seeded result for every case."""
    stored = np.load(path)
    missing = [
        label
        for label in labels
        if not {f"{label}_graphs", f"{label}_components"} <= set(stored.files)
    ]
    if missing:
        held = sorted(
            name.removesuffix("_graphs") for name in stored.files if name.endswith("_graphs")
        )
        raise SystemExit(
            f"{path} holds no result for {', '.join(missing)} "
            f"(it holds {', '.join(held) or 'none'}); write the reference with this script and "
            "the same --query-weeks or --windows (see the module docstring)"
        )
    return stored


def _compare(result, stored, label: str) -> dict[str, float]:
    """Return mismatch statistics against a stored seeded result."""
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


def _cases(args, n_controls: int) -> list[tuple[str, dict[str, object], str]]:
    """``(label, observations, window)`` for every timed call."""
    if args.query_weeks:
        horizon = CONTEXT + MAX_QUERY_WEEKS
        return [
            (
                f"q{weeks}",
                quickstart_observations(n_controls, n_weeks=horizon, query_weeks=weeks),
                "query",
            )
            for weeks in (int(value) for value in args.query_weeks.split(",") if value)
        ]
    observations = quickstart_observations(n_controls)
    return [(window, observations, window) for window in args.windows.split(",") if window]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--windows", default="query,context")
    parser.add_argument(
        "--query-weeks",
        default="",
        help="comma-separated query horizons after the 166-week context, e.g. 10,50,150,205",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument("--check-reference", type=Path)
    parser.add_argument(
        "--max-graph-mismatch",
        type=int,
        default=0,
        help="draws per case whose graph may differ from the reference (default 0)",
    )
    parser.add_argument(
        "--max-rel-err",
        type=float,
        default=1e-4,
        help="largest component difference where the graphs agree, as a fraction of the "
        "largest reference component (default 1e-4)",
    )
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    start = time.perf_counter()
    model = from_pretrained(REPOSITORY, profile=PROFILE, revision=REVISION, local_files_only=True)
    model = model.to(args.device)
    load_s = time.perf_counter() - start
    cases = _cases(args, model.config.n_controls)
    stored = None
    if args.check_reference is not None:
        stored = _reference(args.check_reference, [label for label, _, _ in cases])
    print(
        f"# torch {torch.__version__} | python {platform.python_version()} | "
        f"{platform.machine()} {platform.processor() or platform.platform()} | "
        f"threads {torch.get_num_threads()} | device {args.device} | commit {_commit()} | "
        f"{REPOSITORY}/{PROFILE}@{REVISION[:12]} | load {load_s:.2f}s | "
        f"package {Path(pymc_mfm.__file__).parent}",
        file=sys.stderr,
    )

    # Warm up allocator, kernels and caches with a short call per case.
    for _, observations, window in cases:
        _run(model, observations, 16, window)

    metrics: dict[str, float] = {"load_s": load_s}
    saved: dict[str, np.ndarray] = {}
    failures = []
    times_s, speedups = [], []
    for label, observations, window in cases:
        times = []
        result = None
        for _ in range(args.repeats):
            elapsed, result = _run(model, observations, args.n_samples, window)
            times.append(elapsed)
        median = statistics.median(times)
        print(f"# {label}: {[round(t, 3) for t in times]}", file=sys.stderr)
        metrics[f"{label}_s" if args.query_weeks else f"{label}_{args.n_samples}_s"] = median
        times_s.append(median)
        saved[f"{label}_graphs"] = np.asarray(result.graphs)
        saved[f"{label}_components"] = np.asarray(result.components)
        saved[f"{label}_s"] = np.asarray(median)
        if stored is not None:
            stats = _compare(result, stored, label)
            metrics.update(stats)
            if f"{label}_s" in stored.files:
                speedups.append(float(stored[f"{label}_s"]) / median)
                metrics[f"{label}_speedup"] = speedups[-1]
            if (
                stats[f"{label}_graph_mismatch_draws"] > args.max_graph_mismatch
                or stats[f"{label}_max_rel_err"] > args.max_rel_err
            ):
                failures.append(f"{label}: {stats}")

    geomean = math.exp(statistics.fmean(math.log(t) for t in times_s))
    if args.query_weeks:
        metrics = {"geomean_s": geomean, **metrics}
    if speedups:
        metrics["min_speedup"] = min(speedups)
        metrics["max_speedup"] = max(speedups)
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
