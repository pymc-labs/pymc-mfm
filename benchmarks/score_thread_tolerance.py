"""Replay-score a checkpoint with a different torch thread count and report the drift.

Trains a short run on a synthetic corpus with ``--train-threads`` threads (default 2), then
runs ``python -m pymc_mfm.training score --replay`` in a subprocess with ``OMP_NUM_THREADS``
and ``--threads`` set to ``--score-threads`` (default 1). Prints the number of compared
``holdout/*`` keys, how many are exact, and the largest absolute difference. Same-thread
replay is exact (``tests/test_score.py``); this measures the drift across thread counts.

    uv run --no-sync python benchmarks/score_thread_tolerance.py --out-dir /tmp/devloop-p4/threads
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-threads", type=int, default=2)
    parser.add_argument("--score-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-samples", type=int, default=16)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.train_threads)

    import torch
    from conftest import make_synthetic_corpus
    from pymc_generator import save_corpus

    from pymc_mfm.data.recipes import load_corpora
    from pymc_mfm.model.config import ModelConfig
    from pymc_mfm.training import TrackingConfig, TrainingConfig, train

    torch.set_num_threads(args.train_threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = args.out_dir / "corpus.npz"
    save_corpus(make_synthetic_corpus(n_cells=8, draws_per_cell=2, seed=9), corpus)
    config = ModelConfig(
        d_model=32, n_layers=2, n_heads=4, d_ff=64, flow_hidden=64, n_controls=2, n_demand=2
    )
    run_dir = args.out_dir / "run"
    train(
        [item.corpus for item in load_corpora([corpus])],
        out_dir=run_dir,
        model_config=config,
        training=TrainingConfig(
            steps=args.steps,
            batch_size=2,
            warmup_steps=2,
            eval_every=args.steps,
            eval_worlds=4,
            eval_samples=args.eval_samples,
            log_every=5,
            milestone_steps=(),
            prefetch_batches=2,
        ),
        device="cpu",
        tracking=TrackingConfig(enabled=False, profile="thread-tolerance"),
    )
    results = args.out_dir / "results.csv"
    results.unlink(missing_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(args.score_threads)}
    env["MKL_NUM_THREADS"] = str(args.score_threads)
    subprocess.run(
        [sys.executable, "-m", "pymc_mfm.training", "score", "--checkpoint",
         str(run_dir / "checkpoint_last.pt"), "--corpus", str(corpus), "--replay",
         "--threads", str(args.score_threads), "--results", str(results)],
        check=True, env=env,
    )  # fmt: skip
    with results.open(newline="") as handle:
        (row,) = csv.DictReader(handle)
    print(
        f"train threads {args.train_threads}, score threads {row['torch_threads']}: "
        f"compared {row['replay_keys_compared']}, exact {row['replay_keys_exact']}, "
        f"max abs diff {row['replay_max_abs_diff']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
