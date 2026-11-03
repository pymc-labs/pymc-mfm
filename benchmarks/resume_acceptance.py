"""Resume acceptance: interrupted-then-resumed equals uninterrupted (bitwise on CPU).

Toy geometry (T=52, K=4, M=2, J=1; 16 cells x 4 worlds, 4 held out), the canonical
architecture, 200 updates of 4 worlds, evaluation every 100, prefetch on. The run is
interrupted right after ``checkpoint_last.pt`` of step 100 is written, then resumed
into a new directory. Steps 101..200 of every logged loss key, ``grad_norm`` and
``lr`` must be bitwise equal to the uninterrupted run, and so must the final weights
and the step-200 evaluation. Two negative controls resume with the stream restore, then
the RNG restore, switched off; each must diverge.

On CUDA the script turns on ``torch.use_deterministic_algorithms`` (with
``warn_only``), because default kernels are not bitwise reproducible and training
amplifies their differences. A second uninterrupted run measures the remaining noise
floor. The resume passes when its largest difference from the uninterrupted run is at
most ``max(2 x floor, 1e-5)`` and each negative control differs by more than ten times
that.

    OMP_NUM_THREADS=2 uv run --no-sync python benchmarks/resume_acceptance.py \
        [--device cpu|cuda] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
# Needed by deterministic cuBLAS; read when CUDA initializes, ignored on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import make_synthetic_corpus  # noqa: E402

from pymc_mfm.model.config import ModelConfig  # noqa: E402
from pymc_mfm.training import (  # noqa: E402
    TrackingConfig,
    TrainingConfig,
    TrainingObserver,
    load_checkpoint,
    train,
)
from pymc_mfm.training import trainer as trainer_module  # noqa: E402
from pymc_mfm.training.stream import CorpusStream  # noqa: E402
from pymc_mfm.training.trainer import LOSS_TERMS  # noqa: E402

STEPS, INTERRUPT_AT = 200, 100
LOGGED = (*LOSS_TERMS, "grad_norm", "lr")


class InterruptAfterLast(TrainingObserver):
    def on_checkpoint(self, record) -> None:
        if record["kind"] == "last" and record["step"] == INTERRUPT_AT:
            raise KeyboardInterrupt


def metrics(run_dir: Path) -> dict[int, dict]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return {record["step"]: record for record in map(json.loads, lines)}


def weights_equal(a: Path, b: Path) -> bool:
    first, second = load_checkpoint(a)[0].state_dict(), load_checkpoint(b)[0].state_dict()
    return all(torch.equal(first[key], second[key]) for key in first)


def largest_diff(comparison: dict) -> float:
    return max(comparison["max_abs_diff_by_key"].values())


def compare(full: Path, other: Path) -> dict:
    reference, resumed = metrics(full), metrics(other)
    steps = range(INTERRUPT_AT + 1, STEPS + 1)
    identical = sum(all(resumed[s][k] == reference[s][k] for k in LOGGED) for s in steps)
    worst = {k: max(abs(resumed[s][k] - reference[s][k]) for s in steps) for k in LOGGED}
    final = [json.loads(x) for x in (full / "evaluations.jsonl").read_text().splitlines()][-1]
    final_other = [json.loads(x) for x in (other / "evaluations.jsonl").read_text().splitlines()][
        -1
    ]
    return {
        "steps_compared": len(steps),
        "steps_bitwise_identical": identical,
        "max_abs_diff_by_key": worst,
        "provenance_equal": all(
            resumed[s][k] == reference[s][k]
            for s in steps
            for k in ("row_sequence_sha256", "batch_rows", "epoch", "consumed_rows")
        ),
        "final_weights_equal": weights_equal(
            full / "checkpoint_last.pt", other / "checkpoint_last.pt"
        ),
        "step200_evaluation_flat_equal": final["flat"] == final_other["flat"],
        "step200_seen_rows_equal": final["legs"]["seen"]["rows"]
        == final_other["legs"]["seen"]["rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="run directory (default: temporary)")
    parser.add_argument("--device", default="cpu", help="cpu (bitwise check) or cuda")
    args = parser.parse_args()
    if args.device != "cpu":
        torch.use_deterministic_algorithms(True, warn_only=True)
    root = args.out or Path(tempfile.mkdtemp(prefix="resume_acceptance_"))
    corpus = make_synthetic_corpus(
        n_cells=16, draws_per_cell=4, n_weeks=52, context_lengths=(39,), k=4, m=2, j=1, val_cells=4
    )
    config = TrainingConfig(
        steps=STEPS, batch_size=4, lr=1e-3, warmup_steps=20, eval_every=100, eval_worlds=8,
        eval_samples=16, log_every=1, milestone_steps=(), prefetch_batches=2, seed=31,
    )  # fmt: skip
    kwargs = dict(
        model_config=ModelConfig(n_controls=2, n_demand=1),
        training=config,
        device=args.device,
        tracking=TrackingConfig(enabled=False),
    )
    seconds = {}

    def timed(name, **extra):
        started = time.perf_counter()
        try:
            train([corpus], out_dir=root / name, **kwargs, **extra)
        except KeyboardInterrupt:
            pass
        seconds[name] = round(time.perf_counter() - started, 1)

    timed("full")
    timed("interrupted", observers=[InterruptAfterLast()])
    checkpoint = root / "interrupted" / "checkpoint_last.pt"
    timed("resumed", resume_from=checkpoint)
    report = {"directory": str(root), "resumed": compare(root / "full", root / "resumed")}

    original_restore, original_rng = CorpusStream.restore, trainer_module.restore_rng_state
    try:
        CorpusStream.restore = lambda self, state: None
        timed("control_no_stream_restore", resume_from=checkpoint)
        CorpusStream.restore = original_restore
        trainer_module.restore_rng_state = lambda *args: None
        timed("control_no_rng_restore", resume_from=checkpoint)
    finally:
        CorpusStream.restore, trainer_module.restore_rng_state = original_restore, original_rng
    report["control_no_stream_restore"] = compare(root / "full", root / "control_no_stream_restore")
    report["control_no_rng_restore"] = compare(root / "full", root / "control_no_rng_restore")
    report["seconds"] = seconds

    resumed = report["resumed"]
    controls = ("control_no_stream_restore", "control_no_rng_restore")
    if args.device != "cpu":
        timed("full_repeat")
        floor = largest_diff(compare(root / "full", root / "full_repeat"))
        tolerance = max(2 * floor, 1e-5)
        report["device"] = torch.cuda.get_device_name(0) if args.device == "cuda" else args.device
        report["noise_floor"], report["tolerance"] = floor, tolerance
        report["passed"] = largest_diff(resumed) <= tolerance and all(
            largest_diff(report[name]) > 10 * tolerance for name in controls
        )
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    passed = (
        resumed["steps_bitwise_identical"] == resumed["steps_compared"]
        and resumed["provenance_equal"]
        and resumed["final_weights_equal"]
        and resumed["step200_evaluation_flat_equal"]
        and resumed["step200_seen_rows_equal"]
        and all(
            report[name]["steps_bitwise_identical"] == 0 and not report[name]["final_weights_equal"]
            for name in controls
        )
    )
    report["passed"] = passed
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
