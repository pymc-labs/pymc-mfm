"""Correctness and speed of bf16 autocast and torch.compile on one corpus.

Runs the real ``train()`` (tracking off) for the same number of updates in each mode:
fp32 for ``--seeds`` seeds, then bf16, fp32 + compile and bf16 + compile at the first
seed. Per mode it reports median step seconds after warm-up, first-call seconds,
graph breaks, peak memory, a loss summary and the final evaluation monitor. The checks:

* compiled vs eager: max abs loss difference between fp32 and fp32 + compile (same seed);
* bf16 final loss and final monitor lie within the spread of the fp32 seeds;
* the compiled runs' checkpoint and ``save_pretrained`` export reload to equal weights.

    uv run --no-sync python benchmarks/precision_compile_check.py --device cpu --steps 12 \
        --batch-size 4 --corpus corpus.npz --small-model --out report.json
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import statistics
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from pymc_mfm.data.recipes import load_corpora, provenance
from pymc_mfm.inference import from_pretrained
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.training import TrackingConfig, TrainingConfig, load_checkpoint, train
from pymc_mfm.training.trainer import corpus_geometry

WARMUP_STEPS = 3
#: Mean over this many final updates is the "final loss": one update is too noisy to compare.
FINAL_LOSS_WINDOW = 5
SMALL_MODEL = dict(d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def peak_memory_mb(device: torch.device) -> float:
    """Peak CUDA allocation since the last reset; on CPU the process's peak RSS so far."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def summarize(run_dir: Path, result, peak_mb: float) -> dict:
    steps = read_jsonl(run_dir / "metrics.jsonl")
    losses = [record["loss"] for record in steps]
    step_seconds = [record["step_seconds"] for record in steps]
    monitor = result.run["selection_policy"]["monitor"]
    final_evaluation = read_jsonl(run_dir / "evaluations.jsonl")[-1]
    compile_block = result.run["compile"]
    return {
        "median_step_seconds": statistics.median(step_seconds[WARMUP_STEPS:]),
        "first_call_seconds": compile_block["first_call_seconds"] or step_seconds[0],
        "graphs": compile_block["graphs"],
        "graph_breaks": compile_block["graph_breaks"],
        "peak_memory_mb": round(peak_mb, 1),
        "loss_first": losses[0],
        "loss_final": statistics.mean(losses[-FINAL_LOSS_WINDOW:]),
        "loss_min": min(losses),
        "losses": losses,
        "monitor": monitor,
        "final_monitor": final_evaluation["flat"][monitor],
    }


def weights_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    other = b.state_dict()
    return all(torch.equal(value.cpu(), other[key].cpu()) for key, value in a.state_dict().items())


def round_trip_equal(run_dir: Path, trained: torch.nn.Module) -> bool:
    """``checkpoint_last.pt`` and the ``save_pretrained`` export reload to the trained weights."""
    checkpoint_model, _ = load_checkpoint(run_dir / "checkpoint_last.pt")
    exported = from_pretrained(run_dir / "model")
    return weights_equal(checkpoint_model, trained) and weights_equal(exported, trained)


def within_spread(value: float, reference: list[float]) -> bool:
    return min(reference) <= value <= max(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--corpus", nargs="+", required=True, help="schema-4 corpus .npz files")
    parser.add_argument("--seeds", type=int, default=3, help="fp32 seeds (spread reference)")
    parser.add_argument("--small-model", action="store_true", help="tiny architecture for CPU")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--workdir", type=Path, default=None, help="run directories (temporary)")
    args = parser.parse_args()

    device = torch.device(args.device)
    loaded = load_corpora(args.corpus)
    corpora = [item.corpus for item in loaded]
    corpus_provenance = [provenance(item) for item in loaded]
    geometry = corpus_geometry(corpora)
    model_config = ModelConfig(
        n_controls=geometry["n_controls"],
        n_demand=geometry["n_demand"],
        **(SMALL_MODEL if args.small_model else {}),
    )
    base = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=min(args.steps // 10, 100),
        eval_every=args.steps,
        eval_worlds=4,
        eval_samples=16,
        log_every=1,
        milestone_steps=(),
    )
    root = args.workdir or Path(tempfile.mkdtemp(prefix="precision_compile_"))

    def run_mode(name: str, config: TrainingConfig) -> tuple[dict, Path, torch.nn.Module]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        run_dir = root / name
        result = train(
            corpora,
            out_dir=run_dir,
            corpus_provenance=corpus_provenance,
            model_config=model_config,
            training=config,
            device=device,
            tracking=TrackingConfig(enabled=False),
        )
        return summarize(run_dir, result, peak_memory_mb(device)), run_dir, result.model

    modes = {}
    for i in range(args.seeds):
        modes[f"fp32_seed{base.seed + i}"] = run_mode(
            f"fp32_seed{base.seed + i}", replace(base, seed=base.seed + i)
        )
    for name, precision, compile_ in (
        ("bf16", "bf16", False),
        ("fp32_compile", "fp32", True),
        ("bf16_compile", "bf16", True),
    ):
        modes[name] = run_mode(name, replace(base, precision=precision, compile=compile_))

    eager = modes[f"fp32_seed{base.seed}"][0]["losses"]
    compiled = modes["fp32_compile"][0]["losses"]
    fp32_runs = [summary for key, (summary, _, _) in modes.items() if key.startswith("fp32_seed")]
    bf16 = modes["bf16"][0]
    checks = {
        "compiled_vs_eager_max_abs_loss_diff": max(abs(a - b) for a, b in zip(eager, compiled)),
        "bf16_final_loss": bf16["loss_final"],
        "fp32_final_loss_range": [
            min(r["loss_final"] for r in fp32_runs),
            max(r["loss_final"] for r in fp32_runs),
        ],
        "bf16_final_loss_within_fp32_spread": within_spread(
            bf16["loss_final"], [r["loss_final"] for r in fp32_runs]
        ),
        "bf16_final_monitor": bf16["final_monitor"],
        "fp32_final_monitor_range": [
            min(r["final_monitor"] for r in fp32_runs),
            max(r["final_monitor"] for r in fp32_runs),
        ],
        "bf16_final_monitor_within_fp32_spread": within_spread(
            bf16["final_monitor"], [r["final_monitor"] for r in fp32_runs]
        ),
        "round_trip_weights_equal": {
            name: round_trip_equal(run_dir, model)
            for name, (_, run_dir, model) in modes.items()
            if name.endswith("compile")
        },
    }
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "corpus": [str(path) for path in args.corpus],
        "model": "small" if args.small_model else "default",
        "modes": {name: summary for name, (summary, _, _) in modes.items()},
        "checks": checks,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "modes": {
                    k: {m: v for m, v in s.items() if m != "losses"}
                    for k, s in report["modes"].items()
                },
                "checks": checks,
            },
            indent=2,
        )
    )
    if args.workdir is None:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
