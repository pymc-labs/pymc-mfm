"""Profile training updates with ``torch.profiler``: top ops and per-phase time.

Runs ``--warmup`` unprofiled updates, then profiles ``--steps`` updates with the CPU
activity and, when the device is CUDA, the CUDA activity. Each update is split into the
phases ``forward_loss``, ``backward`` and ``optimizer_step`` with ``record_function``.
The JSON file holds the top ops by self time and the total time of each phase in
microseconds; a top-3 summary is printed.

    uv run --no-sync python benchmarks/train_profile.py --corpus mini/shard_550000.npz \\
        --preset mini --device cuda --out /tmp/devloop-p5/profile.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from pymc_mfm.data.recipes import load_corpora
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.model import MFM
from pymc_mfm.training.config import PRESETS
from pymc_mfm.training.stream import CorpusStream
from pymc_mfm.training.trainer import corpus_geometry, resolve_device

PHASES = ("forward_loss", "backward", "optimizer_step")


def update(model, optimizer, stream, weights, grad_clip, device) -> None:
    """One training update, each phase under its own ``record_function`` label."""
    batch, _ = stream.next_batch()
    batch = {
        key: value if key == "sales_scale" else value.to(device) for key, value in batch.items()
    }
    with record_function("forward_loss"):
        terms = model.loss_terms(batch, weights=weights)
    with record_function("backward"):
        optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    with record_function("optimizer_step"):
        optimizer.step()


def summarize(events, device_type: str, top: int) -> dict:
    """Top ops by self time and the phase totals, in microseconds."""
    time_of = (
        (lambda e: e.self_device_time_total)
        if device_type == "cuda"
        else (lambda e: e.self_cpu_time_total)
    )
    phase_events = {event.key: event for event in events if event.key in PHASES}
    ranked = sorted((e for e in events if e.key not in PHASES), key=time_of, reverse=True)
    return {
        "phases_us": {
            name: {
                "cpu_total": phase_events[name].cpu_time_total,
                "device_total": phase_events[name].device_time_total,
            }
            for name in PHASES
        },
        "top_ops": [
            {
                "op": event.key,
                "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "self_device_us": event.self_device_time_total,
            }
            for event in ranked[:top]
        ],
        "ranked_by": "self device time" if device_type == "cuda" else "self cpu time",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mini")
    parser.add_argument("--batch-size", type=int, default=None, help="default: the preset's")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = PRESETS[args.preset]
    batch_size = args.batch_size or config.batch_size
    device = resolve_device(args.device)
    corpora = [item.corpus for item in load_corpora(args.corpus)]
    geometry = corpus_geometry(corpora)
    torch.manual_seed(config.seed)
    model = MFM(ModelConfig(n_controls=geometry["n_controls"], n_demand=geometry["n_demand"]))
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    stream = CorpusStream(corpora, batch_size=batch_size, seed=config.seed, device="cpu")
    step = (model, optimizer, stream, config.loss_weights, config.grad_clip, device)

    for _ in range(args.warmup):
        update(*step)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as profiler:
        for _ in range(args.steps):
            update(*step)
        if device.type == "cuda":
            torch.cuda.synchronize()

    report = {
        "preset": args.preset,
        "batch_size": batch_size,
        "device": str(device),
        "warmup_steps": args.warmup,
        "profiled_steps": args.steps,
        "torch": torch.__version__,
        **summarize(profiler.key_averages(), device.type, args.top),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, total in report["phases_us"].items():
        print(
            f"phase {name}: cpu {total['cpu_total'] / 1e3:.1f} ms, "
            f"device {total['device_total'] / 1e3:.1f} ms"
        )
    for op in report["top_ops"][:3]:
        print(
            f"top op {op['op']}: {op['calls']} calls, "
            f"self cpu {op['self_cpu_us'] / 1e3:.1f} ms, "
            f"self device {op['self_device_us'] / 1e3:.1f} ms"
        )


if __name__ == "__main__":
    main()
