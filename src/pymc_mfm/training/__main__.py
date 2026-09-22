"""``python -m pymc_mfm.training``: train the canonical model on saved corpora.

Defaults come from :class:`TrainingConfig`; pass a smaller
``--steps``/``--batch-size`` for a clearly identified reduced run that exercises
the same code path. Loading ``.npz`` corpora needs the ``generation`` extra;
the architecture is sized from corpus geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields

from pymc_mfm.training.config import TrainingConfig
from pymc_mfm.training.tracking import TrackingConfig
from pymc_mfm.training.trainer import train


def _milestones(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip()) if value else ()


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        prog="python -m pymc_mfm.training", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--corpus", nargs="+", required=True, help="schema-4 corpus .npz files")
    parser.add_argument("--out-dir", required=True, help="new run directory")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="explicitly disable MLflow and diagnostic delivery",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow URI; defaults to MLFLOW_TRACKING_URI or output-local SQLite",
    )
    parser.add_argument("--experiment-name", default="pymc-mfm")
    parser.add_argument(
        "--profile", default="new-run", help="honest corpus/checkpoint profile label"
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="retain evaluation metrics without rendering reports",
    )
    for field in fields(TrainingConfig):
        flag = "--" + field.name.replace("_", "-")
        default = getattr(defaults, field.name)
        if field.name == "milestone_steps":
            parser.add_argument(
                flag,
                type=_milestones,
                default=None,
                metavar="STEP[,STEP...]",
                help=(
                    "exact steps to retain; default keeps the canonical %s that fall "
                    "within --steps; pass '' for none" % ",".join(map(str, default))
                ),
            )
        elif field.name == "lr_schedule":
            parser.add_argument(flag, choices=("cosine", "constant"), default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pymc_mfm.data.recipes import load_corpora, provenance

    loaded = load_corpora(args.corpus)
    values = {field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
    if values["milestone_steps"] is None:
        values["milestone_steps"] = tuple(
            step for step in TrainingConfig().milestone_steps if step <= values["steps"]
        )
    training = TrainingConfig(**values)
    result = train(
        [item.corpus for item in loaded],
        out_dir=args.out_dir,
        training=training,
        device=args.device,
        corpus_provenance=[provenance(item) for item in loaded],
        run_name=args.run_name,
        tracking=TrackingConfig(
            enabled=not args.no_tracking,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            profile=args.profile,
            diagnostics=not args.no_diagnostics,
        ),
    )
    summary = {
        "run_dir": str(result.run_dir),
        "status": result.status,
        "final_step": result.final_step,
        "budget": result.run["training_config"]["budget"],
        "best": result.run["best"],
        "tracking": result.run["tracking"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
