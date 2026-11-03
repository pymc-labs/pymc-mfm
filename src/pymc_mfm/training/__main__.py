"""``python -m pymc_mfm.training``: train the canonical model on saved corpora.

Defaults come from :class:`TrainingConfig`, or from the named ``--preset``
(``canonical``, ``mini`` or ``smoke``; explicit flags override it). The ``mini``
and ``smoke`` presets are for development on the mini corpus plans and are not
validated surrogates for the canonical run. Pass a smaller
``--steps``/``--batch-size`` for a clearly identified reduced run that exercises
the same code path. Loading ``.npz`` corpora needs the ``generation`` extra;
the architecture is sized from corpus geometry. ``--holdout-archives
KIND[,KIND]`` (``best``, ``terminal``) also keeps the held-out draws of the
promoted and/or final evaluations as evaluation archives under ``archives/``
in the run directory, as ``train(holdout_archives=...)`` does.

``python -m pymc_mfm.training evaluate --model PATH --corpus F [F ...]
--out-dir DIR`` evaluates a model bundle directory or a training checkpoint
file on saved corpora into a new evaluation archive, as
:func:`~pymc_mfm.training.evaluate_into_archive` does: every held-out row by
default, every row with ``--split all``, or the ``--world SEED:ROW`` rows of
the corpora whose verified prior sidecars record those generator seeds. An
``evaluate_rows`` call evaluates at most ``--worlds-per-call`` worlds (4 by
default), which bounds its peak memory, and call ``i`` draws with seed
``--seed`` + ``i``. A request that cannot be archived, such as an unknown
seed, a repeated world or an identity the archive refuses, is a usage error
(exit status 2) and stops before any inference. The published archive is
verified by loading it and summarized as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.inference import from_pretrained
from pymc_mfm.training.archive import _evaluate_identity, _sidecar_seed, archive_identity
from pymc_mfm.training.checkpoint import file_sha256, load_checkpoint
from pymc_mfm.training.config import PRESETS, SELECTION_HEADS, TrainingConfig
from pymc_mfm.training.evaluation import DECOMPOSITION_HEADS
from pymc_mfm.training.tracking import TrackingConfig
from pymc_mfm.training.trainer import (
    HOLDOUT_ARCHIVE_KINDS,
    _holdout_archive_kinds,
    resolve_device,
    train,
)

if TYPE_CHECKING:
    from pymc_mfm.data.recipes import LoadedCorpus
    from pymc_mfm.model.model import MFM


def _milestones(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip()) if value else ()


def _names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(","))


def _archive_kinds(value: str) -> tuple[str, ...]:
    """``--holdout-archives``: distinct :data:`HOLDOUT_ARCHIVE_KINDS`, in canonical order."""
    try:
        return _holdout_archive_kinds(_names(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected distinct names from {', '.join(HOLDOUT_ARCHIVE_KINDS)}, got {value!r}"
        ) from None


def _world(value: str) -> tuple[int, int]:
    """``--world SEED:ROW`` as ``(seed, row)``, two non-negative decimal integers."""
    match = re.fullmatch(r"([0-9]+):([0-9]+)", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"expected SEED:ROW with non-negative integers, got {value!r}"
        )
    return int(match[1]), int(match[2])


def _preset_name(argv: Sequence[str]) -> str:
    """The ``--preset`` value in ``argv``, ``canonical`` when absent; exits on an unknown name."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--preset", choices=sorted(PRESETS), default="canonical")
    return probe.parse_known_args(list(argv))[0].preset


def build_parser(preset: str = "canonical") -> argparse.ArgumentParser:
    """Training parser whose flag defaults are the values of ``PRESETS[preset]``."""
    defaults = PRESETS[preset]
    parser = argparse.ArgumentParser(
        prog="python -m pymc_mfm.training", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--corpus", nargs="+", required=True, help="schema-4 corpus .npz files")
    parser.add_argument("--out-dir", required=True, help="new run directory")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=preset,
        help="named defaults for every training flag below; explicit flags override",
    )
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
                    "exact steps to retain; default keeps the preset's %s that fall "
                    "within --steps; pass '' for none" % (",".join(map(str, default)) or "none")
                ),
            )
        elif field.name == "lr_schedule":
            parser.add_argument(flag, choices=("cosine", "constant"), default=default)
        elif field.name == "selection_head":
            parser.add_argument(flag, choices=SELECTION_HEADS, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    parser.add_argument(
        "--holdout-archives",
        type=_archive_kinds,
        default=(),
        metavar="KIND[,KIND]",
        help="archive held-out evaluation draws at these checkpoints: best, terminal",
    )
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        prog="python -m pymc_mfm.training evaluate",
        description="Evaluate a model on saved corpora into a new evaluation archive.",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="model bundle directory, loaded locally (never from the Hub), or training "
        "checkpoint file",
    )
    parser.add_argument(
        "--corpus", nargs="+", required=True, metavar="F", help="schema-4 corpus .npz files"
    )
    parser.add_argument(
        "--out-dir", required=True, metavar="DIR", help="new or empty archive directory"
    )
    worlds = parser.add_mutually_exclusive_group()
    worlds.add_argument(
        "--split",
        choices=("holdout", "all"),
        help="every is_val == 1 row (holdout, the default) or every row of every corpus, "
        "in corpus order, rows ascending",
    )
    worlds.add_argument(
        "--world",
        action="append",
        type=_world,
        metavar="SEED:ROW",
        help="row ROW of the corpus whose verified prior sidecar records generator seed "
        "SEED; repeatable, evaluated in the given order",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=defaults.eval_samples,
        metavar="N",
        help="conditional draws per world (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.eval_seed + 1,
        metavar="S",
        help="draw seed of the first call; call i uses S + i (default: %(default)s, the "
        "trainer's held-out draw seed)",
    )
    parser.add_argument(
        "--heads",
        type=_names,
        default=DECOMPOSITION_HEADS,
        metavar="H[,H]",
        help="decomposition heads to sample; must include conditional (default: joint,conditional)",
    )
    parser.add_argument(
        "--worlds-per-call",
        type=int,
        default=4,
        metavar="N",
        help="at most N worlds per evaluate_rows call, which bounds its peak memory "
        "(default: %(default)s, about 12 GB on CPU for A/B shards)",
    )
    parser.add_argument(
        "--leg",
        default=None,
        metavar="LABEL",
        help="data_split label of the captures (default: holdout, all or selected, after "
        "--split or --world)",
    )
    parser.add_argument(
        "--device", default="cpu", metavar="D", help="auto | cpu | cuda | mps (default: cpu)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the training CLI, or the ``evaluate`` subcommand when it comes first."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["evaluate"]:
        return _evaluate(argv[1:])
    args = build_parser(_preset_name(argv)).parse_args(argv)
    from pymc_mfm.data.recipes import load_corpora, provenance

    loaded = load_corpora(args.corpus)
    values = {field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
    if values["milestone_steps"] is None:
        values["milestone_steps"] = tuple(
            step for step in PRESETS[args.preset].milestone_steps if step <= values["steps"]
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
        holdout_archives=args.holdout_archives,
    )
    summary = {
        "preset": args.preset,
        "run_dir": str(result.run_dir),
        "status": result.status,
        "final_step": result.final_step,
        "budget": result.run["training_config"]["budget"],
        "best": result.run["best"],
        "tracking": result.run["tracking"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate(argv: list[str]) -> int:
    """``python -m pymc_mfm.training evaluate``: evaluate a model into a new archive.

    Nothing is inferred before the request is checked. A problem with the
    arguments, the device, the selected rows, the ``--model`` path or the
    archive identity is a usage error (exit status 2); a corpus or model that
    fails to load, or a target that is not a new or empty directory, raises as
    in the training CLI. The calls the identity declares then run straight into
    the archive, which is verified by loading it before its summary is printed.
    """
    from pymc_mfm.data.recipes import load_corpora, provenance
    from pymc_mfm.diagnostics.archive import EvaluationArchive

    parser = build_evaluate_parser()
    args = parser.parse_args(argv)
    try:
        device = resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(f"--device {args.device}: {exc}")
    loaded = load_corpora(args.corpus)
    records = [provenance(item) for item in loaded]
    rows, leg = _selected_rows(parser, args, loaded, records)
    model, stated = _evaluated_model(parser, args.model)
    corpora = [item.corpus for item in loaded]
    try:
        identity = archive_identity(
            model,
            corpora,
            rows,
            n_samples=args.n_samples,
            seed=args.seed,
            leg=leg if args.leg is None else args.leg,
            device=device,
            heads=args.heads,
            worlds_per_call=args.worlds_per_call,
            corpus_provenance=records,
            **stated,
        )
    except ValueError as exc:
        parser.error(str(exc))
    path, _ = _evaluate_identity(model, corpora, identity, args.out_dir, device=device)
    archive = EvaluationArchive.load(path)
    identity = archive.identity
    evaluation = identity["evaluation"]
    summary = {
        "archive": str(path),
        "archive_sha256": archive.archive_sha256,
        "worlds": len(evaluation["worlds"]),
        "entries": len(archive),
        "calls": len(evaluation["calls"]),
        "leg": evaluation["leg"],
        "n_samples": evaluation["n_samples"],
        "seeds": [call["seed"] for call in evaluation["calls"]],
        "heads": evaluation["heads"],
        "state_sha256": identity["model"]["state_sha256"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _selected_rows(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    loaded: Sequence[LoadedCorpus],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[int, int]], str]:
    """The ``(corpus_index, row)`` pairs to evaluate and their default leg; exits when invalid.

    ``--split`` takes rows in corpus order, ascending. ``--world`` keeps the given
    order and maps each seed to the corpus whose verified prior sidecar records it.
    """
    if args.world is None:
        split = args.split or "holdout"
        rows = []
        for index, item in enumerate(loaded):
            selected = (
                range(item.n_worlds)
                if split == "all"
                else np.flatnonzero(np.asarray(item.corpus["is_val"]) == 1)
            )
            rows.extend((index, int(row)) for row in selected)
        if not rows and split == "holdout":
            parser.error(
                "--split holdout selects no worlds: no --corpus has is_val == 1 rows; "
                "use --split all or --world"
            )
        return rows, split
    corpus_of: dict[int, int] = {}
    for index, record in enumerate(records):
        seed = _sidecar_seed(record)
        if seed is not None:  # a seed shared by two corpora fails with the identity
            corpus_of.setdefault(seed, index)
    rows, seen = [], set()
    for seed, row in args.world:
        if seed not in corpus_of:
            parser.error(f"--world {seed}:{row}: no --corpus has verified generator seed {seed}")
        index = corpus_of[seed]
        if row >= loaded[index].n_worlds:
            parser.error(
                f"--world {seed}:{row}: the corpus with generator seed {seed} has "
                f"{loaded[index].n_worlds} worlds"
            )
        if (index, row) in seen:
            parser.error(f"--world {seed}:{row} repeats; each world is evaluated once")
        seen.add((index, row))
        rows.append((index, row))
    return rows, "selected"


def _evaluated_model(parser: argparse.ArgumentParser, value: str) -> tuple[MFM, dict[str, Any]]:
    """The ``--model`` and the identity facts it does not record itself; exits when absent.

    A directory is a model bundle, loaded locally (never from the Hub), which
    records its own source checkpoint, step and profile. A file is a training
    checkpoint, described by its file digest, its step and its run's profile.
    """
    path = Path(value).expanduser()
    if path.is_dir():
        return from_pretrained(path), {}
    if not path.is_file():
        parser.error(
            f"--model {value}: neither a model bundle directory nor a training checkpoint file"
        )
    model, payload = load_checkpoint(path)
    return model, {
        "checkpoint_sha256": file_sha256(path),
        "source_step": payload["step"],
        "profile": payload["run"]["profile"],
    }


if __name__ == "__main__":
    sys.exit(main())
