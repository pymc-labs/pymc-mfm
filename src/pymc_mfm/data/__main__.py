"""Corpus generation entry point: ``python -m pymc_mfm.data generate ...``.

``--profile {A,B} --shard I`` writes shard ``I`` of that profile's plan in
``pymc_mfm.data.recipes.SHARD_PLANS`` and ``--evaluation-shard I`` shard ``I``
of the evaluation plan. Without either, ``--profile`` with ``--n-cells``,
``--draws-per-cell``, ``--seed`` and optional ``--override`` values builds a
custom prior with ``profile_prior``. JSON lists in ``--override`` values become
tuples because ``SCMPrior`` ranges are tuples. An override must name an
``SCMPrior`` field other than the three that have their own flags.

``--out`` is an existing directory, which receives ``shard_<seed>.npz``, or a
``.npz`` file path. An existing corpus there is replaced only with ``--force``.
``generate_corpus`` writes the corpus and its ``.prior.json`` sidecar
atomically. Arguments are checked before anything is generated. Requires the
repository's ``generation`` group: ``uv sync --locked --group generation``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from pymc_mfm.data.profiles import PROFILE_A, PROFILE_B
from pymc_mfm.data.recipes import (
    SHARD_PLANS,
    ShardPlan,
    generate_corpus,
    profile_prior,
    recipe_for,
)

_PROFILES = {PROFILE_A.name: PROFILE_A, PROFILE_B.name: PROFILE_B}


def _tuples(value: Any) -> Any:
    """Turn JSON lists into tuples recursively; leave everything else unchanged."""
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


def _parse_override(item: str) -> tuple[str, Any]:
    key, separator, raw = item.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"override must be key=json, got {item!r}")
    try:
        return key, _tuples(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"override {key!r} value is not JSON: {raw!r}") from exc


_FLAG_FIELDS = {"n_cells": "--n-cells", "draws_per_cell": "--draws-per-cell", "seed": "--seed"}


def _check_overrides(command: argparse.ArgumentParser, overrides: list[tuple[str, Any]]) -> None:
    """Usage errors for override keys that are not free ``SCMPrior`` fields."""
    if not overrides:
        return
    try:
        from pymc_generator import SCMPrior
    except ImportError:
        return  # profile_prior reports the missing generator
    known = {field.name for field in dataclasses.fields(SCMPrior)}
    for key, _ in overrides:
        if key in _FLAG_FIELDS:
            command.error(f"--override {key} repeats {_FLAG_FIELDS[key]}; set it with that flag")
        if key not in known:
            command.error(f"--override {key} is not an SCMPrior field")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pymc_mfm.data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate one corpus file")
    generate.add_argument("--profile", choices=sorted(_PROFILES))
    generate.add_argument("--shard", type=int, help="shard index in the profile's plan")
    generate.add_argument("--evaluation-shard", type=int, help="shard index in the evaluation plan")
    generate.add_argument("--n-cells", type=int)
    generate.add_argument("--draws-per-cell", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument(
        "--out", required=True, help="existing directory (gets shard_<seed>.npz) or .npz path"
    )
    generate.add_argument(
        "--override",
        action="append",
        default=[],
        type=_parse_override,
        metavar="KEY=JSON",
        help="SCMPrior field override for a custom corpus, repeatable",
    )
    generate.add_argument("--force", action="store_true", help="replace an existing corpus")
    args = parser.parse_args(argv)

    plan, index = _plan_shard(generate, args)
    _check_overrides(generate, args.override)
    seed = args.seed if plan is None else plan.seed(index)
    target = _target(generate, args.out, seed, force=args.force)
    try:
        if plan is None:
            profile = _PROFILES[args.profile]
            prior = profile_prior(
                profile,
                n_cells=args.n_cells,
                draws_per_cell=args.draws_per_cell,
                seed=seed,
                **dict(args.override),
            )
            source = f"recipe {recipe_for(profile).name}"
        else:
            prior = plan.prior(index)
            label = "" if plan.shard_names is None else f" ({plan.shard_names[index]})"
            source = (
                f"recipe {recipe_for(plan.profile).name}, plan {plan.name} shard {index}{label}"
            )
    except ValueError as error:
        generate.error(str(error))
    corpus = generate_corpus(prior, target)
    n_worlds = int(corpus["is_val"].shape[0])
    n_holdout = int(corpus["is_val"].sum())
    print(f"wrote {target}: {n_worlds} worlds ({n_holdout} holdout) from {source}, seed {seed}")
    return 0


def _plan_shard(
    command: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[ShardPlan | None, int | None]:
    """The requested plan and shard, ``(None, None)`` for a custom corpus; exits when invalid."""
    custom = {
        "--n-cells": args.n_cells,
        "--draws-per-cell": args.draws_per_cell,
        "--seed": args.seed,
        "--override": args.override or None,
    }
    if args.evaluation_shard is not None:
        option, plan, index = "--evaluation-shard", SHARD_PLANS["evaluation"], args.evaluation_shard
        excluded = {**custom, "--profile": args.profile, "--shard": args.shard}
    elif args.shard is not None:
        if args.profile is None:
            command.error("--shard requires --profile")
        option, plan, index = "--shard", SHARD_PLANS[args.profile], args.shard
        excluded = custom
    else:
        required = {
            "--profile": args.profile,
            "--n-cells": args.n_cells,
            "--draws-per-cell": args.draws_per_cell,
            "--seed": args.seed,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            command.error(
                f"a custom corpus needs {', '.join(missing)}; "
                "otherwise pass --shard or --evaluation-shard"
            )
        return None, None
    conflicts = [flag for flag, value in excluded.items() if value is not None]
    if conflicts:
        command.error(f"{option} excludes {', '.join(conflicts)}")
    if not 0 <= index < plan.n_shards:
        command.error(
            f"{option} must be in 0..{plan.n_shards - 1} for plan {plan.name}, got {index}"
        )
    return plan, index


def _target(command: argparse.ArgumentParser, out: str, seed: int, *, force: bool) -> Path:
    """Corpus path for ``--out``; exits unless it may be written."""
    path = Path(out)
    if path.is_dir():
        path = path / f"shard_{seed}.npz"
    elif path.suffix != ".npz":
        command.error(f"--out must be an existing directory or a .npz path, got {out!r}")
    if path.is_dir():
        command.error(f"{path} is a directory")
    if path.exists() and not force:
        command.error(f"{path} exists; pass --force to replace it")
    # Fail before generating, which can take minutes, rather than at the final write.
    # Nothing is created here: generate_corpus makes missing parents.
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        command.error(f"{ancestor} is not a directory")
    if not os.access(ancestor, os.W_OK | os.X_OK):
        command.error(f"{ancestor} is not writable")
    return path


if __name__ == "__main__":
    sys.exit(main())
