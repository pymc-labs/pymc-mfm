"""Corpus generation entry point: ``python -m pymc_mfm.data generate ...``.

Builds the profile's prior with ``pymc_mfm.data.recipes.profile_prior`` and
writes the corpus plus its ``.prior.json`` sidecar with ``generate_corpus``.
Requires the ``generation`` extra. JSON lists in ``--override`` values become
tuples because ``SCMPrior`` ranges are tuples.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pymc_mfm.data.profiles import PROFILE_A, PROFILE_B
from pymc_mfm.data.recipes import generate_corpus, profile_prior, recipe_for

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pymc_mfm.data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="generate one corpus file for a profile")
    generate.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    generate.add_argument("--n-cells", type=int, required=True)
    generate.add_argument("--draws-per-cell", type=int, required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--out", required=True, help="corpus .npz path")
    generate.add_argument(
        "--override",
        action="append",
        default=[],
        type=_parse_override,
        metavar="KEY=JSON",
        help="SCMPrior field override, repeatable",
    )
    args = parser.parse_args(argv)

    profile = _PROFILES[args.profile]
    recipe = recipe_for(profile)
    prior = profile_prior(
        profile,
        n_cells=args.n_cells,
        draws_per_cell=args.draws_per_cell,
        seed=args.seed,
        **dict(args.override),
    )
    corpus = generate_corpus(prior, args.out)
    n_worlds = int(corpus["is_val"].shape[0])
    n_holdout = int(corpus["is_val"].sum())
    flag = " (reconstructed recipe)" if recipe.reconstructed else ""
    print(
        f"wrote {args.out}: {n_worlds} worlds ({n_holdout} holdout) from recipe "
        f"{recipe.name}{flag}, seed {args.seed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
