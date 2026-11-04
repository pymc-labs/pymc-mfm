"""``python -m pymc_mfm.training score``: one reproducible results row per checkpoint.

A checkpoint is scored in one of two modes, and the row records which:

``replay``
    Reproduces the trainer's in-loop held-out evaluation of the checkpoint's
    step: the same held-out rows, draw count and draw seed, all read from the
    checkpoint's ``training_config``. The corpora must be the ones the run
    consumed (their content SHA-256 is checked against the run's record before
    any inference). The scores are compared with the checkpoint's own
    ``evaluation`` record and the comparison is written into the row.

``eval-set``
    Scores a frozen set: every row of every given corpus, in corpus order, with a
    seed and draw count from the command line. It does not reproduce the in-loop
    record. Its id is the SHA-256 of the corpus keys, ordered selected rows, row
    rule, seed and draw count, so the same inputs always give the same id.

Rows are appended to a results CSV with a stable header; a CSV with another header
is refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pymc_mfm.model.graph import STRUCT_EDGE_KEYS
from pymc_mfm.training._provenance import code_record, corpus_identity, corpus_key
from pymc_mfm.training.checkpoint import file_sha256, import_plugins, load_checkpoint
from pymc_mfm.training.config import TrainingConfig
from pymc_mfm.training.evaluation import evaluate_rows, flatten_evaluation, selection_metrics
from pymc_mfm.training.stream import CorpusStream
from pymc_mfm.training.trainer import resolve_device

MODES = ("replay", "eval-set")
#: Row rule of ``eval-set`` mode: every row of every corpus, corpus order, rows ascending.
ALL_ROWS = "all-rows"
#: Row rule of ``replay`` mode: the trainer's in-loop held-out rows.
REPLAY_ROWS = "replay-holdout"
#: Leg label of the scored record. ``replay`` must use the trainer's ``holdout`` label to
#: compare flat keys; both modes strip the leg from the keys they write to the row.
_LEGS = {"replay": "holdout", "eval-set": "evalset"}
_HEADLINE_WINDOW = "query"
_HEADLINE_COMPONENT = "contribution"

_ROUTER_AUROC = tuple(f"router_auroc_{edge}" for edge in STRUCT_EDGE_KEYS)
RESULT_COLUMNS = (
    "run_name",
    "run_id",
    "step",
    "checkpoint_kind",
    "checkpoint_sha256",
    "code_revision",
    "code_modified_from_revision",
    "code_source_sha256",
    "model_config_sha256",
    "components",
    "training_config_sha256",
    "corpus_keys",
    "eval_set_id",
    "mode",
    "row_rule",
    "seed",
    "draws",
    "worlds",
    "device",
    "torch_threads",
    "torch_version",
    "selection_monitor",
    "selection_context_gate",
    "selection_structure_gate",
    "selection_leakage_gate",
    "selection_head",
    "cov80",
    "cov95",
    "crps",
    "router_shd_mean",
    "router_normalized_shd_mean",
    *_ROUTER_AUROC,
    "step_seconds_median",
    "replay_keys_compared",
    "replay_keys_exact",
    "replay_max_abs_diff",
    "metrics_json",
)


class ScoreError(ValueError):
    """A scoring request that cannot be honoured; the CLI reports it as a usage error."""


def canonical_sha256(value: Any) -> str:
    """SHA-256 of the compact, key-sorted JSON of ``value``."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def eval_set_id(
    corpus_keys: Sequence[str], rule: str, seed: int, draws: int, rows: Sequence[tuple[int, int]]
) -> str:
    """Hash the corpus keys, ordered selected rows, rule, seed and draws."""
    return canonical_sha256(
        {
            "corpora": list(corpus_keys),
            "rule": rule,
            "seed": seed,
            "draws": draws,
            "rows": list(rows),
        }
    )


def loaded_corpus_keys(loaded: Sequence[Any]) -> list[str]:
    """:func:`corpus_key` of each loaded corpus file."""
    keys = [
        corpus_key({"sha256": item.sha256, "actual_inputs": corpus_identity(item.corpus)})
        for item in loaded
    ]
    if None in keys:
        raise ScoreError("a corpus has neither a content nor a file SHA-256")
    return keys


def verify_run_corpora(
    payload: Mapping[str, Any], keys: Sequence[str], paths: Sequence[str]
) -> None:
    """Refuse corpora that differ from the ones the checkpoint's run consumed."""
    recorded = [corpus_key(record) for record in payload["run"]["corpus"]]
    if len(recorded) != len(keys):
        raise ScoreError(f"the run consumed {len(recorded)} corpora, but {len(keys)} were given")
    for index, (expected, actual) in enumerate(zip(recorded, keys, strict=True)):
        if expected is None:
            raise ScoreError(f"the run recorded no identity for corpus {index}")
        if expected != actual:
            raise ScoreError(
                f"corpus {index} ({paths[index]}) identity {actual} differs from the "
                f"run's recorded {expected}"
            )


def _finite_or_none(value: Any) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def compare_with_record(
    recorded: Mapping[str, float | None], replayed: Mapping[str, float | None]
) -> dict[str, Any]:
    """Compare the record's ``holdout/*`` flat keys with a replay of them.

    A key is exact when both values are equal or both undefined. A key the replay lacks, or
    defined in only one of the two, is a difference of infinity.
    """
    keys = sorted(key for key in recorded if key.startswith("holdout/"))
    exact, max_diff = 0, 0.0
    missing = object()
    for key in keys:
        old, new = recorded[key], replayed.get(key, missing)
        if old is None and new is None:
            diff = 0.0
        elif old is None or new is None or new is missing:
            diff = math.inf
        else:
            diff = abs(old - new)
        exact += diff == 0.0
        max_diff = max(max_diff, diff)
    return {"compared": len(keys), "exact": exact, "max_abs_diff": max_diff}


def median_step_seconds(checkpoint: Path, step: int) -> float | None:
    """Median ``step_seconds`` of the run's ``metrics.jsonl`` up to ``step``, if it sits beside."""
    path = checkpoint.parent / "metrics.jsonl"
    if not path.is_file():
        return None
    seconds = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if record["step"] <= step and "step_seconds" in record:
            seconds.append(record["step_seconds"])
    return statistics.median(seconds) if seconds else None


def _leg_free(flat: Mapping[str, float | None], leg: str) -> dict[str, float | None]:
    prefix = f"{leg}/"
    return {key.removeprefix(prefix): _finite_or_none(value) for key, value in flat.items()}


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)


def _headline(metrics: Mapping[str, float | None], head: str, name: str) -> float | None:
    return metrics.get(f"{head}/{_HEADLINE_WINDOW}/{_HEADLINE_COMPONENT}_{name}")


def score_checkpoint(
    checkpoint: str | Path,
    corpus_paths: Sequence[str | Path],
    *,
    mode: str,
    seed: int | None = None,
    draws: int | None = None,
    device: str = "cpu",
    threads: int | None = None,
    allow_research: bool = False,
) -> dict[str, Any]:
    """Score ``checkpoint`` on the corpora and return one results row (see the module doc).

    ``seed`` and ``draws`` are used by ``eval-set`` mode (default: ``TrainingConfig()``)
    and are refused in ``replay`` mode, where the checkpoint decides them.
    ``threads`` sets the torch thread count for the call; ``None`` keeps the current one.
    Eval-set scores all rows in one evaluation call per context length, so peak memory
    grows with the number of worlds.
    A checkpoint with research components is refused unless ``allow_research`` is true; the
    module that registers them must already be imported.
    """
    from pymc_mfm.data.recipes import load_corpora

    if mode not in MODES:
        raise ScoreError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "replay" and (seed is not None or draws is not None):
        raise ScoreError("replay mode takes its seed and draws from the checkpoint")
    checkpoint = Path(checkpoint)
    resolved = resolve_device(device)
    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        model, payload = load_checkpoint(checkpoint, device=resolved, allow_research=allow_research)
        training = payload["training_config"]
        loaded = load_corpora([str(path) for path in corpus_paths])
        corpora = [item.corpus for item in loaded]
        content = loaded_corpus_keys(loaded)
        if mode == "replay":
            if payload["evaluation"] is None:
                raise ScoreError(f"{checkpoint} has no evaluation record to replay")
            verify_run_corpora(payload, content, [item.path for item in loaded])
            seed, draws, rule = training["eval_seed"] + 1, training["eval_samples"], REPLAY_ROWS
            rows = _replay_rows(corpora, training)
        else:
            seed = TrainingConfig().eval_seed + 1 if seed is None else seed
            draws = TrainingConfig().eval_samples if draws is None else draws
            rule = ALL_ROWS
            rows = [
                (i, row) for i, corpus in enumerate(corpora) for row in range(_n_worlds(corpus))
            ]
        record = evaluate_rows(
            model,
            corpora,
            rows,
            n_samples=draws,
            seed=seed,
            device=resolved,
            leg=_LEGS[mode],
            per_world=True,
        )
        flat = flatten_evaluation(record)
        threads_used = torch.get_num_threads()
    finally:
        torch.set_num_threads(previous_threads)
    comparison = None
    if mode == "replay":
        comparison = compare_with_record(payload["evaluation"]["flat"], flat)
    return _row(
        payload,
        mode=mode,
        rule=rule,
        seed=seed,
        draws=draws,
        content=content,
        rows=rows,
        worlds=record["n_worlds"],
        device=str(resolved),
        threads=threads_used,
        metrics=_leg_free(flat, _LEGS[mode]),
        comparison=comparison,
        checkpoint_sha256=file_sha256(checkpoint),
        code=code_record(),
        step_seconds=median_step_seconds(checkpoint, payload["step"]),
    )


def _n_worlds(corpus: Mapping[str, Any]) -> int:
    return int(np.asarray(corpus["is_val"]).shape[0])


def _replay_rows(corpora: Sequence[Mapping[str, Any]], training: Mapping[str, Any]) -> list:
    """The trainer's in-loop held-out rows (``trainer.evaluate``)."""
    stream = CorpusStream(
        corpora, batch_size=training["batch_size"], seed=training["seed"], split="train"
    )
    n_holdout = sum(int(np.asarray(corpus["is_val"]).sum()) for corpus in corpora)
    return stream.holdout_rows(min(training["eval_worlds"], n_holdout), seed=training["eval_seed"])


def _row(
    payload: Mapping[str, Any],
    *,
    mode: str,
    rule: str,
    seed: int,
    draws: int,
    content: Sequence[str],
    rows: Sequence[tuple[int, int]],
    worlds: int,
    device: str,
    threads: int,
    metrics: Mapping[str, float | None],
    comparison: Mapping[str, Any] | None,
    checkpoint_sha256: str,
    code: Mapping[str, Any],
    step_seconds: float | None,
) -> dict[str, Any]:
    training = payload["training_config"]
    selection = selection_metrics(training["selection_head"])
    head = training["selection_head"]
    components = payload["schemas"].get("components")
    row: dict[str, Any] = {
        "run_name": payload["run"]["run_name"],
        "run_id": payload["run"]["run_id"],
        "step": payload["step"],
        "checkpoint_kind": payload["kind"],
        "checkpoint_sha256": checkpoint_sha256,
        "code_revision": code["revision"],
        "code_modified_from_revision": code["modified_from_revision"],
        "code_source_sha256": code["source_sha256"],
        "model_config_sha256": canonical_sha256(payload["config"]),
        "components": json.dumps(components, sort_keys=True) if components else None,
        "training_config_sha256": canonical_sha256(training),
        "corpus_keys": json.dumps(list(content)),
        "eval_set_id": eval_set_id(content, rule, seed, draws, rows),
        "mode": mode,
        "row_rule": rule,
        "seed": seed,
        "draws": draws,
        "worlds": worlds,
        "device": device,
        "torch_threads": threads,
        "torch_version": torch.__version__,
        "selection_monitor": selection.monitor,
        "selection_context_gate": selection.context_gate,
        "selection_structure_gate": selection.structure_gate,
        "selection_leakage_gate": selection.leakage_gate,
        "selection_head": head,
        "cov80": _headline(metrics, head, "cov80"),
        "cov95": _headline(metrics, head, "cov95"),
        "crps": _headline(metrics, head, "crps"),
        "router_shd_mean": metrics.get("joint_router/graph/shd_mean"),
        "router_normalized_shd_mean": metrics.get("joint_router/graph/normalized_shd_mean"),
        "step_seconds_median": step_seconds,
        "replay_keys_compared": None if comparison is None else comparison["compared"],
        "replay_keys_exact": None if comparison is None else comparison["exact"],
        "replay_max_abs_diff": None if comparison is None else comparison["max_abs_diff"],
        "metrics_json": json.dumps(metrics, sort_keys=True, allow_nan=False),
    }
    for edge, column in zip(STRUCT_EDGE_KEYS, _ROUTER_AUROC, strict=True):
        row[column] = metrics.get(f"joint_router/edge/{edge}/auroc")
    return {column: _cell(row[column]) for column in RESULT_COLUMNS}


def append_row(path: str | Path, row: Mapping[str, str]) -> None:
    """Append ``row`` to the results CSV, writing the header of a new or empty file.

    Raises :class:`ScoreError` when the file exists with a different header.
    """
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if tuple(header) != RESULT_COLUMNS:
            raise ScoreError(f"{path} has a different header; refusing to append")
        new = False
    else:
        new = True
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def add_research_arguments(parser: argparse.ArgumentParser) -> None:
    """``--allow-research`` and ``--plugin``, which reading a research checkpoint needs."""
    parser.add_argument(
        "--allow-research",
        action="store_true",
        help="read a checkpoint trained with research components (never exportable)",
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE",
        help="import MODULE first so it registers its components; repeatable",
    )


def build_score_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        prog="python -m pymc_mfm.training score",
        description=(
            "Score a training checkpoint into one row of a results CSV. --replay reproduces "
            "the trainer's in-loop held-out record of the checkpoint's step. --eval-set scores "
            "every row of the given corpora with --seed and --draws; it does NOT reproduce the "
            "in-loop record. Eval-set defaults use TrainingConfig(); replay uses the "
            "checkpoint's recorded settings."
        ),
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH", help="training checkpoint")
    parser.add_argument(
        "--corpus", nargs="+", required=True, metavar="F", help="schema-4 corpus .npz files"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--replay",
        action="store_const",
        dest="mode",
        const="replay",
        help="re-evaluate the in-loop held-out rows; the corpora must be the run's own "
        "(content SHA-256 verified) and the row records the comparison with the checkpoint",
    )
    mode.add_argument(
        "--eval-set",
        action="store_const",
        dest="mode",
        const="eval-set",
        help="score every row of every corpus, in corpus order",
    )
    parser.add_argument("--results", required=True, metavar="CSV", help="results CSV to append to")
    parser.add_argument("--device", default="cpu", help="auto | cpu | cuda | mps (default: cpu)")
    parser.add_argument(
        "--threads", type=int, default=None, metavar="N", help="torch thread count (default: as is)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="eval-set draw seed (default: %d from TrainingConfig())" % (defaults.eval_seed + 1),
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        metavar="N",
        help="eval-set draws per world (default: %d from TrainingConfig())" % defaults.eval_samples,
    )
    add_research_arguments(parser)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="score N times and fail unless every row is identical; one row is written",
    )
    return parser


def main(argv: Sequence[str]) -> int:
    """Run ``score``; a request that cannot be honoured is a usage error (exit status 2)."""
    parser = build_score_parser()
    args = parser.parse_args(list(argv))
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    try:
        import_plugins(args.plugin)
        rows = [
            score_checkpoint(
                args.checkpoint,
                args.corpus,
                mode=args.mode,
                seed=args.seed,
                draws=args.draws,
                device=args.device,
                threads=args.threads,
                allow_research=args.allow_research,
            )
            for _ in range(args.repeat)
        ]
        if any(row != rows[0] for row in rows[1:]):
            print("repeated scorings gave different rows", file=sys.stderr)
            return 1
        append_row(args.results, rows[0])
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    summary = {
        key: rows[0][key]
        for key in (
            "mode",
            "step",
            "eval_set_id",
            "worlds",
            "cov80",
            "cov95",
            "crps",
            "replay_keys_compared",
            "replay_keys_exact",
            "replay_max_abs_diff",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0
