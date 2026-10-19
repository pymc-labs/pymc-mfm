"""Run-start summaries of the training corpora and of the model's parameters.

All are JSON-safe. The trainer writes the first two to ``run.json`` as
``corpus_summary`` and ``parameter_counts``; the tracking observer logs every
numeric leaf of the corpus summary as a ``corpus/*`` metric at step 0 and the
parameter counts as ``config/parameter_counts/*`` parameters. The per-world
``corpus_support`` goes to its own ``corpus_support.json`` and is never logged.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

from pymc_mfm.model.graph import STRUCT_EDGE_KEYS, EdgeSlotLayout

if TYPE_CHECKING:
    from torch import nn

    from pymc_mfm.model.model import MFM

# ``parameter_counts`` keys and the ``MFM`` attribute each one counts.
_MODULES = (
    ("encoder", "encoder"),
    ("router", "head"),
    ("structure_head", "struct_head"),
    ("graph_conditioner", "graph_conditioner"),
    ("conditional_head", "conditional_head"),
)
# Per-world counts histogrammed per split, in ``corpus_summary`` order.
_HISTOGRAMS = ("context_lengths", "active_channels", "active_controls", "active_latents")
#: Worlds ``corpus_support`` keeps: all of them up to this many, else this many
#: evenly spaced over every row of the corpora in order.
CORPUS_SUPPORT_MAX_WORLDS = 5000


def parameter_counts(model: MFM) -> dict[str, int]:
    """Parameter count of each fixed ``MFM`` module and of the whole model.

    ``router`` is the joint flow head that samples graphs, ``structure_head``
    the per-edge classifier and ``conditional_head`` the graph-conditional
    decoder. ``total`` counts every parameter, trainable or not.
    """
    counts = {name: _numel(getattr(model, attribute)) for name, attribute in _MODULES}
    counts["total"] = _numel(model)
    return counts


def _numel(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def corpus_summary(corpora: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Split, context, node, edge and graph statistics of schema-4 corpora.

    Worlds split as the trainer splits them: ``train`` holds ``is_val == 0``
    and ``holdout`` ``is_val == 1``. Each split records ``worlds``, ``cells``
    (distinct ``(corpus, cell_id)`` pairs), ``distinct_graphs`` (distinct ``g``
    rows) and ``{value: worlds}`` histograms of the context length
    (``support_mask`` sum), of the active channel, control and latent counts,
    and of each edge family's edge count (``edges``, keyed by the model's
    family names). Histogram keys are decimal strings.
    ``shared_graphs`` counts the graphs present in both splits. ``generator``
    maps every finite numeric leaf of the corpora's ``diagnostics`` (dotted
    path; booleans excluded) to its ``{"min", "max"}`` over the corpora
    that report it. The corpora must share one ``(K, M, J)`` graph geometry.
    """
    if not corpora:
        raise ValueError("corpus_summary needs at least one corpus")
    splits = {
        split: {
            "worlds": 0,
            "cells": 0,
            "graphs": [],
            "histograms": {name: Counter() for name in _HISTOGRAMS},
            "edges": {family: Counter() for family in STRUCT_EDGE_KEYS},
        }
        for split in ("train", "holdout")
    }
    geometries = set()
    ranges: dict[str, list[int | float]] = {}
    for index, corpus in enumerate(corpora):
        masks = [
            np.asarray(corpus[key])
            for key in ("treatment_active_mask", "covariate_active_mask", "latent_active_mask")
        ]
        layout = EdgeSlotLayout(*(mask.shape[1] for mask in masks))
        geometries.add((layout.K, layout.M, layout.J))
        graph = np.asarray(corpus["g"])
        holdout = np.asarray(corpus["is_val"]).astype(bool)
        if graph.shape != (holdout.size, layout.n_slots):
            raise ValueError(
                f"corpus {index}: g must have shape {(holdout.size, layout.n_slots)}, "
                f"got {graph.shape}"
            )
        per_world = [np.asarray(corpus["support_mask"]).sum(axis=1)]
        per_world += [mask.sum(axis=1) for mask in masks]
        cells = np.asarray(corpus["cell_id"])
        packed = np.ascontiguousarray(np.packbits(graph.astype(bool), axis=1))
        graph_keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
        for split, rows in (("train", ~holdout), ("holdout", holdout)):
            state = splits[split]
            state["worlds"] += int(rows.sum())
            state["cells"] += int(np.unique(cells[rows]).size)
            state["graphs"].append(graph_keys[rows])
            for name, values in zip(_HISTOGRAMS, per_world, strict=True):
                _tally(state["histograms"][name], values[rows])
            for family, counter in state["edges"].items():
                _tally(counter, graph[rows, layout.slices[family]].sum(axis=1))
        diagnostics = corpus.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            for path, value in _numeric_leaves(diagnostics):
                bounds = ranges.setdefault(path, [value, value])
                bounds[:] = min(bounds[0], value), max(bounds[1], value)
    if len(geometries) > 1:
        raise ValueError(
            f"corpora must share one (K, M, J) graph geometry, got {sorted(geometries)}"
        )

    distinct = {
        split: np.unique(np.concatenate(state["graphs"])) for split, state in splits.items()
    }
    summary: dict[str, Any] = {"corpora": len(corpora)}
    for split, state in splits.items():
        summary[split] = {
            "worlds": state["worlds"],
            "cells": state["cells"],
            "distinct_graphs": int(distinct[split].size),
            **{name: _histogram(counter) for name, counter in state["histograms"].items()},
            "edges": {family: _histogram(counter) for family, counter in state["edges"].items()},
        }
    summary["shared_graphs"] = int(
        np.intersect1d(distinct["train"], distinct["holdout"], assume_unique=True).size
    )
    summary["generator"] = {
        path: {"min": low, "max": high} for path, (low, high) in sorted(ranges.items())
    }
    return summary


def corpus_support(
    corpora: Sequence[Mapping[str, Any]], *, max_worlds: int = CORPUS_SUPPORT_MAX_WORLDS
) -> dict[str, Any]:
    """Context-outcome location and scale of the corpora's worlds, for ``plot_corpus_support``.

    Each world ``"<corpus>:<row>"`` records the mean and standard deviation of
    its ``outcome_raw`` over the context window (the ``support_mask`` prefix),
    in :func:`pymc_mfm.diagnostics.comparisons.corpus_support_summary`'s layout.
    Beyond ``max_worlds`` rows, that many rows evenly spaced over all corpora
    in order are kept; ``n_worlds`` counts every row of both splits. A kept
    world with an empty or non-finite context window, or whose finite outcomes
    overflow the mean or standard deviation, is left out and named in
    ``skipped`` with its reason, so a defective row never aborts training here.
    """
    from pymc_mfm.diagnostics.comparisons import corpus_support_summary

    if isinstance(max_worlds, bool) or not isinstance(max_worlds, int) or max_worlds < 1:
        raise ValueError("max_worlds must be a positive integer")
    sizes = [len(np.asarray(corpus["is_val"])) for corpus in corpora]
    total = sum(sizes)
    if total == 0:
        raise ValueError("corpus_support needs at least one world")
    keep = np.arange(total)
    if total > max_worlds:
        keep = np.linspace(0, total - 1, max_worlds).round().astype(int)
    outcomes, skipped = {}, {}
    start = 0
    for index, (corpus, size) in enumerate(zip(corpora, sizes, strict=True)):
        rows = keep[(keep >= start) & (keep < start + size)] - start
        start += size
        if not rows.size:
            continue
        context = np.asarray(corpus["support_mask"])[rows].astype(bool)
        outcome = np.asarray(corpus["outcome_raw"])[rows].astype(np.float64)
        for row, mask, values in zip(rows.tolist(), context, outcome, strict=True):
            world, values = f"{index}:{row}", values[mask]
            if not values.size:
                skipped[world] = "empty_context"
            elif not np.isfinite(values).all():
                skipped[world] = "nonfinite_context_outcome"
            else:
                outcomes[world] = values
    summary = {
        "worlds": [],
        "provenance": {},
        "qualification": "No kept world has a nonempty context window with finite statistics.",
    }
    if outcomes:
        # Finite but extreme outcomes (float64 near its range) overflow the statistics.
        with np.errstate(over="ignore", invalid="ignore"):
            described = corpus_support_summary(outcomes)
        worlds = []
        for world in described["worlds"]:
            if math.isfinite(world["mean"]) and math.isfinite(world["std"]):
                worlds.append(world)
            else:
                skipped[world["world"]] = "nonfinite_context_statistics"
        if worlds:
            summary = {**described, "worlds": worlds}
    summary["n_worlds"] = total
    summary["skipped"] = skipped
    return summary


def _tally(counter: Counter, values: np.ndarray) -> None:
    unique, frequency = np.unique(values, return_counts=True)
    counter.update(dict(zip(unique.tolist(), frequency.tolist(), strict=True)))


def _histogram(counter: Counter) -> dict[str, int]:
    return {str(int(value)): int(counter[value]) for value in sorted(counter)}


def _numeric_leaves(value: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, Mapping):
            yield from _numeric_leaves(item, f"{path}.")
        elif isinstance(item, Real) and not isinstance(item, bool) and math.isfinite(item):
            yield path, item.item() if isinstance(item, np.generic) else item
