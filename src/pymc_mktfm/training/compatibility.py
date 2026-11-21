"""Compatibility of a bundle with the training evaluation recorded in its provenance.

A bundle's ``training_evaluation`` provenance names the held-out worlds
(``{"shard_seed", "row"}``), draw count and evaluation seed of its terminal
in-training evaluation, and the joint-head values recorded there with each
key's aggregation label. :func:`check_recorded_evaluation` evaluates those
worlds ``replicates`` times with fresh draw seeds and compares every recorded
value with the same aggregate of each replicate. A ``slot_pooled`` value pools
the slots of every world where the key is defined (``sum(value * weight) /
sum(weight)``); a ``world``, ``slot`` or ``pooled_edges`` value is the
evaluator's leg value, as :func:`~pymc_mktfm.training.evaluation.metric_aggregation`
defines it.

* **exact** keys do not depend on draws (``structure_head/*``, ``counts/*``,
  ``joint_router/edge/*/{n,n_pos,prevalence,unc}``, where ``unc = r (1 - r)``
  of the prevalence ``r``, and ``joint/*/contribution_truth_lag1``) and must
  agree within ``max(EXACT_TOLERANCE, delta_k)`` in every replicate;
* **informational** keys (``joint_router/edge/*/log_score``: clipped at machine
  epsilon, so one confident miss dominates; ``joint_router/edge/*/{mcb,dsc}``:
  they split the compared ``brier`` minus the exact ``unc``, and their
  isotonic term jumps by at least ``1 / (2 n)`` when one draw ties or inverts a
  positive and a negative edge that replicates keep apart) are reported but
  not counted;
* **stochastic** keys are every other key. The recorded value is compared with
  the replicate mean within the key's tolerance
  ``max(t * sd * sqrt(1 + 1/R), delta_k)``, the half-width of a 99% prediction
  interval: ``t`` is the :data:`T_LEVEL` Student-t quantile on ``R - 1``
  degrees of freedom and ``sd`` the replicate standard deviation (``ddof=1``).
  A key whose replicates are all equal has the tolerance
  ``max(EXACT_TOLERANCE, delta_k)``. The 99.9% interval of the family rule uses
  the :data:`FAMILY_T_LEVEL` quantile with the same floor.

``delta_k`` (``event_step``) is the resolution of a discrete key: the change one
elementary event makes to its value, in whichever defined world (or pooled
edge set) it weighs most, or for the co-moments the spacing of their value
grid; it is ``None`` for keys with continuous values. With ``S`` draws per world:

* a key that averages indicators moves by one indicator. A world averages
  ``weight * u`` of them: its ``weight`` slots (1 for a world-level key) of
  ``u`` each, which are the window's weeks for weekly coverage, one for
  time-average coverage, and for router frequencies the ``S`` draws, times the
  node pairs of a two-cycle rate or the active slots of ``normalized_shd_mean``.
  One indicator moves an equal-world mean by ``1 / (n_defined * weight * u)``
  and a slot-pooled value by ``1 / (sum(weights) * u)``; ``null_fpr`` averages
  the negative slots of its edge type, so ``1 / n_neg``;
* one slot of one draw moves that draw's Hamming distance to the true graph by
  one: ``shd_mean`` by ``1 / S`` in a world, and ``shd_median`` by ``1/2`` for
  even ``S`` (the mean of the two middle distances) or ``1`` for odd ``S``. One
  draw moves ``n_distinct`` by at most ``1``;
* a slot pair's co-moment ``|c S - a b| / S**2`` (co-occurrence count ``c`` and
  on-counts ``a``, ``b`` of the pair over the draws) lies on a grid of spacing
  ``1 / S**2``, the step of ``pairwise_comoment_max_abs_err`` in a world; the
  mean ``pairwise_comoment_abs_err`` over the ``L (L - 1) / 2`` pairs of ``L``
  active slots lies on a grid of spacing ``2 / (S**2 L (L - 1))``. These steps
  are the smallest change a co-moment can make; one slot of one draw can move
  a pair's co-moment by up to ``(S - 1) / S**2``;
* the pooled ranking scores see each active slot's draw frequency, which one
  draw moves by ``1 / S``. AUROC is ``U / (n_pos n_neg)`` with a tied
  positive-negative pair counting one half, so a positive slot moves it by up
  to ``1 / (2 n_pos)`` and a negative one by up to ``1 / (2 n_neg)``, when
  every slot of the other class shares its old or new frequency:
  ``1 / (2 min(n_pos, n_neg))``. AUPRC, the mean over the ``n_pos`` positives
  of the precision at their frequency, moves by less than ``1 / n_pos``.

The rules apply in this order, and ``reason`` names the first one that fails:

1. ``no_compared_keys``: no exact or stochastic key was compared;
2. ``exact_failures``: an exact key disagrees in a replicate, is undefined in
   the recording or a replicate but not in all of them, or has a non-finite
   replicate;
3. ``must_pass_failures``: :data:`HEADLINE` lies outside its tolerance, or a
   stochastic key is undefined in the recording or a replicate but not in all
   of them, or has a non-finite replicate;
4. ``gross_failures``: a stochastic key lies more than :data:`GROSS_FACTOR`
   tolerances from the replicate mean;
5. ``family_failures``: two keys of one metric family lie outside their 99.9%
   intervals;
6. ``allowance_exceeded``: more metric families have a key outside its
   tolerance than ``binom.ppf(ALLOWANCE_LEVEL, k, FAILURE_RATE)``, where ``k``
   is the number of families with a compared stochastic key; a failing family
   counts once however many of its keys fail.

A key recorded as undefined passes only when every replicate is undefined too.
A metric family groups the keys that score one quantity, which a regression
moves together:

* ``{head}/{window}/{component}``: ``{component}_{scaled_mae,crps,cov80}`` of one
  decomposition component;
* ``{head}/{window}/temporal``: the lag-1 family ``contribution_{truth_lag1, ...}``;
* ``{head}/{window}/timeavg_coverage``: ``timeavg_contribution_cov80``, its
  ``_live`` and its ``_dead`` variant;
* ``{head}/{window}/attribution_{live,dead}_{media,control}``: one attribution
  group;
* ``{head}/{window}/latent_sales``: ``closure_*`` and ``latent_sales_r2``;
* ``joint_router/graph/distance``: ``shd_mean``, ``shd_median``,
  ``normalized_shd_mean``, ``exact_recovery`` and ``energy_score``;
* ``joint_router/graph/spread``: the diversity and dependence of the router's
  draws, ``entropy``, ``n_distinct``, ``pairwise_comoment_abs_err`` and
  ``pairwise_comoment_max_abs_err``;
* ``joint_router/graph/{block}_cycles``: ``{block}_two_cycle_rate`` and
  ``{block}_cycle_rate`` of the ``cc`` or ``zz`` block;
* ``joint_router/edge/{et}``: the scores of edge type ``et``;

and every other key is a family of its own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch

from pymc_mktfm.inference._artifact_schema import validate_semantic_schema
from pymc_mktfm.model import FLOW_STEPS_SAMPLE
from pymc_mktfm.model.graph import SQUARE_EDGE_TYPES, EdgeSlotLayout
from pymc_mktfm.model.model import MFM
from pymc_mktfm.training.evaluation import (
    DECOMPOSITION_HEADS,
    DECOMPOSITION_METRICS,
    evaluate_rows,
    flatten_evaluation,
    metric_aggregation,
)
from pymc_mktfm.training.metrics import LATENT_SALES_KEYS, TEMPORAL_KEYS, TIME_AVERAGE_KEYS

#: Fewest replicates; the prediction interval needs a spread estimate.
MIN_REPLICATES = 8
#: Largest difference of an exact key in any replicate, and the tolerance of a
#: stochastic key without replicate spread or event step.
EXACT_TOLERANCE = 1e-5
#: Student-t quantile of a key's two-sided 99% prediction interval.
T_LEVEL = 0.995
#: Student-t quantile of the two-sided 99.9% interval of the family rule.
FAMILY_T_LEVEL = 0.9995
#: A stochastic key further than this many tolerances from its replicates fails the check.
GROSS_FACTOR = 3.0
#: Assumed per-family false-failure rate, and the level of the binomial allowance.
FAILURE_RATE = 0.01
ALLOWANCE_LEVEL = 0.99
#: The headline key, which the binomial allowance never covers.
HEADLINE = "joint/query/decomposition_energy_score"
_LEG = "holdout"
_HEADS = ("joint",)
_RECORD_KEYS = {
    "status",
    "source",
    "step",
    "leg",
    "head",
    "eval_seed",
    "n_worlds",
    "n_samples",
    "flow_steps",
    "worlds",
    "metrics",
    "aggregation",
}
_COMPONENT_STATS = ("scaled_mae", "crps", "cov80")
_DECOMPOSITION_NAMES = frozenset(DECOMPOSITION_METRICS)
_TEMPORAL_NAMES = frozenset(f"contribution_{key}" for key in TEMPORAL_KEYS)
_TIME_AVERAGE_NAMES = frozenset(f"timeavg_contribution_{key}" for key in TIME_AVERAGE_KEYS)
_GRAPH_DISTANCES = frozenset(
    ("shd_mean", "shd_median", "normalized_shd_mean", "exact_recovery", "energy_score")
)
_GRAPH_SPREAD = frozenset(
    ("entropy", "n_distinct", "pairwise_comoment_abs_err", "pairwise_comoment_max_abs_err")
)
_CYCLE_BLOCKS = {
    f"{block}_{rate}": block
    for block in SQUARE_EDGE_TYPES
    for rate in ("two_cycle_rate", "cycle_rate")
}


def key_class(key: str) -> str:
    """``"exact"``, ``"informational"`` or ``"stochastic"`` for a leg-free key."""
    parts = key.split("/")
    if parts[0] in ("structure_head", "counts"):
        return "exact"
    if parts[:2] == ["joint_router", "edge"]:
        if parts[-1] in ("n", "n_pos", "prevalence", "unc"):
            return "exact"
        if parts[-1] in ("log_score", "mcb", "dsc"):
            return "informational"
    if parts[0] in DECOMPOSITION_HEADS and parts[-1] == "contribution_truth_lag1":
        return "exact"
    return "stochastic"


def _family(key: str) -> str:
    """The metric family of a leg-free key (see the module doc)."""
    parts = key.split("/")
    scope, name = "/".join(parts[:-1]), parts[-1]
    if parts[0] in DECOMPOSITION_HEADS:
        if name in _TEMPORAL_NAMES:
            return f"{scope}/temporal"
        if name in _TIME_AVERAGE_NAMES:
            return f"{scope}/timeavg_coverage"
        if name in LATENT_SALES_KEYS:
            return f"{scope}/latent_sales"
        if name.startswith("attribution_"):
            return f"{scope}/{'_'.join(name.split('_')[:3])}"
        for stat in _COMPONENT_STATS:
            component = name.removesuffix(f"_{stat}")
            if component != name and all(
                f"{component}_{other}" in _DECOMPOSITION_NAMES for other in _COMPONENT_STATS
            ):
                return f"{scope}/{component}"
    elif parts[:2] == ["joint_router", "graph"]:
        if name in _GRAPH_DISTANCES:
            return f"{scope}/distance"
        if name in _GRAPH_SPREAD:
            return f"{scope}/spread"
        if name in _CYCLE_BLOCKS:
            return f"{scope}/{_CYCLE_BLOCKS[name]}_cycles"
    elif parts[:2] == ["joint_router", "edge"]:
        return scope
    return key


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _training_evaluation(model: MFM) -> Mapping[str, Any]:
    """The recorded evaluation of ``model``'s provenance, checked for this comparison."""
    metadata = getattr(model, "artifact_metadata", {})
    if metadata:
        validate_semantic_schema(metadata.get("schemas"), model.config, label="loaded artifact")
    record = getattr(model, "provenance", {}).get("training_evaluation")
    if not isinstance(record, Mapping):
        raise ValueError("the model's provenance holds no recorded training_evaluation")
    if set(record) != _RECORD_KEYS:
        raise ValueError(f"training_evaluation keys differ from {sorted(_RECORD_KEYS)}")
    if (record["status"], record["leg"], record["head"]) != ("recorded", _LEG, "joint"):
        raise ValueError("training_evaluation must be a recorded holdout joint-head evaluation")
    if record["flow_steps"] != FLOW_STEPS_SAMPLE:
        raise ValueError(
            f"training_evaluation used {record['flow_steps']} flow steps; "
            f"evaluation integrates {FLOW_STEPS_SAMPLE}"
        )
    _integer(record["eval_seed"], "eval_seed", -(2**63))
    _integer(record["n_samples"], "n_samples", 1)
    n_worlds = _integer(record["n_worlds"], "n_worlds", 1)
    if len(record["worlds"]) != n_worlds:
        raise ValueError(f"{len(record['worlds'])} recorded worlds for n_worlds={n_worlds}")
    evaluated = metric_aggregation(_HEADS)
    if set(record["metrics"]) != set(record["aggregation"]):
        raise ValueError("training_evaluation metrics and aggregation name different keys")
    unknown = sorted(set(record["metrics"]) - set(evaluated))
    if unknown:
        raise ValueError(f"recorded keys the joint-head evaluator does not produce: {unknown}")
    # A slot key may also be recorded pooled over the slots of every world.
    mismatched = sorted(
        key
        for key, kind in record["aggregation"].items()
        if kind != evaluated[key] and (kind, evaluated[key]) != ("slot_pooled", "slot")
    )
    if mismatched:
        raise ValueError(f"recorded aggregation differs from the evaluator for {mismatched}")
    return record


def _worlds(
    record: Mapping[str, Any], corpora_by_seed: Mapping[int, Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[tuple[int, int]], dict[tuple[int, int], dict[str, int]]]:
    """Corpora in first-use order, evaluation rows, and each world's event geometry."""
    seeds: list[int] = []
    rows = []
    for world in record["worlds"]:
        seed = _integer(world["shard_seed"], "shard_seed", -(2**63))
        if seed not in corpora_by_seed:
            raise ValueError(f"no corpus for recorded shard seed {seed}")
        if seed not in seeds:
            seeds.append(seed)
        rows.append((seeds.index(seed), _integer(world["row"], "row", 0)))
    corpora = [corpora_by_seed[seed] for seed in seeds]
    geometry = {}
    for corpus_index, row in rows:
        corpus = corpora[corpus_index]
        support = np.asarray(corpus["support_mask"])
        if row >= support.shape[0]:
            raise ValueError(f"row {row} is outside shard {seeds[corpus_index]}")
        n_context = int(support[row].sum())
        masks = [
            np.asarray(corpus[key][row]) > 0
            for key in ("treatment_active_mask", "covariate_active_mask", "latent_active_mask")
        ]
        layout = EdgeSlotLayout(*(mask.size for mask in masks))
        geometry[(corpus_index, row)] = {
            "query": support.shape[1] - n_context,
            "context": n_context,
            "channels": int(masks[0].sum()),
            "controls": int(masks[1].sum()),
            "slots": int(layout.active_slot_mask(*masks).sum()),
        }
    return corpora, rows, geometry


def _event_step(
    key: str,
    aggregation: str,
    record: Mapping[str, Any],
    geometry: Mapping[tuple[int, int], Mapping[str, int]],
) -> float | None:
    """The resolution of a key's value (see the module doc); ``None`` for continuous keys.

    A world where the key is defined moves by ``1 / (weight * u)``: its
    ``weight`` slots (1 for a world-level key) of ``u`` each, which are the
    window's weeks for weekly coverage and one for time-average coverage; for
    router graph keys, the draws ``S`` for per-draw frequencies and
    ``shd_mean``, times the node pairs of a two-cycle rate or the active slots
    of ``normalized_shd_mean``, 2 (1 for odd ``S``) for ``shd_median``, 1 for
    ``n_distinct``, and the grid ``S**2`` of the largest co-moment, times the
    slot pairs for their mean. That moves an equal-world mean by
    ``1 / (n_defined * weight * u)`` in the world where it is largest and a
    slot-pooled value by ``1 / (sum(weights) * u)``. Of the pooled edge scores
    of an edge type, ``null_fpr`` moves by ``1 / n_neg``, AUROC by
    ``1 / (2 min(n_pos, n_neg))`` and AUPRC by ``1 / n_pos``.
    """
    parts = key.split("/")
    name = parts[-1]
    if parts[:2] == ["joint_router", "edge"]:
        edge = record["heads"]["joint_router"]["edge"][parts[2]]
        positives, negatives = edge["n_pos"], edge["n"] - edge["n_pos"]
        if name == "null_fpr" and negatives:
            return 1.0 / negatives
        if name == "auroc" and positives and negatives:
            return 1.0 / (2 * min(positives, negatives))
        if name == "auprc" and positives and negatives:
            return 1.0 / positives
        return None
    defined = [
        (entry, geometry[(entry["corpus_index"], entry["row"])])
        for entry in record["per_world"]
        if entry["values"].get(key) is not None
    ]
    n_samples = record["n_samples"]
    if parts[0] in DECOMPOSITION_HEADS:
        if name.startswith("timeavg_"):
            terms = [1 for _ in defined]
        elif name.endswith("_cov80"):
            terms = [world[parts[1]] for _, world in defined]
        else:
            return None
    elif parts[:2] == ["joint_router", "graph"]:
        if name in ("exact_recovery", "cc_cycle_rate", "zz_cycle_rate", "shd_mean"):
            terms = [n_samples for _ in defined]
        elif name in ("cc_two_cycle_rate", "zz_two_cycle_rate"):
            nodes = "channels" if name.startswith("cc") else "controls"
            terms = [n_samples * math.comb(world[nodes], 2) for _, world in defined]
        elif name == "normalized_shd_mean":
            terms = [n_samples * world["slots"] for _, world in defined]
        elif name == "shd_median":
            terms = [2 if n_samples % 2 == 0 else 1 for _ in defined]
        elif name == "n_distinct":
            terms = [1 for _ in defined]
        elif name == "pairwise_comoment_max_abs_err":
            terms = [n_samples**2 for _ in defined]
        elif name == "pairwise_comoment_abs_err":
            terms = [n_samples**2 * math.comb(world["slots"], 2) for _, world in defined]
        else:
            return None
    else:
        return None
    if not defined:
        return None
    weights = [entry["weights"][key] for entry, _ in defined]
    if aggregation == "slot_pooled":
        return 1.0 / (sum(weights) * min(terms))
    return 1.0 / (len(defined) * min(w * u for w, u in zip(weights, terms, strict=True)))


def _replicate_value(key: str, aggregation: str, record: Mapping[str, Any]) -> float | None:
    """A slot-pooled key pools the slots of its defined worlds; any other key is the leg value."""
    if aggregation != "slot_pooled":
        return flatten_evaluation(record)[f"{record['leg']}/{key}"]
    total = weight = 0.0
    for entry in record["per_world"]:
        if entry["values"][key] is not None:
            total += entry["values"][key] * entry["weights"][key]
            weight += entry["weights"][key]
    return total / weight if weight else None


def _compare(
    key: str,
    kind: str,
    recorded: float | None,
    values: Sequence[float | None],
    nonfinite: bool,
    event_step: float | None,
    t_quantile: float,
    family_t_quantile: float,
) -> dict[str, Any]:
    """One key's verdict, with the Student-t quantiles of its 99% and 99.9% intervals."""
    result: dict[str, Any] = {
        "class": kind,
        "recorded": recorded,
        "replicates": list(values),
        "event_step": event_step,
        "mean": None,
        "sd": None,
        "tolerance": None,
        "family_tolerance": None,
    }
    if recorded is None and all(value is None for value in values):
        return {**result, "passing": True, "reason": "undefined_as_recorded"}
    if recorded is None:
        return {**result, "passing": False, "reason": "recorded_undefined"}
    if any(value is None for value in values):
        return {**result, "passing": False, "reason": "replicate_undefined"}
    if nonfinite:
        return {**result, "passing": False, "reason": "replicate_nonfinite"}
    recorded = float(recorded)
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    result.update(recorded=recorded, mean=mean, sd=sd)
    step = event_step or 0.0
    # One event, with relative slack for round-off in the difference.
    event = step * (1.0 + 1e-9)
    if kind == "exact":
        # Draw-independent keys must agree with the recorded value in every replicate.
        passing = bool(np.max(np.abs(array - recorded)) <= max(EXACT_TOLERANCE, event))
        return {
            **result,
            "tolerance": max(EXACT_TOLERANCE, step),
            "passing": passing,
            "reason": None if passing else "exact_mismatch",
        }
    constant = bool(array.min() == array.max())
    spread = 0.0 if constant else sd * math.sqrt(1.0 + 1.0 / len(array))
    # Without replicate spread the interval is its floor, never below EXACT_TOLERANCE.
    lowest = EXACT_TOLERANCE if constant else 0.0
    distance = abs(recorded - (float(array[0]) if constant else mean))
    limit = max(t_quantile * spread, lowest, event)
    if distance <= limit:
        reason = None
    elif distance > GROSS_FACTOR * limit:
        reason = "gross_mismatch"
    elif distance > max(family_t_quantile * spread, lowest, event):
        reason = "outside_family_interval"
    else:
        reason = "outside_replicate_interval"
    return {
        **result,
        "tolerance": max(t_quantile * spread, lowest, step),
        "family_tolerance": max(family_t_quantile * spread, lowest, step),
        "passing": reason is None,
        "reason": reason,
    }


def _compare_recorded(
    recorded: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    geometry: Mapping[tuple[int, int], Mapping[str, int]],
) -> dict[str, Any]:
    """Compare recorded values with replicate evaluation records (see the module doc)."""
    from scipy.stats import binom
    from scipy.stats import t as student_t

    dof = len(records) - 1
    t_quantile = float(student_t.ppf(T_LEVEL, dof))
    family_t_quantile = float(student_t.ppf(FAMILY_T_LEVEL, dof))
    keys = {}
    for key, value in recorded["metrics"].items():
        aggregation = recorded["aggregation"][key]
        keys[key] = _compare(
            key,
            key_class(key),
            value,
            [_replicate_value(key, aggregation, record) for record in records],
            any(record["undefined"].get(key, "").startswith("nonfinite_in_") for record in records),
            _event_step(key, aggregation, records[0], geometry),
            t_quantile,
            family_t_quantile,
        )
        keys[key]["aggregation"] = aggregation
    exact_failures = [
        key for key, entry in keys.items() if entry["class"] == "exact" and not entry["passing"]
    ]
    stochastic = {key: entry for key, entry in keys.items() if entry["class"] == "stochastic"}
    stochastic_failures = [key for key, entry in stochastic.items() if not entry["passing"]]
    # A key without a tolerance failed on definedness or a non-finite value before comparing.
    must_pass_failures = [
        key
        for key in stochastic_failures
        if key == HEADLINE or stochastic[key]["tolerance"] is None
    ]
    gross_failures = [
        key for key in stochastic_failures if stochastic[key]["reason"] == "gross_mismatch"
    ]
    beyond: dict[str, list[str]] = {}
    for key in stochastic_failures:
        if stochastic[key]["reason"] in ("outside_family_interval", "gross_mismatch"):
            beyond.setdefault(_family(key), []).append(key)
    family_failures = {family: members for family, members in beyond.items() if len(members) > 1}
    # Each family counts once against the allowance, unless an earlier rule failed it.
    failed = {*family_failures, *(_family(key) for key in gross_failures)}
    failing_families = sorted(
        {_family(key) for key in stochastic_failures if key not in must_pass_failures} - failed
    )
    families = {_family(key) for key, entry in stochastic.items() if entry["tolerance"] is not None}
    allowance = int(binom.ppf(ALLOWANCE_LEVEL, len(families), FAILURE_RATE)) if families else 0
    compared = any(
        entry["class"] != "informational" and entry["tolerance"] is not None
        for entry in keys.values()
    )
    if not compared:
        reason = "no_compared_keys"
    elif exact_failures:
        reason = "exact_failures"
    elif must_pass_failures:
        reason = "must_pass_failures"
    elif gross_failures:
        reason = "gross_failures"
    elif family_failures:
        reason = "family_failures"
    elif len(failing_families) > allowance:
        reason = "allowance_exceeded"
    else:
        reason = None
    return {
        "passed": reason is None,
        "reason": reason,
        "replicates": len(records),
        "seeds": [record["seed"] for record in records],
        "n_worlds": recorded["n_worlds"],
        "n_samples": recorded["n_samples"],
        "eval_seed": recorded["eval_seed"],
        "n_families": len(families),
        "allowed_family_failures": allowance,
        "exact_failures": exact_failures,
        "stochastic_failures": stochastic_failures,
        "must_pass_failures": must_pass_failures,
        "gross_failures": gross_failures,
        "family_failures": family_failures,
        "failing_families": failing_families,
        "informational": [key for key, entry in keys.items() if entry["class"] == "informational"],
        "keys": keys,
    }


def check_recorded_evaluation(
    model: MFM,
    corpora_by_seed: Mapping[int, Mapping[str, Any]],
    *,
    replicates: int = MIN_REPLICATES,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Re-evaluate the recorded held-out worlds and compare with the recorded values.

    ``corpora_by_seed`` maps every recorded ``shard_seed`` to its loaded corpus.
    The recorded worlds are evaluated ``replicates`` (at least
    :data:`MIN_REPLICATES`) times with draw seeds ``eval_seed + 1 + r``, on the
    joint head only and with per-world records; ``model`` is moved to
    ``device``. A key recorded as ``slot_pooled`` pools the slots of the worlds
    where it is defined, weighting each world by its slot count; every other key
    uses the evaluated leg value. Raises ``ValueError`` when the provenance has
    no valid recorded evaluation, its flow steps differ from the sampler's, a
    shard seed has no corpus, or the world list disagrees with ``n_worlds``.

    Returns ``{"passed", "reason", "replicates", "seeds", "n_worlds",
    "n_samples", "eval_seed", "n_families", "allowed_family_failures",
    "exact_failures", "stochastic_failures", "must_pass_failures",
    "gross_failures", "family_failures", "failing_families", "informational",
    "keys"}``. ``reason`` is ``None`` when the check passes and otherwise names
    the first rule that failed, in the module doc's order: ``"no_compared_keys"``,
    ``"exact_failures"``, ``"must_pass_failures"``, ``"gross_failures"``,
    ``"family_failures"`` or ``"allowance_exceeded"``. ``stochastic_failures``
    lists every stochastic key outside its tolerance; ``family_failures`` maps
    each family with two keys outside their 99.9% intervals to those keys;
    ``failing_families`` are the families counted against the allowance
    ``allowed_family_failures``, those with a failing key that no earlier rule
    failed, and ``n_families`` is the allowance's ``k``. ``keys[key]`` holds the
    key's class, aggregation, recorded and replicate values, their mean and
    standard deviation, its ``tolerance`` (99%) and ``family_tolerance``
    (99.9%), its resolution ``event_step`` and its verdict. A failing key's
    ``reason`` is ``"exact_mismatch"``, how far a stochastic key lies
    (``"outside_replicate_interval"``, ``"outside_family_interval"`` or, beyond
    :data:`GROSS_FACTOR` tolerances, ``"gross_mismatch"``), or why it is
    undefined or non-finite.
    """
    recorded = _training_evaluation(model)
    count = _integer(replicates, "replicates", MIN_REPLICATES)
    corpora, rows, geometry = _worlds(recorded, corpora_by_seed)
    model.to(device)
    records = [
        evaluate_rows(
            model,
            corpora,
            rows,
            n_samples=recorded["n_samples"],
            seed=recorded["eval_seed"] + 1 + replicate,
            device=device,
            leg=_LEG,
            heads=_HEADS,
            per_world=True,
        )
        for replicate in range(count)
    ]
    return _compare_recorded(recorded, records, geometry)
