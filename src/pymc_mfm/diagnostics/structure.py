"""Explicitly factorized cardinality-constrained edge-set diagnostics.

These readouts are approximations from marginals, not the model's empirical
complete-graph posterior and never a conditioning/routing implementation.
"""

from __future__ import annotations

import itertools
from numbers import Integral

import numpy as np


def cardinality_decode(marginals, cardinality, beta=1.0):
    """Enumerate fixed-cardinality sets with smoothed log probabilities.

    The clipping epsilon is ``1/(2*A*256)`` for A active slots. MAP must use
    beta=1 probabilities; tempering changes set probabilities, not the
    selection rule. k=0 is the single empty hypothesis, not an unavailable
    posterior.
    """
    probabilities = np.asarray(marginals, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("marginals must be a nonempty vector of active edge probabilities")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("marginals must be finite probabilities in [0,1]")
    if isinstance(cardinality, bool) or not isinstance(cardinality, Integral):
        raise ValueError("cardinality must be an integer")
    if not 0 <= cardinality <= probabilities.size:
        raise ValueError("cardinality must lie between zero and the active slot count")
    if isinstance(beta, bool) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    subsets = np.asarray(
        list(itertools.combinations(range(len(probabilities)), cardinality)), dtype=int
    )
    if cardinality == 0:
        subsets = np.empty((1, 0), dtype=int)
    epsilon = 1 / (2 * len(probabilities) * 256)
    p = np.clip(probabilities, epsilon, 1 - epsilon)
    logits = np.log(p) - np.log1p(-p)
    weights = np.log1p(-p).sum() + logits[subsets].sum(axis=1)
    weights *= beta
    return subsets, weights - np.logaddexp.reduce(weights)


def randomized_set_pit(log_probabilities, true_index, *, seed=20260906, rng=None):
    """Randomized mass-rank PIT with near-equal probability masses tied."""
    logp = np.asarray(log_probabilities, dtype=np.float64)
    if logp.ndim != 1 or logp.size == 0 or not np.isfinite(logp).all():
        raise ValueError("log probabilities must be finite and nonempty")
    if (
        isinstance(true_index, bool)
        or not isinstance(true_index, Integral)
        or not 0 <= true_index < len(logp)
    ):
        raise ValueError("true_index must index a represented set")
    probabilities = np.exp(logp)
    if not np.isclose(probabilities.sum(), 1, rtol=1e-10, atol=1e-12):
        raise ValueError("set probabilities must sum to one")
    p_true = probabilities[true_index]
    tied = np.isclose(probabilities, p_true, rtol=1e-9, atol=0)
    below = probabilities[(probabilities < p_true) & ~tied].sum()
    rng = np.random.default_rng(seed) if rng is None else rng
    return float(min(below + rng.random() * probabilities[tied].sum(), 1))


def factorized_set_diagnostics(
    marginals,
    cardinality,
    *,
    truth=None,
    edge_names=None,
    source,
    beta=1.0,
    seed=20260906,
):
    """Score a labeled marginal-source approximation under an explicit design k.

    ``source`` must describe the source (e.g. joint router or auxiliary head).
    The supplied cardinality is a design hypothesis, never estimated from truth.
    Undefined truth under that hypothesis is recorded rather than assigned mass.
    """
    if not isinstance(source, str) or not source:
        raise ValueError("an explicit marginal source label is required")
    p = np.asarray(marginals, dtype=np.float64)
    subsets, untempered = cardinality_decode(p, cardinality)
    _, logp = cardinality_decode(p, cardinality, beta)
    if edge_names is None:
        edge_names = [str(index) for index in range(len(p))]
    if len(edge_names) != len(p) or len(set(edge_names)) != len(p):
        raise ValueError("edge_names must uniquely label all active slots")
    mode = int(np.argmax(untempered))
    top = np.sort(untempered)[-2:]
    metrics = {
        "map_probability": float(np.exp(logp[mode])),
        "chance_hit": float(1 / len(subsets)),
        "map_hit": None,
        "true_set_log_score": None,
        "randomized_set_pit": None,
    }
    statuses = {
        name: {
            "status": "ok" if value is not None else "unavailable",
            "reason": None if value is not None else "graph truth not supplied",
        }
        for name, value in metrics.items()
    }
    true_index = None
    if truth is not None:
        truth = np.asarray(truth)
        if truth.shape != p.shape or not np.isin(truth, (0, 1)).all():
            raise ValueError("truth must be an aligned binary edge vector")
        indices = np.flatnonzero(truth)
        if len(indices) == cardinality:
            true_index = int(np.flatnonzero(np.all(subsets == indices, axis=1))[0])
            metrics.update(
                map_hit=float(mode == true_index),
                true_set_log_score=float(-logp[true_index]),
                randomized_set_pit=randomized_set_pit(logp, true_index, seed=seed),
            )
            for name in ("map_hit", "true_set_log_score", "randomized_set_pit"):
                statuses[name] = {
                    "status": "screening_only",
                    "reason": "one-world design-conditioned approximation",
                }
        else:
            for name in ("map_hit", "true_set_log_score", "randomized_set_pit"):
                statuses[name] = {
                    "status": "unavailable",
                    "reason": "truth has a different design cardinality",
                }
    return {
        "format_version": 1,
        "source": source,
        "kind": "factorized_cardinality_approximation",
        "cardinality": int(cardinality),
        "beta": float(beta),
        "seed": int(seed),
        "smoothing": {"name": "half-draw-256", "epsilon": float(1 / (2 * len(p) * 256))},
        "edge_names": list(edge_names),
        "marginals": p.tolist(),
        "subsets": subsets.tolist(),
        "log_probabilities": logp.tolist(),
        "map_index": mode,
        "true_index": true_index,
        "map_ambiguous": bool(len(top) == 2 and abs(top[-1] - top[-2]) <= 1e-12),
        "map_tie_rule": "first argmax of untempered log probabilities; summation ulps preserved",
        "metrics": metrics,
        "statuses": statuses,
        "qualification": (
            "Factorized marginal approximation conditional on supplied k, NOT "
            "an empirical complete-graph posterior."
        ),
    }


def set_reliability(confidence, correct, *, independent_worlds=False, bins=10):
    """Pooled set-selection ECE with explicit across-world qualification."""
    from pymc_mfm.training.metrics import ece

    probabilities = np.asarray(confidence, dtype=np.float64)
    outcomes = np.asarray(correct)
    if probabilities.ndim != 1 or probabilities.size == 0 or outcomes.shape != probabilities.shape:
        raise ValueError("confidence and correct must be aligned nonempty vectors")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("confidence must contain probabilities")
    if not np.isin(outcomes, (0, 1)).all():
        raise ValueError("correct must contain binary outcomes")
    return {
        "ece": ece(probabilities, outcomes, bins=bins),
        "n": len(probabilities),
        "status": "independent_worlds_declared" if independent_worlds else "screening_only",
        "qualification": "Binned selection reliability is not proof of posterior calibration.",
    }


def graph_support_diagnostics(graphs, edge_families, edge_mask, *, truth=None, support=None):
    """Audit saved complete graphs against an explicitly declared prior support.

    ``support`` is ``{cardinality: {family: [minimum, maximum]},
    forbidden_families: [family, ...]}``. Budget labels use model edge names
    (``zb``/``db``); callers must explicitly map external generator names.
    Truth is used for recovery only, never to define the permitted prior.
    Modal recovery uses the empirical COMPLETE-graph mode, lexicographic ties.
    """
    from collections.abc import Mapping

    draws = np.asarray(graphs)
    mask = np.asarray(edge_mask)
    families = np.asarray(edge_families)
    if draws.ndim != 2 or len(draws) < 1 or not np.isin(draws, (0, 1)).all():
        raise ValueError("graphs must be nonempty binary [S,L] draws")
    if mask.shape != (draws.shape[1],) or not np.isin(mask, (0, 1)).all():
        raise ValueError("edge_mask must be an aligned binary vector")
    if families.shape != mask.shape or not all(
        isinstance(item, str) and item for item in edge_families
    ):
        raise ValueError("edge_families must align with graph slots")
    mask = mask.astype(bool)
    target = None if truth is None else np.asarray(truth)
    if target is not None and (target.shape != mask.shape or not np.isin(target, (0, 1)).all()):
        raise ValueError("truth must be an aligned binary vector")
    if target is not None and np.any(target[~mask]):
        raise ValueError("graph truth contains edges touching padded nodes")
    if mask.any():
        unique, counts = np.unique(draws[:, mask], axis=0, return_counts=True)
        mode = unique[np.flatnonzero(counts == counts.max())[0]]
    else:
        mode = np.empty(0, dtype=int)
    full_mode = np.zeros(draws.shape[1], dtype=int)
    full_mode[mask] = mode
    metrics = {}
    statuses = {}

    def add(key, value, reason=None):
        metrics[key] = None if value is None else float(value)
        statuses[key] = {"status": "ok" if value is not None else "unavailable", "reason": reason}

    add(
        "structure_exact_hit",
        None if target is None else np.array_equal(full_mode[mask], target[mask]),
        None if target is not None else "graph truth not supplied",
    )
    arrays = {
        "mode_graph": full_mode.tolist(),
        "edge_count_histograms": {},
        "support": support,
        "mode_tie_rule": "lexicographic_first_complete_graph",
    }
    counts_by_family = {}
    for family in sorted(set(edge_families)):
        keep = (families == family) & mask
        counts = draws[:, keep].sum(axis=1)
        counts_by_family[family] = counts
        histogram = np.bincount(counts.astype(int), minlength=int(keep.sum()) + 1)
        arrays["edge_count_histograms"][family] = histogram.tolist()
        prefix = f"family/{family}"
        add(f"{prefix}/mean_count", counts.mean())
        add(f"{prefix}/modal_count", full_mode[keep].sum())
        if target is None:
            for key in (
                "true_count",
                "count_mae",
                "count_recovery",
                "modal_count_hit",
                "modal_set_exact_hit",
            ):
                add(f"{prefix}/{key}", None, "graph truth not supplied")
        else:
            true_count = target[keep].sum()
            add(f"{prefix}/true_count", true_count)
            add(f"{prefix}/count_mae", np.abs(counts - true_count).mean())
            add(f"{prefix}/count_recovery", (counts == true_count).mean())
            add(f"{prefix}/modal_count_hit", full_mode[keep].sum() == true_count)
            add(f"{prefix}/modal_set_exact_hit", np.array_equal(full_mode[keep], target[keep]))
    padded = np.any(draws[:, ~mask], axis=1)
    add("padded_edge_mass", padded.mean())
    if support is None:
        for key in (
            "off_prior_mass",
            "forbidden_family_mass",
            "wrong_cy_cardinality_mass",
            "wrong_zb_cardinality_mass",
        ):
            add(key, None, "explicit prior support contract not supplied")
    else:
        if not isinstance(support, Mapping) or set(support) != {
            "cardinality",
            "forbidden_families",
        }:
            raise ValueError("graph support requires cardinality and forbidden_families")
        budgets = support["cardinality"]
        forbidden = support["forbidden_families"]
        if not isinstance(budgets, Mapping) or not isinstance(forbidden, (list, tuple)):
            raise ValueError("invalid graph support cardinality or forbidden_families")
        known = {"cy", "zb", "db", "dc", "dz", "zc", "cc", "zz"} | set(edge_families)
        if not set(budgets) <= known or not set(forbidden) <= known:
            raise ValueError("graph support names an unknown edge family")
        invalid = padded.copy()
        forbidden_draw = np.zeros(len(draws), dtype=bool)
        for family, interval in budgets.items():
            if (
                not isinstance(interval, (list, tuple))
                or len(interval) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, Integral) for value in interval
                )
                or not 0 <= interval[0] <= interval[1]
            ):
                raise ValueError(
                    "graph support cardinalities must be nonnegative integer [min,max] intervals"
                )
            family_counts = counts_by_family.get(family, np.zeros(len(draws), dtype=int))
            wrong = (family_counts < interval[0]) | (family_counts > interval[1])
            invalid |= wrong
            add(f"wrong_{family}_cardinality_mass", wrong.mean())
        for family in forbidden:
            forbidden_draw |= counts_by_family.get(family, np.zeros(len(draws), dtype=int)) != 0
        add("forbidden_family_mass", forbidden_draw.mean())
        add("off_prior_mass", (invalid | forbidden_draw).mean())
        for family in ("cy", "zb"):
            if family not in budgets:
                add(
                    f"wrong_{family}_cardinality_mass",
                    None,
                    "prior contract does not constrain this family",
                )
    return {
        "metrics": metrics,
        "statuses": statuses,
        "arrays": arrays,
        "qualification": (
            "Support checks describe supplied prior constraints; truth is not a prior."
        ),
    }
