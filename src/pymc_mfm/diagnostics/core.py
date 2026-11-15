"""Pure computations for conditional decomposition reports; no renderers or inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import numpy as np

from pymc_mfm.inference._time import _time_to_json

from .calibration import SCREENING_QUALIFICATION, calibration_summary
from .uncertainty import paired_graph_uncertainty

if TYPE_CHECKING:
    from pymc_mfm.inference.result import DecompositionResult

PANEL_NAMES = (
    "target_recovery",
    "channel_contributions",
    "channel_totals",
    "components",
    "indirect_sources",
    "controls",
    "recovery_scatter",
    "attribution",
    "window_totals",
    "cumulative",
    "temporal",
    "residual_true_size",
    "residual_predicted_size",
    "strata",
    "structure_heatmap",
    "dag_recovery",
    "dag_posterior",
    "graph_cardinality",
    "graph_statistics",
    "cycles",
    "reliability",
    "support",
    "input_support",
    "zero_atoms",
    "graph_uncertainty",
    "calibration_pit",
    "calibration_ecdf",
    "coverage",
    "training",
    "evaluation",
    "reference_comparison",
    "set_posterior",
    "posterior_sbc",
    "c2st",
)
FAMILIES = ("baseline", "channel", "control", "latent", "indirect")
#: Effects scored live/dead, as ``(effect, component family, direct edge family)``:
#: media is live through a ``cy`` edge, a control through a ``zb`` edge.
ATTRIBUTION_EFFECTS = (("media", "channel", "cy"), ("control", "control", "zb"))
#: Trajectory families scored for temporal coherence; ``latent_total`` is the
#: sum of every active component (the latent outcome).
TEMPORAL_FAMILIES = ("baseline", "channel", "control", "latent_total", "indirect")
#: Per-family ``graph_support`` metrics that count edges; the others are masses or rates.
_SUPPORT_COUNTS = ("mean_count", "modal_count", "true_count", "count_mae")
#: Panels drawn from the per-effect stratified rows (:mod:`.strata`).
STRATA_PANELS = ("residual_true_size", "residual_predicted_size", "strata")


def json_safe(value):
    """Convert scientific containers, refusing nonfinite values and unknown objects."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            return _time_to_json(value)
        return json_safe(value.tolist())
    if isinstance(value, np.datetime64):
        return _time_to_json(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("evidence object keys must be strings")
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("evidence cannot contain nonfinite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _evidence(value, shape, name, *, binary=False):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    if binary and not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return array


def _series(samples, truth=None):
    low, high = np.quantile(samples, [0.1, 0.9], axis=0)
    return {
        "mean": samples.mean(axis=0).tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "truth": None if truth is None else truth.tolist(),
    }


#: Metric names derived from graph draws; a result without graphs (hosted) cannot score them.
_GRAPH_METRIC_PREFIXES = ("uncertainty/", "graph/", "graph_support/", "edge/", "set_approximation/")
_GRAPH_METRIC_NAMES = ("support/active_edges", "support/padded_edges", "support/edge_padding_mass")
#: ``uncertainty/`` metrics that come from the total variance and covariance only, so they
#: need no graphs and stay computed.
_GRAPH_FREE_UNCERTAINTY = (
    "uncertainty/total_variance_mean",
    "uncertainty/latent_total/total_variance_mean",
)
_GRAPH_FREE_UNCERTAINTY_PREFIX = "uncertainty/time_mean/"
_GRAPH_PANELS = (
    "structure_heatmap",
    "dag_recovery",
    "dag_posterior",
    "graph_cardinality",
    "graph_statistics",
    "cycles",
    "reliability",
    "support",
    "zero_atoms",
    "graph_uncertainty",
    "set_posterior",
)
#: Arrays built from graph draws. ``None`` is how the catalog says an array is unavailable.
_GRAPH_ARRAYS = (
    "graphs",
    "graph_support",
    "reliability",
    "set_posterior",
    "uncertainty",
    "total_uncertainty",
    "aggregate_uncertainty",
)
_NO_GRAPHS = "the result has no graphs (the hosted draws file carries none)"


def _graph_metric(name):
    if name in _GRAPH_FREE_UNCERTAINTY or name.startswith(_GRAPH_FREE_UNCERTAINTY_PREFIX):
        return False
    return name.startswith(_GRAPH_METRIC_PREFIXES) or name in _GRAPH_METRIC_NAMES


def _mark_graphs_unavailable(metrics, panels, arrays):
    """Report every graph-derived metric, panel and array as unavailable, never as a value.

    Placeholder graphs say nothing about graph uncertainty: zero between-graph variance would
    be a claim. Metrics keep no value, panels get status ``unavailable`` and arrays are ``None``.
    Quantities from the draws alone (total variance, component covariance) stay computed.
    """
    for name in metrics.values:
        if _graph_metric(name):
            metrics.values[name] = None
            metrics.statuses[f"metric/{name}"] = {"status": "unavailable", "reason": _NO_GRAPHS}
    for name in _GRAPH_PANELS:
        panels[name] = {"status": "unavailable", "reason": _NO_GRAPHS}
    for name in _GRAPH_ARRAYS:
        arrays[name] = None


class _Metrics:
    def __init__(self):
        self.values = {}
        self.statuses = {}
        self.units = {}

    def add(self, name, value, *, unit="dimensionless", reason=None, status=None):
        if name in self.values:
            raise ValueError(f"duplicate diagnostic metric {name}")
        if value is not None:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"nonfinite diagnostic metric {name}")
        self.values[name] = value
        self.units[name] = unit
        self.statuses[f"metric/{name}"] = {
            "status": status or ("ok" if value is not None else "unavailable"),
            "reason": reason,
        }


def _score(target, samples, truth, metrics, units, scale):
    from pymc_mfm.training import metrics as scoring

    # Compute in context units, then restore only dimensional readouts. Ratios
    # and squared errors must not depend on the native measurement magnitude.
    ruler = 1.0 if scale is None else scale
    samples = samples / ruler
    low, high = np.quantile(samples, [0.1, 0.9], axis=0)
    metrics.add(f"{target}/mean", samples.mean() * ruler, unit=units)
    metrics.add(f"{target}/interval80_width", np.mean(high - low) * ruler, unit=units)
    keys = (
        "mae",
        "bias",
        "crps",
        "cov80",
        "relative_mae",
        "scaled_mae",
        "energy_score",
        "var_ratio",
        "r2",
    )
    if truth is None:
        for key in keys:
            metrics.add(f"{target}/{key}", None, reason="simulation truth not supplied")
        return
    truth = truth / ruler
    error = samples.mean(axis=0) - truth
    mae = float(np.mean(np.abs(error)))
    metrics.add(f"{target}/mae", mae * ruler, unit=units)
    metrics.add(f"{target}/bias", error.mean() * ruler, unit=units)
    metrics.add(f"{target}/crps", scoring.empirical_crps(samples, truth) * ruler, unit=units)
    metrics.add(f"{target}/cov80", scoring.coverage(samples, truth))
    denominator = float(np.abs(truth).mean())
    metrics.add(
        f"{target}/relative_mae",
        mae / denominator if denominator else None,
        reason=None if denominator else "zero-magnitude truth; relative error undefined",
    )
    metrics.add(
        f"{target}/scaled_mae",
        mae if scale is not None else None,
        unit="sales_scale",
        reason=None if scale is not None else "sales_scale provenance absent",
    )
    energy = scoring.energy_score(samples, truth)
    metrics.add(
        f"{target}/energy_score",
        None if energy is None else energy * ruler,
        unit=units,
        reason="requires at least two draws" if len(samples) < 2 else None,
    )
    truth_variance = float(np.var(truth))
    metrics.add(
        f"{target}/var_ratio",
        np.var(samples.mean(axis=0)) / truth_variance if truth_variance else None,
        reason=None if truth_variance else "constant truth; trajectory variance ratio undefined",
    )
    r2, reason = scoring.r2(samples.mean(axis=0), truth)
    metrics.add(
        f"{target}/r2",
        r2,
        reason="constant truth; R-squared undefined" if reason == "constant_series" else reason,
    )


def _add_scores(metrics, world, keys, *, unit="dimensionless", unlabeled=None):
    """Add one world's :mod:`pymc_mfm.training.metrics` scores of ``keys``.

    Undefined keys keep the scorer's reason; ``unlabeled`` states why live/dead
    labels are absent in place of the scorer's ``missing_graph_truth``.
    """
    for key in keys:
        value = world.values.get(key)
        reason = None if value is not None else world.undefined[key]
        if reason == "missing_graph_truth" and unlabeled is not None:
            reason = unlabeled
        metrics.add(key, value, unit=unit, reason=reason)


def _effect_labels(names, indices, edge, graph_truth, edge_active, endpoints, edge_families):
    """Graph-truth labels (1 live, 0 dead) of present effects, read from their direct edges.

    Returns ``(labels, None)``, or ``(None, reason)`` when graph truth, the edge
    metadata or an effect's unique active ``edge`` slot is missing. Zero-valued
    component truth is never read as a causal label.
    """
    if graph_truth is None:
        return None, "graph truth not supplied"
    if endpoints is None or edge_families is None:
        return None, "edge endpoint/family metadata absent"
    labels = []
    for index in indices:
        slots = [
            slot
            for slot, (family, pair) in enumerate(zip(edge_families, endpoints, strict=True))
            if family == edge and pair[0] == names[index] and edge_active[slot]
        ]
        if len(slots) != 1:
            return None, f"{names[index]} has no unique active {edge} edge"
        labels.append(int(graph_truth[slots[0]]))
    return np.asarray(labels, dtype=int), None


def _cyclic_graphs(graphs, endpoints):
    """Whether each complete graph over the ``endpoints`` slots contains a directed cycle."""
    from pymc_mfm.training import metrics as scoring

    flags = []
    for adjacency in scoring.adjacency_draws(graphs, endpoints):
        rate, _ = scoring.cycle_rate(adjacency[None])
        flags.append(bool(rate))  # fewer than two nodes cannot form a cycle
    return flags


def _graph_scores(draws, truth, families, endpoints, metrics):
    """The evaluator's whole-graph statistics (``joint_router/graph``) as ``graph/<key>``.

    :func:`pymc_mfm.training.metrics.graph_scores` of the active-slot draws
    ``(S, L)`` in the evaluator's units, with the ``cc``/``zz`` cycle blocks
    built like the evaluator's from the slots' edge ``families`` and
    ``(source, target)`` ``endpoints``. Without graph truth (``truth=None``)
    only the truth-free keys (entropy, distinct graphs, edge co-moments and
    cycle rates) have values; the cycle rates also need the edge metadata.
    """
    from pymc_mfm.training import metrics as scoring
    from pymc_mfm.training._metric_tables import GRAPH_METRICS, METRIC_UNITS

    if families is None or endpoints is None:
        blocks = {}
    else:  # imported here: without edge metadata (a hosted result) this needs no PyTorch
        from pymc_mfm.model.graph import SQUARE_EDGE_TYPES

        blocks = scoring.adjacency_blocks(draws, families, endpoints, SQUARE_EDGE_TYPES)
    world = scoring.graph_scores(draws, truth, blocks, prefix="graph/")
    for key in GRAPH_METRICS:
        name = f"graph/{key}"
        if name not in world.values and name not in world.undefined:  # cycle rates
            world.skip(name, "edge endpoint/family metadata absent")
        _add_scores(
            metrics, world, [name], unit=METRIC_UNITS[key], unlabeled="graph truth not supplied"
        )


def _numeric_leaves(record, prefix=""):
    for key, value in record.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _numeric_leaves(value, name)
        elif value is None or isinstance(value, (int, float, bool, np.number)):
            yield name, value


def _nonfinite(value):
    """A NaN or infinite float; integers of any size and booleans are finite."""
    return isinstance(value, (float, np.inexact)) and not np.isfinite(value)


def record_series(records: Sequence[Mapping[str, Any]], kind: str = "training") -> dict:
    """Scalar trajectories from training or evaluation records.

    Numeric, boolean and null leaves of nested mappings become series named
    by their ``/``-joined path; strings and lists are skipped. Either every
    record has a strictly increasing ``step`` or none has one, in which case
    the record index is the coordinate. Null values stay recorded gaps.
    """
    records = list(records)
    lines = {}
    steps = []
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError(f"{kind} records must be mappings")
    has_steps = ["step" in record for record in records]
    if any(has_steps) and not all(has_steps):
        raise ValueError(f"{kind} records cannot mix training steps and unnamed record indices")
    coordinate = "step" if has_steps and all(has_steps) else "record_index"
    for index, record in enumerate(records):
        step = record.get("step", index)
        if isinstance(step, bool) or not isinstance(step, (int, float)) or _nonfinite(step):
            raise ValueError(f"{kind} record step must be finite")
        if steps and step <= steps[-1]:
            raise ValueError(f"{kind} records must have strictly increasing steps")
        steps.append(step)
        for name, value in _numeric_leaves(record):
            if name == "step":
                continue
            if _nonfinite(value):
                raise ValueError(f"{kind} record {step}/{name} is nonfinite")
            entry = (
                {"step": step, "value": value}
                if coordinate == "step"
                else {
                    "step": None,
                    "record_index": index,
                    "value": value,
                }
            )
            lines.setdefault(name, []).append(entry)
    return {
        "steps": steps if coordinate == "step" else None,
        "coordinate": coordinate,
        "series": lines,
        "records": json_safe(records),
    }


_INPUT_STATISTICS = ("n", "mean", "std", "min", "max")


def _input_support(result, active):
    """Raw spend/control location and scale per active entity, as recorded at inference.

    ``decompose`` and the evaluator's result capture record these statistics
    under the same keys, only for native-unit observations; encoded inputs
    have no raw scale, so ``None`` marks the panel unavailable.
    """
    recorded = result.provenance.get("input_support")
    if not isinstance(recorded, Mapping) or not any(
        f"ctx_{key}_raw" in recorded for key in ("spend", "controls")
    ):
        return None
    families = np.asarray(result.component_families)
    names = np.asarray(result.component_names)
    rows = []
    for family, key in (("channel", "spend"), ("control", "controls")):
        entities = names[families == family]
        windows = {}
        for window in ("ctx", "qry"):
            statistics = recorded.get(f"{window}_{key}_raw")
            if statistics is None:
                continue
            if not isinstance(statistics, Mapping) or any(
                name not in statistics for name in _INPUT_STATISTICS
            ):
                raise ValueError(f"input_support {window}_{key}_raw lacks {_INPUT_STATISTICS}")
            if any(
                np.shape(statistics[name]) != (len(entities),) for name in _INPUT_STATISTICS[1:]
            ):
                raise ValueError(f"input_support {window}_{key}_raw must align with {family} names")
            windows[window] = statistics
        if "ctx" not in windows:
            continue
        for index, (name, keep) in enumerate(zip(entities, active[families == family])):
            if not keep:
                continue
            row = {"name": str(name)}
            for window, label in (("ctx", "context"), ("qry", "query")):
                statistics = windows.get(window)
                row[label] = (
                    None
                    if statistics is None
                    else {
                        "count": statistics["n"],
                        **{stat: statistics[stat][index] for stat in _INPUT_STATISTICS[1:]},
                    }
                )
            rows.append(row)
    return {
        "inputs": rows,
        "source": "input_support statistics recorded at inference (native input units)",
        "qualification": (
            "Raw input location and scale recorded at inference; no generator draws, "
            "inferred truth or out-of-distribution pass/fail claim."
        ),
    }


def _attribution(names, families, active, samples, truth, scale, effect_labels, metrics):
    """Live/dead attribution of each effect family, in sales-scale units like the evaluator.

    Metrics ``attribution/{live,dead}_{effect}/<key>`` come from
    :func:`pymc_mfm.training.metrics.attribution_scores`; the array holds every
    effect's own scores, whose means over effects are those world values.
    Returns ``(array, None)``, or ``(None, reason)`` when no effect is scored.
    """
    from pymc_mfm.training import metrics as scoring

    effects = {}
    for effect, family, edge in ATTRIBUTION_EFFECTS:
        indices = np.flatnonzero((families == family) & active)
        if truth is None:
            live, reason = None, "simulation truth not supplied"
        elif scale is None:
            live, reason = None, "sales_scale provenance absent"
        elif not indices.size:
            live, reason = None, f"no active {family} components"
        else:
            live, reason = effect_labels[edge]
        live_prefix, dead_prefix = f"attribution/live_{effect}/", f"attribution/dead_{effect}/"
        keys = [
            *(live_prefix + key for key in scoring.LIVE_ATTRIBUTION_KEYS),
            *(dead_prefix + key for key in scoring.DEAD_ATTRIBUTION_KEYS),
        ]
        entries = []
        if reason is not None:
            for key in keys:
                metrics.add(key, None, unit="sales_scale", reason=reason)
        else:
            draws, target = samples[..., indices] / scale, truth[:, indices] / scale
            world = scoring.attribution_scores(
                draws, target, live, live_prefix=live_prefix, dead_prefix=dead_prefix
            )
            _add_scores(metrics, world, keys, unit="sales_scale")
            for position, (index, label) in enumerate(zip(indices, live, strict=True)):
                own = draws[..., [position]], target[:, [position]]
                stats = scoring.attribution_stats(*own) if label else scoring.leakage_stats(*own)
                entries.append(
                    {"name": names[index], "index": int(index), "live": bool(label), **stats}
                )
        effects[effect] = {"edge": edge, "reason": reason, "effects": entries}
    if not any(record["effects"] for record in effects.values()):
        return None, "; ".join(dict.fromkeys(record["reason"] for record in effects.values()))
    return {"unit": "sales_scale", "sales_scale": scale, "effects": effects}, None


def _window_totals(names, channels, samples, truth, labels, scale, metrics):
    """Window-level channel coverage, joint-live coverage and running-total energy score.

    ``labels`` is the ``(live, reason)`` pair of the active channels' ``cy`` labels.
    Pointwise scores keep their ``timeavg/channel/`` names; the simultaneous
    live-family event is ``window_totals/simultaneous_cov80``, dimensionless.
    Returns the window-total and cumulative arrays, or ``(None, None, reason)``.
    """
    from pymc_mfm.training import metrics as scoring

    coverage_keys = [f"timeavg/channel/{key}" for key in ("cov80", "cov80_live", "cov80_dead")]
    simultaneous_key = "window_totals/simultaneous_cov80"
    energy_key = "cumulative/channel/energy_score"
    live, unlabeled = labels
    if truth is None or not channels.size:
        reason = (
            "simulation truth not supplied" if truth is None else "no active channel components"
        )
        for key in coverage_keys:
            metrics.add(key, None, reason=reason)
        simultaneous_reason = (
            reason
            if truth is None
            else unlabeled
            if live is None
            else "fewer_than_two_live_effects"
        )
        metrics.add(simultaneous_key, None, reason=simultaneous_reason)
        metrics.add(energy_key, None, unit="sales_scale", reason=reason)
        return None, None, reason
    draws, target = samples[..., channels], truth[:, channels]
    world = scoring.time_average_scores(draws, target, live, prefix="timeavg/channel/")
    _add_scores(metrics, world, coverage_keys, unlabeled=unlabeled)
    score_key = "timeavg/channel/simcov80_live"
    reason = world.undefined.get(score_key)
    metrics.add(
        simultaneous_key,
        world.values.get(score_key),
        reason=unlabeled if reason == "missing_graph_truth" else reason,
    )
    if scale is None:
        metrics.add(energy_key, None, unit="sales_scale", reason="sales_scale provenance absent")
    else:
        value, reason = scoring.cumulative_energy_score(draws / scale, target / scale)
        metrics.add(energy_key, value, unit="sales_scale", reason=reason)
    live = [None] * channels.size if live is None else [bool(label) for label in live]
    totals, running = [], []
    for position, (index, label) in enumerate(zip(channels, live, strict=True)):
        entry = {"name": names[index], "index": int(index), "live": label}
        covered = scoring.time_average_coverage(draws[..., [position]], target[:, [position]])
        totals.append({**entry, "covered": bool(covered)})
        cumulative = _series(np.cumsum(samples[..., index], axis=1), np.cumsum(truth[:, index]))
        running.append({**entry, **cumulative})
    return {"channels": totals}, {"channels": running}, None


def _temporal(names, families, active, samples, truth, latent, latent_truth, metrics):
    """Temporal coherence of each trajectory family, and of each of its slots.

    Metrics ``temporal/<family>/<key>`` average the family's defined slots
    (:func:`pymc_mfm.training.metrics.temporal_scores`). Returns ``(array,
    None)``, or ``(None, reason)`` without truth or when no score is defined.
    """
    from pymc_mfm.training import metrics as scoring

    if truth is None:
        for family in TEMPORAL_FAMILIES:
            for key in scoring.TEMPORAL_KEYS:
                name = f"temporal/{family}/{key}"
                metrics.add(name, None, reason="simulation truth not supplied")
        return None, "simulation truth not supplied"
    records, reasons, defined = [], set(), False
    for family in TEMPORAL_FAMILIES:
        prefix = f"temporal/{family}/"
        keys = [prefix + key for key in scoring.TEMPORAL_KEYS]
        if family == "latent_total":
            slots, draws, target = ["latent total"], latent[..., None], latent_truth[:, None]
        else:
            group = (families == family) & active
            slots = [names[index] for index in np.flatnonzero(group)]
            draws, target = samples[..., group], truth[:, group]
        if not slots:
            for key in keys:
                metrics.add(key, None, reason=f"no active {family} components")
            continue
        world = scoring.temporal_scores(draws, target, prefix=prefix)
        _add_scores(metrics, world, keys)
        defined = defined or bool(world.values)
        reasons.update(world.undefined.values())
        entries = []
        for position, name in enumerate(slots):
            own = scoring.temporal_scores(draws[..., [position]], target[:, [position]])
            values = {key: own.values.get(key) for key in scoring.TEMPORAL_KEYS}
            entries.append({"name": name, "values": values, "undefined": own.undefined})
        records.append({"family": family, "slots": entries})
    if not defined:
        return None, "no temporal score is defined: " + ", ".join(sorted(reasons))
    return {"families": records}, None


def _strata(result, truth, graph_truth, families, active, scale):
    """Window-total rows of the active channels and controls and their strata.

    ``{"table": strata_summary(rows), "rows": rows}`` in ``sales_scale`` units,
    with this world's own size cuts, or ``(None, reason)`` without truth,
    ``sales_scale`` or an active channel or control.
    """
    from .strata import EFFECT_FAMILIES, result_effect_rows, strata_summary

    if truth is None:
        return None, "simulation truth not supplied"
    if scale is None:
        return None, "sales_scale provenance absent"
    if not (np.isin(families, EFFECT_FAMILIES) & active).any():
        return None, "no active channel or control components"
    rows = result_effect_rows(result, truth, graph_truth)
    return {"table": strata_summary(rows), "rows": rows}, None


def _reliability(probabilities, truth, families, label):
    """Binned and CORP reliability of active-edge frequencies against graph truth.

    Pooled over every active edge and per family: ``rows`` are the non-empty
    equal-width bins, ``corp`` the isotonic-fit blocks and ``decomposition``
    the Brier score's ``{mcb, dsc, unc}``. An edge set with one truth class
    keeps its bins, but has an empty ``corp``, no ``decomposition`` and a
    ``notice`` that its calibration is not evidenced.
    """
    from pymc_mfm.training import metrics as scoring

    def table(keep):
        p, y = probabilities[keep], truth[keep]
        n, n_pos = int(keep.sum()), int(y.sum())
        notice = scoring.one_class_notice(n, n_pos)
        return {
            "rows": scoring.reliability_table(p, y),
            "corp": [] if notice else scoring.isotonic_calibration(p, y),
            "decomposition": None if notice else scoring.brier_decomposition(p, y),
            "notice": notice,
            "n": n,
            "n_pos": n_pos,
        }

    return {
        "pooled": table(np.ones(truth.size, dtype=bool)),
        "families": {}
        if families is None
        else {family: table(families == family) for family in dict.fromkeys(families.tolist())},
        "bins": scoring.RELIABILITY_BINS,
        "label": label,
    }


def compute_diagnostics(
    result: DecompositionResult,
    *,
    truth=None,
    graph_truth=None,
    observed=None,
    training_records: Sequence[Mapping[str, Any]] = (),
    evaluation_records: Sequence[Mapping[str, Any]] = (),
    reference=None,
):
    """Compute the full applicable catalog from ONE saved conditional window.

    No truth is inferred from observations or zero effects. Graph support and
    scalar quantity masks are applied before recovery/uncertainty statistics.
    Natural observation residuals are not labeled latent closure or noise draws.
    Expected unavailable panels are explicit; malformed evidence raises.

    The world's live/dead attribution (``attribution/*``, sales-scale units),
    per-draw closure, temporal coherence per trajectory family (``temporal/*``),
    time-average coverage (``timeavg/*``), running-total energy score
    (``cumulative/*``), R² and edge reliability are scored by the evaluator's
    rules in :mod:`pymc_mfm.training.metrics`; ``share/*`` errors are
    report-only. Live and dead come from graph truth on each effect's direct
    ``cy``/``zb`` edge. ``arrays["strata"]`` holds each active channel and
    control effect's window-total row and their
    :func:`~pymc_mfm.diagnostics.strata_summary` with this world's cuts
    (:mod:`.strata`), for the residual and strata panels; not metrics.

    Whole-graph statistics ``graph/<key>`` for every key of the evaluator's
    ``joint_router/graph`` score the active slots the evaluator's way; the
    truth-free ones (entropy, distinct graphs, edge co-moments, ``cc``/``zz``
    two-cycle and cycle rates) need no graph truth. ``graph/cyclic_draw_fraction``
    is the draw mass whose complete graph (every edge family) has a directed
    cycle, not a per-family rate. ``uncertainty/n_unique_graphs`` counts the
    groups of the paired variance decomposition: one group even without active
    edges, where ``graph/n_distinct`` is undefined.
    """
    from pymc_mfm.training import metrics as scoring

    samples = np.asarray(result.components, dtype=np.float64)
    graph_draws = np.asarray(result.graphs)
    n, times, width = samples.shape
    active = np.asarray(result.component_mask, dtype=bool)
    edge_active = np.asarray(result.edge_mask, dtype=bool)
    if not active.any():
        raise ValueError("diagnostics require at least one active component")
    truth = _evidence(truth, (times, width), "truth")
    graph_truth = _evidence(graph_truth, (graph_draws.shape[1],), "graph_truth", binary=True)
    observed = _evidence(observed, (times,), "observed")
    if truth is not None and np.any(truth[:, ~active]):
        raise ValueError("truth contains nonzero padded components")
    if graph_truth is not None and np.any(graph_truth[~edge_active]):
        raise ValueError("graph_truth contains edges outside active support")
    scale = result.provenance.get("sales_scale")
    if scale is not None and (isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("sales_scale must be finite and positive when supplied")
    metrics = _Metrics()
    arrays = {}
    panels = {name: {"status": "ok", "reason": None} for name in PANEL_NAMES}

    def unavailable(name, reason):
        panels[name] = {"status": "unavailable", "reason": reason}

    components = []
    calibration = {}
    for index, (name, family) in enumerate(zip(result.component_names, result.component_families)):
        values = samples[..., index]
        target = None if truth is None else truth[:, index]
        series = _series(values, target)
        series.update(index=index, name=name, family=family, active=bool(active[index]))
        series["time_mean"] = _series(
            values.mean(axis=1), None if target is None else target.mean()
        )
        series["zero_atom_probability"] = float(np.mean(np.all(values == 0, axis=1)))
        series["weekly_zero_probability"] = np.mean(values == 0, axis=0).tolist()
        components.append(series)
        prefix = f"component/{quote(name, safe='')}"
        if active[index]:
            _score(prefix, values, target, metrics, result.units, scale)
            _score(
                f"{prefix}/time_mean",
                values.mean(axis=1)[:, None],
                None if target is None else np.asarray([target.mean()]),
                metrics,
                result.units,
                scale,
            )
            metrics.add(f"{prefix}/zero_atom_probability", series["zero_atom_probability"])
            if target is not None:
                nonzero = target != 0
                metrics.add(
                    f"{prefix}/zero_mass_at_nonzero_truth",
                    np.mean((values == 0)[:, nonzero]) if nonzero.any() else None,
                    reason=None if nonzero.any() else "no nonzero truth coordinates",
                )
        else:
            metrics.add(
                f"{prefix}/padding_draw_absolute_mean", np.mean(np.abs(values)), unit=result.units
            )
    arrays["components"] = components

    families = []
    labels = np.asarray(result.component_families)
    for family in FAMILIES:
        mask = (labels == family) & active
        if not mask.any():
            families.append(
                {"name": family, "status": "unavailable", "reason": "no active components"}
            )
            continue
        values = samples[..., mask].sum(axis=-1)
        target = None if truth is None else truth[:, mask].sum(axis=-1)
        families.append({"name": family, "status": "ok", **_series(values, target)})
        _score(f"family/{family}/total", values, target, metrics, result.units, scale)
        if truth is not None:
            calibration[family] = calibration_summary(samples[..., mask], truth[:, mask])
            _score(
                f"family/{family}/individual",
                samples[..., mask],
                truth[:, mask],
                metrics,
                result.units,
                scale,
            )
    arrays["families"] = families
    for panel, family in (
        ("channel_contributions", "channel"),
        ("channel_totals", "channel"),
        ("controls", "control"),
        ("indirect_sources", "indirect"),
    ):
        if not ((labels == family) & active).any():
            unavailable(panel, f"no active {family} components")

    latent = samples[..., active].sum(axis=-1)
    latent_truth = None if truth is None else truth[:, active].sum(axis=-1)
    arrays["total"] = {
        **_series(latent, latent_truth),
        "observed": None if observed is None else observed.tolist(),
        "observed_minus_latent_mean": None
        if observed is None
        else (observed - latent.mean(0)).tolist(),
        "realized_noise": None
        if observed is None or latent_truth is None
        else (observed - latent_truth).tolist(),
        "definition": "Sum of active latent components; excludes observation noise",
    }
    _score("latent_total", latent, latent_truth, metrics, result.units, scale)
    # Closure against the structural outcome (latent truth); relative errors
    # divide by the observations when supplied, else by the structural outcome.
    sales = None if latent_truth is None else (latent_truth if observed is None else observed)
    closure_keys = (
        ("closure_rel_err", "closure/latent_relative_error"),
        ("closure_draw_rel_err", "closure/draw_relative_error"),
        ("closure_draw_q90_rel_err", "closure/draw_q90_relative_error"),
    )
    ruler = 1.0 if scale is None else scale
    closure = (
        None
        if latent_truth is None
        else scoring.latent_sales_scores(
            (samples[..., active] / ruler).sum(axis=-1),
            (truth[:, active] / ruler).sum(axis=-1),
            sales / ruler,
        )
    )
    for key, name in closure_keys:
        if closure is None:
            metrics.add(name, None, reason="latent truth not supplied")
        else:
            metrics.add(name, closure.values.get(key), reason=closure.undefined.get(key))
    # Report-only share errors: a group total's posterior-mean error relative to
    # the outcome magnitude, over indirect sources, all declared control slots
    # (padded ones are zero) and the present controls.
    for key, group in (
        ("indirect_share_mae", active & (labels == "indirect")),
        ("control_share_mae", labels == "control"),
        ("active_control_share_mae", active & (labels == "control")),
    ):
        if truth is None:
            metrics.add(f"share/{key}", None, reason="simulation truth not supplied")
        elif not group.any():
            metrics.add(f"share/{key}", None, reason="no components in this group")
        else:
            error = (samples[..., group] / ruler).sum(axis=-1).mean(axis=0) - (
                truth[:, group] / ruler
            ).sum(axis=-1)
            metrics.add(
                f"share/{key}",
                np.mean(np.abs(error) / np.maximum(np.abs(sales / ruler), scoring.CLOSURE_EPS)),
            )
    metrics.add(
        "observation/latent_mean_absolute_residual",
        None if observed is None else np.abs(observed - latent.mean(0)).mean(),
        unit=result.units,
        reason="not a noise estimate or a posterior predictive score"
        if observed is not None
        else "observations not supplied",
    )
    metrics.add("support/active_components", active.sum(), unit="count")
    metrics.add("support/padded_components", (~active).sum(), unit="count")
    metrics.add("support/active_edges", edge_active.sum(), unit="count")
    metrics.add("support/padded_edges", (~edge_active).sum(), unit="count")
    metrics.add(
        "support/component_padding_max_abs",
        np.max(np.abs(samples[..., ~active])) if (~active).any() else None,
        unit=result.units,
        reason=None if (~active).any() else "no padded components",
    )
    metrics.add(
        "support/edge_padding_mass",
        graph_draws[:, ~edge_active].mean() if (~edge_active).any() else None,
        reason=None if (~edge_active).any() else "no padded edges",
    )
    arrays["support"] = {
        "component_names": list(result.component_names),
        "component_mask": active.tolist(),
        "edge_names": list(result.edge_names),
        "edge_mask": edge_active.tolist(),
        "input_support": result.provenance.get("input_support"),
        "prior": result.provenance.get("prior", result.provenance.get("corpus_provenance")),
        "qualification": (
            "Finite/mask checks are not evidence of in-distribution or causal validity."
        ),
    }
    arrays["input_support"] = _input_support(result, active)
    if arrays["input_support"] is None:
        unavailable("input_support", "raw spend/control observations not recorded in input_support")

    paired = paired_graph_uncertainty(graph_draws[:, edge_active], samples[..., active])
    arrays["uncertainty"] = paired
    total_paired = paired_graph_uncertainty(graph_draws[:, edge_active], latent)
    arrays["total_uncertainty"] = total_paired
    for name in (
        "n_unique_graphs",
        "mode_frequency",
        "n_rare_groups",
        "rare_draw_mass",
        "n_singleton_groups",
    ):
        unit = "count" if name.startswith("n_") else "dimensionless"
        metrics.add(f"uncertainty/{name}", paired[name], unit=unit)
    for name in ("total_variance", "within_graph_variance", "between_graph_variance"):
        metrics.add(f"uncertainty/{name}_mean", np.mean(paired[name]), unit=f"({result.units})^2")
    metrics.add(
        "uncertainty/variance_closure_max_abs",
        np.abs(paired["variance_closure_error"]).max(),
        unit=f"({result.units})^2",
    )
    # The uncertainty panel draws these latent-total series, not per-component means.
    for name in ("total_variance", "within_graph_variance", "between_graph_variance"):
        metrics.add(
            f"uncertainty/latent_total/{name}_mean",
            np.mean(total_paired[name]),
            unit=f"({result.units})^2",
        )
    aggregate_names = [family for family in FAMILIES if ((labels == family) & active).any()]
    aggregates = np.stack(
        [
            samples[..., (labels == family) & active].sum(axis=-1).mean(axis=1)
            for family in aggregate_names
        ],
        axis=1,
    )
    aggregate_uncertainty = paired_graph_uncertainty(
        graph_draws[:, edge_active],
        aggregates,
        covariance=True,
    )
    arrays["aggregate_uncertainty"] = {"quantity_names": aggregate_names, **aggregate_uncertainty}
    covariance = np.asarray(aggregate_uncertainty["total_covariance"])
    sd = np.sqrt(np.diag(covariance))
    for index, family in enumerate(aggregate_names):
        metrics.add(f"uncertainty/time_mean/{family}/sd", sd[index], unit=result.units)
        for other in range(index):
            denominator = sd[index] * sd[other]
            metrics.add(
                f"uncertainty/time_mean/corr_{aggregate_names[other]}_{family}",
                covariance[index, other] / denominator if denominator else None,
                reason=None if denominator else "constant aggregate; correlation undefined",
            )
    probabilities = graph_draws.mean(axis=0)
    endpoints = result.provenance.get("edge_endpoints")
    edge_families = result.provenance.get("edge_families")
    if endpoints is not None:
        if len(endpoints) != len(probabilities) or any(
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(node, str) and node for node in pair)
            for pair in endpoints
        ):
            raise ValueError("edge_endpoints must align exactly with the edge axis")
    if edge_families is not None and len(edge_families) != len(probabilities):
        raise ValueError("edge_families must align exactly with the edge axis")
    arrays["graph_support"] = None
    if edge_families is not None:
        from .structure import graph_support_diagnostics

        arrays["graph_support"] = graph_support_diagnostics(
            graph_draws,
            edge_families,
            edge_active,
            truth=graph_truth,
            support=result.provenance.get("graph_support"),
        )
        for name, value in arrays["graph_support"]["metrics"].items():
            status = arrays["graph_support"]["statuses"][name]
            metrics.add(
                f"graph_support/{name}",
                value,
                unit="count" if name.rsplit("/", 1)[-1] in _SUPPORT_COUNTS else "dimensionless",
                status=status["status"],
                reason=status["reason"],
            )
    records = []
    for index, name in enumerate(result.edge_names):
        if not edge_active[index]:
            continue
        prefix = f"edge/{quote(name, safe='')}"
        metrics.add(f"{prefix}/probability", probabilities[index])
        records.append(
            {
                "index": index,
                "name": name,
                "probability": float(probabilities[index]),
                "truth": None if graph_truth is None else int(graph_truth[index]),
                "endpoints": None if endpoints is None else endpoints[index],
                "family": None if edge_families is None else edge_families[index],
            }
        )
    active_prob = probabilities[edge_active]
    truth_active = None if graph_truth is None else graph_truth[edge_active]
    frequency = paired["graph_frequencies"]
    cardinalities = graph_draws[:, edge_active].sum(axis=1).astype(int)
    counts = np.bincount(cardinalities, minlength=int(edge_active.sum()) + 1)
    graph_label = {
        "model_average": "Empirical complete-graph router frequencies",
        "map": "Selected empirical modal hypothesis; frequencies are not posterior mass",
        "fixed": "Supplied fixed hypothesis; frequencies are not posterior mass",
    }[result.mode]
    arrays["graphs"] = {
        "edges": records,
        "complete_graphs": frequency,
        "mode_graph": paired["mode_graph"],
        "tied_mode_graphs": paired["tied_mode_graphs"],
        "cardinality_counts": counts.tolist(),
        "cardinality_probabilities": (counts / n).tolist(),
        "truth": None if truth_active is None else truth_active.tolist(),
        "true_cardinality": None if truth_active is None else int(truth_active.sum()),
        "source": result.provenance.get("graph_source", "conditional paired draws"),
        "label": graph_label,
    }
    if not records:
        # Every whole-graph statistic is undefined (no_active_slots) without an active edge.
        for name in ("structure_heatmap", "dag_recovery", "graph_statistics"):
            unavailable(name, "no active graph edges")
    elif endpoints is None:
        unavailable("dag_recovery", "edge endpoint metadata absent; topology cannot be inferred")
    metrics.add("graph/cardinality_mean", cardinalities.mean(), unit="count")
    metrics.add("graph/cardinality_sd", cardinalities.std(), unit="count")
    active_endpoints = (
        None
        if endpoints is None
        else [pair for pair, keep in zip(endpoints, edge_active, strict=True) if keep]
    )
    _graph_scores(
        graph_draws[:, edge_active],
        truth_active,
        None if edge_families is None else np.asarray(edge_families)[edge_active],
        active_endpoints,
        metrics,
    )
    if active_endpoints is None:
        metrics.add("graph/cyclic_draw_fraction", None, reason="edge endpoint metadata absent")
        unavailable("cycles", "edge endpoint metadata absent; directed cycles cannot be checked")
    else:
        cyclic = _cyclic_graphs([record["graph"] for record in frequency], active_endpoints)
        for record, flag in zip(frequency, cyclic, strict=True):
            record["cyclic"] = flag
        cyclic_mass = sum(record["frequency"] for record in frequency if record["cyclic"])
        metrics.add("graph/cyclic_draw_fraction", cyclic_mass)
    valid = truth_active is not None and len(active_prob) > 0
    notice = scoring.one_class_notice(active_prob.size, int(truth_active.sum())) if valid else None
    for key, function in (
        ("brier", scoring.brier),
        ("log_score", scoring.log_score),
        ("auroc", scoring.auroc),
        ("ece", scoring.ece),
    ):
        value = function(active_prob, truth_active) if valid else None
        if value is None:
            reason = "both truth classes required" if valid else "active graph truth unavailable"
        else:
            reason = notice if key == "ece" else None
        metrics.add(
            f"graph/{key}",
            value,
            unit="nats" if key == "log_score" else "dimensionless",
            reason=reason,
            status="not_evidenced" if key == "ece" and notice else None,
        )
    decomposition = (
        scoring.brier_decomposition(active_prob, truth_active) if valid and not notice else None
    )
    for key in scoring.BRIER_DECOMPOSITION_KEYS:
        metrics.add(
            f"graph/{key}",
            None if decomposition is None else decomposition[key],
            reason=None
            if decomposition is not None
            else (notice or "active graph truth unavailable"),
        )
    if truth_active is not None:
        metrics.add(
            "graph/cardinality_absolute_error",
            np.abs(cardinalities - truth_active.sum()).mean(),
            unit="count",
        )
        live = truth_active == 1
        dead = ~live
        for key, value, keep in (
            ("false_exclusion_mass", 1 - active_prob, live),
            ("false_inclusion_mass", active_prob, dead),
        ):
            metrics.add(
                f"graph/{key}",
                value[keep].mean() if keep.any() else None,
                reason=None if keep.any() else "no applicable true edges",
            )
        for record in frequency:
            record["is_truth"] = bool(np.array_equal(record["graph"], truth_active))
            record["shd"] = int(np.abs(np.asarray(record["graph"]) - truth_active).sum())
    else:
        for key in (
            "cardinality_absolute_error",
            "false_exclusion_mass",
            "false_inclusion_mass",
        ):
            metrics.add(
                f"graph/{key}",
                None,
                unit="count" if key == "cardinality_absolute_error" else "dimensionless",
                reason="graph truth not supplied",
            )

    # Live/dead labels come from graph truth on each effect's direct edge.
    effect_labels = {
        edge: _effect_labels(
            result.component_names,
            np.flatnonzero((labels == family) & active),
            edge,
            graph_truth,
            edge_active,
            endpoints,
            edge_families,
        )
        for _, family, edge in ATTRIBUTION_EFFECTS
    }
    arrays["attribution"], reason = _attribution(
        result.component_names, labels, active, samples, truth, scale, effect_labels, metrics
    )
    if reason is not None:
        unavailable("attribution", reason)
    arrays["window_totals"], arrays["cumulative"], reason = _window_totals(
        result.component_names,
        np.flatnonzero((labels == "channel") & active),
        samples,
        truth,
        effect_labels["cy"],
        scale,
        metrics,
    )
    if reason is not None:
        unavailable("window_totals", reason)
        unavailable("cumulative", reason)
    arrays["strata"], reason = _strata(result, truth, graph_truth, labels, active, scale)
    if reason is not None:
        for name in STRATA_PANELS:
            unavailable(name, reason)
    arrays["temporal"], reason = _temporal(
        result.component_names, labels, active, samples, truth, latent, latent_truth, metrics
    )
    if reason is not None:
        unavailable("temporal", reason)
    if truth_active is None or not active_prob.size:
        arrays["reliability"] = None
        unavailable(
            "reliability",
            "graph truth not supplied" if truth_active is None else "no active graph edges",
        )
    else:
        arrays["reliability"] = _reliability(
            active_prob,
            truth_active,
            None if edge_families is None else np.asarray(edge_families, dtype=object)[edge_active],
            graph_label,
        )
        pooled_notice = arrays["reliability"]["pooled"]["notice"]
        if pooled_notice is not None:
            # Still drawn: its legend states why no curve is shown.
            panels["reliability"] = {"status": "not_evidenced", "reason": pooled_notice}
    arrays["set_posterior"] = {}
    cardinalities_by_family = result.provenance.get("set_cardinalities")
    if cardinalities_by_family is None:
        unavailable(
            "set_posterior", "explicit design cardinalities not supplied; never inferred from truth"
        )
    elif not isinstance(cardinalities_by_family, Mapping) or edge_families is None:
        raise ValueError("set_cardinalities requires a mapping and aligned edge_families")
    else:
        from .structure import factorized_set_diagnostics

        for family, cardinality in cardinalities_by_family.items():
            mask = (np.asarray(edge_families) == family) & edge_active
            if not mask.any():
                raise ValueError(f"design cardinality names no active edges: {family}")
            approximation = factorized_set_diagnostics(
                probabilities[mask],
                cardinality,
                truth=None if graph_truth is None else graph_truth[mask],
                edge_names=[name for name, keep in zip(result.edge_names, mask) if keep],
                source=arrays["graphs"]["source"],
            )
            arrays["set_posterior"][family] = approximation
            for name, value in approximation["metrics"].items():
                status = approximation["statuses"][name]
                metrics.add(
                    f"set_approximation/{family}/{name}",
                    value,
                    status=status["status"],
                    reason=status["reason"],
                )
        panels["set_posterior"] = {
            "status": "approximation",
            "reason": "Factorized cardinality-conditioned marginals, not complete-graph posterior",
        }

    if truth is None:
        for name in ("recovery_scatter", "calibration_pit", "calibration_ecdf", "coverage"):
            unavailable(name, "simulation truth not supplied")
    else:
        calibration["pooled"] = calibration_summary(samples[..., active], truth[:, active])
        for family, summary in calibration.items():
            metrics.add(
                f"calibration/{family}/ks_screening",
                summary["ks"],
                status="screening_only",
                reason=SCREENING_QUALIFICATION,
            )
            for key in (
                "chi2",
                "ks_pvalue",
                "chi2_pvalue",
                "ecdf_band_reject",
                "pot_c_pvalue",
                "n_ranks",
            ):
                metrics.add(
                    f"calibration/{family}/{key}",
                    summary[key],
                    unit="count" if key == "n_ranks" else "dimensionless",
                    status="screening_only" if summary[key] is not None else "unavailable",
                    reason=SCREENING_QUALIFICATION,
                )
            for key, value in summary["rank_coverage"].items():
                metrics.add(
                    f"calibration/{family}/rank_implied/{key}",
                    value,
                    status="screening_only",
                    reason=SCREENING_QUALIFICATION,
                )
            for interval in summary["intervals"]:
                level = f"{100 * interval['level']:g}"
                metrics.add(
                    f"calibration/{family}/cov{level}",
                    interval["coverage"],
                    status="screening_only",
                    reason=SCREENING_QUALIFICATION,
                )
                metrics.add(
                    f"calibration/{family}/width{level}", interval["mean_width"], unit=result.units
                )
        for panel in ("calibration_pit", "calibration_ecdf", "coverage"):
            panels[panel] = {"status": "screening_only", "reason": SCREENING_QUALIFICATION}
    arrays["calibration"] = calibration
    for kind, supplied in (("training", training_records), ("evaluation", evaluation_records)):
        arrays[kind] = recorded = record_series(supplied, kind)
        coordinate = recorded["coordinate"]
        for name, rows in recorded["series"].items():
            for row in rows:
                metrics.add(
                    f"{kind}/{coordinate}/{row[coordinate]}/{quote(name, safe='/')}",
                    row["value"],
                    reason=None if row["value"] is not None else "recorded undefined measurement",
                    unit="as_recorded",
                )
    if not arrays["training"]["records"]:
        unavailable("training", "training records not supplied")
    if not arrays["evaluation"]["records"]:
        unavailable("evaluation", "checkpoint evaluation records not supplied")
    if reference is None:
        unavailable("reference_comparison", "reference draws and fit-health evidence not supplied")
        arrays["reference"] = None
    else:
        from .comparisons import reference_diagnostics

        arrays["reference"] = reference_diagnostics(result, reference)
        for name, value in arrays["reference"]["metrics"].items():
            metrics.add(
                f"reference/{name}",
                value,
                reason="see reference fit-health and qualifications",
                status="reference_dependent",
            )
        panels["reference_comparison"] = {
            "status": "reference_dependent",
            "reason": "see reference fit-health and qualifications",
        }
    for name in ("posterior_sbc", "c2st"):
        detail = None if arrays["reference"] is None else arrays["reference"].get(name)
        if detail is None:
            unavailable(name, f"saved {name} evidence not supplied")
        else:
            panels[name] = {"status": "reference_dependent", "reason": detail["qualification"]}
            for key, value in _numeric_leaves(detail):
                metric = f"reference/{name}/{key}"
                if metric not in metrics.values:
                    metrics.add(
                        metric, value, status="reference_dependent", reason=detail["qualification"]
                    )
    from pymc_mfm.inference.result import graphs_unavailable

    no_graphs = graphs_unavailable(result)
    if no_graphs:
        _mark_graphs_unavailable(metrics, panels, arrays)
    statuses = dict(metrics.statuses)
    if result.mode != "model_average":
        reason = (
            "Frequencies and scores describe a selected/supplied graph hypothesis, not posterior "
            "graph probabilities or routing calibration."
        )

        def hypothesis(record):
            # One truth class evidences no calibration under any graph mode.
            if record["status"] == "not_evidenced":
                return {**record, "reason": f"{record['reason']}; {reason}"}
            return {"status": "hypothesis_conditional", "reason": reason}

        for name, value in metrics.values.items():
            if value is not None and (
                name.startswith(("edge/", "graph/", "set_approximation/"))
                or name == "uncertainty/mode_frequency"
            ):
                statuses[f"metric/{name}"] = hypothesis(statuses[f"metric/{name}"])
        for name in (
            "structure_heatmap",
            "dag_recovery",
            "dag_posterior",
            "graph_cardinality",
            "graph_statistics",
            "cycles",
            "reliability",
            "strata",
            "set_posterior",
        ):
            if panels[name]["status"] != "unavailable":
                panels[name] = hypothesis(panels[name])
    if arrays["reference"] is not None:
        statuses.update(
            {
                f"reference/{name}": record
                for name, record in arrays["reference"]["statuses"].items()
            }
        )
    statuses.update({f"panel/{name}": record for name, record in panels.items()})
    for name, array in arrays.items():
        available = bool(array["records"]) if name in ("training", "evaluation") else bool(array)
        statuses[f"array/{name}"] = {
            "status": "ok" if available else "unavailable",
            "reason": None if available else "required evidence not supplied",
        }
        if no_graphs and name in _GRAPH_ARRAYS:
            statuses[f"array/{name}"]["reason"] = _NO_GRAPHS
    if result.mode != "model_average":
        statuses["array/graphs"] = {
            "status": "hypothesis_conditional",
            "reason": (
                "Selected/supplied graph indicators and conditional-bank "
                "frequencies, not posterior probabilities"
            ),
        }
    for name in ("input_support", "prior"):
        statuses[f"support/{name}"] = {
            "status": "recorded" if arrays["support"][name] is not None else "unavailable",
            "reason": "provenance supplied, not independently verified"
            if arrays["support"][name] is not None
            else f"{name} evidence not supplied",
        }
    has_structure_records = any(
        "/structure_head/" in f"/{name}/" for name in arrays["evaluation"]["series"]
    )
    statuses["source/structure_head"] = {
        "status": "recorded_separately" if has_structure_records else "unavailable",
        "reason": "Auxiliary probabilities remain separately labeled in evaluation records."
        if has_structure_records
        else "no auxiliary structure-head records supplied",
    }
    return json_safe(
        {
            "format_version": 1,
            "metrics": metrics.values,
            "metric_units": metrics.units,
            "statuses": statuses,
            "panels": panels,
            "arrays": arrays,
            "metadata": {
                "time": result.time,
                "head": result.provenance.get("head", "conditional"),
                "window": result.window,
                "mode": result.mode,
                "units": result.units,
                "n_samples": n,
                "context_length": result.context_length,
                "component_names": list(result.component_names),
                "component_families": list(result.component_families),
                "component_mask": active.tolist(),
                "edge_names": list(result.edge_names),
                "edge_mask": edge_active.tolist(),
                "interval": "central 80%, NumPy linear quantiles, inclusive endpoints",
            },
            "qualifications": [
                SCREENING_QUALIFICATION,
                (
                    "Finite paired ddof=0 within/between variance is not calibrated "
                    "conditional uncertainty."
                ),
                (
                    "Graph marginals are not complete-graph probabilities; fixed "
                    "graphs are supplied hypotheses."
                ),
                (
                    "Context and query are separately sampled marginal windows; do "
                    "not join their draw axes."
                ),
                (
                    "Latent component totals exclude observation noise; no predictive "
                    "noise draws were invented."
                ),
                "Mask checks do not certify prior/input support or out-of-distribution validity.",
            ],
        }
    )
