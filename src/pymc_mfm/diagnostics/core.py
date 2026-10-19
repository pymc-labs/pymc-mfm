"""Pure computations for conditional decomposition reports; no renderers or inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import numpy as np

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
    "structure_heatmap",
    "dag_recovery",
    "dag_posterior",
    "graph_cardinality",
    "support",
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


def json_safe(value):
    """Convert scientific containers, refusing nonfinite values and unknown objects."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
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

    low, high = np.quantile(samples, [0.1, 0.9], axis=0)
    metrics.add(f"{target}/mean", samples.mean(), unit=units)
    metrics.add(f"{target}/interval80_width", np.mean(high - low), unit=units)
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
    error = samples.mean(axis=0) - truth
    mae = float(np.mean(np.abs(error)))
    metrics.add(f"{target}/mae", mae, unit=units)
    metrics.add(f"{target}/bias", error.mean(), unit=units)
    metrics.add(f"{target}/crps", scoring.empirical_crps(samples, truth), unit=units)
    metrics.add(f"{target}/cov80", scoring.coverage(samples, truth))
    denominator = float(np.abs(truth).mean())
    metrics.add(
        f"{target}/relative_mae",
        mae / denominator if denominator else None,
        reason=None if denominator else "zero-magnitude truth; relative error undefined",
    )
    metrics.add(
        f"{target}/scaled_mae",
        mae / scale if scale is not None else None,
        unit="sales_scale",
        reason=None if scale is not None else "sales_scale provenance absent",
    )
    metrics.add(
        f"{target}/energy_score",
        scoring.energy_score(samples, truth),
        unit=units,
        reason="requires at least two draws" if len(samples) < 2 else None,
    )
    truth_variance = float(np.var(truth))
    metrics.add(
        f"{target}/var_ratio",
        np.var(samples.mean(axis=0)) / truth_variance if truth_variance else None,
        reason=None if truth_variance else "constant truth; trajectory variance ratio undefined",
    )
    metrics.add(
        f"{target}/r2",
        1 - np.mean(error**2) / truth_variance if truth_variance else None,
        reason=None if truth_variance else "constant truth; R-squared undefined",
    )


def _cycle(graph, endpoints):
    adjacency = {}
    indegree = {}
    for keep, (source, target) in zip(graph, endpoints, strict=True):
        if not keep:
            continue
        adjacency.setdefault(source, []).append(target)
        indegree.setdefault(source, 0)
        indegree[target] = indegree.get(target, 0) + 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency.get(node, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(indegree)


def _numeric_leaves(record, prefix=""):
    for key, value in record.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _numeric_leaves(value, name)
        elif value is None or isinstance(value, (int, float, bool, np.number)):
            yield name, value


def _record_series(records, kind, metrics):
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
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not np.isfinite(step):
            raise ValueError(f"{kind} record step must be finite")
        if steps and step <= steps[-1]:
            raise ValueError(f"{kind} records must have strictly increasing steps")
        steps.append(step)
        for name, value in _numeric_leaves(record):
            if name == "step":
                continue
            if value is not None and not np.isfinite(value):
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
            metrics.add(
                f"{kind}/{coordinate}/{step}/{quote(name, safe='/')}",
                value,
                reason=None if value is not None else "recorded undefined measurement",
                unit="as_recorded",
            )
    return {
        "steps": steps if coordinate == "step" else None,
        "coordinate": coordinate,
        "series": lines,
        "records": json_safe(records),
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
    metrics.add(
        "closure/latent_relative_error",
        None
        if latent_truth is None
        else np.mean(
            np.abs(latent.mean(0) - latent_truth)
            / np.maximum(np.abs(observed if observed is not None else latent_truth), 1e-6)
        ),
        reason=None if latent_truth is not None else "latent truth not supplied",
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
        metrics.add(f"uncertainty/{name}", paired[name])
    for name in ("total_variance", "within_graph_variance", "between_graph_variance"):
        metrics.add(f"uncertainty/{name}_mean", np.mean(paired[name]), unit=f"({result.units})^2")
    metrics.add(
        "uncertainty/variance_closure_max_abs",
        np.abs(paired["variance_closure_error"]).max(),
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
                f"graph_support/{name}", value, status=status["status"], reason=status["reason"]
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
        unavailable("structure_heatmap", "no active graph edges")
        unavailable("dag_recovery", "no active graph edges")
    elif endpoints is None:
        unavailable("dag_recovery", "edge endpoint metadata absent; topology cannot be inferred")
    metrics.add("graph/cardinality_mean", cardinalities.mean(), unit="count")
    metrics.add("graph/cardinality_sd", cardinalities.std(), unit="count")
    frequencies = np.asarray([record["frequency"] for record in frequency])
    metrics.add("graph/empirical_entropy", -np.sum(frequencies * np.log(frequencies)), unit="nats")
    if endpoints is None:
        metrics.add("graph/cyclic_draw_fraction", None, reason="edge endpoint metadata absent")
    else:
        active_endpoints = [pair for pair, keep in zip(endpoints, edge_active, strict=True) if keep]
        cyclic_mass = sum(
            record["frequency"] for record in frequency if _cycle(record["graph"], active_endpoints)
        )
        metrics.add("graph/cyclic_draw_fraction", cyclic_mass)
    for key, function in (
        ("brier", scoring.brier),
        ("log_score", scoring.log_score),
        ("auroc", scoring.auroc),
        ("ece", scoring.ece),
    ):
        valid = truth_active is not None and len(active_prob) > 0
        value = function(active_prob, truth_active) if valid else None
        metrics.add(
            f"graph/{key}",
            value,
            reason=None
            if value is not None
            else ("both truth classes required" if valid else "active graph truth unavailable"),
        )
    if truth_active is not None:
        mismatch = np.abs(graph_draws[:, edge_active] - truth_active)
        metrics.add("graph/shd_mean", mismatch.sum(axis=1).mean(), unit="count")
        metrics.add("graph/exact_recovery", np.all(mismatch == 0, axis=1).mean())
        metrics.add(
            "graph/cardinality_absolute_error", np.abs(cardinalities - truth_active.sum()).mean()
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
    else:
        for key in (
            "shd_mean",
            "exact_recovery",
            "cardinality_absolute_error",
            "false_exclusion_mass",
            "false_inclusion_mass",
        ):
            metrics.add(f"graph/{key}", None, reason="graph truth not supplied")

    # Live/dead groups use graph labels, never zero-valued effects as causal labels.
    if endpoints is not None and edge_families is not None and graph_truth is not None:
        for family, direct_edge in (("channel", "cy"), ("control", "zb")):
            for live_label, live_value in (("live", 1), ("dead", 0)):
                indices = []
                for component in components:
                    if not component["active"] or component["family"] != family:
                        continue
                    matches = [
                        i
                        for i, (edge_family, pair) in enumerate(zip(edge_families, endpoints))
                        if edge_family == direct_edge
                        and pair[0] == component["name"]
                        and edge_active[i]
                    ]
                    if len(matches) == 1 and graph_truth[matches[0]] == live_value:
                        indices.append(component["index"])
                key = f"attribution/{family}/{live_label}"
                if indices and truth is not None:
                    _score(
                        key, samples[..., indices], truth[:, indices], metrics, result.units, scale
                    )
                    metrics.add(
                        f"{key}/draw_absolute_mean",
                        np.abs(samples[..., indices]).mean(),
                        unit=result.units,
                    )
                else:
                    metrics.add(
                        f"{key}/mae",
                        None,
                        reason="no applicable graph-labeled effects or component truth",
                    )
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
    arrays["training"] = _record_series(training_records, "training", metrics)
    arrays["evaluation"] = _record_series(evaluation_records, "evaluation", metrics)
    if not training_records:
        unavailable("training", "training records not supplied")
    if not evaluation_records:
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
    statuses = dict(metrics.statuses)
    if result.mode != "model_average":
        reason = (
            "Frequencies and scores describe a selected/supplied graph hypothesis, not posterior "
            "graph probabilities or routing calibration."
        )
        for name, value in metrics.values.items():
            if value is not None and (
                name.startswith(("edge/", "graph/", "set_approximation/"))
                or name == "uncertainty/mode_frequency"
            ):
                statuses[f"metric/{name}"] = {"status": "hypothesis_conditional", "reason": reason}
        for name in (
            "structure_heatmap",
            "dag_recovery",
            "dag_posterior",
            "graph_cardinality",
            "set_posterior",
        ):
            if panels[name]["status"] != "unavailable":
                panels[name] = {"status": "hypothesis_conditional", "reason": reason}
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
                "time": result.time.tolist(),
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
