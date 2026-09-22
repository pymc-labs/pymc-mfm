"""Numerical comparisons of saved conditional draws and frozen reference evidence.

No function invokes an inference model, generator, or reference sampler.
Component identities, windows and natural units come from ``DecompositionResult``.
A supplied fixed graph is a hypothesis, never an estimated posterior probability.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymc_mfm.inference.result import DecompositionResult
from pymc_mfm.training.metrics import empirical_crps


def _finite(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} requires a lowercase SHA-256 identity")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def _mass(mass: float) -> tuple[float, float]:
    if not math.isfinite(mass) or not 0 < mass < 1:
        raise ValueError("interval mass must lie in (0, 1)")
    return (1 - mass) / 2, (1 + mass) / 2


def _status(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


@dataclass(frozen=True)
class ComparisonRecord:
    """One saved method/world/window, with optional separately supplied truth.

    ``live`` is explicit generating support, not inferred from a nonzero posterior
    mean. ``scale`` is the recorded positive context-outcome ruler. For omission
    contrasts ``world_evidence`` must contain the *original*, full-world arrays
    ``observed``, ``noise``, ``native_inputs``, ``native_truth`` and ``time`` on
    both sides. ``input_names`` labels native columns using the same namespaced
    identities as channel/control components; ``truth_names`` labels every
    original truth column. ``visible_inputs`` records exactly which input IDs
    were supplied. No aggregate CSV or seed substitutes for those raw arrays.
    """

    case: str
    world: str
    method: str
    result: DecompositionResult
    kind: str
    truth: np.ndarray | None = None
    live: np.ndarray | None = None
    graph_truth: np.ndarray | None = None
    observed: np.ndarray | None = None
    latent_truth: np.ndarray | None = None
    scale: float = 1.0
    health: Mapping[str, Any] | None = None
    world_evidence: Mapping[str, np.ndarray] = field(default_factory=dict)
    input_names: tuple[str, ...] = ()
    truth_names: tuple[str, ...] = ()
    visible_inputs: tuple[str, ...] = ()
    omitted_input: str | None = None
    control_edges: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("model", "reference"):
            raise ValueError("kind must explicitly identify model or reference evidence")
        for name in ("case", "world", "method"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.result, DecompositionResult):
            raise ValueError("result must be a DecompositionResult")
        _digest(self.result.provenance.get("data_sha256"), "result.provenance.data_sha256")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        _, weeks, width = self.result.components.shape
        if self.truth is not None:
            object.__setattr__(self, "truth", _finite(self.truth, "truth", (weeks, width)))
            if np.any(self.truth[:, ~self.result.component_mask]):
                raise ValueError("truth must be zero on padded components")
        if self.live is not None:
            live = np.asarray(self.live)
            if live.dtype.kind != "b" or live.shape != (width,):
                raise ValueError("live must be an explicit boolean component mask")
            if np.any(live & ~self.result.component_mask):
                raise ValueError("padded components cannot be generating-live")
            object.__setattr__(self, "live", live)
        if self.graph_truth is not None:
            graph = _finite(self.graph_truth, "graph_truth", (self.result.graphs.shape[1],))
            if not np.isin(graph, (0, 1)).all() or np.any(graph[~self.result.edge_mask]):
                raise ValueError("graph truth must be binary and zero on padded edges")
            object.__setattr__(self, "graph_truth", graph.astype(bool))
        if self.observed is not None:
            object.__setattr__(self, "observed", _finite(self.observed, "observed", (weeks,)))
        if self.latent_truth is not None:
            object.__setattr__(
                self, "latent_truth", _finite(self.latent_truth, "latent_truth", (weeks,))
            )
        evidence = {
            name: _finite(value, f"world_evidence.{name}")
            for name, value in self.world_evidence.items()
        }
        object.__setattr__(self, "world_evidence", evidence)
        for name in ("input_names", "truth_names", "visible_inputs", "control_edges"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(
                not isinstance(v, str) or not v for v in values
            ):
                raise ValueError(f"{name} must contain unique nonempty identities")
            object.__setattr__(self, name, values)
        if not set(self.visible_inputs).issubset(self.input_names):
            raise ValueError("visible inputs must retain original input identities")
        if self.omitted_input is not None and (
            self.omitted_input not in self.input_names or self.omitted_input in self.visible_inputs
        ):
            raise ValueError("omitted input must be a native input excluded from visible inputs")
        if "native_inputs" in evidence:
            native = evidence["native_inputs"]
            if native.ndim != 2 or native.shape[1] != len(self.input_names):
                raise ValueError("native input columns must match input_names")
        if "native_truth" in evidence:
            native = evidence["native_truth"]
            if native.ndim != 2 or native.shape[1] != len(self.truth_names):
                raise ValueError("native truth columns must match truth_names")
        if not set(self.control_edges).issubset(self.result.edge_names):
            raise ValueError("control_edges must identify saved graph columns")


def _score(draws: np.ndarray, truth: np.ndarray | None, mass: float, scale: float) -> dict:
    low, high = np.quantile(draws, _mass(mass), axis=0, method="linear")
    mean = draws.mean(axis=0)
    result = {
        "mean": mean.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "width": float(np.mean(high - low)),
        "zero_mass": np.mean(draws == 0, axis=0).tolist(),
        "crps": None,
        "coverage": None,
        "signed_error": None,
        "mae": None,
    }
    if draws.ndim == 1:
        counts, edges = np.histogram(draws, bins=min(40, max(1, int(math.sqrt(len(draws))))))
        result["distribution"] = {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "density": (counts / (len(draws) * np.diff(edges))).tolist(),
        }
        result["truth"] = None if truth is None else float(truth)
    if truth is not None:
        result.update(
            crps=empirical_crps(draws, truth) / scale,
            coverage=float(np.mean((truth >= low) & (truth <= high))),
            signed_error=float(np.mean(mean - truth)) / scale,
            mae=float(np.mean(np.abs(mean - truth))) / scale,
        )
    return result


def score_record(record: ComparisonRecord, *, mass: float = 0.8) -> dict:
    """Score native trajectories and within-draw window means; never pool windows."""
    _mass(mass)
    result = record.result
    rows, families = [], {}
    for index, (name, family) in enumerate(
        zip(result.component_names, result.component_families, strict=True)
    ):
        if not result.component_mask[index]:
            continue
        draws = result.components[:, :, index]
        truth = None if record.truth is None else record.truth[:, index]
        window_draws = draws.mean(axis=1)
        window_truth = None if truth is None else truth.mean()
        row = {
            "name": name,
            "family": family,
            "live": None if record.live is None else bool(record.live[index]),
            "weekly": _score(draws, truth, mass, record.scale),
            "window_mean": _score(window_draws, window_truth, mass, record.scale),
            "truth": None if truth is None else truth.tolist(),
        }
        row["zero_path_mass"] = float(np.mean(np.all(draws == 0, axis=1)))
        rows.append(row)
    for family in dict.fromkeys(result.component_families):
        selected = [row for row in rows if row["family"] == family]
        live = [row for row in selected if row["live"]]
        metrics = {
            "n_components": len(selected),
            "n_live": None if record.live is None else len(live),
        }
        for name in ("crps", "coverage", "width", "mae", "signed_error"):
            available = [
                row["window_mean"][name] for row in live if row["window_mean"][name] is not None
            ]
            metrics[name] = float(np.mean(available)) if available else None
        effect_family = family in ("channel", "control")
        metrics["total_signed_error"] = (
            None
            if record.truth is None or (effect_family and record.live is None)
            else float(
                sum(
                    row["window_mean"]["signed_error"]
                    for row in (live if effect_family else selected)
                )
            )
        )
        metrics["all_active_total_signed_error"] = (
            None
            if record.truth is None
            else float(sum(row["window_mean"]["signed_error"] for row in selected))
        )
        metrics["false_exclusions"] = (
            None if record.live is None else sum(row["zero_path_mass"] > 0.5 for row in live)
        )
        metrics["all_covered"] = (
            None
            if not live or record.truth is None
            else all(row["window_mean"]["coverage"] == 1 for row in live)
        )
        inactive = [row for row in selected if row["live"] is False]
        metrics["inactive_mean_attribution"] = (
            float(np.mean([abs(row["window_mean"]["mean"]) / record.scale for row in inactive]))
            if inactive
            else None
        )
        families[family] = metrics
    active = result.component_mask
    total = result.components[:, :, active].sum(axis=-1)
    total_truth = record.latent_truth
    if total_truth is None and record.truth is not None and record.omitted_input is None:
        total_truth = record.truth[:, active].sum(axis=-1)
    hidden_mean, absorbed_share = None, None
    if record.omitted_input is not None and total_truth is not None and record.truth is not None:
        hidden_mean = float(np.mean(total_truth - record.truth[:, active].sum(axis=-1)))
        if hidden_mean != 0:
            absorbed_share = 1 + float(np.mean(total) - np.mean(total_truth)) / hidden_mean
    health = reference_fit_health(record.health) if record.kind == "reference" else None
    controls = families.get("control", {})
    structure = {
        "metrics": {},
        "statuses": {
            "edge_families": _status("unavailable", "Saved graph-family labels not supplied")
        },
    }
    if result.provenance.get("edge_families") is not None:
        from pymc_mfm.diagnostics.structure import graph_support_diagnostics

        structure = graph_support_diagnostics(
            result.graphs,
            result.provenance["edge_families"],
            result.edge_mask,
            truth=record.graph_truth,
            support=result.provenance.get("graph_support"),
        )
    modal_control_count = structure["metrics"].get("family/zb/modal_count")
    if record.control_edges:
        columns = [result.edge_names.index(name) for name in record.control_edges]
        if "arrays" in structure:
            mode_graph = np.asarray(structure["arrays"]["mode_graph"])
        else:
            graphs, counts = np.unique(result.graphs, axis=0, return_counts=True)
            mode_graph = graphs[np.argmax(counts)]
        modal_control_count = int(mode_graph[columns].sum())
    live_rows = [row for row in rows if row["live"] and row["family"] in ("channel", "control")]
    recovered = (
        None
        if not live_rows or record.truth is None
        else all(
            row["window_mean"]["coverage"] == 1 and row["zero_path_mass"] <= 0.5
            for row in live_rows
        )
    )
    return _json(
        {
            "case": record.case,
            "world": record.world,
            "method": record.method,
            "kind": record.kind,
            "window": result.window,
            "mode": result.mode,
            "units": result.units,
            "scale": record.scale,
            "provenance": result.provenance,
            "n_samples": result.n_samples,
            "time": result.time,
            "context_length": result.context_length,
            "mass": mass,
            "components": rows,
            "families": families,
            "total": {
                "weekly": _score(total, total_truth, mass, record.scale),
                "window_mean": _score(
                    total.mean(axis=1),
                    None if total_truth is None else total_truth.mean(),
                    mass,
                    record.scale,
                ),
                "truth": total_truth,
            },
            "observed": record.observed,
            "recovered": recovered,
            "n_live_components": len(live_rows) if record.live is not None else None,
            "component_hit_rate": None
            if not live_rows or record.truth is None
            else float(np.mean([row["window_mean"]["coverage"] for row in live_rows])),
            "hidden_mean": hidden_mean,
            "absorbed_share": absorbed_share,
            "five_control_false_exclusions": controls.get("false_exclusions")
            if controls.get("n_live") == 5
            else None,
            "five_control_modal_count_hit": None
            if controls.get("n_live") != 5 or modal_control_count is None
            else modal_control_count == 5,
            "modal_control_count": modal_control_count,
            "reference_health": health,
            "structure": structure,
            "statuses": {
                "truth": _status(
                    "available" if record.truth is not None else "unavailable",
                    "Supplied generating truth"
                    if record.truth is not None
                    else "No generating truth supplied",
                ),
                "support": _status(
                    "available" if record.live is not None else "unavailable",
                    "Explicit generating support"
                    if record.live is not None
                    else "No generating support supplied",
                ),
            },
        }
    )


def _compatible(
    first: ComparisonRecord, second: ComparisonRecord, *, omission: bool = False
) -> None:
    a, b = first.result, second.result
    for name in ("truth", "live"):
        if (getattr(first, name) is None) != (getattr(second, name) is None):
            raise ValueError(f"matched records have asymmetric {name} evidence")
    totals = [
        record.latent_truth
        if record.latent_truth is not None
        else record.truth[:, record.result.component_mask].sum(axis=1)
        if record.truth is not None and record.omitted_input is None
        else None
        for record in (first, second)
    ]
    if (totals[0] is None) != (totals[1] is None):
        raise ValueError("matched records have asymmetric latent total truth evidence")
    if totals[0] is not None and not np.allclose(totals[0], totals[1], rtol=1e-12, atol=1e-12):
        raise ValueError("matched records changed latent total truth")
    if (
        first.world != second.world
        or a.window != b.window
        or a.units != b.units
        or first.scale != second.scale
    ):
        raise ValueError("comparisons require matched world, window, units and scale")
    if not np.array_equal(a.time, b.time) or a.context_length != b.context_length:
        raise ValueError("comparisons require identical time coordinates and context boundary")
    if not omission and a.provenance["data_sha256"] != b.provenance["data_sha256"]:
        raise ValueError("comparison data SHA-256 identities differ")
    a_names = {
        name for name, active in zip(a.component_names, a.component_mask, strict=True) if active
    }
    b_names = {
        name for name, active in zip(b.component_names, b.component_mask, strict=True) if active
    }
    if (not omission and a_names != b_names) or not b_names.issubset(a_names):
        raise ValueError("comparison component identities differ")
    for name in b_names:
        i, j = a.component_names.index(name), b.component_names.index(name)
        if a.component_families[i] != b.component_families[j]:
            raise ValueError("component family changed across methods")
        if (
            first.truth is not None
            and second.truth is not None
            and not np.allclose(first.truth[:, i], second.truth[:, j], rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("generating truth changed across matched records")
        if first.live is not None and second.live is not None and first.live[i] != second.live[j]:
            raise ValueError("generating support changed across matched records")
    for name in ("latent_truth", "observed"):
        left, right = getattr(first, name), getattr(second, name)
        if (
            left is not None
            and right is not None
            and not np.allclose(left, right, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError(f"matched records changed {name}")


def matched_omission(
    base: ComparisonRecord, omitted: ComparisonRecord, *, mass: float = 0.8
) -> list[dict]:
    """Verify original raw worlds and score the same retained live component IDs.

    The pre-omission score excludes the omitted component too. Subtracting two
    aggregate family scores with different denominators is deliberately forbidden.
    """
    _compatible(base, omitted, omission=True)
    if base.method != omitted.method or base.result.mode != omitted.result.mode:
        raise ValueError("omission pairs require the same method and inference mode")
    for record in (base, omitted):
        _digest(record.result.provenance.get("checkpoint_sha256"), "omission checkpoint_sha256")
        if (
            not isinstance(record.result.provenance.get("inference_config"), Mapping)
            or not record.result.provenance["inference_config"]
        ):
            raise ValueError("omission requires a recorded inference configuration")
    for key in ("checkpoint_sha256", "inference_config"):
        if base.result.provenance.get(key) != omitted.result.provenance.get(key):
            raise ValueError(f"omission pairs changed {key}")
    required = {"observed", "noise", "native_inputs", "native_truth", "time"}
    if not required.issubset(base.world_evidence) or set(base.world_evidence) != set(
        omitted.world_evidence
    ):
        raise ValueError(
            "matched omission requires paired raw full-world evidence, not aggregate scores"
        )
    for name, first in base.world_evidence.items():
        second = omitted.world_evidence[name]
        if first.shape != second.shape or not np.allclose(first, second, rtol=1e-12, atol=1e-12):
            raise ValueError(f"unmatched original world evidence: {name}")
    if (
        base.input_names != omitted.input_names
        or base.omitted_input is not None
        or omitted.omitted_input is None
    ):
        raise ValueError("omission requires a complete base and the same original input identities")
    if base.visible_inputs != base.input_names:
        raise ValueError("base world does not expose all native inputs")
    expected = tuple(name for name in base.visible_inputs if name != omitted.omitted_input)
    if omitted.visible_inputs != expected:
        raise ValueError(
            "omission must remove exactly its declared input without reordering retained IDs"
        )
    if base.truth is None or omitted.truth is None or base.live is None or omitted.live is None:
        raise ValueError("matched omission requires explicit truth and live masks on both sides")
    if base.truth_names != omitted.truth_names:
        raise ValueError("original truth identities changed across omission pair")
    original = base.world_evidence
    times = original["time"]
    if times.ndim != 1 or len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("original world time must be strictly increasing")
    shape = (len(times),)
    for name in ("observed", "noise"):
        _finite(original[name], f"original {name}", shape)
    _finite(
        original["native_inputs"], "original native_inputs", (len(times), len(base.input_names))
    )
    _finite(original["native_truth"], "original native_truth", (len(times), len(base.truth_names)))
    if not np.allclose(
        original["native_truth"].sum(axis=1) + original["noise"],
        original["observed"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("original truth, noise and observations do not close")
    positions = np.searchsorted(times, base.result.time)
    if np.any(positions >= len(times)) or not np.array_equal(times[positions], base.result.time):
        raise ValueError("result times do not identify an original world window")
    base_names = {
        name
        for name, active in zip(
            base.result.component_names, base.result.component_mask, strict=True
        )
        if active
    }
    if set(base.truth_names) != base_names:
        raise ValueError("native truth must identify every original active component")
    for record in (base, omitted):
        active_inputs = {
            name
            for name, family, active in zip(
                record.result.component_names,
                record.result.component_families,
                record.result.component_mask,
                strict=True,
            )
            if active and family in ("channel", "control")
        }
        if active_inputs != set(record.visible_inputs):
            raise ValueError(
                "active input components do not exactly match retained visible original IDs"
            )
        active_names = {
            name
            for name, active in zip(
                record.result.component_names, record.result.component_mask, strict=True
            )
            if active
        }
        expected_names = base_names if record is base else base_names - {omitted.omitted_input}
        if active_names != expected_names:
            raise ValueError("omission changed components beyond the declared removed input")
        for column, name in enumerate(record.result.component_names):
            if not record.result.component_mask[column]:
                continue
            expected_truth = original["native_truth"][positions, base.truth_names.index(name)]
            if not np.allclose(record.truth[:, column], expected_truth, rtol=1e-12, atol=1e-12):
                raise ValueError("record truth is not bound to original raw world evidence")
        if record.observed is not None and not np.allclose(
            record.observed, original["observed"][positions], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("record observations are not bound to original raw world evidence")
        if record.latent_truth is not None and not np.allclose(
            record.latent_truth,
            original["native_truth"][positions].sum(axis=1),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("latent total truth is not bound to original raw world evidence")
    before = {row["name"]: row for row in score_record(base, mass=mass)["components"]}
    rows = []
    for row in score_record(omitted, mass=mass)["components"]:
        if not row["live"] or row["family"] not in ("channel", "control"):
            continue
        old = before[row["name"]]
        rows.append(
            {
                "case": omitted.case,
                "base_case": base.case,
                "world": base.world,
                "method": base.method,
                "window": base.result.window,
                "family": row["family"],
                "component_id": row["name"],
                "omitted_input": omitted.omitted_input,
                "before_crps": old["window_mean"]["crps"],
                "after_crps": row["window_mean"]["crps"],
                "delta_crps": row["window_mean"]["crps"] - old["window_mean"]["crps"],
                "base_data_sha256": base.result.provenance["data_sha256"],
                "omitted_data_sha256": omitted.result.provenance["data_sha256"],
            }
        )
    return rows


def _estimate(values: Sequence[float], *, seed: int, bootstrap_samples: int) -> dict:
    values = _finite(values, "world values").astype(float, copy=False)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("estimates require nonempty one-dimensional world values")
    rng = np.random.default_rng(seed)
    # Bound temporary memory independently of the requested bootstrap count.
    means = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 256):
        stop = min(start + 256, bootstrap_samples)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "n_worlds": len(values),
        "n_zero": int(np.sum(values == 0)),
    }


def comparison_summary(
    records: Sequence[ComparisonRecord],
    *,
    mass: float = 0.8,
    method_pairs: Sequence[tuple[str, str]] = (),
    omission_pairs: Sequence[tuple[str, str]] = (),
    bootstrap_samples: int = 2000,
    seed: int = 0,
    false_effect_threshold: float = 0.01,
) -> dict:
    """Compute the complete saved-evidence comparison table used by all renderers.

    Bootstrap units are whole worlds, not posterior draws or component-weeks.
    Every requested contrast requires identical world coverage; unmatched rows
    never disappear through an inner join. Reference fits remain even if unhealthy.
    """
    _mass(mass)
    if (
        not records
        or isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ValueError("records and a positive integer bootstrap_samples are required")
    if not math.isfinite(false_effect_threshold) or false_effect_threshold <= 0:
        raise ValueError("false_effect_threshold must be positive")
    index = {}
    group_units = {}
    for record in records:
        key = (record.case, record.world, record.method, record.result.window)
        if key in index:
            raise ValueError(f"duplicate case/world/method/window: {key}")
        index[key] = record
        group = (record.case, record.method, record.result.window)
        units = group_units.setdefault(group, record.result.units)
        if units != record.result.units:
            raise ValueError(
                "aggregate worlds must share outcome units; native widths cannot mix currencies"
            )
    # Methods on the same case/world/window must see exactly the same observations.
    worlds = defaultdict(list)
    for record in records:
        worlds[(record.case, record.world, record.result.window)].append(record)
    for group in worlds.values():
        for record in group[1:]:
            _compatible(group[0], record)
    scored = [score_record(record, mass=mass) for record in records]
    scored_index = {
        (row["case"], row["world"], row["method"], row["window"]): row for row in scored
    }
    grouped = defaultdict(list)
    for row in scored:
        for family, values in row["families"].items():
            for metric in (
                "crps",
                "coverage",
                "width",
                "mae",
                "signed_error",
                "total_signed_error",
                "all_active_total_signed_error",
                "false_exclusions",
                "inactive_mean_attribution",
            ):
                if values[metric] is not None:
                    grouped[(row["case"], row["method"], row["window"], family, metric)].append(
                        (row["world"], values[metric])
                    )
        for metric in (
            "recovered",
            "component_hit_rate",
            "absorbed_share",
            "five_control_false_exclusions",
            "five_control_modal_count_hit",
            "modal_control_count",
        ):
            if row[metric] is not None:
                grouped[(row["case"], row["method"], row["window"], "all", metric)].append(
                    (row["world"], float(row[metric]))
                )
        for metric in ("crps", "coverage", "mae", "signed_error", "width"):
            if row["total"]["window_mean"][metric] is not None:
                grouped[(row["case"], row["method"], row["window"], "total", metric)].append(
                    (row["world"], row["total"]["window_mean"][metric])
                )
        for metric, value in row["structure"]["metrics"].items():
            if value is not None:
                grouped[(row["case"], row["method"], row["window"], "graph", metric)].append(
                    (row["world"], float(value))
                )
        for family in ("channel", "control"):
            inactive = [
                component
                for component in row["components"]
                if component["family"] == family and component["live"] is False
            ]
            if inactive:
                count = sum(
                    abs(component["window_mean"]["mean"]) / row["scale"] >= false_effect_threshold
                    for component in inactive
                )
                grouped[
                    (row["case"], row["method"], row["window"], family, "false_effect_rate")
                ].append((row["world"], count / len(inactive)))
                grouped[
                    (row["case"], row["method"], row["window"], family, "worlds_with_false_effect")
                ].append((row["world"], float(count > 0)))
    aggregate = []
    for (case, method, window, family, metric), values in sorted(grouped.items()):
        values.sort()
        aggregate.append(
            {
                "case": case,
                "method": method,
                "window": window,
                "family": family,
                "metric": metric,
                **_estimate(
                    [value for _, value in values], seed=seed, bootstrap_samples=bootstrap_samples
                ),
            }
        )
    paired = []
    automatic_pairs = set()
    checkpoint_methods = defaultdict(lambda: defaultdict(set))
    for record in records:
        checkpoint = record.result.provenance.get("checkpoint_sha256")
        if checkpoint:
            _digest(checkpoint, "checkpoint_sha256")
            checkpoint_methods[checkpoint][record.result.mode].add(record.method)
    for modes in checkpoint_methods.values():
        if len(modes["model_average"]) == 1:
            average = next(iter(modes["model_average"]))
            for mode in ("map", "fixed"):
                if len(modes[mode]) == 1:
                    automatic_pairs.add((next(iter(modes[mode])), average))
    method_pairs = tuple(dict.fromkeys((*map(tuple, method_pairs), *sorted(automatic_pairs))))
    for first, second in method_pairs:
        a = {(r.case, r.world, r.result.window): r for r in records if r.method == first}
        b = {(r.case, r.world, r.result.window): r for r in records if r.method == second}
        if not a or set(a) != set(b):
            raise ValueError(
                f"paired methods {first}/{second} require exactly matched case/world/window records"
            )
        differences = defaultdict(list)
        for key in sorted(a):
            _compatible(a[key], b[key])
            ra, rb = (
                scored_index[(key[0], key[1], first, key[2])],
                scored_index[(key[0], key[1], second, key[2])],
            )
            for family in ra["families"]:
                va, vb = ra["families"][family]["crps"], rb["families"].get(family, {}).get("crps")
                if va is not None and vb is not None:
                    differences[(key[0], key[2], family)].append(vb - va)
            va, vb = ra["total"]["window_mean"]["crps"], rb["total"]["window_mean"]["crps"]
            if va is not None and vb is not None:
                differences[(key[0], key[2], "total")].append(vb - va)
        for (case, window, family), values in differences.items():
            paired.append(
                {
                    "case": case,
                    "window": window,
                    "family": family,
                    "first": first,
                    "second": second,
                    "metric": "crps_difference",
                    "definition": "second minus first",
                    "kind": "graph_averaging"
                    if (first, second) in automatic_pairs
                    else "method_comparison",
                    "n_second_better": int(np.sum(np.asarray(values) < 0)),
                    "n_first_better": int(np.sum(np.asarray(values) > 0)),
                    "n_ties": int(np.sum(np.asarray(values) == 0)),
                    **_estimate(values, seed=seed, bootstrap_samples=bootstrap_samples),
                }
            )
    omission = []
    for base_case, omitted_case in omission_pairs:
        a = {(r.world, r.method, r.result.window): r for r in records if r.case == base_case}
        b = {(r.world, r.method, r.result.window): r for r in records if r.case == omitted_case}
        if not a or set(a) != set(b):
            raise ValueError("omission cases require exactly matched world/method/window records")
        for key in sorted(a):
            omission.extend(matched_omission(a[key], b[key], mass=mass))
    component_groups = defaultdict(list)
    for row in omission:
        component_groups[
            (row["case"], row["method"], row["window"], row["family"], row["world"])
        ].append(row["delta_crps"])
    omission_groups = defaultdict(list)
    for key, values in component_groups.items():
        omission_groups[key[:-1]].append(float(np.mean(values)))
    omission_summary = [
        {
            "case": key[0],
            "method": key[1],
            "window": key[2],
            "family": key[3],
            "metric": "matched_delta_crps",
            **_estimate(values, seed=seed, bootstrap_samples=bootstrap_samples),
        }
        for key, values in sorted(omission_groups.items())
    ]
    return {
        "format_version": 1,
        "records": scored,
        "aggregate": aggregate,
        "paired": paired,
        "matched_omission": omission,
        "omission_summary": omission_summary,
        "settings": {
            "mass": mass,
            "interval": "equal-tailed, NumPy linear quantiles",
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "false_effect_threshold": false_effect_threshold,
            "method_pairs": _json(method_pairs),
            "omission_pairs": _json(omission_pairs),
        },
        "qualifications": [
            "Window means average weeks within each draw, never independently sampled windows.",
            (
                "Intervals resample worlds, not training seeds; component-week "
                "coverage is descriptive."
            ),
            "Fixed-graph views describe supplied hypotheses, not posterior graph probabilities.",
            (
                "Reference fits are retained with their health warnings; "
                "different conditioning information is not equalized."
            ),
            (
                "The conjunction recovery rate has no universal mass**k "
                "calibration target without component independence."
            ),
        ],
    }


def reference_fit_health(
    health: Mapping[str, Any] | None, *, max_rhat: float = 1.01, min_ess: float = 400
) -> dict:
    """Flag unhealthy and incomplete fits without dropping or refitting them."""
    required = (
        "divergences",
        "max_rhat",
        "min_ess_bulk",
        "min_ess_tail",
        "tree_depth_hits",
        "min_bfmi",
    )
    if health is None:
        return {
            "status": "unavailable",
            "healthy": None,
            "reasons": ["Reference sampler diagnostics not supplied"],
            "metrics": {},
        }
    values, reasons, missing = {}, [], []
    for name in required:
        value = health.get(name)
        if (
            value is None
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
        ):
            missing.append(name)
            values[name] = None
        else:
            values[name] = float(value)
            if value < 0 or (name in ("divergences", "tree_depth_hits") and value != int(value)):
                raise ValueError(f"invalid reference diagnostic {name}")
    if values["divergences"] is not None and values["divergences"] > 0:
        reasons.append("Divergent transitions")
    if values["tree_depth_hits"] is not None and values["tree_depth_hits"] > 0:
        reasons.append("Tree-depth saturation")
    if values["max_rhat"] is not None and values["max_rhat"] > max_rhat:
        reasons.append(f"R-hat exceeds {max_rhat}")
    for name in ("min_ess_bulk", "min_ess_tail"):
        if values[name] is not None and values[name] < min_ess:
            reasons.append(f"{name} below {min_ess}")
    if values["min_bfmi"] is not None and values["min_bfmi"] < 0.3:
        reasons.append("BFMI below 0.3")
    if missing:
        reasons.append("Missing or nonfinite diagnostics: " + ", ".join(missing))
    return {
        "status": "warning" if reasons else "available",
        "healthy": False if reasons and not missing else None if missing else True,
        "reasons": reasons,
        "metrics": values,
        "thresholds": {"max_rhat": max_rhat, "min_ess": min_ess, "min_bfmi": 0.3},
        "retained": True,
    }


def run_c2st(
    reference_vectors: Any,
    approximate_vectors: Any,
    world_ids: Sequence[str],
    *,
    data_features: Any | None = None,
    classifier: str = "mlp",
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict:
    """Fit a diagnostic classifier on saved world vectors, never an inference model.

    Each row is one reference/approximate pair from an independent world (or
    independently generated reference replication), not a correlated draw from
    one MCMC chain. Both classes of each world stay on one side of the split.
    The optional classifier dependency is imported only on this explicit call.
    ``prediction_evidence`` can be persisted and passed to
    :func:`c2st_from_predictions` without fitting the classifier again.
    """
    reference = _finite(reference_vectors, "reference_vectors")
    approximate = _finite(approximate_vectors, "approximate_vectors", reference.shape)
    if reference.ndim != 2 or len(reference) < 16 or reference.shape[1] < 1:
        raise ValueError("C2ST needs at least 16 independent world vectors")
    ids = tuple(world_ids)
    if (
        len(ids) != len(reference)
        or len(set(ids)) != len(ids)
        or any(not isinstance(v, str) or not v for v in ids)
    ):
        raise ValueError("world_ids must uniquely label independent reference/approximate pairs")
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie in (0, 1)")
    if classifier not in ("mlp", "logistic", "quadratic"):
        raise ValueError("classifier must be mlp, logistic or quadratic")
    if data_features is not None:
        features = _finite(data_features, "data_features")
        if features.ndim != 2 or features.shape[0] != len(reference) or features.shape[1] < 1:
            raise ValueError("data_features must provide one finite vector per world")
        reference = np.concatenate((reference, features), axis=1)
        approximate = np.concatenate((approximate, features), axis=1)
    n_test = max(2, round(len(reference) * test_fraction))
    if len(reference) - n_test < 2:
        raise ValueError("test_fraction leaves fewer than two training worlds")
    permutation = np.random.default_rng(seed).permutation(len(reference))
    test, train = permutation[:n_test], permutation[n_test:]
    x_train = np.concatenate((reference[train], approximate[train]))
    x_test = np.concatenate((reference[test], approximate[test]))
    y_train = np.repeat([0, 1], len(train))
    y_test = np.repeat([0, 1], len(test))
    center, scale = x_train.mean(axis=0), np.maximum(x_train.std(axis=0), 1e-12)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    if classifier == "mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
    elif classifier == "quadratic":
        estimator = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
    else:
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    estimator.fit((x_train - center) / scale, y_train)
    evidence = {
        "labels": y_test.tolist(),
        "probabilities": estimator.predict_proba((x_test - center) / scale)[:, 1].tolist(),
        "world_ids": [ids[i] for i in test] * 2,
        "training_world_ids": [ids[i] for i in train],
        "classifier": classifier,
        "data_features": data_features is not None,
    }
    summary = c2st_from_predictions(**evidence)
    summary.update(prediction_evidence=evidence, seed=seed, test_fraction=test_fraction)
    return summary


def c2st_from_predictions(
    labels: Any,
    probabilities: Any,
    world_ids: Sequence[str],
    *,
    training_world_ids: Sequence[str],
    classifier: str,
    data_features: bool = True,
) -> dict:
    """Score frozen held-out C2ST predictions, preserving world-level pairing.

    One class-0 and class-1 row per held-out world are required. The p-value is
    an exact *paired label-swap* null conditional on the trained classifier,
    not a binomial assertion that the two rows of a world are independent.
    """
    y = _finite(labels, "labels")
    p = _finite(probabilities, "probabilities", y.shape)
    if y.ndim != 1 or len(y) < 4 or not np.isin(y, (0, 1)).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("C2ST requires binary labels and probabilities in [0, 1]")
    ids = tuple(world_ids)
    if len(ids) != len(y) or any(not isinstance(v, str) or not v for v in ids) or not classifier:
        raise ValueError("C2ST requires a classifier name and one world identity per row")
    train = set(training_world_ids)
    if not train or train.intersection(ids):
        raise ValueError("C2ST training and held-out worlds must be nonempty and disjoint")
    correct = (p >= 0.5) == y
    pair_scores = []
    for world in sorted(set(ids)):
        positions = [i for i, value in enumerate(ids) if value == world]
        if len(positions) != 2 or set(y[positions]) != {0, 1}:
            raise ValueError("C2ST requires exactly one reference/approximate pair per world")
        pair_scores.append(int(correct[positions].sum()) - 1)
    nonzero = np.asarray(pair_scores)[np.asarray(pair_scores) != 0]
    successes, trials = int(np.sum(nonzero > 0)), len(nonzero)
    if trials:
        k = min(successes, trials - successes)
        logs = np.asarray(
            [
                math.lgamma(trials + 1)
                - math.lgamma(i + 1)
                - math.lgamma(trials - i + 1)
                - trials * math.log(2)
                for i in range(k + 1)
            ]
        )
        p_value = min(1.0, 2 * float(np.exp(logs).sum()))
    else:
        p_value = 1.0
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    cross_entropy = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    return {
        "accuracy": float(correct.mean()),
        "cross_entropy": cross_entropy,
        "divergence_lb": max(0.0, math.log(2) - cross_entropy),
        "p_value": p_value,
        "n_test": len(y),
        "n_test_worlds": len(set(ids)),
        "n_training_worlds": len(train),
        "classifier": classifier,
        "test": "exact paired label-swap",
        "qualification": (
            "A non-rejection is not proof of calibration; divergence estimate "
            "is descriptive. The paired swap null requires independent worlds "
            "and reference/approximate exchangeability within each world; no "
            "automatic independence claim is made. "
        )
        + (
            "Data features included."
            if data_features
            else "Quantity-marginal-only comparison cannot detect ignored data."
        ),
        "labels": y.tolist(),
        "probabilities": p.tolist(),
        "world_ids": list(ids),
    }


def posterior_sbc_from_draws(
    posterior_draws: Any,
    augmented_draws: Any,
    query_indices: Any,
    quantity_names: Sequence[str],
    *,
    provenance: Mapping[str, Any],
    seed: int = 0,
    levels: Sequence[float] = (0.5, 0.8, 0.95),
) -> dict:
    """Rank saved posterior-world values against saved augmented-posterior draws.

    Rows are replications, not weeks of a single posterior. Each augmentation
    must have been generated from the corresponding *joint* posterior-world
    draw and be valid for the sequence model. This routine never generates the
    replicate or silently appends exchangeable observations to a time series.
    """
    truth = _finite(posterior_draws, "posterior_draws")
    draws = _finite(augmented_draws, "augmented_draws")
    if (
        truth.ndim != 2
        or len(truth) < 2
        or draws.ndim != 3
        or draws.shape[0] != truth.shape[0]
        or draws.shape[2] != truth.shape[1]
        or draws.shape[1] < 1
    ):
        raise ValueError(
            "posterior-SBC needs [replication,quantity] and [replication,draw,quantity]"
        )
    names = tuple(quantity_names)
    if (
        len(names) != truth.shape[1]
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n for n in names)
    ):
        raise ValueError("quantity names must uniquely label the ranked quantities")
    queries = _finite(query_indices, "query_indices", (len(truth),))
    if np.any(queries < 0) or not np.equal(queries, np.floor(queries)).all():
        raise ValueError("query_indices must identify one nonnegative query index per replication")
    _digest(provenance.get("observed_data_sha256"), "observed_data_sha256")
    if (
        provenance.get("replicate_contract")
        not in ("valid_sequential_continuation", "exchangeable")
        or provenance.get("joint_draws") is not True
    ):
        raise ValueError(
            "posterior-SBC requires an explicit valid replicate and joint-draw contract"
        )
    if not levels:
        raise ValueError("at least one coverage level is required")
    for level in levels:
        _mass(level)
    less = np.sum(draws < truth[:, None, :], axis=1)
    equal = np.sum(draws == truth[:, None, :], axis=1)
    # Uniform discrete tie rank includes both endpoints, including zero atoms.
    rng = np.random.default_rng(seed)
    ranks = less + rng.integers(equal + 1)
    pit = (ranks + 0.5) / (draws.shape[1] + 1)
    coverage = {
        name: {
            str(level): float(
                np.mean((pit[:, i] >= (1 - level) / 2) & (pit[:, i] <= (1 + level) / 2))
            )
            for level in levels
        }
        for i, name in enumerate(names)
    }
    return {
        "ranks": {name: ranks[:, i].tolist() for i, name in enumerate(names)},
        "pit": {name: pit[:, i].tolist() for i, name in enumerate(names)},
        "ecdf": {
            name: {
                "x": np.sort(pit[:, i]).tolist(),
                "y": (np.arange(1, len(truth) + 1) / len(truth)).tolist(),
            }
            for i, name in enumerate(names)
        },
        "coverage": coverage,
        "n_replications": len(truth),
        "n_post_samples": draws.shape[1],
        "query_indices": queries.astype(int).tolist(),
        "seed": seed,
        "provenance": _json(provenance),
        "qualification": (
            "Conditional updating diagnostic near one observed dataset; "
            "validity depends on the saved replicate contract. Not prior-wide "
            "calibration proof."
        ),
    }


def reference_diagnostics(result: DecompositionResult, reference: Mapping[str, Any]) -> dict:
    """Report hook for aligned, saved reference draws and optional calibration evidence."""
    required = {"label", "components", "component_names", "time", "window", "units", "provenance"}
    if not required.issubset(reference):
        raise ValueError(
            "reference requires label, components, names, time, window, units and provenance"
        )
    if (
        tuple(reference["component_names"]) != result.component_names
        or reference["window"] != result.window
        or reference["units"] != result.units
        or not np.array_equal(reference["time"], result.time)
    ):
        raise ValueError("reference component identities, time, window and units must align")
    identity = _digest(reference["provenance"].get("data_sha256"), "reference data_sha256")
    if identity != result.provenance.get("data_sha256"):
        raise ValueError("reference data identity differs from the result")
    draws = _finite(reference["components"], "reference components")
    if draws.ndim != 3 or draws.shape[0] < 1 or draws.shape[1:] != result.components.shape[1:]:
        raise ValueError("reference draws must have shape [sample,time,component]")
    if np.any(draws[:, :, ~result.component_mask]):
        raise ValueError("reference draws must be zero on padded components")
    mask = result.component_mask
    first = _score(result.components[:, :, mask], None, 0.8, 1.0)
    second = _score(draws[:, :, mask], None, 0.8, 1.0)
    health = reference_fit_health(reference.get("health"))
    ratio = first["width"] / second["width"] if second["width"] > 0 else None
    summary = {
        "metrics": {
            "width_ratio": ratio,
            **{f"health/{k}": v for k, v in health["metrics"].items()},
        },
        "statuses": {
            "health": _status(
                health["status"],
                "; ".join(health["reasons"]) or "Recorded fit diagnostics meet thresholds",
            ),
            "width_ratio": _status(
                "available" if ratio is not None else "undefined",
                "Descriptive; conditioning information may differ"
                if ratio is not None
                else "Reference interval width is zero",
            ),
        },
        "arrays": {
            "time": result.time.tolist(),
            "names": [
                name for name, active in zip(result.component_names, mask, strict=True) if active
            ],
            "model": first,
            "reference": second,
        },
        "label": str(reference["label"]),
        "window": result.window,
        "units": result.units,
        "reference_health": health,
        "provenance": _json(reference["provenance"]),
        "qualifications": [
            "Reference health failures remain visible and never trigger refitting.",
            (
                "Widths compare potentially differently conditioned approximate "
                "posteriors, not calibration targets."
            ),
        ],
    }
    for name, function in (
        ("posterior_sbc", posterior_sbc_from_draws),
        ("c2st", c2st_from_predictions),
    ):
        if name in reference:
            if (
                name == "posterior_sbc"
                and reference[name].get("provenance", {}).get("observed_data_sha256") != identity
            ):
                raise ValueError("posterior-SBC observed data identity differs from the reference")
            detail = function(**reference[name])
            summary[name] = detail
            summary["statuses"][name] = _status("available", detail["qualification"])
            if name == "c2st":
                for metric in ("accuracy", "cross_entropy", "divergence_lb", "p_value"):
                    summary["metrics"][f"c2st/{metric}"] = detail[metric]
        else:
            summary["statuses"][name] = _status(
                "unavailable", "Required saved diagnostic evidence not supplied"
            )
    return summary


def _metric_units(metric: str, outcome_units: str) -> str:
    if metric == "width":
        return outcome_units
    if metric in (
        "crps",
        "crps_difference",
        "matched_delta_crps",
        "mae",
        "signed_error",
        "total_signed_error",
        "all_active_total_signed_error",
        "inactive_mean_attribution",
    ):
        return "fraction_of_outcome_scale"
    if metric in (
        "false_exclusions",
        "five_control_false_exclusions",
        "modal_control_count",
        "n_live",
        "n_components",
        "n_live_components",
        "router_unique_graphs",
    ):
        return "count"
    if metric.endswith(("/mean_count", "/modal_count", "/true_count", "/count_mae")):
        return "edges"
    if metric == "hidden_mean":
        return outcome_units
    return "fraction"


def _artifact(path: Path, root: Path, kind: str) -> dict:
    return {"path": path.relative_to(root).as_posix(), "kind": kind, "sha256": _sha256(path)}


def build_comparison_report(
    records: Sequence[ComparisonRecord],
    out_dir: str | Path,
    *,
    strict: bool = True,
    **settings: Any,
) -> dict:
    """Save raw results, numerical tables and a self-contained readable report.

    Any private or unknown record makes the entire bundle sensitive. Per-record
    checkpoint/step identities are retained; aggregate provenance never invents
    a shared training step or checkpoint. Figure failures remain in the manifest.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    summary = comparison_summary(records, **settings)
    synthetic = all(
        record.result.provenance.get("synthetic_training_evaluation") is True
        and not record.result.provenance.get("private_data", False)
        for record in records
    )
    artifacts, entries = [], []
    for index, record in enumerate(records):
        folder = root / f"record_{index:04d}"
        record.result.save(folder)
        arrays = {
            "truth": record.truth,
            "live": record.live,
            "observed": record.observed,
            "latent_truth": record.latent_truth,
            "graph_truth": record.graph_truth,
        }
        arrays = {key: value for key, value in arrays.items() if value is not None}
        evidence_keys = {}
        for number, (name, value) in enumerate(record.world_evidence.items()):
            key = f"world_{number}"
            arrays[key] = value
            evidence_keys[name] = key
        np.savez_compressed(folder / "comparison.npz", **arrays)
        entries.append(
            {
                "path": folder.name,
                "case": record.case,
                "world": record.world,
                "method": record.method,
                "kind": record.kind,
                "scale": record.scale,
                "health": _json(record.health),
                "world_evidence": evidence_keys,
                "input_names": list(record.input_names),
                "visible_inputs": list(record.visible_inputs),
                "truth_names": list(record.truth_names),
                "provenance": record.result.provenance,
                "omitted_input": record.omitted_input,
                "control_edges": list(record.control_edges),
            }
        )
        for path in sorted(folder.iterdir()):
            if path.is_file():
                artifacts.append(_artifact(path, root, "raw_evidence"))
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    artifacts.append(_artifact(summary_path, root, "comparison_table"))
    from pymc_mfm.plotting.comparisons import render_comparisons

    panels = render_comparisons(summary, root / "figures")
    for panel in panels:
        if panel.get("path"):
            artifacts.append(_artifact(root / "figures" / panel["path"], root, "figure"))
    metrics, dimensions = {}, {}
    record_groups = defaultdict(list)
    for record in records:
        record_groups[(record.case, record.method, record.result.window)].append(record)
    for section in ("aggregate", "paired", "omission_summary"):
        for row in summary[section]:
            methods = (row["first"], row["second"]) if section == "paired" else (row["method"],)
            method = " minus ".join(reversed(methods))
            parts = (section, row["case"], method, row["window"], row["family"], row["metric"])
            prefix = "/".join(str(value).replace("~", "~0").replace("/", "~1") for value in parts)
            selected = [
                record
                for label in methods
                for record in record_groups[(row["case"], label, row["window"])]
            ]
            detail = {
                "head": "conditional",
                "mode": "multiple",
                "window": row["window"],
                "world_id": "world_aggregate",
                "profile": row["case"],
                "method": method,
                "family": row["family"],
            }
            for name in ("checkpoint_sha256", "source_step", "data_split"):
                values = [record.result.provenance.get(name) for record in selected]
                if values and values[0] is not None and all(value == values[0] for value in values):
                    detail[name] = values[0]
            for name in ("mode", "units"):
                values = {getattr(record.result, name) for record in selected}
                detail[name] = next(iter(values)) if len(values) == 1 else "multiple"
            detail["outcome_units"] = detail["units"]
            detail["units"] = _metric_units(row["metric"], detail["units"])
            for statistic, value in row.items():
                if value is None or isinstance(value, (bool, int, float)):
                    key = f"{prefix}/{statistic}"
                    metrics[key] = value
                    dimensions[key] = dict(
                        detail,
                        statistic=statistic,
                        units="count" if statistic.startswith("n_") else detail["units"],
                    )
    _write_comparison_index(root, summary, panels)
    artifacts.append(_artifact(root / "index.html", root, "report"))
    for artifact in artifacts:
        artifact["sensitive"] = not synthetic
    manifest = {
        "format_version": 1,
        "kind": "conditional_comparison",
        "records": entries,
        "summary": "summary.json",
        "provenance": {
            "synthetic_training_evaluation": synthetic,
            "records": [entry["provenance"] for entry in entries],
        },
        "dimensions": {
            "head": "conditional",
            "mode": "comparison",
            "window": "multiple",
            "units": "multiple",
        },
        "metrics": metrics,
        "metric_dimensions": dimensions,
        "statuses": {
            panel["name"]: _status(panel["status"], panel.get("reason", "Saved-evidence rendering"))
            for panel in panels
        },
        "panels": panels,
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if strict and any(panel["status"] == "failed" for panel in panels):
        raise RuntimeError(
            "Comparison figure generation failed; details are saved in manifest.json"
        )
    return manifest


def _write_comparison_index(
    root: Path, summary: Mapping[str, Any], panels: Sequence[Mapping[str, Any]]
) -> None:
    parts = [
        (
            "<!doctype html><html lang='en'><meta "
            "charset='utf-8'><title>Saved conditional comparison</title>"
        ),
        (
            "<style>body{font:16px sans-serif;max-width:1200px;margin:2em "
            "auto;padding:1em}img{max-width:100%}section{margin:2em "
            "0}li{margin:.5em 0}</style>"
        ),
        (
            "<h1>Saved conditional comparison</h1><p><a "
            "href='summary.json'>Numerical evidence</a></p><ul>"
        ),
    ]
    parts.extend(f"<li>{html.escape(str(note))}</li>" for note in summary["qualifications"])
    parts.append(
        (
            "</ul><p>Approximate reference fits remain included when "
            "unhealthy or incomplete; consult the fit-health panels.</p>"
        )
    )
    for panel in panels:
        parts.append(f"<section><h2>{html.escape(panel['name'])}</h2>")
        if panel.get("path"):
            image = base64.b64encode((root / "figures" / panel["path"]).read_bytes()).decode(
                "ascii"
            )
            parts.append(
                (
                    "<img loading='lazy' src='data:image/png;base64,"
                    f"{image}"
                    "' alt='"
                    f"{html.escape(panel['name'], quote=True)}"
                    "'>"
                )
            )
        else:
            parts.append(
                f"<p>{html.escape(panel['status'])}: {html.escape(panel.get('reason', ''))}</p>"
            )
        parts.append("</section>")
    parts.append("</html>")
    (root / "index.html").write_text("\n".join(parts) + "\n")


def _load_manifest(saved_path: str | Path) -> tuple[Path, dict]:
    path = Path(saved_path)
    root = path if path.is_dir() else path.parent
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("kind") != "conditional_comparison":
        raise ValueError("unsupported comparison bundle")
    paths = [artifact["path"] for artifact in manifest["artifacts"]]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate comparison artifact descriptors")
    required = {manifest["summary"]}
    folders = [entry["path"] for entry in manifest["records"]]
    if len(folders) != len(set(folders)):
        raise ValueError("duplicate comparison record directories")
    for folder in folders:
        required.update(
            f"{folder}/{name}" for name in ("result.json", "arrays.npz", "comparison.npz")
        )
    if not required.issubset(paths):
        raise ValueError("comparison bundle is missing required raw evidence descriptors")
    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        target = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not target.is_relative_to(root.resolve())
        ):
            raise ValueError("comparison artifact escapes bundle")
        if _sha256(target) != artifact["sha256"]:
            raise ValueError(f"comparison artifact hash mismatch: {relative}")
    return root, manifest


def load_comparison_records(saved_path: str | Path) -> tuple[ComparisonRecord, ...]:
    """Load verified raw comparisons without a model or any plotting dependency."""
    root, manifest = _load_manifest(saved_path)
    records = []
    verified = {entry["path"] for entry in manifest["artifacts"]}
    for entry in manifest["records"]:
        folder = Path(entry["path"])
        if (
            folder.is_absolute()
            or ".." in folder.parts
            or (folder / "comparison.npz").as_posix() not in verified
        ):
            raise ValueError("unverified comparison record path")
        result = DecompositionResult.load(root / folder)
        with np.load(root / folder / "comparison.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        records.append(
            ComparisonRecord(
                case=entry["case"],
                world=entry["world"],
                method=entry["method"],
                kind=entry["kind"],
                result=result,
                truth=arrays.get("truth"),
                live=arrays.get("live"),
                observed=arrays.get("observed"),
                latent_truth=arrays.get("latent_truth"),
                graph_truth=arrays.get("graph_truth"),
                scale=entry["scale"],
                health=entry["health"],
                world_evidence={name: arrays[key] for name, key in entry["world_evidence"].items()},
                input_names=tuple(entry["input_names"]),
                visible_inputs=tuple(entry["visible_inputs"]),
                truth_names=tuple(entry["truth_names"]),
                omitted_input=entry["omitted_input"],
                control_edges=tuple(entry["control_edges"]),
            )
        )
    return tuple(records)


def render_comparison_report(saved_path: str | Path, out_dir: str | Path) -> list[dict]:
    """Replot the verified, stored numerical table, without rerunning any score."""
    root, manifest = _load_manifest(saved_path)
    summary_path = manifest["summary"]
    if summary_path not in {entry["path"] for entry in manifest["artifacts"]}:
        raise ValueError("comparison summary is not integrity-bound")
    summary = json.loads((root / summary_path).read_text())
    from pymc_mfm.plotting.comparisons import render_comparisons

    return render_comparisons(summary, out_dir)


def load_replication_summary(
    summary_path: str | Path,
    *,
    expected_sha256: str,
    evidence_root: str | Path,
) -> dict:
    """Read a hash-pinned, verified derived replication table and its raw receipts.

    The version-2 evidence schema retains full window-mean draws and verifies
    matched omissions at original component identity level. Its weekly quantiles
    are NOT a replacement for aligned weekly draws. This adapter verifies every
    declared raw archive and required omission/sparse-world archive; it never
    manufactures weekly paths or graph banks. Additional scalar-table intervals
    bootstrap whole worlds once at import, with the recorded seed/resample count.
    """
    path, root = Path(summary_path), Path(evidence_root).resolve()
    _digest(expected_sha256, "expected_sha256")
    if _sha256(path) != expected_sha256:
        raise ValueError("replication summary hash mismatch")
    source = json.loads(path.read_text())
    if source.get("schema_version") != 2:
        raise ValueError("unsupported verified replication summary schema")
    receipts = source["source"]["raw_npz_sha256"]
    if len(receipts) != source["source"]["raw_npz_count"] or not receipts:
        raise ValueError("raw evidence receipt inventory is incomplete")
    verified = []
    for relative, expected in receipts.items():
        member = Path(relative)
        target = (root / member).resolve()
        if member.is_absolute() or ".." in member.parts or not target.is_relative_to(root):
            raise ValueError("raw evidence path escapes its root")
        _digest(expected, f"raw evidence {relative}")
        if _sha256(target) != expected:
            raise ValueError(f"raw evidence hash mismatch: {relative}")
        verified.append({"path": relative, "sha256": expected, "kind": "raw_evidence"})
    n_worlds = source["population"]["worlds_per_case"]
    if type(n_worlds) is not int or n_worlds < 1:
        raise ValueError("invalid replication world count")
    for case in ("constant_intercept", "omitted_channel", "omitted_control", "sparse_inputs"):
        for world in range(n_worlds):
            for directory in ("worlds", "networks/A", "networks/B", "references", "references_mmm"):
                name = f"{directory}/{case}/w{world:04d}.npz"
                if name not in receipts:
                    raise ValueError(f"insufficient paired raw evidence: {name}")
    checks = source["verification"]
    if (
        checks["matched_world_pairs"] != 2 * n_worlds
        or checks["false_effect_worlds"] != n_worlds
        or checks["aggregate_scores_checked"] < 1
    ):
        raise ValueError("derived evidence lacks complete raw-pair verification")
    if (
        not math.isfinite(checks["max_absolute_crps_difference"])
        or checks["max_absolute_crps_difference"] > checks["score_atol"]
    ):
        raise ValueError("derived scores do not match raw-evidence verification tolerance")
    # Bind aggregate metrics and protocol/checkpoint identities as well.
    for kind in ("metrics", "manifest"):
        relative = Path(source["source"][kind])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid replication metadata path")
        member = path.parent.parent / relative
        if _sha256(member) != source["source"][f"{kind}_sha256"]:
            raise ValueError(f"replication {kind} hash mismatch")
    manifest = json.loads((path.parent.parent / source["source"]["manifest"]).read_text())
    aggregate, paired, omission = [], [], []
    families = {"media": "channel", "controls": "control", "latent_total": "total"}

    def add(case, method, family, metric, stats):
        for key in ("mean", "ci_low", "ci_high"):
            if key in stats and stats[key] is not None and not math.isfinite(stats[key]):
                raise ValueError(f"nonfinite derived statistic: {case}/{method}/{metric}/{key}")
        aggregate.append(
            {
                "case": case,
                "method": method,
                "window": "query",
                "family": family,
                "metric": metric,
                **stats,
            }
        )

    for case, methods in source["metrics"].items():
        for method, values in methods.items():
            for name, stats in values.items():
                family, metric = name.rsplit("_", 1)
                saved = dict(stats)
                if metric == "crps":
                    saved.update(source["failure_size"][case][method][name])
                add(case, method, families[family], metric, saved)
    for case, methods in source["recovery"].items():
        for method, values in methods.items():
            for name, stats in values.items():
                add(case, method, "all", "recovered" if name == "recovery_rate" else name, stats)
    for case, values in source["ab_verdict"].items():
        for name, stats in values.items():
            paired.append(
                {
                    "case": case,
                    "window": "query",
                    "family": families[name.removesuffix("_crps")],
                    "first": stats["a_method"],
                    "second": stats["b_method"],
                    "kind": "method_comparison",
                    "definition": "second minus first",
                    "metric": "crps_difference",
                    **stats,
                }
            )
    for case, values in source["averaging"].items():
        for stats in values.values():
            paired.append(
                {
                    "case": case,
                    "window": "query",
                    "family": "control",
                    "first": stats["modal_method"],
                    "second": stats["average_method"],
                    "kind": "graph_averaging",
                    "definition": "graph average minus modal graph",
                    "metric": "crps_difference",
                    **stats,
                }
            )
    for case, methods in source["omission"].items():
        for method, values in methods.items():
            for family, stats in values.items():
                omission.append(
                    {
                        "case": case,
                        "method": method,
                        "window": "query",
                        "family": family,
                        "metric": "matched_delta_crps",
                        "n_component_pairs": stats["n_component_pairs"],
                        **stats["delta_crps"],
                    }
                )
    for stats in source["five_controls"].values():
        add(
            "five_controls",
            stats["method"],
            "all",
            "five_control_false_exclusions",
            stats["false_exclusions"],
        )
        add(
            "five_controls",
            stats["method"],
            "all",
            "five_control_modal_count_hit",
            stats["modal_control_count_hit"],
        )
    for method, values in source["false_effects"]["methods"].items():
        for family in ("channel", "control"):
            detail = values[family]
            add(
                source["false_effects"]["case"],
                method,
                family,
                "inactive_mean_attribution",
                detail["mean_attribution"],
            )
            add(
                source["false_effects"]["case"],
                method,
                family,
                "worlds_with_false_effect",
                detail["worlds_with_any_false_effect"],
            )
            add(
                source["false_effects"]["case"],
                method,
                family,
                "false_effect_rate",
                {
                    "mean": detail["component_false_effect_rate"],
                    "ci_low": None,
                    "ci_high": None,
                    "n_worlds": n_worlds,
                },
            )
    scalar_rows = _replication_scalar_tables(
        path.parent.parent / source["source"]["metrics"],
        aggregate,
        paired,
        seed=source["bootstrap"]["seed"],
        bootstrap_samples=source["bootstrap"]["resamples"],
    )
    return {
        "format_version": 1,
        "records": [],
        "aggregate": aggregate,
        "paired": paired,
        "matched_omission": [],
        "omission_summary": omission,
        "settings": {
            "mass": 0.8,
            "interval": "equal-tailed, NumPy linear quantiles",
            "seed": source["bootstrap"]["seed"],
            "bootstrap_samples": source["bootstrap"]["resamples"],
            "false_effect_threshold": source["false_effects"]["threshold"],
        },
        "source_evidence": {
            "summary_sha256": expected_sha256,
            "artifacts": verified,
            "verification": checks,
            "checkpoint_identities": manifest["checkpoints"],
        },
        "source_summary": source,
        "source_metrics": scalar_rows,
        "reference_fit_summary": source["diagnostics"],
        "statuses": {
            "weekly_joint_draws": _status(
                "unavailable", "Replication retained weekly quantiles, not aligned weekly draws"
            ),
            "weekly_graph_uncertainty": _status(
                "unavailable", "Complete paired weekly graph/quantity banks are not retained"
            ),
            "matched_omission": _status(
                "available",
                (
                    "Externally hash-pinned derived component-ID pairing; all "
                    "required raw input/draw archives verified"
                ),
            ),
        },
        "qualifications": source["limitations"],
    }


def input_support_summary(
    context_inputs: Mapping[str, Any],
    query_inputs: Mapping[str, Any] | None = None,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict:
    """Summarize saved raw inputs without standardizing away location or scale."""
    if not context_inputs:
        raise ValueError("raw context inputs are required")
    if query_inputs is not None and set(query_inputs) != set(context_inputs):
        raise ValueError("context/query inputs must preserve identical entity identities")
    rows = []
    context_count, query_count = None, None
    for name, values in context_inputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("input names must be nonempty strings")
        context = _finite(values, f"context input {name}")
        if context.ndim != 1 or not context.size:
            raise ValueError("each raw input must be a nonempty time vector")
        if context_count is not None and context_count != len(context):
            raise ValueError("context input time lengths differ")
        context_count = len(context)
        mean, std = float(context.mean()), float(context.std())
        row = {
            "name": name,
            "context": {
                "count": len(context),
                "mean": mean,
                "std": std,
                "min": float(context.min()),
                "max": float(context.max()),
            },
            "query": None,
            "standardized_mean_shift": None,
            "scale_ratio": None,
        }
        if query_inputs is not None:
            query = _finite(query_inputs[name], f"query input {name}")
            if query.ndim != 1 or not query.size:
                raise ValueError("each query input must be a nonempty time vector")
            if query_count is not None and query_count != len(query):
                raise ValueError("query input time lengths differ")
            query_count = len(query)
            row["query"] = {
                "count": len(query),
                "mean": float(query.mean()),
                "std": float(query.std()),
                "min": float(query.min()),
                "max": float(query.max()),
            }
            if std > 0:
                row["standardized_mean_shift"] = (float(query.mean()) - mean) / std
                row["scale_ratio"] = float(query.std()) / std
        row["status"] = _status(
            "available" if std > 0 else "undefined",
            "Raw support descriptors; no OOD pass/fail claim"
            if std > 0
            else "Constant context input: standardized shift/scale ratio undefined",
        )
        rows.append(row)
    return {
        "inputs": rows,
        "provenance": _json(provenance or {}),
        "qualification": (
            "Raw support descriptors preserve location and scale; no "
            "generator or inferred truth is used."
        ),
    }


def _replication_scalar_tables(
    path: Path, aggregate: list, paired: list, *, seed: int, bootstrap_samples: int
) -> list:
    """Import all verified scalar evidence; never reconstruct discarded trajectories."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {
            "case",
            "world",
            "method",
            "window",
            "media_crps",
            "controls_crps",
            "latent_total_crps",
        }.issubset(reader.fieldnames or ()):
            raise ValueError("replication scalar table lacks identity or CRPS columns")
        rows = []
        identities = set()
        for source in reader:
            row = {}
            for name, value in source.items():
                if name in ("case", "world", "method", "window", "network"):
                    row[name] = value
                elif value in ("", "nan", "NaN"):
                    row[name] = None
                elif value in ("True", "False"):
                    row[name] = value == "True"
                else:
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError("nonfinite replication scalar")
                    row[name] = parsed
            key = tuple(row[name] for name in ("case", "world", "method", "window"))
            if key in identities or row["window"] not in ("context", "query"):
                raise ValueError("duplicate or invalid replication scalar identity")
            identities.add(key)
            total_live = (row.get("media_n_live") or 0) + (row.get("controls_n_live") or 0)
            if (
                total_live
                and row.get("media_coverage") is not None
                and row.get("controls_coverage") is not None
            ):
                row["component_hit_rate"] = (
                    row["media_coverage"] * row["media_n_live"]
                    + row["controls_coverage"] * row["controls_n_live"]
                ) / total_live
            rows.append(row)
    if not rows:
        raise ValueError("replication scalar table is empty")
    mapping = {
        "latent_total_error": ("total", "signed_error"),
        "latent_total_crps": ("total", "crps"),
        "latent_total_covered": ("total", "coverage"),
        "latent_total_weekly_coverage": ("total", "weekly_coverage"),
        "baseline_error": ("baseline", "total_signed_error"),
        "recovered": ("all", "recovered"),
        "component_hit_rate": ("all", "component_hit_rate"),
        "n_live_components": ("all", "n_live_components"),
        "hidden_truth_mean": ("all", "hidden_mean"),
        "absorbed_share": ("all", "absorbed_share"),
        "structure_exact_hit": ("graph", "structure_exact_hit"),
        "live_channel_count_hit": ("graph", "family/cy/modal_count_hit"),
        "live_control_count_hit": ("graph", "family/zb/modal_count_hit"),
        "router_mode_share": ("graph", "router_mode_share"),
        "router_unique_graphs": ("graph", "router_unique_graphs"),
    }
    for old, family in (("media", "channel"), ("controls", "control")):
        for suffix, metric in (
            ("crps", "crps"),
            ("abs_error", "mae"),
            ("total_error", "total_signed_error"),
            ("coverage", "coverage"),
            ("all_covered", "all_covered"),
            ("switched_off", "false_exclusions"),
            ("max_zero_share", "max_zero_path_mass"),
            ("n_live", "n_live"),
        ):
            mapping[f"{old}_{suffix}"] = (family, metric)
    grouped = defaultdict(list)
    for row in rows:
        for column, (family, metric) in mapping.items():
            if row.get(column) is not None:
                grouped[(row["case"], row["method"], row["window"], family, metric)].append(
                    float(row[column])
                )
    present = {
        tuple(row[name] for name in ("case", "method", "window", "family", "metric"))
        for row in aggregate
    }
    for key, values in sorted(grouped.items()):
        if key not in present:
            aggregate.append(
                dict(
                    zip(("case", "method", "window", "family", "metric"), key, strict=True),
                    **_estimate(values, seed=seed, bootstrap_samples=bootstrap_samples),
                )
            )
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][(row["case"], row["world"], row["window"])] = row
    oracle = by_method.get("oracle")
    if oracle:
        for method, values in sorted(by_method.items()):
            if method == "oracle":
                continue
            if set(values) != set(oracle):
                raise ValueError("oracle comparisons require identical world/window coverage")
            differences = defaultdict(list)
            for identity, row in sorted(values.items()):
                for family, column in (
                    ("channel", "media_crps"),
                    ("control", "controls_crps"),
                    ("total", "latent_total_crps"),
                ):
                    first, second = oracle[identity].get(column), row.get(column)
                    if (first is None) != (second is None):
                        raise ValueError("oracle comparison has asymmetric scoring evidence")
                    if first is not None:
                        differences[(identity[0], identity[2], family)].append(second - first)
            for (case, window, family), values in sorted(differences.items()):
                paired.append(
                    {
                        "case": case,
                        "window": window,
                        "family": family,
                        "first": "oracle",
                        "second": method,
                        "metric": "crps_difference",
                        "definition": "method minus privileged approximate oracle",
                        "kind": "oracle_comparison",
                        "n_second_better": int(np.sum(np.asarray(values) < 0)),
                        "n_first_better": int(np.sum(np.asarray(values) > 0)),
                        "n_ties": int(np.sum(np.asarray(values) == 0)),
                        **_estimate(values, seed=seed, bootstrap_samples=bootstrap_samples),
                    }
                )
    return rows


def corpus_support_summary(
    context_outcomes: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None
) -> dict:
    """Describe saved world-level context outcomes without generating new tasks."""
    if not context_outcomes:
        raise ValueError("saved context outcomes are required")
    worlds = []
    for world, values in context_outcomes.items():
        if not isinstance(world, str) or not world:
            raise ValueError("world identities must be nonempty strings")
        values = _finite(values, f"context outcomes {world}")
        if values.ndim != 1 or not values.size:
            raise ValueError("each world's context outcomes must be a nonempty vector")
        worlds.append(
            {
                "world": world,
                "n_context": len(values),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )
    return {
        "worlds": worlds,
        "provenance": _json(provenance or {}),
        "qualification": (
            "Descriptive distribution of saved context-outcome "
            "locations/scales across worlds; not an independent calibration "
            "test or a generated prior sample."
        ),
    }
