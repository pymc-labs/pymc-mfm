"""In-loop evaluation for decomposition, structure, and graph readouts.

``evaluate_rows`` scores schema-4 corpus rows in normalized target space and
returns one JSON-safe record with separately labeled readouts:

* ``joint``: joint router marginals for query and context windows;
* ``conditional``: graph-conditional decoder draws;
* ``structure_head``: auxiliary edge probabilities over active slots;
* ``joint_router``: edge marginals and whole-graph statistics from joint draws.

Scores aggregate active coordinates within each world, then aggregate across
worlds. Unavailable or non-finite values are named under ``undefined`` rather
than coerced into valid-looking metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from pymc_mfm.data.observations import observation_identity
from pymc_mfm.model.graph import (
    STRUCT_EDGE_KEYS,
    EdgeSlotLayout,
    binary_graphs_from_flow,
    structure_active_masks,
)
from pymc_mfm.model.model import MFM
from pymc_mfm.training import metrics
from pymc_mfm.training.stream import batches_for_rows

if TYPE_CHECKING:
    from pymc_mfm.inference.result import DecompositionResult

ResultSink = Callable[["DecompositionResult", np.ndarray, np.ndarray, np.ndarray], None]

#: Held-out joint-query energy score minimized during checkpoint selection.
MONITOR_METRIC = "holdout/joint/query/decomposition_energy_score"

#: Additional required checkpoint gates: auxiliary ``cy`` log score and exact
#: padded-channel leakage.
GATE_METRICS = (
    "holdout/structure_head/cy/log_score",
    "holdout/joint/query/contribution_padding_leakage_scaled_mae",
)

HEAD_LABELS = {
    "joint": "joint router marginal; not a conditional-decoder quality claim",
    "conditional": "graph-conditional decoder, model_average mode",
    "structure_head": "auxiliary BCE structure head; independent per-edge probabilities",
    "joint_router": (
        "edge marginals and whole-graph statistics recovered from joint-head graph draws"
    ),
}

DECOMPOSITION_HEADS = ("joint", "conditional")
WINDOWS = ("query", "context")
INDIRECT_SOURCES = ("cc", "zc", "dc")
_FAMILIES = (
    "baseline_intrinsic",
    "contribution",
    "control_contribution",
    "confounder_total",
    "indirect_cc",
    "indirect_zc",
    "indirect_dc",
    "indirect_total",
)
_FAMILY_STATS = ("scaled_mae", "crps", "cov80")
_ENERGY_METRICS = (
    "decomposition_energy_score",
    "latent_sales_temporal_energy_score",
    "contribution_temporal_energy_score",
)
DECOMPOSITION_METRICS: tuple[str, ...] = tuple(
    f"{family}_{stat}" for family in _FAMILIES for stat in _FAMILY_STATS
) + ("contribution_padding_leakage_scaled_mae", "closure_rel_err", *_ENERGY_METRICS)
_EDGE_SCORES = ("log_score", "brier", "auroc", "ece", "prevalence")
EDGE_METRICS = (*_EDGE_SCORES, "n", "n_pos")
GRAPH_METRICS = ("shd_mean", "exact_recovery")

#: Denominator floor for ``closure_rel_err``.
CLOSURE_EPS = 1e-6


@contextmanager
def isolated_rng(device: torch.device, seed: int) -> Iterator[None]:
    """Seed the CPU generator and evaluation device, then restore both.

    This isolates evaluation draws from the caller's CPU, CUDA, or MPS random
    state, so changing evaluation settings cannot change training draws.
    """
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            elif device.type == "mps":
                torch.mps.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class _MeanAccumulator:
    """Finite per-world means over a fixed key set with drop bookkeeping."""

    def __init__(self, keys: Sequence[str]):
        self.keys = tuple(keys)
        self._finite: dict[str, list[float]] = {key: [] for key in self.keys}
        self._nonfinite: dict[str, int] = dict.fromkeys(self.keys, 0)
        self._reasons: dict[str, str] = {}

    def push(self, key: str, value: float) -> None:
        if np.isfinite(value):
            self._finite[key].append(float(value))
        else:
            self._nonfinite[key] += 1

    def skip(self, key: str, reason: str) -> None:
        self._reasons.setdefault(key, reason)

    def finalize(self, prefix: str, undefined: dict[str, str]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for key in self.keys:
            finite = self._finite[key]
            dropped = self._nonfinite[key]
            out[key] = float(np.mean(finite)) if finite else None
            if dropped:
                undefined[f"{prefix}/{key}"] = f"nonfinite_in_{dropped}_worlds"
            elif not finite:
                undefined[f"{prefix}/{key}"] = self._reasons.get(key, "no_scored_worlds")
        return out


class _EdgePool:
    """Active-slot (probability, truth) pairs pooled over worlds per edge type."""

    def __init__(self):
        self._probs: dict[str, list[np.ndarray]] = {et: [] for et in STRUCT_EDGE_KEYS}
        self._truth: dict[str, list[np.ndarray]] = {et: [] for et in STRUCT_EDGE_KEYS}

    def add(self, et: str, probs: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> None:
        self._probs[et].append(probs[keep])
        self._truth[et].append(truth[keep])

    def finalize(self, prefix: str, undefined: dict[str, str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for et in STRUCT_EDGE_KEYS:
            p = np.concatenate(self._probs[et]) if self._probs[et] else np.empty(0)
            y = np.concatenate(self._truth[et]) if self._truth[et] else np.empty(0)
            out[et] = _edge_record(p, y, f"{prefix}/{et}", undefined)
        return out


def _edge_record(
    p: np.ndarray, y: np.ndarray, prefix: str, undefined: dict[str, str]
) -> dict[str, Any]:
    n = int(y.size)
    n_pos = int(y.sum()) if n else 0
    if n == 0:
        for key in _EDGE_SCORES:
            undefined[f"{prefix}/{key}"] = "no_active_slots"
        return {**dict.fromkeys(_EDGE_SCORES), "n": 0, "n_pos": 0}
    invalid = int((~np.isfinite(p)).sum())
    if invalid:
        for key in _EDGE_SCORES:
            if key != "prevalence":
                undefined[f"{prefix}/{key}"] = f"nonfinite_in_{invalid}_slots"
        # Truth summaries remain defined even when predictions cannot be scored.
        return {
            **dict.fromkeys(_EDGE_SCORES),
            "prevalence": float(y.mean()),
            "n": n,
            "n_pos": n_pos,
        }
    auroc = metrics.auroc(p, y)
    if auroc is None:
        undefined[f"{prefix}/auroc"] = "one_class"
    return {
        "log_score": metrics.log_score(p, y),
        "brier": metrics.brier(p, y),
        "auroc": auroc,
        "ece": metrics.ece(p, y),
        "prevalence": float(y.mean()),
        "n": n,
        "n_pos": n_pos,
    }


def _numpy(value: Tensor) -> np.ndarray:
    return value.detach().to(device="cpu", dtype=torch.float64).numpy()


def _score_decomposition(
    acc: _MeanAccumulator,
    pred: np.ndarray,
    truth: np.ndarray,
    active_c: np.ndarray,
    active_m: np.ndarray,
    outcome: np.ndarray,
    noise: np.ndarray,
    scale: float,
) -> None:
    """Push one world's z-space decomposition metrics.

    ``pred`` is ``[S, T, D]``, ``truth`` ``[T, D]`` with
    ``D = 1 + K + M + 4 = [baseline | K direct | M covariate | confounder_total
    | indirect cc, zc, dc]``; ``outcome``/``noise`` are natural ``[T]`` series.
    """
    n_samples = pred.shape[0]
    k, m = active_c.size, active_m.size
    if pred.shape[1:] != truth.shape or truth.shape[-1] != 1 + k + m + 4:
        raise ValueError(
            f"decomposition draws {pred.shape} must match truth {truth.shape} with "
            f"D = 1 + {k} + {m} + 4"
        )
    contrib = slice(1, 1 + k)
    controls = slice(1 + k, 1 + k + m)
    confounder = 1 + k + m
    indirect = slice(confounder + 1, confounder + 4)

    def family(name: str, p: np.ndarray, t: np.ndarray) -> None:
        acc.push(f"{name}_scaled_mae", float(np.mean(np.abs(p.mean(axis=0) - t))))
        acc.push(f"{name}_crps", metrics.empirical_crps(p, t))
        acc.push(f"{name}_cov80", metrics.coverage(p, t))

    def skip_family(name: str, reason: str) -> None:
        for stat in _FAMILY_STATS:
            acc.skip(f"{name}_{stat}", reason)

    pred_contrib = pred[..., contrib]
    true_contrib = truth[:, contrib]
    pred_controls = pred[..., controls][..., active_m]
    true_controls = truth[:, controls][:, active_m]
    pred_indirect_total = pred[..., indirect].sum(axis=-1)
    true_indirect_total = truth[:, indirect].sum(axis=-1)

    family("baseline_intrinsic", pred[..., 0], truth[:, 0])
    if active_c.any():
        family("contribution", pred_contrib[..., active_c], true_contrib[:, active_c])
    else:
        skip_family("contribution", "no_active_channels")
    if active_m.any():
        family("control_contribution", pred_controls, true_controls)
    else:
        skip_family("control_contribution", "no_active_controls")
    family("confounder_total", pred[..., confounder], truth[:, confounder])
    for i, source in enumerate(INDIRECT_SOURCES):
        column = indirect.start + i
        family(f"indirect_{source}", pred[..., column], truth[:, column])
    family("indirect_total", pred_indirect_total, true_indirect_total)

    inactive_c = ~active_c
    if inactive_c.any():
        leakage = np.abs(pred_contrib[..., inactive_c].mean(axis=0) - true_contrib[:, inactive_c])
        acc.push("contribution_padding_leakage_scaled_mae", float(np.mean(leakage)))
    else:
        acc.skip("contribution_padding_leakage_scaled_mae", "no_inactive_channels")

    # Closure of the posterior-mean decomposition against STRUCTURAL outcome
    # (observed net of iid noise), with the observed magnitude as denominator.
    component_draws = np.concatenate(
        [
            pred[..., :1],
            pred_contrib[..., active_c],
            pred_controls,
            pred[..., confounder : confounder + 1],
            pred_indirect_total[..., None],
        ],
        axis=-1,
    )
    component_truth = np.concatenate(
        [
            truth[:, :1],
            true_contrib[:, active_c],
            true_controls,
            truth[:, confounder : confounder + 1],
            true_indirect_total[:, None],
        ],
        axis=-1,
    )
    draw_sales = component_draws.sum(axis=-1)
    structural = outcome - noise
    safe_outcome = np.maximum(np.abs(outcome), CLOSURE_EPS)
    acc.push(
        "closure_rel_err",
        float(np.mean(np.abs(draw_sales.mean(axis=0) * scale - structural) / safe_outcome)),
    )
    if n_samples < 2:
        for key in _ENERGY_METRICS:
            acc.skip(key, "needs_n_samples_ge_2")
        return
    acc.push("decomposition_energy_score", metrics.energy_score(component_draws, component_truth))
    # Latent (structural) sales in z-space: the summed component truth equals
    # (outcome - noise) / scale by the corpus identity, without float32 round-off.
    acc.push(
        "latent_sales_temporal_energy_score",
        metrics.energy_score(draw_sales, component_truth.sum(axis=-1)),
    )
    if active_c.any():
        acc.push(
            "contribution_temporal_energy_score",
            metrics.energy_score(pred_contrib[..., active_c], true_contrib[:, active_c]),
        )
    else:
        acc.skip("contribution_temporal_energy_score", "no_active_channels")


def _active_edge_masks(batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
    """Boolean per-edge activity with cc/zz diagonals excluded."""
    masks = structure_active_masks(batch)
    if masks is None:
        raise ValueError("evaluation batches must carry active_c/m/j masks")
    out = {et: _numpy(mask).astype(bool) for et, mask in masks.items()}
    for et in ("cc", "zz"):
        out[et] &= ~np.eye(out[et].shape[-1], dtype=bool)
    return out


def evaluate_rows(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    rows: Sequence[tuple[int, int]],
    *,
    n_samples: int,
    seed: int,
    device: str | torch.device,
    leg: str,
    result_sink: ResultSink | None = None,
) -> dict[str, Any]:
    """Score ``rows`` (``(corpus_index, row)`` pairs) on all four heads.

    Batches are built per context length with
    :func:`pymc_mfm.training.stream.batches_for_rows`; sampling runs under a
    forked CPU/CUDA RNG seeded with ``seed`` and explicit MPS state restoration.
    Only the CPU and evaluation device are seeded; all caller RNG states and
    the model's ``training`` flag are untouched on return. Truth comes from the batch
    (``targets_z``, ``context_targets_z``, ``sales_scale``, masks, ``g_*``) and
    from the corpus (``outcome_raw``, ``outcome_noise`` for closure).

    The record is JSON-safe (floats, ints, ``None``, strings):
    ``{"leg", "n_worlds", "n_samples", "seed", "heads": {"joint": {window:
    metrics}, "conditional": {...}, "structure_head": {edge: metrics},
    "joint_router": {"edge": {edge: metrics}, "graph": metrics}}, "undefined":
    {"head/.../metric": reason}, "labels": HEAD_LABELS}``. Metric keys are
    fixed (:data:`DECOMPOSITION_METRICS`, :data:`EDGE_METRICS`,
    :data:`GRAPH_METRICS`); a metric with no finite world is ``None`` and named
    in ``undefined`` with its reason; partially non-finite metrics keep the
    finite mean and report ``nonfinite_in_<k>_worlds``.

    Optional ``result_sink(result, truth, graph_truth, observed)`` captures each
    world's context/query conditional draws already computed here. It never
    samples a second time. Capture conversion failures are explicit under
    ``result_capture`` and do not alter numerical scores; callback failures
    propagate. The callback receives natural-unit truth/observations separately.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not rows:
        raise ValueError("evaluate_rows needs at least one row")
    n_samples = int(n_samples)
    device = torch.device(device)

    decomposition = {
        (head, window): _MeanAccumulator(DECOMPOSITION_METRICS)
        for head in DECOMPOSITION_HEADS
        for window in WINDOWS
    }
    structure_pool = _EdgePool()
    router_pool = _EdgePool()
    graph_acc = _MeanAccumulator(GRAPH_METRICS)
    result_capture: dict[str, Any] = {"captured": 0, "failures": []}

    batches = batches_for_rows(corpora, rows, device=device)
    was_training = model.training
    model.eval()
    try:
        with isolated_rng(device, seed), torch.no_grad():
            for batch, batch_rows in batches:
                _score_batch(
                    model,
                    corpora,
                    batch,
                    batch_rows,
                    n_samples,
                    decomposition,
                    structure_pool,
                    router_pool,
                    graph_acc,
                    result_sink=result_sink,
                    result_capture=result_capture,
                    leg=leg,
                    seed=seed,
                )
    finally:
        model.train(was_training)

    undefined: dict[str, str] = {}
    heads: dict[str, Any] = {
        head: {
            window: decomposition[(head, window)].finalize(f"{head}/{window}", undefined)
            for window in WINDOWS
        }
        for head in DECOMPOSITION_HEADS
    }
    heads["structure_head"] = structure_pool.finalize("structure_head", undefined)
    heads["joint_router"] = {
        "edge": router_pool.finalize("joint_router/edge", undefined),
        "graph": graph_acc.finalize("joint_router/graph", undefined),
    }
    return {
        "leg": str(leg),
        "n_worlds": len(rows),
        "n_samples": n_samples,
        "seed": int(seed),
        "heads": heads,
        "undefined": undefined,
        "labels": dict(HEAD_LABELS),
        **({"result_capture": result_capture} if result_sink is not None else {}),
    }


def _score_batch(
    model: MFM,
    corpora: Sequence[Mapping[str, Any]],
    batch: dict[str, Tensor],
    batch_rows: Sequence[tuple[int, int]],
    n_samples: int,
    decomposition: dict[tuple[str, str], _MeanAccumulator],
    structure_pool: _EdgePool,
    router_pool: _EdgePool,
    graph_acc: _MeanAccumulator,
    *,
    result_sink: ResultSink | None = None,
    result_capture: dict[str, Any] | None = None,
    leg: str = "",
    seed: int = 0,
) -> None:
    active_c = _numpy(batch["active_c_mask"]).astype(bool)
    active_m = _numpy(batch["active_m_mask"]).astype(bool)
    active_j = _numpy(batch["active_j_mask"]).astype(bool)
    n_worlds = active_c.shape[0]
    if len(batch_rows) != n_worlds:
        raise ValueError(f"batch has {n_worlds} worlds but {len(batch_rows)} rows were named")
    layout = EdgeSlotLayout(active_c.shape[1], active_m.shape[1], active_j.shape[1])
    scale = _numpy(batch["sales_scale"])
    n_context = int(batch["context_targets_z"].shape[1])
    truth = {"query": _numpy(batch["targets_z"]), "context": _numpy(batch["context_targets_z"])}
    edge_truth = {et: _numpy(batch[f"g_{et}"]) for et in STRUCT_EDGE_KEYS}
    edge_active = _active_edge_masks(batch)

    weekly, channels, controls = model.encoder.forward_with_controls(batch)
    probs = model.struct_head.predictive_probs(model.structure_logits(weekly, channels, controls))
    probs = {et: _numpy(probs[et]) for et in STRUCT_EDGE_KEYS}

    joint_query, g_flow = model.predict(batch, n_samples, return_structure=True)
    joint_context, _ = model.sample_context(batch, n_samples, return_structure=True)
    conditional_query, query_graphs = model.sample_disentangled(
        batch, n_samples, mode="model_average", window="query"
    )
    conditional_context, context_graphs = model.sample_disentangled(
        batch, n_samples, mode="model_average", window="context"
    )
    draws = {
        ("joint", "query"): _numpy(joint_query),
        ("joint", "context"): _numpy(joint_context),
        ("conditional", "query"): _numpy(conditional_query),
        ("conditional", "context"): _numpy(conditional_context),
    }
    conditional_graphs = (
        None
        if result_sink is None
        else {
            "query": _numpy(query_graphs),
            "context": _numpy(context_graphs),
        }
    )

    graphs = _numpy(binary_graphs_from_flow(g_flow))  # [S, B, L]
    # Thresholding NaN/Inf flow values would silently turn them into binary edges.
    finite_graph_slots = torch.isfinite(g_flow).all(dim=2).detach().cpu().numpy()
    graphs[~finite_graph_slots] = np.nan
    if graphs.shape[-1] != layout.n_slots:
        raise ValueError(
            f"joint router emitted {graphs.shape[-1]} slots but the geometry has {layout.n_slots}"
        )
    true_flat = np.asarray(layout.pack(edge_truth), dtype=np.float64)  # [B, L]
    slot_active = np.asarray(layout.active_slot_mask(active_c, active_m, active_j)).astype(bool)
    marginals = layout.unpack(graphs.mean(axis=0))  # blocks [B, ...]

    window_slices = {"query": slice(n_context, None), "context": slice(None, n_context)}
    for b, (corpus_index, row) in enumerate(batch_rows):
        corpus = corpora[corpus_index]
        outcome = np.asarray(corpus["outcome_raw"][row], dtype=np.float64)
        noise = np.asarray(corpus["outcome_noise"][row], dtype=np.float64)
        capture_provenance = None
        if result_sink is not None:
            input_support = {}
            for key in ("ctx_spend_z", "qry_spend_z", "ctx_controls_z", "qry_controls_z"):
                values = _numpy(batch[key][b])
                input_support[key] = {
                    "units": "encoded context-normalized representation",
                    "n": len(values),
                    "mean": values.mean(axis=0).tolist(),
                    "std": values.std(axis=0).tolist(),
                    "min": values.min(axis=0).tolist(),
                    "max": values.max(axis=0).tolist(),
                }
            for key, field in (("channels", "treatment_raw"), ("controls", "covariates")):
                raw = np.asarray(corpus[field][row], dtype=np.float64)
                for window, span in window_slices.items():
                    values = raw[span]
                    input_support[f"{window}_{key}_raw"] = {
                        "units": "native observed input units",
                        "n": len(values),
                        "mean": values.mean(axis=0).tolist(),
                        "std": values.std(axis=0).tolist(),
                        "min": values.min(axis=0).tolist(),
                        "max": values.max(axis=0).tolist(),
                    }
            context_observed = outcome[:n_context]
            input_support["context_outcome_raw"] = {
                "units": "outcome",
                "n": n_context,
                "mean": float(context_observed.mean()),
                "std": float(context_observed.std()),
                "min": float(context_observed.min()),
                "max": float(context_observed.max()),
            }
            capture_provenance = {
                "data_split": str(leg),
                "corpus_index": int(corpus_index),
                "row": int(row),
                "world_id": f"corpus:{corpus_index}/row:{row}",
                "data_sha256": observation_identity(batch, world=b),
                "seed": int(seed),
                "input_support": input_support,
                "capture": "existing conditional evaluator samples; no additional inference",
            }
        for (head, window), samples in draws.items():
            span = window_slices[window]
            _score_decomposition(
                decomposition[(head, window)],
                samples[:, b],
                truth[window][b],
                active_c[b],
                active_m[b],
                outcome[span],
                noise[span],
                float(scale[b]),
            )
            if result_sink is not None and head == "conditional":
                from pymc_mfm.inference.result import result_from_batch

                try:
                    captured = result_from_batch(
                        batch,
                        samples,
                        conditional_graphs[window],
                        model_config=model.config,
                        world=b,
                        window=window,
                        mode="model_average",
                        provenance=capture_provenance,
                    )
                except (ValueError, TypeError) as exc:
                    if result_capture is None:
                        raise
                    result_capture["failures"].append(
                        {
                            "corpus_index": int(corpus_index),
                            "row": int(row),
                            "window": window,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                else:
                    result_sink(
                        captured,
                        truth[window][b] * float(scale[b]),
                        true_flat[b].copy(),
                        outcome[span].copy(),
                    )
                    if result_capture is not None:
                        result_capture["captured"] += 1
        for et in STRUCT_EDGE_KEYS:
            keep = edge_active[et][b].ravel()
            y = edge_truth[et][b].ravel()
            structure_pool.add(et, probs[et][b].ravel(), y, keep)
            router_pool.add(et, np.asarray(marginals[et][b]).ravel(), y, keep)
        keep = slot_active[b]
        if not keep.any():
            for key in GRAPH_METRICS:
                graph_acc.skip(key, "no_active_slots")
            continue
        active_draws = graphs[:, b, keep]
        active_truth = true_flat[b, keep]
        if not np.isfinite(active_draws).all():
            for key in GRAPH_METRICS:
                graph_acc.push(key, float("nan"))
            continue
        graph_acc.push("shd_mean", float(np.abs(active_draws - active_truth).sum(axis=1).mean()))
        graph_acc.push("exact_recovery", float((active_draws == active_truth).all(axis=1).mean()))


def flatten_evaluation(record: Mapping[str, Any]) -> dict[str, float | None]:
    """Flatten a record to ``"{leg}/{head}/.../{metric}"`` scalar keys.

    Key shapes: ``{leg}/{joint|conditional}/{window}/{metric}``,
    ``{leg}/structure_head/{edge}/{metric}``,
    ``{leg}/joint_router/edge/{edge}/{metric}`` and
    ``{leg}/joint_router/graph/{metric}``. Integer counts become floats;
    undefined metrics stay ``None``. ``/`` is the only separator (MLflow-safe).
    """
    leg = record["leg"]
    heads = record["heads"]
    out: dict[str, float | None] = {}

    def put(prefix: str, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            out[f"{prefix}/{key}"] = None if value is None else float(value)

    for head in DECOMPOSITION_HEADS:
        for window, values in heads[head].items():
            put(f"{leg}/{head}/{window}", values)
    for et, values in heads["structure_head"].items():
        put(f"{leg}/structure_head/{et}", values)
    for et, values in heads["joint_router"]["edge"].items():
        put(f"{leg}/joint_router/edge/{et}", values)
    put(f"{leg}/joint_router/graph", heads["joint_router"]["graph"])
    return out
