"""Uncertainty of a finite distribution of aligned graph/quantity draws."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def paired_graph_uncertainty(
    graph_draws: np.ndarray,
    quantity_samples: np.ndarray,
    *,
    selected_graph: np.ndarray | None = None,
    covariance: bool = False,
    rare_count: int = 5,
) -> dict[str, object]:
    """Decompose empirical variance using population (``ddof=0``) moments.

    Inputs have shapes ``[S,L]`` and ``[S,...]`` with the SAME draw ordering.
    Graphs are whole binary vectors, never products of edge marginals. Empty
    active graph geometry is a single deterministic empty graph. Singleton
    groups have zero empirical within-group variance, not certain posteriors.
    Full covariance is optional and only accepts scalar/vector aggregates.
    """
    graphs = np.asarray(graph_draws)
    samples = np.asarray(quantity_samples, dtype=np.float64)
    if graphs.ndim != 2 or graphs.shape[0] < 1 or not np.isin(graphs, (0, 1)).all():
        raise ValueError("graph_draws must have shape [S,L] and contain binary graphs")
    if samples.ndim < 1 or samples.shape[0] != len(graphs) or samples.size == 0:
        raise ValueError("quantity_samples must have nonempty shape [S,...] aligned with graphs")
    if not np.isfinite(samples).all():
        raise ValueError("quantity_samples must be finite")
    if covariance and samples.ndim > 2:
        raise ValueError("covariance requires scalar/vector aggregates, not trajectories")
    if isinstance(rare_count, bool) or not isinstance(rare_count, Integral) or rare_count < 1:
        raise ValueError("rare_count must be a positive integer")

    if graphs.shape[1]:
        unique, inverse, counts = np.unique(
            graphs.astype(np.uint8), axis=0, return_inverse=True, return_counts=True
        )
    else:
        unique = np.empty((1, 0), dtype=np.uint8)
        inverse = np.zeros(len(graphs), dtype=int)
        counts = np.array([len(graphs)])
    n = len(samples)
    shape = samples.shape[1:]
    flat = samples.reshape(n, -1)
    frequencies = counts / n
    means = np.zeros((len(unique), flat.shape[1]), dtype=np.float64)
    np.add.at(means, inverse, flat)
    means /= counts[:, None]
    mean = flat.mean(axis=0)
    residuals = flat - means[inverse]
    group_variances = np.zeros_like(means)
    np.add.at(group_variances, inverse, residuals * residuals)
    group_variances /= counts[:, None]
    total = np.mean((flat - mean) ** 2, axis=0)
    within = frequencies @ group_variances
    between = frequencies @ ((means - mean) ** 2)

    mode_indices = np.flatnonzero(counts == counts.max())
    mode = int(mode_indices[0])
    selected = unique[mode] if selected_graph is None else np.asarray(selected_graph)
    if selected.shape != (graphs.shape[1],) or not np.isin(selected, (0, 1)).all():
        raise ValueError("selected_graph must be a binary vector with shape [L]")
    matched = np.flatnonzero(np.all(unique == selected, axis=1))

    def shaped(values):
        return values.reshape(shape).tolist()

    selected_summary = {
        "graph": selected.tolist(),
        "source": "empirical_full_graph_mode" if selected_graph is None else "explicit",
        "status": "observed" if matched.size else "absent",
        "count": 0,
        "frequency": 0.0,
        "mean": None,
        "sd": None,
    }
    if matched.size:
        index = int(matched[0])
        selected_summary.update(
            count=int(counts[index]),
            frequency=float(frequencies[index]),
            mean=shaped(means[index]),
            sd=shaped(np.sqrt(group_variances[index])),
        )
    rare = counts < rare_count
    report = {
        "estimand": "empirical_paired_draw_uncertainty",
        "ddof": 0,
        "n_draws": n,
        "quantity_shape": list(shape),
        "n_unique_graphs": len(unique),
        "graph_frequencies": [
            {"graph": graph.tolist(), "count": int(count), "frequency": float(frequency)}
            for graph, count, frequency in zip(unique, counts, frequencies, strict=True)
        ],
        "mode_graph": unique[mode].tolist(),
        "mode_count": int(counts[mode]),
        "mode_frequency": float(frequencies[mode]),
        "tied_mode_graphs": unique[mode_indices].tolist(),
        "mode_tie_rule": "lexicographic_first",
        "mean": shaped(mean),
        "group_means": means.reshape((len(unique), *shape)).tolist(),
        "group_variances": group_variances.reshape((len(unique), *shape)).tolist(),
        "total_variance": shaped(total),
        "within_graph_variance": shaped(within),
        "between_graph_variance": shaped(between),
        "variance_closure_error": shaped(total - within - between),
        "selected_group": selected_summary,
        "rare_count_threshold": int(rare_count),
        "n_rare_groups": int(rare.sum()),
        "rare_draw_mass": float(frequencies[rare].sum()),
        "n_singleton_groups": int((counts == 1).sum()),
        "caveats": [
            "Finite paired-draw ddof=0 decomposition, not a conditional calibration measurement.",
            "Unseen graphs and empirical mode selection have Monte Carlo uncertainty.",
            "Rare groups are poorly resolved; singleton sd=0 is not posterior certainty.",
        ],
    }
    if not matched.size:
        report["caveats"].append("Selected graph was not drawn; conditional moments unavailable.")
    if covariance:
        centered = flat - mean
        centered_means = means - mean
        total_covariance = centered.T @ centered / n
        within_covariance = residuals.T @ residuals / n
        between_covariance = centered_means.T @ (frequencies[:, None] * centered_means)
        report.update(
            total_covariance=total_covariance.tolist(),
            within_graph_covariance=within_covariance.tolist(),
            between_graph_covariance=between_covariance.tolist(),
            covariance_closure_error=(
                total_covariance - within_covariance - between_covariance
            ).tolist(),
        )
    return report
