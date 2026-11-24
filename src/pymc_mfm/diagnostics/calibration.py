"""Saved-draw calibration diagnostics with explicit independence qualifications."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
SCREENING_QUALIFICATION = (
    "Component/time ranks in one world are dependent screening diagnostics, "
    "not independent-world SBC evidence. No calibrated pass/fail is assigned."
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _ranks(ranks, n_post_samples):
    n_post_samples = _positive_int(n_post_samples, "n_post_samples")
    values = np.asarray(ranks, dtype=np.float64).ravel()
    if (
        not values.size
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or np.any((values < 0) | (values > n_post_samples))
    ):
        raise ValueError(f"ranks must be nonempty integers in [0,{n_post_samples}]")
    return values.astype(np.int64)


def randomized_ranks(samples, truth, *, seed=0, rng=None):
    """Rank each truth coordinate among aligned draws, randomizing exact ties.

    Draws are ``[S,...]``, truth exactly ``[...]``. The inclusive tie offset is
    uniform on ``0,...,ties``; zeros are atoms, not values to jitter or drop.
    An explicitly supplied generator is advanced; otherwise only a local RNG
    is used and ``seed`` is retained by the calling report.
    """
    values = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    if values.ndim < 1 or len(values) < 1 or values.shape[1:] != targets.shape:
        raise ValueError("truth shape must exactly match samples without their draw axis")
    if targets.size == 0 or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("samples and truth must be nonempty and finite")
    rng = np.random.default_rng(seed) if rng is None else rng
    below = np.sum(values < targets, axis=0)
    ties = np.sum(values == targets, axis=0)
    return below + rng.integers(0, ties + 1, size=targets.shape)


def compute_rank(true_value, posterior_samples, rng=None):
    """Randomized scalar rank, including the upper endpoint when all draws tie."""
    samples = np.asarray(posterior_samples, dtype=np.float64).ravel()
    return int(randomized_ranks(samples, np.asarray(true_value), rng=rng))


def coverage_from_ranks(ranks, n_post_samples, levels=COVERAGE_LEVELS):
    """Central coverage of discrete mid-PIT, distinct from sample quantile coverage."""
    values = _ranks(ranks, n_post_samples)
    levels = np.asarray(levels, dtype=np.float64)
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all():
        raise ValueError("levels must be a nonempty finite vector")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("levels must lie in (0,1)")
    pit = (values + 0.5) / (n_post_samples + 1)
    return {
        f"cov{100 * level:g}": float(np.mean(np.abs(pit - 0.5) <= level / 2)) for level in levels
    }


def _ks(counts, n, grid):
    cumulative = np.cumsum(counts, axis=-1)
    return np.maximum(
        np.max(cumulative / n - grid, axis=-1),
        np.max(grid - (cumulative - counts) / n, axis=-1),
    )


def rank_uniformity_stats(
    ranks,
    n_post_samples,
    *,
    independent_worlds=False,
    n_bins=20,
    simulations=10000,
    seed=0,
    alpha=0.05,
):
    """Rank histogram/PIT/ECDF; null tests require independent-world ranks.

    The KS null is simulated under the exact discrete rank law rather than a
    continuous asymptotic law. Simultaneous ECDF bands use binomial quantiles
    with Monte Carlo calibrated pointwise level. No inference is performed.
    """
    values = _ranks(ranks, n_post_samples)
    n_bins = min(_positive_int(n_bins, "n_bins"), n_post_samples + 1)
    simulations = _positive_int(simulations, "simulations")
    if not isinstance(independent_worlds, bool):
        raise ValueError("independent_worlds must be an explicit boolean")
    if not 0 < alpha < 1 or not np.isfinite(alpha):
        raise ValueError("alpha must lie in (0,1)")
    edges = np.linspace(0, n_post_samples + 1, n_bins + 1)
    histogram = np.histogram(values, bins=edges)[0]
    expected = len(values) * np.histogram(np.arange(n_post_samples + 1), bins=edges)[0]
    expected = expected / (n_post_samples + 1)
    counts = np.bincount(values, minlength=n_post_samples + 1)
    pit_grid = (np.arange(n_post_samples + 1) + 0.5) / (n_post_samples + 1)
    null_cdf = (np.arange(n_post_samples + 1) + 1) / (n_post_samples + 1)
    statistic = float(_ks(counts, len(values), pit_grid))
    report = {
        "ranks": values.tolist(),
        "pit": ((values + 0.5) / (n_post_samples + 1)).tolist(),
        "rank_histogram": histogram.tolist(),
        "rank_bin_edges": edges.tolist(),
        "expected_rank_histogram": expected.tolist(),
        "ecdf_grid": pit_grid.tolist(),
        "ecdf": (np.cumsum(counts) / len(values)).tolist(),
        "null_ecdf": null_cdf.tolist(),
        "ks": statistic,
        "chi2": float(np.sum((histogram - expected) ** 2 / expected)),
        "ks_pvalue": None,
        "chi2_pvalue": None,
        "ecdf_band_reject": None,
        "pot_c_pvalue": None,
        "pot_c_status": {"status": "unavailable", "reason": SCREENING_QUALIFICATION},
        "rank_coverage": coverage_from_ranks(values, n_post_samples),
        "n_ranks": len(values),
        "n_post_samples": n_post_samples,
        "seed": int(seed),
        "alpha": float(alpha),
        "independent_worlds": bool(independent_worlds),
        "qualification": SCREENING_QUALIFICATION,
        "status": "screening_only",
    }
    if not independent_worlds:
        return report
    if simulations < 100:
        raise ValueError("independent-world null tests need at least 100 simulations")
    from scipy.stats import binom, chi2

    rng = np.random.default_rng(seed)
    null_counts = rng.multinomial(
        len(values), np.full(n_post_samples + 1, 1 / (n_post_samples + 1)), size=simulations
    )
    null_statistic = _ks(null_counts, len(values), pit_grid)
    null_cumulative = np.cumsum(null_counts, axis=1)[:, :-1]
    probabilities = null_cdf[:-1]
    low_gamma, high_gamma = 1e-9, alpha
    for _ in range(40):
        gamma = np.sqrt(low_gamma * high_gamma)
        lower = binom.ppf(gamma / 2, len(values), probabilities)
        upper = binom.ppf(1 - gamma / 2, len(values), probabilities)
        cover = np.mean(np.all((null_cumulative >= lower) & (null_cumulative <= upper), axis=1))
        if cover < 1 - alpha:
            high_gamma = gamma
        else:
            low_gamma = gamma
    lower = binom.ppf(low_gamma / 2, len(values), probabilities) / len(values)
    upper = binom.ppf(1 - low_gamma / 2, len(values), probabilities) / len(values)
    actual = np.cumsum(counts)[:-1] / len(values)
    excursion = float(np.maximum(np.maximum(lower - actual, actual - upper), 0).max())
    report.update(
        ks_pvalue=float((1 + np.sum(null_statistic >= statistic)) / (1 + simulations)),
        chi2_pvalue=float(chi2.sf(report["chi2"], n_bins - 1)) if n_bins > 1 else None,
        ecdf_lower=lower.tolist() + [1.0],
        ecdf_upper=upper.tolist() + [1.0],
        ecdf_band_reject=float(excursion > 0),
        ecdf_max_excursion=excursion,
        ecdf_gamma=float(low_gamma),
        simulations=simulations,
        status="independent_worlds_declared",
        qualification=(
            "Independent worlds are a caller-supplied sampling contract, not inferred from array "
            "shape. Monte Carlo null tests have finite resolution; chi-square p-values are "
            "asymptotic and may be unreliable with small expected bin counts. A non-rejection "
            "does not prove calibration or out-of-distribution validity."
        ),
    )

    from arviz_stats.base import array_stats

    jitter = np.random.default_rng(20250211).random(len(values))
    randomized_pit = (values + jitter) / (n_post_samples + 1)
    value = float(array_stats.uniformity_test(randomized_pit)[0])
    if not np.isfinite(value):
        raise ValueError("pot_c uniformity test returned a nonfinite value")
    report["pot_c_pvalue"] = value
    report["pot_c_status"] = {
        "status": "graded_diagnostic",
        "reason": (
            "Order-statistic uniformity test is not a calibrated gate; "
            "finite-sample size can exceed nominal"
        ),
    }
    return report


def calibration_summary(samples, truth, *, seed=0, levels=COVERAGE_LEVELS):
    """One-world screening ranks and actual central-quantile interval coverage."""
    values = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    ranks = randomized_ranks(values, targets, seed=seed)
    report = rank_uniformity_stats(ranks, len(values), seed=seed)
    intervals = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("coverage levels must lie in (0,1)")
        low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2], axis=0)
        intervals.append(
            {
                "level": float(level),
                "coverage": float(np.mean((targets >= low) & (targets <= high))),
                "mean_width": float(np.mean(high - low)),
            }
        )
    report["intervals"] = intervals
    report["rank_shape"] = list(ranks.shape)
    report["interval_definition"] = (
        "central equal-tail, NumPy linear quantiles, inclusive endpoints"
    )
    return report


def sbc_report(
    samples,
    truth,
    *,
    world_ids: Sequence[str],
    quantity_names: Sequence[str],
    provenance: Mapping[str, object],
    seed=0,
    alpha=0.05,
    simulations=10000,
):
    """Report independent-world SBC from SAVED ``[W,S,Q]`` posterior draws.

    ``truth`` is ``[W,Q]``. Each quantity must supply exactly one statistic per
    independent simulated world (aggregate trajectories before calling).
    Explicit unique world IDs and prior/data provenance are required. Cross-
    quantity family-wise control uses Bonferroni, not an independence assumption.
    """
    values = np.asarray(samples, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] < 1:
        raise ValueError("SBC samples must have shape [W>=2,S,Q>=1]")
    if targets.shape != (values.shape[0], values.shape[2]):
        raise ValueError("SBC truth must have shape [W,Q]")
    worlds = tuple(world_ids)
    quantities = tuple(quantity_names)
    if (
        len(worlds) != len(values)
        or not all(isinstance(name, str) and name for name in worlds)
        or len(set(worlds)) != len(worlds)
    ):
        raise ValueError("world_ids must uniquely identify every independent world")
    if (
        len(quantities) != values.shape[2]
        or not all(isinstance(name, str) and name for name in quantities)
        or len(set(quantities)) != len(quantities)
    ):
        raise ValueError("quantity_names must uniquely identify every quantity")
    if not provenance or provenance.get("independent_worlds") is not True:
        raise ValueError("provenance must explicitly declare independent_worlds=True")
    identity = provenance.get("data_sha256")
    if (
        not provenance.get("prior")
        or not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("SBC provenance requires a prior description and data_sha256 identity")
    rng = np.random.default_rng(seed)
    ranks = np.stack(
        [
            randomized_ranks(draws, target, rng=rng)
            for draws, target in zip(values, targets, strict=True)
        ]
    )
    per_quantity = {
        name: rank_uniformity_stats(
            ranks[:, index],
            values.shape[1],
            independent_worlds=True,
            seed=seed,
            alpha=alpha / len(quantities),
            simulations=simulations,
        )
        for index, name in enumerate(quantities)
    }
    return {
        "format_version": 1,
        "kind": "independent_world_sbc",
        "provenance": dict(provenance),
        "world_ids": list(worlds),
        "quantity_names": list(quantities),
        "seed": int(seed),
        "ranks": ranks.tolist(),
        "quantities": per_quantity,
        "family_wise": {
            "method": "Bonferroni",
            "alpha": float(alpha),
            "alpha_per_quantity": float(alpha / len(quantities)),
            "rejecting_quantities": [
                name for name, record in per_quantity.items() if record["ecdf_band_reject"]
            ],
        },
        "qualification": "A non-rejection is not proof of calibration; prior/data support applies.",
    }


def _continuous_rank_stats(rows, view, *, alpha, seed, simulations):
    """Uniform randomized PIT from ranks with possibly different draw counts."""
    available = [row for row in rows if row is not None and row[view] is not None]
    if len(available) < 2:
        return None
    from scipy.stats import binom, kstest

    rng = np.random.default_rng(seed)
    pairs = [row[view] for row in available]
    ranks = np.asarray([rank for rank, _, _ in pairs], dtype=np.int64)
    counts = np.asarray([count for _, count, _ in pairs], dtype=np.int64)
    full_counts = np.asarray([count == total for _, count, total in pairs], dtype=bool)
    pit = (ranks + rng.random(len(pairs))) / (counts + 1)
    grid = np.linspace(0, 1, 101)
    ecdf = np.searchsorted(np.sort(pit), grid, side="right") / len(pit)
    radius = float(np.sqrt(np.log(2 / alpha) / (2 * len(pit))))
    edges = np.linspace(0, 1, 21)
    histogram = np.histogram(pit, bins=edges)[0]
    uniformity = kstest(pit, "uniform")
    full = {}
    for count in sorted(set(counts[full_counts])):
        selected = full_counts & (counts == count)
        if selected.sum() >= 2:
            discrete = rank_uniformity_stats(
                ranks[selected],
                int(count),
                independent_worlds=True,
                alpha=alpha,
                seed=seed,
                simulations=simulations,
            )
            # Each discrete cumulative rank count is binomial under the
            # exchangeable null. A union bound over both tails at every cut
            # gives a valid finite-sample simultaneous band, even when the
            # legacy Monte Carlo band cannot resolve the adjusted level.
            n = int(selected.sum())
            cut_probabilities = np.arange(1, int(count) + 1) / (count + 1)
            tail = alpha / (2 * int(count))
            lower = binom.ppf(tail, n, cut_probabilities) / n
            upper = binom.ppf(1 - tail, n, cut_probabilities) / n
            observed = np.asarray(discrete["ecdf"][:-1])
            excursion = float(np.maximum(np.maximum(lower - observed, observed - upper), 0).max())
            discrete.update(
                ecdf_lower=lower.tolist() + [1.0],
                ecdf_upper=upper.tolist() + [1.0],
                ecdf_band_reject=int(excursion > 0),
                ecdf_max_excursion=excursion,
                ecdf_gamma=float(tail),
                qualification=(
                    discrete["qualification"]
                    + " This card's simultaneous ECDF decision instead uses exact binomial "
                    "quantiles and a two-tail union bound over all discrete rank cuts."
                ),
            )
            full[str(int(count))] = discrete
    return {
        "ranks": ranks.tolist(),
        "world_ids": [row["world_id"] for row in available],
        "cell_ids": [row["cell_id"] for row in available],
        "entity_names": [row["name"] for row in available],
        "n_on": counts.tolist(),
        "pit": pit.tolist(),
        "rank_histogram": histogram.tolist(),
        "rank_bin_edges": edges.tolist(),
        "expected_rank_histogram": [len(pit) / 20] * 20,
        "ecdf_grid": grid.tolist(),
        "ecdf": ecdf.tolist(),
        "null_ecdf": grid.tolist(),
        "ecdf_lower": np.maximum(0, grid - radius).tolist(),
        "ecdf_upper": np.minimum(1, grid + radius).tolist(),
        "rank_coverage": {
            f"cov{100 * level:g}": float(np.mean(np.abs(pit - 0.5) <= level / 2))
            for level in COVERAGE_LEVELS
        },
        "n_ranks": len(pit),
        "n_post_samples": None if len(set(counts)) != 1 else int(counts[0]),
        "ks_pvalue": float(uniformity.pvalue),
        "ecdf_band_reject": int(uniformity.statistic > radius),
        "exact_full_draw_count": full,
        "status": "independent_cells_declared",
        "qualification": (
            "One effect per generator cell; continuous randomized PIT pools varying gate-on "
            "counts. The simultaneous ECDF band uses the conservative DKW inequality."
        ),
    }


def conditional_sbc_report(
    records,
    *,
    world_ids: Sequence[str],
    cell_ids: Sequence[tuple[str | int, str | int]],
    provenance: Mapping[str, object],
    seed=0,
    alpha=0.05,
    simulations=1000,
):
    """Conditional live-effect and graph-averaged SBC from saved evaluation captures.

    ``records`` are evaluator ``(result, truth, graph_truth, observed)`` tuples,
    all from one window. ``cell_ids`` identify ``(corpus, cell_id)`` pairs; only
    the first world of each cell enters formal tests. A family contributes at
    most one selected effect per cell. Dead truths enter graph-averaged ranks
    but never gate-on ranks; signed live truths are retained without filtering.
    Neither view establishes out-of-distribution calibration.
    """
    records = tuple(records)
    worlds, cells = tuple(world_ids), tuple(cell_ids)
    if len(records) < 2 or len(worlds) != len(records) or len(cells) != len(records):
        raise ValueError("records, world_ids and cell_ids must align with at least two worlds")
    if not all(isinstance(name, str) and name for name in worlds) or len(set(worlds)) != len(
        worlds
    ):
        raise ValueError("world_ids must uniquely identify every world")
    if any(
        not isinstance(cell, (tuple, list))
        or len(cell) != 2
        or any(not isinstance(part, (str, Integral)) or isinstance(part, bool) for part in cell)
        for cell in cells
    ):
        raise ValueError("cell_ids must contain (corpus, cell_id) pairs")
    cells = tuple(
        tuple(int(part) if isinstance(part, Integral) else part for part in cell) for cell in cells
    )
    if not isinstance(provenance, Mapping) or provenance.get("independent_worlds") is not True:
        raise ValueError("provenance must explicitly declare independent_worlds=True")
    identity = provenance.get("data_sha256")
    if (
        not provenance.get("prior")
        or not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("SBC provenance requires a prior description and data_sha256 identity")
    from .core import json_safe

    try:
        safe_provenance = json_safe(provenance)
        json.dumps(
            {**safe_provenance, "world_ids": worlds, "cell_ids": cells}, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("SBC provenance and IDs must be strict JSON values") from exc
    if not 0 < alpha < 1 or not np.isfinite(alpha):
        raise ValueError("alpha must lie in (0,1)")
    if _positive_int(simulations, "simulations") < 100:
        raise ValueError("independent-cell null tests need at least 100 simulations")

    selected = []
    seen_cells = set()
    for index, cell in enumerate(cells):
        key = tuple(cell)
        if key not in seen_cells:
            selected.append(index)
            seen_cells.add(key)
    if len(selected) < 2:
        raise ValueError("at least two independent generator cells are required")

    rng = np.random.default_rng(seed)
    selection_rng = np.random.default_rng(int(seed) + 1)
    family_rows = {family: [] for family in ("channel", "control")}
    entity_rows = {family: {} for family in family_rows}
    reasons = (
        "dead_truth_edge",
        "fewer_than_two_edge_on_draws",
        "padded",
        "not_model_average",
    )
    skip_counts = {family: dict.fromkeys(reasons, 0) for family in family_rows}
    entity_skips = {family: {} for family in family_rows}
    mode_applicable = False
    window = units = None
    for index in selected:
        record = records[index]
        if not isinstance(record, (tuple, list)) or len(record) != 4:
            raise ValueError("records must contain (result, truth, graph_truth, observed)")
        result, truth, graph_truth, _ = record
        mode_applicable |= result.mode == "model_average"
        if window is None:
            window, units = result.window, result.units
        elif (result.window, result.units) != (window, units):
            raise ValueError("context and query windows or incompatible units cannot be pooled")
        targets = np.asarray(truth, dtype=np.float64)
        edges = np.asarray(graph_truth, dtype=np.float64)
        if (
            targets.shape != result.components.shape[1:]
            or edges.shape != (result.graphs.shape[1],)
            or not np.isfinite(targets).all()
            or not np.isin(edges, (0, 1)).all()
        ):
            raise ValueError("truth and graph_truth must align with finite saved result axes")
        families = result.provenance.get("edge_families")
        endpoints = result.provenance.get("edge_endpoints")
        if (
            not isinstance(families, list)
            or not isinstance(endpoints, list)
            or len(families) != result.graphs.shape[1]
            or len(endpoints) != len(families)
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in endpoints)
        ):
            raise ValueError("saved result requires aligned edge_families and edge_endpoints")
        for family, edge_family in (("channel", "cy"), ("control", "zb")):
            eligible = []
            for component, (name, kind) in enumerate(
                zip(result.component_names, result.component_families, strict=True)
            ):
                if kind != family:
                    continue
                skips = entity_skips[family].setdefault(name, dict.fromkeys(reasons, 0))
                entity = entity_rows[family].setdefault(name, [])
                if not result.component_mask[component]:
                    skip_counts[family]["padded"] += 1
                    skips["padded"] += 1
                    continue
                slots = [
                    slot
                    for slot, (edge, pair) in enumerate(zip(families, endpoints, strict=True))
                    if edge == edge_family and pair[0] == name and result.edge_mask[slot]
                ]
                if len(slots) != 1:
                    raise ValueError(f"{name} has no unique active {edge_family} gate")
                slot = slots[0]
                draws = result.components[:, :, component].sum(axis=1)
                target = float(targets[:, component].sum())
                row = {
                    "world_id": worlds[index],
                    "cell_id": list(cells[index]),
                    "name": name,
                    "model_average": None,
                    "conditional": None,
                }
                if result.mode != "model_average":
                    skip_counts[family]["not_model_average"] += 1
                    skips["not_model_average"] += 1
                else:
                    average_rank = int(randomized_ranks(draws, np.asarray(target), rng=rng))
                    row["model_average"] = (average_rank, result.n_samples, result.n_samples)
                    if edges[slot] == 0:
                        skip_counts[family]["dead_truth_edge"] += 1
                        skips["dead_truth_edge"] += 1
                    else:
                        on = result.graphs[:, slot] == 1
                        n_on = int(on.sum())
                        if n_on < 2:
                            skip_counts[family]["fewer_than_two_edge_on_draws"] += 1
                            skips["fewer_than_two_edge_on_draws"] += 1
                        else:
                            rank = int(randomized_ranks(draws[on], np.asarray(target), rng=rng))
                            row["conditional"] = (rank, n_on, result.n_samples)
                entity.append(row)
                eligible.append(row)
            applicable = [row for row in eligible if row["model_average"] is not None]
            if applicable:
                # Select once without consulting true gates or edge-on counts:
                # reselecting among live edges biases the family rank when
                # effect magnitudes depend on other edges' truth.
                chosen = applicable[int(selection_rng.integers(len(applicable)))]
                family_rows[family].append({"model_average": chosen, "conditional": chosen})
    # Count every formal view, including each eligible fixed-draw discrete
    # subset; otherwise its independent ECDF decision evades family-wise
    # multiplicity control.
    n_tests = 2 * sum(1 + len(rows) for rows in entity_rows.values())
    for family, rows_by_name in entity_rows.items():
        for view in ("conditional", "model_average"):
            groups = ([entry[view] for entry in family_rows[family]], *rows_by_name.values())
            for rows in groups:
                full_counts = {}
                for row in rows:
                    if row is None or row[view] is None:
                        continue
                    _, count, total = row[view]
                    if count == total:
                        full_counts[count] = full_counts.get(count, 0) + 1
                n_tests += sum(n >= 2 for n in full_counts.values())
    adjusted = alpha / n_tests
    if adjusted < 1e-9:
        raise ValueError("alpha is too small for family-wise discrete ECDF calibration")

    def summarize(rows, view):
        return _continuous_rank_stats(
            rows, view, alpha=adjusted, seed=seed, simulations=simulations
        )

    def collect_rejections(stats, path):
        if stats is None:
            return
        if stats["ecdf_band_reject"]:
            rejecting.append(path)
        for count, subset in stats["exact_full_draw_count"].items():
            if subset["ecdf_band_reject"]:
                rejecting.append(f"{path}/exact_full_draw_count/{count}")

    report_families = {}
    rejecting = []
    for family in family_rows:
        family_data = {}
        for view in ("conditional", "model_average"):
            rows = [entry[view] for entry in family_rows[family]]
            stats = summarize(rows, view)
            family_data[view] = stats
            collect_rejections(stats, f"{family}/{view}")
        family_data["entities"] = {}
        for name, rows in entity_rows[family].items():
            data = {view: summarize(rows, view) for view in ("conditional", "model_average")}
            data["skip_reasons"] = entity_skips[family][name]
            data["status"] = (
                "independent_cells_declared"
                if data["conditional"] is not None or data["model_average"] is not None
                else "not_applicable"
                if not mode_applicable
                else "unavailable"
            )
            family_data["entities"][name] = data
            for view in ("conditional", "model_average"):
                collect_rejections(data[view], f"{family}/{name}/{view}")
        family_data["skip_reasons"] = skip_counts[family]
        family_data["status"] = (
            "independent_cells_declared"
            if family_data["conditional"] is not None or family_data["model_average"] is not None
            else "not_applicable"
            if not mode_applicable
            else "unavailable"
        )
        report_families[family] = family_data
    return {
        "format_version": 1,
        "kind": "independent_world_conditional_sbc",
        "provenance": {
            **safe_provenance,
            "window": window,
            "units": units,
            "mode": "model_average" if mode_applicable else "not_applicable",
        },
        "world_ids": [worlds[index] for index in selected],
        "cell_ids": [list(cells[index]) for index in selected],
        "seed": int(seed),
        "alpha": float(alpha),
        "families": report_families,
        "family_wise": {
            "method": "Bonferroni",
            "alpha": float(alpha),
            "alpha_per_quantity": float(adjusted),
            "rejecting_quantities": rejecting,
        },
        "qualification": (
            "True-live conditional magnitude and unconditional graph-averaged ranks test "
            "different estimands. One world per generator cell and one effect per family/cell "
            "enter formal tests; non-rejection is not proof of calibration or OOD validity."
        ),
    }
