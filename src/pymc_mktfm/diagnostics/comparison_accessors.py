"""Figure accessors over saved comparison and calibration evidence.

:class:`Comparison` scores the saved draws of several methods on one world
once and draws every comparison figure for their common window.
:class:`CalibrationReport` plots and persists saved independent-world SBC
cards, including conditional live-effect and graph-averaged SBC. Every figure
method returns a matplotlib ``Figure`` and imports
matplotlib only when called. Nothing here samples a model, trains or fits a
reference. A method whose evidence is missing raises ``ValueError`` naming it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any

from pymc_mktfm.inference.result import DecompositionResult

from .report import write_calibration_report

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .comparisons import ComparisonRecord


def _supplied(field):
    return lambda comparison: any(
        getattr(record, field) is not None for record in comparison.records
    )


def _declared(setting):
    return lambda comparison: bool(comparison._settings.get(setting))


# Evidence a figure needs, named as the caller supplies it.
_EVIDENCE = {
    "truth": _supplied("truth"),
    "live": _supplied("live"),
    "graph_truth": _supplied("graph_truth"),
    # Declared pairs that coincide with an automatic graph-averaging pair are
    # drawn by ``graph_averaging``, so ``ab`` needs rows of its own kind.
    "method_pairs": lambda comparison: any(
        row["kind"] == "method_comparison" for row in comparison.summary["paired"]
    ),
    "oracle_pairs": _declared("oracle_pairs"),
    "omission_pairs": _declared("omission_pairs"),
    "a record with omitted_input": _supplied("omitted_input"),
    "a reference record": lambda comparison: any(
        record.kind == "reference" for record in comparison.records
    ),
    "a model_average and a map or fixed result of one checkpoint": lambda comparison: any(
        row["kind"] == "graph_averaging" for row in comparison.summary["paired"]
    ),
}


class Comparison:
    """Methods compared on one world and window of the same observations.

    ``results_or_records`` is a mapping ``{method: DecompositionResult}`` or a
    sequence of :class:`~pymc_mktfm.diagnostics.comparisons.ComparisonRecord`.
    A mapping becomes one model record per method, labelled ``case`` and
    ``world``, sharing the optional ``truth[time, component]``,
    ``live[component]``, ``graph_truth[edge]`` and ``observed[time]`` and
    scaled by each result's recorded ``sales_scale``, which every result must
    carry as a finite positive number. ``method_pairs`` then defaults to every
    pair in mapping order (second minus first) that is neither an
    ``oracle_pairs`` contrast nor, in either order, a pair
    ``comparison_summary`` makes itself: one checkpoint's model average and its
    map or fixed view, drawn by ``plot.graph_averaging`` unless declared an
    oracle contrast. Records carry their own labels and evidence.

    All records share one window and world, and one ``data_sha256`` except
    the omitted case of a declared ``omission_pairs`` contrast, which
    :func:`~pymc_mktfm.diagnostics.comparisons.matched_omission` binds to the
    original world instead. ``settings`` go to
    :func:`~pymc_mktfm.diagnostics.comparisons.comparison_summary`, which scores
    the saved draws once; figures only redraw that summary.
    """

    __slots__ = ("_records", "_settings", "_summary", "_window", "_world")

    def __init__(
        self,
        results_or_records: Mapping[str, DecompositionResult] | Sequence[ComparisonRecord],
        *,
        truth=None,
        live=None,
        graph_truth=None,
        observed=None,
        case: str = "case",
        world: int | str = 0,
        **settings: Any,
    ):
        from .comparisons import (
            ComparisonRecord,
            _automatic_pairs,
            _valid_scale,
            comparison_summary,
        )

        for name in ("method_pairs", "oracle_pairs", "omission_pairs"):
            if name in settings:
                settings[name] = tuple(map(tuple, settings[name]))
        if isinstance(results_or_records, Mapping):
            if isinstance(world, bool) or not isinstance(world, (int, str)):
                raise ValueError("world must be an integer or string label")
            records = []
            for method, result in results_or_records.items():
                if not isinstance(result, DecompositionResult):
                    raise ValueError(f"{method!r} must map to a DecompositionResult")
                scale = result.provenance.get("sales_scale")
                if scale is None:
                    raise ValueError(
                        f"{method!r} records no sales_scale; Comparison scales each result's "
                        "scores by its own recorded sales_scale"
                    )
                if not _valid_scale(scale):
                    raise ValueError(
                        f"{method!r} records sales_scale {scale!r}; Comparison needs a finite "
                        "positive number"
                    )
                records.append(
                    ComparisonRecord(
                        case=case,
                        world=str(world),
                        method=method,
                        result=result,
                        kind="model",
                        truth=truth,
                        live=live,
                        graph_truth=graph_truth,
                        observed=observed,
                        scale=scale,
                    )
                )
            # Oracle contrasts and graph-averaging pairs have their own figures.
            excluded = {
                frozenset(pair)
                for pair in (*settings.get("oracle_pairs", ()), *_automatic_pairs(records))
            }
            settings.setdefault(
                "method_pairs",
                tuple(
                    pair
                    for pair in combinations(results_or_records, 2)
                    if frozenset(pair) not in excluded
                ),
            )
        else:
            records = list(results_or_records)
            if not all(isinstance(record, ComparisonRecord) for record in records):
                raise ValueError(
                    "Comparison needs a mapping of results or a sequence of ComparisonRecord"
                )
            if (case, world) != ("case", 0) or any(
                value is not None for value in (truth, live, graph_truth, observed)
            ):
                raise ValueError(
                    "records carry their own case, world, truth, live, graph_truth and observed"
                )
        if not records:
            raise ValueError("Comparison needs at least one result")
        windows = {record.result.window for record in records}
        worlds = {record.world for record in records}
        if len(windows) != 1 or len(worlds) != 1:
            raise ValueError(
                f"Comparison needs one common window and world, got windows {sorted(windows)} "
                f"and worlds {sorted(worlds)}"
            )
        omitted = {pair[1] for pair in settings.get("omission_pairs", ())}
        identities = {
            record.result.provenance["data_sha256"]
            for record in records
            if record.case not in omitted
        }
        if len(identities) != 1:
            raise ValueError(
                "Comparison needs one common data_sha256; only the omitted case of declared "
                "omission_pairs may differ"
            )
        self._records = tuple(records)
        self._settings = settings
        self._window = windows.pop()
        self._world = worlds.pop()
        self._summary = comparison_summary(self._records, **settings)

    @property
    def records(self) -> tuple[ComparisonRecord, ...]:
        """The compared records."""
        return self._records

    @property
    def window(self) -> str:
        """The common window every figure draws."""
        return self._window

    @property
    def world(self) -> str:
        """The common world label."""
        return self._world

    @property
    def summary(self) -> dict[str, Any]:
        """The saved-evidence table every figure draws (``comparison_summary`` output)."""
        return self._summary

    @property
    def plot(self) -> ComparisonPlots:
        return ComparisonPlots(self)


class ComparisonPlots:
    """Every comparison figure of a :class:`Comparison` (``comparison.plot``), for its window."""

    __slots__ = ("_comparison",)

    def __init__(self, comparison: Comparison):
        self._comparison = comparison

    def _draw(self, name, function, *needs, **options) -> Figure:
        comparison = self._comparison
        missing = [need for need in needs if not _EVIDENCE[need](comparison)]
        if missing:
            raise ValueError(f"Comparison.plot.{name} needs {' and '.join(missing)}")
        from pymc_mktfm.plotting import comparisons

        return getattr(comparisons, function)(
            comparison.summary, window=comparison.window, **options
        )

    def performance(self) -> Figure:
        """Window-mean CRPS per family and method; needs truth."""
        return self._draw("performance", "plot_performance", "truth")

    def coverage(self) -> Figure:
        """Coverage of equal-tailed window-mean intervals; needs truth."""
        return self._draw("coverage", "plot_coverage", "truth")

    def recovery(self) -> Figure:
        """Every live effect covered and none falsely excluded; needs truth and live."""
        return self._draw("recovery", "plot_recovery", "truth", "live")

    def failure_size(self) -> Figure:
        """Median and 90th-percentile window-mean CRPS; needs truth."""
        return self._draw("failure_size", "plot_failure_size", "truth")

    def false_effects(self) -> Figure:
        """Share of inactive inputs with effect-sized attribution; needs live."""
        return self._draw("false_effects", "plot_false_effects", "live")

    def five_controls(self) -> Figure:
        """False exclusions among exactly five live controls; needs live."""
        return self._draw("five_controls", "plot_five_controls", "live")

    def error_decomposition(self) -> Figure:
        """Signed baseline and live-family total errors; needs truth."""
        return self._draw("error_decomposition", "plot_error_decomposition", "truth")

    def component_hit_rate(self) -> Figure:
        """Fraction of live components covered; needs truth and live."""
        return self._draw("component_hit_rate", "plot_component_hit_rate", "truth", "live")

    def control_count_recovery(self) -> Figure:
        """Modal graph recovers five live controls; needs live."""
        return self._draw("control_count_recovery", "plot_control_count_recovery", "live")

    def inactive_attribution(self) -> Figure:
        """Absolute attribution to inactive inputs; needs live."""
        return self._draw("inactive_attribution", "plot_inactive_attribution", "live")

    def false_effect_worlds(self) -> Figure:
        """Worlds with any effect-sized false attribution; needs live."""
        return self._draw("false_effect_worlds", "plot_false_effect_worlds", "live")

    def absorbed_share(self) -> Figure:
        """Omitted contribution absorbed elsewhere; needs truth and an omitted-input record."""
        return self._draw(
            "absorbed_share", "plot_absorbed_share", "truth", "a record with omitted_input"
        )

    def structure_recovery(self) -> Figure:
        """Empirical complete-graph mode equals the generating graph; needs graph_truth."""
        return self._draw("structure_recovery", "plot_structure_recovery", "graph_truth")

    def graph_count_recovery(self, family: str) -> Figure:
        """Modal graph recovers the ``family`` edge count (e.g. ``"cy"``); needs graph_truth."""
        return self._draw(
            "graph_count_recovery", "plot_graph_count_recovery", "graph_truth", family=family
        )

    def ab(self) -> Figure:
        """Paired CRPS differences of ``method_pairs``, second minus first; needs truth.

        One checkpoint's model average and map or fixed view are a
        graph-averaging pair, drawn by :meth:`graph_averaging` instead.
        """
        return self._draw("ab", "plot_ab_comparison", "truth", "method_pairs")

    def oracle(self) -> Figure:
        """Paired CRPS differences of methods minus their ``oracle_pairs`` oracle; needs truth."""
        return self._draw("oracle", "plot_oracle_comparison", "truth", "oracle_pairs")

    def graph_averaging(self) -> Figure:
        """Model average minus the map/fixed view of one checkpoint; needs truth.

        A pair declared in ``oracle_pairs`` is drawn by :meth:`oracle` instead.
        """
        return self._draw(
            "graph_averaging",
            "plot_graph_averaging",
            "truth",
            "a model_average and a map or fixed result of one checkpoint",
        )

    def matched_omission(self) -> Figure:
        """After minus before CRPS on the retained identities of ``omission_pairs``."""
        return self._draw("matched_omission", "plot_matched_omission", "omission_pairs")

    def world(self, projection: str = "trajectory", *, case: str | None = None) -> Figure:
        """Stored trajectories (or ``"window_mean"`` distributions) of every method.

        ``case`` is required only when the records span several cases.
        """
        comparison = self._comparison
        cases = list(dict.fromkeys(record.case for record in comparison.records))
        if case is None:
            if len(cases) != 1:
                raise ValueError(f"Comparison.plot.world needs case, one of {cases}")
            case = cases[0]
        return self._draw(
            "world",
            "plot_world_comparison",
            case=case,
            world=comparison.world,
            projection=projection,
        )

    def reference_health(self) -> Figure:
        """Recorded fit health of the reference records."""
        return self._draw("reference_health", "plot_reference_health", "a reference record")


class CalibrationReport:
    """Plot and persist a saved independent-world SBC report card.

    Wraps the output of :func:`~pymc_mktfm.diagnostics.sbc_report` or
    :func:`~pymc_mktfm.diagnostics.conditional_sbc_report`. Nothing is
    resampled: ``plot.card`` redraws saved ranks, PIT, ECDF bands and coverage,
    and ``report`` writes the files of
    :func:`~pymc_mktfm.diagnostics.write_calibration_report`.
    """

    __slots__ = ("_report",)

    def __init__(self, report: Mapping[str, Any]):
        if not isinstance(report, Mapping) or report.get("kind") not in (
            "independent_world_sbc",
            "independent_world_conditional_sbc",
        ):
            raise ValueError("CalibrationReport wraps an independent-world sbc_report result")
        self._report = report

    @property
    def summary(self) -> Mapping[str, Any]:
        """The wrapped ``sbc_report`` or ``conditional_sbc_report`` output."""
        return self._report

    @property
    def plot(self) -> CalibrationPlots:
        return CalibrationPlots(self._report)

    def report(self, out_dir, *, strict: bool = True) -> dict[str, Any]:
        """Persist the report card into a new or empty ``out_dir``."""
        return write_calibration_report(self._report, out_dir, strict=strict)


class CalibrationPlots:
    """SBC report-card figures of a :class:`CalibrationReport` (``calibration.plot``)."""

    __slots__ = ("_report",)

    def __init__(self, report: Mapping[str, Any]):
        self._report = report

    def card(self, kind: str = "ecdf") -> Figure:
        """Saved rank/PIT histogram (``"ranks"``), ECDF (``"ecdf"``) or ``"coverage"``."""
        from pymc_mktfm.plotting.panels import plot_sbc_report

        return plot_sbc_report(self._report, kind=kind)
