"""Numerical diagnostics, calibration, and evaluation metrics.

Names load on first use (PEP 562), so ``pymc_mfm.diagnostics.accessors`` and the result
plots need no PyTorch; the archive and training helpers import modules that do.
"""

from __future__ import annotations

from pkgutil import iter_modules
from typing import TYPE_CHECKING

from pymc_mfm._lazy import import_local

if TYPE_CHECKING:
    from .accessors import (
        Diagnostics,
        DiagnosticsPlots,
        ResultPlots,
        TrainingPlots,
        TrainingRun,
    )
    from .archive import (
        WINDOW_TOTAL_FAMILIES,
        ArchiveEntry,
        ArchiveIntegrityError,
        EvaluationArchive,
        EvaluationArchiveWriter,
        family_window_totals,
    )
    from .calibration import (
        calibration_summary,
        compute_rank,
        conditional_sbc_report,
        coverage_from_ranks,
        randomized_ranks,
        rank_uniformity_stats,
        sbc_report,
    )
    from .comparison_accessors import (
        CalibrationPlots,
        CalibrationReport,
        Comparison,
        ComparisonPlots,
    )
    from .core import PANEL_NAMES, compute_diagnostics, record_series
    from .report import DiagnosticError, build_report, render_report, write_calibration_report
    from .strata import (
        STRATA_AXES,
        effect_rows,
        misclassification_summary,
        result_effect_rows,
        row_strata,
        size_cuts,
        strata_summary,
        stratified_summary,
    )
    from .structure import (
        cardinality_decode,
        factorized_set_diagnostics,
        graph_support_diagnostics,
        randomized_set_pit,
        set_reliability,
    )
    from .uncertainty import paired_graph_uncertainty

#: Public name -> submodule that defines it.
_EXPORTS = {
    "Diagnostics": "accessors",
    "DiagnosticsPlots": "accessors",
    "ResultPlots": "accessors",
    "TrainingPlots": "accessors",
    "TrainingRun": "accessors",
    "WINDOW_TOTAL_FAMILIES": "archive",
    "ArchiveEntry": "archive",
    "ArchiveIntegrityError": "archive",
    "EvaluationArchive": "archive",
    "EvaluationArchiveWriter": "archive",
    "family_window_totals": "archive",
    "calibration_summary": "calibration",
    "compute_rank": "calibration",
    "conditional_sbc_report": "calibration",
    "coverage_from_ranks": "calibration",
    "randomized_ranks": "calibration",
    "rank_uniformity_stats": "calibration",
    "sbc_report": "calibration",
    "CalibrationPlots": "comparison_accessors",
    "CalibrationReport": "comparison_accessors",
    "Comparison": "comparison_accessors",
    "ComparisonPlots": "comparison_accessors",
    "PANEL_NAMES": "core",
    "compute_diagnostics": "core",
    "record_series": "core",
    "DiagnosticError": "report",
    "build_report": "report",
    "render_report": "report",
    "write_calibration_report": "report",
    "STRATA_AXES": "strata",
    "effect_rows": "strata",
    "misclassification_summary": "strata",
    "result_effect_rows": "strata",
    "row_strata": "strata",
    "size_cuts": "strata",
    "strata_summary": "strata",
    "stratified_summary": "strata",
    "cardinality_decode": "structure",
    "factorized_set_diagnostics": "structure",
    "graph_support_diagnostics": "structure",
    "randomized_set_pit": "structure",
    "set_reliability": "structure",
    "paired_graph_uncertainty": "uncertainty",
}

#: Submodules that load on attribute access, as the eager package left them bound.
_SUBMODULES = frozenset(info.name for info in iter_modules(__path__))

__all__ = [
    "ArchiveEntry",
    "ArchiveIntegrityError",
    "CalibrationPlots",
    "CalibrationReport",
    "Comparison",
    "ComparisonPlots",
    "DiagnosticError",
    "Diagnostics",
    "DiagnosticsPlots",
    "EvaluationArchive",
    "EvaluationArchiveWriter",
    "PANEL_NAMES",
    "STRATA_AXES",
    "ResultPlots",
    "TrainingPlots",
    "TrainingRun",
    "WINDOW_TOTAL_FAMILIES",
    "build_report",
    "calibration_summary",
    "conditional_sbc_report",
    "cardinality_decode",
    "compute_diagnostics",
    "effect_rows",
    "compute_rank",
    "coverage_from_ranks",
    "factorized_set_diagnostics",
    "family_window_totals",
    "graph_support_diagnostics",
    "misclassification_summary",
    "paired_graph_uncertainty",
    "randomized_ranks",
    "randomized_set_pit",
    "rank_uniformity_stats",
    "record_series",
    "result_effect_rows",
    "row_strata",
    "render_report",
    "sbc_report",
    "size_cuts",
    "strata_summary",
    "stratified_summary",
    "set_reliability",
    "write_calibration_report",
]


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is None:
        # A submodule the eager package used to import (``package.calibration``).
        if name in _SUBMODULES:
            return import_local(f"{__name__}.{name}", None, f"{__name__}.{name}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_local(f"{__name__}.{submodule}", name, f"{__name__}.{name}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})
