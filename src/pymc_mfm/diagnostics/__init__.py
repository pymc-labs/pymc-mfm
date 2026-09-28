"""Numerical diagnostics, calibration, and evaluation metrics."""

from .accessors import (
    Diagnostics,
    DiagnosticsPlots,
    ResultPlots,
    TrainingPlots,
    TrainingRun,
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

__all__ = [
    "CalibrationPlots",
    "CalibrationReport",
    "Comparison",
    "ComparisonPlots",
    "DiagnosticError",
    "Diagnostics",
    "DiagnosticsPlots",
    "PANEL_NAMES",
    "STRATA_AXES",
    "ResultPlots",
    "TrainingPlots",
    "TrainingRun",
    "build_report",
    "calibration_summary",
    "conditional_sbc_report",
    "cardinality_decode",
    "compute_diagnostics",
    "effect_rows",
    "compute_rank",
    "coverage_from_ranks",
    "factorized_set_diagnostics",
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
