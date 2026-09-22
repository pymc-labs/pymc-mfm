"""Numerical diagnostics, calibration, and evaluation metrics."""

from .calibration import (
    calibration_summary,
    compute_rank,
    coverage_from_ranks,
    randomized_ranks,
    rank_uniformity_stats,
    sbc_report,
)
from .core import PANEL_NAMES, compute_diagnostics
from .report import DiagnosticError, build_report, render_report, write_calibration_report
from .structure import (
    cardinality_decode,
    factorized_set_diagnostics,
    graph_support_diagnostics,
    randomized_set_pit,
    set_reliability,
)
from .uncertainty import paired_graph_uncertainty

__all__ = [
    "DiagnosticError",
    "PANEL_NAMES",
    "build_report",
    "calibration_summary",
    "cardinality_decode",
    "compute_diagnostics",
    "compute_rank",
    "coverage_from_ranks",
    "factorized_set_diagnostics",
    "graph_support_diagnostics",
    "paired_graph_uncertainty",
    "randomized_ranks",
    "randomized_set_pit",
    "rank_uniformity_stats",
    "render_report",
    "sbc_report",
    "set_reliability",
    "write_calibration_report",
]
