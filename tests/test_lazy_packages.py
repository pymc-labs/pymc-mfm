"""The lazy package ``__init__`` files keep every attribute the eager ones bound."""

import subprocess
import sys
import textwrap

import pytest

#: ``vars(package)`` of ``pymc_mfm.<package>`` on the last eager release, without underscore names.
BASE_NAMES = {
    "diagnostics": [
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
        "ResultPlots",
        "STRATA_AXES",
        "TrainingPlots",
        "TrainingRun",
        "WINDOW_TOTAL_FAMILIES",
        "accessors",
        "archive",
        "build_report",
        "calibration",
        "calibration_summary",
        "cardinality_decode",
        "comparison_accessors",
        "compute_diagnostics",
        "compute_rank",
        "conditional_sbc_report",
        "core",
        "coverage",
        "coverage_from_ranks",
        "effect_rows",
        "factorized_set_diagnostics",
        "family_window_totals",
        "graph_support_diagnostics",
        "misclassification_summary",
        "paired_graph_uncertainty",
        "randomized_ranks",
        "randomized_set_pit",
        "rank_uniformity_stats",
        "record_series",
        "render_report",
        "report",
        "result_effect_rows",
        "row_strata",
        "sbc_report",
        "set_reliability",
        "size_cuts",
        "strata",
        "strata_summary",
        "stratified_summary",
        "structure",
        "uncertainty",
        "write_calibration_report",
    ],
    "inference": [
        "DecompositionResult",
        "WarmUp",
        "api",
        "checkpoint",
        "decompose",
        "from_pretrained",
        "hub",
        "list_profiles",
        "observation_batch",
        "push_to_hub",
        "result",
        "save_pretrained",
        "warm_up_observations",
        "warm_up_sampling",
        "warmup",
    ],
    "training": [
        "CorpusStream",
        "METRIC_BOUNDS",
        "MlflowTrainingObserver",
        "TrackingConfig",
        "TrainingConfig",
        "TrainingError",
        "TrainingObserver",
        "TrainingResult",
        "archive",
        "archive_identity",
        "check_recorded_evaluation",
        "checkpoint",
        "compatibility",
        "config",
        "evaluate_into_archive",
        "evaluate_rows",
        "evaluation",
        "flatten_evaluation",
        "load_checkpoint",
        "metric_aggregation",
        "metrics",
        "observer",
        "save_checkpoint",
        "selection_metrics",
        "stream",
        "summary",
        "tracking",
        "train",
        "trainer",
    ],
}


@pytest.mark.parametrize("package", sorted(BASE_NAMES))
def test_every_attribute_of_the_eager_package_still_resolves(package):
    import importlib

    module = importlib.import_module(f"pymc_mfm.{package}")
    missing = [name for name in BASE_NAMES[package] if not hasattr(module, name)]
    assert missing == []
    assert set(BASE_NAMES[package]) <= set(dir(module))
    for name in getattr(module, "__all__", ()):
        assert hasattr(module, name)


def test_importing_the_packages_imports_no_torch_until_a_torch_name_is_used():
    code = textwrap.dedent(
        """
        import sys
        import pymc_mfm.inference, pymc_mfm.diagnostics, pymc_mfm.training
        from pymc_mfm.inference import DecompositionResult
        assert "torch" not in sys.modules
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
