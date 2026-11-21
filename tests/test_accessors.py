"""Result and diagnostics accessors: one computation, the report renderers, stated labels."""

import copy
import io
import json
import pickle
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from test_diagnostics import make_result

from pymc_mktfm.diagnostics import build_report
from pymc_mktfm.diagnostics import report as reporting
from pymc_mktfm.inference.result import DecompositionResult

pytest.importorskip("matplotlib")
from matplotlib.figure import Figure  # noqa: E402

RESULT_FIGURES = (
    "components",
    "channels",
    "channel_totals",
    "controls",
    "indirect",
    "reconstruction",
    "graph",
    "graph_posterior",
    "cardinality",
    "graph_statistics",
    "cycles",
    "zero_atoms",
    "uncertainty",
    "support",
)
SINGLE_PANEL_FIGURES = ("channel_totals", "graph_posterior", "zero_atoms", "uncertainty", "cycles")
EVIDENCE_FIGURES = ("attribution", "window_totals", "cumulative", "temporal", "reliability")


def labels_of(result):
    head = result.provenance.get("head", "conditional")
    return (f"{head} head", result.window, result.mode, result.units, f"{result.n_samples} draws")


def test_result_figures_state_head_window_mode_units_and_draws():
    result = make_result()
    result = replace(result, provenance={**result.provenance, "head": "joint"})
    for name in RESULT_FIGURES:
        text = getattr(result.plot, name)()._suptitle.get_text()
        for label in labels_of(result):
            assert label in text, (name, label)


def test_result_dag_returns_a_native_artifact_of_its_prediction_export():
    from pymc_mktfm.plotting import GraphvizGraph

    result = make_result()
    artifact = result.plot.dag()
    assert isinstance(artifact, GraphvizGraph)
    assert artifact.source == result.plot.dot()


def test_single_panel_figures_draw_into_caller_axes_and_title_it():
    result = make_result()
    for name in SINGLE_PANEL_FIGURES:
        figure = Figure()
        target, untouched = figure.subplots(1, 2)
        assert getattr(result.plot, name)(ax=target) is figure
        assert figure._suptitle is None
        assert all(label in target.get_title() for label in labels_of(result)), name
        assert not (untouched.patches or untouched.lines or untouched.collections), name
    figure = Figure()
    ax = figure.subplots()
    result.plot.zero_atoms(ax=ax)
    zero_paths = np.all(result.components == 0, axis=1).mean(axis=0)[result.component_mask]
    np.testing.assert_allclose([bar.get_width() for bar in ax.patches], zero_paths)


def test_single_panel_figures_on_subfigure_axes_return_the_root_figure():
    result = make_result()
    diagnostics = result.diagnostics(truth=result.components.mean(axis=0), graph_truth=[1, 0, 1])
    methods = [(result.plot, name) for name in SINGLE_PANEL_FIGURES]
    methods += [(diagnostics.plot, "recovery"), (diagnostics.plot, "reliability")]
    for plots, name in methods:
        root = Figure()
        # Axes two subfigures below the figure that saves.
        target = root.subfigures(1, 2)[0].subfigures(2, 1)[1].subplots()
        figure = getattr(plots, name)(ax=target)
        assert figure is root, name
        figure.savefig(io.BytesIO(), format="png")
        assert all(label in target.get_title() for label in labels_of(result)), name


def test_calibration_coverage_draws_into_caller_axes_and_the_family_grids_refuse():
    result = make_result()
    diagnostics = result.diagnostics(truth=result.components.mean(axis=0))
    figure = Figure()
    target, untouched = figure.subplots(1, 2)
    assert diagnostics.plot.calibration("coverage", ax=target) is figure
    assert figure._suptitle is None
    assert all(label in target.get_title() for label in labels_of(result))
    lines = {line.get_label(): line for line in target.lines}
    for family, record in diagnostics.summary["arrays"]["calibration"].items():
        coverage = [row["coverage"] for row in record["intervals"]]
        np.testing.assert_array_equal(lines[family].get_ydata(), coverage)
    assert not untouched.lines
    for kind in ("pit", "ecdf"):
        refused = Figure().subplots()
        with pytest.raises(ValueError):
            diagnostics.plot.calibration(kind, ax=refused)
        assert not (refused.lines or refused.patches or refused.get_title()), kind


def test_truth_figures_state_head_window_mode_units_and_draws():
    result = make_result()
    result = replace(result, provenance={**result.provenance, "head": "joint"})
    diagnostics = result.diagnostics(truth=result.components.mean(axis=0), graph_truth=[1, 0, 1])
    for name in EVIDENCE_FIGURES:
        text = getattr(diagnostics.plot, name)()._suptitle.get_text()
        for label in labels_of(result):
            assert label in text, (name, label)
    figure = Figure()
    target, untouched = figure.subplots(1, 2)
    assert diagnostics.plot.reliability(ax=target) is figure
    assert figure._suptitle is None
    assert all(label in target.get_title() for label in labels_of(result))
    assert not (untouched.patches or untouched.lines or untouched.collections)


def test_result_and_diagnostics_score_once_and_reuse_the_values(monkeypatch, tmp_path):
    calls = []
    original = reporting.compute_diagnostics

    def counting(result, **evidence):
        calls.append(evidence["truth"] is not None)
        return original(result, **evidence)

    monkeypatch.setattr(reporting, "compute_diagnostics", counting)
    result = make_result()
    result.plot.components()
    result.plot.zero_atoms()
    assert result.diagnostics().metrics is result.diagnostics().metrics
    assert calls == [False]
    diagnostics = result.diagnostics(truth=result.components.mean(axis=0))
    diagnostics.plot.recovery()
    diagnostics.plot.calibration("coverage")
    manifest = diagnostics.report(tmp_path / "report")
    assert calls == [False, True]
    assert manifest["metrics"] == diagnostics.metrics
    assert manifest["panels"]["recovery_scatter"]["render_status"] == "ok"
    # Rendering the report leaves the cached values untouched for later figures.
    assert "path" not in diagnostics.summary["panels"]["recovery_scatter"]


def test_diagnostics_report_writes_what_build_report_writes(tmp_path):
    result = make_result()
    truth = result.components.mean(axis=0)
    evidence = {
        "truth": truth,
        "graph_truth": [1, 0, 1],
        "observed": truth.sum(axis=-1) + [1.0, -2.0, 0.5],
        "training_records": [{"step": 1, "loss": 3.0}, {"step": 2, "loss": 2.0}],
        "evaluation_records": [{"step": 2, "flat": {"holdout/conditional/query/crps": 0.4}}],
    }
    reported = result.diagnostics(**evidence).report(tmp_path / "accessor")
    built = build_report(result, tmp_path / "function", **evidence)
    for name in ("diagnostics.json", "evidence.json", "result/result.json"):
        assert (tmp_path / "accessor" / name).read_bytes() == (
            tmp_path / "function" / name
        ).read_bytes()

    def without_hashes(manifest):
        return {key: value for key, value in manifest.items() if key != "artifacts"}

    assert without_hashes(reported) == without_hashes(built)
    assert [item["path"] for item in reported["artifacts"]] == [
        item["path"] for item in built["artifacts"]
    ]


def test_diagnostics_report_saves_the_evidence_it_scored(tmp_path):
    result = make_result()
    truth = result.components.mean(axis=0)
    scored = truth.copy()
    diagnostics = result.diagnostics(truth=truth)
    truth += 100.0  # the caller reuses its array after scoring
    diagnostics.report(tmp_path / "report")
    with np.load(tmp_path / "report" / "evidence.npz") as saved:
        np.testing.assert_array_equal(saved["truth"], scored)


def test_diagnostics_report_saves_the_draws_it_scored_and_result_arrays_are_read_only(tmp_path):
    components = np.array(make_result().components)  # the caller's own buffer
    result = replace(make_result(), components=components)
    diagnostics = result.diagnostics(truth=components.mean(axis=0))
    components[..., 0] += 100.0  # the caller reuses its buffer after scoring
    diagnostics.report(tmp_path / "report")
    saved = DecompositionResult.load(tmp_path / "report" / "result")
    scored = json.loads((tmp_path / "report" / "diagnostics.json").read_text())["metrics"]
    assert saved.components[..., 0].mean() == pytest.approx(scored["component/baseline/mean"])
    with pytest.raises(ValueError):
        result.components[0, 0, 0] = 1.0


@pytest.mark.parametrize(
    ("method", "missing"),
    [
        ("recovery", "truth"),
        ("attribution", "truth"),
        ("window_totals", "truth"),
        ("cumulative", "truth"),
        ("temporal", "truth"),
        ("reliability", "graph truth"),
        ("calibration", "truth"),
        ("training", "training records"),
        ("evaluation", "evaluation records"),
        ("reference", "reference"),
        ("posterior_sbc", "posterior_sbc"),
        ("c2st", "c2st"),
    ],
)
def test_diagnostic_figures_name_missing_evidence(method, missing):
    with pytest.raises(ValueError, match=rf"unavailable: .*{missing}"):
        getattr(make_result().diagnostics().plot, method)()


def test_result_figures_without_recorded_evidence_name_it():
    result = make_result()
    with pytest.raises(ValueError, match=r"unavailable: .*input_support"):
        result.plot.input_support()
    with pytest.raises(ValueError, match="set_cardinalities|design cardinalities"):
        result.plot.set_posterior()
    with pytest.raises(ValueError, match="kind"):
        result.diagnostics().plot.calibration("rank")


def reference_evidence(result):
    data_sha256 = result.provenance["data_sha256"]
    return {
        "label": "Cached reference",
        "components": result.components,
        "component_names": result.component_names,
        "time": result.time,
        "window": result.window,
        "units": result.units,
        "provenance": {"data_sha256": data_sha256},
        "c2st": {
            "labels": [0, 0, 1, 1],
            "probabilities": [0.1, 0.2, 0.8, 0.9],
            "world_ids": ["a", "b", "a", "b"],
            "training_world_ids": ["train"],
            "classifier": "frozen",
        },
        "posterior_sbc": {
            "posterior_draws": np.zeros((256, 1)),
            "augmented_draws": np.zeros((256, 4, 1)),
            "query_indices": np.zeros(256, dtype=int),
            "quantity_names": ["control:weather"],
            "provenance": {
                "observed_data_sha256": data_sha256,
                "replicate_contract": "valid_sequential_continuation",
                "joint_draws": True,
            },
        },
    }


def test_reference_evidence_panels_render_from_diagnostics():
    result = make_result()
    diagnostics = result.diagnostics(reference=reference_evidence(result))
    # Every held-out pair is ranked correctly: accuracy 1.
    assert diagnostics.metrics["reference/c2st/accuracy"] == 1
    for name in ("reference", "posterior_sbc", "c2st"):
        assert isinstance(getattr(diagnostics.plot, name)(), Figure)
    # The saved-evidence figures still state the draws they diagnose.
    for name in ("posterior_sbc", "c2st"):
        text = getattr(diagnostics.plot, name)().get_suptitle()
        assert all(label in text for label in labels_of(result)), name


def test_record_figures_from_diagnostics_carry_no_result_labels():
    records = [{"step": 1, "loss": 3.0, "lr": 0.1}, {"step": 2, "loss": 2.5, "lr": 0.2}]
    figure = make_result().diagnostics(training_records=records).plot.training()
    lines = {line.get_label(): line for ax in figure.axes for line in ax.lines}
    np.testing.assert_array_equal(lines["loss"].get_xdata(), [1, 2])
    np.testing.assert_array_equal(lines["loss"].get_ydata(), [3.0, 2.5])
    text = figure._suptitle.get_text()
    assert "revenue" not in text and "model_average" not in text


def test_reconstruction_with_observations_draws_the_observation_residual():
    result = make_result()
    observed = np.array([12.0, 13.0, 14.0])
    main, residual = result.plot.reconstruction(observed=observed).axes
    lines = {line.get_label(): line for line in main.lines}
    np.testing.assert_allclose(lines["Noisy observations"].get_ydata(), observed)
    latent_mean = result.components[..., result.component_mask].sum(axis=-1).mean(axis=0)
    residual_line = next(
        line for line in residual.lines if np.allclose(line.get_ydata(), observed - latent_mean)
    )
    np.testing.assert_array_equal(residual_line.get_xdata(), result.time)
    unobserved = {line.get_label() for line in result.plot.reconstruction().axes[0].lines}
    assert "Noisy observations" not in unobserved


def test_input_support_describes_the_active_raw_inputs_decompose_saw(model):
    from conftest import make_synthetic_corpus

    from pymc_mktfm.inference.api import decompose

    corpus = make_synthetic_corpus()
    context = int(corpus["support_mask"][0].sum())
    spend, controls = corpus["treatment_raw"][0], corpus["covariates"][0]
    result = decompose(
        model,
        {
            "spend": spend,
            "controls": controls,
            "context_outcome": corpus["outcome_raw"][0, :context],
            "channel_mask": corpus["treatment_active_mask"][0],
            "control_mask": corpus["covariate_active_mask"][0],
            "latent_mask": corpus["latent_active_mask"][0],
        },
        n_samples=2,
    )
    expected = [
        (f"{family}:{family}_{index}", values[:, index].astype(np.float64))
        for family, values, mask in (
            ("channel", spend, corpus["treatment_active_mask"][0]),
            ("control", controls, corpus["covariate_active_mask"][0]),
        )
        for index in np.flatnonzero(mask)
    ]
    rows = result.diagnostics().summary["arrays"]["input_support"]["inputs"]
    assert [row["name"] for row in rows] == [name for name, _ in expected]
    for row, (_, series) in zip(rows, expected):
        assert row["context"]["mean"] == pytest.approx(series[:context].mean())
        assert row["context"]["std"] == pytest.approx(series[:context].std())
        assert row["query"]["mean"] == pytest.approx(series[context:].mean())
        assert row["query"]["max"] == pytest.approx(series[context:].max())
    assert isinstance(result.plot.input_support(), Figure)


def test_decompose_and_diagnostics_import_no_matplotlib_until_a_figure_is_drawn():
    # A notebook shows a cell result through ``_repr_png_``, with no pyplot backend.
    code = """
import sys
import numpy as np
import torch
from pymc_mktfm import MFM
from pymc_mktfm.inference import decompose
from pymc_mktfm.model import ModelConfig

torch.manual_seed(0)
config = ModelConfig(
    d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=1
)
observations = {
    "spend": np.ones((5, 2)),
    "controls": np.zeros((5, 2)),
    "context_outcome": np.array([10.0, 11.0]),
}
result = decompose(MFM(config).eval(), observations, n_samples=2)
result.plot
result.diagnostics()
graph = result.plot.dag()
assert graph.source == result.plot.dot()
loaded = sorted(name for name in sys.modules if name.split(".")[0] == "matplotlib")
assert not loaded, loaded
figure = result.plot.components()
from matplotlib.figure import Figure
assert isinstance(figure, Figure)
assert figure._repr_png_().startswith(b"\\x89PNG\\r\\n\\x1a\\n")
assert "matplotlib.pyplot" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_figures_pickle_as_plain_matplotlib_figures_and_copies_stay_displayable(tmp_path):
    figure = make_result().plot.components()
    path = tmp_path / "figure.pickle"
    path.write_bytes(pickle.dumps(figure))
    # The pickle loads where pymc_mktfm cannot be imported.
    code = f"""
import io, pickle, sys
sys.modules["pymc_mktfm"] = None
from matplotlib.figure import Figure
with open({str(path)!r}, "rb") as handle:
    figure = pickle.load(handle)
assert type(figure) is Figure, type(figure)
assert figure.get_suptitle() == {figure.get_suptitle()!r}
figure.savefig(io.BytesIO(), format="png")
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    clone = copy.deepcopy(figure)
    assert clone is not figure and clone.get_suptitle() == figure.get_suptitle()
    assert clone._repr_png_().startswith(b"\x89PNG\r\n\x1a\n")
