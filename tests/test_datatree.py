"""ArviZ DataTree export of one result: active slots, evidence groups, refusals, optional extra."""

import json
import re
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from conftest import CHART_ROLES, make_synthetic_corpus, needs_arviz
from test_diagnostics import make_result
from test_inference import observed

import pymc_mfm
from pymc_mfm.inference.api import decompose

#: Provenance keys a hand-built result need not record.
OPTIONAL_PROVENANCE = (
    "edge_families",
    "edge_endpoints",
    "sales_scale",
    "data_sha256",
    "state_sha256",
)


def padded_inputs():
    """Corpus row 8 pads channel_2 and control_1; ``latent_mask`` pads latent:1's edge slots."""
    return {**observed(make_synthetic_corpus(), 8), "latent_mask": np.array([1, 0])}


def aligned_truth(result):
    """Truth of ``result``'s shape: one on its active components, zero on padded ones."""
    return np.ones(result.components.shape[1:]) * result.component_mask


@needs_arviz
def test_a_tree_holds_the_active_slots_of_its_window_and_keeps_truth_apart_from_observations(
    model, tmp_path
):
    import arviz_base
    import xarray as xr
    from arviz_stats import summary

    labels = np.array([0, 2, 4, 7, 10, 15, 20, 22, 30, 35, 50, 51])
    result = decompose(model, padded_inputs(), n_samples=6, time=labels)
    active, edges = result.component_mask, result.edge_mask
    names, edge_names = np.array(result.component_names), np.array(result.edge_names)
    latent_1 = np.array(["latent:1" in name for name in result.edge_names])
    assert names[~active].tolist() == ["channel:channel_2", "control:control_1"]
    assert latent_1.any() and not edges[latent_1].any()
    _, times, width = result.components.shape
    truth = np.arange(times * width, dtype=np.float64).reshape(times, width) * active
    graph_truth = edges & (np.arange(edges.size) % 2 == 0)  # boolean, zero on padded slots
    outcome = np.arange(times) + 500  # integer observations
    tree = result.to_datatree(observed=outcome, truth=truth, graph_truth=graph_truth)

    posterior = tree["posterior"]
    assert posterior["components"].dims == ("chain", "draw", "time", "component")
    assert posterior["graph"].dims == ("chain", "draw", "edge")
    np.testing.assert_array_equal(
        posterior["components"].values, result.components[None][..., active]
    )
    np.testing.assert_array_equal(posterior["graph"].values, result.graphs[None][..., edges])
    # Draws keep their stored dtypes: decompose's graphs are float32, not float64.
    assert result.graphs.dtype != np.float64
    assert posterior["components"].dtype == result.components.dtype
    assert posterior["graph"].dtype == result.graphs.dtype
    assert posterior["chain"].values.tolist() == [0]
    assert posterior["draw"].values.tolist() == list(range(6))
    np.testing.assert_array_equal(posterior["time"].values, result.time)
    assert posterior["component"].values.tolist() == names[active].tolist()
    assert posterior["edge"].values.tolist() == edge_names[edges].tolist()
    constants = tree["constant_data"]
    endpoints = np.array(result.provenance["edge_endpoints"])[edges]
    assert {name: constants[name].values.tolist() for name in constants.data_vars} == {
        "component_family": np.array(result.component_families)[active].tolist(),
        "edge_family": np.array(result.provenance["edge_families"])[edges].tolist(),
        "edge_source": endpoints[:, 0].tolist(),
        "edge_target": endpoints[:, 1].tolist(),
    }

    # Observations and synthetic truth go to their own groups, on the posterior's axes.
    assert set(tree.children) == {"posterior", "constant_data", "observed_data", "truth"}
    assert set(tree["observed_data"].data_vars) == {"outcome"}
    assert set(tree["truth"].data_vars) == {"components", "graph"}
    assert tree["observed_data"]["outcome"].dims == ("time",)
    assert tree["truth"]["components"].dims == ("time", "component")
    assert tree["truth"]["graph"].dims == ("edge",)
    np.testing.assert_array_equal(tree["observed_data"]["outcome"].values, outcome)
    np.testing.assert_array_equal(tree["truth"]["components"].values, truth[:, active])
    np.testing.assert_array_equal(tree["truth"]["graph"].values, graph_truth[edges])
    xr.align(*(tree[group].to_dataset() for group in tree.children), join="exact")

    saved = json.loads((result.save(tmp_path / "evidence") / "result.json").read_text())
    expected = {
        "window": "query",
        "mode": "model_average",
        "units": "outcome",
        "n_samples": 6,
        "context_length": 6,
        "metadata_sha256": saved["metadata_sha256"],
        "sales_scale": result.provenance["sales_scale"],
        "data_sha256": result.provenance["data_sha256"],
        "graph_source": "joint_router",
        "mcmc_diagnostics": "not_applicable",
    }
    # These and the three notes, nothing else: a fresh model records no state digest.
    standard = set(arviz_base.make_attrs(inference_library=pymc_mfm, sample_dims=()))
    for attrs in (tree.attrs, posterior.attrs):
        assert expected.items() <= attrs.items()
        assert set(attrs) - standard == {*expected, "graph_draws", "sampling", "marginal_windows"}
    assert all(isinstance(value, (str, int, float)) for value in tree.attrs.values())
    assert len(summary(tree, kind="stats")) == times * active.sum() + edges.sum()

    # The context window of the same world is its own tree over its own rows.
    context = decompose(
        model, padded_inputs(), n_samples=6, window="context", time=labels
    ).to_datatree()
    assert set(context.children) == {"posterior", "constant_data"}
    assert context.attrs["window"] == context["posterior"].attrs["window"] == "context"
    np.testing.assert_array_equal(context["posterior"]["time"].values, labels[:6])


#: One evidence piece per export, refused by the rules ``diagnostics`` applies.
REFUSED_EVIDENCE = {
    "complex truth": (
        lambda result: {"truth": aligned_truth(result) + 0j},
        "truth must be a real array",
    ),
    "NaN truth": (lambda result: {"truth": aligned_truth(result) * np.nan}, "truth must be finite"),
    "truth on a padded component": (
        lambda result: {"truth": np.ones(result.components.shape[1:])},
        "truth is nonzero on padded components",
    ),
    "fractional graph_truth": (
        lambda result: {"graph_truth": result.edge_mask * 0.5},
        "graph_truth must be binary",
    ),
    "graph_truth on a padded edge": (
        lambda result: {"graph_truth": np.ones(result.edge_mask.shape)},
        "graph_truth has edges outside the active support",
    ),
    "short graph_truth": (
        lambda result: {"graph_truth": result.edge_mask[:-1]},
        "graph_truth has shape",
    ),
    "observed of another length": (
        lambda result: {"observed": np.ones(result.components.shape[1] + 1)},
        "observed has shape",
    ),
}


@needs_arviz
@pytest.mark.parametrize("case", REFUSED_EVIDENCE)
def test_each_supplied_evidence_piece_is_checked_on_its_own(case):
    evidence, message = REFUSED_EVIDENCE[case]
    result = make_result()
    with pytest.raises(ValueError, match=re.escape(message)):
        result.to_datatree(**evidence(result))


#: Each evidence piece supplied alone: (its value for a result, its group, its variable).
LONE_EVIDENCE = {
    "observed": (lambda result: np.ones(result.time.shape), "observed_data", "outcome"),
    "truth": (aligned_truth, "truth", "components"),
    "graph_truth": (lambda result: result.edge_mask, "truth", "graph"),
}


@needs_arviz
@pytest.mark.parametrize("piece", LONE_EVIDENCE)
def test_an_evidence_piece_supplied_alone_is_exported_alone(piece):
    value, group, variable = LONE_EVIDENCE[piece]
    result = make_result()
    tree = result.to_datatree(**{piece: value(result)})
    assert set(tree.children) == {"posterior", "constant_data", group}
    assert set(tree[group].data_vars) == {variable}


@needs_arviz
@pytest.mark.parametrize("recorded", ["absent", "null", "recorded", "families", "endpoints"])
def test_edge_constants_and_digests_are_exported_exactly_when_provenance_records_them(recorded):
    result = make_result()
    kept = {
        key: value for key, value in result.provenance.items() if key not in OPTIONAL_PROVENANCE
    }
    provenance = {
        "absent": kept,
        "null": {**kept, **dict.fromkeys(OPTIONAL_PROVENANCE)},
        "recorded": {**result.provenance, "state_sha256": "c" * 64},
        "families": {**kept, "edge_families": result.provenance["edge_families"]},
        "endpoints": {**kept, "edge_endpoints": result.provenance["edge_endpoints"]},
    }[recorded]
    tree = replace(result, provenance=provenance).to_datatree()
    edge_constants = {}
    if recorded in ("recorded", "families"):
        edge_constants["edge_family"] = np.array(result.provenance["edge_families"])[
            result.edge_mask
        ].tolist()
    if recorded in ("recorded", "endpoints"):
        endpoints = np.array(result.provenance["edge_endpoints"])[result.edge_mask]
        edge_constants.update(
            edge_source=endpoints[:, 0].tolist(), edge_target=endpoints[:, 1].tolist()
        )
    digests = (
        {"sales_scale": 10.0, "data_sha256": "b" * 64, "state_sha256": "c" * 64}
        if recorded == "recorded"
        else {}
    )
    assert set(tree["constant_data"].data_vars) == {"component_family", *edge_constants}
    for name, values in edge_constants.items():
        assert tree["constant_data"][name].values.tolist() == values
    for attrs in (tree.attrs, tree["posterior"].attrs):
        # Present exactly when recorded: an unrecorded key is absent, never None.
        assert {key: attrs[key] for key in OPTIONAL_PROVENANCE if key in attrs} == digests
    np.testing.assert_array_equal(
        tree["posterior"]["graph"].values[0], result.graphs[:, result.edge_mask]
    )


#: Each mode with the graph source its results record.
MODES = (
    ("model_average", "joint_router"),
    ("map", "joint_router"),
    ("fixed", "supplied_hypothesis"),
)


@needs_arviz
def test_each_mode_records_its_graph_source_and_consistent_graph_notes():
    result = make_result()
    for mode, source in MODES:
        provenance = {**result.provenance, "graph_source": source}
        tree = replace(result, mode=mode, provenance=provenance).to_datatree()
        for attrs in (tree.attrs, tree["posterior"].attrs):
            assert (attrs["mode"], attrs["graph_source"]) == (mode, source)
        assert tree.attrs["graph_draws"] == tree["posterior"].attrs["graph_draws"]


def recording(**changes):
    """An edit of a result: its provenance with ``changes`` recorded."""
    return lambda result: replace(result, provenance={**result.provenance, **changes})


def without_components(result):
    """An edit of a result: every component padded."""
    return replace(
        result,
        components=np.zeros_like(result.components),
        component_mask=np.zeros_like(result.component_mask),
    )


def ending_in_nul(field, index):
    """An edit of a result: label ``index`` of ``field`` ends in a NUL character."""

    def edit(result):
        labels = list(getattr(result, field))
        labels[index] += "\x00"
        return replace(result, **{field: tuple(labels)})

    return edit


SCALE = "provenance sales_scale must be a finite positive number"
#: Results the export refuses: (edit of ``make_result()``, message).
REFUSED_RESULTS = {
    "short edge_families": (
        recording(edge_families=["cy", "cy"]),
        "provenance edge_families must name one family per edge slot",
    ),
    "endpoint pair of one node": (
        recording(edge_endpoints=[["a", "b"], ["c"], ["d", "e"]]),
        "provenance edge_endpoints must hold one [source, target] pair of node names",
    ),
    "boolean sales_scale": (recording(sales_scale=True), SCALE),
    "zero sales_scale": (recording(sales_scale=0), SCALE),
    "sales_scale beyond float64": (recording(sales_scale=10**400), SCALE),
    "numeric digest": (recording(data_sha256=7), "provenance data_sha256 must be text"),
    "channel label ending in NUL": (
        ending_in_nul("component_names", 1),
        "component labels cannot contain NUL characters",
    ),
    "edge label ending in NUL": (
        ending_in_nul("edge_names", 0),
        "edge labels cannot contain NUL characters",
    ),
    "edge family ending in NUL": (
        recording(edge_families=["cy\x00", "cy", "zb"]),
        "edge_families labels cannot contain NUL characters",
    ),
    "edge endpoint ending in NUL": (
        recording(edge_endpoints=[["a\x00", "b"], ["c", "d"], ["e", "f"]]),
        "edge_endpoints labels cannot contain NUL characters",
    ),
    "no active component": (
        without_components,
        "DataTree export needs at least one active component",
    ),
}


@needs_arviz
@pytest.mark.parametrize("case", REFUSED_RESULTS)
def test_results_the_export_cannot_label_are_refused_and_named(case):
    edit, message = REFUSED_RESULTS[case]
    with pytest.raises(ValueError, match=re.escape(message)):
        edit(make_result()).to_datatree()


@needs_arviz
def test_a_result_without_active_edges_exports_no_edge_axis():
    from arviz_stats import summary

    result = make_result()
    result = replace(
        result, graphs=np.zeros_like(result.graphs), edge_mask=np.zeros_like(result.edge_mask)
    )
    tree = result.to_datatree(graph_truth=np.zeros(result.edge_mask.size))
    # Graph truth of no active edge leaves the truth group without a variable: no group.
    assert set(tree.children) == {"posterior", "constant_data"}
    assert all("edge" not in tree[group].dims for group in tree.children)
    assert set(tree["posterior"].data_vars) == {"components"}
    assert set(tree["constant_data"].data_vars) == {"component_family"}
    assert len(summary(tree, kind="stats")) == 3 * result.component_mask.sum()


@needs_arviz
@pytest.mark.parametrize("labels", [[False], [True], [False, True]])
def test_saved_boolean_time_labels_select_the_original_evidence_rows(tmp_path, labels):
    from arviz_stats import summary

    original = make_result()
    result = replace(
        original, time=np.array(labels), components=original.components[:, : len(labels)]
    )
    result = pymc_mfm.inference.DecompositionResult.load(result.save(tmp_path / "result"))
    truth = aligned_truth(result)
    observations = np.arange(len(labels)) + 10
    tree = result.to_datatree(truth=truth, observed=observations)
    for row, label in enumerate(labels):
        np.testing.assert_array_equal(
            tree["posterior"]["components"].sel(time=label).values[0],
            result.components[:, row][:, result.component_mask],
        )
        np.testing.assert_array_equal(
            tree["truth"]["components"].sel(time=label).values, truth[row, result.component_mask]
        )
        assert tree["observed_data"]["outcome"].sel(time=label).item() == observations[row]
    assert len(summary(tree, kind="stats")) == (
        len(labels) * result.component_mask.sum() + result.edge_mask.sum()
    )


@needs_arviz
@pytest.mark.parametrize("scale", [1e-30, 1e38], ids=["underflow", "overflow"])
def test_float32_components_keep_exact_values_and_finite_correct_summary_spread(scale):
    from arviz_stats import summary

    result = make_result()
    values = (np.arange(1, 6) * scale / (1 if scale < 1 else 5)).astype(np.float32)
    components = np.broadcast_to(
        values[:, None, None] * result.component_mask, result.components.shape
    )
    result = replace(result, components=components)
    tree = result.to_datatree()
    np.testing.assert_array_equal(
        tree["posterior"]["components"].values[0], components[..., result.component_mask]
    )
    table = summary(tree, var_names=["components"], kind="stats", round_to="none")
    exact = values.astype(np.float64)
    np.testing.assert_allclose(table["mean"], exact.mean(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(table["sd"], exact.std(), rtol=1e-12, atol=0)


#: Draws ArviZ summaries cannot use as stored: booleans cannot be subtracted, float16
#: sums overflow (these float16 squared deviations exceed its maximum) and pandas
#: refuses big-endian time labels, as a result saved on a big-endian host loads.
WIDENED_DRAWS = {
    "boolean graphs": lambda result: replace(result, graphs=result.graphs.astype(bool)),
    "boolean components": lambda result: replace(result, components=result.components > 0),
    "float16 components": lambda result: replace(
        result, components=(result.components * 100).astype(np.float16)
    ),
    "big-endian float16 components and time": lambda result: replace(
        result,
        components=(result.components * 100).astype(">f2"),
        time=result.time.astype(">i8"),
    ),
}


@needs_arviz
@pytest.mark.parametrize("draws", WIDENED_DRAWS)
def test_draws_arviz_cannot_summarize_as_stored_keep_their_exact_values_and_summarize(draws):
    from arviz_stats import summary

    result = WIDENED_DRAWS[draws](make_result())
    active, edges = result.component_mask, result.edge_mask
    tree = result.to_datatree()
    np.testing.assert_array_equal(
        tree["posterior"]["components"].values[0], result.components[..., active]
    )
    np.testing.assert_array_equal(tree["posterior"]["graph"].values[0], result.graphs[:, edges])
    table = summary(tree, kind="stats")
    assert len(table) == 3 * active.sum() + edges.sum()
    assert np.isfinite(table[["mean", "sd"]].to_numpy()).all()


@needs_arviz
def test_the_export_layout_ignores_arviz_rcparams():
    from arviz_base import rc_context

    result = make_result()
    with rc_context({"data.sample_dims": ["sample"], "data.index_origin": 1}):
        posterior = result.to_datatree()["posterior"]
    assert posterior["components"].dims == ("chain", "draw", "time", "component")
    assert posterior["graph"].dims == ("chain", "draw", "edge")
    assert posterior["chain"].values.tolist() == [0]
    assert posterior["draw"].values.tolist() == list(range(result.n_samples))


@pytest.mark.parametrize(
    "hidden",
    [("xarray", "arviz_base"), ("arviz_base",)],
    ids=["base install", "xarray without arviz-base"],
)
def test_an_export_without_the_arviz_extra_names_the_extra(monkeypatch, hidden):
    # Import the export module anew with the extra hidden: its own imports must not need it.
    monkeypatch.delitem(sys.modules, "pymc_mfm.inference._datatree", raising=False)
    monkeypatch.setattr(pymc_mfm.inference, "_datatree", None, raising=False)
    for name in hidden:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=re.escape("pymc-mfm[arviz]")):
        make_result().to_datatree()


def test_standalone_decompose_and_saved_evidence_never_import_xarray_or_arviz(tmp_path):
    code = """
import sys
from importlib.util import find_spec

import numpy as np
import torch

import pymc_mfm.diagnostics
from pymc_mfm import MFM
from pymc_mfm.inference import DecompositionResult, decompose
from pymc_mfm.model import ModelConfig

torch.manual_seed(0)
config = ModelConfig(
    d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=1
)
observations = {
    "spend": np.ones((5, 2)),
    "controls": np.zeros((5, 2)),
    "context_outcome": np.array([10.0, 11.0]),
}
model = MFM(config).eval()
result = decompose(model, observations, n_samples=2)
loaded = DecompositionResult.load(result.save(sys.argv[1]))
np.testing.assert_array_equal(loaded.components, result.components)
assert model.idata is None
optional = {"xarray", "arviz_base", "arviz_stats", "arviz", "arviz_plots", "xarray_einstats"}
imported = sorted(name for name in sys.modules if name.split(".")[0] in optional)
assert not imported, imported
if find_spec("arviz_base") is not None:
    loaded.to_datatree()
    assert "xarray" in sys.modules
else:
    try:
        loaded.to_datatree()
    except ImportError as exc:
        assert "pymc-mfm[arviz]" in str(exc), exc
    else:
        raise AssertionError("to_datatree() ran without the arviz extra")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "evidence")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dataframe_inference_constructs_a_tree_without_statistics_or_rendering():
    code = """
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch

optional = {"arviz_stats", "arviz", "arviz_plots", "xarray_einstats", "matplotlib"}
real_find_spec = importlib.util.find_spec

def find_spec_without_analysis_extras(name, package=None):
    if name.split(".")[0] in optional:
        return None
    return real_find_spec(name, package)

importlib.util.find_spec = find_spec_without_analysis_extras

class MissingAnalysisExtras:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in optional:
            raise ModuleNotFoundError(f"optional analysis package blocked: {fullname}")

sys.meta_path.insert(0, MissingAnalysisExtras())

from pymc_mfm import MFM
from pymc_mfm.inference import DecompositionResult
from pymc_mfm.model import ModelConfig

torch.manual_seed(0)
model = MFM(ModelConfig(
    d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=2, n_demand=1
)).eval()
frame = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=5, freq="W-MON"),
    "sales": [10.0, 11.0, np.nan, np.nan, np.nan],
    "search": np.ones(5),
    "social": np.ones(5),
    "seasonality": np.zeros(5),
})
tree = model.inference(
    frame.iloc[:2], frame.iloc[2:], date_column="date", target_column="sales",
    channel_columns=["search", "social"], control_columns=["seasonality"], n_samples=2,
)
assert tree is model.idata
assert set(tree.children) == {"context", "query"}
assert tree["context/posterior"].sizes["draw"] == 2
assert tree["query/posterior"].sizes["draw"] == 2
assert isinstance(model.query_result, DecompositionResult)
assert {"xarray", "arviz_base"} <= sys.modules.keys()
imported = sorted(name for name in sys.modules if name.split(".")[0] in optional)
assert not imported, imported
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("scale", [1e-30, 1e38], ids=["underflow", "overflow"])
def test_named_and_curve_posteriors_preserve_consumer_visible_uncertainty(
    model, chart_frames, scale
):
    from pymc_mfm.inference._datatree import attach_response_curves, prediction_datatree
    from pymc_mfm.inference.curves import ResponseCurves

    history, future = chart_frames
    model.inference(history, future, **CHART_ROLES, n_samples=5)
    prediction = model._prediction_snapshot
    values = (np.arange(1, 6) * scale / (1 if scale < 1 else 5)).astype(np.float32)
    query = replace(
        prediction.query,
        components=np.broadcast_to(
            values[:, None, None] * prediction.query.component_mask,
            prediction.query.components.shape,
        ),
    )
    prediction = replace(prediction, query=query)
    tree = prediction_datatree(
        model._sample_context(prediction, generator=None), prediction, revision=1
    )
    posterior = tree["query/posterior"]
    exact = values.astype(np.float64)
    # Consumers must get finite, nonzero spread, not float32 reduction under/overflow.
    for visible in (
        posterior["baseline"].isel(chain=0, date=0),
        posterior["channel_contribution"].isel(chain=0, date=0, channel=0),
        posterior["control_contribution"].isel(chain=0, date=0, control=0),
    ):
        np.testing.assert_allclose(visible.std("draw").item(), exact.std(), rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            visible.quantile([0.1, 0.9], dim="draw").values,
            np.quantile(exact, [0.1, 0.9]),
            rtol=1e-12,
            atol=0,
        )

    # Different level distributions and draw count expose transposed or inherited axes.
    draws = np.stack((values[:3], values[[0, 2, 4]]))[None]
    curves = ResponseCurves(
        channels=(prediction.fitted_context.channel_columns[0],),
        levels=np.array([0.0, 2.0]),
        media=np.array([100.0]),
        draws=draws,
        weeks=2,
        dates=query.time[:2],
        units=query.units,
        n_samples=3,
    )
    attached = attach_response_curves(tree, curves, prediction)
    visible = attached["response_curves/posterior"]["contribution"].isel(chain=0, channel=0)
    expected = draws[0].astype(np.float64)
    np.testing.assert_allclose(visible.std("draw").values, expected.std(axis=1), rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        visible.quantile([0.1, 0.9], dim="draw").values,
        np.quantile(expected, [0.1, 0.9], axis=1),
        rtol=1e-12,
        atol=0,
    )
