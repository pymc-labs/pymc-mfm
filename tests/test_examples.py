"""The example notebooks run top to bottom without a notebook kernel."""

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
import pytest
import torch
from test_graph_dot import native_dot, native_topology, svg_topology

from pymc_mfm.plotting.charts._data import IdataReader

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOK = EXAMPLES / "dev" / "diagnostics.ipynb"
QUICKSTART = EXAMPLES / "river_1_quickstart.ipynb"


def run_code_cells(notebook: Path) -> dict:
    namespace = {"__name__": "__main__"}
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for number, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{notebook.name}:cell-{number}", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("corpus", ["generated", "synthetic"])
def test_diagnostics_notebook_code_cells_run_headless(corpus, tmp_path, monkeypatch):
    # Matplotlib is the only optional dependency of the notebook's path; it ships
    # with the diagnostics and training extras. The generator extra is optional.
    pytest.importorskip("matplotlib")
    if corpus == "generated":
        pytest.importorskip("pymc_generator")
    else:
        # The notebook then builds its own synthetic corpus.
        monkeypatch.setitem(sys.modules, "pymc_generator", None)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    namespace = run_code_cells(NOTEBOOK)
    assert namespace["source"].startswith("synthetic") is (corpus == "synthetic")
    manifest = namespace["manifest"]
    assert manifest["status"] == "complete"
    rendered = {
        name for name, panel in manifest["panels"].items() if panel.get("render_status") == "ok"
    }
    assert {"attribution", "window_totals", "cumulative", "temporal", "reliability"} <= rendered


def assert_complete_quickstart_tree(model, context_result, tree, history, future, roles, n_samples):
    """Compare both named windows with their completed draws and original inputs."""
    assert set(tree.children) == {"context", "query"}
    assert not tree.coords
    assert tree.attrs["date_column"] == roles["date_column"]
    assert tree.attrs["target_column"] == roles["target_column"]
    for role in ("channel_columns", "control_columns"):
        assert json.loads(tree.attrs[role]) == roles[role]
    scalar_names = {
        "baseline": "baseline",
        "latent": "latent",
        "indirect:cc": "indirect_cc",
        "indirect:zc": "indirect_zc",
        "indirect:dc": "indirect_dc",
    }
    for window, result, frame in (
        ("context", context_result, history),
        ("query", model.query_result, future),
    ):
        assert result.n_samples == n_samples
        posterior = tree[window]["posterior"]
        assert posterior.sizes["chain"] == 1
        assert posterior.sizes["draw"] == n_samples
        assert posterior.sizes["date"] == len(frame)
        np.testing.assert_array_equal(posterior["date"].values, result.time)
        np.testing.assert_array_equal(
            posterior["graph"].values, result.graphs[None][..., result.edge_mask]
        )
        families = np.asarray(result.component_families)
        for family, columns, input_name in (
            ("channel", roles["channel_columns"], "media"),
            ("control", roles["control_columns"], "controls"),
        ):
            active = (families == family) & result.component_mask
            assert posterior[family].values.tolist() == columns
            np.testing.assert_array_equal(
                posterior[f"{family}_contribution"].values,
                result.components[None][..., active],
            )
            constants = tree[window]["constant_data"]
            assert constants[input_name].dims == ("date", family)
            np.testing.assert_array_equal(constants[input_name].values, frame[columns].to_numpy())
        for index, name in enumerate(result.component_names):
            if name in scalar_names and result.component_mask[index]:
                np.testing.assert_array_equal(
                    posterior[scalar_names[name]].values, result.components[None, ..., index]
                )
        np.testing.assert_array_equal(
            tree[window]["observed_data"]["outcome"].values,
            frame[roles["target_column"]].to_numpy(),
        )
        assert json.loads(tree[window].attrs["provenance"]) == result.provenance


def test_river_1_quickstart_runs_headless_with_a_stand_in_model(tmp_path, monkeypatch):
    # No model download: retain River-1's control/latent widths in a tiny random model.
    pytest.importorskip("matplotlib")
    pytest.importorskip("pymc_marketing")
    if shutil.which("dot") is None:
        pytest.skip("The quickstart's native SVG export requires Graphviz")
    import pymc_mfm
    from pymc_mfm import MFM, DecompositionResult, ModelConfig

    requests = []
    exports = {}

    class CapturedWorkspace(TemporaryDirectory):
        def cleanup(self):
            directory = Path(self.name) / "graphs"
            if directory.exists():
                exports.update(
                    (path.name, path.read_text(encoding="utf-8")) for path in directory.iterdir()
                )
            super().cleanup()

    def stand_in(repository, **options):
        requests.append((repository, options))
        torch.manual_seed(0)
        config = ModelConfig(
            d_model=8, n_layers=1, n_heads=2, d_ff=16, flow_hidden=16, n_controls=10, n_demand=1
        )
        return MFM(config).eval()

    contexts = []
    sample_context = MFM._sample_context

    def capture_context(self, prediction):
        contexts.append(sample_context(self, prediction))
        return contexts[-1]

    monkeypatch.setattr(MFM, "_sample_context", capture_context)
    monkeypatch.setattr("tempfile.TemporaryDirectory", CapturedWorkspace)
    monkeypatch.setattr(pymc_mfm, "from_pretrained", stand_in)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    namespace = run_code_cells(QUICKSTART)
    [(repository, options)] = requests
    assert (repository, options["profile"]) == ("pymc-labs/mfm", "river-1")
    assert re.fullmatch(r"[0-9a-f]{40}", options["revision"])
    n_weeks, context = namespace["N_WEEKS"], namespace["CONTEXT"]
    assert context == int(0.8 * n_weeks)
    model, query, base_tree = namespace["model"], namespace["query"], namespace["idata"]
    assert model.query_result is query
    [context_result] = contexts
    assert_complete_quickstart_tree(
        model,
        context_result,
        base_tree,
        namespace["history"],
        namespace["future"],
        namespace["roles"],
        query.n_samples,
    )
    assert query.provenance["input_usage"] == namespace["usage"]
    for result, weeks in (
        (context_result, context),
        (query, n_weeks - context),
    ):
        assert result.components.shape[:2] == (1_000, weeks)
        assert np.isfinite(result.components).all()
    curves = namespace["curves"]
    assert curves.draws.shape == (2, 41, 1_000) and curves.weeks == 8
    np.testing.assert_array_equal(IdataReader(model.idata).response_curves().draws, curves.draws)
    assert model.idata is not base_tree
    assert set(model.idata.children) == {"context", "query", "response_curves"}
    assert model.idata.attrs["inference_revision"] == base_tree.attrs["inference_revision"] == 1
    for window in ("context", "query"):
        assert model.idata[window].identical(base_tree[window])
    curve_tree = model.idata["response_curves"]
    np.testing.assert_array_equal(
        curve_tree["posterior"]["contribution"].values, curves.draws.transpose(2, 0, 1)[None]
    )
    np.testing.assert_array_equal(
        curve_tree["constant_data"]["media"].values, curves.media[:, None] * curves.levels
    )
    np.testing.assert_array_equal(curve_tree["constant_data"]["date"].values, curves.dates)
    assert json.loads(curve_tree.attrs["source_provenance"]) == query.provenance
    assert curve_tree.attrs["source_inference_revision"] == 1

    # Independently derive every catalog row and topology from this exact query bank.
    active_graphs = query.graphs[:, query.edge_mask]
    graphs, counts = np.unique(active_graphs, axis=0, return_counts=True)
    expected = sorted(
        zip(graphs.astype(int).tolist(), counts.tolist(), strict=True),
        key=lambda record: (-record[1], record[0]),
    )
    records, catalog = namespace["graph_records"], namespace["graph_catalog"]
    assert [record["graph"] for record in records] == [graph for graph, _ in expected]
    assert [record["count"] for record in records] == [count for _, count in expected]
    assert catalog.index.tolist() == list(range(len(expected)))
    np.testing.assert_array_equal(catalog["draw_count"], [count for _, count in expected])
    np.testing.assert_array_equal(catalog["edge_count"], [sum(graph) for graph, _ in expected])
    frequencies = np.asarray([count for _, count in expected]) / query.n_samples
    np.testing.assert_array_equal(catalog["frequency"], frequencies)
    np.testing.assert_array_equal([record["frequency"] for record in records], frequencies)
    endpoints = np.asarray(query.provenance["edge_endpoints"])[query.edge_mask].tolist()
    evidence = namespace["graph_evidence"]
    assert [edge["endpoints"] for edge in evidence["edges"]] == endpoints
    np.testing.assert_array_equal(
        [edge["probability"] for edge in evidence["edges"]], active_graphs.mean(axis=0)
    )
    rank = namespace["selected_graph_rank"]
    expected_connections = [
        {"source": pair[0], "target": pair[1]}
        for pair, present in zip(endpoints, expected[rank][0], strict=True)
        if present
    ]
    assert namespace["selected_connections"] == expected_connections
    assert namespace["ranks_to_plot"] == list(catalog.index[:3])
    nodes = {node for pair in endpoints for node in pair}
    selected_pairs = [(entry["source"], entry["target"]) for entry in expected_connections]
    average_pairs = [
        tuple(pair)
        for pair, probability in zip(endpoints, active_graphs.mean(axis=0), strict=True)
        if probability
    ]
    assert set(exports) == {"graph-average.dot", f"graph-{rank}.dot", f"graph-{rank}.svg"}
    assert exports["graph-average.dot"] == namespace["average_dot"] == query.plot.dot()
    assert exports[f"graph-{rank}.dot"] == namespace["selected_dot"] == query.plot.dot(graph=rank)
    average_document = native_dot(exports["graph-average.dot"])
    selected_document = native_dot(exports[f"graph-{rank}.dot"])
    average_nodes, average_edges = native_topology(average_document)
    selected_nodes, selected_edges = native_topology(selected_document)
    assert average_nodes == selected_nodes == nodes
    assert Counter(average_edges) == Counter(average_pairs)
    assert Counter(selected_edges) == Counter(selected_pairs)
    assert Counter(
        (*pair, float(edge["probability"]), edge["label"])
        for pair, edge in zip(average_edges, average_document.get("edges", []), strict=True)
    ) == Counter(
        (*pair, float(probability), f"{float(probability):.3g}")
        for pair, probability in zip(endpoints, active_graphs.mean(axis=0), strict=True)
        if probability
    )
    assert all("label" not in edge for edge in selected_document.get("edges", []))
    svg_nodes, svg_edges = svg_topology(ElementTree.fromstring(exports[f"graph-{rank}.svg"]))
    assert svg_nodes == nodes
    assert Counter(svg_edges) == Counter(selected_pairs)
    restored = DecompositionResult.load(query.save(tmp_path / "query-evidence"))
    np.testing.assert_array_equal(restored.components, query.components)
    np.testing.assert_array_equal(restored.graphs, query.graphs)
    assert restored.provenance == query.provenance
    assert restored.plot.dot(graph=rank) == namespace["selected_dot"]
