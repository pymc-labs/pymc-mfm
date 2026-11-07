"""Complete-graph topology selection and safe, portable inspector artifacts."""

import copy
import itertools
import json
import re
from collections import Counter
from dataclasses import replace
from html.parser import HTMLParser

import numpy as np
import pytest

from pymc_mfm.inference.result import DecompositionResult, result_from_batch
from pymc_mfm.model.config import ModelConfig
from pymc_mfm.model.graph import EdgeSlotLayout
from pymc_mfm.plotting import plot_dag, plot_graph_inspector

pytest.importorskip("matplotlib")

PAIRS = (
    ("channel:paid_search", "outcome"),
    ("control:price", "channel:paid_search"),
    ("channel:paid_search", "channel:tv_spend"),
    ("channel:tv_spend", "outcome"),
    ("control:price", "control:temperature"),
    ("control:temperature", "outcome"),
)
A = [1, 1, 0, 0, 0, 0]
B = [0, 0, 1, 0, 0, 1]
EMPTY = [0] * 6


def synthetic_result(patterns=None, *, mode="model_average", extra_channels=0):
    """Named live nodes around a padded middle channel; known complete topologies."""
    patterns = [A, A, A, B, B, EMPTY, EMPTY] if patterns is None else patterns
    patterns = np.asarray(patterns, dtype=int)
    width = 3 + extra_channels
    layout = EdgeSlotLayout(width, 2, 1)
    draws = len(patterns)
    components = np.zeros((draws, 1, 3, width + 7))
    components[..., 0] = 10
    result = result_from_batch(
        {
            "active_c_mask": np.array([[1, 0, 1, *([1] * extra_channels)]]),
            "active_m_mask": np.array([[1, 1]]),
            "active_j_mask": np.array([[0]]),
            "ctx_spend_z": np.zeros((1, 4, width)),
            "qry_spend_z": np.zeros((1, 3, width)),
            "sales_scale": np.ones(1),
        },
        components,
        np.zeros((draws, 1, layout.n_slots), dtype=int),
        model_config=ModelConfig(n_controls=2, n_demand=1),
        channel_names=(
            "paid_search",
            "padding",
            "tv_spend",
            *(f"extra_{index}" for index in range(extra_channels)),
        ),
        control_names=("price", "temperature"),
        mode=mode,
        units="revenue",
    )
    indices = {
        tuple(pair): index
        for index, pair in enumerate(result.provenance["edge_endpoints"])
        if result.edge_mask[index]
    }
    graphs = result.graphs.copy()
    for column, pair in enumerate(PAIRS):
        graphs[:, indices[pair]] = patterns[:, column]
    return replace(result, graphs=graphs)


def arrow_pairs(axis, result):
    endpoints = result.provenance["edge_endpoints"]
    return {
        tuple(endpoints[int(patch.get_gid().removeprefix("edge-"))]): patch
        for patch in axis.patches
    }


def expected_pairs(pattern):
    return {pair for pair, present in zip(PAIRS, pattern, strict=True) if present}


def test_rank_selects_complete_topology_with_padding_ties_and_isolated_nodes():
    result = synthetic_result()
    diagnostics = result.diagnostics()
    before = copy.deepcopy(diagnostics.summary)
    original = result.graphs.copy()
    # Count 3 wins; among the two count-2 graphs, the empty vector sorts first.
    for rank, pattern in enumerate((A, EMPTY, B)):
        for plots in (result.plot, diagnostics.plot):
            figure = plots.dag(graph=rank)
            assert set(arrow_pairs(figure.axes[0], result)) == expected_pairs(pattern)
            assert {text.get_text() for text in figure.axes[0].texts} == {
                "channel:paid_search",
                "channel:tv_spend",
                "control:price",
                "control:temperature",
                "outcome",
            }
            for patch in figure.axes[0].patches:
                assert patch.get_linewidth() == pytest.approx(2.6)
                assert patch.get_alpha() == pytest.approx(1)
            title = figure._suptitle.get_text()
            assert int(re.search(r"rank (\d+)", title)[1]) == rank
            counts = re.search(r"count (\d+) / (\d+) draws", title)
            assert tuple(map(int, counts.groups())) == ((3, 2, 2)[rank], 7)
            frequency = float(re.search(r"observed frequency ([\d.e+-]+)", title)[1])
            assert frequency == pytest.approx((3, 2, 2)[rank] / 7, rel=0.01)
            assert "binary" in title and "indicator" in title
            assert all(
                value in title for value in ("conditional", "query", "model_average", "revenue")
            )
    assert diagnostics.summary == before
    np.testing.assert_array_equal(result.graphs, original)


def test_default_view_preserves_marginal_edges_and_frequency_encoding():
    result = synthetic_result()
    patches = arrow_pairs(result.plot.dag().axes[0], result)
    expected = {
        **dict.fromkeys(expected_pairs(A), 3 / 7),
        **dict.fromkeys(expected_pairs(B), 2 / 7),
    }
    assert set(patches) == set(expected)
    for pair, frequency in expected.items():
        assert patches[pair].get_linewidth() == pytest.approx(0.6 + 2 * frequency)
        assert patches[pair].get_alpha() == pytest.approx(0.3 + 0.7 * frequency)


@pytest.mark.parametrize("mode", ["map", "fixed"])
def test_selected_hypothesis_has_no_invented_alternative_graphs(mode):
    result = synthetic_result([A, A], mode=mode)
    figure = result.plot.dag(graph=0)
    assert set(arrow_pairs(figure.axes[0], result)) == expected_pairs(A)
    title = figure._suptitle.get_text()
    assert all(value in title for value in (mode, "hypothesis", "2 draws", "revenue"))
    assert re.search(r"\bnot\b.*\bposterior\b", title)
    with pytest.raises(ValueError):
        result.plot.dag(graph=1)


def test_long_vector_chart_identifiers_select_the_same_complete_topologies():
    result = synthetic_result(extra_channels=3)
    assert result.edge_mask.sum() > 28
    chart = result.plot.graph_posterior().axes[0]
    identifiers = [
        int(re.search(r"\b(\d+)\b", tick.get_text())[1]) for tick in chart.get_yticklabels()
    ]
    assert identifiers == [0, 1, 2]
    for rank, pattern, bar in zip(identifiers, (A, EMPTY, B), chart.patches, strict=True):
        assert set(arrow_pairs(result.plot.dag(graph=rank).axes[0], result)) == expected_pairs(
            pattern
        )
        assert bar.get_width() == pytest.approx((3, 2, 2)[rank] / 7)


def test_selected_graph_and_supplied_truth_have_independent_topologies():
    result = synthetic_result()
    truth, prediction = result.diagnostics(graph_truth=result.graphs[3]).plot.dag(graph=0).axes
    assert set(arrow_pairs(truth, result)) == expected_pairs(B)
    assert set(arrow_pairs(prediction, result)) == expected_pairs(A)


@pytest.mark.parametrize("rank", [True, False, np.bool_(True), -1, 3, 0.0, "0", [0]])
def test_invalid_rank_is_not_silently_coerced_or_indexed_from_end(rank):
    with pytest.raises(ValueError, match="graph"):
        synthetic_result().plot.dag(graph=rank)


def test_numpy_integer_rank_and_rendering_from_serialized_summary(tmp_path):
    result = synthetic_result()
    summary = json.loads(json.dumps(result.diagnostics().summary))
    figure = plot_dag(summary, graph=np.int64(2))
    assert set(arrow_pairs(figure.axes[0], result)) == expected_pairs(B)
    output = tmp_path / "selected.svg"
    figure.savefig(output)
    # The exported artifact retains the selected topology's exact edge identities.
    from xml.etree import ElementTree

    root = ElementTree.parse(output).getroot()
    exported = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
    expected = {
        f"edge-{index}"
        for index, pair in enumerate(result.provenance["edge_endpoints"])
        if tuple(pair) in expected_pairs(B)
    }
    assert {identity for identity in exported if identity.startswith("edge-")} == expected


def test_singleton_zero_graph_retains_nodes_and_saved_result_can_render_it(tmp_path):
    result = synthetic_result([EMPTY])
    result.save(tmp_path / "result")
    restored = DecompositionResult.load(tmp_path / "result")
    for graph in (None, 0):
        axis = restored.plot.dag(graph=graph).axes[0]
        assert arrow_pairs(axis, restored) == {}
        assert {text.get_text() for text in axis.texts} == {node for pair in PAIRS for node in pair}
    inspector = restored.plot.graph_inspector()
    assert inspector.save(tmp_path / "zero.html") == tmp_path / "zero.html"


@pytest.mark.parametrize("missing", ["endpoints", "active_slots"])
def test_unavailable_topology_fails_for_both_renderers(missing):
    result = synthetic_result()
    if missing == "endpoints":
        provenance = {
            key: value for key, value in result.provenance.items() if key != "edge_endpoints"
        }
        result = replace(result, provenance=provenance)
    else:
        result = replace(
            result, edge_mask=np.zeros_like(result.edge_mask), graphs=np.zeros_like(result.graphs)
        )
    for render in (result.plot.dag, result.plot.graph_inspector):
        with pytest.raises(ValueError):
            render()


class InspectorDocument(HTMLParser):
    """Parse the exported document, not its JavaScript implementation."""

    def __init__(self, content):
        super().__init__(convert_charrefs=True)
        self.frames = []
        self.script_attrs = []
        self.payload = ""
        self._data = False
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "iframe":
            self.frames.append(attrs)
        if tag == "script":
            self.script_attrs.append(attrs)
            self._data = attrs.get("id") == "graph-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self._data = False

    def handle_data(self, data):
        if self._data:
            self.payload += data


def test_inspector_includes_every_graph_without_mutating_or_embedding_component_draws(tmp_path):
    patterns = list(itertools.product((0, 1), repeat=6))
    result = synthetic_result([*patterns, A, A, A, A, B, B, EMPTY, EMPTY])
    summary = result.diagnostics().summary
    before = copy.deepcopy(summary)
    inspector = plot_graph_inspector(summary)
    path = inspector.save(tmp_path / "graphs.html")
    document = InspectorDocument(path.read_text())
    payload = json.loads(document.payload)
    # Raw draws independently determine every ranked vector's count and frequency.
    counts = Counter(tuple(row[result.edge_mask]) for row in result.graphs)
    ordered = sorted(counts, key=lambda vector: (-counts[vector], vector))
    expected = [
        {
            "rank": rank,
            "graph": list(vector),
            "count": counts[vector],
            "frequency": counts[vector] / result.n_samples,
        }
        for rank, vector in enumerate(ordered)
    ]
    assert payload["graphs"] == expected
    assert summary == before
    assert "components" not in payload and "observed" not in payload and "provenance" not in payload


def test_hostile_column_names_remain_literal_data_in_standalone_and_isolated_notebook(tmp_path):
    result = synthetic_result()
    hostile = (
        '</script><script src="https://invalid.example/injected.js"></script>'
        "\u2028&<svg onload=alert(1)>"
    )
    provenance = copy.deepcopy(result.provenance)
    provenance["edge_endpoints"] = [
        [node.replace("paid_search", hostile) for node in pair]
        for pair in provenance["edge_endpoints"]
    ]
    result = replace(result, provenance=provenance)
    inspector = result.plot.graph_inspector()
    standalone = inspector.save(tmp_path / "hostile.html").read_text()
    document = InspectorDocument(standalone)
    payload = json.loads(document.payload)
    assert not any(attrs.get("src") for attrs in document.script_attrs)
    assert any(hostile in node["name"] for node in payload["nodes"])
    notebook = InspectorDocument(inspector._repr_html_())
    assert len(notebook.frames) == 1
    frame = notebook.frames[0]
    assert frame.get("title")
    assert {"allow-scripts", "allow-downloads"} <= set(frame["sandbox"].split())
    assert "allow-same-origin" not in frame["sandbox"].split()
    embedded = InspectorDocument(frame["srcdoc"])
    assert json.loads(embedded.payload) == json.loads(document.payload)
