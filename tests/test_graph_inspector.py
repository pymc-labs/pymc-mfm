"""Complete-graph topology selection and safe, portable inspector artifacts."""

import copy
import itertools
import json
import shutil
import subprocess
from collections import Counter
from dataclasses import replace
from html.parser import HTMLParser
from xml.etree import ElementTree

import numpy as np
import pytest

from pymc_mktfm.inference.result import DecompositionResult, result_from_batch
from pymc_mktfm.model.config import ModelConfig
from pymc_mktfm.model.graph import EdgeSlotLayout
from pymc_mktfm.plotting import graph_to_dot, plot_dag, plot_graph_inspector

pytestmark = pytest.mark.skipif(
    shutil.which("dot") is None, reason="Native Graphviz is not installed"
)

SVG = "{http://www.w3.org/2000/svg}"

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


def svg_topology(root):
    """Read literal labels and topology from native Graphviz SVG groups."""
    groups = list(root.iter(f"{SVG}g"))
    names = {
        group.findtext(f"{SVG}title"): "\n".join(
            text.text or "" for text in group.findall(f"{SVG}text")
        )
        for group in groups
        if group.get("class") == "node"
    }
    pairs, labels = [], []
    for group in groups:
        if group.get("class") != "edge":
            continue
        source, target = group.findtext(f"{SVG}title").split("->")
        pairs.append((names[source], names[target]))
        labels.append([text.text or "" for text in group.findall(f"{SVG}text")])
    return list(names.values()), pairs, labels


def rendered_svg(graph):
    return ElementTree.fromstring(graph.render(format="svg"))


def inspector_payload(inspector):
    return json.loads(InspectorDocument(inspector.html).payload)


def inspector_dot(payload, rank):
    """Materialize an exported selection from the portable DOT template."""
    template = payload["dot"]
    values = (
        [edge["value"] for edge in payload["edges"]]
        if rank is None
        else payload["graphs"][rank]["graph"]
    )
    fragments = template["average" if rank is None else "individual"]
    view = "// Average\n" if rank is None else f"// Complete graph rank {rank}\n"
    body = "\n".join(
        fragment for fragment, value in zip(fragments, values, strict=True) if value > 0
    )
    return view + template["preamble"] + body + template["suffix"]


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
            nodes, pairs, labels = svg_topology(rendered_svg(plots.dag(graph=rank)))
            assert set(pairs) == expected_pairs(pattern)
            assert len(pairs) == len(expected_pairs(pattern))
            assert Counter(nodes) == Counter({node for pair in PAIRS for node in pair})
            assert all(not label for label in labels)
    assert diagnostics.summary == before
    np.testing.assert_array_equal(result.graphs, original)


def test_default_view_preserves_marginal_edges_and_numeric_frequencies():
    result = synthetic_result()
    nodes, pairs, labels = svg_topology(rendered_svg(result.plot.dag()))
    expected = {
        **dict.fromkeys(expected_pairs(A), 3 / 7),
        **dict.fromkeys(expected_pairs(B), 2 / 7),
    }
    assert set(pairs) == set(expected)
    assert len(pairs) == len(expected)
    assert Counter(nodes) == Counter({node for pair in PAIRS for node in pair})
    for pair, label in zip(pairs, labels, strict=True):
        assert len(label) == 1
        assert float(label[0]) == pytest.approx(expected[pair], abs=0.005)


@pytest.mark.parametrize("mode", ["map", "fixed"])
def test_selected_hypothesis_has_no_invented_alternative_graphs(mode):
    result = synthetic_result([A, A], mode=mode)
    _, pairs, _ = svg_topology(rendered_svg(result.plot.dag(graph=0)))
    assert set(pairs) == expected_pairs(A)
    payload = inspector_payload(result.plot.graph_inspector())
    assert payload["metadata"]["mode"] == mode
    assert len(payload["graphs"]) == 1
    assert payload["graphs"][0]["count"] == 2
    assert payload["graphs"][0]["frequency"] == 1
    with pytest.raises(ValueError):
        result.plot.dag(graph=1)


def test_long_binary_vectors_keep_global_ranks_and_complete_topologies():
    result = synthetic_result(extra_channels=3)
    assert result.edge_mask.sum() > 28
    payload = inspector_payload(result.plot.graph_inspector())
    assert [record["rank"] for record in payload["graphs"]] == [0, 1, 2]
    for rank, pattern, record in zip(range(3), (A, EMPTY, B), payload["graphs"], strict=True):
        _, pairs, _ = svg_topology(rendered_svg(result.plot.dag(graph=rank)))
        assert set(pairs) == expected_pairs(pattern)
        assert record["frequency"] == pytest.approx((3, 2, 2)[rank] / 7)
        _, inspector_pairs, _ = svg_topology(ElementTree.fromstring(payload["svg"][rank + 1]))
        assert Counter(inspector_pairs) == Counter(pairs)


@pytest.mark.parametrize("rank", [True, False, np.bool_(True), -1, 3, 0.0, "0", [0]])
def test_invalid_rank_is_not_silently_coerced_or_indexed_from_end(rank):
    with pytest.raises(ValueError, match="graph"):
        synthetic_result().plot.dag(graph=rank)


def test_numpy_integer_rank_and_rendering_from_serialized_summary(tmp_path):
    result = synthetic_result()
    summary = json.loads(json.dumps(result.diagnostics().summary))
    graph = plot_dag(summary, graph=np.int64(2))
    assert set(svg_topology(rendered_svg(graph))[1]) == expected_pairs(B)
    output = tmp_path / "selected.svg"
    graph.savefig(output)
    nodes, pairs, labels = svg_topology(ElementTree.parse(output).getroot())
    assert Counter(nodes) == Counter({node for pair in PAIRS for node in pair})
    assert set(pairs) == expected_pairs(B)
    assert len(pairs) == len(expected_pairs(B))
    assert all(not label for label in labels)


def test_singleton_zero_graph_retains_nodes_and_saved_result_can_render_it(tmp_path):
    result = synthetic_result([EMPTY])
    result.save(tmp_path / "result")
    restored = DecompositionResult.load(tmp_path / "result")
    for graph in (None, 0):
        nodes, pairs, _ = svg_topology(rendered_svg(restored.plot.dag(graph=graph)))
        assert pairs == []
        assert Counter(nodes) == Counter({node for pair in PAIRS for node in pair})
    inspector = restored.plot.graph_inspector()
    for source in inspector_payload(inspector)["svg"]:
        nodes, pairs, _ = svg_topology(ElementTree.fromstring(source))
        assert pairs == []
        assert Counter(nodes) == Counter({node for pair in PAIRS for node in pair})
    output = tmp_path / "zero.html"
    assert inspector.save(output) == output
    assert output.read_bytes() == inspector.html.encode("utf-8")


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
        self.policies = []
        self.resources = []
        self._data = False
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "iframe":
            self.frames.append(attrs)
        if tag == "script":
            self.script_attrs.append(attrs)
            self._data = attrs.get("id") == "graph-data"
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "content-security-policy":
            self.policies.append(attrs["content"])
        for attribute in ("src", "href", "data", "poster"):
            if attrs.get(attribute):
                self.resources.append((tag, attribute, attrs[attribute]))

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
    before_graphs = result.graphs.copy()
    inspector = plot_graph_inspector(summary)
    path = inspector.save(tmp_path / "graphs.html")
    document = InspectorDocument(path.read_text(encoding="utf-8"))
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
    assert len(payload["graphs"]) == 64
    assert len(payload["svg"]) == len(payload["graphs"]) + 1
    for record, source in zip(payload["graphs"], payload["svg"][1:], strict=True):
        nodes, pairs, labels = svg_topology(ElementTree.fromstring(source))
        assert Counter(nodes) == Counter({node for pair in PAIRS for node in pair})
        expected_edges = [
            (edge["source"], edge["target"])
            for edge, bit in zip(payload["edges"], record["graph"], strict=True)
            if bit
        ]
        assert Counter(pairs) == Counter(expected_edges)
        assert all(not label for label in labels)
    assert {edge["index"] for edge in payload["edges"]} == set(np.flatnonzero(result.edge_mask))
    assert summary == before
    assert "components" not in payload and "observed" not in payload and "provenance" not in payload
    np.testing.assert_array_equal(result.graphs, before_graphs)


def test_hostile_column_names_remain_literal_data_in_standalone_and_isolated_notebook(tmp_path):
    result = synthetic_result()
    hostile = (
        '</script><script src="https://invalid.example/injected.js"></script>'
        "\u2028&<svg onload=alert(1)> café 🐝"
    )
    provenance = copy.deepcopy(result.provenance)
    provenance["edge_endpoints"] = [
        [node.replace("paid_search", hostile) for node in pair]
        for pair in provenance["edge_endpoints"]
    ]
    result = replace(result, provenance=provenance)
    inspector = result.plot.graph_inspector()
    standalone = inspector.save(tmp_path / "hostile.html").read_text(encoding="utf-8")
    document = InspectorDocument(standalone)
    payload = json.loads(document.payload)
    assert not any(attrs.get("src") for attrs in document.script_attrs)
    assert any(hostile in node["name"] for node in payload["nodes"])
    assert len(document.script_attrs) == 2
    assert document.resources == []
    assert len(document.policies) == 1
    directives = {directive.strip() for directive in document.policies[0].split(";")}
    assert {"default-src 'none'", "base-uri 'none'", "form-action 'none'"} <= directives
    for source in payload["svg"]:
        root = ElementTree.fromstring(source)
        nodes, _, _ = svg_topology(root)
        assert "channel:" + hostile in nodes
        assert not any(
            element.tag in {f"{SVG}{tag}" for tag in ("script", "image", "foreignObject")}
            for element in root.iter()
        )
        assert not any(
            key.endswith("href") or key.lower().startswith("on")
            for element in root.iter()
            for key in element.attrib
        )
    sources = []
    for rank in (None, 0, 1, 2):
        source = inspector_dot(payload, rank)
        assert source == result.plot.dot(graph=rank)
        assert "\r" not in source
        assert source.endswith("\n") and not source.endswith("\n\n")
        sources.append(source)
    native = subprocess.run(
        [shutil.which("dot"), "-Tsvg"],
        input="\n".join(sources).encode("utf-8"),
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    assert "".join(payload["svg"]) == native
    notebook = InspectorDocument(inspector._repr_html_())
    assert len(notebook.frames) == 1
    frame = notebook.frames[0]
    assert frame.get("title")
    assert set(frame["sandbox"].split()) == {"allow-scripts", "allow-downloads"}
    assert frame["referrerpolicy"] == "no-referrer"
    embedded = InspectorDocument(frame["srcdoc"])
    assert json.loads(embedded.payload) == json.loads(document.payload)


def test_inspector_truth_does_not_change_prediction_svg_or_dot():
    result = synthetic_result()
    original = inspector_payload(result.plot.graph_inspector())
    with_truth = inspector_payload(
        result.diagnostics(graph_truth=result.graphs[3]).plot.graph_inspector()
    )
    assert original["has_graph_truth"] is False
    assert with_truth["has_graph_truth"] is True
    for key in ("nodes", "edges", "graphs", "dot", "svg"):
        assert with_truth[key] == original[key]


@pytest.mark.parametrize("codepoint", [0, 1, 8, 11, 12, 31, 0xD800, 0xDFFF, 0xFFFE, 0xFFFF])
def test_inspector_and_dot_reject_labels_native_svg_cannot_represent(codepoint):
    summary = copy.deepcopy(synthetic_result().diagnostics().summary)
    summary["arrays"]["graphs"]["edges"][0]["endpoints"][0] = f"bad{chr(codepoint)}label"
    for export in (graph_to_dot, plot_graph_inspector):
        with pytest.raises(ValueError, match="label|character|XML"):
            export(summary)


def test_inspector_rejects_misaligned_complete_graph_vectors():
    summary = copy.deepcopy(synthetic_result().diagnostics().summary)
    summary["arrays"]["graphs"]["complete_graphs"][0]["graph"].append(1)
    with pytest.raises(ValueError, match="align"):
        plot_graph_inspector(summary)
