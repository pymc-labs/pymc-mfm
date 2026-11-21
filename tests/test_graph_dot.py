"""Canonical DOT semantics, literal labels and native Graphviz artifacts."""

import copy
import io
import json
import re
import shutil
import subprocess
from xml.etree import ElementTree

import numpy as np
import pytest

from pymc_mktfm.inference.result import DecompositionResult
from pymc_mktfm.plotting import GraphvizGraph, graph_to_dot, plot_dag
from pymc_mktfm.plotting._graphviz import _render_svg_batch

ENDPOINTS = (
    ("feature", "target"),
    ("padding", "padding"),
    ("other", "target"),
    ("isolated", "target"),
)
DRAWS = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
NATIVE_GRAPHVIZ = pytest.mark.skipif(
    shutil.which("dot") is None, reason="Native graph rendering requires Graphviz"
)


def graph_result(draws=DRAWS, endpoints=ENDPOINTS, *, mask=(1, 0, 1, 1), mode="model_average"):
    """Small synthetic draw bank; a padded middle slot and isolated active endpoint."""
    return DecompositionResult(
        components=np.ones((len(draws), 2, 1)),
        graphs=np.asarray(draws),
        component_names=("baseline",),
        component_families=("baseline",),
        component_mask=np.ones(1),
        edge_names=tuple(f"edge_{index}" for index in range(len(endpoints))),
        edge_mask=np.asarray(mask),
        channel_names=(),
        control_names=(),
        time=np.arange(2),
        window="query",
        mode=mode,
        units="synthetic outcome",
        provenance={
            "edge_endpoints": [list(pair) for pair in endpoints],
            "edge_families": ["cy"] * len(endpoints),
            "sales_scale": 1.0,
        },
        context_length=2,
    )


def native_bytes(source, format):
    executable = shutil.which("dot")
    if executable is None:
        pytest.skip("Native graph rendering requires Graphviz")
    return subprocess.run(
        [executable, f"-T{format}"],
        input=source.encode("utf-8"),
        capture_output=True,
        check=True,
    ).stdout


def native_dot(source, format="json"):
    rendered = native_bytes(source, format)
    return json.loads(rendered) if format == "json" else ElementTree.fromstring(rendered)


def native_topology(document):
    names = {node["_gvid"]: node["label"] for node in document["objects"]}
    return set(names.values()), [
        (names[edge["tail"]], names[edge["head"]]) for edge in document.get("edges", [])
    ]


@pytest.mark.parametrize("api", ["standalone", "result", "diagnostics"])
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (None, {("feature", "target"), ("other", "target")}),
        (np.int64(0), {("feature", "target")}),
        (1, set()),
        (2, {("other", "target")}),
    ],
)
def test_export_prediction_topology_is_independent_of_truth_padding_and_rank_ties(
    api, rank, expected
):
    result = graph_result([[1, 0, 0, 0]] * 5 + [[0, 0, 1, 0], [0, 0, 0, 0]])
    diagnostics = result.diagnostics(graph_truth=np.asarray([0, 0, 1, 0]))
    summary = diagnostics.summary
    source = {
        "standalone": lambda: graph_to_dot(summary, graph=rank),
        "result": lambda: result.plot.dot(graph=rank),
        "diagnostics": lambda: diagnostics.plot.dot(graph=rank),
    }[api]()
    # Parse this ASCII fixture's statements even when native Graphviz is absent.
    identifier = r"[A-Za-z_]\w*"
    statements = re.findall(
        rf"\b({identifier})(?:\s*->\s*({identifier}))?\s*(?:\[([^\[\]]*)\])?\s*;",
        source,
    )
    names, exported_edges = {}, []
    for left, right, attributes in statements:
        attributes = dict(re.findall(r'(\w+)\s*=\s*"([^"\\]*)"', attributes))
        if right:
            exported_edges.append((left, right, attributes))
        elif left not in {"node", "edge", "graph"} and "label" in attributes:
            names[left] = attributes["label"]
    assert set(names.values()) == {"feature", "other", "isolated", "target"}
    pairs = [(names[left], names[right]) for left, right, _ in exported_edges]
    assert set(pairs) == expected
    assert len(pairs) == len(expected)
    if rank is None:
        frequencies = {
            pair: float(attrs["probability"])
            for pair, (_, _, attrs) in zip(pairs, exported_edges, strict=True)
        }
        assert frequencies == {("feature", "target"): 5 / 7, ("other", "target"): 1 / 7}
        for pair, (_, _, attrs) in zip(pairs, exported_edges, strict=True):
            assert float(attrs["label"]) == pytest.approx(frequencies[pair], abs=0.005)
    else:
        assert all("label" not in attrs for _, _, attrs in exported_edges)
    if shutil.which("dot") is None:
        return
    document = native_dot(source)
    nodes, edges = native_topology(document)
    assert nodes == {"feature", "other", "isolated", "target"}
    assert set(edges) == expected
    assert len(edges) == len(expected)
    if rank is None:
        values = {tuple(pair): edge for pair, edge in zip(edges, document["edges"], strict=True)}
        for pair, probability in {("feature", "target"): 5 / 7, ("other", "target"): 1 / 7}.items():
            assert float(values[pair]["probability"]) == probability
            assert float(values[pair]["label"]) == pytest.approx(probability, abs=0.005)
    else:
        assert all("label" not in edge for edge in document.get("edges", []))


@pytest.mark.parametrize("rank", [True, False, np.bool_(True), -1, 3, 0.0, "0", [0]])
def test_dot_rank_validation_across_public_apis(rank):
    result = graph_result()
    diagnostics = result.diagnostics()
    for export in (
        lambda: graph_to_dot(diagnostics.summary, graph=rank),
        lambda: result.plot.dot(graph=rank),
        lambda: diagnostics.plot.dot(graph=rank),
    ):
        with pytest.raises(ValueError, match="graph"):
            export()


def test_export_is_canonical_utf8_without_mutating_evidence(tmp_path):
    result = graph_result()
    summary = result.diagnostics().summary
    before = copy.deepcopy(summary)
    graphs = result.graphs.copy()
    for rank in (None, 0, 1, 2):
        source = graph_to_dot(summary, graph=rank)
        assert source.endswith("\n") and not source.endswith("\n\n")
        assert "\r" not in source
        path = tmp_path / f"{rank}.dot"
        path.write_text(source, encoding="utf-8", newline="\n")
        assert path.read_bytes() == source.encode("utf-8")
    assert summary == before
    np.testing.assert_array_equal(result.graphs, graphs)


@pytest.mark.parametrize("mode", ["model_average", "map", "fixed"])
def test_singleton_empty_graph_exports_all_active_nodes_without_edges(mode):
    result = graph_result([[0, 0, 0, 0]], mode=mode)
    for rank in (None, 0):
        nodes, edges = native_topology(native_dot(result.plot.dot(graph=rank)))
        assert nodes == {"feature", "other", "isolated", "target"}
        assert edges == []


def test_parallel_slots_are_not_merged_into_a_strict_graph():
    result = graph_result([[1, 1]], (("a", "b"), ("a", "b")), mask=(1, 1))
    for rank in (None, 0):
        nodes, edges = native_topology(native_dot(result.plot.dot(graph=rank)))
        assert nodes == {"a", "b"}
        assert edges == [("a", "b"), ("a", "b")]


@pytest.mark.parametrize(
    "label",
    [
        'quote " and colon:a',
        r"literal \N \G \E \T \H \L \n",
        "line one\n\nline two\rline three",
        "tab\there 測定 🐝",
        '<img src="file:///not-a-resource">',
        '"]; injected [image="https://not-a-resource.invalid"]; //',
    ],
)
@NATIVE_GRAPHVIZ
def test_native_svg_preserves_supported_hostile_labels_as_text(label):
    result = graph_result([[1]], ((label, "target"),), mask=(1,))
    root = ElementTree.fromstring(result.plot.dag().render("svg"))
    namespace = {"s": "http://www.w3.org/2000/svg"}
    groups = [group for group in root.findall(".//s:g", namespace) if group.get("class") == "node"]
    rendered_labels = [
        [text.text or "" for text in group.findall("s:text", namespace)] for group in groups
    ]
    expected_lines = [line for line in label.replace("\r", "\n").split("\n") if line]
    assert expected_lines in rendered_labels
    assert ["target"] in rendered_labels
    assert len(groups) == 2
    assert not root.findall(".//s:image", namespace)
    assert not any(key.endswith("href") for element in root.iter() for key in element.attrib)


@pytest.mark.parametrize("codepoint", [0, 1, 8, 11, 12, 31, 0xD800, 0xDFFF, 0xFFFE, 0xFFFF])
def test_dot_rejects_characters_that_native_svg_cannot_represent(codepoint):
    summary = copy.deepcopy(graph_result().diagnostics().summary)
    summary["arrays"]["graphs"]["edges"][0]["endpoints"][0] = f"bad{chr(codepoint)}label"
    for render in (graph_to_dot, plot_dag):
        with pytest.raises(ValueError, match="label|character|XML"):
            render(summary)


def svg_topology(document):
    namespace = {"s": "http://www.w3.org/2000/svg"}
    groups = document.findall(".//s:g", namespace)
    names = {
        group.find("s:title", namespace).text: "\n".join(
            text.text or "" for text in group.findall("s:text", namespace)
        )
        for group in groups
        if group.get("class") == "node"
    }
    return set(names.values()), [
        tuple(names[identifier] for identifier in group.find("s:title", namespace).text.split("->"))
        for group in groups
        if group.get("class") == "edge"
    ]


@NATIVE_GRAPHVIZ
@pytest.mark.parametrize("api", ["result", "diagnostics", "standalone"])
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (None, {("feature", "target"), ("other", "target")}),
        (np.int64(0), {("feature", "target")}),
        (1, set()),
        (2, {("other", "target")}),
    ],
)
def test_native_dag_renders_canonical_prediction_source_without_a_custom_template(
    api, rank, expected
):
    result = graph_result([[1, 0, 0, 0]] * 5 + [[0, 0, 1, 0], [0, 0, 0, 0]])
    diagnostics = result.diagnostics()
    summary = json.loads(json.dumps(diagnostics.summary))
    artifact = {
        "result": lambda: result.plot.dag(graph=rank),
        "diagnostics": lambda: diagnostics.plot.dag(graph=rank),
        "standalone": lambda: plot_dag(summary, graph=rank),
    }[api]()
    assert isinstance(artifact, GraphvizGraph)
    assert artifact.source == graph_to_dot(summary, graph=rank)
    svg = artifact.render("svg")
    assert svg == native_bytes(artifact.source, "svg")
    nodes, edges = svg_topology(ElementTree.fromstring(svg))
    assert nodes == {"feature", "other", "isolated", "target"}
    assert set(edges) == expected
    assert len(edges) == len(expected)
    png = artifact._repr_png_()
    assert png == native_bytes(artifact.source, "png")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


@NATIVE_GRAPHVIZ
@pytest.mark.parametrize("rank", [None, 0, 1, 2])
def test_truth_uses_separate_labeled_clusters_without_changing_prediction_export(rank):
    result = graph_result()
    diagnostics = result.diagnostics(graph_truth=np.array([0, 0, 1, 0]))
    artifact = diagnostics.plot.dag(graph=rank)
    document = native_dot(artifact.source)
    nodes = {node["_gvid"]: node for node in document["objects"] if "nodes" not in node}
    clusters = {
        cluster["label"]: (
            {nodes[index]["label"] for index in cluster["nodes"]},
            [
                (nodes[edge["tail"]]["label"], nodes[edge["head"]]["label"])
                for edge in document.get("edges", [])
                if edge["tail"] in cluster["nodes"] and edge["head"] in cluster["nodes"]
            ],
        )
        for cluster in document["objects"]
        if "nodes" in cluster
    }
    prediction_label = "Average" if rank is None else f"Complete graph rank {rank}"
    expected_nodes, expected_edges = native_topology(native_dot(result.plot.dot(graph=rank)))
    assert clusters == {
        "True graph": (expected_nodes, [("other", "target")]),
        prediction_label: (expected_nodes, expected_edges),
    }
    svg = artifact.render("svg")
    assert svg == native_bytes(artifact.source, "svg")
    root = ElementTree.fromstring(svg)
    namespace = {"s": "http://www.w3.org/2000/svg"}
    labels = {
        group.find("s:text", namespace).text
        for group in root.findall(".//s:g", namespace)
        if group.get("class") == "cluster"
    }
    assert labels == {"True graph", prediction_label}
    assert diagnostics.plot.dot(graph=rank) == result.plot.dot(graph=rank)


@NATIVE_GRAPHVIZ
def test_native_dag_keeps_parallel_slots_cycles_and_self_loops():
    pairs = (("a", "b"), ("a", "b"), ("b", "c"), ("c", "a"), ("b", "a"), ("a", "a"))
    result = graph_result([[1] * len(pairs)], pairs, mask=(1,) * len(pairs))
    artifact = result.plot.dag(graph=0)
    nodes, edges = svg_topology(ElementTree.fromstring(artifact.render("svg")))
    assert nodes == {"a", "b", "c"}
    assert sorted(edges) == sorted(pairs)


@NATIVE_GRAPHVIZ
@pytest.mark.parametrize("rank", [None, 0])
def test_native_empty_selection_retains_every_active_endpoint(rank):
    artifact = graph_result([[0, 0, 0, 0]]).plot.dag(graph=rank)
    nodes, edges = svg_topology(ElementTree.fromstring(artifact.render("svg")))
    assert nodes == {"feature", "other", "isolated", "target"}
    assert edges == []


@NATIVE_GRAPHVIZ
def test_native_artifact_exports_real_vector_svg_and_png_to_paths_and_binary_files(tmp_path):
    artifact = graph_result().plot.dag()
    for suffix, format in ((".png", "png"), (".SVG", "svg"), ("", "png")):
        path = tmp_path / f"graph{suffix}"
        artifact.savefig(path)
        expected = native_bytes(artifact.source, format)
        assert path.read_bytes() == expected
        buffer = io.BytesIO()
        artifact.savefig(buffer, format=format)
        assert buffer.getvalue() == expected
    root = ElementTree.fromstring((tmp_path / "graph.SVG").read_bytes())
    namespace = {"s": "http://www.w3.org/2000/svg"}
    assert root.findall(".//s:ellipse", namespace)
    assert root.findall(".//s:path", namespace)
    assert not root.findall(".//s:image", namespace)
    default_buffer = io.BytesIO()
    artifact.savefig(default_buffer)
    assert default_buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    overridden = tmp_path / "vector.png"
    with overridden.open("wb") as handle:
        artifact.savefig(handle, format="svg")
    assert ElementTree.fromstring(overridden.read_bytes()).tag.endswith("svg")
    with pytest.raises(TypeError, match="dpi"):
        artifact.savefig(tmp_path / "ignored.png", dpi=120)
    with pytest.raises(TypeError, match="bbox_inches"):
        artifact.savefig(tmp_path / "ignored.svg", bbox_inches="tight")


@NATIVE_GRAPHVIZ
def test_native_artifact_can_render_a_graph_without_nodes():
    artifact = GraphvizGraph("digraph {}\n")
    assert svg_topology(ElementTree.fromstring(artifact.render("svg"))) == (set(), [])
    assert artifact.render().startswith(b"\x89PNG\r\n\x1a\n")


@NATIVE_GRAPHVIZ
@pytest.mark.parametrize("source", ["not a graph", "digraph { a -> }", ""])
@pytest.mark.parametrize("format", ["png", "svg"])
def test_native_artifact_surfaces_invalid_dot(source, format):
    with pytest.raises(ValueError, match="Graphviz dot"):
        GraphvizGraph(source).render(format)


def test_missing_dot_is_explicit_but_dot_export_still_works(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("dot")

    monkeypatch.setattr("pymc_mktfm.plotting._graphviz.subprocess.run", missing)
    result = graph_result()
    artifact = result.plot.dag()
    assert artifact.source == result.plot.dot()
    for render in (artifact.render, lambda: _render_svg_batch([artifact.source])):
        with pytest.raises(RuntimeError, match="Graphviz.*dot.*PATH"):
            render()


def test_native_artifact_rejects_unsupported_formats(tmp_path):
    artifact = GraphvizGraph("digraph {}")
    with pytest.raises(ValueError, match="format"):
        artifact.render("jpeg")
    with pytest.raises(ValueError, match="format"):
        artifact.savefig(tmp_path / "graph.jpeg")


@NATIVE_GRAPHVIZ
def test_svg_batch_preserves_complete_documents_order_and_bytes_in_one_process(monkeypatch):
    result = graph_result()
    sources = [result.plot.dot(graph=rank) for rank in (None, 2, 1, 0)]
    expected = native_bytes("\n".join(sources), "svg").decode("utf-8")
    expected_topologies = [native_topology(native_dot(source)) for source in sources]
    run = subprocess.run
    calls = []

    def counting(*args, **kwargs):
        calls.append(None)
        return run(*args, **kwargs)

    monkeypatch.setattr("pymc_mktfm.plotting._graphviz.subprocess.run", counting)
    documents = _render_svg_batch(sources)
    assert "".join(documents) == expected
    assert [
        svg_topology(ElementTree.fromstring(document)) for document in documents
    ] == expected_topologies
    assert len(calls) == 1
    assert _render_svg_batch([]) == []
    assert len(calls) == 1


@NATIVE_GRAPHVIZ
@pytest.mark.parametrize(
    "sources",
    [["digraph { a -> b; }\ndigraph { c -> d; }\n"], ["digraph { a; }\n", ""]],
)
def test_svg_batch_rejects_missing_or_extra_graph_documents(sources):
    with pytest.raises(RuntimeError, match="document count"):
        _render_svg_batch(sources)
