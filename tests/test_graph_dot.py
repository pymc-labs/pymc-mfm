"""Independent DOT semantics, literal labels and observed-support layout invariants."""

import copy
import json
import re
import shutil
import subprocess
from xml.etree import ElementTree

import numpy as np
import pytest

from pymc_mfm.inference.result import DecompositionResult
from pymc_mfm.plotting import graph_to_dot
from pymc_mfm.plotting._graphs import graph_positions

ENDPOINTS = (
    ("feature", "target"),
    ("padding", "padding"),
    ("other", "target"),
    ("isolated", "target"),
)
DRAWS = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]


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


def native_dot(source, format="json"):
    executable = shutil.which("dot")
    if executable is None:
        pytest.skip("Optional native Graphviz is not installed")
    rendered = subprocess.run(
        [executable, f"-T{format}"],
        input=source.encode("utf-8"),
        capture_output=True,
        check=True,
    ).stdout
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
def test_native_svg_preserves_supported_hostile_labels_as_text(label):
    result = graph_result([[1]], ((label, "target"),), mask=(1,))
    root = native_dot(result.plot.dot(), "svg")
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
    with pytest.raises(ValueError, match="label|character|XML"):
        graph_to_dot(summary)


def edge(source, target, probability=1):
    return {"endpoints": [source, target], "probability": probability}


def test_layout_uses_observed_scc_longest_paths_and_retains_isolated_nodes():
    edges = [
        edge("a", "b"),
        edge("b", "c"),
        edge("a", "c"),
        edge("c", "d"),
        edge("d", "e"),
        edge("e", "d"),
        edge("e", "f"),
        edge("f", "a", 0),
        edge("isolated", "f", 0),
    ]
    positions = graph_positions(edges)
    ranks = {name: position[0] for name, position in positions.items()}
    assert ranks["a"] == ranks["isolated"] == 0
    assert ranks["b"] == ranks["a"] + 1
    assert ranks["c"] == ranks["b"] + 1
    assert ranks["d"] == ranks["e"] == ranks["c"] + 1
    assert ranks["f"] == ranks["e"] + 1
    assert positions == graph_positions(list(reversed(edges)))
    assert list(positions) == list(graph_positions(list(reversed(edges))))


def test_deep_path_layout_has_no_python_recursion_ceiling():
    positions = graph_positions([edge(f"n{index}", f"n{index + 1}") for index in range(1200)])
    assert len(positions) == 1201
    assert positions["n1200"][0] - positions["n0"][0] == 1200


def rendered_graph(result, graph=0):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Ellipse, FancyArrowPatch

    figure = result.plot.dag(graph=graph)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    axis = figure.axes[0]
    ellipses = [patch for patch in axis.patches if isinstance(patch, Ellipse)]
    arrows = [patch for patch in axis.patches if isinstance(patch, FancyArrowPatch)]
    return figure, axis, canvas.get_renderer(), ellipses, arrows


def test_minimal_graph_matches_vertical_reference_and_stable_selected_geometry():
    result = graph_result([[1], [0]], (("a", "b"),), mask=(1,))
    centers = []
    for rank in (None, 0, 1):
        _, axis, _, ellipses, arrows = rendered_graph(result, rank)
        labels = {
            text.get_text(): axis.transData.transform(text.get_position())
            for text in axis.texts
            if text.get_text() in {"a", "b"}
        }
        assert labels["a"][0] == pytest.approx(labels["b"][0])
        assert labels["a"][1] > labels["b"][1]
        centers.append([ellipse.center for ellipse in ellipses])
        # The count tie ranks the empty vector first, and the one-edge vector second.
        assert len(arrows) == (0 if rank == 0 else 1)
        for arrow in arrows:
            path = arrow.get_path().transformed(arrow.get_transform())
            # Exclude the triangular arrow head; the connector itself is straight.
            connector, _ = next(path.iter_bezier())
            start_x = connector(0)[0]
            for index, (segment, code) in enumerate(path.iter_bezier()):
                if index and code == 1:
                    break
                points = segment(np.linspace(0, 1, 10))
                assert points[:, 0] == pytest.approx(np.full(len(points), start_x))
    assert centers[0] == centers[1] == centers[2]


def test_multiline_literal_labels_fit_inside_ellipses_without_colliding_nodes():
    labels = [
        f"long feature {index}: Δωρεά $x$ and literal \\\\n\n\nlast line" for index in range(4)
    ]
    result = graph_result([[1] * 4], tuple((label, "target") for label in labels), mask=(1,) * 4)
    _, axis, renderer, ellipses, _ = rendered_graph(result)
    assert sorted(text.get_text() for text in axis.texts) == sorted([*labels, "target"])
    bounds = [ellipse.get_window_extent(renderer) for ellipse in ellipses]
    for text in axis.texts:
        assert not text.get_parse_math()
        text_bounds = text.get_window_extent(renderer)
        center = axis.transData.transform(text.get_position())
        matching = [
            bound
            for bound in bounds
            if np.allclose([(bound.x0 + bound.x1) / 2, (bound.y0 + bound.y1) / 2], center)
        ]
        assert len(matching) == 1
        ellipse = matching[0]
        for x in (text_bounds.x0, text_bounds.x1):
            for y in (text_bounds.y0, text_bounds.y1):
                distance = ((x - center[0]) / (ellipse.width / 2)) ** 2
                distance += ((y - center[1]) / (ellipse.height / 2)) ** 2
                assert distance < 1
    for index, first in enumerate(bounds):
        for second in bounds[index + 1 :]:
            assert not first.overlaps(second)


def test_cycles_reciprocal_edges_and_self_loop_stay_visible_outside_node_interiors():
    pairs = (("a", "b"), ("b", "c"), ("c", "a"), ("b", "a"), ("a", "a"))
    result = graph_result([[1] * len(pairs)], pairs, mask=(1,) * len(pairs))
    _, axis, renderer, ellipses, arrows = rendered_graph(result)
    assert len(arrows) == len(pairs)
    viewport = axis.get_window_extent(renderer)
    node_bounds = [ellipse.get_window_extent(renderer) for ellipse in ellipses]
    named_bounds = {
        text.get_text(): next(
            node
            for node in node_bounds
            if np.allclose(
                [(node.x0 + node.x1) / 2, (node.y0 + node.y1) / 2],
                axis.transData.transform(text.get_position()),
            )
        )
        for text in axis.texts
    }
    connectors = {}
    for arrow in arrows:
        path = arrow.get_path().transformed(arrow.get_transform())
        source, target = pairs[int(arrow.get_gid().removeprefix("edge-"))]
        segments = list(path.iter_bezier())
        head_start = next(index for index, (_, code) in enumerate(segments) if index and code == 1)
        connector = np.concatenate(
            [segment(np.linspace(0, 1, 31)) for segment, _ in segments[:head_start]]
        )
        connectors[source, target] = connector
        source_bounds, target_bounds = named_bounds[source], named_bounds[target]

        def radial_distance(points, node):
            x = (points[..., 0] - (node.x0 + node.x1) / 2) / (node.width / 2)
            y = (points[..., 1] - (node.y0 + node.y1) / 2) / (node.height / 2)
            return np.sqrt(x * x + y * y)

        assert radial_distance(connector[0], source_bounds) == pytest.approx(1, abs=1e-5)
        head = np.concatenate(
            [segment(np.linspace(0, 1, 31)) for segment, _ in segments[head_start:]]
        )
        stroke_pixels = arrow.get_linewidth() * axis.figure.dpi / 72
        target_error = abs(np.min(radial_distance(head, target_bounds)) - 1)
        assert (
            target_error * min(target_bounds.width, target_bounds.height) / 2 <= 2 * stroke_pixels
        )
        bounds = path.get_extents()
        assert bounds.x0 >= viewport.x0 and bounds.x1 <= viewport.x1
        assert bounds.y0 >= viewport.y0 and bounds.y1 <= viewport.y1
        for segment, _ in path.iter_bezier():
            points = segment(np.linspace(0, 1, 31))
            for node in node_bounds:
                x = (points[:, 0] - (node.x0 + node.x1) / 2) / (node.width / 2)
                y = (points[:, 1] - (node.y0 + node.y1) / 2) / (node.height / 2)
                assert np.min(x * x + y * y) >= 1 - 1e-5
    forward, reverse = connectors["a", "b"], connectors["b", "a"]
    distances = np.linalg.norm(forward[:, None, :] - reverse[None, :, :], axis=-1)
    assert np.median(np.min(distances, axis=1)) > 2 * stroke_pixels


def test_tall_diagram_keeps_heading_above_graph_content():
    pairs = tuple((f"n{index:02}", f"n{index + 1:02}") for index in range(79))
    result = graph_result([[1] * len(pairs)], pairs, mask=(1,) * len(pairs))
    figure, axis, renderer, ellipses, _ = rendered_graph(result, graph=None)
    heading = figure._suptitle.get_window_extent(renderer)
    assert heading.y0 > axis.get_window_extent(renderer).y1
    assert heading.y0 > axis.title.get_window_extent(renderer).y1
    assert all(heading.y0 > ellipse.get_window_extent(renderer).y1 for ellipse in ellipses)
