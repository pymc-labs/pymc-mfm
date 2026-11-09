"""Self-contained, offline inspection of empirical complete-graph distributions."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from ._graphs import graph_positions, ranked_graphs
from .dot import _DotFormatError, dot_template


class GraphInspector:
    """An interactive HTML document with isolated notebook display and offline saving.

    The document contains active graph topology, complete-graph counts/frequencies,
    and the display metadata needed to interpret them. It does not contain observed
    data, contribution arrays, or full result provenance. Review node labels before
    sharing an exported document.
    """

    def __init__(self, html: str):
        self.html = html

    def _repr_html_(self) -> str:
        """Display in an independent frame without access to the notebook's origin."""
        return (
            '<iframe title="Graph distribution inspector" '
            'sandbox="allow-scripts allow-downloads" '
            'referrerpolicy="no-referrer" '
            'style="width:100%;height:950px;border:0" '
            f'srcdoc="{escape(self.html, quote=True)}"></iframe>'
        )

    def save(self, path: str | Path) -> Path:
        """Write the standalone HTML document as UTF-8 and return its path."""
        path = Path(path)
        path.write_text(self.html, encoding="utf-8")
        return path


def plot_graph_inspector(summary) -> GraphInspector:
    """Inspect Average or any zero-based complete-graph rank without a server.

    Ranks use descending draw count, breaking ties by ascending binary vector.
    Average uses the existing marginal edge frequencies; a ranked view uses the
    exact complete binary vector over active slots. Supplied graph truth is not
    overlaid: the inspector explicitly identifies its prediction-only view.
    """
    graph = summary["arrays"]["graphs"]
    edges = graph["edges"]
    positions = graph_positions(edges)
    records = ranked_graphs(summary)
    if any(len(record["graph"]) != len(edges) for record in records):
        raise ValueError("Complete graph vectors must align with active edge records")
    try:
        dot = dot_template(summary)
        dot_error = None
    except _DotFormatError as error:
        dot = None
        dot_error = str(error)
    metadata = summary["metadata"]
    payload = {
        "metadata": {
            "head": str(metadata.get("head", "conditional")),
            "window": str(metadata["window"]),
            "mode": str(metadata["mode"]),
            "units": str(metadata["units"]),
            "n_samples": int(metadata["n_samples"]),
        },
        "nodes": [
            {"name": name, "position": [float(x), float(y)]} for name, (x, y) in positions.items()
        ],
        "edges": [
            {
                "source": entry["endpoints"][0],
                "target": entry["endpoints"][1],
                "value": float(entry["probability"]),
                "index": int(entry["index"]),
                "label": f"{float(entry['probability']):.3g}",
                "exact": repr(float(entry["probability"])),
            }
            for entry in edges
        ],
        "graphs": [
            {
                "rank": rank,
                "graph": [int(value) for value in record["graph"]],
                "count": int(record["count"]),
                "frequency": float(record["frequency"]),
            }
            for rank, record in enumerate(records)
        ],
        "has_graph_truth": graph.get("truth") is not None
        or any(entry.get("truth") is not None for entry in edges),
        "dot": dot,
        "dot_error": dot_error,
    }
    # JSON is data, not executable source. Escape HTML/script delimiters even
    # inside strings, including labels such as </script> and Unicode separators.
    data = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    for character, replacement in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        data = data.replace(character, replacement)
    return GraphInspector(_DOCUMENT.replace("__GRAPH_DATA__", data))


_DOCUMENT = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline';
               base-uri 'none'; form-action 'none'">
<title>Graph distribution inspector</title>
<style>
:root { color-scheme: light; font: 16px/1.5 system-ui, sans-serif; color: #172635; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f7fa; }
main { max-width: 100rem; margin: 0 auto; padding: clamp(.75rem, 3vw, 2rem); }
h1, h2 { line-height: 1.25; }
h1 { font-size: 1.6rem; margin: 0 0 .5rem; }
h2 { font-size: 1.2rem; margin-top: 0; }
p { overflow-wrap: anywhere; }
.card { background: white; border: 1px solid #c4d0dc; border-radius: .6rem;
        margin: 1rem 0; padding: clamp(.75rem, 2vw, 1.25rem); min-width: 0; }
.metadata { display: flex; flex-wrap: wrap; gap: .4rem 1.5rem; margin: .75rem 0; }
.metadata div { min-width: 0; overflow-wrap: anywhere; }
.metadata dt { display: inline; font-weight: 650; }
.metadata dd { display: inline; margin: 0 0 0 .35rem; }
.note { color: #3b5064; font-size: .93rem; }
.qualification { border-left: .3rem solid #0072b2; padding-left: .8rem; }
.controls { display: flex; align-items: end; flex-wrap: wrap; gap: .75rem; }
.selector { flex: 1 1 22rem; min-width: 0; }
.selector label { display: block; font-weight: 650; margin-bottom: .3rem; }
button, select { font: inherit; color: inherit; background: white; border: 1px solid #62798d;
                 border-radius: .3rem; min-height: 2.75rem; padding: .45rem .7rem; }
select { width: 100%; max-width: 100%; }
button { cursor: pointer; }
button:hover:not(:disabled) { background: #edf6fc; }
button:disabled { color: #687783; cursor: default; border-color: #bbc6cf; }
button[aria-pressed="true"] { background: #dceffc; border-color: #005b91; font-weight: 700; }
button:focus-visible, select:focus-visible, [tabindex]:focus-visible {
    outline: 3px solid #b44b00; outline-offset: 3px;
}
#selected-statistics { font-variant-numeric: tabular-nums; }
#zero-edge { background: #f0f6fa; padding: .6rem; border-radius: .25rem; }
.diagram-scroll { overflow: auto; max-height: 65vh; min-height: 20rem;
                  border: 1px solid #b8c8d6; border-radius: .3rem; background: white;
                  overscroll-behavior: contain; scrollbar-gutter: stable; }
#graph-svg { display: block; max-width: none; }
.distribution-scroll { overflow: auto; max-height: 32rem; }
table { border-collapse: collapse; width: 100%; min-width: 29rem;
        font-variant-numeric: tabular-nums; }
caption { text-align: left; padding-bottom: .65rem; }
th, td { text-align: left; padding: .5rem .65rem; border-bottom: 1px solid #d9e2ea; }
th { background: #f0f5f9; position: sticky; top: 0; z-index: 1; }
tr[aria-selected="true"] { background: #edf6fc; }
td button { width: 100%; text-align: left; white-space: nowrap; }
.frequency-track { display: block; height: .35rem; min-width: 6rem; background: #e3ebf1;
                   border-radius: .2rem; margin-top: .2rem; overflow: hidden; }
.frequency-bar { display: block; height: 100%; background: #0072b2; }
.pagination { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem; margin-top: 1rem; }
.pagination p { flex: 1 1 15rem; margin: 0; }
footer { margin-top: 1rem; }
[hidden] { display: none !important; }
</style>
</head>
<body>
<main>
<header>
<h1>Graph distribution inspector</h1>
<p>Inspect the marginal Average or any complete graph, ranked from zero by draw count.</p>
<dl id="metadata" class="metadata" aria-label="Result metadata"></dl>
<p id="mode-qualification" class="note qualification"></p>
<p id="truth-note" class="note qualification" hidden></p>
</header>
<noscript><p>
This offline inspector needs JavaScript enabled for graph selection and SVG/DOT downloads.
</p></noscript>
<section class="card" aria-labelledby="view-heading">
<h2 id="view-heading">Prediction graph</h2>
<div class="controls">
<div class="selector">
<label for="graph-select">Graph view (zero-based ranks)</label>
<select id="graph-select" aria-describedby="rank-note"></select>
</div>
<button id="average-view" type="button" aria-pressed="true">Average</button>
<button id="download-svg" type="button"
        aria-describedby="svg-unavailable">Download current SVG</button>
<button id="download-dot" type="button"
        aria-describedby="dot-unavailable">Download current DOT</button>
</div>
<p id="rank-note" class="note">
Rank 0 is the most frequent complete graph; ties use ascending binary vectors.
No graphs are omitted.
</p>
<p id="dot-unavailable" class="note" hidden></p>
<p id="svg-unavailable" class="note" role="status" hidden></p>
<div id="selection-summary" role="status" aria-live="polite" aria-atomic="true"
     data-graph-rank="average">
<h2 id="selected-title">Average</h2>
<p id="selected-statistics"></p>
<p id="edge-semantics" class="note"></p>
</div>
<p id="zero-edge" hidden>No edges in this view. All active endpoint nodes remain visible.</p>
<div class="diagram-scroll" tabindex="0" role="region" aria-label="Scrollable named graph diagram">
<svg id="graph-svg" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-labelledby="svg-title svg-description" data-graph-rank="average" style="background:white">
<title id="svg-title">Average graph</title>
<desc id="svg-description"></desc>
<defs>
<marker id="graph-arrowhead" viewBox="0 0 10 10" refX="10" refY="5"
        markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
<path d="M 0 0 L 10 5 L 0 10 z" fill="black"/>
</marker>
</defs>
<g id="graph-edges"></g>
<g id="graph-nodes"></g>
</svg>
</div>
<p class="note">
Scroll the diagram to inspect dense graphs or long labels.
Node and arrow tooltips retain their full names.
</p>
</section>
<section class="card" aria-labelledby="distribution-heading">
<h2 id="distribution-heading">Complete-graph distribution</h2>
<div class="distribution-scroll" tabindex="0" role="region"
     aria-label="Scrollable graph distribution">
<table aria-describedby="rank-note page-status">
<caption>
Choose a graph by its global zero-based rank.
Bars show observed draw frequencies, not fit scores.
</caption>
<thead><tr>
<th scope="col">Graph rank</th><th scope="col">Draw count</th>
<th scope="col">Observed frequency</th><th scope="col">Edges</th>
</tr></thead>
<tbody id="distribution-rows"></tbody>
</table>
</div>
<nav class="pagination" aria-label="Graph distribution pages">
<button id="previous-page" type="button">Previous</button>
<p id="page-status" role="status" aria-live="polite"></p>
<button id="next-page" type="button">Next</button>
</nav>
</section>
<footer class="note">
<p>
HTML exports contain all active node labels, graph topology, draw counts/frequencies,
and the display metadata above, but no observations, contribution arrays, or full provenance.
Review labels before sharing. SVG and DOT downloads contain the current prediction view only.
</p>
</footer>
</main>
<script id="graph-data" type="application/json">__GRAPH_DATA__</script>
<script>
"use strict";
(() => {
    const data = JSON.parse(document.getElementById("graph-data").textContent);
    const byId = id => document.getElementById(id);
    const select = byId("graph-select");
    const rows = byId("distribution-rows");
    const distributionScroll = rows.closest(".distribution-scroll");
    const svg = byId("graph-svg");
    const edgeLayer = byId("graph-edges");
    const nodeLayer = byId("graph-nodes");
    const previous = byId("previous-page");
    const next = byId("next-page");
    const pageSize = 50;
    const pageCount = Math.max(1, Math.ceil(data.graphs.length / pageSize));
    const marginalValues = data.edges.map(edge => edge.value);
    const edgeCounts = data.graphs.map(record => record.graph.reduce((sum, bit) => sum + bit, 0));
    const percentFormat = new Intl.NumberFormat(undefined, {
        style: "percent", maximumSignificantDigits: 6
    });
    const percent = value => percentFormat.format(value);
    const count = value => value.toLocaleString();
    let selection = "average";
    let page = 0;

    function textElement(tag, text, parent) {
        const element = document.createElement(tag);
        element.textContent = text;
        if (parent) parent.append(element);
        return element;
    }

    function svgElement(tag, attributes = {}) {
        const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
        for (const [name, value] of Object.entries(attributes)) {
            element.setAttribute(name, String(value));
        }
        return element;
    }

    const metadata = data.metadata;
    for (const [label, value] of [
        ["Head", metadata.head], ["Window", metadata.window], ["Mode", metadata.mode],
        ["Units", metadata.units], ["Draws", count(metadata.n_samples)]
    ]) {
        const item = document.createElement("div");
        textElement("dt", label, item);
        textElement("dd", value, item);
        byId("metadata").append(item);
    }
    const qualifications = {
        model_average: "Frequencies describe the empirical distribution of " +
            "sampled complete graphs. Average shows marginal edge frequencies, " +
            "may combine edges never sampled together, and may contain cycles. " +
            "Frequencies are not fit scores or evidence of causal validity.",
        map: "MAP mode shows the selected empirical modal hypothesis only. " +
            "Counts and frequencies describe the selected draws, not pre-selection " +
            "router mass or posterior probabilities. They are not fit scores or " +
            "evidence of causal validity.",
        fixed: "Fixed mode shows the supplied hypothesis only. Counts and frequencies describe " +
            "the supplied draws, not router mass or posterior probabilities. They are not fit " +
            "scores or evidence of causal validity."
    };
    byId("mode-qualification").textContent = qualifications[metadata.mode];
    if (data.has_graph_truth) {
        byId("truth-note").hidden = false;
        byId("truth-note").textContent = "Prediction-only view: graph truth was supplied, but is " +
            "not overlaid or used to change the selected graph.";
    }
    if (data.dot_error !== null) {
        byId("download-dot").disabled = true;
        byId("dot-unavailable").hidden = false;
        byId("dot-unavailable").textContent = "DOT download unavailable: " + data.dot_error;
    }

    const averageOption = textElement("option", "Average — marginal edge frequencies", select);
    averageOption.value = "average";
    for (const record of data.graphs) {
        const option = textElement("option", "Graph " + record.rank + " · " +
            count(record.count) + "/" + count(metadata.n_samples) + " draws · " +
            percent(record.frequency), select);
        option.value = String(record.rank);
    }

    // Measure literal, multiline serif text before packing rows. Ovals contain
    // the padded text rectangle's corners; the scroll region never shrinks it.
    const nodeViews = new Map();
    const ranks = new Map();
    const probe = svgElement("text", {"font-size": 16, "font-family": "serif"});
    probe.textContent = "Mg";
    nodeLayer.append(probe);
    const fontBox = probe.getBBox();
    const lineHeight = Math.max(20, fontBox.height * 1.25);
    const baseline = -(fontBox.y + fontBox.height / 2);
    probe.remove();
    for (const node of data.nodes) {
        const group = svgElement("g", {"data-node-name": node.name});
        const title = svgElement("title");
        title.textContent = node.name;
        const oval = svgElement("ellipse", {
            fill: "white", stroke: "black", "stroke-width": 1
        });
        const label = svgElement("text", {
            "text-anchor": "middle", "font-size": 16, "font-family": "serif",
            fill: "black", style: "white-space:pre"
        });
        const lines = node.name.split("\n");
        lines.forEach((line, index) => {
            const span = svgElement("tspan", {
                x: 0, y: (index - (lines.length - 1) / 2) * lineHeight + baseline
            });
            span.textContent = line;
            label.append(span);
        });
        group.append(title, oval, label);
        nodeLayer.append(group);
        const box = label.getBBox();
        const textWidth = 2 * Math.max(Math.abs(box.x), Math.abs(box.x + box.width));
        const textHeight = Math.max(
            lines.length * lineHeight, 2 * Math.abs(box.y), 2 * Math.abs(box.y + box.height)
        );
        const rx = Math.max(30, Math.SQRT2 * (textWidth / 2 + 10));
        const ry = Math.max(20, Math.SQRT2 * (textHeight / 2 + 5));
        oval.setAttribute("rx", rx);
        oval.setAttribute("ry", ry);
        nodeViews.set(node.name, {group, rx, ry});
        const rank = node.position[0];
        if (!ranks.has(rank)) ranks.set(rank, []);
        ranks.get(rank).push(node);
    }
    const rowNodes = [...ranks.entries()].sort((a, b) => a[0] - b[0]).map(([, nodes]) =>
        nodes.sort((a, b) => a.position[1] - b.position[1]));
    const rowFor = new Map();
    const rowWidths = [];
    const rowHeights = [];
    const positions = new Map();
    rowNodes.forEach((nodes, row) => {
        const width = nodes.reduce((sum, node) => sum + 2 * nodeViews.get(node.name).rx, 0) +
            48 * (nodes.length - 1);
        rowWidths.push(width);
        rowHeights.push(Math.max(...nodes.map(node => 2 * nodeViews.get(node.name).ry)));
        let x = -width / 2;
        for (const node of nodes) {
            const view = nodeViews.get(node.name);
            positions.set(node.name, [x + view.rx, 0]);
            rowFor.set(node.name, row);
            x += 2 * view.rx + 48;
        }
    });
    const labelSizes = data.edges.map(edge => {
        const label = svgElement("text", {"font-size": 12, "font-family": "serif"});
        label.textContent = edge.label;
        edgeLayer.append(label);
        const box = label.getBBox();
        label.remove();
        return [box.width, Math.max(14, box.height)];
    });

    // All active slots reserve stable ports/lanes, independently of selection.
    // Every segment stays in its endpoint column, a row gutter, or an outer
    // rail; unlike a generic bent connector it cannot cut another oval.
    const ports = new Map();
    function portEntries(name, side) {
        if (!ports.has(name)) ports.set(name, {top: [], bottom: []});
        return ports.get(name)[side];
    }
    const routes = data.edges.map(edge => {
        const sourceRow = rowFor.get(edge.source);
        const targetRow = rowFor.get(edge.target);
        let kind, sourceSide, targetSide;
        if (sourceRow === targetRow) {
            kind = "row";
            sourceSide = targetSide =
                positions.get(edge.source)[0] <= positions.get(edge.target)[0] ? "top" : "bottom";
        } else {
            const down = sourceRow < targetRow;
            sourceSide = down ? "bottom" : "top";
            targetSide = down ? "top" : "bottom";
            kind = targetRow === sourceRow + 1 ? "adjacent" : "outer";
        }
        const route = {source: edge.source, target: edge.target, sourceRow, targetRow,
            sourceSide, targetSide, kind};
        portEntries(edge.source, sourceSide).push([route, "source"]);
        portEntries(edge.target, targetSide).push([route, "target"]);
        return route;
    });
    for (const [name, sides] of ports) {
        for (const entries of Object.values(sides)) {
            const hasLoops = entries.some(([route]) => route.source === route.target);
            entries.forEach(([route, end], index) => {
                const step = index / Math.max(1, entries.length - 1);
                let fraction;
                if (route.source === route.target) {
                    fraction = (end === "source" ? -0.8 : 0.6) + 0.2 * step;
                } else {
                    const span = hasLoops ? 0.4 : 0.65;
                    fraction = entries.length === 1 ? 0 : span * (2 * step - 1);
                }
                route[end + "Offset"] = nodeViews.get(name).rx * fraction;
            });
        }
    }
    const laneCounts = rowNodes.map(() => ({top: 0, bottom: 0}));
    const outerCounts = {left: 0, right: 0};
    for (const route of routes) {
        if (route.kind === "adjacent" &&
            positions.get(route.source)[0] === positions.get(route.target)[0] &&
            portEntries(route.source, "bottom").length === 1 &&
            portEntries(route.target, "top").length === 1) {
            route.kind = "direct";
            continue;
        }
        for (const end of route.kind === "outer" ? ["source", "target"] : ["source"]) {
            const counts = laneCounts[route[end + "Row"]];
            const side = route[end + "Side"];
            route[end + "Lane"] = counts[side]++;
        }
        if (route.kind === "outer") {
            route.outerSide = outerCounts.left <= outerCounts.right ? "left" : "right";
            route.outerLane = outerCounts[route.outerSide]++;
        }
    }
    const laneGap = Math.max(28, ...labelSizes.map(([, height]) => height + 16));
    const rowCenters = [];
    let cursor = 0;
    rowNodes.forEach((nodes, row) => {
        cursor += laneCounts[row].top * laneGap + laneGap + rowHeights[row] / 2;
        rowCenters.push(cursor);
        for (const node of nodes) positions.get(node.name)[1] = cursor;
        cursor += rowHeights[row] / 2 + laneGap + laneCounts[row].bottom * laneGap;
    });
    function endpoint(route, end) {
        const name = route[end];
        const [x, y] = positions.get(name);
        const {rx, ry} = nodeViews.get(name);
        const offset = route[end + "Offset"];
        const sign = route[end + "Side"] === "top" ? -1 : 1;
        return [x + offset, y + sign * ry * Math.sqrt(1 - (offset / rx) ** 2)];
    }
    function rail(route, end) {
        const row = route[end + "Row"];
        const sign = route[end + "Side"] === "top" ? -1 : 1;
        return rowCenters[row] + sign *
            (rowHeights[row] / 2 + laneGap * (route[end + "Lane"] + 1));
    }
    const left = -Math.max(...rowWidths) / 2;
    const right = Math.max(...rowWidths) / 2;
    routes.forEach((route, slot) => {
        const start = endpoint(route, "source");
        const end = endpoint(route, "target");
        if (route.kind === "direct") {
            route.points = [start, end];
            route.label = [start[0] + 9 + labelSizes[slot][0] / 2, (start[1] + end[1]) / 2];
        } else {
            const sourceRail = rail(route, "source");
            if (route.kind === "outer") {
                const sign = route.outerSide === "left" ? -1 : 1;
                const outerX = (sign < 0 ? left : right) + sign *
                    (42 + route.outerLane * laneGap);
                const targetRail = rail(route, "target");
                route.points = [start, [start[0], sourceRail], [outerX, sourceRail],
                    [outerX, targetRail], [end[0], targetRail], end];
                route.label = [(start[0] + outerX) / 2, sourceRail - 9];
            } else {
                route.points = [start, [start[0], sourceRail], [end[0], sourceRail], end];
                route.label = [(start[0] + end[0]) / 2, sourceRail - 9];
            }
        }
    });
    let minimumX = Infinity, minimumY = Infinity, maximumX = -Infinity, maximumY = -Infinity;
    function include(x1, y1, x2 = x1, y2 = y1) {
        minimumX = Math.min(minimumX, x1);
        minimumY = Math.min(minimumY, y1);
        maximumX = Math.max(maximumX, x2);
        maximumY = Math.max(maximumY, y2);
    }
    for (const [name, [x, y]] of positions) {
        const {rx, ry} = nodeViews.get(name);
        include(x - rx, y - ry, x + rx, y + ry);
    }
    routes.forEach((route, slot) => {
        for (const [x, y] of route.points) include(x, y);
        const [x, y] = route.label;
        const [width, height] = labelSizes[slot];
        include(x - width / 2, y - height / 2, x + width / 2, y + height / 2);
    });
    const canvasWidth = Math.max(480, Math.ceil(maximumX - minimumX + 64));
    const canvasHeight = Math.max(320, Math.ceil(maximumY - minimumY + 64));
    const offsetX = (canvasWidth - (maximumX - minimumX)) / 2;
    const offsetY = (canvasHeight - (maximumY - minimumY)) / 2;
    const translate = ([x, y]) => [x - minimumX + offsetX, y - minimumY + offsetY];
    for (const [name, point] of positions) {
        const [x, y] = translate(point);
        nodeViews.get(name).group.setAttribute("transform", "translate(" + x + " " + y + ")");
    }
    for (const route of routes) {
        route.path = route.points.map((point, index) =>
            (index ? "L " : "M ") + translate(point).join(" ")).join(" ");
        route.label = translate(route.label);
    }
    svg.setAttribute("width", canvasWidth);
    svg.setAttribute("height", canvasHeight);
    svg.setAttribute("viewBox", "0 0 " + canvasWidth + " " + canvasHeight);

    function renderGraph() {
        const average = selection === "average";
        const record = average ? null : data.graphs[Number(selection)];
        const values = average ? marginalValues : record.graph;
        const edgeCount = average ?
            values.filter(value => value > 0).length : edgeCounts[record.rank];
        const title = average ? "Average — marginal edge frequencies" : "Graph rank " + record.rank;
        const statistics = average ? count(metadata.n_samples) + " draws across " +
            count(data.graphs.length) + " distinct complete graphs · " + count(edgeCount) +
            " nonzero-frequency edges" : count(record.count) + " / " + count(metadata.n_samples) +
            " draws · observed frequency " + percent(record.frequency) + " · " + count(edgeCount) +
            " selected edges";
        byId("selected-title").textContent = title;
        byId("selected-statistics").textContent = statistics;
        byId("edge-semantics").textContent = average ?
            "Numeric labels show marginal edge frequencies; tooltips retain exact values. " +
            "Average is not necessarily a sampled complete graph." :
            "Exact complete graph: plain arrows are binary edge indicators, not marginal " +
            "frequencies. Absent edges are not drawn.";
        const summary = byId("selection-summary");
        summary.dataset.graphRank = selection;
        summary.dataset.count = String(average ? metadata.n_samples : record.count);
        summary.dataset.edgeCount = String(edgeCount);
        if (average) delete summary.dataset.frequency;
        else summary.dataset.frequency = String(record.frequency);
        byId("zero-edge").hidden = edgeCount !== 0;
        byId("average-view").setAttribute("aria-pressed", String(average));
        byId("svg-title").textContent = title + " — prediction view";
        byId("svg-description").textContent = statistics + ". " + metadata.head + " head, " +
            metadata.window + " window, " + metadata.mode + " mode, units: " + metadata.units +
            ". " + count(data.nodes.length) + " named active endpoint nodes. " +
            (data.has_graph_truth ? "Prediction only; supplied graph truth is not shown." : "");
        svg.dataset.graphRank = selection;
        svg.dataset.edgeCount = String(edgeCount);
        edgeLayer.replaceChildren();
        const frequencyLabels = window.document.createDocumentFragment();
        data.edges.forEach((edge, slot) => {
            const value = values[slot];
            if (!value) return;
            const arrow = svgElement("path", {
                d: routes[slot].path, fill: "none", stroke: "black",
                "stroke-width": 1, opacity: 1, "stroke-linejoin": "round",
                "stroke-linecap": "round", id: "edge-" + edge.index,
                "marker-end": "url(#graph-arrowhead)", "data-source": edge.source,
                "data-target": edge.target, "data-edge-index": edge.index,
                "data-active-index": slot, "data-value": value, "data-graph-rank": selection
            });
            const label = edge.source + " → " + edge.target + " · " +
                (average ? "marginal frequency " + edge.exact : "edge indicator 1");
            arrow.setAttribute("aria-label", label);
            const tooltip = svgElement("title");
            tooltip.textContent = label;
            arrow.append(tooltip);
            edgeLayer.append(arrow);
            if (average) {
                const [x, y] = routes[slot].label;
                const frequency = svgElement("text", {
                    x, y, "text-anchor": "middle", "dominant-baseline": "central",
                    "font-size": 12, "font-family": "serif", fill: "black",
                    "paint-order": "stroke", stroke: "white", "stroke-width": 4,
                    "stroke-linejoin": "round", "data-edge-index": edge.index,
                    "data-value": value
                });
                frequency.textContent = edge.label;
                const exact = svgElement("title");
                exact.textContent = label;
                frequency.append(exact);
                frequencyLabels.append(frequency);
            }
        });
        edgeLayer.append(frequencyLabels);
    }

    function updateRowSelection() {
        for (const row of rows.children) {
            const selected = row.dataset.rank === selection;
            row.setAttribute("aria-selected", String(selected));
            row.querySelector("button").setAttribute("aria-pressed", String(selected));
        }
    }

    function renderRows() {
        rows.replaceChildren();
        const first = page * pageSize;
        const last = Math.min(first + pageSize, data.graphs.length);
        for (let rank = first; rank < last; rank++) {
            const record = data.graphs[rank];
            const row = document.createElement("tr");
            row.dataset.rank = String(rank);
            row.dataset.count = String(record.count);
            row.dataset.frequency = String(record.frequency);
            const rankCell = document.createElement("td");
            const button = textElement("button", "Graph " + rank, rankCell);
            button.type = "button";
            button.dataset.rank = String(rank);
            button.setAttribute("aria-label", "Select graph rank " + rank + ", " +
                count(record.count) + " of " + count(metadata.n_samples) +
                " draws, observed frequency " + percent(record.frequency));
            button.addEventListener("click", () => selectGraph(String(rank)));
            row.append(rankCell);
            textElement("td", count(record.count), row);
            const frequencyCell = textElement("td", percent(record.frequency), row);
            const track = textElement("span", "", frequencyCell);
            track.className = "frequency-track";
            track.setAttribute("aria-hidden", "true");
            const bar = textElement("span", "", track);
            bar.className = "frequency-bar";
            bar.style.width = (record.frequency * 100) + "%";
            textElement("td", count(edgeCounts[rank]), row);
            rows.append(row);
        }
        previous.disabled = page === 0;
        next.disabled = page === pageCount - 1;
        byId("page-status").textContent = data.graphs.length ?
            "Ranks " + first + "–" + (last - 1) + " of " + count(data.graphs.length) +
            " graphs · Page " + (page + 1) + " of " + pageCount : "No complete graphs";
        updateRowSelection();
    }

    function selectGraph(value) {
        selection = value;
        select.value = value;
        const selectedPage = value === "average" ? page : Math.floor(Number(value) / pageSize);
        if (selectedPage !== page) {
            page = selectedPage;
            renderRows();
            distributionScroll.scrollTop = 0;
        } else {
            updateRowSelection();
        }
        renderGraph();
    }

    function changePage(direction) {
        const focused = document.activeElement;
        page = Math.max(0, Math.min(pageCount - 1, page + direction));
        renderRows();
        distributionScroll.scrollTop = 0;
        // A disabled pagination control cannot keep keyboard focus. Move it to
        // the enabled counterpart at a boundary, rather than dropping focus.
        if (focused === next && next.disabled && !previous.disabled) previous.focus();
        if (focused === previous && previous.disabled && !next.disabled) next.focus();
    }

    select.addEventListener("change", () => selectGraph(select.value));
    byId("average-view").addEventListener("click", () => selectGraph("average"));
    previous.addEventListener("click", () => changePage(-1));
    next.addEventListener("click", () => changePage(1));
    byId("download-svg").addEventListener("click", () => {
        const image = svg.cloneNode(true);
        image.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        const source = '<?xml version="1.0" encoding="UTF-8"?>\n' +
            new XMLSerializer().serializeToString(image);
        const invalid = source.match(
            /[^\t\n\r\u0020-\uD7FF\uE000-\uFFFD\u{10000}-\u{10FFFF}]/u);
        if (invalid) {
            const codepoint = invalid[0].codePointAt(0).toString(16).toUpperCase().padStart(4, "0");
            const note = byId("svg-unavailable");
            note.hidden = false;
            note.textContent = "SVG download unavailable: XML 1.0-incompatible text or metadata " +
                "(U+" + codepoint + "). Viewing and graph selection remain available.";
            byId("download-svg").disabled = true;
            return;
        }
        const url = URL.createObjectURL(
            new Blob([source], {type: "image/svg+xml;charset=utf-8"}));
        const link = window.document.createElement("a");
        link.href = url;
        link.download = selection === "average" ?
            "graph-average.svg" : "graph-rank-" + selection + ".svg";
        window.document.body.append(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    byId("download-dot").addEventListener("click", () => {
        const average = selection === "average";
        const values = average ? marginalValues : data.graphs[Number(selection)].graph;
        const fragments = average ? data.dot.average : data.dot.individual;
        const source = (average ? "// Average\n" : "// Complete graph rank " + selection + "\n") +
            data.dot.preamble + fragments.filter((_, slot) => values[slot] > 0).join("\n") +
            data.dot.suffix;
        const url = URL.createObjectURL(new Blob([source], {
            type: "text/vnd.graphviz;charset=utf-8", endings: "transparent"
        }));
        const link = window.document.createElement("a");
        link.href = url;
        link.download = average ? "graph-average.dot" : "graph-rank-" + selection + ".dot";
        window.document.body.append(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    renderRows();
    renderGraph();
})();
</script>
</body>
</html>
"""
