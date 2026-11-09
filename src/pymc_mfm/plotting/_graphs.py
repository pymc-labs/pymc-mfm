"""Shared complete-graph selection and observed-support layout for graph views."""

from collections import deque
from numbers import Integral


def ranked_graphs(summary):
    """All complete graphs, ranked by descending count then ascending binary vector."""
    return sorted(
        summary["arrays"]["graphs"]["complete_graphs"],
        key=lambda item: (-item["count"], item["graph"]),
    )


def select_graph(summary, graph=None):
    """Select a complete graph by its zero-based frequency rank, or the Average."""
    if graph is None:
        return None
    if isinstance(graph, bool) or not isinstance(graph, Integral):
        raise ValueError("graph must be None or a zero-based integer rank")
    records = ranked_graphs(summary)
    if graph < 0 or graph >= len(records):
        raise ValueError(
            f"graph rank {graph} is out of range for {len(records)} complete graphs (zero-based)"
        )
    selected = records[graph]
    if len(selected["graph"]) != len(summary["arrays"]["graphs"]["edges"]):
        raise ValueError("Complete graph vector must align with the active edge records")
    return selected


def _graph_nodes(edges):
    """Validate active endpoints and return unique names in first-appearance order."""
    if not edges:
        raise ValueError("DAG view requires aligned edge endpoints")
    nodes = {}
    for entry in edges:
        endpoints = entry.get("endpoints")
        if (
            not isinstance(endpoints, (list, tuple))
            or len(endpoints) != 2
            or not all(isinstance(node, str) and node for node in endpoints)
        ):
            raise ValueError("DAG view requires aligned edge endpoints")
        for node in endpoints:
            nodes[node] = None
    return nodes


def graph_positions(edges):
    """Top-to-bottom longest-path ranks of the observed-support SCC condensation.

    All active endpoint names remain, even when their edge frequency is zero.
    Sorting names makes row order independent of edge order or selected graph.
    Both SCC traversals are iterative so long paths do not consume the call stack.
    """
    nodes = sorted(_graph_nodes(edges))
    outgoing = {node: [] for node in nodes}
    incoming = {node: [] for node in nodes}
    for entry in edges:
        if entry["probability"] > 0:
            source, target = entry["endpoints"]
            outgoing[source].append(target)
            incoming[target].append(source)

    visited = set()
    finished = []
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(outgoing[start]))]
        while stack:
            node, neighbors = stack[-1]
            neighbor = next(neighbors, None)
            if neighbor is None:
                finished.append(node)
                stack.pop()
            elif neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, iter(outgoing[neighbor])))

    components = {}
    component_count = 0
    for start in reversed(finished):
        if start in components:
            continue
        components[start] = component_count
        pending = [start]
        while pending:
            node = pending.pop()
            for neighbor in incoming[node]:
                if neighbor not in components:
                    components[neighbor] = component_count
                    pending.append(neighbor)
        component_count += 1

    successors = [[] for _ in range(component_count)]
    indegree = [0] * component_count
    for source in nodes:
        source_component = components[source]
        for target in outgoing[source]:
            target_component = components[target]
            if source_component != target_component:
                successors[source_component].append(target_component)
                indegree[target_component] += 1

    ranks = [0] * component_count
    ready = deque(index for index, degree in enumerate(indegree) if degree == 0)
    while ready:
        source = ready.popleft()
        for target in successors[source]:
            ranks[target] = max(ranks[target], ranks[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    row_counts = {}
    for node in nodes:
        rank = ranks[components[node]]
        row_counts[rank] = row_counts.get(rank, 0) + 1
    row_indices = {}
    positions = {}
    for node in nodes:
        rank = ranks[components[node]]
        count = row_counts[rank]
        index = row_indices.get(rank, 0)
        fraction = 0.5 if count == 1 else 0.1 + 0.8 * index / (count - 1)
        positions[node] = (float(rank), fraction)
        row_indices[rank] = index + 1
    return positions
