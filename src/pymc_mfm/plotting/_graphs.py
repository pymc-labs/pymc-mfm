"""Shared complete-graph selection and active endpoint validation for graph views."""

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
