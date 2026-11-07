"""Shared complete-graph ranking and active-node layout for graph views."""


def ranked_graphs(summary):
    """All complete graphs, ranked by descending count then ascending binary vector."""
    return sorted(
        summary["arrays"]["graphs"]["complete_graphs"],
        key=lambda item: (-item["count"], item["graph"]),
    )


def graph_positions(edges):
    """Named nodes from every active endpoint, laid out by bank even without arrows."""
    if not edges:
        raise ValueError("DAG view requires aligned edge endpoints")
    nodes = set()
    for entry in edges:
        endpoints = entry.get("endpoints")
        if (
            not isinstance(endpoints, (list, tuple))
            or len(endpoints) != 2
            or not all(isinstance(node, str) and node for node in endpoints)
        ):
            raise ValueError("DAG view requires aligned edge endpoints")
        nodes.update(endpoints)
    banks = {}
    for node in sorted(nodes):
        bank = node.split(":", 1)[0]
        banks.setdefault(bank, []).append(node)
    order = sorted(
        banks,
        key=lambda bank: (
            {"latent": 0, "control": 1, "channel": 2, "outcome": 3, "sales": 4}.get(bank, 5),
            bank,
        ),
    )
    positions = {}
    for column, bank in enumerate(order):
        count = len(banks[bank])
        for row, node in enumerate(banks[bank]):
            y = 0.5 if count == 1 else 0.1 + 0.8 * row / (count - 1)
            positions[node] = (float(column), y)
    return positions
