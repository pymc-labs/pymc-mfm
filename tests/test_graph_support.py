"""Graph-support recovery metrics do not depend on the integer dtype of saved graphs."""

import numpy as np
import pytest

from pymc_mktfm.diagnostics import graph_support_diagnostics


def test_unsigned_graph_arrays_score_count_error_without_wraparound():
    # One two-slot family: draw counts 1, 2, 2 against a true count of 2, so the
    # count MAE is (1 + 0 + 0) / 3 and one draw in three misses the [2, 2] budget.
    graphs = np.array([[1, 0], [1, 1], [1, 1]], dtype=np.uint8)
    truth = np.array([1, 1], dtype=np.uint8)
    mask = np.array([1, 1], dtype=np.uint8)
    support = {"cardinality": {"cy": [2, 2]}, "forbidden_families": []}
    unsigned = graph_support_diagnostics(graphs, ["cy", "cy"], mask, truth=truth, support=support)
    assert unsigned["metrics"]["family/cy/count_mae"] == pytest.approx(1 / 3)
    assert unsigned["metrics"]["wrong_cy_cardinality_mass"] == pytest.approx(1 / 3)
    signed = graph_support_diagnostics(
        graphs.astype(np.int64),
        ["cy", "cy"],
        mask.astype(np.int64),
        truth=truth.astype(np.int64),
        support=support,
    )
    assert unsigned["metrics"] == signed["metrics"]
