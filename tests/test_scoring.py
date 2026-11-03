import numpy as np
import pytest

from demo.scoring import fair_crps


def test_fair_crps_matches_pairwise_definition():
    draws = np.array([[0.0, 2.0], [1.0, 4.0], [3.0, 3.0]])
    truth = np.array([1.0, 3.0])
    n = len(draws)
    first = np.abs(draws - truth).mean(axis=0)
    pairs = np.abs(draws[:, None, :] - draws[None, :, :]).sum(axis=(0, 1))
    expected = np.mean(first - pairs / (2 * n * (n - 1)))
    assert fair_crps(draws, truth) == pytest.approx(expected)


def test_fair_crps_requires_valid_shape():
    with pytest.raises(ValueError):
        fair_crps([[1.0]], [1.0])
