"""Proper scoring of sampled scalar distributions against simulator truth."""

import numpy as np


def fair_crps(draws, truth):
    """Mean fair CRPS across time, with posterior draws on axis zero.

    The finite-ensemble correction excludes self-pairs, reducing bias when the
    models provide different numbers of draws. Scores are in target units.
    """
    draws = np.asarray(draws, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if draws.ndim != 2 or draws.shape[0] < 2 or truth.shape != (draws.shape[1],):
        raise ValueError("Expected at least two draws by time and one truth per time")
    n = draws.shape[0]
    ordered = np.sort(draws, axis=0)
    weights = (2 * np.arange(n) - n + 1)[:, None]
    first = np.mean(np.abs(draws - truth), axis=0)
    half_pairwise = np.sum(weights * ordered, axis=0) / (n * (n - 1))
    return float(np.mean(first - half_pairwise))
