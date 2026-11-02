"""Fixed synthetic world for comparing model assumptions, not a benchmark."""

from dataclasses import dataclass

import numpy as np

N_WEEKS = 208
N_CONTEXT = 166
SEED = 17
CHANNELS = ("TV", "Search")


@dataclass(frozen=True)
class ComparisonWorld:
    spend: np.ndarray
    promotion: np.ndarray
    outcome: np.ndarray
    baseline: np.ndarray
    channel_truth: np.ndarray
    promotion_truth: np.ndarray


def make_world():
    rng = np.random.default_rng(SEED)
    t = np.arange(N_WEEKS)
    promotion = (rng.random(N_WEEKS) < 0.27).astype(float)
    tv = rng.uniform(0.1, 1.0, N_WEEKS)
    search = np.clip(rng.uniform(0.05, 0.55, N_WEEKS) + 0.42 * promotion, 0, 1.0)
    spend = np.column_stack((tv, search))
    adstock = np.zeros_like(spend)
    for j, alpha in enumerate((0.4, 0.2)):
        mass = sum(alpha**i for i in range(8))
        for lag in range(8):
            adstock[lag:, j] += alpha**lag / mass * spend[: N_WEEKS - lag, j]
    channel_truth = 1000 * np.tanh(adstock * [4.0, 3.0] / 2) * [2.0, 1.2]
    baseline = 3200 + 300 * np.sin(2 * np.pi * t / 52) + 150 * np.cos(2 * np.pi * t / 52)
    promo_truth = 1200 * promotion
    outcome = baseline + channel_truth.sum(axis=1) + promo_truth + rng.normal(0, 180, N_WEEKS)
    return ComparisonWorld(spend * 1000, promotion, outcome, baseline, channel_truth, promo_truth)
