"""Three deterministic worlds for evaluating held-out marketing forecasts."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Scenario = Literal["baseline", "promotion", "halo"]
CHANNELS = ("Search", "Social", "Video", "TV", "Audio", "Display")
N_CONTEXT = 166
N_WEEKS = 208


@dataclass(frozen=True)
class ScenarioWorld:
    spend: np.ndarray
    controls: np.ndarray
    outcome: np.ndarray
    expected: np.ndarray
    contributions: np.ndarray


def make_scenario(scenario: Scenario, seed: int = 21) -> ScenarioWorld:
    if scenario not in ("baseline", "promotion", "halo"):
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    t = np.arange(N_WEEKS)
    promotion = ((t % 13) < 3).astype(float)
    shared = rng.normal(0, 0.45, N_WEEKS)
    spend = np.exp(rng.normal(0, 0.55, (N_WEEKS, len(CHANNELS))) + shared[:, None])
    if scenario == "promotion":
        spend[:, 0] *= 1 + 1.4 * promotion
        spend[:, 1] *= 1 + 0.8 * promotion
    if scenario == "halo":
        spend[:, 0] += 0.7 * spend[:, 3]
    spend *= 100
    adstock = np.zeros_like(spend)
    for lag in range(8):
        adstock[lag:] += (0.4**lag) * spend[: N_WEEKS - lag]
    adstock /= sum(0.4**lag for lag in range(8))
    contributions = 1000 * np.tanh(adstock / 180) * np.array([1.3, 0.9, 0.7, 1.1, 0.5, 0.4])
    baseline = 4500 + 450 * np.sin(2 * np.pi * t / 52)
    if scenario == "promotion":
        baseline += 1300 * promotion
    if scenario == "halo":
        # A separate observed interaction makes the additive default misspecified.
        baseline += 800 * np.tanh(adstock[:, 0] / 180) * np.tanh(adstock[:, 3] / 180)
    expected = baseline + contributions.sum(axis=1)
    outcome = expected + rng.normal(0, 180, N_WEEKS)
    return ScenarioWorld(spend, promotion[:, None], outcome, expected, contributions)
