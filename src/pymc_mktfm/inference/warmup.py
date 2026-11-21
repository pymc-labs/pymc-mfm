"""Warm up compiled sampling before the first real call."""

from __future__ import annotations

import contextlib
import time as _time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from pymc_mktfm.inference.api import decompose
from pymc_mktfm.model.model import MFM


@dataclass(frozen=True)
class WarmUp:
    """What a warm-up ran: ``calls`` decompose calls in ``seconds``, adding ``compiles``.

    ``compiles`` is ``None`` when the model is not compiled or PyTorch does not report
    compilations. A warm-up covers the inputs it ran; a later input of a new kind (for
    example a new channel count) can still compile on first use.
    """

    calls: int
    seconds: float
    compiles: int | None


def warm_up_observations(
    n_channels: int, *, n_controls: int, n_weeks: int = 208, context_weeks: int = 166, seed: int = 0
) -> dict[str, np.ndarray]:
    """A simple synthetic world for warm-up: ``n_channels`` channels and one live control.

    The values only need valid shapes and ranges; they carry no meaning. Padded controls
    beyond the first are zero and masked.
    """
    if n_channels < 1 or n_controls < 1 or not 0 < context_weeks < n_weeks:
        raise ValueError("need at least one channel and control, and 0 < context_weeks < n_weeks")
    rng = np.random.default_rng(seed)
    spend = rng.gamma(2.0, 1.0, size=(n_weeks, n_channels))
    controls = np.zeros((n_weeks, n_controls))
    controls[:, 0] = np.sin(np.arange(n_weeks) * 2 * np.pi / 52)
    outcome = 10.0 + spend[:context_weeks] @ rng.uniform(0.5, 1.5, n_channels)
    outcome = outcome + rng.normal(0.0, 0.5, context_weeks).clip(-5.0, 5.0)
    control_mask = np.zeros(n_controls, dtype=bool)
    control_mask[0] = True
    return {
        "spend": spend,
        "controls": controls,
        "context_outcome": outcome,
        "control_mask": control_mask,
    }


def warm_up_sampling(
    model: MFM,
    observations: Iterable[Mapping[str, object]] | None = None,
    *,
    channels: Iterable[int] = (),
    windows: Iterable[str] = ("query", "context"),
    modes: Iterable[str] = ("model_average",),
    n_samples: int = 1_000,
    seeds: Iterable[int] = (0, 1),
) -> WarmUp:
    """Run ``decompose`` on representative inputs so compilation happens now, not later.

    Each world in ``observations`` (``decompose`` observation mappings), plus one
    :func:`warm_up_observations` world per entry of ``channels``, runs once per window,
    mode and seed. Use worlds, draw counts and modes like the ones you will serve: which
    graphs the router samples also decides which variants compile, so several seeds
    cover more of them. The global random state is restored afterwards, so warm-up does
    not change later seeded draws. Warm up before other threads sample or draw random
    numbers: warm-up seeds and restores the process-wide generators.
    """
    windows, modes, seeds = tuple(windows), tuple(modes), tuple(seeds)
    worlds = list(observations or [])
    worlds += [warm_up_observations(int(k), n_controls=model.config.n_controls) for k in channels]
    if not worlds:
        raise ValueError("warm_up_sampling needs observations or channels")
    device = next(model.parameters()).device
    # ``torch.manual_seed`` seeds every device, so restore every CUDA generator, not only
    # the model's.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    before = model.sampling_status.compiles
    calls = 0
    start = _time.perf_counter()
    with contextlib.ExitStack() as restore:
        restore.enter_context(torch.random.fork_rng(devices=devices))
        if torch.backends.mps.is_available():
            restore.enter_context(torch.random.fork_rng(device_type="mps"))
        for world in worlds:
            for window in windows:
                for mode in modes:
                    for seed in seeds:
                        torch.manual_seed(seed)
                        decompose(model, world, n_samples=n_samples, window=window, mode=mode)
                        calls += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    after = model.sampling_status.compiles
    added = None if before is None or after is None else after - before
    return WarmUp(calls=calls, seconds=_time.perf_counter() - start, compiles=added)
