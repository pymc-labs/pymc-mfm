"""Opt-in compiled sampling: same draws as eager up to round-off, same seeded draw order.

Compilation must not consume the global random state, change a checkpoint or leave
results that depend on whether a call happened to compile. Most tests trace with
Dynamo's ``eager`` backend so they run fast; one test runs the real default compiler.
"""

import contextlib
import copy
import pickle

import numpy as np
import pytest
import torch

import pymc_mfm.model.compiled as compiled_module
from pymc_mfm.inference import decompose, warm_up_observations, warm_up_sampling
from pymc_mfm.model import SamplingStatus

ROUND_OFF = {"rtol": 1e-5, "atol": 1e-6}


@pytest.fixture
def traced(monkeypatch):
    """Trace with Dynamo but run the captured graph eagerly (fast, CPU-only)."""
    torch._dynamo.reset()
    real_compile = torch.compile

    def compile_eager(function, *, mode=None, dynamic=None):
        return real_compile(function, backend="eager", dynamic=dynamic)

    monkeypatch.setattr(compiled_module.torch, "compile", compile_eager)
    yield
    torch._dynamo.reset()


def _draws(model, batch, seed=3, n=19, window="query"):
    torch.manual_seed(seed)
    return model.sample_disentangled(batch, n, window=window)


def test_sampling_is_eager_until_compiled(model):
    assert model.sampling_status == SamplingStatus(compiled=False)


@pytest.mark.parametrize("window", ["query", "context"])
def test_compiled_draws_match_eager_draws(model, batch, traced, window):
    eager, eager_graphs = _draws(model, batch, window=window)
    model.compile_sampling()
    compiled, compiled_graphs = _draws(model, batch, window=window)
    torch.testing.assert_close(compiled_graphs, eager_graphs, rtol=0, atol=0)
    torch.testing.assert_close(compiled, eager, **ROUND_OFF)
    status = model.sampling_status
    assert status.compiled and status.mode == "default" and status.fallback is None
    assert status.compiles is None or status.compiles >= 1


def test_a_compiling_call_keeps_the_seeded_draw_order(model, batch, traced, monkeypatch):
    model.compile_sampling()
    real = compiled_module.CompiledVelocity._function
    consumed = []

    def compile_that_draws(self):
        function = real(self)

        def run(*args):
            # Autotuning compilers draw random benchmark inputs; the sampler must not see it.
            if not consumed:
                consumed.append(torch.randn(7))
            return function(*args)

        return run

    monkeypatch.setattr(compiled_module.CompiledVelocity, "_function", compile_that_draws)
    torch.manual_seed(3)
    cold = model.sample_disentangled(batch, 19)
    after_cold = torch.get_rng_state()
    torch.manual_seed(3)
    warm = model.sample_disentangled(batch, 19)
    torch.testing.assert_close(cold, warm, rtol=0, atol=0)
    torch.testing.assert_close(after_cold, torch.get_rng_state(), rtol=0, atol=0)


def test_compiled_calls_preserve_the_global_random_state(model, batch, traced):
    torch.manual_seed(3)
    model.sample_disentangled(batch, 19)
    eager_state = torch.get_rng_state()
    model.compile_sampling()
    torch.manual_seed(3)
    model.sample_disentangled(batch, 19)
    torch.testing.assert_close(torch.get_rng_state(), eager_state, rtol=0, atol=0)


def test_disable_returns_to_eager(model, batch, traced):
    eager, _ = _draws(model, batch)
    model.compile_sampling().disable_compiled_sampling()
    assert model.sampling_status == SamplingStatus(compiled=False)
    again, _ = _draws(model, batch)
    torch.testing.assert_close(again, eager, rtol=0, atol=0)


def test_compiler_failure_warns_and_falls_back_to_eager(model, batch, monkeypatch):
    def broken(function, *, mode=None, dynamic=None):
        def fail(*args):
            raise RuntimeError("no compiler here")

        return fail

    monkeypatch.setattr(compiled_module.torch, "compile", broken)
    eager, _ = _draws(model, batch)
    model.compile_sampling()
    with pytest.warns(RuntimeWarning, match="samples eagerly"):
        fallback, _ = _draws(model, batch)
    torch.testing.assert_close(fallback, eager, rtol=0, atol=0)
    status = model.sampling_status
    assert status.compiled and status.fallback == "RuntimeError: no compiler here"


def test_compiling_leaves_weights_and_copies_unchanged(model, batch, traced):
    keys = list(model.state_dict())
    model.compile_sampling(mode="reduce-overhead")
    _draws(model, batch)
    assert list(model.state_dict()) == keys
    for clone in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        status = clone.sampling_status
        assert status.compiled and status.mode == "reduce-overhead"
        assert status.compiles in (0, None)


def test_compile_count_survives_a_compiler_reset(model, batch, traced):
    model.compile_sampling()
    _draws(model, batch)
    first = model.sampling_status.compiles
    torch.compiler.reset()  # drops compiled code and every compile callback
    _draws(model, batch)
    assert first is None or model.sampling_status.compiles > first


def test_compile_mode_is_checked(model):
    with pytest.raises(ValueError, match="mode must be one of"):
        model.compile_sampling(mode="fastest")


def test_warm_up_runs_each_world_and_keeps_the_random_state(model, traced):
    model.compile_sampling()
    torch.manual_seed(9)
    state = torch.get_rng_state()
    report = warm_up_sampling(model, channels=(1, 2), seeds=(0,), n_samples=8)
    assert report.calls == 2 * 2  # two worlds, two windows, one mode, one seed
    assert report.compiles is None or report.compiles >= 1
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)


def test_warm_up_accepts_one_shot_iterables(model, traced):
    report = warm_up_sampling(
        model, channels=iter((1, 2)), windows=iter(("query",)), seeds=iter((0, 1)), n_samples=8
    )
    assert report.calls == 2 * 1 * 2


def test_warm_up_restores_every_cuda_generator(model, monkeypatch):
    """``torch.manual_seed`` seeds all devices, so warm-up must restore all of them."""
    forked = []

    @contextlib.contextmanager
    def record(devices=None, enabled=True, device_type="cuda"):
        forked.append((device_type, list(devices) if devices is not None else None))
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.random, "fork_rng", record)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    warm_up_sampling(model, channels=(1,), windows=("query",), seeds=(0,), n_samples=4)
    assert ("cuda", [0, 1]) in forked


def test_warm_up_needs_inputs(model):
    with pytest.raises(ValueError, match="needs observations or channels"):
        warm_up_sampling(model)


def test_warm_up_world_is_a_valid_observation(model):
    world = warm_up_observations(3, n_controls=model.config.n_controls, seed=1)
    result = decompose(model, world, n_samples=4)
    assert np.isfinite(result.components).all()


def test_default_compiler_matches_eager(model, batch):
    """The real ``torch.compile`` path (Inductor on CPU) agrees with eager sampling."""
    torch._dynamo.reset()
    try:
        eager, eager_graphs = _draws(model, batch, window="context")
        model.compile_sampling()
        compiled, compiled_graphs = _draws(model, batch, window="context")
        assert model.sampling_status.fallback is None
        torch.testing.assert_close(compiled_graphs, eager_graphs, rtol=0, atol=0)
        torch.testing.assert_close(compiled, eager, **ROUND_OFF)
        repeat, _ = _draws(model, batch, window="context")
        torch.testing.assert_close(repeat, compiled, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
