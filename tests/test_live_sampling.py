"""Sampling evaluates each flow only on coordinates live in some task.

A coordinate masked in every task is projected to zero after each Euler step, so
the sampler never evaluates its velocity. On every coordinate it does evaluate,
the velocity must equal the trained ``JointFlowHead.velocity`` up to float64
round-off; otherwise seeded draws would silently leave the trained sampler.
Noise is drawn per chunk of draws; how draws are then batched for transport, and
whether batches run concurrently on a share of the threads, must not change any graph
and changes draws by float round-off at most.
"""

import copy
import threading

import pytest
import torch

import pymc_mfm.model.flow as flow
from pymc_mfm.model import _metal
from pymc_mfm.model.encoder import BiAttentionBlock
from pymc_mfm.model.flow import _LiveVelocity
from pymc_mfm.model.graph import EdgeSlotLayout

ROUND_OFF = {"rtol": 1e-10, "atol": 1e-12}
#: Sampling and the encoder's kernels on an Apple GPU need PyTorch ``_metal.MIN_TORCH``;
#: older releases raise a clear error (sampling) or keep PyTorch's attention (encoder).
#: Sampling also refuses a GPU whose MPS linear layers are wrong (macOS virtual machines).
APPLE_GPU = (
    torch.backends.mps.is_available()
    and torch.__version__ >= _metal.MIN_TORCH
    and _metal.linear_correct()
)
#: Why the Apple GPU cases skip.
NO_APPLE_GPU = "needs an Apple GPU with PyTorch 2.9 or newer and correct MPS linear layers"
GPUS = [
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not APPLE_GPU, reason=NO_APPLE_GPU),
    ),
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


def _fresh_thread_count() -> int:
    """PyTorch's thread count in a thread started now."""
    seen = []
    thread = threading.Thread(target=lambda: seen.append(torch.get_num_threads()))
    thread.start()
    thread.join()
    return seen[0]


@pytest.fixture
def model64(model):
    return model.double()


@pytest.fixture
def batch64(batch):
    batch = {key: value.double() for key, value in batch.items()}
    # Channel 2 is padding in both tasks, so its coordinate and slots are dead;
    # control 1 and demand 1 are live in one task only.
    batch["active_c_mask"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float64)
    batch["active_m_mask"] = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    batch["active_j_mask"] = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
    return batch


def _assert_live_velocity_matches(head, weekly, channels, controls, mask, *, n_samples=3):
    d = mask.shape[-1]
    generator = torch.Generator().manual_seed(5)
    shape = (n_samples, *weekly.shape[:-1], d)
    x = torch.randn(shape, generator=generator, dtype=torch.float64) * mask
    latent = torch.randn(shape, generator=generator, dtype=torch.float64) * mask
    live = _LiveVelocity(head, weekly, channels, controls, mask)
    offset = live.trajectory_offset(latent)

    live_somewhere = set(torch.nonzero((mask != 0).reshape(-1, d).any(dim=0)).flatten().tolist())
    assert 0 in live_somewhere, "the baseline coordinate is never masked"
    assert set(live.index.tolist()) == live_somewhere, "evaluated coordinates must be the live ones"

    def expand(value):
        return value.unsqueeze(0).expand(n_samples, *value.shape)

    for t in (0.0, 0.3125, 0.9375):
        full = head.velocity(
            x,
            torch.full(shape[:-1], t, dtype=torch.float64),
            expand(weekly),
            expand(channels),
            expand(controls),
            latent,
            mask,
        )
        state = x.index_select(-1, live.index)
        torch.testing.assert_close(
            live.velocity(state, t, offset), full.index_select(-1, live.index), **ROUND_OFF
        )
        # The form traced by compiled sampling computes the same velocity.
        time = flow._fourier_time(torch.tensor(t, dtype=torch.float64))
        torch.testing.assert_close(
            live._velocity(state, time, offset),
            full.index_select(-1, live.index),
            **ROUND_OFF,
        )
    return live_somewhere


@pytest.mark.parametrize("window", ["query", "context"])
def test_router_live_velocity_matches_the_trained_velocity(model64, batch64, window):
    timeline = model64.encoder.forward_timeline(batch64)
    representations = timeline.query if window == "query" else timeline.context
    mask = model64._flow_state_mask(batch64)
    live = _assert_live_velocity_matches(
        model64.head,
        representations.weekly,
        representations.channels,
        representations.controls,
        mask,
    )
    assert len(live) < mask.shape[-1], "the fixture must leave coordinates dead in every task"


@pytest.mark.parametrize("graphs", ["active_and_empty", "both_empty"])
def test_decoder_live_velocity_matches_the_trained_velocity(model64, batch64, graphs):
    weekly, channels, controls = model64.encoder.forward_with_controls(batch64)
    layout = EdgeSlotLayout(3, 2, 2)
    state_mask = model64._flow_state_mask(batch64)
    possible = state_mask[:, 0, -layout.n_slots :]
    # An empty graph forbids every gated contribution: only the baseline stays stochastic.
    graph = possible * torch.tensor([[1.0], [0.0]] if graphs == "active_and_empty" else 0.0)
    n_dec = model64.conditional_head.state_dim(layout.K, layout.M)
    mask = model64._conditioned_state_mask(state_mask[..., :n_dec], graph, layout)
    _assert_live_velocity_matches(model64.conditional_head, weekly, channels, controls, mask)


def _draws(model, batch, monkeypatch, *, rows, workers, window):
    monkeypatch.setattr(flow, "_FLOW_SAMPLE_ROWS", rows)
    monkeypatch.setattr(flow, "_SAMPLE_BATCH_WORKERS", workers)
    torch.manual_seed(41)
    # 19 draws cross the 16-draw noise chunk.
    return model.sample_disentangled(batch, 19, window=window)


@pytest.mark.parametrize("window", ["query", "context"])
def test_transport_batches_and_their_schedule_never_change_a_draw(
    model64, batch64, monkeypatch, window
):
    # Concurrent batches split the caller's threads between them on every platform here.
    monkeypatch.setattr(flow, "_SPLIT_SAMPLE_THREADS", True)
    threads = torch.get_num_threads()
    torch.set_num_threads(max(threads, 4))
    try:
        whole, whole_graphs = _draws(
            model64, batch64, monkeypatch, rows=2**30, workers=1, window=window
        )
        # A one-row budget transports every draw in its own batch.
        serial, serial_graphs = _draws(
            model64, batch64, monkeypatch, rows=1, workers=1, window=window
        )
        concurrent, concurrent_graphs = _draws(
            model64, batch64, monkeypatch, rows=1, workers=4, window=window
        )
        # The workers' share of threads leaks neither into the caller nor into threads
        # started afterwards.
        assert torch.get_num_threads() == max(threads, 4)
        assert _fresh_thread_count() == max(threads, 4)
    finally:
        torch.set_num_threads(threads)
    assert torch.equal(serial_graphs, whole_graphs)
    torch.testing.assert_close(serial, whole, **ROUND_OFF)
    assert torch.equal(concurrent_graphs, serial_graphs)
    # Fewer threads per batch may reorder floating-point sums.
    torch.testing.assert_close(concurrent, serial, **ROUND_OFF)


def test_concurrent_batches_get_equal_thread_shares(monkeypatch):
    # Unequal shares let a batch run on 2 threads in one call and 3 in the next, which
    # changed round-off between identical calls.
    if not flow._thread_counts_per_thread():
        pytest.skip("thread counts are process-wide here")
    monkeypatch.setattr(flow, "_SPLIT_SAMPLE_THREADS", True)
    threads = torch.get_num_threads()
    torch.set_num_threads(flow._SAMPLE_BATCH_WORKERS + 3)
    try:
        counts = flow._map_batches(
            lambda start, stop: torch.get_num_threads(), 16, 1, torch.device("cpu")
        )
    finally:
        torch.set_num_threads(threads)
    assert len(counts) == 16
    assert set(counts) == {1}


@pytest.mark.parametrize("window", ["query", "context"])
def test_x86_network_layout_never_changes_a_draw(model64, batch64, monkeypatch, window):
    monkeypatch.setattr(flow, "_HIDDEN_CHUNK_ROWS", None)
    monkeypatch.setattr(flow, "_FOLD_FIRST_LAYER", False)
    whole, whole_graphs = _draws(
        model64, batch64, monkeypatch, rows=2**30, workers=1, window=window
    )
    # Seven-row chunks split every network's rows; most counts leave a short last chunk.
    # Each first layer is filled with its static and context terms before the scalar
    # product accumulates into it.
    monkeypatch.setattr(flow, "_HIDDEN_CHUNK_ROWS", 7)
    monkeypatch.setattr(flow, "_FOLD_FIRST_LAYER", True)
    chunked, chunked_graphs = _draws(
        model64, batch64, monkeypatch, rows=2**30, workers=1, window=window
    )
    assert torch.equal(chunked_graphs, whole_graphs)
    torch.testing.assert_close(chunked, whole, **ROUND_OFF)


@pytest.mark.parametrize("device", GPUS)
@pytest.mark.parametrize("rows", [2**30, 1])
@pytest.mark.parametrize("window", ["query", "context"])
def test_gpu_draws_the_cpu_samples(model, batch, monkeypatch, device, rows, window):
    # Noise comes from the CPU generator on every device, so seeded draws agree up to
    # float32 round-off; a one-row budget draws noise and transports one draw at a time.
    # The fused kernels transport the draws wherever the GPU has them.
    monkeypatch.setattr(flow, "_FLOW_SAMPLE_ROWS_GPU", rows)
    integrate = flow._metal.integrate
    calls = []
    monkeypatch.setattr(
        flow._metal, "integrate", lambda *a, **k: calls.append(1) or integrate(*a, **k)
    )
    torch.manual_seed(41)
    cpu, cpu_graphs = model.sample_disentangled(batch, 19, window=window)
    gpu_model = copy.deepcopy(model).to(device)
    torch.manual_seed(41)
    gpu, gpu_graphs = gpu_model.sample_disentangled(
        {key: value.to(device) for key, value in batch.items()}, 19, window=window
    )
    hidden = model.conditional_head.base_net[2].weight.shape[0]
    assert bool(calls) == flow._metal.has_kernels(torch.device(device), hidden)
    assert torch.equal(gpu_graphs.cpu(), cpu_graphs)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", GPUS)
def test_gpu_decodes_a_graph_that_turns_every_entity_off(model, batch, device):
    # With every edge off no entity contribution is live: the GPU kernels transport only
    # the baseline and fixed totals, as the CPU path does.
    tasks, _, n_channels = batch["qry_spend_z"].shape
    layout = EdgeSlotLayout(n_channels, model.config.n_controls, model.config.n_demand)
    graph = torch.zeros(tasks, layout.n_slots)
    torch.manual_seed(41)
    cpu, _ = model.sample_disentangled(batch, 19, mode="fixed", graph=graph)
    gpu_model = copy.deepcopy(model).to(device)
    torch.manual_seed(41)
    gpu, _ = gpu_model.sample_disentangled(
        {key: value.to(device) for key, value in batch.items()}, 19, mode="fixed", graph=graph
    )
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("uniform", [False, True])
def test_host_noise_from_a_generator_equals_the_global_stream_and_leaves_it_alone(uniform):
    # The GPU path draws noise on the host, directly or on the prefetch worker.
    shapes = [(1, 6, 3, 5), (1, 2, 3, 5)]

    def chunks(**kwargs):
        return [flow._draw_host_noise(shape, torch.float32, uniform, **kwargs) for shape in shapes]

    def flat(drawn):
        return [part.values.reshape(-1) for pair in drawn for part in pair]

    torch.manual_seed(8)
    expected = flat(chunks())
    torch.manual_seed(1)
    before = torch.get_rng_state()
    direct = flat(chunks(generator=torch.Generator().manual_seed(8)))
    assert torch.equal(torch.get_rng_state(), before)
    assert all(map(torch.equal, direct, expected))


def test_prefetched_noise_from_a_generator_leaves_the_global_stream_alone():
    shapes = [(1, 6, 3, 5), (1, 2, 3, 5)]
    prefetch = flow._NoisePrefetch()
    try:
        torch.manual_seed(1)
        before = torch.get_rng_state()
        seeded = list(prefetch.request(shapes, torch.float32, torch.Generator().manual_seed(8)))
        assert torch.equal(torch.get_rng_state(), before)
        replay = list(prefetch.request(shapes, torch.float32, torch.Generator().manual_seed(8)))
    finally:
        prefetch.close()
    for pair, other in zip(seeded, replay, strict=True):
        assert all(torch.equal(a.values, b.values) for a, b in zip(pair, other, strict=True))


@pytest.mark.parametrize("device", GPUS)
@pytest.mark.parametrize("window", ["query", "context"])
def test_a_seeded_gpu_call_draws_the_cpu_stream_and_leaves_the_global_state(
    model, batch, device, window
):
    # Not run where no GPU exists: the seeded noise is drawn on the host with the call's
    # CPU generator, so the GPU agrees with the CPU up to round-off and no global state moves.
    cpu, cpu_graphs = model.sample_disentangled(batch, 19, window=window, random_seed=41)
    gpu_model = copy.deepcopy(model).to(device)
    torch.manual_seed(5)
    before = torch.get_rng_state()
    gpu, gpu_graphs = gpu_model.sample_disentangled(
        {key: value.to(device) for key, value in batch.items()},
        19,
        window=window,
        random_seed=41,
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(gpu_graphs.cpu(), cpu_graphs)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not APPLE_GPU, reason=NO_APPLE_GPU)
def test_apple_gpu_attention_kernels_match_masked_attention(monkeypatch):
    # 16-wide heads run the Metal kernels. 70 weeks span three runs of 32 causal keys;
    # the two tasks pad different entities and the last entity is active in neither.
    torch.manual_seed(12)
    block = BiAttentionBlock(d_model=32, n_heads=2, d_ff=64).to("mps").eval()
    x = torch.randn(2, 70, 5, 32, device="mps")
    time_mask = torch.ones(70, 70, dtype=torch.bool, device="mps").tril()
    active = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 0]], dtype=torch.bool, device="mps")
    calls = []
    for name in ("causal_attention", "masked_attention"):
        kernel = getattr(_metal, name)
        monkeypatch.setattr(
            _metal, name, lambda *a, kernel=kernel, name=name: calls.append(name) or kernel(*a)
        )
    with torch.no_grad():
        kernels = block(x, time_mask, active, causal=True)
        monkeypatch.setattr(_metal, "attention_supported", lambda *_: False)
        reference = block(x, time_mask, active, causal=True)
    assert calls == ["causal_attention", "masked_attention"]
    torch.testing.assert_close(kernels.cpu(), reference.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("window", ["query", "context"])
def test_structure_only_router_draws_the_same_graph_flow(model64, batch64, window):
    timeline = model64.encoder.forward_timeline(batch64)
    representations = timeline.query if window == "query" else timeline.context
    flows = []
    for structure_only in (False, True):
        torch.manual_seed(43)
        decomposition, graph_flow = model64.sample(
            batch64,
            representations.weekly,
            representations.channels,
            n_samples=19,
            return_structure=True,
            control_repr=representations.controls,
            structure_only=structure_only,
        )
        assert (decomposition is None) == structure_only
        flows.append(graph_flow)
    torch.testing.assert_close(flows[1], flows[0], **ROUND_OFF)
