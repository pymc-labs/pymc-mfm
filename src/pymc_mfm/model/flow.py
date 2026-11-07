"""Equivariant trajectory flows for graph routing and conditional decomposition."""

from __future__ import annotations

import itertools
import math
import platform
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import _metal
from .compiled import CompiledVelocity
from .config import ModelConfig
from .graph import N_NODE_TYPES, STRUCT_EDGE_KEYS, EdgeSlotLayout
from .ode import integrate_flow

DECOMPOSITION_FIXED_TARGETS = (
    "confounder_contribution_total",
    "indirect_cc",
    "indirect_zc",
    "indirect_dc",
)
ENTITY_CHANNEL, ENTITY_CONTROL = 0, 1
_ENTITY_EMBED_DIM = 8
_TRAJECTORY_EMBED_DIM = 8
_TRAJECTORY_RANK = 4
_TRAJECTORY_CONSTANT_LOADING_SCALE = 3.5
_TRAJECTORY_FEATURE_VERSION = 5
_FLOW_SAMPLE_CHUNK = 16
#: State rows (draws × tasks × weeks × live coordinates) per sampling velocity evaluation.
#: It bounds memory without changing base-noise draws; kernel shapes may change rounding.
_FLOW_SAMPLE_ROWS = 2**15
#: GPUs with the fused sampling kernels (see :mod:`._metal`) run few, large batches best:
#: the kernels keep no per-row activations.
_FLOW_SAMPLE_ROWS_GPU = 2**19
# Transporting several batches at once overlaps one batch's matrix products with the others'
# elementwise work and Python overhead. Off Apple silicon each batch worker runs on its own
# share of PyTorch's threads: on Linux x86 capped at 8 CPUs, eight one-thread workers
# sampled about 2x faster than one batch on every thread on AMD EPYC and about as fast on
# Intel Xeon, while workers that each used every thread oversubscribed the cores and lost.
# On Apple silicon, where Accelerate runs the products on a separate matrix unit, eight
# workers share every thread; splitting gained nothing there. In-flight memory scales with
# the count.
_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
_SAMPLE_BATCH_WORKERS = 8
_SPLIT_SAMPLE_THREADS = not _APPLE_SILICON
#: Off Apple silicon, CPU sampling runs each network's hidden and output layers on row chunks
#: of this size, so a chunk's activations stay in cache through the product, bias, SiLU and
#: output layer instead of streaming through memory between them (Apple silicon ran slower).
_HIDDEN_CHUNK_ROWS = None if _APPLE_SILICON else 4096

# Fixed architecture numerics shared by A/B and every new training.
#: Fourier frequencies of the flow-time embedding; they set layer input widths.
FLOW_N_FREQ = 2
#: Uniform Euler steps of the sampling grid.
FLOW_STEPS_SAMPLE = 16
#: Base-noise variance share of the rank-four trajectory component.
FLOW_TRAJECTORY_CORRELATION = 0.98
#: Draws per sampling call when ``n_samples`` is omitted.
DEFAULT_N_SAMPLES = 32


@dataclass(frozen=True)
class ZeroAtomGate:
    """Contribution block and the graph family required for nonzero support.

    Channel and control contributions require their own cy and zb edge.
    Confounder and indirect totals require any edge in their source family.
    Baseline is never gated; dz and zz have no decomposition column.
    Every gated column's natural zero is exactly zero on the shared ruler.
    """

    column: str
    family: str
    reduction: str


ZERO_ATOM_GATES: tuple[ZeroAtomGate, ...] = (
    ZeroAtomGate("contribution", "cy", "elementwise"),
    ZeroAtomGate("control_contribution", "zb", "elementwise"),
    ZeroAtomGate("confounder_contribution_total", "db", "any"),
    ZeroAtomGate("indirect_cc", "cc", "any"),
    ZeroAtomGate("indirect_zc", "zc", "any"),
    ZeroAtomGate("indirect_dc", "dc", "any"),
)


def _fourier_time(t: Tensor) -> Tensor:
    """Map scalar t in [0,1] to [t, sin(2^k pi t), cos(2^k pi t)] for k < FLOW_N_FREQ."""
    t = t.unsqueeze(-1)
    freqs = (2.0 ** torch.arange(FLOW_N_FREQ, device=t.device, dtype=t.dtype)) * torch.pi
    ang = t * freqs
    return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=-1)


@lru_cache(maxsize=32)
def _edge_slot_layout(n_channels: int, n_controls: int, n_demand: int) -> EdgeSlotLayout:
    """Cache immutable edge geometry used by every velocity evaluation."""
    return EdgeSlotLayout(n_channels, n_controls, n_demand)


def _sample_count(value: object, name: str = "n_samples") -> int:
    """Validate a draw count: a positive integer, never a bool or a fraction."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sample_rows(device: torch.device) -> int:
    """State rows per sampling velocity evaluation on ``device``."""
    return _FLOW_SAMPLE_ROWS_GPU if _metal.has_kernels(device) else _FLOW_SAMPLE_ROWS


def _noise_device(device: torch.device) -> torch.device:
    """Where sampling draws base noise for ``device``.

    GPU sampling (MPS and CUDA) draws with the CPU generator and copies the noise to the
    GPU, so a seeded call draws the same base noise, and the same samples up to round-off,
    on every device.
    """
    return torch.device("cpu") if device.type in ("mps", "cuda") else device


def _to_device(value: Tensor, device: torch.device) -> Tensor:
    """``value`` on ``device``; a host-to-GPU copy does not wait for queued GPU work."""
    if value.device != device and device.type in ("mps", "cuda") and value.device.type == "cpu":
        return _metal.to_device_async(value, device)
    return value.to(device)


@lru_cache(maxsize=1)
def _thread_counts_per_thread() -> bool:
    """Whether ``torch.set_num_threads`` in one thread leaves other threads' count alone.

    It does under PyTorch's OpenMP backend, whose thread count is per thread; CPU batch
    workers split the threads only then. The call also sets the count that threads started
    later begin with, so the caller sets its own count again afterwards.
    """
    before = torch.get_num_threads()
    probe = threading.Thread(target=torch.set_num_threads, args=(1,))
    probe.start()
    probe.join()
    per_thread = torch.get_num_threads() == before
    torch.set_num_threads(before)
    return per_thread


def _map_batches[T](
    fn: Callable[[int, int], T], total: int, max_size: int, device: torch.device
) -> list[T]:
    """Split ``range(total)`` into balanced batches and return ``fn(start, stop)`` for each.

    Batches hold at most ``max_size`` items and run several at a time where that helps;
    their count is then a multiple of the workers, so no worker idles in a short last
    round. Batches share no state and draw no randomness, preserving sample identity.
    Different batch shapes may introduce floating-point round-off. Grad mode is
    thread-local and carried into workers, which never outnumber ``torch.get_num_threads()``;
    off Apple silicon they split those threads between them.
    """
    threads = torch.get_num_threads()
    workers = min(_SAMPLE_BATCH_WORKERS, threads) if device.type == "cpu" else 1
    count = -(-total // max_size)
    if workers > 1:
        count = min(total, -(-count // workers) * workers)
    bounds = [(total * i // count, total * (i + 1) // count) for i in range(count)]
    if workers < 2 or count < 2:
        return [fn(start, stop) for start, stop in bounds]
    grad_enabled = torch.is_grad_enabled()
    pool_size = min(workers, count)
    initializer = None
    if _SPLIT_SAMPLE_THREADS and _thread_counts_per_thread():
        share, extra = divmod(threads, pool_size)
        slots = itertools.count()

        def split_threads() -> None:
            # Per pool thread, which ends with the pool.
            torch.set_num_threads(share + (next(slots) < extra))

        initializer = split_threads

    def run(bound: tuple[int, int]) -> T:
        with torch.set_grad_enabled(grad_enabled):
            return fn(*bound)

    try:
        with ThreadPoolExecutor(max_workers=pool_size, initializer=initializer) as pool:
            return list(pool.map(run, bounds))
    finally:
        if initializer is not None:
            # The workers' calls also set the count that threads started later begin with;
            # setting the caller's count again restores it.
            torch.set_num_threads(threads)


class _OrderedNoise:
    """Base draws for consecutive draw ranges, taken chunk by chunk as transport asks.

    ``chunks`` yields the host noise chunks ``(white, z)`` (see :func:`_draw_host_noise`)
    in draw order, exactly as when all are drawn up front. Each range is finished and
    mixed on the GPU, which transports one batch while the CPU draws the next batch's
    noise. Ranges must be taken in order. The copies to the GPU do not wait for queued
    work (see :func:`_metal.to_device_async`).
    """

    def __init__(
        self,
        head: JointFlowHead,
        chunks: Iterator[tuple[_metal.HostNoise, _metal.HostNoise]],
        weekly_repr: Tensor,
    ):
        self.head = head
        self.chunks = chunks
        self.weekly_repr = weekly_repr
        self.drawn = self.taken = 0
        self.white: list[Tensor] = []
        self.z: list[Tensor] = []

    def __call__(
        self, start: int, stop: int, columns: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Mixed base draws ``[start, stop)``, restricted to state ``columns`` if given."""
        if start != self.taken:
            raise RuntimeError("base draws must be taken in draw order")
        pending = []
        while self.drawn < stop:
            pending.append(next(self.chunks))
            self.drawn += pending[-1][0].shape[0]
        if pending:
            device = self.weekly_repr.device
            rows = sum(white.shape[0] for white, _ in pending)
            for part, taken in ((0, self.white), (1, self.z)):
                parts = [chunk[part] for chunk in pending]
                values = _metal.to_device(parts, device)
                taken.append(values.view(rows, *parts[0].shape[1:]))
        white = torch.cat(self.white) if len(self.white) > 1 else self.white[0]
        z = torch.cat(self.z) if len(self.z) > 1 else self.z[0]
        n = stop - start
        self.white, self.z = [white[n:]], [z[n:]]
        self.taken = stop
        white, z = white[:n], z[:n]
        if columns is not None:
            # Mixing acts on each state column separately: mix only the ones transported.
            white, z = white.index_select(-1, columns), z.index_select(-2, columns)
        return self.head._mix_noise(white, z, self.weekly_repr)


def _draw_host_noise(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    uniform: bool,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[_metal.HostNoise, _metal.HostNoise]:
    """:meth:`JointFlowHead._draw_noise` drawn on the host for the GPU to finish.

    White noise, then the rank-four latents; see :func:`_metal.draw_host`. ``out``
    optionally holds the two flat host tensors that receive them.
    """
    white, z = (None, None) if out is None else out
    return (
        _metal.draw_host(shape, dtype, uniform, white),
        _metal.draw_host(_latent_shape(shape), dtype, uniform, z),
    )


def _latent_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shape of the rank-four trajectory latents drawn with white noise of ``shape``."""
    return (*shape[:-2], shape[-1], _TRAJECTORY_RANK)


@dataclass(frozen=True)
class _NoiseRequest:
    """Noise chunks of the given shapes, drawn in order by a :class:`_NoisePrefetch`."""

    shapes: tuple[tuple[int, ...], ...]
    dtype: torch.dtype
    chunks: tuple[Future, ...]

    def __iter__(self) -> Iterator[tuple[_metal.HostNoise, _metal.HostNoise]]:
        return (chunk.result() for chunk in self.chunks)


class _NoisePrefetch:
    """Draws base-noise chunks with the CPU generator on a worker thread, in request order.

    GPU sampling draws its noise on the CPU (see :func:`_noise_device`). Requesting the
    chunks as soon as their shapes are known overlaps the generator with GPU work. One
    worker draws the requests in submission order, so the noise equals drawing it inline,
    provided nothing else draws random numbers until :meth:`close` returns. A request's
    chunks land end to end in one buffer per kind, which the GPU copy reads directly;
    ``pin`` (for CUDA) page-locks those buffers so the copy does not wait.
    """

    def __init__(self, *, pin: bool = False):
        self._worker = ThreadPoolExecutor(max_workers=1)
        self._pin = pin
        # Decided here: the check runs GPU work, which stays on the calling thread.
        self._uniform = _metal.uniform_normals()

    def request(self, shapes: list[tuple[int, ...]], dtype: torch.dtype) -> _NoiseRequest:
        sizes = [
            (
                _metal.host_size(tuple(shape), dtype, self._uniform),
                _metal.host_size(_latent_shape(tuple(shape)), dtype, self._uniform),
            )
            for shape in shapes
        ]
        values_dtype = torch.float32 if self._uniform and dtype == torch.float32 else dtype
        white = torch.empty(
            sum(size for size, _ in sizes), dtype=values_dtype, pin_memory=self._pin
        )
        z = torch.empty(sum(size for _, size in sizes), dtype=values_dtype, pin_memory=self._pin)
        chunks, white_start, z_start = [], 0, 0
        for shape, (white_size, z_size) in zip(shapes, sizes, strict=True):
            out = (white[white_start : white_start + white_size], z[z_start : z_start + z_size])
            chunks.append(
                self._worker.submit(_draw_host_noise, tuple(shape), dtype, self._uniform, out)
            )
            white_start += white_size
            z_start += z_size
        return _NoiseRequest(tuple(map(tuple, shapes)), dtype, tuple(chunks))

    def close(self) -> None:
        """Wait for the worker; requests not yet started are dropped."""
        self._worker.shutdown(wait=True, cancel_futures=True)


class JointFlowHead(nn.Module):
    """Rectified flow over the full additive decomposition of each week.

    The router appends the complete graph in ``EdgeSlotLayout`` order and
    couples graph and contribution velocities. The conditional decoder carries
    only decomposition coordinates; its representations and support mask are
    supplied by the model's graph conditioner. Both use a rank-four coherent
    base distribution and pool the live trajectory state at every drift step.
    """

    #: Compiled sampling velocity set by ``MFM.compile_sampling``; ``None`` samples eagerly.
    #: It is neither a submodule nor a buffer, so weights and checkpoints never see it.
    _compiled_velocity: CompiledVelocity | None = None

    def __init__(self, config: ModelConfig, *, graph_router: bool):
        super().__init__()
        self.graph_router = graph_router
        self.n_controls = config.n_controls
        self.n_demand = config.n_demand
        d_model = d_channel = config.d_model
        hidden = config.flow_hidden
        t_dim = 1 + 2 * FLOW_N_FREQ
        entity_edge_state = int(graph_router)

        self.trajectory_state_proj = nn.Linear(2, _TRAJECTORY_EMBED_DIM)
        self.trajectory_week_proj = nn.Linear(d_model, _TRAJECTORY_EMBED_DIM)
        self.trajectory_token_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(_TRAJECTORY_EMBED_DIM, _TRAJECTORY_EMBED_DIM),
        )
        self.trajectory_score = nn.Linear(_TRAJECTORY_EMBED_DIM, 1)
        self.trajectory_loading_net = nn.Sequential(
            nn.Linear(d_model, _TRAJECTORY_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(_TRAJECTORY_EMBED_DIM, _TRAJECTORY_RANK - 1),
        )
        self.register_buffer(
            "trajectory_feature_version",
            torch.tensor(_TRAJECTORY_FEATURE_VERSION, dtype=torch.int64),
        )
        self.base_net = nn.Sequential(
            nn.Linear(1 + t_dim + d_model + 2 + _TRAJECTORY_EMBED_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.contrib_net = nn.Sequential(
            nn.Linear(
                1
                + t_dim
                + d_model
                + d_channel
                + 3
                + _ENTITY_EMBED_DIM
                + entity_edge_state
                + _TRAJECTORY_EMBED_DIM
                + entity_edge_state * _TRAJECTORY_EMBED_DIM,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.entity_type_embed = nn.Embedding(2, _ENTITY_EMBED_DIM)
        self.fixed_slot_embed = nn.Embedding(len(DECOMPOSITION_FIXED_TARGETS), _ENTITY_EMBED_DIM)
        self.fixed_net = nn.Sequential(
            nn.Linear(
                1 + t_dim + d_model + 2 + _ENTITY_EMBED_DIM + _TRAJECTORY_EMBED_DIM,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        if graph_router:
            self.node_type_embed = nn.Embedding(N_NODE_TYPES, _ENTITY_EMBED_DIM)
            self.demand_node_embed = nn.Embedding(max(self.n_demand, 1), d_channel)
            self.baseline_node_embed = nn.Embedding(1, d_channel)
            self.sales_node_proj = nn.Linear(d_model, d_channel)
            self.edge_net = nn.Sequential(
                nn.Linear(
                    3
                    + t_dim
                    + d_model
                    + 2 * d_channel
                    + 2 * _ENTITY_EMBED_DIM
                    + 3 * _TRAJECTORY_EMBED_DIM,
                    hidden,
                ),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            self._edge_tensor_cache: dict[
                tuple[int, int, int, str],
                tuple[Tensor, Tensor, Tensor, Tensor],
            ] = {}

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Equal-shaped older weights can encode different trajectory features.
        # Guard here so direct state_dict loading and artifact loading agree.
        key = prefix + "trajectory_feature_version"
        version = state_dict.get(key)
        if (
            not isinstance(version, Tensor)
            or version.shape != torch.Size([])
            or version.dtype != torch.int64
            or version.item() != _TRAJECTORY_FEATURE_VERSION
        ):
            error_msgs.append(
                f"{key} must be scalar int64 feature version {_TRAJECTORY_FEATURE_VERSION}"
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @property
    def n_fixed(self) -> int:
        return len(DECOMPOSITION_FIXED_TARGETS)

    def state_dim(self, c: int, m: int | None = None) -> int:
        """State width for observed entity counts, defaulting to training controls."""
        m_eff = self.n_controls if m is None else m
        d = 1 + c + m_eff + self.n_fixed
        if self.graph_router:
            d += _edge_slot_layout(c, m_eff, self.n_demand).n_slots
        return d

    def _edge_layout_tensors(
        self, layout: EdgeSlotLayout, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Cache immutable slot indices/types outside the ODE hot loop."""
        key = (layout.K, layout.M, layout.J, str(device))
        cached = self._edge_tensor_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(layout.slot_src_index, dtype=torch.long, device=device),
                torch.as_tensor(layout.slot_dst_index, dtype=torch.long, device=device),
                torch.as_tensor(layout.slot_src_type, dtype=torch.long, device=device),
                torch.as_tensor(layout.slot_dst_type, dtype=torch.long, device=device),
            )
            self._edge_tensor_cache[key] = cached
        return cached

    def _entity_repr(self, channel_repr: Tensor, control_repr: Tensor | None) -> Tensor:
        """Concatenate count-agnostic channel and control representations."""
        if control_repr is None:
            raise ValueError("decomposition flow head requires control_repr")
        return torch.cat([channel_repr, control_repr], dim=-2)

    def _trajectory_context(
        self,
        weekly_repr: Tensor,
        trajectory_latent: Tensor,
        state_mask: Tensor | None = None,
        x: Tensor | None = None,
    ) -> Tensor:
        """Learn one live-state trajectory summary per state coordinate."""
        if weekly_repr.shape[:-1] != trajectory_latent.shape[:-1]:
            raise ValueError(
                "weekly_repr and trajectory_latent must share leading trajectory axes, "
                f"got {tuple(weekly_repr.shape)} and {tuple(trajectory_latent.shape)}"
            )
        if x is None:
            raise ValueError("trajectory context requires the live flow state")
        state_inputs = torch.stack([x, trajectory_latent], dim=-1)
        state_tokens = self.trajectory_state_proj(state_inputs)
        week_tokens = self.trajectory_week_proj(weekly_repr).unsqueeze(-2)
        tokens = self.trajectory_token_net(state_tokens + week_tokens)
        weights = torch.softmax(self.trajectory_score(tokens), dim=-3)
        context = (weights * tokens).sum(dim=-3, keepdim=True).expand_as(tokens)
        if state_mask is not None:
            context = context * state_mask.to(context.dtype).unsqueeze(-1)
        return context

    def velocity(
        self,
        x: Tensor,
        t: Tensor,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor | None = None,
        trajectory_latent: Tensor | None = None,
        state_mask: Tensor | None = None,
    ) -> Tensor:
        if trajectory_latent is None:
            raise ValueError("trajectory_latent is required by the coherent flow")
        if trajectory_latent.shape != x.shape:
            raise ValueError(
                "trajectory_latent must match the flow state shape, "
                f"got {tuple(trajectory_latent.shape)} for {tuple(x.shape)}"
            )
        trajectory_context = self._trajectory_context(weekly_repr, trajectory_latent, state_mask, x)
        c = channel_repr.shape[-2]
        entity_repr = self._entity_repr(channel_repr, control_repr)
        n_entities = entity_repr.shape[-2]
        x_base = x[..., :1]
        x_entity = x[..., 1 : 1 + n_entities]
        if state_mask is None:
            entity_active = torch.ones_like(x_entity)
        else:
            entity_active = state_mask[..., 1 : 1 + n_entities].to(x.dtype)
            try:
                entity_active = torch.broadcast_to(entity_active, x_entity.shape)
            except RuntimeError as error:
                raise ValueError(
                    "state_mask entity prefix does not broadcast to the flow state: "
                    f"{tuple(state_mask.shape)} versus {tuple(x.shape)}"
                ) from error
        entity_count = entity_active.sum(dim=-1, keepdim=True)
        if self.graph_router:
            if (entity_count == 0).any():
                raise ValueError("flow state has a task/week with no active entities")
        else:
            # An all-off conditioning graph leaves only baseline stochastic.
            entity_count = entity_count.clamp_min(1)
        entity_repr = entity_repr * entity_active.unsqueeze(-1)
        entity_mean = (x_entity * entity_active).sum(dim=-1, keepdim=True) / entity_count
        entity_std = (
            torch.linalg.vector_norm((x_entity - entity_mean) * entity_active, dim=-1, keepdim=True)
            / entity_count.sqrt()
        )
        te = _fourier_time(t)
        base_parts = [x_base, te, weekly_repr, entity_mean, entity_std]
        base_parts.append(trajectory_context[..., 0, :])
        v_base = self.base_net(torch.cat(base_parts, dim=-1))

        weekly_exp = weekly_repr.unsqueeze(-2).expand(
            *weekly_repr.shape[:-1], n_entities, weekly_repr.shape[-1]
        )
        te_exp = te.unsqueeze(-2).expand(*te.shape[:-1], n_entities, te.shape[-1])
        x_base_exp = x_base.unsqueeze(-2).expand(*x_base.shape[:-1], n_entities, 1)
        mean_exp = entity_mean.unsqueeze(-2).expand(*entity_mean.shape[:-1], n_entities, 1)
        std_exp = entity_std.unsqueeze(-2).expand(*entity_std.shape[:-1], n_entities, 1)
        layout = None
        x_edges = None
        if self.graph_router:
            layout = _edge_slot_layout(c, n_entities - c, self.n_demand)
            x_edges = x[..., 1 + n_entities + self.n_fixed :]
            if x_edges.shape[-1] != layout.n_slots:
                raise ValueError(
                    f"joint graph state carries {x_edges.shape[-1]} edge columns but the "
                    f"layout for K={c}, M={n_entities - c}, J={self.n_demand} has "
                    f"{layout.n_slots} slots"
                )
        entity_in = [
            x_entity.unsqueeze(-1),
            te_exp,
            weekly_exp,
            entity_repr,
            x_base_exp,
            mean_exp,
            std_exp,
        ]
        type_ids = torch.cat(
            [
                torch.full((c,), ENTITY_CHANNEL, dtype=torch.long, device=x.device),
                torch.full((n_entities - c,), ENTITY_CONTROL, dtype=torch.long, device=x.device),
            ]
        )
        type_emb = self.entity_type_embed(type_ids).expand(
            *x.shape[:-1], n_entities, _ENTITY_EMBED_DIM
        )
        entity_in.append(type_emb.to(x.dtype))
        if self.graph_router:
            own_edge = torch.cat(
                [x_edges[..., layout.slices["cy"]], x_edges[..., layout.slices["zb"]]], dim=-1
            )
            entity_in.append(own_edge.unsqueeze(-1))
        entity_in.append(trajectory_context[..., 1 : 1 + n_entities, :])
        if self.graph_router:
            edge_context = trajectory_context[..., 1 + n_entities + self.n_fixed :, :]
            entity_in.append(
                torch.cat(
                    [
                        edge_context[..., layout.slices["cy"], :],
                        edge_context[..., layout.slices["zb"], :],
                    ],
                    dim=-2,
                )
            )
        v_per_entity = self.contrib_net(torch.cat(entity_in, dim=-1))

        x_fixed = x[..., 1 + n_entities : 1 + n_entities + self.n_fixed]
        slot_ids = torch.arange(self.n_fixed, dtype=torch.long, device=x.device)
        slot_emb = self.fixed_slot_embed(slot_ids).expand(
            *x.shape[:-1], self.n_fixed, _ENTITY_EMBED_DIM
        )
        weekly_fixed = weekly_repr.unsqueeze(-2).expand(
            *weekly_repr.shape[:-1], self.n_fixed, weekly_repr.shape[-1]
        )
        te_fixed = te.unsqueeze(-2).expand(*te.shape[:-1], self.n_fixed, te.shape[-1])
        mean_fixed = entity_mean.unsqueeze(-2).expand(*entity_mean.shape[:-1], self.n_fixed, 1)
        std_fixed = entity_std.unsqueeze(-2).expand(*entity_std.shape[:-1], self.n_fixed, 1)
        fixed_in = torch.cat(
            [
                x_fixed.unsqueeze(-1),
                te_fixed,
                weekly_fixed,
                mean_fixed,
                std_fixed,
                slot_emb.to(x.dtype),
                trajectory_context[..., 1 + n_entities : 1 + n_entities + self.n_fixed, :],
            ],
            dim=-1,
        )
        v_fixed = self.fixed_net(fixed_in).squeeze(-1)
        if not self.graph_router:
            return torch.cat([v_base, v_per_entity.squeeze(-1), v_fixed], dim=-1)

        # Node bank follows the edge layout: channels, controls, demand, baseline, sales.
        n_slots = layout.n_slots
        lead = x.shape[:-1]
        demand_repr = self.demand_node_embed.weight[: self.n_demand].expand(
            *lead, self.n_demand, entity_repr.shape[-1]
        )
        baseline_repr = self.baseline_node_embed.weight.expand(*lead, 1, entity_repr.shape[-1])
        sales_repr = self.sales_node_proj(weekly_repr).unsqueeze(-2)
        node_bank = torch.cat(
            [entity_repr, demand_repr.to(x.dtype), baseline_repr.to(x.dtype), sales_repr], dim=-2
        )
        zeros_j = x.new_zeros(*lead, self.n_demand)
        zeros_y = x.new_zeros(*lead, 1)
        state_bank = torch.cat([x_entity, zeros_j, x_base, zeros_y], dim=-1)
        entity_context = trajectory_context[..., 1 : 1 + n_entities, :]
        zeros_j_context = x.new_zeros(*lead, self.n_demand, _TRAJECTORY_EMBED_DIM)
        zeros_y_context = x.new_zeros(*lead, 1, _TRAJECTORY_EMBED_DIM)
        context_bank = torch.cat(
            [entity_context, zeros_j_context, trajectory_context[..., :1, :], zeros_y_context],
            dim=-2,
        )
        src_idx, dst_idx, src_type_idx, dst_type_idx = self._edge_layout_tensors(layout, x.device)
        src_repr = node_bank[..., src_idx, :]
        dst_repr = node_bank[..., dst_idx, :]
        src_state = state_bank[..., src_idx].unsqueeze(-1)
        dst_state = state_bank[..., dst_idx].unsqueeze(-1)
        src_context = context_bank[..., src_idx, :]
        dst_context = context_bank[..., dst_idx, :]
        src_type = self.node_type_embed(src_type_idx)
        dst_type = self.node_type_embed(dst_type_idx)
        te_edge = te.unsqueeze(-2).expand(*te.shape[:-1], n_slots, te.shape[-1])
        weekly_edge = weekly_repr.unsqueeze(-2).expand(
            *weekly_repr.shape[:-1], n_slots, weekly_repr.shape[-1]
        )
        edge_inputs = [
            x_edges.unsqueeze(-1),
            src_state,
            dst_state,
            te_edge,
            weekly_edge,
            src_repr,
            dst_repr,
            src_type.expand(*lead, n_slots, _ENTITY_EMBED_DIM).to(x.dtype),
            dst_type.expand(*lead, n_slots, _ENTITY_EMBED_DIM).to(x.dtype),
        ]
        edge_inputs.extend(
            [
                trajectory_context[..., 1 + n_entities + self.n_fixed :, :],
                src_context,
                dst_context,
            ]
        )
        edge_in = torch.cat(edge_inputs, dim=-1)
        v_edges = self.edge_net(edge_in).squeeze(-1)
        return torch.cat([v_base, v_per_entity.squeeze(-1), v_fixed, v_edges], dim=-1)

    def _trajectory_loadings(self, weekly_repr: Tensor) -> Tensor:
        """Return row-unit week loadings with an identifiable constant basis."""
        learned = self.trajectory_loading_net(weekly_repr)
        centered = learned - learned.mean(dim=-2, keepdim=True)
        eps = torch.finfo(centered.dtype).eps
        # Masking sqrt(0) afterwards still produces NaN gradients. Floor before
        # sqrt; below-threshold loadings remain zero under the existing mask.
        rms = centered.square().mean(dim=-2, keepdim=True).clamp_min(eps * eps).sqrt()
        valid_rms = torch.isfinite(rms) & (rms > eps)
        safe_centered = torch.where(torch.isfinite(centered), centered, torch.zeros_like(centered))
        safe_rms = torch.where(valid_rms, rms, torch.ones_like(rms))
        learned = torch.where(
            valid_rms,
            safe_centered / safe_rms,
            torch.zeros_like(safe_centered),
        )
        constant = torch.full_like(learned[..., :1], _TRAJECTORY_CONSTANT_LOADING_SCALE)
        return F.normalize(torch.cat([constant, learned], dim=-1), dim=-1)

    def _base_noise(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        weekly_repr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Draw full-rank noise and its shared rank-four trajectory component.

        Unit-norm week loadings preserve marginal unit variance; the white
        component keeps the complete base covariance full rank. Draw order is
        white noise first, then one latent per task, coordinate, and rank.
        """
        white, z = self._draw_noise(shape, device=device, dtype=dtype)
        return self._mix_noise(white, z, weekly_repr)

    @staticmethod
    def _draw_noise(
        shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """The random part of :meth:`_base_noise`: white noise, then rank-four latents."""
        white = torch.randn(shape, device=device, dtype=dtype)
        z = torch.randn((*shape[:-2], shape[-1], _TRAJECTORY_RANK), device=device, dtype=dtype)
        return white, z

    def _mix_noise(self, white: Tensor, z: Tensor, weekly_repr: Tensor) -> tuple[Tensor, Tensor]:
        """The deterministic part of :meth:`_base_noise`: load latents onto weeks and mix."""
        shape = tuple(white.shape)
        if weekly_repr.shape[-2] != shape[-2]:
            raise ValueError(
                "weekly_repr and flow state must share the query-week axis, "
                f"got {tuple(weekly_repr.shape)} and {shape}"
            )
        try:
            torch.broadcast_shapes(weekly_repr.shape[:-2], shape[:-2])
        except RuntimeError as error:
            raise ValueError(
                "weekly_repr must broadcast over the flow state's leading axes, "
                f"got {tuple(weekly_repr.shape)} and {shape}"
            ) from error
        loadings = self._trajectory_loadings(weekly_repr)
        latent = torch.matmul(loadings, z.transpose(-1, -2))
        rho = FLOW_TRAJECTORY_CORRELATION
        x0 = math.sqrt(rho) * latent + math.sqrt(1.0 - rho) * white
        return x0, latent

    def zero_atom_gate_slices(
        self, n_channels: int, n_controls: int
    ) -> tuple[tuple[ZeroAtomGate, slice, slice], ...]:
        """Resolve contribution and absolute edge slices for the router's gates."""
        if not self.graph_router:
            return ()
        edge_slices = _edge_slot_layout(n_channels, n_controls, self.n_demand).slices
        n_decomposition = 1 + n_channels + n_controls + self.n_fixed
        fixed_start = 1 + n_channels + n_controls
        column_slices = {
            "contribution": slice(1, 1 + n_channels),
            "control_contribution": slice(1 + n_channels, fixed_start),
        }
        for offset, column in enumerate(DECOMPOSITION_FIXED_TARGETS):
            column_slices[column] = slice(fixed_start + offset, fixed_start + offset + 1)
        return tuple(
            (
                gate,
                column_slices[gate.column],
                slice(
                    n_decomposition + edge_slices[gate.family].start,
                    n_decomposition + edge_slices[gate.family].stop,
                ),
            )
            for gate in ZERO_ATOM_GATES
        )

    def project_zero_atom(
        self,
        raw_state: Tensor,
        *,
        n_channels: int,
        n_controls: int | None = None,
        state_mask: Tensor | None = None,
    ) -> Tensor:
        """Project graph-off contributions to zero without mutating the input.

        Gates decode the sign of each draw's mean graph coordinate across weeks,
        exactly as the binary graph decoder. Aggregate gates reduce over every
        slot in their family, including all ordered channel pairs in cc.
        """
        n_controls_eff = self.n_controls if n_controls is None else int(n_controls)
        gates = self.zero_atom_gate_slices(n_channels, n_controls_eff)
        if not gates:
            raise ValueError("zero-atom projection requires a joint graph router state")
        expected_dim = self.state_dim(n_channels, n_controls_eff)
        if raw_state.shape[-1] != expected_dim:
            raise ValueError(
                "raw_state does not match the requested zero-atom geometry: "
                f"expected {expected_dim} state coordinates for {n_channels} channels "
                f"and {n_controls_eff} controls, got {raw_state.shape[-1]}"
            )
        mask = (
            None
            if state_mask is None
            else torch.as_tensor(state_mask, device=raw_state.device, dtype=raw_state.dtype)
        )
        raw_zero = raw_state.new_zeros(())
        projected = raw_state.clone()
        for gate, contribution_slice, edge_slice in gates:
            family_state = raw_state[..., edge_slice]
            contribution = raw_state[..., contribution_slice]
            if gate.reduction == "elementwise" and contribution.shape[-1] != family_state.shape[-1]:
                raise ValueError(
                    "raw_state does not match the requested zero-atom geometry: "
                    f"expected equal {gate.column} and {gate.family} widths, "
                    f"got {contribution.shape[-1]} and {family_state.shape[-1]}"
                )
            live = family_state.mean(dim=-2, keepdim=True) > 0.0
            if mask is not None:
                edge_active = mask[..., edge_slice]
                try:
                    edge_active = torch.broadcast_to(edge_active, family_state.shape)
                except RuntimeError as error:
                    raise ValueError(
                        f"state_mask {gate.family} coordinates must broadcast to the raw "
                        f"{gate.family} state, got {tuple(edge_active.shape)} "
                        f"for {tuple(family_state.shape)}"
                    ) from error
                live = live & (edge_active > 0.0)
            if gate.reduction == "elementwise":
                atom = ~live.expand_as(family_state)
            else:
                # Empty families have no live edge and therefore force the atom.
                atom = ~live.any(dim=-1, keepdim=True)
            projected[..., contribution_slice] = torch.where(atom, raw_zero, contribution)
        return projected

    def _family_normalized_flow_mask(
        self,
        effective_mask: Tensor,
        *,
        n_channels: int,
        n_controls: int,
    ) -> Tensor:
        """Give decomposition and each graph family unit mass per task.

        Normalizing each task separately prevents active-slot counts in other
        tasks from changing its coordinate weights. Empty groups stay empty.
        """
        n_decomposition = 1 + n_channels + n_controls + self.n_fixed
        layout = _edge_slot_layout(n_channels, n_controls, self.n_demand)
        groups = [slice(0, n_decomposition)]
        groups.extend(
            slice(
                n_decomposition + layout.slices[family].start,
                n_decomposition + layout.slices[family].stop,
            )
            for family in STRUCT_EDGE_KEYS
        )
        for coordinate_slice in groups:
            weights = effective_mask[..., coordinate_slice]
            denominator = weights.sum(dim=(-2, -1), keepdim=True)
            scale = torch.where(
                denominator > 0.0,
                denominator.clamp_min(torch.finfo(weights.dtype).tiny).reciprocal(),
                torch.ones_like(denominator),
            )
            weights.mul_(scale)
        return effective_mask

    def _decomposition_width(self, n_channels: int, n_controls: int) -> int:
        return 1 + n_channels + n_controls + self.n_fixed

    @staticmethod
    def _time_grid(n_steps: int) -> list[tuple[float, float, float]]:
        """Uniform fixed-grid steps, retaining the trained sampler's arithmetic."""
        dt = 1.0 / n_steps
        return [(i * dt, dt, (i + 1) * dt) for i in range(n_steps)]

    def loss(
        self,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        targets_z: Tensor,
        control_repr: Tensor | None = None,
        state_mask: Tensor | None = None,
        *,
        graph_loss_weight: float | None = None,
    ) -> Tensor:
        """Flow-matching loss with shared, stratified time across each trajectory.

        The router normalizes its decomposition and graph families separately;
        the conditional decoder uses active-coordinate MSE. An explicit graph
        weight gives ``L_dec + graph_loss_weight * L_graph``, preserving the
        decomposition coefficient for context reconstruction.
        """
        n_channels = channel_repr.shape[-2]
        n_controls = control_repr.shape[-2] if control_repr is not None else self.n_controls
        graph_budget = None if graph_loss_weight is None else float(graph_loss_weight)
        if graph_budget is not None and (not math.isfinite(graph_budget) or graph_budget < 0.0):
            raise ValueError(
                f"graph_loss_weight must be finite and nonnegative, got {graph_budget}"
            )
        x1 = targets_z
        x0, trajectory_latent = self._base_noise(
            tuple(x1.shape),
            device=x1.device,
            dtype=x1.dtype,
            weekly_repr=weekly_repr,
        )
        base_mask = torch.ones_like(x1)
        if state_mask is not None:
            base_mask = state_mask.to(x1.dtype).expand_as(x1)
            x1 = x1 * base_mask
            x0 = x0 * base_mask
            trajectory_latent = trajectory_latent * base_mask
        n_tasks = math.prod(x1.shape[:-2])
        strata = torch.arange(n_tasks, device=x1.device, dtype=x1.dtype)
        t_task = (strata + torch.rand(n_tasks, device=x1.device, dtype=x1.dtype)) / n_tasks
        t_task = t_task[torch.randperm(n_tasks, device=x1.device)]
        t = t_task.reshape(*x1.shape[:-2], 1).expand(*x1.shape[:-1])
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target_velocity = x1 - x0
        pred_velocity = self.velocity(
            xt,
            t,
            weekly_repr,
            channel_repr,
            control_repr,
            trajectory_latent,
            base_mask,
        )
        effective_mask = base_mask.clone()
        if self.graph_router:
            effective_mask = self._family_normalized_flow_mask(
                effective_mask, n_channels=n_channels, n_controls=n_controls
            )
        if graph_budget is not None:
            if not self.graph_router:
                raise ValueError("graph_loss_weight requires a joint decomposition graph state")
            n_dec = self._decomposition_width(n_channels, n_controls)
            denominator = effective_mask[..., :n_dec].sum()
            graph_weights = effective_mask[..., n_dec:]
            graph_mass = graph_weights.sum()
            if float(graph_mass.detach()) <= 0.0:
                raise ValueError("graph_loss_weight requires active graph coordinates")
            graph_weights.mul_(graph_budget * denominator / graph_mass)
        else:
            denominator = effective_mask.sum()
        sq_err = (pred_velocity - target_velocity).square()
        numerator = (sq_err * effective_mask).sum()
        if float(denominator.detach().item()) <= 0.0:
            raise ValueError("flow objective has no active state coordinates")
        return numerator / denominator

    def noise_shapes(
        self, lead: tuple[int, ...], n_channels: int, n_controls: int | None, n_samples: int
    ) -> list[tuple[int, ...]]:
        """Shapes of the base-noise chunks :meth:`sample` draws, in draw order.

        ``lead`` is the representations' ``weekly_repr.shape[:-1]``.
        """
        s = _sample_count(n_samples)
        d = self.state_dim(n_channels, n_controls)
        chunk = _FLOW_SAMPLE_CHUNK if self.graph_router else s
        return [(min(chunk, s - start), *lead, d) for start in range(0, s, chunk)]

    @torch.no_grad()
    def sample(
        self,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        n_samples: int = DEFAULT_N_SAMPLES,
        control_repr: Tensor | None = None,
        state_mask: Tensor | None = None,
        apply_zero_atom: bool = True,
        *,
        noise: _NoiseRequest | None = None,
        graph_only: bool = False,
    ) -> Tensor:
        """Draw decomposition trajectories and, for the router, full graph states.

        Each draw integrates ``FLOW_STEPS_SAMPLE`` uniform Euler steps. The router
        draws base noise in chunks of ``_FLOW_SAMPLE_CHUNK`` draws; seeded draws
        depend on that chunking but not on how draws are batched for transport.
        ``noise`` optionally supplies those chunks, prefetched for :meth:`noise_shapes`.
        With ``graph_only`` only the graph coordinates are draws: the decomposition
        coordinates skip work no graph coordinate depends on (their final Euler step on
        the GPU, the support rules), so they are left unfinished.
        """
        s = _sample_count(n_samples)
        shapes = self.noise_shapes(
            tuple(weekly_repr.shape[:-1]),
            channel_repr.shape[-2],
            control_repr.shape[-2] if control_repr is not None else None,
            s,
        )
        device, noise_device = weekly_repr.device, _noise_device(weekly_repr.device)
        if noise is not None and (
            noise_device == device
            or noise.shapes != tuple(shapes)
            or noise.dtype != weekly_repr.dtype
        ):
            raise ValueError("prefetched noise does not match the sampled draws")
        if noise_device == device:
            drawn = [
                self._draw_noise(shape, device=device, dtype=weekly_repr.dtype) for shape in shapes
            ]
            x, trajectory_latent = self._mix_noise(
                torch.cat([white for white, _ in drawn]),
                torch.cat([z for _, z in drawn]),
                weekly_repr,
            )

            def draws(start: int, stop: int) -> tuple[Tensor, Tensor]:
                return x[start:stop], trajectory_latent[start:stop]

        else:
            if noise is None:
                uniform = _metal.uniform_normals()
                chunks = (_draw_host_noise(shape, weekly_repr.dtype, uniform) for shape in shapes)
            else:
                chunks = iter(noise)
            draws = _OrderedNoise(self, chunks, weekly_repr)
        return self._transport_draws(
            draws,
            s,
            weekly_repr,
            channel_repr,
            control_repr,
            state_mask,
            apply_zero_atom=apply_zero_atom,
            graph_only=graph_only,
        )

    def _transport(
        self,
        x: Tensor,
        trajectory_latent: Tensor,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor | None,
        state_mask: Tensor | None,
        *,
        apply_zero_atom: bool,
        representation_index: Tensor | None = None,
    ) -> Tensor:
        """Transport base draws ``x[S, ..., T, D]`` and apply the sampler's support rules.

        ``representation_index`` optionally maps the batch axis of ``x`` to rows of the
        representations and ``state_mask``, given once per distinct conditioning.
        """
        return self._transport_draws(
            lambda start, stop: (x[start:stop], trajectory_latent[start:stop]),
            x.shape[0],
            weekly_repr,
            channel_repr,
            control_repr,
            state_mask,
            apply_zero_atom=apply_zero_atom,
            representation_index=representation_index,
        )

    def _transport_draws(
        self,
        draws: Callable[[int, int], tuple[Tensor, Tensor]],
        s: int,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor | None,
        state_mask: Tensor | None,
        *,
        apply_zero_atom: bool,
        representation_index: Tensor | None = None,
        graph_only: bool = False,
    ) -> Tensor:
        """Transport ``s`` base draws and apply the sampler's support rules.

        ``draws(start, stop)`` returns the base state and trajectory latent of draws
        ``[start, stop)``, ``[stop - start, ..., T, D]``. Draws are integrated in batches
        of about ``_FLOW_SAMPLE_ROWS`` evaluated state rows. Every draw is integrated
        independently, so batching bounds memory without changing any draw.
        """
        c = channel_repr.shape[-2]
        n_controls = control_repr.shape[-2] if control_repr is not None else self.n_controls
        device = weekly_repr.device
        if device.type == "mps" and torch.__version__ < _metal.MIN_TORCH:
            raise RuntimeError(
                f"sampling on an Apple GPU (MPS) needs PyTorch 2.9 or newer, found "
                f"{torch.__version__}: older MPS releases miscompute or lack operations this "
                "sampler uses. Upgrade PyTorch or sample on the CPU."
            )
        weight = None if state_mask is None else state_mask.to(weekly_repr.dtype)
        host_weight = None
        if weight is not None and weight.device != device:
            # A host state mask decides the live coordinates and support rules without
            # waiting for the device, and its copy to the device does not wait either.
            host_weight, weight = weight, _to_device(weight, device)
        if representation_index is not None and (
            weight is None
            or self._compiled_velocity is not None
            or not _metal.supported(self, weight)
        ):
            # Only the GPU kernels read per-conditioning rows; give the rest one row per draw.
            weekly_repr, channel_repr, control_repr, weight = (
                None if value is None else value[representation_index]
                for value in (weekly_repr, channel_repr, control_repr, weight)
            )
            representation_index = host_weight = None
        if representation_index is None:
            lead = weekly_repr.shape[:-1]
            draw_weight = weight
        else:
            lead = (representation_index.numel(), *weekly_repr.shape[1:-1])
            draw_weight = weight[representation_index]
        if draw_weight is not None and not graph_only:
            # Decide the support rules before transport: reading a GPU tensor here would
            # wait for every batch queued below. A host mask has a row for every draw's.
            channel_active = draw_weight[..., 1 : 1 + c] > 0.0
            control_slice = slice(1 + c, 1 + c + n_controls)
            control_active = draw_weight[..., control_slice] > 0.0
            if host_weight is not None:
                inactive_controls = bool((~(host_weight[..., control_slice] > 0.0)).any())
            else:
                inactive_controls = bool((~control_active).any())

        grid = self._time_grid(FLOW_STEPS_SAMPLE)
        if weight is None:

            def flow(start: int, stop: int) -> Tensor:
                x, latent = draws(start, stop)

                def expand(value: Tensor | None) -> Tensor | None:
                    if value is None:
                        return None
                    return value.unsqueeze(0).expand(x.shape[0], *value.shape)

                weekly, channels, controls = map(expand, (weekly_repr, channel_repr, control_repr))

                def drift(x: Tensor, t_val: float) -> Tensor:
                    t = torch.full(x.shape[:-1], t_val, device=x.device, dtype=x.dtype)
                    return self.velocity(x, t, weekly, channels, controls, latent)

                return integrate_flow(x, drift, grid)

            width = self.state_dim(c, n_controls)
        else:
            live = _LiveVelocity(self, weekly_repr, channel_repr, control_repr, weight, host_weight)
            # With the GPU kernels, ordered host noise is mixed for the live columns only.
            columns = live.index if live.metal and isinstance(draws, _OrderedNoise) else None

            def flow(start: int, stop: int) -> Tensor:
                if columns is None:
                    x, latent = draws(start, stop)
                    return live.integrate(
                        x * draw_weight,
                        latent * draw_weight,
                        grid,
                        representation_index,
                        graph_only,
                    )
                x, latent = draws(start, stop, columns)
                return live.integrate_live(
                    x * live.weight, latent * live.weight, grid, representation_index, graph_only
                )

            width = live.index.numel()
        x = torch.cat(
            _map_batches(flow, s, max(1, _sample_rows(device) // (math.prod(lead) * width)), device)
        )
        if graph_only:
            return x
        if draw_weight is not None:
            # Direct contributions are nonnegative; padding uses the same natural zero.
            contribution = x[..., 1 : 1 + c].clamp_min(0.0)
            contribution = torch.where(channel_active, contribution, contribution.new_zeros(()))
            x = torch.cat([x[..., :1], contribution, x[..., 1 + c :]], dim=-1)
            if inactive_controls:
                controls_zeroed = torch.where(
                    control_active,
                    x[..., control_slice],
                    torch.zeros_like(x[..., control_slice]),
                )
                x = torch.cat(
                    [
                        x[..., : control_slice.start],
                        controls_zeroed,
                        x[..., control_slice.stop :],
                    ],
                    dim=-1,
                )
        if self.graph_router and apply_zero_atom:
            x = self.project_zero_atom(
                x,
                n_channels=c,
                n_controls=n_controls,
                state_mask=draw_weight,
            )
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _input_blocks(layer: nn.Linear, widths: dict[str, int]) -> dict[str, Tensor]:
    """Split a first layer's weight by the blocks of its concatenated input, in order."""
    if sum(widths.values()) != layer.in_features:
        raise ValueError(f"input blocks {widths} do not sum to {layer.in_features} features")
    return dict(zip(widths, layer.weight.split(list(widths.values()), dim=1), strict=True))


class _LiveNetwork:
    """One per-coordinate network whose step-invariant first-layer blocks are precomputed.

    ``static`` holds bias plus the projection of every input fixed during
    integration, ``[..., T, rows, hidden]``. At each step only the flow-state
    scalars (``scalar_index`` into the scalar bank), the week-pooled trajectory
    contexts (``context_index`` into the context bank) and the flow time are projected.
    """

    def __init__(
        self,
        net: nn.Sequential,
        static: Tensor,
        *,
        time_weight: Tensor,
        scalar_weight: Tensor,
        scalar_index: Tensor,
        context_weight: Tensor,
        context_index: Tensor,
        chunk_rows: int | None = None,
    ):
        _, act, hidden_layer, act2, output_layer = net
        if not (isinstance(act, nn.SiLU) and isinstance(act2, nn.SiLU)):
            raise ValueError("coordinate networks must be Linear, SiLU, Linear, SiLU, Linear")
        # Bias added after the product: CPU addmm with a broadcast bias is markedly slower.
        self.hidden_weight = hidden_layer.weight.t()
        self.hidden_bias = hidden_layer.bias
        self.output_weight = output_layer.weight[0]
        self.output_bias = output_layer.bias
        self.static = static
        self.time_weight = time_weight
        self.scalar_weight = scalar_weight
        self.scalar_index = scalar_index
        self.context_weight = context_weight
        self.context_index = context_index
        self.rows = scalar_index.shape[0]
        #: Rows per chunk of the hidden and output layers; ``None`` runs them in one call.
        self.chunk_rows = chunk_rows

    def __call__(self, scalars: Tensor, contexts: Tensor, time: Tensor) -> Tensor:
        if self.rows == 0:
            return scalars.new_zeros(*scalars.shape[:-1], 0)
        state = scalars.index_select(-1, self.scalar_index.flatten())
        state = state.unflatten(-1, tuple(self.scalar_index.shape))
        context = contexts.index_select(-2, self.context_index.flatten())
        context = context.unflatten(-2, tuple(self.context_index.shape)).flatten(-2)
        hidden = F.linear(state, self.scalar_weight)
        hidden += self.static
        hidden += F.linear(context, self.context_weight, F.linear(time, self.time_weight))
        hidden = hidden.view(-1, hidden.shape[-1])
        chunk = self.chunk_rows
        if chunk is None or hidden.shape[0] <= chunk:
            hidden = F.silu(hidden, inplace=True)
            hidden = torch.mm(hidden, self.hidden_weight).add_(self.hidden_bias)
            hidden = F.silu(hidden, inplace=True)
            output = torch.mv(hidden, self.output_weight)
        else:
            output = hidden.new_empty(hidden.shape[0])
            for start in range(0, hidden.shape[0], chunk):
                # The first layer's SiLU too, while the product is about to read the chunk.
                part = F.silu(hidden[start : start + chunk], inplace=True)
                part = torch.mm(part, self.hidden_weight)
                part = F.silu(part.add_(self.hidden_bias), inplace=True)
                torch.mv(part, self.output_weight, out=output[start : start + chunk])
        return output.add_(self.output_bias).view(state.shape[:-1])


class _LiveVelocity:
    """Sampling-time :meth:`JointFlowHead.velocity` restricted to live coordinates.

    Equal to ``velocity`` on every coordinate that is live in some task, up to
    float round-off. Every network acts on one coordinate at a time, entity
    moments are mask-weighted and each coordinate's trajectory context pools only
    its own weeks, so a coordinate masked in every task cannot reach a live one.
    The sampler projects such coordinates to zero after every step; their
    velocity is never evaluated. Each network's first layer is affine in its
    concatenated input, so the blocks fixed during integration (week, node and
    type representations, the trajectory latent) are projected once per call and
    only flow-state inputs are projected at each step.
    """

    def __init__(
        self,
        head: JointFlowHead,
        weekly_repr: Tensor,
        channel_repr: Tensor,
        control_repr: Tensor | None,
        weight: Tensor,
        host_weight: Tensor | None = None,
    ):
        entity_repr = head._entity_repr(channel_repr, control_repr)
        self.compiled = head._compiled_velocity
        self.metal = self.compiled is None and _metal.supported(head, weight)
        device = weekly_repr.device
        # Eager CPU sampling chunks the networks' hidden layers (see _HIDDEN_CHUNK_ROWS);
        # compiled sampling traces one call per layer.
        chunk_rows = _HIDDEN_CHUNK_ROWS if self.compiled is None and device.type == "cpu" else None
        n_channels, n_entities = channel_repr.shape[-2], entity_repr.shape[-2]
        n_fixed = head.n_fixed
        n_decomposition = 1 + n_entities + n_fixed
        d = weight.shape[-1]
        #: Full flow-state width.
        self.width = d
        layout = None
        if head.graph_router:
            layout = _edge_slot_layout(n_channels, n_entities - n_channels, head.n_demand)
            if d - n_decomposition != layout.n_slots:
                raise ValueError(
                    f"joint graph state carries {d - n_decomposition} edge columns but the "
                    f"layout for K={n_channels}, M={n_entities - n_channels}, "
                    f"J={head.n_demand} has {layout.n_slots} slots"
                )
        entity_mask = weight[..., 1 : 1 + n_entities]
        try:
            torch.broadcast_shapes(entity_mask.shape, (*weekly_repr.shape[:-1], n_entities))
        except RuntimeError as error:
            raise ValueError(
                "state_mask entity prefix does not broadcast to the flow state: "
                f"{tuple(weight.shape)} versus {tuple(weekly_repr.shape[:-1])}"
            ) from error
        entity_count = entity_mask.sum(dim=-1, keepdim=True)
        # Live state order: baseline, then live entities, fixed slots and edge slots. The
        # host decides which coordinates are live from the caller's host copy of the mask,
        # if any, else from one copy of these small masks.
        decided = weight if host_weight is None else host_weight
        live = (decided != 0).expand(*decided.shape[:-1], d).reshape(-1, d).any(dim=0)
        if host_weight is None:
            empty = (entity_count == 0).any()
        else:
            empty = (host_weight[..., 1 : 1 + n_entities].sum(dim=-1) == 0).any()
        host = torch.cat([live, empty.reshape(1)]).cpu()
        if head.graph_router:
            if host[-1]:
                raise ValueError("flow state has a task/week with no active entities")
        else:
            # An all-off conditioning graph leaves only baseline stochastic.
            entity_count = entity_count.clamp_min(1)
        live = host[:-1]
        live_entities = torch.nonzero(live[1 : 1 + n_entities]).squeeze(-1)
        live_fixed = torch.nonzero(live[1 + n_entities : n_decomposition]).squeeze(-1)
        live_slots = torch.nonzero(live[n_decomposition:]).squeeze(-1)
        widths = (1, live_entities.numel(), live_fixed.numel(), live_slots.numel())
        self.index = _to_device(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.long),
                    1 + live_entities,
                    1 + n_entities + live_fixed,
                    n_decomposition + live_slots,
                ]
            ),
            device,
        )
        _, live_entities, live_fixed, live_slots = self.index.split(widths)
        live_entities = live_entities - 1
        live_fixed = live_fixed - (1 + n_entities)
        live_slots = live_slots - n_decomposition
        n_live = self.index.numel()
        n_live_entities = live_entities.numel()
        #: Live columns before the graph slots (baseline, entities, fixed totals).
        self.decomposition_width = n_live - live_slots.numel()
        # Bank positions: a shared zero column after the live state, then the moments.
        zero, mean, std = n_live, n_live + 1, n_live + 2
        position = torch.full((d,), zero, dtype=torch.long, device=device)
        position[self.index] = torch.arange(n_live, device=device)

        self.weight = weight.index_select(-1, self.index)
        self.entity_active = weight.index_select(-1, 1 + live_entities)
        self.entity_count = entity_count
        self.entity_count_sqrt = entity_count.sqrt()
        self.context_mask = self.weight.unsqueeze(-1)

        state_weight = head.trajectory_state_proj.weight
        self.token_state_weight = state_weight[:, 0]
        self.token_latent_weight = state_weight[:, 1]
        self.token_base = head.trajectory_state_proj.bias + head.trajectory_week_proj(
            weekly_repr
        ).unsqueeze(-2)
        # Tokens are Linear(SiLU(.)): score each week from SiLU(.) directly and apply
        # the Linear after pooling, since the softmax weights over weeks sum to one.
        _, token_layer = head.trajectory_token_net
        score_layer = head.trajectory_score
        self.token_layer = token_layer
        self.score_weight = token_layer.weight.t() @ score_layer.weight[0]
        self.score_bias = score_layer.weight[0] @ token_layer.bias + score_layer.bias

        t_dim = 1 + 2 * FLOW_N_FREQ
        d_model = weekly_repr.shape[-1]
        d_node = entity_repr.shape[-1]
        embed = _TRAJECTORY_EMBED_DIM

        def week_static(layer: nn.Linear, week_weight: Tensor) -> Tensor:
            return F.linear(weekly_repr, week_weight, layer.bias).unsqueeze(-2)

        def columns(rows: int, *values: int | Tensor) -> Tensor:
            """Bank positions ``[rows, len(values)]``; an integer repeats down its column."""
            return torch.stack(
                [
                    value
                    if isinstance(value, Tensor)
                    else torch.full((rows,), value, dtype=torch.long, device=device)
                    for value in values
                ],
                dim=-1,
            )

        base_layer = head.base_net[0]
        base = _input_blocks(
            base_layer,
            {"x": 1, "time": t_dim, "week": d_model, "mean": 1, "std": 1, "context": embed},
        )
        networks = [
            _LiveNetwork(
                head.base_net,
                week_static(base_layer, base["week"]),
                time_weight=base["time"],
                scalar_weight=torch.cat([base["x"], base["mean"], base["std"]], dim=1),
                scalar_index=columns(1, 0, mean, std),
                context_weight=base["context"],
                context_index=columns(1, 0),
                chunk_rows=chunk_rows,
            )
        ]

        # Entities: a masked node representation and a channel/control type.
        masked_repr = entity_repr * entity_mask.unsqueeze(-1).to(entity_repr.dtype)
        type_ids = torch.cat(
            [
                torch.full((n_channels,), ENTITY_CHANNEL, dtype=torch.long, device=device),
                torch.full(
                    (n_entities - n_channels,), ENTITY_CONTROL, dtype=torch.long, device=device
                ),
            ]
        )
        contrib_layer = head.contrib_net[0]
        contrib_widths = {
            "x": 1,
            "time": t_dim,
            "week": d_model,
            "repr": d_node,
            "base": 1,
            "mean": 1,
            "std": 1,
            "type": _ENTITY_EMBED_DIM,
        }
        if layout is not None:
            contrib_widths["own_edge"] = 1
        contrib_widths["context"] = embed
        if layout is not None:
            contrib_widths["edge_context"] = embed
        contrib = _input_blocks(contrib_layer, contrib_widths)
        entity_static = (
            week_static(contrib_layer, contrib["week"])
            + F.linear(masked_repr.index_select(-2, live_entities), contrib["repr"])
            + F.linear(head.entity_type_embed(type_ids[live_entities]), contrib["type"])
        )
        entity_positions = position[1 + live_entities]
        scalar_columns = [entity_positions, 0, mean, std]
        context_columns = [entity_positions]
        scalar_blocks = [contrib["x"], contrib["base"], contrib["mean"], contrib["std"]]
        context_blocks = [contrib["context"]]
        if layout is not None:
            # Each entity also reads its own outcome edge: cy for channels, zb for controls.
            cy, zb = layout.slices["cy"], layout.slices["zb"]
            own_slot = torch.cat(
                [
                    torch.arange(cy.start, cy.stop, device=device),
                    torch.arange(zb.start, zb.stop, device=device),
                ]
            )
            own_positions = position[n_decomposition + own_slot[live_entities]]
            scalar_columns.append(own_positions)
            context_columns.append(own_positions)
            scalar_blocks.append(contrib["own_edge"])
            context_blocks.append(contrib["edge_context"])
        networks.append(
            _LiveNetwork(
                head.contrib_net,
                entity_static,
                time_weight=contrib["time"],
                scalar_weight=torch.cat(scalar_blocks, dim=1),
                scalar_index=columns(n_live_entities, *scalar_columns),
                context_weight=torch.cat(context_blocks, dim=1),
                context_index=columns(n_live_entities, *context_columns),
                chunk_rows=chunk_rows,
            )
        )

        fixed_layer = head.fixed_net[0]
        fixed = _input_blocks(
            fixed_layer,
            {
                "x": 1,
                "time": t_dim,
                "week": d_model,
                "mean": 1,
                "std": 1,
                "slot": _ENTITY_EMBED_DIM,
                "context": embed,
            },
        )
        fixed_positions = position[1 + n_entities + live_fixed]
        networks.append(
            _LiveNetwork(
                head.fixed_net,
                week_static(fixed_layer, fixed["week"])
                + F.linear(head.fixed_slot_embed.weight[live_fixed], fixed["slot"]),
                time_weight=fixed["time"],
                scalar_weight=torch.cat([fixed["x"], fixed["mean"], fixed["std"]], dim=1),
                scalar_index=columns(live_fixed.numel(), fixed_positions, mean, std),
                context_weight=fixed["context"],
                context_index=columns(live_fixed.numel(), fixed_positions),
                chunk_rows=chunk_rows,
            )
        )

        if layout is not None:
            edge_layer = head.edge_net[0]
            edge = _input_blocks(
                edge_layer,
                {
                    "x": 1,
                    "src_state": 1,
                    "dst_state": 1,
                    "time": t_dim,
                    "week": d_model,
                    "src": d_node,
                    "dst": d_node,
                    "src_type": _ENTITY_EMBED_DIM,
                    "dst_type": _ENTITY_EMBED_DIM,
                    "context": embed,
                    "src_context": embed,
                    "dst_context": embed,
                },
            )
            # Node bank follows the edge layout: channels, controls, demand, baseline, sales.
            lead = weekly_repr.shape[:-1]
            node_bank = torch.cat(
                [
                    masked_repr.expand(*lead, n_entities, d_node),
                    head.demand_node_embed.weight[: head.n_demand].expand(
                        *lead, head.n_demand, d_node
                    ),
                    head.baseline_node_embed.weight.expand(*lead, 1, d_node),
                    head.sales_node_proj(weekly_repr).unsqueeze(-2),
                ],
                dim=-2,
            )
            src_index, dst_index, src_type, dst_type = head._edge_layout_tensors(layout, device)
            src_index, dst_index = src_index[live_slots], dst_index[live_slots]
            # Node states: entities and baseline carry flow coordinates; demand and sales are zero.
            node_coordinate = torch.cat(
                [
                    position[1 : 1 + n_entities],
                    torch.full((head.n_demand,), zero, dtype=torch.long, device=device),
                    position[:1],
                    torch.full((1,), zero, dtype=torch.long, device=device),
                ]
            )
            slot_rows = torch.stack(
                [
                    position[n_decomposition + live_slots],
                    node_coordinate[src_index],
                    node_coordinate[dst_index],
                ],
                dim=-1,
            )
            networks.append(
                _LiveNetwork(
                    head.edge_net,
                    week_static(edge_layer, edge["week"])
                    + F.linear(node_bank, edge["src"]).index_select(-2, src_index)
                    + F.linear(node_bank, edge["dst"]).index_select(-2, dst_index)
                    + F.linear(head.node_type_embed(src_type[live_slots]), edge["src_type"])
                    + F.linear(head.node_type_embed(dst_type[live_slots]), edge["dst_type"]),
                    time_weight=edge["time"],
                    scalar_weight=torch.cat([edge["x"], edge["src_state"], edge["dst_state"]], 1),
                    scalar_index=slot_rows,
                    context_weight=torch.cat(
                        [edge["context"], edge["src_context"], edge["dst_context"]], dim=1
                    ),
                    context_index=slot_rows,
                    chunk_rows=chunk_rows,
                )
            )
        self.networks = networks

    def trajectory_offset(self, trajectory_latent: Tensor) -> Tensor:
        """Step-invariant part of the live trajectory tokens for one batch of draws."""
        return torch.addcmul(
            self.token_base,
            trajectory_latent.index_select(-1, self.index).unsqueeze(-1),
            self.token_latent_weight,
        )

    def velocity(self, x: Tensor, t_val: float, trajectory_offset: Tensor) -> Tensor:
        """Velocity of the live state ``x[..., index]`` at flow time ``t_val``."""
        return self._velocity(x, _fourier_time(x.new_full((), t_val)), trajectory_offset)

    def _velocity(self, x: Tensor, time: Tensor, trajectory_offset: Tensor) -> Tensor:
        """:meth:`velocity` at Fourier flow time ``time``, a tensor, as compilation needs."""
        hidden = F.silu(
            torch.addcmul(trajectory_offset, x.unsqueeze(-1), self.token_state_weight),
            inplace=True,
        )
        score = torch.matmul(hidden, self.score_weight).add_(self.score_bias)
        weights = torch.softmax(score, dim=-2).unsqueeze(-1)
        pooled = (weights * hidden).sum(dim=-3, keepdim=True)
        context = self.token_layer(pooled) * self.context_mask
        zero_context = context.new_zeros(*context.shape[:-2], 1, _TRAJECTORY_EMBED_DIM)
        contexts = torch.cat([context, zero_context], dim=-2)

        # Live entities follow the baseline. Their count comes from a shape, not a Python
        # constant, so compiled sampling does not recompile for each channel count.
        entity = x[..., 1 : 1 + self.entity_active.shape[-1]]
        mean = (entity * self.entity_active).sum(dim=-1, keepdim=True) / self.entity_count
        std = (
            torch.linalg.vector_norm((entity - mean) * self.entity_active, dim=-1, keepdim=True)
            / self.entity_count_sqrt
        )
        scalars = torch.cat([x, x.new_zeros(*x.shape[:-1], 1), mean, std], dim=-1)
        return torch.cat([network(scalars, contexts, time) for network in self.networks], dim=-1)

    def integrate(
        self,
        x: Tensor,
        trajectory_latent: Tensor,
        grid,
        index: Tensor | None = None,
        graph_only: bool = False,
    ) -> Tensor:
        """Euler-integrate masked ``x``; coordinates masked in every task stay zero.

        ``index`` maps the last lead axis of ``x`` to the representations' batch rows when
        those were given once per distinct conditioning (GPU kernels only). With
        ``graph_only`` the GPU kernels skip the decomposition networks' final step, whose
        result no graph coordinate reads.
        """
        weight = self.weight
        compiled = self.compiled
        state = x.index_select(-1, self.index)
        if self.metal:
            return self.integrate_live(
                state, trajectory_latent.index_select(-1, self.index), grid, index, graph_only
            )
        if index is not None:
            raise ValueError("indexed representations need the GPU sampling kernels")
        offset = self.trajectory_offset(trajectory_latent)
        # Flow times as tensors: a Python float would recompile at every Euler step.
        times = (
            {} if compiled is None else {t: _fourier_time(x.new_full((), t)) for t, _, _ in grid}
        )

        def drift(state: Tensor, t_val: float) -> Tensor:
            if compiled is None:
                return self.velocity(state, t_val, offset)
            return compiled(self, state, times[t_val], offset)

        live = integrate_flow(
            state,
            drift,
            grid,
            project=lambda state, _t_next: state * weight,
        )
        return x.new_zeros(x.shape).index_copy_(-1, self.index, live)

    def integrate_live(
        self,
        state: Tensor,
        trajectory_latent: Tensor,
        grid,
        index: Tensor | None = None,
        graph_only: bool = False,
    ) -> Tensor:
        """:meth:`integrate` with the GPU kernels, given only the live columns ``index``.

        Returns the full-width state, zero in coordinates masked in every task.
        """
        live = _metal.integrate(
            self,
            state,
            trajectory_latent,
            grid,
            index,
            final_from=self.decomposition_width if graph_only else 0,
        )
        return state.new_zeros(*state.shape[:-1], self.width).index_copy_(-1, self.index, live)
