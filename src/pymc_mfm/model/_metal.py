"""Apple GPU (MPS) kernels for the sampling-time live velocity.

On MPS, PyTorch runs each small elementwise operation of the live velocity as its own
kernel and writes every intermediate ``[rows, hidden]`` tensor to memory. These kernels
evaluate the arithmetic of :class:`pymc_mfm.model.flow._LiveVelocity` with a few passes
per Euler step: the week-pooled trajectory contexts, the entity moments, the per-row
context projections and one fused kernel per coordinate network that computes its
first layer, its ``hidden x hidden`` product on simdgroup matrices and its output layer
without writing activations to memory. Results agree with the eager velocity up to
float32 round-off.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .flow import JointFlowHead, _LiveNetwork, _LiveVelocity

#: Trajectory token width the kernels are written for (``flow._TRAJECTORY_EMBED_DIM``).
_TOKEN = 8

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Fast-math exponential and division: within a few float32 ulp of v / (1 + exp(-v)),
// about an eighth faster per network row than the correctly rounded division.
inline float silu(float v) { return fast::divide(v, 1.0f + fast::exp(-v)); }

// Online-softmax state of a run of weeks: max score m, denominator d and the weighted
// token sum a[8], all relative to m.
struct Pool {
    float m;
    float d;
    float a[8];
};

inline void merge(thread Pool& p, const thread Pool& q) {
    if (q.d == 0.0f) return;
    if (p.d == 0.0f) {
        p = q;
        return;
    }
    const float m = max(p.m, q.m);
    const float sp = exp(p.m - m), sq = exp(q.m - m);
    p.d = p.d * sp + q.d * sq;
    for (int e = 0; e < 8; ++e) p.a[e] = p.a[e] * sp + q.a[e] * sq;
    p.m = m;
}

// Week-pooled trajectory context of each live coordinate. S lanes of a simdgroup share
// one (l, c) pair: each pools every S-th week with an online softmax, then the lanes
// merge their states. ctx[l, NL] is the shared zero context.
// p: state weight [8], latent weight [8], score weight [8], score bias, token W [8x8],
// token bias [8].
kernel void trajectory_context(
    device float* ctx [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* latent [[buffer(2)]],
    device const float* token_base [[buffer(3)]],
    device const int* token_lead [[buffer(4)]],
    device const float* weight [[buffer(5)]],
    constant float* p [[buffer(6)]],
    constant int& L [[buffer(7)]],
    constant int& T [[buffer(8)]],
    constant int& NL [[buffer(9)]],
    constant int& S [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const int pair = gid / S, lane = gid % S;
    const bool live = pair < L * NL;
    const int l = live ? pair / NL : 0, c = live ? pair % NL : 0;
    Pool pool;
    pool.m = 0.0f;
    pool.d = 0.0f;
    for (int e = 0; e < 8; ++e) pool.a[e] = 0.0f;
    device const float* tb = token_base + (long)token_lead[l] * T * 8;
    for (int w = lane; live && w < T; w += S) {
        const long at = ((long)l * T + w) * NL + c;
        const float xv = x[at], zv = latent[at];
        float h[8];
        float s = p[24];
        for (int e = 0; e < 8; ++e) {
            h[e] = silu(tb[w * 8 + e] + zv * p[8 + e] + xv * p[e]);
            s += h[e] * p[16 + e];
        }
        Pool one;
        one.m = s;
        one.d = 1.0f;
        for (int e = 0; e < 8; ++e) one.a[e] = h[e];
        merge(pool, one);
    }
    // Butterfly over the S lanes of the pair (S divides the simdgroup width).
    for (int offset = 1; offset < S; offset <<= 1) {
        Pool other;
        other.m = simd_shuffle_xor(pool.m, offset);
        other.d = simd_shuffle_xor(pool.d, offset);
        for (int e = 0; e < 8; ++e) other.a[e] = simd_shuffle_xor(pool.a[e], offset);
        merge(pool, other);
    }
    if (!live || lane != 0) return;
    const float mask = weight[(long)l * NL + c];
    device float* out = ctx + ((long)l * (NL + 1) + c) * 8;
    for (int o = 0; o < 8; ++o) {
        float v = p[89 + o];
        for (int e = 0; e < 8; ++e) v += p[25 + o * 8 + e] * (pool.a[e] / pool.d);
        out[o] = v * mask;
    }
    if (c == 0) {
        device float* zero = ctx + ((long)l * (NL + 1) + NL) * 8;
        for (int o = 0; o < 8; ++o) zero[o] = 0.0f;
    }
}

// Normal draws from the uniforms PyTorch's CPU generator produced for them. PyTorch's
// float32 normal_ (contiguous, >= 16 values) turns each 16-value block of uniforms into
// Box-Muller pairs (j, j + 8), and when the size is not a multiple of 16 redraws the last
// 16 values from 16 more uniforms. u holds `count` runs of size (+16) uniforms each.
kernel void normal_from_uniform(
    device float* out [[buffer(0)]],
    device const float* u [[buffer(1)]],
    constant int& size [[buffer(2)]],
    constant int& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const int blocks = size / 16;
    const bool tail = size % 16 != 0;
    const int pairs = (blocks + (tail ? 1 : 0)) * 8;
    if ((int)gid >= count * pairs) return;
    const int run = gid / pairs, p = gid % pairs, j = p % 8;
    device const float* ur = u + (long)run * (size + (tail ? 16 : 0));
    device float* o = out + (long)run * size;
    const bool redrawn = p >= blocks * 8;
    const int base = redrawn ? size - 16 : (p / 8) * 16;
    device const float* ub = redrawn ? ur + size : ur + base;
    const float radius = precise::sqrt(-2.0f * precise::log(1.0f - ub[j]));
    // float(2 * pi * u2) with 2 * pi in double precision, as PyTorch rounds it.
    const float u2 = ub[j + 8];
    const float theta = fma(6.28318548f, u2, -1.74845553e-07f * u2);
    const int i0 = base + j, i1 = base + j + 8;
    if (redrawn || !tail || i0 < size - 16) o[i0] = radius * precise::cos(theta);
    if (redrawn || !tail || i1 < size - 16) o[i1] = radius * precise::sin(theta);
}

// Optional Euler step x <- (x + dt * v) * mask, then the scalar bank row
// [x_live..., 0, entity mean, entity std] of the state. One thread per (l, w).
kernel void step_moments(
    device float* bank [[buffer(0)]],
    device float* x [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device const float* active [[buffer(4)]],
    device const float* count [[buffer(5)]],
    constant float& dt [[buffer(6)]],
    constant int& L [[buffer(7)]],
    constant int& T [[buffer(8)]],
    constant int& NL [[buffer(9)]],
    constant int& NE [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= L * T) return;
    const int l = gid / T;
    device float* xr = x + (long)gid * NL;
    if (dt != 0.0f) {
        device const float* vr = v + (long)gid * NL;
        device const float* mr = mask + (long)l * NL;
        for (int i = 0; i < NL; ++i) xr[i] = (xr[i] + dt * vr[i]) * mr[i];
    }
    device const float* a = active + (long)l * NE;
    device float* b = bank + (long)gid * (NL + 3);
    float s = 0.0f;
    for (int i = 0; i < NE; ++i) s += xr[1 + i] * a[i];
    const float n = count[l];
    const float mean = s / n;
    float ss = 0.0f;
    for (int i = 0; i < NE; ++i) {
        const float d = (xr[1 + i] - mean) * a[i];
        ss += d * d;
    }
    for (int i = 0; i < NL; ++i) b[i] = xr[i];
    b[NL] = 0.0f;
    b[NL + 1] = mean;
    b[NL + 2] = sqrt(ss) / sqrt(n);
}

// First-layer terms of row c that do not vary over weeks: the flow-time projection and
// the pooled contexts the row reads. One thread per (l, c, 4 hidden units); wct is the
// context weight transposed to [NC * 8, H].
kernel void context_projection(
    device float4* out [[buffer(0)]],
    device const float* ctx [[buffer(1)]],
    constant int* cidx [[buffer(2)]],
    device const float4* wct [[buffer(3)]],
    device const float4* tproj [[buffer(4)]],
    constant int& L [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& NC [[buffer(7)]],
    constant int& NL [[buffer(8)]],
    constant int& H4 [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int j4 = gid.x;
    const int lc = gid.y;
    if (lc >= L * C || j4 >= H4) return;
    const int l = lc / C, c = lc % C;
    float4 acc = tproj[j4];
    for (int k = 0; k < NC; ++k) {
        device const float* cb = ctx + ((long)l * (NL + 1) + cidx[c * NC + k]) * 8;
        for (int e = 0; e < 8; ++e) acc += cb[e] * wct[(k * 8 + e) * H4 + j4];
    }
    out[(long)lc * H4 + j4] = acc;
}

// One coordinate network on a block of BM rows r = (l * T + w) * C + c:
//   v[l, w, col0 + c] = w3 . silu(silu(first(r)) @ W2t + b2) + b3,
//   first(r) = static[r] + cproj[l, c] + scalars(r) @ ws.
// Each BK-wide slice of the first layer is computed into threadgroup memory as the
// product needs it, the HN x HN product runs on simdgroup matrices with W2t read through
// the cache, and the output layer reduces in registers, so no [rows, HN] activation
// reaches device memory. WM x WN simdgroups own SM x SN tiles of the BM x HN block.
template <int HN, int BM, int BK, int WM, int WN>
kernel void network(
    device float* v [[buffer(0)]],
    device const float* bank [[buffer(1)]],
    constant int* sidx [[buffer(2)]],
    device const float* ws [[buffer(3)]],
    device const float* stat [[buffer(4)]],
    device const int* static_lead [[buffer(5)]],
    device const float* cproj [[buffer(6)]],
    device const float* w2t [[buffer(7)]],
    device const float* b2 [[buffer(8)]],
    device const float* w3 [[buffer(9)]],
    device const float* b3 [[buffer(10)]],
    constant int& R [[buffer(11)]],
    constant int& T [[buffer(12)]],
    constant int& C [[buffer(13)]],
    constant int& NS [[buffer(14)]],
    constant int& NB [[buffer(15)]],
    constant int& NL [[buffer(16)]],
    constant int& col0 [[buffer(17)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]])
{
    constexpr int NSMAX = 8;
    constexpr int SM = BM / WM, SN = HN / WN, FM = SM / 8, FN = SN / 8;
    constexpr int NT = WM * WN * 32;
    constexpr int PE = BM * BK / NT;  // first-layer values per thread and slice
    static_assert(PE * NT == BM * BK && NT >= BM && BK % PE == 0, "tile shape");
    threadgroup float As[BM * BK];
    threadgroup float scal[BM * NSMAX];
    threadgroup long soff[BM];
    threadgroup long coff[BM];
    threadgroup float partial[WN * BM];

    const int row0 = tgid * BM;
    if (tid < BM) {
        const int r = row0 + tid;
        if (r < R) {
            const int c = r % C, lw = r / C, w = lw % T, l = lw / T;
            soff[tid] = (((long)static_lead[l] * T + w) * C + c) * HN;
            coff[tid] = ((long)l * C + c) * HN;
            for (int i = 0; i < NS; ++i) {
                scal[tid * NSMAX + i] = bank[(long)lw * NB + sidx[c * NS + i]];
            }
        } else {
            soff[tid] = -1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int wm = sg / WN, wn = sg % WN;
    simdgroup_float8x8 acc[FM][FN];
    #pragma clang loop unroll(full)
    for (int i = 0; i < FM; ++i) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < FN; ++j) acc[i][j] = simdgroup_float8x8(0);
    }
    // Each thread computes PE consecutive first-layer values of one row per slice.
    const int pe0 = tid * PE, prow = pe0 / BK, pk = pe0 % BK;
    const long so = soff[prow], co = coff[prow];
    for (int k0 = 0; k0 < HN; k0 += BK) {
        float a[PE];
        #pragma clang loop unroll(full)
        for (int p = 0; p < PE; ++p) a[p] = 0.0f;
        if (so >= 0) {
            const int k = k0 + pk;
            #pragma clang loop unroll(full)
            for (int p = 0; p < PE; ++p) a[p] = stat[so + k + p] + cproj[co + k + p];
            for (int i = 0; i < NS; ++i) {
                const float s = scal[prow * NSMAX + i];
                #pragma clang loop unroll(full)
                for (int p = 0; p < PE; ++p) a[p] += s * ws[i * HN + k + p];
            }
            #pragma clang loop unroll(full)
            for (int p = 0; p < PE; ++p) a[p] = silu(a[p]);
        }
        #pragma clang loop unroll(full)
        for (int p = 0; p < PE; ++p) As[pe0 + p] = a[p];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 am[FM], bm[FN];
            #pragma clang loop unroll(full)
            for (int i = 0; i < FM; ++i) {
                simdgroup_load(am[i], As + (wm * SM + i * 8) * BK + kk, BK);
            }
            #pragma clang loop unroll(full)
            for (int j = 0; j < FN; ++j) {
                simdgroup_load(bm[j], w2t + (long)(k0 + kk) * HN + wn * SN + j * 8, HN);
            }
            #pragma clang loop unroll(full)
            for (int i = 0; i < FM; ++i) {
                #pragma clang loop unroll(full)
                for (int j = 0; j < FN; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane layout of an 8x8 simdgroup matrix: row fm, columns fn and fn + 1.
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    #pragma clang loop unroll(full)
    for (int i = 0; i < FM; ++i) {
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (int j = 0; j < FN; ++j) {
            const int col = wn * SN + j * 8 + fn;
            auto e = acc[i][j].thread_elements();
            s += silu(e[0] + b2[col]) * w3[col] + silu(e[1] + b2[col + 1]) * w3[col + 1];
        }
        // Lanes 1 and 8 apart hold the other columns of the same row.
        s += simd_shuffle_xor(s, 1);
        s += simd_shuffle_xor(s, 8);
        if ((lane & 9) == 0) partial[wn * BM + wm * SM + i * 8 + fm] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < BM && soff[tid] >= 0) {
        float total = b3[0];
        for (int q = 0; q < WN; ++q) total += partial[q * BM + tid];
        const int r = row0 + tid;
        v[(long)(r / C) * NL + col0 + r % C] = total;
    }
}

#define NETWORK(HN, WN)                                                                  \
template [[host_name("network_" #HN)]] kernel void network<HN, 32, 16, 1, WN>(          \
    device float*, device const float*, constant int*, device const float*,              \
    device const float*, device const int*, device const float*, device const float*,    \
    device const float*, device const float*, device const float*, constant int&,        \
    constant int&, constant int&, constant int&, constant int&, constant int&,           \
    constant int&, uint, uint, uint, uint);
NETWORK(16, 2)
NETWORK(32, 4)
NETWORK(64, 4)
NETWORK(128, 4)
NETWORK(256, 4)
"""

#: Threads per network threadgroup for each compiled hidden width (32 rows per group).
_NETWORK_THREADS = {16: 64, 32: 128, 64: 128, 128: 128, 256: 128}
_BLOCK_ROWS = 32
#: Most scalar inputs a network row reads.
_MAX_SCALARS = 8
#: Threads per Apple GPU simdgroup.
_SIMD = 32


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


#: Host tensors that queued copies have yet to read, each with an event recorded after it.
_READING: list[tuple[torch.mps.Event, Tensor]] = []
_READING_LOCK = threading.Lock()


def to_device_async(host: Tensor, device: torch.device) -> Tensor:
    """Copy ``host`` to the GPU ``device`` without waiting for the work queued before it.

    A blocking copy waits for every queued kernel. A non-blocking copy reads the host
    memory only when it runs on the GPU, so ``host`` stays referenced until an event
    recorded after the copy has completed.
    """
    host = host.contiguous()
    moved = host.to(device, non_blocking=True)
    event = torch.mps.Event()
    event.record()
    with _READING_LOCK:
        _READING[:] = [entry for entry in _READING if not entry[0].query()]
        _READING.append((event, host))
    return moved


@dataclass(frozen=True)
class HostNoise:
    """Standard normal draws of ``shape`` made on the host by PyTorch's CPU generator.

    With ``uniform`` the tensor holds the generator's uniforms that ``torch.randn`` would
    have turned into those normals; :func:`to_device` finishes them on the GPU.
    """

    values: Tensor
    shape: tuple[int, ...]
    uniform: bool


def draw_host(shape: tuple[int, ...], dtype: torch.dtype, uniform: bool) -> HostNoise:
    """Consume the CPU generator exactly as ``torch.randn(shape, dtype=dtype)`` does.

    ``uniform`` (from :func:`uniform_normals`) draws only the uniforms, about 3.5 times
    faster on the CPU, for the GPU to transform.
    """
    count = math.prod(shape)
    if uniform and dtype == torch.float32 and count >= 16:
        return HostNoise(torch.rand(count + (16 if count % 16 else 0)), tuple(shape), True)
    return HostNoise(torch.randn(shape, dtype=dtype), tuple(shape), False)


def to_device(parts: list[HostNoise], device: torch.device) -> Tensor:
    """Normal draws of consecutive host ``parts`` on ``device``, flattened and joined.

    The copies do not wait for queued device work (see :func:`to_device_async`).
    """
    out = []
    start = 0
    while start < len(parts):
        stop = start + 1
        first = parts[start]
        while (
            stop < len(parts)
            and parts[stop].uniform == first.uniform
            and parts[stop].values.numel() == first.values.numel()
        ):
            stop += 1
        group = parts[start:stop]
        moved = to_device_async(torch.cat([part.values.reshape(-1) for part in group]), device)
        if first.uniform:
            size = math.prod(first.shape)
            normals = torch.empty(len(group) * size, device=device, dtype=moved.dtype)
            pairs = (size // 16 + (1 if size % 16 else 0)) * 8
            _library().normal_from_uniform(
                normals, moved, size, len(group), threads=len(group) * pairs
            )
            moved = normals
        out.append(moved)
        start = stop
    return torch.cat(out) if len(out) > 1 else out[0]


@lru_cache(maxsize=1)
def uniform_normals() -> bool:
    """Whether :func:`draw_host` may draw uniforms for the GPU to finish into normals.

    Checks once that PyTorch's CPU ``normal_`` consumes and transforms its uniforms as
    ``normal_from_uniform`` assumes, so the GPU draws equal ``torch.randn`` up to float32
    round-off; otherwise sampling keeps drawing normals on the CPU.
    """
    if not torch.backends.mps.is_available():
        return False
    device = torch.device("mps")
    normal, uniform = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    for count in (16, 37, 1000):
        expected = torch.randn(count, generator=normal)
        values = torch.rand(count + (16 if count % 16 else 0), generator=uniform)
        got = to_device([HostNoise(values, (count,), True)], device).cpu()
        if not torch.allclose(got, expected, rtol=1e-5, atol=1e-5):
            return False
    return torch.equal(normal.get_state(), uniform.get_state())


def supported(head: JointFlowHead, weight: Tensor) -> bool:
    """Whether :func:`integrate` can transport ``head``'s flow under state mask ``weight``."""
    hidden = head.base_net[2].weight.shape
    return (
        weight.device.type == "mps"
        and weight.dtype == torch.float32
        and weight.dim() >= 2
        and weight.shape[-2] == 1
        and head.trajectory_token_net[1].weight.shape == (_TOKEN, _TOKEN)
        and hidden[0] == hidden[1]
        and hidden[0] in _NETWORK_THREADS
    )


def _lead_index(shape: torch.Size, lead: torch.Size, device: torch.device) -> Tensor:
    """For each flattened ``lead`` index, the flattened index into broadcast ``shape``."""
    index = torch.arange(math.prod(shape), dtype=torch.int32, device=device).reshape(shape)
    return index.expand(lead).reshape(-1).contiguous()


class _Network:
    """Kernel-ready tensors of one live coordinate network."""

    def __init__(self, network: _LiveNetwork, weeks: int, column: int, times: Tensor):
        static = network.static
        self.rows = network.rows
        self.column = column
        self.static_shape = static.shape[:-3]
        self.static = static.reshape(-1, weeks, self.rows, static.shape[-1]).contiguous()
        if network.scalar_index.shape[1] > _MAX_SCALARS:
            raise ValueError(f"GPU sampling kernels read at most {_MAX_SCALARS} scalar inputs")
        self.scalar_index = network.scalar_index.to(torch.int32).contiguous()
        self.scalar_weight = network.scalar_weight.t().contiguous()
        self.context_index = network.context_index.to(torch.int32).contiguous()
        self.context_weight = network.context_weight.t().contiguous()
        self.hidden_weight = network.hidden_weight.contiguous()
        self.hidden_bias = network.hidden_bias.contiguous()
        self.output_weight = network.output_weight.contiguous()
        self.output_bias = network.output_bias.contiguous()
        # Flow-time projection of the first layer at every grid time, one tensor per step.
        projected = torch.mm(times, network.time_weight.t())
        self.time_projections = [row.clone() for row in projected.unbind()]


class _Prepared:
    """Inputs of one live velocity shared by every batch it transports."""

    def __init__(self, live: _LiveVelocity, weeks: int, grid):
        from .flow import _fourier_time

        self.weeks = weeks
        self.grid = tuple(grid)
        reference = live.weight
        times = torch.stack([_fourier_time(reference.new_full((1,), t)) for t, _, _ in grid])
        token_base = live.token_base  # [..., T, 1, token]
        self.token_shape = token_base.shape[:-3]
        self.token_base = token_base.reshape(-1, weeks, _TOKEN).contiguous()
        self.params = torch.cat(
            [
                live.token_state_weight,
                live.token_latent_weight,
                live.score_weight,
                live.score_bias.reshape(1),
                live.token_layer.weight.reshape(-1),
                live.token_layer.bias,
            ]
        ).contiguous()
        self.networks = []
        column = 0
        for network in live.networks:
            if network.rows:
                self.networks.append(_Network(network, weeks, column, times.squeeze(1)))
            column += network.rows
        self.hidden = self.networks[0].hidden_weight.shape[0]


def _prepared(live: _LiveVelocity, weeks: int, grid) -> _Prepared:
    prepared = getattr(live, "_kernel_inputs", None)
    if prepared is None or prepared.weeks != weeks or prepared.grid != tuple(grid):
        prepared = live._kernel_inputs = _Prepared(live, weeks, grid)
    return prepared


def integrate(
    live: _LiveVelocity,
    state: Tensor,
    trajectory_latent: Tensor,
    grid,
    index: Tensor | None = None,
    final_from: int = 0,
) -> Tensor:
    """Euler-integrate the live state ``[..., T, NL]`` as ``_LiveVelocity.integrate`` does.

    Without ``index`` the live velocity's representations and masks broadcast over the
    state's lead axes. With it, they hold one row per distinct conditioning and
    ``index`` gives the row of each entry of the state's last lead axis. Live columns
    before ``final_from`` are not wanted after the last step: networks that only write
    them skip it, so those columns end one step short.
    """
    lib = _library()
    device, dtype = state.device, state.dtype
    *lead_shape, weeks, n_live = state.shape
    lead = torch.Size(lead_shape)
    n_lead = math.prod(lead)
    prepared = _prepared(live, weeks, grid)
    x = state.reshape(n_lead, weeks, n_live).clone(memory_format=torch.contiguous_format)
    latent = trajectory_latent.reshape(n_lead, weeks, n_live).contiguous()

    n_entities = live.entity_active.shape[-1]
    if index is None:
        weight = live.weight.expand(*lead, 1, n_live).reshape(n_lead, n_live)
        active = live.entity_active.expand(*lead, 1, n_entities).reshape(n_lead, n_entities)
        count = live.entity_count.expand(*lead, 1, 1).reshape(n_lead)
        token_lead = _lead_index(prepared.token_shape, lead, device)
        static_leads = [_lead_index(net.static_shape, lead, device) for net in prepared.networks]
    else:
        lead_rows = index.repeat(n_lead // index.numel())
        weight = live.weight.reshape(-1, n_live)[lead_rows]
        active = live.entity_active.reshape(-1, n_entities)[lead_rows]
        count = live.entity_count.reshape(-1)[lead_rows]
        token_lead = lead_rows.to(torch.int32)
        static_leads = [token_lead] * len(prepared.networks)
    weight, active, count = weight.contiguous(), active.contiguous(), count.contiguous()

    hidden = prepared.hidden
    h4 = hidden // 4
    kernel, group = getattr(lib, f"network_{hidden}"), _NETWORK_THREADS[hidden]
    bank = torch.empty(n_lead, weeks, n_live + 3, device=device, dtype=dtype)
    contexts = torch.empty(n_lead, n_live + 1, _TOKEN, device=device, dtype=dtype)
    velocity = torch.empty(n_lead, weeks, n_live, device=device, dtype=dtype)
    projections = [
        torch.empty(n_lead, network.rows, hidden, device=device, dtype=dtype)
        for network in prepared.networks
    ]
    # Lanes pooling one coordinate's weeks: about eight weeks per lane, at most a simdgroup.
    lanes = min(_SIMD, 1 << max(0, math.ceil(math.log2(max(1, weeks / 8)))))
    pooled_threads = -(-n_lead * n_live * lanes // _SIMD) * _SIMD
    lib.step_moments(
        bank, x, velocity, weight, active, count, 0.0, n_lead, weeks, n_live, n_entities,
        threads=n_lead * weeks,
    )  # fmt: skip
    for step, (_, dt, _) in enumerate(grid):
        lib.trajectory_context(
            contexts, x, latent, prepared.token_base, token_lead, weight, prepared.params,
            n_lead, weeks, n_live, lanes,
            threads=pooled_threads, group_size=_SIMD,
        )  # fmt: skip
        last = step == len(grid) - 1
        for network, static_lead, projection in zip(
            prepared.networks, static_leads, projections, strict=True
        ):
            if last and network.column + network.rows <= final_from:
                continue
            rows = n_lead * weeks * network.rows
            lib.context_projection(
                projection, contexts, network.context_index, network.context_weight,
                network.time_projections[step], n_lead, network.rows,
                network.context_index.shape[1], n_live, h4,
                threads=(h4, n_lead * network.rows),
            )  # fmt: skip
            kernel(
                velocity, bank, network.scalar_index, network.scalar_weight, network.static,
                static_lead, projection, network.hidden_weight, network.hidden_bias,
                network.output_weight, network.output_bias, rows, weeks, network.rows,
                network.scalar_index.shape[1], n_live + 3, n_live, network.column,
                threads=-(-rows // _BLOCK_ROWS) * group, group_size=group,
            )  # fmt: skip
        # x <- (x + dt * v) * mask, then the next step's scalar bank.
        lib.step_moments(
            bank, x, velocity, weight, active, count, float(dt), n_lead, weeks, n_live,
            n_entities,
            threads=n_lead * weeks,
        )  # fmt: skip
    return x.reshape(state.shape)
