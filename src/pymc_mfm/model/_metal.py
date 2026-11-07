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

inline float silu(float v) { return v / (1.0f + exp(-v)); }

// Week-pooled trajectory context of each live coordinate, one thread per (l, c), with an
// online softmax over weeks. ctx[l, NL] is the shared zero context.
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
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= L * NL) return;
    const int l = gid / NL, c = gid % NL;
    float m = -INFINITY, denom = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    device const float* tb = token_base + (long)token_lead[l] * T * 8;
    for (int w = 0; w < T; ++w) {
        const long at = ((long)l * T + w) * NL + c;
        const float xv = x[at], zv = latent[at];
        float h[8];
        float s = p[24];
        for (int e = 0; e < 8; ++e) {
            h[e] = silu(tb[w * 8 + e] + zv * p[8 + e] + xv * p[e]);
            s += h[e] * p[16 + e];
        }
        if (s > m) {
            const float scale = exp(m - s);
            denom = denom * scale + 1.0f;
            for (int e = 0; e < 8; ++e) acc[e] = acc[e] * scale + h[e];
            m = s;
        } else {
            const float q = exp(s - m);
            denom += q;
            for (int e = 0; e < 8; ++e) acc[e] += q * h[e];
        }
    }
    const float mask = weight[(long)l * NL + c];
    device float* out = ctx + ((long)l * (NL + 1) + c) * 8;
    for (int o = 0; o < 8; ++o) {
        float v = p[89 + o];
        for (int e = 0; e < 8; ++e) v += p[25 + o * 8 + e] * (acc[e] / denom);
        out[o] = v * mask;
    }
    if (c == 0) {
        device float* zero = ctx + ((long)l * (NL + 1) + NL) * 8;
        for (int o = 0; o < 8; ++o) zero[o] = 0.0f;
    }
}

// Scalar bank row [x_live..., 0, entity mean, entity std], one thread per (l, w).
kernel void moments(
    device float* bank [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* active [[buffer(2)]],
    device const float* count [[buffer(3)]],
    constant int& L [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& NL [[buffer(6)]],
    constant int& NE [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= L * T) return;
    const int l = gid / T;
    device const float* xr = x + (long)gid * NL;
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
// the pooled contexts the row reads. One thread per (l, c, 4 hidden units).
kernel void context_projection(
    device float4* out [[buffer(0)]],
    device const float* ctx [[buffer(1)]],
    constant int* cidx [[buffer(2)]],
    device const float* wc [[buffer(3)]],
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
    const int width = NC * 8;
    device const float* w0 = wc + (long)(j4 * 4) * width;
    for (int k = 0; k < NC; ++k) {
        device const float* cb = ctx + ((long)l * (NL + 1) + cidx[c * NC + k]) * 8;
        for (int e = 0; e < 8; ++e) {
            const int col = k * 8 + e;
            acc += cb[e] * float4(w0[col], w0[width + col], w0[2 * width + col],
                                  w0[3 * width + col]);
        }
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


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE)


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
        self.context_weight = network.context_weight.contiguous()
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
) -> Tensor:
    """Euler-integrate the live state ``[..., T, NL]`` as ``_LiveVelocity.integrate`` does.

    Without ``index`` the live velocity's representations and masks broadcast over the
    state's lead axes. With it, they hold one row per distinct conditioning and
    ``index`` gives the row of each entry of the state's last lead axis.
    """
    lib = _library()
    device, dtype = state.device, state.dtype
    *lead_shape, weeks, n_live = state.shape
    lead = torch.Size(lead_shape)
    n_lead = math.prod(lead)
    prepared = _prepared(live, weeks, grid)
    x = state.reshape(n_lead, weeks, n_live).contiguous()
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
    mask = weight.unsqueeze(1)
    for step, (_, dt, _) in enumerate(grid):
        lib.trajectory_context(
            contexts, x, latent, prepared.token_base, token_lead, weight, prepared.params,
            n_lead, weeks, n_live,
            threads=n_lead * n_live,
        )  # fmt: skip
        lib.moments(
            bank, x, active, count, n_lead, weeks, n_live, n_entities,
            threads=n_lead * weeks,
        )  # fmt: skip
        for network, static_lead, projection in zip(
            prepared.networks, static_leads, projections, strict=True
        ):
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
        x = (x + dt * velocity) * mask
    return x.reshape(state.shape)
