"""Fused GPU kernels for the sampling-time live velocity: Apple (Metal) and NVIDIA (CUDA).

On a GPU, PyTorch runs each small elementwise operation of the live velocity as its own
kernel and writes every intermediate ``[rows, hidden]`` tensor to memory. These kernels
evaluate the arithmetic of :class:`pymc_mktfm.model.flow._LiveVelocity` with a few passes
per Euler step: the week-pooled trajectory contexts, the entity moments, the per-row
context projections and one fused kernel per coordinate network that computes its
first layer, its ``hidden x hidden`` product and its output layer without writing
activations to memory. This module holds the Metal sources (simdgroup matrices) and the
transport that launches them; :mod:`pymc_mktfm.model._cuda` holds the CUDA sources it
launches on NVIDIA GPUs. Results agree with the eager velocity up to float32 round-off.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import _cuda

if TYPE_CHECKING:
    from .flow import JointFlowHead, _LiveNetwork, _LiveVelocity

#: Trajectory token width the kernels are written for (``flow._TRAJECTORY_EMBED_DIM``).
_TOKEN = 8
#: Oldest PyTorch whose Apple GPU backend samples this model correctly: on 2.8, MPS computes
#: linear layers of column-sliced weights wrongly (every draw changes); 2.7 lacks MPS
#: ``index_copy`` and 2.6 lacks ``torch.mps.compile_shader``. Sampling refuses older releases
#: and the encoder keeps PyTorch's attention there.
MIN_TORCH = (2, 9)

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
// the pooled contexts the row reads, for the rows c in [c0, CT) of every network of one
// step, side by side as [L, CT, H]. Row c reads rows[c] = (contexts NC, first row of its
// network's context weights in wct (transposed to [rows, H]), network), the contexts
// cidx[c, :NC] and its network's flow-time projection. One thread per (4 hidden units,
// row c, run of D draws) keeps the row's weights in registers while the draws stream
// past, instead of reloading them for every draw.
kernel void context_projection(
    device float4* out [[buffer(0)]],
    device const float* ctx [[buffer(1)]],
    constant int* cidx [[buffer(2)]],
    constant int* rows [[buffer(3)]],
    device const float4* wct [[buffer(4)]],
    device const float4* tproj [[buffer(5)]],
    constant int& L [[buffer(6)]],
    constant int& CT [[buffer(7)]],
    constant int& c0 [[buffer(8)]],
    constant int& NL [[buffer(9)]],
    constant int& H4 [[buffer(10)]],
    constant int& D [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    constexpr int NCMAX = 3;
    const int width = CT - c0;
    const int j4 = gid.x, c = c0 + (int)gid.y % width, l0 = ((int)gid.y / width) * D;
    if (j4 >= H4 || l0 >= L) return;
    const int NC = rows[c * 3], wrow = rows[c * 3 + 1], net = rows[c * 3 + 2];
    float4 w[NCMAX * 8];
    int ci[NCMAX];
    #pragma clang loop unroll(full)
    for (int k = 0; k < NCMAX; ++k) {
        if (k < NC) {
            ci[k] = cidx[c * NCMAX + k];
            #pragma clang loop unroll(full)
            for (int e = 0; e < 8; ++e) w[k * 8 + e] = wct[(wrow + k * 8 + e) * H4 + j4];
        }
    }
    const float4 t = tproj[net * H4 + j4];
    const int l1 = min(L, l0 + D);
    for (int l = l0; l < l1; ++l) {
        float4 acc = t;
        #pragma clang loop unroll(full)
        for (int k = 0; k < NCMAX; ++k) {
            if (k < NC) {
                device const float* cb = ctx + ((long)l * (NL + 1) + ci[k]) * 8;
                #pragma clang loop unroll(full)
                for (int e = 0; e < 8; ++e) acc += cb[e] * w[k * 8 + e];
            }
        }
        out[((long)l * CT + c) * H4 + j4] = acc;
    }
}

// One coordinate network on a block of BM rows r = (l * T + w) * C + c:
//   v[l, w, col0 + c] = w3 . silu(silu(first(r)) @ W2t + b2) + b3,
//   first(r) = static[r] + cproj[l, c0 + c] + scalars(r) @ ws,
// where cproj [L, CT, HN] holds every network's context projections side by side.
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
    constant int& CT [[buffer(18)]],
    constant int& c0 [[buffer(19)]],
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
            coff[tid] = ((long)l * CT + c0 + c) * HN;
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
    constant int&, constant int&, constant int&, uint, uint, uint, uint);
NETWORK(16, 2)
NETWORK(32, 4)
NETWORK(64, 4)
NETWORK(128, 4)
NETWORK(256, 4)

// Encoder self-attention over 16-wide heads, as scaled_dot_product_attention computes it,
// reading q, k and v from the packed projection [sequences, N, 3, heads, 16] and writing
// [sequences, N, heads, 16]. Each thread keeps one query's online-softmax state.
//
// Causal (attention over weeks): query i attends keys j <= i. One threadgroup of 32
// threads per (sequence, head, run of 32 queries); runs of 32 keys pass through
// threadgroup memory.
kernel void causal_attention16(
    device float* out [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    constant int& N [[buffer(2)]],
    constant int& heads [[buffer(3)]],
    constant int& tiles [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    threadgroup float K[32 * 16];
    threadgroup float V[32 * 16];
    const int tile = tgid % tiles, head = (tgid / tiles) % heads, seq = tgid / (tiles * heads);
    const int width = heads * 16, stride = 3 * width;
    device const float* rows = qkv + (long)seq * N * stride + head * 16;
    const int i = tile * 32 + (int)lane;
    const bool valid = i < N;
    float q[16], acc[16];
    for (int d = 0; d < 16; ++d) {
        q[d] = valid ? rows[(long)i * stride + d] : 0.0f;
        acc[d] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    const int keys = min(N, tile * 32 + 32);
    for (int j0 = 0; j0 < keys; j0 += 32) {
        const int j = j0 + (int)lane;
        for (int d = 0; d < 16; ++d) {
            K[lane * 16 + d] = j < N ? rows[(long)j * stride + width + d] : 0.0f;
            V[lane * 16 + d] = j < N ? rows[(long)j * stride + 2 * width + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int count = valid ? min(32, i - j0 + 1) : 0;
        for (int c = 0; c < count; ++c) {
            float s = 0.0f;
            for (int d = 0; d < 16; ++d) s = fma(q[d], K[c * 16 + d], s);
            s *= 0.25f;
            if (s > m) {
                const float correction = precise::exp(m - s);
                l *= correction;
                for (int d = 0; d < 16; ++d) acc[d] *= correction;
                m = s;
            }
            const float p = precise::exp(s - m);
            l += p;
            for (int d = 0; d < 16; ++d) acc[d] = fma(p, V[c * 16 + d], acc[d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        device float* o = out + ((long)seq * N + i) * width + head * 16;
        for (int d = 0; d < 16; ++d) o[d] = acc[d] / l;
    }
}

// Key-masked (attention over a week's entities): query i attends the keys j active for
// its sequence's task, active[(seq / per_task) * N + j]. One thread per (sequence, head,
// query).
kernel void masked_attention16(
    device float* out [[buffer(0)]],
    device const float* qkv [[buffer(1)]],
    device const bool* active [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& heads [[buffer(4)]],
    constant int& per_task [[buffer(5)]],
    constant int& sequences [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const int i = gid % N, head = (gid / N) % heads, seq = gid / (N * heads);
    if (seq >= sequences) return;
    const int width = heads * 16, stride = 3 * width;
    device const float* rows = qkv + (long)seq * N * stride + head * 16;
    device const bool* keys = active + (long)(seq / per_task) * N;
    float q[16], acc[16];
    for (int d = 0; d < 16; ++d) {
        q[d] = rows[(long)i * stride + d];
        acc[d] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (!keys[j]) continue;
        device const float* k = rows + (long)j * stride + width;
        float s = 0.0f;
        for (int d = 0; d < 16; ++d) s = fma(q[d], k[d], s);
        s *= 0.25f;
        if (s > m) {
            const float correction = precise::exp(m - s);
            l *= correction;
            for (int d = 0; d < 16; ++d) acc[d] *= correction;
            m = s;
        }
        const float p = precise::exp(s - m);
        l += p;
        device const float* v = k + width;
        for (int d = 0; d < 16; ++d) acc[d] = fma(p, v[d], acc[d]);
    }
    device float* o = out + ((long)seq * N + i) * width + head * 16;
    for (int d = 0; d < 16; ++d) o[d] = acc[d] / l;
}
"""

#: Threads per network threadgroup for each compiled hidden width (32 rows per group).
_NETWORK_THREADS = {16: 64, 32: 128, 64: 128, 128: 128, 256: 128}
_BLOCK_ROWS = 32
#: Most scalar inputs a network row reads.
_MAX_SCALARS = 8
#: Most pooled contexts a network row reads (NCMAX of the projection kernel).
_MAX_CONTEXTS = 3
#: Draws per context-projection thread.
_PROJECTION_DRAWS = 8
#: Threads per Apple GPU simdgroup.
_SIMD = 32


@lru_cache(maxsize=1)
def _metal_library():
    return torch.mps.compile_shader(_SOURCE)


def _library(device: torch.device):
    """The compiled kernels for ``device``: Metal on an Apple GPU, CUDA on an NVIDIA GPU."""
    return _cuda.library() if device.type == "cuda" else _metal_library()


#: Host tensors that queued copies have yet to read, each with an event recorded after it.
_READING: list[tuple[torch.mps.Event, Tensor]] = []
_READING_LOCK = threading.Lock()


def to_device_async(host: Tensor, device: torch.device) -> Tensor:
    """Copy ``host`` to the GPU ``device`` without waiting for the work queued before it.

    A blocking copy waits for every queued kernel. A non-blocking copy reads the host
    memory only when it runs on the GPU, so ``host`` stays referenced until an event
    recorded after the copy has completed (on CUDA, PyTorch's pinned memory does this).
    """
    if device.type == "cuda":
        return _cuda.to_device_async(host, device)
    host = host.contiguous()
    moved = host.to(device, non_blocking=True)
    event = torch.mps.Event()
    event.record()
    with _READING_LOCK:
        _READING[:] = [entry for entry in _READING if not entry[0].query()]
        _READING.append((event, host))
    return moved


def flush() -> None:
    """Commit the GPU work queued so far without waiting for it.

    MPS encodes kernels into a command buffer that the GPU starts only once something
    commits it, such as a read that waits; recording an event commits it at once.
    """
    torch.mps.Event().record()


@dataclass(frozen=True)
class HostNoise:
    """Standard normal draws of ``shape`` made on the host by PyTorch's CPU generator.

    With ``uniform`` the tensor holds the generator's uniforms that ``torch.randn`` would
    have turned into those normals; :func:`to_device` finishes them on the GPU.
    """

    values: Tensor
    shape: tuple[int, ...]
    uniform: bool


def host_size(shape: tuple[int, ...], dtype: torch.dtype, uniform: bool) -> int:
    """Number of values :func:`draw_host` keeps for a draw of ``shape``."""
    count = math.prod(shape)
    if uniform and dtype == torch.float32 and count >= 16:
        return count + (16 if count % 16 else 0)
    return count


def draw_host(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    uniform: bool,
    out: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> HostNoise:
    """Consume the CPU generator exactly as ``torch.randn(shape, dtype=dtype)`` does.

    ``uniform`` (from :func:`uniform_normals`) draws only the uniforms, about 3.5 times
    faster on the CPU, for the GPU to transform. ``out`` optionally receives the values:
    a contiguous 1-D host tensor of :func:`host_size` elements and the drawn dtype.
    ``generator`` is a CPU generator to draw from; ``None`` uses the global one.
    """
    if uniform and dtype == torch.float32 and math.prod(shape) >= 16:
        size = host_size(shape, dtype, uniform)
        values = (
            torch.rand(size, generator=generator)
            if out is None
            else out.uniform_(generator=generator)
        )
        return HostNoise(values, tuple(shape), True)
    values = (
        torch.randn(shape, dtype=dtype, generator=generator)
        if out is None
        else out.normal_(generator=generator)
    )
    return HostNoise(values, tuple(shape), False)


def _joined(values: list[Tensor]) -> Tensor:
    """The flat concatenation of ``values``; a view when they lie end to end in one buffer."""
    first = values[0]
    position = first.storage_offset()
    pointer = first.untyped_storage().data_ptr()
    for value in values:
        if (
            value.dim() != 1
            or value.untyped_storage().data_ptr() != pointer
            or value.storage_offset() != position
        ):
            return torch.cat([value.reshape(-1) for value in values])
        position += value.numel()
    return first.as_strided((position - first.storage_offset(),), (1,))


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
        moved = to_device_async(_joined([part.values for part in group]), device)
        if first.uniform:
            size = math.prod(first.shape)
            normals = torch.empty(len(group) * size, device=device, dtype=moved.dtype)
            pairs = (size // 16 + (1 if size % 16 else 0)) * 8
            _metal_library().normal_from_uniform(
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
    if not torch.backends.mps.is_available() or torch.__version__ < MIN_TORCH:
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


@lru_cache(maxsize=1)
def linear_correct() -> bool:
    """Whether PyTorch's MPS linear layer with a bias agrees with the CPU on this Apple GPU.

    In a macOS virtual machine (GPU "Apple Paravirtual device"), PyTorch 2.14's MPS computes
    ``F.linear`` with a bias wrongly, 30-80% off at every shape, while products and bias
    additions on their own are right; every MKTFM layer is such a linear layer. Checked once,
    on a small layer, with its own generator.
    """
    generator = torch.Generator().manual_seed(0)
    x, weight, bias = (torch.randn(shape, generator=generator) for shape in ((4, 8), (8, 8), (8,)))
    device = torch.device("mps")
    got = torch.nn.functional.linear(x.to(device), weight.to(device), bias.to(device)).cpu()
    expected = torch.nn.functional.linear(x, weight, bias)
    return torch.allclose(got, expected, rtol=1e-4, atol=1e-5)


def attention_supported(qkv: Tensor, head_width: int) -> bool:
    """Whether the attention kernels can read the packed projection ``qkv[S, N, 3 * D]``."""
    return (
        qkv.device.type == "mps"
        and torch.__version__ >= MIN_TORCH
        and qkv.dtype == torch.float32
        and qkv.dim() == 3
        and qkv.is_contiguous()
        and head_width == 16
        and not (torch.is_grad_enabled() and qkv.requires_grad)
    )


def causal_attention(qkv: Tensor, heads: int) -> Tensor:
    """Causal self-attention of 16-wide heads from ``qkv[S, N, 3 * heads * 16]``.

    Equals ``scaled_dot_product_attention`` under a lower-triangular mask up to float32
    round-off, without materializing the ``N x N`` scores; returns ``[S, N, heads * 16]``.
    """
    sequences, length, _ = qkv.shape
    out = torch.empty(sequences, length, heads * 16, device=qkv.device, dtype=qkv.dtype)
    tiles = -(-length // _SIMD)
    _metal_library().causal_attention16(
        out, qkv, length, heads, tiles,
        threads=sequences * heads * tiles * _SIMD, group_size=_SIMD,
    )  # fmt: skip
    return out


def masked_attention(qkv: Tensor, heads: int, active: Tensor) -> Tensor:
    """Self-attention of 16-wide heads over each task's active keys.

    ``qkv[S, N, 3 * heads * 16]`` holds ``S = tasks * per_task`` sequences, task-major;
    ``active[tasks, N]`` marks the keys of each task. Equals
    ``scaled_dot_product_attention`` with that key mask up to float32 round-off; returns
    ``[S, N, heads * 16]``.
    """
    sequences, length, _ = qkv.shape
    active = active.to(torch.bool).contiguous()
    out = torch.empty(sequences, length, heads * 16, device=qkv.device, dtype=qkv.dtype)
    _metal_library().masked_attention16(
        out, qkv, active, length, heads, sequences // active.shape[0], sequences,
        threads=sequences * heads * length,
    )  # fmt: skip
    return out


def has_kernels(device: torch.device, hidden: int) -> bool:
    """Whether sampling on ``device`` runs these fused kernels for ``hidden``-wide networks: an
    Apple GPU on PyTorch 2.9 or newer, or an NVIDIA GPU on which every kernel builds."""
    if device.type == "mps":
        return hidden in _NETWORK_THREADS and torch.__version__ >= MIN_TORCH
    return device.type == "cuda" and _cuda.available(device, hidden)


def prepare(device: torch.device, heads: Iterable[JointFlowHead]) -> None:
    """Start building the kernels that sampling ``heads`` on the GPU ``device`` launches.

    NVIDIA kernels build with NVRTC in background threads (see :func:`_cuda.prefetch`);
    Apple GPU kernels compile in milliseconds on first use.
    """
    if device.type != "cuda":
        return
    widths = {
        head.base_net[2].weight.shape[0]
        for head in heads
        if head.base_net[2].weight.dtype == torch.float32
    }
    if widths:
        _cuda.prefetch(device, widths)


def supported(
    head: JointFlowHead, device: torch.device, dtype: torch.dtype, mask_shape: torch.Size
) -> bool:
    """Whether :func:`integrate` can transport ``head``'s flow on ``device`` in ``dtype`` under
    a state mask of ``mask_shape``. The model and mask are checked before the device, so
    sampling the kernels cannot serve never waits for them to build."""
    hidden = head.base_net[2].weight.shape
    return (
        dtype == torch.float32
        and len(mask_shape) >= 2
        and mask_shape[-2] == 1
        and head.trajectory_token_net[1].weight.shape == (_TOKEN, _TOKEN)
        and hidden[0] == hidden[1]
        and has_kernels(device, hidden[0])
    )


def _lead_index(shape: torch.Size, lead: torch.Size, device: torch.device) -> Tensor:
    """For each flattened ``lead`` index, the flattened index into broadcast ``shape``."""
    index = torch.arange(math.prod(shape), dtype=torch.int32, device=device).reshape(shape)
    return index.expand(lead).reshape(-1).contiguous()


@lru_cache(maxsize=8)
def _grid_times(times: tuple[float, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
    """Fourier flow-time features ``[steps, t_dim]`` of the grid times, as the live velocity
    computes them, made once per grid, device and dtype."""
    from .flow import _fourier_time

    return torch.stack(
        [_fourier_time(torch.full((1,), t, device=device, dtype=dtype)) for t in times]
    ).squeeze(1)


class _Network:
    """Kernel-ready tensors of one live coordinate network."""

    def __init__(self, network: _LiveNetwork, weeks: int, column: int, offset: int, times: Tensor):
        static = network.static
        self.rows = network.rows
        #: First live state column the network writes; first row of its context projections.
        self.column, self.offset = column, offset
        self.static_shape = static.shape[:-3]
        self.static = static.reshape(-1, weeks, self.rows, static.shape[-1]).contiguous()
        if network.scalar_index.shape[1] > _MAX_SCALARS:
            raise ValueError(f"GPU sampling kernels read at most {_MAX_SCALARS} scalar inputs")
        if network.context_index.shape[1] > _MAX_CONTEXTS:
            raise ValueError(f"GPU sampling kernels read at most {_MAX_CONTEXTS} contexts")
        self.scalar_index = network.scalar_index.to(torch.int32).contiguous()
        self.scalar_weight = network.scalar_weight.t().contiguous()
        self.context_index = network.context_index.to(torch.int32).contiguous()
        self.context_weight = network.context_weight.t().contiguous()
        self.hidden_weight = network.hidden_weight.contiguous()
        self.hidden_bias = network.hidden_bias.contiguous()
        self.output_weight = network.output_weight.contiguous()
        self.output_bias = network.output_bias.contiguous()
        #: Flow-time projection of the first layer at every grid time, ``[steps, hidden]``.
        self.time_projection = torch.mm(times, network.time_weight.t())


class _Prepared:
    """Inputs of one live velocity shared by every batch it transports."""

    def __init__(self, live: _LiveVelocity, weeks: int, grid):
        self.weeks = weeks
        self.grid = tuple(grid)
        reference = live.weight
        times = _grid_times(tuple(t for t, _, _ in grid), reference.device, reference.dtype)
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
        column = offset = 0
        for network in live.networks:
            if network.rows:
                self.networks.append(_Network(network, weeks, column, offset, times))
                offset += network.rows
            column += network.rows
        self.hidden = self.networks[0].hidden_weight.shape[0]
        # Every network's context projections side by side, [lead, coordinates, hidden], in
        # one dispatch per step. Row c's table entry: contexts, first row of its network's
        # context weights, network.
        device = reference.device
        self.coordinates = offset
        self.context_index = torch.zeros(offset, _MAX_CONTEXTS, dtype=torch.int32, device=device)
        table, weight_rows = [], 0
        for k, network in enumerate(self.networks):
            n_contexts = network.context_index.shape[1]
            rows = slice(network.offset, network.offset + network.rows)
            self.context_index[rows, :n_contexts] = network.context_index
            table += [(n_contexts, weight_rows, k)] * network.rows
            weight_rows += n_contexts * 8
        self.context_rows = to_device_async(torch.tensor(table, dtype=torch.int32), device)
        self.context_weight = torch.cat([network.context_weight for network in self.networks])
        #: Every network's time projections, ``[steps, networks, hidden]``.
        self.time_projections = torch.stack(
            [network.time_projection for network in self.networks], dim=1
        )


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
    device, dtype = state.device, state.dtype
    lib = _library(device)
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
        # One row per distinct conditioning; a graph may leave no entity live (width 0).
        conditionings = live.weight.numel() // n_live
        lead_rows = index.repeat(n_lead // index.numel())
        weight = live.weight.reshape(conditionings, n_live)[lead_rows]
        active = live.entity_active.reshape(conditionings, n_entities)[lead_rows]
        count = live.entity_count.reshape(conditionings)[lead_rows]
        token_lead = lead_rows.to(torch.int32)
        static_leads = [token_lead] * len(prepared.networks)
    weight, active, count = weight.contiguous(), active.contiguous(), count.contiguous()

    hidden = prepared.hidden
    h4 = hidden // 4
    if device.type == "cuda":
        block_rows, group = _cuda.network_launch(hidden, device)
    else:
        block_rows, group = _BLOCK_ROWS, _NETWORK_THREADS[hidden]
    kernel = getattr(lib, f"network_{hidden}")
    bank = torch.empty(n_lead, weeks, n_live + 3, device=device, dtype=dtype)
    contexts = torch.empty(n_lead, n_live + 1, _TOKEN, device=device, dtype=dtype)
    velocity = torch.empty(n_lead, weeks, n_live, device=device, dtype=dtype)
    projection = torch.empty(n_lead, prepared.coordinates, hidden, device=device, dtype=dtype)
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
        running = [
            (network, static_lead)
            for network, static_lead in zip(prepared.networks, static_leads, strict=True)
            if not (last and network.column + network.rows <= final_from)
        ]
        if running:
            first = running[0][0].offset
            lib.context_projection(
                projection, contexts, prepared.context_index, prepared.context_rows,
                prepared.context_weight, prepared.time_projections[step], n_lead,
                prepared.coordinates, first, n_live, h4, _PROJECTION_DRAWS,
                threads=(h4, (prepared.coordinates - first) * -(-n_lead // _PROJECTION_DRAWS)),
            )  # fmt: skip
        for network, static_lead in running:
            rows = n_lead * weeks * network.rows
            kernel(
                velocity, bank, network.scalar_index, network.scalar_weight, network.static,
                static_lead, projection, network.hidden_weight, network.hidden_bias,
                network.output_weight, network.output_bias, rows, weeks, network.rows,
                network.scalar_index.shape[1], n_live + 3, n_live, network.column,
                prepared.coordinates, network.offset,
                threads=-(-rows // block_rows) * group, group_size=group,
            )  # fmt: skip
        # x <- (x + dt * v) * mask, then the next step's scalar bank.
        lib.step_moments(
            bank, x, velocity, weight, active, count, float(dt), n_lead, weeks, n_live,
            n_entities,
            threads=n_lead * weeks,
        )  # fmt: skip
    if device.type == "mps":
        # Let the GPU start on this transport while the host prepares the next one.
        flush()
    return x.reshape(state.shape)
