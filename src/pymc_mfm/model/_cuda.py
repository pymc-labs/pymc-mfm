"""NVIDIA GPU (CUDA) kernels for the sampling-time live velocity.

The CUDA counterparts of the Apple GPU kernels in :mod:`pymc_mfm.model._metal`, compiled at
run time by NVRTC, the CUDA runtime compiler that PyTorch's CUDA wheels ship, so no CUDA
toolkit or compiler is needed on Linux or Windows. :func:`pymc_mfm.model._metal.integrate`
launches them in the same order with the same inputs. Results agree with the eager velocity
up to float32 round-off.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import warnings
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache

import torch
from torch import Tensor

#: Network tile shape: rows of one block, rows per thread, first-layer slice width and the
#: minimum blocks per multiprocessor the compiler budgets registers for.
_TILES = (64, 8, 16, 2)
#: 256-wide networks on GPUs whose multiprocessors hold four double-buffered 32-row blocks in
#: shared memory run four of them at a time, with the same threads and registers: 3-7% faster
#: on an A100, H100 and B200. An A10 or L4 holds two such blocks and runs them 17-26% slower
#: than 64-row blocks; the draws are the same either way.
_WIDE_TILES = {(8, 0): (32, 8, 16, 4), (9, 0): (32, 8, 16, 4), (10, 0): (32, 8, 16, 4)}
#: Hidden widths with a network kernel.
NETWORK_WIDTHS = (16, 32, 64, 128, 256)


def _tiles(hidden: int, arch: tuple[int, int]) -> tuple[int, int, int, int]:
    return _WIDE_TILES.get(arch, _TILES) if hidden == 256 else _TILES


def network_launch(hidden: int, device: torch.device) -> tuple[int, int]:
    """Rows and threads of one ``network_<hidden>`` block on the CUDA ``device``."""
    block_rows, rows_per_thread, _, _ = _tiles(hidden, torch.cuda.get_device_capability(device))
    return block_rows, (block_rows // rows_per_thread) * min(hidden, 32)


_COMMON = r"""
#define FULL_MASK 0xffffffffu
typedef long long i64;

// Fast exponential and division: within a few float32 ulp of v / (1 + exp(-v)).
__device__ __forceinline__ float silu(float v) { return __fdividef(v, 1.0f + __expf(-v)); }
"""

_SOURCES = {
    "trajectory_context": r"""
// Online-softmax state of a run of weeks: max score m, denominator d and the weighted
// token sum a[8], all relative to m.
struct Pool {
    float m;
    float d;
    float a[8];
};

__device__ __forceinline__ void merge(Pool& p, const Pool& q) {
    if (q.d == 0.0f) return;
    if (p.d == 0.0f) {
        p = q;
        return;
    }
    const float m = fmaxf(p.m, q.m);
    const float sp = expf(p.m - m), sq = expf(q.m - m);
    p.d = p.d * sp + q.d * sq;
    for (int e = 0; e < 8; ++e) p.a[e] = p.a[e] * sp + q.a[e] * sq;
    p.m = m;
}

// Week-pooled trajectory context of each live coordinate. S lanes of a warp share one
// (l, c) pair: each pools every S-th week with an online softmax, then the lanes merge
// their states. ctx[l, NL] is the shared zero context. Every thread reaches the shuffles.
extern "C" __global__ void trajectory_context(
    float* ctx, const float* x, const float* latent, const float* token_base,
    const int* token_lead, const float* weight, const float* p, int L, int T, int NL, int S)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = gid / S, lane = gid % S;
    const bool live = pair < L * NL;
    const int l = live ? pair / NL : 0, c = live ? pair % NL : 0;
    Pool pool;
    pool.m = 0.0f;
    pool.d = 0.0f;
    for (int e = 0; e < 8; ++e) pool.a[e] = 0.0f;
    const float* tb = token_base + (i64)token_lead[l] * T * 8;
    for (int w = lane; live && w < T; w += S) {
        const i64 at = ((i64)l * T + w) * NL + c;
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
    for (int offset = 1; offset < S; offset <<= 1) {
        Pool other;
        other.m = __shfl_xor_sync(FULL_MASK, pool.m, offset);
        other.d = __shfl_xor_sync(FULL_MASK, pool.d, offset);
        for (int e = 0; e < 8; ++e) other.a[e] = __shfl_xor_sync(FULL_MASK, pool.a[e], offset);
        merge(pool, other);
    }
    if (!live || lane != 0) return;
    const float mask = weight[(i64)l * NL + c];
    float* out = ctx + ((i64)l * (NL + 1) + c) * 8;
    for (int o = 0; o < 8; ++o) {
        float v = p[89 + o];
        for (int e = 0; e < 8; ++e) v += p[25 + o * 8 + e] * (pool.a[e] / pool.d);
        out[o] = v * mask;
    }
    if (c == 0) {
        float* zero = ctx + ((i64)l * (NL + 1) + NL) * 8;
        for (int o = 0; o < 8; ++o) zero[o] = 0.0f;
    }
}
""",
    "step_moments": r"""
// Optional Euler step x <- (x + dt * v) * mask, then the scalar bank row
// [x_live..., 0, entity mean, entity std] of the state. One thread per (l, w).
extern "C" __global__ void step_moments(
    float* bank, float* x, const float* v, const float* mask, const float* active,
    const float* count, double dt, int L, int T, int NL, int NE)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= L * T) return;
    const int l = gid / T;
    const float step = (float)dt;
    float* xr = x + (i64)gid * NL;
    if (step != 0.0f) {
        const float* vr = v + (i64)gid * NL;
        const float* mr = mask + (i64)l * NL;
        for (int i = 0; i < NL; ++i) xr[i] = (xr[i] + step * vr[i]) * mr[i];
    }
    const float* a = active + (i64)l * NE;
    float* b = bank + (i64)gid * (NL + 3);
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
    b[NL + 2] = sqrtf(ss) / sqrtf(n);
}
""",
    "context_projection": r"""
// First-layer terms of row c that do not vary over weeks, for the rows c in [c0, CT) of
// every network of one step, side by side as [L, CT, H] (see the Metal kernel). One
// thread per (4 hidden units, row c, run of D draws) keeps the row's weights in registers.
extern "C" __global__ void context_projection(
    float4* out, const float* ctx, const int* cidx, const int* rows, const float4* wct,
    const float4* tproj, int L, int CT, int c0, int NL, int H4, int D)
{
    const int NCMAX = 3;
    const int j4 = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int width = CT - c0;
    const int c = c0 + gy % width, l0 = (gy / width) * D;
    if (j4 >= H4 || l0 >= L) return;
    const int NC = rows[c * 3], wrow = rows[c * 3 + 1], net = rows[c * 3 + 2];
    float4 w[NCMAX * 8];
    int ci[NCMAX];
#pragma unroll
    for (int k = 0; k < NCMAX; ++k) {
        if (k < NC) {
            ci[k] = cidx[c * NCMAX + k];
#pragma unroll
            for (int e = 0; e < 8; ++e) w[k * 8 + e] = wct[(wrow + k * 8 + e) * H4 + j4];
        }
    }
    const float4 t = tproj[net * H4 + j4];
    const int l1 = min(L, l0 + D);
    for (int l = l0; l < l1; ++l) {
        float4 acc = t;
#pragma unroll
        for (int k = 0; k < NCMAX; ++k) {
            if (k < NC) {
                const float* cb = ctx + ((i64)l * (NL + 1) + ci[k]) * 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float s = cb[e];
                    acc.x += s * w[k * 8 + e].x;
                    acc.y += s * w[k * 8 + e].y;
                    acc.z += s * w[k * 8 + e].z;
                    acc.w += s * w[k * 8 + e].w;
                }
            }
        }
        out[((i64)l * CT + c) * H4 + j4] = acc;
    }
}
""",
}

# One coordinate network on a block of BM rows r = (l * T + w) * C + c:
#   v[l, w, col0 + c] = w3 . silu(silu(first(r)) @ W2t + b2) + b3,
#   first(r) = static[r] + cproj[l, c0 + c] + scalars(r) @ ws.
# Each BK-wide slice of the first layer is computed into shared memory as the product needs
# it, with the matching slice of W2t; each thread accumulates TM rows x TN columns in
# registers and the output layer reduces across the warp, so no [rows, HN] activation
# reaches device memory. On sm_80 and newer the slices are double-buffered: the next W2t
# slice is copied asynchronously (cp.async) during the current product and the next
# first-layer slice is built right after it, with one barrier per slice.
_NETWORK = r"""
#if __CUDA_ARCH__ >= 800
#define BUFFERS 2
#else
#define BUFFERS 1
#endif

// 16-byte global-to-shared copy, asynchronous on sm_80 and newer.
__device__ __forceinline__ void copy16(float* shared, const float* global) {
#if __CUDA_ARCH__ >= 800
    unsigned address;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
                 : "=r"(address) : "l"(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(global));
#else
    *reinterpret_cast<float4*>(shared) = *reinterpret_cast<const float4*>(global);
#endif
}

__device__ __forceinline__ void copies_done() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

template <int HN, int BM, int BK, int TM>
__device__ __forceinline__ void network_body(
    float* __restrict__ v, const float* __restrict__ bank, const int* __restrict__ sidx,
    const float* __restrict__ ws, const float* __restrict__ stat,
    const int* __restrict__ static_lead, const float* __restrict__ cproj,
    const float* __restrict__ w2t, const float* __restrict__ b2, const float* __restrict__ w3,
    const float* __restrict__ b3, int R, int T, int C, int NS, int NB, int NL, int col0, int CT,
    int c0)
{
    const int NSMAX = 8;
    const int COLS = HN < 32 ? HN : 32;
    const int TN = HN / COLS;
    const int NT = (BM / TM) * COLS;
    const int STEPS = HN / BK;
    // With TN a multiple of 4 a thread owns float4 runs of columns, one run per 4*COLS.
    const bool VEC = TN % 4 == 0;
    // First-layer rows padded by 4 floats: fewer bank conflicts when they are stored.
    __shared__ alignas(16) float As[BUFFERS][BK][BM + 4];
    __shared__ alignas(16) float Bs[BUFFERS][BK][HN];
    __shared__ float scal[BM][NSMAX];
    __shared__ i64 soff[BM];
    __shared__ i64 coff[BM];

    const int tid = threadIdx.x, tx = tid % COLS, ty = tid / COLS;
    auto column = [&](int j) {
        return VEC ? (j / 4) * (COLS * 4) + tx * 4 + j % 4 : j * COLS + tx;
    };
    const int row0 = blockIdx.x * BM;
    for (int r = tid; r < BM; r += NT) {
        const int row = row0 + r;
        if (row < R) {
            const int c = row % C, lw = row / C, w = lw % T, l = lw / T;
            soff[r] = (((i64)static_lead[l] * T + w) * C + c) * HN;
            coff[r] = ((i64)l * CT + c0 + c) * HN;
            for (int i = 0; i < NS; ++i) scal[r][i] = bank[(i64)lw * NB + sidx[c * NS + i]];
        } else {
            soff[r] = -1;
        }
    }
    __syncthreads();
    // First-layer slice at k0 into buffer buf: each thread builds 4 consecutive values of one
    // row from 128-bit loads.
    auto first = [&](int k0, int buf) {
        for (int e = tid; e < BM * BK / 4; e += NT) {
            const int r = e / (BK / 4), q = (e % (BK / 4)) * 4, k = k0 + q;
            const i64 so = soff[r];
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            if (so >= 0) {
                const float4 s = *reinterpret_cast<const float4*>(stat + so + k);
                const float4 p = *reinterpret_cast<const float4*>(cproj + coff[r] + k);
                a0 = s.x + p.x;
                a1 = s.y + p.y;
                a2 = s.z + p.z;
                a3 = s.w + p.w;
                for (int i = 0; i < NS; ++i) {
                    const float c = scal[r][i];
                    const float4 w = *reinterpret_cast<const float4*>(ws + i * HN + k);
                    a0 += c * w.x;
                    a1 += c * w.y;
                    a2 += c * w.z;
                    a3 += c * w.w;
                }
                a0 = silu(a0);
                a1 = silu(a1);
                a2 = silu(a2);
                a3 = silu(a3);
            }
            As[buf][q][r] = a0;
            As[buf][q + 1][r] = a1;
            As[buf][q + 2][r] = a2;
            As[buf][q + 3][r] = a3;
        }
    };
    auto weights = [&](int k0, int buf) {
        float* to = &Bs[buf][0][0];
        const float* from = w2t + (i64)k0 * HN;
        for (int e = tid; e < BK * HN / 4; e += NT) copy16(to + 4 * e, from + 4 * e);
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    }
    weights(0, 0);
    first(0, 0);
    copies_done();
    __syncthreads();
    for (int step = 0; step < STEPS; ++step) {
        const int cur = BUFFERS == 2 ? step & 1 : 0, next = BUFFERS == 2 ? cur ^ 1 : 0;
        const bool more = step + 1 < STEPS;
        if (BUFFERS == 2 && more) weights((step + 1) * BK, next);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int h = 0; h < TM / 4; ++h) {
                const float4 q = *reinterpret_cast<const float4*>(&As[cur][kk][ty * TM + 4 * h]);
                a[4 * h] = q.x;
                a[4 * h + 1] = q.y;
                a[4 * h + 2] = q.z;
                a[4 * h + 3] = q.w;
            }
            if (VEC) {
#pragma unroll
                for (int g = 0; g < TN / 4; ++g) {
                    const float4 q =
                        *reinterpret_cast<const float4*>(&Bs[cur][kk][g * COLS * 4 + tx * 4]);
                    b[4 * g] = q.x;
                    b[4 * g + 1] = q.y;
                    b[4 * g + 2] = q.z;
                    b[4 * g + 3] = q.w;
                }
            } else {
#pragma unroll
                for (int j = 0; j < TN; ++j) b[j] = Bs[cur][kk][j * COLS + tx];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
        }
        if (more) {
            if (BUFFERS == 1) {
                // One buffer: every thread must finish reading it before it is refilled.
                __syncthreads();
                weights((step + 1) * BK, 0);
            }
            first((step + 1) * BK, next);
            copies_done();
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = column(j);
            s += silu(acc[i][j] + b2[col]) * w3[col];
        }
#pragma unroll
        for (int offset = COLS / 2; offset > 0; offset >>= 1) {
            s += __shfl_xor_sync(FULL_MASK, s, offset);
        }
        const int r = ty * TM + i;
        if (tx == 0 && soff[r] >= 0) {
            const int row = row0 + r;
            v[(i64)(row / C) * NL + col0 + row % C] = s + b3[0];
        }
    }
}

extern "C" __global__ void __launch_bounds__(@THREADS@, @MIN_BLOCKS@) network_@HIDDEN@(
    float* v, const float* bank, const int* sidx, const float* ws, const float* stat,
    const int* static_lead, const float* cproj, const float* w2t, const float* b2,
    const float* w3, const float* b3, int R, int T, int C, int NS, int NB, int NL, int col0,
    int CT, int c0)
{
    network_body<@HIDDEN@, @BLOCK_ROWS@, @SLICE@, @ROWS_PER_THREAD@>(
        v, bank, sidx, ws, stat, static_lead, cproj, w2t, b2, w3, b3, R, T, C, NS, NB, NL,
        col0, CT, c0);
}
"""


def available(device: torch.device) -> bool:
    """Whether sampling on the CUDA ``device`` can run these kernels.

    The kernels assume 32-lane warps (CUDA, not ROCm) and run on the current CUDA device,
    whose context loads them. They compile once per device and process; if NVRTC cannot
    build them for a GPU, sampling there warns once and keeps PyTorch operations.
    """
    return (
        torch.version.cuda is not None
        and torch.version.hip is None
        and torch.cuda.is_available()
        and (device.index is None or device.index == torch.cuda.current_device())
        and _compiles(torch.cuda.current_device())
    )


@lru_cache(maxsize=None)
def _compiles(index: int) -> bool:
    try:
        _compiled("step_moments", index)
    except (AttributeError, OSError, RuntimeError) as error:
        warnings.warn(
            f"NVIDIA GPU sampling kernels are unavailable on cuda:{index} ({error}); sampling "
            "runs PyTorch operations on the GPU instead",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


@lru_cache(maxsize=1)
def _nvrtc() -> ctypes.CDLL:
    """The NVRTC library of PyTorch's CUDA build.

    PyTorch 2.9 and newer find it themselves. PyTorch 2.8 looks only for an unversioned
    ``libnvrtc.so``, which pip's CUDA wheels do not ship, and 2.6 and 2.7 have no lookup at
    all; their CUDA libraries load NVRTC under its versioned name, so that name finds the
    same copy. Every CUDA 11 release names it 11.2.
    """
    from torch.cuda import _utils

    try:
        return _utils._get_nvrtc_library()
    except (AttributeError, OSError):
        pass
    major = int(torch.version.cuda.split(".")[0])
    if sys.platform == "win32":
        names = ["nvrtc64_112_0.dll" if major == 11 else f"nvrtc64_{major}0_0.dll"]
    else:
        names = [f"libnvrtc.so.{'11.2' if major == 11 else major}", "libnvrtc.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError(f"NVRTC not found (tried {', '.join(names)})")


def _cubin(source: str, name: str, arch: tuple[int, int], nvrtc: ctypes.CDLL) -> bytes:
    """Kernel ``name`` compiled by NVRTC into machine code for compute capability ``arch``.

    The sources use only built-in CUDA types and functions, so no CUDA toolkit or headers
    are needed: PyTorch's CUDA wheels ship the NVRTC library. NVRTC needs no CUDA context,
    so this runs on any thread.
    """
    major, minor = arch
    # C++17 is NVRTC's default from CUDA 12; CUDA 11's NVRTC needs it spelled out.
    options = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
    program = ctypes.c_void_p()
    _check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), source.encode(), name.encode(), 0, None, None
        ),
        "NVRTC",
    )
    try:
        status = nvrtc.nvrtcCompileProgram(
            program, len(options), (ctypes.c_char_p * len(options))(*options)
        )
        if status != 0:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"NVRTC could not compile {name}: {log.value.decode()}")
        size = ctypes.c_size_t()
        _check(nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)), "NVRTC")
        binary = ctypes.create_string_buffer(size.value)
        _check(nvrtc.nvrtcGetCUBIN(program, binary), "NVRTC")
        return binary.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def _load(binary: bytes, name: str) -> ctypes.c_void_p:
    """Load compiled kernel ``name`` into the context PyTorch made current; its function."""
    torch.cuda.current_stream()  # PyTorch's context is current on this thread
    driver = _driver()
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    _check(driver.cuModuleLoadData(ctypes.byref(module), binary), "CUDA driver")
    _check(driver.cuModuleGetFunction(ctypes.byref(function), module, name.encode()), "CUDA driver")
    return function


#: NVRTC builds by (kernel, compute capability), running or done; see :func:`prefetch`.
_BUILDS: dict[tuple[str, int, int], Future[bytes]] = {}
_BUILDS_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _builder() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(4, thread_name_prefix="pymc-mfm-nvrtc")


def _build(name: str, index: int) -> Future[bytes]:
    """The NVRTC build of kernel ``name`` for device ``index``, started on first request."""
    arch = torch.cuda.get_device_capability(index)
    with _BUILDS_LOCK:
        build = _BUILDS.get((name, *arch))
        if build is None:
            build = _builder().submit(_cubin, _source(name, arch), name, arch, _nvrtc())
            _BUILDS[(name, *arch)] = build
    return build


def prefetch(device: torch.device, hidden_sizes: Iterable[int]) -> None:
    """Start building every kernel that sampling on ``device`` launches, in the background.

    ``hidden_sizes`` are the widths of the flow networks. NVRTC takes about a second for
    these kernels when no compiled copy is cached on the machine yet (a machine's first
    process; fresh notebook, CI or container sessions); built in background threads, they
    overlap each other and the call's encoder and other first-use work. First uses wait
    only for unfinished builds.
    """
    if torch.version.cuda is None or torch.version.hip is not None:
        return
    index = torch.cuda.current_device()
    if device.index not in (None, index):
        return
    networks = [f"network_{hidden}" for hidden in sorted(hidden_sizes) if hidden in NETWORK_WIDTHS]
    try:
        for name in (*networks, "step_moments", "trajectory_context", "context_projection"):
            _build(name, index)
    except (AttributeError, OSError, RuntimeError):
        pass  # the availability check reports why the kernels are unavailable


@lru_cache(maxsize=1)
def _driver() -> ctypes.CDLL:
    """The NVIDIA driver library, with the launch call's argument types declared."""
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    driver.cuLaunchKernel.argtypes = [ctypes.c_void_p, *[ctypes.c_uint] * 7, ctypes.c_void_p]
    driver.cuLaunchKernel.argtypes += [ctypes.c_void_p, ctypes.c_void_p]
    return driver


def _check(status: int, library: str) -> None:
    if status != 0:
        raise RuntimeError(f"{library} error {status}")


class _Kernel:
    """Launch one compiled kernel with the Metal library's call convention.

    ``threads`` is the total thread count, or ``(x, y)`` for a 2-D grid; ``group_size`` is
    the block size the kernel relies on, if any. Python ints pass as ``int``, floats as
    ``double`` and tensors as device pointers, on PyTorch's current stream.
    """

    def __init__(self, function: ctypes.c_void_p):
        self._function = function

    def __call__(self, *args, threads: int | tuple[int, int], group_size: int | None = None):
        if isinstance(threads, tuple):
            columns, rows = threads
            block_x = min(columns, 64)
            block = (block_x, max(1, 256 // block_x), 1)
            grid = (-(-columns // block[0]), -(-rows // block[1]), 1)
        else:
            width = group_size if group_size is not None and group_size >= 128 else 256
            block = (width, 1, 1)
            grid = (-(-threads // width), 1, 1)
        values = [
            ctypes.c_void_p(arg.data_ptr())
            if isinstance(arg, Tensor)
            else ctypes.c_double(arg)
            if isinstance(arg, float)
            else ctypes.c_int(arg)
            for arg in args
        ]
        pointers = (ctypes.c_void_p * len(values))(*map(ctypes.addressof, values))
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            _driver().cuLaunchKernel(self._function, *grid, *block, 0, stream, pointers, None),
            "CUDA launch",
        )


class _Library:
    """The CUDA kernels by name for the current device, each compiled by NVRTC on first use."""

    def __getattr__(self, name: str) -> _Kernel:
        return _compiled(name, torch.cuda.current_device())


def _network_source(
    hidden: int, block_rows: int, rows_per_thread: int, slice_: int, min_blocks: int
) -> str:
    """Source of ``network_<hidden>`` for one tile shape."""
    threads = (block_rows // rows_per_thread) * min(hidden, 32)
    source = _NETWORK
    for placeholder, value in (
        ("HIDDEN", hidden),
        ("BLOCK_ROWS", block_rows),
        ("ROWS_PER_THREAD", rows_per_thread),
        ("SLICE", slice_),
        ("MIN_BLOCKS", min_blocks),
        ("THREADS", threads),
    ):
        source = source.replace(f"@{placeholder}@", str(value))
    return _COMMON + source


def _source(name: str, arch: tuple[int, int]) -> str:
    if name.startswith("network_"):
        hidden = int(name.removeprefix("network_"))
        return _network_source(hidden, *_tiles(hidden, arch))
    return _COMMON + _SOURCES[name]


@lru_cache(maxsize=None)
def _compiled(name: str, index: int) -> _Kernel:
    """Kernel ``name`` loaded on CUDA device ``index``, which must be the current device."""
    return _Kernel(_load(_build(name, index).result(), name))


_LIBRARY = _Library()


def library() -> _Library:
    return _LIBRARY


def to_device_async(host: Tensor, device: torch.device) -> Tensor:
    """Copy ``host`` to the CUDA ``device`` without waiting for queued work.

    A copy from pinned memory is asynchronous, and PyTorch keeps a pinned block alive until
    the copies that read it have run; pageable memory is pinned first.
    """
    if not host.is_pinned():
        host = host.pin_memory()
    return host.to(device, non_blocking=True)
