"""P0 spike: the scoring segment of the D chain as custom Metal kernels (torch.mps.compile_shader).

One layer's D step begins with three small PyTorch ops that are each a separate GPU kernel: z of the newest key
(project_keys + a strided column write), the index query u (project_queries) and the candidate scores
(candidate_scores). `MetalScore.fused` does all of that in ONE kernel launch; `MetalScore.two_launch` splits the
prologue (u and the z column) from the scan so the two can be compared. Two kernel versions are kept: v1 (one column
per thread, 2-byte loads; the first attempt, 79 GB/s effective at 32K in the smoke run) and v2 (four columns per
thread with 8-byte loads, four times fewer threadgroups, a cheaper u reduction). Scores come back in float32; the z column is
written to the index table in bfloat16 with round-to-nearest-even, exactly where the PyTorch path writes it.

Scope and limits (docs/tasks/04_FUSED_D_KERNEL.md): batch 1, bfloat16 index table [G, r, C] contiguous, r <= 32,
threadgroup of 256 threads. Nothing in the model path uses this module; it exists to measure whether a fused kernel
can approach the bandwidth floor of the scan. bfloat16 tensors are bound as raw 16-bit buffers and converted in the
kernel, so no Metal bfloat type is required.
"""
from __future__ import annotations

import math

import torch

TG = 256          # threads per threadgroup; the kernels assume exactly this (8 simdgroups of 32)
MAX_R = 32        # index width supported by the threadgroup buffers

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline float bf2f(ushort x) { return as_type<float>(((uint)x) << 16); }

inline ushort f2bf(float f) {
    // round to nearest even; NaN keeps a quiet payload, inf stays inf
    uint u = as_type<uint>(f);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        ushort hi = (ushort)(u >> 16);
        return (u & 0xffffu) ? (hi | (ushort)0x40) : hi;
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// One launch: every threadgroup (g, b) rebuilds u[g] cooperatively (the 16 KB of weights per group stay cache
// resident), block 0 of each group writes the newest key's z column, then each thread scores one candidate column.
kernel void score_fused(device ushort* zT            [[buffer(0)]],   // [G, r, C] bf16, read + one column written
                        device const ushort* k_new   [[buffer(1)]],   // [G, d]
                        device const ushort* wk      [[buffer(2)]],   // [G, r, d]
                        device const ushort* q_cat   [[buffer(3)]],   // [G, hd]
                        device const ushort* wq_t    [[buffer(4)]],   // [G, hd, r]  (1/sqrt(r) already folded in)
                        device float* out            [[buffer(5)]],   // [G, M] fp32 scores of columns lo .. lo+M-1
                        constant int& r              [[buffer(6)]],
                        constant int& d              [[buffer(7)]],
                        constant int& hd             [[buffer(8)]],
                        constant int& C              [[buffer(9)]],
                        constant int& lo             [[buffer(10)]],
                        constant int& M              [[buffer(11)]],
                        constant int& write_col      [[buffer(12)]],
                        constant int& nblocks        [[buffer(13)]],
                        uint tg  [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float u_sh[32];
    threadgroup float part[8][32];
    const int g = (int)tg / nblocks;
    const int b = (int)tg - g * nblocks;
    // u[g, j] = sum_f q_cat[g, f] * wq_t[g, f, j]; thread lid handles the features f = lid, lid + tgsize, ...
    float acc[32];
    for (int j = 0; j < 32; ++j) acc[j] = 0.0f;
    for (int f = (int)lid; f < hd; f += (int)tgsize) {
        const float q = bf2f(q_cat[g * hd + f]);
        const device ushort* w = wq_t + ((size_t)g * hd + f) * r;
        for (int j = 0; j < r; ++j) acc[j] += q * bf2f(w[j]);
    }
    const uint sg = lid / 32, lane = lid % 32;
    for (int j = 0; j < r; ++j) {
        const float v = simd_sum(acc[j]);
        if (lane == 0) part[sg][j] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < (uint)r) {
        float s = 0.0f;
        for (uint i = 0; i < tgsize / 32; ++i) s += part[i][lid];
        u_sh[lid] = s;
    }
    if (b == 0 && lid < (uint)r) {   // z column of the newest key: z[g, j] = sum_e wk[g, j, e] * k_new[g, e]
        const device ushort* wrow = wk + ((size_t)g * r + lid) * d;
        float s = 0.0f;
        for (int e = 0; e < d; ++e) s += bf2f(wrow[e]) * bf2f(k_new[g * d + e]);
        zT[((size_t)g * r + lid) * C + write_col] = f2bf(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int m = b * (int)tgsize + (int)lid;
    if (m < M) {   // columns lo+m < hi <= write_col: the scan never touches the column written above
        const device ushort* col = zT + (size_t)g * r * C + (size_t)(lo + m);
        float s = 0.0f;
        for (int j = 0; j < r; ++j) s += u_sh[j] * bf2f(col[(size_t)j * C]);
        out[(size_t)g * M + m] = s;
    }
}

// Two launches: a tiny prologue (G*r threads) computes u and the z column, then the scan reads u from a buffer.
kernel void prologue(device ushort* zT            [[buffer(0)]],
                     device const ushort* k_new   [[buffer(1)]],
                     device const ushort* wk      [[buffer(2)]],
                     device const ushort* q_cat   [[buffer(3)]],
                     device const ushort* wq_t    [[buffer(4)]],
                     device float* u_out          [[buffer(5)]],   // [G, r]
                     constant int& G              [[buffer(6)]],
                     constant int& r              [[buffer(7)]],
                     constant int& d              [[buffer(8)]],
                     constant int& hd             [[buffer(9)]],
                     constant int& C              [[buffer(10)]],
                     constant int& write_col      [[buffer(11)]],
                     uint id [[thread_position_in_grid]]) {
    const int g = (int)id / r, j = (int)id % r;
    if (g >= G) return;
    float s = 0.0f;
    for (int f = 0; f < hd; ++f) s += bf2f(q_cat[g * hd + f]) * bf2f(wq_t[((size_t)g * hd + f) * r + j]);
    u_out[g * r + j] = s;
    const device ushort* wrow = wk + ((size_t)g * r + j) * d;
    float z = 0.0f;
    for (int e = 0; e < d; ++e) z += bf2f(wrow[e]) * bf2f(k_new[g * d + e]);
    zT[((size_t)g * r + j) * C + write_col] = f2bf(z);
}

kernel void score_only(device const ushort* zT     [[buffer(0)]],
                       device const float* u       [[buffer(1)]],   // [G, r]
                       device float* out           [[buffer(2)]],
                       constant int& r             [[buffer(3)]],
                       constant int& C             [[buffer(4)]],
                       constant int& lo            [[buffer(5)]],
                       constant int& M             [[buffer(6)]],
                       constant int& nblocks       [[buffer(7)]],
                       uint tg  [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float u_sh[32];
    const int g = (int)tg / nblocks;
    const int b = (int)tg - g * nblocks;
    if (lid < (uint)r) u_sh[lid] = u[g * r + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int m = b * (int)tgsize + (int)lid;
    if (m < M) {
        const device ushort* col = zT + (size_t)g * r * C + (size_t)(lo + m);
        float s = 0.0f;
        for (int j = 0; j < r; ++j) s += u_sh[j] * bf2f(col[(size_t)j * C]);
        out[(size_t)g * M + m] = s;
    }
}
// ---- v2: each thread scores 4 adjacent columns with 8-byte loads; a threadgroup covers 4*tgsize columns, so a group
// of M columns needs ceil(M / 1024) threadgroups instead of ceil(M / 256). The prologue sums u with 8 feature slices
// per component and one shared-memory reduction instead of 32 simd reductions. Requires tgsize == 256, r <= 32,
// and (C % 4 == 0 and lo % 4 == 0) for aligned vector loads (the host checks and falls back to v1 otherwise).
inline void u_prologue(threadgroup float part[8][32], threadgroup float u_sh[32], device const ushort* q_cat, device const ushort* wq_t,
                       int g, int r, int hd, uint lid) {
    const int j = (int)(lid & 31), s = (int)(lid >> 5);
    float acc = 0.0f;
    if (j < r) {
        for (int f = s; f < hd; f += 8) acc += bf2f(q_cat[g * hd + f]) * bf2f(wq_t[((size_t)g * hd + f) * r + j]);
    }
    part[s][j] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < (uint)r) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += part[i][lid];
        u_sh[lid] = t;
    }
}

inline void scan4(device const ushort* zT, threadgroup float u_sh[32], device float* out, int g, int r, int C, int lo, int M, int m0) {
    const device ushort* base = zT + (size_t)g * r * C + (size_t)(lo + m0);
    device float* o = out + (size_t)g * M + m0;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    if (m0 + 4 <= M) {                       // full vector: columns lo+m0 .. lo+m0+3 < hi
        for (int j = 0; j < r; ++j) {
            const ushort4 v = *((const device ushort4*)(base + (size_t)j * C));
            const float u = u_sh[j];
            s0 += u * bf2f(v.x); s1 += u * bf2f(v.y); s2 += u * bf2f(v.z); s3 += u * bf2f(v.w);
        }
        o[0] = s0; o[1] = s1; o[2] = s2; o[3] = s3;
    } else {                                 // tail: scalar loads, never at or beyond column hi
        const int n = M - m0;
        for (int j = 0; j < r; ++j) {
            const float u = u_sh[j];
            const device ushort* row = base + (size_t)j * C;
            s0 += u * bf2f(row[0]);
            if (n > 1) s1 += u * bf2f(row[1]);
            if (n > 2) s2 += u * bf2f(row[2]);
        }
        o[0] = s0;
        if (n > 1) o[1] = s1;
        if (n > 2) o[2] = s2;
    }
}

kernel void score_fused_v2(device ushort* zT            [[buffer(0)]],
                           device const ushort* k_new   [[buffer(1)]],
                           device const ushort* wk      [[buffer(2)]],
                           device const ushort* q_cat   [[buffer(3)]],
                           device const ushort* wq_t    [[buffer(4)]],
                           device float* out            [[buffer(5)]],
                           constant int& r              [[buffer(6)]],
                           constant int& d              [[buffer(7)]],
                           constant int& hd             [[buffer(8)]],
                           constant int& C              [[buffer(9)]],
                           constant int& lo             [[buffer(10)]],
                           constant int& M              [[buffer(11)]],
                           constant int& write_col      [[buffer(12)]],
                           constant int& nblocks        [[buffer(13)]],
                           uint tg  [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float u_sh[32];
    threadgroup float part[8][32];
    const int g = (int)tg / nblocks;
    const int b = (int)tg - g * nblocks;
    u_prologue(part, u_sh, q_cat, wq_t, g, r, hd, lid);
    if (b == 0 && lid < (uint)r) {
        const device ushort* wrow = wk + ((size_t)g * r + lid) * d;
        float s = 0.0f;
        for (int e = 0; e < d; ++e) s += bf2f(wrow[e]) * bf2f(k_new[g * d + e]);
        zT[((size_t)g * r + lid) * C + write_col] = f2bf(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int m0 = (b * (int)tgsize + (int)lid) * 4;
    if (m0 < M) scan4(zT, u_sh, out, g, r, C, lo, M, m0);
}

kernel void score_only_v2(device const ushort* zT     [[buffer(0)]],
                          device const float* u       [[buffer(1)]],
                          device float* out           [[buffer(2)]],
                          constant int& r             [[buffer(3)]],
                          constant int& C             [[buffer(4)]],
                          constant int& lo            [[buffer(5)]],
                          constant int& M             [[buffer(6)]],
                          constant int& nblocks       [[buffer(7)]],
                          uint tg  [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float u_sh[32];
    const int g = (int)tg / nblocks;
    const int b = (int)tg - g * nblocks;
    if (lid < (uint)r) u_sh[lid] = u[g * r + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int m0 = (b * (int)tgsize + (int)lid) * 4;
    if (m0 < M) scan4(zT, u_sh, out, g, r, C, lo, M, m0);
}
"""


def available() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def _check(zT, k_new, wk, q_cat, wq_t, lo, hi, write_col):
    if zT.dim() != 3 or zT.dtype != torch.bfloat16 or not zT.is_contiguous():
        raise ValueError("zT must be a contiguous bfloat16 [G, r, C] tensor (batch 1)")
    G, r, C = zT.shape
    if r > MAX_R:
        raise ValueError(f"index width {r} exceeds the kernel's {MAX_R}")
    if tuple(k_new.shape) != (G, wk.shape[2]) or tuple(wk.shape[:2]) != (G, r):
        raise ValueError("k_new must be [G, d] and wk [G, r, d]")
    if q_cat.dim() != 2 or q_cat.shape[0] != G or tuple(wq_t.shape) != (G, q_cat.shape[1], r):
        raise ValueError("q_cat must be [G, hd] and wq_t [G, hd, r]")
    for t in (k_new, wk, q_cat, wq_t):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != zT.device:
            raise ValueError("every input must be contiguous bfloat16 on the same device as zT")
    if not (0 <= lo <= hi <= C and 0 <= write_col < C):
        raise ValueError(f"need 0 <= lo <= hi <= C and 0 <= write_col < C, got lo={lo} hi={hi} write_col={write_col} C={C}")
    if lo <= write_col < hi:
        raise ValueError("write_col must lie outside the scanned range [lo, hi)")
    return G, r, C, int(wk.shape[2]), int(q_cat.shape[1])


class MetalScore:
    """Compiled kernels for the scoring segment. `fused` is one launch, `two_launch` is prologue + scan."""

    def __init__(self):
        if not available():
            raise RuntimeError("needs MPS and torch.mps.compile_shader")
        self.lib = torch.mps.compile_shader(SOURCE)
        if self.lib.score_fused.max_threads_per_threadgroup < TG:
            raise RuntimeError("device cannot run 256-thread threadgroups")

    @staticmethod
    def _grid(G: int, M: int, cols_per_thread: int):
        nblocks = max(1, math.ceil(M / (TG * cols_per_thread)))
        return nblocks, G * nblocks * TG

    @staticmethod
    def vector_ok(C: int, lo: int) -> bool:
        """v2 needs 8-byte aligned column quads: C % 4 == 0 and lo % 4 == 0 (lo is the fixed prefix 4 in this project)."""
        return C % 4 == 0 and lo % 4 == 0

    def fused(self, zT, k_new, wk, q_cat, wq_t, lo: int, hi: int, write_col: int, out=None, version: int = 2) -> torch.Tensor:
        """Write z of the newest key into zT[:, :, write_col] and return fp32 scores [G, hi - lo] of the candidate columns.

        version 2 (default): 4 columns per thread with 8-byte loads, falls back to version 1 when the layout is not
        aligned; version 1: one column per thread with 2-byte loads (kept as the recorded first attempt).
        """
        G, r, C, d, hd = _check(zT, k_new, wk, q_cat, wq_t, lo, hi, write_col)
        M = hi - lo
        if out is None:
            out = torch.empty(G, max(M, 1), dtype=torch.float32, device=zT.device)
        if M <= 0:
            return out[:, :0]
        if version == 2 and self.vector_ok(C, lo):
            nblocks, threads = self._grid(G, M, 4)
            self.lib.score_fused_v2(zT, k_new, wk, q_cat, wq_t, out, r, d, hd, C, lo, M, write_col, nblocks, threads=threads, group_size=TG)
        else:
            nblocks, threads = self._grid(G, M, 1)
            self.lib.score_fused(zT, k_new, wk, q_cat, wq_t, out, r, d, hd, C, lo, M, write_col, nblocks, threads=threads, group_size=TG)
        return out

    def two_launch(self, zT, k_new, wk, q_cat, wq_t, lo: int, hi: int, write_col: int, out=None, u_buf=None, version: int = 2) -> torch.Tensor:
        G, r, C, d, hd = _check(zT, k_new, wk, q_cat, wq_t, lo, hi, write_col)
        M = hi - lo
        if u_buf is None:
            u_buf = torch.empty(G, r, dtype=torch.float32, device=zT.device)
        if out is None:
            out = torch.empty(G, max(M, 1), dtype=torch.float32, device=zT.device)
        self.lib.prologue(zT, k_new, wk, q_cat, wq_t, u_buf, G, r, d, hd, C, write_col, threads=G * r, group_size=min(G * r, TG))
        if M <= 0:
            return out[:, :0]
        if version == 2 and self.vector_ok(C, lo):
            nblocks, threads = self._grid(G, M, 4)
            self.lib.score_only_v2(zT, u_buf, out, r, C, lo, M, nblocks, threads=threads, group_size=TG)
        else:
            nblocks, threads = self._grid(G, M, 1)
            self.lib.score_only(zT, u_buf, out, r, C, lo, M, nblocks, threads=threads, group_size=TG)
        return out


def model_views(zT4, k_pre_new4, q_pre5, indexer):
    """Adapt the model-side tensors (batch 1) to the kernel's 2-D / 3-D views. No copies except q_cat when needed."""
    if zT4.shape[0] != 1:
        raise ValueError("P0 kernels take batch 1")
    zT = zT4[0]                                                  # [G, r, C]
    k_new = k_pre_new4[0, :, 0, :]                               # [G, d]
    B, G, h, T, d = q_pre5.shape
    q_cat = q_pre5[0].permute(0, 2, 1, 3).reshape(G, T * h * d)  # [G, h*d] for T == 1 (same order as project_queries)
    return zT, k_new.contiguous(), indexer.wk, q_cat.contiguous(), indexer.wq_t_scaled


def reference_scores(zT, k_new, wk, q_cat, wq_t, lo: int, hi: int):
    """float32 CPU reference from the same bf16 inputs: u, z_new and the candidate scores."""
    zf, kf, wkf, qf, wqf = (t.detach().float().cpu() for t in (zT, k_new, wk, q_cat, wq_t))
    u = torch.einsum("gf,gfr->gr", qf, wqf)                      # [G, r]
    z_new = torch.einsum("grd,gd->gr", wkf, kf)                  # [G, r]
    scores = torch.einsum("gr,grm->gm", u, zf[:, :, lo:hi])      # [G, M]
    return u, z_new, scores
