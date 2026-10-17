"""P1 spike: the D chain's gather-free subset attention as Metal kernels (torch.mps.compile_shader).

The G1a path reads the selected K rows and V rows with two flattened index_select calls and then runs SDPA over the
gathered subset (three kernels, two materialised copies). `MetalAttend` instead walks the row list and reads each K/V
row straight from the cache by its original position: one simdgroup handles one row at a time with 32 lanes x 4 dims,
the h query heads of a group share every row load (GQA), each simdgroup keeps an online-softmax state per head, the
simdgroups of a threadgroup are merged in threadgroup memory, and with more than one partition per group a tiny combine
kernel merges the partial (m, l, acc) triples. Accumulation is float32; the output is float32 or bfloat16 (RNE).

Scope (docs/tasks/04_FUSED_D_KERNEL.md, P1): batch 1, head_dim 128 (32 lanes x 4), h <= 4, bfloat16 K/V/q, int32 row
positions without duplicates (F ∪ top-k never repeats a position). Rows are read by original position; nothing is
renumbered and no RoPE is applied. Nothing in the model path uses this module.
"""
from __future__ import annotations

import math

import torch

TG = 256          # 8 simdgroups of 32 lanes
HEAD_DIM = 128    # 32 lanes x 4 dims
MAX_H = 4
MAX_P = 64

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

inline float bf2f(ushort x) { return as_type<float>(((uint)x) << 16); }
inline ushort f2bf(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7f800000u) == 0x7f800000u) { ushort hi = (ushort)(u >> 16); return (u & 0xffffu) ? (hi | (ushort)0x40) : hi; }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}
inline float4 load4(const device ushort* p) {
    const ushort4 v = *((const device ushort4*)p);
    return float4(bf2f(v.x), bf2f(v.y), bf2f(v.z), bf2f(v.w));
}

#define MAXH 4

// One threadgroup per (group g, partition p). Lane = 4 dims of the 128-dim head. Each simdgroup walks every 8th row of
// the partition, four rows at a time (eight independent 8-byte loads in flight per lane), scores all h heads from one
// K row read and accumulates all h heads from one V row read.
kernel void attend_partial(device const ushort* K      [[buffer(0)]],   // [G, C, 128]
                           device const ushort* V      [[buffer(1)]],   // [G, C, 128]
                           device const ushort* q      [[buffer(2)]],   // [G, h, 128]
                           device const int*    rows   [[buffer(3)]],   // [G, S] original positions
                           device float* part          [[buffer(4)]],   // [G, P, h, 130]: m, l, acc[128]
                           device ushort* out_bf       [[buffer(5)]],   // [G, h, 128] bf16 (P == 1 and out_bf16)
                           device float*  out_f        [[buffer(6)]],   // [G, h, 128] fp32 (P == 1 and !out_bf16)
                           constant int& C             [[buffer(7)]],
                           constant int& h             [[buffer(8)]],
                           constant int& S             [[buffer(9)]],
                           constant int& P             [[buffer(10)]],
                           constant float& scale       [[buffer(11)]],
                           constant int& out_bf16      [[buffer(12)]],
                           constant int& row_stride    [[buffer(13)]],   // elements between consecutive positions (128, or 256 when K and V are interleaved)
                           constant int& v_off         [[buffer(14)]],   // element offset of the V row inside a position (0, or 128 when interleaved)
                           uint tg  [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float sh_m[8][MAXH];
    threadgroup float sh_l[8][MAXH];
    threadgroup float sh_acc[8][MAXH][128];
    const int g = (int)tg / P, p = (int)tg - g * P;
    const int sg = (int)(lid >> 5), lane = (int)(lid & 31);
    const int Sp = (S + P - 1) / P;
    const int s_begin = p * Sp, s_end = min(S, s_begin + Sp);
    float4 qv[MAXH];
    for (int i = 0; i < MAXH; ++i) qv[i] = float4(0.0f);
    for (int i = 0; i < h; ++i) qv[i] = load4(q + ((size_t)g * h + i) * 128 + lane * 4) * scale;   // scale folded into q
    float m[MAXH], l[MAXH];
    float4 acc[MAXH];
    for (int i = 0; i < MAXH; ++i) { m[i] = -INFINITY; l[i] = 0.0f; acc[i] = float4(0.0f); }
    const device int* rg = rows + (size_t)g * S;
    const size_t gbase = (size_t)g * C * row_stride;
    // four rows per iteration: positions s_begin + sg + 8*t, t = 4*iter .. 4*iter+3
    int s = s_begin + sg;
    for (; s + 24 < s_end; s += 32) {
        const int p0 = rg[s], p1 = rg[s + 8], p2 = rg[s + 16], p3 = rg[s + 24];
        const float4 k0 = load4(K + gbase + (size_t)p0 * row_stride + lane * 4);
        const float4 k1 = load4(K + gbase + (size_t)p1 * row_stride + lane * 4);
        const float4 k2 = load4(K + gbase + (size_t)p2 * row_stride + lane * 4);
        const float4 k3 = load4(K + gbase + (size_t)p3 * row_stride + lane * 4);
        const float4 v0 = load4(V + gbase + (size_t)p0 * row_stride + v_off + lane * 4);
        const float4 v1 = load4(V + gbase + (size_t)p1 * row_stride + v_off + lane * 4);
        const float4 v2 = load4(V + gbase + (size_t)p2 * row_stride + v_off + lane * 4);
        const float4 v3 = load4(V + gbase + (size_t)p3 * row_stride + v_off + lane * 4);
        for (int i = 0; i < h; ++i) {
            const float s0 = simd_sum(dot(qv[i], k0)), s1 = simd_sum(dot(qv[i], k1));
            const float s2 = simd_sum(dot(qv[i], k2)), s3 = simd_sum(dot(qv[i], k3));
            const float mn = max(m[i], max(max(s0, s1), max(s2, s3)));
            const float a = exp(m[i] - mn);
            const float e0 = exp(s0 - mn), e1 = exp(s1 - mn), e2 = exp(s2 - mn), e3 = exp(s3 - mn);
            l[i] = l[i] * a + (e0 + e1 + e2 + e3);
            acc[i] = acc[i] * a + e0 * v0 + e1 * v1 + e2 * v2 + e3 * v3;
            m[i] = mn;
        }
    }
    for (; s < s_end; s += 8) {   // tail, one row at a time
        const int p0 = rg[s];
        const float4 k0 = load4(K + gbase + (size_t)p0 * row_stride + lane * 4);
        const float4 v0 = load4(V + gbase + (size_t)p0 * row_stride + v_off + lane * 4);
        for (int i = 0; i < h; ++i) {
            const float s0 = simd_sum(dot(qv[i], k0));
            const float mn = max(m[i], s0);
            const float a = exp(m[i] - mn), e0 = exp(s0 - mn);
            l[i] = l[i] * a + e0;
            acc[i] = acc[i] * a + e0 * v0;
            m[i] = mn;
        }
    }
    // merge the 8 simdgroups of this threadgroup
    for (int i = 0; i < h; ++i) {
        if (lane == 0) { sh_m[sg][i] = m[i]; sh_l[sg][i] = l[i]; }
        sh_acc[sg][i][lane * 4 + 0] = acc[i].x; sh_acc[sg][i][lane * 4 + 1] = acc[i].y;
        sh_acc[sg][i][lane * 4 + 2] = acc[i].z; sh_acc[sg][i][lane * 4 + 3] = acc[i].w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int i = 0; i < h; ++i) {
            float mstar = -INFINITY;
            for (int k = 0; k < 8; ++k) mstar = max(mstar, sh_m[k][i]);
            float lstar = 0.0f;
            float4 a4 = float4(0.0f);
            for (int k = 0; k < 8; ++k) {
                const float w = (sh_l[k][i] > 0.0f) ? exp(sh_m[k][i] - mstar) : 0.0f;
                lstar += w * sh_l[k][i];
                a4 += w * float4(sh_acc[k][i][lane * 4], sh_acc[k][i][lane * 4 + 1], sh_acc[k][i][lane * 4 + 2], sh_acc[k][i][lane * 4 + 3]);
            }
            if (P == 1) {
                const float4 o = a4 / lstar;
                const size_t ob = ((size_t)g * h + i) * 128 + lane * 4;
                if (out_bf16) { out_bf[ob] = f2bf(o.x); out_bf[ob + 1] = f2bf(o.y); out_bf[ob + 2] = f2bf(o.z); out_bf[ob + 3] = f2bf(o.w); }
                else { out_f[ob] = o.x; out_f[ob + 1] = o.y; out_f[ob + 2] = o.z; out_f[ob + 3] = o.w; }
            } else {
                device float* pb = part + (((size_t)g * P + p) * h + i) * 130;
                if (lane == 0) { pb[0] = mstar; pb[1] = lstar; }
                pb[2 + lane * 4] = a4.x; pb[3 + lane * 4] = a4.y; pb[4 + lane * 4] = a4.z; pb[5 + lane * 4] = a4.w;
            }
        }
    }
}

// Merge the P partials of every (g, i): one simdgroup per (g, i), lane = 4 dims.
kernel void attend_combine(device const float* part   [[buffer(0)]],
                           device ushort* out_bf       [[buffer(1)]],
                           device float*  out_f        [[buffer(2)]],
                           constant int& h             [[buffer(3)]],
                           constant int& P             [[buffer(4)]],
                           constant int& out_bf16      [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    const int gi = (int)(id >> 5), lane = (int)(id & 31);   // gi = g * h + i
    float mstar = -INFINITY;
    for (int p = 0; p < P; ++p) mstar = max(mstar, part[((size_t)(gi / h) * P + p) * h * 130 + (size_t)(gi % h) * 130]);
    float lstar = 0.0f;
    float4 a4 = float4(0.0f);
    for (int p = 0; p < P; ++p) {
        const device float* pb = part + (((size_t)(gi / h) * P + p) * h + (gi % h)) * 130;
        const float w = (pb[1] > 0.0f) ? exp(pb[0] - mstar) : 0.0f;
        lstar += w * pb[1];
        a4 += w * float4(pb[2 + lane * 4], pb[3 + lane * 4], pb[4 + lane * 4], pb[5 + lane * 4]);
    }
    const float4 o = a4 / lstar;
    const size_t ob = (size_t)gi * 128 + lane * 4;
    if (out_bf16) { out_bf[ob] = f2bf(o.x); out_bf[ob + 1] = f2bf(o.y); out_bf[ob + 2] = f2bf(o.z); out_bf[ob + 3] = f2bf(o.w); }
    else { out_f[ob] = o.x; out_f[ob + 1] = o.y; out_f[ob + 2] = o.z; out_f[ob + 3] = o.w; }
}
"""


def available() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def _check_kv(KV, q, rows, partitions):
    """Validation for the interleaved layout: KV [G, C, 2*d]."""
    G, C, dd = KV.shape
    if dd != 2 * HEAD_DIM:
        raise ValueError("interleaved rows must hold K and V of head_dim 128")
    h = int(q.shape[1])
    if q.dim() != 3 or q.shape[0] != G or q.shape[2] != HEAD_DIM or q.dtype != torch.bfloat16 or not q.is_contiguous() or not 1 <= h <= MAX_H:
        raise ValueError("q must be a contiguous bfloat16 [G, h, 128] tensor with h <= 4")
    if rows.dim() != 2 or rows.shape[0] != G or rows.dtype != torch.int32 or not rows.is_contiguous() or rows.shape[1] < 1:
        raise ValueError("rows must be a contiguous int32 [G, S] tensor with S >= 1")
    if not 1 <= int(partitions) <= MAX_P:
        raise ValueError(f"partitions must be in [1, {MAX_P}]")
    return G, C, h, int(rows.shape[1])


def _check(K, V, q, rows, partitions):
    if K.dim() != 3 or K.dtype != torch.bfloat16 or not K.is_contiguous():
        raise ValueError("K must be a contiguous bfloat16 [G, C, d] tensor (batch 1)")
    if V.shape != K.shape or V.dtype != torch.bfloat16 or not V.is_contiguous():
        raise ValueError("V must match K")
    G, C, d = K.shape
    if d != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM} (32 lanes x 4)")
    if q.dim() != 3 or q.shape[0] != G or q.shape[2] != d or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("q must be a contiguous bfloat16 [G, h, d] tensor")
    h = int(q.shape[1])
    if not 1 <= h <= MAX_H:
        raise ValueError(f"h must be in [1, {MAX_H}]")
    if rows.dim() != 2 or rows.shape[0] != G or rows.dtype != torch.int32 or not rows.is_contiguous():
        raise ValueError("rows must be a contiguous int32 [G, S] tensor of original positions")
    S = int(rows.shape[1])
    if S < 1:
        raise ValueError("at least one row must be selected")
    if not 1 <= int(partitions) <= MAX_P:
        raise ValueError(f"partitions must be in [1, {MAX_P}]")
    for t in (V, q, rows):
        if t.device != K.device:
            raise ValueError("all inputs must live on the same device")
    return G, C, h, S


class MetalAttend:
    """Gather-free subset attention: out[g, i] = softmax(q[g, i] · K[g, rows[g]] * scale) @ V[g, rows[g]]."""

    def __init__(self):
        if not available():
            raise RuntimeError("needs MPS and torch.mps.compile_shader")
        self.lib = torch.mps.compile_shader(SOURCE)

    def __call__(self, K, V, q, rows, *, scale=None, partitions: int = 4, out_dtype=torch.float32, out=None, part=None, kv_interleaved=None):
        """K, V [G, C, d] bf16; or `kv_interleaved` [G, C, 2, d] bf16 (K then V per position) with K and V ignored (layout diagnostic)."""
        if kv_interleaved is not None:
            kv = kv_interleaved
            if kv.dim() != 4 or kv.shape[2] != 2 or kv.dtype != torch.bfloat16 or not kv.is_contiguous():
                raise ValueError("kv_interleaved must be a contiguous bfloat16 [G, C, 2, d] tensor")
            K = V = kv.view(kv.shape[0], kv.shape[1], 2 * kv.shape[3])      # [G, C, 256]: row stride 256, V at offset 128
            G, C, h, S = _check_kv(K, q, rows, partitions)
            row_stride, v_off = 2 * HEAD_DIM, HEAD_DIM
        else:
            G, C, h, S = _check(K, V, q, rows, partitions)
            row_stride, v_off = HEAD_DIM, 0
        P = int(partitions)
        sc = float(1.0 / math.sqrt(HEAD_DIM) if scale is None else scale)
        bf = out_dtype == torch.bfloat16
        if out is None:
            out = torch.empty(G, h, HEAD_DIM, dtype=out_dtype, device=K.device)
        if part is None:
            part = torch.empty(G, P, h, 2 + HEAD_DIM, dtype=torch.float32, device=K.device) if P > 1 else torch.empty(1, device=K.device)
        out_bf = out if bf else torch.empty(1, dtype=torch.bfloat16, device=K.device)
        out_f = out if not bf else torch.empty(1, dtype=torch.float32, device=K.device)
        self.lib.attend_partial(K, V, q, rows, part, out_bf, out_f, C, h, S, P, sc, int(bf), row_stride, v_off, threads=G * P * TG, group_size=TG)
        if P > 1:
            self.lib.attend_combine(part, out_bf, out_f, h, P, int(bf), threads=G * h * 32, group_size=min(G * h * 32, TG))
        return out


def model_views(K4, V4, q5, rows3):
    """Model-side tensors (batch 1) -> kernel views: K/V [G, C, d], q [G, h, d], rows int32 [G, S]."""
    if K4.shape[0] != 1 or q5.shape[0] != 1 or rows3.shape[0] != 1:
        raise ValueError("P1 kernels take batch 1")
    if q5.shape[3] != 1:
        raise ValueError("one query position per step")
    return K4[0], V4[0], q5[0, :, :, 0, :].contiguous(), rows3[0].to(torch.int32).contiguous()


def reference_attend(K, V, q, rows, *, scale=None):
    """float32 CPU reference: softmax over the listed rows only, read by original position (dense_mask_reference semantics)."""
    Kf, Vf, qf = (t.detach().float().cpu() for t in (K, V, q))
    r = rows.detach().long().cpu()
    sc = 1.0 / math.sqrt(K.shape[-1]) if scale is None else float(scale)
    G, h, d = qf.shape
    out = torch.empty(G, h, d)
    for g in range(G):
        ks, vs = Kf[g][r[g]], Vf[g][r[g]]                       # [S, d]
        s = (qf[g] @ ks.T) * sc                                  # [h, S]
        out[g] = torch.softmax(s, dim=-1) @ vs
    return out
