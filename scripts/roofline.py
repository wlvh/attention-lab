#!/usr/bin/env python3
"""Attention traffic scenarios, NOT latency measurements.

Derived from the supplied roofline.py. Changes are recorded in docs/DECISIONS.md:
explicit bandwidth, validated inputs, bounded candidates, honest dtype labels,
and separate stored-index versus scanned-index bytes. No third-party imports.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Model:
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    parameters: int

# Source: supplied PLAN_v3 / roofline; confirm actual numel at model integration.
MODELS = {
    "Qwen3-0.6B": Model(28, 16, 8, 128, 596_049_920),
    "Qwen3-1.7B": Model(28, 16, 8, 128, 1_720_574_976),
    "Qwen3-4B": Model(36, 32, 8, 128, 4_022_468_096),
}

def estimate(model: Model, n: int, *, rho: float = 1 / 32, r: int = 32,
             fixed: int = 68, element_bytes: int = 2,
             index_bytes: int | None = None, bandwidth: float | None = None,
             scan_scope: str = "candidates", extra_ops: int = 7,
             launch_us: float | None = None) -> dict[str, Any]:
    """One batch-1 decode read stage. The current position is included in n.

    S1: index scan + selected KV read once.
    S2: index scan + selected KV read, gather write, attention reread.
    Neither accounts for every scratch access, projection, top-k pass or cache hit.
    """
    index_bytes = element_bytes if index_bytes is None else index_bytes
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    for name, value, minimum in (("r", r, 1), ("fixed", fixed, 0),
                                 ("element_bytes", element_bytes, 1),
                                 ("index_bytes", index_bytes, 1), ("extra_ops", extra_ops, 0)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("rho must be finite and between 0 and 1 inclusive")
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError("bandwidth must be finite and positive")
    if launch_us is not None and (not math.isfinite(launch_us) or launch_us < 0):
        raise ValueError("launch_us must be finite and nonnegative")
    if scan_scope not in ("all", "candidates"):
        raise ValueError("scan_scope must be 'all' or 'candidates'")
    if min(model.layers, model.q_heads, model.kv_heads, model.head_dim, model.parameters) <= 0:
        raise ValueError("all model dimensions must be positive")
    if model.q_heads % model.kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    fixed_count = min(fixed, n)
    candidate_count = n - fixed_count
    k_requested = round(rho * n)  # Python ties-to-even, matching the supplied script.
    k = min(k_requested, candidate_count)
    selected = fixed_count + k
    per_position = 2 * model.layers * model.kv_heads * model.head_dim * element_bytes
    weights = model.parameters * element_bytes
    dense_kv = per_position * n
    scanned_n = n if scan_scope == "all" else candidate_count
    index_per_position = model.layers * model.kv_heads * r * index_bytes
    index_stored = n * index_per_position
    index_read = scanned_n * index_per_position
    selected_kv = selected * per_position
    d_s1 = index_read + selected_kv
    d_s2 = index_read + 3 * selected_kv
    denominator = weights + dense_kv
    to_ms = lambda b: None if bandwidth is None else b / bandwidth * 1000
    return {
        "kind": "prediction", "scope": "batch1_decode_read_stage_scenario",
        "n": n, "rho": rho, "r": r, "fixed_count": fixed_count,
        "candidate_count": candidate_count, "k_requested": k_requested,
        "k": k, "selected_count": selected, "effective_ratio": selected / n,
        "degenerate_full_selection": selected == n,
        "element_bytes": element_bytes, "index_element_bytes": index_bytes,
        "scan_scope": scan_scope, "index_scanned_positions": scanned_n,
        "weights_bytes": weights, "dense_kv_bytes": dense_kv,
        "selected_kv_bytes": selected_kv, "index_stored_bytes": index_stored,
        "index_read_bytes": index_read, "d_s1_bytes": d_s1, "d_s2_bytes": d_s2,
        "saved_s1_bytes": dense_kv - d_s1, "saved_s2_bytes": dense_kv - d_s2,
        "attention_fraction_of_assumed_total_traffic": dense_kv / denominator,
        "index_parameter_count": model.layers * model.kv_heads * r *
                                 ((model.q_heads // model.kv_heads + 1) * model.head_dim),
        "bandwidth_assumption_bytes_per_second": bandwidth,
        "dense_scenario_ms": to_ms(denominator),
        "saved_s1_scenario_ms": to_ms(dense_kv - d_s1),
        "saved_s2_scenario_ms": to_ms(dense_kv - d_s2),
        "extra_cpu_submit_scenario_ms": None if launch_us is None else
                                          model.layers * extra_ops * launch_us / 1000,
        "caveat": "Traffic scenarios, not measured wall time or speedup bounds. CPU submission may overlap GPU work.",
    }

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bw", type=float, default=None, help="Explicit bandwidth assumption in bytes/sec. Default: no time estimate.")
    p.add_argument("--rho", type=float, default=1 / 32)
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--fixed", type=int, default=68)
    p.add_argument("--bytes", type=int, default=2, dest="element_bytes")
    p.add_argument("--index-bytes", type=int, default=None)
    p.add_argument("--scan-scope", choices=["all", "candidates"], default="candidates")
    p.add_argument("--launch-us", "--launch_us", type=float, default=None)
    p.add_argument("--extra-ops", "--extra_ops", type=int, default=7)
    p.add_argument("--n", type=int, nargs="+", default=[8192, 16384, 32768])
    p.add_argument("--models", choices=list(MODELS), nargs="+", default=list(MODELS))
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    a = p.parse_args(argv)
    rows = []
    try:
        for name in a.models:
            for n in a.n:
                rows.append({"model": name, **estimate(MODELS[name], n, rho=a.rho, r=a.r,
                    fixed=a.fixed, element_bytes=a.element_bytes, index_bytes=a.index_bytes,
                    bandwidth=a.bw, scan_scope=a.scan_scope, extra_ops=a.extra_ops, launch_us=a.launch_us)})
    except ValueError as e:
        p.error(str(e))
    if a.format == "json":
        print(json.dumps({"schema_version": 1, "kind": "prediction", "rows": rows}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    print(f"预测而非测量；batch=1；主数值{a.element_bytes}字节；索引数值{a.index_bytes or a.element_bytes}字节；ρ={a.rho:g}；索引扫描={a.scan_scope}。")
    print("未指定格式时不能仅由2字节推断BF16或FP16。默认不填Mac带宽。")
    print("| 模型 | N | k | 实际读取数 | 权重GB | 主KV GB | 索引保存GB | S1读取GB | S2读取GB | S1节省ms情景 | S2节省ms情景 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for x in rows:
        fmt = lambda v: "未估计" if v is None else f"{v:.3f}"
        print(f"| {x['model']} | {x['n']} | {x['k']} | {x['selected_count']} | {x['weights_bytes']/1e9:.3f} | {x['dense_kv_bytes']/1e9:.3f} | {x['index_stored_bytes']/1e9:.3f} | {x['d_s1_bytes']/1e9:.3f} | {x['d_s2_bytes']/1e9:.3f} | {fmt(x['saved_s1_scenario_ms'])} | {fmt(x['saved_s2_scenario_ms'])} |")
    print("\n同精度、扫描全部位置时索引载荷/主KV=r/(2d)，这是载荷模型，不是所有内核的实际DRAM下界。")
    print("下界相减不是差值上界；负节省可发生；全选格属于正确性诊断。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
