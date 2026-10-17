#!/usr/bin/env python3
"""Aggregate P0 score-kernel runs (runs/p0-score-*/summary.json, non-quick) into reports/p0_score_kernel_mac_m4max.{md,json}.

The stop gate of docs/tasks/04_FUSED_D_KERNEL.md needs every independent process to satisfy: fused_1launch <= 80 us/layer
at 32K (conservative median = slower of the two orders), fused/pytorch_chain <= 0.75, numerics within the preset.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATS = ["pytorch_chain", "pytorch_score_only", "fused_1launch", "fused_2launch", "fused_1launch_v1", "fused_2launch_v1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/p0_score_kernel_mac_m4max.md")
    ap.add_argument("--json", type=Path, default=ROOT / "reports/p0_score_kernel_mac_m4max.json")
    ap.add_argument("--include-quick", action="store_true")
    a = ap.parse_args()
    paths = a.runs or sorted(glob.glob(str(ROOT / "runs/p0-score-*/summary.json")))
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    runs = [r for r in runs if a.include_quick or not r.get("quick")]
    if not runs:
        print("no non-quick P0 runs")
        return 1
    lengths = sorted({b["N0"] for r in runs for b in r["lengths"]})
    agg = {}
    for N0 in lengths:
        agg[N0] = {}
        for s in STRATS:
            vals = [b["per_strategy_us_conservative"][s] for r in runs for b in r["lengths"] if b["N0"] == N0]
            best = [b["per_strategy_us_best"][s] for r in runs for b in r["lengths"] if b["N0"] == N0]
            agg[N0][s] = {"median_us": statistics.median(vals), "min_us": min(vals), "max_us": max(vals), "best_order_median_us": statistics.median(best), "n_runs": len(vals)}
        agg[N0]["ratio_fused1_chain"] = [b["ratios_conservative"]["best_fused/pytorch_chain"] for r in runs for b in r["lengths"] if b["N0"] == N0]
        agg[N0]["best_fused"] = [b["best_fused"]["strategy"] for r in runs for b in r["lengths"] if b["N0"] == N0]
        agg[N0]["numerics"] = [b["numerics"] for r in runs for b in r["lengths"] if b["N0"] == N0]
        agg[N0]["scan_bytes_per_layer"] = next(b["scan_bytes_per_layer"] for r in runs for b in r["lengths"] if b["N0"] == N0)
        agg[N0]["M"] = next(b["M"] for r in runs for b in r["lengths"] if b["N0"] == N0)
        agg[N0]["k"] = next(b["k"] for r in runs for b in r["lengths"] if b["N0"] == N0)
    gates = [r["stop_gate"] for r in runs]
    verdict = {"runs": [r["run_id"] for r in runs], "every_run_passes_gate": all(g.get("pass") for g in gates), "per_run": gates}
    out = {"runs": [r["run_id"] for r in runs], "gate": runs[0]["gate"], "aggregate": {str(k): v for k, v in agg.items()}, "verdict": verdict,
           "git": [r["git"] for r in runs], "source_hashes": runs[0]["source_hashes"]}
    a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    H, S = runs[0]["host"], runs[0]["software"]
    L = ["# P0 实测报告 · D 链打分段的自定义 Metal 核（Mac M4 Max）\n",
         f"由 `scripts/p0_report.py` 从 {len(runs)} 个独立进程生成（{', '.join(out['runs'])}）。**随机张量、单层的一段、batch 1；没有模型、没有任务、没有质量结论；MPS 结果不推 CUDA。** 预测见 `reports/cards/P0_fused_score_mac_m4max.md`。\n",
         "## 1. 环境与设置\n", "| 项 | 值 |\n|---|---|",
         f"| 机器 | {H.get('chip')}，{H.get('gpu_cores')} 核 GPU，{(H.get('physical_memory_bytes') or 0)/2**30:.0f} GiB，macOS {H.get('macos')} |",
         f"| torch / Python | {S.get('torch')} / {H.get('python')}；MPS + `torch.mps.compile_shader`；fallback 与内存保护环境变量未设置 |",
         f"| 形状 | G={runs[0]['args']['groups']}，h={runs[0]['args']['heads_per_group']}，d={runs[0]['args']['head_dim']}，r={runs[0]['args']['index_dim']}，bf16，ρ={runs[0]['args']['rho']:.5f}；容量 N0+128，合法长度 N0+1，候选域 [4, L−64) |",
         f"| 计时 | 每 pass {runs[0]['args']['layers']} 层轮换；预热 {runs[0]['args']['warmup']}、正式 {runs[0]['args']['repeats']}；每 pass 前后 `torch.mps.synchronize()`；每策略按正序与反序各测一次，取较慢者为保守值 |",
         f"| 代码 | " + "；".join(f"{r['run_id']}: commit {r['git'].get('commit')} dirty {r['git'].get('dirty_diff_sha256')}" for r in runs) + " |\n",
         "## 2. 每层 µs（保守中位数；括号内为进程间最小..最大）\n",
         "| N0 | M | k | 扫描 MB/层 | pytorch_chain | pytorch_score_only | fused_1launch (v2) | fused_2launch (v2) | v1 一次 / 两次启动 | 最优 fused/chain（各进程） | 最优 fused 有效带宽 GB/s |\n|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for N0 in lengths:
        g = agg[N0]
        cell = lambda s: f"{g[s]['median_us']:.1f} ({g[s]['min_us']:.1f}..{g[s]['max_us']:.1f})"  # noqa: E731
        best = min(("fused_1launch", "fused_2launch"), key=lambda n: g[n]["median_us"])
        bw = g["scan_bytes_per_layer"] / (g[best]["median_us"] * 1e-6) / 1e9
        L.append(f"| {N0} | {g['M']} | {g['k']} | {g['scan_bytes_per_layer']/1e6:.1f} | {cell('pytorch_chain')} | {cell('pytorch_score_only')} | **{cell('fused_1launch')}** | **{cell('fused_2launch')}** | {g['fused_1launch_v1']['median_us']:.1f} / {g['fused_2launch_v1']['median_us']:.1f} | {', '.join(f'{x:.2f}' for x in g['ratio_fused1_chain'])}（{', '.join(g['best_fused'])}） | {bw:.0f} |")
    L.append("\n## 3. 数值（每进程第 0 层，fused fp32 与 PyTorch bf16 各对 CPU float32 参照）\n")
    L.append("| N0 | run | fused max abs | fused 相对范数 | PyTorch bf16 max abs | PyTorch bf16 相对范数 | z 列逐位相同比例 | z 最大相对差（ulp 单位 2^-8） | top-k 集合一致 均值/最小 | 通过 |\n|---:|---|---:|---:|---:|---:|---:|---:|---|---|")
    for N0 in lengths:
        for rid, n in zip(out["runs"], agg[N0]["numerics"]):
            L.append(f"| {N0} | {rid} | {n['fused_vs_ref_max_abs']:.2e} | {n['fused_vs_ref_rel_norm']:.2e} | {n['pytorch_bf16_vs_ref_max_abs']:.2e} | {n['pytorch_bf16_vs_ref_rel_norm']:.2e} | {n['z_fused_bitwise_fraction']:.3f} | {n['z_fused_vs_pytorch_max_rel_ulp']/2**-8:.2f} | {n['topk_set_agreement_mean']:.4f} / {n['topk_set_agreement_min']:.4f} | {'✓' if n['pass'] else '✗'} |")
    L.append("\n## 4. 停止门\n")
    L.append("| run | 评估 N0 | 最优 fused 变体 | µs（≤ 80） | fused/chain（≤ 0.75） | 数值 | 通过 |\n|---|---:|---|---:|---:|---|---|")
    for rid, g in zip(out["runs"], gates):
        L.append(f"| {rid} | {g.get('evaluated_at_N0')} | {g.get('fused_strategy')} | {g.get('fused_us_conservative') if g.get('fused_us_conservative') is None else round(g['fused_us_conservative'], 1)} | {g.get('ratio_vs_chain') if g.get('ratio_vs_chain') is None else round(g['ratio_vs_chain'], 3)} | {g.get('numerics_pass_all')} | {'**通过**' if g.get('pass') else '不通过'} |")
    L.append(f"\n每个独立进程都通过：**{verdict['every_run_passes_gate']}**。这只回答打分段能否接近带宽下界；top-k、取行、子集注意力三段未动，整模型比值不由本报告推出。\n")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out.relative_to(ROOT)}, {a.json.relative_to(ROOT)}; every run passes gate: {verdict['every_run_passes_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
