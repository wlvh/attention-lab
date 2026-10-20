#!/usr/bin/env python3
"""Aggregate P1 attention-kernel runs (runs/p1-attend-*/summary.json, non-quick) into reports/p1_attend_kernel_mac_m4max.{md,json}.

Stop gate (docs/tasks/04_FUSED_D_KERNEL.md P1): in every independent process the best fused variant <= 60 us/layer at
S = 1076 (conservative median = slower of the two orders), fused / pytorch_gather_sdpa <= 0.80, numerics within preset.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATS = ["pytorch_gather_sdpa", "fused_attend_P1", "fused_attend_P2", "fused_attend_P4", "fused_attend_P8", "fused_attend_P16", "fused_attend_P32", "diag_kv_interleaved_P4", "diag_kv_interleaved_P8", "diag_sorted_rows_P8", "pytorch_full_sdpa"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/p1_attend_kernel_mac_m4max.md")
    ap.add_argument("--json", type=Path, default=ROOT / "reports/p1_attend_kernel_mac_m4max.json")
    ap.add_argument("--include-quick", action="store_true")
    a = ap.parse_args()
    paths = a.runs or sorted(glob.glob(str(ROOT / "runs/p1-attend-*/summary.json")))
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    runs = [r for r in runs if a.include_quick or not r.get("quick")]
    if not runs:
        print("no non-quick P1 runs")
        return 1
    lengths = sorted({b["N0"] for r in runs for b in r["lengths"]})
    agg = {}
    for N0 in lengths:
        blocks = [b for r in runs for b in r["lengths"] if b["N0"] == N0]
        agg[N0] = {s: {"median_us": statistics.median(b["per_strategy_us_conservative"][s] for b in blocks),
                       "min_us": min(b["per_strategy_us_conservative"][s] for b in blocks), "max_us": max(b["per_strategy_us_conservative"][s] for b in blocks),
                       "best_order_median_us": statistics.median(b["per_strategy_us_best"][s] for b in blocks), "n_runs": len(blocks)} for s in STRATS}
        agg[N0].update({"S": blocks[0]["S"], "k": blocks[0]["k"], "kv_bytes_read_per_layer": blocks[0]["kv_bytes_read_per_layer"], "full_read_bytes_per_layer": blocks[0]["full_read_bytes_per_layer"],
                        "best_fused": [b["best_fused"]["strategy"] for b in blocks], "ratio_best_vs_gather": [b["ratios_conservative"]["best_fused/pytorch_gather_sdpa"] for b in blocks],
                        "ratio_best_vs_full": [b["ratios_conservative"]["best_fused/pytorch_full_sdpa"] for b in blocks], "numerics": [b["numerics"] for b in blocks]})
    gates = [r["stop_gate"] for r in runs]
    verdict = {"runs": [r["run_id"] for r in runs], "every_run_passes_gate": all(g.get("pass") for g in gates), "per_run": gates}
    out = {"runs": verdict["runs"], "gate": runs[0]["gate"], "aggregate": {str(k): v for k, v in agg.items()}, "verdict": verdict, "git": [r["git"] for r in runs], "source_hashes": runs[0]["source_hashes"]}
    a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    H, S_ = runs[0]["host"], runs[0]["software"]
    L = ["# P1 实测报告 · D 链取行注意力段的自定义 Metal 核（Mac M4 Max）\n",
         f"由 `scripts/p1_report.py` 从 {len(runs)} 个独立进程生成（{', '.join(out['runs'])}）。**随机张量、单层的一段、batch 1；行表在计时之外准备；没有模型、没有任务、没有质量结论；MPS 结果不推 CUDA。** 预测见 `reports/cards/P1_fused_attend_mac_m4max.md`。\n",
         "## 1. 环境与设置\n", "| 项 | 值 |\n|---|---|",
         f"| 机器 | {H.get('chip')}，{H.get('gpu_cores')} 核 GPU，{(H.get('physical_memory_bytes') or 0)/2**30:.0f} GiB，macOS {H.get('macos')} |",
         f"| torch / Python | {S_.get('torch')} / {H.get('python')}；MPS + `torch.mps.compile_shader`；fallback 与内存保护环境变量未设置 |",
         f"| 形状 | G={runs[0]['args']['groups']}，h={runs[0]['args']['heads_per_group']}，d={runs[0]['args']['head_dim']}，bf16，ρ={runs[0]['args']['rho']:.5f}；容量 N0+128，合法长度 N0+1；行表 = F ∪ 候选域随机 k 个 |",
         f"| 计时 | 每 pass {runs[0]['args']['layers']} 层轮换；预热 {runs[0]['args']['warmup']}、正式 {runs[0]['args']['repeats']}；每 pass 前后 `torch.mps.synchronize()`；每策略正序与反序各一次，取较慢者为保守值；融合核输出 bf16 |",
         "| 代码 | " + "；".join(f"{r['run_id']}: commit {r['git'].get('commit')} dirty {r['git'].get('dirty_diff_sha256')}" for r in runs) + " |\n",
         "## 2. 每层 µs（保守中位数；括号内为进程间最小..最大）\n",
         "| N0 | S | 读行字节 MB/层 | pytorch_gather_sdpa | 融合：1分区 | 2分区 | 4分区 | 8分区 | 16分区 | 32分区 | A 的完整 SDPA（上下文） | 最优 fused / gather_sdpa（各进程） | 最优 fused / 完整 SDPA | 最优 fused 有效带宽 GB/s |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    L.insert(1, '这是已停止的 Metal 取行注意力原型记录。文件名里的 P0/P1 是当时的原型阶段名，与训练课程中“索引位置设置 P0/P1”不同；没有接入正式模型路径。 行表已经在计时外给定。pytorch_gather_sdpa 先把入选 K/V 拷到子数组再做注意力；融合实现按原位置直接读主缓存并累加注意力。分区数表示把入选位置拆成多少段处理，随后合并部分结果；表中的 1/2/4/8 分区不是新的实验阶段。\n')
    for N0 in lengths:
        g = agg[N0]
        cell = lambda s: f"{g[s]['median_us']:.1f} ({g[s]['min_us']:.1f}..{g[s]['max_us']:.1f})"  # noqa: E731
        best = min((f"fused_attend_P{P}" for P in (1, 2, 4, 8, 16, 32)), key=lambda n: g[n]["median_us"])
        bw = g["kv_bytes_read_per_layer"] / (g[best]["median_us"] * 1e-6) / 1e9
        L.append(f"| {N0} | {g['S']} | {g['kv_bytes_read_per_layer']/1e6:.2f} | {cell('pytorch_gather_sdpa')} | {cell('fused_attend_P1')} | {cell('fused_attend_P2')} | {cell('fused_attend_P4')} | {cell('fused_attend_P8')} | **{cell('fused_attend_P16')}** | **{cell('fused_attend_P32')}** | {cell('pytorch_full_sdpa')} | "
                 f"{', '.join(f'{x:.2f}' for x in g['ratio_best_vs_gather'])}（{', '.join(g['best_fused'])}） | {', '.join(f'{x:.2f}' for x in g['ratio_best_vs_full'])} | {bw:.0f} |")
    L.append("\n诊断（不参与门槛；布局与行表结构的影响）：")
    for N0 in lengths:
        g = agg[N0]
        L.append(f"- N0={N0}（S={g['S']}）：K/V 交错布局（每位置一段 512 B）P4 {g['diag_kv_interleaved_P4']['median_us']:.1f} µs、P8 {g['diag_kv_interleaved_P8']['median_us']:.1f} µs；行表排序后 P8 {g['diag_sorted_rows_P8']['median_us']:.1f} µs；对照分离布局 P4 {g['fused_attend_P4']['median_us']:.1f}、P8 {g['fused_attend_P8']['median_us']:.1f} µs。")
    L.append("\n## 3. 数值（每进程第 0 层；fused fp32 输出对 `dense_mask_reference`（CPU float32），PyTorch bf16 路径对同一参照）\n")
    L.append("| N0 | run | 分区 P | fused max abs | fused 相对范数 | bf16 输出 1 ulp 内 | bf16 逐位相同比例 | PyTorch bf16 max abs | PyTorch bf16 相对范数 | 通过 |\n|---:|---|---:|---:|---:|---|---:|---:|---:|---|")
    for N0 in lengths:
        for rid, n in zip(out["runs"], agg[N0]["numerics"]):
            for P, v in n["per_partition"].items():
                L.append(f"| {N0} | {rid} | {P} | {v['fused_vs_ref_max_abs']:.2e} | {v['fused_vs_ref_rel_norm']:.2e} | {v['bf16_within_one_ulp']} | {v['bf16_bitwise_fraction']:.3f} | {n['pytorch_bf16_vs_ref_max_abs']:.2e} | {n['pytorch_bf16_vs_ref_rel_norm']:.2e} | {'✓' if v['pass'] else '✗'} |")
    L.append("\n## 4. 停止门（S = 1076）\n")
    L.append("| run | N0 | 最优 fused 变体 | µs（≤ 60） | fused / gather_sdpa（≤ 0.80） | 数值 | 通过 |\n|---|---:|---|---:|---:|---|---|")
    for rid, g in zip(out["runs"], gates):
        L.append(f"| {rid} | {g.get('evaluated_at_N0')} | {g.get('fused_strategy')} | {g.get('fused_us_conservative') if g.get('fused_us_conservative') is None else round(g['fused_us_conservative'], 1)} | {g.get('ratio_vs_gather_sdpa') if g.get('ratio_vs_gather_sdpa') is None else round(g['ratio_vs_gather_sdpa'], 3)} | {g.get('numerics_pass_all')} | {'**通过**' if g.get('pass') else '不通过'} |")
    L.append(f"\n每个独立进程都通过：**{verdict['every_run_passes_gate']}**。这只比较所测取行注意力实现的时间与数值；选择段（top-k、行号装配）未动，整模型比值不由本报告推出。\n")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out.relative_to(ROOT)}, {a.json.relative_to(ROOT)}; every run passes gate: {verdict['every_run_passes_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
