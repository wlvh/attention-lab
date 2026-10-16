#!/usr/bin/env python3
"""Aggregate the G1b runs into reports/: numerics verdicts, full-step timings, cache growth, profile evidence.

Inputs (non-quick unless --include-quick): runs/g1b-numerics-*/results.json (latest, or --numerics) and every
runs/g1b-mac-*/summary.json (or --runs ...). Outputs: reports/g1b_mac_m4max.md, reports/g1b_summary_mac_m4max.json,
reports/g1b_numerics_mac_m4max.json and reports/g1b_manifest_mac_m4max.json. Numbers come only from the runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_RATIO = 0.90
PATH_LABEL = {"native_dynamic": "原生 A（DynamicCache）", "native_static": "原生 A（StaticCache）", "wrapped_full": "封装 A（静态 LabCache）", "sparse": "D（索引读取，含全部新增成本）"}


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def numerics_table(res):
    rows = []
    for rec in res["lengths"]:
        for name, c in rec["checks"].items():
            rows.append({"dtype": rec["dtype"], "device": rec["device"], "L": rec["L"], "check": name, "max_abs": c.get("max_abs", max(c.get("max_abs_k", 0), c.get("max_abs_v", 0), c.get("max_abs_z", 0))),
                         "rel_norm": c.get("rel_norm"), "argmax_agreement": c.get("argmax_agreement"), "tol_atol": c.get("tol_atol"), "pass": c.get("pass"),
                         "bitwise_equal": c.get("bitwise_equal"), "k0": c.get("k0")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numerics", default=None, help="runs/g1b-numerics-*/results.json (default: latest non-quick)")
    ap.add_argument("--runs", nargs="*", default=None, help="runs/g1b-mac-*/summary.json (default: every non-quick)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/g1b_mac_m4max.md")
    ap.add_argument("--json", type=Path, default=ROOT / "reports/g1b_summary_mac_m4max.json")
    ap.add_argument("--numerics-json", type=Path, default=ROOT / "reports/g1b_numerics_mac_m4max.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "reports/g1b_manifest_mac_m4max.json")
    ap.add_argument("--card", default="reports/cards/G1b_mac_m4max.md")
    ap.add_argument("--include-quick", action="store_true")
    a = ap.parse_args()

    num_paths = [a.numerics] if a.numerics else sorted(glob.glob(str(ROOT / "runs/g1b-numerics-*/results.json")))
    numerics = [load_json(p) | {"_path": rel(p)} for p in num_paths]
    numerics = [n for n in numerics if a.include_quick or not n.get("quick")]
    num = numerics[-1] if numerics else None
    run_paths = a.runs or sorted(glob.glob(str(ROOT / "runs/g1b-mac-*/summary.json")))
    runs = [load_json(p) | {"_path": rel(p)} for p in run_paths]
    runs = [r for r in runs if a.include_quick or not r.get("quick")]
    if not num and not runs:
        print("no non-quick G1b runs found")
        return 1
    manifests = {"numerics": [load_json(Path(ROOT / n["_path"]).parent / "manifest.json") for n in ([num] if num else [])],
                 "benchmarks": [load_json(Path(ROOT / r["_path"]).parent / "manifest.json") for r in runs]}

    # ---- condensed benchmark summary: per N0, mode, path ----
    cond = {"runs": [r["run_id"] for r in runs], "cells": [], "verdicts": []}
    for r in runs:
        for block in r["lengths"]:
            if "modes" not in block:
                cond["cells"].append({"run_id": r["run_id"], "N0": block["N0"], "skipped": block.get("skipped")})
                continue
            for mode, res in block["modes"].items():
                for path, pr in res.items():
                    s, q = pr.get("single"), pr.get("sequence")
                    cell = {"run_id": r["run_id"], "N0": block["N0"], "capacity": block["capacity"], "mode": mode, "path": path,
                            "prefill_ms": block["paths"].get(path, {}).get("prefill_ms"), "k0": block["paths"].get(path, {}).get("k0"),
                            "cache_bytes": block["paths"].get(path, {}).get("cache_bytes", {}).get("total_bytes"),
                            "single_median_ms": s and s["median_ms"], "single_p95_ms": s and s["p95_ms"], "single_min_ms": s and s["min_ms"], "single_n": s and s["n"],
                            "paired_ratio_vs_wrapped_full": pr.get("paired_ratio_vs_wrapped_full"),
                            "seq_step_median_ms": q and q["step_median_ms"], "seq_total_ms_median": q and q["total_ms_median"],
                            "seq_first16_ms": q and q["first16_median_ms"], "seq_last16_ms": q and q["last16_median_ms"], "seq_slope_us_per_step": q and q["slope_ms_per_step_median"] * 1e3,
                            "seq_n": q and q["n"], "k_last": (pr.get("sequences") or [{}])[-1].get("k_last"), "selected_last": (pr.get("sequences") or [{}])[-1].get("selected_last")}
                    cond["cells"].append(cell)
                if "sparse" in res and "wrapped_full" in res:
                    pr = res["sparse"].get("paired_ratio_vs_wrapped_full") or {}
                    prof = (block.get("profiles") or {}).get("sparse", {})
                    cond["verdicts"].append({"run_id": r["run_id"], "N0": block["N0"], "mode": mode, "ratio_median": pr.get("median"), "ratio_min": pr.get("min"), "ratio_max": pr.get("max"),
                                             "n_cycles": pr.get("n_cycles"), "cycles_at_or_below_0.90": pr.get("cycles_at_or_below_0.90"),
                                             "d_full_length_matmul_detected": prof.get("full_length_matmul_detected"),
                                             "pass": bool(pr) and pr.get("max") is not None and pr["max"] <= PASS_RATIO and prof.get("full_length_matmul_detected") is False})
    # ---- numerics condensed ----
    num_cond = None
    if num:
        num_cond = {"run_id": num["run_id"], "numerics_pass": num["numerics_pass"], "gated_checks": num["gated_checks"], "failed_checks": num["failed_checks"],
                    "model_info": num["model_info"], "tolerances": num["config"]["tolerances"], "checks": numerics_table(num),
                    "reference_prompt": num["reference_prompt"], "smoke_generation": num["smoke_generation"], "cpu_vs_mps_unmodified": num["cpu_vs_mps_unmodified"]}
    a.numerics_json.write_text(json.dumps(num_cond, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a.json.write_text(json.dumps(cond, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a.manifest.write_text(json.dumps(manifests, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---- markdown ----
    L = []
    L.append("# G1b 实测报告 · Mac M4 Max：把索引注意力装进 Qwen3-0.6B，整模型一步的 A 与 D\n")
    L.append(f"由 `scripts/g1b_report.py` 从 {'1 个数值 run（' + num['run_id'] + '）' if num else '无数值 run'} 与 {len(runs)} 个测速 run（{', '.join(r['run_id'] for r in runs) or '无'}）生成。"
             f"**随机 token 只驱动执行；没有任务、没有质量结论；MPS 结果不推 CUDA。** 预测见 `{a.card}`。\n")
    src = runs[0] if runs else None
    H = (src or num or {}).get("host") or (manifests["numerics"][0]["host"] if manifests["numerics"] else {})
    S = (src or {}).get("software") or (manifests["numerics"][0]["software"] if manifests["numerics"] else {})
    info = (src["model_info"] if src else (num["model_info"] if num else {}))
    L.append("## 1. 环境、模型与设置\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append(f"| 机器 | {H.get('chip')}，{H.get('gpu_cores')} 核 GPU，{(H.get('physical_memory_bytes') or 0)/2**30:.0f} GiB，macOS {H.get('macos')} |")
    L.append(f"| torch / transformers / Python | {S.get('torch')} / {S.get('transformers')} / {H.get('python')}；MPS；fallback 与内存保护环境变量未设置 |")
    L.append(f"| 模型 | {info.get('model_id')} @ `{info.get('config_commit_hash')}`（tokenizer 同 revision）；{info.get('architecture')}，attn `{info.get('attn_implementation')}` |")
    L.append(f"| 形状（从 config 读取并与参数形状核对） | {info.get('num_hidden_layers')} 层，{info.get('num_attention_heads')} 查询头 / {info.get('num_key_value_heads')} KV 头，head_dim {info.get('head_dim')}，hidden {info.get('hidden_size')}，词表 {info.get('vocab_size')}，参数 {(info.get('num_parameters_total') or 0)/1e6:.1f} M（独立存储 {(info.get('num_parameters_unique_storage') or 0)/1e6:.1f} M） |")
    if manifests["benchmarks"]:
        m0 = manifests["benchmarks"][0]
        L.append(f"| 权重文件 | model.safetensors sha256 `{(m0.get('weights') or {}).get('sha256')}`，{(m0.get('weights') or {}).get('bytes')} bytes |")
        L.append(f"| 代码 | commit {m0['git'].get('commit')}，dirty diff sha256 {m0['git'].get('dirty_diff_sha256')}；源码哈希见 manifest |")
    if src:
        ar = src["args"]
        L.append(f"| 测速 | dtype {ar['dtype']}，batch 1，N0 {ar['lengths']}，{ar['steps']} 步，ρ {ar['rho']}，r {ar['index_dim']}，容量 N0+{ar['steps']}，Prefill 块 {ar['chunk']}，单步预热 {ar['warmup']} / 循环 {ar['repeats']}（回文顺序），序列 {ar['seq_repeats']} 次；每步前后 `torch.mps.synchronize()`；`logits_to_keep=1` |")
    L.append("")

    if num_cond:
        L.append("## 2. 数值对照（三段 + 索引接入契约）\n")
        L.append(f"判定：**{'通过' if num_cond['numerics_pass'] else '未通过'}**（{num_cond['gated_checks']} 项门控检查" + (f"，失败：{num_cond['failed_checks']}" if num_cond["failed_checks"] else "") + "）。容差来自 `configs/mac_gate1b.json`，运行前预设。\n")
        L.append("| dtype | device | L | 检查 | max abs | 相对范数 | argmax 一致 | 容差 atol | 通过 |\n|---|---|---:|---|---:|---:|---:|---:|---|")
        for row in num_cond["checks"]:
            L.append(f"| {row['dtype']} | {row['device']} | {row['L']} | {row['check']} | {row['max_abs']:.3e} | {f(row['rel_norm'], 3) if row['rel_norm'] is None else format(row['rel_norm'], '.3e')} | "
                     f"{f(row['argmax_agreement'], 3)} | {row['tol_atol']} | {'✓' if row['pass'] else ('记录' if row['pass'] is None else '✗')}{' 逐位相同' if row.get('bitwise_equal') else ''} |")
        if num_cond["cpu_vs_mps_unmodified"]:
            c = num_cond["cpu_vs_mps_unmodified"]
            L.append(f"\n未改模型 MPS float32 对 CPU float32（平台噪声底，不门控）：max abs {c['max_abs']:.3e}，相对范数 {c['rel_norm']:.3e}，argmax 一致 {c['argmax_agreement']:.3f}。\n")
        L.append("短提示参照（未改模型，top-1 与贪心续写；只是数值参照，不评价质量）：\n")
        for rp in num_cond["reference_prompt"]:
            L.append(f"- {rp['dtype']} {rp['device']}：top-5 {rp['top5_tokens']} logits {rp['top5_logits']}；贪心：{rp['greedy_text'][:80]!r}")
        for sm in num_cond["smoke_generation"]:
            L.append(f"\n冒烟生成（{sm['dtype']}，提示 {sm['prompt_tokens']} token，贪心 {sm['new_tokens']} 新 token，D 的 k0={sm['sparse']['k0']}）：")
            for k, v in sm["texts"].items():
                L.append(f"- {k}（与 native 首次分歧步 {sm['first_divergence_step'][k]}）：{v[:100]!r}")
        L.append("")

    if runs:
        L.append("## 3. Prefill 与首步（每条路径一次，ms）\n")
        L.append("| run | N0 | 路径 | Prefill ms | 首步（冷）ms | k0 | 缓存字节 |\n|---|---:|---|---:|---:|---:|---:|")
        for r in runs:
            for block in r["lengths"]:
                for path, p in (block.get("paths") or {}).items():
                    L.append(f"| {r['run_id']} | {block['N0']} | {PATH_LABEL.get(path, path)} | {f(p['prefill_ms'], 1)} | {f(p['first_step_ms_cold'])} | {p.get('k0') if p.get('k0') is not None else '—'} | {(p.get('cache_bytes') or {}).get('total_bytes')} |")
        L.append("\n## 4. 固定 N 单步：整步耗时（ms）与配对比值\n")
        L.append("配对比值 = 同一循环内该路径中位数 ÷ 封装 A 中位数（回文顺序，每循环每路径两次）。D/A ≤ 0.90 为通过线，要求每个循环都 ≤ 0.90 且剖析无全长 matmul。\n")
        L.append("| run | N0 | 模式 | 路径 | 中位 ms | p95 ms | 最小 ms | n | 配对比值中位（范围） | ≤0.90 的循环数 |\n|---|---:|---|---|---:|---:|---:|---:|---|---:|")
        for c in cond["cells"]:
            if c.get("skipped") or c.get("single_median_ms") is None:
                continue
            pr = c["paired_ratio_vs_wrapped_full"] or {}
            L.append(f"| {c['run_id']} | {c['N0']} | {c['mode']} | {PATH_LABEL.get(c['path'], c['path'])} | {f(c['single_median_ms'])} | {f(c['single_p95_ms'])} | {f(c['single_min_ms'])} | {c['single_n']} | "
                     f"{f(pr.get('median'))} ({f(pr.get('min'))}–{f(pr.get('max'))}) | {pr.get('cycles_at_or_below_0.90', '—')}/{pr.get('n_cycles', '—')} |")
        L.append("\n## 5. 连续 128 步：每步耗时、增长与 k 规则\n")
        L.append("| run | N0 | 模式 | 路径 | 每步中位 ms | 首 16 步中位 | 末 16 步中位 | 斜率 µs/步 | 128 步合计中位 ms | k（末步） | 读取数（末步） | 序列数 |\n|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in cond["cells"]:
            if c.get("skipped") or c.get("seq_step_median_ms") is None:
                continue
            L.append(f"| {c['run_id']} | {c['N0']} | {c['mode']} | {PATH_LABEL.get(c['path'], c['path'])} | {f(c['seq_step_median_ms'])} | {f(c['seq_first16_ms'])} | {f(c['seq_last16_ms'])} | {f(c['seq_slope_us_per_step'], 1)} | {f(c['seq_total_ms_median'], 0)} | "
                     f"{c['k_last'] if c['k_last'] is not None else '—'} | {c['selected_last'] if c['selected_last'] is not None else '—'} | {c['seq_n']} |")
        L.append("\n## 6. 内存与剖析\n")
        for r in runs:
            for block in r["lengths"]:
                if "modes" not in block:
                    continue
                mb = block["memory"]
                L.append(f"- {r['run_id']} N0={block['N0']}：加载后 {r['memory_after_load']['current_allocated_bytes']/1e9:.2f} GB，Prefill 后各路径 " +
                         ", ".join(f"{p}: {v['memory_after_prefill']['current_allocated_bytes']/1e9:.2f} GB" for p, v in block["paths"].items()) +
                         f"；eager 测完 {mb['after_eager']['current_allocated_bytes']/1e9:.2f} GB（driver {mb['after_eager']['driver_allocated_bytes']/1e9:.2f} GB）；进程 ru_maxrss {mb['end']['ru_maxrss_bytes']/1e9:.2f} GB")
                for path, pf in (block.get("profiles") or {}).items():
                    if pf.get("status") == "error":
                        L.append(f"  - 剖析 {path}：错误 {pf['error_type']}")
                        continue
                    L.append(f"  - 剖析 {path}：aten 算子 {pf['aten_ops']}，触及长度的 matmul 类算子 {len(pf['length_matmul_ops'])}（{'有全长 matmul' if pf['full_length_matmul_detected'] else '无全长 matmul'}），top-k {pf['topk_calls']}，index_select {pf['index_select_calls']}，cat {pf['cat_calls']}，SDPA 调度 {pf['sdpa_dispatch']}，触及长度的算子 {pf['ops_touching_length']}")
                if block.get("compile"):
                    ci = block["compile"]
                    L.append(f"  - 块编译（MLP + RMSNorm，四路径共用）：{ci.get('status')}" + (f"，首调用 {ci.get('first_call_s', 0):.1f} s，与 eager 词表分数 max abs 差 {ci.get('max_abs_diff_vs_eager_logits')}" if ci.get("status") == "ok" else f"，{ci.get('error_type')}: {ci.get('error', '')[:200]}"))
                    if ci.get("dynamo_counters_delta"):
                        L.append(f"    dynamo 计数变化：{json.dumps(ci['dynamo_counters_delta'], ensure_ascii=False)[:300]}")
        L.append("\n## 7. 判定\n")
        L.append("| run | N0 | 模式 | D/A 配对中位 | 范围 | ≤0.90 循环 | D 无全长 matmul | 性能通过 |\n|---|---:|---|---:|---|---:|---|---|")
        for v in cond["verdicts"]:
            L.append(f"| {v['run_id']} | {v['N0']} | {v['mode']} | {f(v['ratio_median'])} | {f(v['ratio_min'])}–{f(v['ratio_max'])} | {v['cycles_at_or_below_0.90']}/{v['n_cycles']} | {v['d_full_length_matmul_detected'] is False} | {'**通过**' if v['pass'] else '不通过'} |")
        passing = [v for v in cond["verdicts"] if v["pass"]]
        L.append(f"\n性能判定：{'有' if passing else '没有'}设置在同模型、同设备、同工作量下呈现每个循环都 ≤ 0.90 的整步收益" + (f"：{[(v['N0'], v['mode']) for v in passing]}" if passing else "") + "。工程完成判定见数值一节与卡片。\n")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {rel(a.out)}, {rel(a.json)}, {rel(a.numerics_json)}, {rel(a.manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
