#!/usr/bin/env python3
"""Render reports/g1a_supplement_mac_m4max.md from a runs/g1a-supp-*/summary.json (default: latest non-quick).

The supplement answers three review questions (order effects, capacity slack, folded explicit A control). The formal
G1a ratios quoted for comparison are read from reports/g1a_summary_mac_m4max.json. Numbers come only from those files.
"""
from __future__ import annotations
import argparse
import glob
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/g1a_supplement_mac_m4max.md")
    a = ap.parse_args()
    if a.summary is None:
        cands = [p for p in sorted(glob.glob(str(ROOT / "runs/g1a-supp-*/summary.json"))) if "quick" not in p]
        if not cands:
            print("no non-quick supplement run found")
            return 1
        a.summary = Path(cands[-1])
    S = json.loads(a.summary.read_text(encoding="utf-8"))
    formal = json.loads((ROOT / "reports/g1a_summary_mac_m4max.json").read_text(encoding="utf-8"))
    shutil.copyfile(a.summary, ROOT / "reports/g1a_supplement_summary_mac_m4max.json")
    mp = a.summary.parent / "manifest.json"
    if mp.exists():
        home = str(Path.home())
        (ROOT / "reports/g1a_supplement_manifest_mac_m4max.json").write_text(mp.read_text(encoding="utf-8").replace(home, "~"), encoding="utf-8")
    cells = [c for c in S["cells"] if c.get("status") == "ok"]
    by = defaultdict(dict)
    for c in cells:
        by[(c["dtype"], c["N0"], c.get("rho_block", c["rho"]) if c["strategy"].startswith("order") else c["rho"], c["strategy"], c.get("block"))] = c
    dtypes = sorted({c["dtype"] for c in cells}, key=lambda s: s != "bfloat16")
    lengths = sorted({c["N0"] for c in cells})
    rhos = sorted({round(c["rho"], 6) for c in cells if c["strategy"].startswith("order:D")})
    formal_ratio = {(v["dtype"], v["N"], round(v["rho"], 6), v["strategy"]): v for v in formal["verdicts"]}

    def cell(dtype, N, rho, strat, block):
        for k, v in by.items():
            if k[0] == dtype and k[1] == N and k[3] == strat and k[4] == block and (rho is None or round(k[2], 6) == round(rho, 6)):
                return v
        return None

    L = []
    L.append("# G1a 补测报告 · 交错顺序、容量四格、折叠手写 A 对照（Mac M4 Max）\n")
    L.append(f"run_id `{S['run_id']}`，UTC {S['utc_start']} 到 {S['utc_end']}，设备活跃 {S['device_seconds']:.0f} s。由 `scripts/g1a_supplement_report.py` 从 `reports/g1a_supplement_summary_mac_m4max.json` 生成。"
             f"代码 commit `{S['git'].get('commit')}`，dirty diff sha256 `{S['git'].get('dirty_diff_sha256')}`。**随机张量、单层、batch 1、一步解码；MPS 结果不推 CUDA。**\n")
    L.append("## 0. 为什么补测\n")
    L.append("PR #1 评审（GPT 6 Pro，2026-10-17）指出三点：正式六进程固定按 A、D_full、D_read 顺序且同种子，顺序效应不会因为重启进程消失；容量松弛诊断同时改变了容量、数据、常驻内存与时序；A 的 `explicit` 参照是另一种五维广播写法，不能排除折叠布局下手写 QK→softmax→PV 更快。本补测在同一进程内把这三件事各自隔离。\n")

    L.append("## 1. 顺序：交错与顺序执行的 D/A 比值\n")
    L.append("每个 cycle 按 A, D_full, D_read, D_read, D_full, A（ABBA）或其反序（BAAB）各跑一个 pass，比值取 cycle 内 D 均值除以 A 均值；\"sequential\" 为正式运行的顺序，在同一进程、同一数据上重测。表中为每层 µs 的中位数；配对比值给中位数与范围。\n")
    L.append("| dtype | N | ρ | 块 | A µs | D_full µs | D_read µs | 配对 D_full/A 中位（范围） | 全部 cycle ≤ 0.90 | 配对 D_read/A 中位 | 正式六进程 D_full/A |\n|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|")
    for dtype in dtypes:
        for N in lengths:
            for rho in rhos:
                fr = formal_ratio.get((dtype, N, rho, "D_full"))
                for block in ("ABBA", "BAAB", "sequential"):
                    strat = "order_seq" if block == "sequential" else "order"
                    A = cell(dtype, N, rho, f"{strat}:A", block); Df = cell(dtype, N, rho, f"{strat}:D_full", block); Dr = cell(dtype, N, rho, f"{strat}:D_read", block)
                    if not (A and Df):
                        continue
                    pr = S["paired_ratios"].get(f"{dtype}:N{N}:rho{rho:.5f}:{block}:D_full")
                    prr = S["paired_ratios"].get(f"{dtype}:N{N}:rho{rho:.5f}:{block}:D_read")
                    paired = f"{pr['median']:.2f}（{pr['min']:.2f}–{pr['max']:.2f}）" if pr else f"{Df['us_per_layer_median']/A['us_per_layer_median']:.2f}（顺序执行，无配对）"
                    allb = ("是" if pr["all_below_0.90"] else "否") if pr else "—"
                    L.append(f"| {dtype} | {N} | 1/{round(1/rho)} | {block} | {f(A['us_per_layer_median'])} | {f(Df['us_per_layer_median'])} | {f(Dr['us_per_layer_median']) if Dr else '—'} | {paired} | {allb} | "
                             f"{f(prr['median'], 2) if prr else (f(Dr['us_per_layer_median']/A['us_per_layer_median'], 2) if Dr else '—')} | {f(fr['ratio_median'], 2) if fr else '—'}（{f(fr['ratio_min'], 2) if fr else '—'}–{f(fr['ratio_max'], 2) if fr else '—'}） |")

    L.append("\n## 2. 容量四格：同一组数据，容量恰好 N 与 N+2048，交错执行\n")
    L.append("两套缓冲同时常驻；松弛缓冲的前 N 列复制自紧凑缓冲，输出先核对相等再计时。顺序为 A_tight, A_slack, D_tight, D_slack, D_slack, D_tight, A_slack, A_tight。ρ 取首档。\n")
    L.append("| dtype | N | A 紧凑 µs | A 松弛 µs | 松弛/紧凑 | D_full 紧凑 µs | D_full 松弛 µs | 松弛/紧凑 | 输出差 A | 输出差 D |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dtype in dtypes:
        for N in lengths:
            At = cell(dtype, N, None, "cap:A_tight", "interleaved"); As = cell(dtype, N, None, "cap:A_slack", "interleaved")
            Dt = cell(dtype, N, None, "cap:D_tight", "interleaved"); Ds = cell(dtype, N, None, "cap:D_slack", "interleaved")
            eq = S["equality_checks"].get(f"{dtype}:N{N}:capacity", {})
            if not (At and As and Dt and Ds):
                continue
            L.append(f"| {dtype} | {N} | {f(At['us_per_layer_median'])} | {f(As['us_per_layer_median'])} | {As['us_per_layer_median']/At['us_per_layer_median']:.2f} | "
                     f"{f(Dt['us_per_layer_median'])} | {f(Ds['us_per_layer_median'])} | {Ds['us_per_layer_median']/Dt['us_per_layer_median']:.2f} | "
                     f"{eq.get('A_tight_vs_slack_max_abs', float('nan')):.1e} | {eq.get('D_tight_vs_slack_max_abs', float('nan')):.1e} |")

    L.append("\n## 3. A 对照：折叠布局下手写 QK→softmax→PV 与两种 SDPA\n")
    L.append("同一折叠布局（组内 h 个查询头折进序列轴），交错执行。数值差为对 fp32 `dense_reference` 的最大绝对差。\n")
    L.append("| dtype | N | sdpa_fold µs | explicit_fold µs | sdpa_gqa µs | explicit/fold | 数值差 explicit | 数值差 fold | 数值差 gqa |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dtype in dtypes:
        for N in lengths:
            fo = cell(dtype, N, None, "actrl:sdpa_fold", "interleaved"); ex = cell(dtype, N, None, "actrl:explicit_fold", "interleaved"); gq = cell(dtype, N, None, "actrl:sdpa_gqa", "interleaved")
            eq = S["equality_checks"].get(f"{dtype}:N{N}:a_control", {})
            if not (fo and ex and gq):
                continue
            L.append(f"| {dtype} | {N} | {f(fo['us_per_layer_median'])} | {f(ex['us_per_layer_median'])} | {f(gq['us_per_layer_median'])} | {ex['us_per_layer_median']/fo['us_per_layer_median']:.2f} | "
                     f"{eq.get('explicit_fold_vs_dense_reference_fp32_max_abs', float('nan')):.1e} | {eq.get('sdpa_fold_vs_dense_reference_fp32_max_abs', float('nan')):.1e} | {eq.get('sdpa_gqa_vs_dense_reference_fp32_max_abs', float('nan')):.1e} |")

    L.append("\n## 4. 读法与边界\n")
    L.append("- 第 1 节回答\"正式比值是不是顺序造成的\"：看交错块的配对比值是否与正式六进程的比值落在同一范围。")
    L.append("- 第 2 节回答\"容量松弛是否自带惩罚\"：同数据、同时常驻、交错执行下，松弛/紧凑接近 1 表示正式诊断里的差异来自数据、内存或时序而不是跨步视图本身。")
    L.append("- 第 3 节回答\"是否存在更快的合法 A\"：只在所比较的实现中，不是穷尽。")
    L.append("- 每层 µs 仍是单层读取阶段；p95 为 pass 耗时 ÷ 层数的 p95。没有模型、任务、质量结论。")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    rel = lambda q: str(q.resolve().relative_to(ROOT)) if q.resolve().is_relative_to(ROOT) else str(q)  # noqa: E731
    print(f"wrote {rel(a.out)} from {rel(a.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
