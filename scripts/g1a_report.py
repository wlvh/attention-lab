#!/usr/bin/env python3
"""Aggregate G1a microbenchmark runs into reports/g1a_mac_m4max.md plus condensed JSON. Numbers come only from the runs.

Usage: python scripts/g1a_report.py [--runs runs/g1a-mac-*/summary.json ...]
Default: every non-quick runs/g1a-mac-*/summary.json. Independent processes are aggregated as median-of-medians with
the min..max spread across processes; the pass rule needs every included process to clear the 10% threshold.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS_RATIO = 0.90
LAYERS = {"Qwen3-0.6B": 28, "Qwen3-1.7B": 28, "Qwen3-4B": 36}


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def load_runs(paths, include_quick=False):
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        if d.get("quick") and not include_quick:
            continue
        d["_path"] = str(Path(p).relative_to(ROOT)) if Path(p).resolve().is_relative_to(ROOT) else str(p)
        d["_abs_path"] = str(Path(p).resolve())
        runs.append(d)
    return runs


def key_of(c):
    return (c["dtype"], c["mode"], c["workset"], c["N0"], round(c["rho"], 6), c["strategy"], c.get("model_shape"), c.get("capacity"))


def aggregate(runs):
    """key -> {per_run: [...medians], median, min, max, p95_median, n_runs, backend, cell_example}."""
    per = defaultdict(list)
    for r in runs:
        for c in r["cells"]:
            if c.get("status") != "ok":
                continue
            per[key_of(c)].append((r["run_id"], c))
    agg = {}
    for k, items in per.items():
        meds = [c["us_per_layer_median"] for _, c in items]
        agg[k] = {"per_run": {rid: c["us_per_layer_median"] for rid, c in items}, "median": statistics.median(meds), "min": min(meds), "max": max(meds),
                  "p95_median": statistics.median([c["us_per_layer_p95"] for _, c in items]), "n_runs": len(items), "backend": items[0][1].get("backend"),
                  "cell": {kk: vv for kk, vv in items[0][1].items() if kk not in ("raw_ms_per_pass", "memory_before", "compile_info")}}
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/g1a_mac_m4max.md")
    ap.add_argument("--json", type=Path, default=ROOT / "reports/g1a_summary_mac_m4max.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "reports/g1a_manifest_mac_m4max.json")
    ap.add_argument("--card", default="reports/cards/G1a_mac_m4max.md")
    ap.add_argument("--include-quick", action="store_true", help="dry runs only: include smoke-test runs")
    a = ap.parse_args()
    paths = a.runs or sorted(glob.glob(str(ROOT / "runs/g1a-mac-*/summary.json")))
    runs = load_runs(paths, a.include_quick)
    if not runs:
        print("no non-quick runs found")
        return 1
    agg = aggregate(runs)
    eager = [r for r in runs if "compile" not in r["args"].get("modes", [])]
    compile_runs = [r for r in runs if "compile" in r["args"].get("modes", [])]
    dtypes = sorted({k[0] for k in agg}, key=lambda s: s != "bfloat16")
    lengths = sorted({k[3] for k in agg})
    rhos = sorted({k[4] for k in agg if k[4] > 0})
    first = runs[0]
    H, S = first["host"], first["software"]
    shape_main = f"B1_G{first['args']['groups']}_H{first['args']['heads']}_d{first['args']['head_dim']}_r{first['args']['index_dim']}"

    def get(dtype, mode, workset, N, rho, strategy, shape=shape_main, capacity=None):
        for k, v in agg.items():
            if k[:6] == (dtype, mode, workset, N, round(rho, 6), strategy) and k[6] == shape and (capacity is None or k[7] == capacity):
                return v
        return None

    # pass evaluation: bf16 (and fp32) eager rotating, D_full and D_read vs A, every independent process must clear 10%
    verdicts = []
    for dtype in dtypes:
        for N in lengths:
            for rho in rhos:
                A = get(dtype, "eager", "rotating", N, rho, "A")   # A is re-measured inside every rho block
                for strat in ("D_full", "D_read"):
                    D = get(dtype, "eager", "rotating", N, rho, strat)
                    if not A or not D:
                        continue
                    ratios = {rid: D["per_run"][rid] / A["per_run"][rid] for rid in D["per_run"] if rid in A["per_run"]}
                    prof = None
                    for r in runs:
                        pr = r.get("profiles", {}).get(f"{dtype}:N{N}")
                        if pr and "d_full_qk_detected" in pr:
                            prof = pr["d_full_qk_detected"]
                            break
                    ok = bool(ratios) and all(x <= PASS_RATIO for x in ratios.values()) and len(ratios) >= 2 and prof is False
                    verdicts.append({"dtype": dtype, "N": N, "rho": rho, "strategy": strat, "A_us": A["median"], "D_us": D["median"], "ratio_median": statistics.median(ratios.values()) if ratios else None,
                                     "ratio_min": min(ratios.values()) if ratios else None, "ratio_max": max(ratios.values()) if ratios else None, "n_runs": len(ratios),
                                     "every_run_below_0.9": bool(ratios) and all(x <= PASS_RATIO for x in ratios.values()), "d_full_qk_detected": prof, "pass": ok})
    passing = [v for v in verdicts if v["pass"] and v["dtype"] == "bfloat16" and v["strategy"] == "D_full"]
    passing_any = [v for v in verdicts if v["pass"]]

    L = []
    L.append("# G1a 实测报告 · Mac M4 Max 单层读取阶段：完整注意力 A 与索引子集 D\n")
    L.append('本报告比较已经准备好 Q/K/V 后的一次注意力读取，项目编号为 G1a。A 用当前 Q 给全部合法 K 打分、softmax 后加权 V。D 先为最新位置补索引，再用当前查询的低维投影给候选打分、选位、取出原 K/V，最后由主 Q/K 重新计算权重并组合 V。两者输出都是注意力向量，不含输出投影、残差、MLP 或答案生成。\n\n本档为 32 个查询头、8 个 K/V 头，每个 KV 组的 4 个查询头共享一套历史 K/V，索引读取时共用一份名单。以下名额和位置数均按本步某一层、某一 KV 组计算。以 32K 为例，D 保留开头 4 个和最近 64 个位置，从其余 32,700 个位置中再选 1,024 个，合并后读 1,092 个。额外 1,024 来自起始长度 32,768 乘名额参数 ρ=1/32；不是把必留 68 个算进这个 1/32。索引每组把当前查询投影成 32 维 u，与各历史位置的 32 维 z 点积，选最高分；它的分数只决定名单，不直接加权 V。\n\n这里的主 Q/K/V、索引查询输入和已有历史索引表 zT 都直接随机生成，索引参数未训练，没有实际文本或 RoPE 计算。D_full 计入新 z 写入、u 投影、打分、选位、取行和主注意力；D_read 只省新 z 写入，其余照做。已有历史 zT 的完整构建不计入这一步。\n\n每轮执行 36 次：hot 反复访问同一套数组，rotating 依次访问 36 套独立数组，前一次输出不送给下一次。将整轮时间除以 36 得到每套数据的平均读取时间。讲解与源码对应见[第一课](01_environment_and_attention.html)。\n')
    L.append("<details><summary>运行版本与报告来源</summary>\n")
    L.append(f"由 `scripts/g1a_report.py` 从 {len(runs)} 个独立进程的 `runs/g1a-mac-*/summary.json` 生成（{', '.join(r['run_id'] for r in runs)}）。"
             f"**随机张量、单层、batch 1、一步解码；没有模型、没有任务、没有质量结论；MPS 结果不推 CUDA。** 预测见 `{a.card}`。\n")
    L.append("</details>\n")
    L.append("## 1. 环境与设置\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append(f"| 机器 | {H.get('chip')}，{H.get('gpu_cores')} 核 GPU，{(H.get('physical_memory_bytes') or 0)/2**30:.0f} GiB，macOS {H.get('macos')} |")
    L.append(f"| torch / Python | {S.get('torch')} / {H.get('python')}，MPS；fallback 与内存保护环境变量均未设置 |")
    L.append(f"| 形状 | {shape_main}（G=8 组，H=32 查询头，h=4，d=128，r=32），B=1，T=1，query_position=N−1 |")
    L.append(f"| 固定集合 | 首 {4} + 近 {64}；候选域 = 合法域 − F；k0 = min(round(ρN0), |候选|) |")
    L.append(f"| 计时 | 每 pass {first['args']['layers']} 层步；预热 {first['args']['warmup']}、正式 {first['args']['repeats']}；每 pass 前后 `torch.mps.synchronize()`；中位数与 p95。**p95 是 pass 耗时 ÷ 层数的 p95，不是单个解码步延迟的 p95** |")
    L.append(f"| 工作集 | rotating：{first['args']['layers']} 层各自缓冲轮换；hot：同一层重复 |")
    L.append(f"| 代码 | commit {first['git'].get('commit')}，dirty diff sha256 {first['git'].get('dirty_diff_sha256')} |")
    L.append(f"| 独立进程 | eager {len(eager)} 个；含 compile 的 {len(compile_runs)} 个 |\n")

    L.append("## 2. 主表：每层 µs（跨进程中位数，括号内为进程间最小..最大）\n")
    L.append("A = 完整注意力（本进程实测最快的合法 GQA 变体，见第 4 节）；D_full = z 追加 + u 投影 + 候选打分 + top-k + 合并 + index_select K/V + 子集注意力；D_read = 不含 z 追加。比值 < 0.90 为通过线。\n")
    for dtype in dtypes:
        for workset in ("rotating", "hot"):
            L.append(f"\n### {dtype} · {workset} · eager\n")
            L.append("| N | ρ | k | 读取数 S | A µs | A p95（pass÷层数） | D_full µs | D_full/A | D_read µs | D_read/A | 进程数 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for N in lengths:
                for rho in rhos:
                    A = get(dtype, "eager", workset, N, rho, "A")
                    Df = get(dtype, "eager", workset, N, rho, "D_full")
                    Dr = get(dtype, "eager", workset, N, rho, "D_read")
                    if not A or not Df:
                        continue
                    c = Df["cell"]
                    L.append(f"| {N} | 1/{round(1/rho)} | {c['k_actual']} | {c['selected_count']} | {f(A['median'])} ({f(A['min'])}..{f(A['max'])}) | {f(A['p95_median'])} | "
                             f"{f(Df['median'])} ({f(Df['min'])}..{f(Df['max'])}) | **{Df['median']/A['median']:.2f}** | {f(Dr['median']) if Dr else '—'} | "
                             f"{(Dr['median']/A['median']):.2f} | {Df['n_runs']} |" if Dr else
                             f"| {N} | 1/{round(1/rho)} | {c['k_actual']} | {c['selected_count']} | {f(A['median'])} | {f(A['p95_median'])} | {f(Df['median'])} | **{Df['median']/A['median']:.2f}** | — | — | {Df['n_runs']} |")

    L.append("\n## 3. 通过判定（eager、rotating、每个独立进程分别算比值）\n")
    L.append("| dtype | N | ρ | 策略 | A µs | D µs | 比值中位 | 比值范围 | 进程数 | 每个进程都 ≤ 0.90 | 剖析 D 无完整 QK | 通过 |\n|---|---:|---:|---|---:|---:|---:|---|---:|---|---|---|")
    for v in verdicts:
        L.append(f"| {v['dtype']} | {v['N']} | 1/{round(1/v['rho'])} | {v['strategy']} | {f(v['A_us'])} | {f(v['D_us'])} | {f(v['ratio_median'], 2)} | {f(v['ratio_min'], 2)}..{f(v['ratio_max'], 2)} | {v['n_runs']} | "
                 f"{'是' if v['every_run_below_0.9'] else '否'} | {'是' if v['d_full_qk_detected'] is False else ('否' if v['d_full_qk_detected'] else '未剖析')} | {'**通过**' if v['pass'] else '不通过'} |")
    if passing:
        items = ", ".join("N=%d ρ=1/%d（%.2f）" % (v["N"], round(1 / v["rho"]), v["ratio_median"]) for v in passing)
        L.append(f"\n**结论：bf16 eager 下 D_full 通过的组合：{items}。** 投资门槛达到，当时据此进入 G1b；真实模型结果已在第二课交付；10% 是门槛，不是架构定理。")
    elif passing_any:
        items = ", ".join("%s %s N=%d ρ=1/%d" % (v["dtype"], v["strategy"], v["N"], round(1 / v["rho"])) for v in passing_any)
        L.append(f"\n**结论：bf16 D_full 没有组合通过；通过的只有 {items}。**")
    else:
        L.append("\n**结论：没有任何组合在重复试验中达到 10%，速度主线按方案收口为负结果。**")

    L.append("\n## 4. A 的实现变体（eager、rotating）：enable_gqa 是否内部重复 KV\n")
    L.append("| dtype | N | sdpa_fold | sdpa_gqa | sdpa_repeat（物化重复，诊断） | explicit（参照） | 选用 |\n|---|---:|---:|---:|---:|---:|---|")
    for dtype in dtypes:
        for N in lengths:
            vals = {b: get(dtype, "eager", "rotating", N, 0.0, f"A_variant:{b}") for b in ("sdpa_fold", "sdpa_gqa", "sdpa_repeat", "explicit")}
            chosen = None
            for r in runs:
                ch = r.get("a_backend_chosen", {}).get(f"{dtype}:N{N}")
                if ch:
                    chosen = ch["backend"]
                    break
            L.append(f"| {dtype} | {N} | " + " | ".join(f(v["median"]) if v else "—" for v in vals.values()) + f" | {chosen} |")

    L.append("\n## 5. D 的分段成本（eager、rotating、每层 µs）\n")
    L.append("各段单独计时，且 top-k、合并、gather、子集注意力复用同一份预先算好的中间量（s0、top0、rows0、ksub0/vsub0）；完整链式则在链中产生并消费本层的新中间量。因此分段之和与链式时间不相等，差异可能来自数据依赖、分配、缓存状态和提交方式，**现有证据不能分离这些来源**。\n")
    comps = ["comp:z_append", "comp:u_proj", "comp:score_candidates", "comp:topk", "comp:merge_rows", "comp:gather_k", "comp:gather_v", "comp:subset_attention"]
    L.append("| dtype | N | ρ | " + " | ".join(c.split(':')[1] for c in comps) + " | 分段之和 | D_full 链式 |\n|---|---:|---:|" + "---:|" * (len(comps) + 2))
    for dtype in dtypes:
        for N in lengths:
            for rho in rhos:
                vals = [get(dtype, "eager", "rotating", N, rho, c) for c in comps]
                if not any(vals):
                    continue
                Df = get(dtype, "eager", "rotating", N, rho, "D_full")
                total = sum(v["median"] for v in vals if v)
                L.append(f"| {dtype} | {N} | 1/{round(1/rho)} | " + " | ".join(f(v["median"]) if v else "—" for v in vals) + f" | {f(total)} | {f(Df['median']) if Df else '—'} |")

    L.append("\n## 6. torch.compile（两边同时加；记录 graph break、首次编译时间、重编译与数值差）\n")
    if compile_runs:
        L.append("| dtype | N | ρ | 策略 | 状态 | graphs | breaks | 首次编译 s | 对 eager 最大绝对差 | eager µs | compile µs | compile/eager |\n|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in compile_runs:
            for key, info in r.get("compile", {}).items():
                dtype, Ns, rs = key.split(":")
                N = int(Ns[1:]); rho = float(rs[3:])
                for strat, ci in info.items():
                    ce = get(dtype, "compile", "rotating", N, rho, strat)
                    ee = get(dtype, "eager", "rotating", N, rho, strat)
                    cu = ce["per_run"].get(r["run_id"]) if ce else None
                    eu = ee["per_run"].get(r["run_id"]) if ee else None
                    diff = ci.get("max_abs_diff_vs_eager")
                    diff_s = "—" if diff is None else "%.2e" % diff
                    status = str(ci.get("status")) + ((" " + str(ci.get("error_type"))) if ci.get("status") != "ok" else "")
                    L.append(f"| {dtype} | {N} | 1/{round(1/rho)} | {strat} | {status} | {ci.get('graph_count', '—')} | {ci.get('graph_break_count', '—')} | "
                             f"{f(ci.get('first_call_s'))} | {diff_s} | {f(eu)} | {f(cu)} | {f(cu/eu, 2) if cu and eu else '—'} |")
        recomp = [c for r in compile_runs for c in r["cells"] if c.get("mode") == "compile" and c.get("status") == "error"]
        L.append(f"\ncompile 模式下出错或重编译的单元：{len(recomp)}" + ("；" + "; ".join(f"{c['strategy']} N={c['N0']} {c.get('error_type')}" for c in recomp[:8]) if recomp else "。"))
    else:
        L.append("未运行 compile 模式。")

    L.append("\n## 7. gather 敏感性诊断（同样 S 行，不同名单结构；eager、rotating、每层 µs）\n")
    L.append("| dtype | N | ρ | top-k 实选 | 均匀随机 | 均匀随机已排序 | 成簇连续 |\n|---|---:|---:|---:|---:|---:|---:|")
    for dtype in dtypes:
        for N in lengths:
            for rho in rhos:
                vals = [get(dtype, "eager", "rotating", N, rho, f"gather_sens:{n}") for n in ("topk_actual", "uniform", "uniform_sorted", "clustered")]
                if any(vals):
                    L.append(f"| {dtype} | {N} | 1/{round(1/rho)} | " + " | ".join(f(v["median"]) if v else "—" for v in vals) + " |")
    L.append("\n这是 gather 对名单结构的敏感性，不是学习型 D 的真实选择。")

    L.append("\n## 8. 其他诊断\n")
    L.append("| dtype | N | A（容量=N） | A（容量 N+2048，切片视图） | A（H=16） | D_full（H=16，ρ=首档） |\n|---|---:|---:|---:|---:|---:|")
    shape16 = shape_main.replace(f"_H{first['args']['heads']}_", "_H16_")
    for dtype in dtypes:
        for N in lengths:
            A = get(dtype, "eager", "rotating", N, rhos[0], "A") if rhos else None
            Sl = get(dtype, "eager", "rotating", N, 0.0, "A_capacity_slack", capacity=N + 2048)
            A16 = get(dtype, "eager", "rotating", N, 0.0, "A", shape=shape16)
            D16 = get(dtype, "eager", "rotating", N, rhos[0], "D_full", shape=shape16) if rhos else None
            L.append(f"| {dtype} | {N} | {f(A['median']) if A else '—'} | {f(Sl['median']) if Sl else '—'} | {f(A16['median']) if A16 else '—'} | {f(D16['median']) if D16 else '—'} |")

    L.append("\n## 9. 剖析：D 没有先算完整 QK\n")
    for r in runs[:1]:
        for key, pr in r.get("profiles", {}).items():
            if "d_full_qk_detected" not in pr:
                L.append(f"- {key}: 剖析失败 {pr.get('error_type')}")
                continue
            L.append(f"- {key}: A 的 SDPA 调度 {pr['A_sdpa_dispatch']}；A 中含 N×d 输入的 matmul 类算子 {len(pr['A_full_qk_ops'])} 个；"
                     f"D_full 中 {len(pr['D_full_qk_ops'])} 个（d_full_qk_detected = {pr['d_full_qk_detected']}）；D 中触及长度 N 的算子："
                     + ("; ".join(f"{o['name']} {o['input_shapes']}" for o in pr["D_ops_touching_N"][:6]) or "无") + "。")
    L.append("\n完整算子表见各 run 的 `profiles.txt`。")

    L.append("\n## 10. 边界\n")
    L.append("- 单层读取阶段的时间，不是整模型每步时间；整模型瓶颈由 G1b 实测。按层数外推只能作为量级：36 层 × 每层差值。")
    L.append("- 随机张量下 top-k 名单近似均匀；真实轨迹的局部性、ρ 与 k 规则需重测。")
    L.append("- A 采用本机实测最快的合法 GQA 变体；SDPA 内核名称以 aten 调度层为准，Metal 内核级别未用 Instruments 核对。")
    L.append("- 没有模型、任务、质量结论；MPS 结果不推 CUDA。")
    L.append("- summary.json 中的 `memory_peaks` 是各 (dtype, N) 结束时 `torch.mps` 分配量的时点快照，不是运行期真实峰值。")
    L.append("- 六个正式进程按 A、D_full、D_read 的固定顺序、相同数据种子执行；顺序效应未在本轮消除。交错顺序的补测见 `reports/g1a_supplement_mac_m4max.md`。")

    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    cond = {"schema_version": 1, "kind": "benchmark_summary", "runs": [r["run_id"] for r in runs], "pass_ratio": PASS_RATIO, "verdicts": verdicts,
            "aggregate": {"|".join(str(x) for x in k): {kk: vv for kk, vv in v.items() if kk != "cell"} | {"cell": v["cell"]} for k, v in agg.items()},
            "a_backend_chosen": {k: v for r in runs for k, v in r.get("a_backend_chosen", {}).items()},
            "compile": {r["run_id"]: r.get("compile", {}) for r in compile_runs},
            "profiles": {r["run_id"]: {k: {kk: vv for kk, vv in p.items() if kk not in ("A_ops", "D_full_ops")} for k, p in r.get("profiles", {}).items()} for r in runs}}
    a.json.write_text(json.dumps(cond, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifests = []
    home = str(Path.home())
    for r in runs:
        mp = Path(r["_abs_path"]).parent / "manifest.json"      # next to its summary.json, so an extracted archive works too
        if mp.exists():
            manifests.append(json.loads(mp.read_text().replace(home, "~")))   # no absolute home paths in public files
    a.manifest.write_text(json.dumps({"schema_version": 1, "kind": "benchmark", "runs": manifests}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    rel = lambda q: str(q.relative_to(ROOT)) if q.resolve().is_relative_to(ROOT) else str(q)  # noqa: E731
    print(f"wrote {rel(a.out)}, {rel(a.json)}, {rel(a.manifest)} from {len(runs)} runs; "
          f"passing bf16 D_full combos: {[(v['N'], v['rho']) for v in passing]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
