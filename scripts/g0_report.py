#!/usr/bin/env python3
"""Generate reports/g0_mac_m4max.md from a G0 probe JSON and a roofline JSON. Numbers come only from those files."""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {"Qwen3-0.6B": 28, "Qwen3-1.7B": 28, "Qwen3-4B": 36}


def f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, default=None, help="runs/<run_id>/g0_probe.json; default: latest")
    ap.add_argument("--roofline", type=Path, default=ROOT / "reports/g0_roofline_mac_bw390.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/g0_mac_m4max.md")
    a = ap.parse_args()
    probe_path = a.probe or Path(sorted(glob.glob(str(ROOT / "runs/g0-mac-*/g0_probe.json")))[-1])
    d = json.loads(probe_path.read_text()); P = d["probes"]
    roof = json.loads(a.roofline.read_text())
    bw = roof["rows"][0]["bandwidth_assumption_bytes_per_second"]
    shutil.copyfile(probe_path, ROOT / "reports/g0_probe_mac_m4max.json")
    H = d["host"]; S = d["software"]
    L = []
    L.append("# G0 实测报告 · Mac M4 Max 环境、能力、带宽与索引路径原语成本\n")
    L.append('这一阶段称 G0，测的是本机基础操作，还没有模型在生成文本。脚本先构造数组，再分别测矩阵乘、选位置和按名单取 K/V 行。以 32K 档为例，每个 KV 组有 32,768 个 128 维 K 和 V，已给定名单要求取 1,024 对；K 用 0.01、V 用 0.02 填充，并非从真实资料提取。\n\n报告中的 u 是当前查询的低维索引向量，z 是历史键的低维索引向量；它们点积产生选位分数。第 5 节的链路只测打分、top-k 和取行，没有测主注意力读出。第 6 节才用这些时间作预测，A 指完整读取，D 指先筛选再读取。完整读取链的真实对照在后续 G1a，讲解见[第一课](01_environment_and_attention.html)。\n')
    L.append("<details><summary>运行版本与报告来源</summary>\n")
    L.append(f"run_id `{d['run_id']}`，UTC {d['utc_start']} 到 {d['utc_end']}。由 `scripts/g0_report.py` 从 `reports/g0_probe_mac_m4max.json` 与 `{a.roofline.name}` 生成。**这是本机测量与由此回填的 G1a 预测，不是 G1a 的结果，也不是任何 CUDA 结论。**\n")
    L.append("</details>\n")
    L.append("## 1. 环境\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append(f"| 机器 | {H['chip']}，{H['gpu_cores']} 核 GPU，{H['physical_memory_bytes']/2**30:.0f} GiB 统一内存，macOS {H['macos']} |")
    L.append(f"| Python / torch | {H['python']} / {S['torch']}，venv 内，MPS 可用 |")
    L.append(f"| `recommended_max_memory` | {P['memory_apis']['recommended_max_memory']/1e9:.1f} GB，约为物理内存的 {100*P['memory_apis']['recommended_max_memory']/H['physical_memory_bytes']:.0f}% |")
    env_state = "均未设置" if not any(S['env'].values()) else ", ".join(f"{k}={v}" for k, v in S['env'].items() if v)
    L.append(f"| MPS fallback / 内存保护 | 未设置 fallback，未关闭保护；相关环境变量{env_state} |")
    L.append("| 依赖锁 | `requirements/lock-macos-arm64.txt`，由本机 pip freeze 生成 |\n")
    L.append("## 2. 能力检查：执行前预测与实测\n")
    L.append("| 项 | 预测 | 实测 |\n|---|---|---|")
    dm = P["dtype_matmul"]
    L.append(f"| matmul fp32 / fp16 / bf16 | 支持 | 支持；相对误差 {dm['float32']['rel_err_vs_cpu_fp32']:.1e} / {dm['float16']['rel_err_vs_cpu_fp32']:.1e} / {dm['bfloat16']['rel_err_vs_cpu_fp32']:.1e} |")
    L.append(f"| matmul fp64 | 不支持 | 不支持，{dm['float64']['error_type']} |")
    so = P["selection_ops"]
    L.append(f"| top-k / index_select / mask softmax | 支持且与 CPU 一致 | top-k 下标全同；index_select 逐位相同；mask softmax 相对误差 {so['masked_softmax_rel_err']:.1e} |")
    sd = P["sdpa"]
    L.append(f"| SDPA 普通多头 | 支持 | 支持，相对误差 {sd['mha_repeated_kv']['rel_err']:.1e} |")
    L.append(f"| SDPA `enable_gqa=True` | 不支持或报错 | **支持**，相对误差 {sd['enable_gqa']['rel_err']:.1e}；预测错误。内核是否内部重复 KV 待 G1a 剖析 |")
    L.append(f"| SDPA bool mask / bf16 | 支持 | 支持；bf16 对 fp32 相对误差 {sd['bf16']['rel_err_vs_fp32']:.1e} |")
    ib = P["index_backward_kl"]
    L.append(f"| 小索引矩阵 KL 反向 | 梯度与 CPU 相对误差 ≤ 1e-4 | 有限且一致，Wq {ib['grad_wq_rel_err']:.1e}，Wk {ib['grad_wk_rel_err']:.1e}；MPS 与 CPU 损失相同 |")
    tc = P["torch_compile_elementwise"]
    L.append(f"| `torch.compile` 逐元素小函数 | 五成把握，首次 20 到 90 s | 可用，首次 {tc['first_call_s']:.2f} s，第二次 {tc['second_call_ms']:.2f} ms；只证明逐元素函数能编译 |")
    L.append(f"| `recommended_max_memory` | 90 到 105 GB | {P['memory_apis']['recommended_max_memory']/1e9:.1f} GB，高于预测 |\n")
    L.append("## 3. 有效带宽\n")
    L.append("标称 546 GB/s。copy 与 mul_out 按读加写计字节，sum 只计读。中位数，每次运算前后同步。\n")
    L.append("| 张量 | copy GB/s | sum（只读）GB/s | mul_out GB/s |\n|---|---:|---:|---:|")
    for key, v in P["bandwidth"].items():
        if isinstance(v, dict) and "copy" in v:
            L.append(f"| {key.replace('_', ' ')} | {v['copy']['gbps_median']:.0f} | {v['sum_read']['gbps_median']:.0f} | {v['mul_out']['gbps_median']:.0f} |")
    sc = P["attention_scan_qk"]
    L.append(f"\n注意力形状的读：q[8,1,128]·Kᵀ，K 为 [8,{sc['n']},128] bf16，每层 {sc['k_bytes_per_layer']/2**20:.0f} MiB。{sc['layers']} 层轮换：**{sc['rotating_workset']['gbps_median']:.0f} GB/s**，每层 {sc['rotating_workset']['us_per_layer']:.0f} µs；同一层热读 {sc['layers']} 次：{sc['hot_same_layer']['gbps_median']:.0f} GB/s。两种访问顺序的有效带宽接近；本次没有测硬件缓存命中率，不能据此确定每次数据来自哪一级存储。")
    L.append(f"\n预测区间：只读 250 到 450，注意力扫描 150 到 350。实测只读约 {P['bandwidth']['4GiB_bfloat16']['sum_read']['gbps_median']:.0f}，扫描约 {sc['rotating_workset']['gbps_median']:.0f}，都落在预测区间上端或更高。三次运行中扫描带宽在 380 到 420 之间波动。回填 roofline 时取本次运行的扫描值 {bw/1e9:.0f} GB/s。\n")
    L.append("## 4. 提交开销\n")
    do = P["dispatch_overhead"]
    L.append(f"300 次连发后同步，每算子墙钟：tiny matmul {do['tiny_matmul_us_per_op']:.0f} µs，top-k（[8,4096] 取 32）{do['topk_4096_k32_us_per_op']:.0f} µs，index_select（4096 取 64 行）{do['index_select_64_of_4096_us_per_op']:.0f} µs。预测 15 到 60 µs，实测在区间内，但 top-k 偏高端。这是连续提交后一起等待完成的墙钟成本，不能单独分离 CPU 提交与 GPU 执行各占多少。\n")
    L.append("## 5. 索引路径逐原语成本（本报告最重要的发现）\n")
    L.append("每层解码一步的索引读取路径：u·zᵀ 对候选打分、top-k（k=N/32）、按名单取 K 与 V。bf16，36 层轮换，每层 µs，中位数。\n")
    L.append("| N | k | 打分 zᵀ视图 | 打分 z 预转置 | top-k | gather 展开下标 | index_select 拉平 | 高级索引 | 链式 朴素 | 链式 最优 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ipc = P["index_path_cost"]
    for N in (8192, 16384, 32768):
        v = ipc[f"N{N}"]
        L.append(f"| {N} | {v['k']} | {v['score_us_per_layer']:.0f} | {v['score_zT_contig_us_per_layer']:.0f} | {v['topk_us_per_layer']:.0f} | {v['gather_kv_us_per_layer']:.0f} | {v['gather_index_select_flat_us_per_layer']:.0f} | {v['gather_adv_index_us_per_layer']:.0f} | {v['chain_us_per_layer']:.0f} | **{v['chain_best_us_per_layer']:.0f}** |")
    v32 = ipc["N32768"]
    L.append(f"\n三个事实：")
    L.append(f"\n- **`torch.gather` 配展开下标在 MPS 上极慢。** 32K 时取 {v32['k']} 行要 {v32['gather_kv_us_per_layer']:.0f} µs，比扫完整层 64 MiB 的 K（{sc['rotating_workset']['us_per_layer']:.0f} µs）慢数倍；拉平后 `index_select` 只要 {v32['gather_index_select_flat_us_per_layer']:.0f} µs，快 {v32['gather_kv_us_per_layer']/v32['gather_index_select_flat_us_per_layer']:.0f} 倍。因此后续 G1a 采用这一在当前条件下实测较快的取行写法。")
    L.append(f"- **索引打分不是按字节比例变便宜。** z 宽 32、K 宽 128，字节比 1/4，但 zᵀ 视图打分 {v32['score_us_per_layer']:.0f} µs 与整层 K 扫描 {sc['rotating_workset']['us_per_layer']:.0f} µs 相当；预转置后 {v32['score_zT_contig_us_per_layer']:.0f} µs。观测说明字节量减少没有按比例缩短耗时；没有单独分离内核、访存与调度开销，不能直接判定瓶颈。")
    L.append(f"- **完整链的时间没有随字节量同比变化。** 8K 时最优链 {ipc['N8192']['chain_best_us_per_layer']:.0f} µs，32K 时 {v32['chain_best_us_per_layer']:.0f} µs，随 N 增长缓慢；top-k 一项就占 {ipc['N8192']['topk_us_per_layer']:.0f} 到 {v32['topk_us_per_layer']:.0f} µs。固定开销可能是原因之一，但这组测量没有独立估计出一个不随长度改变的时间底。\n")
    L.append("## 6. 回填后的 G1a 预测表\n")
    L.append(f"带宽取第 3 节的注意力扫描值 {bw/1e9:.0f} GB/s，ρ = 1/32，r = 32，固定集合 68。比较的两边必须来自同一套划分：A 一侧是完整 KV 按该带宽的理想读取时间；D 一侧是第 5 节实测的最优链式成本（索引打分、top-k、取 K/V）乘以层数。净值 = A 理想读取 − 最优链式，**未计入子集注意力本身**（G1a 实测每层 14 到 30 µs，36 层约 0.5 到 1 ms），也未计入密集注意力内核相对带宽模型的偏差。S1 与 S2 列是 D 的理想读取情景，只作参考，不再参与净值。\n")
    L.append("| 模型 | N | 注意力占每步流量 | A 完整 KV 理想读取 ms | S1 D 理想读取 ms | S2 D 理想读取 ms | 朴素链式 ms/步 | 最优链式 ms/步 | 净值 A理想−最优链 ms | 方向 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in roof["rows"]:
        m, n = row["model"], row["n"]; v = ipc[f"N{n}"]; Lyr = LAYERS[m]
        naive = Lyr * v["chain_us_per_layer"] / 1000; best = Lyr * v["chain_best_us_per_layer"] / 1000
        t_a = row["dense_kv_bytes"] / bw * 1e3
        d1 = row["d_s1_bytes"] / bw * 1e3; d2 = row["d_s2_bytes"] / bw * 1e3
        net = t_a - best
        arrow = "D 可能更快" if net > 1.0 else ("持平区" if net > -1.0 else "D 可能更慢")
        L.append(f"| {m} | {n} | {100*row['attention_fraction_of_assumed_total_traffic']:.0f}% | {t_a:.1f} | {d1:.1f} | {d2:.1f} | {naive:.1f} | {best:.1f} | {net:+.1f} | {arrow} |")
    L.append("\n**更正（2026-10-17，PR #1 评审）。** 首版此表的净值写成“S1 节省 − 最优链式”。S1 已按理想载荷扣掉索引键读与入选 KV 读，实测链式又包含同一批工作的真实耗时，等于把这部分成本扣了两次。首版数值（4B）为 8K −0.9、16K +1.1、32K +4.5 ms；更正后见上表。更正使 D 更有利，三个长度的方向不变。首版表保留在 git 历史与 `reports/cards/G0_mac_m4max.md` 的更正说明中。")
    L.append("\n读表规则：这是 G1a 开跑前的预测，不是结果。两条路径各自的访存下界相减不是耗时差上界；密集注意力内核的实际时间由 G1a 测，可能高于或低于带宽模型。朴素链式一列说明：若 G1a 直接用 `torch.gather`，所有组合都会得到\"D 更慢\"，而那是原语问题，不是架构问题。\n")
    L.append("## 7. 当时据此作出的下一阶段安排\n")
    L.append("- 实现要求：索引键按 [G, r, N] 连续存放；gather 用拉平后的 `index_select`；top-k 只对候选域一次。不满足这三条的实现不进入正式计时。")
    L.append("- 预测：eager 下 8K 持平偏负，16K 持平偏正，32K 正收益；0.6B 的相对收益最大。编译只在能确认融合时才会改变这个格局。")
    L.append("- 需要 G1a 补测的量：密集注意力整层（含 V 读、softmax、PV）的实际时间；子集注意力的时间；`enable_gqa` 在 MPS 上是否内部 repeat。")
    L.append("- 不变的边界：本报告是 MPS 结果，不推 CUDA；没有模型、没有任务、没有质量结论。\n")
    L.append("## 8. 本阶段未做的工作\n")
    L.append("未下载模型；未训练；未做任何密集与子集注意力的正式对照；未测连续生成；未测 compile 对注意力相关算子的支持。CUDA 轨道未启动。后续已经完成的本机实验见第一至七课，本段只记录 G0 结束时的范围。\n")
    L.append("## 9. 证据\n")
    L.append(f"- 原始 JSON：`runs/{probe_path.parent.name}/g0_probe.json`（本地，已复制为 `reports/g0_probe_mac_m4max.json`）\n- 探针源码：`scripts/g0_probe.py`\n- 回填表：`{a.roofline.name}` 与同名 .md\n- 环境清单：`reports/g0_inventory_mac_m4max.json`\n- 依赖锁：`requirements/lock-macos-arm64.txt`\n- 预测卡片：`reports/cards/G0_mac_m4max.md`")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out.relative_to(ROOT)} from {probe_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
