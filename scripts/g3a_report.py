#!/usr/bin/env python3
"""Rebuild G3a tables from archived per-query records. Standard library; no torch/model.

Plot helpers import matplotlib only on demand. Raw frozen feature arrays remain in
ignored runs/ with hashes; the compact archive includes all metrics, positions,
documents, index weights, source snapshots and both preflight outcomes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/provenance/g3a_evidence_20261018.tar.gz"
INDEX = ROOT / "reports/provenance/g3a_archive_manifest.json"
METRICS = ("kl", "coverage", "candidate_mass", "missed_full_mass", "teacher_entropy", "output_rmse", "output_relative_l2")
METHODS = ("C", "D0", "D", "R")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def package(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "done" or manifest["mode"] != "formal":
        raise ValueError("only complete formal evidence may be packaged as the result")
    files = {}
    for directory in sorted((ROOT / "runs").glob("g3a-*-preflight")) + [run]:
        for p in directory.iterdir():
            if p.is_file():
                files[f'{directory.name}/{p.name}'] = p.read_bytes()
    files["budget.json"] = (ROOT / "runs/g3a-budget-20261018.json").read_bytes()
    for path, digest in manifest["sources"].items():
        content = subprocess.check_output(["git", "show", f'{manifest["git"]["commit"]}:{path}'], cwd=ROOT)
        if sha(content) != digest:
            raise ValueError(f"as-run source does not match commit: {path}")
        files["source/" + path] = content
    files["source/configs/mac_g2lite.json"] = subprocess.check_output(["git", "show", f'{manifest["git"]["commit"]}:configs/mac_g2lite.json'], cwd=ROOT)
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
    dump(INDEX, {"archive": str(ARCHIVE.relative_to(ROOT)), "sha256": sha(ARCHIVE.read_bytes()),
                 "formal_run": run.name, "files": {k: {"bytes": len(v), "sha256": sha(v)} for k, v in files.items()},
                 "excluded": "features/*.npz remain local in runs/, individually hashed in the original manifest; not uploaded"})


def read_evidence():
    index = json.loads(INDEX.read_text())
    if sha(ARCHIVE.read_bytes()) != index["sha256"]:
        raise ValueError("archive hash mismatch")
    with tarfile.open(ARCHIVE) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    if set(files) != set(index["files"]):
        raise ValueError("archive member set mismatch")
    for name, content in files.items():
        meta = index["files"][name]
        if len(content) != meta["bytes"] or sha(content) != meta["sha256"]:
            raise ValueError(f"member mismatch: {name}")
    prefix = index["formal_run"] + "/"
    get = lambda name: json.loads(files[prefix + name])
    records = [json.loads(line) for line in files[prefix + "metrics.jsonl"].splitlines()]
    docs = [json.loads(line) for line in files[prefix + "data.jsonl"].splitlines()]
    manifest, status = get("manifest.json"), get("summary.json")
    if status["status"] != "done" or manifest["status"] != "done" or status["mode"] != "formal" or status["completed"] != status["planned"]:
        raise ValueError("incomplete/diagnostic run is not a formal result")
    for name, digest in manifest["output_hashes"].items():
        if name in ("summary.json", "manifest.json"):
            continue
        if sha(files[prefix + name]) != digest:
            raise ValueError(f"original output hash mismatch: {name}")
    for path, digest in manifest["sources"].items():
        if sha(files["source/" + path]) != digest:
            raise ValueError(f"source mismatch: {path}")
    if sha(files[prefix + "index_final.npz"]) != manifest["final_checkpoint_sha256"]:
        raise ValueError("checkpoint mismatch")
    return dict(index=index, manifest=manifest, status=status, records=records, docs=docs,
                training=get("training.json"), files=files, budget=json.loads(files["budget.json"]))


def summarize(evidence):
    docs, records = evidence["docs"], evidence["records"]
    cfg = evidence["manifest"]["config"]
    if len({d["doc_id"] for d in docs}) != len(docs) or any(
        sum(d["split"] == split for d in docs) != cfg[key]
        for split, key in (("train", "train_docs"), ("heldout", "heldout_docs"))
    ):
        raise ValueError("document split/count mismatch")
    expected = {(d["doc_id"], qi, m) for d in docs for qi in range(len(cfg["query_offsets_from_suffix_end"])) for m in METHODS}
    actual = [(r["doc_id"], r["query_index"], r["method"]) for r in records]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("missing/duplicate query/method records")
    by_doc = {d["doc_id"]: d for d in docs}
    pools = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["split"] != by_doc[r["doc_id"]]["split"]:
            raise ValueError("split mismatch")
        for positions in r["positions"]:
            if len(positions) != r["selected_count"] or len(set(positions)) != len(positions) or not all(0 <= p < r["legal"] for p in positions):
                raise ValueError("invalid original-position selection")
        for metric in METRICS:
            if r[metric] is None:
                continue
            for value, eligible in zip(r[metric], r["eligible"]):
                if metric in ("kl", "coverage", "teacher_entropy") and not eligible:
                    continue
                pools[(r["split"], r["doc_id"], r["method"])][metric].append(value)
    document_results = []
    for (split, doc_id, method), metrics in sorted(pools.items()):
        document_results.append(dict(split=split, doc_id=doc_id, method=method,
                                     **{k: statistics.mean(v) for k, v in metrics.items()}))
    aggregate = {}
    for split in ("train", "heldout"):
        aggregate[split] = {}
        for method in METHODS:
            rows = [r for r in document_results if r["split"] == split and r["method"] == method]
            aggregate[split][method] = {metric: statistics.mean(r[metric] for r in rows if metric in r) if any(metric in r for r in rows) else None for metric in METRICS}
            aggregate[split][method]["documents"] = len(rows)
    paired = []
    for d in docs:
        methods = {r["method"]: r for r in document_results if r["doc_id"] == d["doc_id"]}
        paired.append(dict(doc_id=d["doc_id"], split=d["split"],
                           delta_kl=methods["D"]["kl"] - methods["D0"]["kl"],
                           delta_coverage=methods["D"]["coverage"] - methods["D0"]["coverage"],
                           delta_output_error=methods["D"]["output_relative_l2"] - methods["D0"]["output_relative_l2"]))
    final_qi = len(cfg["query_offsets_from_suffix_end"]) - 1
    held = [r for r in records if r["split"] == "heldout" and r["query_index"] == final_qi]
    gains = {}
    for d in docs:
        if d["split"] == "heldout":
            selected = {r["method"]: r for r in held if r["doc_id"] == d["doc_id"]}
            gains[d["doc_id"]] = statistics.mean(selected["D"]["coverage"]) - statistics.mean(selected["D0"]["coverage"])
    cases = []
    for label, doc_id in (("最大改善", max(gains, key=gains.get)), ("最差变化", min(gains, key=gains.get))):
        d = by_doc[doc_id]
        cases.append({"label": label, "doc_id": doc_id, "selection_rule": "留出最终查询、8组平均 coverage 的 D-D0 极值；展示组固定为0",
                      "question": d["question"], "target_records": [f'编号 {d["ids"][i]}：{d["values"][i]}' for i in d["target_indices"]],
                      "target_spans": d["target_spans"], "prefix_tokens": d["prefix_token_count"],
                      "group": 0, "delta_coverage": gains[doc_id], "records": [r for r in held if r["doc_id"] == doc_id]})
    h = aggregate["heldout"]
    support = h["D"]["kl"] < min(h["D0"]["kl"], h["C"]["kl"]) and h["D"]["coverage"] > max(h["D0"]["coverage"], h["C"]["coverage"])
    held_rows = [r for r in records if r["split"] == "heldout"]
    by_group = []
    for group in range(8):
        item = {"group": group, "methods": {}}
        for method in METHODS:
            item["methods"][method] = {}
            for metric in METRICS:
                values = [r[metric][group] for r in held_rows if r["method"] == method and r[metric] is not None
                          and (metric not in ("kl", "coverage", "teacher_entropy") or r["eligible"][group])]
                item["methods"][method][metric] = statistics.mean(values) if values else None
        by_group.append(item)
    dr = [r for r in held_rows if r["method"] == "D"]
    _, residual_row, group = max((r["output_relative_l2"][g], i, g) for i, r in enumerate(dr) for g in range(8))
    worst = dr[residual_row]
    residual = {"doc_id": worst["doc_id"], "query_index": worst["query_index"], "group": group,
                "selection_rule": "事后展示留出集中 D 的相对输出误差最大行；没有据此调整训练",
                "records": [r for r in held_rows if r["doc_id"] == worst["doc_id"] and r["query_index"] == worst["query_index"]]}
    paired_rows = {(r["doc_id"], r["query_index"]): r for r in held_rows if r["method"] == "D0"}
    regressions = {metric: sum((r[metric][g] < paired_rows[r["doc_id"], r["query_index"]][metric][g]) if metric == "coverage"
                              else (r[metric][g] > paired_rows[r["doc_id"], r["query_index"]][metric][g])
                              for r in dr for g in range(8)) for metric in ("coverage", "output_relative_l2")}
    return {"run_id": evidence["status"]["run_id"], "run_commit": evidence["manifest"]["git"]["commit"],
            "report_source_sha256": sha(Path(__file__).read_bytes()), "archive_sha256": evidence["index"]["sha256"],
            "status": "single_layer_descriptive_support" if support else "insufficient_evidence",
            "scope": "full-path frozen features, one layer; not task accuracy, sparse trajectory or speed",
            "aggregate": aggregate, "by_document": document_results, "paired": paired, "cases": cases,
            "heldout_by_group": by_group, "residual_case": residual, "query_group_regressions_vs_D0": regressions,
            "training": evidence["training"], "budget": evidence["budget"],
            "range": {k: [min(r[k] for r in records), max(r[k] for r in records)] for k in ("k0", "k", "selected_count", "candidate_count", "legal")},
            "prefix_tokens_range": [min(d["prefix_token_count"] for d in docs), max(d["prefix_token_count"] for d in docs)],
            "rows": len(records), "skipped_train_rows": evidence["manifest"]["skipped_train_rows"]}


def table(summary, split):
    lines = ["| 方法 | KL（nats）↓ | 条件覆盖率 ↑ | 候选总权重 m_C | 遗漏完整权重 ↓ | 输出 RMSE ↓ | 输出相对 L2 ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        a = summary["aggregate"][split][m]
        fmt = lambda x: "—" if x is None else f"{x:.5f}"
        lines.append(f'| {m} | ' + " | ".join(fmt(a[k]) for k in ("kl", "coverage", "candidate_mass", "missed_full_mass", "output_rmse", "output_relative_l2")) + " |")
    return "\n".join(lines)


def markdown(summary, evidence):
    m = evidence["manifest"]
    paired = [r for r in summary["paired"] if r["split"] == "heldout"]
    residual = summary["residual_case"]
    residual_d = next(r for r in residual["records"] if r["method"] == "D")
    g = residual["group"]
    return f'''# G3a：单层索引器是否学会了选择

完整模型处理按编号取四位值的 T1 资料，再逐 token 处理问题和模板尾部。在第 14 层保存倒数第 24/16/8/1 个位置的 Q，以及该层整段历史 K/V；四个查询是同一份输入中的四个位置，不是四道题。评测位置 p 时只取截至 p 的 K/V，不能读取后续问题位置。本批 64 份输入的问句占 26 个 token，加模板后缀后共 35 个；四个采样点分别是“数值”“四位”、用户消息结束后的换行、空 thinking 模板后的两个换行（后缀零基下标 11、19、27、34）。最后一个位置才是首个答案预测处。这是采样位置说明，不能由 token 的表面文本断言查询只包含这些词的信息。

随后把这些数组固定：一份训练资料的四个查询各有 8 个 KV 组，用当前 Wq/Wk 计算候选分数与 softmax，汇总它们相对教师的 KL，再更新 Wq/Wk。48 份训练资料逐份更新，固定 50 轮；16 份留出只在最终参数固定后评测。语言模型不参与反向更新，也没有让稀疏模型自行回答。

评测时必留开头 4 个和最近 64 个位置，额外名额按请求第一步长度的 1/8 固定。C 均匀随机选候选，D0 用训练前参数，D 用最终参数，R 直接用完整教师权重选。R 已付完整打分成本，只作诊断。索引把主 RoPE 前 Q/K 投影到 32 维，不再旋转，称“索引位置设置 P0”。它与旧 Metal 原型阶段的 P0 无关。

教师在同一 KV 组内先平均各查询头的注意力概率，再去掉必留位置并将候选权重归一化。索引器只需学习候选之间如何竞争额外名额。表中条件覆盖率数的是选中候选保住多少教师权重；m_C 是候选原本占完整权重的多少。训练分数、选位和读出的对应过程见[第四课](04_can_an_indexer_learn.html)。

判定：**{summary['status']}**。这是预登记条件下的描述性检查，不是总体泛化保证。只训练第 14 层的 98,304 个索引参数，主体、其他层和旧实验未改变。没有生成新答案，没有部署全层稀疏读取，也没有测加速。

评测还检查选位是否改变实际读出：取 F 与入选候选上的原主注意力概率，在这些位置内重新归一化，再加权原 V，与完整读出比较。它等价于在同一 Q/K/V 上执行子集主注意力；索引概率不直接加权 V。每次比较对应一个查询位置和一个 KV 组，将本组两个头的 128 维读出合成 256 维后计算 RMSE、相对 L2；再在资料内平均查询/组，最后平均资料。比较止于输出投影之前，不把差异继续送进模型其余层。

## 留出：16 份完整资料

{table(summary, 'heldout')}

每份先对查询/组求均值，再对资料求均值。m_C<1e-4 的查询/组不用于训练；汇总时 KL、条件覆盖率和熵排除这些行，但候选总权重、遗漏完整权重及输出 RMSE/相对 L2 仍计算并保留。跳过不是把整行指标填成零。C 的 KL 是均匀全候选分布的 KL，coverage 来自实际随机名单；R 只有选位诊断。误差在注意力输出 o_proj 前计算，不能替代任务分数。

相对 D0，留出资料中 KL 下降 {sum(r['delta_kl'] < 0 for r in paired)}/16；coverage 上升 {sum(r['delta_coverage'] > 0 for r in paired)}/16；相对输出误差下降 {sum(r['delta_output_error'] < 0 for r in paired)}/16。完整逐资料差异见 JSON，不把这些相关行当独立验证样本。

完整权重口径下，固定集合平均已占 {1-summary['aggregate']['heldout']['D']['candidate_mass']:.5f}；D 的平均遗漏完整权重为 {summary['aggregate']['heldout']['D']['missed_full_mass']:.5f}。条件 coverage 不等于全部注意力覆盖率，更不是答案准确率。m_C×(1−coverage) 必须逐行计算再平均，不能用两个均值相乘代替。

## 训练：48 份完整资料

{table(summary, 'train')}

训练曲线记录每轮更新过程中遇到的损失，不是固定轮末参数的整集损失；上表才是同一最终参数重评的结果。固定第 50 轮，无留出早停或挑选。低候选质量跳过训练行数：{summary['skipped_train_rows']}。

## 名额、来源与正确性

- ρ=1/8；资料 token 范围 {summary['prefix_tokens_range']}；k0/实际 k/总读取数分别为 {summary['range']['k0']} / {summary['range']['k']} / {summary['range']['selected_count']}。候选数 {summary['range']['candidate_count']}，合法长度 {summary['range']['legal']}；k/M 与总读取比例按原始逐查询记录分别计算，不把 ρ 当作总比例。
- 特征来自完整 A-step 路径；主体 bf16/MPS；教师与局部输出用保存特征在 CPU float32 重算；索引器 MPS float32。只抓取一个层、每资料四个问题/模板后缀位置，不保存全层注意力。
- 主体提取前后 hash：`{m['body_hash_before']}` / `{m['body_hash_after_extraction']}`；主体在索引训练前释放；优化器仅含 Wq/Wk。
- 原始逐查询/组数据共 {summary['rows']} 条方法记录；64 份资料与旧归档 60 个唯一资料 hash 重合为 0。新种子不使这 16 份成为更大总体的保证。
- 两小时预算含失败预检、通过预检及正式运行，累计 {summary['budget']['used_seconds']:.1f} 秒；现金 0。第一次预检在布局振荡处停止，未训练，证据保留。G3a 单独采用 2048±64 token，未改旧生成器或旧 run。

## 案例与解释边界

案例按留出最终查询、8组平均 coverage 的 D-D0 最大/最小变化选取；展示组固定为0，极值不是典型样本。具体题目、目标记录、原位置名单与指标见 JSON 和 Notebook。目标位置覆盖、教师注意力覆盖、局部输出误差与答题正确性不是同一指标；R 覆盖最优也不保证输出误差最小。

本轮留出的 512 个查询/组组合（16份资料×4个查询位置×8个KV组）中，D 相对 D0 的 coverage 退化 {summary['query_group_regressions_vs_D0']['coverage']} 行，相对输出误差退化 {summary['query_group_regressions_vs_D0']['output_relative_l2']} 行；没有生成答案，因此不能制造“答题失败案例”。保留残余误差最大的现有行：`{residual['doc_id']}`、查询 {residual['query_index']}、KV 组 {g}，D 的条件 coverage {residual_d['coverage'][g]:.5f}、m_C {residual_d['candidate_mass'][g]:.5f}、相对输出误差 {residual_d['output_relative_l2'][g]:.5f}。这是事后解释用例，未用于调参。8 组完整汇总在 JSON 的 heldout_by_group 中。

留出资料仍来自同一生成器、提示和填充句库；学到稳定的读取模式与学会寻找答案不能等同。改善不保证迁移到别的模板、长度、层或自由生成轨迹。

运行前预测保留在 `docs/tasks/06_G3A_SINGLE_LAYER.md`（首次提交 ee7e2f1）。全层稀疏轨迹、任务分数、训练种子稳定性、T2、速度与从资料开始稀疏均未检查。这描述 G3a 结束时的范围；后来另行授权的各层训练、真实推理与新题评测见第五至七课，不倒算进本阶段结果。

## 复现与展示

正式 run：`{summary['run_id']}`；运行代码 `{summary['run_commit']}`；报告生成源码 SHA-256 `{summary['report_source_sha256']}`。配置与原始数据在证据归档中，源自新的 48/16 T1 划分。

`python scripts/g3a_report.py` 只读归档重建本报告/JSON，不导入 torch。第四课 Notebook 不加载真实模型，但会执行 12 参数、20 步的 CPU 教学更新；它不启动正式索引训练。完整特征 NPZ 留在本机原始 run 的 features/（逐文件 hash 在 manifest），精简归档包含全部逐查询指标/名单位图的原位置列表、数据、初始/最终参数、运行源码、失败记录和预算账。

HTML 与图像的实际展示检查范围见 PROJECT_STATUS.md；执行无错误不等于整页已视觉验收。
'''


def plot_overview(summary):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    trace = summary["training"]
    axes[0].plot([x["epoch"] for x in trace], [x["online_train_kl"] for x in trace], color="#0072b2")
    axes[0].set(xlabel="Epoch (fixed 50)", ylabel="Online train KL (nats)", title="Training updates; no heldout tuning")
    colors = ["#999999", "#e69f00", "#0072b2", "#009e73"]
    for ax, key, title in zip(axes[1:], ("coverage", "output_relative_l2"), ("Heldout teacher coverage (higher)", "Heldout output relative L2 (lower)")):
        ax.bar(METHODS, [summary["aggregate"]["heldout"][m][key] for m in METHODS], color=colors)
        for i, method in enumerate(METHODS):
            values = [r[key] for r in summary["by_document"] if r["split"] == "heldout" and r["method"] == method]
            ax.plot([i] * len(values), values, "k.", alpha=.5, markersize=4)
        ax.set(title=title, ylabel="Document mean (dots: 16 documents)", ylim=(0, None))
    return fig


def plot_case(case):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 3.3), layout="constrained")
    colors = {"C": "#999999", "D0": "#e69f00", "D": "#0072b2", "R": "#009e73"}
    for i, row in enumerate(case["records"]):
        positions = row["positions"][case["group"]]
        ax.plot(positions, [i] * len(positions), "|", markersize=9, label=row["method"], color=colors[row["method"]])
    for j, (start, end) in enumerate(case["target_spans"]):
        ax.axvspan(start, end, color="#cc79a7", alpha=.35, label="Target record" if j == 0 else None)
    ax.set(yticks=range(4), yticklabels=[r["method"] for r in case["records"]], xlabel="Original token position (no renumbering)",
           title=f'{case["doc_id"]}: final query; layer 14, KV group 0', ylim=(-.6, 3.7))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-run", type=Path)
    args = parser.parse_args()
    if args.package_run:
        package(args.package_run)
    evidence = read_evidence()
    result = summarize(evidence)
    dump(ROOT / "reports/g3a_summary_mac_m4max.json", result)
    (ROOT / "reports/g3a_mac_m4max.md").write_text(markdown(result, evidence))
    print(json.dumps({"status": result["status"], "heldout": result["aggregate"]["heldout"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
