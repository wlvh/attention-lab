#!/usr/bin/env python3
"""Aggregate the G2-lite runs into reports/: baseline usability, four-method quality tables, paired 2x2 against A,
coverage, representative cases with their real row lists, counterfactual rewrites and recorded costs.

Inputs are run directories (quality.jsonl, data.jsonl, summary.json, manifest.json, positions/*.npz when present):
  --baseline    the first pass (A on every cell)                     default: earliest non-quick run whose rows are all A
  --comparison  the second pass (A + B/C/D0 on the usable cells)     default: latest non-quick run with B/C/D0 base rows
  --rewrites    third-pass runs (value / irrelevant / move variants)  default: every non-quick run with non-base rows
Outputs: reports/g2lite_mac_m4max.md, reports/g2lite_summary_mac_m4max.json (tables + cases), reports/g2lite_answers_mac_m4max.json
(every answer with its parse result) and reports/g2lite_manifest_mac_m4max.json. Numbers come only from the runs; no torch.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The scorer is pure Python, but importing its package would eagerly import torch.
# Keep this report-only entry local and explicit; don't change package initialization.
_spec = importlib.util.spec_from_file_location("g2lite_report_eval", ROOT / "src/attention_lab/g2lite_eval.py")
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)
cell_key, method_key = evaluation.cell_key, evaluation.method_key

METHOD_LABEL = {"A": "A 完整读取", "B": "B 近期窗口", "C": "C 均匀随机", "D0": "D0 未训练索引"}
BUCKET_LABEL = {"front": "前段", "middle": "中段", "back_near": "后段·窗口内", "back_far": "后段·远窗口"}


def load_run(d: Path) -> dict:
    d = d if d.is_dir() else d.parent
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "done" or summary.get("status", "done") != "done":
        raise ValueError(f"incomplete run cannot be reported as complete evidence: {d.name}")
    completion = manifest.get("completion")
    if completion and (completion["completed_requests"] != completion["planned_requests"]
                       or completion["skipped_requests"] or completion["pending_requests"]):
        raise ValueError(f"incomplete request accounting: {d.name}")
    rows = [json.loads(l) for l in (d / "quality.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [json.loads(l) for l in (d / "data.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (d / "data.jsonl").exists() else []   # smoke / calibration archives carry no data.jsonl
    return {"dir": d, "rows": rows, "docs": {x["doc_id"]: x for x in docs}, "summary": summary,
            "manifest": manifest, "config": json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else None}


def rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def value_range(values, *, percent=False):
    if not values:
        return "—"
    lo, hi = min(values), max(values)
    fmt = (lambda x: f"{100*x:.2f}%") if percent else str
    return fmt(lo) if lo == hi else f"{fmt(lo)}–{fmt(hi)}"


def render_rewrite_observations(observations):
    """A shared calculated table for the report, review card and teaching notebook."""
    lines = ["| 改写 | 方法 | 配对请求 | 回答不变 | 回答改变 | 正误改变 | 两次都错 | 改后答编号 | 从非编号变为编号 | C 随机条件不同 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant, methods in observations.items():
        for method, c in methods.items():
            keys = ("n", "answer_unchanged", "answer_changed", "correctness_changed", "both_wrong", "rewrite_answered_with_ids", "became_id_echo", "confounded_uniform_conditions")
            lines.append(f"| {variant} | {method} | " + " | ".join(str(c[k]) for k in keys) + " |")
    value = observations.get("value", {})
    sparse = [value[m] for m in ("B", "C", "D0") if m in value]
    n = sum(c["n"] for c in sparse)
    lines.append(f"\nvalue 的 B/C/D0 合计：{sum(c['answer_unchanged'] for c in sparse)}/{n} 个解析后回答不变，"
                 f"{sum(c['answer_changed'] for c in sparse)}/{n} 个改变；{sum(c['both_wrong'] for c in sparse)}/{n} 个两次都错，"
                 f"改后 {sum(c['rewrite_answered_with_ids'] for c in sparse)}/{n} 个答成编号。四个量不能互相替代。")
    lines.append("\n旧 C 改写的随机种子随资料 ID 后缀改变：这些记录是真实观察，但不是固定随机读取条件下的单变量证据。"
                 "新代码控制父资料的外部随机性，没有重跑旧 C，也没有倒填旧 manifest。D0 仍按改变后的输入重新打分，不强制固定名单。")
    return "\n".join(lines)


def report_build(runs, comp, details):
    """Separate generating-code identity from immutable model-run manifests."""
    sources = ("scripts/g2lite_report.py", "src/attention_lab/g2lite_eval.py")
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = [p for p in sources if hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{p}"], cwd=ROOT)).hexdigest() != hashes[p]]
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    inputs = {}
    for run in runs:
        rid = run["manifest"]["run_id"]
        for name in ("data.jsonl", "quality.jsonl", "summary.json", "manifest.json", "config.json"):
            p = run["dir"] / name
            if p.exists():
                inputs[f"{rid}/{name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    if comp:
        for doc_id, detail in details.items():
            for mk in detail["answers"]:
                p = comp["dir"] / "positions" / f"{doc_id}__{mk.replace('@', '_rho')}.npz"
                if p.exists():
                    inputs[f"{comp['manifest']['run_id']}/positions/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"schema_version": 1, "kind": "report_regeneration_not_model_run", "source_commit": commit,
            "report_source_hashes": hashes, "uncommitted_report_sources": dirty, "input_files_sha256": dict(sorted(inputs.items())),
            "model_run_commits": sorted({r["manifest"]["git"]["commit"] for r in runs}),
            "model_run_config_versions": sorted({r["manifest"]["config_version"] for r in runs}),
            "new_model_runs": False, "dependencies": ["Python standard library", "NumPy (position bitsets)"],
            "historical_C_rewrites": "confounded: variant-specific seeds; retained without rerunning",
            "early_calibration_gap": "v1 calib-w2/w4/w5 generator hashes differ from both 4bccbb5 and af62df4; exact source not recovered"}


def runs_of_kind(all_runs):
    base = [r for r in all_runs if r["rows"] and all(x["method"] == "A" and x.get("variant", "base") == "base" for x in r["rows"])]
    comp = [r for r in all_runs if any(x["method"] != "A" and x.get("variant", "base") == "base" for x in r["rows"])]
    rew = [r for r in all_runs if any(x.get("variant", "base") != "base" for x in r["rows"])]
    return base, comp, rew


def runs_of_positions(run, doc_id, mk):
    p = run["dir"] / "positions" / f"{doc_id}__{mk.replace('@', '_rho')}.npz"
    if not p.exists():
        return None
    import numpy as np
    z = np.load(p)
    bits, cap = z["bits"], int(z["capacity"])
    sel = np.unpackbits(bits, axis=-1)[..., :cap].astype(bool)          # [L, G, C]
    spans = z["target_spans"].tolist()
    L, G, _ = sel.shape
    tgt = np.zeros(cap, dtype=bool)
    for s, e in spans:
        tgt[s:e] = True
    cover = sel[:, :, tgt].mean(-1)                                       # [L, G] fraction of target positions read
    full = (sel[:, :, tgt].all(-1)).mean()
    out = {"layers": L, "groups": G, "capacity": cap, "prefix_len": int(z["prefix_len"]), "legal_first_gen_step": int(z["legal_first_gen_step"]),
           "target_spans": spans, "coverage_first_gen_step_layer_group": cover.round(3).tolist(), "fraction_layer_groups_full_coverage": float(full),
           "example_rows": {}}
    for l in sorted({0, L // 2, L - 1}):
        pos = np.flatnonzero(sel[l, 0])
        runs = []
        if pos.size:
            start = prev = int(pos[0])
            for p_ in pos[1:]:
                p_ = int(p_)
                if p_ != prev + 1:
                    runs.append([start, prev + 1])
                    start = p_
                prev = p_
            runs.append([start, prev + 1])
        out["example_rows"][f"layer{l}_group0"] = {"n_rows": int(pos.size), "ranges": runs[:80], "n_ranges": len(runs)}
    return out


def pick_cases(comp, usable, rhos):
    """Pre-registered: per usable cell, up to two lowest-index documents per category (docs/tasks/05_G2_LITE.md)."""
    rows = [r for r in comp["rows"] if r.get("variant", "base") == "base"]
    cases = []
    for c in sorted(usable):
        in_cell = [r for r in rows if cell_key(r["task"], r["length_tag"]) == c]
        by = {}
        for r in in_cell:
            by.setdefault(r["doc_id"], {})[method_key(r["method"], r["rho"])] = r
        docs = sorted(by, key=lambda d: by[d]["A"]["doc_index"] if "A" in by[d] else 99)
        cats = {"A_right_B_wrong": [], "A_right_C_or_D0_wrong": [], "all_four_right": []}
        for d in docs:
            m = by[d]
            if "A" not in m or not m["A"]["exact_match"]:
                continue
            for rho in rhos:
                b, cc, dd = m.get(method_key("B", rho)), m.get(method_key("C", rho)), m.get(method_key("D0", rho))
                if b and cc and dd:
                    if not b["exact_match"] and len(cats["A_right_B_wrong"]) < 2 and all(x["doc_id"] != d for x in cats["A_right_B_wrong"]):
                        cats["A_right_B_wrong"].append({"doc_id": d, "rho": rho})
                    if (not cc["exact_match"] or not dd["exact_match"]) and len(cats["A_right_C_or_D0_wrong"]) < 2 and all(x["doc_id"] != d for x in cats["A_right_C_or_D0_wrong"]):
                        cats["A_right_C_or_D0_wrong"].append({"doc_id": d, "rho": rho})
                    if b["exact_match"] and cc["exact_match"] and dd["exact_match"] and len(cats["all_four_right"]) < 2 and all(x["doc_id"] != d for x in cats["all_four_right"]):
                        cats["all_four_right"].append({"doc_id": d, "rho": rho})
        for cat, lst in cats.items():
            for item in lst:
                cases.append({"cell": c, "category": cat, **item})
        # added after the results were seen (not pre-registered; stated in the report): the two contrasts the first run showed
        post = {"B_right_C_and_D0_wrong": [], "B_wrong_target_in_window": []}
        for d in docs:
            m = by[d]
            if "A" not in m or not m["A"]["exact_match"]:
                continue
            for rho in rhos:
                b, cc, dd = m.get(method_key("B", rho)), m.get(method_key("C", rho)), m.get(method_key("D0", rho))
                if not (b and cc and dd):
                    continue
                if b["exact_match"] and not cc["exact_match"] and not dd["exact_match"] and len(post["B_right_C_and_D0_wrong"]) < 2 and all(x["doc_id"] != d for x in post["B_right_C_and_D0_wrong"]):
                    post["B_right_C_and_D0_wrong"].append({"doc_id": d, "rho": rho})
                if not b["exact_match"] and b["coverage"].get("first_gen_step_mean", 0) >= 0.999 and len(post["B_wrong_target_in_window"]) < 2 and all(x["doc_id"] != d for x in post["B_wrong_target_in_window"]):
                    post["B_wrong_target_in_window"].append({"doc_id": d, "rho": rho})
        for cat, lst in post.items():
            for item in lst:
                cases.append({"cell": c, "category": cat, "post_hoc": True, **item})
    return cases


def case_detail(comp, rew_runs, doc_id, rhos):
    doc = comp["docs"][doc_id]
    rows = [r for r in comp["rows"] if r["doc_id"] == doc_id]
    answers = {}
    for r in sorted(rows, key=lambda r: (r["method"] != "A", r["method"], r["rho"] or 0)):
        mk = method_key(r["method"], r["rho"])
        answers[mk] = {"predicted": r["predicted"], "text": r["text"], "exact": r["exact_match"], "answered_with_ids": r.get("answered_with_ids"), "k0": r["k0"], "selected_count": r["selected_count"],
                       "coverage": r["coverage"], "positions": runs_of_positions(comp, doc_id, mk) if r["method"] != "A" else None}
    rewrites = []
    for rr in rew_runs:
        for r in rr["rows"]:
            if r.get("parent_doc_id") == doc_id:
                rewrites.append({"variant": r["variant"], "method": r["method"], "rho": r["rho"], "predicted": r["predicted"], "exact": r["exact_match"], "targets": r["targets"], "note": r.get("note"),
                                 "doc_id": r["doc_id"], "run_id": rr["summary"]["run_id"]})
    return {"doc": {k: doc.get(k) for k in ("doc_id", "task", "length_tag", "buckets", "n_records", "prefix_token_count", "suffix_token_count", "question", "targets", "target_ids", "target_indices",
                                           "target_records_from_end", "target_distance_from_prefix_end", "target_spans")},
            "records_around_targets": {str(j): [doc["ids"][j], doc["values"][j]] for i in doc["target_indices"] for j in range(max(0, i - 2), min(doc["n_records"], i + 3))},
            "last_records": [[doc["ids"][j], doc["values"][j]] for j in range(max(0, doc["n_records"] - 6), doc["n_records"])],
            "answers": answers, "rewrites": rewrites}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--comparison", type=Path, default=None)
    ap.add_argument("--rewrites", type=Path, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/g2lite_mac_m4max.md")
    ap.add_argument("--json", type=Path, default=ROOT / "reports/g2lite_summary_mac_m4max.json")
    ap.add_argument("--answers", type=Path, default=ROOT / "reports/g2lite_answers_mac_m4max.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "reports/g2lite_manifest_mac_m4max.json")
    ap.add_argument("--build-manifest", type=Path, default=ROOT / "reports/g2lite_report_build.json", help="new report-code/input identity; never changes the old run manifests")
    ap.add_argument("--review-card", type=Path, default=None, help="refresh only the marked calculated review block, preserving registered predictions")
    ap.add_argument("--card", default="reports/cards/G2lite_mac_m4max.md")
    ap.add_argument("--include-quick", action="store_true")
    a = ap.parse_args()
    all_runs = [load_run(Path(p)) for p in sorted(glob.glob(str(a.runs_dir / "g2lite-*/summary.json")))]
    all_runs = [r for r in all_runs if (a.include_quick or not r["summary"].get("quick")) and not r["summary"].get("calibration")]
    base_runs, comp_runs, rew_runs = runs_of_kind(all_runs)
    base = load_run(a.baseline) if a.baseline else (base_runs[0] if base_runs else None)
    comp = load_run(a.comparison) if a.comparison else (comp_runs[-1] if comp_runs else None)
    rews = [load_run(p) for p in a.rewrites] if a.rewrites is not None else rew_runs
    if base is None:
        print("no baseline run found")
        return 1
    cfg = base["config"] or comp["config"]
    rhos = sorted({r["rho"] for r in (comp["rows"] if comp else []) if r["rho"] is not None}) or cfg["rho"]
    gate = base["summary"]["aggregate"]["gate"]
    usable = sorted(c for c, g in gate.items() if g["usable"])
    agg_comp = comp["summary"]["aggregate"] if comp else None
    cases = pick_cases(comp, usable, rhos) if comp else []
    details = {c["doc_id"]: case_detail(comp, rews, c["doc_id"], rhos) for c in cases} if comp else {}
    rew_rows = [r for rr in rews for r in rr["rows"]]
    comp_rows = comp["rows"] if comp else []
    rew_table = evaluation.rewrite_table(comp_rows + rew_rows) if rews else []
    observations = evaluation.rewrite_observations(rew_table)
    budgets = evaluation.budget_tables(comp_rows)
    build = report_build([base] + ([comp] if comp else []) + rews, comp, details)
    # ---- condensed JSON ----
    cond = {"baseline_run": base["summary"]["run_id"], "comparison_run": comp["summary"]["run_id"] if comp else None, "rewrite_runs": [r["summary"]["run_id"] for r in rews],
            "config_version": cfg["config_version"], "rho": rhos, "gate": gate, "usable_cells": usable,
            "baseline_methods": base["summary"]["aggregate"]["methods"], "comparison_methods": agg_comp["methods"] if agg_comp else None,
            "paired": agg_comp["paired"] if agg_comp else None, "a_consistency_vs_baseline": agg_comp.get("a_consistency_vs_prev") if agg_comp else None,
            "rewrites": rew_table, "rewrite_observations": observations, "read_budgets": budgets,
            "move_cases": [r["summary"].get("move_cases") for r in rews], "cases": cases, "case_details": details,
            "timing": {"baseline": base["summary"]["aggregate"]["timing"], "comparison": agg_comp["timing"] if agg_comp else None, "rewrites": [r["summary"]["aggregate"]["timing"] for r in rews]},
            "device_seconds": {"baseline": base["summary"]["device_seconds"], "comparison": comp["summary"]["device_seconds"] if comp else None, "rewrites": [r["summary"]["device_seconds"] for r in rews],
                               "total": base["summary"]["device_seconds"] + (comp["summary"]["device_seconds"] if comp else 0) + sum(r["summary"]["device_seconds"] for r in rews)},
            "docs": {d["doc_id"]: {k: d.get(k) for k in ("task", "length_tag", "variant", "parent_doc_id", "buckets", "n_records", "prefix_token_count", "suffix_token_count", "targets", "target_ids",
                                                            "target_records_from_end", "target_distance_from_prefix_end", "question", "changed_record_index", "note")}
                     for run in [base] + ([comp] if comp else []) + rews for d in run["docs"].values()}}
    a.json.write_text(json.dumps(cond, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    skip = {"coverage"}
    answers = [{**{k: v for k, v in r.items() if k not in skip}, "coverage_first_gen_step_mean": r["coverage"].get("first_gen_step_mean"), "coverage_question_steps": r["coverage"].get("mean_fraction_question_steps"),
                "coverage_generation_steps": r["coverage"].get("mean_fraction_generation_steps"), "coverage_first_gen_step_per_layer": r["coverage"].get("first_gen_step_per_layer")}
               for run in [base] + ([comp] if comp else []) + rews for r in run["rows"]]
    a.answers.write_text(json.dumps(answers, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    manifest_text = json.dumps({"runs": [r["manifest"] for r in [base] + ([comp] if comp else []) + rews]}, ensure_ascii=False, indent=1) + "\n"
    if not a.manifest.exists() or a.manifest.read_text(encoding="utf-8") != manifest_text:
        a.manifest.write_text(manifest_text, encoding="utf-8")
    a.build_manifest.write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if a.review_card:
        card = a.review_card.read_text(encoding="utf-8")
        start, end = "<!-- review-counts:start -->", "<!-- review-counts:end -->"
        if card.count(start) != 1 or card.count(end) != 1:
            raise ValueError("review card must contain one start/end pair for calculated counts")
        before, tail = card.split(start)
        _, after = tail.split(end)
        a.review_card.write_text(before + start + "\n" + render_rewrite_observations(observations) + "\n" + end + after, encoding="utf-8")

    # ---- markdown ----
    L = []
    m0, s0 = base["manifest"], base["summary"]
    H, S, info = s0["host"], s0["software"], s0["model_info"]
    L.append("# G2-lite 实测报告 · Mac M4 Max：少看历史会丢掉什么——四种读取方式在 T1 / T2 上的质量对照\n")
    L.append('本报告让冻结的 Qwen3-0.6B 读资料、回答记录值，项目编号为 G2-lite。T1 按编号返回一个四位值，T2 按顺序返回两个；例如资料中的 `1563 → 8422`，问 1563 时应回答 8422。模型先完整处理不含问题的资料，随后各方法取得前缀缓存副本；从问题开始，各自读取并写入自己的 K/V，生成时将自己选出的 token 再送回。\n\nA 读全部历史。B/C/D0 都保留首 4 个和最近 64 个位置，这部分合称 F；再从剩余位置取相同数量：B 取最近的，C 均匀随机取，D0 用未训练参数按内容打分。C 随机的是位置，D0 随机的是初始化参数。\n\n这里的名额和读取数指本步某一层、某一 KV 组；同组查询头共用名单。不同层、不同组的额外名单可以不同，不能把这个数当成整个模型所有名单的去重总数。名额在第一个问题 token 写入后确定。例如 T1-2K-00 的前缀长 2,050，起始合法长度为 2,051；ρ=1/8 时额外名额 round(2051/8)=256，以后不变。首个答案预测处已有 2,085 个位置，必留 68 个加额外 256 个，共读 324 个。故 ρ 用于分配额外名额，不是总读取比例；候选数 M 是本步去掉 F 后剩下的位置数。\n\n以 T1-2K-00 为例，问句有 26 个 token，问句后的聊天模板另有 9 个，合计 35 个。模板包含用户结束标记、assistant 开头和空 thinking 段；首个答案由整个后缀末尾的换行 token 预测，不是问句句号处。第三课展示了实际字符串。\n\n48 份探索资料分成 T1/T2×2K/8K 四组，各 12 份，先全部运行 A。只在 A 达到预定门槛的 T1 两组上，复用同一批 24 份资料比较两档名额的 B/C/D0；不是每档重新取得一批独立资料。旧 C 的资料改写还同时改变了随机名单，仅保留混杂观察。过程和真实案例见[第三课](03_what_is_lost_when_reading_less.html)。\n')
    L.append("<details><summary>运行版本与报告来源</summary>\n")
    L.append(f"由 `scripts/g2lite_report.py` 从基线 run `{s0['run_id']}`、对照 run `{comp['summary']['run_id'] if comp else '无'}` 与 {len(rews)} 个改写 run（{', '.join(r['summary']['run_id'] for r in rews) or '无'}）生成。"
             f"**48 份探索资料，不做区间统计；D0 是未训练的随机索引器；B 的名单按窗口几何确定，正确性另测；MPS 结果不推 CUDA。** 预测见 `{a.card}`。\n")
    L.append(f"**2026-10-18 审查修订：仅由旧记录重生成，没有新增模型运行。** 旧运行配置 v{cfg['config_version']}、运行源码 `af62df4` 与今日报告代码分开标识：生成提交 `{build['source_commit']}`，"
             "文件 SHA-256 与输入文件清单见 `reports/g2lite_report_build.json`。旧 run manifest 保持原样；当前种子修复没有追溯性地修复旧 C 改写。\n")
    L.append("</details>\n")
    L.append("## 1. 环境、模型、数据与协议\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append(f"| 机器 | {H.get('chip')}，{H.get('gpu_cores')} 核 GPU，{(H.get('physical_memory_bytes') or 0)/2**30:.0f} GiB，macOS {H.get('macos')} |")
    L.append(f"| torch / transformers / Python | {S.get('torch')} / {S.get('transformers')} / {H.get('python')}；MPS；fallback 与内存保护环境变量未设置 |")
    L.append(f"| 模型 | {info.get('model_id')} @ `{info.get('resolved_model_commit')}`（权重与 tokenizer 快照同一 commit：{info.get('resolved_weights_commit') == info.get('resolved_model_commit') == info.get('resolved_tokenizer_commit')}），{m0['dtype']}，attn `{info.get('attn_implementation')}`；权重 sha256 `{(m0.get('weights') or {}).get('sha256')}` |")
    L.append(f"| 代码 | 基线 run commit `{m0['git'].get('commit')}`，dirty diff sha256 {m0['git'].get('dirty_diff_sha256')}；对照 run commit `{comp['manifest']['git'].get('commit') if comp else '—'}`；配置 v{cfg['config_version']} sha256 `{m0['config_sha256'][:16]}…`；chat template sha256 `{m0['chat_template_sha256'][:16]}…`；HF_HUB_OFFLINE={m0.get('hf_hub_offline')} |")
    bd = [d for d in base["docs"].values() if d["variant"] == "base"]
    geo = "；".join(f"{lt} 前缀 {min(d['prefix_token_count'] for d in bd if d['length_tag'] == lt)}–{max(d['prefix_token_count'] for d in bd if d['length_tag'] == lt)} token" for lt in sorted({d["length_tag"] for d in bd}))
    suf = "；".join(f"{t} 问题+模板尾 {min(d['suffix_token_count'] for d in bd if d['task'] == t)}–{max(d['suffix_token_count'] for d in bd if d['task'] == t)} token" for t in sorted({d["task"] for d in bd}))
    L.append(f"| 数据 | 生成器 v{cfg['data']['generator_version']}，种子 {cfg['data']['seed']}，digest `{m0['data_sha256'][:16]}…`；每份 {bd[0]['n_records']} 条记录穿插在无数字填充句之间，记录行 {bd[0].get('tokens_per_record', 12)} token；{geo}；{suf}；尾部两条记录距前缀末尾 {cfg['data']['layout']['back_near_tokens_from_end']} / {cfg['data']['layout']['back_far_tokens_from_end']} token |")
    L.append(f"| 协议 | `{cfg['protocol']['name']}`：前缀完整读取一次并建索引表，问题不进前缀；每个请求克隆前缀缓存，从问题第一个 token 起逐 token 用自己的读取方式；贪心，≤ {cfg['generation']['max_new_tokens']} 新 token，EOS 停止；解析：{cfg['parsing']['T1']}；{cfg['parsing']['T2']} |")
    L.append(f"| 名额 | ρ ∈ {cfg['rho']}；k0 = round(ρ·N0)，N0 = 前缀 + 1；B、C、D0 同 F（首 4 ∪ 近 64）同名额 68 + k0 |")
    L.append("")
    # ---- baseline ----
    L.append("## 2. 基线可用性（第一遍：A 在全部格子）\n")
    L.append(f"门槛（预设）：每格完整匹配率 ≥ {pct(cfg['gate']['baseline_usable_min_exact'])}，小样本可用性检查，不是质量结论。\n")
    L.append("| 格子 | n | 完整匹配 | 率 | 答成编号 | 任意顺序含全部目标 | 格式错误 | 截断 | 按目标位置（前/中/后近/后远 或 成对） | 可用 |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for c, tbl in s0["aggregate"]["methods"].items():
        t = tbl["A"]
        rows_c = [r for r in base["rows"] if cell_key(r["task"], r["length_tag"]) == c]
        ids = sum(1 for r in rows_c if r.get("answered_with_ids"))
        bb = "；".join(f"{'/'.join(BUCKET_LABEL.get(x, x) for x in b.split('/'))} {pct(v)}" for b, v in t["by_bucket"].items())
        L.append(f"| {c} | {t['n']} | {t['hits']} | {pct(t['rate'])} | {ids} | {pct(t['any_order'])} | {t['format_errors']} | {t['truncated']} | {bb} | {'✓' if gate[c]['usable'] else '✗'} |")
    L.append("")
    # ---- comparison ----
    if comp:
        s1 = comp["summary"]
        L.append(f"## 3. 四方法质量对照（第二遍，只在可用格子：{', '.join(usable) or '无'}）\n")
        ac = s1["aggregate"].get("a_consistency_vs_prev") or {}
        L.append(f"A 在第二遍重跑：{ac.get('n_compared')} 份中 {ac.get('n_identical_generated_ids')} 份生成的 token 序列与第一遍逐位相同" + (f"（不同：{ac.get('differing_doc_ids')}）" if ac.get("differing_doc_ids") else "") + "。\n")
        L.append("| 格子 | 方法 | k0 | 读取数 | n | 完整匹配 | 率 | 答成编号 | 任意顺序 | 格式错误 | 按目标位置 | 对 A：同对 / A 对它错 / A 错它对 / 同错 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for c, tbl in s1["aggregate"]["methods"].items():
            for mk, t in tbl.items():
                rows_c = [r for r in comp["rows"] if cell_key(r["task"], r["length_tag"]) == c and method_key(r["method"], r["rho"]) == mk and r.get("variant", "base") == "base"]
                ids = sum(1 for r in rows_c if r.get("answered_with_ids"))
                pr = (s1["aggregate"]["paired"].get(c) or {}).get(mk)
                p2 = f"{pr['counts']['both_right']} / {pr['counts']['A_right_other_wrong']} / {pr['counts']['A_wrong_other_right']} / {pr['counts']['both_wrong']}" if pr else "—"
                bb = "；".join(f"{'/'.join(BUCKET_LABEL.get(x, x) for x in b.split('/'))} {pct(v)}" for b, v in t["by_bucket"].items())
                L.append(f"| {c} | {METHOD_LABEL[mk.split('@')[0]]}{' ρ=' + mk.split('@')[1] if '@' in mk else ''} | {value_range(t['k0'])} | {value_range(t['selected']) if t['selected'] else '全部'} | {t['n']} | {t['hits']} | {pct(t['rate'])} | {ids} | {pct(t['any_order'])} | {t['format_errors']} | {bb} | {p2} |")
        L.append("")
        # coverage
        L.append("### 3.1 目标片段覆盖率（每步每层每组读到的被查询记录 token 占比；结构性数字，不是学习结果）\n")
        L.append("| 长度 | 方法 | k / M（期望） | 问题阶段均值 | 生成阶段均值 | 第一个生成 token 那一步：均值 / 最低层 / 最高层 |\n|---|---|---:|---:|---:|---|")
        for lt in sorted({r["length_tag"] for r in comp["rows"]}):
            for mk in sorted({method_key(r["method"], r["rho"]) for r in comp["rows"] if r["method"] != "A"}):
                rs = [r for r in comp["rows"] if r["length_tag"] == lt and method_key(r["method"], r["rho"]) == mk and r.get("variant", "base") == "base"]
                if not rs:
                    continue
                budget = budgets[cell_key(rs[0]["task"], lt)][mk]
                cq = statistics.mean(r["coverage"]["mean_fraction_question_steps"] for r in rs)
                cg = statistics.mean(r["coverage"]["mean_fraction_generation_steps"] for r in rs if r["coverage"].get("mean_fraction_generation_steps") is not None)
                fm = statistics.mean(r["coverage"]["first_gen_step_mean"] for r in rs)
                fmin = statistics.mean(r["coverage"]["first_gen_step_min_layer"] for r in rs)
                fmax = statistics.mean(r["coverage"]["first_gen_step_max_layer"] for r in rs)
                L.append(f"| {lt} | {METHOD_LABEL[mk.split('@')[0]]} ρ={mk.split('@')[1]} | {100*budget['candidate_fraction_mean']:.2f}%（逐请求均值，首生成步） | {pct(cq)} | {pct(cg)} | {pct(fm)} / {pct(fmin)} / {pct(fmax)} |")
        L.append("\n本次 T1 的 B 首生成步目标覆盖率为 0 或 1；其他窗口可能切开一条记录，不能把二值性当成通用规则。k/M 是均匀 C 的结构参考，不是 B/D0 的覆盖率预测。\n")
        b_rows = [r for r in comp["rows"] if r["method"] == "B" and r.get("variant", "base") == "base"]
        inside = [r for r in b_rows if r["coverage"].get("first_gen_step_mean", 0) >= 0.999]
        outside = [r for r in b_rows if r["coverage"].get("first_gen_step_mean", 0) < 0.001]
        L.append(f"**B 的窗口内命中率。** 目标在窗口内的请求 {len(inside)} 个，答对 {sum(r['exact_match'] for r in inside)} 个" +
                 (f"（答错的：{', '.join(f'{r[chr(100)+chr(111)+chr(99)+chr(95)+chr(105)+chr(100)]} ρ={r[chr(114)+chr(104)+chr(111)]:g} 答 {r[chr(116)+chr(101)+chr(120)+chr(116)][:12]!r}' for r in inside if not r['exact_match'])}）" if any(not r["exact_match"] for r in inside) else "") +
                 f"；目标在窗口外的请求 {len(outside)} 个，答对 {sum(r['exact_match'] for r in outside)} 个，其中答成编号 {sum(1 for r in outside if r.get('answered_with_ids'))} 个。\n")
        L.append("### 3.2 实际读取预算（预测第一个答案 token 的步骤）\n")
        L.append('ρ 用请求起始长度 N0=P+1 决定 k0；首生成步的合法长度 N=P+Q，候选域 M=N−|F|。下表分别列 k/M 和 (|F|+k)/N，范围按逐请求计算。B/C/D0 同题同档相同。这些数均按一层、一个 KV 组计；Q 包括问句与聊天模板尾部，首生成步指整个后缀最后一个 token 的计算。\n')
        L.append("| 格子 | ρ | k0 范围 | 总读取数范围 | N 范围 | k/M 范围 | 总读取比例范围 |\n|---|---:|---|---|---|---|---|")
        for cell, table in budgets.items():
            for mk, b in table.items():
                if mk.startswith("B@"):
                    L.append(f"| {cell} | {b['rho']:g} | {value_range(b['k0'])} | {value_range(b['selected'])} | {value_range(b['legal'])} | {value_range(b['candidate_fraction'], percent=True)} | {value_range(b['total_read_fraction'], percent=True)} |")
        L.append('\n覆盖率只是选中目标位置的比例，不是注意力权重，也不是检索的必要或充分条件。目标记录“编号 1563：8422”在保存输入中占 11 个 token 位置，答案 8422 只占其中最后 4 个数字 token。一条文本记录对应多行 K/V，不能当作一行。整条记录覆盖与这四个数字位置的覆盖不同；即便某层组读齐答案数字 token 仍错，也只能说明读齐不足以保证答对，不能据此证明读齐不是必要条件。\n')
        # cases
        L.append("## 4. 代表案例（预登记规则：每个可用格子每类最多两份，按资料序号从小到大；标 † 的两类是看到结果后追加的对比，不在预登记内）\n")
        if not cases:
            L.append("没有可用格子，或没有满足类别条件的案例。\n")
        for cs in cases:
            d = details[cs["doc_id"]]
            doc = d["doc"]
            L.append(f"**{cs['category']}{' †' if cs.get('post_hoc') else ''} · {cs['doc_id']}**（{cs['cell']}，ρ={cs['rho']:g} 处入选，目标位置 {'/'.join(BUCKET_LABEL.get(x, x) for x in doc['buckets'])}，倒数第 {doc['target_records_from_end']} 条，距前缀末尾 {doc['target_distance_from_prefix_end']} token）。问题：{doc['question']}；目标值 {doc['targets']}。")
            L.append("")
            L.append("| 方法 | 读取数 | 回答 | 判定 | 目标片段覆盖（第一个生成 token 那一步：均值 / 层组全覆盖比例） |\n|---|---:|---|---|---|")
            for mk, an in d["answers"].items():
                cov = an["coverage"]
                covs = "全部" if mk == "A" else f"{pct(cov.get('first_gen_step_mean'))} / {pct(cov.get('fraction_of_layer_groups_with_full_coverage_first_gen_step'))}"
                L.append(f"| {METHOD_LABEL[mk.split('@')[0]]}{' ρ=' + mk.split('@')[1] if '@' in mk else ''} | {an['selected_count'] or '全部'} | `{an['text'][:60]}` | {'✓' if an['exact'] else '✗'}{'（答成编号）' if an.get('answered_with_ids') else ''} | {covs} |")
            if d["rewrites"]:
                L.append("")
                L.append("改写：" + "；".join(f"{r['variant']} → {METHOD_LABEL[r['method']]}{' ρ=' + format(r['rho'], 'g') if r['rho'] is not None else ''} 答 {r['predicted']}（目标 {r['targets']}，{'✓' if r['exact'] else '✗'}）" for r in d["rewrites"]))
            L.append("")
    # ---- rewrites ----
    if rew_table:
        L.append("## 5. 反事实改写（重建前缀，不复用 KV）\n")
        L.append(render_rewrite_observations(observations) + "\n")
        L.append("资料前缀先用完整注意力处理，后方位置的深层 K/V 可能受前方记录影响。因此没有直接选择目标位置不等于目标信息无法影响答案。现有例外否定了‘窗口外必不变’，尚未定位具体层或路径。\n")
        L.append("| 改写 | 资料 | 方法 | 原答 | 原判定 | 改后答 | 改后目标 | 改后判定 | 检查 |\n|---|---|---|---|---|---|---|---|---|")
        for r in rew_table:
            chk = {"value": f"跟着改：{r.get('answer_followed_change')}；黏在旧值：{r.get('answer_stuck_to_old_value')}", "irrelevant": f"不变：{r.get('answer_unchanged')}", "move": f"移近后变对：{r.get('became_right')}"}[r["variant"]]
            L.append(f"| {r['variant']} | {r['parent_doc_id']} | {METHOD_LABEL[r['method']]}{' ρ=' + format(r['rho'], 'g') if r['rho'] is not None else ''} | {r['base_predicted']} | {'✓' if r['base_exact'] else '✗'} | {r['rewrite_predicted']} | {r['rewrite_targets']} | {'✓' if r['rewrite_exact'] else '✗'} | {chk} |")
        L.append("")
        # summary counts
        for v in ("value", "irrelevant", "move"):
            rs = [r for r in rew_table if r["variant"] == v]
            if not rs:
                continue
            per = {}
            for r in rs:
                mk = method_key(r["method"], r["rho"])
                key = {"value": "answer_followed_change", "irrelevant": "answer_unchanged", "move": "became_right"}[v]
                per.setdefault(mk, [0, 0])
                per[mk][0] += int(bool(r.get(key)))
                per[mk][1] += 1
            L.append(f"- {v}：" + "；".join(f"{METHOD_LABEL[mk.split('@')[0]]}{' ρ=' + mk.split('@')[1] if '@' in mk else ''} {x}/{n}" for mk, (x, n) in per.items()))
        L.append("")
    # ---- cost ----
    L.append("## 6. 实际运行成本\n")
    L.append("| run | 设备时间 s | Prefill 中位 ms（2K / 8K） | 每步 ms 中位（按长度 / 方法） |\n|---|---:|---|---|")
    for run in [base] + ([comp] if comp else []) + rews:
        tm = run["summary"]["aggregate"]["timing"]
        pf = tm["prefill_ms_median_by_length"]
        L.append(f"| {run['summary']['run_id']} | {run['summary']['device_seconds']:.0f} | {f(pf.get('2K'), 0)} / {f(pf.get('8K'), 0)} | " + "；".join(f"{k} {v:.1f}" for k, v in tm["ms_per_step_median"].items()) + " |")
    L.append(f"\n合计设备时间 {cond['device_seconds']['total']:.0f} s（含模型加载），现金 0。每步耗时含覆盖率记账与逐步同步，不是速度基准。\n")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {rel(a.out)}, {rel(a.json)}, {rel(a.answers)}, {rel(a.manifest)}; usable cells {usable}; cases {len(cases)}; rewrites {len(rew_table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
