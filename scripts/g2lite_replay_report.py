#!/usr/bin/env python3
"""Pair a G2-lite replay run (B re-run plus pinned diagnostic variants) against the archived pass-2 run.

Header replay (reports/cards/G2lite_replay_header_mac_m4max.md): BH = recent + header, BM = recent minus |header|.
Filler replay (reports/cards/G2lite_replay_filler_mac_m4max.md): BF = recent + the same number of filler positions right
after the header, BH again for the pairing.
Header-split replay (reports/cards/G2lite_replay_header_split_mac_m4max.md): BG = recent + the header's format clause,
BA = recent + the header's first clause (different pinned counts, hence windows that differ from each other and from
BH's), B and BH paired again.
Format-split replay (reports/cards/G2lite_replay_format_split_mac_m4max.md): BG1 = recent + 「格式为「编号 X：Y」，」, BG2 =
recent + 「Y 是编号 X 对应的四位数值。」, B, BH and BG paired again. The variants present in the run decide the columns; the
paired in-window set is the set of requests whose target lies inside every variant's window. Inputs are run dirs:
  --run        the replay run (quality.jsonl, data.jsonl, summary.json, manifest.json, positions/*.npz)
               default: the latest non-quick runs/g2lite-* whose summary is marked diagnostic_replay
  --reference  the archived pass-2 run with A and the original B rows (default runs/g2lite-20261017T130319Z-pass2-ABCD0;
               extract it from reports/provenance/g2lite_runs_20261017.tar.gz in a clean checkout)
Outputs (defaults are the header replay's files; the filler replay passes its own --out/--json/--manifest/--title/--card).
Numbers come only from the runs; no torch, no model.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attention_lab.g2lite_eval import cell_key, method_key  # noqa: E402
from attention_lab.selection import FIXED_RECENT  # noqa: E402

LABEL = {"A": "A 完整读取", "B": "B 近期窗口", "BH": "BH 近期窗口+表头", "BM": "BM 近期窗口减名额（对照）", "BF": "BF 近期窗口+填充句",
         "BG": "BG 近期窗口+格式句", "BA": "BA 近期窗口+前半句", "BG1": "BG1 近期窗口+格式句前段", "BG2": "BG2 近期窗口+格式句后段"}
DESC = {"BH": "BH：F ∪ 表头 ∪ 最近 k0 − {n} 个候选（读取数与 B 相同）", "BM": "BM：F ∪ 最近 k0 − {n} 个候选（BH 的窗口、没有表头，读取数比 B 少 {n}）",
        "BF": "BF：F ∪ 表头之后的 {n} 个填充句位置 {span} ∪ 最近 k0 − {n} 个候选（读取数与 B 相同，窗口与 BH 相同）",
        "BG": "BG：F ∪ 表头格式句 {span} 的 {n} 个位置 ∪ 最近 k0 − {n} 个候选（读取数与 B 相同，窗口比 B 少 {n}）",
        "BA": "BA：F ∪ 表头前半句 {span} 的 {n} 个候选位置（位置 3 在 F 里）∪ 最近 k0 − {n} 个候选（读取数与 B 相同，窗口比 B 少 {n}）",
        "BG1": "BG1：F ∪ 格式句前段「格式为「编号 X：Y」，」{span} 的 {n} 个位置 ∪ 最近 k0 − {n} 个候选（读取数与 B 相同，窗口比 B 少 {n}）",
        "BG2": "BG2：F ∪ 格式句后段「Y 是编号 X 对应的四位数值。」{span} 的 {n} 个位置 ∪ 最近 k0 − {n} 个候选（读取数与 B 相同，窗口比 B 少 {n}）"}
VARIANT_ORDER = ("BH", "BM", "BF", "BG", "BA", "BG1", "BG2")
BUCKET = {"front": "前段", "middle": "中段", "back_near": "后段·近", "back_far": "后段·远"}
IN_WINDOW_MISSES = [("T1-2K-08", 0.125), ("T1-2K-10", 0.125), ("T1-8K-09", 0.03125), ("T1-8K-11", 0.03125), ("T1-8K-11", 0.125)]   # pass-2 B: inside the window, wrong
DEFAULT_TITLE = "# G2-lite 单变量重放 · Mac M4 Max：B 在窗口内仍答错，是否因为「表头不在名单里」\n"


def load_run(d: Path) -> dict:
    d = d if d.is_dir() else d.parent
    rows = [json.loads(l) for l in (d / "quality.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = {x["doc_id"]: x for x in (json.loads(l) for l in (d / "data.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    return {"dir": d, "rows": rows, "docs": docs, "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
            "manifest": json.loads((d / "manifest.json").read_text(encoding="utf-8"))}


def span_coverage(run, doc_id, mk, span):
    """Fraction of the span's positions in the saved row lists of the first generated token's step (mean over layers and groups)."""
    p = run["dir"] / "positions" / f"{doc_id}__{mk.replace('@', '_rho')}.npz"
    if p.exists() is False or span is None:
        return None
    import numpy as np
    z = np.load(p)
    sel = np.unpackbits(z["bits"], axis=-1)[..., :int(z["capacity"])].astype(bool)
    s, e = span
    return float(sel[:, :, s:e].mean())


def mark(r):
    if r is None:
        return "—"
    exact = r["exact_match"] if "exact_match" in r else r["exact"]
    return f"{r['predicted'][0] if r['predicted'] else '∅'} {'✓' if exact else '✗'}{'（编号）' if r.get('answered_with_ids') else ''}"


def brief(r):
    return {"predicted": r["predicted"], "exact": r["exact_match"], "answered_with_ids": r.get("answered_with_ids"), "text": r["text"], "generated_ids": r["generated_ids"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--reference", type=Path, default=ROOT / "runs/g2lite-20261017T130319Z-pass2-ABCD0")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/g2lite_replay_header_mac_m4max.md")
    ap.add_argument("--json", type=Path, default=ROOT / "reports/g2lite_replay_header_summary_mac_m4max.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "reports/g2lite_replay_header_manifest_mac_m4max.json")
    ap.add_argument("--card", default="reports/cards/G2lite_replay_header_mac_m4max.md")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    a = ap.parse_args()
    if a.run is None:
        cands = [load_run(Path(p)) for p in sorted(glob.glob(str(a.runs_dir / "g2lite-*/summary.json")))]
        cands = [r for r in cands if r["summary"].get("diagnostic_replay") and not r["summary"].get("quick")]
        if not cands:
            print("no replay run found")
            return 1
        rep = cands[-1]
    else:
        rep = load_run(a.run)
    ref = load_run(a.reference)
    if rep["summary"]["data_sha256"] != ref["manifest"]["data_sha256"]:
        print(f"data digests differ: replay {rep['summary']['data_sha256'][:16]} vs reference {ref['manifest']['data_sha256'][:16]}")
        return 1
    rhos = sorted({r["rho"] for r in rep["rows"] if r["rho"] is not None})
    variants = [v for v in VARIANT_ORDER if any(r["method"] == v for r in rep["rows"])]
    if not variants or not any(r["method"] == "B" for r in rep["rows"]):
        print("the replay run needs B and at least one pinned variant")
        return 1
    pinned_ref = "BH" if "BH" in variants else variants[0]          # the variant whose window defines the paired in-window set
    extra_pin = any(v not in ("BH", "BM") for v in variants)        # a variant that pins something other than the header: record each pinned span and its coverage
    by = {}
    for r in ref["rows"]:
        if r.get("variant", "base") == "base":
            by.setdefault(r["doc_id"], {})["ref:" + method_key(r["method"], r["rho"])] = r
    for r in rep["rows"]:
        by.setdefault(r["doc_id"], {})[method_key(r["method"], r["rho"])] = r
    docs = sorted(rep["docs"], key=lambda d: (rep["docs"][d]["length_tag"], rep["docs"][d]["doc_index"]))
    pairs = []
    for d in docs:
        doc = rep["docs"][d]
        P, Q = doc["prefix_token_count"], doc["suffix_token_count"]
        ts, te = doc["target_spans"][0]
        hs, he = doc["header_span"]
        for rho in rhos:
            m = by[d]
            b_ref, b, a_ref = m.get("ref:" + method_key("B", rho)), m.get(method_key("B", rho)), m.get("ref:A")
            vr = {v: m.get(method_key(v, rho)) for v in variants}
            if b is None or any(vr[v] is None for v in variants):
                continue
            k0 = b["k0"]
            N = P + Q                                                   # legal length at the step that produces the first generated token
            k_eff = {"B": k0, **{v: (vr[v]["k0"] if vr[v]["method"] == "BM" or vr[v]["n_pinned_candidates"] is None else vr[v]["k0"] - vr[v]["n_pinned_candidates"]) for v in variants}}
            win = {k: 64 + ke - Q for k, ke in k_eff.items()}
            start = {k: N - FIXED_RECENT - ke for k, ke in k_eff.items()}
            inside = {k: bool(ts >= s_) for k, s_ in start.items()}
            n_pin = next((vr[v]["n_pinned_candidates"] for v in variants if vr[v]["n_pinned_candidates"] is not None), None)
            pair = {"doc_id": d, "cell": cell_key(doc["task"], doc["length_tag"]), "rho": rho, "bucket": doc["buckets"][0], "P": P, "Q": Q, "k0_B": k0,
                    "k0_BM": vr["BM"]["k0"] if "BM" in variants else None, "n_pinned_candidates": n_pin, "target_span": [ts, te], "target_dist_end": P - te, "target_dist_start": P - ts,
                    "window_doc_tokens": win, "window_start": start, "inside_geometric": inside,
                    "coverage_first_gen_step": {"B": b["coverage"]["first_gen_step_mean"], **{v: vr[v]["coverage"]["first_gen_step_mean"] for v in variants}},
                    "coverage_ref_B": b_ref["coverage"]["first_gen_step_mean"] if b_ref else None,
                    "header_coverage": {"B": span_coverage(rep, d, method_key("B", rho), (hs, he)), **{v: span_coverage(rep, d, method_key(v, rho), (hs, he)) for v in variants}},
                    **({"pinned_span": {v: vr[v].get("pinned_span") for v in variants},
                        "pinned_coverage": {v: span_coverage(rep, d, method_key(v, rho), vr[v].get("pinned_span")) for v in variants}} if extra_pin else {}),
                    "A_ref": {"predicted": a_ref["predicted"], "exact": a_ref["exact_match"]} if a_ref else None,
                    "B_ref": {"predicted": b_ref["predicted"], "exact": b_ref["exact_match"], "answered_with_ids": b_ref.get("answered_with_ids"), "generated_ids": b_ref["generated_ids"]} if b_ref else None,
                    "B": brief(b), **{v: brief(vr[v]) for v in variants},
                    "B_identical_to_ref": (b_ref is not None and b["generated_ids"] == b_ref["generated_ids"]), "targets": b["targets"], "target_ids": b["target_ids"],
                    "is_pass2_in_window_miss": (d, rho) in IN_WINDOW_MISSES}
            for k in ("B", *variants):   # the measured coverage must agree with the geometry (1.0 inside, 0.0 outside, in between when the line straddles the window start)
                c = pair["coverage_first_gen_step"][k]
                pair.setdefault("coverage_agrees", {})[k] = (c >= 0.999) if inside[k] else (c < 0.999)
            pairs.append(pair)
    # ---- sets ----
    in_B = [p for p in pairs if p["inside_geometric"]["B"]]
    in_both = [p for p in in_B if all(p["inside_geometric"][v] for v in variants)]     # inside every variant's window: the paired set
    straddle = [p for p in in_B if not all(p["inside_geometric"][v] for v in variants)]
    win_vars = [pinned_ref] + [v for v in variants if v != pinned_ref and any(p["window_doc_tokens"][v] != p["window_doc_tokens"][pinned_ref] for p in pairs)]   # variants whose window differs from the reference variant's
    out_B = [p for p in pairs if not p["inside_geometric"]["B"]]
    misses = [p for p in pairs if p["is_pass2_in_window_miss"]]

    def hits(ps, k):
        return sum(1 for p in ps if p[k] and p[k]["exact"])

    def echoes(ps, k):
        return sum(1 for p in ps if p[k] and not p[k]["exact"] and p[k].get("answered_with_ids"))

    def two_by_two(ps, x, y):
        c = {"both_right": [], f"{x}_right_{y}_wrong": [], f"{x}_wrong_{y}_right": [], "both_wrong": []}
        for p in ps:
            if not (p[x] and p[y]):
                continue
            xr, yr = p[x]["exact"], p[y]["exact"]
            key = "both_right" if xr and yr else f"{x}_right_{y}_wrong" if xr else f"{x}_wrong_{y}_right" if yr else "both_wrong"
            c[key].append(f"{p['doc_id']}@{p['rho']:g}")
        return {k: {"n": len(v), "ids": v} for k, v in c.items()}

    cols = ["B", *variants]
    comparisons = [("B", v) for v in variants] + [(variants[j], variants[i]) for i in range(len(variants)) for j in range(i + 1, len(variants))]   # every later variant against every earlier one
    sets = {"in_window_B": in_B, "in_window_both": in_both, "straddle_BH": straddle, "out_of_window": out_B, "pass2_in_window_misses": misses, "all": pairs}
    summary_sets = {name: {"n": len(ps), **{f"{k}_hits": hits(ps, k) for k in ("B_ref", *cols)}, **{f"{k}_echoes": echoes(ps, k) for k in cols},
                           "ids": [f"{p['doc_id']}@{p['rho']:g}" for p in ps]} for name, ps in sets.items()}
    changed_vs_B = {k: [{"id": f"{p['doc_id']}@{p['rho']:g}", "inside_B": p["inside_geometric"]["B"], "inside_" + k: p["inside_geometric"][k], "B": p["B"]["predicted"], k: p[k]["predicted"], k + "_exact": p[k]["exact"], "B_exact": p["B"]["exact"], "targets": p["targets"]}
                        for p in pairs if p[k] and p[k]["predicted"] != p["B"]["predicted"]] for k in variants}
    by_cell = {}
    for c in sorted({p["cell"] for p in pairs}):
        for rho in rhos:
            ps = [p for p in pairs if p["cell"] == c and p["rho"] == rho]
            by_cell[f"{c}@{rho:g}"] = {"n": len(ps), **{f"{k}_hits": hits(ps, k) for k in ("A_ref", "B_ref", *cols)}, **{f"{x}_vs_{y}": two_by_two(ps, x, y) for x, y in comparisons},
                                      "k0_B": sorted({p["k0_B"] for p in ps}), "k0_BM": sorted({p["k0_BM"] for p in ps if p["k0_BM"] is not None}), "window_doc_tokens_B": sorted({p["window_doc_tokens"]["B"] for p in ps}),
                                      **{f"window_doc_tokens_{v}": sorted({p["window_doc_tokens"][v] for p in ps}) for v in win_vars}}
    b_identical = sum(1 for p in pairs if p["B_identical_to_ref"])
    cov_ok = all(all(v for v in p.get("coverage_agrees", {}).values()) for p in pairs)
    hc = {k: [p["header_coverage"][k] for p in pairs if p["header_coverage"][k] is not None] for k in cols}
    hc_stats = {k: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None} for k, v in hc.items()}
    pc = {v: [p["pinned_coverage"][v] for p in pairs if p.get("pinned_coverage") and p["pinned_coverage"][v] is not None] for v in variants}
    pc_stats = {v: {"n": len(x), "min": min(x) if x else None, "max": max(x) if x else None, "span": sorted({tuple(p["pinned_span"][v]) for p in pairs if p.get("pinned_span") and p["pinned_span"][v]})} for v, x in pc.items()}
    tm = rep["summary"]["aggregate"]["timing"]
    n_pin = sorted({p["n_pinned_candidates"] for p in pairs})
    n_by_v = {v: sorted({r["n_pinned_candidates"] for r in rep["rows"] if r["method"] == v and r.get("n_pinned_candidates") is not None}) for v in variants}   # pinned candidates per variant
    cond = {"replay_run": rep["summary"]["run_id"], "reference_run": ref["summary"]["run_id"], "replay_commit": rep["manifest"]["git"].get("commit"), "replay_dirty": rep["manifest"]["git"].get("dirty_files"),
            "reference_commit": ref["manifest"]["git"].get("commit"), "config_sha256": rep["summary"]["config_sha256"], "config_sha256_reference": ref["manifest"]["config_sha256"], "data_sha256": rep["summary"]["data_sha256"],
            "diagnostic_methods": rep["summary"].get("diagnostic_methods"), "rho": rhos, "header_span": sorted({tuple(rep["docs"][d]["header_span"]) for d in docs}), "n_pinned_candidates": n_pin,
            "B_rerun_identical_to_pass2": {"n": len(pairs), "identical": b_identical, "differing": [f"{p['doc_id']}@{p['rho']:g}" for p in pairs if not p["B_identical_to_ref"]]},
            "coverage_agrees_with_geometry": cov_ok, "header_coverage_first_gen_step": hc_stats, "sets": summary_sets, "by_cell_rho": by_cell, "changed_vs_B": changed_vs_B,
            "pass2_in_window_misses": [{"id": f"{p['doc_id']}@{p['rho']:g}", "B_ref": p["B_ref"]["predicted"] if p["B_ref"] else None, "B": p["B"]["predicted"],
                                        **{kk: vv for v in variants for kk, vv in ((v, p[v]["predicted"]), (f"{v}_exact", p[v]["exact"]))}, "targets": p["targets"]} for p in misses],
            "pairs": pairs, "timing": tm, "device_seconds": rep["summary"]["device_seconds"], "memory_end": rep["summary"].get("memory_end")}
    if extra_pin:
        cond["pinned_coverage_first_gen_step"] = pc_stats
    if len({tuple(x) for x in n_by_v.values()}) > 1:               # variants that pin different counts (hence different windows)
        cond["n_pinned_by_variant"] = n_by_v
    a.json.write_text(json.dumps(cond, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a.manifest.write_text(json.dumps({"runs": [rep["manifest"]], "reference_runs": [{"run_id": ref["manifest"]["run_id"], "git": ref["manifest"]["git"], "config_sha256": ref["manifest"]["config_sha256"], "data_sha256": ref["manifest"]["data_sha256"],
                                                                                   "source_hashes": ref["manifest"]["source_hashes"]}]}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # ---- markdown ----
    H, S, M = rep["summary"]["host"], rep["summary"]["software"], rep["manifest"]
    wv = "/".join(win_vars)
    ref_label = pinned_ref if len(win_vars) == 1 else wv
    st_out = [v for v in variants if any(not p["inside_geometric"][v] for p in straddle)] or list(variants)   # variants whose window start cuts the straddling records
    st_in = [v for v in variants if v not in st_out]
    descs = "；".join(DESC[v].format(n=n_by_v[v][0], span=f"[{pc_stats[v]['span'][0][0]}, {pc_stats[v]['span'][0][1]})" if pc_stats.get(v, {}).get("span") else "") for v in variants)

    def win_text(v_):   # variants grouped by their window set: "BH/BM [60]" or "BH [60]，BG [74]，BA [79]"
        groups = []
        for v in variants:
            w = v_.get(f"window_doc_tokens_{v}", v_[f"window_doc_tokens_{pinned_ref}"])
            g = next((g for g in groups if g[1] == w), None)
            if g is None:
                groups.append([[v], w])
            else:
                g[0].append(v)
        return "，".join(f"{'/'.join(vs)} {w}" for vs, w in groups)
    win_row = "；".join(f"{k}: B {v['window_doc_tokens_B']} → {win_text(v)}" for k, v in by_cell.items())
    L = [a.title,
         f"由 `scripts/g2lite_replay_report.py` 从重放 run `{rep['summary']['run_id']}`（commit `{M['git'].get('commit')}`，dirty {M['git'].get('dirty_files')}）与参照 run `{ref['summary']['run_id']}`（G2-lite 第二遍，commit `{ref['manifest']['git'].get('commit')}`）生成。"
         f"**24 份 T1 资料、每份两档 ρ，探索样本，不做区间统计；{' 与 '.join(variants)} 是诊断变体，不是读取方法的提案。** 预测见 `{a.card}`。\n",
         "## 1. 设置\n", "| 项 | 值 |\n|---|---|",
         f"| 机器 / 软件 | {H.get('chip')}，{H.get('gpu_cores')} 核 GPU，{(H.get('physical_memory_bytes') or 0)/2**30:.0f} GiB，macOS {H.get('macos')}；torch {S.get('torch')}，transformers {S.get('transformers')}，Python {H.get('python')}；MPS，bf16；fallback 与内存保护环境变量未设置 |",
         f"| 模型 / 配置 / 数据 | {M['model_id']} @ `{M['model_revision']}`；配置 v{M['config_version']} sha256 `{M['config_sha256'][:16]}…`（与参照 run 相同：{M['config_sha256'] == ref['manifest']['config_sha256']}）；数据 digest `{M['data_sha256'][:16]}…`（与参照 run 相同：{M['data_sha256'] == ref['manifest']['data_sha256']}）；HF_HUB_OFFLINE={M.get('hf_hub_offline')} |",
         f"| 表头 | 文本「{rep['docs'][docs[0]]['prefix_text'][rep['docs'][docs[0]]['prefix_text'].index('下面'):][:20]}…」，token 位置 {cond['header_span']}（位置 0–2 是聊天模板头；位置 3 在 F 的首 4 里），候选域内的表头位置数 {cond['n_pinned_candidates']} |",
         f"| {'三个' if len(variants) == 2 else '各'}请求 | B：F ∪ 最近 k0 个候选（与第二遍相同）；{descs}。{'三者' if len(variants) == 2 else '各请求'}各自克隆同一前缀缓存，从问题第一个 token 起逐 token 读取，贪心生成 |",
         f"| 窗口（第一个生成 token 那一步覆盖的资料 token 数） | {win_row} |", ""]
    L.append("## 2. 一致性检查\n")
    L.append(f"- B 重跑与第二遍：{b_identical}/{len(pairs)} 个请求生成的 token 序列逐位相同" + (f"（不同：{cond['B_rerun_identical_to_pass2']['differing']}）" if b_identical != len(pairs) else "") + "。")
    L.append(f"- 覆盖率与几何一致（目标在窗口内 ⇔ 第一个生成 token 那一步的目标覆盖率 = 100%）：{cov_ok}。")
    L.append(f"- 表头覆盖率（第一个生成 token 那一步，层组平均）：B {hc_stats['B']['min']:.3f}–{hc_stats['B']['max']:.3f}（只有位置 3 在 F 里），"
             + "，".join(f"{v} {hc_stats[v]['min']:.3f}–{hc_stats[v]['max']:.3f}" for v in variants) + "。"
             + "".join(f" {v} 钉住的位置 {[list(s_) for s_ in pc_stats[v]['span']]} 的覆盖率 {pc_stats[v]['min']:.3f}–{pc_stats[v]['max']:.3f}。" for v in variants if v not in ("BH", "BM")) + "\n")
    L.append("## 3. 配对表（按资料与 ρ；A 与「B 第二遍」来自参照 run，其余来自重放 run）\n")
    L.append(f"| 资料 | ρ | 目标位置 | 行首距前缀末尾 | k0(B) | 窗口 B→{wv} | 在窗口内 B / {wv} | A | B 第二遍 | B 重跑 | " + " | ".join(variants) + " |\n|---|---|---|---:|---:|---|---|---|---|---|" + "---|" * len(variants))
    for p in pairs:
        L.append(f"| {p['doc_id']} | {p['rho']:g} | {BUCKET.get(p['bucket'], p['bucket'])} | {p['target_dist_start']} | {p['k0_B']} | {p['window_doc_tokens']['B']}→{'/'.join(str(p['window_doc_tokens'][v]) for v in win_vars)} | {'内' if p['inside_geometric']['B'] else '外'} / {'/'.join('内' if p['inside_geometric'][v] else '外' for v in win_vars)} | "
                 f"{mark({'predicted': p['A_ref']['predicted'], 'exact': p['A_ref']['exact']}) if p['A_ref'] else '—'} | {mark(p['B_ref'])} | {mark(p['B'])}{'' if p['B_identical_to_ref'] else ' ≠'} | " + " | ".join(mark(p[v]) for v in variants) + " |")
    L.append(f"\n「（编号）」= 答成问题里的编号；{' / '.join(variants)} 的「内/外」按各自的窗口几何。\n")
    L.append("## 4. 窗口内命中率与第二遍的五个窗口内错误\n")
    L.append("| 集合 | n | B 第二遍 | B 重跑 | " + " | ".join(variants) + " |\n|---|---:|---:|---:|" + "---:|" * len(variants))
    for name, zh in (("in_window_B", "目标在 B 窗口内"), ("in_window_both", f"目标在 B 与 {ref_label} 窗口内（配对集合）"),
                     ("straddle_BH", f"目标在 B 窗口内、在 {'/'.join(st_out)} 窗口起点处被切开" + (f"（在 {'/'.join(st_in)} 窗口内）" if st_in else "")), ("out_of_window", "目标在 B 窗口外"), ("all", "全部")):
        s_ = summary_sets[name]
        L.append(f"| {zh} | {s_['n']} | {s_['B_ref_hits']} | {s_['B_hits']} | " + " | ".join(str(s_[f'{v}_hits']) for v in variants) + " |")
    L.append("")
    L.append("第二遍 B 在窗口内答错的五个请求：\n")
    L.append("| 请求 | 目标 | B 第二遍 | B 重跑 | " + " | ".join(variants) + " |\n|---|---|---|---|" + "---|" * len(variants))
    for p in misses:
        L.append(f"| {p['doc_id']} ρ={p['rho']:g} | {p['targets'][0]} | {mark(p['B_ref'])} | {mark(p['B'])} | " + " | ".join(mark(p[v]) for v in variants) + " |")
    L.append("")
    L.append("## 5. 配对四格（按格子与 ρ）\n")
    L.append("| 格子 | ρ | n | " + " | ".join(f"{x} 对 {y}：同对 / {x} 对 {y} 错 / {x} 错 {y} 对 / 同错" for x, y in comparisons) + " |\n|---|---|---:|" + "---|" * len(comparisons))
    for key, v in by_cell.items():
        c, rho = key.split("@")

        def fmt(t, x, y):
            return f"{t['both_right']['n']} / {t[f'{x}_right_{y}_wrong']['n']} / {t[f'{x}_wrong_{y}_right']['n']} / {t['both_wrong']['n']}"
        L.append(f"| {c} | {rho} | {v['n']} | " + " | ".join(fmt(v[f"{x}_vs_{y}"], x, y) for x, y in comparisons) + " |")
    L.append("")
    L.append("## 6. 与 B 回答不同的请求\n")
    for k in variants:
        ch = changed_vs_B[k]
        L.append(f"- {LABEL[k]} 与 B 重跑回答不同：{len(ch)} 个" + ("。" if not ch else "：" + "；".join(f"{x['id']}（B 答 {x['B'][0] if x['B'] else '∅'} {'✓' if x['B_exact'] else '✗'} → {k} 答 {x[k][0] if x[k] else '∅'} {'✓' if x[k + '_exact'] else '✗'}，目标 {x['targets'][0]}，目标在 {k} 窗口{'内' if x['inside_' + k] else '外'}）" for x in ch) + "。"))
    L.append("")
    L.append("## 7. 实际运行成本\n")
    L.append(f"重放 run 设备时间 {rep['summary']['device_seconds']:.0f} s（含模型加载 {rep['summary']['model_load_seconds']:.1f} s）；Prefill 中位 ms：" + "、".join(f"{k} {v:.0f}" for k, v in tm["prefill_ms_median_by_length"].items()) +
             "；每步 ms 中位：" + "、".join(f"{k} {v:.1f}" for k, v in tm["ms_per_step_median"].items()) + f"；结束时分配内存 {(rep['summary'].get('memory_end') or {}).get('current_allocated_bytes', 0)/1e9:.2f} GB。每步耗时含覆盖率记账与逐步同步，不是速度基准。现金 0。\n")
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out.relative_to(ROOT) if a.out.is_relative_to(ROOT) else a.out}; pairs {len(pairs)}; B identical {b_identical}/{len(pairs)}; in-window-both " +
          ", ".join(f"{v} {summary_sets['in_window_both'][f'{v}_hits']}/{summary_sets['in_window_both']['n']}" for v in variants) + "; misses fixed: " + ", ".join(f"{v} {sum(1 for p in misses if p[v]['exact'])}/{len(misses)}" for v in variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
