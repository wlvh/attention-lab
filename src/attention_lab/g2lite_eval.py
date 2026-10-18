"""G2-lite answer parsing and scoring. Pure Python; the rules are frozen in configs/mac_g2lite.json before any run.

T1 asks for one four-digit value, T2 for two values in a stated order. The model's free generation is decoded (special
tokens dropped, text cut at the first EOS), every standalone four-digit number is extracted in order of appearance, and
the first `len(targets)` of them are compared with the targets position by position. Nothing is normalised beyond
whitespace; a five-digit run is not a four-digit number; "7305。" and "值是 7305" both yield 7305.
"""
from __future__ import annotations

import re

FOUR_DIGITS = re.compile(r"(?<!\d)\d{4}(?!\d)")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def clean_text(text: str) -> str:
    """Drop a stray think block (the template already closes thinking; the model should not emit another) and trim."""
    return THINK_BLOCK.sub("", text).strip()


def extract_numbers(text: str) -> list[str]:
    """Standalone four-digit numbers in order of appearance (digits are single tokens in Qwen3, so the model's answer
    arrives as a run of digit tokens that decodes to a plain number)."""
    return FOUR_DIGITS.findall(clean_text(text))


def score_answer(task: str, targets, text: str, *, truncated: bool) -> dict:
    """Exact match = the first len(targets) standalone four-digit numbers equal the targets in order.

    Also reported: per-item correctness, whether every target appears anywhere (any order), whether a two-item answer is
    exactly the targets swapped, format_error (fewer numbers than required) and the truncation flag passed by the caller.
    """
    targets = [str(t) for t in targets]
    need = len(targets)
    if (task == "T1" and need != 1) or (task == "T2" and need != 2):
        raise ValueError(f"{task} expects {1 if task == 'T1' else 2} target(s), got {need}")
    numbers = extract_numbers(text)
    predicted = numbers[:need]
    return {
        "numbers": numbers,
        "predicted": predicted,
        "exact_match": predicted == targets,
        "item_correct": [i < len(predicted) and predicted[i] == t for i, t in enumerate(targets)],
        "targets_present_any_order": all(t in numbers for t in targets),
        "order_swapped": need == 2 and len(predicted) == 2 and predicted == list(reversed(targets)),
        "format_error": len(numbers) < need,
        "truncated": bool(truncated),
        "text": clean_text(text),
    }


def cell_rate(rows, key: str = "exact_match") -> dict:
    """n and rate of a boolean field over rows (used for the per-cell tables and the baseline usability check)."""
    rows = list(rows)
    n = len(rows)
    hits = sum(1 for r in rows if r.get(key))
    return {"n": n, "hits": hits, "rate": (hits / n) if n else None}


# ----------------------------------------------------------------------------------------------------------------------
# aggregation of quality rows (pure Python; used by the harness at the end of a run and by scripts/g2lite_report.py)
# ----------------------------------------------------------------------------------------------------------------------
def cell_key(task: str, length_tag: str) -> str:
    return f"{task}/{length_tag}"


def method_key(method: str, rho) -> str:
    return method if rho is None else f"{method}@{rho:g}"


def usable_cells(rows, min_exact: float) -> dict:
    """Per cell: the full read's exact-match rate over base documents and whether the cell passes the usability check."""
    out = {}
    cells = sorted({cell_key(r["task"], r["length_tag"]) for r in rows})
    for c in cells:
        a_rows = [r for r in rows if cell_key(r["task"], r["length_tag"]) == c and r["method"] == "A" and r.get("variant", "base") == "base"]
        rate = cell_rate(a_rows)
        out[c] = {**rate, "usable": (rate["rate"] is not None and rate["rate"] >= min_exact), "min_exact": min_exact}
    return out


def paired_table(rows, cells=None) -> dict:
    """2x2 agreement of every sparse (method, rho) against A per cell over base documents, joined by doc_id."""
    rows = [r for r in rows if r.get("variant", "base") == "base"]
    out = {}
    for c in sorted({cell_key(r["task"], r["length_tag"]) for r in rows}):
        if cells is not None and c not in cells:
            continue
        in_cell = [r for r in rows if cell_key(r["task"], r["length_tag"]) == c]
        a = {r["doc_id"]: r for r in in_cell if r["method"] == "A"}
        out[c] = {}
        for mk in sorted({method_key(r["method"], r["rho"]) for r in in_cell if r["method"] != "A"}):
            others = {r["doc_id"]: r for r in in_cell if method_key(r["method"], r["rho"]) == mk}
            ids = sorted(set(a) & set(others))
            cell = {"n": len(ids), "both_right": [], "A_right_other_wrong": [], "A_wrong_other_right": [], "both_wrong": []}
            for d in ids:
                ar, orr = bool(a[d]["exact_match"]), bool(others[d]["exact_match"])
                key = "both_right" if ar and orr else "A_right_other_wrong" if ar else "A_wrong_other_right" if orr else "both_wrong"
                cell[key].append(d)
            cell["other_exact_rate"] = cell_rate([others[d] for d in ids])["rate"]
            cell["A_exact_rate"] = cell_rate([a[d] for d in ids])["rate"]
            cell["delta_other_minus_A"] = None if cell["n"] == 0 else cell["other_exact_rate"] - cell["A_exact_rate"]
            cell["counts"] = {k: len(cell[k]) for k in ("both_right", "A_right_other_wrong", "A_wrong_other_right", "both_wrong")}
            out[c][mk] = cell
    return out


def method_tables(rows) -> dict:
    """Per cell and (method, rho): n, exact rate, format errors, truncations, item-level rates (base documents)."""
    rows = [r for r in rows if r.get("variant", "base") == "base"]
    out = {}
    for c in sorted({cell_key(r["task"], r["length_tag"]) for r in rows}):
        out[c] = {}
        in_cell = [r for r in rows if cell_key(r["task"], r["length_tag"]) == c]
        for mk in sorted({method_key(r["method"], r["rho"]) for r in in_cell}, key=lambda s: (s.split("@")[0] != "A", s)):
            rs = [r for r in in_cell if method_key(r["method"], r["rho"]) == mk]
            n_items = sum(len(r["item_correct"]) for r in rs)
            out[c][mk] = {**cell_rate(rs), "format_errors": sum(1 for r in rs if r["format_error"]), "truncated": sum(1 for r in rs if r["truncated"]),
                          "any_order": cell_rate(rs, "targets_present_any_order")["rate"],
                          "item_rate": (sum(sum(r["item_correct"]) for r in rs) / n_items) if n_items else None,
                          "k0": sorted({r["k0"] for r in rs if r["k0"] is not None}), "selected": sorted({r["selected_count"] for r in rs if r.get("selected_count") is not None}),
                          "by_bucket": {b: cell_rate([r for r in rs if "/".join(r["buckets"]) == b])["rate"] for b in sorted({"/".join(r["buckets"]) for r in rs})}}
    return out


def rewrite_table(rows) -> list:
    """Counterfactual checks, joined by parent doc and (method, rho): does the answer follow a change of the queried value,
    stay put under an irrelevant change, and become right after the record is moved into the window."""
    base = {(r["doc_id"], method_key(r["method"], r["rho"])): r for r in rows if r.get("variant", "base") == "base"}
    out = []
    for r in rows:
        v = r.get("variant", "base")
        if v == "base":
            continue
        mk = method_key(r["method"], r["rho"])
        b = base.get((r["parent_doc_id"], mk))
        rec = {"doc_id": r["doc_id"], "parent_doc_id": r["parent_doc_id"], "variant": v, "task": r["task"], "length_tag": r["length_tag"], "method": r["method"], "rho": r["rho"],
               "base_predicted": b["predicted"] if b else None, "base_exact": b["exact_match"] if b else None,
               "rewrite_predicted": r["predicted"], "rewrite_exact": r["exact_match"], "base_targets": b["targets"] if b else None, "rewrite_targets": r["targets"],
               "changed_record_index": r.get("changed_record_index"), "note": r.get("note")}
        rec.update({"paired": b is not None,
                    "answer_unchanged": b is not None and r["predicted"] == b["predicted"],
                    "answer_changed": b is not None and r["predicted"] != b["predicted"],
                    "correctness_changed": b is not None and r["exact_match"] != b["exact_match"],
                    "both_wrong": b is not None and not b["exact_match"] and not r["exact_match"],
                    "base_answered_with_ids": b.get("answered_with_ids") if b else None,
                    "rewrite_answered_with_ids": r.get("answered_with_ids"),
                    "became_id_echo": b is not None and not b.get("answered_with_ids") and bool(r.get("answered_with_ids"))})
        if r["method"] == "C":
            fields = ("uniform_seed", "uniform_request_counter", "legal_start", "k0", "suffix_tokens")
            comparable = b is not None and all(b.get(k) is not None and b.get(k) == r.get(k) for k in fields)
            rec.update({"base_uniform_seed": b.get("uniform_seed") if b else None,
                        "rewrite_uniform_seed": r.get("uniform_seed"),
                        "uniform_conditions_equal": comparable,
                        "counterfactual_control": "matched_uniform_conditions" if comparable else "confounded_uniform_conditions"})
        else:
            rec["counterfactual_control"] = "content_responsive_indexer" if r["method"] == "D0" else "deterministic_read_rule"
        if v == "value":
            rec["answer_followed_change"] = (b is not None) and r["exact_match"] and r["targets"] != b["targets"]
            rec["answer_stuck_to_old_value"] = (b is not None) and (r["predicted"] == b["targets"])
        elif v == "irrelevant":
            rec["answer_unchanged"] = (b is not None) and (r["predicted"] == b["predicted"])
        elif v == "move":
            rec["became_right"] = (b is not None) and (not b["exact_match"]) and r["exact_match"]
        out.append(rec)
    return out


def rewrite_observations(rewrites: list) -> dict:
    """Counts of distinct events, never equating 'still wrong' with 'unchanged'.

Each unit is a paired (document, method, rho) request, not an independent document.
The C control flag describes random-stream conditions, not a new model result.
    """
    out = {}
    for r in rewrites:
        if not r["paired"]:
            continue
        cell = out.setdefault(r["variant"], {}).setdefault(r["method"], {"n": 0, **{k: 0 for k in (
            "answer_unchanged", "answer_changed", "correctness_changed", "both_wrong", "rewrite_answered_with_ids",
            "became_id_echo", "answer_followed_change", "base_exact", "confounded_uniform_conditions")}})
        cell["n"] += 1
        for k in cell.keys() - {"n", "confounded_uniform_conditions"}:
            cell[k] += int(bool(r.get(k)))
        cell["confounded_uniform_conditions"] += int(r["counterfactual_control"] == "confounded_uniform_conditions")
    return out


def budget_tables(rows: list) -> dict:
    """Actual budgets at the step predicting the first generated token.

rho fixes k0 at N0=P+1; at this later step N=P+Q, M=N-|F|. Thus rho,
k/M, and (|F|+k)/N are three different quantities. Compute ratios per request
before averaging; don't use the first document as a proxy for the whole cell.
    """
    import statistics
    groups = {}
    for r in rows:
        if r["method"] == "A" or r.get("variant", "base") != "base":
            continue
        groups.setdefault(cell_key(r["task"], r["length_tag"]), {}).setdefault(method_key(r["method"], r["rho"]), []).append(r)
    out = {}
    for cell, methods in groups.items():
        out[cell] = {}
        for mk, rs in methods.items():
            legal = [r["prefix_tokens"] + r["suffix_tokens"] for r in rs]
            fixed = [r["selected_count"] - r["k0"] for r in rs]
            candidate = [n - f for n, f in zip(legal, fixed)]
            fractions = [r["k0"] / m for r, m in zip(rs, candidate)]
            total = [r["selected_count"] / n for r, n in zip(rs, legal)]
            out[cell][mk] = {"step": "first_generated_token_prediction", "n": len(rs), "rho": rs[0]["rho"],
                            "k0": sorted({r["k0"] for r in rs}), "selected": sorted({r["selected_count"] for r in rs}),
                            "legal": [min(legal), max(legal)], "candidate_count": [min(candidate), max(candidate)],
                            "candidate_fraction": [min(fractions), max(fractions)], "candidate_fraction_mean": statistics.mean(fractions),
                            "total_read_fraction": [min(total), max(total)]}
    return out


def select_move_cases(rows, cfg: dict, cells) -> list:
    """Pre-registered rule (configs/mac_g2lite.json data.move_case_rule): per usable cell, the lowest doc indices where A is
    exact and B is not at rho 1/32 (else at rho 1/8), at most max_per_cell; returns (doc_id, which) pairs."""
    rule = cfg["data"]["move_case_rule"]
    rows = [r for r in rows if r.get("variant", "base") == "base"]
    rhos = sorted({r["rho"] for r in rows if r["rho"] is not None})
    out = []
    for c in sorted(cells):
        in_cell = [r for r in rows if cell_key(r["task"], r["length_tag"]) == c]
        a = {r["doc_id"]: r for r in in_cell if r["method"] == "A"}
        chosen = []
        for rho in rhos:
            b = {r["doc_id"]: r for r in in_cell if r["method"] == "B" and r["rho"] == rho}
            cands = sorted(d for d in a if d in b and a[d]["exact_match"] and not b[d]["exact_match"])
            cands = sorted(cands, key=lambda d: a[d]["doc_index"])
            chosen = cands[:rule["max_per_cell"]]
            if chosen:
                break
        for d in chosen:
            r = a[d]
            which = 0 if r["task"] == "T1" else max(range(len(r["target_distance_from_prefix_end"])), key=lambda i: r["target_distance_from_prefix_end"][i])
            out.append({"doc_id": d, "which": which, "rho_selected_at": rho})
    return out


def timing_table(rows, prefill: dict) -> dict:
    """Medians of the recorded costs: prefill per length, decode ms per step per (method, length), totals."""
    import statistics
    out = {"prefill_ms_median_by_length": {}, "ms_per_step_median": {}, "request_ms_median": {}, "total_request_ms": sum(r["question_ms"] + r["generation_ms"] for r in rows)}
    for lt in sorted({r["length_tag"] for r in rows}):
        vals = [v for k, v in prefill.items() if k.split("::")[0].split("-")[1] == lt] if prefill else []
        out["prefill_ms_median_by_length"][lt] = statistics.median(vals) if vals else None
        for mk in sorted({method_key(r["method"], r["rho"]) for r in rows}):
            rs = [r for r in rows if r["length_tag"] == lt and method_key(r["method"], r["rho"]) == mk]
            if rs:
                out["ms_per_step_median"][f"{lt}/{mk}"] = statistics.median(r["ms_per_step"] for r in rs)
                out["request_ms_median"][f"{lt}/{mk}"] = statistics.median(r["question_ms"] + r["generation_ms"] for r in rs)
    return out


def aggregate(rows, docs, cfg: dict, prefill_ms: dict, prev_rows=None) -> dict:
    """Everything the report needs from one run's rows: per-cell tables, gate, paired 2x2, rewrites, timing, consistency."""
    gate = usable_cells(rows, cfg["gate"]["baseline_usable_min_exact"])
    out = {"docs": [{k: d.get(k) for k in ("doc_id", "task", "length_tag", "variant", "parent_doc_id", "buckets", "n_records", "prefix_token_count", "suffix_token_count",
                                            "prompt_token_count", "targets", "target_ids", "target_indices", "target_records_from_end", "target_distance_from_prefix_end",
                                            "target_spans", "changed_record_index", "note", "question")} for d in docs],
           "gate": gate, "methods": method_tables(rows), "paired": paired_table(rows), "rewrites": rewrite_table(rows), "timing": timing_table(rows, prefill_ms)}
    if prev_rows is not None:
        prev_a = {r["doc_id"]: r for r in prev_rows if r["method"] == "A" and r.get("variant", "base") == "base"}
        cur_a = [r for r in rows if r["method"] == "A" and r.get("variant", "base") == "base" and r["doc_id"] in prev_a]
        out["a_consistency_vs_prev"] = {"n_compared": len(cur_a), "n_identical_generated_ids": sum(1 for r in cur_a if r["generated_ids"] == prev_a[r["doc_id"]]["generated_ids"]),
                                        "differing_doc_ids": [r["doc_id"] for r in cur_a if r["generated_ids"] != prev_a[r["doc_id"]]["generated_ids"]]}
    return out
