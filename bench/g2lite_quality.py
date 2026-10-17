#!/usr/bin/env python3
"""G2-lite: what does the model lose when it reads less of its history? Four read paths on real retrieval questions.

Pinned Qwen3-0.6B, bf16, MPS, batch 1, body frozen, untrained indexer (docs/tasks/05_G2_LITE.md). For every document:
  1. the shared prefix (chat-template head + header + record table; the question is NOT part of it) is processed once
     with the full read, in chunks, while the index table is built;
  2. every (method, rho) request gets an independent clone of that prefix cache and, from the first question token on,
     reads every token (question, template suffix, generated) as one decode step with its own selector:
       A   full read                      B   recent window (F + the k most recent candidates)
       C   uniform random (F + a seeded uniform draw of k candidates, per layer and group)
       D0  untrained indexer (F + the random GroupedIndexer's top-k)
     B, C and D0 share F and the budget k; nobody borrows another path's K/V; the answer is free greedy generation
     (<= 64 new tokens, EOS stops), parsed with the frozen rules of configs/mac_g2lite.json.
  3. coverage bookkeeping: per decode step and layer, how many of the queried records' token positions each KV group
     read; the per-layer, per-group row lists of the step that produces the first generated token are saved as bitsets.
Rewrites (value / irrelevant / move) rebuild the prefix from scratch. Pure MPS (refuses CPU fallback or a disabled
memory guard). Costs are recorded, but this is a quality run, not a speed benchmark: the bookkeeping is inside the
request time and every step syncs for the EOS check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_lab import g2lite_data as gd  # noqa: E402
from attention_lab import runinfo  # noqa: E402
from attention_lab.g2lite_eval import aggregate, cell_key, method_key, score_answer, select_move_cases, usable_cells  # noqa: E402

METHODS = ("A", "B", "C", "D0")
VARIANTS = ("base", "value", "irrelevant", "move")
WARN_BYTES, STOP_BYTES = 80_000_000_000, 100_000_000_000


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/mac_g2lite.json")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--rhos", type=float, nargs="+", default=None, help="read budgets for B / C / D0 (default: config)")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--lengths", nargs="+", default=None, help="length tags, e.g. 2K 8K")
    ap.add_argument("--docs-per-cell", type=int, default=None)
    ap.add_argument("--doc-indices", type=int, nargs="+", default=None, help="restrict to these document indices within every cell")
    ap.add_argument("--variants", nargs="+", default=["base"], choices=list(VARIANTS))
    ap.add_argument("--usable-from", type=Path, default=None, help="summary.json of an earlier run: restrict B/C/D0 (and A) to cells whose full read passed the usability check there")
    ap.add_argument("--cases-from", type=Path, default=None, help="quality.jsonl of an earlier run: select the 'move' cases by the pre-registered rule")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="pipeline smoke: T1 and T2 at 2K, one document per cell, run_id marked quick")
    a = ap.parse_args()
    a.cfg = json.loads(a.config.read_text(encoding="utf-8"))
    a.rhos = a.rhos or a.cfg["rho"]
    a.chunk = a.chunk or a.cfg["run"]["prefill_chunk"]
    if a.quick:
        a.lengths, a.docs_per_cell = ["2K"], 1
    return a


def mem(torch):
    return {"current_allocated_bytes": int(torch.mps.current_allocated_memory()), "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def uniform_seed_for(cfg, doc_id: str) -> int:
    return int(cfg["seeds"]["uniform_base"]) + int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % (1 << 20)


def read_rows(path: Path) -> list:
    p = path if path.name == "quality.jsonl" else path.parent / "quality.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_docs(tok, cfg, a, prev_rows, usable):
    """Base documents of the requested grid (filtered by usable cells / indices) plus the requested rewrites."""
    plan = gd.plan_docs(cfg, a.tasks, a.lengths, a.docs_per_cell)
    if a.doc_indices is not None:
        plan = [p for p in plan if p["doc_index"] in a.doc_indices]
    if usable is not None:
        plan = [p for p in plan if cell_key(p["task"], p["length_tag"]) in usable]
    docs = []
    for p in plan:
        base = gd.build_document(tok, cfg, p["task"], p["length_tag"], p["doc_index"])
        if "base" in a.variants:
            docs.append(base)
        for v in ("value", "irrelevant"):
            if v in a.variants:
                docs.append(gd.rewrite_document(tok, cfg, base, v, which=0))
    move_cases = []
    if "move" in a.variants:
        if prev_rows is None:
            raise SystemExit("--variants move needs --cases-from (the pre-registered rule selects cases from an earlier run)")
        cells = usable if usable is not None else {cell_key(p["task"], p["length_tag"]) for p in plan}
        move_cases = select_move_cases(prev_rows, cfg, cells)
        for case in move_cases:
            task, lt, idx = case["doc_id"].split("-")
            base = gd.build_document(tok, cfg, task, lt, int(idx))
            docs.append(gd.rewrite_document(tok, cfg, base, "move", which=case["which"]))
    return docs, move_cases


def main() -> int:
    a = parse_args()
    try:
        runinfo.refuse_unsafe_env()
    except RuntimeError as e:
        print(f"Refusing: {e}")
        return 2
    import numpy as np
    import torch
    if not torch.backends.mps.is_available():
        print("MPS not available; this harness is MPS-only.")
        return 1
    import transformers
    from attention_lab.qwen_wrap import LabRunner, LabState, StateHolder, geometry_of, load_model, make_cache, make_indexers, weights_file_sha256, wrap_model
    cfg, mcfg = a.cfg, a.cfg["model"]
    dev = torch.device("mps")
    sync = torch.mps.synchronize
    run_id = "g2lite-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    (out_dir / "positions").mkdir(parents=True, exist_ok=True)
    args_rec = {k: (runinfo.sanitize(str(v)) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "cfg"}
    gen_cfg, max_new, eos = cfg["generation"], int(cfg["generation"]["max_new_tokens"]), set(cfg["generation"]["eos_token_ids"])
    r = int(cfg["index"]["r"])
    prev_rows = read_rows(a.cases_from) if a.cases_from else None
    usable = None
    usable_src = None
    if a.usable_from:
        prev_summary = json.loads((a.usable_from if a.usable_from.name.endswith(".json") else a.usable_from / "summary.json").read_text(encoding="utf-8"))
        usable = {c for c, g in prev_summary["aggregate"]["gate"].items() if g["usable"]}
        usable_src = {"run_id": prev_summary["run_id"], "usable_cells": sorted(usable), "gate": prev_summary["aggregate"]["gate"]}
        if prev_rows is None:
            prev_rows = read_rows(Path(ROOT / prev_summary["_quality_path"]) if "_quality_path" in prev_summary else (a.usable_from.parent / "quality.jsonl" if a.usable_from.name.endswith(".json") else a.usable_from / "quality.jsonl"))
    t_start = time.perf_counter()
    dtype = getattr(torch, mcfg["dtype"])
    t0 = time.perf_counter()
    model, tok, info = load_model(mcfg["id"], mcfg["revision"], dtype=dtype, device=dev)
    sync()
    load_s = time.perf_counter() - t0
    geo = geometry_of(model)
    mism = {k: (info.get(k), v) for k, v in mcfg["expected_from_config_json"].items() if info.get(k) is not None and info.get(k) != v}
    print(f"run_id {run_id}  torch {torch.__version__}  transformers {transformers.__version__}  model loaded in {load_s:.1f} s: {geo}  mismatch_vs_config {mism or 'none'}")
    docs, move_cases = build_docs(tok, cfg, a, prev_rows, usable)
    if not docs:
        print("no documents selected (check --usable-from / --doc-indices / --variants)")
        return 1
    data_digest = gd.data_digest(docs)
    with open(out_dir / "data.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    chat_template_sha = hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest()
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "quality", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": runinfo.sha256_file(a.config), "config_version": cfg["config_version"],
                "model_id": mcfg["id"], "model_revision": mcfg["revision"], "tokenizer_revision": mcfg["tokenizer_revision"], "weights": weights_file_sha256(mcfg["id"], mcfg["revision"]),
                "chat_template_sha256": chat_template_sha, "data_sha256": data_digest, "data_seed": cfg["data"]["seed"], "generator_version": cfg["data"]["generator_version"],
                "indexer_seed": cfg["index"]["indexer_seed"], "uniform_base_seed": cfg["seeds"]["uniform_base"], "device": "mps", "backend": "mps", "dtype": mcfg["dtype"],
                "host": runinfo.host_info(), "software": {**runinfo.torch_env(torch), "transformers": transformers.__version__},
                "command": "python " + " ".join(runinfo.sanitize(x) for x in sys.argv), "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
                "timing_scope": "per request: question + template-suffix decode steps, then greedy generation, torch.mps.synchronize() at segment boundaries plus an implicit sync per step (EOS check); "
                                "coverage bookkeeping ops are inside the request time; prefix prefill (chunked, index table built) timed separately; model load separate. Quality run, not a speed benchmark.",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["data.jsonl", "quality.jsonl", "summary.json", "manifest.json", "positions/*.npz"], "args": args_rec,
                "usable_from": usable_src, "move_cases": move_cases, "n_docs": len(docs),
                "limitations": ["48 exploratory documents at most; T1/T2 record lookup only; Qwen3-0.6B bf16 on MPS; untrained indexer; no training; no statistical interval claims"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    holder = wrap_model(model)
    indexers = make_indexers(model, r=r, seed=int(cfg["index"]["indexer_seed"]))
    rec_max = manifest["software"].get("recommended_max_memory") or 0
    rows, prefill_ms_by_doc = [], {}
    qf = open(out_dir / "quality.jsonl", "w", encoding="utf-8")
    plan_requests = [("A", None)] if "A" in a.methods else []
    plan_requests += [(m, rho) for m in a.methods if m != "A" for rho in a.rhos]
    print(f"{len(docs)} documents x {len(plan_requests)} requests {[method_key(m, rho) for m, rho in plan_requests]}; usable cells: {sorted(usable) if usable is not None else 'all'}; move cases: {move_cases}")

    def timed(fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        return (time.perf_counter() - t) * 1e3, out

    with torch.inference_mode():
        for di, doc in enumerate(docs):
            prefix = torch.tensor([doc["prefix_ids"]], device=dev)
            suffix = torch.tensor([doc["suffix_ids"]], device=dev)
            P, Q = prefix.shape[1], suffix.shape[1]
            cap = doc["prompt_token_count"] + max_new + int(cfg["run"]["capacity_margin"])
            cache0 = make_cache(model, cap, with_index=True, index_dim=r)
            need = 2 * cache0.bytes()["total_bytes"]
            if need > STOP_BYTES or (rec_max and need > 0.8 * rec_max):
                print(f"skip {doc['doc_id']}: {need/1e9:.1f} GB of caches exceeds limits")
                continue
            st0 = LabState(cache0, indexers, mode="full")               # dense prefix, index table filled
            run0 = LabRunner(model, holder, st0)
            pre_ms, _ = timed(lambda: run0.prefill(prefix, a.chunk))
            if cache0.valid_length != P:
                raise RuntimeError("prefix length mismatch after prefill")
            prefill_ms_by_doc[doc["doc_id"]] = pre_ms
            G, L = cache0.groups, cache0.layers
            span_pos = torch.tensor([p for s, e in doc["target_spans"] for p in range(s, e)], device=dev, dtype=torch.long)
            n_target = int(span_pos.numel())
            lut = torch.zeros(cache0.batch * G * cap, dtype=torch.int16, device=dev)
            lut[(span_pos.view(1, -1) + cache0.offsets.view(-1, 1)).reshape(-1)] = 1
            line = f"[{di+1}/{len(docs)}] {doc['doc_id']:<22s} P={P} Q={Q} n={doc['n_records']} buckets={doc['buckets']} prefill {pre_ms:7.0f} ms |"
            for method, rho in plan_requests:
                spec = cfg["methods"][method]
                cache = cache0.clone()
                useed = uniform_seed_for(cfg, doc["doc_id"])
                st = LabState(cache, indexers if method == "D0" else None, mode=spec["mode"], selector=spec["selector"] or "index", uniform_seed=useed)
                run = LabRunner(model, holder, st)
                k0 = st.begin_request(rho) if method != "A" else None
                st.record_selection = method != "A"
                max_steps = Q + max_new
                cov = torch.zeros(max_steps, L, G, dtype=torch.int32, device=dev) if method != "A" else None

                def account(step):
                    if cov is None:
                        return
                    for l in range(L):
                        cov[step, l] = lut.index_select(0, st.last_rows[l]).view(G, -1).sum(-1)

                sync()
                t_q = time.perf_counter()
                logits = None
                for j in range(Q):
                    logits = run.step(suffix[:, j:j + 1])
                    account(j)
                sync()
                q_ms = (time.perf_counter() - t_q) * 1e3
                first_rows = list(st.last_rows) if method != "A" else None     # row lists of the step that produces the first generated token
                gen = []
                t_g = time.perf_counter()
                for s in range(max_new):
                    nxt = logits[0, -1].argmax()
                    tid = int(nxt)
                    gen.append(tid)
                    if tid in eos or len(gen) >= max_new:
                        break
                    logits = run.step(nxt.view(1, 1))
                    account(Q + s)
                sync()
                g_ms = (time.perf_counter() - t_g) * 1e3
                steps = Q + len(gen) - 1
                text = tok.decode([t for t in gen if t not in eos], skip_special_tokens=True)
                truncated = gen[-1] not in eos
                sc = score_answer(doc["task"], doc["targets"], text, truncated=truncated)
                row = {"run_id": run_id, "doc_id": doc["doc_id"], "task": doc["task"], "length_tag": doc["length_tag"], "doc_index": doc["doc_index"], "variant": doc["variant"],
                       "parent_doc_id": doc.get("parent_doc_id"), "buckets": doc["buckets"], "method": method, "rho": rho, "k0": k0,
                       "k_last": st.plan.k if st.plan is not None else None, "selected_count": st.plan.selected_count if st.plan is not None else None,
                       "legal_start": P + 1, "legal_end": cache.valid_length, "capacity": cap, "prefix_tokens": P, "suffix_tokens": Q, "generated_tokens": len(gen),
                       "generated_ids": gen, "targets": doc["targets"], "target_ids": doc["target_ids"], "target_records_from_end": doc["target_records_from_end"],
                       "target_distance_from_prefix_end": doc["target_distance_from_prefix_end"], "changed_record_index": doc.get("changed_record_index"), "note": doc.get("note"),
                       **{k: v for k, v in sc.items()}, "question_ms": q_ms, "generation_ms": g_ms, "steps": steps, "ms_per_step": (q_ms + g_ms) / max(steps, 1),
                       "uniform_seed": useed if method == "C" else None, "uniform_request_counter": st.request_counter if method == "C" else None,
                       "coverage": {"target_tokens": n_target}}
                if cov is not None:
                    c = cov[:steps].float() / max(n_target, 1)                           # fraction of the queried records' positions read, per step/layer/group
                    cq, cg = c[:Q], c[Q:]
                    first = c[Q - 1]                                                     # the step that produces the first generated token
                    row["coverage"].update({"mean_fraction_question_steps": float(cq.mean()), "mean_fraction_generation_steps": float(cg.mean()) if cg.numel() else None,
                                            "first_gen_step_mean": float(first.mean()), "first_gen_step_min_layer": float(first.mean(-1).min()), "first_gen_step_max_layer": float(first.mean(-1).max()),
                                            "first_gen_step_per_layer": [round(float(x), 4) for x in first.mean(-1)],
                                            "fraction_of_layer_groups_with_full_coverage_first_gen_step": float((first >= 1.0 - 1e-6).float().mean())})
                    sel = np.zeros((L, G, cap), dtype=bool)
                    for l in range(L):
                        pos = (first_rows[l].view(G, -1) - cache.offsets.view(G, 1)).cpu().numpy()
                        for g in range(G):
                            sel[l, g, pos[g]] = True
                    np.savez_compressed(out_dir / "positions" / f"{doc['doc_id']}__{method_key(method, rho).replace('@', '_rho')}.npz", bits=np.packbits(sel, axis=-1),
                                        coverage=cov[:steps].cpu().numpy().astype(np.int16), capacity=cap, prefix_len=P, suffix_len=Q, steps=steps,
                                        target_spans=np.array(doc["target_spans"], dtype=np.int32), legal_first_gen_step=P + Q)
                rows.append(row)
                qf.write(json.dumps(row, ensure_ascii=False) + "\n")
                qf.flush()
                line += f" {method_key(method, rho)}:{'✓' if sc['exact_match'] else '✗'}{'(trunc)' if truncated else ''}"
                del cache, st, run, cov
            m = mem(torch)
            print(line + f" | {m['current_allocated_bytes']/1e9:.2f} GB", flush=True)
            del cache0, st0, run0, lut
            torch.mps.empty_cache()
            if m["current_allocated_bytes"] > WARN_BYTES:
                print("WARNING: allocated memory above the 80 GB warning line; stopping")
                break
    qf.close()
    device_s = time.perf_counter() - t_start
    agg = aggregate(rows, docs, cfg, prefill_ms_by_doc, prev_rows=prev_rows)
    summary = {"run_id": run_id, "quick": a.quick, "utc_start": manifest["created_at_utc"], "utc_end": runinfo.utc_now(), "device_seconds": device_s, "model_load_seconds": load_s,
               "args": args_rec, "config_version": cfg["config_version"], "config_sha256": manifest["config_sha256"], "data_sha256": data_digest, "model_info": {**info, "mismatch_vs_config_json": mism},
               "host": manifest["host"], "software": manifest["software"], "git": manifest["git"], "source_hashes": manifest["source_hashes"], "usable_from": usable_src, "move_cases": move_cases,
               "n_docs": len(docs), "n_rows": len(rows), "prefill_ms_by_doc": prefill_ms_by_doc, "aggregate": agg, "memory_end": mem(torch), "_quality_path": runinfo.sanitize(str((out_dir / "quality.jsonl").relative_to(ROOT)))}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest.update({"status": "done", "finished_at_utc": summary["utc_end"], "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": device_s}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n== exact-match rate per cell (base documents) ==")
    for c, table in agg["methods"].items():
        print(f"  {c:<8s} " + "  ".join(f"{mk}: {t['hits']}/{t['n']}" for mk, t in table.items()) + f"   A usable: {agg['gate'][c]['usable']}")
    if agg["rewrites"]:
        print(f"  rewrites: {len(agg['rewrites'])} rows")
    print(f"\nwrote {runinfo.sanitize(str(out_dir))}  ({device_s:.0f} s device time incl. load {load_s:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
