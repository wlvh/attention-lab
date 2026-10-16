#!/usr/bin/env python3
"""G1a: single-layer decode read-stage microbenchmark on this Mac's MPS: full attention A vs index-selected D.

Timed per layer, per decode step (one new query, batch 1, random tensors):
  A       full attention over all N valid positions: SDPA reading K and V, softmax, PV. The backend is the
          fastest *legitimate* GQA variant measured in-process (sdpa_fold or sdpa_gqa); repeat_interleave and the
          explicit reference are timed as diagnostics only, never used as A.
  D_full  z append for the newest key + u projection + one matmul over the candidate columns + one top-k over the
          candidate domain + merge with the fixed set + flattened index_select of K and V rows + subset SDPA.
  D_read  D_full without the z append (read stage from an existing index table).
Not timed in either path: appending the new K/V row (identical in both), q/k/v/o projections, MLP, other layers.

One pass = `layers` layer-steps. Rotating workset cycles through distinct per-layer buffers; hot repeats one
layer's buffers. torch.mps.synchronize() before and after every pass; warm-up passes are separate from the timed
ones. Raw per-pass times go to runs/<run_id>/microbench.jsonl, summaries to summary.json, provenance to
manifest.json. torch.compile is applied to both A and D with graph-break and recompile detection; compile failures
are recorded, not hidden. Pure MPS: the script refuses to run with CPU fallback enabled or the memory guard off.

Numbers are measurements of this host under the recorded software. They are not model results, not quality
results and not CUDA conclusions.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_lab import runinfo  # noqa: E402
from attention_lab.decode import prepare_request, select_rows_kernel, sparse_decode_step_kernel  # noqa: E402
from attention_lab.full import full_attention  # noqa: E402
from attention_lab.indexer import GroupedIndexer, candidate_scores, project_keys, project_queries  # noqa: E402
from attention_lab.sparse import flatten_cache, subset_attention  # noqa: E402

A_LEGIT = ("sdpa_fold", "sdpa_gqa")
A_DIAG = ("sdpa_repeat", "explicit")
MATMUL_OPS = {"aten::matmul", "aten::bmm", "aten::mm", "aten::baddbmm", "aten::addmm", "aten::linear", "aten::einsum",
              "aten::scaled_dot_product_attention", "aten::_scaled_dot_product_attention_math",
              "aten::_scaled_dot_product_attention_math_for_mps", "aten::_scaled_dot_product_flash_attention",
              "aten::_scaled_dot_product_efficient_attention", "aten::_scaled_dot_product_fused_attention_overrideable",
              "aten::_fused_sdp_choice"}
WARN_BYTES, STOP_BYTES = 80_000_000_000, 100_000_000_000


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"], choices=["bfloat16", "float32", "float16"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--rhos", type=float, nargs="+", default=[1 / 32, 1 / 8])
    ap.add_argument("--layers", type=int, default=36, help="layer-steps per pass; also the rotating workset size")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--heads", type=int, default=32, help="query heads H; h = H / groups")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--index-dim", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--modes", nargs="+", default=["eager"], choices=["eager", "compile"])
    ap.add_argument("--worksets", nargs="+", default=["rotating", "hot"], choices=["rotating", "hot"])
    ap.add_argument("--skip-diagnostics", action="store_true", help="skip component, gather-sensitivity, slack and heads diagnostics")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 warm-ups, 5 repeats, run_id marked quick")
    a = ap.parse_args()
    if a.quick:
        a.warmup, a.repeats = 3, 5
    return a


def percentile(xs, p):
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def stats(ms, layers):
    return {"n": len(ms), "ms_per_pass_median": statistics.median(ms), "ms_per_pass_p95": percentile(ms, 0.95),
            "ms_per_pass_min": min(ms), "ms_per_pass_max": max(ms), "ms_per_pass_mean": statistics.fmean(ms),
            "us_per_layer_median": statistics.median(ms) / layers * 1e3, "us_per_layer_p95": percentile(ms, 0.95) / layers * 1e3}


def make_pass(step, order):
    def run():
        for i in order:
            step(i)
    return run


class LayerSet:
    """Per-layer buffers of one decode state: K/V caches, index tables, indexer weights, this step's q and new k."""

    def __init__(self, torch, dev, dtype, layers, G, h, d, r, N, seed, capacity=None):
        C = N if capacity is None else capacity
        torch.manual_seed(seed)
        self.N, self.C, self.G, self.h, self.d, self.r, self.layers = N, C, G, h, d, r, layers
        self.K = [torch.randn(1, G, C, d, device=dev, dtype=dtype) for _ in range(layers)]
        self.V = [torch.randn(1, G, C, d, device=dev, dtype=dtype) for _ in range(layers)]
        self.zT = [torch.randn(1, G, r, C, device=dev, dtype=dtype) for _ in range(layers)]   # [B,G,r,C] index tables
        self.idx = [GroupedIndexer(G, h, d, r, device=dev, dtype=dtype, seed=seed * 1000 + i) for i in range(layers)]
        self.q_main = [torch.randn(1, G, h, 1, d, device=dev, dtype=dtype) for _ in range(layers)]
        self.q_pre = [torch.randn(1, G, h, 1, d, device=dev, dtype=dtype) for _ in range(layers)]
        self.k_pre_new = [torch.randn(1, G, 1, d, device=dev, dtype=dtype) for _ in range(layers)]
        es = torch.tensor([], dtype=dtype).element_size()
        self.kv_bytes_per_layer = 2 * G * C * d * es
        self.index_bytes_per_layer = G * r * C * es
        self.bytes_total = layers * (self.kv_bytes_per_layer + self.index_bytes_per_layer)

    @staticmethod
    def estimate_bytes(layers, G, C, d, r, es):
        return layers * (2 * G * C * d * es + G * r * C * es)


class Bench:
    def __init__(self, a, torch, dev, out_dir, run_id):
        self.a, self.torch, self.dev, self.out_dir, self.run_id = a, torch, dev, out_dir, run_id
        self.sync = torch.mps.synchronize
        self.jsonl = open(out_dir / "microbench.jsonl", "w", encoding="utf-8")
        self.cells = []

    def time_pass(self, fn, warmup=None, repeats=None):
        warmup = self.a.warmup if warmup is None else warmup
        repeats = self.a.repeats if repeats is None else repeats
        for _ in range(warmup):
            fn()
        self.sync()
        ms = []
        for _ in range(repeats):
            self.sync()
            t0 = time.perf_counter()
            fn()
            self.sync()
            ms.append((time.perf_counter() - t0) * 1e3)
        return ms

    def memory(self):
        return {"current_allocated_bytes": int(self.torch.mps.current_allocated_memory()),
                "driver_allocated_bytes": int(self.torch.mps.driver_allocated_memory())}

    def record(self, meta: dict, fn, warmup=None, repeats=None, extra=None):
        """Time one cell, write one JSONL row per repeat and one summary cell. Exceptions become recorded cells."""
        cell = dict(meta)
        cell.update({"run_id": self.run_id, "warmup": self.a.warmup if warmup is None else warmup,
                     "sync": "torch.mps.synchronize() before and after each pass", "memory_before": self.memory()})
        try:
            ms = self.time_pass(fn, warmup, repeats)
            cell.update(stats(ms, meta["layers_per_pass"]))
            cell["raw_ms_per_pass"] = ms
            cell["status"] = "ok"
            for i, x in enumerate(ms):
                row = {k: v for k, v in meta.items()}
                row.update({"run_id": self.run_id, "repeat": i, "elapsed_ms": x, "us_per_layer": x / meta["layers_per_pass"] * 1e3,
                            "sync": cell["sync"], "exception": None})
                self.jsonl.write(json.dumps(row) + "\n")
        except Exception as e:
            cell.update({"status": "error", "error_type": type(e).__name__, "error": runinfo.sanitize(str(e))[:500]})
            self.jsonl.write(json.dumps({**meta, "run_id": self.run_id, "repeat": None, "elapsed_ms": None,
                                         "exception": f"{type(e).__name__}: {runinfo.sanitize(str(e))[:300]}"}) + "\n")
        if extra:
            cell.update(extra)
        self.jsonl.flush()
        self.cells.append(cell)
        if cell["status"] == "ok":
            print(f"  {meta['strategy']:<28s} {meta['mode']:<7s} {meta['workset']:<8s} N={meta['N0']:<6d} rho={meta['rho']:.5f} "
                  f"{meta['dtype']:<8s} median {cell['us_per_layer_median']:7.1f} us/layer  p95 {cell['us_per_layer_p95']:7.1f}", flush=True)
        else:
            print(f"  {meta['strategy']:<28s} {meta['mode']:<7s} {meta['workset']:<8s} N={meta['N0']:<6d} ERROR {cell['error_type']}: {cell['error'][:160]}", flush=True)
        return cell


def compile_function(torch, fn, example_args, sync):
    """torch.compile with graph-break accounting. Returns (compiled_fn or None, info)."""
    import torch._dynamo as dynamo
    info = {"status": "ok"}
    try:
        t0 = time.perf_counter()
        ex = dynamo.explain(fn)(*example_args)
        sync()
        info.update({"explain_s": time.perf_counter() - t0, "graph_count": ex.graph_count, "graph_break_count": ex.graph_break_count,
                     "op_count": ex.op_count, "break_reasons": [runinfo.sanitize(str(r.reason))[:200] for r in ex.break_reasons][:5]})
        dynamo.reset()
        cfn = torch.compile(fn, dynamic=False)
        t0 = time.perf_counter()
        out = cfn(*example_args)
        sync()
        info["first_call_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out2 = cfn(*example_args)
        sync()
        info["second_call_ms"] = (time.perf_counter() - t0) * 1e3
        ref = fn(*example_args)
        sync()
        info["max_abs_diff_vs_eager"] = float((out2.float() - ref.float()).abs().max().item())
        return cfn, info
    except Exception as e:
        info.update({"status": "error", "error_type": type(e).__name__, "error": runinfo.sanitize(str(e))[:600],
                     "traceback_tail": runinfo.sanitize(traceback.format_exc())[-800:]})
        return None, info


def profile_step(torch, step, sync):
    from torch.profiler import ProfilerActivity, profile
    step()
    sync()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        step()
        sync()
    ops = [{"name": e.name, "input_shapes": [list(s) for s in (e.input_shapes or [])], "cpu_time_us": float(e.cpu_time_total)}
           for e in prof.events() if e.name.startswith("aten::")]
    table = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=60)
    return ops, table


def full_qk_evidence(ops, N, d):
    """Matmul-class ops whose inputs carry both the full length N and the head dim d: the signature of a full QK/PV."""
    hits = []
    for op in ops:
        if op["name"] in MATMUL_OPS and any((N in s and d in s) for s in op["input_shapes"]):
            hits.append({"name": op["name"], "input_shapes": op["input_shapes"]})
    return hits


def main() -> int:
    a = parse_args()
    try:
        runinfo.refuse_unsafe_env()
    except RuntimeError as e:
        print(f"Refusing: {e}")
        return 2
    import torch
    if not torch.backends.mps.is_available():
        print("MPS not available; this benchmark is MPS-only (CPU results must not pass the MPS gate).")
        return 1
    dev = torch.device("mps")
    sync = torch.mps.synchronize
    run_id = "g1a-mac-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    G, H, d, r = a.groups, a.heads, a.head_dim, a.index_dim
    if H % G:
        print("heads must be a multiple of groups")
        return 2
    h = H // G
    command = "python " + " ".join(runinfo.sanitize(x) for x in sys.argv)
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "benchmark", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": None,
                "model_revision": None, "tokenizer_revision": None, "data_sha256": None, "seed": a.seed, "device": "mps",
                "host": runinfo.host_info(), "software": runinfo.torch_env(torch), "backend": "mps", "dtype": a.dtypes,
                "command": command, "timing_scope": "single layer decode read stage; per pass = layers layer-steps; sync before/after pass",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["microbench.jsonl", "summary.json", "manifest.json", "profiles.txt"],
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "limitations": ["random tensors, one layer, batch 1, T=1; no model, no task, no quality statement; MPS only"]}
    manifest["config_sha256"] = runinfo.sha256_file(ROOT / "configs/mac_gate1.json") if (ROOT / "configs/mac_gate1.json").exists() else None
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_id {run_id}  torch {torch.__version__}  recommended_max_memory {manifest['software'].get('recommended_max_memory', 0)/1e9:.1f} GB")
    t_start = time.perf_counter()
    bench = Bench(a, torch, dev, out_dir, run_id)
    summary = {"run_id": run_id, "quick": a.quick, "utc_start": runinfo.utc_now(), "args": manifest["args"], "host": manifest["host"],
               "software": manifest["software"], "git": manifest["git"], "source_hashes": manifest["source_hashes"],
               "a_backend_chosen": {}, "compile": {}, "profiles": {}, "cells": bench.cells, "memory_notes": []}
    profiles_txt = []
    model_shape = f"B1_G{G}_H{H}_d{d}_r{r}"
    rec_max = manifest["software"].get("recommended_max_memory") or 0

    for dtype_name in a.dtypes:
        dtype = getattr(torch, dtype_name)
        es = torch.tensor([], dtype=dtype).element_size()
        for N in a.lengths:
            need = LayerSet.estimate_bytes(a.layers, G, N, d, r, es)
            note = {"dtype": dtype_name, "N": N, "estimated_bytes": need, "recommended_max_memory": rec_max}
            if need > STOP_BYTES or (rec_max and need > 0.8 * rec_max):
                note["action"] = "skipped: exceeds hard limit or 80% of recommended_max_memory"
                summary["memory_notes"].append(note)
                print(f"skip N={N} {dtype_name}: {need/1e9:.1f} GB exceeds limits")
                continue
            if need > WARN_BYTES:
                note["action"] = "warning: above 80 GB soft limit"
            summary["memory_notes"].append(note)
            print(f"\n== dtype {dtype_name}  N={N}  buffers {need/1e9:.2f} GB for {a.layers} layers ==")
            ls = LayerSet(torch, dev, dtype, a.layers, G, h, d, r, N, a.seed)
            sync()
            rotating = list(range(a.layers))
            hot = [0] * a.layers
            orders = {"rotating": rotating, "hot": hot}
            base = {"dtype": dtype_name, "model_shape": model_shape, "N0": N, "valid_length": N, "query_length": 1, "query_position": N - 1,
                    "capacity": N, "layers_per_pass": a.layers, "repeats": a.repeats, "kv_bytes_per_layer": ls.kv_bytes_per_layer,
                    "index_bytes_per_layer": ls.index_bytes_per_layer, "compiled": False, "diagnostic": False}

            # --- A variants (eager, rotating): choose the fastest legitimate one as A for this (dtype, N) ---
            a_medians = {}
            for backend in A_LEGIT + A_DIAG:
                def a_step(i, backend=backend):
                    return full_attention(ls.q_main[i], ls.K[i], ls.V[i], valid_length=N, backend=backend)
                meta = {**base, "strategy": f"A_variant:{backend}", "backend": backend, "mode": "eager", "workset": "rotating", "rho": 0.0,
                        "k0": 0, "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0, "index_scanned_positions": 0,
                        "diagnostic": backend in A_DIAG}
                cell = bench.record(meta, make_pass(a_step, rotating))
                if cell["status"] == "ok" and backend in A_LEGIT:
                    a_medians[backend] = cell["us_per_layer_median"]
            if not a_medians:
                print("no legitimate A backend ran; skipping this N")
                del ls
                torch.mps.empty_cache()
                continue
            a_backend = min(a_medians, key=a_medians.get)
            summary["a_backend_chosen"][f"{dtype_name}:N{N}"] = {"backend": a_backend, "medians_us_per_layer": a_medians}
            print(f"  A backend for {dtype_name} N={N}: {a_backend}")

            def a_step(i, backend=a_backend):
                return full_attention(ls.q_main[i], ls.K[i], ls.V[i], valid_length=N, backend=backend)

            for rho in a.rhos:
                req = prepare_request(1, G, N, valid_length=N, query_position=N - 1, rho=rho, device=dev)
                lo, hi, kk, S = req.layout.candidate_lo, req.layout.candidate_hi, req.k, req.selected_count
                sel = {"rho": rho, "k0": req.k0, "k_actual": kk, "fixed_count": req.layout.fixed_count, "selected_count": S,
                       "candidate_count": req.layout.candidate_count, "index_scanned_positions": req.layout.candidate_count}
                print(f"  -- rho={rho:.5f}: k0={req.k0} k={kk} fixed={req.layout.fixed_count} selected={S} candidates={req.layout.candidate_count}")

                def d_full_step(i):
                    return sparse_decode_step_kernel(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk,
                                                     req.fixed_rows, req.off_lo, lo, hi, kk, S, k_pre_new=ls.k_pre_new[i], write_col=req.write_col)

                def d_read_step(i):
                    return sparse_decode_step_kernel(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk,
                                                     req.fixed_rows, req.off_lo, lo, hi, kk, S)

                strategies = {"A": (a_step, a_backend, {"selected_count": N, "candidate_count": 0, "index_scanned_positions": 0, "k_actual": 0, "fixed_count": 0}),
                              "D_full": (d_full_step, "sdpa", {}), "D_read": (d_read_step, "sdpa", {})}
                for mode in a.modes:
                    steps = {}
                    if mode == "eager":
                        steps = {name: fn for name, (fn, _, _) in strategies.items()}
                        cinfo = {}
                    else:
                        cinfo = {}
                        a_fn = (lambda q, k, v, backend=a_backend: full_attention(q, k, v, valid_length=N, backend=backend))
                        ca, ia = compile_function(torch, a_fn, (ls.q_main[0], ls.K[0], ls.V[0]), sync)
                        cinfo["A"] = ia
                        d_fn = (lambda qm, qp, k, v, zT, wq, wk, knew: sparse_decode_step_kernel(qm, qp, k, v, zT, wq, wk, req.fixed_rows, req.off_lo,
                                                                                               lo, hi, kk, S, k_pre_new=knew, write_col=req.write_col))
                        cd, idf = compile_function(torch, d_fn, (ls.q_main[0], ls.q_pre[0], ls.K[0], ls.V[0], ls.zT[0], ls.idx[0].wq_t_scaled, ls.idx[0].wk, ls.k_pre_new[0]), sync)
                        cinfo["D_full"] = idf
                        dr_fn = (lambda qm, qp, k, v, zT, wq, wk: sparse_decode_step_kernel(qm, qp, k, v, zT, wq, wk, req.fixed_rows, req.off_lo, lo, hi, kk, S))
                        cdr, idr = compile_function(torch, dr_fn, (ls.q_main[0], ls.q_pre[0], ls.K[0], ls.V[0], ls.zT[0], ls.idx[0].wq_t_scaled, ls.idx[0].wk), sync)
                        cinfo["D_read"] = idr
                        summary["compile"][f"{dtype_name}:N{N}:rho{rho:.5f}"] = cinfo
                        for name, info in cinfo.items():
                            print(f"  compile {name}: {info.get('status')} graphs={info.get('graph_count')} breaks={info.get('graph_break_count')} "
                                  f"first_call={info.get('first_call_s', float('nan')):.1f}s max_abs_diff={info.get('max_abs_diff_vs_eager')}"
                                  + (f" ERROR {info.get('error_type')}: {info.get('error', '')[:200]}" if info.get("status") != "ok" else ""))
                        if ca is not None:
                            steps["A"] = lambda i, ca=ca: ca(ls.q_main[i], ls.K[i], ls.V[i])
                        if cd is not None:
                            steps["D_full"] = lambda i, cd=cd: cd(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk, ls.k_pre_new[i])
                        if cdr is not None:
                            steps["D_read"] = lambda i, cdr=cdr: cdr(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk)
                    for workset in a.worksets:
                        for name, (_, backend, override) in strategies.items():
                            meta = {**base, **sel, **override, "strategy": name, "backend": backend, "mode": mode, "workset": workset, "compiled": mode == "compile"}
                            if name not in steps:
                                meta["recompile_error"] = None
                                bench.cells.append({**meta, "run_id": run_id, "status": "error", "error_type": "CompileFailed",
                                                    "error": cinfo.get(name, {}).get("error", "compile failed")})
                                continue
                            extra = {}
                            if mode == "compile":
                                import torch._dynamo as dynamo
                                dynamo.config.error_on_recompile = True
                                try:
                                    bench.record(meta, make_pass(steps[name], orders[workset]), extra={"recompile_error": None, "compile_info": cinfo.get(name)})
                                finally:
                                    dynamo.config.error_on_recompile = False
                            else:
                                bench.record(meta, make_pass(steps[name], orders[workset]), extra=extra)
                    if mode == "compile":
                        import torch._dynamo as dynamo
                        dynamo.reset()

                if not a.skip_diagnostics:
                    # components of D (eager, rotating), each timed alone from layer-0 intermediates of identical shape
                    u0 = project_queries(ls.q_pre[0], ls.idx[0].wq_t_scaled)
                    s0 = candidate_scores(u0, ls.zT[0], lo, hi)
                    rows0 = select_rows_kernel(u0, ls.zT[0], req.fixed_rows, req.off_lo, lo, hi, kk)
                    top0 = torch.topk(s0, max(kk, 1), dim=-1).indices.view(1, G, max(kk, 1))
                    ksub0 = flatten_cache(ls.K[0]).index_select(0, rows0).view(1, G, S, d)
                    vsub0 = flatten_cache(ls.V[0]).index_select(0, rows0).view(1, G, S, d)
                    comps = {
                        "comp:z_append": lambda i: ls.zT[i][..., N - 1:N].copy_(project_keys(ls.k_pre_new[i], ls.idx[i].wk)),
                        "comp:u_proj": lambda i: project_queries(ls.q_pre[i], ls.idx[i].wq_t_scaled),
                        "comp:score_candidates": lambda i: candidate_scores(u0, ls.zT[i], lo, hi),
                        "comp:topk": lambda i: torch.topk(s0, kk, dim=-1),
                        "comp:merge_rows": lambda i: torch.cat([req.fixed_rows, top0 + req.off_lo], dim=-1).view(-1),
                        "comp:gather_k": lambda i: flatten_cache(ls.K[i]).index_select(0, rows0),
                        "comp:gather_v": lambda i: flatten_cache(ls.V[i]).index_select(0, rows0),
                        "comp:subset_attention": lambda i: subset_attention(ls.q_main[i], ksub0, vsub0),
                    }
                    for name, fn in comps.items():
                        meta = {**base, **sel, "strategy": name, "backend": "eager-op", "mode": "eager", "workset": "rotating", "diagnostic": True}
                        bench.record(meta, make_pass(fn, rotating))
                    # gather sensitivity: same S rows per layer, different list structure (diagnostic, not a D result)
                    gen = torch.Generator().manual_seed(a.seed + 1)
                    M = req.layout.candidate_count
                    lists = {"topk_actual": rows0}
                    if kk > 0:
                        uni = torch.stack([torch.randperm(M, generator=gen)[:kk] for _ in range(G)]).view(1, G, kk)
                        start = int(torch.randint(0, M - kk + 1, (1,), generator=gen))
                        clu = torch.arange(start, start + kk).view(1, 1, kk).expand(1, G, kk)
                        for name, cand in (("uniform", uni), ("uniform_sorted", uni.sort(dim=-1).values), ("clustered", clu)):
                            lists[name] = torch.cat([req.fixed_rows, cand.to(dev) + req.off_lo], dim=-1).view(-1)
                    for name, rows in lists.items():
                        def g_step(i, rows=rows):
                            ks = flatten_cache(ls.K[i]).index_select(0, rows).view(1, G, S, d)
                            vs = flatten_cache(ls.V[i]).index_select(0, rows).view(1, G, S, d)
                            return subset_attention(ls.q_main[i], ks, vs)
                        meta = {**base, **sel, "strategy": f"gather_sens:{name}", "backend": "index_select+sdpa", "mode": "eager", "workset": "rotating", "diagnostic": True}
                        bench.record(meta, make_pass(g_step, rotating))

            # --- profile one A step and one D_full step (eager): op list with shapes; D must show no full QK ---
            if not a.skip_profile:
                req = prepare_request(1, G, N, valid_length=N, query_position=N - 1, rho=a.rhos[0], device=dev)
                lo, hi, kk, S = req.layout.candidate_lo, req.layout.candidate_hi, req.k, req.selected_count
                prof = {}
                try:
                    ops_a, tab_a = profile_step(torch, lambda: a_step(0), sync)
                    ops_d, tab_d = profile_step(torch, lambda: sparse_decode_step_kernel(ls.q_main[0], ls.q_pre[0], ls.K[0], ls.V[0], ls.zT[0], ls.idx[0].wq_t_scaled, ls.idx[0].wk,
                                                                                       req.fixed_rows, req.off_lo, lo, hi, kk, S, k_pre_new=ls.k_pre_new[0], write_col=req.write_col), sync)
                    prof = {"rho": a.rhos[0], "A_backend": a_backend, "A_ops": ops_a, "D_full_ops": ops_d,
                            "A_full_qk_ops": full_qk_evidence(ops_a, N, d), "D_full_qk_ops": full_qk_evidence(ops_d, N, d),
                            "D_ops_touching_N": [o for o in ops_d if any(N in s for s in o["input_shapes"])],
                            "A_sdpa_dispatch": sorted({o["name"] for o in ops_a if "scaled_dot_product" in o["name"] or "sdp" in o["name"]}),
                            "D_sdpa_dispatch": sorted({o["name"] for o in ops_d if "scaled_dot_product" in o["name"] or "sdp" in o["name"]})}
                    prof["d_full_qk_detected"] = bool(prof["D_full_qk_ops"])
                    prof["a_full_qk_detected"] = bool(prof["A_full_qk_ops"])
                    profiles_txt.append(f"=== {dtype_name} N={N} A[{a_backend}] ===\n{tab_a}\n=== {dtype_name} N={N} D_full rho={a.rhos[0]} ===\n{tab_d}\n")
                    print(f"  profile: A full-QK ops {len(prof['A_full_qk_ops'])}, D full-QK ops {len(prof['D_full_qk_ops'])} -> d_full_qk_detected={prof['d_full_qk_detected']}; "
                          f"A sdpa dispatch {prof['A_sdpa_dispatch']}")
                except Exception as e:
                    prof = {"status": "error", "error_type": type(e).__name__, "error": runinfo.sanitize(str(e))[:400]}
                    print(f"  profile ERROR {type(e).__name__}: {e}")
                summary["profiles"][f"{dtype_name}:N{N}"] = prof

            if not a.skip_diagnostics:
                # capacity slack: A on a strided view k[:, :, :N] of buffers allocated with capacity N+2048 (static-cache situation)
                slack = 2048
                need2 = LayerSet.estimate_bytes(a.layers, G, N + slack, d, r, es)
                if need2 + need < STOP_BYTES and (not rec_max or need2 + need < 0.8 * rec_max):
                    ls2 = LayerSet(torch, dev, dtype, a.layers, G, h, d, r, N, a.seed + 7, capacity=N + slack)
                    sync()
                    def slack_step(i):
                        return full_attention(ls2.q_main[i], ls2.K[i], ls2.V[i], valid_length=N, backend=a_backend)
                    meta = {**base, "capacity": N + slack, "strategy": "A_capacity_slack", "backend": a_backend, "mode": "eager", "workset": "rotating",
                            "rho": 0.0, "k0": 0, "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0, "index_scanned_positions": 0, "diagnostic": True}
                    bench.record(meta, make_pass(slack_step, rotating))
                    del ls2
                    torch.mps.empty_cache()
                # heads sensitivity: H=16 (h=2) geometry of the smaller models, eager rotating, first rho
                if H != 16 and G <= 16:
                    h2 = 16 // G
                    q2 = [torch.randn(1, G, h2, 1, d, device=dev, dtype=dtype) for _ in range(a.layers)]
                    qp2 = [torch.randn(1, G, h2, 1, d, device=dev, dtype=dtype) for _ in range(a.layers)]
                    idx2 = [GroupedIndexer(G, h2, d, r, device=dev, dtype=dtype, seed=a.seed * 1000 + 500 + i) for i in range(a.layers)]
                    req = prepare_request(1, G, N, valid_length=N, query_position=N - 1, rho=a.rhos[0], device=dev)
                    lo, hi, kk, S = req.layout.candidate_lo, req.layout.candidate_hi, req.k, req.selected_count
                    sel = {"rho": a.rhos[0], "k0": req.k0, "k_actual": kk, "fixed_count": req.layout.fixed_count, "selected_count": S,
                           "candidate_count": req.layout.candidate_count, "index_scanned_positions": req.layout.candidate_count}
                    shape16 = f"B1_G{G}_H16_d{d}_r{r}"
                    bench.record({**base, "model_shape": shape16, "strategy": "A", "backend": a_backend, "mode": "eager", "workset": "rotating", "rho": 0.0, "k0": 0,
                                  "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0, "index_scanned_positions": 0, "diagnostic": True},
                                 make_pass(lambda i: full_attention(q2[i], ls.K[i], ls.V[i], valid_length=N, backend=a_backend), rotating))
                    bench.record({**base, **sel, "model_shape": shape16, "strategy": "D_full", "backend": "sdpa", "mode": "eager", "workset": "rotating", "diagnostic": True},
                                 make_pass(lambda i: sparse_decode_step_kernel(q2[i], qp2[i], ls.K[i], ls.V[i], ls.zT[i], idx2[i].wq_t_scaled, idx2[i].wk, req.fixed_rows,
                                                                               req.off_lo, lo, hi, kk, S, k_pre_new=ls.k_pre_new[i], write_col=req.write_col), rotating))
                    del q2, qp2, idx2
            summary.setdefault("memory_peaks", []).append({"dtype": dtype_name, "N": N, **bench.memory()})
            del ls
            torch.mps.empty_cache()

    summary["utc_end"] = runinfo.utc_now()
    summary["device_seconds"] = time.perf_counter() - t_start
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / "profiles.txt").write_text("\n".join(profiles_txt), encoding="utf-8")
    manifest.update({"status": "done", "finished_at_utc": summary["utc_end"], "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": summary["device_seconds"]}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bench.jsonl.close()
    # console summary of the main comparison
    print("\n== main comparison (us per layer, median; rotating / hot) ==")
    main_cells = [c for c in bench.cells if c.get("status") == "ok" and not c.get("diagnostic") and c["strategy"] in ("A", "D_full", "D_read")]
    keys = sorted({(c["dtype"], c["mode"], c["N0"], c["rho"]) for c in main_cells if c["strategy"] != "A"})
    for dtype_name, mode, N, rho in keys:
        def med(strategy, workset):
            for c in main_cells:
                if (c["dtype"], c["mode"], c["N0"], c["workset"], c["strategy"]) == (dtype_name, mode, N, workset, strategy) and (strategy == "A" or c["rho"] == rho):
                    return c["us_per_layer_median"]
            return float("nan")
        ar, dr, drr = med("A", "rotating"), med("D_full", "rotating"), med("D_read", "rotating")
        ah, dh = med("A", "hot"), med("D_full", "hot")
        print(f"  {dtype_name:<8s} {mode:<7s} N={N:<6d} rho={rho:.5f}  A {ar:7.1f}  D_full {dr:7.1f} ({dr/ar:5.2f}x)  D_read {drr:7.1f} ({drr/ar:5.2f}x)   hot: A {ah:7.1f} D_full {dh:7.1f}")
    print(f"\nwrote {runinfo.sanitize(str(out_dir))}  ({summary['device_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
