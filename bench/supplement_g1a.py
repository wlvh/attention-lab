#!/usr/bin/env python3
"""G1a supplement after the PR #1 review. Three bounded questions, MPS only, random tensors, same conventions as
bench/microbench.py (Bench/LayerSet/JSONL are imported from it, not re-implemented).

1. Order. The formal runs timed A, then D_full, then D_read, with one data seed. Here every timed cycle interleaves
   single passes as A, D_full, D_read, D_read, D_full, A (and the reversed sequence) on the same LayerSet, so each
   cycle yields a paired D/A ratio; the formal sequential order is also timed in the same process for comparison.
2. Capacity. 2x2 of {A, D_full} x {capacity = N, capacity = N + 2048} on IDENTICAL K/V/index values, both buffer sets
   resident for the whole measurement, interleaved. Tight and slack outputs are checked for equality.
3. A control. A hand-written folded QK -> softmax -> PV on the same folded layout, timed against sdpa_fold and
   sdpa_gqa, with its numerical difference to dense_reference recorded.

Numbers are measurements of this host; not model, task or CUDA results.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from attention_lab import runinfo  # noqa: E402
from attention_lab.decode import prepare_request, sparse_decode_step_kernel  # noqa: E402
from attention_lab.full import full_attention  # noqa: E402
from attention_lab.reference import default_scale, dense_reference  # noqa: E402
from microbench import Bench, LayerSet, make_pass, percentile, stats  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"], choices=["bfloat16", "float32"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[32768])
    ap.add_argument("--rhos", type=float, nargs="+", default=[1 / 32, 1 / 8])
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--index-dim", type=int, default=32)
    ap.add_argument("--a-backend", default="sdpa_fold", choices=["sdpa_fold", "sdpa_gqa"])
    ap.add_argument("--warmup", type=int, default=10, help="warm-up passes per step before each block")
    ap.add_argument("--cycles", type=int, default=40, help="interleaved cycles per block")
    ap.add_argument("--slack", type=int, default=2048)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.warmup, a.cycles = 2, 4
    return a


def timed_pass(sync, fn):
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - t0) * 1e3


def record_samples(bench, meta, ms, extra=None):
    """Store an already-timed sample list with the same cell/JSONL layout Bench.record uses."""
    cell = dict(meta)
    cell.update({"run_id": bench.run_id, "warmup": bench.a.warmup, "sync": "torch.mps.synchronize() before and after each pass",
                 "memory_before": bench.memory(), "status": "ok", "raw_ms_per_pass": list(ms)})
    cell.update(stats(ms, meta["layers_per_pass"]))
    for i, x in enumerate(ms):
        row = dict(meta)
        row.update({"run_id": bench.run_id, "repeat": i, "elapsed_ms": x, "us_per_layer": x / meta["layers_per_pass"] * 1e3,
                    "sync": cell["sync"], "exception": None})
        bench.jsonl.write(json.dumps(row) + "\n")
    if extra:
        cell.update(extra)
    bench.jsonl.flush()
    bench.cells.append(cell)
    print(f"  {meta['strategy']:<22s} {meta.get('block', ''):<12s} N={meta['N0']:<6d} rho={meta['rho']:.5f} {meta['dtype']:<8s} "
          f"median {cell['us_per_layer_median']:7.1f} us/layer  p95 {cell['us_per_layer_p95']:7.1f}  n={len(ms)}", flush=True)
    return cell


def interleaved_block(bench, sync, steps: dict, sequence: list[str], order: list[int], cycles: int, warmup: int):
    """Run `cycles` cycles of the sequence; return samples per strategy and the per-cycle paired ratios D/A."""
    passes = {name: make_pass(fn, order) for name, fn in steps.items()}
    for name in sequence:
        for _ in range(warmup):
            passes[name]()
    sync()
    samples = {name: [] for name in steps}
    ratios = {name: [] for name in steps if name != "A"}
    for _ in range(cycles):
        cyc = {name: [] for name in steps}
        for name in sequence:
            ms = timed_pass(sync, passes[name])
            samples[name].append(ms)
            cyc[name].append(ms)
        a_mean = statistics.fmean(cyc["A"]) if cyc.get("A") else None
        if a_mean:
            for name in ratios:
                if cyc[name]:
                    ratios[name].append(statistics.fmean(cyc[name]) / a_mean)
    return samples, ratios


def main() -> int:
    a = parse_args()
    try:
        runinfo.refuse_unsafe_env()
    except RuntimeError as e:
        print(f"Refusing: {e}")
        return 2
    import torch
    if not torch.backends.mps.is_available():
        print("MPS not available; this supplement is MPS-only.")
        return 1
    dev = torch.device("mps")
    sync = torch.mps.synchronize
    run_id = "g1a-supp-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    G, H, d, r = a.groups, a.heads, a.head_dim, a.index_dim
    h = H // G
    command = "python " + " ".join(runinfo.sanitize(x) for x in sys.argv)
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "benchmark_supplement", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": None,
                "model_revision": None, "tokenizer_revision": None, "data_sha256": None, "seed": a.seed, "device": "mps",
                "host": runinfo.host_info(), "software": runinfo.torch_env(torch), "backend": "mps", "dtype": a.dtypes,
                "command": command, "timing_scope": "single layer decode read stage; per pass = layers layer-steps; interleaved single passes; sync before/after pass",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["microbench.jsonl", "summary.json", "manifest.json"],
                "args": {k: (runinfo.sanitize(str(v)) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "limitations": ["random tensors, one layer, batch 1, T=1; no model, no task, no quality statement; MPS only"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_id {run_id}  torch {torch.__version__}")
    t_start = time.perf_counter()
    bench = Bench(SimpleNamespace(warmup=a.warmup, repeats=a.cycles), torch, dev, out_dir, run_id)
    summary = {"run_id": run_id, "quick": a.quick, "kind": "g1a_supplement", "utc_start": runinfo.utc_now(), "args": manifest["args"],
               "host": manifest["host"], "software": manifest["software"], "git": manifest["git"], "source_hashes": manifest["source_hashes"],
               "cells": bench.cells, "paired_ratios": {}, "equality_checks": {}, "memory_notes": []}
    model_shape = f"B1_G{G}_H{H}_d{d}_r{r}"

    for dtype_name in a.dtypes:
        dtype = getattr(torch, dtype_name)
        for N in a.lengths:
            print(f"\n== {dtype_name} N={N} ==")
            ls = LayerSet(torch, dev, dtype, a.layers, G, h, d, r, N, a.seed)
            sync()
            order = list(range(a.layers))
            base = {"dtype": dtype_name, "model_shape": model_shape, "N0": N, "valid_length": N, "query_length": 1, "query_position": N - 1,
                    "capacity": N, "layers_per_pass": a.layers, "repeats": a.cycles, "kv_bytes_per_layer": ls.kv_bytes_per_layer,
                    "index_bytes_per_layer": ls.index_bytes_per_layer, "compiled": False, "diagnostic": True, "mode": "eager", "workset": "rotating"}

            def a_step(i, backend=a.a_backend):
                return full_attention(ls.q_main[i], ls.K[i], ls.V[i], valid_length=N, backend=backend)

            # ---------- 1. order ----------
            for rho in a.rhos:
                req = prepare_request(1, G, N, valid_length=N, query_position=N - 1, rho=rho, device=dev)
                lo, hi, kk, S = req.layout.candidate_lo, req.layout.candidate_hi, req.k, req.selected_count
                sel = {"rho": rho, "k0": req.k0, "k_actual": kk, "fixed_count": req.layout.fixed_count, "selected_count": S,
                       "candidate_count": req.layout.candidate_count, "index_scanned_positions": req.layout.candidate_count}

                def d_full(i):
                    return sparse_decode_step_kernel(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk,
                                                     req.fixed_rows, req.off_lo, lo, hi, kk, S, k_pre_new=ls.k_pre_new[i], write_col=req.write_col)

                def d_read(i):
                    return sparse_decode_step_kernel(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk,
                                                     req.fixed_rows, req.off_lo, lo, hi, kk, S)
                steps = {"A": a_step, "D_full": d_full, "D_read": d_read}
                for block, seq in (("ABBA", ["A", "D_full", "D_read", "D_read", "D_full", "A"]),
                                   ("BAAB", ["D_read", "D_full", "A", "A", "D_full", "D_read"])):
                    samples, ratios = interleaved_block(bench, sync, steps, seq, order, a.cycles, a.warmup)
                    for name, ms in samples.items():
                        meta = {**base, **({"rho": 0.0, "k0": 0, "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0,
                                            "index_scanned_positions": 0} if name == "A" else sel),
                                "strategy": f"order:{name}", "block": block, "backend": a.a_backend if name == "A" else "sdpa"}
                        if name == "A":
                            meta["rho_block"] = rho
                        record_samples(bench, meta, ms)
                    for name, rs in ratios.items():
                        key = f"{dtype_name}:N{N}:rho{rho:.5f}:{block}:{name}"
                        summary["paired_ratios"][key] = {"n_cycles": len(rs), "median": statistics.median(rs), "min": min(rs), "max": max(rs),
                                                         "p95": percentile(rs, 0.95), "all_below_0.90": all(x <= 0.90 for x in rs)}
                        print(f"    paired {name}/A [{block}] median {statistics.median(rs):.3f}  range {min(rs):.3f}..{max(rs):.3f}  n={len(rs)}")
                # the formal sequential order, same process, for comparison
                for name, fn in steps.items():
                    ms = bench.time_pass(make_pass(fn, order), warmup=a.warmup, repeats=a.cycles * 2)
                    meta = {**base, **({"rho": 0.0, "k0": 0, "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0,
                                        "index_scanned_positions": 0} if name == "A" else sel),
                            "strategy": f"order_seq:{name}", "block": "sequential", "backend": a.a_backend if name == "A" else "sdpa"}
                    if name == "A":
                        meta["rho_block"] = rho
                    record_samples(bench, meta, ms)

            # ---------- 2. capacity 2x2 ----------
            rho = a.rhos[0]
            C2 = N + a.slack
            K_s = [torch.zeros(1, G, C2, d, device=dev, dtype=dtype) for _ in range(a.layers)]
            V_s = [torch.zeros(1, G, C2, d, device=dev, dtype=dtype) for _ in range(a.layers)]
            zT_s = [torch.zeros(1, G, r, C2, device=dev, dtype=dtype) for _ in range(a.layers)]
            for i in range(a.layers):
                K_s[i][:, :, :N].copy_(ls.K[i]); V_s[i][:, :, :N].copy_(ls.V[i]); zT_s[i][..., :N].copy_(ls.zT[i])
            sync()
            req_t = prepare_request(1, G, N, valid_length=N, query_position=N - 1, rho=rho, device=dev)
            req_s = prepare_request(1, G, C2, valid_length=N, query_position=N - 1, rho=rho, device=dev)
            lo, hi, kk, S = req_t.layout.candidate_lo, req_t.layout.candidate_hi, req_t.k, req_t.selected_count
            sel = {"rho": rho, "k0": req_t.k0, "k_actual": kk, "fixed_count": req_t.layout.fixed_count, "selected_count": S,
                   "candidate_count": req_t.layout.candidate_count, "index_scanned_positions": req_t.layout.candidate_count}

            def a_tight(i):
                return full_attention(ls.q_main[i], ls.K[i], ls.V[i], valid_length=N, backend=a.a_backend)

            def a_slack(i):
                return full_attention(ls.q_main[i], K_s[i], V_s[i], valid_length=N, backend=a.a_backend)

            def d_tight(i):
                return sparse_decode_step_kernel(ls.q_main[i], ls.q_pre[i], ls.K[i], ls.V[i], ls.zT[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk,
                                                 req_t.fixed_rows, req_t.off_lo, lo, hi, kk, S, k_pre_new=ls.k_pre_new[i], write_col=req_t.write_col)

            def d_slack(i):
                return sparse_decode_step_kernel(ls.q_main[i], ls.q_pre[i], K_s[i], V_s[i], zT_s[i], ls.idx[i].wq_t_scaled, ls.idx[i].wk,
                                                 req_s.fixed_rows, req_s.off_lo, lo, hi, kk, S, k_pre_new=ls.k_pre_new[i], write_col=req_s.write_col)
            eq = {}
            with torch.no_grad():
                eq["A_tight_vs_slack_max_abs"] = float((a_tight(0).float() - a_slack(0).float()).abs().max())
                eq["D_tight_vs_slack_max_abs"] = float((d_tight(0).float() - d_slack(0).float()).abs().max())
            sync()
            summary["equality_checks"][f"{dtype_name}:N{N}:capacity"] = eq
            print(f"    capacity equality: A {eq['A_tight_vs_slack_max_abs']:.2e}  D {eq['D_tight_vs_slack_max_abs']:.2e}")
            steps = {"A_tight": a_tight, "A_slack": a_slack, "D_tight": d_tight, "D_slack": d_slack}
            seq = ["A_tight", "A_slack", "D_tight", "D_slack", "D_slack", "D_tight", "A_slack", "A_tight"]
            passes = {name: make_pass(fn, order) for name, fn in steps.items()}
            for name in seq:
                for _ in range(a.warmup):
                    passes[name]()
            sync()
            samples = {name: [] for name in steps}
            for _ in range(a.cycles):
                for name in seq:
                    samples[name].append(timed_pass(sync, passes[name]))
            for name, ms in samples.items():
                cap = N if name.endswith("tight") else C2
                meta = {**base, **({"rho": 0.0, "k0": 0, "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0,
                                    "index_scanned_positions": 0} if name.startswith("A") else sel),
                        "capacity": cap, "strategy": f"cap:{name}", "block": "interleaved", "backend": a.a_backend if name.startswith("A") else "sdpa"}
                record_samples(bench, meta, ms)
            del K_s, V_s, zT_s
            torch.mps.empty_cache()

            # ---------- 3. folded explicit A control ----------
            scale = default_scale(d, None)

            def explicit_fold(i):
                qf = ls.q_main[i].reshape(1, G, h, d)
                scores = torch.matmul(qf, ls.K[i].transpose(-1, -2)) * scale        # [1,G,h,N]
                probs = torch.softmax(scores, dim=-1)
                return torch.matmul(probs, ls.V[i]).view(1, G, h, 1, d)

            def fold(i):
                return full_attention(ls.q_main[i], ls.K[i], ls.V[i], valid_length=N, backend="sdpa_fold")

            def gqa(i):
                return full_attention(ls.q_main[i], ls.K[i], ls.V[i], valid_length=N, backend="sdpa_gqa")
            with torch.no_grad():
                ref = dense_reference(ls.q_main[0].float(), ls.K[0].float(), ls.V[0].float(), valid_length=N)
                eqa = {"explicit_fold_vs_dense_reference_fp32_max_abs": float((explicit_fold(0).float() - ref).abs().max()),
                       "sdpa_fold_vs_dense_reference_fp32_max_abs": float((fold(0).float() - ref).abs().max()),
                       "sdpa_gqa_vs_dense_reference_fp32_max_abs": float((gqa(0).float() - ref).abs().max())}
            sync()
            summary["equality_checks"][f"{dtype_name}:N{N}:a_control"] = eqa
            steps = {"sdpa_fold": fold, "explicit_fold": explicit_fold, "sdpa_gqa": gqa}
            seq = ["sdpa_fold", "explicit_fold", "sdpa_gqa", "sdpa_gqa", "explicit_fold", "sdpa_fold"]
            passes = {name: make_pass(fn, order) for name, fn in steps.items()}
            for name in seq:
                for _ in range(a.warmup):
                    passes[name]()
            sync()
            samples = {name: [] for name in steps}
            for _ in range(a.cycles):
                for name in seq:
                    samples[name].append(timed_pass(sync, passes[name]))
            for name, ms in samples.items():
                meta = {**base, "rho": 0.0, "k0": 0, "k_actual": 0, "fixed_count": 0, "selected_count": N, "candidate_count": 0,
                        "index_scanned_positions": 0, "strategy": f"actrl:{name}", "block": "interleaved", "backend": name}
                record_samples(bench, meta, ms)
            summary["memory_notes"].append({"dtype": dtype_name, "N": N, **bench.memory(), "note": "snapshot after this (dtype, N), not a peak"})
            del ls
            torch.mps.empty_cache()

    summary["utc_end"] = runinfo.utc_now()
    summary["device_seconds"] = time.perf_counter() - t_start
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest.update({"status": "done", "finished_at_utc": summary["utc_end"], "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": summary["device_seconds"]}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bench.jsonl.close()
    print(f"\nwrote {runinfo.sanitize(str(out_dir))}  ({summary['device_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
