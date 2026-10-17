#!/usr/bin/env python3
"""P0 spike: the D chain's scoring segment as one Metal kernel vs the PyTorch path, random tensors on this Mac's MPS.

Per layer and decode step the segment is: z of the newest key written into its index-table column, the index query u,
and the scores of every candidate column. Strategies (all bf16 inputs, batch 1):
  pytorch_chain        project_keys + strided column copy_ + project_queries + candidate_scores (3 bmm + 1 copy)
  pytorch_score_only   candidate_scores alone from a precomputed u (the read-stage lower bound of the PyTorch path)
  fused_1launch        MetalScore.fused: one kernel launch does all three
  fused_2launch        MetalScore.two_launch: prologue kernel (u, z column) + scan kernel
Rotating workset of `layers` independent index tables; warm-up then timed passes with torch.mps.synchronize() before
and after each pass; every strategy is timed twice (forward and reverse order) to expose drift. Numerics per length:
fused fp32 scores and the PyTorch bf16 scores against a CPU float32 reference, the z column against project_keys,
and the top-k set agreement between the two score sources. Pure MPS; refuses CPU fallback / disabled memory guard.
Stop gate (docs/tasks/04_FUSED_D_KERNEL.md) is evaluated at the end and written to summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_lab import runinfo  # noqa: E402

GATE = {"fused_us_max_32k": 80.0, "ratio_vs_chain_max": 0.75, "scores_atol": 1e-5, "scores_rel_norm": 1e-5, "z_rel_ulp": 2.0 ** -8}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32256], help="prompt lengths N0 (legal length N0+1, capacity N0+128)")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--heads-per-group", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--index-dim", type=int, default=32)
    ap.add_argument("--rho", type=float, default=1 / 32)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="pipeline smoke: warm-up 3, repeats 5, run_id marked quick")
    a = ap.parse_args()
    if a.quick:
        a.warmup, a.repeats = 3, 5
    return a


def percentile(xs, p):
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def main() -> int:
    a = parse_args()
    try:
        runinfo.refuse_unsafe_env()
    except RuntimeError as e:
        print(f"Refusing: {e}")
        return 2
    import torch
    from attention_lab import metal_score
    from attention_lab.decode import prepare_request
    from attention_lab.indexer import GroupedIndexer, candidate_scores, project_keys, project_queries
    if not metal_score.available():
        print("needs MPS and torch.mps.compile_shader")
        return 1
    dev, sync = torch.device("mps"), torch.mps.synchronize
    G, h, d, r = a.groups, a.heads_per_group, a.head_dim, a.index_dim
    run_id = "p0-score-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "benchmark", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": None, "model_revision": None,
                "tokenizer_revision": None, "data_sha256": None, "seed": a.seed, "device": "mps", "host": runinfo.host_info(),
                "software": runinfo.torch_env(torch), "backend": "mps+metal(compile_shader)", "dtype": "bfloat16",
                "command": "python " + " ".join(runinfo.sanitize(x) for x in sys.argv),
                "timing_scope": "single layer scoring segment per decode step; pass = layers layer-steps; sync before/after pass",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["p0.jsonl", "summary.json", "manifest.json"],
                "args": {k: (runinfo.sanitize(str(v)) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "limitations": ["random tensors, one layer segment, batch 1; no model, no task; MPS only; P0 of docs/tasks/04_FUSED_D_KERNEL.md"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_id {run_id}  torch {torch.__version__}  layers {a.layers}  warmup {a.warmup} repeats {a.repeats}")
    t_start = time.perf_counter()
    ms_kernel = metal_score.MetalScore()
    jsonl = open(out_dir / "p0.jsonl", "w", encoding="utf-8")
    summary = {"run_id": run_id, "quick": a.quick, "utc_start": runinfo.utc_now(), "args": manifest["args"], "host": manifest["host"],
               "software": manifest["software"], "git": manifest["git"], "source_hashes": manifest["source_hashes"], "gate": GATE, "lengths": []}

    def time_pass(fn, order):
        for _ in range(a.warmup):
            for i in order:
                fn(i)
        sync()
        ms = []
        for _ in range(a.repeats):
            sync()
            t0 = time.perf_counter()
            for i in order:
                fn(i)
            sync()
            ms.append((time.perf_counter() - t0) * 1e3)
        return ms

    with torch.inference_mode():
        for N0 in a.lengths:
            C, L = N0 + 128, N0 + 1
            req = prepare_request(1, G, C, valid_length=L, query_position=L - 1, rho=a.rho, device=dev)
            lo, hi, wc, k = req.layout.candidate_lo, req.layout.candidate_hi, req.write_col, req.k
            M = hi - lo
            torch.manual_seed(a.seed + N0)
            zT = [torch.randn(1, G, r, C, device=dev, dtype=torch.bfloat16) for _ in range(a.layers)]
            knew = [torch.randn(1, G, 1, d, device=dev, dtype=torch.bfloat16) for _ in range(a.layers)]
            qpre = [torch.randn(1, G, h, 1, d, device=dev, dtype=torch.bfloat16) for _ in range(a.layers)]
            idx = [GroupedIndexer(G, h, d, r, device=dev, dtype=torch.bfloat16, seed=a.seed * 1000 + i) for i in range(a.layers)]
            u0 = [project_queries(qpre[i], idx[i].wq_t_scaled) for i in range(a.layers)]
            views = [metal_score.model_views(zT[i], knew[i], qpre[i], idx[i]) for i in range(a.layers)]
            sync()
            bytes_scan = G * r * M * 2
            block = {"N0": N0, "capacity": C, "legal_length": L, "lo": lo, "hi": hi, "M": M, "k": k, "write_col": wc, "layers": a.layers,
                     "scan_bytes_per_layer": bytes_scan, "workset_bytes": a.layers * G * r * C * 2, "cells": {}, "numerics": {}}
            print(f"\n== N0={N0} C={C} L={L} candidates [{lo},{hi}) M={M} k={k}  scan {bytes_scan/1e6:.1f} MB/layer  workset {block['workset_bytes']/1e9:.2f} GB ==")

            # ---- numerics (layer 0), before timing ----
            z0, kn0, wk0, qc0, wq0 = views[0]
            u_ref, z_ref, s_ref = metal_score.reference_scores(z0, kn0, wk0, qc0, wq0, lo, hi)
            zc = zT[0].clone()
            s_fused = ms_kernel.fused(zc[0], kn0, wk0, qc0, wq0, lo, hi, wc)
            s_two = ms_kernel.two_launch(zT[0].clone()[0], kn0, wk0, qc0, wq0, lo, hi, wc)
            zp = zT[0].clone()
            zp[..., wc:wc + 1].copy_(project_keys(knew[0], idx[0].wk))
            s_pt = candidate_scores(project_queries(qpre[0], idx[0].wq_t_scaled), zp, lo, hi)[0, :, 0, :]
            sync()
            z_pt = zp[0, :, :, wc].float().cpu()
            z_fu = zc[0, :, :, wc].float().cpu()
            num = {"fused_vs_ref_max_abs": float((s_fused.cpu() - s_ref).abs().max()), "fused_vs_ref_rel_norm": float((s_fused.cpu() - s_ref).norm() / s_ref.norm()),
                   "two_launch_vs_fused_max_abs": float((s_two.cpu() - s_fused.cpu()).abs().max()),
                   "pytorch_bf16_vs_ref_max_abs": float((s_pt.float().cpu() - s_ref).abs().max()), "pytorch_bf16_vs_ref_rel_norm": float((s_pt.float().cpu() - s_ref).norm() / s_ref.norm()),
                   "z_fused_vs_pytorch_max_rel_ulp": float(((z_fu - z_pt).abs() / (z_pt.abs() + 1e-12)).max()), "z_fused_bitwise_fraction": float((z_fu == z_pt).float().mean()),
                   "z_fused_vs_ref_max_abs": float((z_fu - z_ref).abs().max()), "z_pytorch_vs_ref_max_abs": float((z_pt - z_ref).abs().max()),
                   "ref_scores_max_abs": float(s_ref.abs().max())}
            agree = []
            for g in range(G):
                top_f = set(torch.topk(s_fused[g], k).indices.cpu().tolist())
                top_p = set(torch.topk(s_pt[g].float(), k).indices.cpu().tolist())
                agree.append(len(top_f & top_p) / k)
            num["topk_set_agreement_mean"] = statistics.fmean(agree)
            num["topk_set_agreement_min"] = min(agree)
            num["pass"] = (num["fused_vs_ref_max_abs"] <= GATE["scores_atol"] and num["fused_vs_ref_rel_norm"] <= GATE["scores_rel_norm"]
                           and num["z_fused_vs_pytorch_max_rel_ulp"] <= GATE["z_rel_ulp"] + 1e-9 and num["two_launch_vs_fused_max_abs"] <= 1e-6)
            block["numerics"] = num
            print(f"  numerics: fused vs fp32 ref max {num['fused_vs_ref_max_abs']:.2e} rel {num['fused_vs_ref_rel_norm']:.2e} | pytorch bf16 vs ref max {num['pytorch_bf16_vs_ref_max_abs']:.2e} rel {num['pytorch_bf16_vs_ref_rel_norm']:.2e} | "
                  f"z bitwise {num['z_fused_bitwise_fraction']:.3f} max rel-ulp {num['z_fused_vs_pytorch_max_rel_ulp']:.2e} | top-k agreement {num['topk_set_agreement_mean']:.4f} (min {num['topk_set_agreement_min']:.4f}) | pass={num['pass']}")

            # ---- strategies ----
            def pytorch_chain(i):
                zT[i][..., wc:wc + 1].copy_(project_keys(knew[i], idx[i].wk))
                u = project_queries(qpre[i], idx[i].wq_t_scaled)
                return candidate_scores(u, zT[i], lo, hi)

            def pytorch_score_only(i):
                return candidate_scores(u0[i], zT[i], lo, hi)

            def fused_1launch(i):
                z, kn, wk, qc, wq = metal_score.model_views(zT[i], knew[i], qpre[i], idx[i])
                return ms_kernel.fused(z, kn, wk, qc, wq, lo, hi, wc)

            def fused_2launch(i):
                z, kn, wk, qc, wq = metal_score.model_views(zT[i], knew[i], qpre[i], idx[i])
                return ms_kernel.two_launch(z, kn, wk, qc, wq, lo, hi, wc)

            def fused_1launch_v1(i):
                z, kn, wk, qc, wq = metal_score.model_views(zT[i], knew[i], qpre[i], idx[i])
                return ms_kernel.fused(z, kn, wk, qc, wq, lo, hi, wc, version=1)

            def fused_2launch_v1(i):
                z, kn, wk, qc, wq = metal_score.model_views(zT[i], knew[i], qpre[i], idx[i])
                return ms_kernel.two_launch(z, kn, wk, qc, wq, lo, hi, wc, version=1)

            strategies = {"pytorch_chain": pytorch_chain, "pytorch_score_only": pytorch_score_only, "fused_1launch": fused_1launch, "fused_2launch": fused_2launch,
                          "fused_1launch_v1": fused_1launch_v1, "fused_2launch_v1": fused_2launch_v1}
            order = list(range(a.layers))
            for pass_name, names in (("forward", list(strategies)), ("reverse", list(reversed(list(strategies))))):
                for name in names:
                    ms = time_pass(strategies[name], order)
                    us = [x / a.layers * 1e3 for x in ms]
                    cell = {"strategy": name, "order": pass_name, "n": len(us), "us_per_layer_median": statistics.median(us), "us_per_layer_p95": percentile(us, 0.95),
                            "us_per_layer_min": min(us), "us_per_layer_max": max(us), "raw_ms_per_pass": ms,
                            "effective_scan_gbps": bytes_scan / (statistics.median(us) * 1e-6) / 1e9}
                    block["cells"][f"{name}:{pass_name}"] = cell
                    for i, x in enumerate(ms):
                        jsonl.write(json.dumps({"run_id": run_id, "N0": N0, "strategy": name, "order": pass_name, "repeat": i, "elapsed_ms": x, "us_per_layer": x / a.layers * 1e3}) + "\n")
                    print(f"  {name:<20s} {pass_name:<8s} median {cell['us_per_layer_median']:7.1f} us/layer  p95 {cell['us_per_layer_p95']:7.1f}  (scan {cell['effective_scan_gbps']:5.0f} GB/s eff.)", flush=True)
            jsonl.flush()
            # conservative per-strategy figure = the slower of the two orders
            block["per_strategy_us_conservative"] = {n: max(block["cells"][f"{n}:forward"]["us_per_layer_median"], block["cells"][f"{n}:reverse"]["us_per_layer_median"]) for n in strategies}
            block["per_strategy_us_best"] = {n: min(block["cells"][f"{n}:forward"]["us_per_layer_median"], block["cells"][f"{n}:reverse"]["us_per_layer_median"]) for n in strategies}
            c = block["per_strategy_us_conservative"]
            best_name = min(("fused_1launch", "fused_2launch"), key=lambda n: c[n])
            block["best_fused"] = {"strategy": best_name, "us_conservative": c[best_name]}
            block["ratios_conservative"] = {"fused_1launch/pytorch_chain": c["fused_1launch"] / c["pytorch_chain"], "fused_2launch/pytorch_chain": c["fused_2launch"] / c["pytorch_chain"],
                                            "best_fused/pytorch_chain": c[best_name] / c["pytorch_chain"], "best_fused/pytorch_score_only": c[best_name] / c["pytorch_score_only"],
                                            "fused_1launch_v1/pytorch_chain": c["fused_1launch_v1"] / c["pytorch_chain"]}
            block["memory"] = {"current_allocated_bytes": int(torch.mps.current_allocated_memory()), "driver_allocated_bytes": int(torch.mps.driver_allocated_memory())}
            summary["lengths"].append(block)
            del zT, knew, qpre, idx, u0, views
            torch.mps.empty_cache()

    # ---- stop gate ----
    gate = {"evaluated_at_N0": None, "fused_us_conservative": None, "ratio_vs_chain": None, "numerics_pass_all": all(b["numerics"]["pass"] for b in summary["lengths"])}
    big = [b for b in summary["lengths"] if b["N0"] >= 32000]
    if big:
        b = big[-1]
        gate.update({"evaluated_at_N0": b["N0"], "fused_strategy": b["best_fused"]["strategy"], "fused_us_conservative": b["best_fused"]["us_conservative"],
                     "ratio_vs_chain": b["ratios_conservative"]["best_fused/pytorch_chain"]})
        gate["pass"] = gate["numerics_pass_all"] and gate["fused_us_conservative"] <= GATE["fused_us_max_32k"] and gate["ratio_vs_chain"] <= GATE["ratio_vs_chain_max"]
    else:
        gate["pass"] = None
    summary["stop_gate"] = gate
    summary["utc_end"] = runinfo.utc_now()
    summary["device_seconds"] = time.perf_counter() - t_start
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest.update({"status": "done", "finished_at_utc": summary["utc_end"], "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": summary["device_seconds"]}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl.close()
    print("\n== conservative medians (us/layer) and stop gate ==")
    for b in summary["lengths"]:
        c = b["per_strategy_us_conservative"]
        print(f"  N0={b['N0']:<6d} chain {c['pytorch_chain']:6.1f}  score_only {c['pytorch_score_only']:6.1f}  fused1 {c['fused_1launch']:6.1f}  fused2 {c['fused_2launch']:6.1f}  "
              f"(v1: {c['fused_1launch_v1']:6.1f} / {c['fused_2launch_v1']:6.1f})  best_fused/chain {b['ratios_conservative']['best_fused/pytorch_chain']:.2f}  numerics {'ok' if b['numerics']['pass'] else 'FAIL'}")
    print(f"  stop gate at N0={gate['evaluated_at_N0']}: {gate.get('fused_strategy')} {gate['fused_us_conservative']} us (<= {GATE['fused_us_max_32k']}), ratio {gate['ratio_vs_chain']} (<= {GATE['ratio_vs_chain_max']}), numerics {gate['numerics_pass_all']} -> pass={gate['pass']}")
    print(f"wrote {runinfo.sanitize(str(out_dir))}  ({summary['device_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
