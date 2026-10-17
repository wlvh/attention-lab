#!/usr/bin/env python3
"""P1 spike: gather-free subset attention (Metal) vs the G1a index_select + SDPA path, random tensors on this Mac's MPS.

Per layer and decode step the segment is: read the S selected K rows and V rows and attend over them. Strategies
(bf16 inputs, batch 1, G = 8, h = 2, d = 128):
  pytorch_gather_sdpa   flattened index_select of K, of V, then subset SDPA (the G1a D path: three kernels)
  fused_attend_P{1,2,4,8}  MetalAttend with P partitions per group (P > 1 adds a tiny combine kernel), bf16 output
  pytorch_full_sdpa     context only: A's full read over the legal length with sdpa_fold (not part of the gate)
Row lists are F ∪ k random distinct candidates per layer (the shape of a D selection; k from the rho = 1/32 rule), kept
as flat int64 row ids for the PyTorch path and as int32 positions for the kernel; both are prepared outside the timed
region (in the full chain the selection stage produces them). Rotating workset of `layers` caches; warm-up then timed
passes with torch.mps.synchronize() before and after each pass; every strategy timed in forward and reverse order.
Numerics per length: fp32 kernel output vs dense_mask_reference (float32 CPU), the PyTorch bf16 output vs the same
reference, bf16 kernel output vs the RNE rounding of the fp32 output. Stop gate (docs/tasks/04_FUSED_D_KERNEL.md P1)
is evaluated at S = 1076 and written to summary.json.
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

GATE = {"fused_us_max_at_S1076": 60.0, "ratio_vs_gather_sdpa_max": 0.80, "out_atol": 5e-5, "out_rel_norm": 1e-5, "bf16_ulp": 2.0 ** -8}
PARTS = (1, 2, 4, 8, 16, 32)   # 16 and 32 added after the smoke run (P8 = 37 us, ratio 0.79): same gate, wider sweep of the pre-declared parameter


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32256])
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--heads-per-group", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
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
    from attention_lab import metal_attend
    from attention_lab.decode import prepare_request
    from attention_lab.full import full_attention
    from attention_lab.reference import dense_mask_reference
    from attention_lab.sparse import flatten_cache, subset_attention
    if not metal_attend.available():
        print("needs MPS and torch.mps.compile_shader")
        return 1
    dev, sync = torch.device("mps"), torch.mps.synchronize
    G, h, d = a.groups, a.heads_per_group, a.head_dim
    run_id = "p1-attend-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "benchmark", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": None, "model_revision": None,
                "tokenizer_revision": None, "data_sha256": None, "seed": a.seed, "device": "mps", "host": runinfo.host_info(),
                "software": runinfo.torch_env(torch), "backend": "mps+metal(compile_shader)", "dtype": "bfloat16",
                "command": "python " + " ".join(runinfo.sanitize(x) for x in sys.argv),
                "timing_scope": "single layer subset-attention segment per decode step; pass = layers layer-steps; sync before/after pass",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["p1.jsonl", "summary.json", "manifest.json"],
                "args": {k: (runinfo.sanitize(str(v)) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "limitations": ["random tensors, one layer segment, batch 1; row lists prepared outside timing; no model, no task; MPS only; P1 of docs/tasks/04_FUSED_D_KERNEL.md"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_id {run_id}  torch {torch.__version__}  layers {a.layers}  warmup {a.warmup} repeats {a.repeats}")
    t_start = time.perf_counter()
    ma = metal_attend.MetalAttend()
    jsonl = open(out_dir / "p1.jsonl", "w", encoding="utf-8")
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
            req = prepare_request(1, G, C, valid_length=L, query_position=L - 1, rho=a.rho, device="cpu")
            lo, hi, k, S = req.layout.candidate_lo, req.layout.candidate_hi, req.k, req.selected_count
            fixed = torch.tensor(req.layout.fixed_positions(), dtype=torch.long)
            gen = torch.Generator().manual_seed(a.seed + N0)
            torch.manual_seed(a.seed + N0)
            K4 = [torch.randn(1, G, C, d, device=dev, dtype=torch.bfloat16) for _ in range(a.layers)]
            V4 = [torch.randn(1, G, C, d, device=dev, dtype=torch.bfloat16) for _ in range(a.layers)]
            q5 = [torch.randn(1, G, h, 1, d, device=dev, dtype=torch.bfloat16) for _ in range(a.layers)]
            rows3 = [torch.stack([torch.cat([fixed, lo + torch.randperm(hi - lo, generator=gen)[:k]]) for _ in range(G)]).unsqueeze(0) for _ in range(a.layers)]
            offsets = (torch.arange(G, dtype=torch.long) * C).view(1, G, 1)
            rows_flat = [(r + offsets).reshape(-1).to(dev) for r in rows3]                        # PyTorch path: flat int64 row ids
            views = [metal_attend.model_views(K4[i], V4[i], q5[i], rows3[i].to(dev)) for i in range(a.layers)]   # kernel: int32 positions
            kvi = [torch.stack([K4[i][0], V4[i][0]], dim=2).contiguous() for i in range(a.layers)]               # diagnostic layout [G, C, 2, d]
            rows_sorted = [rows3[i][0].sort(dim=-1).values.to(torch.int32).contiguous().to(dev) for i in range(a.layers)]   # diagnostic: sorted lists
            sync()
            bytes_rows = S * d * 2 * 2 * G
            block = {"N0": N0, "capacity": C, "legal_length": L, "k": k, "S": S, "layers": a.layers, "kv_bytes_read_per_layer": bytes_rows,
                     "full_read_bytes_per_layer": L * d * 2 * 2 * G, "cells": {}, "numerics": {}}
            print(f"\n== N0={N0} C={C} L={L} k={k} S={S}  rows read {bytes_rows/1e6:.2f} MB/layer (A reads {block['full_read_bytes_per_layer']/1e6:.1f} MB) ==")

            # ---- numerics (layer 0) ----
            K, V, q, rows = views[0]
            ref = dense_mask_reference(q5[0].float().cpu(), K4[0].float().cpu(), V4[0].float().cpu(), rows3[0], valid_length=C)[0, :, :, 0, :]
            ks = flatten_cache(K4[0]).index_select(0, rows_flat[0]).view(1, G, S, d)
            vs = flatten_cache(V4[0]).index_select(0, rows_flat[0]).view(1, G, S, d)
            pt = subset_attention(q5[0], ks, vs)[0, :, :, 0, :]
            num = {"pytorch_bf16_vs_ref_max_abs": float((pt.float().cpu() - ref).abs().max()), "pytorch_bf16_vs_ref_rel_norm": float((pt.float().cpu() - ref).norm() / ref.norm()),
                   "ref_out_max_abs": float(ref.abs().max()), "per_partition": {}}
            ok = True
            for P in PARTS:
                o32 = ma(K, V, q, rows, partitions=P)
                o16 = ma(K, V, q, rows, partitions=P, out_dtype=torch.bfloat16)
                sync()
                diff = float((o32.cpu() - ref).abs().max())
                rel = float((o32.cpu() - ref).norm() / ref.norm())
                rne = o32.to(torch.bfloat16).float().cpu()
                ulp_ok = bool(((o16.float().cpu() - rne).abs() <= GATE["bf16_ulp"] * rne.abs() + 1e-7).all())
                bit = float((o16.float().cpu() == rne).float().mean())
                p_ok = diff <= GATE["out_atol"] and rel <= GATE["out_rel_norm"] and ulp_ok
                ok = ok and p_ok
                num["per_partition"][str(P)] = {"fused_vs_ref_max_abs": diff, "fused_vs_ref_rel_norm": rel, "bf16_within_one_ulp": ulp_ok, "bf16_bitwise_fraction": bit, "pass": p_ok}
            num["pass"] = ok
            block["numerics"] = num
            print("  numerics: " + " | ".join(f"P={P} max {v['fused_vs_ref_max_abs']:.2e} rel {v['fused_vs_ref_rel_norm']:.2e} bf16 bitwise {v['bf16_bitwise_fraction']:.3f}" for P, v in num["per_partition"].items())
                  + f" | pytorch bf16 vs ref max {num['pytorch_bf16_vs_ref_max_abs']:.2e} rel {num['pytorch_bf16_vs_ref_rel_norm']:.2e} | pass={ok}")

            # ---- strategies ----
            def pytorch_gather_sdpa(i):
                ks = flatten_cache(K4[i]).index_select(0, rows_flat[i]).view(1, G, S, d)
                vs = flatten_cache(V4[i]).index_select(0, rows_flat[i]).view(1, G, S, d)
                return subset_attention(q5[i], ks, vs)

            def make_fused(P):
                def fn(i):
                    K, V, q, rows = views[i]
                    return ma(K, V, q, rows, partitions=P, out_dtype=torch.bfloat16)
                return fn

            def pytorch_full_sdpa(i):
                return full_attention(q5[i], K4[i], V4[i], valid_length=L, backend="sdpa_fold")

            def make_fused_kvi(P):
                def fn(i):
                    _, _, q, rows = views[i]
                    return ma(None, None, q, rows, partitions=P, out_dtype=torch.bfloat16, kv_interleaved=kvi[i])
                return fn

            def fused_p8_sorted(i):
                K, V, q, _ = views[i]
                return ma(K, V, q, rows_sorted[i], partitions=8, out_dtype=torch.bfloat16)

            strategies = {"pytorch_gather_sdpa": pytorch_gather_sdpa, **{f"fused_attend_P{P}": make_fused(P) for P in PARTS},
                          "diag_kv_interleaved_P4": make_fused_kvi(4), "diag_kv_interleaved_P8": make_fused_kvi(8), "diag_sorted_rows_P8": fused_p8_sorted,
                          "pytorch_full_sdpa": pytorch_full_sdpa}
            order = list(range(a.layers))
            for pass_name, names in (("forward", list(strategies)), ("reverse", list(reversed(list(strategies))))):
                for name in names:
                    ms = time_pass(strategies[name], order)
                    us = [x / a.layers * 1e3 for x in ms]
                    nbytes = block["full_read_bytes_per_layer"] if name == "pytorch_full_sdpa" else bytes_rows
                    cell = {"strategy": name, "order": pass_name, "n": len(us), "us_per_layer_median": statistics.median(us), "us_per_layer_p95": percentile(us, 0.95),
                            "us_per_layer_min": min(us), "us_per_layer_max": max(us), "raw_ms_per_pass": ms, "effective_gbps": nbytes / (statistics.median(us) * 1e-6) / 1e9}
                    block["cells"][f"{name}:{pass_name}"] = cell
                    for i, x in enumerate(ms):
                        jsonl.write(json.dumps({"run_id": run_id, "N0": N0, "S": S, "strategy": name, "order": pass_name, "repeat": i, "elapsed_ms": x, "us_per_layer": x / a.layers * 1e3}) + "\n")
                    print(f"  {name:<20s} {pass_name:<8s} median {cell['us_per_layer_median']:7.1f} us/layer  p95 {cell['us_per_layer_p95']:7.1f}  ({cell['effective_gbps']:5.0f} GB/s eff.)", flush=True)
            jsonl.flush()
            block["per_strategy_us_conservative"] = {n: max(block["cells"][f"{n}:forward"]["us_per_layer_median"], block["cells"][f"{n}:reverse"]["us_per_layer_median"]) for n in strategies}
            block["per_strategy_us_best"] = {n: min(block["cells"][f"{n}:forward"]["us_per_layer_median"], block["cells"][f"{n}:reverse"]["us_per_layer_median"]) for n in strategies}
            c = block["per_strategy_us_conservative"]
            best = min((f"fused_attend_P{P}" for P in PARTS), key=lambda n: c[n])
            block["best_fused"] = {"strategy": best, "us_conservative": c[best]}
            block["ratios_conservative"] = {"best_fused/pytorch_gather_sdpa": c[best] / c["pytorch_gather_sdpa"], "best_fused/pytorch_full_sdpa": c[best] / c["pytorch_full_sdpa"],
                                            **{f"fused_attend_P{P}/pytorch_gather_sdpa": c[f"fused_attend_P{P}"] / c["pytorch_gather_sdpa"] for P in PARTS}}
            block["memory"] = {"current_allocated_bytes": int(torch.mps.current_allocated_memory()), "driver_allocated_bytes": int(torch.mps.driver_allocated_memory())}
            summary["lengths"].append(block)
            del K4, V4, q5, rows_flat, views, kvi, rows_sorted
            torch.mps.empty_cache()

    gate = {"evaluated_at_N0": None, "S": None, "fused_strategy": None, "fused_us_conservative": None, "ratio_vs_gather_sdpa": None,
            "numerics_pass_all": all(b["numerics"]["pass"] for b in summary["lengths"])}
    big = [b for b in summary["lengths"] if b["S"] >= 1076]
    if big:
        b = big[-1]
        gate.update({"evaluated_at_N0": b["N0"], "S": b["S"], "fused_strategy": b["best_fused"]["strategy"], "fused_us_conservative": b["best_fused"]["us_conservative"],
                     "ratio_vs_gather_sdpa": b["ratios_conservative"]["best_fused/pytorch_gather_sdpa"]})
        gate["pass"] = gate["numerics_pass_all"] and gate["fused_us_conservative"] <= GATE["fused_us_max_at_S1076"] and gate["ratio_vs_gather_sdpa"] <= GATE["ratio_vs_gather_sdpa_max"]
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
        print(f"  N0={b['N0']:<6d} S={b['S']:<5d} gather_sdpa {c['pytorch_gather_sdpa']:6.1f}  " + "  ".join(f"P{P} {c[f'fused_attend_P{P}']:6.1f}" for P in PARTS)
              + f"  | diag kvi P4 {c['diag_kv_interleaved_P4']:6.1f} P8 {c['diag_kv_interleaved_P8']:6.1f} sorted P8 {c['diag_sorted_rows_P8']:6.1f}"
              + f"  | full_sdpa {c['pytorch_full_sdpa']:6.1f}  best {b['best_fused']['strategy']} ratio {b['ratios_conservative']['best_fused/pytorch_gather_sdpa']:.2f}  numerics {'ok' if b['numerics']['pass'] else 'FAIL'}")
    print(f"  stop gate at N0={gate['evaluated_at_N0']} S={gate['S']}: {gate.get('fused_strategy')} {gate['fused_us_conservative']} us (<= {GATE['fused_us_max_at_S1076']}), ratio {gate['ratio_vs_gather_sdpa']} (<= {GATE['ratio_vs_gather_sdpa_max']}), numerics {gate['numerics_pass_all']} -> pass={gate['pass']}")
    print(f"wrote {runinfo.sanitize(str(out_dir))}  ({summary['device_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
