#!/usr/bin/env python3
"""G1b full-model decode benchmark on this Mac's MPS: the pinned Qwen3 checkpoint, batch 1, bf16.

Paths (same model object, same tokens, same device; each keeps its own cache, D never borrows A's K/V):
  native_dynamic  unmodified model + transformers DynamicCache (the library default: K/V grow by torch.cat)
  native_static   unmodified model + transformers StaticCache(max_cache_len = capacity)
  wrapped_full    LabAttention wrapper, static LabCache, full read over the legal range (A)
  sparse          LabAttention wrapper, static LabCache + index table, G1a index read with every D-only cost (D)

Timed quantity: one full forward step from the token id to the next-token vocabulary scores (logits_to_keep=1),
torch.mps.synchronize() before and after. Prefill (chunked) and, for D, the index-table build inside it are timed
separately. Two measurements per (N0, path): a fixed-N single step repeated after restoring the state to N0, in
palindromic cycles over the paths; and 128 continuous steps feeding the same fixed continuation tokens one at a time,
with per-step times, legal length, k, selected count, capacity and memory recorded. Profiling (op lists) and the
optional block compile are separate from timing. Pure MPS: refuses to run with CPU fallback enabled or the memory
guard off. Random token ids drive execution only; nothing here is a quality result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_lab import runinfo  # noqa: E402

FIRST_SPECIAL_TOKEN = 151643
PATHS = ("native_dynamic", "native_static", "wrapped_full", "sparse")
MATMUL_OPS = {"aten::matmul", "aten::bmm", "aten::mm", "aten::baddbmm", "aten::addmm", "aten::linear", "aten::einsum",
              "aten::scaled_dot_product_attention", "aten::_scaled_dot_product_attention_math",
              "aten::_scaled_dot_product_attention_math_for_mps", "aten::_scaled_dot_product_flash_attention",
              "aten::_scaled_dot_product_efficient_attention", "aten::_scaled_dot_product_fused_attention_overrideable"}
WARN_BYTES, STOP_BYTES = 80_000_000_000, 100_000_000_000


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/mac_gate1b.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=None, help="prompt lengths N0")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--index-dim", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"])
    ap.add_argument("--chunk", type=int, default=None, help="prefill chunk")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None, help="single-step cycles")
    ap.add_argument("--seq-repeats", type=int, default=None)
    ap.add_argument("--paths", nargs="+", default=list(PATHS), choices=list(PATHS))
    ap.add_argument("--compile-blocks", action="store_true", help="after eager: torch.compile the MLP and RMSNorm blocks shared by every path and re-measure")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="pipeline smoke: warmup 2, 3 cycles, 1 sequence of 16 steps, run_id marked quick")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    b = cfg["benchmark"]
    a.lengths = a.lengths or b["prompt_lengths"]
    a.steps = a.steps or b["steps"]
    a.rho = b["rho"][0] if a.rho is None else a.rho
    a.index_dim = a.index_dim or cfg["index"]["r"]
    a.dtype = a.dtype or b["dtype"]
    a.chunk = a.chunk or b["prefill_chunk"]
    a.warmup = b["single_step"]["warmup"] if a.warmup is None else a.warmup
    a.repeats = b["single_step"]["repeats"] if a.repeats is None else a.repeats
    a.seq_repeats = b["sequence_repeats"] if a.seq_repeats is None else a.seq_repeats
    if a.quick:
        a.warmup, a.repeats, a.seq_repeats, a.steps = 2, 3, 1, min(a.steps, 16)
    a.cfg = cfg
    return a


def percentile(xs, p):
    s = sorted(xs)
    return s[max(0, math.ceil(p * len(s)) - 1)]


def summarize(ms):
    return {"n": len(ms), "median_ms": statistics.median(ms), "p95_ms": percentile(ms, 0.95), "min_ms": min(ms), "max_ms": max(ms),
            "mean_ms": statistics.fmean(ms)}


def slope(ys):
    """Least-squares ms per step over a sequence (growth check)."""
    n = len(ys)
    if n < 2:
        return None
    xs = range(n)
    mx, my = (n - 1) / 2, statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None


def mem(torch):
    return {"current_allocated_bytes": int(torch.mps.current_allocated_memory()), "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


class Paths:
    """Runners for every measured path over one wrapped-capable model; toggles wrapped / native per call."""

    def __init__(self, torch, model, holder, lab_layers, names, N0, steps, rho, r, device):
        from transformers import DynamicCache, StaticCache
        from attention_lab.qwen_wrap import LabRunner, NativeRunner, make_state
        self.torch, self.model, self.holder, self.lab_layers = torch, model, holder, lab_layers
        self.names, self.N0, self.steps, self.rho = list(names), N0, steps, rho
        self.capacity = N0 + steps
        self.runners, self.states, self.kind = {}, {}, {}
        for name in self.names:
            if name == "native_dynamic":
                self.set_wrapped(False)
                self.runners[name] = NativeRunner(model, DynamicCache(config=model.config), self.capacity)
                self.kind[name] = "native"
            elif name == "native_static":
                self.set_wrapped(False)
                self.runners[name] = NativeRunner(model, StaticCache(config=model.config, max_cache_len=self.capacity), self.capacity)
                self.kind[name] = "native"
            else:
                self.set_wrapped(True)
                st = make_state(model, self.capacity, "full" if name == "wrapped_full" else "sparse", r=r)
                self.states[name] = st
                self.runners[name] = LabRunner(model, holder, st)
                self.kind[name] = "wrapped"

    def set_wrapped(self, flag: bool):
        for layer, lab in zip(self.model.model.layers, self.lab_layers):
            layer.self_attn = lab if flag else lab.inner

    def select(self, name):
        self.set_wrapped(self.kind[name] == "wrapped")
        return self.runners[name]

    def restore(self, name):
        self.select(name).rollback(self.N0)

    def selection_fields(self, name):
        st = self.states.get(name)
        if st is None or st.mode != "sparse" or st.plan is None:
            return {"k0": None, "k": None, "selected_count": None, "candidate_count": None, "legal_length_plan": None}
        p = st.plan
        return {"k0": p.k0, "k": p.k, "selected_count": p.selected_count, "candidate_count": p.candidate_count, "legal_length_plan": p.legal_length}

    def cache_bytes(self, name):
        st = self.states.get(name)
        if st is not None:
            return st.cache.bytes()
        es = self.torch.tensor([], dtype=next(self.model.parameters()).dtype).element_size()
        geo = self._geo
        kv = geo.layers * 2 * geo.groups * geo.head_dim * es * (self.capacity if name == "native_static" else self.N0)
        return {"kv_bytes": kv, "index_bytes": 0, "total_bytes": kv, "note": "native_dynamic grows with the sequence; native_static is preallocated to capacity"}


def profile_step(torch, fn, sync):
    from torch.profiler import ProfilerActivity, profile
    fn()
    sync()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        fn()
        sync()
    ops = [{"name": e.name, "input_shapes": [list(s) for s in (e.input_shapes or [])], "cpu_time_us": float(e.cpu_time_total)}
           for e in prof.events() if e.name.startswith("aten::")]
    table = prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time_total", row_limit=45)
    return ops, table


def length_evidence(ops, lengths, d):
    """Matmul-class ops whose input shapes carry one of `lengths` (legal length or capacity) together with the head dim."""
    hits = []
    for op in ops:
        if op["name"] in MATMUL_OPS and any(any(L in s for L in lengths) and d in s for s in op["input_shapes"]):
            hits.append({"name": op["name"], "input_shapes": op["input_shapes"]})
    return hits


def make_block_compiler(torch):
    """Shared compiled functions for the MLP and RMSNorm math (identical op order to the transformers modules)."""
    import torch.nn.functional as F
    from torch import nn

    def mlp_math(x, gate_w, up_w, down_w):
        return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)

    def rms_math(x, weight, eps: float):
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        return weight * (xf * torch.rsqrt(var + eps)).to(x.dtype)

    c_mlp = torch.compile(mlp_math, dynamic=False)
    c_rms = torch.compile(rms_math, dynamic=False)

    class CompiledMLP(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return c_mlp(x, self.inner.gate_proj.weight, self.inner.up_proj.weight, self.inner.down_proj.weight)

    class CompiledRMS(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return c_rms(x, self.inner.weight, self.inner.variance_epsilon)

    return CompiledMLP, CompiledRMS


def swap_blocks(model, CompiledMLP, CompiledRMS):
    originals = []
    for layer in model.model.layers:
        originals.append((layer, layer.mlp, layer.input_layernorm, layer.post_attention_layernorm))
        layer.mlp = CompiledMLP(layer.mlp)
        layer.input_layernorm = CompiledRMS(layer.input_layernorm)
        layer.post_attention_layernorm = CompiledRMS(layer.post_attention_layernorm)
    norm = model.model.norm
    model.model.norm = CompiledRMS(norm)
    return originals, norm


def restore_blocks(model, originals, norm):
    for layer, mlp, ln1, ln2 in originals:
        layer.mlp, layer.input_layernorm, layer.post_attention_layernorm = mlp, ln1, ln2
    model.model.norm = norm


def main() -> int:
    a = parse_args()
    try:
        runinfo.refuse_unsafe_env()
    except RuntimeError as e:
        print(f"Refusing: {e}")
        return 2
    import torch
    if not torch.backends.mps.is_available():
        print("MPS not available; this benchmark is MPS-only.")
        return 1
    import transformers
    from attention_lab.qwen_wrap import LabAttention, StateHolder, geometry_of, load_model, weights_file_sha256
    dev = torch.device("mps")
    sync = torch.mps.synchronize
    cfg, mcfg = a.cfg, a.cfg["model"]
    run_id = "g1b-mac-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    args_rec = {k: (runinfo.sanitize(str(v)) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "cfg"}
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "benchmark", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": runinfo.sha256_file(a.config),
                "model_id": mcfg["id"], "model_revision": mcfg["revision"], "tokenizer_revision": mcfg["tokenizer_revision"],
                "weights": weights_file_sha256(mcfg["id"], mcfg["revision"]), "data_sha256": {}, "seed": a.seed, "device": "mps",
                "host": runinfo.host_info(), "software": {**runinfo.torch_env(torch), "transformers": transformers.__version__}, "backend": "mps",
                "dtype": a.dtype, "command": "python " + " ".join(runinfo.sanitize(x) for x in sys.argv),
                "timing_scope": "one full forward step (token id -> next-token logits via logits_to_keep=1), torch.mps.synchronize() before and after; prefill timed separately",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["steps.jsonl", "summary.json", "manifest.json", "profiles.txt"], "args": args_rec,
                "limitations": ["random token ids drive execution only; no task, no quality statement; MPS only; batch 1"]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_id {run_id}  torch {torch.__version__}  transformers {transformers.__version__}")
    t_start = time.perf_counter()
    dtype = getattr(torch, a.dtype)
    t0 = time.perf_counter()
    model, tok, info = load_model(mcfg["id"], mcfg["revision"], dtype=dtype, device=dev)
    sync()
    load_s = time.perf_counter() - t0
    geo = geometry_of(model)
    mism = {k: (info.get(k), v) for k, v in mcfg["expected_from_config_json"].items() if info.get(k) is not None and info.get(k) != v}
    print(f"model loaded in {load_s:.1f} s: {geo}  params {info['num_parameters_total']/1e6:.1f} M  mismatch_vs_config {mism or 'none'}")
    holder = StateHolder()
    lab_layers = [LabAttention(layer.self_attn, holder, i) for i, layer in enumerate(model.model.layers)]   # created once, toggled per path
    jsonl = open(out_dir / "steps.jsonl", "w", encoding="utf-8")
    summary = {"run_id": run_id, "quick": a.quick, "utc_start": runinfo.utc_now(), "args": args_rec, "host": manifest["host"], "software": manifest["software"],
               "git": manifest["git"], "source_hashes": manifest["source_hashes"], "model_info": {**info, "mismatch_vs_config_json": mism, "load_seconds": load_s},
               "lengths": [], "memory_after_load": mem(torch)}
    profiles_txt = []
    rec_max = manifest["software"].get("recommended_max_memory") or 0

    def write_row(row):
        jsonl.write(json.dumps(row) + "\n")

    def timed(fn):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        return (time.perf_counter() - t) * 1e3, out

    with torch.inference_mode():
        for N0 in a.lengths:
            steps, cap = a.steps, N0 + a.steps
            gen = torch.Generator().manual_seed(a.seed + N0)
            ids = torch.randint(0, FIRST_SPECIAL_TOKEN, (1, N0 + steps), generator=gen)
            data_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            manifest["data_sha256"][str(N0)] = data_sha
            ids_dev = ids.to(dev)
            es = torch.tensor([], dtype=dtype).element_size()
            per_cache = geo.layers * 2 * geo.groups * cap * geo.head_dim * es
            need = per_cache * len(a.paths) + geo.layers * geo.groups * a.index_dim * cap * es
            if need > STOP_BYTES or (rec_max and need > 0.8 * rec_max):
                summary["lengths"].append({"N0": N0, "skipped": f"estimated cache bytes {need} exceed limits"})
                print(f"skip N0={N0}: {need/1e9:.1f} GB of caches exceeds limits")
                continue
            print(f"\n== N0={N0} steps={steps} capacity={cap} rho={a.rho} r={a.index_dim} paths={a.paths}  caches ~{need/1e9:.1f} GB ==")
            block = {"N0": N0, "steps": steps, "capacity": cap, "rho": a.rho, "index_dim": a.index_dim, "data_sha256": data_sha, "paths": {},
                     "modes": {}, "memory": {"before": mem(torch)}}
            P = Paths(torch, model, holder, lab_layers, a.paths, N0, steps, a.rho, a.index_dim, dev)
            P._geo = geo
            # --- prefill, timed per path (D's includes the index-table build) ---
            first_logits = {}
            for name in a.paths:
                run = P.select(name)
                ms, _ = timed(lambda: run.prefill(ids_dev[:, :N0], a.chunk))
                if name == "sparse":
                    k0 = P.states[name].begin_request(a.rho)
                else:
                    k0 = None
                ms1, lg = timed(lambda: run.step(ids_dev[:, N0:N0 + 1]))
                first_logits[name] = lg[0, -1].float()
                P.restore(name)
                block["paths"][name] = {"prefill_ms": ms, "prefill_chunk": a.chunk, "first_step_ms_cold": ms1, "k0": k0,
                                        "cache_bytes": P.cache_bytes(name), "memory_after_prefill": mem(torch), **P.selection_fields(name)}
                print(f"  prefill {name:<15s} {ms:9.1f} ms   first step {ms1:7.2f} ms   k0={k0}  mem {mem(torch)['current_allocated_bytes']/1e9:.2f} GB", flush=True)
            # bf16 sanity across paths at the first step (information only; the formal numerics are bench/model_numerics.py)
            base = first_logits[a.paths[0]]
            block["first_step_agreement"] = {name: {"top1_equal": bool(lg.argmax() == base.argmax()), "max_abs_vs_first_path": float((lg - base).abs().max())}
                                             for name, lg in first_logits.items()}
            print(f"  first-step top-1 agreement vs {a.paths[0]}: " + ", ".join(f"{n}={v['top1_equal']}" for n, v in block["first_step_agreement"].items()))

            def measure(mode, error_on_recompile=False):
                import torch._dynamo as dynamo
                res = {name: {"single_ms": [], "sequences": []} for name in a.paths}
                order = list(a.paths) + list(reversed(a.paths))
                # warm-up: a few steps per path from N0, then restore
                for name in a.paths:
                    run = P.select(name)
                    for i in range(a.warmup):
                        run.step(ids_dev[:, N0 + i:N0 + i + 1])
                    P.restore(name)
                sync()
                if error_on_recompile:
                    dynamo.config.error_on_recompile = True
                try:
                    # fixed-N single step: restore -> time one step at position N0, palindromic cycles over the paths
                    for cyc in range(a.repeats):
                        for name in order:
                            P.restore(name)
                            run = P.runners[name]
                            ms, _ = timed(lambda: run.step(ids_dev[:, N0:N0 + 1]))
                            res[name]["single_ms"].append(ms)
                            res[name].setdefault("single_cycles", [[] for _ in range(a.repeats)])[cyc].append(ms)
                            write_row({"run_id": run_id, "phase": "single", "mode": mode, "N0": N0, "path": name, "cycle": cyc, "legal_before": N0,
                                       "legal_after": N0 + 1, "capacity": cap, "elapsed_ms": ms, **P.selection_fields(name)})
                    # continuous steps: the same fixed continuation tokens, one at a time, per-step timing
                    for rep in range(a.seq_repeats):
                        seq_order = list(a.paths) if rep % 2 == 0 else list(reversed(a.paths))
                        for name in seq_order:
                            P.restore(name)
                            run = P.runners[name]
                            series, mems = [], []
                            for i in range(steps):
                                ms, _ = timed(lambda: run.step(ids_dev[:, N0 + i:N0 + i + 1]))
                                series.append(ms)
                                sel = P.selection_fields(name)
                                row = {"run_id": run_id, "phase": "sequence", "mode": mode, "N0": N0, "path": name, "rep": rep, "step": i,
                                       "legal_before": N0 + i, "legal_after": N0 + i + 1, "capacity": cap, "elapsed_ms": ms, **sel}
                                if i % 16 == 0 or i == steps - 1:
                                    row["memory"] = mem(torch)
                                    mems.append({"step": i, **row["memory"]})
                                write_row(row)
                            res[name]["sequences"].append({"rep": rep, "ms": series, "total_ms": sum(series), "median_ms": statistics.median(series),
                                                           "first16_median_ms": statistics.median(series[:16]), "last16_median_ms": statistics.median(series[-16:]),
                                                           "slope_ms_per_step": slope(series), "memory": mems, "k_last": P.selection_fields(name)["k"],
                                                           "selected_last": P.selection_fields(name)["selected_count"]})
                            P.restore(name)
                finally:
                    dynamo.config.error_on_recompile = False
                for name in a.paths:
                    r = res[name]
                    r["single"] = summarize(r["single_ms"]) if r["single_ms"] else None
                    if r["sequences"]:
                        r["sequence"] = {"n": len(r["sequences"]), "total_ms_median": statistics.median(s["total_ms"] for s in r["sequences"]),
                                         "step_median_ms": statistics.median(x for s in r["sequences"] for x in s["ms"]),
                                         "step_p95_ms": percentile([x for s in r["sequences"] for x in s["ms"]], 0.95),
                                         "first16_median_ms": statistics.median(s["first16_median_ms"] for s in r["sequences"]),
                                         "last16_median_ms": statistics.median(s["last16_median_ms"] for s in r["sequences"]),
                                         "slope_ms_per_step_median": statistics.median(s["slope_ms_per_step"] for s in r["sequences"] if s["slope_ms_per_step"] is not None)}
                # paired ratios per cycle (same cycle, same position, G1a-supplement style): path / wrapped_full
                if "wrapped_full" in a.paths and res["wrapped_full"].get("single_cycles"):
                    A_cycles = [statistics.median(c) for c in res["wrapped_full"]["single_cycles"] if c]
                    for name in a.paths:
                        cycles = [statistics.median(c) for c in res[name].get("single_cycles", []) if c]
                        ratios = [x / y for x, y in zip(cycles, A_cycles)]
                        if ratios:
                            res[name]["paired_ratio_vs_wrapped_full"] = {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios),
                                                                         "n_cycles": len(ratios), "cycles_at_or_below_0.90": sum(r <= 0.90 for r in ratios),
                                                                         "per_cycle": ratios}
                return res

            # eager
            res = measure("eager")
            block["modes"]["eager"] = res
            block["memory"]["after_eager"] = mem(torch)
            for name in a.paths:
                r = res[name]
                s, q = r["single"], r.get("sequence")
                print(f"  eager {name:<15s} single median {s['median_ms']:7.2f} ms (p95 {s['p95_ms']:7.2f}, n={s['n']})"
                      + (f"   128-step median {q['step_median_ms']:7.2f} ms  first16 {q['first16_median_ms']:7.2f} last16 {q['last16_median_ms']:7.2f}  slope {q['slope_ms_per_step_median']*1e3:+.1f} us/step" if q else ""), flush=True)
            # --- profile one step of each wrapped path (separate from timing) ---
            if not a.skip_profile:
                prof = {}
                for name in a.paths:
                    try:
                        run = P.select(name)
                        P.restore(name)
                        ops, table = profile_step(torch, lambda: run.step(ids_dev[:, N0:N0 + 1]), sync)
                        P.restore(name)
                        prof[name] = {"aten_ops": len(ops), "distinct_ops": len({o["name"] for o in ops}),
                                      "length_matmul_ops": length_evidence(ops, (N0 + 1, cap), geo.head_dim),
                                      "ops_touching_length": sorted({o["name"] for o in ops if any((N0 + 1) in s or cap in s for s in o["input_shapes"])}),
                                      "sdpa_dispatch": sorted({o["name"] for o in ops if "scaled_dot_product" in o["name"] or "sdp" in o["name"]}),
                                      "topk_calls": sum(1 for o in ops if o["name"] == "aten::topk"),
                                      "index_select_calls": sum(1 for o in ops if o["name"] == "aten::index_select"),
                                      "cat_calls": sum(1 for o in ops if o["name"] == "aten::cat")}
                        prof[name]["full_length_matmul_detected"] = bool(prof[name]["length_matmul_ops"])
                        profiles_txt.append(f"=== N0={N0} {name} ===\n{table}\n")
                        print(f"  profile {name:<15s} aten ops {prof[name]['aten_ops']:5d}  length-matmul ops {len(prof[name]['length_matmul_ops']):3d}  topk {prof[name]['topk_calls']}  index_select {prof[name]['index_select_calls']}  cat {prof[name]['cat_calls']}  sdpa {prof[name]['sdpa_dispatch']}")
                    except Exception as e:
                        prof[name] = {"status": "error", "error_type": type(e).__name__, "error": runinfo.sanitize(str(e))[:400]}
                        print(f"  profile {name} ERROR {type(e).__name__}: {e}")
                block["profiles"] = prof
            # --- block compile (optional): same compiled MLP / RMSNorm for every path, re-measure ---
            if a.compile_blocks:
                import torch._dynamo as dynamo
                cinfo = {"status": "ok"}
                originals = None
                try:
                    dynamo.reset()
                    CompiledMLP, CompiledRMS = make_block_compiler(torch)
                    originals, norm = swap_blocks(model, CompiledMLP, CompiledRMS)
                    t0 = time.perf_counter()
                    run = P.select(a.paths[0])
                    out_c = run.step(ids_dev[:, N0:N0 + 1])
                    sync()
                    cinfo["first_call_s"] = time.perf_counter() - t0
                    P.restore(a.paths[0])
                    restore_blocks(model, originals, norm)
                    out_e = P.select(a.paths[0]).step(ids_dev[:, N0:N0 + 1])
                    sync()
                    P.restore(a.paths[0])
                    cinfo["max_abs_diff_vs_eager_logits"] = float((out_c.float() - out_e.float()).abs().max())
                    originals, norm = swap_blocks(model, CompiledMLP, CompiledRMS)
                    before = {k: dict(v) for k, v in dynamo.utils.counters.items()}
                    res_c = measure("compile", error_on_recompile=True)
                    after = {k: dict(v) for k, v in dynamo.utils.counters.items()}
                    cinfo["dynamo_counters_delta"] = {k: {kk: after[k].get(kk, 0) - before.get(k, {}).get(kk, 0) for kk in after[k]} for k in after}
                    block["modes"]["compile"] = res_c
                    for name in a.paths:
                        s, q = res_c[name]["single"], res_c[name].get("sequence")
                        print(f"  compile {name:<15s} single median {s['median_ms']:7.2f} ms (p95 {s['p95_ms']:7.2f})"
                              + (f"   128-step median {q['step_median_ms']:7.2f} ms  slope {q['slope_ms_per_step_median']*1e3:+.1f} us/step" if q else ""), flush=True)
                except Exception as e:
                    cinfo.update({"status": "error", "error_type": type(e).__name__, "error": runinfo.sanitize(str(e))[:800],
                                  "traceback_tail": runinfo.sanitize(traceback.format_exc())[-1200:]})
                    print(f"  compile ERROR {type(e).__name__}: {runinfo.sanitize(str(e))[:300]}")
                finally:
                    if originals is not None:
                        restore_blocks(model, originals, norm)
                    dynamo.config.error_on_recompile = False
                    dynamo.reset()
                block["compile"] = cinfo
            block["memory"]["end"] = mem(torch)
            summary["lengths"].append(block)
            P.set_wrapped(False)
            del P
            torch.mps.empty_cache()
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    summary["utc_end"] = runinfo.utc_now()
    summary["device_seconds"] = time.perf_counter() - t_start
    summary["memory_end"] = mem(torch)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / "profiles.txt").write_text("\n".join(profiles_txt), encoding="utf-8")
    manifest.update({"status": "done", "finished_at_utc": summary["utc_end"], "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": summary["device_seconds"]}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl.close()
    print("\n== full-step medians (ms) ==")
    for block in summary["lengths"]:
        if "modes" not in block:
            continue
        for mode, res in block["modes"].items():
            A = res.get("wrapped_full", {}).get("single", {}).get("median_ms")
            line = f"  N0={block['N0']:<6d} {mode:<7s} " + "  ".join(f"{n} {res[n]['single']['median_ms']:7.2f}" + (f" ({res[n]['single']['median_ms']/A:4.2f}x A)" if A and n != "wrapped_full" else "") for n in res if res[n].get("single"))
            print(line)
    print(f"\nwrote {runinfo.sanitize(str(out_dir))}  ({summary['device_seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
