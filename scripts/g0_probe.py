#!/usr/bin/env python3
"""G0: capability, memory, effective-bandwidth and dispatch-overhead probe for one Mac.

Scope: this machine only. No model download, no training, no cloud. Every probe is
wrapped so the JSON is always written; an unsupported operation is recorded as such,
never silently routed to the CPU (the script refuses to run with MPS fallback enabled).

Outputs runs/<run_id>/g0_probe.json and prints a short summary. Numbers are measurements
of this host under the recorded software; they are not architecture conclusions.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GiB = 1024 ** 3
MiB = 1024 ** 2


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize(text: str) -> str:
    home = os.path.expanduser("~")
    return text.replace(home, "~")


def host_info() -> dict:
    info = {"system": platform.system(), "machine": platform.machine(), "macos": platform.mac_ver()[0] or None,
            "python": platform.python_version(), "chip": None, "gpu_cores": None, "physical_memory_bytes": None}
    try:
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip() or None
        info["physical_memory_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip())
        disp = json.loads(subprocess.run(["system_profiler", "-json", "SPDisplaysDataType"], capture_output=True, text=True, timeout=30).stdout)
        for gpu in disp.get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores")
            if cores:
                info["gpu_cores"] = int(cores)
    except Exception as e:  # inventory only
        info["host_info_error"] = f"{type(e).__name__}: {e}"
    return info


def probe(name: str, results: dict, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        out = {} if out is None else out
        out.setdefault("status", "ok")
    except Exception as e:
        out = {"status": "error", "error_type": type(e).__name__, "error": sanitize(str(e))[:400]}
    out["probe_seconds"] = round(time.perf_counter() - t0, 3)
    results[name] = out
    flag = out["status"]
    print(f"[{flag:5s}] {name} ({out['probe_seconds']}s)" + (f" -> {out.get('error_type')}: {out.get('error', '')[:120]}" if flag == "error" else ""))
    return out


def rel_err(a, b) -> float:
    a = a.detach().float().cpu(); b = b.detach().float().cpu()
    return float(((a - b).norm() / (b.norm() + 1e-12)).item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--sizes-gib", type=float, nargs="+", default=[0.25, 1.0, 4.0])
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--scan-layers", type=int, default=36)
    ap.add_argument("--scan-n", type=int, default=32768)
    ap.add_argument("--skip-compile", action="store_true")
    a = ap.parse_args()

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        print("Refusing: PYTORCH_ENABLE_MPS_FALLBACK=1 would hide unsupported MPS ops behind CPU fallback.")
        return 2
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0":
        print("Refusing: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 disables the MPS memory guard.")
        return 2

    import torch
    import torch.nn.functional as F

    run_id = "g0-mac-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {"kind": "capability_and_bandwidth_probe", "run_id": run_id, "utc_start": utc(), "host": host_info(),
               "software": {"torch": torch.__version__, "python_executable": sanitize(sys.executable),
                            "env": {k: os.environ.get(k) for k in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")}},
               "args": vars(a) | {"runs_dir": str(a.runs_dir)}, "probes": {}}
    P = R["probes"]

    mps_ok = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    P["mps_available"] = {"status": "ok" if mps_ok else "error", "is_built": torch.backends.mps.is_built(), "is_available": torch.backends.mps.is_available()}
    print(f"[{'ok' if mps_ok else 'error':5s}] mps_available built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}")
    if not mps_ok:
        (out_dir / "g0_probe.json").write_text(json.dumps(R, ensure_ascii=False, indent=2))
        return 1
    dev = torch.device("mps")
    sync = torch.mps.synchronize

    def memory_apis():
        out = {}
        for n in ("recommended_max_memory", "current_allocated_memory", "driver_allocated_memory"):
            fn = getattr(torch.mps, n, None)
            out[n] = int(fn()) if fn else None
        return out
    probe("memory_apis", P, memory_apis)

    def dtype_matmul():
        out = {}
        a0 = torch.randn(256, 512); b0 = torch.randn(512, 128); ref = a0 @ b0
        for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            try:
                y = (a0.to(dev, dt) @ b0.to(dev, dt)); sync()
                out[str(dt).split(".")[-1]] = {"supported": True, "rel_err_vs_cpu_fp32": rel_err(y, ref)}
            except Exception as e:
                out[str(dt).split(".")[-1]] = {"supported": False, "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}
        return out
    probe("dtype_matmul", P, dtype_matmul)

    def selection_ops():
        s = torch.randn(8, 4096); ref_v, ref_i = s.topk(64, dim=-1)
        v, i = s.to(dev).topk(64, dim=-1); sync()
        kv = torch.randn(8, 4096, 128); idx = torch.randint(0, 4096, (300,))
        g_ref = kv.index_select(1, idx); g = kv.to(dev).index_select(1, idx.to(dev)); sync()
        mask = torch.rand(8, 4096) < 0.5
        sm_ref = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        sm = torch.softmax(s.to(dev).masked_fill(~mask.to(dev), float("-inf")), -1); sync()
        return {"topk_indices_match": bool((i.cpu() == ref_i).all()), "topk_values_rel_err": rel_err(v, ref_v),
                "index_select_exact": bool(torch.equal(g.cpu(), g_ref)), "masked_softmax_rel_err": rel_err(sm, sm_ref),
                "masked_softmax_finite": bool(torch.isfinite(sm).all())}
    probe("selection_ops", P, selection_ops)

    def sdpa():
        out = {}
        B, H, G, N, d = 1, 32, 8, 1024, 128
        q = torch.randn(B, H, 1, d); k = torch.randn(B, G, N, d); v = torch.randn(B, G, N, d)
        k_rep = k.repeat_interleave(H // G, dim=1); v_rep = v.repeat_interleave(H // G, dim=1)
        ref = F.scaled_dot_product_attention(q, k_rep, v_rep)
        try:
            y = F.scaled_dot_product_attention(q.to(dev), k_rep.to(dev), v_rep.to(dev)); sync()
            out["mha_repeated_kv"] = {"supported": True, "rel_err": rel_err(y, ref)}
        except Exception as e:
            out["mha_repeated_kv"] = {"supported": False, "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}
        try:
            y = F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev), enable_gqa=True); sync()
            out["enable_gqa"] = {"supported": True, "rel_err": rel_err(y, ref)}
        except Exception as e:
            out["enable_gqa"] = {"supported": False, "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}
        mask = torch.zeros(1, 1, 1, N, dtype=torch.bool); mask[..., : N // 2] = True
        ref_m = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
        try:
            y = F.scaled_dot_product_attention(q.to(dev), k_rep.to(dev), v_rep.to(dev), attn_mask=mask.to(dev)); sync()
            out["bool_attn_mask"] = {"supported": True, "rel_err": rel_err(y, ref_m)}
        except Exception as e:
            out["bool_attn_mask"] = {"supported": False, "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}
        try:
            y = F.scaled_dot_product_attention(q.to(dev, torch.bfloat16), k_rep.to(dev, torch.bfloat16), v_rep.to(dev, torch.bfloat16)); sync()
            out["bf16"] = {"supported": True, "rel_err_vs_fp32": rel_err(y, ref)}
        except Exception as e:
            out["bf16"] = {"supported": False, "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}
        return out
    probe("sdpa", P, sdpa)

    def index_backward():
        torch.manual_seed(0)
        hq, d, r, N = 4, 128, 32, 2048
        q_cat = torch.randn(1, hq * d); k_pre = torch.randn(N, d)
        target = torch.softmax(torch.randn(1, N) * 3, -1)
        def run(device):
            Wq = torch.nn.Linear(hq * d, r, bias=False); Wk = torch.nn.Linear(d, r, bias=False)
            torch.manual_seed(1); torch.nn.init.normal_(Wq.weight, std=0.02); torch.nn.init.normal_(Wk.weight, std=0.02)
            Wq.to(device); Wk.to(device)
            u = Wq(q_cat.to(device)); z = Wk(k_pre.to(device))
            logp = torch.log_softmax(u @ z.T / math.sqrt(r), -1)
            loss = F.kl_div(logp, target.to(device), reduction="batchmean")
            loss.backward()
            return loss.item(), Wq.weight.grad.detach().cpu(), Wk.weight.grad.detach().cpu()
        l_cpu, gq_cpu, gk_cpu = run("cpu"); l_mps, gq_mps, gk_mps = run(dev); sync()
        return {"loss_cpu": l_cpu, "loss_mps": l_mps, "grad_finite": bool(torch.isfinite(gq_mps).all() and torch.isfinite(gk_mps).all()),
                "grad_wq_rel_err": rel_err(gq_mps, gq_cpu), "grad_wk_rel_err": rel_err(gk_mps, gk_cpu)}
    probe("index_backward_kl", P, index_backward)

    def dispatch_overhead():
        out = {}
        iters = 300
        x = torch.randn(8, 1, 32, device=dev); w = torch.randn(8, 32, 64, device=dev)
        s = torch.randn(8, 4096, device=dev); kv = torch.randn(8, 4096, 128, device=dev); idx = torch.randint(0, 4096, (64,), device=dev)
        ops = {"tiny_matmul": lambda: torch.matmul(x, w), "topk_4096_k32": lambda: torch.topk(s, 32, dim=-1), "index_select_64_of_4096": lambda: kv.index_select(1, idx)}
        for name, fn in ops.items():
            for _ in range(20): fn()
            sync(); t0 = time.perf_counter()
            for _ in range(iters): fn()
            sync(); out[name + "_us_per_op"] = (time.perf_counter() - t0) / iters * 1e6
        t0 = time.perf_counter()
        for _ in range(200): sync()
        out["synchronize_us"] = (time.perf_counter() - t0) / 200 * 1e6
        return out
    probe("dispatch_overhead", P, dispatch_overhead)

    def index_path_cost():
        """Per-layer decode cost of the index read path only: u·zᵀ over N, top-k (k=N/32), gather K and V rows.
        36 rotating layers so nothing stays cache-hot. Dense attention timing is G1a, not measured here."""
        out = {"dtype": "bfloat16", "layers": a.scan_layers, "rho": 1 / 32,
               "note": "us_per_layer medians over 8 passes; chain = score+topk+gather(K)+gather(V) in one stream"}
        G, d, r = 8, 128, 32
        for N in (8192, 16384, 32768):
            k = N // 32
            try:
                zs = [(torch.rand(G, N, r, device=dev) - 0.5).to(torch.bfloat16) for _ in range(a.scan_layers)]
                Ks = [torch.empty(G, N, d, dtype=torch.bfloat16, device=dev).fill_(0.01) for _ in range(a.scan_layers)]
                Vs = [torch.empty(G, N, d, dtype=torch.bfloat16, device=dev).fill_(0.02) for _ in range(a.scan_layers)]
                u = (torch.rand(G, 1, r, device=dev) - 0.5).to(torch.bfloat16)
                s0 = torch.matmul(u, zs[0].transpose(-1, -2)).squeeze(1)
                idx0 = torch.topk(s0, k, dim=-1).indices
                e0 = idx0[..., None].expand(G, k, d); sync()
                def score():
                    for z in zs: torch.matmul(u, z.transpose(-1, -2))
                def topk():
                    for _ in range(a.scan_layers): torch.topk(s0, k, dim=-1)
                def gather_kv():
                    for K, V in zip(Ks, Vs): K.gather(1, e0); V.gather(1, e0)
                def chain():
                    for z, K, V in zip(zs, Ks, Vs):
                        s = torch.matmul(u, z.transpose(-1, -2)).squeeze(1)
                        idx = torch.topk(s, k, dim=-1).indices
                        e = idx[..., None].expand(G, k, d)
                        K.gather(1, e); V.gather(1, e)
                # implementation variants of the same two primitives
                zTs = [z.transpose(-1, -2).contiguous() for z in zs]          # index keys stored [G, r, N]
                offs = (torch.arange(G, device=dev) * N)[:, None]
                rows0 = (idx0 + offs).reshape(-1)                              # flat row ids into [G*N, d]
                gi = torch.arange(G, device=dev)[:, None].expand(G, k)
                def score_zT_contig():
                    for zT in zTs: torch.matmul(u, zT)
                def gather_index_select_flat():
                    for K, V in zip(Ks, Vs): K.view(G * N, d).index_select(0, rows0); V.view(G * N, d).index_select(0, rows0)
                def gather_adv_index():
                    for K, V in zip(Ks, Vs): K[gi, idx0]; V[gi, idx0]
                def chain_best():
                    for zT, K, V in zip(zTs, Ks, Vs):
                        s = torch.matmul(u, zT).squeeze(1)
                        idx = torch.topk(s, k, dim=-1).indices
                        rows = (idx + offs).reshape(-1)
                        K.view(G * N, d).index_select(0, rows); V.view(G * N, d).index_select(0, rows)
                def timed(fn):
                    for _ in range(2): fn()
                    sync(); ts = []
                    for _ in range(8):
                        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
                    return statistics.median(ts) / a.scan_layers * 1e6
                out[f"N{N}"] = {"k": k, "z_bytes_per_layer": G * N * r * 2, "selected_kv_bytes_per_layer": 2 * G * k * d * 2,
                                "dense_kv_bytes_per_layer": 2 * G * N * d * 2,
                                "score_us_per_layer": timed(score), "score_zT_contig_us_per_layer": timed(score_zT_contig),
                                "topk_us_per_layer": timed(topk),
                                "gather_kv_us_per_layer": timed(gather_kv),
                                "gather_index_select_flat_us_per_layer": timed(gather_index_select_flat),
                                "gather_adv_index_us_per_layer": timed(gather_adv_index),
                                "chain_us_per_layer": timed(chain), "chain_best_us_per_layer": timed(chain_best)}
                del zs, zTs, Ks, Vs, u, s0, idx0, e0, rows0, gi; torch.mps.empty_cache()
            except Exception as e:
                out[f"N{N}"] = {"status": "error", "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}; torch.mps.empty_cache()
        return out
    probe("index_path_cost", P, index_path_cost)

    def bandwidth():
        out = {"note": "copy counts read+write bytes; sum counts read bytes; mul_out counts read+write. Median of repeats; synchronize around each op."}
        dts = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]
        for gib in a.sizes_gib:
            nbytes = int(gib * GiB)
            for dname, dt in dts:
                es = torch.tensor([], dtype=dt).element_size(); n = nbytes // es
                key = f"{gib:g}GiB_{dname}"
                try:
                    x = torch.empty(n, dtype=dt, device=dev).fill_(1.0); y = torch.empty_like(x); sync()
                    reps = a.repeats if gib <= 1 else max(5, a.repeats // 2)
                    def timed(fn, moved):
                        for _ in range(2): fn()
                        sync(); ts = []
                        for _ in range(reps):
                            sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
                        med = statistics.median(ts)
                        return {"gbps_median": moved / med / 1e9, "ms_median": med * 1e3, "ms_min": min(ts) * 1e3, "ms_max": max(ts) * 1e3, "repeats": reps}
                    out[key] = {"bytes": nbytes, "copy": timed(lambda: y.copy_(x), 2 * nbytes), "sum_read": timed(lambda: x.sum(), nbytes),
                                "mul_out": timed(lambda: torch.mul(x, 1.0001, out=y), 2 * nbytes)}
                    del x, y; torch.mps.empty_cache()
                except Exception as e:
                    out[key] = {"status": "error", "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}
                    torch.mps.empty_cache()
        return out
    probe("bandwidth", P, bandwidth)

    def attention_scan():
        """q[8,1,128] against K[8,N,128] per layer, bf16: the read pattern of one dense decode step across layers."""
        out = {"layers": a.scan_layers, "n": a.scan_n, "dtype": "bfloat16"}
        G, d = 8, 128
        per_layer = 2 * G * a.scan_n * d
        try:
            Ks = [torch.empty(G, a.scan_n, d, dtype=torch.bfloat16, device=dev).fill_(0.01) for _ in range(a.scan_layers)]
            q = torch.empty(G, 1, d, dtype=torch.bfloat16, device=dev).fill_(0.5); sync()
            def pass_rotating():
                for K in Ks: torch.matmul(q, K.transpose(-1, -2))
            def pass_hot():
                K = Ks[0]
                for _ in range(a.scan_layers): torch.matmul(q, K.transpose(-1, -2))
            def timed(fn):
                for _ in range(2): fn()
                sync(); ts = []
                for _ in range(8):
                    sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
                med = statistics.median(ts)
                return {"ms_median_per_pass": med * 1e3, "gbps_median": per_layer * a.scan_layers / med / 1e9, "us_per_layer": med / a.scan_layers * 1e6}
            out["k_bytes_per_layer"] = per_layer; out["k_bytes_total"] = per_layer * a.scan_layers
            out["rotating_workset"] = timed(pass_rotating); out["hot_same_layer"] = timed(pass_hot)
            del Ks, q; torch.mps.empty_cache()
        except Exception as e:
            out.update({"status": "error", "error_type": type(e).__name__, "error": sanitize(str(e))[:200]}); torch.mps.empty_cache()
        return out
    probe("attention_scan_qk", P, attention_scan)

    if not a.skip_compile:
        def compile_probe():
            def fn(x): return torch.relu(x * 2.0 + 1.0).sum()
            x = torch.randn(1 << 20, device=dev)
            ref = fn(x); t0 = time.perf_counter()
            cf = torch.compile(fn)
            y = cf(x); sync(); t1 = time.perf_counter()
            y2 = cf(x); sync(); t2 = time.perf_counter()
            return {"supported": True, "first_call_s": t1 - t0, "second_call_ms": (t2 - t1) * 1e3, "rel_err": rel_err(y, ref)}
        probe("torch_compile_elementwise", P, compile_probe)

    R["utc_end"] = utc()
    R["memory_after"] = {n: (int(getattr(torch.mps, n)()) if hasattr(torch.mps, n) else None) for n in ("current_allocated_memory", "driver_allocated_memory")}
    path = out_dir / "g0_probe.json"
    path.write_text(json.dumps(R, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {sanitize(str(path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
