#!/usr/bin/env python3
"""G1b numerics: did the wrapper change what the model computes? Three-way comparison on the pinned Qwen3 checkpoint.

For each dtype (float32, bfloat16) on MPS and float32 on CPU, and each length L in {32, 128, 512} (random token ids):
  (1) unmodified model, no cache, one forward over L positions            -> logits at every position (reference)
      unmodified model with its own DynamicCache                          -> K/V per layer (cache reference)
  (2) wrapped model, full read, one forward over L positions              -> logits + static cache vs (1)
  (3) wrapped model + static cache fed chunk by chunk then token by token -> logits of the last 8 positions + cache vs (1)
  (4) index path D with every candidate selected (k0 = capacity)         -> logits vs (3) and (1)
  plus: D at rho = 1/8 (recorded, not required to match), rollback + re-decode, chunked vs one-shot index table build,
  a short-prompt reference (top-5 logits, greedy continuation) and a free-generation smoke (text only, no quality claim).

Tolerances come from configs/mac_gate1b.json and were written before this script first ran. Results go to
runs/<run_id>/results.json with a manifest; scripts/g1b_report.py condenses them into reports/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from attention_lab import runinfo  # noqa: E402

FIRST_SPECIAL_TOKEN = 151643   # Qwen3 vocabulary: ids below this are ordinary BPE tokens


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/mac_gate1b.json")
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--dtypes", nargs="+", default=None, choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cpu-length", type=int, default=None, help="float32 CPU reference length (0 disables)")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--tail", type=int, default=8, help="tokens decoded one at a time after the chunked prefill")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-tokens", type=int, default=32)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="smoke test of the pipeline: shortest length only, marked quick")
    return ap.parse_args()


def compare(a, b, tol, *, kind):
    """max|a-b|, relative norm and argmax agreement, computed in float32 on the CPU, with the preset verdict."""
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    finite = bool(torch.isfinite(a32).all())
    diff = float((a32 - b32).abs().max()) if finite else float("inf")
    rel = float((a32 - b32).norm() / (b32.norm() + 1e-30)) if finite else float("inf")
    out = {"finite": finite, "max_abs": diff, "rel_norm": rel}
    if kind == "logits":
        agree = float((a32.argmax(-1) == b32.argmax(-1)).float().mean())
        out.update({"argmax_agreement": agree, "tol_atol": tol["logits_atol"], "tol_rel_norm": tol["logits_rel_norm"],
                    "tol_argmax_min": tol["argmax_agreement_min"],
                    "pass": finite and diff <= tol["logits_atol"] and rel <= tol["logits_rel_norm"] and agree >= tol["argmax_agreement_min"]})
    else:
        out.update({"tol_atol": tol["cache_atol"], "pass": finite and diff <= tol["cache_atol"]})
    return out


def cache_compare(cache, dc, L, tol):
    """Per-layer K/V of the static cache against transformers' DynamicCache: the worst layer decides."""
    from attention_lab.qwen_wrap import hf_cache_kv
    worst_k = worst_v = 0.0
    finite = True
    for l in range(cache.layers):
        k, v = hf_cache_kv(dc, l)
        ck, cv = cache.k[l][:, :, :L].float().cpu(), cache.v[l][:, :, :L].float().cpu()
        finite = finite and bool(torch.isfinite(ck).all()) and bool(torch.isfinite(cv).all())
        worst_k = max(worst_k, float((ck - k.float().cpu()).abs().max()))
        worst_v = max(worst_v, float((cv - v.float().cpu()).abs().max()))
    return {"finite": finite, "max_abs_k": worst_k, "max_abs_v": worst_v, "tol_atol": tol["cache_atol"],
            "pass": finite and max(worst_k, worst_v) <= tol["cache_atol"]}


def poison(cache):
    L = cache.valid_length
    for l in range(cache.layers):
        cache.k[l][:, :, L:] = float("nan")
        cache.v[l][:, :, L:] = float("nan")
        if cache.z is not None:
            cache.z[l][..., L:] = float("nan")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def run_length(model, holder, device, dtype_name, L, tol, a, results):
    """All comparisons for one (dtype, L). The model arrives unwrapped and leaves unwrapped."""
    from transformers import DynamicCache
    from attention_lab.qwen_wrap import LabRunner, make_state, unwrap_model, wrap_model
    gen = torch.Generator().manual_seed(a.seed + L)
    ids = torch.randint(0, FIRST_SPECIAL_TOKEN, (1, L), generator=gen).to(device)
    tail, chunk, cap = a.tail, a.chunk, L + 16
    rec = {"dtype": dtype_name, "device": str(device), "L": L, "capacity": cap, "chunk": chunk, "tail": tail,
           "data_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(), "checks": {}}
    t0 = time.perf_counter()
    ref = model(input_ids=ids).logits                                          # (1) unmodified, no cache
    dc = DynamicCache(config=model.config)
    model(input_ids=ids, past_key_values=dc, use_cache=True)                  # (1) unmodified with its own cache
    sync(device)
    wrap_model(model, holder)
    # (2) wrapped full path, one forward
    stA = make_state(model, cap, "full")
    runA = LabRunner(model, holder, stA)
    poison(stA.cache)
    one = runA.prefill(ids, L, keep_logits=True)
    rec["checks"]["wrapped_one_shot_vs_unmodified_logits"] = compare(one, ref, tol, kind="logits")
    rec["checks"]["wrapped_one_shot_cache_vs_hf_cache"] = cache_compare(stA.cache, dc, L, tol)
    # (3) wrapped + static cache, chunked prefill then token by token
    stB = make_state(model, cap, "full")
    runB = LabRunner(model, holder, stB)
    poison(stB.cache)
    head = runB.prefill(ids[:, :L - tail], chunk, keep_logits=True)
    tailB = torch.cat([runB.step(ids[:, i:i + 1]) for i in range(L - tail, L)], dim=1)
    rec["checks"]["wrapped_incremental_vs_unmodified_logits"] = compare(torch.cat([head, tailB], 1), ref, tol, kind="logits")
    rec["checks"]["wrapped_incremental_last_tokens_vs_unmodified_logits"] = compare(tailB, ref[:, L - tail:], tol, kind="logits")
    rec["checks"]["wrapped_incremental_cache_vs_hf_cache"] = cache_compare(stB.cache, dc, L, tol)
    rec["checks"]["wrapped_incremental_vs_one_shot_logits"] = compare(torch.cat([head, tailB], 1), one, tol, kind="logits")
    # rollback: same tokens again must reproduce the first pass; the rolled-back suffix is poisoned first
    runB.rollback(L - tail)
    poison(stB.cache)
    again = torch.cat([runB.step(ids[:, i:i + 1]) for i in range(L - tail, L)], dim=1)
    rb = compare(again, tailB, tol, kind="logits")
    rb["bitwise_equal"] = bool(torch.equal(again, tailB))
    rec["checks"]["rollback_redecode_vs_first_pass"] = rb
    # (4) D with every candidate selected, and D at rho = 1/8 (recorded only)
    stD = make_state(model, cap, "sparse")
    runD = LabRunner(model, holder, stD)
    poison(stD.cache)
    runD.prefill(ids[:, :L - tail], chunk)
    stD.begin_request(None, k0=cap)
    tailD = torch.cat([runD.step(ids[:, i:i + 1]) for i in range(L - tail, L)], dim=1)
    rec["checks"]["D_full_selection_vs_wrapped_incremental_logits"] = compare(tailD, tailB, tol, kind="logits")
    rec["checks"]["D_full_selection_vs_unmodified_logits"] = compare(tailD, ref[:, L - tail:], tol, kind="logits")
    rec["checks"]["D_full_selection_cache_vs_hf_cache"] = cache_compare(stD.cache, dc, L, tol)
    runD.rollback(L - tail)
    k0 = stD.begin_request(1 / 8)
    tailP = torch.cat([runD.step(ids[:, i:i + 1]) for i in range(L - tail, L)], dim=1)
    partial = compare(tailP, ref[:, L - tail:], tol, kind="logits")
    partial.update({"k0": k0, "k_last": stD.plan.k, "selected_last": stD.plan.selected_count, "legal_last": stD.plan.legal_length,
                    "gated": False, "pass": None, "note": "partial read is not required to match the full path; recorded for information"})
    rec["checks"]["D_rho_1_8_vs_unmodified_logits_information_only"] = partial
    # chunked vs one-shot index table (and K/V) build
    stE = make_state(model, cap, "sparse")
    LabRunner(model, holder, stE).prefill(ids, L)
    worst = {"k": 0.0, "v": 0.0, "z": 0.0}
    for l in range(stD.cache.layers):
        stF = stE.cache
        worst["k"] = max(worst["k"], float((stD.cache.k[l][:, :, :L].float() - stF.k[l][:, :, :L].float()).abs().max()))
        worst["v"] = max(worst["v"], float((stD.cache.v[l][:, :, :L].float() - stF.v[l][:, :, :L].float()).abs().max()))
        worst["z"] = max(worst["z"], float((stD.cache.z[l][..., :L].float() - stF.z[l][..., :L].float()).abs().max()))
    rec["checks"]["chunked_vs_one_shot_build_kv_and_index"] = {"max_abs_k": worst["k"], "max_abs_v": worst["v"], "max_abs_z": worst["z"],
                                                              "tol_atol": tol["cache_atol"], "pass": max(worst.values()) <= tol["cache_atol"],
                                                              "note": "D cache: chunked prefill + 8 single steps vs one forward over L"}
    unwrap_model(model)
    sync(device)
    rec["seconds"] = time.perf_counter() - t0
    rec["gated_pass"] = all(c["pass"] for c in rec["checks"].values() if c.get("pass") is not None)
    results["lengths"].append(rec)
    flags = " ".join(f"{k}={'ok' if c['pass'] else ('--' if c['pass'] is None else 'FAIL')}" for k, c in rec["checks"].items())
    print(f"  {dtype_name:<9s} {str(device):<4s} L={L:<4d} gated_pass={rec['gated_pass']}  {flags}", flush=True)
    return rec


def reference_prompt(model, tok, device, dtype_name, results, n_new=16):
    """Short-prompt numerical reference of the unmodified model: top-5 logits and a greedy continuation."""
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    logits = model(input_ids=ids).logits[0, -1].float()
    top = torch.topk(logits, 5)
    gen = model.generate(ids, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    rec = {"dtype": dtype_name, "device": str(device), "prompt": prompt, "prompt_ids": ids[0].tolist(),
           "top5_ids": top.indices.cpu().tolist(), "top5_logits": [round(x, 4) for x in top.values.cpu().tolist()],
           "top5_tokens": [tok.decode([i]) for i in top.indices.cpu().tolist()],
           "greedy_ids": gen[0, ids.shape[1]:].tolist(), "greedy_text": text}
    results["reference_prompt"].append(rec)
    print(f"  reference {dtype_name} {device}: top1 {rec['top5_tokens'][0]!r} logit {rec['top5_logits'][0]}  greedy: {text[:60]!r}")
    return rec


def smoke_generation(model, tok, holder, device, dtype_name, results, n_new):
    """Free greedy generation after a ~600-token prompt: native vs wrapped full vs D (rho 1/8). Text only; no quality claim."""
    from transformers import DynamicCache
    from attention_lab.qwen_wrap import LabRunner, NativeRunner, make_state, unwrap_model, wrap_model
    base = ("Attention Lab studies the trade-off between reading every past position and reading an indexed subset. "
            "The body of the model is frozen; only how it reads its own history changes. ")
    msgs = [{"role": "user", "content": base * 12 + "\nSummarise the paragraph above in one sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"].to(device)
    L = ids.shape[1]
    cap = L + n_new + 8
    outs = {}
    nat = NativeRunner(model, DynamicCache(config=model.config), cap)
    logits = nat.prefill(ids, 256)
    seq = []
    for _ in range(n_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        seq.append(int(nxt))
        logits = nat.step(nxt)
    outs["native_dynamic"] = seq
    wrap_model(model, holder)
    for mode, rho in (("full", None), ("sparse", 1 / 8)):
        st = make_state(model, cap, mode)
        run = LabRunner(model, holder, st)
        logits = run.prefill(ids, 256)
        if mode == "sparse":
            st.begin_request(rho)
        seq = []
        for _ in range(n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            seq.append(int(nxt))
            logits = run.step(nxt)
        outs["wrapped_full" if mode == "full" else "D_rho_1_8"] = seq
        if mode == "sparse":
            k_info = {"k0": st.k0, "k_last": st.plan.k, "selected_last": st.plan.selected_count, "legal_last": st.plan.legal_length}
    unwrap_model(model)
    rec = {"dtype": dtype_name, "device": str(device), "prompt_tokens": L, "new_tokens": n_new, "sparse": k_info,
           "texts": {k: tok.decode(v, skip_special_tokens=True) for k, v in outs.items()}, "ids": outs,
           "first_divergence_step": {k: next((i for i, (x, y) in enumerate(zip(v, outs["native_dynamic"])) if x != y), None) for k, v in outs.items()},
           "note": "smoke only: greedy text agreement is not a correctness criterion and no task quality is evaluated"}
    results["smoke_generation"].append(rec)
    for k, v in rec["texts"].items():
        print(f"  smoke {dtype_name} {k:<14s} divergence@{rec['first_divergence_step'][k]}: {v[:80]!r}")


def main() -> int:
    a = parse_args()
    try:
        runinfo.refuse_unsafe_env()
    except RuntimeError as e:
        print(f"Refusing: {e}")
        return 2
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    tols = cfg["numerics"]["tolerances"]
    lengths = a.lengths or cfg["numerics"]["lengths"]
    dtypes = a.dtypes or cfg["numerics"]["dtypes"]
    cpu_len = cfg["numerics"]["cpu_float32_reference_length"] if a.cpu_length is None else a.cpu_length
    if a.quick:
        lengths, dtypes, cpu_len = lengths[:1], dtypes[:1], 0
    device = torch.device(a.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS not available")
        return 1
    from attention_lab.qwen_wrap import StateHolder, load_model, weights_file_sha256
    run_id = "g1b-numerics-" + runinfo.utc_compact() + ("-quick" if a.quick else "") + (f"-{a.tag}" if a.tag else "")
    out_dir = a.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = cfg["model"]
    manifest = {"schema_version": 1, "run_id": run_id, "kind": "correctness", "status": "running", "created_at_utc": runinfo.utc_now(),
                "git": runinfo.git_info(ROOT), "source_hashes": runinfo.source_hashes(ROOT), "config_sha256": runinfo.sha256_file(a.config),
                "model_revision": model_cfg["revision"], "tokenizer_revision": model_cfg["tokenizer_revision"], "model_id": model_cfg["id"],
                "weights": weights_file_sha256(model_cfg["id"], model_cfg["revision"]), "data_sha256": None, "seed": a.seed,
                "device": str(device), "host": runinfo.host_info(), "software": None, "backend": device.type, "dtype": dtypes,
                "command": "python " + " ".join(runinfo.sanitize(x) for x in sys.argv), "timing_scope": "no timing: numerics only",
                "memory_units": "bytes", "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": None},
                "artifacts": ["results.json", "manifest.json"], "args": {k: (runinfo.sanitize(str(v)) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "limitations": ["random token ids and one short prompt; equivalence of computation, not task quality"]}
    manifest["software"] = runinfo.torch_env(torch)
    import transformers
    manifest["software"]["transformers"] = transformers.__version__
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_id {run_id}  torch {torch.__version__}  transformers {transformers.__version__}")
    results = {"run_id": run_id, "quick": a.quick, "utc_start": runinfo.utc_now(), "config": cfg["numerics"], "model_info": None,
               "reference_prompt": [], "lengths": [], "smoke_generation": [], "cpu_vs_mps_unmodified": None}
    t_start = time.perf_counter()
    holder = StateHolder()
    with torch.inference_mode():
        for dtype_name in dtypes:
            dtype = getattr(torch, dtype_name)
            print(f"\n== {dtype_name} on {device} ==")
            model, tok, info = load_model(model_cfg["id"], model_cfg["revision"], dtype=dtype, device=device)
            if results["model_info"] is None:
                results["model_info"] = info
                mism = {k: (info.get(k), v) for k, v in model_cfg["expected_from_config_json"].items() if info.get(k) is not None and info.get(k) != v}
                results["model_info"]["mismatch_vs_config_json"] = mism
                if mism:
                    print(f"  WARNING config mismatch: {mism}")
            reference_prompt(model, tok, device, dtype_name, results)
            for L in lengths:
                run_length(model, holder, device, dtype_name, L, tols[dtype_name], a, results)
            if not a.skip_smoke and dtype_name == dtypes[-1]:
                smoke_generation(model, tok, holder, device, dtype_name, results, a.smoke_tokens)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        if cpu_len:
            print(f"\n== float32 on cpu (reference length {cpu_len}) ==")
            cpu = torch.device("cpu")
            model, tok, _ = load_model(model_cfg["id"], model_cfg["revision"], dtype=torch.float32, device=cpu)
            reference_prompt(model, tok, cpu, "float32", results)
            rec = run_length(model, holder, cpu, "float32", cpu_len, tols["float32"], a, results)
            # platform noise floor of the unmodified model: MPS float32 vs CPU float32 on the same tokens (information only)
            if device.type == "mps" and "float32" in dtypes:
                gen = torch.Generator().manual_seed(a.seed + cpu_len)
                ids = torch.randint(0, FIRST_SPECIAL_TOKEN, (1, cpu_len), generator=gen)
                cpu_logits = model(input_ids=ids).logits
                del model
                model, _, _ = load_model(model_cfg["id"], model_cfg["revision"], dtype=torch.float32, device=device)
                mps_logits = model(input_ids=ids.to(device)).logits
                c = compare(mps_logits, cpu_logits, tols["float32"], kind="logits")
                c.update({"gated": False, "pass": None, "note": "unmodified model, MPS float32 vs CPU float32: platform noise floor for the float32 tolerances"})
                results["cpu_vs_mps_unmodified"] = c
                print(f"  unmodified MPS vs CPU float32 L={cpu_len}: max_abs {c['max_abs']:.3e} rel {c['rel_norm']:.3e} argmax agreement {c['argmax_agreement']:.3f}")
            del model
    results["utc_end"] = runinfo.utc_now()
    results["device_seconds"] = time.perf_counter() - t_start
    gated = [c for rec in results["lengths"] for c in rec["checks"].values() if c.get("pass") is not None]
    results["numerics_pass"] = all(c["pass"] for c in gated) and bool(gated)
    results["gated_checks"] = len(gated)
    results["failed_checks"] = [(rec["dtype"], str(rec["device"]), rec["L"], name) for rec in results["lengths"] for name, c in rec["checks"].items() if c.get("pass") is False]
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest.update({"status": "done", "finished_at_utc": results["utc_end"], "budget": {"cloud_usd_actual": 0, "cloud_usd_committed": 0, "device_seconds": results["device_seconds"]}})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nnumerics_pass={results['numerics_pass']} ({len(gated)} gated checks, failed: {results['failed_checks']})  wrote {runinfo.sanitize(str(out_dir))}  ({results['device_seconds']:.0f} s)")
    return 0 if results["numerics_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
