#!/usr/bin/env python3
"""Explicit bounded G3a run. Never invoked by the teaching notebook. Offline only."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attention_lab import runinfo


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def completion_status(planned, completed, reason=None):
    return "done" if reason is None and planned == completed else "incomplete"


def new_documents(tok, cfg, count):
    from attention_lab.g2lite_data import build_document
    base = copy.deepcopy(json.loads((ROOT / cfg["base_config"]).read_text()))
    base["data"]["seed"] = cfg["data_seed"]
    base["data"]["bucket_plan"]["T1"] = (cfg["buckets_cycle"] * 16)
    docs = []
    for i in range(count):
        doc = build_document(tok, base, "T1", "2K", i)
        doc["doc_id"] = "G3a-" + doc["doc_id"]
        doc["split"] = "train" if i < cfg["train_docs"] else "heldout"
        docs.append(doc)
    hashes = [d["prefix_sha256"] for d in docs]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate new documents")
    old = set()
    archive = ROOT / "reports/provenance/g2lite_runs_20261017.tar.gz"
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.endswith("/data.jsonl"):
                for line in tar.extractfile(member):
                    item = json.loads(line)
                    if "prefix_sha256" in item:
                        old.add(item["prefix_sha256"])
    if not old or old.intersection(hashes):
        raise ValueError("old/new data separation could not be established")
    return docs, {"new_unique": len(hashes), "old_unique_checked": len(old), "overlap": 0,
                  "old_archive_sha256": runinfo.sha256_file(archive)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mac_g3a.json")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runinfo.refuse_unsafe_env()
    import numpy as np
    import psutil
    import torch
    import transformers
    from attention_lab.g3a import (TrainableIndexer, body_digest, evaluate_document, extract_features,
                                  load_features, loss_for_document, query_rows)
    from attention_lab.qwen_wrap import geometry_of, load_model, weights_file_sha256, wrap_model

    cfg = json.loads(args.config.read_text())
    cfg_hash = runinfo.sha256_file(args.config)
    git = runinfo.git_info(ROOT)
    if git["dirty_files"]:
        raise RuntimeError("commit all implementation/config changes before real runs")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no implicit backend change")
    torch.set_num_threads(4)
    ledger_path = ROOT / "runs/g3a-budget-20261018.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"used_seconds": 0.0, "runs": []}
    if ledger.get("active_pid"):
        try:
            os.kill(ledger["active_pid"], 0)
        except ProcessLookupError:
            raise RuntimeError("interrupted run ledger: reconcile its elapsed budget before resuming")
        raise RuntimeError("another registered G3a process is active")
    if ledger["used_seconds"] >= cfg["budget_seconds"]:
        raise RuntimeError("cumulative budget exhausted")
    if args.mode == "formal":
        if args.preflight is None:
            raise RuntimeError("formal requires an explicit completed preflight")
        pre = json.loads((args.preflight / "summary.json").read_text())
        if pre["status"] != "done" or pre["mode"] != "preflight" or pre["config_sha256"] != cfg_hash:
            raise RuntimeError("preflight incomplete or config differs")
        if pre["projected_formal_seconds"] * 1.5 > cfg["budget_seconds"] - ledger["used_seconds"]:
            raise RuntimeError("preflight projection exceeds remaining budget with 1.5x margin")
    run_id = f'g3a-{runinfo.utc_compact()}-{args.mode}'
    out = ROOT / "runs" / run_id
    (out / "features").mkdir(parents=True)
    start = time.monotonic()
    initial_swap = psutil.swap_memory().used
    ledger.update(active_pid=os.getpid(), active_run=run_id)
    write_json(ledger_path, ledger)
    count = cfg["preflight_docs"] if args.mode == "preflight" else cfg["train_docs"] + cfg["heldout_docs"]
    epochs = cfg["preflight_epochs"] if args.mode == "preflight" else cfg["epochs"]
    planned = {"documents": count, "epochs": epochs, "evaluated_documents": count}
    completed = {key: 0 for key in planned}
    manifest = {"run_id": run_id, "mode": args.mode, "status": "running", "utc_start": runinfo.utc_now(),
                "pid": os.getpid(), "git": git, "config": cfg, "config_sha256": cfg_hash,
                "base_config_sha256": runinfo.sha256_file(ROOT / cfg["base_config"]),
                "sources": runinfo.source_hashes(ROOT), "environment": runinfo.torch_env(torch),
                "transformers": transformers.__version__, "host": runinfo.host_info(),
                "offline": True, "planned": planned, "completed": completed, "feature_files": {},
                "memory_samples": [], "elapsed_budget_before_seconds": ledger["used_seconds"],
                "protocol": "full body A-step; bf16 captured features; CPU fp32 teacher/local outputs; MPS fp32 index training"}
    write_json(out / "manifest.json", manifest)
    write_json(out / "config.json", cfg)
    reason = None
    result = {}

    def guard():
        elapsed = time.monotonic() - start
        if ledger["used_seconds"] + elapsed >= cfg["budget_seconds"]:
            raise RuntimeError("budget_stop")
        mem = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        mps = torch.mps.driver_allocated_memory()
        if max(rss, mps) >= cfg["stop_bytes"] or mem.available < 4_000_000_000:
            raise RuntimeError("memory_limit")
        if psutil.swap_memory().used - initial_swap > cfg["swap_growth_stop_bytes"]:
            raise RuntimeError("swap_growth_stop")
        return {"rss": rss, "mps_driver": mps, "available": mem.available, "swap": psutil.swap_memory().used,
                "warning": max(rss, mps) >= cfg["warning_bytes"]}

    try:
        guard()
        model, tok, info = load_model(cfg["model_id"], cfg["revision"], dtype=torch.bfloat16, device="mps")
        holder = wrap_model(model)
        geo = geometry_of(model)
        if (geo.layers, geo.groups, geo.heads_per_group, geo.head_dim) != (28, 8, 2, 128):
            raise RuntimeError("unexpected model geometry")
        manifest["model"] = info
        manifest["checkpoint"] = weights_file_sha256(cfg["model_id"], cfg["revision"])
        manifest["body_hash_before"] = body_digest(model)
        docs, isolation = new_documents(tok, cfg, count)
        manifest["data_isolation"] = isolation
        with (out / "data.jsonl").open("w") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        extraction_start = time.monotonic()
        for i, doc in enumerate(docs):
            manifest["memory_samples"].append(guard())
            features = extract_features(model, holder, doc, cfg, guard)
            path = out / "features" / f'{doc["doc_id"]}.npz'
            np.savez_compressed(path, **{key: value.numpy() for key, value in features.items()})
            manifest["feature_files"][path.name] = {"sha256": runinfo.sha256_file(path), "bytes": path.stat().st_size,
                                                    "shapes": {key: list(value.shape) for key, value in features.items()}}
            completed["documents"] += 1
            print(f'features {i + 1}/{count} {doc["doc_id"]} P={doc["prefix_token_count"]}', flush=True)
            write_json(out / "manifest.json", manifest)
        extraction_seconds = time.monotonic() - extraction_start
        manifest["body_hash_after_extraction"] = body_digest(model)
        if manifest["body_hash_after_extraction"] != manifest["body_hash_before"] or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("body_mutated_or_gradient_attached")
        manifest["body_released_before_training"] = True
        del model, holder, features
        torch.mps.empty_cache()
        training_docs = [d for d in docs if d["split"] == "train"]
        # Heldout feature files are not used to form a loss or select a checkpoint.
        training_rows = [list(query_rows(load_features(out / "features" / f'{d["doc_id"]}.npz'), cfg)) for d in training_docs]
        trained = TrainableIndexer(r=cfg["r"], seed=cfg["index_seed"], device=cfg["training_device"])
        initial = TrainableIndexer(r=cfg["r"], seed=cfg["index_seed"])
        initial.save(out / "index_initial.npz")
        optimizer = torch.optim.Adam(trained.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        manifest["optimizer_parameters"] = {name: list(p.shape) for name, p in trained.named_parameters()}
        manifest["trainable_parameters"] = sum(p.numel() for p in trained.parameters())
        manifest["eligible_train_rows"] = sum(int(row["eligible"].sum()) for rows in training_rows for row in rows)
        manifest["skipped_train_rows"] = sum(int((~row["eligible"]).sum()) for rows in training_rows for row in rows)
        if manifest["eligible_train_rows"] == 0:
            raise RuntimeError("no_eligible_training_rows")
        trace = []
        train_start = time.monotonic()
        rng = np.random.default_rng(cfg["shuffle_seed"])
        for epoch in range(1, epochs + 1):
            losses, gradnorms = [], []
            for di in rng.permutation(len(training_docs)):
                guard()
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for_document(trained, training_rows[di])
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite_loss")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trained.parameters()):
                    raise RuntimeError("disconnected_or_nonfinite_gradient")
                norm = torch.nn.utils.clip_grad_norm_(trained.parameters(), cfg["gradient_clip_norm"])
                if not torch.isfinite(norm) or (epoch == 1 and float(norm) == 0):
                    raise RuntimeError("nonfinite_or_zero_initial_gradient")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in trained.parameters()):
                    raise RuntimeError("nonfinite_parameters")
                losses.append(float(loss.detach().cpu()))
                gradnorms.append(float(norm.cpu()))
            trace.append({"epoch": epoch, "online_train_kl": float(np.mean(losses)), "updates": len(losses),
                          "gradient_norm_min": min(gradnorms), "gradient_norm_max": max(gradnorms)})
            completed["epochs"] = epoch
            write_json(out / "training.json", trace)
            print(f'epoch {epoch}/{epochs} online_train_KL={trace[-1]["online_train_kl"]:.6f}', flush=True)
        train_seconds = time.monotonic() - train_start
        trained.save(out / "index_final.npz")
        manifest["final_checkpoint_sha256"] = runinfo.sha256_file(out / "index_final.npz")
        if manifest["final_checkpoint_sha256"] == runinfo.sha256_file(out / "index_initial.npz"):
            raise RuntimeError("index_parameters_unchanged")
        # Freeze final checkpoint BEFORE any heldout metric is computed.
        manifest["checkpoint_frozen_utc"] = runinfo.utc_now()
        write_json(out / "manifest.json", manifest)
        trained_cpu = TrainableIndexer(r=cfg["r"], seed=cfg["index_seed"])
        trained_cpu.restore(out / "index_final.npz")
        del optimizer, trained, training_rows
        torch.mps.empty_cache()
        eval_start = time.monotonic()
        with (out / "metrics.jsonl").open("w") as f:
            for doc in docs:
                guard()
                feature = load_features(out / "features" / f'{doc["doc_id"]}.npz')
                for row in evaluate_document(doc["doc_id"], feature, cfg, initial, trained_cpu):
                    row["split"] = doc["split"]
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                completed["evaluated_documents"] += 1
        eval_seconds = time.monotonic() - eval_start
        result = {"extraction_seconds": extraction_seconds, "training_seconds": train_seconds, "evaluation_seconds": eval_seconds,
                  "projected_formal_seconds": (extraction_seconds + eval_seconds) / count * 64 + train_seconds / (epochs * len(training_docs)) * 50 * 48 + 120}
        manifest["memory_samples"].append(guard())
    except Exception as error:
        reason = f"{type(error).__name__}: {runinfo.sanitize(str(error))}"
        (out / "error.txt").write_text(runinfo.sanitize(traceback.format_exc()))
        print(reason, flush=True)
    finally:
        elapsed = time.monotonic() - start
        status = completion_status(planned, completed, reason)
        manifest.update(status=status, reason=reason, completed=completed.copy(), elapsed_seconds=elapsed, utc_end=runinfo.utc_now())
        manifest["output_hashes"] = {p.name: runinfo.sha256_file(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
        write_json(out / "summary.json", {"run_id": run_id, "mode": args.mode, "status": status, "reason": reason,
                                        "config_sha256": cfg_hash, "planned": planned, "completed": completed,
                                        "pending": {k: planned[k] - completed[k] for k in planned}, "elapsed_seconds": elapsed, **result})
        ledger.update(used_seconds=ledger["used_seconds"] + elapsed, active_pid=None, active_run=None)
        ledger["runs"].append({"run_id": run_id, "elapsed_seconds": elapsed, "status": status})
        write_json(ledger_path, ledger)
        print(f'{run_id}: {status}; cumulative device-run wall time {ledger["used_seconds"]:.1f}/7200s', flush=True)
    return 0 if status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
