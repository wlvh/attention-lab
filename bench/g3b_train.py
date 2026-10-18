#!/usr/bin/env python3
"""G3b-lite: train each new layer, reuse audited layer 13, share a four-hour budget."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attention_lab import runinfo
from attention_lab.campaign import CampaignBudget, ResourceGuard, digest, execution_identity, require_preflight, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "formal"), required=True)
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    runinfo.refuse_unsafe_env()
    import numpy as np
    import torch
    import transformers
    from attention_lab.g3a import body_digest, extract_features, load_features
    from attention_lab.g3b import layer_config, feature_fingerprint, verify_reuse, train_layer, evaluate_layer, summarize_rows
    from attention_lab.qwen_wrap import load_model, wrap_model, weights_file_sha256
    cfg_path = ROOT / "configs/mac_g3bg4lite.json"
    cfg = json.loads(cfg_path.read_text())
    git = runinfo.git_info(ROOT)
    if git["dirty_files"]:
        raise RuntimeError("commit execution/config changes before a real run")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.set_num_threads(4)
    versions = {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__}
    execution = execution_identity(ROOT, cfg_path, "bench/g3b_train.py", versions)
    execution["configs"]["configs/g3bg4lite_test_plan.json"] = digest(ROOT / "configs/g3bg4lite_test_plan.json")
    old_dir = ROOT / cfg["g3a_run"]
    old_manifest = json.loads((old_dir / "manifest.json").read_text())
    verify_reuse(ROOT, cfg, old_manifest)
    if digest(old_dir / "data.jsonl") != old_manifest["output_hashes"]["data.jsonl"]:
        raise RuntimeError("G3a data hash differs")
    if digest(old_dir / "index_final.npz") != old_manifest["final_checkpoint_sha256"]:
        raise RuntimeError("G3a checkpoint hash differs")
    docs = [json.loads(line) for line in (old_dir / "data.jsonl").read_text().splitlines()]
    for doc in docs:
        doc["split"] = "train" if doc["doc_index"] < cfg["train_docs"] else "development"
    layers = cfg["preflight_layers"] if args.mode == "preflight" else list(range(cfg["layers"]))
    if args.mode == "preflight":
        docs = docs[:cfg["preflight_docs"]]
    epochs = cfg["preflight_epochs"] if args.mode == "preflight" else cfg["epochs"]
    reused = int(cfg["reuse_layer_index"] in layers)
    planned = {"layers_delivered":len(layers), "new_layers_trained":len(layers)-reused, "reused_layers":reused,
               "new_feature_docs":(len(layers)-reused)*len(docs), "new_train_epochs":(len(layers)-reused)*epochs,
               "new_evaluation_docs":(len(layers)-reused)*len(docs)}
    completed = {key:0 for key in planned}
    if args.mode == "formal":
        if args.preflight is None:
            raise RuntimeError("explicit preflight required")
        pre = json.loads((args.preflight / "manifest.json").read_text())
        require_preflight(pre, execution)
    run_id = f'g3b-{runinfo.utc_compact()}-{args.mode}'
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    manifest = {"run_id":run_id,"stage":"G3b-lite","mode":args.mode,"status":"running","git":git,
                "utc_start":runinfo.utc_now(),"execution":execution,"config":cfg,"planned":planned,"completed":completed,
                "layers":{},"data_source":cfg["g3a_run"],"data_source_sha256":digest(old_dir/"data.jsonl"),
                "test_plan_sha256":execution["configs"]["configs/g3bg4lite_test_plan.json"],
                "feature_retention":cfg["feature_retention"]}
    write_json(out/"manifest.json",manifest)
    write_json(out/"config.json",cfg)
    (out/"data.jsonl").write_text("".join(json.dumps(d,ensure_ascii=False)+"\n" for d in docs))
    reason = None
    stage_start = time.monotonic()
    try:
        with CampaignBudget(ROOT,cfg,run_id) as budget:
            guard = ResourceGuard(budget,torch)
            if args.mode == "formal":
                guard.check(estimated_next=pre["projected_formal_seconds"]*1.2,reserve=cfg["inference_reserve_seconds"])
            guard.check(record=True)
            model,tok,info=load_model(cfg["model_id"],cfg["revision"],dtype=torch.bfloat16,device="mps")
            holder=wrap_model(model)
            manifest["model"],manifest["checkpoint"] = info,weights_file_sha256(cfg["model_id"],cfg["revision"])
            manifest["body_hash_before"] = body_digest(model)
            preflight_predictions=[]
            for layer in layers:
                lc=layer_config(cfg,layer)
                directory=out/f"layer-{layer:02d}";directory.mkdir()
                guard.check(record=True)
                if layer == cfg["reuse_layer_index"]:
                    for name in ("index_initial.npz","index_final.npz","training.json"):
                        shutil.copyfile(old_dir/name,directory/name)
                    rows=[json.loads(line) for line in (old_dir/"metrics.jsonl").read_text().splitlines()]
                    for row in rows:
                        row["split"]="train" if row["split"]=="train" else "development"
                        if int(row["doc_id"].rsplit("-",1)[1]) not in cfg["audit_feature_doc_indices"]:
                            row.pop("positions")
                    (directory/"metrics.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
                    write_json(directory/"metrics_summary.json",summarize_rows(rows))
                    layer_record={"status":"done","origin":"reused_G3a","source_run":cfg["g3a_run"],
                                  "source_manifest_sha256":digest(old_dir/"manifest.json"),"index_seed":lc["index_seed"]}
                    completed["reused_layers"]+=1
                else:
                    features, fingerprints, extract_times = {},{},[]
                    layer_start=time.monotonic()
                    if args.mode == "formal" and manifest["layers"]:
                        previous=[x["elapsed_seconds"] for x in manifest["layers"].values() if x.get("origin")=="new_training"]
                        if previous:
                            guard.check(estimated_next=max(previous)*1.2,reserve=cfg["inference_reserve_seconds"])
                    for doc in docs:
                        guard.check()
                        then=time.monotonic()
                        feat=extract_features(model,holder,doc,lc,guard.check)
                        extract_times.append(time.monotonic()-then)
                        features[doc["doc_id"]]=feat
                        fingerprints[doc["doc_id"]]=feature_fingerprint(feat)
                        if doc["doc_index"] in cfg["audit_feature_doc_indices"]:
                            np.savez_compressed(directory/f'audit-{doc["doc_index"]:02d}.npz',**{k:v.numpy() for k,v in feat.items()})
                        completed["new_feature_docs"]+=1
                    write_json(directory/"feature_hashes.json",fingerprints)
                    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
                        raise RuntimeError("body gradient attached")
                    def epoch_progress(epoch):
                        completed["new_train_epochs"]+=1
                        if epoch%10==0 or epoch==epochs:
                            write_json(out/"manifest.json",manifest)
                    initial,trained,training = train_layer(features,docs,lc,guard,directory,epochs,epoch_progress)
                    completed["new_layers_trained"]+=1
                    evaluate_layer(features,docs,lc,initial,trained,directory,guard)
                    completed["new_evaluation_docs"]+=len(docs)
                    layer_record={"status":"done","origin":"new_training","index_seed":lc["index_seed"],
                                  "elapsed_seconds":time.monotonic()-layer_start,"training":training,
                                  "feature_extract_seconds":extract_times}
                    if args.mode == "preflight":
                        steady=training["steady_step_seconds"]
                        preflight_predictions.append(max(extract_times[1:])*64 + (steady or 0.1)*48*50 + 15)
                    del features,feat,initial,trained
                    gc.collect();torch.mps.empty_cache()
                layer_record["files"]={p.name:{"sha256":digest(p),"bytes":p.stat().st_size} for p in directory.iterdir() if p.is_file()}
                write_json(directory/"layer_manifest.json",layer_record)
                manifest["layers"][str(layer)]=layer_record
                completed["layers_delivered"]+=1
                manifest["next_layer"] = next((l for l in layers if str(l) not in manifest["layers"]),None)
                write_json(out/"manifest.json",manifest)
                print(f'layer {layer+1}/28 {layer_record["origin"]}; complete={completed["layers_delivered"]}/{len(layers)}; remaining={budget.remaining:.0f}s',flush=True)
            manifest["body_hash_after"] = body_digest(model)
            if manifest["body_hash_after"] != manifest["body_hash_before"] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("body weights changed")
            manifest["memory_samples"] = guard.samples
            if args.mode == "preflight":
                manifest["projected_formal_seconds"] = max(preflight_predictions)*27 + 60
            if completed != planned:
                raise RuntimeError("incomplete planned counts")
    except BaseException as error:
        reason=f"{type(error).__name__}: {runinfo.sanitize(str(error))}"
        (out/"error.txt").write_text(runinfo.sanitize(traceback.format_exc()))
        print(reason,flush=True)
    finally:
        manifest.update(status="done" if reason is None and completed==planned else "incomplete",reason=reason,
                        utc_end=runinfo.utc_now(),elapsed_seconds=time.monotonic()-stage_start)
        manifest["completed"] = completed
        manifest["pending"] = {k:planned[k]-completed[k] for k in planned}
        write_json(out/"manifest.json",manifest)
        print(f'{run_id}: {manifest["status"]}',flush=True)
    return 0 if manifest["status"]=="done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
