#!/usr/bin/env python3
"""Freeze new final-test identities before any G3b/G3c/G4 model run; tokenizer only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def prepare(root=ROOT):
    from transformers import AutoTokenizer
    from attention_lab.g2lite_data import build_document
    from attention_lab.campaign import write_json, digest
    os.environ["HF_HUB_OFFLINE"] = "1"
    cfg = json.loads((root / "configs/mac_g3bg4lite.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    base = copy.deepcopy(json.loads((root / cfg["base_config"]).read_text()))
    base["data"]["seed"] = cfg["test_seed"]
    base["data"]["layout"]["length_tolerance_tokens"] = cfg["length_tolerance_tokens"]
    base["data"]["bucket_plan"]["T1"] = cfg["buckets_cycle"] * (cfg["test_docs"] // 4)
    docs = []
    for i in range(cfg["test_docs"]):
        d = build_document(tok, base, "T1", "2K", i)
        d["doc_id"], d["split"] = "G4lite-" + d["doc_id"], "final_test"
        docs.append(d)
    prior = [json.loads(line) for line in (root / cfg["g3a_run"] / "data.jsonl").read_text().splitlines()]
    with tarfile.open(root / "reports/provenance/g2lite_runs_20261017.tar.gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("/data.jsonl"):
                prior.extend(json.loads(line) for line in tar.extractfile(member))
    hashes = {d["prefix_sha256"] for d in docs}
    assert len(hashes) == cfg["test_docs"] and not hashes.intersection(d["prefix_sha256"] for d in prior)
    bindings = lambda d: hashlib.sha256(json.dumps([d["ids"],d["values"]]).encode()).hexdigest()
    assert not {bindings(d) for d in docs}.intersection(bindings(d) for d in prior)
    content = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    out = root / "runs/g3bg4lite-frozen-test-20261018"
    out.mkdir(exist_ok=True)
    data_path = out / "data.jsonl"
    if data_path.exists() and data_path.read_text() != content:
        raise RuntimeError("existing final test differs; refuse overwrite")
    data_path.write_text(content)
    plan = {"status_at_registration": "FROZEN_NOT_EVALUATED", "data_seed": cfg["test_seed"], "count": len(docs),
            "base_config_sha256": digest(root / cfg["base_config"]), "data_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "local_data": str(data_path.relative_to(root)), "known_old_prefix_count": len({d["prefix_sha256"] for d in prior}),
            "old_prefix_overlap": 0, "old_record_bindings_overlap": 0,
            "documents": [{k: d[k] for k in ("doc_id", "prefix_sha256", "prompt_sha256", "prefix_token_count", "buckets")} for d in docs]}
    target = root / "configs/g3bg4lite_test_plan.json"
    if target.exists() and json.loads(target.read_text()) != plan:
        raise RuntimeError("frozen plan differs; refuse overwrite")
    write_json(target, plan)
    print(json.dumps({"count":len(docs), "token_range":[min(d["prefix_token_count"] for d in docs),max(d["prefix_token_count"] for d in docs)],"data_sha256":plan["data_sha256"]}))


if __name__ == "__main__":
    prepare()
