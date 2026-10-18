"""G3b CPU fixtures only; never run the real model or MPS in these tests."""
import json
from pathlib import Path
import sys
import unittest
try:
    import torch
    import pytest
except ImportError as error:
    raise unittest.SkipTest("G3b contracts require torch and pytest") from error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from attention_lab.g3b import layer_config,feature_fingerprint,verify_reuse,train_layer,evaluate_layer


def test_layer_seed_and_existing_reuse_identity(tmp_path):
    cfg=json.loads((ROOT/"configs/mac_g3bg4lite.json").read_text())
    assert len({layer_config(cfg,l)["index_seed"] for l in range(28)})==28
    assert layer_config(cfg,13)["index_seed"]==1803
    import tarfile
    with tarfile.open(ROOT/"reports/provenance/g3a_evidence_20261018.tar.gz") as tar:
        m=json.loads(tar.extractfile("g3a-20261018T122927Z-formal/manifest.json").read())
    # Historical training reuse remains valid on its as-run sources. Later inference
    # changes must fail the future reuse gate, rather than weakening that gate.
    with tarfile.open(ROOT/"reports/provenance/g3b_evidence_20261018.tar.gz") as tar:
        for item in tar.getmembers():
            if item.name.startswith("source/") and item.isfile():
                target=tmp_path/item.name.removeprefix("source/")
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(tar.extractfile(item).read())
    verify_reuse(tmp_path,cfg,m)
    with pytest.raises(ValueError):verify_reuse(tmp_path,dict(cfg,epochs=49),m)
    with pytest.raises(ValueError):verify_reuse(ROOT,cfg,m)


def test_feature_hash_detects_mutation_and_shapes():
    a={"x":torch.arange(8,dtype=torch.float32).view(2,4)}
    first=feature_fingerprint(a)
    assert first==feature_fingerprint(a)
    a["x"][0,0]+=1
    assert first!=feature_fingerprint(a)
    assert first["x"]["shape"]==[2,4]


def test_layer_training_and_compact_metrics_cpu(tmp_path):
    from unittest.mock import patch
    from attention_lab.g3a import TrainableIndexer
    # Use the real orchestration with small tensors, not Qwen inference.
    torch.manual_seed(12)
    f={"q_pre":torch.randn(8,2,1,128),"q_main":torch.randn(8,2,1,128),
       "k_pre":torch.randn(8,96,128),"k_main":torch.randn(8,96,128),"v":torch.randn(8,96,128),
       "query_positions":torch.tensor([95]),"prefix_len":torch.tensor(80)}
    cfg=dict(r=32,index_seed=1803,training_device="cpu",rho=.125,fixed_prefix=4,fixed_recent=8,
             min_candidate_mass=1e-4,learning_rate=.001,gradient_clip_norm=1.,shuffle_seed=1805,uniform_seed=1804,audit_feature_doc_indices=[0])
    docs=[dict(doc_id="train",doc_index=0,split="train"),dict(doc_id="dev",doc_index=48,split="development")]
    class Guard:
        def check(self):pass
    epochs=[]
    initial,trained,record=train_layer({"train":f,"dev":f},docs,cfg,Guard(),tmp_path,2,epochs.append)
    assert epochs==[1,2] and record["steps"]==2 and min(record["parameter_change_l2"].values())>0
    result=evaluate_layer({"train":f,"dev":f},docs,cfg,initial,trained,tmp_path,Guard())
    rows=[json.loads(x) for x in (tmp_path/"metrics.jsonl").read_text().splitlines()]
    assert all("positions" in x for x in rows if x["split"]=="train")
    assert all("positions" not in x for x in rows if x["split"]=="development")
    assert result["aggregate"]["development"]["D"]["kl"] is not None
