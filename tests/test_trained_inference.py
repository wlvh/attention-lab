"""CPU contracts for frozen bank inference; no real checkpoint or MPS work."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
try:
    import numpy as np
    import pytest
    import torch
    import transformers
except ImportError as error:
    raise unittest.SkipTest("trained inference CPU contracts require torch/numpy/pytest/transformers") from error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from attention_lab.indexer import GroupedIndexer,project_queries,project_keys
from attention_lab.qwen_wrap import tiny_qwen3,wrap_model,geometry_of,LabRunner,LabState,make_cache
from attention_lab.trained import load_bank,make_trained_state


def setup_bank(tmp_path,model):
    geo=geometry_of(model);entries=[]
    for layer in range(geo.layers):
        idx=GroupedIndexer(geo.groups,geo.heads_per_group,geo.head_dim,r=4,seed=100+layer)
        path=tmp_path/f'layer-{layer}.npz'
        np.savez(path,wq=idx.wq.numpy(),wk=idx.wk.numpy())
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(dict(layer_index=layer,initial_file=path.name,final_file=path.name,initial_sha256=sha,final_sha256=sha))
    meta=dict(model_id="tiny-test",revision="fixture",r=4,layers=entries)
    meta["bank_id"]=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    return load_bank(tmp_path,meta,geo,device="cpu"),meta


@pytest.mark.parametrize('active_layers',[{0},None])
def test_single_and_all_layer_full_selection_equivalence(tmp_path,active_layers):
    model=tiny_qwen3(layers=2,dtype=torch.float64)
    holder=wrap_model(model);bank,_=setup_bank(tmp_path,model)
    state=make_trained_state(model,96,bank,sparse_layers=active_layers)
    state.record_selection=True
    for z in state.cache.z:z.fill_(float('nan'))
    full_state=LabState(make_cache(model,96,with_index=False),mode="full")
    a,d=LabRunner(model,holder,full_state),LabRunner(model,holder,state)
    prefix=torch.arange(1,81).view(1,-1)
    a.prefill(prefix,16);d.prefill(prefix,16)
    state.begin_request(.125,k0=96)
    for token in (81,82,83):
        expected=a.step(torch.tensor([[token]]));got=d.step(torch.tensor([[token]]))
        torch.testing.assert_close(got,expected,atol=1e-10,rtol=1e-10)
        for k,v in zip(a.cache.k,d.cache.k):torch.testing.assert_close(k[:,:,:a.cache.valid_length],v[:,:,:d.cache.valid_length],atol=1e-10,rtol=1e-10)
    assert state.cache.k[0].dtype==torch.float64 and state.cache.z[0].dtype==torch.float32
    assert all(torch.isfinite(z[...,:state.cache.valid_length]).all() for z in state.cache.z)
    if active_layers=={0}:assert state.last_rows[0] is not None and state.last_rows[1] is None
    else:assert all(x is not None for x in state.last_rows)


def test_mixed_precision_index_matches_explicit_fp32_formula(tmp_path):
    model=tiny_qwen3(layers=1,dtype=torch.float64);bank,_=setup_bank(tmp_path,model)
    idx=bank.indexes[0]
    q=torch.randn(1,2,2,1,16,dtype=torch.float64);k=torch.randn(1,2,80,16,dtype=torch.float64)
    scores=project_queries(q.float(),idx.wq_t_scaled)@project_keys(k.float(),idx.wk)
    qflat=q.float().permute(0,1,3,2,4).reshape(1,2,1,32)
    expected=(qflat@idx.wq.transpose(1,2).unsqueeze(0))@(idx.wk.unsqueeze(0)@k.float().transpose(-1,-2))/2
    torch.testing.assert_close(scores,expected,atol=1e-6,rtol=1e-5)


def test_parameter_mutation_rejects_before_cache_write(tmp_path):
    model=tiny_qwen3(layers=2);holder=wrap_model(model);bank,_=setup_bank(tmp_path,model)
    state=make_trained_state(model,96,bank);run=LabRunner(model,holder,state)
    run.prefill(torch.ones(1,80,dtype=torch.long),16);state.begin_request(.125)
    length=state.cache.valid_length;writes=list(state.cache.writes)
    bank.indexes[0].wk.add_(.01)
    with pytest.raises(RuntimeError,match="parameters changed"):run.step(torch.tensor([[2]]))
    assert state.cache.valid_length==length and state.cache.writes==writes


def test_bank_identity_replacement_and_parameter_file_tampering(tmp_path):
    model=tiny_qwen3(layers=2);holder=wrap_model(model);bank,meta=setup_bank(tmp_path,model)
    state=make_trained_state(model,96,bank);run=LabRunner(model,holder,state)
    run.prefill(torch.ones(1,80,dtype=torch.long),16)
    other=load_bank(tmp_path,meta,geometry_of(model),device="cpu",initial=True)
    mismatch=LabState(state.cache.clone(),other.indexes,mode="sparse",cache_guard=other.guard)
    mismatch.begin_request(.125)
    with pytest.raises(RuntimeError,match="different parameter identity"):
        LabRunner(model,holder,mismatch).step(torch.tensor([[2]]))
    state.indexers=list(other.indexes);state.begin_request(.125)
    with pytest.raises(RuntimeError,match="state index bank changed"):run.step(torch.tensor([[2]]))
    first=tmp_path/meta['layers'][0]['final_file'];first.write_bytes(first.read_bytes()+b'changed')
    with pytest.raises(ValueError,match="file hash mismatch"):load_bank(tmp_path,meta,geometry_of(model),device="cpu")


def test_independent_suffixes_and_cache_bytes(tmp_path):
    model=tiny_qwen3(layers=2);holder=wrap_model(model);bank,_=setup_bank(tmp_path,model)
    prefix=make_trained_state(model,96,bank);LabRunner(model,holder,prefix).prefill(torch.ones(1,80,dtype=torch.long),16)
    a=LabState(prefix.cache.clone(),bank.indexes,mode="sparse",cache_guard=bank.guard)
    b=LabState(prefix.cache.clone(),bank.indexes,mode="sparse",cache_guard=bank.guard)
    a.begin_request(.125);b.begin_request(.125)
    LabRunner(model,holder,a).step(torch.tensor([[2]]));LabRunner(model,holder,b).step(torch.tensor([[3]]))
    assert prefix.cache.valid_length==80 and a.cache.valid_length==b.cache.valid_length==81
    assert not torch.equal(a.cache.k[0][:,:,80],b.cache.k[0][:,:,80])
    assert a.cache.k[0].data_ptr()!=b.cache.k[0].data_ptr()
    for x,y in zip(prefix.cache.k,a.cache.k):torch.testing.assert_close(x[:,:,:80],y[:,:,:80])
    expected=sum(x.numel()*x.element_size() for x in prefix.cache.k+prefix.cache.v+prefix.cache.z)
    assert prefix.cache.bytes()['total_bytes']==expected


def test_index_cache_uses_pre_rope_keys_including_inactive_layers(tmp_path):
    model=tiny_qwen3(layers=2);holder=wrap_model(model);bank,_=setup_bank(tmp_path,model)
    state=make_trained_state(model,96,bank,sparse_layers={0});run=LabRunner(model,holder,state)
    captured=[[],[]];hooks=[]
    for layer,block in enumerate(model.model.layers):
        def record(module,args,output,l=layer):captured[l].append(output.detach().float().transpose(1,2))
        hooks.append(block.self_attn.inner.k_norm.register_forward_hook(record))
    try:
        run.prefill(torch.arange(1,81).view(1,-1),16);state.begin_request(.125)
        run.step(torch.tensor([[81]]));run.step(torch.tensor([[82]]))
    finally:
        for hook in hooks:hook.remove()
    for layer in range(2):
        keys=torch.cat(captured[layer],dim=2)[0].numpy().astype(np.float64)
        weights=bank.indexes[layer].wk.numpy().astype(np.float64)
        expected=np.einsum('grd,gnd->grn',weights,keys)
        np.testing.assert_allclose(state.cache.z[layer][0,...,:82].numpy(),expected,atol=1e-6,rtol=1e-5)


def test_canonical_order_preserves_selected_set_and_float64_attention(tmp_path):
    model=tiny_qwen3(layers=2,dtype=torch.float64);holder=wrap_model(model);bank,_=setup_bank(tmp_path,model)
    old=make_trained_state(model,96,bank);new=make_trained_state(model,96,bank)
    old.canonical_order=False
    old.record_selection=new.record_selection=True
    a,b=LabRunner(model,holder,old),LabRunner(model,holder,new)
    ids=torch.arange(1,81).view(1,-1)
    a.prefill(ids,16);b.prefill(ids,16)
    old.begin_request(.125);new.begin_request(.125)
    left=a.step(torch.tensor([[81]]));right=b.step(torch.tensor([[81]]))
    torch.testing.assert_close(left,right,atol=1e-10,rtol=1e-10)
    for x,y in zip(old.last_rows,new.last_rows):
        assert torch.equal(x.sort().values,y.sort().values)
        assert torch.equal(y,y.sort().values)
