"""Shared real-inference routines for G3c diagnostics and G4-lite quality/cost.

Generation semantics are explicit: free_generate stops at EOS, fixed_greedy_tokens
ignores EOS only to define controlled diagnostic inputs. Never score the latter.
"""
from __future__ import annotations
import hashlib
import time
import numpy as np
import torch
from .qwen_wrap import LabRunner,make_state,geometry_of
from .trained import make_trained_state
from .g2lite_control import uniform_seed_for


def runner_for(model,holder,capacity,method,trained=None,initial=None,*,doc=None,uniform_seed=1804):
    if method in ("A","A-best"):
        state=make_state(model,capacity,"full")
    elif method in ("B","C"):
        seed=uniform_seed_for({"seeds":{"uniform_base":uniform_seed}},doc)
        state=make_state(model,capacity,"sparse",selector="recent" if method=="B" else "uniform",uniform_seed=seed)
    elif method in ("D","D-one","D0"):
        state=make_trained_state(model,capacity,initial if method=="D0" else trained,
                                 sparse_layers={13} if method=="D-one" else None)
    else:raise ValueError(method)
    return LabRunner(model,holder,state)


@torch.no_grad()
def question_forward(runner,suffix,*,batch=False):
    if batch:return runner.forward(suffix,phase="prefill")
    logits=None
    for i in range(suffix.shape[1]):logits=runner.step(suffix[:,i:i+1])
    return logits


@torch.no_grad()
def free_generate(runner,logits,*,max_new,eos,guard=lambda:None):
    generated=[];stopped=False
    for i in range(max_new):
        guard()
        token=int(logits[0,-1].argmax().item());generated.append(token)
        if token in eos:stopped=True;break
        if i+1<max_new:
            logits=runner.step(torch.tensor([[token]],device=runner.cache.device))
    return generated,stopped


@torch.no_grad()
def fixed_greedy_tokens(runner,logits,length,guard=lambda:None):
    """Controlled continuation: greedy for exactly length tokens, ignores EOS; NOT quality."""
    tokens=[]
    for i in range(length):
        guard();token=int(logits[0,-1].argmax().item());tokens.append(token)
        if i+1<length:logits=runner.step(torch.tensor([[token]],device=runner.cache.device))
    return tokens


def prefix_digest(cache,length):
    h=hashlib.sha256()
    for arrays in (cache.k,cache.v):
        for x in arrays:h.update(x[:,:,:length].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def selection_bitsets(state):
    """Actual original positions; full/inactive layers are explicitly all legal rows."""
    cache=state.cache;legal=cache.valid_length
    bits=np.zeros((cache.layers,cache.groups,cache.capacity),dtype=np.uint8)
    counts=[]
    for layer,rows in enumerate(state.last_rows):
        if rows is None:
            bits[layer,:,:legal]=1;counts.append(legal)
        else:
            count=state.plan.selected_count
            positions=(rows.view(1,cache.groups,count)-cache.offsets).cpu().numpy()[0]
            if positions.min()<0 or positions.max()>=legal:raise RuntimeError("invalid selected position")
            for group in range(cache.groups):bits[layer,group,positions[group]]=1
            if not np.all(bits[layer].sum(-1)==count):raise RuntimeError("duplicate selection")
            counts.append(count)
    return {"bitsets":np.packbits(bits,axis=-1),"counts":np.array(counts),"legal":np.array(legal),"capacity":np.array(cache.capacity)}


class StepCapture:
    """Only new-position Q/K and pre-o_proj output vectors; no NxN attention matrix."""
    def __init__(self,model):
        self.enabled=False;self.rows={};self.hooks=[]
        for layer,block in enumerate(model.model.layers):
            for kind,module in (("q",block.self_attn.inner.q_norm),("k",block.self_attn.inner.k_norm)):
                def capture(module,args,output,l=layer,k=kind):
                    if self.enabled:self.rows[l,k]=output.detach().float().cpu().flatten()
                self.hooks.append(module.register_forward_hook(capture))
            def output_capture(module,args,l=layer):
                if self.enabled:self.rows[l,"attention_output"]=args[0].detach().float().cpu().flatten()
            self.hooks.append(block.self_attn.inner.o_proj.register_forward_pre_hook(output_capture))
        self.layers=len(model.model.layers)

    def start(self):self.rows={};self.enabled=True
    def finish(self,logits):
        self.enabled=False
        result={kind:torch.stack([self.rows[layer,kind] for layer in range(self.layers)]) for kind in ("q","k","attention_output")}
        result["logits"]=logits.detach().float().cpu().flatten()
        if any(not torch.isfinite(v).all() for v in result.values()):raise RuntimeError("nonfinite captured state")
        return result
    def close(self):
        for hook in self.hooks:hook.remove()


def state_difference(current,reference):
    result={}
    for kind in ("q","k","attention_output","logits"):
        a,b=current[kind],reference[kind];delta=a-b
        if kind=="logits":a,b,delta=a[None],b[None],delta[None]
        result[kind]={"max_abs":delta.abs().amax(-1).tolist(),"rmse":delta.square().mean(-1).sqrt().tolist(),
                      "relative_l2":(delta.norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-12)).tolist()}
    result["argmax_equal"]=int(current["logits"].argmax())==int(reference["logits"].argmax())
    return result
