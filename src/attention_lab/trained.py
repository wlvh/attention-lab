"""Frozen learned indexes: verified files, float32 scoring, parameter-bound z caches."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


class FrozenIndexer:
    def __init__(self, wq, wk, *, identity, device):
        if wq.dtype != np.float32 or wk.dtype != np.float32:
            raise ValueError("frozen index parameters must be float32")
        self.wq = torch.from_numpy(wq.copy()).to(device).detach()
        self.wk = torch.from_numpy(wk.copy()).to(device).detach()
        self.r = self.wk.shape[1]
        self.wq_t_scaled = (self.wq / math.sqrt(self.r)).transpose(1,2).contiguous()
        self.identity = identity
        self._versions = self.signature()

    def signature(self):
        return tuple((id(x), x._version) for x in (self.wq,self.wk,self.wq_t_scaled))

    def verify(self):
        if self.signature() != self._versions:
            raise RuntimeError("frozen index parameters changed; rebuild bank and prefix cache")


class FrozenBank:
    def __init__(self, indexes, *, identity):
        self.indexes = list(indexes)
        self.identity = identity
        self._objects = tuple(id(x) for x in self.indexes)

    def guard(self, state):
        if tuple(id(x) for x in state.indexers) != self._objects:
            raise RuntimeError("state index bank changed")
        for index in self.indexes:
            index.verify()
            if index.wk.device != state.cache.device:
                raise RuntimeError("index/cache devices differ")
        cache = state.cache
        if not cache.has_index or cache.index_dtype != torch.float32:
            raise RuntimeError("bank requires a float32 index cache")
        if cache.valid_length and getattr(cache,"index_identity",None) != self.identity:
            raise RuntimeError("prefix index cache belongs to different parameter identity")
        cache.index_identity = self.identity


def load_bank(run_dir, metadata, geometry, *, device, initial=False):
    """Verify every file before using it. Bank role is part of cache identity."""
    root=Path(run_dir).resolve()
    unsigned={k:v for k,v in metadata.items() if k!="bank_id"}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True).encode()).hexdigest()!=metadata["bank_id"]:
        raise ValueError("bank metadata identity mismatch")
    entries=metadata["layers"]
    if len(entries)!=geometry.layers or {x["layer_index"] for x in entries}!=set(range(geometry.layers)):
        raise ValueError("incomplete index bank")
    r=metadata["r"];role="initial" if initial else "final"
    indexes=[]
    for entry in sorted(entries,key=lambda x:x["layer_index"]):
        path=(root/entry[role+"_file"]).resolve()
        if not path.is_relative_to(root):raise ValueError("parameter path outside run")
        identity=hashlib.sha256(path.read_bytes()).hexdigest()
        if identity!=entry[role+"_sha256"]:raise ValueError("parameter file hash mismatch")
        with np.load(path,allow_pickle=False) as f:
            wq,wk=f["wq"],f["wk"]
            if wq.shape!=(geometry.groups,r,geometry.heads_per_group*geometry.head_dim) or wk.shape!=(geometry.groups,r,geometry.head_dim):
                raise ValueError("index/model geometry differs")
            if not np.isfinite(wq).all() or not np.isfinite(wk).all():raise ValueError("nonfinite parameters")
            indexes.append(FrozenIndexer(wq,wk,identity=identity,device=device))
    identity=hashlib.sha256(json.dumps({"bank_id":metadata["bank_id"],"role":role,"files":[x.identity for x in indexes],
                                      "r":r,"dtype":"float32"},sort_keys=True).encode()).hexdigest()
    return FrozenBank(indexes,identity=identity)


def make_trained_state(model, capacity, bank, *, sparse_layers=None, mode="sparse"):
    from .qwen_wrap import LabState,make_cache
    cache=make_cache(model,capacity,with_index=True,index_dim=bank.indexes[0].r,index_dtype=torch.float32)
    return LabState(cache,bank.indexes,mode=mode,selector="index",sparse_layers=sparse_layers,cache_guard=bank.guard)
