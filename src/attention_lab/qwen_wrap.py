"""Qwen3 integration (G1b): wrap every layer's self_attn around the frozen body, add a static K/V cache and an index
table, and drive prefill / decode with the same read paths G1a measured on random tensors.

Reused from the installed model, untouched: q_proj / k_proj / v_proj / o_proj, q_norm / k_norm, the model's RoPE
(cos/sin computed by Qwen3Model.forward, applied with transformers' apply_rotary_pos_emb), embeddings, MLP, norms and
lm_head. The wrapper replaces only the read: which K/V rows the new query attends to, and from which buffer.

Per layer (docs/IMPLEMENTATION.md §1):
  hidden -> q_proj / k_proj / v_proj -> q_norm / k_norm -> (q_pre, k_pre: index features, BEFORE the main RoPE)
         -> main RoPE -> (q_main, k_main) -> append k_main, v (and z = Wk·k_pre) exactly once -> read -> o_proj

Positions: a token's cache slot is its absolute position; rows are read by original position and never renumbered, so
no second RoPE is applied. z is built from k_pre, never from the rotated K held in the cache. A request fixes k0 from
the legal length of its first decode step; the legal range grows by one per step; after a rollback the old suffix is
still in memory but lies outside the legal range and can neither be read nor selected.

Nothing here trains anything, modifies the installed library, or re-initialises body weights.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .decode import sparse_decode_step_kernel
from .full import full_attention
from .indexer import GroupedIndexer, project_keys
from .selection import FIXED_PREFIX, FIXED_RECENT, fixed_budget, selection_layout
from .sparse import row_offsets

try:  # transformers is a runtime dependency of this module only; the rest of the package stays importable without it
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError as _e:  # pragma: no cover
    apply_rotary_pos_emb = None
    _TRANSFORMERS_IMPORT_ERROR = _e

MASK_BYPASS = {"full_attention": None}   # Qwen3Model.forward skips mask creation when attention_mask is already a dict


# ----------------------------------------------------------------------------------------------------------------------
# static cache: K/V per layer [B,G,C,d], index table per layer zT [B,G,r,C], one valid_length for all layers
# ----------------------------------------------------------------------------------------------------------------------
class LabCache:
    """Preallocated per-layer K/V buffers and (optionally) index tables; one legal length shared by every layer.

    Buffers come from torch.empty on purpose: unwritten slots hold garbage and the contracts poison them with NaN to
    prove that no read or selection ever touches a slot at or beyond valid_length.
    """

    def __init__(self, *, layers: int, batch: int, groups: int, capacity: int, head_dim: int, index_dim=None, device,
                 dtype=torch.float32):
        if int(capacity) <= 0 or int(layers) <= 0:
            raise ValueError("capacity and layers must be positive")
        self.layers, self.batch, self.groups = int(layers), int(batch), int(groups)
        self.capacity, self.head_dim = int(capacity), int(head_dim)
        self.index_dim = None if index_dim is None else int(index_dim)
        self.device, self.dtype = torch.device(device), dtype
        kv_shape = (self.batch, self.groups, self.capacity, self.head_dim)
        self.k = [torch.empty(kv_shape, device=self.device, dtype=dtype) for _ in range(self.layers)]
        self.v = [torch.empty(kv_shape, device=self.device, dtype=dtype) for _ in range(self.layers)]
        if self.index_dim is None:
            self.z = None
        else:
            z_shape = (self.batch, self.groups, self.index_dim, self.capacity)                 # [B,G,r,C]: G0 layout
            self.z = [torch.empty(z_shape, device=self.device, dtype=dtype) for _ in range(self.layers)]
        self.valid_length = 0
        self.offsets = row_offsets(self.batch, self.groups, self.capacity, self.device)          # [B,G,1]
        ar = torch.arange(FIXED_PREFIX, device=self.device).view(1, 1, -1)
        self.prefix_rows = (ar + self.offsets).contiguous()                                      # rows of positions 0..3
        self.recent_base = (torch.arange(-FIXED_RECENT, 0, device=self.device).view(1, 1, -1) + self.offsets).contiguous()
        self.off_lo = (self.offsets + FIXED_PREFIX).contiguous()                                 # candidate_lo = 4 when legal >= 68
        self.writes = [0] * self.layers          # append calls per layer; one per forward is the contract
        self.last_write = [None] * self.layers   # (start, length) of each layer's latest append
        self._pending = None                      # length of the forward in flight

    @property
    def has_index(self) -> bool:
        return self.z is not None

    def append(self, layer: int, k_main: torch.Tensor, v: torch.Tensor, z=None) -> None:
        """Write this layer's new rows at [valid_length, valid_length + T). Called exactly once per layer per forward."""
        start, T = self.valid_length, int(k_main.shape[2])
        if start + T > self.capacity:
            raise RuntimeError(f"cache capacity {self.capacity} exceeded: valid {start} + new {T}")
        if self._pending is None:
            self._pending = T
        elif self._pending != T:
            raise RuntimeError("layers appended different lengths within one forward")
        self.k[layer][:, :, start:start + T].copy_(k_main)
        self.v[layer][:, :, start:start + T].copy_(v)
        if z is not None:
            if self.z is None:
                raise RuntimeError("this cache was created without an index table")
            self.z[layer][..., start:start + T].copy_(z)
        self.writes[layer] += 1
        self.last_write[layer] = (start, T)

    def advance(self, T: int) -> None:
        """Commit a forward of T tokens after every layer appended exactly once."""
        if self._pending != int(T):
            raise RuntimeError(f"advance({T}) does not match the appended length {self._pending}")
        if any(w != self.writes[0] for w in self.writes):
            raise RuntimeError(f"unequal append counts across layers: {self.writes}")
        self.valid_length += int(T)
        self._pending = None

    def rollback(self, length: int) -> None:
        """Shrink the legal range. The old suffix stays in memory; reads and selections are bounded by valid_length."""
        if not 0 <= int(length) <= self.valid_length:
            raise ValueError(f"rollback to {length} outside [0, {self.valid_length}]")
        if self._pending is not None:
            raise RuntimeError("rollback during a forward")
        self.valid_length = int(length)

    def clone(self) -> "LabCache":
        """Independent copy (a request branch). No storage is shared with the original."""
        new = LabCache.__new__(LabCache)
        new.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("k", "v", "z", "writes", "last_write")})
        new.k = [t.clone() for t in self.k]
        new.v = [t.clone() for t in self.v]
        new.z = None if self.z is None else [t.clone() for t in self.z]
        new.writes, new.last_write = list(self.writes), list(self.last_write)
        return new

    def bytes(self) -> dict:
        es = self.k[0].element_size()
        kv = self.layers * 2 * self.batch * self.groups * self.capacity * self.head_dim * es
        z = 0 if self.z is None else self.layers * self.batch * self.groups * self.index_dim * self.capacity * es
        return {"kv_bytes": kv, "index_bytes": z, "total_bytes": kv + z}


# ----------------------------------------------------------------------------------------------------------------------
# per-step selection plan (shared by all layers of one decode step)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class StepPlan:
    legal_length: int          # positions 0..legal_length-1 are legal for this step's query (its own key included)
    k0: int                    # fixed per request
    k: int                     # min(k0, candidates now)
    lo: int                    # candidate domain [lo, hi)
    hi: int
    fixed_count: int
    selected_count: int
    fixed_rows: torch.Tensor   # [B,G,F] row ids in the flattened cache
    off_lo: torch.Tensor       # [B,G,1] offsets + lo

    @property
    def write_col(self) -> int:
        return self.legal_length - 1

    @property
    def candidate_count(self) -> int:
        return self.hi - self.lo


def build_step_plan(cache: LabCache, legal_length: int, k0: int) -> StepPlan:
    """F ∪ candidate domain for a decode step whose legal range is [0, legal_length). Device-side when legal >= 68."""
    layout = selection_layout(legal_length)
    k = layout.clamp_k(k0)
    if layout.fixed_prefix == FIXED_PREFIX and layout.recent_start == legal_length - FIXED_RECENT:
        fixed_rows = torch.cat([cache.prefix_rows, cache.recent_base + legal_length], dim=-1)   # no host->device copy
        off_lo = cache.off_lo
    else:  # short history: F covers (almost) everything; build the general list
        pos = torch.tensor(layout.fixed_positions(), dtype=torch.long, device=cache.device).view(1, 1, -1)
        fixed_rows = (pos + cache.offsets).contiguous()
        off_lo = (cache.offsets + layout.candidate_lo).contiguous()
    return StepPlan(legal_length=int(legal_length), k0=int(k0), k=int(k), lo=layout.candidate_lo, hi=layout.candidate_hi,
                    fixed_count=layout.fixed_count, selected_count=layout.fixed_count + k, fixed_rows=fixed_rows, off_lo=off_lo)


# ----------------------------------------------------------------------------------------------------------------------
# state shared by the wrapped layers and the wrapper module itself
# ----------------------------------------------------------------------------------------------------------------------
class LabState:
    """One read configuration: a cache, optional indexers, the mode (full | sparse) and the plan of the step in flight."""

    def __init__(self, cache: LabCache, indexers=None, *, mode: str = "full", decode_backend: str = "sdpa_fold",
                 prefill_backend: str = "sdpa_fold"):
        if mode not in ("full", "sparse"):
            raise ValueError("mode must be 'full' or 'sparse'")
        if mode == "sparse" and (indexers is None or not cache.has_index):
            raise ValueError("sparse mode needs indexers and a cache with an index table")
        if indexers is not None and len(indexers) != cache.layers:
            raise ValueError("one indexer per layer")
        self.cache, self.indexers, self.mode = cache, indexers, mode
        self.decode_backend, self.prefill_backend = decode_backend, prefill_backend
        self.plan: StepPlan | None = None
        self.rho = None
        self.k0 = None
        self.request_start_length = None

    def begin_request(self, rho: float, *, k0=None) -> int:
        """Fix k0 from the legal length of the first decode step: the prefilled tokens plus the first new token.

        `k0` overrides the rho rule (diagnostics only, e.g. k0 = capacity forces every candidate on every step so the
        sparse path can be checked against the full path); k is still clamped to the candidate count at each step.
        """
        self.rho = None if rho is None else float(rho)
        self.request_start_length = self.cache.valid_length + 1
        self.k0 = fixed_budget(self.request_start_length, self.rho) if k0 is None else int(k0)
        return self.k0


class StateHolder:
    """Mutable pointer so one wrapped model can alternate between read configurations (A cache, D cache)."""

    def __init__(self, state: LabState | None = None):
        self.state = state


class LabAttention(nn.Module):
    """Replacement for one Qwen3Attention: identical projections, norms, RoPE and output path; only the read differs.

    `inner` is the original module object (its parameters are reused, never copied or re-initialised).
    """

    def __init__(self, inner: nn.Module, holder: StateHolder, layer_idx: int):
        super().__init__()
        if apply_rotary_pos_emb is None:  # pragma: no cover
            raise _TRANSFORMERS_IMPORT_ERROR
        self.inner = inner
        self.holder = holder
        self.layer_idx = int(layer_idx)
        cfg = inner.config
        self.num_heads = int(cfg.num_attention_heads)
        self.num_groups = int(cfg.num_key_value_heads)
        self.head_dim = int(inner.head_dim)
        if self.num_heads % self.num_groups:
            raise ValueError("query heads must be a multiple of KV heads")
        self.heads_per_group = self.num_heads // self.num_groups
        self.scaling = float(inner.scaling)
        self.is_causal = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        inner, st = self.inner, self.holder.state
        if st is None:
            raise RuntimeError("no LabState set on the wrapped model")
        B, T, _ = hidden_states.shape
        shape = (B, T, -1, self.head_dim)
        q_pre = inner.q_norm(inner.q_proj(hidden_states).view(shape)).transpose(1, 2)   # [B,H,T,d] post-norm, pre-RoPE
        k_pre = inner.k_norm(inner.k_proj(hidden_states).view(shape)).transpose(1, 2)   # [B,G,T,d] post-norm, pre-RoPE
        v = inner.v_proj(hidden_states).view(shape).transpose(1, 2)                     # [B,G,T,d]
        cos, sin = position_embeddings
        q_main, k_main = apply_rotary_pos_emb(q_pre, k_pre, cos, sin)                   # the model's own RoPE
        cache, l = st.cache, self.layer_idx
        L0 = cache.valid_length
        G, h, d = self.num_groups, self.heads_per_group, self.head_dim
        q5 = q_main.view(B, G, h, T, d)                                                 # query head g*h+i reads group g
        if T == 1 and st.mode == "sparse":
            plan = st.plan
            if plan is None or plan.legal_length != L0 + 1:
                raise RuntimeError("sparse decode needs a StepPlan built for this step's legal length")
            idx = st.indexers[l]
            cache.append(l, k_main, v)                        # K/V once; z of the new key is written by the kernel below
            out = sparse_decode_step_kernel(q5, q_pre.view(B, G, h, 1, d), cache.k[l], cache.v[l], cache.z[l],
                                            idx.wq_t_scaled, idx.wk, plan.fixed_rows, plan.off_lo, plan.lo, plan.hi,
                                            plan.k, plan.selected_count, k_pre_new=k_pre, write_col=plan.write_col,
                                            scale=self.scaling)
        else:
            z = project_keys(k_pre, st.indexers[l].wk) if cache.has_index else None     # index keys from pre-RoPE K
            cache.append(l, k_main, v, z)
            backend = st.decode_backend if T == 1 else st.prefill_backend
            out = full_attention(q5, cache.k[l], cache.v[l], valid_length=L0 + T, backend=backend, scale=self.scaling)
        attn = out.view(B, self.num_heads, T, d).transpose(1, 2).reshape(B, T, self.num_heads * d)
        return inner.o_proj(attn), None


# ----------------------------------------------------------------------------------------------------------------------
# model geometry, construction helpers, wrap / unwrap
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelGeometry:
    layers: int
    hidden: int
    heads: int
    groups: int
    head_dim: int
    vocab: int
    max_positions: int

    @property
    def heads_per_group(self) -> int:
        return self.heads // self.groups


def _attention_module(layer):
    return layer.self_attn.inner if isinstance(layer.self_attn, LabAttention) else layer.self_attn


def geometry_of(model) -> ModelGeometry:
    """Read the geometry from the config and check it against the actual parameter shapes (not from G1a defaults)."""
    cfg = model.config
    attn = _attention_module(model.model.layers[0])
    heads, groups, head_dim, hidden = int(cfg.num_attention_heads), int(cfg.num_key_value_heads), int(attn.head_dim), int(cfg.hidden_size)
    expect = {"q_proj": (heads * head_dim, hidden), "k_proj": (groups * head_dim, hidden), "v_proj": (groups * head_dim, hidden),
              "o_proj": (hidden, heads * head_dim)}
    for name, shape in expect.items():
        got = tuple(getattr(attn, name).weight.shape)
        if got != shape:
            raise ValueError(f"{name} weight shape {got} != {shape} implied by the config")
    if tuple(attn.q_norm.weight.shape) != (head_dim,) or tuple(attn.k_norm.weight.shape) != (head_dim,):
        raise ValueError("q_norm / k_norm act on head_dim in Qwen3; shapes disagree")
    return ModelGeometry(layers=len(model.model.layers), hidden=hidden, heads=heads, groups=groups, head_dim=head_dim,
                         vocab=int(cfg.vocab_size), max_positions=int(cfg.max_position_embeddings))


def make_cache(model, capacity: int, *, with_index: bool, index_dim: int = 32, batch: int = 1, dtype=None, device=None) -> LabCache:
    geo = geometry_of(model)
    p = next(model.parameters())
    return LabCache(layers=geo.layers, batch=batch, groups=geo.groups, capacity=capacity, head_dim=geo.head_dim,
                    index_dim=index_dim if with_index else None, device=p.device if device is None else device,
                    dtype=p.dtype if dtype is None else dtype)


def make_indexers(model, *, r: int = 32, seed: int = 0, dtype=None, device=None) -> list[GroupedIndexer]:
    """One untrained (D0-style) grouped indexer per layer, seeded as in G1a (seed * 1000 + layer)."""
    geo = geometry_of(model)
    p = next(model.parameters())
    return [GroupedIndexer(geo.groups, geo.heads_per_group, geo.head_dim, r, device=p.device if device is None else device,
                           dtype=p.dtype if dtype is None else dtype, seed=seed * 1000 + i) for i in range(geo.layers)]


def make_state(model, capacity: int, mode: str, *, r: int = 32, seed: int = 0, batch: int = 1, **kw) -> LabState:
    with_index = mode == "sparse"
    cache = make_cache(model, capacity, with_index=with_index, index_dim=r, batch=batch)
    idx = make_indexers(model, r=r, seed=seed) if with_index else None
    return LabState(cache, idx, mode=mode, **kw)


def wrap_model(model, holder: StateHolder | None = None) -> StateHolder:
    """Replace every layer's self_attn with LabAttention around the original module. Returns the shared holder."""
    holder = StateHolder() if holder is None else holder
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.self_attn, LabAttention):
            raise RuntimeError("model is already wrapped")
        layer.self_attn = LabAttention(layer.self_attn, holder, i)
    return holder


def unwrap_model(model) -> None:
    for layer in model.model.layers:
        if isinstance(layer.self_attn, LabAttention):
            layer.self_attn = layer.self_attn.inner


def is_wrapped(model) -> bool:
    return isinstance(model.model.layers[0].self_attn, LabAttention)


# ----------------------------------------------------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------------------------------------------------
class LabRunner:
    """Prefill / decode driver for a wrapped model and one LabState.

    Every forward goes through the model's own entry (embeddings -> layers -> norm -> lm_head); only self_attn differs.
    position_ids are views into one preallocated arange, so a decode step issues no host->device copy for positions.
    """

    def __init__(self, model, holder: StateHolder, state: LabState):
        if not is_wrapped(model):
            raise RuntimeError("wrap_model first")
        self.model, self.holder, self.state = model, holder, state
        self.positions = torch.arange(state.cache.capacity, device=state.cache.device)

    @property
    def cache(self) -> LabCache:
        return self.state.cache

    def forward(self, input_ids: torch.Tensor, *, logits_to_keep: int = 1) -> torch.Tensor:
        st, cache = self.state, self.state.cache
        B, T = input_ids.shape
        if B != cache.batch:
            raise ValueError("batch size differs from the cache")
        start = cache.valid_length
        if T == 1 and st.mode == "sparse":
            if st.k0 is None:
                raise RuntimeError("call state.begin_request(rho) before the first sparse decode step")
            st.plan = build_step_plan(cache, start + 1, st.k0)
        self.holder.state = st
        out = self.model(input_ids=input_ids, position_ids=self.positions[start:start + T].unsqueeze(0),
                         attention_mask=MASK_BYPASS, use_cache=False, logits_to_keep=logits_to_keep)
        cache.advance(T)
        return out.logits

    def prefill(self, input_ids: torch.Tensor, chunk: int, *, keep_logits: bool = False):
        """Dense prefill in chunks (the index table is filled chunk by chunk when the cache has one)."""
        T = input_ids.shape[1]
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        outs = []
        logits = None
        for a in range(0, T, chunk):
            logits = self.forward(input_ids[:, a:a + chunk], logits_to_keep=0 if keep_logits else 1)
            if keep_logits:
                outs.append(logits)
        return torch.cat(outs, dim=1) if keep_logits else logits

    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """One decode step: token ids [B,1] -> vocabulary scores [B,1,V] of the next token."""
        if token_ids.shape[1] != 1:
            raise ValueError("a decode step takes one token per sequence")
        return self.forward(token_ids, logits_to_keep=1)

    def rollback(self, length: int) -> None:
        self.state.cache.rollback(length)


class NativeRunner:
    """The unmodified model with transformers' own cache (DynamicCache or StaticCache): the native A reference."""

    def __init__(self, model, cache, capacity: int):
        if is_wrapped(model):
            raise RuntimeError("NativeRunner needs the unwrapped model")
        self.model, self.cache = model, cache
        self.length = 0
        self.positions = torch.arange(int(capacity), device=next(model.parameters()).device)

    def forward(self, input_ids: torch.Tensor, *, logits_to_keep: int = 1) -> torch.Tensor:
        T = input_ids.shape[1]
        out = self.model(input_ids=input_ids, position_ids=self.positions[self.length:self.length + T].unsqueeze(0),
                         past_key_values=self.cache, use_cache=True, logits_to_keep=logits_to_keep)
        self.length += T
        return out.logits

    def prefill(self, input_ids: torch.Tensor, chunk: int, *, keep_logits: bool = False):
        T = input_ids.shape[1]
        outs, logits = [], None
        for a in range(0, T, chunk):
            logits = self.forward(input_ids[:, a:a + chunk], logits_to_keep=0 if keep_logits else 1)
            if keep_logits:
                outs.append(logits)
        return torch.cat(outs, dim=1) if keep_logits else logits

    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids, logits_to_keep=1)

    def rollback(self, length: int) -> None:
        removed = self.length - int(length)
        if removed < 0:
            raise ValueError("cannot roll forward")
        if removed == 0:
            return
        layers = self.cache.layers
        if all(hasattr(lay, "cumulative_length") and isinstance(lay.cumulative_length, torch.Tensor) for lay in layers):
            for lay in layers:                      # StaticLayer: slots beyond `length` fall outside the causal mask
                lay.cumulative_length.fill_(int(length))
        else:
            self.cache.crop(-removed)               # DynamicLayer: drop the suffix
        self.length = int(length)


# ----------------------------------------------------------------------------------------------------------------------
# loading and provenance
# ----------------------------------------------------------------------------------------------------------------------
def load_model(model_id: str, revision: str, *, dtype, device, attn_implementation: str = "sdpa"):
    """Load the pinned checkpoint and tokenizer, freeze every parameter, return (model, tokenizer, info)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype, attn_implementation=attn_implementation)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, model_info(model, tok, model_id, revision)


def model_info(model, tok, model_id: str, revision: str) -> dict:
    cfg = model.config
    geo = geometry_of(model)
    layer0 = model.model.layers[0]
    attn = _attention_module(layer0)
    seen, unique = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique += p.numel()
    rope = getattr(cfg, "rope_parameters", None)
    rope_theta = rope.get("rope_theta") if isinstance(rope, dict) else getattr(cfg, "rope_theta", None)
    return {
        "model_id": model_id, "revision_requested": revision,
        "config_commit_hash": getattr(cfg, "_commit_hash", None),
        "tokenizer_class": type(tok).__name__ if tok is not None else None,
        "tokenizer_commit_hash": getattr(tok, "_commit_hash", None) if tok is not None else None,
        "architecture": type(model).__name__, "attn_implementation": getattr(cfg, "_attn_implementation", None),
        "num_hidden_layers": geo.layers, "num_attention_heads": geo.heads, "num_key_value_heads": geo.groups,
        "head_dim": geo.head_dim, "hidden_size": geo.hidden, "intermediate_size": int(cfg.intermediate_size),
        "vocab_size": geo.vocab, "max_position_embeddings": geo.max_positions, "rope_theta": rope_theta,
        "rms_norm_eps": float(cfg.rms_norm_eps), "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "num_parameters_total": int(sum(p.numel() for p in model.parameters())), "num_parameters_unique_storage": int(unique),
        "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
        "modules": {"self_attn": type(layer0.self_attn).__name__, "inner_attention": type(attn).__name__,
                    "q_proj": list(attn.q_proj.weight.shape), "k_proj": list(attn.k_proj.weight.shape),
                    "v_proj": list(attn.v_proj.weight.shape), "o_proj": list(attn.o_proj.weight.shape),
                    "q_norm": list(attn.q_norm.weight.shape), "k_norm": list(attn.k_norm.weight.shape),
                    "mlp": type(layer0.mlp).__name__, "rotary": type(model.model.rotary_emb).__name__,
                    "lm_head": list(model.lm_head.weight.shape)},
    }


def weights_file_sha256(model_id: str, revision: str, filename: str = "model.safetensors") -> dict:
    """SHA-256 of the cached checkpoint file at the pinned revision (the actual weight identity)."""
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(model_id, filename, revision=revision)
    if not isinstance(p, str):
        return {"file": filename, "path": None, "sha256": None, "bytes": None}
    real = Path(p).resolve()
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return {"file": filename, "sha256": h.hexdigest(), "bytes": real.stat().st_size}


def tiny_qwen3(*, seed: int = 0, layers: int = 2, hidden: int = 64, heads: int = 4, groups: int = 2, head_dim: int = 16,
               intermediate: int = 128, vocab: int = 256, max_positions: int = 2048, dtype=torch.float64):
    """Randomly initialised Qwen3 with the real module code, for contracts that need no download (CI, CPU float64)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=groups, head_dim=head_dim, max_position_embeddings=max_positions,
                      tie_word_embeddings=False, attn_implementation="sdpa")
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def hf_cache_kv(cache, layer: int):
    """K/V [B,G,L,d] held by a transformers cache for one layer (the comparison target for the static cache)."""
    lay = cache.layers[layer]
    return lay.keys, lay.values
