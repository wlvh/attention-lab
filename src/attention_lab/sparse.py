"""Gather-first attention: read only the listed rows, then attend over them. This is the D performance path.

G0 constraint (reports/cards/G0_mac_m4max.md): rows are fetched with index_select on the flattened [B*G*C, d]
view of a contiguous cache, never with torch.gather on expanded indices.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import default_scale


def row_offsets(batch: int, groups: int, capacity: int, device) -> torch.Tensor:
    """[B,G,1] int64 offsets that turn a position into a row id of the flattened cache."""
    return (torch.arange(batch * groups, device=device, dtype=torch.long) * capacity).view(batch, groups, 1)


def flatten_cache(x: torch.Tensor) -> torch.Tensor:
    """[B,G,C,d] contiguous -> [B*G*C, d] view. No copy; raises instead of copying silently."""
    if x.dim() != 4:
        raise ValueError(f"cache must be [B,G,C,d], got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("cache must be contiguous so the flattened view is free")
    B, G, C, d = x.shape
    return x.view(B * G * C, d)


def gather_rows(x: torch.Tensor, selected: torch.Tensor, offsets=None) -> torch.Tensor:
    """x [B,G,C,d], selected [B,G,S] original positions -> [B,G,S,d]; row s of group (b,g) is x[b,g,selected[b,g,s]]."""
    B, G, C, d = x.shape
    if selected.shape[:2] != (B, G) or selected.dim() != 3:
        raise ValueError(f"selected must be [B,G,S] matching the cache, got {tuple(selected.shape)}")
    S = selected.shape[-1]
    off = row_offsets(B, G, C, x.device) if offsets is None else offsets
    rows = (selected + off).reshape(-1)
    return flatten_cache(x).index_select(0, rows).view(B, G, S, d)


def subset_attention(q, k_sub, v_sub, valid_mask=None, *, scale=None, backend="sdpa"):
    """Attention of q [B,G,h,T,d] over an already gathered subset k_sub, v_sub [B,G,S,d] -> [B,G,h,T,d].

    The h*T query rows of a group are folded into the sequence axis, so the S rows are read once per group and
    never repeated per query head. `valid_mask` [B,G,S] bool masks padded slots; the subset is shared by all
    query rows of the step, so callers use T == 1 or a selection that is legal for every query.
    """
    B, G, h, T, d = q.shape
    S = k_sub.shape[2]
    qf = q.reshape(B, G, h * T, d)
    mask = None if valid_mask is None else valid_mask.view(B, G, 1, S)
    if backend == "sdpa":
        out = F.scaled_dot_product_attention(qf, k_sub, v_sub, attn_mask=mask, scale=scale)
    elif backend == "explicit":
        scores = torch.matmul(qf, k_sub.transpose(-1, -2)) * default_scale(d, scale)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        out = torch.matmul(torch.softmax(scores, dim=-1), v_sub)
    else:
        raise ValueError(f"unknown subset backend {backend!r}")
    return out.view(B, G, h, T, d)


def gather_attention(q, k, v, selected, selected_valid=None, *, scale=None, backend="sdpa", offsets=None):
    """Gather the listed K/V rows (original positions, no renumbering, no second RoPE), then attend over them."""
    k_sub = gather_rows(k, selected, offsets)
    v_sub = gather_rows(v, selected, offsets)
    return subset_attention(q, k_sub, v_sub, selected_valid, scale=scale, backend=backend)
