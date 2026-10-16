"""Explicit mathematical references: QK^T, mask, softmax, PV written out. Comparison targets, not performance paths.

Shapes (docs/IMPLEMENTATION.md §2): q [B,G,h,T,d]; k, v [B,G,C,d]; output [B,G,h,T,d].
B batch, G KV groups, h query heads per group, T queries in this call, C allocated capacity, d head dim.
"""
from __future__ import annotations

import math

import torch


def check_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if q.dim() != 5:
        raise ValueError(f"q must be [B,G,h,T,d], got {tuple(q.shape)}")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"k and v must be [B,G,C,d], got {tuple(k.shape)} and {tuple(v.shape)}")
    B, G, h, T, d = q.shape
    if k.shape[0] != B or k.shape[1] != G or k.shape[3] != d or v.shape != k.shape:
        raise ValueError(f"shape mismatch: q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)}")
    return B, G, h, T, d, k.shape[2]


def default_scale(d: int, scale):
    return 1.0 / math.sqrt(d) if scale is None else float(scale)


def resolve_query_positions(T: int, legal: int, query_positions) -> list[int]:
    """Default: the T queries sit at the last T positions of the legal range (decode step or tail of a prefill)."""
    if query_positions is None:
        qp = list(range(legal - T, legal))
    else:
        qp = [int(p) for p in query_positions]
    if len(qp) != T:
        raise ValueError(f"need {T} query positions, got {len(qp)}")
    if min(qp) < 0:
        raise ValueError("query positions must be non-negative")
    return qp


def legality_mask(query_positions: list[int], legal: int, device) -> torch.Tensor:
    """[T, legal] bool: key j is legal for query t iff j <= query_positions[t]. Positions >= legal are already excluded."""
    pos = torch.arange(legal, device=device)
    qp = torch.tensor(query_positions, device=device, dtype=torch.long)
    mask = pos[None, :] <= qp[:, None]
    if not bool(mask.any(dim=1).all()):
        raise ValueError("a query has no legal key position")
    return mask


def dense_reference(q, k, v, *, valid_length=None, query_positions=None, scale=None):
    """Full attention written out. Slots at or beyond valid_length are never touched; future positions are masked."""
    B, G, h, T, d, C = check_shapes(q, k, v)
    legal = C if valid_length is None else int(valid_length)
    if not 0 < legal <= C:
        raise ValueError(f"valid_length={legal} must be in (0, capacity={C}]")
    qp = resolve_query_positions(T, legal, query_positions)
    kk, vv = k[:, :, :legal], v[:, :, :legal]
    scores = torch.matmul(q, kk.unsqueeze(2).transpose(-1, -2)) * default_scale(d, scale)  # [B,G,h,T,legal]
    scores = scores.masked_fill(~legality_mask(qp, legal, q.device), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vv.unsqueeze(2))


def selection_mask(selected: torch.Tensor, legal: int, selected_valid=None) -> torch.Tensor:
    """[B,G,legal] bool from a position list [B,G,S]; padded slots are dropped through selected_valid [B,G,S]."""
    if selected.dim() != 3 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected must be an integer tensor [B,G,S]")
    if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= legal):
        raise ValueError(f"selected positions must lie in [0, {legal})")
    B, G, S = selected.shape
    counts = torch.zeros(B, G, legal, dtype=torch.int32, device=selected.device)
    src = torch.ones(B, G, S, dtype=torch.int32, device=selected.device)
    if selected_valid is not None:
        src = src * selected_valid.to(torch.int32)
    counts.scatter_add_(2, selected.long(), src)
    return counts > 0


def dense_mask_reference(q, k, v, selected, selected_valid=None, *, valid_length=None, scale=None):
    """Attention restricted to a position subset, as dense scores over every column followed by masking.

    `selected` [B,G,S] holds original positions shared by the h*T query rows of each group. Positions at or beyond
    valid_length are never touched; the subset must lie below valid_length.
    """
    B, G, h, T, d, C = check_shapes(q, k, v)
    legal = C if valid_length is None else int(valid_length)
    if not 0 < legal <= C:
        raise ValueError(f"valid_length={legal} must be in (0, capacity={C}]")
    keep = selection_mask(selected, legal, selected_valid)
    if not bool(keep.any(dim=-1).all()):
        raise ValueError("empty selection for some (batch, group)")
    kk, vv = k[:, :, :legal], v[:, :, :legal]
    scores = torch.matmul(q, kk.unsqueeze(2).transpose(-1, -2)) * default_scale(d, scale)
    scores = scores.masked_fill(~keep[:, :, None, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vv.unsqueeze(2))
