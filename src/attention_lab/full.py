"""Full attention A over every legal position, with the implementation variants the benchmark compares.

- sdpa_fold   : the h query heads of a group are folded into the sequence axis; K/V read once per group.
- sdpa_gqa    : torch's enable_gqa=True with H query heads against G KV heads.
- sdpa_repeat : K/V materialised per query head with repeat_interleave (diagnostic; deliberately heavier).
- explicit    : dense_reference.
"""
from __future__ import annotations

import torch.nn.functional as F

from .reference import check_shapes, dense_reference, legality_mask, resolve_query_positions

FULL_BACKENDS = ("sdpa_fold", "sdpa_gqa", "sdpa_repeat", "explicit")


def full_attention(q, k, v, *, valid_length=None, query_positions=None, scale=None, backend="sdpa_fold"):
    """q [B,G,h,T,d]; k, v [B,G,C,d]; reads k[:, :, :valid_length]. Returns [B,G,h,T,d]."""
    B, G, h, T, d, C = check_shapes(q, k, v)
    legal = C if valid_length is None else int(valid_length)
    if not 0 < legal <= C:
        raise ValueError(f"valid_length={legal} must be in (0, capacity={C}]")
    if backend == "explicit":
        return dense_reference(q, k, v, valid_length=legal, query_positions=query_positions, scale=scale)
    qp = resolve_query_positions(T, legal, query_positions)
    kk, vv = k[:, :, :legal], v[:, :, :legal]
    # A single query that may read the whole legal range needs no mask; everything else gets an explicit
    # legality mask, never a square top-left causal mask.
    needs_mask = not (T == 1 and qp[0] >= legal - 1)
    legal_mask = legality_mask(qp, legal, q.device) if needs_mask else None  # [T, legal]
    if backend == "sdpa_fold":
        mask = None if legal_mask is None else legal_mask.repeat(h, 1).view(1, 1, h * T, legal)  # row = i*T + t
        out = F.scaled_dot_product_attention(q.reshape(B, G, h * T, d), kk, vv, attn_mask=mask, scale=scale)
        return out.view(B, G, h, T, d)
    mask = None if legal_mask is None else legal_mask.view(1, 1, T, legal)
    qh = q.reshape(B, G * h, T, d)  # query head g*h+i belongs to group g
    if backend == "sdpa_gqa":
        out = F.scaled_dot_product_attention(qh, kk, vv, attn_mask=mask, scale=scale, enable_gqa=True)
    elif backend == "sdpa_repeat":
        out = F.scaled_dot_product_attention(qh, kk.repeat_interleave(h, dim=1), vv.repeat_interleave(h, dim=1),
                                             attn_mask=mask, scale=scale)
    else:
        raise ValueError(f"unknown full-attention backend {backend!r}; choose from {FULL_BACKENDS}")
    return out.view(B, G, h, T, d)
