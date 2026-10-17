"""One decode step of a single layer: the dense read (A) and the index-selected read (D).

D per step: [z append for the newest key] -> u projection -> one matmul over the candidate columns -> one top-k
over the candidate domain -> merge with the fixed set as row ids -> index_select of K and V rows -> subset
attention. The selection is a list of original positions; nothing is renumbered and the main K is read as stored.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .full import full_attention
from .indexer import GroupedIndexer, candidate_scores, project_keys, project_queries
from .selection import FIXED_PREFIX, FIXED_RECENT, SelectionLayout, fixed_budget, legal_length, selection_layout
from .sparse import flatten_cache, row_offsets, subset_attention


@dataclass
class DecodeRequest:
    """Per-request constants for a fixed legal length: layout, budget and precomputed row ids of the fixed set."""

    layout: SelectionLayout
    k0: int
    k: int
    batch: int
    groups: int
    capacity: int
    offsets: torch.Tensor          # [B,G,1] row offsets into the flattened [B*G*C, d] cache
    fixed_positions: torch.Tensor  # [B,G,F] original positions of F
    fixed_rows: torch.Tensor       # [B,G,F] = fixed_positions + offsets
    off_lo: torch.Tensor           # [B,G,1] = offsets + candidate_lo (maps candidate-relative top-k indices to rows)

    @property
    def selected_count(self) -> int:
        return self.layout.fixed_count + self.k

    @property
    def write_col(self) -> int:
        """Column of the index table that belongs to the newest key (the query's own position)."""
        return self.layout.legal_length - 1

    def topk_candidates(self, u: torch.Tensor, zT: torch.Tensor) -> torch.Tensor:
        """[B,G,k] candidate-relative indices from one top-k over the candidate domain."""
        if self.k <= 0:
            return torch.empty(self.batch, self.groups, 0, dtype=torch.long, device=u.device)
        scores = candidate_scores(u, zT, self.layout.candidate_lo, self.layout.candidate_hi)
        return torch.topk(scores, self.k, dim=-1).indices.view(self.batch, self.groups, self.k)

    def select_positions(self, u: torch.Tensor, zT: torch.Tensor) -> torch.Tensor:
        """[B,G,S] original positions: F first, then the chosen candidates."""
        top = self.topk_candidates(u, zT) + self.layout.candidate_lo
        return torch.cat([self.fixed_positions, top], dim=-1)

    def select_rows(self, u: torch.Tensor, zT: torch.Tensor) -> torch.Tensor:
        """Flat row ids [B*G*S] for index_select on the flattened cache (what the step actually uses)."""
        return select_rows_kernel(u, zT, self.fixed_rows, self.off_lo, self.layout.candidate_lo,
                                  self.layout.candidate_hi, self.k)


def prepare_request(batch: int, groups: int, capacity: int, *, valid_length: int, query_position: int, rho: float,
                    device, n0=None, fixed_prefix: int = FIXED_PREFIX, fixed_recent: int = FIXED_RECENT) -> DecodeRequest:
    """Fix k0 from the request's starting length n0 (default: the current legal length) and clamp k to this step."""
    legal = legal_length(query_position, valid_length)
    if legal > capacity:
        raise ValueError(f"legal length {legal} exceeds allocated capacity {capacity}")
    n0 = legal if n0 is None else int(n0)
    k0 = fixed_budget(n0, rho, fixed_prefix, fixed_recent)
    layout = selection_layout(legal, fixed_prefix, fixed_recent)
    k = layout.clamp_k(k0)
    offsets = row_offsets(batch, groups, capacity, device)
    fixed_positions = torch.tensor(layout.fixed_positions(), dtype=torch.long, device=device)
    fixed_positions = fixed_positions.view(1, 1, -1).expand(batch, groups, -1).contiguous()
    return DecodeRequest(layout=layout, k0=k0, k=k, batch=batch, groups=groups, capacity=capacity, offsets=offsets,
                         fixed_positions=fixed_positions, fixed_rows=(fixed_positions + offsets).contiguous(),
                         off_lo=(offsets + layout.candidate_lo).contiguous())


def select_rows_kernel(u, zT, fixed_rows, off_lo, lo: int, hi: int, k: int) -> torch.Tensor:
    """Score the candidate columns once, top-k once, merge with the fixed rows. Returns flat row ids [B*G*S]."""
    B, G = fixed_rows.shape[:2]
    if k <= 0:
        return fixed_rows.reshape(-1)
    scores = candidate_scores(u, zT, lo, hi)                       # [B,G,1,M]
    top = torch.topk(scores, k, dim=-1).indices.view(B, G, k)      # the only top-k, over the candidate domain
    return torch.cat([fixed_rows, top + off_lo], dim=-1).view(-1)  # original positions as row ids


def sparse_decode_step_kernel(q_main, q_pre, k_cache, v_cache, zT, wq_t_scaled, wk, fixed_rows, off_lo, lo: int,
                              hi: int, k: int, selected_count: int, *, k_pre_new=None, write_col: int = -1,
                              scale=None, backend="sdpa"):
    """One layer's D read for one new query (T == 1). Every D-only cost is inside this function."""
    B, G, h, T, d = q_main.shape
    if T != 1:
        raise ValueError("a decode step takes exactly one new query position")
    if k_pre_new is not None:
        zT[..., write_col:write_col + 1].copy_(project_keys(k_pre_new, wk))   # z of the newest key, its own column
    u = project_queries(q_pre, wq_t_scaled)                                     # [B,G,1,r]
    rows = select_rows_kernel(u, zT, fixed_rows, off_lo, lo, hi, k)
    k_sub = flatten_cache(k_cache).index_select(0, rows).view(B, G, selected_count, d)
    v_sub = flatten_cache(v_cache).index_select(0, rows).view(B, G, selected_count, d)
    return subset_attention(q_main, k_sub, v_sub, scale=scale, backend=backend)


def sparse_decode_step(q_main, q_pre, k_cache, v_cache, zT, indexer: GroupedIndexer, request: DecodeRequest, *,
                       k_pre_new=None, scale=None, backend="sdpa"):
    return sparse_decode_step_kernel(q_main, q_pre, k_cache, v_cache, zT, indexer.wq_t_scaled, indexer.wk,
                                     request.fixed_rows, request.off_lo, request.layout.candidate_lo,
                                     request.layout.candidate_hi, request.k, request.selected_count,
                                     k_pre_new=k_pre_new, write_col=request.write_col, scale=scale, backend=backend)


def dense_decode_step(q_main, k_cache, v_cache, valid_length: int, *, backend="sdpa_fold", scale=None):
    """One layer's A read for one new query: every valid position, K and V read, softmax, PV."""
    return full_attention(q_main, k_cache, v_cache, valid_length=valid_length, backend=backend, scale=scale)
