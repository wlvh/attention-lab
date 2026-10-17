"""One decode step of a single layer: the dense read (A) and the index-selected read (D).

D per step: [z append for the newest key] -> u projection -> one matmul over the candidate columns -> one top-k
over the candidate domain -> merge with the fixed set as row ids -> index_select of K and V rows -> subset
attention. The selection is a list of original positions; nothing is renumbered and the main K is read as stored.

G2-lite adds two alternative row-list generators next to `select_rows_kernel`, with the same fixed set F and the same
budget k, differing only in who receives the k slots of the candidate domain:
  select_rows_kernel   D0  the untrained indexer's top-k scores (content-dependent, weights random)
  select_rows_recent   B   the k most recent candidate positions (a sliding window)
  select_rows_uniform  C   k positions drawn uniformly without replacement from the candidate domain (seeded)
All three feed `gathered_decode_step_kernel`: one flattened index_select of K and of V, then subset attention.
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
    """D0: score the candidate columns once, top-k once, merge with the fixed rows. Returns flat row ids [B*G*S]."""
    B, G = fixed_rows.shape[:2]
    if k <= 0:
        return fixed_rows.reshape(-1)
    scores = candidate_scores(u, zT, lo, hi)                       # [B,G,1,M]
    top = torch.topk(scores, k, dim=-1).indices.view(B, G, k)      # the only top-k, over the candidate domain
    return torch.cat([fixed_rows, top + off_lo], dim=-1).view(-1)  # original positions as row ids


def select_rows_recent(fixed_rows, off_lo, lo: int, hi: int, k: int) -> torch.Tensor:
    """B: the k most recent candidate positions [hi-k, hi) merged with the fixed rows. Same F and count as D0.

    Candidate-relative index i in [0, hi-lo) maps to row off_lo + i, exactly as the top-k indices do in
    `select_rows_kernel`; nothing is scored, nothing is renumbered.
    """
    B, G = fixed_rows.shape[:2]
    if k <= 0:
        return fixed_rows.reshape(-1)
    M = hi - lo
    if k > M:
        raise ValueError(f"k={k} exceeds the candidate count {M}")
    rel = torch.arange(M - k, M, device=fixed_rows.device, dtype=torch.long).view(1, 1, k).expand(B, G, k)
    return torch.cat([fixed_rows, rel + off_lo], dim=-1).view(-1)


def select_rows_uniform(fixed_rows, off_lo, lo: int, hi: int, k: int, *, rng) -> torch.Tensor:
    """C: k candidate positions drawn uniformly without replacement, independently per (batch, group), merged with F.

    `rng` is a numpy Generator (numpy.random.default_rng seeded from (uniform_seed, request, step) by the caller); the
    draw is the top-k of i.i.d. float64 uniforms, i.e. a uniformly random k-subset. The layers of one step consume the
    same generator in order, so a step replayed after a rollback draws the same rows. Not a learned choice.
    """
    import numpy as np  # local import: the package stays importable without numpy
    B, G = fixed_rows.shape[:2]
    if k <= 0:
        return fixed_rows.reshape(-1)
    M = hi - lo
    if k > M:
        raise ValueError(f"k={k} exceeds the candidate count {M}")
    u = torch.from_numpy(rng.random((B, G, M), dtype=np.float64))                # CPU, deterministic given the generator state
    rel = torch.topk(u, k, dim=-1).indices.to(fixed_rows.device)               # [B,G,k] candidate-relative
    return torch.cat([fixed_rows, rel + off_lo], dim=-1).view(-1)


def gathered_decode_step_kernel(q_main, k_cache, v_cache, rows, selected_count: int, *, scale=None, backend="sdpa"):
    """Read exactly the listed rows (flat ids into the flattened [B*G*C, d] cache) and attend over them.

    The shared tail of every selective read: D0's rows come from `select_rows_kernel`, B's from `select_rows_recent`,
    C's from `select_rows_uniform`. Two flattened index_selects (K, V), then subset attention; no torch.gather.
    """
    B, G, h, T, d = q_main.shape
    if T != 1:
        raise ValueError("a decode step takes exactly one new query position")
    k_sub = flatten_cache(k_cache).index_select(0, rows).view(B, G, selected_count, d)
    v_sub = flatten_cache(v_cache).index_select(0, rows).view(B, G, selected_count, d)
    return subset_attention(q_main, k_sub, v_sub, scale=scale, backend=backend)


def sparse_decode_step_kernel(q_main, q_pre, k_cache, v_cache, zT, wq_t_scaled, wk, fixed_rows, off_lo, lo: int,
                              hi: int, k: int, selected_count: int, *, k_pre_new=None, write_col: int = -1,
                              scale=None, backend="sdpa", return_rows: bool = False):
    """One layer's D read for one new query (T == 1). Every D-only cost is inside this function.

    `return_rows=True` also returns the flat row ids that were read (for selection diagnostics); the device work is the
    same either way.
    """
    B, G, h, T, d = q_main.shape
    if T != 1:
        raise ValueError("a decode step takes exactly one new query position")
    if k_pre_new is not None:
        zT[..., write_col:write_col + 1].copy_(project_keys(k_pre_new, wk))   # z of the newest key, its own column
    u = project_queries(q_pre, wq_t_scaled)                                     # [B,G,1,r]
    rows = select_rows_kernel(u, zT, fixed_rows, off_lo, lo, hi, k)
    out = gathered_decode_step_kernel(q_main, k_cache, v_cache, rows, selected_count, scale=scale, backend=backend)
    return (out, rows) if return_rows else out


def sparse_decode_step(q_main, q_pre, k_cache, v_cache, zT, indexer: GroupedIndexer, request: DecodeRequest, *,
                       k_pre_new=None, scale=None, backend="sdpa"):
    return sparse_decode_step_kernel(q_main, q_pre, k_cache, v_cache, zT, indexer.wq_t_scaled, indexer.wk,
                                     request.fixed_rows, request.off_lo, request.layout.candidate_lo,
                                     request.layout.candidate_hi, request.k, request.selected_count,
                                     k_pre_new=k_pre_new, write_col=request.write_col, scale=scale, backend=backend)


def dense_decode_step(q_main, k_cache, v_cache, valid_length: int, *, backend="sdpa_fold", scale=None):
    """One layer's A read for one new query: every valid position, K and V read, softmax, PV."""
    return full_attention(q_main, k_cache, v_cache, valid_length=valid_length, backend=backend, scale=scale)
