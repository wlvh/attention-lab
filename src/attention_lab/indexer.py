"""Grouped indexer: one (Wq, Wk) pair per KV group. u = Wq·concat_h(q_pre[g]); z = Wk·k_pre; score = u·z/√r.

Index keys are stored transposed and contiguous as zT [B,G,r,C] (G0 constraint: [G, r, N] layout), so one
decode step scores the candidate range with a single batched matmul over the contiguous slice zT[..., lo:hi].
Inputs are the pre-RoPE, post-norm q/k (docs/PLAN_CURRENT.md §5); with random tensors that distinction is
only a naming convention, and it becomes a tested requirement in G1b.
"""
from __future__ import annotations

import math

import torch


def project_keys(k_pre: torch.Tensor, wk: torch.Tensor) -> torch.Tensor:
    """k_pre [B,G,C,d], wk [G,r,d] -> zT [B,G,r,C] (contiguous matmul output)."""
    return torch.matmul(wk.unsqueeze(0), k_pre.transpose(-1, -2))


def project_queries(q_pre: torch.Tensor, wq_t_scaled: torch.Tensor) -> torch.Tensor:
    """q_pre [B,G,h,T,d], wq_t_scaled [G,h*d,r] -> u [B,G,T,r], with 1/√r already folded into the weights."""
    B, G, h, T, d = q_pre.shape
    q_cat = q_pre.permute(0, 1, 3, 2, 4).reshape(B, G, T, h * d)
    return torch.matmul(q_cat, wq_t_scaled.unsqueeze(0))


def candidate_scores(u: torch.Tensor, zT: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """u [B,G,T,r] against the candidate columns only -> [B,G,T,hi-lo]. Nothing outside [lo, hi) is scanned."""
    return torch.matmul(u, zT[..., lo:hi])


class GroupedIndexer:
    """Untrained (D0-style) random indexer for one layer. G1a measures its cost; training is G3."""

    def __init__(self, groups: int, heads_per_group: int, head_dim: int, r: int = 32, *, device="cpu",
                 dtype=torch.float32, seed: int = 0, init_std: float = 0.02):
        self.groups, self.heads_per_group, self.head_dim, self.r = int(groups), int(heads_per_group), int(head_dim), int(r)
        gen = torch.Generator().manual_seed(int(seed))
        # weights are drawn and scaled in float64 on the CPU, then cast once, so every working dtype starts from the
        # same values and the float64 reference path is exact
        wq = torch.randn(groups, r, heads_per_group * head_dim, generator=gen, dtype=torch.float64) * init_std
        wk = torch.randn(groups, r, head_dim, generator=gen, dtype=torch.float64) * init_std
        self.wq = wq.to(device=device, dtype=dtype)                      # [G, r, h*d], unscaled, reference use
        self.wk = wk.to(device=device, dtype=dtype)                      # [G, r, d]
        self.wq_t_scaled = (wq / math.sqrt(r)).transpose(1, 2).contiguous().to(device=device, dtype=dtype)  # [G, h*d, r]

    def keys(self, k_pre: torch.Tensor) -> torch.Tensor:
        """Build (or extend, chunk by chunk) the transposed index table: [B,G,C,d] -> [B,G,r,C]."""
        return project_keys(k_pre, self.wk)

    def query(self, q_pre: torch.Tensor) -> torch.Tensor:
        return project_queries(q_pre, self.wq_t_scaled)

    def scores(self, u: torch.Tensor, zT: torch.Tensor, lo: int = 0, hi=None) -> torch.Tensor:
        return candidate_scores(u, zT, lo, zT.shape[-1] if hi is None else hi)

    def reference_scores(self, q_pre: torch.Tensor, k_pre: torch.Tensor, lo: int = 0, hi=None) -> torch.Tensor:
        """Brute force in float64 on CPU from the stored weights: (Wq·concat q)·(Wk·k)/√r over [lo, hi)."""
        B, G, h, T, d = q_pre.shape
        hi = k_pre.shape[2] if hi is None else hi
        q_cat = q_pre.permute(0, 1, 3, 2, 4).reshape(B, G, T, h * d).cpu().double()   # MPS has no float64: move first
        u = torch.einsum("bgtf,grf->bgtr", q_cat, self.wq.cpu().double())
        z = torch.einsum("grd,bgcd->bgcr", self.wk.cpu().double(), k_pre.cpu().double())
        return torch.einsum("bgtr,bgcr->bgtc", u, z)[..., lo:hi] / math.sqrt(self.r)
