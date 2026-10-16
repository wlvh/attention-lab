"""Fixed set, candidate domain and read-budget rules. Pure Python, no torch dependency.

Rules (docs/PLAN_CURRENT.md §4, docs/IMPLEMENTATION.md §3):

- A key position j is legal for a query at position p when 0 <= j < valid_length and j <= p.
- F = first ``fixed_prefix`` legal positions ∪ last ``fixed_recent`` legal positions, deduplicated.
- The candidate domain is the legal range minus F. It is always one contiguous range [lo, hi).
- k0 = min(round(rho * N0), |candidates at N0|) is fixed once per request from its starting length N0.
- At every later step k = min(k0, |candidates now|). The read list is F ∪ top-k(candidates), kept as
  original positions; nothing is renumbered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FIXED_PREFIX = 4
FIXED_RECENT = 64


def legal_length(query_position: int, valid_length: int) -> int:
    """Number of legal key positions, i.e. the legal range is [0, legal_length)."""
    query_position = int(query_position)
    valid_length = int(valid_length)
    if query_position < 0 or valid_length < 0:
        raise ValueError(f"negative query_position={query_position} or valid_length={valid_length}")
    n = min(valid_length, query_position + 1)
    if n <= 0:
        raise ValueError(f"no legal key position for query_position={query_position}, valid_length={valid_length}")
    return n


@dataclass(frozen=True)
class SelectionLayout:
    """Where F and the candidate domain sit inside the legal range [0, legal_length)."""

    legal_length: int
    fixed_prefix: int      # leading fixed positions are [0, fixed_prefix)
    recent_start: int      # trailing fixed positions are [recent_start, legal_length)

    @property
    def candidate_lo(self) -> int:
        return self.fixed_prefix

    @property
    def candidate_hi(self) -> int:
        return self.recent_start

    @property
    def candidate_count(self) -> int:
        return self.recent_start - self.fixed_prefix

    @property
    def fixed_count(self) -> int:
        return self.fixed_prefix + (self.legal_length - self.recent_start)

    def fixed_positions(self) -> list[int]:
        return list(range(self.fixed_prefix)) + list(range(self.recent_start, self.legal_length))

    def clamp_k(self, k: int) -> int:
        if k < 0:
            raise ValueError("k must be non-negative")
        return min(int(k), self.candidate_count)

    def selected_count(self, k: int) -> int:
        return self.fixed_count + self.clamp_k(k)


def selection_layout(legal: int, fixed_prefix: int = FIXED_PREFIX, fixed_recent: int = FIXED_RECENT) -> SelectionLayout:
    legal = int(legal)
    if legal <= 0:
        raise ValueError("legal length must be positive")
    if fixed_prefix < 0 or fixed_recent < 0:
        raise ValueError("fixed set sizes must be non-negative")
    prefix = min(fixed_prefix, legal)
    recent_start = max(prefix, legal - fixed_recent)
    return SelectionLayout(legal, prefix, recent_start)


def fixed_budget(n0: int, rho: float, fixed_prefix: int = FIXED_PREFIX, fixed_recent: int = FIXED_RECENT) -> int:
    """k0 for a request that starts with n0 legal positions. ``round`` is Python ties-to-even, as in scripts/roofline.py."""
    rho = float(rho)
    if math.isnan(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    layout = selection_layout(n0, fixed_prefix, fixed_recent)
    return min(round(rho * int(n0)), layout.candidate_count)
