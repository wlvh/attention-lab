"""Attention Lab: the formal implementations of the read paths this project compares.

Scope at this version (G2-lite): the G1a single-layer read paths, the Qwen3 integration in `qwen_wrap` (per-layer
self_attn wrapper around the frozen body, static K/V cache, index table, prefill / decode drivers) and, from G2-lite,
the B (recent window) and C (uniform random) row-list generators beside the D0 index selector, plus the task data
generator and scorer in `g2lite_data` / `g2lite_eval`.
No training. `qwen_wrap` needs transformers and is imported explicitly (`from attention_lab.qwen_wrap import ...`)
so the random-tensor modules stay importable without it. Notebooks display and call this code; they do not keep a
second copy of the algorithms.
"""
__version__ = "0.3.0"
IMPLEMENTATION_STATUS = "g2lite_selectors_recent_uniform_index"

from .selection import (FIXED_PREFIX, FIXED_RECENT, SelectionLayout, fixed_budget,  # noqa: E402
                        legal_length, selection_layout)
from .reference import dense_mask_reference, dense_reference  # noqa: E402
from .sparse import flatten_cache, gather_attention, gather_rows, row_offsets, subset_attention  # noqa: E402
from .full import FULL_BACKENDS, full_attention  # noqa: E402
from .indexer import GroupedIndexer, candidate_scores, project_keys, project_queries  # noqa: E402
from .decode import (DecodeRequest, dense_decode_step, gathered_decode_step_kernel, prepare_request,  # noqa: E402
                     select_rows_kernel, select_rows_recent, select_rows_uniform, sparse_decode_step,
                     sparse_decode_step_kernel)

__all__ = [
    "FIXED_PREFIX", "FIXED_RECENT", "SelectionLayout", "fixed_budget", "legal_length", "selection_layout",
    "dense_reference", "dense_mask_reference",
    "flatten_cache", "gather_attention", "gather_rows", "row_offsets", "subset_attention",
    "FULL_BACKENDS", "full_attention",
    "GroupedIndexer", "candidate_scores", "project_keys", "project_queries",
    "DecodeRequest", "dense_decode_step", "gathered_decode_step_kernel", "prepare_request", "select_rows_kernel",
    "select_rows_recent", "select_rows_uniform", "sparse_decode_step", "sparse_decode_step_kernel",
]
