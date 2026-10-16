"""Attention Lab: the formal implementations of the read paths this project compares.

Scope at this version (G1a): single-layer, random-tensor read paths for one decode step.
No model wrapper, no training, no cache lifecycle yet. Notebooks display and call this code;
they do not keep a second copy of the algorithms.
"""
__version__ = "0.1.0"
IMPLEMENTATION_STATUS = "g1a_single_layer_read_paths"

from .selection import (FIXED_PREFIX, FIXED_RECENT, SelectionLayout, fixed_budget,  # noqa: E402
                        legal_length, selection_layout)
from .reference import dense_mask_reference, dense_reference  # noqa: E402
from .sparse import flatten_cache, gather_attention, gather_rows, row_offsets, subset_attention  # noqa: E402
from .full import FULL_BACKENDS, full_attention  # noqa: E402
from .indexer import GroupedIndexer, candidate_scores, project_keys, project_queries  # noqa: E402
from .decode import (DecodeRequest, dense_decode_step, prepare_request, select_rows_kernel,  # noqa: E402
                     sparse_decode_step, sparse_decode_step_kernel)

__all__ = [
    "FIXED_PREFIX", "FIXED_RECENT", "SelectionLayout", "fixed_budget", "legal_length", "selection_layout",
    "dense_reference", "dense_mask_reference",
    "flatten_cache", "gather_attention", "gather_rows", "row_offsets", "subset_attention",
    "FULL_BACKENDS", "full_attention",
    "GroupedIndexer", "candidate_scores", "project_keys", "project_queries",
    "DecodeRequest", "dense_decode_step", "prepare_request", "select_rows_kernel",
    "sparse_decode_step", "sparse_decode_step_kernel",
]
