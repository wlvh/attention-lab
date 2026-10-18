"""CPU-only request identity and completion bookkeeping for G2-lite.

These rules apply to future runs. They do not relabel or repair the archived v2
runs, whose C rewrites used different random streams from their parent documents.
"""
from __future__ import annotations

import hashlib


UNIFORM_SEED_RULE = "parent_doc_id_or_doc_id/v2"


def uniform_seed_for(cfg: dict, doc: dict) -> int:
    """Hold C's external randomness constant across rewrites of one document.

The selector still keys its stream by (seed, request counter, step), so matching
selections also requires matching candidate domains and k. D0 does not use this
seed: it continues to score the changed inputs with its fixed indexer parameters.
    """
    identity = doc.get("parent_doc_id") or doc["doc_id"]
    return int(cfg["seeds"]["uniform_base"]) + int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % (1 << 20)


class RunProgress:
    """Account for every planned request, including work omitted at a stop gate."""

    def __init__(self, doc_ids, requests):
        self.doc_ids = list(doc_ids)
        self.requests = list(requests)
        self.planned = {(d, m, rho) for d in self.doc_ids for m, rho in self.requests}
        if len(self.planned) != len(self.doc_ids) * len(self.requests):
            raise ValueError("duplicate document or request in run plan")
        self.completed = set()
        self.skipped = {}
        self.stop_reason = None

    def complete(self, doc_id, method, rho):
        key = (doc_id, method, rho)
        if key not in self.planned or key in self.completed or key in self.skipped:
            raise ValueError(f"unplanned or already accounted request: {key}")
        self.completed.add(key)

    def skip_document(self, doc_id, reason):
        if doc_id not in self.doc_ids:
            raise ValueError(f"unplanned document: {doc_id}")
        for method, rho in self.requests:
            key = (doc_id, method, rho)
            if key not in self.completed and key not in self.skipped:
                self.skipped[key] = reason

    def stop(self, reason):
        self.stop_reason = reason
        for doc_id in self.doc_ids:
            self.skip_document(doc_id, reason)

    def snapshot(self, *, final=False):
        pending = self.planned - self.completed - self.skipped.keys()
        complete_docs = sum(all((d, m, r) in self.completed for m, r in self.requests) for d in self.doc_ids)
        partial_docs = sum(any((d, m, r) in self.completed for m, r in self.requests)
                           and not all((d, m, r) in self.completed for m, r in self.requests) for d in self.doc_ids)
        skipped_docs = sum(any((d, m, r) in self.skipped for m, r in self.requests) for d in self.doc_ids)
        done = bool(self.planned) and self.completed == self.planned
        return {"status": ("done" if done else "incomplete") if final else "running",
                "planned_docs": len(self.doc_ids), "completed_docs": complete_docs,
                "partial_docs": partial_docs, "skipped_docs": skipped_docs,
                "planned_requests": len(self.planned), "completed_requests": len(self.completed),
                "skipped_requests": len(self.skipped), "pending_requests": len(pending),
                "stop_reason": self.stop_reason,
                "skipped": [{"doc_id": d, "method": m, "rho": r, "reason": why}
                            for (d, m, r), why in self.skipped.items()]}
