"""Shared wall-clock budget and immutable execution identity for G3b/G3c/G4-lite.

No model execution on import. One registered process at a time; every real run,
including failed runs, charges the same ledger. Existing G3a accounting is read-only.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execution_identity(root, config_path, runner, versions=None):
    root = Path(root)
    cfg = json.loads(Path(config_path).read_text())
    sources = sorted(root.glob("src/attention_lab/*.py")) + [root / runner]
    return {"sources": {str(p.relative_to(root)): digest(p) for p in sources},
            "configs": {str(Path(config_path).relative_to(root)): digest(config_path),
                        cfg["base_config"]: digest(root / cfg["base_config"]),
                        cfg["g3a_config"]: digest(root / cfg["g3a_config"])},
            "versions": versions or {}}


def require_preflight(manifest, execution):
    if manifest.get("status") != "done" or manifest.get("mode") != "preflight":
        raise RuntimeError("completed preflight required")
    if not manifest.get("planned") or manifest.get("completed") != manifest["planned"]:
        raise RuntimeError("incomplete preflight counts")
    if manifest.get("execution") != execution:
        raise RuntimeError("preflight execution sources/configs/versions differ")


class BudgetStop(RuntimeError):
    pass


class CampaignBudget:
    def __init__(self, root, cfg, run_id, *, clock=time.monotonic):
        self.root, self.cfg, self.run_id, self.clock = Path(root), cfg, run_id, clock
        self.path = self.root / "runs" / f'{cfg["campaign"]}-budget.json'
        self.start = None

    def __enter__(self):
        if self.path.exists():
            self.ledger = json.loads(self.path.read_text())
        else:
            old = self.root / "runs/g3a-budget-20261018.json"
            self.ledger = {"campaign": self.cfg["campaign"], "limit_seconds": self.cfg["budget_seconds"],
                           "used_seconds": 0.0, "runs": [], "prior_ledger": str(old.relative_to(self.root)),
                           "prior_ledger_sha256": digest(old), "prior_g3a_seconds": json.loads(old.read_text())["used_seconds"]}
        if self.ledger["limit_seconds"] != self.cfg["budget_seconds"]:
            raise RuntimeError("cannot silently reset campaign budget")
        if self.ledger.get("active_pid"):
            raise RuntimeError("registered run still active or interrupted; reconcile PID/ledger before continuing")
        if self.ledger["used_seconds"] >= self.ledger["limit_seconds"]:
            raise BudgetStop("shared campaign budget exhausted")
        self.start = self.clock()
        self.ledger.update(active_pid=os.getpid(), active_run=self.run_id, active_started_unix=time.time())
        write_json(self.path, self.ledger)
        return self

    @property
    def remaining(self):
        return self.ledger["limit_seconds"] - self.ledger["used_seconds"] - (self.clock() - self.start)

    def check(self, *, estimated_next=0, reserve=0):
        if self.remaining <= estimated_next + reserve:
            raise BudgetStop(f"remaining {self.remaining:.1f}s cannot cover next batch {estimated_next:.1f}s + reserve {reserve:.1f}s")
        if shutil.disk_usage(self.root).free < self.cfg["minimum_free_disk_bytes"]:
            raise BudgetStop("disk safety reserve reached")

    def __exit__(self, kind, value, tb):
        elapsed = self.clock() - self.start
        self.ledger["used_seconds"] += elapsed
        self.ledger["runs"].append({"run_id": self.run_id, "elapsed_seconds": elapsed,
                                    "status": "done" if kind is None else "incomplete",
                                    "reason": None if value is None else str(value)})
        self.ledger.update(active_pid=None, active_run=None, active_started_unix=None)
        write_json(self.path, self.ledger)
        return False


class ResourceGuard:
    def __init__(self, budget, torch):
        import psutil
        self.budget, self.torch, self.psutil = budget, torch, psutil
        self.swap0 = psutil.swap_memory().used
        self.samples = []

    def check(self, *, record=False, estimated_next=0, reserve=0):
        self.budget.check(estimated_next=estimated_next, reserve=reserve)
        p, t, cfg = self.psutil, self.torch, self.budget.cfg
        sample = {"rss": p.Process().memory_info().rss, "mps_driver": t.mps.driver_allocated_memory(),
                  "available": p.virtual_memory().available, "swap": p.swap_memory().used}
        if max(sample["rss"], sample["mps_driver"]) >= cfg["stop_bytes"] or sample["available"] < 4_000_000_000:
            raise BudgetStop("memory safety boundary reached")
        if sample["swap"] - self.swap0 > cfg["swap_growth_stop_bytes"]:
            raise BudgetStop("swap growth boundary reached")
        sample["warning"] = max(sample["rss"], sample["mps_driver"]) >= cfg["warning_bytes"]
        if record:
            self.samples.append(sample)
        return sample
