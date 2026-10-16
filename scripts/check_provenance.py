#!/usr/bin/env python3
"""Verify that the archived benchmark evidence matches what the committed reports say was run. No torch, no network.

Checks (PR #1 review, finding 1):
1. The as-run source snapshot reports/provenance/microbench_at_formal_runs.py has the SHA-256 recorded for
   bench/microbench.py in every formal run manifest aggregated in reports/g1a_manifest_mac_m4max.json.
2. Every file listed in reports/provenance/archive_manifest.json has the recorded size and SHA-256.
3. The committed bench/microbench.py differs from the as-run snapshot only by the lines stored in
   reports/provenance/microbench_post_run_edit.diff (so any later edit must update that diff deliberately).
"""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "reports/provenance"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_lines(diff_text: str) -> list[str]:
    return sorted(l for l in diff_text.splitlines() if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))


def main() -> int:
    errors = []
    snap = PROV / "microbench_at_formal_runs.py"
    manifest = json.loads((ROOT / "reports/g1a_manifest_mac_m4max.json").read_text(encoding="utf-8"))
    snap_sha = sha256(snap)
    for run in manifest["runs"]:
        recorded = run.get("source_hashes", {}).get("bench/microbench.py")
        if recorded != snap_sha:
            errors.append(f"{run.get('run_id')}: manifest hash {str(recorded)[:16]} != snapshot {snap_sha[:16]}")
    am = json.loads((PROV / "archive_manifest.json").read_text(encoding="utf-8"))
    for item in am["files"]:
        p = ROOT / item["path"]
        if not p.is_file():
            errors.append(f"missing {item['path']}")
            continue
        if p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            errors.append(f"changed {item['path']}")
    stored = changed_lines((PROV / "microbench_post_run_edit.diff").read_text(encoding="utf-8"))
    current = changed_lines("".join(difflib.unified_diff(snap.read_text(encoding="utf-8").splitlines(True),
                                                         (ROOT / "bench/microbench.py").read_text(encoding="utf-8").splitlines(True))))
    if stored != current:
        errors.append("bench/microbench.py differs from the as-run snapshot by lines not recorded in microbench_post_run_edit.diff; "
                      "regenerate the diff file deliberately if the edit is intended")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Provenance OK: snapshot {snap_sha[:16]} matches {len(manifest['runs'])} run manifests; "
          f"{len(am['files'])} archived files verified; post-run edit limited to the recorded diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
