#!/usr/bin/env python3
"""Verify archived source bytes. No network, model loading or experiment execution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def verify(root: Path = ROOT) -> list[str]:
    root = root.resolve()
    manifest = json.loads((root / "docs/source_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"Missing or invalid source: {item['path']}")
            continue
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            errors.append(f"Source changed: {item['path']}")
    return errors

if __name__ == "__main__":
    failures = verify()
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("Archived source sizes and SHA-256 hashes match the manifest. Source claims are not thereby verified.")
