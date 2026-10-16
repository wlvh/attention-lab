#!/usr/bin/env python3
"""Bounded inventory only. Does not benchmark, train, download or verify kernels."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def inventory() -> dict:
    info = {
        "kind": "capability_inventory_only", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(), "machine": platform.machine(), "os_release": platform.release(),
        "macos": platform.mac_ver()[0] or None, "python_version": platform.python_version(),
        "python_executable": sys.executable, "physical_memory_bytes": None,
        "packages": {}, "mps": {"is_built": False, "is_available": False, "operations_tested": False},
        "compile_tested": False, "cuda": {"is_available": False, "device_count": 0},
        "relevant_env": {k: os.environ.get(k) for k in (
            "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO", "PYTORCH_MPS_FAST_MATH")},
        "limitations": ["Inventory is not a numerical test, memory peak measurement or performance result."],
    }
    try:
        if platform.system() == "Darwin":
            proc = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=True)
            info["physical_memory_bytes"] = int(proc.stdout.strip())
        elif hasattr(os, "sysconf"):
            info["physical_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    for name in ("torch", "numpy", "transformers", "jupyterlab", "nbformat", "nbclient", "nbconvert"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch
            info["mps"]["is_built"] = bool(torch.backends.mps.is_built())
            info["mps"]["is_available"] = bool(torch.backends.mps.is_available())
            info["mps"]["api_presence"] = {n: hasattr(torch.mps, n) for n in (
                "synchronize", "current_allocated_memory", "driver_allocated_memory", "recommended_max_memory")}
            info["cuda"]["is_available"] = bool(torch.cuda.is_available())
            info["cuda"]["device_count"] = torch.cuda.device_count() if info["cuda"]["is_available"] else 0
        except Exception as e:
            info["torch_inventory_error"] = f"{type(e).__name__}: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--require-mps", action="store_true")
    args = ap.parse_args()
    data = inventory()
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 2 if args.require_mps and not data["mps"]["is_available"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
