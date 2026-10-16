"""Run provenance helpers: host, software, git state, source hashes, environment guards. No measurement here."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize(text: str) -> str:
    return str(text).replace(os.path.expanduser("~"), "~")


def _run(args, timeout=10) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()


def host_info() -> dict:
    info = {"system": platform.system(), "machine": platform.machine(), "macos": platform.mac_ver()[0] or None,
            "python": platform.python_version(), "chip": None, "gpu_cores": None, "physical_memory_bytes": None}
    try:
        info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or None
        info["physical_memory_bytes"] = int(_run(["sysctl", "-n", "hw.memsize"]))
        disp = json.loads(_run(["system_profiler", "-json", "SPDisplaysDataType"], timeout=30))
        for gpu in disp.get("SPDisplaysDataType", []):
            if gpu.get("sppci_cores"):
                info["gpu_cores"] = int(gpu["sppci_cores"])
    except Exception as e:  # inventory only
        info["host_info_error"] = f"{type(e).__name__}: {sanitize(e)}"
    return info


def git_info(root: Path) -> dict:
    out = {"commit": None, "branch": None, "dirty_files": None, "dirty_diff_sha256": None}
    try:
        out["commit"] = _run(["git", "-C", str(root), "rev-parse", "HEAD"]) or None
        out["branch"] = _run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]) or None
        status = _run(["git", "-C", str(root), "status", "--porcelain"])
        out["dirty_files"] = [line[3:] for line in status.splitlines()] if status else []
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD"], capture_output=True, timeout=30).stdout
        out["dirty_diff_sha256"] = hashlib.sha256(diff).hexdigest() if diff else None
    except Exception as e:
        out["git_error"] = f"{type(e).__name__}: {sanitize(e)}"
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root: Path, patterns=("src/attention_lab/*.py", "bench/*.py")) -> dict:
    out = {}
    for pattern in patterns:
        for p in sorted(Path(root).glob(pattern)):
            out[str(p.relative_to(root))] = sha256_file(p)
    return out


ENV_KEYS = ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            "PYTORCH_MPS_FAST_MATH", "TORCH_COMPILE_DEBUG")


def torch_env(torch) -> dict:
    mps = torch.backends.mps
    out = {"torch": torch.__version__, "python_executable": sanitize(sys.executable),
           "mps_built": bool(mps.is_built()), "mps_available": bool(mps.is_available()),
           "env": {k: os.environ.get(k) for k in ENV_KEYS}}
    if out["mps_available"]:
        for name in ("recommended_max_memory", "current_allocated_memory", "driver_allocated_memory"):
            fn = getattr(torch.mps, name, None)
            out[name] = int(fn()) if fn else None
    return out


def refuse_unsafe_env() -> None:
    """Pure-MPS runs must not hide unsupported ops behind CPU fallback or disable the memory guard."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 would hide unsupported MPS ops behind CPU fallback")
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") == "0":
        raise RuntimeError("PYTORCH_MPS_HIGH_WATERMARK_RATIO=0 disables the MPS memory guard")
