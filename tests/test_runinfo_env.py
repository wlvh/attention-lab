"""Environment guards must reject every spelling of zero and must not need torch (PR #1 re-review)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

# Load runinfo.py by path: importing the attention_lab package would pull in its torch-dependent modules, and this
# guard must be testable on the CI image that has no torch.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("attention_lab_runinfo_under_test", ROOT / "src/attention_lab/runinfo.py")
runinfo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runinfo
_spec.loader.exec_module(runinfo)


class EnvGuardTests(unittest.TestCase):
    def test_zero_spellings_are_rejected(self):
        for value in ("0", "0.0", "0e0", "-0.0", " 0 ", "0.00"):
            with self.subTest(value=value), patch.dict(os.environ, {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": value}, clear=False):
                self.assertTrue(runinfo.is_zero_ratio(value))
                with self.assertRaises(RuntimeError):
                    runinfo.refuse_unsafe_env()

    def test_non_zero_or_unset_passes(self):
        for value in ("0.5", "1.7", "2", "abc", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": value}, clear=False):
                self.assertFalse(runinfo.is_zero_ratio(value))
                runinfo.refuse_unsafe_env()
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTORCH_")}
        with patch.dict(os.environ, env, clear=True):
            runinfo.refuse_unsafe_env()

    def test_cpu_fallback_flag_is_rejected(self):
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, clear=False):
            with self.assertRaises(RuntimeError):
                runinfo.refuse_unsafe_env()
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}, clear=False):
            runinfo.refuse_unsafe_env()


if __name__ == "__main__":
    unittest.main()
