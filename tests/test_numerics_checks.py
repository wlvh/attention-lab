"""The G1b numerics checks must fail on a NaN on either side (PR #2 review: Python's max(0.0, nan) returned 0.0).

bench/model_numerics.py is a script, so it is loaded by path. Only the comparison helpers are exercised: small LabCaches
on the CPU, a stand-in for the transformers cache, and logits tensors. No model, no MPS.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
except ImportError:
    torch = None
try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = torch is not None
except ImportError:
    HAVE_TRANSFORMERS = False

if HAVE_TRANSFORMERS:
    _spec = importlib.util.spec_from_file_location("model_numerics_under_test", ROOT / "bench/model_numerics.py")
    mn = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = mn
    _spec.loader.exec_module(mn)
    from attention_lab.qwen_wrap import LabCache

TOL = {"logits_atol": 1e-2, "logits_rel_norm": 1e-3, "cache_atol": 1e-3, "argmax_agreement_min": 0.99}
RULE = {"factor": 2.0}


def make_caches(L=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    c1 = LabCache(layers=2, batch=1, groups=2, capacity=16, head_dim=8, index_dim=4, device="cpu", dtype=torch.float32)
    c2 = LabCache(layers=2, batch=1, groups=2, capacity=16, head_dim=8, index_dim=4, device="cpu", dtype=torch.float32)
    for l in range(2):
        k, v, z = torch.randn(1, 2, 16, 8, generator=gen), torch.randn(1, 2, 16, 8, generator=gen), torch.randn(1, 2, 4, 16, generator=gen)
        for c in (c1, c2):
            c.k[l].copy_(k); c.v[l].copy_(v); c.z[l].copy_(z)
            c.k[l][:, :, L:] = float("nan"); c.v[l][:, :, L:] = float("nan"); c.z[l][..., L:] = float("nan")   # unwritten slots: allowed
        c1.valid_length = c2.valid_length = L
    return c1, c2


def fake_hf_cache(c, L):
    return SimpleNamespace(layers=[SimpleNamespace(keys=c.k[l][:, :, :L].clone(), values=c.v[l][:, :, :L].clone()) for l in range(c.layers)])


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed (or torch not installed)")
class NumericsCheckTests(unittest.TestCase):
    def test_worst_propagates_nan_and_python_max_does_not(self):
        self.assertEqual(max(0.0, float("nan")), 0.0, "sanity: the bug this guards against")
        self.assertTrue(mn.worst([0.0, float("nan"), 1.0]) != mn.worst([0.0, float("nan"), 1.0]), "worst() must return NaN")
        self.assertEqual(mn.worst([0.5, 0.25]), 0.5)
        self.assertEqual(mn.worst([]), 0.0)

    def test_cache_vs_cache_passes_when_finite_and_fails_on_nan_on_either_side(self):
        L = 10
        c1, c2 = make_caches(L)
        ok = mn.cache_vs_cache(c1, c2, L, TOL, with_z=True)
        self.assertTrue(ok["pass"] and ok["finite"] and ok["pass_preset"], "identical finite caches must pass (NaN beyond L is allowed)")
        for side in (0, 1):
            for field in ("k", "v", "z"):
                with self.subTest(side=side, field=field):
                    a, b = make_caches(L)
                    target = (a, b)[side]
                    if field == "z":
                        getattr(target, field)[1][0, 1, 2, L - 1] = float("nan")       # inside the compared region
                    else:
                        getattr(target, field)[1][0, 1, L - 1, 3] = float("nan")
                    r = mn.cache_vs_cache(a, b, L, TOL, with_z=True)
                    self.assertFalse(r["finite"], "NaN not detected")
                    self.assertFalse(r["pass"])
                    self.assertFalse(r["pass_preset"])
                    # the revised bf16 rule must not turn it into a pass, whatever the floor says
                    mn.apply_revised_rule(r, {"max_abs_k": 1.0, "max_abs_v": 1.0, "finite": True}, RULE, kind="cache")
                    self.assertFalse(r["pass"])
                    self.assertFalse(r["pass_revised"])

    def test_cache_compare_fails_on_nan_on_either_side(self):
        L = 10
        c1, _ = make_caches(L)
        hf = fake_hf_cache(c1, L)
        self.assertTrue(mn.cache_compare(c1, hf, L, TOL)["pass"])
        for side in ("static", "hf"):
            for field in ("k", "v"):
                with self.subTest(side=side, field=field):
                    a, _ = make_caches(L)
                    hf = fake_hf_cache(a, L)
                    if side == "static":
                        getattr(a, field)[0][0, 0, 2, 1] = float("nan")
                    else:
                        (hf.layers[0].keys if field == "k" else hf.layers[0].values)[0, 0, 2, 1] = float("nan")
                    r = mn.cache_compare(a, hf, L, TOL)
                    self.assertFalse(r["finite"])
                    self.assertFalse(r["pass"])

    def test_logits_compare_fails_on_nan_on_either_side_and_revised_rule_cannot_flip_it(self):
        gen = torch.Generator().manual_seed(1)
        ref = torch.randn(1, 6, 50, generator=gen)
        good = mn.compare(ref + 1e-6, ref, TOL, kind="logits")
        self.assertTrue(good["pass"] and good["finite"])
        for side in (0, 1):
            a, b = ref.clone(), ref.clone()
            (a, b)[side][0, 3, 7] = float("nan")
            r = mn.compare(a, b, TOL, kind="logits")
            self.assertFalse(r["finite"])
            self.assertFalse(r["pass"])
            mn.apply_revised_rule(r, {"max_abs": 10.0, "rel_norm": 10.0, "finite": True}, RULE, kind="logits")
            self.assertFalse(r["pass"])
        # a non-finite noise floor cannot justify a pass either
        r = mn.compare(ref + 1e-6, ref, TOL, kind="logits")
        mn.apply_revised_rule(r, {"max_abs": float("nan"), "rel_norm": float("nan"), "finite": False}, RULE, kind="logits")
        self.assertFalse(r["pass_revised"])


if __name__ == "__main__":
    unittest.main()
