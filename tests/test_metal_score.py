"""Contracts for the P0 Metal score kernel (docs/tasks/04_FUSED_D_KERNEL.md). MPS + torch.mps.compile_shader only.

Tolerances are preset here before the first measurement: fused fp32 scores vs the CPU float32 reference computed
from the same bf16 inputs: max |Δ| <= 1e-5 and relative norm <= 1e-5; the z column vs project_keys: within one bf16
ulp (relative 2^-8) element-wise. The two launch variants must agree with each other. Columns outside the scanned
range (including unwritten slots) may hold NaN without affecting the scores.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from attention_lab import metal_score
    from attention_lab.decode import prepare_request
    from attention_lab.indexer import GroupedIndexer, candidate_scores, project_keys, project_queries

READY = torch is not None and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
TOL = {"scores_atol": 1e-5, "scores_rel_norm": 1e-5, "z_rel_ulp": 2.0 ** -8}


def make(G=8, h=2, d=128, r=32, C=600, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dev = torch.device("mps")
    zT4 = torch.randn(1, G, r, C, generator=gen).to(dev, torch.bfloat16)
    k_pre_new = torch.randn(1, G, 1, d, generator=gen).to(dev, torch.bfloat16)
    q_pre = torch.randn(1, G, h, 1, d, generator=gen).to(dev, torch.bfloat16)
    idx = GroupedIndexer(G, h, d, r, device=dev, dtype=torch.bfloat16, seed=seed)
    return zT4, k_pre_new, q_pre, idx


@unittest.skipUnless(READY, "needs MPS and torch.mps.compile_shader (P0 Metal kernel)")
class MetalScoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ms = metal_score.MetalScore()

    def _case(self, C, L, seed=0):
        zT4, k_pre_new, q_pre, idx = make(C=C, seed=seed)
        req = prepare_request(1, 8, C, valid_length=L, query_position=L - 1, rho=1 / 8, device="mps")
        lo, hi, write_col = req.layout.candidate_lo, req.layout.candidate_hi, req.write_col
        return zT4, k_pre_new, q_pre, idx, lo, hi, write_col

    def test_fused_matches_float32_reference_and_two_launch(self):
        # (C, L): M = 512, 932, 2173, 2175, 1003 -> tails of 0, 3, 1, 3, 3 columns; C=601 / 2301 force the scalar fallback
        for C, L in ((600, 584), (1024, 1000), (2304, 2176 + 65), (2304, 2176 + 67), (601, 585), (2301, 2176 + 65)):
            with self.subTest(C=C, L=L), torch.inference_mode():
                zT4, k_pre_new, q_pre, idx, lo, hi, write_col = self._case(C, L)
                zT, k_new, wk, q_cat, wq_t = metal_score.model_views(zT4, k_pre_new, q_pre, idx)
                u_ref, z_ref, s_ref = metal_score.reference_scores(zT, k_new, wk, q_cat, wq_t, lo, hi)
                outs = {}
                for version in (1, 2):
                    s1 = self.ms.fused(zT.clone(), k_new, wk, q_cat, wq_t, lo, hi, write_col, version=version)
                    torch.mps.synchronize()
                    self.assertEqual(tuple(s1.shape), (8, hi - lo))
                    diff = float((s1.cpu() - s_ref).abs().max())
                    rel = float((s1.cpu() - s_ref).norm() / s_ref.norm())
                    self.assertLessEqual(diff, TOL["scores_atol"], f"v{version} fused scores max abs diff {diff:.3e}")
                    self.assertLessEqual(rel, TOL["scores_rel_norm"], f"v{version} fused scores rel norm {rel:.3e}")
                    s2 = self.ms.two_launch(zT.clone(), k_new, wk, q_cat, wq_t, lo, hi, write_col, version=version)
                    torch.mps.synchronize()
                    self.assertLessEqual(float((s2.cpu() - s1.cpu()).abs().max()), 1e-6, f"v{version} two-launch variant disagrees with fused")
                    outs[version] = s1.cpu()
                self.assertLessEqual(float((outs[1] - outs[2]).abs().max()), 1e-6, "v1 and v2 disagree")
                diff = float((outs[2] - s_ref).abs().max())
                # the PyTorch path (bf16 output) against the same reference: recorded, and the fused kernel must not be worse
                z4 = zT4.clone()
                z4[..., write_col:write_col + 1].copy_(project_keys(k_pre_new, idx.wk))
                s_pt = candidate_scores(project_queries(q_pre, idx.wq_t_scaled), z4, lo, hi)[0, :, 0, :]
                pt_diff = float((s_pt.float().cpu() - s_ref).abs().max())
                self.assertLessEqual(diff, pt_diff, "fused kernel less accurate than the bf16 PyTorch path")

    def test_z_column_written_once_within_one_bf16_ulp(self):
        with torch.inference_mode():
            zT4, k_pre_new, q_pre, idx, lo, hi, write_col = self._case(1024, 1000)
            zT, k_new, wk, q_cat, wq_t = metal_score.model_views(zT4, k_pre_new, q_pre, idx)
            before = zT.clone()
            self.ms.fused(zT, k_new, wk, q_cat, wq_t, lo, hi, write_col)
            torch.mps.synchronize()
            want = project_keys(k_pre_new, idx.wk)[0, :, :, 0]                       # [G, r] bf16 from the PyTorch path
            got = zT[:, :, write_col]
            tol = TOL["z_rel_ulp"] * want.float().abs() + 1e-6
            self.assertTrue(bool(((got.float() - want.float()).abs() <= tol).all()), "z column differs by more than one bf16 ulp")
            mask = torch.ones(zT.shape[-1], dtype=torch.bool, device=zT.device)
            mask[write_col] = False
            self.assertTrue(torch.equal(zT[:, :, mask], before[:, :, mask]), "a column other than write_col was modified")
            exact = float((got == want).float().mean())
            self.assertGreaterEqual(exact, 0.9, f"only {exact:.3f} of z entries bitwise equal to project_keys")

    def test_columns_outside_the_scan_may_be_garbage(self):
        with torch.inference_mode():
            zT4, k_pre_new, q_pre, idx, lo, hi, write_col = self._case(1024, 1000)
            zT, k_new, wk, q_cat, wq_t = metal_score.model_views(zT4, k_pre_new, q_pre, idx)
            poisoned = zT.clone()
            poisoned[:, :, :lo] = float("nan")
            poisoned[:, :, hi:] = float("nan")                                        # recent window, write_col, unwritten slots
            for version in (1, 2):                                                    # v1 and v2 differ by fp32 fma rounding, so compare each with itself
                clean = self.ms.fused(zT.clone(), k_new, wk, q_cat, wq_t, lo, hi, write_col, version=version)
                out = self.ms.fused(poisoned.clone(), k_new, wk, q_cat, wq_t, lo, hi, write_col, version=version)
                torch.mps.synchronize()
                self.assertTrue(bool(torch.isfinite(out).all()), f"v{version}: a column outside [lo, hi) was read")
                self.assertTrue(torch.equal(out, clean), f"v{version}: poison changed the scores")
            p2 = poisoned.clone()
            self.ms.fused(p2, k_new, wk, q_cat, wq_t, lo, hi, write_col)
            torch.mps.synchronize()
            self.assertTrue(bool(torch.isfinite(p2[:, :, write_col].float()).all()), "write_col must hold the new z, not the poison")
            # a tail whose vector would cross hi: L chosen so that M % 4 == 3 and the last quad would read columns >= hi
            zT4b, k_pre_new, q_pre, idx, lo, hi, write_col = self._case(1024, 1000 - 1)
            zTb, k_new, wk, q_cat, wq_t = metal_score.model_views(zT4b, k_pre_new, q_pre, idx)
            self.assertEqual((hi - lo) % 4, 3)
            pb = zTb.clone()
            pb[:, :, hi:] = float("nan")
            outb = self.ms.fused(pb, k_new, wk, q_cat, wq_t, lo, hi, write_col)
            torch.mps.synchronize()
            self.assertTrue(bool(torch.isfinite(outb).all()), "the tail quad read at or beyond hi")

    def test_validation(self):
        with torch.inference_mode():
            zT4, k_pre_new, q_pre, idx, lo, hi, write_col = self._case(600, 584)
            zT, k_new, wk, q_cat, wq_t = metal_score.model_views(zT4, k_pre_new, q_pre, idx)
            with self.assertRaises(ValueError):
                self.ms.fused(zT, k_new, wk, q_cat, wq_t, lo, hi, (lo + hi) // 2)     # write inside the scanned range
            with self.assertRaises(ValueError):
                self.ms.fused(zT.float(), k_new, wk, q_cat, wq_t, lo, hi, write_col)  # not bf16
            with self.assertRaises(ValueError):
                self.ms.fused(zT.transpose(1, 2), k_new, wk, q_cat, wq_t, lo, hi, write_col)
            with self.assertRaises(ValueError):
                metal_score.model_views(torch.cat([zT4, zT4]), k_pre_new, q_pre, idx)


if __name__ == "__main__":
    unittest.main()
