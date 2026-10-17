"""Contracts for the P1 gather-free attention kernel (docs/tasks/04_FUSED_D_KERNEL.md). MPS + compile_shader only.

Preset before the first measurement: fp32 output vs G1a's dense_mask_reference (float32 CPU on the same bf16 inputs)
max |Δ| <= 5e-5 and relative norm <= 1e-5 for every partition count; bf16 output within one bf16 ulp of the RNE
rounding of the fp32 output; poisoned unselected rows never change the output; shuffled and sorted row lists agree;
rows are read by original position (compared with the index_select path); boundary sizes.
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
    from attention_lab import metal_attend
    from attention_lab.decode import prepare_request
    from attention_lab.reference import dense_mask_reference
    from attention_lab.sparse import gather_attention, gather_rows

READY = torch is not None and torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
TOL = {"out_atol": 5e-5, "out_rel_norm": 1e-5, "bf16_ulp": 2.0 ** -8}
PARTS = (1, 2, 4, 8, 16, 32)


def make(G=8, h=2, d=128, C=2304, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dev = torch.device("mps")
    K4 = torch.randn(1, G, C, d, generator=gen).to(dev, torch.bfloat16)
    V4 = torch.randn(1, G, C, d, generator=gen).to(dev, torch.bfloat16)
    q5 = torch.randn(1, G, h, 1, d, generator=gen).to(dev, torch.bfloat16)
    return K4, V4, q5


def row_list(G, C, L, rho, seed=0):
    """F ∪ k random distinct candidates per group (the shape of a D selection), int64 [1, G, S]."""
    req = prepare_request(1, G, C, valid_length=L, query_position=L - 1, rho=rho, device="cpu")
    gen = torch.Generator().manual_seed(seed)
    lo, hi, k = req.layout.candidate_lo, req.layout.candidate_hi, req.k
    fixed = torch.tensor(req.layout.fixed_positions(), dtype=torch.long)
    rows = []
    for g in range(G):
        cand = lo + torch.randperm(hi - lo, generator=gen)[:k]
        rows.append(torch.cat([fixed, cand]))
    return torch.stack(rows).unsqueeze(0), req


@unittest.skipUnless(READY, "needs MPS and torch.mps.compile_shader (P1 Metal kernel)")
class MetalAttendContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ma = metal_attend.MetalAttend()

    def _ref(self, q5, K4, V4, rows3):
        return dense_mask_reference(q5.float().cpu(), K4.float().cpu(), V4.float().cpu(), rows3, valid_length=K4.shape[2])[0, :, :, 0, :]

    def test_matches_dense_mask_reference_for_every_partition_count(self):
        for C, L, rho in ((2304, 2176 + 65, 1 / 8), (2304, 400, 1 / 32), (2304, 300, 1 / 8), (2304, 73, 1.0)):   # S = 348, 78, 97, 73
            with self.subTest(C=C, L=L, rho=rho), torch.inference_mode():
                K4, V4, q5 = make(C=C)
                rows3, req = row_list(8, C, L, rho)
                self.assertEqual(rows3.shape[-1], req.selected_count)
                K, V, q, rows = metal_attend.model_views(K4, V4, q5, rows3.to("mps"))
                ref = self._ref(q5, K4, V4, rows3)
                pt = gather_attention(q5, K4, V4, rows3.to("mps"))[0, :, :, 0, :].float().cpu()           # G1a path, bf16 output
                pt_err = float((pt - ref).abs().max())
                for P in PARTS:
                    out = self.ma(K, V, q, rows, partitions=P)
                    torch.mps.synchronize()
                    diff = float((out.cpu() - ref).abs().max())
                    rel = float((out.cpu() - ref).norm() / ref.norm())
                    self.assertLessEqual(diff, TOL["out_atol"], f"P={P}: max abs diff {diff:.3e} vs dense_mask_reference")
                    self.assertLessEqual(rel, TOL["out_rel_norm"], f"P={P}: rel norm {rel:.3e}")
                    self.assertLessEqual(diff, pt_err, f"P={P}: fused less accurate than the bf16 PyTorch path ({pt_err:.3e})")
                    ob = self.ma(K, V, q, rows, partitions=P, out_dtype=torch.bfloat16)
                    torch.mps.synchronize()
                    rne = out.to(torch.bfloat16).float().cpu()
                    tol = TOL["bf16_ulp"] * rne.abs() + 1e-7
                    self.assertTrue(bool(((ob.float().cpu() - rne).abs() <= tol).all()), f"P={P}: bf16 output off by more than one ulp")
                    self.assertGreaterEqual(float((ob.float().cpu() == rne).float().mean()), 0.99)

    def test_boundary_sizes(self):
        with torch.inference_mode():
            K4, V4, q5 = make(C=600)
            for S in (1, 5, 31, 32, 33, 255, 257):
                rows3 = torch.stack([torch.randperm(600, generator=torch.Generator().manual_seed(S + g))[:S] for g in range(8)]).unsqueeze(0)
                K, V, q, rows = metal_attend.model_views(K4, V4, q5, rows3.to("mps"))
                ref = self._ref(q5, K4, V4, rows3)
                for P in PARTS:
                    out = self.ma(K, V, q, rows, partitions=P)
                    torch.mps.synchronize()
                    self.assertTrue(bool(torch.isfinite(out).all()), f"S={S} P={P}: non-finite output (empty partition handling)")
                    self.assertLessEqual(float((out.cpu() - ref).abs().max()), TOL["out_atol"], f"S={S} P={P}")

    def test_poisoned_unselected_rows_never_change_the_output(self):
        with torch.inference_mode():
            K4, V4, q5 = make(C=2304)
            rows3, _ = row_list(8, 2304, 2176 + 65, 1 / 8)
            K, V, q, rows = metal_attend.model_views(K4, V4, q5, rows3.to("mps"))
            Kp, Vp = K4.clone(), V4.clone()
            for g in range(8):
                keep = torch.zeros(2304, dtype=torch.bool)
                keep[rows3[0, g]] = True
                Kp[0, g, ~keep.to("mps")] = float("nan")
                Vp[0, g, ~keep.to("mps")] = float("nan")
            for P in PARTS:
                clean = self.ma(K, V, q, rows, partitions=P)
                poisoned = self.ma(Kp[0], Vp[0], q, rows, partitions=P)
                torch.mps.synchronize()
                self.assertTrue(bool(torch.isfinite(poisoned).all()), f"P={P}: an unselected row was read")
                self.assertTrue(torch.equal(poisoned, clean), f"P={P}: poison changed the output")

    def test_rows_are_original_positions_and_order_does_not_matter(self):
        with torch.inference_mode():
            K4, V4, q5 = make(C=1024)
            rows3, _ = row_list(8, 1024, 1000, 1 / 8)
            perm = torch.stack([torch.randperm(rows3.shape[-1], generator=torch.Generator().manual_seed(g)) for g in range(8)])
            shuffled = torch.stack([rows3[0, g][perm[g]] for g in range(8)]).unsqueeze(0)
            K, V, q, rows = metal_attend.model_views(K4, V4, q5, rows3.to("mps"))
            _, _, _, rows_sh = metal_attend.model_views(K4, V4, q5, shuffled.to("mps"))
            a = self.ma(K, V, q, rows, partitions=4)
            b = self.ma(K, V, q, rows_sh, partitions=4)
            torch.mps.synchronize()
            self.assertLessEqual(float((a.cpu() - b.cpu()).abs().max()), TOL["out_atol"], "row order changed the result beyond rounding")
            # positions 2 and 9 only: the kernel must read rows 2 and 9 of each group, not rows 0 and 1
            two = torch.tensor([2, 9]).view(1, 1, 2).expand(1, 8, 2).contiguous()
            _, _, _, r2 = metal_attend.model_views(K4, V4, q5, two.to("mps"))
            out2 = self.ma(K, V, q, r2, partitions=1)
            torch.mps.synchronize()
            want = self._ref(q5, K4, V4, two)
            self.assertLessEqual(float((out2.cpu() - want).abs().max()), TOL["out_atol"])
            wrong = self._ref(q5, K4, V4, torch.tensor([0, 1]).view(1, 1, 2).expand(1, 8, 2).contiguous())   # renumbered counterexample
            self.assertGreater(float((out2.cpu() - wrong).abs().max()), 1e-2)
            self.assertTrue(torch.equal(gather_rows(K4, two.to("mps"))[0, 0], K4[0, 0][[2, 9]]))
            # the two heads of a group share the rows but must not share the answer
            self.assertGreater(float((a[:, 0] - a[:, 1]).abs().max()), 1e-2)

    def test_interleaved_kv_layout_diagnostic_matches(self):
        with torch.inference_mode():
            K4, V4, q5 = make(C=2304)
            rows3, _ = row_list(8, 2304, 2176 + 65, 1 / 8)
            K, V, q, rows = metal_attend.model_views(K4, V4, q5, rows3.to("mps"))
            kv = torch.stack([K, V], dim=2).contiguous()                                    # [G, C, 2, d]
            ref = self._ref(q5, K4, V4, rows3)
            for P in (1, 8):
                a = self.ma(K, V, q, rows, partitions=P)
                b = self.ma(None, None, q, rows, partitions=P, kv_interleaved=kv)
                torch.mps.synchronize()
                self.assertLessEqual(float((b.cpu() - ref).abs().max()), TOL["out_atol"], f"P={P}: interleaved layout vs reference")
                self.assertLessEqual(float((a.cpu() - b.cpu()).abs().max()), 1e-6, f"P={P}: interleaved vs separate buffers")
            with self.assertRaises(ValueError):
                self.ma(None, None, q, rows, partitions=1, kv_interleaved=kv.view(8, 2304, 256))

    def test_validation(self):
        with torch.inference_mode():
            K4, V4, q5 = make(C=600)
            rows3, _ = row_list(8, 600, 584, 1 / 8)
            K, V, q, rows = metal_attend.model_views(K4, V4, q5, rows3.to("mps"))
            with self.assertRaises(ValueError):
                self.ma(K, V, q, rows3[0].to("mps"), partitions=4)                   # int64 rows
            with self.assertRaises(ValueError):
                self.ma(K.float(), V, q, rows, partitions=4)
            with self.assertRaises(ValueError):
                self.ma(K, V, q, rows, partitions=0)
            with self.assertRaises(ValueError):
                self.ma(K[:, :, :64].contiguous(), V[:, :, :64].contiguous(), q[:, :, :64].contiguous(), rows, partitions=1)   # head_dim != 128
            with self.assertRaises(ValueError):
                metal_attend.model_views(torch.cat([K4, K4]), V4, q5, rows3.to("mps"))


if __name__ == "__main__":
    unittest.main()
