"""Correctness contracts for the G1a read paths (docs/IMPLEMENTATION.md §6).

Order: CPU float64 small examples first, then MPS float32 and bfloat16 on this machine when MPS is available.
Tolerances are fixed here before the first run and are not loosened to make a path pass. bfloat16 paths are
compared against a float32 reference computed on the same bfloat16-rounded inputs. Without torch (CI bootstrap
image) the whole module is skipped; without MPS only the MPS cases are skipped.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
except ImportError:  # CI bootstrap image has no torch
    torch = None

if torch is not None:
    import torch.nn.functional as F
    from attention_lab import (GroupedIndexer, candidate_scores, dense_mask_reference, dense_reference, fixed_budget,
                               full_attention, gather_attention, gather_rows, legal_length, prepare_request,
                               selection_layout, sparse_decode_step, subset_attention)

TOL = {  # preset; see module docstring
    "float64": {"atol": 1e-10, "rel_norm": 1e-12},
    "float32": {"atol": 1e-4, "rel_norm": 1e-5},
    "bfloat16": {"atol": 5e-2, "rel_norm": 2e-2},
}
MPS = torch is not None and torch.backends.mps.is_available()
CASES = [("cpu", "float64", C, 16, 2, 2) for C in (32, 128)]          # device, dtype, C, d, G, h
if MPS:
    CASES += [("mps", dt, C, 128, 8, 4) for dt in ("float32", "bfloat16") for C in (32, 128, 512)]


def dt_of(name):
    return getattr(torch, name)


def make_inputs(device, dtype, *, B=1, G=2, h=2, T=1, C=64, d=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, G, h, T, d, generator=gen)
    k = torch.randn(B, G, C, d, generator=gen)
    v = torch.randn(B, G, C, d, generator=gen)
    q_pre = torch.randn(B, G, h, T, d, generator=gen)
    k_pre = torch.randn(B, G, C, d, generator=gen)
    cast = lambda x: x.to(device=device, dtype=dt_of(dtype)).contiguous()  # noqa: E731
    return cast(q), cast(k), cast(v), cast(q_pre), cast(k_pre)


def ref_dtype(dtype):
    return "float64" if dtype == "float64" else "float32"


def up(x, dtype):
    """Reference precision for a given working dtype: float64 stays, MPS dtypes are compared in float32."""
    return x.to(dt_of(ref_dtype(dtype)))


class Checks(unittest.TestCase):
    def assert_close(self, a, b, dtype, what):
        a = a.detach().float().cpu()
        b = b.detach().float().cpu()
        self.assertTrue(torch.isfinite(a).all(), f"{what}: non-finite output")
        diff = float((a - b).abs().max())
        rel = float((a - b).norm() / (b.norm() + 1e-30))
        self.assertLessEqual(diff, TOL[dtype]["atol"], f"{what}: max abs diff {diff:.3e} > {TOL[dtype]['atol']}")
        self.assertLessEqual(rel, TOL[dtype]["rel_norm"], f"{what}: rel norm diff {rel:.3e} > {TOL[dtype]['rel_norm']}")

    def assert_not_close(self, a, b, dtype, what):
        diff = float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())
        self.assertGreater(diff, 10 * TOL[dtype]["atol"], f"{what}: expected a detectable difference, got {diff:.3e}")


@unittest.skipUnless(torch is not None, "torch not installed")
class ContractTests(Checks):
    def test_full_selection_matches_dense_reference(self):
        for device, dtype, C, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype, C=C):
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=C, d=d)
                ref = dense_reference(up(q, dtype), up(k, dtype), up(v, dtype), valid_length=C)
                idx = GroupedIndexer(G, h, d, r=32, device=device, dtype=dt_of(dtype))
                req = prepare_request(1, G, C, valid_length=C, query_position=C - 1, rho=1.0, device=device)
                self.assertEqual(req.selected_count, C)
                zT = idx.keys(k_pre)
                out = sparse_decode_step(q, q_pre, k, v, zT, idx, req, k_pre_new=k_pre[:, :, C - 1:C])
                self.assert_close(out, ref, dtype, "sparse step with every position selected vs dense_reference")
                pos = req.select_positions(idx.query(q_pre), zT)
                self.assertEqual(sorted(pos[0, 0].tolist()), list(range(C)))
                for backend in ("sdpa_fold", "sdpa_gqa", "sdpa_repeat", "explicit"):
                    self.assert_close(full_attention(q, k, v, valid_length=C, backend=backend), ref, dtype, f"full_attention[{backend}]")

    def test_same_subset_gather_matches_dense_mask(self):
        for device, dtype, C, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype, C=C):
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=C, d=d)
                idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
                req = prepare_request(1, G, C, valid_length=C, query_position=C - 1, rho=1 / 8, device=device)
                sel = req.select_positions(idx.query(q_pre), idx.keys(k_pre))          # [1,G,S] original positions
                hand = torch.tensor([9, 2, C - 1, 17 % C, 5], device=device).view(1, 1, 5).expand(1, G, 5).contiguous()
                for name, s in (("indexer selection", sel), ("hand-made unsorted subset", hand)):
                    out_g = gather_attention(q, k, v, s)
                    out_m = dense_mask_reference(q, k, v, s)
                    ref = dense_mask_reference(up(q, dtype), up(k, dtype), up(v, dtype), s)
                    self.assert_close(out_g, out_m, dtype, f"{name}: gather vs dense_mask")
                    self.assert_close(out_g, ref, dtype, f"{name}: gather vs reference precision")
                    self.assert_close(gather_attention(q, k, v, s, backend="explicit"), ref, dtype, f"{name}: explicit subset")
                if device == "cpu":  # padded slots: the mask must drop them, duplicates must not double count
                    padded = torch.cat([hand, torch.zeros(1, G, 3, dtype=torch.long)], dim=-1)
                    valid = torch.cat([torch.ones(1, G, 5, dtype=torch.bool), torch.zeros(1, G, 3, dtype=torch.bool)], dim=-1)
                    self.assert_close(gather_attention(q, k, v, padded, valid), dense_mask_reference(q, k, v, hand), dtype, "padded gather")
                    self.assert_close(dense_mask_reference(q, k, v, padded, valid), dense_mask_reference(q, k, v, hand), dtype, "padded mask")

    def test_unselected_rows_are_never_read(self):
        for device, dtype, C, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype, C=C):
                if C <= 68:
                    continue  # every position is in F: nothing is unselected, see test_rules_and_edge_cases
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=C, d=d)
                idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
                req = prepare_request(1, G, C, valid_length=C, query_position=C - 1, rho=1 / 8, device=device)
                zT = idx.keys(k_pre)
                clean = sparse_decode_step(q, q_pre, k, v, zT, idx, req)
                pos = req.select_positions(idx.query(q_pre), zT)
                poison_k, poison_v = k.clone(), v.clone()
                for g in range(G):
                    keep = torch.zeros(C, dtype=torch.bool, device=device)
                    keep[pos[0, g]] = True
                    poison_k[0, g, ~keep] = float("nan")
                    poison_v[0, g, ~keep] = float("nan")
                self.assertLess(req.selected_count, C)
                out = sparse_decode_step(q, q_pre, poison_k, poison_v, zT, idx, req)
                self.assertTrue(torch.isfinite(out).all(), "NaN leaked from a row that must not be read")
                self.assert_close(out, clean, dtype, "poisoned unselected rows")
                self.assertFalse(torch.isfinite(dense_mask_reference(q, poison_k, poison_v, pos)).all(),
                                 "the dense-mask reference multiplies masked rows by zero and should expose the poison")

    def test_unwritten_slots_and_future_positions_do_not_change_result(self):
        for device, dtype, C, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype, C=C):
                L, cap = C, C + 32
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=cap, d=d)
                for x in (k, v, k_pre):
                    x[:, :, L:] = float("nan")                              # unwritten slots
                base = dense_reference(q, k[:, :, :L].contiguous(), v[:, :, :L].contiguous())
                self.assert_close(dense_reference(q, k, v, valid_length=L), base, dtype, "dense_reference capacity > valid")
                for backend in ("sdpa_fold", "sdpa_gqa"):
                    self.assert_close(full_attention(q, k, v, valid_length=L, backend=backend), base, dtype, f"full_attention[{backend}] capacity > valid")
                idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
                zT = torch.full((1, G, 32, cap), float("nan"), device=device, dtype=dt_of(dtype))
                zT[..., :L] = idx.keys(k_pre[:, :, :L].contiguous())
                req = prepare_request(1, G, cap, valid_length=L, query_position=L - 1, rho=1 / 8, device=device)
                req_tight = prepare_request(1, G, L, valid_length=L, query_position=L - 1, rho=1 / 8, device=device)
                out = sparse_decode_step(q, q_pre, k, v, zT, idx, req, k_pre_new=k_pre[:, :, L - 1:L])
                tight = sparse_decode_step(q, q_pre, k[:, :, :L].contiguous(), v[:, :, :L].contiguous(), zT[..., :L].contiguous(), idx, req_tight)
                self.assert_close(out, tight, dtype, "sparse step capacity > valid")
                # three queries at the tail: a future-but-written position may change only the queries that see it
                q3 = torch.randn(1, G, h, 3, d, generator=torch.Generator().manual_seed(7)).to(device=device, dtype=dt_of(dtype))
                kk, vv = k[:, :, :L].contiguous(), v[:, :, :L].contiguous()
                before = dense_reference(q3, kk, vv)                          # positions L-3, L-2, L-1
                for backend in ("sdpa_fold", "sdpa_gqa"):
                    self.assert_close(full_attention(q3, kk, vv, backend=backend), before, dtype, f"causal tail [{backend}]")
                kk2, vv2 = kk.clone(), vv.clone()
                kk2[:, :, L - 2:] += 1000.0
                vv2[:, :, L - 2:] += 1000.0
                after = dense_reference(q3, kk2, vv2)
                self.assert_close(after[..., 0, :], before[..., 0, :], dtype, "query at L-3 must not see L-2, L-1")
                self.assert_not_close(after[..., 2, :], before[..., 2, :], dtype, "query at L-1 must see the change")

    def test_single_query_reads_whole_history_without_square_mask(self):
        for device, dtype, _, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype):
                L = 37
                q, k, v, _, _ = make_inputs(device, dtype, G=G, h=h, C=L, d=d)
                scores = torch.matmul(up(q, dtype), up(k, dtype).unsqueeze(2).transpose(-1, -2)) / d ** 0.5
                brute = torch.matmul(torch.softmax(scores, -1), up(v, dtype).unsqueeze(2))   # every one of the 37 keys
                for backend in ("sdpa_fold", "sdpa_gqa", "explicit"):
                    self.assert_close(full_attention(q, k, v, valid_length=L, backend=backend), brute, dtype, f"T=1 over {L} keys [{backend}]")
                wrong = dense_reference(q, k, v, valid_length=L, query_positions=[0])          # square top-left causal mask
                self.assert_not_close(wrong, brute, dtype, "square causal mask on a 1-row query must be detected")
                q3 = torch.randn(1, G, h, 3, d, generator=torch.Generator().manual_seed(3)).to(device=device, dtype=dt_of(dtype))
                sc3 = torch.matmul(up(q3, dtype), up(k, dtype).unsqueeze(2).transpose(-1, -2)) / d ** 0.5
                legal = torch.arange(L)[None, :] <= torch.tensor([L - 3, L - 2, L - 1])[:, None]
                sc3 = sc3.masked_fill(~legal.to(device), float("-inf"))
                brute3 = torch.matmul(torch.softmax(sc3, -1), up(v, dtype).unsqueeze(2))
                for backend in ("sdpa_fold", "sdpa_gqa", "explicit"):
                    self.assert_close(full_attention(q3, k, v, valid_length=L, backend=backend), brute3, dtype, f"T=3 tail [{backend}]")

    def test_positions_are_original_not_renumbered(self):
        for device, dtype, _, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype):
                L = 128
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=L, d=d)
                idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
                req = prepare_request(1, G, L, valid_length=L, query_position=L - 1, rho=1 / 8, device=device)
                self.assertEqual((req.layout.fixed_count, req.k, req.selected_count), (68, 16, 84))
                pos = req.select_positions(idx.query(q_pre), idx.keys(k_pre))
                self.assertEqual(pos[0, 0, :68].tolist(), list(range(4)) + list(range(64, 128)))
                top = pos[..., 68:]
                self.assertTrue(bool((top >= 4).all() and (top < 64).all()), "chosen candidates must stay inside the candidate domain")
                for g in range(G):
                    self.assertEqual(len(set(pos[0, g].tolist())), 84, "duplicate position in the read list")
                    self.assertTrue(torch.equal(gather_rows(k, pos)[0, g], k[0, g][pos[0, g]]), "gathered row is not the original row")
                two = torch.tensor([2, 9], device=device).view(1, 1, 2).expand(1, G, 2).contiguous()
                self.assertTrue(torch.equal(gather_rows(k, two), k[:, :, [2, 9]]))
                # counterexample: dropping the group offsets reads group 0's rows for every group
                flat_wrong = k.view(-1, d).index_select(0, two.reshape(-1)).view(1, G, 2, d)
                self.assert_not_close(subset_attention(q, flat_wrong, flat_wrong, backend="explicit"),
                                      gather_attention(q, k, k, two, backend="explicit"), dtype, "missing group offset")
                rows = req.select_rows(idx.query(q_pre), idx.keys(k_pre))
                self.assertTrue(torch.equal(rows.view(1, G, -1) - req.offsets, pos), "row ids and positions disagree")

    def test_indexer_topk_matches_bruteforce_and_scans_candidates_only(self):
        for device, dtype, _, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype):
                L = 256
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=L, d=d)
                idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
                req = prepare_request(1, G, L, valid_length=L, query_position=L - 1, rho=1 / 8, device=device)
                lo, hi = req.layout.candidate_lo, req.layout.candidate_hi
                self.assertEqual((lo, hi, req.k), (4, 192, 32))
                u, zT = idx.query(q_pre), idx.keys(k_pre)
                s = candidate_scores(u, zT, lo, hi)
                self.assertEqual(tuple(s.shape), (1, G, 1, hi - lo), "only the candidate columns may be scanned")
                brute = idx.reference_scores(q_pre, k_pre, lo, hi)                      # float64, unscaled weights / sqrt(r)
                pos = req.select_positions(u, zT)
                if dtype != "bfloat16":
                    self.assert_close(s, brute, dtype, "candidate scores vs brute force")
                    for g in range(G):
                        want = set((torch.topk(brute[0, g, 0], req.k).indices + lo).tolist())
                        self.assertEqual(set(pos[0, g, 68:].tolist()), want, "top-k set differs from brute force")
                else:  # bf16 ties are common; the chosen scores must still be no smaller than the k-th largest
                    for g in range(G):
                        kth = torch.topk(s[0, g, 0].float(), req.k).values[-1]
                        chosen = s[0, g, 0].float()[pos[0, g, 68:] - lo]
                        self.assertTrue(bool((chosen >= kth).all()))

    def test_topk_once_and_no_torch_gather(self):
        for device, dtype, _, d, G, h in CASES[:1] + CASES[2:3]:
            with self.subTest(device=device, dtype=dtype):
                L = 256
                q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=L, d=d)
                idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
                req = prepare_request(1, G, L, valid_length=L, query_position=L - 1, rho=1 / 32, device=device)
                zT = idx.keys(k_pre)
                with patch("torch.topk", wraps=torch.topk) as topk, \
                        patch("torch.gather", side_effect=AssertionError("torch.gather must not be used")), \
                        patch.object(torch.Tensor, "gather", side_effect=AssertionError("Tensor.gather must not be used")):
                    sparse_decode_step(q, q_pre, k, v, zT, idx, req, k_pre_new=k_pre[:, :, L - 1:L])
                self.assertEqual(topk.call_count, 1, "top-k must run exactly once, over the whole candidate domain")

    def test_index_table_layout_and_chunked_build(self):
        for device, dtype, C, d, G, h in CASES:
            with self.subTest(device=device, dtype=dtype, C=C):
                _, _, _, _, k_pre = make_inputs(device, dtype, G=G, h=h, C=C, d=d)
                idx = GroupedIndexer(G, h, d, r=32, device=device, dtype=dt_of(dtype))
                zT = idx.keys(k_pre)
                self.assertEqual(tuple(zT.shape), (1, G, 32, C))
                self.assertTrue(zT.is_contiguous() and zT.stride(-1) == 1, "index keys must be stored [B,G,r,C] contiguous")
                bounds = [0, 5, C // 2, C]
                chunks = torch.cat([idx.keys(k_pre[:, :, a:b].contiguous()) for a, b in zip(bounds, bounds[1:])], dim=-1)
                self.assert_close(chunks, zT, dtype, "chunked index build vs one-shot build")
                table = torch.empty_like(zT)
                for a, b in zip(bounds, bounds[1:]):
                    table[..., a:b].copy_(idx.keys(k_pre[:, :, a:b].contiguous()))
                self.assert_close(table, zT, dtype, "chunked write into a preallocated table")

    def test_rules_and_edge_cases(self):
        self.assertEqual([fixed_budget(8192, 1 / 32), fixed_budget(8192, 1 / 8), fixed_budget(32768, 1 / 32)], [256, 1024, 1024])
        self.assertEqual([fixed_budget(100, 1 / 32), fixed_budget(50, 0.5), fixed_budget(4, 1.0)], [3, 0, 0])
        lay = selection_layout(100)
        self.assertEqual((lay.fixed_count, lay.candidate_count, lay.candidate_lo, lay.candidate_hi), (68, 32, 4, 36))
        self.assertEqual(lay.fixed_positions(), list(range(4)) + list(range(36, 100)))
        self.assertEqual((selection_layout(50).fixed_count, selection_layout(50).candidate_count), (50, 0))
        self.assertEqual((selection_layout(3).fixed_count, selection_layout(3).fixed_prefix), (3, 3))
        self.assertEqual((legal_length(5, 10), legal_length(20, 10)), (6, 10))
        for bad in ((-1, 10), (3, 0)):
            with self.assertRaises(ValueError):
                legal_length(*bad)
        with self.assertRaises(ValueError):
            selection_layout(0)
        device, dtype, d, G, h = "cpu", "float64", 16, 2, 2
        q, k, v, q_pre, k_pre = make_inputs(device, dtype, G=G, h=h, C=128, d=d)
        idx = GroupedIndexer(G, h, d, device=device, dtype=dt_of(dtype))
        with self.assertRaises(ValueError):
            prepare_request(1, G, 128, valid_length=0, query_position=0, rho=1 / 8, device=device)
        shrunk = prepare_request(1, G, 128, valid_length=100, query_position=99, rho=1 / 32, device=device, n0=8192)
        self.assertEqual((shrunk.k0, shrunk.k, shrunk.selected_count), (256, 32, 100))   # k0 fixed by N0, clamped now
        only_f = prepare_request(1, G, 128, valid_length=128, query_position=127, rho=0.0, device=device)
        self.assertEqual((only_f.k, only_f.selected_count), (0, 68))
        out = sparse_decode_step(q, q_pre, k, v, idx.keys(k_pre), idx, only_f)
        self.assert_close(out, dense_mask_reference(q, k, v, only_f.fixed_positions), dtype, "k=0 reads exactly F")
        short = prepare_request(1, G, 128, valid_length=60, query_position=59, rho=1 / 8, device=device)
        self.assertEqual((short.k, short.selected_count), (0, 60))
        self.assert_close(sparse_decode_step(q, q_pre, k, v, idx.keys(k_pre), idx, short),
                          dense_reference(q, k, v, valid_length=60), dtype, "legal length <= 68 degenerates to full")
        with self.assertRaises(ValueError):
            dense_mask_reference(q, k, v, torch.tensor([[[3, 70]] * G]), valid_length=64)
        with self.assertRaises(ValueError):
            dense_mask_reference(q, k, v, torch.tensor([[[0]] * G]), torch.zeros(1, G, 1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            gather_rows(k.transpose(2, 3).transpose(2, 3)[:, :, ::2], torch.tensor([[[1, 2]] * G]))
        with self.assertRaises(ValueError):
            full_attention(q, k, v, valid_length=200)
        with self.assertRaises(ValueError):
            dense_reference(torch.randn(1, G, h, 2, d, dtype=torch.float64), k, v, valid_length=1)


if __name__ == "__main__":
    unittest.main()
