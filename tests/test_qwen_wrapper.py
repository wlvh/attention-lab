"""Contracts for the Qwen3 wrapper, static cache and index connection (G1b; docs/IMPLEMENTATION.md §6).

Order: a tiny randomly initialised Qwen3 (the real module code, 2 layers) on CPU float64 first, then MPS float32 and
bfloat16 on this machine when MPS is available. The pinned real checkpoint is used only when its snapshot is already
in the local Hub cache (the tests never download). Tolerances are preset: the tiny model uses the G1a contract
tolerances, the real checkpoint uses configs/mac_gate1b.json; neither is loosened to make a path pass.

Without torch (CI bootstrap image) the module is skipped; without transformers the wrapper tests are skipped with a
distinct reason so CI can tell the two apart.
"""
from __future__ import annotations

import json
import os
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
try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = torch is not None
except ImportError:
    HAVE_TRANSFORMERS = False

if HAVE_TRANSFORMERS:
    from transformers import DynamicCache, StaticCache
    from attention_lab.indexer import project_keys
    from attention_lab.qwen_wrap import (LabRunner, NativeRunner, geometry_of, hf_cache_kv, is_wrapped, make_state, tiny_qwen3,
                                         unwrap_model, wrap_model)
    from attention_lab.selection import fixed_budget

CONFIG = json.loads((ROOT / "configs/mac_gate1b.json").read_text(encoding="utf-8"))
REAL_TOL = CONFIG["numerics"]["tolerances"]
TINY_TOL = {  # G1a contract tolerances (tests/test_attention_contracts.py), reused for the tiny model
    "float64": {"atol": 1e-10, "rel_norm": 1e-12},
    "float32": {"atol": 1e-4, "rel_norm": 1e-5},
    "bfloat16": {"atol": 5e-2, "rel_norm": 2e-2},
}
MPS = torch is not None and torch.backends.mps.is_available()
CASES = [("cpu", "float64")] + ([("mps", "float32"), ("mps", "bfloat16")] if MPS else [])
SKIP_NO_TF = "transformers not installed (or torch not installed)"


def dt_of(name):
    return getattr(torch, name)


def snapshot_present(model_id: str, revision: str) -> bool:
    """True when the pinned checkpoint is already in the local Hub cache. Never triggers a download."""
    if not HAVE_TRANSFORMERS:
        return False
    from huggingface_hub import try_to_load_from_cache
    return all(isinstance(try_to_load_from_cache(model_id, f, revision=revision), str)
               for f in ("config.json", "model.safetensors", "tokenizer.json"))


class Checks(unittest.TestCase):
    @staticmethod
    def _pair(a, b):
        return a.detach().cpu().double(), b.detach().cpu().double()

    def assert_close(self, a, b, tol, what):
        a, b = self._pair(a, b)
        self.assertTrue(torch.isfinite(a).all(), f"{what}: non-finite output")
        diff = float((a - b).abs().max())
        rel = float((a - b).norm() / (b.norm() + 1e-30))
        self.assertLessEqual(diff, tol["atol"], f"{what}: max abs diff {diff:.3e} > {tol['atol']}")
        self.assertLessEqual(rel, tol["rel_norm"], f"{what}: rel norm diff {rel:.3e} > {tol['rel_norm']}")

    def assert_not_close(self, a, b, tol, what):
        a, b = self._pair(a, b)
        diff = float((a - b).abs().max())
        self.assertGreater(diff, 10 * tol["atol"], f"{what}: expected a detectable difference, got {diff:.3e}")

    def assert_relatively_different(self, a, b, what, min_rel=0.1):
        """Counterexample check for small-magnitude tensors (index keys): relative norm difference above 10%."""
        a, b = self._pair(a, b)
        rel = float((a - b).norm() / (b.norm() + 1e-30))
        self.assertGreater(rel, min_rel, f"{what}: expected a relative difference above {min_rel}, got {rel:.3e}")


def tiny(device, dtype):
    return tiny_qwen3(dtype=dt_of(dtype)).to(device)


def token_ids(n, seed=1, vocab=256):
    return torch.randint(0, vocab, (1, n), generator=torch.Generator().manual_seed(seed))


def poison_unwritten(cache):
    """NaN into every slot at or beyond valid_length (K, V and z). A correct path never reads them."""
    L = cache.valid_length
    for l in range(cache.layers):
        cache.k[l][:, :, L:] = float("nan")
        cache.v[l][:, :, L:] = float("nan")
        if cache.z is not None:
            cache.z[l][..., L:] = float("nan")


@unittest.skipUnless(HAVE_TRANSFORMERS, SKIP_NO_TF)
class WrapperContracts(Checks):
    def test_geometry_comes_from_config_and_matches_parameters(self):
        m = tiny("cpu", "float64")
        geo = geometry_of(m)
        self.assertEqual((geo.layers, geo.hidden, geo.heads, geo.groups, geo.head_dim, geo.heads_per_group), (2, 64, 4, 2, 16, 2))
        m.config.num_key_value_heads = 4           # config no longer matches k_proj: must be detected, not silently used
        with self.assertRaises(ValueError):
            geometry_of(m)

    def test_wrapped_full_path_matches_unmodified_model_and_hf_cache(self):
        for device, dtype in CASES:
            tol = TINY_TOL[dtype]
            for L in (32, 128, 300):
                with self.subTest(device=device, dtype=dtype, L=L), torch.inference_mode():
                    m = tiny(device, dtype)
                    ids = token_ids(L).to(device)
                    ref = m(ids).logits                                        # unmodified, no cache
                    dc = DynamicCache(config=m.config)
                    m(ids, past_key_values=dc, use_cache=True)                 # unmodified with its own cache -> K/V reference
                    holder = wrap_model(m)
                    self.assertTrue(is_wrapped(m))
                    st = make_state(m, L + 40, "full")
                    run = LabRunner(m, holder, st)
                    poison_unwritten(st.cache)
                    one_shot = run.prefill(ids, L, keep_logits=True)           # wrapped, one forward over every position
                    self.assert_close(one_shot, ref, tol, "wrapped one-shot vs unmodified logits")
                    for l in range(st.cache.layers):
                        k, v = hf_cache_kv(dc, l)
                        self.assert_close(st.cache.k[l][:, :, :L], k, tol, f"layer {l} K: static cache vs transformers cache")
                        self.assert_close(st.cache.v[l][:, :, :L], v, tol, f"layer {l} V: static cache vs transformers cache")
                    st2 = make_state(m, L + 40, "full")                        # wrapped + static cache fed chunk by chunk, then token by token
                    run2 = LabRunner(m, holder, st2)
                    poison_unwritten(st2.cache)
                    head = run2.prefill(ids[:, :L - 8], 37, keep_logits=True)
                    tail = torch.cat([run2.step(ids[:, i:i + 1]) for i in range(L - 8, L)], dim=1)
                    self.assert_close(torch.cat([head, tail], 1), ref, tol, "wrapped incremental vs unmodified logits")
                    self.assertEqual(st2.cache.valid_length, L)
                    self.assertEqual(st2.cache.writes, [-(-(L - 8) // 37) + 8] * st2.cache.layers, "one append per layer per forward")
                    unwrap_model(m)
                    self.assertFalse(is_wrapped(m))
                    self.assert_close(m(ids).logits, ref, {"atol": 0.0, "rel_norm": 0.0}, "unwrapped model is the original object graph")

    def test_full_selection_matches_full_path_and_poison_proves_the_subset_is_read(self):
        for device, dtype in CASES:
            tol = TINY_TOL[dtype]
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                L = 160
                ids = token_ids(L, seed=2).to(device)
                ref = m(ids).logits
                holder = wrap_model(m)
                stA = make_state(m, L + 16, "full")
                runA = LabRunner(m, holder, stA)
                runA.prefill(ids[:, :L - 6], 50)
                stD = make_state(m, L + 16, "sparse")
                runD = LabRunner(m, holder, stD)
                runD.prefill(ids[:, :L - 6], 50)
                stD.begin_request(None, k0=stD.cache.capacity)                # every candidate on every step
                outA, outD = [], []
                for i in range(L - 6, L):
                    outA.append(runA.step(ids[:, i:i + 1]))
                    outD.append(runD.step(ids[:, i:i + 1]))
                    self.assertEqual(stD.plan.selected_count, stD.plan.legal_length, "full selection must cover the legal range")
                outA, outD = torch.cat(outA, 1), torch.cat(outD, 1)
                self.assert_close(outA, ref[:, L - 6:], tol, "wrapped full decode vs unmodified")
                self.assert_close(outD, outA, tol, "sparse path with every position selected vs wrapped full path")
                # partial read: record which candidates each layer picked, poison every other row -> D unchanged, A broken
                runD.rollback(L - 6)
                runA.rollback(L - 6)
                stD.begin_request(1 / 8)
                self.assertEqual(stD.k0, fixed_budget(L - 5, 1 / 8))
                picked = []
                real_topk = torch.topk

                def spy(*a, **k):
                    r = real_topk(*a, **k)
                    picked.append(r.indices.detach().clone())
                    return r

                with patch("torch.topk", side_effect=spy):
                    clean = runD.step(ids[:, L - 6:L - 5])
                plan = stD.plan
                self.assertEqual(len(picked), stD.cache.layers, "one top-k per layer")
                self.assertLess(plan.selected_count, plan.legal_length)
                G = stD.cache.groups
                for l, top in enumerate(picked):
                    top = top.view(G, plan.k) + plan.lo
                    for g in range(G):
                        keep = torch.zeros(stD.cache.capacity, dtype=torch.bool, device=device)
                        keep[torch.tensor(list(range(plan.lo)) + list(range(plan.hi, plan.legal_length)), device=device)] = True
                        keep[top[g]] = True
                        stD.cache.k[l][0, g, ~keep] = float("nan")
                        stD.cache.v[l][0, g, ~keep] = float("nan")
                        stA.cache.k[l][0, g, ~keep] = float("nan")                  # same rows poisoned in the full path's cache
                        stA.cache.v[l][0, g, ~keep] = float("nan")
                runD.rollback(L - 6)
                poisoned = runD.step(ids[:, L - 6:L - 5])
                self.assertTrue(torch.isfinite(poisoned).all(), "D read a row it did not select")
                self.assert_close(poisoned, clean, {"atol": 0.0, "rel_norm": 0.0}, "D with unselected rows poisoned")
                self.assertFalse(torch.isfinite(runA.step(ids[:, L - 6:L - 5])).all(), "the full path must see the poison")
                unwrap_model(m)

    def test_unwritten_slots_and_future_positions_never_change_a_step(self):
        for device, dtype in CASES:
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                L = 120
                ids = token_ids(L, seed=3).to(device)
                holder = wrap_model(m)
                for mode in ("full", "sparse"):
                    st = make_state(m, L + 64, mode)
                    run = LabRunner(m, holder, st)
                    run.prefill(ids[:, :L - 5], 40)
                    if mode == "sparse":
                        st.begin_request(1 / 8)
                    clean = torch.cat([run.step(ids[:, i:i + 1]) for i in range(L - 5, L)], 1)
                    run.rollback(L - 5)
                    for l in range(st.cache.layers):                           # garbage AND a huge finite value in the future slots
                        st.cache.k[l][:, :, L - 5:L - 2] = float("nan")
                        st.cache.k[l][:, :, L - 2:] = 1e6
                        st.cache.v[l][:, :, L - 5:] = float("nan")
                        if st.cache.z is not None:
                            st.cache.z[l][..., L - 5:] = float("nan")
                    again = torch.cat([run.step(ids[:, i:i + 1]) for i in range(L - 5, L)], 1)
                    self.assertTrue(torch.isfinite(again).all(), f"{mode}: an unwritten or future slot leaked")
                    self.assert_close(again, clean, {"atol": 0.0, "rel_norm": 0.0}, f"{mode}: poisoned future slots")
                unwrap_model(m)

    def test_append_rollback_and_old_suffix_is_unreadable(self):
        for device, dtype in CASES:
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                ids = token_ids(140, seed=4).to(device)
                alt = token_ids(140, seed=5).to(device)
                holder = wrap_model(m)
                for mode in ("full", "sparse"):
                    st = make_state(m, 160, mode)
                    run = LabRunner(m, holder, st)
                    run.prefill(ids[:, :100], 37)
                    if mode == "sparse":
                        st.begin_request(1 / 8)
                    first = torch.cat([run.step(ids[:, i:i + 1]) for i in range(100, 105)], 1)
                    self.assertEqual(st.cache.valid_length, 105)
                    run.rollback(100)
                    self.assertEqual(st.cache.valid_length, 100)
                    poison_unwritten(st.cache)                                 # the rolled-back suffix must be invisible
                    second = torch.cat([run.step(ids[:, i:i + 1]) for i in range(100, 105)], 1)
                    self.assert_close(second, first, {"atol": 0.0, "rel_norm": 0.0}, f"{mode}: same tokens after rollback")
                    run.rollback(100)
                    branch = torch.cat([run.step(alt[:, i:i + 1]) for i in range(100, 105)], 1)
                    fresh = make_state(m, 160, mode)
                    runf = LabRunner(m, holder, fresh)
                    runf.prefill(ids[:, :100], 100)
                    if mode == "sparse":
                        fresh.begin_request(1 / 8)
                    fresh_out = torch.cat([runf.step(alt[:, i:i + 1]) for i in range(100, 105)], 1)
                    self.assert_close(branch, fresh_out, TINY_TOL[dtype], f"{mode}: overwritten suffix vs fresh cache")
                    with self.assertRaises(ValueError):
                        run.rollback(200)
                unwrap_model(m)

    def test_chunked_prefill_matches_one_shot_for_kv_and_index_table(self):
        for device, dtype in CASES:
            tol = TINY_TOL[dtype]
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                L = 100
                ids = token_ids(L, seed=6).to(device)
                holder = wrap_model(m)
                one = make_state(m, L + 8, "sparse")
                LabRunner(m, holder, one).prefill(ids, L)
                chunked = make_state(m, L + 8, "sparse")
                runc = LabRunner(m, holder, chunked)
                for a, b in zip([0, 5, 50], [5, 50, 100]):
                    runc.forward(ids[:, a:b], phase="prefill", logits_to_keep=1)
                self.assertEqual(chunked.cache.valid_length, L)
                for l in range(one.cache.layers):
                    self.assert_close(chunked.cache.k[l][:, :, :L], one.cache.k[l][:, :, :L], tol, f"layer {l} K chunked vs one-shot")
                    self.assert_close(chunked.cache.v[l][:, :, :L], one.cache.v[l][:, :, :L], tol, f"layer {l} V chunked vs one-shot")
                    self.assert_close(chunked.cache.z[l][..., :L], one.cache.z[l][..., :L], tol, f"layer {l} z chunked vs one-shot")
                unwrap_model(m)

    def test_index_features_are_pre_rope_and_each_z_column_is_written_once(self):
        for device, dtype in CASES:
            tol = TINY_TOL[dtype]
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                L = 90
                ids = token_ids(L + 3, seed=7).to(device)
                holder = wrap_model(m)
                st = make_state(m, L + 16, "sparse")
                run = LabRunner(m, holder, st)
                captured = {}
                hooks = [layer.self_attn.inner.k_norm.register_forward_hook(lambda mod, inp, out, l=l: captured.__setitem__(l, out.detach().clone()))
                         for l, layer in enumerate(m.model.layers)]
                run.prefill(ids[:, :L], L)
                for l in range(st.cache.layers):
                    k_pre = captured[l].transpose(1, 2)                                   # [B,G,T,d] after k_norm, before RoPE
                    want = project_keys(k_pre, st.indexers[l].wk)
                    self.assert_close(st.cache.z[l][..., :L], want, tol, f"layer {l} z = Wk·k_pre")
                    from_rotated = project_keys(st.cache.k[l][:, :, :L], st.indexers[l].wk)   # counterexample: rotated K is not the feature
                    self.assert_relatively_different(st.cache.z[l][..., :L], from_rotated, f"layer {l} z must not come from the RoPE'd cache")
                st.begin_request(1 / 8)
                snapshot = [st.cache.z[l][..., :L].clone() for l in range(st.cache.layers)]
                for i in range(L, L + 3):
                    run.step(ids[:, i:i + 1])
                    for l in range(st.cache.layers):
                        k_pre_new = captured[l].transpose(1, 2)
                        self.assert_close(st.cache.z[l][..., i:i + 1], project_keys(k_pre_new, st.indexers[l].wk), tol, f"step {i} layer {l} new z column")
                        self.assertTrue(torch.equal(st.cache.z[l][..., :L], snapshot[l]), "older z columns were rewritten")
                        self.assertEqual(st.cache.last_write[l], (i, 1))
                self.assertEqual(st.cache.writes, [1 + 3] * st.cache.layers)
                for h in hooks:
                    h.remove()
                unwrap_model(m)

    def test_topk_once_per_layer_and_no_torch_gather_in_a_model_step(self):
        device, dtype = CASES[0]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(130, seed=8)
            holder = wrap_model(m)
            st = make_state(m, 160, "sparse")
            run = LabRunner(m, holder, st)
            run.prefill(ids[:, :129], 129)
            st.begin_request(1 / 8)
            with patch("torch.topk", wraps=torch.topk) as topk, \
                    patch("torch.gather", side_effect=AssertionError("torch.gather must not be used")), \
                    patch.object(torch.Tensor, "gather", side_effect=AssertionError("Tensor.gather must not be used")):
                run.step(ids[:, 129:130])
            self.assertEqual(topk.call_count, st.cache.layers, "exactly one top-k per layer per step")
            stA = make_state(m, 160, "full")
            runA = LabRunner(m, holder, stA)
            runA.prefill(ids[:, :129], 129)
            with patch("torch.topk", wraps=torch.topk) as topk:
                runA.step(ids[:, 129:130])
            self.assertEqual(topk.call_count, 0)
            unwrap_model(m)

    def test_branches_do_not_alias(self):
        device, dtype = CASES[0]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(110, seed=9)
            alt = token_ids(110, seed=10)
            holder = wrap_model(m)
            st = make_state(m, 128, "sparse")
            run = LabRunner(m, holder, st)
            run.prefill(ids[:, :100], 100)
            st.begin_request(1 / 8)
            st2 = make_state(m, 128, "sparse")
            st2.cache = st.cache.clone()
            st2.indexers = st.indexers
            st2.begin_request(1 / 8)
            run2 = LabRunner(m, holder, st2)
            ptrs = {t.data_ptr() for t in st.cache.k + st.cache.v + st.cache.z}
            ptrs2 = {t.data_ptr() for t in st2.cache.k + st2.cache.v + st2.cache.z}
            self.assertTrue(ptrs.isdisjoint(ptrs2), "cloned cache shares storage")
            out1 = run.step(ids[:, 100:101])
            out2 = run2.step(alt[:, 100:101])
            self.assertFalse(torch.equal(st.cache.k[0][:, :, 100], st2.cache.k[0][:, :, 100]), "branches wrote the same key")
            self.assertTrue(torch.equal(st.cache.k[0][:, :, :100], st2.cache.k[0][:, :, :100]), "shared prefix must be identical")
            run.rollback(100)
            self.assert_close(run.step(ids[:, 100:101]), out1, {"atol": 0.0, "rel_norm": 0.0}, "branch 1 unaffected by branch 2")
            run2.rollback(100)
            self.assert_close(run2.step(alt[:, 100:101]), out2, {"atol": 0.0, "rel_norm": 0.0}, "branch 2 unaffected by branch 1")
            unwrap_model(m)

    def test_k_rule_fixed_per_request_and_short_history_degenerates_to_full(self):
        device, dtype = CASES[0]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(130, seed=11)
            ref = m(ids).logits
            holder = wrap_model(m)
            st = make_state(m, 160, "sparse")
            run = LabRunner(m, holder, st)
            run.prefill(ids[:, :100], 100)
            k0 = st.begin_request(1 / 8)
            self.assertEqual((st.request_start_length, k0), (101, fixed_budget(101, 1 / 8)))
            self.assertEqual(k0, 13)
            for i in range(100, 110):
                run.step(ids[:, i:i + 1])
                p = st.plan
                self.assertEqual((p.k, p.k0, p.legal_length, p.selected_count, p.candidate_count), (13, 13, i + 1, 68 + 13, i + 1 - 68))
                self.assertEqual(p.write_col, i)
            short = make_state(m, 160, "sparse")
            runs = LabRunner(m, holder, short)
            runs.prefill(ids[:, :20], 20)
            self.assertEqual(short.begin_request(1 / 8), 0)                    # candidates at legal 21: none
            out = torch.cat([runs.step(ids[:, i:i + 1]) for i in range(20, 30)], 1)
            self.assertEqual((short.plan.k, short.plan.selected_count), (0, 30))
            self.assert_close(out, ref[:, 20:30], TINY_TOL[dtype], "legal length <= 68 reads everything (degenerate full)")
            with self.assertRaises(RuntimeError):
                fresh = make_state(m, 160, "sparse")
                LabRunner(m, holder, fresh).step(ids[:, :1])                   # sparse decode without begin_request
            unwrap_model(m)

    def test_phase_is_explicit_and_a_one_token_prefill_chunk_never_selects(self):
        """PR #2 review: the phase used to be inferred from T == 1, so a prompt whose length is not a multiple of the
        chunk ended with a one-token chunk that was treated as a decode step. The phase is now explicit."""
        device, dtype = CASES[0]
        tol = TINY_TOL[dtype]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(140, seed=21)
            ref = m(ids).logits
            holder = wrap_model(m)
            # (a) one-token prompt, sparse state, no begin_request: must be a prefill, not a failing decode step
            st = make_state(m, 64, "sparse")
            run = LabRunner(m, holder, st)
            out = run.prefill(ids[:, :1], 2048, keep_logits=True)
            self.assert_close(out, ref[:, :1], tol, "one-token prompt")
            self.assertIsNone(st.plan)
            self.assertEqual(st.cache.valid_length, 1)
            # (b) chunk = 1 over 120 tokens, sparse state: every chunk is a prefill chunk
            st1 = make_state(m, 160, "sparse")
            run1 = LabRunner(m, holder, st1)
            out1 = run1.prefill(ids[:, :120], 1, keep_logits=True)
            self.assert_close(out1, ref[:, :120], tol, "chunk = 1 prefill")
            self.assertIsNone(st1.plan)
            self.assertTrue(all(bool(torch.isfinite(st1.cache.z[l][..., :120]).all()) for l in range(st1.cache.layers)), "index table not filled")
            # (c) divisible (120 / 40) and non-divisible (120 / 50 -> last chunk 20) chunking agree with the one-shot build
            one = make_state(m, 160, "sparse")
            LabRunner(m, holder, one).prefill(ids[:, :120], 120)
            for chunk in (40, 50):
                stc = make_state(m, 160, "sparse")
                LabRunner(m, holder, stc).prefill(ids[:, :120], chunk)
                for l in range(stc.cache.layers):
                    self.assert_close(stc.cache.k[l][:, :, :120], one.cache.k[l][:, :, :120], tol, f"chunk {chunk} K")
                    self.assert_close(stc.cache.z[l][..., :120], one.cache.z[l][..., :120], tol, f"chunk {chunk} z")
            # (d) begin_request called before a prefill whose last chunk is one token: that chunk must still read everything
            st2 = make_state(m, 160, "sparse")
            run2 = LabRunner(m, holder, st2)
            st2.begin_request(1 / 8)
            out2 = run2.prefill(ids[:, :101], 50, keep_logits=True)                # chunks 50, 50, 1
            self.assert_close(out2, ref[:, :101], tol, "prefill ending in a one-token chunk with k0 already set")
            self.assertIsNone(st2.plan, "the one-token prefill chunk built a selection plan")
            # (e) the phase is an argument, never inferred: forward() needs it, decode rejects T > 1
            with self.assertRaises(TypeError):
                run2.forward(ids[:, 101:102])
            with self.assertRaises(ValueError):
                run2.forward(ids[:, 101:103], phase="decode")
            with self.assertRaises(ValueError):
                run2.forward(ids[:, 101:102], phase="generate")
            unwrap_model(m)

    def test_long_prompt_ending_in_a_one_token_chunk(self):
        """8193 tokens in chunks of 2048 (last chunk = 1 token) vs chunks of 1000, sparse state: the same cache."""
        device = "mps" if MPS else "cpu"
        with torch.inference_mode():
            m = tiny_qwen3(dtype=torch.float32, max_positions=16384).to(device)
            ids = token_ids(8193, seed=22).to(device)
            holder = wrap_model(m)
            caches = {}
            for chunk in (2048, 1000):
                st = make_state(m, 8193 + 8, "sparse")
                st.begin_request(1 / 32)                                          # k0 set early on purpose
                run = LabRunner(m, holder, st)
                run.prefill(ids, chunk)
                self.assertEqual(st.cache.valid_length, 8193)
                self.assertIsNone(st.plan, f"chunk {chunk}: a prefill chunk selected rows")
                caches[chunk] = st
            tol = TINY_TOL["float32"]
            for l in range(caches[2048].cache.layers):
                self.assert_close(caches[2048].cache.k[l][:, :, :8193], caches[1000].cache.k[l][:, :, :8193], tol, f"layer {l} K 2048 vs 1000")
                self.assert_close(caches[2048].cache.v[l][:, :, :8193], caches[1000].cache.v[l][:, :, :8193], tol, f"layer {l} V 2048 vs 1000")
                self.assert_close(caches[2048].cache.z[l][..., :8193], caches[1000].cache.z[l][..., :8193], tol, f"layer {l} z 2048 vs 1000")
            step_a = LabRunner(m, holder, caches[2048]).step(ids[:, :1])
            step_b = LabRunner(m, holder, caches[1000]).step(ids[:, :1])
            self.assert_close(step_a, step_b, tol, "first decode step after the two prefills")
            self.assertEqual(caches[2048].plan.k, caches[2048].k0)
            unwrap_model(m)

    def test_native_runners_match_and_roll_back(self):
        for device, dtype in CASES:
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                ids = token_ids(100, seed=12).to(device)
                ref = m(ids).logits
                for cache in (DynamicCache(config=m.config), StaticCache(config=m.config, max_cache_len=128)):
                    nat = NativeRunner(m, cache, 128)
                    head = nat.prefill(ids[:, :92], 37, keep_logits=True)
                    tail = torch.cat([nat.step(ids[:, i:i + 1]) for i in range(92, 100)], 1)
                    self.assert_close(torch.cat([head, tail], 1), ref, TINY_TOL[dtype], f"{type(cache).__name__} incremental")
                    nat.rollback(92)
                    again = torch.cat([nat.step(ids[:, i:i + 1]) for i in range(92, 100)], 1)
                    self.assert_close(again, tail, {"atol": 0.0, "rel_norm": 0.0}, f"{type(cache).__name__} after rollback")


@unittest.skipUnless(HAVE_TRANSFORMERS, SKIP_NO_TF)
@unittest.skipUnless(HAVE_TRANSFORMERS and snapshot_present(CONFIG["model"]["id"], CONFIG["model"]["revision"]),
                     "pinned Qwen3 snapshot not in the local Hub cache (tests never download)")
class RealCheckpointCPU(Checks):
    """The pinned checkpoint on CPU float32 at 128 positions: shapes from the config, three-way numerics, full-selection D."""

    def test_three_way_numerics_cpu_float32(self):
        from attention_lab.qwen_wrap import load_model
        cfg = CONFIG["model"]
        tol = REAL_TOL["float32"]
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), torch.inference_mode():
            model, tok, info = load_model(cfg["id"], cfg["revision"], dtype=torch.float32, device="cpu")
            for key, want in cfg["expected_from_config_json"].items():
                got = info.get(key)
                if got is not None:
                    self.assertEqual(got, want, key)
            self.assertEqual(info["resolved_model_commit"], cfg["revision"], "snapshot commit must be the pinned revision")
            self.assertEqual(info["resolved_weights_commit"], cfg["revision"])
            self.assertEqual(info["resolved_tokenizer_commit"], cfg["revision"])
            L = CONFIG["numerics"]["cpu_float32_reference_length"]
            ids = torch.randint(0, 151643, (1, L), generator=torch.Generator().manual_seed(0))
            ref = model(ids).logits
            dc = DynamicCache(config=model.config)
            model(ids, past_key_values=dc, use_cache=True)
            holder = wrap_model(model)
            st = make_state(model, L + 16, "full")
            run = LabRunner(model, holder, st)
            one_shot = run.prefill(ids, L, keep_logits=True)
            self.assert_close(one_shot, ref, {"atol": tol["logits_atol"], "rel_norm": tol["logits_rel_norm"]}, "real model: wrapped one-shot vs unmodified")
            self.assertGreaterEqual(float((one_shot.argmax(-1) == ref.argmax(-1)).float().mean()), tol["argmax_agreement_min"])
            for l in range(st.cache.layers):
                k, v = hf_cache_kv(dc, l)
                self.assert_close(st.cache.k[l][:, :, :L], k, {"atol": tol["cache_atol"], "rel_norm": 1.0}, f"real layer {l} K")
                self.assert_close(st.cache.v[l][:, :, :L], v, {"atol": tol["cache_atol"], "rel_norm": 1.0}, f"real layer {l} V")
            st2 = make_state(model, L + 16, "full")
            run2 = LabRunner(model, holder, st2)
            head = run2.prefill(ids[:, :L - 8], 50, keep_logits=True)
            tail = torch.cat([run2.step(ids[:, i:i + 1]) for i in range(L - 8, L)], 1)
            self.assert_close(torch.cat([head, tail], 1), ref, {"atol": tol["logits_atol"], "rel_norm": tol["logits_rel_norm"]}, "real model: incremental vs unmodified")
            stD = make_state(model, L + 16, "sparse")
            runD = LabRunner(model, holder, stD)
            runD.prefill(ids[:, :L - 8], 50)
            stD.begin_request(None, k0=stD.cache.capacity)
            tailD = torch.cat([runD.step(ids[:, i:i + 1]) for i in range(L - 8, L)], 1)
            self.assert_close(tailD, tail, {"atol": tol["logits_atol"], "rel_norm": tol["logits_rel_norm"]}, "real model: D full selection vs wrapped full")
            unwrap_model(model)


if __name__ == "__main__":
    unittest.main()
