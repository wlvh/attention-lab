"""G2-lite contracts: the B (recent) and C (uniform) selectors beside D0 (index), the answer scorer, the data generator.

Selectors (tiny random Qwen3, CPU float64 first, MPS float32 / bfloat16 when available): with the same fixed set F and
the same budget k the three sparse reads touch exactly the same number of rows, every row list contains F, the extra
rows lie inside the candidate domain without duplicates, B's extra rows are exactly the k most recent candidates, C's
draw is fixed by its seed, and with k0 = capacity all three degenerate to the full read A. Poisoning every row a method
did not list leaves its output bitwise unchanged while the full read breaks, which proves each method reads only its
list. B and C never score (no index projections) and nothing uses torch.gather.

The scorer tests need no torch. The data tests need the pinned tokenizer in the local Hub cache (never downloaded).
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

try:
    from attention_lab.g2lite_eval import cell_rate, extract_numbers, score_answer  # noqa: E402
except ImportError:  # the package __init__ imports torch; on the torch-less CI image load the pure-Python scorer directly
    import importlib.util
    _spec = importlib.util.spec_from_file_location("g2lite_eval_standalone", ROOT / "src/attention_lab/g2lite_eval.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    cell_rate, extract_numbers, score_answer = _mod.cell_rate, _mod.extract_numbers, _mod.score_answer

if HAVE_TRANSFORMERS:
    from attention_lab.qwen_wrap import LabRunner, LabState, make_cache, make_indexers, make_state, tiny_qwen3, unwrap_model, wrap_model
    from attention_lab.selection import FIXED_PREFIX, FIXED_RECENT, fixed_budget

CONFIG = json.loads((ROOT / "configs/mac_g2lite.json").read_text(encoding="utf-8"))
TOL = {"float64": {"atol": 1e-10, "rel_norm": 1e-12}, "float32": {"atol": 1e-4, "rel_norm": 1e-5}, "bfloat16": {"atol": 5e-2, "rel_norm": 2e-2}}
MPS = torch is not None and os.environ.get("ATTENTION_LAB_TEST_CPU_ONLY") != "1" and torch.backends.mps.is_available()
CASES = [("cpu", "float64")] + ([("mps", "float32"), ("mps", "bfloat16")] if MPS else [])
SKIP_NO_TF = "transformers not installed (or torch not installed)"
SELECTOR_OF = {"B": "recent", "C": "uniform", "D0": "index"}


def tokenizer_present() -> bool:
    if not HAVE_TRANSFORMERS:
        return False
    from huggingface_hub import try_to_load_from_cache
    m = CONFIG["model"]
    return all(isinstance(try_to_load_from_cache(m["id"], f, revision=m["revision"]), str) for f in ("tokenizer.json", "tokenizer_config.json"))


class ScorerTests(unittest.TestCase):
    def test_extract_standalone_four_digit_numbers(self):
        self.assertEqual(extract_numbers("7305。"), ["7305"])
        self.assertEqual(extract_numbers("编号 4821 的值是 7305"), ["4821", "7305"])
        self.assertEqual(extract_numbers("73051"), [])                       # five digits are not a four-digit number
        self.assertEqual(extract_numbers("2210, 7305"), ["2210", "7305"])
        self.assertEqual(extract_numbers("<think>\n9999\n</think>\n\n7305"), ["7305"])   # a stray think block is dropped
        self.assertEqual(extract_numbers(""), [])

    def test_t1_scoring(self):
        s = score_answer("T1", ["7305"], "7305", truncated=False)
        self.assertTrue(s["exact_match"] and not s["format_error"] and s["item_correct"] == [True])
        s = score_answer("T1", ["7305"], "编号 4821 的值是 7305。", truncated=False)
        self.assertFalse(s["exact_match"], "the first number decides; echoing the id first is wrong by the frozen rule")
        self.assertTrue(s["targets_present_any_order"])
        s = score_answer("T1", ["7305"], "我不知道", truncated=True)
        self.assertTrue(s["format_error"] and s["truncated"] and not s["exact_match"])
        with self.assertRaises(ValueError):
            score_answer("T1", ["1", "2"], "x", truncated=False)

    def test_t2_scoring_order_matters(self):
        ok = score_answer("T2", ["7305", "2210"], "7305, 2210", truncated=False)
        self.assertTrue(ok["exact_match"] and ok["item_correct"] == [True, True] and not ok["order_swapped"])
        sw = score_answer("T2", ["7305", "2210"], "2210,7305", truncated=False)
        self.assertFalse(sw["exact_match"])
        self.assertTrue(sw["order_swapped"] and sw["targets_present_any_order"] and sw["item_correct"] == [False, False])
        half = score_answer("T2", ["7305", "2210"], "7305", truncated=False)
        self.assertTrue(half["format_error"] and half["item_correct"] == [True, False] and not half["exact_match"])

    def test_cell_rate(self):
        self.assertEqual(cell_rate([{"exact_match": True}, {"exact_match": False}]), {"n": 2, "hits": 1, "rate": 0.5})
        self.assertEqual(cell_rate([])["rate"], None)


def tiny(device, dtype):
    return tiny_qwen3(dtype=getattr(torch, dtype)).to(device)


def token_ids(n, seed=1, vocab=256):
    return torch.randint(0, vocab, (1, n), generator=torch.Generator().manual_seed(seed))


def positions_of(state, layer):
    """Flat row ids of the step in flight -> original positions [G, S] (rows = position + group offset)."""
    rows = state.last_rows[layer]
    G, C = state.cache.groups, state.cache.capacity
    return (rows.view(G, -1) - state.cache.offsets.view(G, 1)).cpu(), C


def close(a, b, tol):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return bool(torch.isfinite(a).all()) and float((a - b).abs().max()) <= tol["atol"] and float((a - b).norm() / (b.norm() + 1e-30)) <= tol["rel_norm"]


@unittest.skipUnless(HAVE_TRANSFORMERS, SKIP_NO_TF)
class SelectorContracts(unittest.TestCase):
    L, PRE = 130, 100     # prompt of 100 tokens, then decode steps

    def states(self, m, holder, ids, selectors, *, k0_override=None, rho=1 / 8, uniform_seed=3):
        """One state per selector, each prefilled with the same tokens; the index path owns the index table."""
        out = {}
        for name in selectors:
            sel = SELECTOR_OF[name]
            st = make_state(m, self.L + 32, "sparse", selector=sel, uniform_seed=uniform_seed)
            run = LabRunner(m, holder, st)
            run.prefill(ids[:, :self.PRE], 37)
            st.begin_request(None if k0_override is not None else rho, k0=k0_override)
            st.record_selection = True
            out[name] = (st, run)
        return out

    def test_same_fixed_set_same_budget_same_read_count_and_b_is_the_recent_window(self):
        for device, dtype in CASES:
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                ids = token_ids(self.L, seed=31).to(device)
                holder = wrap_model(m)
                S = self.states(m, holder, ids, ("B", "C", "D0"))
                k0 = fixed_budget(self.PRE + 1, 1 / 8)
                self.assertEqual(k0, 13)
                for step in range(3):
                    tok = ids[:, self.PRE + step:self.PRE + step + 1]
                    for name, (st, run) in S.items():
                        run.step(tok)
                    plans = {name: st.plan for name, (st, _) in S.items()}
                    counts = {name: (p.k, p.selected_count, p.fixed_count) for name, p in plans.items()}
                    self.assertEqual(len(set(counts.values())), 1, f"step {step}: budgets differ {counts}")
                    legal = self.PRE + step + 1
                    F = set(range(FIXED_PREFIX)) | set(range(legal - FIXED_RECENT, legal))
                    lo, hi = FIXED_PREFIX, legal - FIXED_RECENT
                    for name, (st, _) in S.items():
                        for l in range(st.cache.layers):
                            pos, C = positions_of(st, l)
                            self.assertEqual(pos.shape[1], 68 + 13, f"{name} layer {l}: read count")
                            for g in range(pos.shape[0]):
                                got = pos[g].tolist()
                                self.assertEqual(len(set(got)), len(got), f"{name} layer {l} group {g}: duplicate rows")
                                self.assertTrue(F.issubset(got), f"{name} layer {l} group {g}: fixed set missing")
                                extra = sorted(set(got) - F)
                                self.assertEqual(len(extra), 13)
                                self.assertTrue(all(lo <= p < hi for p in extra), f"{name}: a row outside the candidate domain")
                                if name == "B":
                                    self.assertEqual(extra, list(range(hi - 13, hi)), "B must take exactly the k most recent candidates")
                    # C's rows differ from B's and from D0's (13 of 33 candidates: identical sets are astronomically unlikely)
                    pc, _ = positions_of(S["C"][0], 0)
                    pb, _ = positions_of(S["B"][0], 0)
                    pd, _ = positions_of(S["D0"][0], 0)
                    self.assertNotEqual(sorted(pc[0].tolist()), sorted(pb[0].tolist()))
                    self.assertNotEqual(sorted(pc[0].tolist()), sorted(pd[0].tolist()))
                unwrap_model(m)

    def test_uniform_draw_is_fixed_by_its_seed_and_differs_across_layers_steps_and_seeds(self):
        device, dtype = CASES[0]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(self.L, seed=32)
            holder = wrap_model(m)
            a = self.states(m, holder, ids, ("C",), uniform_seed=5)["C"]
            b = self.states(m, holder, ids, ("C",), uniform_seed=5)["C"]
            c = self.states(m, holder, ids, ("C",), uniform_seed=6)["C"]
            outs = {}
            for name, (st, run) in (("a", a), ("b", b), ("c", c)):
                outs[name] = run.step(ids[:, self.PRE:self.PRE + 1])
            for l in range(a[0].cache.layers):
                self.assertTrue(torch.equal(a[0].last_rows[l], b[0].last_rows[l]), "same seed must give the same draw")
                self.assertFalse(torch.equal(a[0].last_rows[l], c[0].last_rows[l]), "a different seed must give a different draw")
            self.assertTrue(torch.equal(outs["a"], outs["b"]))
            self.assertFalse(torch.equal(a[0].last_rows[0], a[0].last_rows[1]), "layers draw independently")
            first = a[0].last_rows[0].clone()
            a[1].step(ids[:, self.PRE + 1:self.PRE + 2])
            self.assertFalse(torch.equal(first[:-1], a[0].last_rows[0][:-1]) and first.numel() == a[0].last_rows[0].numel(), "steps draw independently")
            # replay: rolling back to the same legal length and stepping again must draw the same rows (same request, same step)
            a[1].rollback(self.PRE)
            replay = a[1].step(ids[:, self.PRE:self.PRE + 1])
            self.assertTrue(torch.equal(a[0].last_rows[0], first), "a replayed step must draw the same rows")
            self.assertTrue(torch.equal(replay, outs["a"]), "a replayed C step must reproduce its output")
            unwrap_model(m)

    def test_full_selection_degenerates_to_the_full_read_for_every_selector(self):
        for device, dtype in CASES:
            tol = TOL[dtype]
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                ids = token_ids(self.L, seed=33).to(device)
                holder = wrap_model(m)
                stA = make_state(m, self.L + 32, "full")
                runA = LabRunner(m, holder, stA)
                runA.prefill(ids[:, :self.PRE], 37)
                S = self.states(m, holder, ids, ("B", "C", "D0"), k0_override=self.L + 32)
                for step in range(4):
                    tok = ids[:, self.PRE + step:self.PRE + step + 1]
                    outA = runA.step(tok)
                    for name, (st, run) in S.items():
                        out = run.step(tok)
                        self.assertEqual(st.plan.selected_count, st.plan.legal_length, f"{name}: full selection must cover the legal range")
                        self.assertTrue(close(out, outA, tol), f"{name} with every position selected must match A ({dtype})")
                unwrap_model(m)

    def test_poisoning_unlisted_rows_changes_nothing_for_b_and_c_but_breaks_a(self):
        for device, dtype in CASES:
            with self.subTest(device=device, dtype=dtype), torch.inference_mode():
                m = tiny(device, dtype)
                ids = token_ids(self.L, seed=34).to(device)
                holder = wrap_model(m)
                for name in ("B", "C"):
                    st, run = self.states(m, holder, ids, (name,))[name]
                    stA = make_state(m, self.L + 32, "full")
                    runA = LabRunner(m, holder, stA)
                    runA.prefill(ids[:, :self.PRE], 37)
                    tok = ids[:, self.PRE:self.PRE + 1]
                    clean = run.step(tok)
                    runA.step(tok)
                    G, C = st.cache.groups, st.cache.capacity
                    for l in range(st.cache.layers):
                        pos, _ = positions_of(st, l)
                        for g in range(G):
                            keep = torch.zeros(C, dtype=torch.bool)
                            keep[pos[g]] = True
                            keep = keep.to(device)
                            st.cache.k[l][0, g, ~keep] = float("nan")
                            st.cache.v[l][0, g, ~keep] = float("nan")
                            stA.cache.k[l][0, g, ~keep] = float("nan")
                            stA.cache.v[l][0, g, ~keep] = float("nan")
                    run.rollback(self.PRE)
                    runA.rollback(self.PRE)
                    again = run.step(tok)
                    self.assertTrue(torch.isfinite(again).all(), f"{name} read a row it did not list")
                    self.assertTrue(torch.equal(again, clean), f"{name}: the listed rows alone determine the output")
                    self.assertFalse(torch.isfinite(runA.step(tok)).all(), "the full read must see the poison")
                unwrap_model(m)

    def test_b_and_c_never_score_and_nothing_gathers(self):
        device, dtype = CASES[0]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(self.L, seed=35)
            holder = wrap_model(m)
            S = self.states(m, holder, ids, ("B", "C"))
            with patch("attention_lab.decode.candidate_scores", side_effect=AssertionError("B/C must not score candidates")), \
                    patch("attention_lab.decode.project_queries", side_effect=AssertionError("B/C must not project queries")), \
                    patch("torch.gather", side_effect=AssertionError("torch.gather must not be used")), \
                    patch.object(torch.Tensor, "gather", side_effect=AssertionError("Tensor.gather must not be used")):
                for name, (st, run) in S.items():
                    run.step(ids[:, self.PRE:self.PRE + 1])
                    self.assertIsNone(st.cache.z, f"{name} has no index table")
            unwrap_model(m)

    def test_state_rules_and_prefix_clone_protocol(self):
        device, dtype = CASES[0]
        with torch.inference_mode():
            m = tiny(device, dtype)
            ids = token_ids(self.L, seed=36)
            holder = wrap_model(m)
            with self.assertRaises(ValueError):
                LabState(make_cache(m, 64, with_index=False), None, mode="sparse", selector="index")   # D0 needs an index table
            with self.assertRaises(ValueError):
                LabState(make_cache(m, 64, with_index=False), None, mode="sparse", selector="window")  # unknown selector
            self.assertIsNone(make_state(m, 64, "sparse", selector="recent").cache.z)
            self.assertIsNotNone(make_state(m, 64, "sparse", selector="index").cache.z)
            # the G2-lite protocol: one dense prefix with the index table, then an independent clone per method
            idx = make_indexers(m, r=32, seed=0)
            cache0 = make_cache(m, self.L + 32, with_index=True, index_dim=32)
            st0 = LabState(cache0, idx, mode="full")
            LabRunner(m, holder, st0).prefill(ids[:, :self.PRE], 37)
            self.assertTrue(torch.isfinite(cache0.z[0][..., :self.PRE]).all(), "the dense prefix must build the index table")
            outs = {}
            for name in ("A", "B", "C", "D0"):
                mode, sel = ("full", "index") if name == "A" else ("sparse", SELECTOR_OF[name])
                st = LabState(cache0.clone(), idx if name == "D0" else None, mode=mode, selector=sel)
                run = LabRunner(m, holder, st)
                if name != "A":
                    st.begin_request(1 / 8)
                outs[name] = run.step(ids[:, self.PRE:self.PRE + 1])
                self.assertEqual(cache0.valid_length, self.PRE, "the prefix cache itself must not advance")
            self.assertTrue(all(torch.isfinite(o).all() for o in outs.values()))
            unwrap_model(m)


@unittest.skipUnless(HAVE_TRANSFORMERS and tokenizer_present(), "pinned Qwen3 tokenizer not in the local Hub cache (tests never download)")
class DataGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from attention_lab import g2lite_data as gd
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            cls.tok = AutoTokenizer.from_pretrained(CONFIG["model"]["id"], revision=CONFIG["model"]["revision"])
        cls.gd = gd

    def test_grid_and_determinism(self):
        plan = self.gd.plan_docs(CONFIG)
        self.assertEqual(len(plan), 48)
        self.assertEqual(sum(1 for p in plan if p["task"] == "T1" and p["length_tag"] == "2K"), 12)
        d1 = self.gd.build_document(self.tok, CONFIG, "T1", "2K", 0)
        d2 = self.gd.build_document(self.tok, CONFIG, "T1", "2K", 0)
        self.assertEqual(d1["prompt_sha256"], d2["prompt_sha256"])
        self.assertEqual(d1["targets"], d2["targets"])
        d3 = self.gd.build_document(self.tok, CONFIG, "T1", "2K", 1)
        self.assertNotEqual(d1["ids"], d3["ids"], "each document binds its own ids and values")

    def test_prefix_is_token_closed_and_near_the_target_length(self):
        lay = CONFIG["data"]["layout"]
        for task, length_tag, i in (("T1", "2K", 0), ("T2", "2K", 1), ("T1", "8K", 11), ("T2", "8K", 10)):
            d = self.gd.build_document(self.tok, CONFIG, task, length_tag, i)
            self.assertTrue(d["prefix_closed"])
            self.assertEqual(d["n_records"], CONFIG["data"]["n_records"])
            self.assertLessEqual(abs(d["prefix_token_count"] - d["target_tokens"]), lay["length_tolerance_tokens"], f"{d['doc_id']}: {d['prefix_token_count']} tokens")
            self.assertEqual(d["tokens_per_record"], 12)
            self.assertNotIn(self.gd.QUESTION_MARKER, d["prefix_text"])
            self.assertTrue(d["prompt_text"].endswith("<think>\n\n</think>\n\n"), "thinking must be closed by the template")
            self.assertEqual(len(set(d["ids"])), d["n_records"], "ids are distinct")
            self.assertFalse(any(ch.isdigit() for ch in "".join(CONFIG["data"]["filler_pool"])), "filler must be digit-free")
            for t, span in zip(d["target_indices"], d["target_spans"]):          # the target span decodes to the queried record line
                text = self.tok.decode(d["prefix_ids"][span[0]:span[1]])
                self.assertIn(d["ids"][t], text)
                self.assertIn(d["values"][t], text)
                self.assertTrue(span[1] <= d["prefix_token_count"])
            # the two tail records sit at the configured distances from the end, in every document
            dist = d["record_distance_from_prefix_end"]
            self.assertTrue(lay["back_near_tokens_from_end"][0] <= dist[-1] <= lay["back_near_tokens_from_end"][1], f"{d['doc_id']}: near tail at {dist[-1]}")
            self.assertTrue(lay["back_far_tokens_from_end"][0] <= dist[-2] <= lay["back_far_tokens_from_end"][1], f"{d['doc_id']}: far tail at {dist[-2]}")
            self.assertTrue(all(a > b for a, b in zip(dist, dist[1:])), "records are in document order")

    def test_buckets_place_targets_where_the_plan_says(self):
        n = CONFIG["data"]["n_records"]
        body = n - 2
        for i, bucket in enumerate(CONFIG["data"]["bucket_plan"]["T1"]):
            d = self.gd.build_document(self.tok, CONFIG, "T1", "2K", i)
            t = d["target_indices"][0]
            spec = CONFIG["data"]["buckets"][bucket]
            if "fraction" in spec:
                self.assertTrue(int(spec["fraction"][0] * body) <= t < max(int(spec["fraction"][0] * body) + 1, int(spec["fraction"][1] * body)), f"{d['doc_id']} {bucket}: index {t}")
            else:
                self.assertEqual(n - 1 - t, spec["from_end"], f"{d['doc_id']} {bucket}")
        for i in (0, 1):
            d = self.gd.build_document(self.tok, CONFIG, "T2", "2K", i)
            a, b = d["target_indices"]
            self.assertTrue(a < b if i % 2 == 0 else a > b, "T2 asks in ascending position order for even indices, descending for odd")
            self.assertEqual(d["targets"], [d["values"][a], d["values"][b]])
            self.assertIn(d["ids"][a], d["question"])
            self.assertTrue(d["question"].index(d["ids"][a]) < d["question"].index(d["ids"][b]))

    def test_rewrites_change_exactly_what_they_claim(self):
        base = self.gd.build_document(self.tok, CONFIG, "T1", "2K", 2)
        t = base["target_indices"][0]
        val = self.gd.rewrite_document(self.tok, CONFIG, base, "value")
        self.assertEqual(val["ids"], base["ids"])
        self.assertNotEqual(val["values"][t], base["values"][t])
        self.assertEqual([v for i, v in enumerate(val["values"]) if i != t], [v for i, v in enumerate(base["values"]) if i != t])
        self.assertEqual(val["targets"], [val["values"][t]])
        self.assertEqual(val["prefix_token_count"], base["prefix_token_count"])
        self.assertEqual(val["question"], base["question"])
        fmt = CONFIG["data"]["record_format"]
        self.assertEqual(val["prefix_text"].replace(fmt.format(id=base["ids"][t], value=val["values"][t]), fmt.format(id=base["ids"][t], value=base["values"][t]), 1),
                         base["prefix_text"], "only the queried record line differs")
        irr = self.gd.rewrite_document(self.tok, CONFIG, base, "irrelevant")
        j = irr["changed_record_index"]
        self.assertGreaterEqual(abs(j - t), CONFIG["data"]["irrelevant_min_distance_records"])
        self.assertEqual(irr["targets"], base["targets"])
        self.assertEqual(sum(1 for a, b in zip(irr["values"], base["values"]) if a != b), 1)
        self.assertEqual(irr["target_spans"], base["target_spans"], "the queried record keeps its position")
        mv = self.gd.rewrite_document(self.tok, CONFIG, base, "move")
        self.assertEqual(mv["targets"], base["targets"])
        self.assertEqual(mv["target_records_from_end"], [0])
        self.assertTrue(CONFIG["data"]["layout"]["back_near_tokens_from_end"][0] <= mv["target_distance_from_prefix_end"][0] <= CONFIG["data"]["layout"]["back_near_tokens_from_end"][1])
        self.assertEqual(sorted(mv["ids"]), sorted(base["ids"]))
        self.assertEqual(mv["ids"][t], base["ids"][-1], "the former tail record takes the queried record's old slot")
        self.assertEqual(mv["parent_doc_id"], base["doc_id"])
        self.assertEqual(mv["prefix_token_count"], base["prefix_token_count"])
        tail = self.gd.build_document(self.tok, CONFIG, "T1", "2K", 8)             # back_near target: nothing to move
        with self.assertRaises(ValueError):
            self.gd.rewrite_document(self.tok, CONFIG, tail, "move")
        with self.assertRaises(ValueError):
            self.gd.rewrite_document(self.tok, CONFIG, base, "base")

    def test_c_rewrites_replay_actual_original_positions_on_cpu(self):
        """Same seed alone is insufficient: replay the real per-step/layer draws."""
        from attention_lab.g2lite_control import uniform_seed_for
        from attention_lab.decode import select_rows_uniform
        from attention_lab.qwen_wrap import build_step_plan
        base = self.gd.build_document(self.tok, CONFIG, "T1", "2K", 0)
        docs = [base] + [self.gd.rewrite_document(self.tok, CONFIG, base, v) for v in ("value", "irrelevant")]
        m = tiny("cpu", "float64")
        states = []
        for doc in docs:
            self.assertEqual((doc["prefix_token_count"], doc["suffix_token_count"]), (base["prefix_token_count"], base["suffix_token_count"]))
            st = LabState(make_cache(m, doc["prompt_token_count"] + 16, with_index=False), mode="sparse", selector="uniform", uniform_seed=uniform_seed_for(CONFIG, doc))
            st.cache.valid_length = doc["prefix_token_count"]
            states.append(st)
        first_stream = None
        for request_counter in (1, 2):
            for st in states:
                st.begin_request(1 / 8)
                self.assertEqual(st.request_counter, request_counter)
            for step in (0, base["suffix_token_count"] - 1, base["suffix_token_count"] + 2):
                draws = []
                for st in states:
                    p = build_step_plan(st.cache, st.request_start_length + step, st.k0)
                    rng = st.uniform_rng(step)
                    per_layer = []
                    for _ in range(st.cache.layers):
                        rows = select_rows_uniform(p.fixed_rows, p.off_lo, p.lo, p.hi, p.k, rng=rng)
                        pos = rows.view(st.cache.groups, -1) - st.cache.offsets.view(st.cache.groups, 1)
                        per_layer.append(pos.sort(dim=-1).values)
                    draws.append(torch.stack(per_layer))
                self.assertTrue(torch.equal(draws[0], draws[1]))
                self.assertTrue(torch.equal(draws[0], draws[2]))
                if step == 0:
                    if first_stream is None:
                        first_stream = draws[0]
                    else:
                        self.assertFalse(torch.equal(first_stream, draws[0]), "a genuinely new request uses a new stream")


if __name__ == "__main__":
    unittest.main()
