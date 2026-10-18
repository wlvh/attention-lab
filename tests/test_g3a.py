"""CPU contracts for G3a; no checkpoint download or real model execution."""
import importlib.util
from pathlib import Path
import sys

import unittest
try:
    import numpy as np
    import pytest
    import torch
except ImportError as error:
    raise unittest.SkipTest("G3a CPU contracts require numpy, pytest and torch") from error
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from attention_lab.g3a import (TrainableIndexer, body_digest, evaluate_document, extract_features,
                              kl_rows, loss_for_document, query_rows, selection_metrics, teacher)
from attention_lab.indexer import GroupedIndexer


def fixture():
    gen = torch.Generator().manual_seed(15)
    q = torch.randn(2, 2, 2, 8, generator=gen)
    k = torch.randn(2, 96, 8, generator=gen)
    features = dict(q_pre=q, q_main=q, k_pre=k, k_main=k,
                    v=torch.randn(2, 96, 8, generator=gen), query_positions=torch.tensor([84, 95]), prefix_len=torch.tensor(80))
    cfg = dict(rho=.125, fixed_prefix=4, fixed_recent=8, min_candidate_mass=1e-4, uniform_seed=123)
    return features, cfg


def test_initial_formula_and_rebuilt_derived_weights():
    f, cfg = fixture()
    original = GroupedIndexer(2, 2, 8, 4, seed=7)
    trainable = TrainableIndexer(2, 2, 8, 4, seed=7)
    q, k = f["q_pre"].unsqueeze(0), f["k_pre"].unsqueeze(0)
    torch.testing.assert_close(trainable(q, k), original.reference_scores(q, k).float(), atol=1e-6, rtol=1e-5)
    before = trainable(q, k).detach().clone()
    with torch.no_grad():
        trainable.wq.mul_(2)
        trainable.wk.mul_(3)
    torch.testing.assert_close(trainable(q, k), before * 6)


def test_teacher_head_average_and_candidate_normalization():
    f, cfg = fixture()
    row = next(query_rows(f, cfg))
    q, k = f["q_main"][:, :, :1], f["k_main"][:, :85]
    reference = torch.stack([torch.stack([(q[g, h] @ k[g].T / np.sqrt(8)).softmax(-1) for h in range(2)]) for g in range(2)])
    torch.testing.assert_close(row["weights"], reference)
    avg = reference.mean(1).squeeze(1)
    mass = avg[:, 4:77].sum(-1)
    torch.testing.assert_close(row["mass"], mass)
    torch.testing.assert_close(row["target"].sum(-1), torch.ones(2))
    assert row["k0"] == 10 and row["k"] == 10


def test_gathered_output_matches_independent_attention_and_full_degeneracy():
    f, cfg = fixture()
    row = next(query_rows(f, cfg))
    rel = row["target"].topk(row["k"], -1).indices
    result = selection_metrics(row, rel)
    for g in range(2):
        positions = result["positions"][g]
        q, k, v = f["q_main"][g, :, :1], f["k_main"][g, positions], f["v"][g, positions]
        expected = (q @ k.T / np.sqrt(8)).softmax(-1) @ v
        error = (expected - row["full"][g]).square().mean().sqrt()
        assert abs(float(error) - result["output_rmse"][g]) < 1e-6
    all_candidates = torch.arange(row["hi"] - row["lo"]).expand(2, -1)
    full = selection_metrics(row, all_candidates)
    assert max(full["output_relative_l2"]) < 1e-6
    assert max(abs(x - 1) for x in full["coverage"]) < 1e-6


def test_gradient_only_in_index_and_train_loss_can_fall(tmp_path):
    f, cfg = fixture()
    f["q_pre"].requires_grad_(True)
    f["k_pre"].requires_grad_(True)
    rows = list(query_rows(f, cfg))
    model = TrainableIndexer(2, 2, 8, 4)
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    start = float(loss_for_document(model, rows).detach())
    for _ in range(20):
        opt.zero_grad()
        loss = loss_for_document(model, rows)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in model.parameters())
        opt.step()
    assert f["q_pre"].grad is None and f["k_pre"].grad is None
    assert float(loss_for_document(model, rows).detach()) < start
    model.save(tmp_path / "weights.npz")
    clone = TrainableIndexer(2, 2, 8, 4)
    clone.restore(tmp_path / "weights.npz")
    torch.testing.assert_close(model.wq, clone.wq)
    torch.testing.assert_close(model.wk, clone.wk)


def test_uniform_teacher_and_constant_values_counterexample():
    f, cfg = fixture()
    f["q_main"].zero_()
    f["v"].fill_(2)
    idx = TrainableIndexer(2, 2, 8, 4)
    results = evaluate_document("same", f, cfg, idx, idx)
    for r in results:
        assert max(abs(c - r["k"] / r["candidate_count"]) for c in r["coverage"]) < 1e-6
        # fp32 weighted sums are not bitwise equal to the analytical constant.
        assert max(r["output_rmse"]) < r["legal"] * np.finfo(np.float32).eps * 2
    assert [r["positions"] for r in results] == [r["positions"] for r in evaluate_document("same", f, cfg, idx, idx)]


def test_common_budget_unique_positions_R_coverage_and_low_mass_skip():
    f, cfg = fixture()
    idx = TrainableIndexer(2, 2, 8, 4)
    results = evaluate_document("same", f, cfg, idx, idx)
    for qi in range(2):
        rows = [r for r in results if r["query_index"] == qi]
        assert len({(r["k0"], r["k"], r["selected_count"]) for r in rows}) == 1
        oracle = rows[-1]
        for r in rows:
            for g, positions in enumerate(r["positions"]):
                assert len(set(positions)) == r["selected_count"]
                assert min(positions) >= 0 and max(positions) < r["legal"]
                assert r["coverage"][g] <= oracle["coverage"][g] + 1e-6
    cfg["min_candidate_mass"] = 2
    assert loss_for_document(idx, list(query_rows(f, cfg))) is None


def test_capture_single_layer_frozen_body_and_legal_shapes():
    pytest.importorskip("transformers")
    from attention_lab.qwen_wrap import tiny_qwen3, wrap_model
    model = tiny_qwen3(layers=2, hidden=32, heads=4, groups=2, head_dim=8, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    holder = wrap_model(model)
    before = body_digest(model)
    cfg = dict(layer_index=1, query_offsets_from_suffix_end=[4, 1], prefill_chunk=16)
    doc = dict(prefix_token_count=80, prompt_token_count=88, prefix_ids=[2] * 80, suffix_ids=[3] * 8)
    features = extract_features(model, holder, doc, cfg)
    assert features["q_pre"].shape == (2, 2, 2, 8)
    assert features["k_pre"].shape == (2, 88, 8)
    assert features["query_positions"].tolist() == [84, 87]
    assert all(not x.requires_grad for x in features.values())
    assert body_digest(model) == before
    assert all(p.grad is None for p in model.parameters())


def test_incomplete_never_done():
    spec = importlib.util.spec_from_file_location("g3a_runner", ROOT / "bench/g3a_train.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    p = dict(documents=64, epochs=50, evaluated_documents=64)
    assert runner.completion_status(p, p) == "done"
    assert runner.completion_status(p, dict(p, evaluated_documents=63)) == "incomplete"
    assert runner.completion_status(p, p, "budget_stop") == "incomplete"


def test_parameter_update_check_uses_tensor_values():
    spec = importlib.util.spec_from_file_location("g3a_runner", ROOT / "bench/g3a_train.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    initial = TrainableIndexer(2, 2, 8, 4)
    current = TrainableIndexer(2, 2, 8, 4)
    assert runner.parameter_changes(current, initial) == {"wq": 0.0, "wk": 0.0}
    with torch.no_grad():
        current.wq.add_(.01)
        current.wk.add_(.02)
    assert min(runner.parameter_changes(current, initial).values()) > 0
