"""One-layer teacher imitation on frozen full-path features; no sparse model deployment.

q_pre/q_main: [G,h,Q,d]; k_pre/k_main/v: [G,N,d]. Only four question rows,
one layer and one document are processed at a time. Diagnostics use float32.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from .indexer import GroupedIndexer, project_keys, project_queries
from .selection import fixed_budget, selection_layout


class TrainableIndexer(torch.nn.Module):
    """The existing P0 formula, with exactly Wq/Wk trainable (no stale derived weights/z)."""

    def __init__(self, groups=8, heads_per_group=2, head_dim=128, r=32, seed=1803, device="cpu"):
        super().__init__()
        initial = GroupedIndexer(groups, heads_per_group, head_dim, r, seed=seed)
        self.r = r
        self.wq = torch.nn.Parameter(initial.wq.to(device))
        self.wk = torch.nn.Parameter(initial.wk.to(device))

    def forward(self, q_pre, k_pre):
        q = q_pre.detach().to(device=self.wq.device, dtype=self.wq.dtype)
        k = k_pre.detach().to(device=self.wk.device, dtype=self.wk.dtype)
        u = project_queries(q, (self.wq / math.sqrt(self.r)).transpose(1, 2))
        # z must be rebuilt after EVERY optimizer step; only body features are reusable.
        z = project_keys(k, self.wk)
        return torch.matmul(u, z)

    def save(self, path):
        np.savez_compressed(path, wq=self.wq.detach().cpu().numpy(), wk=self.wk.detach().cpu().numpy())

    def restore(self, path):
        with np.load(path, allow_pickle=False) as a, torch.no_grad():
            self.wq.copy_(torch.from_numpy(a["wq"]).to(self.wq.device))
            self.wk.copy_(torch.from_numpy(a["wk"]).to(self.wk.device))


def teacher(q_main, k_main, v):
    """One query: [G,h,1,d], [G,N,d] -> per-head probabilities, group mean, full output.

    Average probabilities AFTER per-head softmax, never average logits first.
    Inputs already contain the main model's RoPE; only legal K/V are supplied.
    """
    q, k, value = (x.detach().float() for x in (q_main, k_main, v))
    logits = torch.matmul(q, k[:, None].transpose(-1, -2)) / math.sqrt(q.shape[-1])
    weights = logits.softmax(-1)
    return weights, weights.mean(1).squeeze(1), torch.matmul(weights, value[:, None])


def query_rows(features, cfg):
    """Produce detached teachers with F removed and fixed request-start budget."""
    k0 = fixed_budget(int(features["prefix_len"]) + 1, cfg["rho"], cfg["fixed_prefix"], cfg["fixed_recent"])
    for qi, pos in enumerate(features["query_positions"].tolist()):
        n = int(pos) + 1
        layout = selection_layout(n, cfg["fixed_prefix"], cfg["fixed_recent"])
        lo, hi = layout.candidate_lo, layout.candidate_hi
        weights, avg, full = teacher(features["q_main"][:, :, qi:qi + 1], features["k_main"][:, :n], features["v"][:, :n])
        mass = avg[:, lo:hi].sum(-1)
        target = avg[:, lo:hi] / mass[:, None].clamp_min(torch.finfo(torch.float32).tiny)
        yield {"qi": qi, "position": int(pos), "legal": n, "lo": lo, "hi": hi, "k0": k0,
               "k": layout.clamp_k(k0), "fixed": layout.fixed_positions(), "mass": mass,
               "eligible": mass >= cfg["min_candidate_mass"], "target": target,
               "weights": weights, "full": full,
               "q": features["q_pre"][:, :, qi:qi + 1].unsqueeze(0),
               "k_pre": features["k_pre"][:, lo:hi].unsqueeze(0), "v": features["v"][:, :n]}


def kl_rows(logits, target):
    """KL(teacher || index) per group, in nats; no gradient into the teacher."""
    p = target.detach().to(logits.device)
    return (torch.xlogy(p, p) - p * logits.log_softmax(-1)).sum(-1)


def loss_for_document(indexer, rows):
    losses = []
    for row in rows:
        logits = indexer(row["q"], row["k_pre"])[0, :, 0]
        valid = row["eligible"].to(logits.device)
        if bool(valid.any()):
            losses.append(kl_rows(logits, row["target"])[valid])
    return torch.cat(losses).mean() if losses else None


def uniform_positions(groups, size, k, seed):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(np.stack([rng.choice(size, k, replace=False) for _ in range(groups)]))


def selection_metrics(row, relative_positions):
    """Reweight original main-head probabilities on F + selection (same Q/K/V).

    This equals softmax restricted to the selected logits; the explicit test compares
    it with independently recomputed gathered attention. Output is BEFORE o_proj.
    """
    g = row["target"].shape[0]
    rel = relative_positions.long()
    fixed = torch.tensor(row["fixed"]).expand(g, -1)
    selected = torch.cat([fixed, rel + row["lo"]], -1)
    coverage = row["target"].gather(1, rel).sum(-1)
    weights = row["weights"]
    h = weights.shape[1]
    chosen = weights.gather(-1, selected[:, None, None].expand(-1, h, 1, -1))
    chosen = chosen / chosen.sum(-1, keepdim=True)
    values = row["v"].gather(1, selected[..., None].expand(-1, -1, row["v"].shape[-1]))
    output = torch.matmul(chosen, values[:, None])
    delta = (output - row["full"]).flatten(1)
    ref = row["full"].flatten(1)
    return {"coverage": coverage.tolist(), "missed_full_mass": (row["mass"] * (1 - coverage)).tolist(),
            "output_rmse": delta.square().mean(-1).sqrt().tolist(),
            "output_relative_l2": (delta.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).tolist(),
            "positions": selected.tolist()}


@torch.no_grad()
def evaluate_document(doc_id, features, cfg, initial, trained):
    records = []
    for row in query_rows(features, cfg):
        g, m = row["target"].shape
        seed = cfg["uniform_seed"] + int(hashlib.sha256(f'{doc_id}/{row["qi"]}'.encode()).hexdigest()[:8], 16)
        logits = {"D0": initial(row["q"], row["k_pre"])[0, :, 0].cpu(),
                  "D": trained(row["q"], row["k_pre"])[0, :, 0].cpu(), "C": torch.zeros(g, m)}
        choices = {method: s.topk(row["k"], dim=-1).indices for method, s in logits.items() if method != "C"}
        choices["C"] = uniform_positions(g, m, row["k"], seed)
        choices["R"] = row["target"].topk(row["k"], dim=-1).indices
        for method in ("C", "D0", "D", "R"):
            records.append({"doc_id": doc_id, "method": method, "query_index": row["qi"], "query_position": row["position"],
                            "legal": row["legal"], "candidate_count": m, "k0": row["k0"], "k": row["k"],
                            "fixed_count": len(row["fixed"]), "selected_count": len(row["fixed"]) + row["k"],
                            "candidate_mass": row["mass"].tolist(), "eligible": row["eligible"].tolist(),
                            "teacher_entropy": (-torch.xlogy(row["target"], row["target"]).sum(-1)).tolist(),
                            "kl": kl_rows(logits[method], row["target"]).tolist() if method != "R" else None,
                            **selection_metrics(row, choices[method])})
    return records


def load_features(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: torch.from_numpy(a[k].copy()) for k in a.files}


def body_digest(model):
    """Hash the actual in-memory tensors (not just the original checkpoint file)."""
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(str((tuple(p.shape), p.dtype)).encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def extract_features(model, holder, doc, cfg, check_budget=lambda: None):
    """Read-only hooks on the existing wrapper; body follows the complete A-step path."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from .qwen_wrap import LabRunner, LabState, make_cache
    n = doc["prompt_token_count"]
    p = doc["prefix_token_count"]
    positions = [n - o for o in cfg["query_offsets_from_suffix_end"]]
    if min(positions) < p or len(set(positions)) != len(positions):
        raise ValueError("query rows must be distinct question/suffix positions")
    cache = make_cache(model, batch=1, capacity=n, with_index=False)
    state = LabState(cache, None, mode="full")
    runner = LabRunner(model, holder, state)
    attn = model.model.layers[cfg["layer_index"]].self_attn
    keys, queries, main_queries = [], [], []
    active = {"position": -1, "rotary": None}

    def before_attention(module, args, kwargs):
        active["rotary"] = kwargs["position_embeddings"]

    def on_key(module, args, output):
        keys.append(output.detach().transpose(1, 2)[0].float().cpu())

    def on_query(module, args, output):
        if active["position"] in positions:
            q = output.transpose(1, 2)
            cos, sin = active["rotary"]
            qm, _ = apply_rotary_pos_emb(q, q, cos, sin)
            shape = (cache.groups, attn.heads_per_group, 1, cache.head_dim)
            queries.append(q[0].reshape(shape).float().cpu())
            main_queries.append(qm[0].reshape(shape).float().cpu())

    hooks = [attn.register_forward_pre_hook(before_attention, with_kwargs=True),
             attn.inner.k_norm.register_forward_hook(on_key), attn.inner.q_norm.register_forward_hook(on_query)]
    device = next(model.parameters()).device
    try:
        runner.prefill(torch.tensor([doc["prefix_ids"]], device=device), cfg["prefill_chunk"])
        for j, token in enumerate(doc["suffix_ids"]):
            check_budget()
            active["position"] = p + j
            runner.step(torch.tensor([[token]], device=device))
    finally:
        for hook in hooks:
            hook.remove()
    if len(queries) != len(positions) or cache.valid_length != n:
        raise RuntimeError("incomplete feature extraction")
    layer = cfg["layer_index"]
    result = {"q_pre": torch.cat(queries, 2), "q_main": torch.cat(main_queries, 2),
              "k_pre": torch.cat(keys, 1), "k_main": cache.k[layer][0, :, :n].float().cpu(),
              "v": cache.v[layer][0, :, :n].float().cpu(), "query_positions": torch.tensor(positions),
              "prefix_len": torch.tensor(p)}
    if any(t.requires_grad or not torch.isfinite(t).all() for t in result.values()):
        raise RuntimeError("nonfinite or attached body features")
    return result
