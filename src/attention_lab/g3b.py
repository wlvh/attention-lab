"""All-layer orchestration reuses the reviewed G3a math; one layer of features in RAM."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .g3a import TrainableIndexer, evaluate_document, loss_for_document, query_rows


def layer_config(cfg, layer):
    return dict(cfg, layer_index=layer, index_seed=cfg["index_seed"] + layer - 13,
                uniform_seed=cfg["uniform_seed"] + layer - 13)


def feature_fingerprint(features):
    out = {}
    for name, tensor in features.items():
        a = tensor.detach().cpu().numpy()
        out[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                     "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
    return out


def verify_reuse(root, cfg, manifest):
    old = manifest["config"]
    fields = ("model_id", "revision", "r", "rho", "fixed_prefix", "fixed_recent", "index_seed", "uniform_seed",
              "shuffle_seed", "epochs", "learning_rate", "gradient_clip_norm", "min_candidate_mass",
              "query_offsets_from_suffix_end", "prefill_chunk", "body_dtype", "body_device")
    for name in fields:
        if cfg[name] != old[name]:
            raise ValueError(f"G3a reuse condition differs: {name}")
    if cfg["reuse_layer_index"] != old["layer_index"] or cfg["train_docs"] != old["train_docs"]:
        raise ValueError("G3a reuse layer/data differs")
    for name in ("g3a.py", "g2lite_data.py", "indexer.py", "selection.py", "qwen_wrap.py", "full.py", "reference.py", "sparse.py", "decode.py"):
        rel = "src/attention_lab/" + name
        if hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() != manifest["sources"][rel]:
            raise ValueError(f"G3a reuse source differs: {rel}")


def summarize_rows(rows):
    """Equal document weights; preserve train vs previously-seen development distinction."""
    from collections import defaultdict
    from statistics import mean
    metrics = ("kl", "coverage", "candidate_mass", "missed_full_mass", "output_rmse", "output_relative_l2")
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in metrics:
            if row[metric] is None:
                continue
            values = [v for v, valid in zip(row[metric], row["eligible"]) if valid or metric not in ("kl", "coverage")]
            cells[row["split"], row["doc_id"], row["method"]][metric].extend(values)
    by_doc = [dict(split=s, doc_id=d, method=m, **{k: mean(v) for k, v in data.items() if v}) for (s,d,m),data in sorted(cells.items())]
    aggregate = {}
    for split in ("train", "development"):
        aggregate[split] = {}
        for method in ("C", "D0", "D", "R"):
            selected = [r for r in by_doc if r["split"] == split and r["method"] == method]
            aggregate[split][method] = {key: mean(r[key] for r in selected if key in r) if any(key in r for r in selected) else None for key in metrics}
    return {"aggregate": aggregate, "by_document": by_doc}


def train_layer(features, docs, cfg, guard, output_dir, epochs, progress=lambda epoch: None):
    import time
    from .campaign import write_json
    output_dir = Path(output_dir)
    trained = TrainableIndexer(r=cfg["r"], seed=cfg["index_seed"], device=cfg.get("training_device", "mps"))
    initial = TrainableIndexer(r=cfg["r"], seed=cfg["index_seed"])
    initial.save(output_dir / "index_initial.npz")
    rows = [list(query_rows(features[d["doc_id"]], cfg)) for d in docs if d["split"] == "train"]
    eligible = sum(int(r["eligible"].sum()) for group in rows for r in group)
    skipped = sum(int((~r["eligible"]).sum()) for group in rows for r in group)
    optimizer = torch.optim.Adam(trained.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["shuffle_seed"])
    trace, step_seconds = [], []
    for epoch in range(1, epochs + 1):
        losses, gradnorms = [], []
        for i in rng.permutation(len(rows)):
            guard.check()
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_document(trained, rows[i])
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trained.parameters()):
                raise RuntimeError("disconnected/nonfinite index gradient")
            norm = torch.nn.utils.clip_grad_norm_(trained.parameters(), cfg["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite gradient norm")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in trained.parameters()):
                raise RuntimeError("nonfinite index parameters")
            losses.append(float(loss.detach().cpu())); gradnorms.append(float(norm.cpu()))
            step_seconds.append(time.monotonic() - begin)
        trace.append({"epoch": epoch, "online_train_kl": float(np.mean(losses)) if losses else None,
                      "updates": len(losses), "gradient_norm_max": max(gradnorms) if gradnorms else None})
        write_json(output_dir / "training.json", trace)
        progress(epoch)
    trained.save(output_dir / "index_final.npz")
    changes = {name: float((p.detach().cpu() - dict(initial.named_parameters())[name].detach()).norm()) for name, p in trained.named_parameters()}
    if eligible and min(changes.values()) <= 0:
        raise RuntimeError("eligible layer did not update parameters")
    trained_cpu = TrainableIndexer(r=cfg["r"], seed=cfg["index_seed"])
    trained_cpu.restore(output_dir / "index_final.npz")
    return initial, trained_cpu, {"eligible_rows": eligible, "skipped_rows": skipped, "parameter_change_l2": changes,
                                  "training_state": "trained" if eligible else "no_eligible_rows_initial_retained",
                                  "steps": len(step_seconds), "steady_step_seconds": float(np.median(step_seconds[2:])) if len(step_seconds)>2 else None}


def evaluate_layer(features, docs, cfg, initial, trained, output_dir, guard):
    from .campaign import write_json
    rows = []
    with (Path(output_dir) / "metrics.jsonl").open("w") as handle:
        for doc in docs:
            guard.check()
            for row in evaluate_document(doc["doc_id"], features[doc["doc_id"]], cfg, initial, trained):
                row["split"] = doc["split"]
                # All metrics retained; full lists only for predetermined audit docs.
                if doc["doc_index"] not in cfg["audit_feature_doc_indices"]:
                    row.pop("positions")
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                rows.append(row)
    result = summarize_rows(rows)
    write_json(Path(output_dir) / "metrics_summary.json", result)
    return result
