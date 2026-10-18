#!/usr/bin/env python3
"""Independent NumPy float64 audit of saved G3a heldout tensors; no torch or GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-4  # numerical reference tolerance, fixed before reading outcomes


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = manifest["config"]
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    heldout = sorted({r["doc_id"] for r in rows if r["split"] == "heldout"})
    params = {m: dict(np.load(run / name, allow_pickle=False)) for m, name in (("D0", "index_initial.npz"), ("D", "index_final.npz"))}
    change = {k: float(np.linalg.norm(params["D"][k] - params["D0"][k])) for k in ("wq", "wk")}
    assert min(change.values()) > 0
    maxima = {k: 0.0 for k in ("kl", "candidate_mass", "coverage", "missed_full_mass", "output_rmse", "output_relative_l2")}
    checked = 0
    for doc_id in heldout:
        path = run / "features" / (doc_id + ".npz")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["feature_files"][path.name]["sha256"]
        with np.load(path, allow_pickle=False) as file:
            f = {k: file[k].astype(np.float64) for k in file.files}
        for qi, position in enumerate(f["query_positions"].astype(int)):
            legal = position + 1
            lo, hi = cfg["fixed_prefix"], legal - cfg["fixed_recent"]
            q = f["q_main"][:, :, qi]
            keys, values = f["k_main"][:, :legal], f["v"][:, :legal]
            weights = softmax(np.einsum("ghd,gnd->ghn", q, keys) / np.sqrt(q.shape[-1]))
            full = np.einsum("ghn,gnd->ghd", weights, values)
            avg = weights.mean(axis=1)
            mass = avg[:, lo:hi].sum(axis=-1)
            target = avg[:, lo:hi] / mass[:, None]
            for row in (r for r in rows if r["doc_id"] == doc_id and r["query_index"] == qi):
                if row["method"] in params:
                    w = params[row["method"]]
                    query = f["q_pre"][:, :, qi].reshape(8, -1)
                    u = np.einsum("grf,gf->gr", w["wq"].astype(float), query) / np.sqrt(cfg["r"])
                    z = np.einsum("grd,gnd->grn", w["wk"].astype(float), f["k_pre"][:, lo:hi])
                    prediction = softmax(np.einsum("gr,grn->gn", u, z))
                else:
                    prediction = np.ones_like(target) / target.shape[-1]
                kl = (target * (np.log(target) - np.log(prediction))).sum(axis=-1)
                for g, positions in enumerate(row["positions"]):
                    chosen = weights[g][:, positions]
                    output = (chosen / chosen.sum(-1, keepdims=True)) @ values[g, positions]
                    delta = output - full[g]
                    candidates = [p for p in positions if lo <= p < hi]
                    coverage = target[g, np.array(candidates) - lo].sum()
                    reference = {"candidate_mass": mass[g], "coverage": coverage,
                                 "missed_full_mass": mass[g] * (1 - coverage),
                                 "output_rmse": np.sqrt((delta ** 2).mean()),
                                 "output_relative_l2": np.linalg.norm(delta) / np.linalg.norm(full[g])}
                    if row["method"] != "R":
                        reference["kl"] = kl[g]
                    for name, value in reference.items():
                        difference = abs(float(value) - row[name][g])
                        maxima[name] = max(maxima[name], difference)
                        assert np.isfinite(value) and difference < TOLERANCE, (doc_id, qi, g, row["method"], name, difference)
                    checked += 1
    return {"status": "passed", "run_id": run.name, "reference": "independent NumPy float64 from saved features and weights",
            "documents": len(heldout), "query_group_method_comparisons": checked, "tolerance": TOLERANCE,
            "maximum_absolute_difference": maxima, "actual_parameter_change_l2": change,
            "scope": "all 16 heldout docs, 4 query rows, 8 groups, 4 methods; no fresh model inference",
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    (ROOT / "reports/g3a_cpu_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
