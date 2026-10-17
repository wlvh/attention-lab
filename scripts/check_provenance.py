#!/usr/bin/env python3
"""Verify that the archived benchmark evidence matches what the committed reports say was run. No torch, no network.

Checks (PR #1 review, finding 1):
1. The as-run source snapshot reports/provenance/microbench_at_formal_runs.py has the SHA-256 recorded for
   bench/microbench.py in every formal run manifest aggregated in reports/g1a_manifest_mac_m4max.json.
2. Every file listed in reports/provenance/archive_manifest.json has the recorded size and SHA-256.
3. The committed bench/microbench.py differs from the as-run snapshot only by the lines stored in
   reports/provenance/microbench_post_run_edit.diff (so any later edit must update that diff deliberately).

G1b (PR #2 review, finding 3):
4. The formal decode run's manifest hashes of src/attention_lab/qwen_wrap.py and bench/model_decode_bench.py equal the
   snapshots under reports/provenance/g1b/, and the committed files differ from the snapshots only by the recorded diffs.
5. The appendix (phase-fix) run and the current numerics run ran on the code committed at PR #2: their manifest hashes
   equal the committed bench files, and for src/attention_lab/qwen_wrap.py the snapshot
   reports/provenance/g1b/qwen_wrap_at_g1b_numerics4_and_phasefix_runs.py (G2-lite edited the file afterwards; the
   committed file may differ from that snapshot only by the lines in g1b/qwen_wrap_post_g1b_edit.diff).
5b. Every G1a and G1b manifest names the same src/attention_lab/decode.py, archived as g1b/decode_at_g1a_g1b_runs.py;
   the committed decode.py differs from it only by g1b/decode_post_g1b_edit.diff (G2-lite's B / C selectors).
6. The three earlier numerics runs inside g1b_runs_20261017.tar.gz ran on the three archived script versions.

G2-lite (PR #5), only once reports/g2lite_manifest_mac_m4max.json exists:
8. Every formal G2-lite run manifest names the as-run selector, data, wrapper and harness sources archived under
   reports/provenance/g2lite/ (the runs were made on committed code at af62df4; the header replay edited these four files
   afterwards, so the committed files may differ from the snapshots only by the recorded g2lite/*_post_g2lite_edit.diff),
   and the committed scorer src/attention_lab/g2lite_eval.py (unchanged since).

Header replay (reports/g2lite_replay_header_manifest_mac_m4max.json), only once it exists:
9. Every formal header-replay run manifest names the committed selector, data, scorer and wrapper sources and the as-run
   harness snapshot reports/provenance/g2lite_replay/g2lite_quality_at_replay_header_run.py (the filler replay edited the
   harness afterwards; the committed harness differs from that snapshot only by the recorded diff).

Filler replay (reports/g2lite_replay_filler_manifest_mac_m4max.json), only once it exists:
10. Every formal filler-replay run manifest names the committed selector, data, scorer and wrapper sources and the as-run
    harness snapshot reports/provenance/g2lite_replay/g2lite_quality_at_replay_filler_run.py (the header-split replay
    edited the harness afterwards; the committed harness differs from that snapshot only by the recorded diff).

Header-split replay (reports/g2lite_replay_header_split_manifest_mac_m4max.json), only once it exists:
11. Every formal header-split run manifest names the committed selector, data, scorer and wrapper sources and the as-run
    harness snapshot reports/provenance/g2lite_replay/g2lite_quality_at_replay_header_split_run.py (the format-split
    replay edited the harness afterwards; the committed harness differs from that snapshot only by the recorded diff).

Format-split replay (reports/g2lite_replay_format_split_manifest_mac_m4max.json), only once it exists:
12. Every formal format-split run manifest names the committed sources (made on committed code).

P0 / P1 (PR #3, stopped prototype), only where the kernel sources exist:
7. The four formal P0/P1 run manifests inside p0p1_runs_20261017.tar.gz name the committed metal_score.py /
   metal_attend.py hashes.
"""
from __future__ import annotations
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "reports/provenance"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_lines(diff_text: str) -> list[str]:
    return sorted(l for l in diff_text.splitlines() if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))


def main() -> int:
    errors = []
    snap = PROV / "microbench_at_formal_runs.py"
    manifest = json.loads((ROOT / "reports/g1a_manifest_mac_m4max.json").read_text(encoding="utf-8"))
    snap_sha = sha256(snap)
    for run in manifest["runs"]:
        recorded = run.get("source_hashes", {}).get("bench/microbench.py")
        if recorded != snap_sha:
            errors.append(f"{run.get('run_id')}: manifest hash {str(recorded)[:16]} != snapshot {snap_sha[:16]}")
    am = json.loads((PROV / "archive_manifest.json").read_text(encoding="utf-8"))
    for item in am["files"]:
        p = ROOT / item["path"]
        if not p.is_file():
            errors.append(f"missing {item['path']}")
            continue
        if p.stat().st_size != item["bytes"] or sha256(p) != item["sha256"]:
            errors.append(f"changed {item['path']}")
    stored = changed_lines((PROV / "microbench_post_run_edit.diff").read_text(encoding="utf-8"))
    current = changed_lines("".join(difflib.unified_diff(snap.read_text(encoding="utf-8").splitlines(True),
                                                         (ROOT / "bench/microbench.py").read_text(encoding="utf-8").splitlines(True))))
    if stored != current:
        errors.append("bench/microbench.py differs from the as-run snapshot by lines not recorded in microbench_post_run_edit.diff; "
                      "regenerate the diff file deliberately if the edit is intended")
    # ---- G1b (PR #2): decode run on the pre-fix wrapper snapshot, numerics runs on three recorded script versions ----
    import tarfile
    g1b = json.loads((ROOT / "reports/g1b_manifest_mac_m4max.json").read_text(encoding="utf-8"))
    pg = PROV / "g1b"
    snap_wrap, snap_bench = sha256(pg / "qwen_wrap_at_formal_decode_run.py"), sha256(pg / "model_decode_bench_at_formal_decode_run.py")
    for run in g1b["benchmarks"]:
        sh = run.get("source_hashes", {})
        if sh.get("src/attention_lab/qwen_wrap.py") != snap_wrap:
            errors.append(f"{run.get('run_id')}: qwen_wrap.py hash {str(sh.get('src/attention_lab/qwen_wrap.py'))[:16]} != snapshot {snap_wrap[:16]}")
        if sh.get("bench/model_decode_bench.py") != snap_bench:
            errors.append(f"{run.get('run_id')}: model_decode_bench.py hash != snapshot")
    snap_wrap_pr2 = sha256(pg / "qwen_wrap_at_g1b_numerics4_and_phasefix_runs.py")   # the wrapper as committed at PR #2 (edited by G2-lite)
    committed = {"src/attention_lab/qwen_wrap.py": snap_wrap_pr2, "bench/model_decode_bench.py": sha256(ROOT / "bench/model_decode_bench.py"),
                 "bench/model_numerics.py": sha256(ROOT / "bench/model_numerics.py")}
    for run in g1b.get("appendix_benchmarks", []) + g1b.get("numerics", []):   # these ran on the code committed at PR #2
        for f_, h in committed.items():
            if run.get("source_hashes", {}).get(f_) != h:
                errors.append(f"{run.get('run_id')}: {f_} hash != {'PR #2 snapshot' if 'qwen_wrap' in f_ else 'committed file'}")
    # decode.py: unchanged from G1a through G1b (every manifest names the archived snapshot); G2-lite's edit is the recorded diff
    snap_decode = sha256(pg / "decode_at_g1a_g1b_runs.py")
    for run in manifest["runs"] + g1b["benchmarks"] + g1b.get("appendix_benchmarks", []) + g1b.get("numerics", []):
        if run.get("source_hashes", {}).get("src/attention_lab/decode.py") != snap_decode:
            errors.append(f"{run.get('run_id')}: decode.py hash != archived G1a/G1b snapshot")
    # snapshot chains: each as-run snapshot differs from the next snapshot (or the committed file) only by the recorded diff
    chains = (("g1b/qwen_wrap_at_formal_decode_run.py", "g1b/qwen_wrap_post_run_edit.diff", "g1b/qwen_wrap_at_g1b_numerics4_and_phasefix_runs.py"),
              ("g1b/model_decode_bench_at_formal_decode_run.py", "g1b/model_decode_bench_post_run_edit.diff", "bench/model_decode_bench.py"),
              ("g1b/qwen_wrap_at_g1b_numerics4_and_phasefix_runs.py", "g1b/qwen_wrap_post_g1b_edit.diff", "g2lite/qwen_wrap_at_g2lite_runs.py"),
              ("g1b/decode_at_g1a_g1b_runs.py", "g1b/decode_post_g1b_edit.diff", "g2lite/decode_at_g2lite_runs.py"),
              ("g2lite/qwen_wrap_at_g2lite_runs.py", "g2lite/qwen_wrap_post_g2lite_edit.diff", "src/attention_lab/qwen_wrap.py"),
              ("g2lite/decode_at_g2lite_runs.py", "g2lite/decode_post_g2lite_edit.diff", "src/attention_lab/decode.py"),
              ("g2lite/g2lite_data_at_g2lite_runs.py", "g2lite/g2lite_data_post_g2lite_edit.diff", "src/attention_lab/g2lite_data.py"),
              ("g2lite/g2lite_quality_at_g2lite_runs.py", "g2lite/g2lite_quality_post_g2lite_edit.diff", "g2lite_replay/g2lite_quality_at_replay_header_run.py"),
              ("g2lite_replay/g2lite_quality_at_replay_header_run.py", "g2lite_replay/g2lite_quality_post_replay_header_edit.diff", "g2lite_replay/g2lite_quality_at_replay_filler_run.py"),
              ("g2lite_replay/g2lite_quality_at_replay_filler_run.py", "g2lite_replay/g2lite_quality_post_replay_filler_edit.diff", "g2lite_replay/g2lite_quality_at_replay_header_split_run.py"),
              ("g2lite_replay/g2lite_quality_at_replay_header_split_run.py", "g2lite_replay/g2lite_quality_post_replay_header_split_edit.diff", "bench/g2lite_quality.py"))
    for snap, diff_file, current in chains:
        target = (ROOT / current) if current.startswith(("src/", "bench/")) else (PROV / current)
        stored_ = changed_lines((PROV / diff_file).read_text(encoding="utf-8"))
        current_ = changed_lines("".join(difflib.unified_diff((PROV / snap).read_text(encoding="utf-8").splitlines(True), target.read_text(encoding="utf-8").splitlines(True))))
        if stored_ != current_:
            errors.append(f"{current} differs from its as-run snapshot {snap} by lines not recorded in {diff_file}")
    # the three earlier numerics runs inside the archive ran on three recorded versions of bench/model_numerics.py
    expected_numerics = {"g1b-numerics-20261016T202211Z-formal": "model_numerics_at_formal_run.py", "g1b-numerics-20261016T202822Z-formal2": "model_numerics_at_formal2_run.py",
                         "g1b-numerics-20261016T204139Z-formal3": "model_numerics_at_formal3_run.py"}
    seen = set()
    with tarfile.open(PROV / "g1b_runs_20261017.tar.gz") as tf:
        for member in tf.getmembers():
            parts = Path(member.name).parts
            if len(parts) >= 3 and parts[-1] == "manifest.json" and parts[-2] in expected_numerics:
                mf = json.loads(tf.extractfile(member).read().decode("utf-8"))
                want = sha256(pg / expected_numerics[parts[-2]])
                if mf.get("source_hashes", {}).get("bench/model_numerics.py") != want:
                    errors.append(f"{parts[-2]}: archived manifest hash of model_numerics.py != snapshot {expected_numerics[parts[-2]]}")
                seen.add(parts[-2])
    for rid in expected_numerics:
        if rid not in seen:
            errors.append(f"archive lacks runs/{rid}/manifest.json")
    # ---- P0 / P1 (PR #3, stopped prototype): the archived run manifests must name the committed kernel sources ----
    p0p1 = PROV / "p0p1_runs_20261017.tar.gz"
    kernels = {"src/attention_lab/metal_score.py": ROOT / "src/attention_lab/metal_score.py", "src/attention_lab/metal_attend.py": ROOT / "src/attention_lab/metal_attend.py"}
    p0p1_checked = 0
    if p0p1.is_file() and all(path.is_file() for path in kernels.values()):
        committed_k = {k: sha256(v) for k, v in kernels.items()}
        with tarfile.open(p0p1) as tf:
            for member in tf.getmembers():
                parts = Path(member.name).parts
                if len(parts) >= 3 and parts[-1] == "manifest.json" and ("-p1" in parts[-2] or "-p2" in parts[-2]):
                    mf = json.loads(tf.extractfile(member).read().decode("utf-8"))
                    for k, h in committed_k.items():
                        if k in mf.get("source_hashes", {}) and mf["source_hashes"][k] != h:
                            errors.append(f"{parts[-2]}: archived manifest hash of {k} != committed file")
                    p0p1_checked += 1
        if p0p1_checked != 4:
            errors.append(f"expected 4 formal P0/P1 run manifests in the archive, found {p0p1_checked}")
    # ---- G2-lite (PR #5): formal quality runs on the code committed at af62df4, archived as g2lite/* snapshots ----
    g2_path = ROOT / "reports/g2lite_manifest_mac_m4max.json"
    g2_checked = 0
    g2_files = ["src/attention_lab/decode.py", "src/attention_lab/qwen_wrap.py", "src/attention_lab/g2lite_data.py", "src/attention_lab/g2lite_eval.py", "bench/g2lite_quality.py"]
    if g2_path.is_file():
        g2 = json.loads(g2_path.read_text(encoding="utf-8"))
        g2_snapshots = {"src/attention_lab/decode.py": sha256(PROV / "g2lite/decode_at_g2lite_runs.py"), "src/attention_lab/qwen_wrap.py": sha256(PROV / "g2lite/qwen_wrap_at_g2lite_runs.py"),
                        "src/attention_lab/g2lite_data.py": sha256(PROV / "g2lite/g2lite_data_at_g2lite_runs.py"), "bench/g2lite_quality.py": sha256(PROV / "g2lite/g2lite_quality_at_g2lite_runs.py"),
                        "src/attention_lab/g2lite_eval.py": sha256(ROOT / "src/attention_lab/g2lite_eval.py")}
        for run in g2.get("runs", []):
            for f_, h in g2_snapshots.items():
                if run.get("source_hashes", {}).get(f_) != h:
                    errors.append(f"{run.get('run_id')}: {f_} hash != {'as-run snapshot' if 'eval' not in f_ else 'committed file'} (G2-lite runs were made on committed code at af62df4)")
            g2_checked += 1
    # ---- header replay: formal run on the code committed at e3296ec (harness archived as g2lite_replay/* snapshot) ----
    rp_path = ROOT / "reports/g2lite_replay_header_manifest_mac_m4max.json"
    rp_checked = 0
    if rp_path.is_file():
        rp = json.loads(rp_path.read_text(encoding="utf-8"))
        rp_expect = {f_: sha256(ROOT / f_) for f_ in g2_files}
        rp_expect["bench/g2lite_quality.py"] = sha256(PROV / "g2lite_replay/g2lite_quality_at_replay_header_run.py")
        for run in rp.get("runs", []):
            for f_, h in rp_expect.items():
                if run.get("source_hashes", {}).get(f_) != h:
                    errors.append(f"{run.get('run_id')}: {f_} hash != {'as-run snapshot' if 'quality' in f_ else 'committed file'} (header-replay run was made on committed code at e3296ec)")
            rp_checked += 1
    # ---- filler replay: formal run on the code committed at 5d9b9c7 (harness archived as g2lite_replay/* snapshot) ----
    fp_path = ROOT / "reports/g2lite_replay_filler_manifest_mac_m4max.json"
    fp_checked = 0
    if fp_path.is_file():
        fp = json.loads(fp_path.read_text(encoding="utf-8"))
        fp_expect = {f_: sha256(ROOT / f_) for f_ in g2_files}
        fp_expect["bench/g2lite_quality.py"] = sha256(PROV / "g2lite_replay/g2lite_quality_at_replay_filler_run.py")
        for run in fp.get("runs", []):
            for f_, h in fp_expect.items():
                if run.get("source_hashes", {}).get(f_) != h:
                    errors.append(f"{run.get('run_id')}: {f_} hash != {'as-run snapshot' if 'quality' in f_ else 'committed file'} (filler-replay run was made on committed code at 5d9b9c7)")
            fp_checked += 1
    # ---- header-split replay: formal run on the code committed at 3aa1ec8 (harness archived as g2lite_replay/* snapshot) ----
    hs_path = ROOT / "reports/g2lite_replay_header_split_manifest_mac_m4max.json"
    hs_checked = 0
    if hs_path.is_file():
        hs = json.loads(hs_path.read_text(encoding="utf-8"))
        hs_expect = {f_: sha256(ROOT / f_) for f_ in g2_files}
        hs_expect["bench/g2lite_quality.py"] = sha256(PROV / "g2lite_replay/g2lite_quality_at_replay_header_split_run.py")
        for run in hs.get("runs", []):
            for f_, h in hs_expect.items():
                if run.get("source_hashes", {}).get(f_) != h:
                    errors.append(f"{run.get('run_id')}: {f_} hash != {'as-run snapshot' if 'quality' in f_ else 'committed file'} (header-split replay run was made on committed code at 3aa1ec8)")
            hs_checked += 1
    # ---- format-split replay: formal run on committed code ----
    fs_path = ROOT / "reports/g2lite_replay_format_split_manifest_mac_m4max.json"
    fs_checked = 0
    if fs_path.is_file():
        fs = json.loads(fs_path.read_text(encoding="utf-8"))
        fs_committed = {f_: sha256(ROOT / f_) for f_ in g2_files}
        for run in fs.get("runs", []):
            for f_, h in fs_committed.items():
                if run.get("source_hashes", {}).get(f_) != h:
                    errors.append(f"{run.get('run_id')}: {f_} hash != committed file (format-split replay runs must be on committed code)")
            fs_checked += 1
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Provenance OK: G1a snapshot {snap_sha[:16]} matches {len(manifest['runs'])} run manifests; G1b decode snapshot matches "
          f"{len(g1b['benchmarks'])} manifest(s), {len(g1b.get('appendix_benchmarks', []))} appendix + {len(g1b.get('numerics', []))} numerics run(s) on committed code, "
          f"{len(seen)} archived numerics runs on their snapshots; {len(am['files'])} archived files verified; post-run edits limited to the recorded diffs"
          + (f"; {p0p1_checked} archived P0/P1 manifests name the committed kernels" if p0p1_checked else "")
          + (f"; {g2_checked} G2-lite run manifest(s) on the af62df4 snapshots" if g2_checked else "")
          + (f"; {rp_checked} header-replay run manifest(s) on the e3296ec snapshot" if rp_checked else "")
          + (f"; {fp_checked} filler-replay run manifest(s) on the 5d9b9c7 snapshot" if fp_checked else "")
          + (f"; {hs_checked} header-split replay run manifest(s) on the 3aa1ec8 snapshot" if hs_checked else "")
          + (f"; {fs_checked} format-split replay run manifest(s) on committed code." if fs_checked else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
