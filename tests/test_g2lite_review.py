"""PR #5 review regressions. All execution is CPU-only; no model download."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def standalone(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = standalone("review_control", "src/attention_lab/g2lite_control.py")
evaluation = standalone("review_evaluation", "src/attention_lab/g2lite_eval.py")
report = standalone("review_report", "scripts/g2lite_report.py")
ARCHIVE = ROOT / "reports/provenance/g2lite_runs_20261017.tar.gz"
COMP = "g2lite-20261017T130319Z-pass2-ABCD0"


def archived_jsonl(tf, run, name):
    return [json.loads(line) for line in tf.extractfile(f"runs/{run}/{name}").read().splitlines() if line]


class ProgressTests(unittest.TestCase):
    def plan(self):
        return control.RunProgress(["one", "two"], [("A", None), ("C", 0.125)])

    def test_completed_plan_only_is_done(self):
        p = self.plan()
        for d in p.doc_ids:
            for m, rho in p.requests:
                p.complete(d, m, rho)
        s = p.snapshot(final=True)
        self.assertEqual((s["status"], s["planned_requests"], s["completed_requests"], s["skipped_requests"]), ("done", 4, 4, 0))
        with self.assertRaises(ValueError):
            p.complete("one", "A", None)

    def test_cache_limit_skip_is_incomplete(self):
        p = self.plan()
        p.skip_document("one", "cache_estimate_exceeds_memory_limit")
        for m, rho in p.requests:
            p.complete("two", m, rho)
        s = p.snapshot(final=True)
        self.assertEqual((s["status"], s["completed_docs"], s["skipped_docs"], s["completed_requests"], s["skipped_requests"]), ("incomplete", 1, 1, 2, 2))
        self.assertTrue(all(r["reason"] == "cache_estimate_exceeds_memory_limit" for r in s["skipped"]))

    def test_warning_stop_accounts_for_unstarted_work(self):
        p = self.plan()
        for m, rho in p.requests:
            p.complete("one", m, rho)
        p.stop("allocated_memory_above_warning_limit")
        s = p.snapshot(final=True)
        self.assertEqual((s["status"], s["completed_requests"], s["skipped_requests"], s["pending_requests"]), ("incomplete", 2, 2, 0))
        self.assertEqual({r["doc_id"] for r in s["skipped"]}, {"two"})

    def test_missing_requests_do_not_become_done(self):
        p = self.plan()
        p.complete("one", "A", None)
        s = p.snapshot(final=True)
        self.assertEqual((s["status"], s["partial_docs"], s["pending_requests"]), ("incomplete", 1, 3))

    def test_report_rejects_an_incomplete_or_mislabelled_run(self):
        for status in ("incomplete", "done"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                d = Path(td)
                p = self.plan()
                p.stop("memory_limit")
                (d / "manifest.json").write_text(json.dumps({"status": status, "completion": p.snapshot(final=True)}))
                (d / "summary.json").write_text(json.dumps({"status": status}))
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    report.load_run(d)


class HistoricalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(ARCHIVE) as tf:
            cls.baseline = archived_jsonl(tf, "g2lite-20261017T130007Z-pass1-A", "quality.jsonl")
            cls.rows = archived_jsonl(tf, COMP, "quality.jsonl")
            cls.docs = archived_jsonl(tf, COMP, "data.jsonl")
            cls.rewrites = []
            for run in ("g2lite-20261017T130551Z-pass3a-rewrites", "g2lite-20261017T130715Z-pass3b-move"):
                cls.rewrites += archived_jsonl(tf, run, "quality.jsonl")

    def test_rescoring_preserves_every_archived_answer(self):
        for r in self.baseline + self.rows + self.rewrites:
            scored = evaluation.score_answer(r["task"], r["targets"], r["text"], truncated=r["truncated"])
            for key, value in scored.items():
                self.assertEqual(value, r[key], (r["doc_id"], r["method"], key))

    def test_changed_wrong_answer_is_not_counted_as_unchanged(self):
        pairs = evaluation.rewrite_table(self.rows + self.rewrites)
        counts = evaluation.rewrite_observations(pairs)
        self.assertEqual(counts["value"]["B"]["answer_unchanged"], 7)
        self.assertEqual(counts["value"]["B"]["both_wrong"], 8)
        self.assertEqual(sum(counts["value"][m]["answer_unchanged"] for m in ("B", "C", "D0")), 23)
        exception = next(r for r in pairs if (r["doc_id"], r["method"], r["rho"]) == ("T1-2K-00-value", "B", 0.125))
        self.assertTrue(exception["answer_changed"] and exception["both_wrong"] and exception["became_id_echo"])
        self.assertFalse(exception["correctness_changed"])
        c = [r for r in pairs if r["method"] == "C"]
        self.assertEqual(len(c), 24)
        self.assertTrue(all(r["counterfactual_control"] == "confounded_uniform_conditions" for r in c))
        self.assertTrue(all(r["counterfactual_control"] == "content_responsive_indexer" for r in pairs if r["method"] == "D0"))

    def test_neighborhood_retains_target_and_every_neighbor(self):
        with tempfile.TemporaryDirectory() as td:
            comp = {"dir": Path(td), "rows": self.rows, "docs": {d["doc_id"]: d for d in self.docs}}
            for d in self.docs:
                detail = report.case_detail(comp, [], d["doc_id"], [1/32, 1/8])
                want = {str(j): [d["ids"][j], d["values"][j]] for i in d["target_indices"] for j in range(max(0, i-2), min(d["n_records"], i+3))}
                self.assertEqual(detail["records_around_targets"], want)
                for i in d["target_indices"]:
                    self.assertEqual(detail["records_around_targets"][str(i)], [d["ids"][i], d["values"][i]])

    def test_actual_budgets_and_first_generation_denominators(self):
        table = evaluation.budget_tables(self.rows)
        for cell, expected in (("T1/2K", [255, 256, 257, 258]), ("T1/8K", [1023, 1024, 1025, 1026])):
            for m in ("B", "C", "D0"):
                self.assertEqual(table[cell][f"{m}@0.125"]["k0"], expected)
        r = self.rows[1]
        b = evaluation.budget_tables([r])["T1/2K"]["B@0.03125"]
        n = r["prefix_tokens"] + r["suffix_tokens"]
        self.assertEqual(b["candidate_fraction_mean"], r["k0"]/(n-68))
        self.assertEqual(b["total_read_fraction"], [r["selected_count"]/n]*2)
        self.assertNotEqual(b["rho"], b["total_read_fraction"][0])
        self.assertEqual(report.value_range([255, 256, 257, 258]), "255–258")

    def test_no_torch_help_and_archive_regeneration(self):
        # -S omits site-packages. For the full report expose only the existing
        # NumPy package, never torch or transformers; do not install anything.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            env = {**os.environ, "PYTHONPATH": str(td / "packages"), "PYTHONDONTWRITEBYTECODE": "1"}
            cmd = [sys.executable, "-S", str(ROOT / "scripts/g2lite_report.py")]
            help_result = subprocess.run(cmd + ["--help"], cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            numpy = importlib.util.find_spec("numpy")
            if numpy is None:
                self.skipTest("NumPy absent: no-torch --help passed; NPZ regeneration tested in CPU job")
            packages = td / "packages"
            packages.mkdir()
            (packages / "numpy").symlink_to(Path(numpy.origin).parent, target_is_directory=True)
            libs = Path(numpy.origin).parent.parent / "numpy.libs"
            if libs.exists():
                (packages / "numpy.libs").symlink_to(libs, target_is_directory=True)
            probe = subprocess.run([sys.executable, "-S", "-c", "import importlib.util,numpy; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('transformers') is None"], cwd=td, env=env, text=True, capture_output=True)
            self.assertEqual(probe.returncode, 0, probe.stderr)
            with tarfile.open(ARCHIVE) as tf:
                tf.extractall(td / "input", filter="data")
            args = ["--runs-dir", str(td / "input/runs")]
            outputs = {"--out": "report.md", "--json": "summary.json", "--answers": "answers.json", "--manifest": "manifest.json", "--build-manifest": "build.json"}
            for flag, name in outputs.items():
                args += [flag, str(td / name)]
            result = subprocess.run(cmd + args, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((td / "answers.json").read_bytes(), (ROOT / "reports/g2lite_answers_mac_m4max.json").read_bytes())
            self.assertEqual((td / "manifest.json").read_bytes(), (ROOT / "reports/g2lite_manifest_mac_m4max.json").read_bytes())
            out = json.loads((td / "summary.json").read_text())
            with tarfile.open(ARCHIVE) as tf:
                frozen = json.load(tf.extractfile(f"runs/{COMP}/summary.json"))["aggregate"]
            self.assertEqual(out["comparison_methods"], frozen["methods"])
            self.assertEqual(out["paired"], frozen["paired"])
            self.assertIn("confounded_uniform_conditions", (td / "summary.json").read_text())


if __name__ == "__main__":
    unittest.main()
