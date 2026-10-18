"""Read-only archived evidence checks; standard-library tests run in bootstrap CI."""
import copy
import importlib.util
import math
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g3a_report", ROOT / "scripts/g3a_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@unittest.skipUnless(report.ARCHIVE.exists(), "G3a formal archive not generated yet")
class EvidenceChecks(unittest.TestCase):
    def test_hashes_counts_full_source_and_frozen_body(self):
        e = report.read_evidence()
        s = report.summarize(e)
        self.assertEqual(s["rows"], 64 * 4 * 4)
        self.assertEqual(e["manifest"]["body_hash_before"], e["manifest"]["body_hash_after_extraction"])
        self.assertEqual(e["manifest"]["trainable_parameters"], 98304)
        self.assertTrue(e["manifest"]["body_released_before_training"])
        self.assertEqual(set(e["manifest"]["optimizer_parameters"]), {"wq", "wk"})
        self.assertLess(e["budget"]["used_seconds"], 7200)
        self.assertEqual(e["manifest"]["data_isolation"]["overlap"], 0)
        self.assertEqual(len(e["manifest"]["feature_files"]), 64)

    def test_teacher_identity_R_optimal_coverage_and_independent_doc_mean(self):
        e = report.read_evidence()
        s = report.summarize(e)
        rows = e["records"]
        for r in rows:
            if r["method"] == "C":
                for kl, entropy in zip(r["kl"], r["teacher_entropy"]):
                    self.assertAlmostEqual(kl, math.log(r["candidate_count"]) - entropy, places=4)
        groups = {}
        for r in rows:
            groups.setdefault((r["doc_id"], r["query_index"]), {})[r["method"]] = r
        for methods in groups.values():
            for method, r in methods.items():
                self.assertEqual(r["candidate_mass"], methods["R"]["candidate_mass"])
                self.assertEqual(r["selected_count"], methods["R"]["selected_count"])
                for value, upper in zip(r["coverage"], methods["R"]["coverage"]):
                    self.assertLessEqual(value, upper + 1e-6)
        for method in report.METHODS:
            per_doc = []
            for doc in e["docs"]:
                if doc["split"] != "heldout":
                    continue
                values = [v for r in rows if r["doc_id"] == doc["doc_id"] and r["method"] == method
                          for v, eligible in zip(r["coverage"], r["eligible"]) if eligible]
                per_doc.append(sum(values) / len(values))
            self.assertAlmostEqual(sum(per_doc) / 16, s["aggregate"]["heldout"][method]["coverage"], places=12)

    def test_reject_missing_duplicate_rows_and_changed_split(self):
        e = report.read_evidence()
        for records in (e["records"][:-1], e["records"] + [e["records"][0]]):
            with self.assertRaises(ValueError):
                report.summarize(dict(e, records=records))
        damaged = copy.deepcopy(e)
        damaged["docs"][0]["split"] = "heldout"
        with self.assertRaises(ValueError):
            report.summarize(damaged)
