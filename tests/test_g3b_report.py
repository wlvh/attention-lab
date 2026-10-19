"""Archived G3b coverage/source tests; no model execution."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("g3b_report",ROOT/"scripts/g3b_report.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


@unittest.skipUnless(r.ARCHIVE.exists(),"G3b formal archive not generated yet")
class G3bEvidenceTests(unittest.TestCase):
    def test_complete_bank_raw_aggregation_and_body_unchanged(self):
        e=r.read_evidence();s=r.summarize(e)
        self.assertEqual(len(s["layers"]),28)
        self.assertEqual([x["layer_index"] for x in s["layers"] if x["origin"]=="reused_G3a"],[13])
        self.assertEqual(e["manifest"]["body_hash_before"],e["manifest"]["body_hash_after"])
        self.assertLess(e["budget"]["used_seconds"],14400)
        for layer in s["layers"]:
            self.assertEqual({x["split"] for x in layer["by_document"]},{"train","development"})
            self.assertEqual(len(layer["by_document"]),64*4)

    def test_all_methods_have_common_budget_and_R_bounds_coverage(self):
        import json
        e=r.read_evidence();prefix=e["index"]["formal_run"]+"/"
        for layer in range(28):
            rows=[json.loads(x) for x in e["files"][prefix+f"layer-{layer:02d}/metrics.jsonl"].splitlines()]
            grouped={}
            for row in rows:grouped.setdefault((row["doc_id"],row["query_index"]),{})[row["method"]]=row
            for methods in grouped.values():
                for row in methods.values():
                    self.assertEqual((row["k"],row["selected_count"]),(methods["R"]["k"],methods["R"]["selected_count"]))
                    for got,bound in zip(row["coverage"],methods["R"]["coverage"]):self.assertLessEqual(got,bound+1e-6)


if __name__=="__main__":unittest.main()
