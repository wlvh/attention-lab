"""Standard-library CPU tests for the shared budget and execution gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("campaign",ROOT/"src/attention_lab/campaign.py")
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def test_failed_run_charges_same_ledger_and_no_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"runs").mkdir()
            (root/"runs/g3a-budget-20261018.json").write_text(json.dumps({"used_seconds":313.8}))
            cfg=dict(campaign="test",budget_seconds=60,minimum_free_disk_bytes=0)
            clock=[0.0]
            with self.assertRaises(RuntimeError):
                with c.CampaignBudget(root,cfg,"first",clock=lambda:clock[0]) as b:
                    clock[0]=20
                    raise RuntimeError("simulated failure")
            ledger=json.loads(b.path.read_text());self.assertEqual(ledger["used_seconds"],20)
            self.assertIsNone(ledger["active_pid"]);self.assertEqual(ledger["runs"][0]["status"],"incomplete")
            with c.CampaignBudget(root,cfg,"second",clock=lambda:clock[0]) as b:
                self.assertEqual(b.remaining,40)
                with self.assertRaises(c.BudgetStop):b.check(estimated_next=35,reserve=10)
                clock[0]=30
            self.assertEqual(json.loads(b.path.read_text())["used_seconds"],30)
            with self.assertRaises(RuntimeError):
                with c.CampaignBudget(root,dict(cfg,budget_seconds=100),"reset"):pass
            self.assertEqual(json.loads((root/"runs/g3a-budget-20261018.json").read_text())["used_seconds"],313.8)

    def test_active_run_refused_and_budget_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"runs").mkdir()
            (root/"runs/g3a-budget-20261018.json").write_text('{"used_seconds":313.8}')
            cfg=dict(campaign="test",budget_seconds=10,minimum_free_disk_bytes=0)
            clock=[0.0]
            with c.CampaignBudget(root,cfg,"active",clock=lambda:clock[0]) as b:
                with self.assertRaises(RuntimeError):
                    with c.CampaignBudget(root,cfg,"collision"):pass
                clock[0]=10
                with self.assertRaises(c.BudgetStop):b.check()
            with self.assertRaises(c.BudgetStop):
                with c.CampaignBudget(root,cfg,"third"):pass

    def test_execution_gate_ignores_git_and_rejects_missing_or_changed_inputs(self):
        identity={"sources":{"runner.py":"a"},"configs":{"data.json":"b"},"versions":{"torch":"x"}}
        manifest={"status":"done","mode":"preflight","planned":{"layers":2},"completed":{"layers":2},"execution":identity,"git":{"commit":"old"}}
        c.require_preflight(manifest,identity)
        c.require_preflight(dict(manifest,git={"commit":"doc-only"}),identity)
        for bad in ({},dict(identity,sources={"runner.py":"changed"}),dict(identity,configs={}),dict(identity,versions={})):
            with self.assertRaises(RuntimeError):c.require_preflight(manifest,bad)
        with self.assertRaises(RuntimeError):c.require_preflight(dict(manifest,completed={"layers":1}),identity)


if __name__=="__main__":unittest.main()
