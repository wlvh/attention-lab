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



# Spawn fresh OS processes: no torch import, GPU, sleep-based scheduling, or real run ledger.
def _budget_worker(root, run_id, seconds, start, read_release, exit_release, events,
                   gate_read=False, fail=False):
    from unittest.mock import patch
    entered = False
    clock = [0.0]
    cfg = dict(campaign="race", budget_seconds=100, minimum_free_disk_bytes=0)
    read_text = Path.read_text

    def controlled_read(path, *args, **kwargs):
        value = read_text(path, *args, **kwargs)
        if gate_read and path.name in ("race-budget.json", "g3a-budget-20261018.json"):
            events.put(("read", run_id))
            if not read_release.wait(15):
                raise RuntimeError("test read rendezvous timed out")
        return value

    try:
        if not start.wait(15):
            raise RuntimeError("test start rendezvous timed out")
        with patch.object(Path, "read_text", controlled_read):
            with c.CampaignBudget(root, cfg, run_id, clock=lambda: clock[0]):
                entered = True
                events.put(("entered", run_id))
                if not exit_release.wait(15):
                    raise RuntimeError("test exit rendezvous timed out")
                clock[0] = seconds
                if fail:
                    raise RuntimeError("synthetic worker failure")
        events.put(("done", run_id))
    except BaseException as exc:
        events.put(("failed" if entered else "rejected", run_id, str(exc)))


class CampaignProcessTests(unittest.TestCase):
    def setUp(self):
        import multiprocessing
        self.ctx = multiprocessing.get_context("spawn")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "runs").mkdir()
        (self.root / "runs/g3a-budget-20261018.json").write_text('{"used_seconds":313.8}')
        self.cfg = dict(campaign="race", budget_seconds=100, minimum_free_disk_bytes=0)
        self.path = self.root / "runs/race-budget.json"
        self.processes = []
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        for process in self.processes:
            process.join(5)
            if process.is_alive():
                process.terminate()  # only this test's own CPU child
                process.join(5)
        self.temp.cleanup()

    def launch(self, run_id, seconds, *, gate_read=False, fail=False, controls=None):
        if controls is None:
            controls = [self.ctx.Event() for _ in range(3)] + [self.ctx.Queue()]
            for event in controls[:3]:
                event.set()
        start, read_release, exit_release, events = controls
        self.release_events.extend(controls[:3])
        process = self.ctx.Process(target=_budget_worker,
                                   args=(self.root, run_id, seconds, start, read_release,
                                         exit_release, events, gate_read, fail))
        self.processes.append(process)
        process.start()
        return process, events

    def test_simultaneous_first_creation_is_exclusive_before_read(self):
        self.assert_exclusive_read_window()

    def test_simultaneous_existing_ledger_read_is_exclusive(self):
        with c.CampaignBudget(self.root, self.cfg, "existing", clock=lambda: 0):
            pass
        self.assert_exclusive_read_window()

    def assert_exclusive_read_window(self):
        controls = [self.ctx.Event() for _ in range(3)] + [self.ctx.Queue()]
        a, events = self.launch("A", 10, gate_read=True, controls=controls)
        b, _ = self.launch("B", 20, gate_read=True, controls=controls)
        controls[0].set()
        # Winner is paused AFTER reading an idle ledger but BEFORE registration.
        # The other process must already be rejected; the old code instead emits two reads.
        messages = [events.get(timeout=15), events.get(timeout=15)]
        self.assertEqual(sorted(m[0] for m in messages), ["read", "rejected"])
        winner = next(m[1] for m in messages if m[0] == "read")
        self.assertIn("budget lock", next(m[2] for m in messages if m[0] == "rejected"))
        controls[1].set()
        self.assertEqual(events.get(timeout=15), ("entered", winner))
        controls[2].set()
        self.assertEqual(events.get(timeout=15), ("done", winner))
        for process in (a, b):
            process.join(15)
            self.assertEqual(process.exitcode, 0)
        ledger = json.loads(self.path.read_text())
        self.assertEqual(ledger["used_seconds"], 10 if winner == "A" else 20)
        self.assertEqual([r["run_id"] for r in ledger["runs"] if r["run_id"] != "existing"], [winner])
        self.assertIsNone(ledger["active_pid"])

    def test_sequential_processes_keep_both_charges_after_worker_failure(self):
        for run_id, seconds, fail in (("A", 10, False), ("B", 20, True)):
            process, events = self.launch(run_id, seconds, fail=fail)
            self.assertEqual(events.get(timeout=15), ("entered", run_id))
            self.assertEqual(events.get(timeout=15)[0], "failed" if fail else "done")
            process.join(15)
            self.assertEqual(process.exitcode, 0)
        ledger = json.loads(self.path.read_text())
        self.assertEqual(ledger["used_seconds"], 30)
        self.assertEqual([r["run_id"] for r in ledger["runs"]], ["A", "B"])
        self.assertEqual([r["status"] for r in ledger["runs"]], ["done", "incomplete"])
        self.assertEqual(ledger["runs"][1]["reason"], "synthetic worker failure")
        self.assertIsNone(ledger["active_pid"])
        # A new OS process can acquire the lock after the exception path.
        process, events = self.launch("C", 0)
        self.assertEqual(events.get(timeout=15), ("entered", "C"))
        self.assertEqual(events.get(timeout=15), ("done", "C"))
        process.join(15)
        self.assertEqual(json.loads(self.path.read_text())["used_seconds"], 30)

    def test_lock_spans_the_final_ledger_write(self):
        from unittest.mock import patch
        original = c.write_json

        def write_then_contend(path, value):
            original(path, value)
            if value.get("active_pid") is None:
                process, events = self.launch("during_exit", 20)
                message = events.get(timeout=15)
                self.assertEqual(message[0], "rejected")
                self.assertIn("budget lock", message[2])
                process.join(15)
                self.assertEqual(process.exitcode, 0)

        clock = [0.0]
        with patch.object(c, "write_json", write_then_contend):
            with c.CampaignBudget(self.root, self.cfg, "owner", clock=lambda: clock[0]):
                clock[0] = 10
        self.assertEqual(json.loads(self.path.read_text())["used_seconds"], 10)

    def test_failed_entry_and_failed_write_release_os_lock(self):
        from unittest.mock import patch
        with patch.object(c, "write_json", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                with c.CampaignBudget(self.root, self.cfg, "failed_entry"):
                    self.fail("entry must fail")
        # Entry failed before a ledger existed; a fresh process can proceed.
        process, events = self.launch("after_entry_failure", 0)
        self.assertEqual(events.get(timeout=15), ("entered", "after_entry_failure"))
        self.assertEqual(events.get(timeout=15), ("done", "after_entry_failure"))
        process.join(15)
        budget = c.CampaignBudget(self.root, self.cfg, "failed_exit", clock=lambda: 0)
        budget.__enter__()
        with patch.object(c, "write_json", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                budget.__exit__(None, None, None)
        self.assert_released_but_unreconciled(budget.lock_path)

    def test_forced_process_exit_releases_lock_but_does_not_erase_unfinished_run(self):
        controls = [self.ctx.Event() for _ in range(3)] + [self.ctx.Queue()]
        controls[0].set(); controls[1].set()
        process, events = self.launch("interrupted", 10, controls=controls)
        self.assertEqual(events.get(timeout=15), ("entered", "interrupted"))
        # A killed child can leave multiprocessing.Event's internal condition locked.
        # Do not touch those test rendezvous objects again after termination.
        self.release_events = [e for e in self.release_events
                               if all(e is not control for control in controls[:3])]
        process.terminate()  # owned CPU child only; simulates an uncatchable interruption
        process.join(15)
        self.assertIsNotNone(process.exitcode)
        self.assert_released_but_unreconciled(self.path.with_suffix(".json.lock"))

    def assert_released_but_unreconciled(self, lock_path):
        import fcntl
        with lock_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "reconcile PID/ledger"):
            with c.CampaignBudget(self.root, self.cfg, "must_not_start"):
                self.fail("unfinished accounting must block a fresh run")
        self.assertEqual(before, self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
