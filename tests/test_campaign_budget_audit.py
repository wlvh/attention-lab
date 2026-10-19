"""Historical accounting is checked without torch, MPS or mutable run files."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_campaign_budget import audit,timeline


class CampaignHistoryTests(unittest.TestCase):
 def fixture(self):
  ledger={'active_pid':None,'used_seconds':30.,'runs':[{'run_id':'A','status':'done','elapsed_seconds':10.},{'run_id':'B','status':'incomplete','elapsed_seconds':20.}]}
  manifests={'A':{'status':'done','utc_start':'2026-10-18T10:00:00+00:00','utc_end':'2026-10-18T10:00:10+00:00','elapsed_seconds':10.001},
             'B':{'status':'incomplete','utc_start':'2026-10-18T10:00:20+00:00','utc_end':'2026-10-18T10:00:40+00:00','elapsed_seconds':20.001}}
  return ledger,manifests

 def test_preserves_different_timer_scopes(self):
  ledger,manifests=self.fixture();r=timeline(ledger,manifests)
  self.assertEqual(r['ledger_total_seconds'],30.)
  self.assertAlmostEqual(r['outer_minus_ledger_total_seconds'],.002)
  self.assertEqual(r['minimum_conservative_gap_seconds'],9.)

 def test_lost_charge_duplicate_missing_and_overlap_are_rejected(self):
  ledger,manifests=self.fixture()
  variants=[]
  bad=copy.deepcopy(ledger);bad['used_seconds']=20.;variants.append((bad,manifests))
  bad=copy.deepcopy(ledger);bad['runs'][1]['run_id']='A';variants.append((bad,manifests))
  variants.append((ledger,{'A':manifests['A']}))
  bad=copy.deepcopy(manifests);bad['B']['utc_start']='2026-10-18T10:00:05+00:00';bad['B']['utc_end']='2026-10-18T10:00:25+00:00';variants.append((ledger,bad))
  for args in variants:
   with self.subTest(args=args),self.assertRaises(ValueError):timeline(*args)

 def test_existing_five_runs_and_source_versions(self):
  result=audit()
  self.assertEqual(len(result['runs']),5)
  self.assertEqual(result['ledger_total_seconds'],3897.345024082926)
  self.assertEqual(result['limit_seconds'],14400)
  self.assertEqual(sum(x['status']=='incomplete' for x in result['runs']),1)
  self.assertTrue(result['no_recorded_overlap'])
  self.assertGreater(result['minimum_conservative_gap_seconds'],0)
  self.assertEqual(len({x['campaign_source_sha256'] for x in result['as_run_sources'].values()}),1)
  self.assertNotEqual(result['current_campaign_source_sha256'],next(iter(result['as_run_sources'].values()))['campaign_source_sha256'])
