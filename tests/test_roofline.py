import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('attention_roofline',ROOT/'scripts/roofline.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
M=m.MODELS['Qwen3-4B']
class RooflineTests(unittest.TestCase):
    def test_no_bandwidth_no_latency(self):
        self.assertIsNone(m.estimate(M,8192)['saved_s1_scenario_ms'])
    def test_index_parameter_counts(self):
        self.assertEqual(m.estimate(m.MODELS['Qwen3-0.6B'],8192)['index_parameter_count'],2752512)
        self.assertEqual(m.estimate(M,8192)['index_parameter_count'],5898240)
    def test_zero_rho(self):
        x=m.estimate(M,512,rho=0);self.assertEqual((x['k'],x['selected_count']),(0,68))
    def test_full_clamped(self):
        x=m.estimate(M,512,rho=1);self.assertEqual(x['selected_count'],512);self.assertTrue(x['degenerate_full_selection'])
    def test_short_fixed_overlap(self):
        x=m.estimate(M,4);self.assertEqual((x['fixed_count'],x['candidate_count'],x['k']),(4,0,0))
    def test_s2_adds_two_selected_passes(self):
        x=m.estimate(M,8192);self.assertEqual(x['d_s2_bytes']-x['d_s1_bytes'],2*x['selected_kv_bytes'])
    def test_candidate_vs_all_scan(self):
        a=m.estimate(M,8192,scan_scope='all');b=m.estimate(M,8192)
        self.assertEqual(a['index_stored_bytes'],b['index_stored_bytes']);self.assertGreater(a['index_read_bytes'],b['index_read_bytes'])
    def test_same_precision_ratio(self):
        x=m.estimate(M,32768,scan_scope='all');self.assertEqual(x['index_read_bytes']/x['dense_kv_bytes'],0.125)
    def test_invalid_values(self):
        for kw in ({'rho':-1},{'rho':float('nan')},{'rho':1.1},{'bandwidth':0},{'r':0},{'element_bytes':0}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):m.estimate(M,8192,**kw)
    def test_fp32_payload(self):
        a=m.estimate(M,8192);b=m.estimate(M,8192,element_bytes=4);self.assertEqual(b['dense_kv_bytes'],2*a['dense_kv_bytes'])
    def test_original_scenario_4b32k(self):
        x=m.estimate(M,32768,scan_scope='all',bandwidth=716.8e9)
        self.assertAlmostEqual(x['saved_s1_scenario_ms'],5.67,places=2);self.assertAlmostEqual(x['saved_s2_scenario_ms'],5.22,places=2)
    def test_cli_json_and_zero(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/roofline.py'),'--rho','0','--format','json'],capture_output=True,text=True,check=True)
        d=json.loads(r.stdout);self.assertEqual(len(d['rows']),9);self.assertTrue(all(x['k']==0 for x in d['rows']))
if __name__=='__main__':unittest.main()
