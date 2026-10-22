"""Independent checks for the small teaching trace, without model checkpoints or MPS."""
import copy,importlib.util,math,unittest
from pathlib import Path
try:
 import torch
except ImportError as error:
 raise unittest.SkipTest('CPU teaching checks require torch') from error
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cpu_lesson',ROOT/'scripts/g3a_cpu_lesson.py')
lesson=importlib.util.module_from_spec(spec);spec.loader.exec_module(lesson)


class TeachingTraceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.result=lesson.run_demo()

 def test_first_gradient_matches_independent_finite_difference_and_adam(self):
  r=self.result;first=r['trace'][0];initial=r['initial']
  query=r['inputs']['concatenated_query'];keys=r['inputs']['keys'];target=r['teacher']
  def independent_kl(wq,wk):
   u=[sum(a*b for a,b in zip(row,query))/math.sqrt(2) for row in wq]
   z=[[sum(a*b for a,b in zip(row,key)) for row in wk] for key in keys]
   scores=[sum(a*b for a,b in zip(u,key)) for key in z]
   ex=[math.exp(x-max(scores)) for x in scores];prob=[x/sum(ex) for x in ex]
   return sum(p*math.log(p/q) for p,q in zip(target,prob))
  wq=copy.deepcopy(initial['parameters']['wq']);wk=initial['parameters']['wk'];eps=1e-5
  wq[0][0]+=eps;high=independent_kl(wq,wk)
  wq[0][0]-=2*eps;low=independent_kl(wq,wk)
  self.assertAlmostEqual(first['gradients']['wq'][0][0],(high-low)/(2*eps),delta=1e-9)
  for name in ('wq','wk'):
   for i,row in enumerate(first['gradients'][name]):
    for j,g in enumerate(row):
     expected=-.02*g/(abs(g)+r['optimizer']['epsilon'])
     self.assertAlmostEqual(first['parameter_update'][name][i][j],expected,delta=1e-8)
     self.assertAlmostEqual(first['parameters'][name][i][j]-first['before']['parameters'][name][i][j],expected,delta=1e-8)

 def test_positive_scaling_preserves_selection_and_single_value_read(self):
  r=self.result;a,b=r['initial'],r['scaled_Wq_times_2']
  self.assertEqual(r['fixed_positions'],[]);self.assertEqual(r['selected_count'],1)
  self.assertEqual(a['positions'],b['positions']);self.assertEqual(a['main_output'],b['main_output'])
  self.assertNotEqual(a['probabilities'],b['probabilities'])
  for x,y in zip(a['scores'],b['scores']):self.assertAlmostEqual(2*x,y,delta=1e-10)
  for row in [a,b,*r['trace']]:
   value=r['inputs']['values'][row['positions'][0]]
   self.assertEqual(row['main_output'],[value,value])
   self.assertEqual(row['selected_main_weights'],[[1.],[1.]])

 def test_zero_query_coordinates_have_zero_gradient_and_inputs_stay_frozen(self):
  r=self.result;self.assertEqual(r['parameter_count'],12);self.assertTrue(r['body_features_unchanged'])
  self.assertEqual(len(r['trace']),20)
  for step in r['trace']:
   for gradient,change in zip(step['gradients']['wq'],step['parameter_update']['wq']):
    self.assertEqual(gradient[1:3],[0.,0.]);self.assertEqual(change[1:3],[0.,0.])
    self.assertEqual(gradient[0],gradient[3])
  self.assertLess(r['trained']['kl'],r['initial']['kl'])
