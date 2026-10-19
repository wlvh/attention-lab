"""Frozen new-test scoring and actual-position audit; CPU only."""
import sys,json,io,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from g4lite_report import read_evidence,summarize


class G4Evidence(unittest.TestCase):
 def test_all_answers_regraded_and_cost_grid_complete(self):
  e=read_evidence();s=summarize(e)
  self.assertEqual(len(e['quality']),160);self.assertEqual(len(e['costs']),96)
  self.assertEqual(e['manifest']['completed'],dict(quality_requests=160,cost_sequences=4,cost_requests=96))
  self.assertEqual(e['manifest']['body_hash_before'],e['manifest']['body_hash_after'])
  self.assertLessEqual(e['budget']['used_seconds'],14400)
  for method,cells in s['paired_against_A'].items():self.assertEqual(sum(map(len,cells.values())),32)
  self.assertEqual({d['doc_index'] for d in e['docs']},set(range(32)))

 def test_actual_first_prediction_positions_have_shared_budget_and_fixed_set(self):
  try:import numpy as np
  except ImportError:self.skipTest('position bitmap audit requires numpy')
  e=read_evidence();prefix=e['index']['formal_run']+'/'
  by={(r['doc_id'],r['method']):r for r in e['quality']}
  for doc in e['docs']:
   sparse_counts=[]
   for method in e['manifest']['config']['quality_methods']:
    row=by[doc['doc_id'],method]
    with np.load(io.BytesIO(e['files'][prefix+f'positions-{doc["doc_id"]}-{method}.npz']),allow_pickle=False) as z:
     bits=np.unpackbits(z['bitsets'],axis=-1);legal=int(z['legal'])
     self.assertEqual(legal,doc['prompt_token_count']);self.assertFalse(bits[:,:,legal:].any())
     np.testing.assert_array_equal(bits.sum(-1),np.broadcast_to(z['counts'][:,None],(28,8)))
     self.assertTrue(bits[:,:,:4].all() and bits[:,:,legal-64:legal].all())
     count=int(z['counts'][0]);self.assertEqual(count,row['first_prediction_read_count'])
     if method=='A':self.assertEqual(count,legal)
     else:
      self.assertEqual(count,row['k0']+68);sparse_counts.append(count)
     self.assertAlmostEqual(row['total_read_ratio'],count/legal)
   self.assertEqual(len(set(sparse_counts)),1)

 def test_saved_token_ids_decode_to_recorded_raw_text_when_local_tokenizer_exists(self):
  try:
   from huggingface_hub import try_to_load_from_cache
   from transformers import AutoTokenizer
  except ImportError:self.skipTest('local tokenizer dependencies unavailable')
  e=read_evidence();cfg=e['manifest']['config']
  if not isinstance(try_to_load_from_cache(cfg['model_id'],'tokenizer.json',revision=cfg['revision']),str):self.skipTest('pinned local tokenizer unavailable; no download')
  tokenizer=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
  for row in e['quality']:
   self.assertEqual(tokenizer.decode(row['token_ids'],skip_special_tokens=False),row['raw_text'])
   self.assertEqual(tokenizer.decode(row['token_ids'],skip_special_tokens=True),row['answer_text'])
