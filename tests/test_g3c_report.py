"""Saved G3c source/data and small CPU vector audits; no model invocation."""
import sys,json,unittest,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from g3c_report import read_evidence,summarize


class G3cEvidence(unittest.TestCase):
    def test_completion_and_isolation_evidence(self):
        e=read_evidence();s=summarize(e)
        self.assertEqual(e['manifest']['body_hash_before'],e['manifest']['body_hash_after'])
        self.assertEqual(len(s['full_selection']),2)
        self.assertEqual(len(s['failed_runs']),1)
        for p in s['points'].values():
            self.assertEqual(p['D-one']['q'][:14],[0]*14)
        self.assertEqual(e['manifest']['completed'],dict(sequence_generations=4,correctness_requests=2,diagnostic_requests=16))

    def test_saved_vectors_independently_reproduce_state_differences(self):
        try:import numpy as np
        except ImportError:self.skipTest('saved vector audit requires numpy')
        e=read_evidence();prefix=e['index']['formal_run']+'/'
        doc=e['docs'][0]
        for step in [len(doc['suffix_ids'])-1,len(doc['suffix_ids'])+15]:
            with np.load(io.BytesIO(e['files'][prefix+f'audit-{doc["doc_id"]}-A-step{step}.npz']),allow_pickle=False) as z:
                ref={k:z[k].astype(float) for k in z.files}
            for method in ('D0','D-one','D'):
                row=next(x for x in e['rows'] if x['method']==method and x['step']==step and x['doc_id']==doc['doc_id'])
                with np.load(io.BytesIO(e['files'][prefix+f'audit-{doc["doc_id"]}-{method}-step{step}.npz']),allow_pickle=False) as z:
                    for kind in ('q','k','attention_output','logits'):
                        a,b=z[kind].astype(float),ref[kind]
                        if kind=='logits':a,b=a[None],b[None]
                        delta=a-b
                        for key,v in [('max_abs',abs(delta).max(-1)),('rmse',np.sqrt((delta**2).mean(-1))),('relative_l2',np.linalg.norm(delta,axis=-1)/np.maximum(np.linalg.norm(b,axis=-1),1e-12))]:
                            # Reuse G3a's 1e-4 absolute metric-audit tolerance; logits norms
                            # reduce 151936 float32 values, unlike small feature vectors.
                            np.testing.assert_allclose(row[kind][key],v,rtol=0,atol=1e-4)
        for doc in e['docs']:
            for step in [len(doc['suffix_ids'])-1,len(doc['suffix_ids'])+15]:
                sets=[]
                for method in ('D0','D'):
                    with np.load(io.BytesIO(e['files'][prefix+f'positions-{doc["doc_id"]}-{method}-step{step}.npz']),allow_pickle=False) as z:
                        bits=np.unpackbits(z['bitsets'],axis=-1);legal=int(z['legal'])
                        self.assertFalse(bits[:,:,legal:].any())
                        np.testing.assert_array_equal(bits.sum(-1),np.broadcast_to(z['counts'][:,None],(28,8)))
                        sets.append(z['counts'].copy())
                np.testing.assert_array_equal(*sets)
