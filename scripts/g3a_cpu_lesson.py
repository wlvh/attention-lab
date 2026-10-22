"""Lesson 04's three-position CPU example, using the actual index/teacher/read code.

Only 12 disposable index parameters are updated. No Qwen checkpoint is loaded,
no research configuration or saved index bank is changed, and no GPU is used.
"""
from __future__ import annotations
import copy
import math
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from attention_lab.g3a import TrainableIndexer,teacher,kl_rows
from attention_lab.indexer import project_keys,project_queries
from attention_lab.sparse import gather_attention


def run_demo():
    # Reuse the Q/K/V already shown in lesson 04. All three are candidates here;
    # this small example does not simulate the real first-4/recent-64 budget.
    q=torch.tensor([[[[1.,0.]],[[0.,1.]]]],device='cpu',requires_grad=True)
    k=torch.tensor([[[1.,0.],[0.,1.],[1.,1.]]],device='cpu',requires_grad=True)
    v=torch.tensor([[[2.,0.],[0.,2.],[1.,1.]]],device='cpu',requires_grad=True)
    original=[x.detach().clone() for x in (q,k,v)]
    head_weights,target,full=teacher(q,k,v)
    index=TrainableIndexer(groups=1,heads_per_group=2,head_dim=2,r=2,seed=1803,device='cpu')

    def loss(model):
        scores=model(q.unsqueeze(0),k.unsqueeze(0))[0,:,0]
        return kl_rows(scores,target).mean()

    def observe(model):
        with torch.no_grad():
            scores=model(q.unsqueeze(0),k.unsqueeze(0))[0,:,0]
            # Same projection functions as TrainableIndexer.forward. The implementation
            # folds 1/sqrt(r) into the query projection; do not divide a second time.
            u=project_queries(q.unsqueeze(0),(model.wq/math.sqrt(model.r)).transpose(1,2))
            z=project_keys(k.unsqueeze(0),model.wk)
            torch.testing.assert_close((u@z)[0,:,0],scores)
            positions=scores.topk(1,dim=-1).indices.unsqueeze(0)
            # Index scores select rows. The original Q/K recompute attention weights.
            output=gather_attention(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),positions)
            # Exactly one selected position and NO fixed positions: softmax([x]) = [1].
            # This example cannot demonstrate weighting among several selected values.
            selected_value=v[0,positions[0,0,0]]
            torch.testing.assert_close(output[0,0,:,0],selected_value.expand(2,-1))
            error=(output[0]-full).norm()/full.norm()
            return dict(scores=scores[0].tolist(),probabilities=scores.softmax(-1)[0].tolist(),
                        kl=float(loss(model)),positions=positions[0,0].tolist(),
                        main_output=output[0,0,:,0].tolist(),relative_output_error=float(error),
                        parameters={'wq':model.wq[0].tolist(),'wk':model.wk[0].tolist()},
                        query_projection_scaled=u[0,0,0].tolist(),index_keys_by_position=z[0,0].T.tolist(),
                        selected_main_weights=[[1.],[1.]])

    initial=observe(index)
    scaled=copy.deepcopy(index)
    with torch.no_grad():scaled.wq.mul_(2)
    scaled_result=observe(scaled)
    torch.testing.assert_close(torch.tensor(scaled_result['scores']),2*torch.tensor(initial['scores']))
    assert scaled_result['positions']==initial['positions']
    assert scaled_result['main_output']==initial['main_output']

    # Same small Adam/20-step setup as the existing G3a CPU gradient test.
    optimizer=torch.optim.Adam(index.parameters(),lr=.02)
    trace=[]
    for step in range(20):
        before=observe(index)
        optimizer.zero_grad()
        scores=index(q.unsqueeze(0),k.unsqueeze(0))[0,:,0]
        scores.retain_grad()
        current_loss=kl_rows(scores,target).mean()
        current_loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in index.parameters())
        # For this one-group KL, dL/dscore = index probability - teacher probability.
        torch.testing.assert_close(scores.grad,scores.detach().softmax(-1)-target,atol=1e-7,rtol=1e-5)
        gradients={'scores':scores.grad[0].tolist(),'wq':index.wq.grad[0].tolist(),'wk':index.wk.grad[0].tolist()}
        old_wq,old_wk=index.wq.detach().clone(),index.wk.detach().clone()
        optimizer.step()
        change={'wq':(index.wq-old_wq)[0].detach().tolist(),'wk':(index.wk-old_wk)[0].detach().tolist()}
        trace.append(dict(step=step+1,before=before,gradients=gradients,parameter_update=change,**observe(index)))
    for x,before in zip((q,k,v),original):
        assert x.grad is None
        torch.testing.assert_close(x,before)
    return dict(scope='12-parameter CPU teaching example, not a Qwen run',teacher=target[0].tolist(),
                full_main_output=full[0,:,0].tolist(),initial=initial,scaled_Wq_times_2=scaled_result,
                trained=trace[-1],trace=trace,body_features_unchanged=True,
                parameter_count=sum(p.numel() for p in index.parameters()),device='cpu',
                inputs={'query_heads':q[0,:,0].detach().tolist(),'keys':k[0].detach().tolist(),
                        'values':v[0].detach().tolist(),'concatenated_query':q[0,:,0].flatten().detach().tolist()},
                teacher_per_head=head_weights[0,:,0].tolist(),
                optimizer={'name':'Adam','learning_rate':.02,'betas':list(optimizer.defaults['betas']),
                           'epsilon':optimizer.defaults['eps'],'steps':20},
                selected_count=1,fixed_positions=[],
                limitation='One selected position without F: main weight is 1 and output equals its V; no multi-position reweighting is demonstrated.')


if __name__=='__main__':
    import hashlib,json
    result=run_demo()
    result['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'scripts/g3a_cpu_lesson.py','src/attention_lab/g3a.py','src/attention_lab/indexer.py','src/attention_lab/sparse.py')}
    result['environment']={'torch':torch.__version__,'device':'cpu','dtype':'float32'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
