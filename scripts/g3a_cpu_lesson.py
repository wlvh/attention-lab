"""Lesson 04's three-position CPU example, using the actual index/teacher/read code.

Only 12 disposable index parameters are updated. No Qwen checkpoint is loaded,
no research configuration or saved index bank is changed, and no GPU is used.
"""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from attention_lab.g3a import TrainableIndexer,teacher,kl_rows
from attention_lab.sparse import gather_attention


def run_demo():
    # Reuse the Q/K/V already shown in lesson 04. All three are candidates here;
    # this small example does not simulate the real first-4/recent-64 budget.
    q=torch.tensor([[[[1.,0.]],[[0.,1.]]]],device='cpu',requires_grad=True)
    k=torch.tensor([[[1.,0.],[0.,1.],[1.,1.]]],device='cpu',requires_grad=True)
    v=torch.tensor([[[2.,0.],[0.,2.],[1.,1.]]],device='cpu',requires_grad=True)
    original=[x.detach().clone() for x in (q,k,v)]
    _,target,full=teacher(q,k,v)
    index=TrainableIndexer(groups=1,heads_per_group=2,head_dim=2,r=2,seed=1803,device='cpu')

    def loss(model):
        scores=model(q.unsqueeze(0),k.unsqueeze(0))[0,:,0]
        return kl_rows(scores,target).mean()

    def observe(model):
        with torch.no_grad():
            scores=model(q.unsqueeze(0),k.unsqueeze(0))[0,:,0]
            positions=scores.topk(1,dim=-1).indices.unsqueeze(0)
            # Index scores select rows. The original Q/K recompute attention weights.
            output=gather_attention(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),positions)
            error=(output[0]-full).norm()/full.norm()
            return dict(scores=scores[0].tolist(),probabilities=scores.softmax(-1)[0].tolist(),
                        kl=float(loss(model)),positions=positions[0,0].tolist(),
                        main_output=output[0,0,:,0].tolist(),relative_output_error=float(error))

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
        optimizer.zero_grad();current_loss=loss(index);current_loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in index.parameters())
        optimizer.step();trace.append(dict(step=step+1,**observe(index)))
    for x,before in zip((q,k,v),original):
        assert x.grad is None
        torch.testing.assert_close(x,before)
    return dict(scope='12-parameter CPU teaching example, not a Qwen run',teacher=target[0].tolist(),
                full_main_output=full[0,:,0].tolist(),initial=initial,scaled_Wq_times_2=scaled_result,
                trained=trace[-1],trace=trace,body_features_unchanged=True,
                parameter_count=sum(p.numel() for p in index.parameters()),device='cpu')


if __name__=='__main__':
    import json
    print(json.dumps(run_demo(),ensure_ascii=False,indent=2))
