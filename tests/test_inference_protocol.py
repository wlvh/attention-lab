"""Small CPU protocol fixtures: free outputs differ from forced cost inputs."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
try:
 import torch,pytest
 import transformers
except ImportError as error:
 raise unittest.SkipTest('inference protocol requires torch/pytest/transformers') from error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from attention_lab.inference import free_generate,fixed_greedy_tokens,question_forward


class FakeRunner:
 def __init__(self):self.cache=SimpleNamespace(device='cpu');self.calls=[]
 def step(self,ids):
  self.calls.append(('step',ids.tolist()))
  return torch.tensor([[[0.,1.,0.]]])
 def forward(self,ids,*,phase):
  self.calls.append((phase,ids.tolist()))
  return torch.tensor([[[0.,1.,0.]]])


def test_eos_stops_free_generation_before_another_forward():
 r=FakeRunner();tokens,stopped=free_generate(r,torch.tensor([[[0.,0.,1.]]]),max_new=64,eos={2})
 assert tokens==[2] and stopped and r.calls==[]


def test_maximum_generation_has_one_fewer_forward_and_preserves_raw_tokens():
 r=FakeRunner();tokens,stopped=free_generate(r,torch.tensor([[[1.,0.,0.]]]),max_new=3,eos={2})
 assert tokens==[0,1,1] and not stopped and len(r.calls)==2


def test_controlled_sequence_can_include_eos_but_is_not_a_free_answer():
 r=FakeRunner();tokens=fixed_greedy_tokens(r,torch.tensor([[[0.,0.,1.]]]),4)
 assert tokens==[2,1,1,1] and len(r.calls)==3


def test_known_question_batch_vs_step_protocol():
 a,b=FakeRunner(),FakeRunner();ids=torch.tensor([[1,2,3]])
 question_forward(a,ids);question_forward(b,ids,batch=True)
 assert len(a.calls)==3 and all(c[0]=='step' for c in a.calls)
 assert b.calls==[('prefill',[[1,2,3]])]
