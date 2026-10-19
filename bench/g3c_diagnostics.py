#!/usr/bin/env python3
"""G3c fixed-token state diagnostics. No answer scores and no trajectory retraining."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from attention_lab import runinfo
from attention_lab.campaign import CampaignBudget,ResourceGuard,digest,execution_identity,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank-run',type=Path,required=True);args=p.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';runinfo.refuse_unsafe_env()
    import numpy as np
    import torch
    import transformers
    from attention_lab.qwen_wrap import load_model,wrap_model,geometry_of
    from attention_lab.trained import load_bank
    from attention_lab.inference import (runner_for,question_forward,fixed_greedy_tokens,prefix_digest,
                                         StepCapture,state_difference,selection_bitsets)
    from attention_lab.g3a import body_digest
    cfg_path=ROOT/'configs/mac_g3bg4lite.json';cfg=json.loads(cfg_path.read_text())
    git=runinfo.git_info(ROOT)
    if git['dirty_files']:raise RuntimeError('commit execution/config changes before real diagnostics')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    metadata=json.loads((ROOT/'reports/g3b_parameters.json').read_text())
    if metadata['run_id']!=args.bank_run.name or (metadata['model_id'],metadata['revision'])!=(cfg['model_id'],cfg['revision']):
        raise RuntimeError('bank/run/model identity mismatch')
    parent=json.loads((args.bank_run/'manifest.json').read_text())
    if parent['status']!='done' or parent['completed']!=parent['planned']:raise RuntimeError('G3b incomplete')
    docs=[json.loads(s) for s in (args.bank_run/'data.jsonl').read_text().splitlines()]
    docs=[d for d in docs if d['doc_index'] in cfg['state_diagnostic_doc_indices']]
    if len(docs)!=4 or any(d['split']!='development' for d in docs):raise RuntimeError('diagnostic data selection differs')
    execution=execution_identity(ROOT,cfg_path,'bench/g3c_diagnostics.py',{'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__})
    execution['configs']['configs/mac_gate1b.json']=digest(ROOT/'configs/mac_gate1b.json')
    tol=json.loads((ROOT/'configs/mac_gate1b.json').read_text())['numerics']['tolerances']['bfloat16']
    run_id='g3c-'+runinfo.utc_compact();out=ROOT/'runs'/run_id;out.mkdir(parents=True)
    planned=dict(sequence_generations=4,correctness_requests=2,diagnostic_requests=16)
    completed={k:0 for k in planned}
    manifest=dict(run_id=run_id,stage='G3c',status='running',git=git,execution=execution,config=cfg,
                  bank=metadata,bank_manifest_sha256=digest(args.bank_run/'manifest.json'),planned=planned,completed=completed,
                  utc_start=runinfo.utc_now(),precision='bf16 body, fp32 index; captured-vector comparisons on CPU float32',
                  sequence_rule='A greedy for exactly 16 continuation tokens, ignores EOS; controlled inputs only, never scored as answers',
                  full_selection_tolerance=tol)
    write_json(out/'manifest.json',manifest);write_json(out/'config.json',cfg)
    (out/'data.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
    reason=None;begin=time.monotonic();capture=None
    try:
        with CampaignBudget(ROOT,cfg,run_id) as budget:
            guard=ResourceGuard(budget,torch);guard.check(record=True,estimated_next=600,reserve=1800)
            model,tok,info=load_model(cfg['model_id'],cfg['revision'],dtype=torch.bfloat16,device='mps');holder=wrap_model(model)
            manifest['model']=info;manifest['body_hash_before']=body_digest(model)
            bank=load_bank(args.bank_run,metadata,geometry_of(model),device='mps')
            initial=load_bank(args.bank_run,metadata,geometry_of(model),device='mps',initial=True)
            manifest['parameter_identities']={'D':bank.identity,'D0':initial.identity}
            capture=StepCapture(model)
            sequences=[];checks=[]
            def new_path(doc,method,all_select=False):
                cap=doc['prompt_token_count']+cfg['max_new_tokens']
                runner=runner_for(model,holder,cap,method,bank,initial,doc=doc,uniform_seed=cfg['uniform_seed'])
                runner.prefill(torch.tensor([doc['prefix_ids']],device='mps'),cfg['prefill_chunk'])
                prefix=prefix_digest(runner.cache,doc['prefix_token_count'])
                if method!='A':runner.state.begin_request(cfg['rho'],k0=cap if all_select else None)
                runner.state.record_selection=True
                return runner,prefix
            for di,doc in enumerate(docs):
                guard.check(record=True)
                a,_=new_path(doc,'A')
                logits=question_forward(a,torch.tensor([doc['suffix_ids']],device='mps'))
                continuation=fixed_greedy_tokens(a,logits,cfg['state_diagnostic_generated_tokens'],guard.check)
                completed['sequence_generations']+=1
                sequence=doc['suffix_ids']+continuation
                sequences.append(dict(doc_id=doc['doc_id'],suffix_ids=doc['suffix_ids'],continuation_ids=continuation,
                                      continuation_text=tok.decode(continuation,skip_special_tokens=False),role='controlled_not_quality'))
                write_json(out/'sequences.json',sequences)
                references=[]
                a,aprefix=new_path(doc,'A')
                for step,token in enumerate(sequence):
                    guard.check();capture.start();logits=a.step(torch.tensor([[token]],device='mps'))
                    record=capture.finish(logits);references.append(record)
                    if di==0 and step in (len(doc['suffix_ids'])-1,len(sequence)-1):
                        np.savez_compressed(out/f'audit-{doc["doc_id"]}-A-step{step}.npz',**{k:v.numpy() for k,v in record.items()})
                completed['diagnostic_requests']+=1
                if di==0:
                    # Correctness before any all-layer sparse trajectory: CPU strict tests are supplemented by existing bf16 tolerances.
                    for method in ('D-one','D'):
                        run,prefix=new_path(doc,method,all_select=True)
                        if prefix!=aprefix:raise RuntimeError('full prefill cache differs between methods')
                        differences=[]
                        for step,token in enumerate(sequence):
                            guard.check();capture.start();logits=run.step(torch.tensor([[token]],device='mps'))
                            differences.append(state_difference(capture.finish(logits),references[step]))
                        check={'method':method,'full_selection':True,'steps':len(sequence),
                               'max_abs_logits':max(r['logits']['max_abs'][0] for r in differences),
                               'max_relative_logits':max(r['logits']['relative_l2'][0] for r in differences),
                               'argmax_agreement':sum(r['argmax_equal'] for r in differences)/len(differences)}
                        write_json(out/f'full-selection-{method}.json',differences)
                        checks.append(check);write_json(out/'correctness.json',checks)
                        if check['max_abs_logits']>tol['logits_atol'] or check['max_relative_logits']>tol['logits_rel_norm'] or check['argmax_agreement']<tol['argmax_agreement_min']:
                            raise RuntimeError(f'full selection bf16 check failed: {method}')
                        completed['correctness_requests']+=1
                for method in ('D0','D-one','D'):
                    guard.check();run,prefix=new_path(doc,method)
                    if prefix!=aprefix:raise RuntimeError('full prefill cache differs between methods')
                    with (out/f'trajectory-{doc["doc_id"]}-{method}.jsonl').open('w') as f:
                        for step,token in enumerate(sequence):
                            guard.check();capture.start();logits=run.step(torch.tensor([[token]],device='mps'))
                            actual=capture.finish(logits)
                            row=dict(doc_id=doc['doc_id'],method=method,step=step,query_position=doc['prefix_token_count']+step,
                                     input_token=token,phase='question' if step<len(doc['suffix_ids']) else 'controlled_continuation',
                                     **state_difference(actual,references[step]))
                            f.write(json.dumps(row,allow_nan=False)+'\n')
                            if step in (len(doc['suffix_ids'])-1,len(sequence)-1):
                                np.savez_compressed(out/f'positions-{doc["doc_id"]}-{method}-step{step}.npz',**selection_bitsets(run.state))
                                if di==0:np.savez_compressed(out/f'audit-{doc["doc_id"]}-{method}-step{step}.npz',**{k:v.numpy() for k,v in actual.items()})
                    completed['diagnostic_requests']+=1
                write_json(out/'sequences.json',sequences)
                write_json(out/'manifest.json',manifest)
                print(f'{doc["doc_id"]}: controlled trajectories saved; remaining={budget.remaining:.0f}s',flush=True)
                del references,record,actual,a,run
                torch.mps.empty_cache()
            manifest['body_hash_after']=body_digest(model)
            if manifest['body_hash_after']!=manifest['body_hash_before']:raise RuntimeError('body weights changed')
            manifest['memory_samples']=guard.samples
            if completed!=planned:raise RuntimeError('incomplete G3c counts')
    except BaseException as error:
        reason=f'{type(error).__name__}: {runinfo.sanitize(str(error))}'
        (out/'error.txt').write_text(runinfo.sanitize(traceback.format_exc()));print(reason,flush=True)
    finally:
        if capture is not None:capture.close()
        manifest.update(status='done' if reason is None and completed==planned else 'incomplete',reason=reason,
                        utc_end=runinfo.utc_now(),elapsed_seconds=time.monotonic()-begin,pending={k:planned[k]-completed[k] for k in planned})
        manifest['output_hashes']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest);print(run_id,manifest['status'],flush=True)
    return 0 if manifest['status']=='done' else 1


if __name__=='__main__':raise SystemExit(main())
