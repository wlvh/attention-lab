#!/usr/bin/env python3
"""Frozen new T1 test: independent free generation and separate fixed-token cost."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from attention_lab import runinfo
from attention_lab.campaign import CampaignBudget,ResourceGuard,digest,execution_identity,write_json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';runinfo.refuse_unsafe_env()
    import numpy as np
    import torch
    import transformers
    from attention_lab.qwen_wrap import load_model,wrap_model,geometry_of,weights_file_sha256
    from attention_lab.trained import load_bank
    from attention_lab.inference import runner_for,question_forward,free_generate,fixed_greedy_tokens,prefix_digest,selection_bitsets
    from attention_lab.g2lite_eval import score_answer
    from attention_lab.g3a import body_digest
    cfg_path=ROOT/'configs/mac_g3bg4lite.json';cfg=json.loads(cfg_path.read_text())
    protocol=json.loads((ROOT/'configs/g4lite_execution.json').read_text())
    plan=json.loads((ROOT/protocol['final_data_plan']).read_text())
    git=runinfo.git_info(ROOT)
    if git['dirty_files']:raise RuntimeError('commit execution/configuration before final test')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    bank_dir=ROOT/protocol['g3b_run'];gate_dir=ROOT/protocol['g3c_run']
    parent=json.loads((gate_dir/'manifest.json').read_text())
    if parent['status']!='done' or parent['completed']!=parent['planned']:raise RuntimeError('G3c incomplete')
    # Reject modified inference code or changed tested precision/geometry without new correctness evidence.
    for name in ('qwen_wrap.py','decode.py','sparse.py','full.py','trained.py','inference.py','indexer.py','selection.py','reference.py','g2lite_control.py'):
        path='src/attention_lab/'+name
        if digest(ROOT/path)!=parent['execution']['sources'][path]:raise RuntimeError('G3c execution changed: '+path)
    for key in ('model_id','revision','r','rho','body_dtype','index_dtype','g3c_g4_row_order'):
        if cfg[key]!=parent['config'][key]:raise RuntimeError('G3c configuration changed: '+key)
    metadata=json.loads((ROOT/'reports/g3b_parameters.json').read_text())
    if metadata!=parent['bank']:raise RuntimeError('index bank differs from G3c')
    data_path=ROOT/plan['local_data']
    if digest(data_path)!=plan['data_sha256']:raise RuntimeError('frozen final data changed')
    docs=[json.loads(x) for x in data_path.read_text().splitlines()]
    if len(docs)!=cfg['test_docs'] or len({d['doc_id'] for d in docs})!=len(docs):raise RuntimeError('test identity/count mismatch')
    execution=execution_identity(ROOT,cfg_path,'bench/g4lite_quality_cost.py',{'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__})
    for file in ('configs/g4lite_execution.json',protocol['final_data_plan']):execution['configs'][file]=digest(ROOT/file)
    run_id='g4lite-'+runinfo.utc_compact();out=ROOT/'runs'/run_id;out.mkdir(parents=True)
    planned=dict(quality_requests=len(docs)*len(cfg['quality_methods']),cost_sequences=len(cfg['cost_doc_indices']),
                 cost_requests=len(cfg['cost_doc_indices'])*len(cfg['cost_methods'])*(cfg['cost_warmup_repeats']+cfg['cost_measured_repeats']))
    completed={k:0 for k in planned};reason=None;start=time.monotonic()
    manifest=dict(run_id=run_id,stage='G4-lite',status='running',utc_start=runinfo.utc_now(),git=git,execution=execution,
                  config=cfg,protocol=protocol,bank=metadata,upstream_manifest_sha256=digest(gate_dir/'manifest.json'),
                  test_data_sha256=plan['data_sha256'],planned=planned,completed=completed)
    write_json(out/'manifest.json',manifest);write_json(out/'config.json',cfg);(out/'data.jsonl').write_bytes(data_path.read_bytes())
    try:
        with CampaignBudget(ROOT,cfg,run_id) as budget:
            torch.set_num_threads(4);guard=ResourceGuard(budget,torch);guard.check(record=True,estimated_next=1800)
            load_start=time.perf_counter()
            model,tok,info=load_model(cfg['model_id'],cfg['revision'],dtype=torch.bfloat16,device='mps');holder=wrap_model(model)
            bank=load_bank(bank_dir,metadata,geometry_of(model),device='mps');initial=load_bank(bank_dir,metadata,geometry_of(model),device='mps',initial=True)
            torch.mps.synchronize();manifest['load_model_and_banks_seconds']=time.perf_counter()-load_start
            manifest['model']=info;manifest['checkpoint']=weights_file_sha256(cfg['model_id'],cfg['revision'])
            manifest['body_hash_before']=body_digest(model)
            eos=set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id,list) else [model.generation_config.eos_token_id])
            if tok.eos_token_id is not None:eos.add(tok.eos_token_id)
            manifest['eos_token_ids']=sorted(x for x in eos if x is not None);eos=set(manifest['eos_token_ids'])
            def timed(fn):
                torch.mps.synchronize();t=time.perf_counter();result=fn();torch.mps.synchronize();return result,time.perf_counter()-t
            def setup(doc,method):
                return runner_for(model,holder,doc['prompt_token_count']+cfg['max_new_tokens'],method,bank,initial,doc=doc,uniform_seed=cfg['uniform_seed'])
            first_doc_times=[]
            with (out/'quality.jsonl').open('w') as records:
                for i,doc in enumerate(docs):
                    if first_doc_times:guard.check(record=True,estimated_next=max(first_doc_times)*1.5,reserve=900)
                    else:guard.check(record=True)
                    doc_start=time.monotonic();prefix=torch.tensor([doc['prefix_ids']],device='mps');suffix=torch.tensor([doc['suffix_ids']],device='mps')
                    methods=cfg['quality_methods'];shift=i%len(methods);order=methods[shift:]+methods[:shift]
                    prefix_hashes={}
                    for method in order:
                        guard.check();runner,setup_s=timed(lambda:setup(doc,method))
                        _,prefill_s=timed(lambda:runner.prefill(prefix,cfg['prefill_chunk']))
                        prefix_hashes[method]=prefix_digest(runner.cache,doc['prefix_token_count'])
                        runner.state.record_selection=True
                        def question():
                            if method!='A':runner.state.begin_request(cfg['rho'])
                            return question_forward(runner,suffix)
                        logits,question_s=timed(question)
                        selected=selection_bitsets(runner.state)
                        np.savez_compressed(out/f'positions-{doc["doc_id"]}-{method}.npz',**selected)
                        legal=int(selected['legal']);candidate=legal-68;k0=runner.state.k0
                        row=dict(doc_id=doc['doc_id'],doc_index=doc['doc_index'],method=method,task='T1',targets=doc['targets'],target_ids=doc['target_ids'],
                                 buckets=doc['buckets'],rho=None if method=='A' else cfg['rho'],k0=k0,first_prediction_legal=legal,
                                 first_prediction_candidate_count=candidate,first_prediction_read_count=int(selected['counts'][0]),
                                 candidate_extraction_ratio=None if method=='A' else k0/candidate,
                                 total_read_ratio=float(selected['counts'][0])/legal,cache_bytes=runner.cache.bytes(),
                                 uniform_seed=runner.state.uniform_seed if method=='C' else None,uniform_request_counter=runner.state.request_counter if method=='C' else None)
                        runner.state.record_selection=False
                        (tokens,stopped),generate_s=timed(lambda:free_generate(runner,logits,max_new=cfg['max_new_tokens'],eos=eos,guard=guard.check))
                        raw=tok.decode(tokens,skip_special_tokens=False);text=tok.decode(tokens,skip_special_tokens=True)
                        row.update(token_ids=tokens,raw_text=raw,answer_text=text,stopped_on_eos=stopped,generated_tokens_including_eos=len(tokens),
                                   score=score_answer('T1',doc['targets'],text,truncated=not stopped),
                                   times=dict(setup_s=setup_s,prefill_s=prefill_s,question_s=question_s,free_generation_s=generate_s),
                                   memory_after=guard.check(record=True))
                        row['answered_with_ids']=row['score']['predicted']==doc['target_ids']
                        records.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');records.flush();completed['quality_requests']+=1
                        del runner,logits,selected
                        holder.state=None
                    if len(set(prefix_hashes.values()))!=1:raise RuntimeError('main prefix K/V differ across methods')
                    manifest.setdefault('prefix_identity_checks',{})[doc['doc_id']]=prefix_hashes
                    first_doc_times.append(time.monotonic()-doc_start)
                    write_json(out/'manifest.json',manifest)
                    print(f'quality {i+1}/{len(docs)} saved; remaining={budget.remaining:.0f}s',flush=True)
                    torch.mps.empty_cache()
            # Fixed-input cost is separate from quality. Controlled A sequences are never graded.
            cost_sequences=[];doc_cost_times=[]
            with (out/'cost.jsonl').open('w') as records:
                for doc in (d for d in docs if d['doc_index'] in cfg['cost_doc_indices']):
                    guard.check(record=True,estimated_next=(max(doc_cost_times)*1.5 if doc_cost_times else 600))
                    doc_start=time.monotonic();prefix=torch.tensor([doc['prefix_ids']],device='mps');suffix=torch.tensor([doc['suffix_ids']],device='mps')
                    runner=setup(doc,'A');runner.prefill(prefix,cfg['prefill_chunk']);logits=question_forward(runner,suffix)
                    seq=fixed_greedy_tokens(runner,logits,cfg['cost_generated_tokens'],guard.check)
                    cost_sequences.append(dict(doc_id=doc['doc_id'],token_ids=seq,role='controlled_not_quality',text=tok.decode(seq,skip_special_tokens=False)))
                    write_json(out/'cost_sequences.json',cost_sequences);completed['cost_sequences']+=1
                    del runner,logits;holder.state=None
                    inputs=torch.tensor([seq],device='mps')
                    for repeat in range(cfg['cost_warmup_repeats']+cfg['cost_measured_repeats']):
                        methods=cfg['cost_methods'];shift=(doc['doc_index']+repeat)%len(methods);order=methods[shift:]+methods[:shift]
                        for ordinal,method in enumerate(order):
                            guard.check();runner,setup_s=timed(lambda:setup(doc,method))
                            _,prefill_s=timed(lambda:runner.prefill(prefix,cfg['prefill_chunk']))
                            def question():
                                if method not in ('A','A-best'):runner.state.begin_request(cfg['rho'])
                                return question_forward(runner,suffix,batch=method=='A-best')
                            logits,question_s=timed(question)
                            def controlled():
                                logits=None
                                for step in range(inputs.shape[1]):logits=runner.step(inputs[:,step:step+1])
                                return logits
                            logits,decode_s=timed(controlled)
                            row=dict(doc_id=doc['doc_id'],method=method,repeat=repeat,order=ordinal,
                                     measured=repeat>=cfg['cost_warmup_repeats'],controlled_input_steps=len(seq),
                                     setup_s=setup_s,prefill_s=prefill_s,question_s=question_s,controlled_decode_s=decode_s,
                                     hot_s=question_s+decode_s,fresh_cache_request_s=setup_s+prefill_s+question_s+decode_s,
                                     cache_bytes=runner.cache.bytes(),memory_after=guard.check(record=True))
                            records.write(json.dumps(row,allow_nan=False)+'\n');records.flush();completed['cost_requests']+=1
                            del runner,logits;holder.state=None
                    doc_cost_times.append(time.monotonic()-doc_start);write_json(out/'manifest.json',manifest)
                    print(f'cost {len(doc_cost_times)}/{len(cfg["cost_doc_indices"])} saved; remaining={budget.remaining:.0f}s',flush=True)
                    torch.mps.empty_cache()
            manifest['body_hash_after']=body_digest(model);manifest['memory_samples']=guard.samples
            if manifest['body_hash_after']!=manifest['body_hash_before']:raise RuntimeError('body weights changed')
            if completed!=planned:raise RuntimeError('incomplete G4 counts')
    except BaseException as error:
        reason=f'{type(error).__name__}: {runinfo.sanitize(str(error))}'
        (out/'error.txt').write_text(runinfo.sanitize(traceback.format_exc()));print(reason,flush=True)
    finally:
        manifest.update(status='done' if reason is None and completed==planned else 'incomplete',reason=reason,
                        utc_end=runinfo.utc_now(),elapsed_seconds=time.monotonic()-start,pending={k:planned[k]-completed[k] for k in planned})
        manifest['output_hashes']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest);print(run_id,manifest['status'],flush=True)
    return 0 if manifest['status']=='done' else 1

if __name__=='__main__':raise SystemExit(main())
