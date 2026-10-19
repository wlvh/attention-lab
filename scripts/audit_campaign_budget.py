#!/usr/bin/env python3
"""Audit the existing five campaign runs; archives and optional local records are read-only."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import hashlib,json,math
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(value):return hashlib.sha256(value).hexdigest()


def timeline(ledger,manifests):
    entries=ledger['runs'];ids=[x['run_id'] for x in entries]
    if len(ids)!=len(set(ids)) or set(ids)!=set(manifests):raise ValueError('duplicate or missing run accounting')
    if ledger.get('active_pid') is not None:raise ValueError('unfinished active run')
    total=sum(x['elapsed_seconds'] for x in entries)
    if not math.isclose(total,ledger['used_seconds'],rel_tol=0,abs_tol=1e-9):raise ValueError('elapsed total differs')
    rows=[];previous_end=None
    for entry in entries:
        m=manifests[entry['run_id']]
        if m['status']!=entry['status']:raise ValueError('run status differs')
        start,end=datetime.fromisoformat(m['utc_start']),datetime.fromisoformat(m['utc_end'])
        if start.tzinfo is None or end.tzinfo is None or end<start:raise ValueError('invalid UTC interval')
        # Original UTC fields have whole-second precision. Use [start, end+1s)
        # when excluding overlap; monotonic elapsed values retain their precision.
        gap=None if previous_end is None else (start-previous_end).total_seconds()
        if gap is not None and gap<0:raise ValueError('recorded time intervals overlap or order differs')
        elapsed=entry['elapsed_seconds'];outer=m['elapsed_seconds'];utc_seconds=(end-start).total_seconds()
        if not math.isfinite(elapsed) or not 0<=elapsed<=outer or not utc_seconds-1<=outer<=utc_seconds+1:
            raise ValueError('elapsed value outside recorded timing scopes')
        rows.append(dict(run_id=entry['run_id'],status=entry['status'],utc_start=m['utc_start'],utc_end=m['utc_end'],
                         ledger_seconds=elapsed,manifest_outer_seconds=outer,outer_minus_ledger_seconds=outer-elapsed,
                         utc_whole_seconds=utc_seconds,conservative_gap_from_previous_seconds=gap))
        previous_end=end+timedelta(seconds=1)
    return dict(runs=rows,ledger_total_seconds=total,manifest_outer_total_seconds=sum(x['manifest_outer_seconds'] for x in rows),
                outer_minus_ledger_total_seconds=sum(x['outer_minus_ledger_seconds'] for x in rows),
                minimum_conservative_gap_seconds=min(x['conservative_gap_from_previous_seconds'] for x in rows[1:]),
                no_recorded_overlap=True)


def audit(root=ROOT,*,verify_local=False):
    root=Path(root);manifests={};raw_manifests={};snapshots=[];source_versions={};inputs={}
    for stage,budget_name in [('g3b','budget_at_G3b_completion.json'),('g3c','budget_at_stage_completion.json'),('g4lite','budget_at_stage_completion.json')]:
        index_path=root/f'reports/provenance/{stage}_archive_manifest.json';index_bytes=index_path.read_bytes();index=json.loads(index_bytes)
        archive_bytes=(root/index['archive']).read_bytes()
        if sha(archive_bytes)!=index['sha256']:raise ValueError('archive hash differs')
        inputs[str(index_path.relative_to(root))]=sha(index_bytes);inputs[index['archive']]=index['sha256']
        with tarfile.open(root/index['archive']) as tar:
            def read(name):
                data=tar.extractfile(name).read();rec=index['files'][name]
                if sha(data)!=rec['sha256'] or len(data)!=rec['bytes']:raise ValueError('archive member hash differs')
                return data
            budget_bytes=read(budget_name);snapshots.append((stage,json.loads(budget_bytes),budget_bytes))
            for member in tar.getmembers():
                if not member.isfile() or not member.name.startswith(stage+'-') or not member.name.endswith('/manifest.json'):continue
                raw=read(member.name);m=json.loads(raw);run=m['run_id']
                if run in manifests:raise ValueError('duplicate manifest')
                manifests[run]=m;raw_manifests[run]=raw
                source='src/attention_lab/campaign.py'
                snapshot='source/'+(m['git']['commit']+'/' if stage!='g3b' else '')+source
                original=read(snapshot)
                if sha(original)!=m['execution']['sources'][source]:raise ValueError('original campaign source hash differs')
                source_versions[run]=dict(commit=m['git']['commit'],campaign_source_sha256=sha(original),manifest_sha256=sha(raw))
    final=snapshots[-1][1]
    for stage,earlier,_ in snapshots:
        if earlier['runs']!=final['runs'][:len(earlier['runs'])]:raise ValueError('stage ledger is not a prefix of final accounting')
        if not math.isclose(sum(x['elapsed_seconds'] for x in earlier['runs']),earlier['used_seconds'],rel_tol=0,abs_tol=1e-9):raise ValueError('stage sum differs')
        for key in ('limit_seconds','prior_ledger','prior_ledger_sha256','prior_g3a_seconds'):
            if earlier[key]!=final[key]:raise ValueError('budget or prior accounting changed between stages')
    result=timeline(final,manifests)
    if verify_local:
        local=root/'runs/g3bg4lite-20261018-budget.json'
        if local.read_bytes()!=snapshots[-1][2]:raise ValueError('live ledger differs from final archived snapshot')
        for run,raw in raw_manifests.items():
            if (root/'runs'/run/'manifest.json').read_bytes()!=raw:raise ValueError('live manifest differs from archive')
        if sha((root/final['prior_ledger']).read_bytes())!=final['prior_ledger_sha256']:raise ValueError('prior G3a ledger changed')
    result.update(status='passed',scope='five recorded campaign runs; no new model execution, no original ledger/manifest edits',
                  utc_resolution='whole seconds; conservative end = recorded end + 1 second',
                  timing_scopes='manifest outer timer brackets CampaignBudget context; ledger timer starts inside entry and ends before final bookkeeping',
                  limit_seconds=final['limit_seconds'],prior_g3a_seconds=final['prior_g3a_seconds'],
                  historical_ledger_sha256=sha(snapshots[-1][2]),archive_inputs=inputs,as_run_sources=source_versions,
                  current_campaign_source_sha256=sha((root/'src/attention_lab/campaign.py').read_bytes()),
                  audit_source_sha256=sha(Path(__file__).read_bytes()),
                  limitation='The existing records support serial execution and complete accounting for these five runs; they are not an independent OS process-monitoring trace.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-local',action='store_true');a=p.parse_args()
    result=audit(verify_local=a.verify_local)
    (ROOT/'reports/campaign_budget_history_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','ledger_total_seconds','outer_minus_ledger_total_seconds','minimum_conservative_gap_seconds')},ensure_ascii=False))
    if a.verify_local:print('Existing local ledger, five manifests and prior G3a ledger match their archived bytes/hashes.')
