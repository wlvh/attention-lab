#!/usr/bin/env python3
"""Recompute G3c controlled-trajectory summaries; no model, torch or answer scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean
from campaign_evidence import ROOT,package,read,dump,sha


def read_evidence():
    e=read('g3c');prefix=e['index']['formal_run']+'/'
    docs=[json.loads(x) for x in e['files'][prefix+'data.jsonl'].splitlines()]
    seq=json.loads(e['files'][prefix+'sequences.json'])
    if {d['doc_id'] for d in docs}!={x['doc_id'] for x in seq} or len(docs)!=4:raise ValueError('diagnostic identities differ')
    checks=json.loads(e['files'][prefix+'correctness.json'])
    if {x['method'] for x in checks}!={'D','D-one'}:raise ValueError('missing full-selection check')
    tol=e['manifest']['full_selection_tolerance']
    for x in checks:
        if x['max_abs_logits']>tol['logits_atol'] or x['max_relative_logits']>tol['logits_rel_norm'] or x['argmax_agreement']<tol['argmax_agreement_min']:raise ValueError('failed full selection')
    rows=[]
    for doc in docs:
        length=len(doc['suffix_ids'])+e['manifest']['config']['state_diagnostic_generated_tokens']
        for method in ('D0','D-one','D'):
            part=[json.loads(x) for x in e['files'][prefix+f'trajectory-{doc["doc_id"]}-{method}.jsonl'].splitlines()]
            if [x['step'] for x in part]!=list(range(length)):raise ValueError('missing or duplicate trajectory steps')
            if any(x['method']!=method or x['doc_id']!=doc['doc_id'] for x in part):raise ValueError('row identity mismatch')
            rows.extend(part)
    e.update(docs=docs,sequences=seq,checks=checks,rows=rows)
    return e


def summarize(e):
    points={}
    for moment in ('first_answer_prediction','last_controlled_step'):
        points[moment]={}
        for method in ('D0','D-one','D'):
            rows=[]
            for doc in e['docs']:
                step=len(doc['suffix_ids'])-1+(e['manifest']['config']['state_diagnostic_generated_tokens'] if moment=='last_controlled_step' else 0)
                rows.append(next(r for r in e['rows'] if r['method']==method and r['doc_id']==doc['doc_id'] and r['step']==step))
            points[moment][method]={kind:[mean(x[kind]['relative_l2'][layer] for x in rows) for layer in range(28)] for kind in ('q','k','attention_output')}
            points[moment][method].update(logits_relative_l2=mean(x['logits']['relative_l2'][0] for x in rows),
                                          argmax_equal=sum(x['argmax_equal'] for x in rows),documents=4)
    return dict(run_id=e['index']['formal_run'],source_commit=e['manifest']['git']['commit'],
                report_source_sha256=sha(Path(__file__).read_bytes()),archive_sha256=e['index']['sha256'],
                scope='4 old development docs, identical controlled token inputs; NOT free-generation accuracy',
                full_selection=e['checks'],points=points,raw_step_rows=len(e['rows']),budget=e['budget'],
                failed_runs={k:v['reason'] for k,v in e['manifests'].items() if v['status']!='done'})


def table(s):
    lines=['| 对齐位置 | 方法 | 第14层Q差异 | 第28层Q差异 | 第28层读出差异 | logits差异 | argmax相同 |','|---|---|---:|---:|---:|---:|---:|']
    for name,methods in s['points'].items():
        for m,x in methods.items():
            lines.append(f'| {name} | {m} | {x["q"][13]:.4f} | {x["q"][27]:.4f} | {x["attention_output"][27]:.4f} | {x["logits_relative_l2"]:.4f} | {x["argmax_equal"]}/4 |')
    return '\n'.join(lines)


def plot(s):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for ax,kind,title in zip(axes,('q','attention_output'),('Incoming query difference from A','Attention output difference from A')):
        for m,c in [('D0','#e69f00'),('D-one','#009e73'),('D','#0072b2')]:
            ax.plot(range(1,29),s['points']['first_answer_prediction'][m][kind],'o-',color=c,label=m,markersize=3)
        ax.axvline(14,color='gray',ls=':',label='single sparse layer');ax.legend(ncol=4);ax.set(ylabel='Relative L2',title=title,ylim=(0,None));ax.grid(alpha=.2)
    axes[-1].set(xlabel='Transformer layer (1-based)',xticks=list(range(1,29)))
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-run',type=Path);p.add_argument('--include-run',type=Path,action='append',default=[]);a=p.parse_args()
    if a.package_run:package('g3c',a.package_run,a.include_run)
    e=read_evidence();s=summarize(e);dump(ROOT/'reports/g3c_summary_mac_m4max.json',s)
    body=f'''# G3c：把冻结索引器接入真实稀疏路径

本阶段在4份旧开发资料上，独立建立每种方法的完整资料缓存，然后从问题开始按各自状态推进。D-one只有第14层稀疏；D0与D所有层稀疏。相同token输入仅用于对齐内部计算，**下表不是答题准确率**，argmax相同也不是正确率。

正式run `{s['run_id']}`，计算源码 `{s['source_commit']}`。主体bf16/MPS，索引fp32，P0/r32/rho=1/8。参数身份与缓存绑定，更新/替换参数会在写缓存前拒绝。主体前后hash一致，所有方法的初始主K/V前缀逐字节一致，后续缓存独立，不借用A的新后缀。

## 正确性与失败记录

{json.dumps(s['full_selection'],ensure_ascii=False)}

初次run在单层全选处失败，旧记录保留：{s['failed_runs']}。有限精度下，按索引分数排列与按原位置排列虽是同一集合，计算顺序不同。一次修复为本轮所有B/C/D0/D选位后按原位置升序读取；主注意力仍实际gather该集合，不走全选旁路，不改集合或名额。CPU检查集合相同与float64数学等价，真实MPS检查沿用原容差。不能把这种数值变化冒称学习效果，G4全部新方法使用同一排序规则；旧G2记录不回写。

## 同token状态差异

每格先计算单份资料的向量相对L2，再对4份平均。Q/K是当前层主RoPE之前的新位置表示，读出是o_proj之前的向量；不同路径使用各自实际特征，不是G3b的同特征局部误差。

{table(s)}

原预测是前层少读会改变后层Q/K，完整路径上的好指标不保证部署后相同。D-one在第14层读入之前没有前层稀疏；D的前层差异可以传到后层。曲线只描述实际状态差异，不定位哪一层导致哪一个答案，也不追加轨迹修正训练。

## 证据与接续

归档含完整逐步差异、固定序列、原位置名单位图、第一份两步的小型Q/K/读出/logits数组、全部成功/失败manifest及各自源码/配置。累计共享设备预算 {e['budget']['used_seconds']:.2f}/14400秒，现金0。32份最终测试尚未在本阶段评分。

下一阶段在冻结新测试上真正自由生成，并独立测同token成本。Notebook默认只读存量证据；整页视觉验收范围见PROJECT_STATUS。
'''
    (ROOT/'reports/g3c_mac_m4max.md').write_text(body);print(json.dumps({'run':s['run_id'],'full_selection':s['full_selection'],'points':s['points']['first_answer_prediction']},ensure_ascii=False))

if __name__=='__main__':main()
