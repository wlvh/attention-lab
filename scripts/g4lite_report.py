#!/usr/bin/env python3
"""Frozen G4 answer re-scoring and controlled-cost aggregation; standard library only."""
from __future__ import annotations
import argparse,json,math,types
from pathlib import Path
from statistics import mean,median
from campaign_evidence import ROOT,package,read,dump,sha


def read_evidence():
    e=read('g4lite');run=e['index']['formal_run'];prefix=run+'/'
    docs=[json.loads(x) for x in e['files'][prefix+'data.jsonl'].splitlines()];by_id={d['doc_id']:d for d in docs}
    if sha(e['files'][prefix+'data.jsonl'])!=e['manifest']['test_data_sha256']:raise ValueError('test data hash differs')
    cfg=e['manifest']['config'];source_prefix=f'source/{e["manifest"]["git"]["commit"]}/'
    plan=json.loads(e['files'][source_prefix+e['manifest']['protocol']['final_data_plan']])
    if plan['data_sha256']!=e['manifest']['test_data_sha256'] or plan['count']!=len(docs):raise ValueError('preregistered data differs')
    # Execute only the archived, hash-verified pure scoring module, not whichever version happens to be current.
    scorer=types.ModuleType('frozen_g4_scorer');exec(compile(e['files'][source_prefix+'src/attention_lab/g2lite_eval.py'],'<archived frozen scorer>','exec'),scorer.__dict__)
    quality=[json.loads(x) for x in e['files'][prefix+'quality.jsonl'].splitlines()]
    keys=[(r['doc_id'],r['method']) for r in quality]
    expected={(d['doc_id'],m) for d in docs for m in cfg['quality_methods']}
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('missing/duplicate quality rows')
    eos=set(e['manifest']['eos_token_ids'])
    for r in quality:
        d=by_id[r['doc_id']]
        if r['targets']!=d['targets'] or r['target_ids']!=d['target_ids']:raise ValueError('grade target differs from frozen data')
        score=scorer.score_answer('T1',d['targets'],r['answer_text'],truncated=not r['stopped_on_eos'])
        if score!=r['score'] or r['answered_with_ids']!=(score['predicted']==d['target_ids']):raise ValueError('raw answer regrade differs')
        tokens=r['token_ids']
        if len(tokens)!=r['generated_tokens_including_eos'] or not 1<=len(tokens)<=cfg['max_new_tokens']:raise ValueError('invalid free token count')
        if any(t in eos for t in tokens[:-1]) or (tokens[-1] in eos)!=r['stopped_on_eos']:raise ValueError('EOS rule differs')
        if not r['stopped_on_eos'] and len(tokens)!=cfg['max_new_tokens']:raise ValueError('incomplete free generation')
        for key in ('setup_s','prefill_s','question_s','free_generation_s'):
            if not math.isfinite(r['times'][key]) or r['times'][key]<0:raise ValueError('invalid quality timing')
        if len(set(e['manifest']['prefix_identity_checks'][d['doc_id']].values()))!=1:raise ValueError('prefix differs')
    costs=[json.loads(x) for x in e['files'][prefix+'cost.jsonl'].splitlines()]
    cost_ids={d['doc_id'] for d in docs if d['doc_index'] in cfg['cost_doc_indices']}
    expected={(d,m,rep) for d in cost_ids for m in cfg['cost_methods'] for rep in range(cfg['cost_warmup_repeats']+cfg['cost_measured_repeats'])}
    keys=[(r['doc_id'],r['method'],r['repeat']) for r in costs]
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('cost grid incomplete/duplicate')
    for r in costs:
        if r['controlled_input_steps']!=cfg['cost_generated_tokens'] or r['measured']!=(r['repeat']>=cfg['cost_warmup_repeats']):raise ValueError('cost protocol differs')
        if abs(r['hot_s']-r['question_s']-r['controlled_decode_s'])>1e-9 or abs(r['fresh_cache_request_s']-r['hot_s']-r['prefill_s']-r['setup_s'])>1e-9:raise ValueError('timing arithmetic differs')
    seq=json.loads(e['files'][prefix+'cost_sequences.json'])
    if {x['doc_id'] for x in seq}!=cost_ids or any(len(x['token_ids'])!=cfg['cost_generated_tokens'] or x['role']!='controlled_not_quality' for x in seq):raise ValueError('cost sequence provenance differs')
    if e['manifest']['body_hash_before']!=e['manifest']['body_hash_after']:raise ValueError('body weights changed')
    e.update(docs=docs,quality=quality,costs=costs,sequences=seq)
    return e


def extent(values):
    values=list(values)
    return {'min':min(values),'median':median(values),'max':max(values)}


def summarize(e):
    quality={};paired={};q=e['quality'];by={(r['doc_id'],r['method']):r for r in q}
    for method in e['manifest']['config']['quality_methods']:
        rows=[x for x in q if x['method']==method]
        quality[method]=dict(n=len(rows),correct=sum(x['score']['exact_match'] for x in rows),
                             format_errors=sum(x['score']['format_error'] for x in rows),id_echoes=sum(x['answered_with_ids'] for x in rows),
                             truncated=sum(x['score']['truncated'] for x in rows),token_count=extent(x['generated_tokens_including_eos'] for x in rows),
                             k0=sorted({x['k0'] for x in rows if x['k0'] is not None}),read_count=extent(x['first_prediction_read_count'] for x in rows),
                             candidate_extraction_ratio=None if method=='A' else extent(x['candidate_extraction_ratio'] for x in rows),
                             total_read_ratio=extent(x['total_read_ratio'] for x in rows),cache_bytes=extent(x['cache_bytes']['total_bytes'] for x in rows),
                             times={key:extent(x['times'][key] for x in rows) for key in rows[0]['times']},
                             by_bucket={b:dict(n=sum(x['buckets']==[b] for x in rows),correct=sum(x['score']['exact_match'] for x in rows if x['buckets']==[b])) for b in e['manifest']['config']['buckets_cycle']})
        if method!='A':
            paired[method]={k:[] for k in ('both_right','A_right_other_wrong','A_wrong_other_right','both_wrong')}
            for r in rows:
                a=by[r['doc_id'],'A']['score']['exact_match'];b=r['score']['exact_match']
                key='both_right' if a and b else 'A_right_other_wrong' if a else 'A_wrong_other_right' if b else 'both_wrong'
                paired[method][key].append(r['doc_id'])
    costs={};measured=[r for r in e['costs'] if r['measured']];cost_lookup={(x['doc_id'],x['repeat'],x['method']):x for x in measured}
    for method in e['manifest']['config']['cost_methods']:
        rows=[r for r in measured if r['method']==method]
        costs[method]=dict(n=len(rows),**{key:extent(r[key] for r in rows) for key in ('setup_s','prefill_s','question_s','controlled_decode_s','hot_s','fresh_cache_request_s')},
                           paired_ratio_to_A={key:extent(r[key]/cost_lookup[r['doc_id'],r['repeat'],'A'][key] for r in rows) for key in ('controlled_decode_s','hot_s','fresh_cache_request_s')},
                           cache_bytes={key:extent(r['cache_bytes'][key] for r in rows) for key in ('kv_bytes','index_bytes','total_bytes')})
    cases={}
    predicates={'A_right_D_wrong':lambda d:by[d,'A']['score']['exact_match'] and not by[d,'D']['score']['exact_match'],
                'D_right_D0_wrong':lambda d:by[d,'D']['score']['exact_match'] and not by[d,'D0']['score']['exact_match'],
                'A_wrong':lambda d:not by[d,'A']['score']['exact_match']}
    for name,predicate in predicates.items():
        chosen=next((d for d in e['docs'] if predicate(d['doc_id'])),None)
        cases[name]=None if chosen is None else {'document':chosen,'outputs':{m:by[chosen['doc_id'],m] for m in e['manifest']['config']['quality_methods']}}
    return dict(run_id=e['index']['formal_run'],source_commit=e['manifest']['git']['commit'],
                report_source_sha256=sha(Path(__file__).read_bytes()),archive_sha256=e['index']['sha256'],
                quality=quality,paired_against_A=paired,costs=costs,cases=cases,budget=e['budget'],
                scope='32 frozen new T1 documents, ~2K, one seed, same generator/template; descriptive not general capability')


def quality_table(s):
    lines=['| 方法 | 完整匹配 | 格式不足 | 编号回声 | 截断 | 输出token数中位（含EOS） |','|---|---:|---:|---:|---:|---:|']
    for m,x in s['quality'].items():lines.append(f'| {m} | {x["correct"]}/{x["n"]} | {x["format_errors"]} | {x["id_echoes"]} | {x["truncated"]} | {x["token_count"]["median"]} |')
    return '\n'.join(lines)


def cost_table(s):
    lines=['| 方法 | 新缓存请求秒，中位 [min,max] | 问题秒，中位 | 32次控制前向秒，中位 | 配对控制前向/A 中位 | KV+z MiB |','|---|---:|---:|---:|---:|---:|']
    for m,x in s['costs'].items():
        v=x['fresh_cache_request_s'];lines.append(f'| {m} | {v["median"]:.3f} [{v["min"]:.3f},{v["max"]:.3f}] | {x["question_s"]["median"]:.3f} | {x["controlled_decode_s"]["median"]:.3f} | {x["paired_ratio_to_A"]["controlled_decode_s"]["median"]:.3f} | {x["cache_bytes"]["total_bytes"]["median"]/2**20:.1f} |')
    return '\n'.join(lines)


def plot(s):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    methods=list(s['quality']);colors={'A':'#333333','B':'#cc79a7','C':'#999999','D0':'#e69f00','D':'#0072b2'}
    hits=[s['quality'][m]['correct'] for m in methods]
    bars=axes[0].bar(methods,hits,color=[colors[m] for m in methods]);axes[0].bar_label(bars,labels=[f'{x}/32' for x in hits]);axes[0].set(ylim=(0,36),ylabel='Exact matches',title='New T1 free generation (32 documents)')
    methods=list(s['costs']);values=[s['costs'][m]['controlled_decode_s']['median'] for m in methods]
    bars=axes[1].bar(methods,values,color=[colors.get(m,'#555555') for m in methods]);axes[1].bar_label(bars,fmt='%.3f');axes[1].set(ylabel='Seconds (median of 12)',title='Same 32 input steps; NOT free generation');axes[1].set_ylim(0,max(values)*1.2)
    return fig



def plot_case(e,s,case_name='A_right_D_wrong'):
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    case=s['cases'].get(case_name)
    if case is None:return None
    d=case['document'];prefix=e['index']['formal_run']+'/'
    fig,ax=plt.subplots(figsize=(11,4),layout='constrained');labels=[]
    for mi,method in enumerate(('D0','D')):
        with np.load(io.BytesIO(e['files'][prefix+f'positions-{d["doc_id"]}-{method}.npz']),allow_pickle=False) as z:
            bits=np.unpackbits(z['bitsets'],axis=-1);legal=int(z['legal'])
        for j,layer in enumerate((0,13,27)):
            y=mi*3+j;pos=np.flatnonzero(bits[layer,0,:legal])
            ax.scatter(pos,np.full(len(pos),y),s=8,marker='|',color='#e69f00' if method=='D0' else '#0072b2')
            labels.append(f'{method}: layer {layer+1}, KV group 0')
    for start,end in d['target_spans']:ax.axvspan(start,end,color='#cc79a7',alpha=.35,label='Whole target record (not only value)')
    ax.set(yticks=list(range(6)),yticklabels=labels,xlabel='Original token position at first answer prediction',
           title=d['doc_id']+'; independently evolved states',xlim=(0,legal-1))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18));return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-run',type=Path);a=p.parse_args()
    if a.package_run:package('g4lite',a.package_run)
    e=read_evidence();s=summarize(e);dump(ROOT/'reports/g4lite_summary_mac_m4max.json',s);dump(ROOT/'reports/g4lite_answers_mac_m4max.json',e['quality'])
    budget_rows=[]
    for m,x in s['quality'].items():
        ratio=x['candidate_extraction_ratio'];budget_rows.append(f'| {m} | {x["k0"] or "NA"} | {x["read_count"]["min"]}–{x["read_count"]["max"]} | '+('NA' if ratio is None else f'{ratio["min"]:.4f}–{ratio["max"]:.4f}')+f' | {x["total_read_ratio"]["min"]:.4f}–{x["total_read_ratio"]["max"]:.4f} |')
    paired_rows=['| 方法对A | 两者对 | A对/方法错 | A错/方法对 | 两者错 |','|---|---:|---:|---:|---:|']
    for method,cells in s['paired_against_A'].items():
        paired_rows.append('| '+method+' | '+' | '.join(str(len(cells[k])) for k in ('both_right','A_right_other_wrong','A_wrong_other_right','both_wrong'))+' |')
    case_text=[]
    for name,c in s['cases'].items():
        if c is None:case_text.append(f'- {name}：本批未出现。');continue
        d=c['document'];case_text.append(f'### {name}：{d["doc_id"]}\n\n目标记录 `{d["target_ids"][0]} → {d["targets"][0]}`，{d["question"]}\n\n'+ '\n'.join(f'- {m} 原回答：`{r["raw_text"]}`；解析 {r["score"]["predicted"]}；完整匹配 {r["score"]["exact_match"]}。' for m,r in c['outputs'].items()))
    body=f'''# G4-lite：训练后的稀疏读取能保住多少答题能力？

模型收到一份含编号和四位值的资料，先完整建立资料缓存，再独立处理问题并生成回答。这个单记录取值任务称 T1。A 读全部历史；B/C/D0/D 都必留开头 4 个和最近 64 个位置，再从其余位置取相同数量：B 取最近的，C 均匀随机取，D0 按初始索引参数打分，D 按训练后参数打分。每种方法把自己预测的 token 送回，并保存自己算出的后缀 K/V。

额外名额按请求第一步长度的 1/8 固定，不包含必留位置。索引把主 RoPE 前 Q/K 投影到 32 维，不额外旋转；这就是“索引位置设置 P0，r=32”。正文分别报告新题回答和另一次等工作量计时；读取数量、教师权重覆盖与答题分数是不同指标。完整讲解见[第七课](07_trained_sparse_quality_and_cost.html)。

## 真实自由生成

{quality_table(s)}

每种方法重新完整Prefill资料，然后独立推进问题/生成缓存。贪心、thinking关闭、最多64个输出token，遇EOS结束；原始文本及特殊token都保留，评分仍用冻结规则取最前面的一个独立四位数。格式不足、编号回声与截断分别列示，可以有重叠，不把正确答案出现在后面就改成完整匹配。

A在这批新题上只有{s['quality']['A']['correct']}/{s['quality']['A']['n']}，不能继续套用旧小样本“T1基线全部正确”的印象。完整读取本身的错误也保留，不能把D/A正确数之比当逐题保留率。

{chr(10).join(paired_rows)}

配对A的全部资料ID见JSON。局部指标来自[G3b](g3b_mac_m4max.md)的完整路径特征和[G3c](g3c_mac_m4max.md)的4份开发资料控制轨迹，单独报告，不能混成新测试准确率。R仅为G3b昂贵教师诊断，没有作为G4任务质量上界。

## 读取预算：首个答案预测对应的最后一个问题token

| 方法 | 请求固定k0范围 | 实际总读取数范围 | k/M范围 | 总读取数/合法长度范围 |
|---|---|---|---|---|
{chr(10).join(budget_rows)}

ρ=1/8用于请求第一步确定k0，之后不随生成扩大名额；M是当步候选数量。F为首4个+最近64个，问题/生成逐步进入合法历史。k/M不等于ρ，总读取比例又包含F。原位置名单位图保留每层每组的实际集合。各稀疏方法同题同名额；本轮选后升序规则相同，不与旧G2的分数排列数字混作配对。

## 实际成本：同输入、同工作量

{cost_table(s)}

4份预定资料，每方法1轮预热、3轮测量，按资料/轮次轮换顺序。表中中位及范围取12次测量；比值先按同资料/同轮与A配对，再取中位，不是两个中位数相除。A-best只将已知问题批量处理；资料和控制延续长度相同。

这里是向模型输入32个固定控制token的32次前向，首个答案预测在问题阶段，不能直接称为“自然输出32个token的耗时”。控制序列来自A、可越过EOS，仅作成本输入，绝不评分。新缓存请求=新分配+资料Prefill（D含建z）+问题+32次前向，进程/分配器已预热。输入转设备、资源/预算检查、计分和写文件在成本计时外，每阶段前后同步。D/D0前向内部的参数与缓存身份检查仍计入当前推理开销；本轮未单独测量其耗时，不能据此定位变慢的主要原因。默认没有编译或融合Metal。自由生成的耗时单独保存在逐题记录，包含argmax/搬运和逐步保护开销，不能拿短输出直接当等工作量加速。

保留完整主K/V并额外存z，所以“少读”不是KV存储同比减少。上表内存为实际分配的KV+z字节；RSS/MPS采样不是每方法峰值，模型与两套索引参数常驻，其他系统占用也未计入算法净收益。

## 失败与改善案例

以下按资料顺序取每类第一个案例，未据此修改配置。完整160条原文在`g4lite_answers_mac_m4max.json`及归档。

{(chr(10)*2).join(case_text)}

这些案例只能提出解释线索，不能仅凭目标位置未入选就断言信息无影响，也不能把条件覆盖率直接当答案所需信息。主体Prefill始终完整，深层位置可能包含前文影响；本轮没有从资料阶段开始稀疏。

## 预测、边界与交付

运行 `{s['run_id']}`，计算源码 `{s['source_commit']}`。Qwen3-0.6B主体冻结，bf16/MPS/eager，P0/r32/rho=1/8，D0和D索引fp32。新32份T1约2K资料在训练前登记，未调参、未删题、未重试找更好分数。它们与旧资料去重，但仍来自相同生成器/模板；只有一个训练种子，结论是小样本描述。

原预测见`docs/tasks/09_G4_LITE.md`。观察支持了两个受限预测：D比D0/C答对更多（17/32对0/32），而约2K下索引/选择路径没有速度收益（控制前向D/A配对中位约1.58）。不支持“各层局部改善就能保持完整任务能力”：A对的21题中D只保住15题，另答对A错的2题。它们不是泛化定律，也没有把所有耗时差异定位到单一操作。没有根据结果追加轨迹修正、ρ、T2、长上下文或更大模型。

主体前后hash相同，原始资料SHA256 `{e['manifest']['test_data_sha256']}` 与登记一致。全部160条自由回答重判、96条成本网格及预算算术从原记录重算；归档含输入、输出、名单、manifest及运行源码，参数引用G3b相同bank。共享新增设备累计 {e['budget']['used_seconds']:.2f}/14400秒（含失败、预检、训练、推理、成本），新增现金0；旧G3a独立账保留。

报告和Notebook默认只读已有数据；整页展示验收范围见PROJECT_STATUS。本阶段及前两阶段已完成正式交付并进入主干。当前重建只更新阅读说明，不启动新的研究运行。
'''
    (ROOT/'reports/g4lite_mac_m4max.md').write_text(body)
    print(json.dumps({'run':s['run_id'],'accuracy':{m:(x['correct'],x['n']) for m,x in s['quality'].items()},'device_seconds':e['budget']['used_seconds']},ensure_ascii=False))

if __name__=='__main__':main()
