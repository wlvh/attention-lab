#!/usr/bin/env python3
"""Read/aggregate G3b records without torch; compact evidence excludes large audit arrays."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import statistics

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"reports/provenance/g3b_evidence_20261018.tar.gz"
INDEX=ROOT/"reports/provenance/g3b_archive_manifest.json"
METRICS=("kl","coverage","candidate_mass","missed_full_mass","output_rmse","output_relative_l2")


def sha(data):return hashlib.sha256(data).hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def aggregate(rows):
    """Recompute document means from raw query/group records, no published means as inputs."""
    from collections import defaultdict
    cells=defaultdict(lambda:defaultdict(list))
    for r in rows:
        for key in METRICS:
            if r[key] is None:continue
            cells[r["split"],r["doc_id"],r["method"]][key].extend(v for v,valid in zip(r[key],r["eligible"]) if valid or key not in ("kl","coverage"))
    docs=[dict(split=s,doc_id=d,method=m,**{k:statistics.mean(v) for k,v in values.items() if v}) for (s,d,m),values in sorted(cells.items())]
    result={}
    for split in ("train","development"):
        result[split]={}
        for method in ("C","D0","D","R"):
            subset=[r for r in docs if r["split"]==split and r["method"]==method]
            result[split][method]={key:statistics.mean(r[key] for r in subset if key in r) if any(key in r for r in subset) else None for key in METRICS}
    return result,docs


def package(run):
    manifest=json.loads((run/"manifest.json").read_text())
    if manifest["status"]!="done" or manifest["mode"]!="formal" or manifest["completed"]!=manifest["planned"]:
        raise ValueError("only complete formal G3b evidence can be published")
    files={}
    directories=sorted((ROOT/"runs").glob("g3b-*-preflight"))+[run]
    for directory in directories:
        for p in directory.rglob("*"):
            if p.is_file() and not p.name.startswith("audit-"):
                files[f'{directory.name}/{p.relative_to(directory)}']=p.read_bytes()
    files["budget_at_G3b_completion.json"]=(ROOT/"runs/g3bg4lite-20261018-budget.json").read_bytes()
    for path,digest in manifest["execution"]["sources"].items():
        content=subprocess.check_output(["git","show",f'{manifest["git"]["commit"]}:{path}'],cwd=ROOT)
        if sha(content)!=digest:raise ValueError(f"as-run source mismatch: {path}")
        files["source/"+path]=content
    for path,digest in manifest["execution"]["configs"].items():
        content=subprocess.check_output(["git","show",f'{manifest["git"]["commit"]}:{path}'],cwd=ROOT)
        if sha(content)!=digest:raise ValueError(f"as-run config mismatch: {path}")
        files["source/"+path]=content
    with tarfile.open(ARCHIVE,"w:gz") as tar:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            tar.addfile(info,io.BytesIO(data))
    dump(INDEX,{"archive":str(ARCHIVE.relative_to(ROOT)),"sha256":sha(ARCHIVE.read_bytes()),"formal_run":run.name,
                "files":{name:{"bytes":len(data),"sha256":sha(data)} for name,data in files.items()},
                "excluded":"audit-*.npz retained locally; remaining full features were hashed then released from RAM by preregistered disk-limited policy"})


def read_evidence():
    index=json.loads(INDEX.read_text())
    if sha(ARCHIVE.read_bytes())!=index["sha256"]:raise ValueError("archive hash mismatch")
    with tarfile.open(ARCHIVE) as tar:files={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    if set(files)!=set(index["files"]):raise ValueError("archive members differ")
    for name,data in files.items():
        if sha(data)!=index["files"][name]["sha256"] or len(data)!=index["files"][name]["bytes"]:raise ValueError(f"member mismatch: {name}")
    prefix=index["formal_run"]+"/"
    manifest=json.loads(files[prefix+"manifest.json"])
    if manifest["status"]!="done" or manifest["completed"]!=manifest["planned"]:raise ValueError("incomplete G3b")
    if set(manifest["layers"])!={str(i) for i in range(28)}:raise ValueError("missing layer")
    docs=[json.loads(s) for s in files[prefix+"data.jsonl"].splitlines()]
    by_layer=[];bank=[]
    for layer in range(28):
        meta=manifest["layers"][str(layer)]
        lp=prefix+f"layer-{layer:02d}/"
        for filename,rec in meta["files"].items():
            if filename.startswith("audit-"):continue
            if sha(files[lp+filename])!=rec["sha256"]:raise ValueError("original file hash mismatch")
        rows=[json.loads(s) for s in files[lp+"metrics.jsonl"].splitlines()]
        keys=[(r["doc_id"],r["query_index"],r["method"]) for r in rows]
        expected={(d["doc_id"],q,m) for d in docs for q in range(4) for m in ("C","D0","D","R")}
        if set(keys)!=expected or len(keys)!=len(expected):raise ValueError("missing/duplicate query records")
        agg,by_doc=aggregate(rows)
        old=json.loads(files[lp+"metrics_summary.json"])
        if agg!=old["aggregate"]:raise ValueError("stored aggregate differs from raw records")
        by_layer.append({"layer_index":layer,"origin":meta["origin"],"training_state":meta.get("training",{}).get("training_state","reused_G3a"),
                         "training":meta.get("training"),"aggregate":agg,"by_document":by_doc})
        bank.append({"layer_index":layer,"initial_file":f'layer-{layer:02d}/index_initial.npz',"final_file":f'layer-{layer:02d}/index_final.npz',
                     "initial_sha256":sha(files[lp+"index_initial.npz"]),"final_sha256":sha(files[lp+"index_final.npz"]),"origin":meta["origin"]})
    for path,digest in manifest["execution"]["sources"].items():
        if sha(files["source/"+path])!=digest:raise ValueError("source snapshot mismatch")
    return {"index":index,"manifest":manifest,"files":files,"layers":by_layer,"bank":bank,
            "budget":json.loads(files["budget_at_G3b_completion.json"])}


def summarize(evidence):
    layers=evidence["layers"]
    return {"run_id":evidence["index"]["formal_run"],"run_commit":evidence["manifest"]["git"]["commit"],
            "report_source_sha256":sha(Path(__file__).read_bytes()),"archive_sha256":evidence["index"]["sha256"],
            "scope":"28 layers on frozen full-path features; old development documents, not a new independent validation set",
            "layers":layers,"budget_at_completion":evidence["budget"],
            "development_improved_layers":{k:sum(r["aggregate"]["development"]["D"][k] is not None and
                (r["aggregate"]["development"]["D"][k]>r["aggregate"]["development"]["D0"][k] if k=="coverage" else r["aggregate"]["development"]["D"][k]<r["aggregate"]["development"]["D0"][k]) for r in layers) for k in ("kl","coverage","output_relative_l2")}}


def table(summary):
    origin_labels = {"reused_G3a": "复用第14层参数", "new_training": "本阶段新训练"}
    lines=["| 层 | 来源 | 开发 KL：D0→D | 开发 coverage：D0→D | 开发相对输出误差：D0→D | m_C |","|---|---|---:|---:|---:|---:|"]
    fmt=lambda x:"NA" if x is None else f"{x:.4f}"
    for layer in summary["layers"]:
        a,b=layer["aggregate"]["development"]["D0"],layer["aggregate"]["development"]["D"]
        lines.append(f'| {layer["layer_index"]+1} | {origin_labels[layer["origin"]]} | '+" | ".join(f'{fmt(a[k])} → {fmt(b[k])}' for k in ("kl","coverage","output_relative_l2"))+f' | {fmt(b["candidate_mass"])} |')
    return "\n".join(lines)


def plot_layers(summary):
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,layout="constrained")
    colors={"C":"#999999","D0":"#e69f00","D":"#0072b2","R":"#009e73"}
    for ax,key,label in zip(axes,("kl","coverage","output_relative_l2"),("KL (nats), lower","Conditional coverage, higher","Relative output L2, lower")):
        for method in ("C","D0","D","R"):
            if key=="kl" and method=="R":continue
            values=[r["aggregate"]["development"][method][key] for r in summary["layers"]]
            ax.plot(range(1,29),[np.nan if v is None else v for v in values],"o-",markersize=3,color=colors[method],label=method)
        ax.axvspan(13.6,14.4,color="black",alpha=.08)
        ax.set(ylabel=label,ylim=(0,None));ax.legend(ncol=4);ax.grid(axis="y",alpha=.2)
    axes[0].set_title("Previously inspected development documents; layer 14 reused from G3a")
    axes[-1].set(xlabel="Transformer layer (1-based)",xticks=list(range(1,29)))
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--package-run",type=Path);args=p.parse_args()
    if args.package_run:package(args.package_run)
    evidence=read_evidence();summary=summarize(evidence)
    dump(ROOT/"reports/g3b_summary_mac_m4max.json",summary)
    bank={"run_id":summary["run_id"],"source_commit":summary["run_commit"],"model_id":evidence["manifest"]["config"]["model_id"],
          "revision":evidence["manifest"]["config"]["revision"],"index_dtype":"float32","r":32,"layers":evidence["bank"]}
    bank["bank_id"]=sha(json.dumps(bank,sort_keys=True).encode())
    dump(ROOT/"reports/g3b_parameters.json",bank)
    body=f'''# G3b-lite：各层索引器训练

完整模型为每一层分别产生固定 Q/K/V，随后只用这批数组训练该层自己的索引器。保存这一层参数和指标后，释放临时特征，再处理下一层；前一层训练好的索引器没有接回主体改变后一层的教师。

T1 要求按资料编号取一个四位值。C 均匀随机选择候选，D0 是训练前索引，D 是训练后索引，R 直接按完整教师权重选择。它们都必留开头 4 个和最近 64 个位置，额外名额按请求第一步长度的 1/8 固定。索引位置设置 P0 指使用主 RoPE 前 Q/K、索引空间不额外旋转；r=32 是投影宽度。

下表比较同一份完整路径特征上的局部计算。条件 coverage 是入选候选保留的教师权重占候选总权重的比例，m_C 是候选的原始总权重，相对输出误差在输出投影之前计算。D0→D 表示训练前后；并非整模型答题成绩。逐层过程见[第五课](05_training_all_layer_indexers.html)。

28层均交付参数：27层新训练，第14层复用已审G3a。主体冻结，P0/r32/rho=1/8，索引float32；每层原始训练规则固定50轮。旧48份用于训练，旧16份仅为开发诊断，**不能再称新的独立验证集**。32份全新最终T1测试已登记，尚未在本阶段评分。

{table(summary)}

## 原预测与解释

预测见 `docs/tasks/07_G3B_LITE.md`：多数层拟合可能改善，但幅度不同，不保证共同部署后回答恢复。这里只观察冻结完整路径特征；下一阶段检查真实稀疏状态变化，不做轨迹修正训练。

每个数字先在同一资料内平均查询/组，再平均资料。m_C很小时条件coverage的意义受限；不足1e-4的行不训练并记录跳过计数。R是昂贵的完整教师选位诊断，不是任务质量上界。

## 证据与预算

正式运行 `{summary['run_id']}`，源码 `{summary['run_commit']}`。各层均报告，不挑好层；开发指标改善层数为 {summary['development_improved_layers']}。局部指标不直接代表28层共同稀疏推理的质量。

主体前后hash相同：`{evidence['manifest']['body_hash_before']}`。归档保留每层全部逐查询指标、初始/最终参数、预定审计资料的名单、训练曲线、源码/配置与原始manifest。因磁盘约束，完整特征逐层在RAM处理，仅每层资料0/48落盘供CPU审计；其余数组按预登记规则记录hash后释放，不声称精简归档能直接重做全部特征级审计。旧G3a特征未删除。

共享新增预算到本阶段结束累计 {evidence['budget']['used_seconds']:.1f}/14400 秒，现金0。此前G3a账本独立保留。参数bank身份 `{bank['bank_id']}` 在后续G3c中绑定缓存，不能在参数变化后继续复用旧索引z。

默认报告/Notebook只读保存结果；整页展示的实际验收范围见PROJECT_STATUS。本阶段已经交付；后续状态对照和新题评测已在第六、七课完成。旧登记状态只描述当时，不是当前项目待办。
'''
    (ROOT/"reports/g3b_mac_m4max.md").write_text(body)
    print(json.dumps({"run":summary["run_id"],"improved_layers":summary["development_improved_layers"],"bank_id":bank["bank_id"]},ensure_ascii=False))


if __name__=="__main__":main()
