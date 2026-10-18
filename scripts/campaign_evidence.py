"""Immutable compact archives for the G3c/G4 campaign; standard library only."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def package(stage,run,prior=()):
    run=Path(run);main=json.loads((run/'manifest.json').read_text())
    if main['status']!='done' or main['completed']!=main['planned']:
        raise ValueError('completed current stage required')
    files={};runs=[]
    for directory in [*map(Path,prior),run]:
        manifest=json.loads((directory/'manifest.json').read_text());runs.append(directory.name)
        for p in directory.iterdir():
            if p.is_file():files[directory.name+'/'+p.name]=p.read_bytes()
        for rel,digest in manifest['output_hashes'].items():
            if sha(files[directory.name+'/'+rel])!=digest:raise ValueError('raw run hash mismatch: '+rel)
        commit=manifest['git']['commit']
        for rel,digest in {**manifest['execution']['sources'],**manifest['execution']['configs']}.items():
            value=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)
            if sha(value)!=digest:raise ValueError('as-run source/config mismatch: '+rel)
            files[f'source/{commit}/{rel}']=value
    files['budget_at_stage_completion.json']=(ROOT/'runs/g3bg4lite-20261018-budget.json').read_bytes()
    archive=ROOT/f'reports/provenance/{stage}_evidence_20261018.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name,value in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;t.addfile(item,io.BytesIO(value))
    index=dict(archive=str(archive.relative_to(ROOT)),sha256=sha(archive.read_bytes()),formal_run=run.name,runs=runs,
               files={name:dict(bytes=len(value),sha256=sha(value)) for name,value in files.items()})
    dump(ROOT/f'reports/provenance/{stage}_archive_manifest.json',index)


def read(stage):
    index=json.loads((ROOT/f'reports/provenance/{stage}_archive_manifest.json').read_text())
    archive=ROOT/index['archive']
    if sha(archive.read_bytes())!=index['sha256']:raise ValueError('archive hash mismatch')
    with tarfile.open(archive) as t:files={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    if set(files)!=set(index['files']):raise ValueError('archive member mismatch')
    for name,value in files.items():
        if len(value)!=index['files'][name]['bytes'] or sha(value)!=index['files'][name]['sha256']:raise ValueError('member hash mismatch: '+name)
    manifests={}
    for run in index['runs']:
        m=json.loads(files[run+'/manifest.json']);manifests[run]=m
        for name,digest in m['output_hashes'].items():
            if sha(files[run+'/'+name])!=digest:raise ValueError('original run hash mismatch')
        for name,digest in {**m['execution']['sources'],**m['execution']['configs']}.items():
            if sha(files[f'source/{m["git"]["commit"]}/{name}'])!=digest:raise ValueError('original source hash mismatch')
    main=manifests[index['formal_run']]
    if main['status']!='done' or main['completed']!=main['planned']:raise ValueError('incomplete current stage')
    return dict(index=index,files=files,manifest=main,manifests=manifests,budget=json.loads(files['budget_at_stage_completion.json']))
