"""Publish previously unshared selected historical checkpoints and compact evidence."""
import json
import os
from pathlib import Path
from export_campaign import ROOT, clean_lfs, sha, scrub

TARGET = ROOT/'artifacts/historical_models'
RUNS = ('bart_base','flan_t5_small','gvr_next','planning_v2','planning_v3','rlvr_reinforce_001')

def main():
    if (TARGET/'manifest.json').exists():
        raise FileExistsError('Historical export already complete')
    TARGET.mkdir(parents=True,exist_ok=True)
    records=[]
    weights=0
    for run in RUNS:
        for folder,dirs,names in os.walk(ROOT/'outputs'/run):
            dirs[:]=sorted(d for d in dirs if d not in ('sessions','__pycache__'))
            for name in sorted(names):
                src=Path(folder)/name
                if src.suffix not in ('.json','.jsonl','.log','.md','.txt','.safetensors','.model'):
                    continue
                relative=src.relative_to(ROOT/'outputs').as_posix()
                dst=TARGET/relative
                dst.parent.mkdir(parents=True,exist_ok=True)
                original=sha(src)
                if src.suffix=='.safetensors':
                    clean_lfs(src,dst)
                    assert f'oid sha256:{original}' in dst.read_text()
                    exported=original
                    weights+=1
                    print('HISTORICAL WEIGHT',weights,relative,flush=True)
                else:
                    raw=src.read_bytes()
                    try:
                        text=scrub(raw.decode('utf-8'))
                        if name=='adapter_config.json':
                            value=json.loads(text)
                            base=value.get('base_model_name_or_path','').replace('\\','/')
                            if base.startswith('outputs/gvr/'):
                                value['base_model_name_or_path']='artifacts/gvr/models/generation'
                            elif base.startswith('outputs/'):
                                value['base_model_name_or_path']=base.replace('outputs/','artifacts/historical_models/',1)
                            text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
                        raw=text.encode('utf-8')
                    except UnicodeDecodeError:
                        pass
                    dst.write_bytes(raw)
                    exported=sha(dst)
                records.append({'path':relative,'bytes':src.stat().st_size if src.suffix=='.safetensors' else dst.stat().st_size,
                                'source_sha256':original,'sha256':exported,
                                'storage':'git-lfs' if src.suffix=='.safetensors' else 'git',
                                'transformation':'byte-identical' if exported==original else 'machine metadata redacted / adapter base relocated'})
    assert weights==9,weights
    (TARGET/'manifest.json').write_text(json.dumps({'schema_version':1,'weight_files':weights,'files':records,
        'notes':['Historical selected checkpoints, distinct from the new three-model campaign.',
                 'Earlier GVR weights already published under artifacts/gvr are not duplicated.',
                 'Compact existing reports included; per-step session trees are omitted here.',
                 'Weights are byte-identical; metadata transformations are declared.']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Historical export complete:',weights,'weights,',len(records),'files',flush=True)

if __name__=='__main__': main()
