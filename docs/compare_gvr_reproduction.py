import json, zipfile, math
from importlib.metadata import version
from pathlib import Path
root=Path('outputs/gvr_repair_study')
z=zipfile.ZipFile('docs/gvr_repair_study/evidence.zip')
def old(p): return json.loads(z.read('outputs/gvr_repair_study/'+p))
def new(p): return json.loads((root/p).read_text())
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if 'seconds' not in k}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,float) and isinstance(b,(int,float)): return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    return a==b
report={'development_identical':new('dev_decision.json')['metrics']==old('dev_decision.json')['metrics'],
        'numeric_absolute_tolerance':1e-12,'excluded':'elapsed seconds; freeze identities differ by design',
        'packages':{p:version(p) for p in ('torch','transformers','peft')},'splits':{}}
for split in ('test_id','test_ood'):
    a,b=new(split+'/results.json'),old(split+'/results.json')
    groups=['generation','candidate_repair','verification','legacy_modes']
    report['splits'][split]={k:{'identical':equal(clean(a.get(k)),clean(b.get(k))), 'local':clean(a.get(k)), 'peer':clean(b.get(k))} for k in groups if k in a or k in b}
report['per_case_plans']={}
for name,key in [('test_id/generation_traces.jsonl','source_id'),('test_id/candidate_traces.jsonl','id'),('test_ood/candidate_traces.jsonl','id')]:
    a=[json.loads(s) for s in (root/name).read_text().splitlines()]
    b={r[key]:r for r in [json.loads(s) for s in z.read('outputs/gvr_repair_study/'+name).decode().splitlines()]}
    def plans(r):
        if 'methods' in r:return {'initial':r['initial'],**{k:v['plan'] for k,v in r['methods'].items()}}
        return {k:r[k]['plan'] for k in ('E1','B4','C4','U4','H4')}
    differences=[{'id':r[key],'methods':{m:{'local':p,'peer':plans(b[r[key]])[m]} for m,p in plans(r).items() if p!=plans(b[r[key]])[m]}} for r in a if plans(r)!=plans(b[r[key]])]
    report['per_case_plans'][name]={'n':len(a),'different':len(differences),'details':differences}
dest=Path('docs/gvr_local_gpu_comparison.json')
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'dev':report['development_identical'],**{s:{k:v['identical'] for k,v in r.items()} for s,r in report['splits'].items()}}))
