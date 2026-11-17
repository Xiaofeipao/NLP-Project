"""Paired old/new R on single/double-error size/density shifts; frozen V variants."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

from ppnl.core import Grid,execute,parse_verdict,OfficialOracle
from ppnl.data import read_jsonl,write_json,write_jsonl
from ppnl.evaluation import plan_metrics,verification_metrics
from ppnl.modeling import Seq2Seq,gpu_budget,require_cuda
from ppnl.policy import NeuralPolicy
from ppnl.prompts import prompt
from ppnl.verifier_rules import RULES,enhance
from part3.run_feedback_backoff_sweep import cut_repair
from scripts.run_phase2_feedback import continue_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    require_cuda()
    checkpoint=Path('outputs/phase2_reproduction/run/train/flan_t5_base/best')
    expected=json.loads(Path('part3/best_rollback.json').read_text())['weights_sha256']
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    assert sha(checkpoint/'model.safetensors')==expected
    previous=Path('outputs/generalization_verifiers/run_001')
    prev_protocol=json.loads((previous/'protocol.json').read_text())
    assert prev_protocol['weights_sha256']==expected
    datasets={}
    for name in ['7x7','dense_6x6']:
        datasets[f'{name}_single']=Path(f'data/generalization_v1/test_{name}.jsonl')
        datasets[f'{name}_double']=Path(f'data/generalization_double/test_{name}_double.jsonl')
    code=[__file__,'scripts/build_generalization_double.py','ppnl/verifier_rules.py',
          'scripts/run_phase2_feedback.py','part3/run_feedback_backoff_sweep.py',
          'part2/run_rollback.py','ppnl/repair_variants.py','ppnl/policy.py','ppnl/modeling.py','ppnl/core.py','ppnl/prompts.py']
    spec={'mode':'beam_soft','score_cap':108,'erase':True}
    args.out.mkdir(parents=True)
    write_json(args.out/'protocol.json',{'weights_sha256':expected,'data_sha256':{str(f):sha(f) for f in datasets.values()},
        'code_sha256':{f:sha(f) for f in code},'rules':RULES,'v_sources':['vanilla','structured','review','executor'],
        'new_r':{'backoff':36,'backend':spec},'old_r':'continue_plan raw greedy predicted prefix, visit_guard=False',
        'max_new_actions':36,'single_v':'exact cached predictions from prior same-checkpoint GPU run; candidate/grid equality required',
        'double_v':'fresh GPU V inference; vanilla/structured/same-model review; no executor evidence in review',
        'selection':'no selection, training or tuning; frozen R and V; optional test only',
        'input':'512 token cap, no truncation','precision':'float32','source_prediction_sha256':{
            n:sha(previous/f'test_{n}_candidates_inputs.jsonl') for n in ['7x7','dense_6x6']}})
    with zipfile.ZipFile(args.out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in code:z.write(f)
    results={};audit=0;started=time.monotonic()
    with gpu_budget(args.out/'gpu_ledger.json',2,'generalization single-double old-new R paired comparison') as check:
        runner=Seq2Seq(str(checkpoint),json.loads(Path('part2/protocol.json').read_text()))
        scorer=NeuralPolicy(runner,check);oracle=OfficialOracle('vendor/ppnl/executor-point-sg.py')
        newcache={}
        for dataset,file in datasets.items():
            rows=read_jsonl(file);assert len(rows)==200
            grids=[Grid.from_dict(r['grid']) for r in rows];candidates=[r['candidate'] for r in rows]
            predictions={}
            if dataset.endswith('_single'):
                name=dataset.removesuffix('_single')
                saved=read_jsonl(previous/f'test_{name}_candidates_inputs.jsonl')
                assert [(r['id'],r['grid'],r['candidate']) for r in rows]==[(r['id'],r['grid'],r['candidate']) for r in saved]
                predictions={s:[r['predictions'][s] for r in saved] for s in ['vanilla','structured','review']}
            else:
                for style in ['vanilla','structured','review']:
                    print('V',dataset,style,flush=True)
                    inputs=[prompt('verification',g,c,input_format='structured' if style=='structured' else 'vanilla') for g,c in zip(grids,candidates)]
                    if style=='review':inputs=[s+'\nYour previous diagnosis: '+v+'\nCheck the action indices and endpoint again. Return only the diagnosis in the required format.' for s,v in zip(inputs,predictions['vanilla'])]
                    predictions[style]=runner.predict(inputs,check)
            truth=[execute(g,c) for g,c in zip(grids,candidates)]
            predictions['executor']=[v.label() for v in truth]
            write_jsonl(args.out/f'{dataset}_inputs.jsonl',[{**r,'predictions':{s:v[i] for s,v in predictions.items()}} for i,r in enumerate(rows)])
            bad=[i for i,t in enumerate(truth) if not t.valid];good=[i for i,t in enumerate(truth) if t.valid]
            assert len(bad)==len(good)==100
            results[dataset]={};oldcache={}
            for source,vs in predictions.items():
                for rule in RULES:
                    feedbacks=[enhance(g,c,v,rule) for g,c,v in zip(grids,candidates,vs)]
                    parsed=[parse_verdict(f) for f in feedbacks]
                    verification=verification_metrics([{'valid':v.valid,'step':v.step,'reason':v.reason} for v in truth],feedbacks)
                    accuracy=sum(p is not None and p[0]==v.valid for p,v in zip(parsed,truth))/200
                    for backend in ['old','new']:
                        name=f'{source}+{rule}+{backend}';plans=[];traces=[]
                        for i,(g,c,fb) in enumerate(zip(grids,candidates,feedbacks)):
                            check()
                            if backend=='old':
                                key=(i,fb)
                                if key not in oldcache:oldcache[key]=continue_plan(g,c,fb,scorer,max_steps=36,visit_guard=False)
                                plan,status=oldcache[key]
                            else:plan,status,_=cut_repair(g,c,fb,36,scorer,spec,newcache)
                            v=execute(g,plan);assert oracle.check(g,plan).label()==v.label();audit+=1
                            plans.append(plan);traces.append({'id':rows[i]['id'],'prediction':vs[i],'feedback':fb,'output':plan,'status':status,'gold':truth[i].label(),'final':v.label()})
                        results[dataset][name]={'planning':plan_metrics(grids,plans),
                            'invalid_only':plan_metrics([grids[i] for i in bad],[plans[i] for i in bad]),
                            'verification':verification,'judgment_accuracy':accuracy,
                            'false_accepts':sum(parsed[i] is not None and parsed[i][0] for i in bad),
                            'false_rejects':sum(parsed[i] is not None and not parsed[i][0] for i in good),
                            'valid_routes_broken':sum(not execute(grids[i],plans[i]).valid for i in good),
                            'statuses':dict(Counter(t['status'] for t in traces))}
                        write_jsonl(args.out/f'{dataset}_{source}_{rule}_{backend}_traces.jsonl',traces)
                print(dataset,source,'done',flush=True)
                write_json(args.out/'results.json',results)
        write_json(args.out/'completed.json',{'seconds':time.monotonic()-started,'official_agreement_count':audit,'usage':scorer.usage,'training_updates':0})


if __name__=='__main__':main()
