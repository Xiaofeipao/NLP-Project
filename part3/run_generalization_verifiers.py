"""Generalization: verifier-only ablations, original prefix-preserving greedy R."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import zipfile

from ppnl.core import Grid, execute, parse_verdict, OfficialOracle
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.evaluation import plan_metrics, verification_metrics
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.policy import NeuralPolicy, greedy_plans
from ppnl.prompts import prompt
from ppnl.verifier_rules import RULES, enhance
from scripts.run_phase2_feedback import continue_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    require_cuda()
    checkpoint = Path('outputs/phase2_reproduction/run/train/flan_t5_base/best')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected = json.loads(Path('part3/best_rollback.json').read_text())['weights_sha256']
    assert sha(checkpoint/'model.safetensors') == expected
    config = json.loads(Path('part2/protocol.json').read_text())
    files = [Path('data/generalization_v1')/f'test_{s}.jsonl' for s in ['7x7','dense_6x6']]
    code = [__file__, 'ppnl/verifier_rules.py','scripts/run_phase2_feedback.py',
            'ppnl/core.py','ppnl/modeling.py','ppnl/policy.py','ppnl/prompts.py']
    args.out.mkdir(parents=True)
    write_json(args.out/'protocol.json', {
        'weights_sha256':expected,'data_sha256':{str(f):sha(f) for f in files},
        'code_sha256':{f:sha(f) for f in code},'rules':RULES,
        'verifiers':['vanilla','structured','review','executor'],
        'review':'same frozen V, second call with its vanilla answer; no executor evidence',
        'r':'original continue_plan; predicted prefix; raw greedy original action model; visit_guard=False; 36 NEW actions; one repair',
        'g':'raw unmasked greedy; 36 actions on both sizes; fresh GPU inference',
        'selection':'all prespecified groups reported; test-only; no tuning or training',
        'rule_scope':'all partial rules include token check; stop at unsafe transition; safety_terminal supplies all invalid diagnoses but keeps model false rejections',
        'executor_scope':'full exact diagnostic source; same rule transformations and same R',
        'precision':config['training']['precision'],'device':'cuda',
    })
    with zipfile.ZipFile(args.out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in code:z.write(f)
    results = {}
    started = time.monotonic()
    audit_count = 0
    oracle = OfficialOracle('vendor/ppnl/executor-point-sg.py')
    with gpu_budget(args.out/'gpu_ledger.json', 2, 'generalization verifier ablations original R') as check:
        runner = Seq2Seq(str(checkpoint), config)
        scorer = NeuralPolicy(runner,check)
        for file in files:
            rows = read_jsonl(file)
            assert len(rows)==200 and sum(not r['valid'] for r in rows)==100
            grids = [Grid.from_dict(r['grid']) for r in rows]
            print('G',file.stem,flush=True)
            generated = greedy_plans(grids,scorer,masked=False,max_steps=36)
            for pool,candidates in [('candidates',[r['candidate'] for r in rows]),('generated',generated)]:
                key=f'{file.stem}_{pool}'
                results[key]={'direct':plan_metrics(grids,candidates)}
                predictions={}
                for style in ('vanilla','structured','review'):
                    inputs=[prompt('verification',g,c,input_format='structured' if style=='structured' else 'vanilla') for g,c in zip(grids,candidates)]
                    if style=='review':
                        inputs=[s+'\nYour previous diagnosis: '+v+'\nCheck the action indices and endpoint again. Return only the diagnosis in the required format.' for s,v in zip(inputs,predictions['vanilla'])]
                    print('V',key,style,flush=True)
                    predictions[style]=runner.predict(inputs,check)
                truth=[execute(g,c) for g,c in zip(grids,candidates)]
                predictions['executor']=[v.label() for v in truth]
                write_jsonl(args.out/f'{key}_inputs.jsonl',[
                    {'id':r['id'],'grid':r['grid'],'candidate':c,'predictions':{s:vs[i] for s,vs in predictions.items()}}
                    for i,(r,c) in enumerate(zip(rows,candidates))])
                repair_cache={}
                executor_outputs=None
                for source,vs in predictions.items():
                    for rule in RULES:
                        name=f'{source}+{rule}'
                        outputs=[]; feedbacks=[]; traces=[]
                        for i,(g,c,v) in enumerate(zip(grids,candidates,vs)):
                            check()
                            fb=enhance(g,c,v,rule)
                            cachekey=(i,fb)
                            if cachekey not in repair_cache:
                                repair_cache[cachekey]=continue_plan(g,c,fb,scorer,max_steps=36,visit_guard=False)
                            out,status=repair_cache[cachekey]
                            outputs.append(out);feedbacks.append(fb)
                            final=execute(g,out)
                            assert oracle.check(g,out).label()==final.label()
                            audit_count+=1
                            traces.append({'id':rows[i]['id'],'prediction':v,'feedback':fb,'output':out,'status':status,'gold':truth[i].label(),'final':final.label()})
                        bad=[i for i,t in enumerate(truth) if not t.valid]
                        good=[i for i,t in enumerate(truth) if t.valid]
                        parsed=[parse_verdict(v) for v in feedbacks]
                        results[key][name]={
                            'planning':plan_metrics(grids,outputs),
                            'invalid_only':plan_metrics([grids[i] for i in bad],[outputs[i] for i in bad]),
                            'verification':verification_metrics([{'valid':t.valid,'step':t.step,'reason':t.reason} for t in truth],feedbacks),
                            'valid_controls':len(good),'invalid_controls':len(bad),
                            'false_accepts':sum(parsed[i] is not None and parsed[i][0] for i in bad),
                            'false_rejects':sum(parsed[i] is not None and not parsed[i][0] for i in good),
                            'valid_routes_broken':sum(not execute(grids[i],outputs[i]).valid for i in good),
                            'statuses':dict(Counter(t['status'] for t in traces)),
                        }
                        if source=='executor':
                            if executor_outputs is None:executor_outputs=outputs
                            assert outputs==executor_outputs,'Rules must not change exact diagnoses'
                        write_jsonl(args.out/f'{key}_{source}_{rule}_traces.jsonl',traces)
                        write_json(args.out/'results.json',results)
                    print(key,source,'done',flush=True)
        write_json(args.out/'audit.json',{'official_agreement_count':audit_count,'executor_rules_idempotent':True})
        write_json(args.out/'completed.json',{'seconds':time.monotonic()-started,'usage':scorer.usage,'training_updates':0})


if __name__=='__main__':
    main()
