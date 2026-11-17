"""Sweep K=1..36 for fixed enhanced R, without oracle relocation of model feedback."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile

from part3.run_rollback import make_cases
from part2.run_rollback import reported_prefix
from ppnl.core import Grid, execute, shortest_path, OfficialOracle
from ppnl.data import write_json
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.policy import NeuralPolicy
from ppnl.repair_variants import repair
from ppnl.verifier_guard import guard_verdict

SOURCES=('model','vocabulary','terminal','executor')


def cut_repair(grid,candidate,feedback,k,scorer,spec,cache):
    prefix,status=reported_prefix(grid,candidate,feedback,k)
    if prefix is None:
        return candidate,status,0
    key=(grid.layout_id,grid.start,grid.goal,prefix)
    if key not in cache:
        control=f'invalid | step={len(prefix.split())+1} | reason=goal_not_reached'
        out,status,info=repair(grid,prefix,control,scorer,spec)
        cache[key]=(out,status,info['score_rows'])
    return cache[key]


def select_k(metrics):
    return max(range(1,37),key=lambda k:(metrics[str(k)]['success_rate'],
               metrics[str(k)]['optimality'],-metrics[str(k)]['score_rows'],-k))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    require_cuda()
    checkpoint=Path('outputs/phase2_reproduction/run/train/flan_t5_base/best')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected=json.loads(Path('part3/best_rollback.json').read_text())['weights_sha256']
    assert sha(checkpoint/'model.safetensors')==expected
    cases=make_cases()
    spec={'mode':'beam_soft','score_cap':108,'erase':True}
    code=[__file__,'part2/run_rollback.py','part3/run_rollback.py','ppnl/repair_variants.py',
          'ppnl/verifier_guard.py','ppnl/policy.py','ppnl/modeling.py','ppnl/core.py']
    args.out.mkdir(parents=True)
    write_json(args.out/'protocol.json',{
        'weights_sha256':expected,'k_range':[1,36],'sources':SOURCES,'backend':spec,
        'counts':{s:len(r) for s,r in cases.items()},
        'selection':'per feedback source, dev_candidates50 only: success, optimality, fewer score rows, smaller K; selected before evaluating tests',
        'controls':'candidate and V outputs replayed from same frozen checkpoint; GPU fresh R scoring; max36 new actions, one repair',
        'prefix':'predicted step only; invalid index rejected; illegal retained prefix rejected; never true-error correction',
        'scope':'exploratory reused test sets; all K reported for audit but not selected on tests; only dev winner promoted',
        'code_sha256':{f:sha(f) for f in code},
        'cases_sha256':{s:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for s,v in cases.items()},
    })
    with zipfile.ZipFile(args.out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in code:z.write(f)
    config=json.loads(Path('part2/protocol.json').read_text())
    results={};selected={}; audit=0
    started=time.monotonic()
    with gpu_budget(args.out/'gpu_ledger.json',2,'feedback source enhanced-R backoff 1..36') as check:
        runner=Seq2Seq(str(checkpoint),config)
        scorer=NeuralPolicy(runner,check)
        cache={}
        oracle=OfficialOracle('vendor/ppnl/executor-point-sg.py')
        for split,rows in cases.items():
            results[split]={}
            counters={source:{k:Counter() for k in range(1,37)} for source in SOURCES}
            with gzip.open(args.out/f'{split}_outcomes.jsonl.gz','wt',encoding='utf-8') as stream:
                for index,row in enumerate(rows):
                    g=Grid.from_dict(row['grid']);candidate=row['candidate']
                    bestlen=len(shortest_path(g))  # evaluation only; never passed to R
                    exact=oracle.check(g,candidate).label()
                    for source in SOURCES:
                        fb=exact if source=='executor' else guard_verdict(g,candidate,row['model_feedback'],source)
                        outputs=[]
                        for k in range(1,37):
                            check()
                            plan,status,cost=cut_repair(g,candidate,fb,k,scorer,spec,cache)
                            verdict=execute(g,plan)
                            assert oracle.check(g,plan).label()==verdict.label()
                            audit+=1
                            outcome={'k':k,'plan':plan,'status':status,'success':verdict.valid,
                                     'feasible':verdict.feasible,'optimal':verdict.valid and len(plan.split())==bestlen,'score_rows':cost}
                            outputs.append(outcome)
                            for metric in ('success','feasible','optimal','score_rows'):
                                counters[source][k][metric]+=outcome[metric]
                        stream.write(json.dumps({**row,'source':source,'feedback':fb,'outcomes':outputs})+'\n')
                    if (index+1)%25==0:print(split,index+1,'/',len(rows),flush=True)
            for source in SOURCES:
                results[split][source]={str(k):{'n':len(rows),
                    'success_rate':v['success']/len(rows),'feasibility':v['feasible']/len(rows),
                    'optimality':v['optimal']/len(rows),'score_rows':v['score_rows']} for k,v in counters[source].items()}
            if split=='dev_candidates':
                selected={s:select_k(results[split][s]) for s in SOURCES}
                write_json(args.out/'selection.json',{'source':'dev_candidates only','selected':selected})
                print('SELECTED',selected,flush=True)
            write_json(args.out/'results.json',results)
            print(split,'done',flush=True)
        write_json(args.out/'selected_results.json',{split:{s:{'k':selected[s],**groups[s][str(selected[s])]} for s in SOURCES} for split,groups in results.items()})
        write_json(args.out/'completed.json',{'seconds':time.monotonic()-started,'official_agreement_count':audit,'usage':scorer.usage,'training_updates':0})


if __name__=='__main__':main()
