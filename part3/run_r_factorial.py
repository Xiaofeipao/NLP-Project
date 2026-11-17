"""GPU 2x2x2 R ablation: branch width, cycle penalty, loop erasure; K=36."""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time
import zipfile

from part3.run_rollback import make_cases
from part3.run_feedback_backoff_sweep import cut_repair,SOURCES
from ppnl.core import Grid,execute,shortest_path,OfficialOracle
from ppnl.data import write_json
from ppnl.modeling import Seq2Seq,gpu_budget,require_cuda
from ppnl.policy import NeuralPolicy
from ppnl.verifier_guard import guard_verdict


def variants():
    return {f'T{t}P{p}E{e}':{'mode':('beam_soft' if p else 'beam') if t else ('soft' if p else 'legal'),
            'score_cap':108,'erase':bool(e)} for t,p,e in itertools.product((0,1),repeat=3)}


def choose(groups):
    return max(groups,key=lambda n:(groups[n]['success_rate'],groups[n]['optimality'],
               -groups[n]['score_rows'],-sum(map(int,n[1::2])),n))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    require_cuda()
    checkpoint=Path('outputs/phase2_reproduction/run/train/flan_t5_base/best')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected=json.loads(Path('part3/best_rollback.json').read_text())['weights_sha256']
    assert sha(checkpoint/'model.safetensors')==expected
    cases=make_cases();specs=variants()
    # Separate budget control is not included in the factorial selection.
    specs['budget36']={**specs['T1P1E1'],'score_cap':36}
    code=[__file__,'part3/run_feedback_backoff_sweep.py','part3/run_rollback.py',
          'part2/run_rollback.py','ppnl/repair_variants.py','ppnl/verifier_guard.py','ppnl/policy.py','ppnl/modeling.py','ppnl/core.py']
    args.out.mkdir(parents=True)
    write_json(args.out/'protocol.json',{
        'weights_sha256':expected,'backoff':36,'variants':specs,'sources':SOURCES,
        'counts':{s:len(v) for s,v in cases.items()},
        'selection':'dev_candidates50 only: success, optimality, fewer score rows, fewer enabled features; budget36 excluded from selection',
        'fixed':'same replayed V diagnoses and checkpoint; 36 new actions; one repair; legal action filtering fixed across all 8 variants',
        'budget':'common upper bound108 scored states; greedy uses at most36; budget36 extra control for triple-on beam',
        'none':'T0P0E0 is legal greedy with K36, not original unmasked prefix-preserving Part2 R',
        'code_sha256':{f:sha(f) for f in code},
        'cases_sha256':{s:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for s,v in cases.items()},
        'scope':'exploratory reused tests; no training; no test re-selection',
    })
    with zipfile.ZipFile(args.out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in code:z.write(f)
    config=json.loads(Path('part2/protocol.json').read_text())
    results={};selected={};audit=0;started=time.monotonic()
    with gpu_budget(args.out/'gpu_ledger.json',2,'R factorial top3 penalty erasure') as check:
        runner=Seq2Seq(str(checkpoint),config);scorer=NeuralPolicy(runner,check)
        oracle=OfficialOracle('vendor/ppnl/executor-point-sg.py')
        caches={n:{} for n in specs} # each cache is isolated by algorithm spec
        for split,rows in cases.items():
            counters={s:{n:Counter() for n in specs} for s in SOURCES}
            with gzip.open(args.out/f'{split}_outcomes.jsonl.gz','wt',encoding='utf-8') as stream:
                for i,r in enumerate(rows):
                    g=Grid.from_dict(r['grid']);c=r['candidate']
                    bestlen=len(shortest_path(g));exact=oracle.check(g,c).label()
                    for source in SOURCES:
                        fb=exact if source=='executor' else guard_verdict(g,c,r['model_feedback'],source)
                        outcomes={}
                        for name,spec in specs.items():
                            check()
                            plan,status,cost=cut_repair(g,c,fb,36,scorer,spec,caches[name])
                            v=execute(g,plan)
                            assert oracle.check(g,plan).label()==v.label();audit+=1
                            outcome={'plan':plan,'status':status,'success':v.valid,'feasible':v.feasible,
                                     'optimal':v.valid and len(plan.split())==bestlen,'score_rows':cost}
                            outcomes[name]=outcome
                            for field in ('success','feasible','optimal','score_rows'):
                                counters[source][name][field]+=outcome[field]
                        for t,p in itertools.product((0,1),repeat=2):
                            a=outcomes[f'T{t}P{p}E0'];b=outcomes[f'T{t}P{p}E1']
                            assert a['success']==b['success'] and a['feasible']==b['feasible']
                            assert not a['optimal'] or b['optimal']
                        stream.write(json.dumps({**r,'source':source,'feedback':fb,'outcomes':outcomes})+'\n')
                    if (i+1)%25==0:print(split,i+1,'/',len(rows),flush=True)
            results[split]={s:{n:{'n':len(rows),'success_rate':v['success']/len(rows),
                'feasibility':v['feasible']/len(rows),'optimality':v['optimal']/len(rows),
                'score_rows':v['score_rows']} for n,v in counters[s].items()} for s in SOURCES}
            if split=='dev_candidates':
                selected={s:choose({n:v for n,v in results[split][s].items() if n!='budget36'}) for s in SOURCES}
                write_json(args.out/'selection.json',{'source':'dev_candidates only','selected':selected})
                print('SELECTED',selected,flush=True)
            write_json(args.out/'results.json',results)
            print(split,'done',flush=True)
        write_json(args.out/'selected_results.json',{split:{s:{'variant':selected[s],**groups[s][selected[s]]} for s in SOURCES} for split,groups in results.items()})
        write_json(args.out/'completed.json',{'seconds':time.monotonic()-started,'official_agreement_count':audit,
            'erase_invariants_pass':True,'usage':scorer.usage,'training_updates':0})


if __name__=='__main__':main()
