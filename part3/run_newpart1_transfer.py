"""NewPart1 local reproduction: fresh V/G, original/enhanced neural R, all maps."""
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
from ppnl.policy import NeuralPolicy,greedy_plans
from ppnl.prompts import prompt
from ppnl.verifier_rules import RULES,enhance
from part3.run_feedback_backoff_sweep import cut_repair
from scripts.run_phase2_feedback import continue_plan

EXTRA=('valid_confirmation','terminal_valid_confirmation','eos_index','terminal_valid_eos')


def feedback_rule(g,c,p,rule):
    if rule=='terminal_valid_eos':
        p=feedback_rule(g,c,p,'terminal_valid_confirmation')
        return feedback_rule(g,c,p,'eos_index')
    if rule=='eos_index':
        parsed=parse_verdict(p)
        if parsed is not None and not parsed[0] and parsed[2]=='goal_not_reached':
            return f'invalid | step={len(c.split())+1} | reason=goal_not_reached'
        return p
    if rule in EXTRA:
        # Explicit full simulation for positive confirmation only. Never expose
        # the executor's first-error position to the model-source controller.
        if execute(g,c).valid:return 'valid'
        return enhance(g,c,p,'terminal' if rule=='terminal_valid_confirmation' else 'none')
    return enhance(g,c,p,rule)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compat-no-checkpoint',action='store_true')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    require_cuda()
    training=json.loads((a.checkpoint.parent/'training.json').read_text())
    assert training['updates']==500 and training['pretrained']['split_verification_loss']
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    files={'original_single':Path('data/policy_v3/test_id.jsonl'),'original_double':Path('data/policy_v3/test_ood.jsonl')}
    for size in ['7x7','dense_6x6']:
        files[size+'_single']=Path(f'data/generalization_v1/test_{size}.jsonl')
        files[size+'_double']=Path(f'data/generalization_double/test_{size}_double.jsonl')
    config=json.loads(Path('part2/protocol.json').read_text())
    rules=(*RULES,*EXTRA);spec={'mode':'beam_soft','score_cap':108,'erase':True}
    a.out.mkdir(parents=True)
    write_json(a.out/'protocol.json',{'peer_commit':'6939390','origin':'independent local reproduction, not peer checkpoint',
        'checkpoint_sha256':sha(a.checkpoint/'model.safetensors'),'training':training,
        'training_runtime_compatibility':{'activation_checkpoint_recomputation_disabled':a.compat_no_checkpoint,
            'reason':'two CUDA backward failures in unmodified peer runtime; same objective/data/budget; independent local reproduction'},
        'files_sha256':{str(f):sha(f) for f in files.values()},'code_sha256':{f:sha(f) for f in [__file__,'ppnl/verifier_rules.py','ppnl/repair_variants.py','scripts/run_phase2_feedback.py','ppnl/modeling.py']},
        'rules':rules,'old_r':'predicted prefix plus raw greedy, max36 new actions','new_r':{'backoff':36,'spec':spec},
        'v':'fresh vanilla, structured, same-model second-call review; executor source explicit',
        'new_rules':'valid_confirmation uses full positive-path simulation only, keeps model on failures; terminal variant also adds terminal rule. eos_index fixes claimed goal_not_reached step to action_count+1 without confirming the claimed reason; combined variant adds all three. Symbolic contribution, not neural reasoning.',
        'selection':'frozen methods; all tests exploratory; no selection on tests','generation':'fresh raw greedy36 per layout pool, shared single/double environments'})
    with zipfile.ZipFile(a.out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [__file__,'ppnl/verifier_rules.py','ppnl/repair_variants.py','scripts/run_phase2_feedback.py','ppnl/modeling.py']:z.write(f)
    results={};audit=0;throughput={}
    with gpu_budget(a.out/'gpu_ledger.json',2,'NewPart1 transfer all map old-new R') as check:
        runner=Seq2Seq(str(a.checkpoint),config);scorer=NeuralPolicy(runner,check)
        oracle=OfficialOracle('vendor/ppnl/executor-point-sg.py');newcache={}
        pools={name:read_jsonl(file) for name,file in files.items()}
        for name in ['original_single','7x7_single','dense_6x6_single']:
            rows=pools[name];gs=[Grid.from_dict(r['grid']) for r in rows]
            print('G',name,flush=True)
            plans=greedy_plans(gs,scorer,masked=False,max_steps=36)
            pools[name.removesuffix('_single')+'_generated']=[{**r,'candidate':c} for r,c in zip(rows,plans)]
        for name,rows in pools.items():
            grids=[Grid.from_dict(r['grid']) for r in rows];candidates=[r['candidate'] for r in rows]
            truth=[execute(g,c) for g,c in zip(grids,candidates)]
            predictions={}
            for style in ['vanilla','structured','review']:
                print('V',name,style,flush=True)
                inputs=[prompt('verification',g,c,input_format='structured' if style=='structured' else 'vanilla') for g,c in zip(grids,candidates)]
                if style=='review':inputs=[s+'\nYour previous diagnosis: '+v+'\nCheck the action indices and endpoint again. Return only the diagnosis in the required format.' for s,v in zip(inputs,predictions['vanilla'])]
                tick=time.monotonic()
                predictions[style]=runner.predict(inputs,check)
                elapsed=time.monotonic()-tick
                throughput[f'{name}:{style}']={'examples':len(inputs),'seconds':elapsed,
                    'examples_per_second':len(inputs)/elapsed,
                    'max_input_tokens':max(len(runner.tokenizer(s)['input_ids']) for s in inputs)}
                write_json(a.out/'v_throughput.json',throughput)
            predictions['executor']=[t.label() for t in truth]
            write_jsonl(a.out/f'{name}_inputs.jsonl',[{'id':r['id'],'grid':r['grid'],'candidate':c,'gold':t.label(),'predictions':{s:v[i] for s,v in predictions.items()}} for i,(r,c,t) in enumerate(zip(rows,candidates,truth))])
            bad=[i for i,t in enumerate(truth) if not t.valid];good=[i for i,t in enumerate(truth) if t.valid]
            results[name]={'direct':plan_metrics(grids,candidates)};oldcache={}
            for source,vs in predictions.items():
                for rule in rules:
                    feedbacks=[feedback_rule(g,c,v,rule) for g,c,v in zip(grids,candidates,vs)]
                    verification=verification_metrics([{'valid':t.valid,'step':t.step,'reason':t.reason} for t in truth],feedbacks)
                    for backend in ['old','new']:
                        outputs=[];traces=[]
                        for i,(g,c,fb) in enumerate(zip(grids,candidates,feedbacks)):
                            check()
                            if backend=='old':
                                key=(i,fb)
                                if key not in oldcache:oldcache[key]=continue_plan(g,c,fb,scorer,36,False)
                                out,status=oldcache[key]
                            else:out,status,_=cut_repair(g,c,fb,36,scorer,spec,newcache)
                            verdict=execute(g,out);assert oracle.check(g,out).label()==verdict.label();audit+=1
                            outputs.append(out);traces.append({'id':rows[i]['id'],'feedback':fb,'output':out,'status':status,'gold':truth[i].label(),'final':verdict.label()})
                        parsed=[parse_verdict(f) for f in feedbacks]
                        results[name][f'{source}+{rule}+{backend}']={'planning':plan_metrics(grids,outputs),
                            'invalid_only':plan_metrics([grids[i] for i in bad],[outputs[i] for i in bad]),'verification':verification,
                            'judgment_accuracy':sum(p is not None and p[0]==t.valid for p,t in zip(parsed,truth))/len(rows),
                            'false_accepts':sum(parsed[i] is not None and parsed[i][0] for i in bad),
                            'false_rejects':sum(parsed[i] is not None and not parsed[i][0] for i in good),
                            'valid_routes_broken':sum(not execute(grids[i],outputs[i]).valid for i in good),
                            'statuses':dict(Counter(t['status'] for t in traces))}
                        write_jsonl(a.out/f'{name}_{source}_{rule}_{backend}.jsonl',traces)
                write_json(a.out/'results.json',results);print(name,source,'done',flush=True)
        write_json(a.out/'completed.json',{'official_agreement_count':audit,'usage':scorer.usage,'training_updates':0,
            'peak_reserved_bytes':runner.torch.cuda.max_memory_reserved(),'peak_allocated_bytes':runner.torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
