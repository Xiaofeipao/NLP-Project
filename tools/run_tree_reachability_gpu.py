"""Frozen-weight diagnostic and paired feedback comparison; no optimizer updates."""
import hashlib
import json
from pathlib import Path
import sys
import gc
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reachability_probe import load_cases, score
from run_tree_verifier import predict
from ppnl.core import Grid, execute
from ppnl.data import read_jsonl, write_json
from ppnl.prompts import prompt
from ppnl.evaluation import plan_metrics
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda
from ppnl.policy import NeuralPolicy
from ppnl.harness_backends import PolicyBackend
from ppnl.harness import Budgets, create_session, run_session
from ppnl.harness_backends import file_hash


def main():
    print('Starting frozen-weight GPU evaluation', flush=True)
    out=ROOT/'outputs/tree_reachability_gpu_001'
    out.mkdir(exist_ok=True)
    config=json.loads((ROOT/'artifacts/campaign_20261103/baselines.json').read_text())
    config['local_files_only']=True
    tree_path=ROOT/'outputs/tree_verifier_001/tree.joblib'
    tree=joblib.load(tree_path)
    grids,prompts,_=load_cases()
    torch=require_cuda()
    torch.set_num_threads(1)
    for key in ('t5_small','t5_base','bart_base'):
        destination=out/f'{key}.json'
        if destination.exists():
            print(f'{key}: complete report exists; skipping', flush=True)
            continue
        checkpoint=ROOT/'artifacts/campaign_20261103/baselines'/key/'train/best'
        policy_checkpoint=ROOT/f'artifacts/campaign_20261103/policies/{key}/{key}/train/best'
        for folder in (checkpoint,policy_checkpoint):
            if (folder/'model.safetensors').stat().st_size < 1024:
                raise RuntimeError(f'Restore Git LFS weights before evaluation: {folder.relative_to(ROOT)}')
        with gpu_budget(config['gpu_ledger'],config['gpu_hours_cap'],f'tree_reachability:{key}') as check:
            print(f'{key}: loading baseline', flush=True)
            runner=Seq2Seq(str(checkpoint),config)
            predictions=dict(zip(prompts,runner.predict(list(prompts.values()),check)))
            report={'kind':'diagnostic_only','probe':score(grids,predictions),'predictions':predictions,
                    'tree_sha256':hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                    'baseline_sha256':file_hash(checkpoint/'model.safetensors'),
                    'policy_sha256':file_hash(policy_checkpoint/'model.safetensors'),
                    'test_status':'eight public curated probes; reused repair tests'}
            if key=='t5_base':
                report['repair']={}
                for split in ('dev','test_id','test_ood'):
                    print(f'{key}: paired repair evaluation on {split}', flush=True)
                    rows=read_jsonl(ROOT/'data/derived'/f'{split}.jsonl')
                    rows=[r for r in rows if r.get('task')=='repair'] if split=='dev' else [r for r in rows if not r['valid']]
                    worlds=[Grid.from_dict(r['grid']) for r in rows]
                    feedback={'tree':[predict(tree,g,r['candidate']) for g,r in zip(worlds,rows)],
                              'executor':[execute(g,r['candidate']).label() for g,r in zip(worlds,rows)],
                              'model':runner.predict([prompt('verification',g,r['candidate']) for g,r in zip(worlds,rows)],check)}
                    comparison={}
                    for name,diagnoses in feedback.items():
                        plans=[r['candidate'] for r in rows]
                        indices=[i for i,d in enumerate(diagnoses) if d.strip()!='valid']
                        fixed=runner.predict([prompt('repair',worlds[i],rows[i]['candidate'],diagnoses[i]) for i in indices],check) if indices else []
                        for i,plan in zip(indices,fixed): plans[i]=plan
                        comparison[name]={'metrics':plan_metrics(worlds,plans),'repair_calls':len(indices),
                                          'cases':[{'id':r['id'],'feedback':d,'plan':p} for r,d,p in zip(rows,diagnoses,plans)]}
                    report['repair'][split]=comparison
            del runner
            gc.collect()
            torch.cuda.empty_cache()
            policy_config=json.loads((ROOT/f'artifacts/campaign_20261103/{key}_policy.json').read_text())
            policy_config['local_files_only']=True
            runner=Seq2Seq(str(policy_checkpoint),policy_config)
            print(f'{key}: evaluating policy on eight probes', flush=True)
            scorer=NeuralPolicy(runner,check)
            backend=PolicyBackend(scorer,True,True)
            policy_cases={}
            for case,grid in grids.items():
                directory=out/'sessions'/key/case
                identity={'experiment':'reachability_probe','model':key,'weight_sha256':report['policy_sha256']}
                create_session(directory,grid,Budgets(model_calls=36,tool_calls=36),identity,{'kind':'local'})
                state=run_session(directory,backend,identity,check=check)
                policy_cases[case]={'plan':state['plan'],'status':state['status'],
                                    'last_feedback':state['feedback'],'model_calls':state['model_calls']}
            report['policy_cases']=policy_cases
            report['policy_probe']=score(grids,{k:v['plan'] for k,v in policy_cases.items()})
            write_json(destination,report)
            print(key,json.dumps(report['probe']),json.dumps(report['policy_probe']),flush=True)
            del backend,scorer,runner
            gc.collect()
            torch.cuda.empty_cache()


if __name__=='__main__':main()
