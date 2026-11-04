"""Frozen full-model Harness or generation-only RL evaluation; no training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.core import Grid, OfficialOracle
from ppnl.data import read_jsonl, write_json, write_jsonl
from ppnl.experiment_freeze import audit_splits, _tree_hashes, file_hash, ROOT, write_new_json
from ppnl.prefix_evaluation import create_lock, validate_lock
from ppnl.prompts import check_checkpoint_prompt_format, prompt
from ppnl.harness import fingerprint


def identity_for(spec):
    config = json.loads(Path(spec['config']).read_text(encoding='utf-8'))
    style = spec.get('prompt_format', 'vanilla')
    check_checkpoint_prompt_format(spec['base'], style)
    dataset = config.get('policy_source_dataset', config['dataset_root'])
    audit = audit_splits(dataset)
    manifest = json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    oracle = OfficialOracle(manifest['executor_path'])
    if oracle.sha256 != manifest['executor_sha256']:
        raise ValueError('Executor drift')
    identity = {'spec':spec, 'config':config, 'split_audit':audit,
                'models':{'base':_tree_hashes(spec['base'])}, 'executor':oracle.sha256,
                'source':{p.relative_to(ROOT).as_posix():file_hash(p) for folder in ('ppnl','scripts')
                          for p in sorted((ROOT/folder).rglob('*.py'))}}
    if spec.get('adapter'):
        identity['models']['adapter'] = _tree_hashes(spec['adapter'])
    if spec['kind'] == 'state_policy':
        identity['policy_data'] = {name:file_hash(Path(config['dataset_root'])/name)
                                   for name in ('manifest.json','train.jsonl','dev.jsonl')}
    return config, oracle, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('evaluate','freeze'))
    parser.add_argument('--spec', required=True)
    parser.add_argument('--split', choices=('dev','test_id','test_ood'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--freeze')
    parser.add_argument('--dev-report')
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    if args.command == 'evaluate' and args.split is None:
        parser.error('--split required')
    if args.command == 'evaluate' and args.split != 'dev' and not args.freeze:
        parser.error('test inference requires --freeze')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    config, oracle, identity = identity_for(spec)
    # Store the full audit once in results.json, not in every trial manifest.
    runtime_identity = {'experiment_sha256': fingerprint(identity)}
    if args.command == 'freeze':
        if not args.dev_report: parser.error('--dev-report required')
        write_new_json(out, create_lock(identity, args.dev_report))
        return
    receipt = validate_lock(args.freeze, identity) if args.split != 'dev' else None
    rows = read_jsonl(Path(config.get('policy_source_dataset', config['dataset_root']))/(args.split+'.jsonl'))
    from ppnl.modeling import Seq2Seq, require_cuda, gpu_budget
    from ppnl.full_finetune_gvr import FullFinetuneGVRModel
    from ppnl.gvr_evaluation import evaluate_gvr
    from ppnl.evaluation import plan_metrics
    torch = require_cuda()
    torch.set_num_threads(1)
    with gpu_budget(config['gpu_ledger'], config['gpu_hours_cap'], f"campaign:{spec['kind']}:{args.split}") as check:
        runner = Seq2Seq(spec['base'], config)
        if spec['kind'] == 'full_harness':
            metrics = evaluate_gvr(FullFinetuneGVRModel(runner), rows, out, runtime_identity, oracle, check)
        elif spec['kind'] == 'rl_generation':
            if spec.get('adapter'):
                from peft import PeftModel
                runner.model = PeftModel.from_pretrained(runner.model, spec['adapter'])
            runner.output_limit = spec['max_new_tokens']
            selected = [r for r in rows if r.get('task') == 'generation'] if args.split == 'dev' else rows
            grids = [Grid.from_dict(r['grid']) for r in selected]
            plans = runner.predict([prompt('generation',g,input_format=spec['prompt_format']) for g in grids],check)
            for g,p in zip(grids,plans): oracle.check(g,p)
            metrics = {'generation':plan_metrics(grids,plans)}
            out.mkdir(parents=True,exist_ok=False)
            write_jsonl(out/'generation_traces.jsonl',[{'id':r['id'],'grid':r['grid'],'plan':p} for r,p in zip(selected,plans)])
        elif spec['kind'] == 'state_policy':
            from ppnl.policy import NeuralPolicy, greedy_plans, optimal_search_plans, search_plans
            from ppnl.harness import Budgets, create_session, run_session
            from ppnl.harness_backends import PolicyBackend
            selected = [r for r in rows if r.get('task') == 'generation'] if args.split == 'dev' else rows
            grids = [Grid.from_dict(r['grid']) for r in selected]
            scorer = NeuralPolicy(runner,check)
            methods = {'raw':greedy_plans(grids,scorer,False), 'legal_no_revisit':greedy_plans(grids,scorer,True)}
            methods['neural_best_first'] = search_plans(grids,scorer,weight=.25)[0]
            methods['neural_optimal_astar'] = optimal_search_plans(grids,scorer)[0]
            methods['symbolic_astar'] = optimal_search_plans(grids,None,goal_tiebreak=True)[0]
            for revisit in (False,True):
                name = 'harness_visit' if revisit else 'harness_legal'
                backend = PolicyBackend(scorer,True,revisit)
                plans=[]
                for row,g in zip(selected,grids):
                    check()
                    directory=out/'sessions'/name/row['id']
                    create_session(directory,g,Budgets(model_calls=36,tool_calls=36),runtime_identity,{'kind':'official','sha256':oracle.sha256})
                    state=run_session(directory,backend,runtime_identity,oracle,check)
                    plans.append(state['plan'])
                methods[name]=plans
            metrics={name:plan_metrics(grids,plans) for name,plans in methods.items()}
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/'policy_usage.json',scorer.usage)
            write_jsonl(out/'policy_traces.jsonl',[{'id':r['id'],'grid':r['grid'],
                         'plans':{name:plans[i] for name,plans in methods.items()}} for i,r in enumerate(selected)])
        else:
            raise ValueError('Unknown evaluation kind')
    report = {'split':args.split,'identity':identity,'metrics':metrics,'freeze_receipt':receipt,
              'test_status':'reused_exploratory'}
    write_json(out/'results.json',report)
    print(json.dumps(metrics),flush=True)


if __name__ == '__main__': main()
