"""Serial frozen evaluation of every predeclared campaign arm, including negatives."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.data import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/campaign_20261103')
    args=parser.parse_args()
    root=Path(args.root)
    config_path=root/'baselines.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if not json.loads((root/'training_status.json').read_text()).get('complete'):
        raise ValueError('Complete train/dev phase first')
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    logs=root/'logs'
    status_path=root/'evaluation_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'stages':[]}
    def run(name,cmd,artifact):
        if Path(artifact).exists():
            print('REUSE',name,flush=True);return
        print('START',name,flush=True)
        with (logs/(name+'.log')).open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,'-u',*cmd],stdout=f,stderr=subprocess.STDOUT)
        status['stages'].append({'name':name,'returncode':result.returncode})
        write_json(status_path,status)
        if result.returncode:
            print((logs/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-5000:],flush=True)
            raise RuntimeError(name)
        print('DONE',name,flush=True)
    prefix_sources=[]
    for key in config['models']:
        full=root/'full_harness'/key
        spec_path=root/f'{key}_full_harness.json'
        write_json(spec_path,{'kind':'full_harness','config':str(config_path),
                             'base':str(Path(config['output_root'])/key/'train/best')})
        run(f'{key}_full_dev',['scripts/evaluate_campaign_model.py','evaluate','--spec',str(spec_path),'--split','dev','--out',str(full/'dev')],full/'dev/results.json')
        run(f'{key}_full_freeze',['scripts/evaluate_campaign_model.py','freeze','--spec',str(spec_path),'--dev-report',str(full/'dev/results.json'),'--out',str(full/'freeze.json')],full/'freeze.json')
        for split in ('test_id','test_ood'):
            run(f'{key}_full_{split}',['scripts/evaluate_campaign_model.py','evaluate','--spec',str(spec_path),'--split',split,'--freeze',str(full/'freeze.json'),'--out',str(full/split)],full/split/'results.json')
        prefix_sources.append((key,key,'full_finetune',full))
        for style in ('vanilla','structured','status'):
            protocol_path=root/f'{key}_{style}.json'
            gvr=root/'gvr'/key/style
            run(f'{key}_{style}_freeze',['scripts/freeze_experiment.py','freeze','--config',str(config_path),'--protocol',str(protocol_path),
                '--selection-report',str(gvr/'verify_dev.json'),'--note','Predeclared format ablation; checkpoint selection uses dev only','--out',str(gvr/'freeze.json')],gvr/'freeze.json')
            for split in ('dev','test_id','test_ood'):
                run(f'{key}_{style}_{split}',['scripts/run_gvr.py','evaluate','--config',str(config_path),'--protocol',str(protocol_path),
                    '--split',split,'--freeze',str(gvr/'freeze.json')],gvr/'evaluation'/split/'metrics.json')
            if style != 'status':
                prefix_sources.append((key+'_'+style,key,'lora_gvr',gvr/'evaluation'))
        for algorithm in ('reference','reinforce','grpo'):
            source_algorithm = 'reinforce' if algorithm == 'reference' else algorithm
            rl_spec=json.loads((root/f'{key}_{source_algorithm}.json').read_text())
            out=root/'rl_evaluation'/key/algorithm
            spec_path=root/f'{key}_{algorithm}_evaluation.json'
            write_json(spec_path,{'kind':'rl_generation','config':str(config_path),'base':rl_spec['base'],
                **({'adapter':str(Path(rl_spec['output'])/'best')} if algorithm != 'reference' else {}),
                'prompt_format':rl_spec['prompt_format'],'max_new_tokens':rl_spec['max_new_tokens']})
            run(f'{key}_{algorithm}_eval_dev',['scripts/evaluate_campaign_model.py','evaluate','--spec',str(spec_path),'--split','dev','--out',str(out/'dev')],out/'dev/results.json')
            run(f'{key}_{algorithm}_freeze',['scripts/evaluate_campaign_model.py','freeze','--spec',str(spec_path),'--dev-report',str(out/'dev/results.json'),'--out',str(out/'freeze.json')],out/'freeze.json')
            for split in ('test_id','test_ood'):
                run(f'{key}_{algorithm}_{split}',['scripts/evaluate_campaign_model.py','evaluate','--spec',str(spec_path),'--split',split,'--freeze',str(out/'freeze.json'),'--out',str(out/split)],out/split/'results.json')
    for name,key,family,traces in prefix_sources:
        spec_path=root/f'{name}_prefix.json'
        write_json(spec_path,{'family':family,'trace_format':'gvr','config':str(root/f'{key}_policy.json'),
            'dataset_root':config['dataset_root'],'max_actions':36,'symbolic_only':False,
            'traces':{s:str(traces/s/'pipeline_traces.jsonl') for s in ('dev','test_id','test_ood')}})
        out=root/'prefix'/name
        run(name+'_prefix_dev',['scripts/run_prefix_experiment.py','evaluate','--spec',str(spec_path),'--split','dev','--out',str(out/'dev')],out/'dev/results.json')
        run(name+'_prefix_freeze',['scripts/run_prefix_experiment.py','freeze','--spec',str(spec_path),'--dev-report',str(out/'dev/results.json'),'--out',str(out/'freeze.json')],out/'freeze.json')
        for split in ('test_id','test_ood'):
            run(name+'_prefix_'+split,['scripts/run_prefix_experiment.py','evaluate','--spec',str(spec_path),'--split',split,'--freeze',str(out/'freeze.json'),'--out',str(out/split)],out/split/'results.json')
    status['complete']=True
    write_json(status_path,status)


if __name__ == '__main__':main()
