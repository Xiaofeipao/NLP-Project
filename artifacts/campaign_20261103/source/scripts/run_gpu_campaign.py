"""Serial, resumable train/dev campaign; never evaluates a test split here."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppnl.data import write_json
from ppnl.experiment_freeze import audit_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='outputs/campaign_20261103')
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True,exist_ok=True)
    config_path = root / 'baselines.json'
    if not config_path.exists():
        template = json.loads((ROOT/'configs/campaign_20261103/baselines.json').read_text(encoding='utf-8'))
        template['output_root'] = str(root/'baselines')
        write_json(config_path,template)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    audit_splits(config['dataset_root'])
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    status_path = root / 'training_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'stages': []}

    def run(name, command, artifact):
        if Path(artifact).exists():
            print('REUSE', name, flush=True)
            return
        print('START', name, flush=True)
        started = time.monotonic()
        with (logs / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status['stages'].append({'name': name, 'returncode': result.returncode, 'seconds': time.monotonic()-started})
        write_json(status_path, status)
        if result.returncode:
            print((logs/(name+'.log')).read_text(encoding='utf-8', errors='replace')[-5000:], flush=True)
            raise RuntimeError(name)
        print('DONE', name, flush=True)

    for key in config['models']:
        for operation, filename in (('pilot','pilot.json'), ('train','training.json')):
            run(f'{key}_baseline_{operation}', ['-m','ppnl.cli','--config',str(config_path),operation,'--model',key],
                Path(config['output_root'])/key/operation/filename)
        run(f'{key}_baseline_dev', ['-m','ppnl.cli','--config',str(config_path),'evaluate','--model',key,'--split','dev'],
            Path(config['output_root'])/key/'eval_dev/metrics.json')
    run('baseline_select', ['-m','ppnl.cli','--config',str(config_path),'select'], Path(config['output_root'])/'selection.json')
    for key in config['models']:
        for style in ('vanilla','structured'):
            out = root / 'gvr' / key / style
            protocol = {'model_key': key, 'dataset_root': config['dataset_root'],
                        'task_index': 'data/derived/gvr_task_index.json', 'output_root': str(out),
                        'generation_base': str(out/'generation/train/best'),
                        'verification_adapter': str(out/'verification/train/best/verification'),
                        'repair_adapter': str(out/'repair/train/best/repair'), 'prompt_format': style,
                        'evidence': dict.fromkeys(('gpu_ledger_reconciled','checkpoint_confirmed','baseline_outputs_verified'), True),
                        'lora': {'r':8, 'alpha':16, 'dropout':.05},
                        'stages': {stage:{'selection':'task_metrics', **({'learning_rate':.001} if stage!='generation' else {})}
                                   for stage in ('generation','verification','repair')}}
            protocol_path = root/f'{key}_{style}.json'
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
                raise ValueError('Protocol changed: '+str(protocol_path))
            write_json(protocol_path, protocol)
            for stage in ('generation','verification','repair'):
                for operation in ('pilot','train'):
                    run(f'{key}_{style}_{stage}_{operation}', ['scripts/run_gvr.py',operation,'--config',str(config_path),
                        '--protocol',str(protocol_path),'--stage',stage], out/stage/operation/'training.json')
            run(f'{key}_{style}_verify', ['scripts/run_gvr.py','verify','--config',str(config_path),'--protocol',str(protocol_path)], out/'verify_dev.json')
        # Select the generation input format using only dev G success/optimality.
        scores = {}
        for style in ('vanilla','structured'):
            report = json.loads((root/'gvr'/key/style/'verify_dev.json').read_text())
            metric = report['dev_diagnostics']['generation']
            scores[style] = (metric['success_rate'], metric['optimality'])
        selected = max(('vanilla','structured'), key=scores.get)
        selection = {'format':selected,'dev_scores':scores,'rule':'dev G success, optimality; tie keeps vanilla'}
        write_json(root/f'{key}_format_selection.json', selection)
        for algorithm in ('reinforce','grpo'):
            out = root/'rl'/key/algorithm
            spec = {'config':str(config_path), 'base':str(root/'gvr'/key/selected/'generation/train/best'),
                    'output':str(out), 'prompt_format':selected, 'algorithm':algorithm,
                    'steps':64, 'group_size':4, 'max_new_tokens':48, 'eval_every':16,
                    'learning_rate':.00005,'kl_beta':.02, 'clip_epsilon':.2,
                    'updates_per_rollout':2 if algorithm=='grpo' else 1,
                    'lora':{'r':8,'alpha':16,'dropout':0.}}
            spec_path = root/f'{key}_{algorithm}.json'
            write_json(spec_path,spec)
            run(f'{key}_{algorithm}', ['scripts/run_rlvr.py','--spec',str(spec_path)], out/'report.json')
            run(f'{key}_{algorithm}_reload', ['scripts/verify_rlvr.py','--spec',str(spec_path)], out/'reload_check.json')
    status['complete'] = True
    write_json(status_path,status)


if __name__ == '__main__':
    main()
