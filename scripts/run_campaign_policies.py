"""Train same-backbone state policies; dev-only before freezing test comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ppnl.data import write_json
from ppnl.experiment_freeze import file_hash


def main():
    root=Path('outputs/campaign_20261103')
    baseline=json.loads((root/'baselines.json').read_text())
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    def run(name,cmd,artifact):
        if artifact.exists():
            print('REUSE',name,flush=True);return
        print('START',name,flush=True)
        log=root/'logs'/f'{name}.log'
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,'-u',*cmd],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text(encoding='utf-8',errors='replace')[-5000:],flush=True)
            raise RuntimeError(name)
        print('DONE',name,flush=True)
    for key in baseline['models']:
        config=json.loads(json.dumps(baseline))
        out=root/'policies'/key
        config.update(output_root=str(out),dataset_root='data/policy_v3',policy_source_dataset='data/derived',
                      policy_checkpoint=str(out/key/'train/best'),policy_local_status=True)
        config['training']['policy_loss']='optimal_actions'
        cfg=root/f'{key}_policy.json'
        write_json(cfg,config)
        for operation,file in (('pilot','pilot.json'),('train','training.json')):
            run(f'{key}_policy_{operation}',['-m','ppnl.cli','--config',str(cfg),operation,'--model',key],out/key/operation/file)
        spec=root/f'{key}_policy_evaluation.json'
        write_json(spec,{'kind':'state_policy','config':str(cfg),'base':config['policy_checkpoint']})
        run(f'{key}_policy_dev',['scripts/evaluate_campaign_model.py','evaluate','--spec',str(spec),'--split','dev','--out',str(out/'dev')],out/'dev/results.json')
        selection=out/'policy_selection.json'
        if not selection.exists():
            report=json.loads((out/'dev/results.json').read_text())
            methods=('raw','legal_no_revisit','harness_legal','harness_visit')
            chosen=max(methods,key=lambda name:(report['metrics'][name]['success_rate'],report['metrics'][name]['optimality']))
            write_json(selection,{'method':chosen,'weights_sha256':file_hash(Path(config['policy_checkpoint'])/'model.safetensors'),
                'dev_report_sha256':file_hash(out/'dev/results.json'),'rule':'Dev-only non-search success, optimality; ties keep simpler method'})
        run(f'{key}_policy_freeze',['scripts/evaluate_campaign_model.py','freeze','--spec',str(spec),'--dev-report',str(out/'dev/results.json'),'--out',str(out/'freeze.json')],out/'freeze.json')
        for split in ('test_id','test_ood'):
            run(f'{key}_policy_{split}',['scripts/evaluate_campaign_model.py','evaluate','--spec',str(spec),'--split',split,
                '--freeze',str(out/'freeze.json'),'--out',str(out/split)],out/split/'results.json')


if __name__=='__main__':main()
