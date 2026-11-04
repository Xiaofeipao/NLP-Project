"""Fixed auxiliary-verdict-loss ablation, including BART after its BOS token."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ppnl.data import write_json


def main():
    root=Path('outputs/campaign_20261103')
    config=root/'baselines.json'
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1')
    for key in json.loads(config.read_text())['models']:
        style=json.loads((root/f'{key}_format_selection.json').read_text())['format']
        protocol=json.loads((root/f'{key}_{style}.json').read_text())
        out=root/'gvr'/key/'status'
        protocol['output_root']=str(out)
        protocol['verification_adapter']=str(out/'verification/train/best/verification')
        protocol['stages']['verification'].update(loss='sequence_and_status',status_weight=1.)
        path=root/f'{key}_status.json'
        write_json(path,protocol)
        for operation in ('pilot','train','verify'):
            artifact=out/'verify_dev.json' if operation=='verify' else out/'verification'/operation/'training.json'
            if artifact.exists(): continue
            print('START',key,'status',operation,flush=True)
            cmd=[sys.executable,'-u','scripts/run_gvr.py',operation,'--config',str(config),'--protocol',str(path)]
            if operation!='verify':cmd+=['--stage','verification']
            log=root/'logs'/f'{key}_status_{operation}.log'
            with log.open('w',encoding='utf-8') as f:
                result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text(encoding='utf-8',errors='replace')[-5000:],flush=True)
                raise RuntimeError(f'{key} status {operation}')
            print('DONE',key,'status',operation,flush=True)


if __name__=='__main__':main()
