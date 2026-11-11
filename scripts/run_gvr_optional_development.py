"""Paired dev-only inference ablations; no training and no test-set access."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.smoke_gvr_repair import ROOT, CHECKPOINT, preflight, WEIGHT_SHA256
from ppnl.core import Grid, OfficialOracle
from ppnl.data import write_json
from ppnl.evaluation import plan_metrics
from ppnl.experiment_freeze import file_hash
from ppnl.constrained_decoding import DecodeSpec
from ppnl.full_finetune_gvr import FullFinetuneGVRModel
from ppnl.modeling import Seq2Seq, gpu_budget, require_cuda

def main():
    out=ROOT/'outputs/gvr_optional_dev_001'
    if out.exists(): raise FileExistsError(out)
    config,manifest,generation,ledger,_=preflight(CHECKPOINT,40,'generation')
    repair=preflight(CHECKPOINT,60,'repair')[2]
    torch=require_cuda()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    oracle=OfficialOracle(ROOT/manifest['executor_path'])
    specs={'ordinary_original':(None,'original')}
    for method in ('constrained','backtrack'):
        for feedback in ('original','none','generic'):
            specs[f'{method}_{feedback}']=(DecodeSpec(backtrack=method=='backtrack',max_score_rows=148),feedback)
    out.mkdir()
    write_json(out/'protocol.json',{'checkpoint_sha256':WEIGHT_SHA256,'seed':42,'split':'dev',
        'generation_n':40,'repair_n':60,'beam_width':4,'max_actions':36,'max_score_rows':148,
        'methods':list(specs),'training':False,'test_access':False,'repairs':1,
        'budget_note':'Same scored-prefix ceiling for constrained/backtrack; tokens and time may differ. Ordinary is not compute-matched.',
        'source_sha256':{p.relative_to(ROOT).as_posix():file_hash(p) for d in ('ppnl','scripts') for p in (ROOT/d).glob('*.py')},
        'dev_sha256':file_hash(ROOT/'data/derived/dev.jsonl')})
    records=[]; summary={}
    with gpu_budget(ledger,20,'gvr_optional:dev_only') as check:
        runner=Seq2Seq(str(CHECKPOINT),config)
        models={k:FullFinetuneGVRModel(runner,d,f) for k,(d,f) in specs.items()}
        for task,rows in (('generation',generation),('repair',repair)):
            worlds=[]; predictions={k:[] for k in specs}; costs={k:{'seconds':0.,'decoder_row_steps':0,'decoder_input_tokens_processed':0} for k in specs}
            for row in rows:
                check(); grid=Grid.from_dict(row['grid']); worlds.append(grid)
                initial=models['ordinary_original'].predict('generation',grid,check=check) if task=='generation' else row['candidate']
                verdict=oracle.check(grid,initial)
                record={'task':task,'id':row['id'],'initial':initial,'feedback':verdict.label(),'methods':{}}
                for name,model in models.items():
                    start=time.perf_counter()
                    plan=initial if verdict.valid else model.predict('repair',grid,initial,verdict.label(),check)
                    elapsed=time.perf_counter()-start
                    usage={} if verdict.valid else dict(model.last_usage)
                    outcome=oracle.check(grid,plan)
                    record['methods'][name]={'plan':plan,'success':outcome.valid,'seconds':elapsed,'usage':usage}
                    predictions[name].append(plan)
                    costs[name]['seconds']+=elapsed
                    for field in ('decoder_row_steps','decoder_input_tokens_processed'): costs[name][field]+=usage.get(field,0)
                records.append(record)
                write_json(out/'progress.json',{'rows':records})
            summary[task]={k:{'metrics':plan_metrics(worlds,predictions[k]),'cost':costs[k]} for k in specs}
            write_json(out/'summary.json',summary)
            print(task,json.dumps({k:v['metrics']['success_rate'] for k,v in summary[task].items()}),flush=True)
    write_json(out/'results.json',{'summary':summary,'rows':records,'test_evaluated':False})

if __name__=='__main__': main()
