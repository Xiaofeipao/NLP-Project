"""Export compact metrics/status only; exclude weights and machine-specific identity."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/campaign_20261103')
    parser.add_argument('--out',default='docs/gpu_campaign_results.json')
    args=parser.parse_args()
    root=Path(args.root)
    config=json.loads((root/'baselines.json').read_text(encoding='utf-8'))
    report={'models':{},'completed_reports':0,'expected_reports':99,
            'test_status':'reused_exploratory','default_deployment_changed':False,
            'limitation':'Single seed. Multiple independent candidates; no new blind holdout. All negatives retained.'}
    def metrics(path):
        if not path.exists(): return {'status':'missing'}
        value=json.loads(path.read_text(encoding='utf-8'))
        report['completed_reports']+=1
        return {'status':'completed','metrics':value.get('metrics',value)}
    for key,item in config['models'].items():
        row={'checkpoint':item['checkpoint'],'revision':item['revision'],'full_harness':{},'gvr':{},'rl':{},'policy':{},'prefix':{}}
        for split in ('dev','test_id','test_ood'):
            row['full_harness'][split]=metrics(root/'full_harness'/key/split/'results.json')
            row['policy'][split]=metrics(root/'policies'/key/split/'results.json')
        for style in ('vanilla','structured','status'):
            row['gvr'][style]={split:metrics(root/'gvr'/key/style/'evaluation'/split/'metrics.json') for split in ('dev','test_id','test_ood')}
        for algorithm in ('reference','reinforce','grpo'):
            trial=root/'rl'/key/algorithm/'report.json'
            value=json.loads(trial.read_text()) if trial.exists() else None
            row['rl'][algorithm]={'training':value,'evaluation':{split:metrics(root/'rl_evaluation'/key/algorithm/split/'results.json') for split in ('dev','test_id','test_ood')}}
            if value:
                improved=value['selected']['success_rate']>value['history'][0]['success_rate']
                row['rl'][algorithm]['disposition']='development_success_improved_candidate' if improved else 'negative_ablation_keep_reference'
        for name in (key,key+'_vanilla',key+'_structured'):
            row['prefix'][name]={split:metrics(root/'prefix'/name/split/'results.json') for split in ('dev','test_id','test_ood')}
        report['models'][key]=row
    ledger=json.loads(Path(config['gpu_ledger']).read_text())
    report['total_gpu_hours_in_shared_ledger']=sum(r['seconds'] for r in ledger['runs'])/3600
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'completed_reports':report['completed_reports'],'expected_reports':99,'output':str(out)},indent=2))


if __name__=='__main__':main()
