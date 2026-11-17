"""Weight-free audit/export for the isolated NewPart1 transfer study."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from ppnl.core import Grid,execute
from ppnl.data import read_jsonl,write_json
from ppnl.evaluation import plan_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();assert (a.run/'completed.json').exists()
    results=json.loads((a.run/'results.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    for category in ['code_sha256','files_sha256']:
        for f,h in protocol[category].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    export=[];failure_audit={};checked=0
    lines=['# NewPart1新代码：本地复现与V/R跨地图对照','',
        '同伴分支NewPart1，提交6939390。主要改动：验证首个valid/invalid token CE和后续token平均CE各占一半，其他动作损失和两轮500更新协议不变。同伴未上传该模型实体权重，因此本轮在outputs/newpart1_code_6939390隔离导出源码，用冻结预训练revision独立复现。不是同伴原权重，不覆盖原实验。训练仅在开发集按原规则选epoch，测试全部复用且标为探索，不在新测试选策略。','',
        '## 诊断与R边界','',
        'V包括原提示、结构化提示、同V答案追加复核一轮；执行器作为准确诊断对照。原7组规则定义沿用此前报告。新增valid_confirmation：完整模拟确认路线确实合法且到达终点时覆盖为valid，对失败路径保留模型原诊断；terminal_valid_confirmation再加终点增强。新增规则直接使用符号正例确认，不能宣称模型V自身能力提升。','',
        '运行兼容性：同伴原运行方式在本机反向传播中两次出现CUDA错误，失败耗时单独保留。最终独立复现关闭激活检查点重计算，损失、数据、模型和更新预算不变；tiny T5 CPU梯度一致性检查通过，不宣称已确定CUDA根因。源码差异与权重哈希可追溯。','',
        '原始R：预测首错前缀＋同一动作模型原始贪心续走。新R：回退36步＋Top3＋软循环惩罚＋最终删环。两者最多新增36动作，一次修复；新R有合法动作过滤和最多108状态评分，成本不同。V输出valid不调用R，错误步号或非法前缀只拒绝，不用执行器真首错纠正。模型来源的reason文本并不语义输入续走策略。','',
        '新增规则：valid_confirmation仅对完整模拟确认为成功的路径改为valid，失败时保留模型；terminal_valid_confirmation再加原终点规则。eos_index只把模型声称goal_not_reached的索引规范为动作数+1，不证明模型原因正确；terminal_valid_eos将上述三项组合。前两项含明确符号执行贡献，最后的索引规范也可能触发从起点重规划，不能视为模型学会定位。','',
        '## 数据与指标','',
        '原6×6 ID/OOD与额外7×7、密集6×6均各200固定候选：100有效、100单错/双错，单/双地图与有效控制配对。额外组仅测试，不混入必做数量。另对三类地图各200环境新生成G路线，再诊断修复；这不是额外独立地图。候选修复与自由生成端到端成绩分开。','',
        '主表修复分母为全部错误候选（候选组100，生成组以实际失败数计）；判断准确率/宏F1基于全200，定位/原因基于全部错误。F/S/O=可行/成功/最优，参考最短路只评分。', '']
    fmt=lambda m:' / '.join('—' if m[k] is None else f'{100*m[k]:.1f}' for k in ['feasibility','success_rate','optimality'])
    for pool,groups in results.items():
        inputs=read_jsonl(a.run/f'{pool}_inputs.jsonl');grids=[Grid.from_dict(r['grid']) for r in inputs]
        truth=[execute(g,r['candidate']) for g,r in zip(grids,inputs)]
        bad=[i for i,t in enumerate(truth) if not t.valid]
        failure_audit[pool]={'candidate_failures':len(bad),'first_failure_reasons':dict(Counter(truth[i].reason for i in bad))}
        if pool.endswith('_generated'):
            failures=[]
            for i in bad:
                t=truth[i];positions=[grids[i].start]+[tuple(x['to']) for x in t.trace if not x['reason']]
                failures.append({'id':inputs[i]['id'],'reason':t.reason,'action_count':len(inputs[i]['candidate'].split()),
                                 'has_repeated_position':len(set(positions))<len(positions)})
            failure_audit[pool]['generated_failures']=failures
        lines += [f'## {pool}','',f'初始路线 F/S/O：{fmt(groups["direct"])}%；初始失败{len(bad)}/200。','',
                  '| V＋规则 | 判断准确率 | V宏F1 | 首错/原因正确率 | FA/FR | 原始R 错误子集F/S/O | 新R 错误子集F/S/O | 新R全200 F/S/O |',
                  '|---|---|---|---|---|---|---|---|']
        for source in ['vanilla','structured','review','executor']:
            executor_outputs={}
            for rule in protocol['rules']:
                both={}
                for backend in ['old','new']:
                    key=f'{source}+{rule}+{backend}';v=groups[key];both[backend]=v
                    traces=read_jsonl(a.run/f'{pool}_{source}_{rule}_{backend}.jsonl')
                    plans=[r['output'] for r in traces]
                    assert plan_metrics(grids,plans)==v['planning']
                    assert plan_metrics([grids[i] for i in bad],[plans[i] for i in bad])==v['invalid_only']
                    checked+=len(plans)
                    if source=='executor':
                        if backend not in executor_outputs:executor_outputs[backend]=plans
                        assert plans==executor_outputs[backend]
                    export.append({'pool':pool,'v':source,'rule':rule,'r':backend,
                        **{'all_'+k:val for k,val in v['planning'].items()},
                        **{'invalid_'+k:val for k,val in v['invalid_only'].items()},
                        **{'v_'+k:val for k,val in v['verification'].items()},
                        'judgment_accuracy':v['judgment_accuracy'],'false_accepts':v['false_accepts'],
                        'false_rejects':v['false_rejects'],'valid_routes_broken':v['valid_routes_broken']})
                x,y=both['old'],both['new'];q=x['verification']
                assert q==y['verification']
                lines.append(f'| {source}+{rule} | {100*x["judgment_accuracy"]:.1f} | {100*q["macro_f1"]:.1f} | {100*q["first_error_localization_accuracy"]:.1f}/{100*q["failure_type_accuracy"]:.1f} | {x["false_accepts"]}/{x["false_rejects"]} | {fmt(x["invalid_only"])} | {fmt(y["invalid_only"])} | {fmt(y["planning"])} |')
    write_json(a.run/'failure_cases.json',failure_audit)
    write_json(a.run/'recompute_audit.json',{'outputs_recomputed':checked,'hashes_match':True,'executor_rules_idempotent':True})
    with Path('docs/newpart1_transfer_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    write_json('docs/newpart1_transfer_results.json',results)
    lines+=['','## 运行证据','',f'独立重算{checked}条输出，原运行逐条与官方执行器一致；源码、数据、检查点SHA与GPU账本都保留。CPU构造/审计与GPU训练/推理分开。权重仅本地保存；本报告及发布证据不包含实体权重。原同伴的规则桥接修复不是本研究两种R，不能直接将其修复比例当作同一算法对比。']
    Path('docs/newpart1_transfer_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('AUDIT PASS',checked)


if __name__=='__main__':main()
