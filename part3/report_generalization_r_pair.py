"""Audit paired old/new R outputs and export all verifier comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from ppnl.core import Grid,execute
from ppnl.data import read_jsonl,write_json
from ppnl.evaluation import plan_metrics
from ppnl.verifier_rules import RULES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();assert (a.run/'completed.json').exists()
    results=json.loads((a.run/'results.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    for category in ['data_sha256','code_sha256']:
        for f,h in protocol[category].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    previous=json.loads(Path('outputs/generalization_verifiers/run_001/results.json').read_text())
    lines=['# 7×7与密集6×6：单错/双错、原始R/增强R配对评测','',
        '## 数据与固定协议','',
        '两类地图各200个环境；每类单错、双错各200条候选，均为100有效＋100错误。双错只替换原100条错误候选；地图、起终点、有效控制均与单错配对，parent_id记录来源。四组800条候选来自400个独立地图，属于可选测试，不加入原必做6×6 ID/OOD、不用于训练/开发/选型。双错是两个独立原始编辑位置、token编辑距离2，不要求可观察到两个首错。CPU构造与官方检查单独记录于data/generalization_double/manifest.json。','',
        '原始R：同一动作模型、按预测首错保留前缀，原始贪心续走，最多新增36步，一次修复，无访问约束/分支/惩罚/删环。新R：固定回退36＋Top3（宽度3、状态评分上限108）＋循环软惩罚＋最终删环，最多新增36步。未按新测试重新选型。新R带合法动作过滤，原始R只在非法后停止，预算与算法贡献均不同。','',
        '模型为同一份本地独立复现Flan-T5-base冻结权重。单错V复用此前同checkpoint的GPU逐例输出并校验ID/地图/候选相等；双错V在GPU新推理。模型预测最大输入512 tokens，超长报错不截断。没有训练或新权重。','',
        '## V方法分别增加什么','',
        '- vanilla：原始微调V；structured：同一V用结构化坐标/障碍提示；review：原V答案再交给同一个V复核一轮，增加一次调用，不输入真实执行反馈。',
        '- none：不增强；vocabulary：首个安全可确认非法动作词；terminal：词表检查＋合法执行后未到终点的EOS诊断。',
        '- bounds：词表＋首个越界；collision：词表＋首个碰撞；safety：词表＋越界＋碰撞。遇到不覆盖的早期错误，立即保留模型反馈，不把晚错当首错。',
        '- safety_terminal：安全组合再加终点，已经由规则提供所有错误路径诊断，仍保留模型对有效路线的误拒，不能把此组收益归功于模型。executor为完整真实诊断来源，接相同规则应幂等。',
        '- 所有组诊断valid不触发R；错误步号或非法保留前缀拒绝，不用真实首错暗中调整。BFS参考路只评分；执行器来源是明确允许的诊断对照。','',
        '## 指标','',
        'F/S/O为可行/成功/最优，分母为行内全部样本。主表错误子集每组100，未被V检出也算未修好；V判断准确率、宏F1基于200条；定位/原因基于100条错误；FA/FR为误接受/误拒数量。R前的验证指标对两种R必须相同。下列值均为百分比，F/S/O顺序固定。','']
    fmt=lambda m:' / '.join(f'{100*m[k]:.1f}' for k in ['feasibility','success_rate','optimality'])
    audit=0;export=[];breakdown={}
    for dataset,groups in results.items():
        inputs=read_jsonl(a.run/f'{dataset}_inputs.jsonl');grids=[Grid.from_dict(r['grid']) for r in inputs]
        bad=[i for i,r in enumerate(inputs) if not r['valid']]
        breakdown[dataset]={}
        lines += [f'## {dataset}','', '| V来源＋增强 | 判断准确率 | V宏F1 | 首错/原因正确率 | FA/FR | 原始R：错误100 F/S/O | 新R：错误100 F/S/O |',
                  '|---|---|---|---|---|---|---|']
        for source in ['vanilla','structured','review','executor']:
            executor_reference={}
            for rule in RULES:
                both={}
                for backend in ['old','new']:
                    name=f'{source}+{rule}+{backend}';v=groups[name];both[backend]=v
                    traces=read_jsonl(a.run/f'{dataset}_{source}_{rule}_{backend}_traces.jsonl')
                    plans=[t['output'] for t in traces]
                    assert plan_metrics(grids,plans)==v['planning']
                    assert plan_metrics([grids[i] for i in bad],[plans[i] for i in bad])==v['invalid_only']
                    audit+=200
                    if source=='executor':
                        if backend not in executor_reference:executor_reference[backend]=plans
                        assert plans==executor_reference[backend]
                    if dataset.endswith('_single') and backend=='old':
                        base=dataset.removesuffix('_single')
                        assert v['planning']==previous[f'test_{base}_candidates'][f'{source}+{rule}']['planning']
                    q=v['verification']
                    export.append({'dataset':dataset,'v_source':source,'rule':rule,'r':backend,
                        **{'all_'+k:val for k,val in v['planning'].items()},
                        **{'invalid_'+k:val for k,val in v['invalid_only'].items()},
                        'judgment_accuracy':v['judgment_accuracy'],**{'v_'+k:val for k,val in q.items()},
                        'false_accepts':v['false_accepts'],'false_rejects':v['false_rejects'],'valid_routes_broken':v['valid_routes_broken']})
                    breakdown[dataset][name]={}
                    for reason in sorted({inputs[i]['reason'] for i in bad}):
                        ix=[i for i in bad if inputs[i]['reason']==reason]
                        breakdown[dataset][name][reason]=plan_metrics([grids[i] for i in ix],[plans[i] for i in ix])
                old,new=both['old'],both['new'];q=old['verification']
                assert q==new['verification'] and old['judgment_accuracy']==new['judgment_accuracy']
                lines.append(f'| {source}+{rule} | {old["judgment_accuracy"]*100:.1f} | {q["macro_f1"]*100:.1f} | {q["first_error_localization_accuracy"]*100:.1f}/{q["failure_type_accuracy"]*100:.1f} | {old["false_accepts"]}/{old["false_rejects"]} | {fmt(old["invalid_only"])} | {fmt(new["invalid_only"])} |')
    with Path('docs/generalization_r_pair_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    write_json(a.run/'by_failure_type.json',breakdown)
    write_json(a.run/'recompute_audit.json',{'outputs_recomputed':audit,'hashes_match':True,'single_old_r_previous_results_match':True,'executor_rules_idempotent':True})
    write_json('docs/generalization_r_pair_results.json',results)
    lines += ['', '有效控制被修坏的数量、全部200候选成绩、格式与联合定位原因准确率，均完整导出CSV和JSON；按真实首错类型细分见输出目录by_failure_type.json。该实验不衡量NO_PATH，也不把人工扰动候选修复成绩当作自由生成端到端成绩。没有按新测试调参；多轮探索的旧测试结果仍属探索性证据。未push。']
    Path('docs/generalization_r_pair_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('AUDIT PASS',audit)


if __name__=='__main__':main()
