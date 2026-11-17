"""Recompute saved inference metrics and write a weight-free comparison report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from ppnl.core import Grid, execute
from ppnl.data import read_jsonl, write_json
from ppnl.evaluation import plan_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    assert (args.run/'completed.json').exists()
    results=json.loads((args.run/'results.json').read_text())
    protocol=json.loads((args.run/'protocol.json').read_text())
    for file,digest in protocol['code_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    for file,digest in protocol['data_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    lines=['# 新地图测试：固定原始R，对比模型V增强和执行器', '',
           '本轮仅做GPU推理，不训练、不选超参数。模型为本地独立复现Flan-T5-base。7×7和密集6×6各200张地图，只用于测试；每组人工候选100有效/100单错。另各生成200条原G路线，不能将候选与G路线当成400张独立地图。两组均可达，不包含NO_PATH。', '',
           '## 原始R保持不变', '',
           '使用scripts/run_phase2_feedback.py的continue_plan，模型反馈valid即保留路径；invalid按预测首错保留之前前缀，再用同一个动作模型贪心续走。错误步号拒绝、非法保留前缀拒绝，不用真实首错纠正模型。无回退36步、无Top3、无软循环惩罚、无访问约束、无删环；G最多36动作，R最多新增36动作、修复一次。新动作不靠规则过滤，只在非法后停止。动作模型原输入仍有Legal moves提示。reason文本不被续走动作策略语义读取，控制主要依赖valid/invalid与step。', '',
           '## V增强分别加了什么', '',
           '- vanilla：原始提示、同一微调V。structured：相同V换结构化坐标/障碍提示，没有新训练。review：原提示输出后，再用同一V独立一轮复核（增加一次模型调用），不提供执行器观察。',
           '- none：不加规则。vocabulary：仅可安全确认的首个非法动作词。',
           '- terminal：词表检查＋整条路径可合法执行且未到终点时，给出EOS位置goal_not_reached；遇到越界/碰撞不替模型定位。',
           '- bounds：词表检查＋首个越界诊断；collision：词表检查＋首个碰撞诊断。若先遇到本组不覆盖的错误，停止规则扫描并保留模型反馈，避免把后续错误当首错。',
           '- safety：词表＋越界＋碰撞；safety_terminal：再加终点。这组对所有实际错误供给确定性诊断，已接近完整符号验证，不应将其改进归因于模型V；它仍保留模型对有效路径的误拒。',
           '- executor：完整执行器作为诊断来源，再接完全相同的7种规则与同一个R。规则应幂等，7组规划输出应完全相同。', '',
           '## 指标口径', '',
           'F=Feasibility动作全程合法；S=Success合法且到终点；O=Optimality成功且等长最短路，均以本行全部样本为分母。V列为macro-F1、非法样本首错准确率与原因准确率；FA=把错误判valid，FR=把有效判invalid。BFS最短路只用于评分；executor组明确例外，允许将其真实诊断用于R。', '',]
    audit=0
    breakdown={}
    csvrows=[]
    fmt=lambda m:' / '.join(f'{100*m[k]:.1f}' for k in ['feasibility','success_rate','optimality'])
    for pool,groups in results.items():
        inputs=read_jsonl(args.run/f'{pool}_inputs.jsonl')
        grids=[Grid.from_dict(r['grid']) for r in inputs]
        truth=[execute(g,r['candidate']) for g,r in zip(grids,inputs)]
        bad=[i for i,t in enumerate(truth) if not t.valid]
        lines += [f'## {pool}', '',f'原候选/直接生成 F/S/O：{fmt(groups["direct"])}%；错误{len(bad)}/200。', '',
                  '| V来源＋规则 | 全200 F/S/O % | 错误子集 F/S/O % | V F1/首错/原因 % | FA/FR | 有效路径被修坏 |',
                  '|---|---|---|---|---|---|']
        breakdown[pool]={}
        for name,v in groups.items():
            if name=='direct':continue
            source,rule=name.split('+')
            traces=read_jsonl(args.run/f'{pool}_{source}_{rule}_traces.jsonl')
            outputs=[r['output'] for r in traces]
            assert plan_metrics(grids,outputs)==v['planning']
            assert plan_metrics([grids[i] for i in bad],[outputs[i] for i in bad])==v['invalid_only']
            audit+=len(outputs)
            q=v['verification']
            csvrows.append({'dataset':pool,'method':name,
                            **{'all_'+k:val for k,val in v['planning'].items()},
                            **{'invalid_'+k:val for k,val in v['invalid_only'].items()},
                            **{'v_'+k:val for k,val in q.items()},
                            'false_accepts':v['false_accepts'],'false_rejects':v['false_rejects'],
                            'valid_routes_broken':v['valid_routes_broken']})
            vm=' / '.join(f'{100*q[k]:.1f}' for k in ['macro_f1','first_error_localization_accuracy','failure_type_accuracy'])
            lines.append(f'| {name} | {fmt(v["planning"])} | {fmt(v["invalid_only"])} | {vm} | {v["false_accepts"]}/{v["false_rejects"]} | {v["valid_routes_broken"]} |')
            breakdown[pool][name]={}
            for reason in sorted({truth[i].reason for i in bad}):
                ix=[i for i in bad if truth[i].reason==reason]
                breakdown[pool][name][reason]=plan_metrics([grids[i] for i in ix],[outputs[i] for i in ix])
    write_json(args.run/'by_failure_type.json',breakdown)
    write_json(args.run/'recompute_audit.json',{'plans_recomputed':audit,'code_and_data_hashes_match':True})
    with Path('docs/generalization_verifier_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]))
        writer.writeheader();writer.writerows(csvrows)
    write_json(args.run/'metric_recompute_audit.json',{'outputs_recomputed':audit,'exact_match':True})
    lines+=['','## 复现与限制','',f'主结果：{args.run.as_posix()}/results.json；诊断和最终路径逐例留档；按错误类型的指标在by_failure_type.json。审计重算全部输出指标，与官方执行器逐例交叉检查见audit.json。结果不混入旧6×6 ID/OOD，不依据这些测试重新选模型或规则；同一模型贪心策略可能持续循环，所以V诊断准确并不保证R能恢复。新模型权重未产生，未push。']
    Path('docs/generalization_verifier_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json('docs/generalization_verifier_results.json',results)
    print('Recomputed',audit,'outputs; report: docs/generalization_verifier_results.md')


if __name__=='__main__':main()
