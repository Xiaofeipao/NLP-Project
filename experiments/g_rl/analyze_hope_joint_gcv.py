"""Paired outcome audit for the completed, fixed-budget joint experiment."""
from collections import Counter
import json
from pathlib import Path
import sys

from hope_joint_data import read, rows, save, write_rows

import argparse
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('--out', type=Path, required=True)
ROOT = cli.parse_args().out.resolve()
ARMS = ('supervised', 'expected', 'rloo')


def paired(before, after, key, describe=None):
    a, b = {r['id']: r for r in before}, {r['id']: r for r in after}
    assert a.keys() == b.keys()
    improved, degraded, examples = [], [], []
    for ident in a:
        old, new = bool(key(a[ident])), bool(key(b[ident]))
        if new and not old:
            improved.append(ident)
        if old and not new:
            degraded.append(ident)
        if old != new:
            item = dict(id=ident, old_correct=old, new_correct=new)
            if describe:
                item.update(describe(a[ident], b[ident]))
            examples.append(item)
    return dict(n=len(a), improved=len(improved), degraded=len(degraded),
                net=len(improved)-len(degraded), improved_ids=improved,
                degraded_ids=degraded, cases=examples)


def main():
    assert (ROOT/'completed.json').exists(), 'Wait until all predeclared arms and evaluations complete'
    result = read(ROOT/'comparison.json')
    transitions = {}
    for arm in ARMS[1:]:
        arm_pairs = {}
        for task in ('G', 'C', 'V'):
            before = [r for r in rows(ROOT/'supervised/update375/step_dev/predictions.jsonl') if r['task'] == task]
            after = [r for r in rows(ROOT/arm/'update375/step_dev/predictions.jsonl') if r['task'] == task]
            arm_pairs[task] = paired(before, after, lambda r:r['correct'],
                lambda a,b:dict(gold=a['gold'],old=a['prediction'],new=b['prediction'],
                               old_result=a.get('move_result'),new_result=b.get('move_result')))
            if task == 'G':
                cases = arm_pairs[task]['cases']
                arm_pairs[task]['corrected_error_types'] = dict(Counter(r['old_result'] for r in cases if r['new_correct']))
                arm_pairs[task]['new_error_types'] = dict(Counter(r['new_result'] for r in cases if not r['new_correct']))
        for split in ('dev','double_error','single_error'):
            before_dir = ROOT/'supervised/evaluation/scores/supervised'/split
            after_dir = ROOT/arm/'evaluation/scores'/arm/split
            before, after = rows(before_dir/'candidate_scores.jsonl'), rows(after_dir/'candidate_scores.jsonl')
            arm_pairs[split] = {
                'verification':paired(before,after,lambda r:r['prediction']==r['gold']),
                'candidate_repair':paired([r for r in before if r['gold']!='valid'],
                                          [r for r in after if r['gold']!='valid'],lambda r:r['repair']['success'])}
            gb, ga = rows(before_dir/'generated_scores.jsonl'), rows(after_dir/'generated_scores.jsonl')
            arm_pairs[split]['direct'] = paired(gb,ga,lambda r:r['direct']['success'])
            if split != 'dev':
                arm_pairs[split]['self_repair'] = paired(gb,ga,lambda r:r['self_repair']['success'])
        transitions[arm] = arm_pairs
    save(ROOT/'paired_comparison.json',transitions)
    phase = {arm:{str(update):read(ROOT/arm/f'update{update}/step_dev/metrics.json')
                  for update in (125,250,375)} for arm in ARMS}
    save(ROOT/'learning_curve.json',phase)
    lines = ['Hope 共享 G/C/V 从预训练起点联合训练：单种子对照实验',
             '代码 Hope@9ebecebe54e94e3bd474a5522c23df86ad66852d',
             '数据 hope_dataset@d88f85b872560f9313ac4319cdd5fa86151c0780',
             '三个实验均从原始 Flan-T5-base 初始化；未冻结 C/V，未使用旧 Hope 微调权重作为起点。',
             '相同 4,000 个状态、36,800 条视图、375 次更新、微批16、有效32状态、seed42。',
             'C遍历一次；G/V各遍历两次。G奖惩强度0.2，最短+1、合法绕路-1、撞墙/越界-5。',
             'supervised：原监督对照；expected：精确四动作期望奖励；rloo：真实采样8动作的RLOO。',
             '所有组均执行同样的四动作额外前向；监督组该项梯度系数为0；额外随机数由fork_rng隔离。',
             '所有组固定第375次更新作为最终检查点；全部训练完成后才评估双错验证及单错测试。',
             '本次微批不同于发布基线，因此收益以本次同配置监督重训为对照。', '']
    for arm in ARMS:
        step = result[arm]['step_dev']
        lines += [arm, '单步开发：'+json.dumps(step,ensure_ascii=False)]
        for split in ('dev','double_error','single_error'):
            role = {'dev':'综合开发','double_error':'双错验证','single_error':'单错测试'}[split]
            lines.append(role+'：'+json.dumps(result[arm][split],ensure_ascii=False))
        lines.append('')
    for arm in ARMS[1:]:
        lines.append(arm+' 相对 supervised 的逐题变化：')
        for task in ('G','C','V'):
            p = transitions[arm][task]
            lines.append(f"{task}: 新增正确{p['improved']}，新增错误{p['degraded']}，净变化{p['net']}，分母{p['n']}")
        for split in ('double_error','single_error'):
            for metric, p in transitions[arm][split].items():
                lines.append(f"{split}/{metric}: 新增正确{p['improved']}，新增错误{p['degraded']}，净变化{p['net']}，分母{p['n']}")
        lines.append('')
    lines += ['边界：这是单种子固定预算试验，不能据此宣称训练稳定性或跨种子统计显著性。',
              '原71%等发布结果仅作历史参考；当前新版错误词和统一微批的分数均重新测量。',
              '完整逐题收益与退化见 paired_comparison.json；训练中三阶段分数见 learning_curve.json。']
    (ROOT/'联合训练结果报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(comparison=result, paired_summary={a:{t:{k:v for k,v in p.items()
          if k in ('n','improved','degraded','net','corrected_error_types','new_error_types')}
          for t,p in ps.items() if t in ('G','C','V')} for a,ps in transitions.items()}),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
