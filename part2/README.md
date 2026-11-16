# Part2 生成 验证 修复

本目录对应题目中“实现生成—验证—修复推理方案”这一项。沿用同伴 [FirstPart 的 state_baselines](https://github.com/Xiaofeipao/NLP-Project/tree/FirstPart/state_baselines) 的 `run.py / workflow.py / protocol.json / README` 组织方式；公共算法仍放在 `ppnl`，已有脚本兼容保留，不复制算法。

先读 [三个阶段的逻辑关系](../docs/part1_part2_part3.md)，再读 [本阶段结果](../docs/phase2_feedback_results.md)。这里的“Part3”是后续改进与消融目录，对应课程 Part 2 的第3项，不是课程 Part 3 的报告展示。

## 方法与结果

第一部分提供开发集选定的 Flan-T5-base 模型家族、布局隔离的数据及错误诊断证据。本地没有同伴的实体权重，经授权按固定 revision 独立训练两轮，开发选择第2轮。后续方法之间共用本地权重；不能把同伴84.5%与本地85.5%的差值当作方法提升。

G逐步生成；V比较模型诊断与符号执行器诊断；R按反馈保留首错前缀，然后由相同微调策略续走，最多36个新增动作。R没有整路径文本修复监督，reason文本不输入动作策略。R不是BFS/A*补路，也不包含成功路径缩短S。

200环境直接生成成功85.5%、可行100%、最优85%。准确执行器反馈使固定非法候选修复成功达到ID94%/OOD92%，模型反馈为20%/28%；但两者都不能单独解除自生成循环。终点校验与访问约束相结合后完整成功98.5%、最优85%。

## 无权重复核

```powershell
python -m part2.run verify-results
```

只用公开证据JSON/JSONL重算全部三率、验证指标和修复分母，无需GPU或权重；若有 `vendor/ppnl/executor-point-sg.py`，同时与官方执行器交叉验证。缺少官方文件时明确显示 `official_executor=False`，仍检查严格本地语义。获取官方文件可用已有 `python scripts/fetch_official.py`。

## 重新训练

```powershell
python -m part2.run prepare --out outputs/part2_reproduction
python -m part2.run audit --out outputs/part2_reproduction --executor vendor/ppnl/executor-point-sg.py
python -m part2.run pilot --out outputs/part2_reproduction --model flan_t5_base --pretrained PRETRAINED_REVISION_SNAPSHOT --ledger SHARED_LEDGER
python -m part2.run train --out outputs/part2_reproduction --model flan_t5_base --pretrained PRETRAINED_REVISION_SNAPSHOT --ledger SHARED_LEDGER
```

`prepare`解压精确的FirstPart训练源码快照到独立输出目录，复用仓库 `data/policy_v3`；不覆盖主代码、数据或旧实验。原始预训练快照需为 `google/flan-t5-base` revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`，目录名为该revision或有同值的 `upstream_revision.txt`。GPU命令必须显式传共用账本；不得新建空账本掩盖历史用量。

重新训练不保证复现历史权重的字节哈希。用于第二/三部分的主模型家族仍由FirstPart开发证据选定，不能依据本地测试重选家族。本地重训包含2,000状态动作与2,000验证例，不增加逻辑训练样本。

## 运行完整反馈比较

```powershell
python -m part2.run evaluate --checkpoint outputs/part2_reproduction/reproduction/train/flan_t5_base/best --out outputs/part2/evaluation_001 --ledger SHARED_LEDGER
```

输出开发候选、ID/OOD候选、ID生成轨迹及结果。OOD共用ID地图，不能把生成成绩相加为400环境。输出目录必须是新的；原实验标为重复使用测试集的探索性证据。

记录格式：每行JSONL保留grid、candidate、模型反馈、采用反馈、最终路径、状态和gold；结果将全体候选与原非法候选分母分开。证据包不含权重，公开路径/设备脱敏改变字节，导出锁不是新的可执行冻结锁。
