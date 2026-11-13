# 状态策略基线

当前 `evaluate` 入口会执行早先实现的规则修复分支。该修复方案未经用户确认，相关结果仅用于复现历史运行。正式修复方案待用户指定。

三模型共享 `data/policy_v3` 和监督目标。模型逐步选择四个动作，解码时不屏蔽非法动作。对每个初始计划比较直接输出、模型自验证加规则修复、执行器验证加同一规则修复。修复使用提供的首错位置，最多回退两步并桥接三步。

在仓库根目录运行。配置与阶段顺序见 [实验计划](../docs/phase1_state_policy_baseline_plan.md)，正式训练和后续评测进展见 [实验记录](../docs/state_policy_experiment_log.md)。

```powershell
python -m state_baselines.run audit --executor EXECUTOR --out RUN
python -m state_baselines.run pilot --model flan_t5_small --pretrained PRETRAINED_SNAPSHOT --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
python -m state_baselines.run train --model flan_t5_small --pretrained PRETRAINED_SNAPSHOT --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
```

对 `flan_t5_base`、`bart_base` 分别运行 pilot 与 train。原始权重目录应为固定 revision 命名的快照，或包含记录固定 revision 的 `upstream_revision.txt`。每个 pilot 用32条训练样例验证两次优化器更新和开发推理。正式训练独立加载原始权重。

```powershell
python -m state_baselines.run select --executor EXECUTOR --out RUN
python -m state_baselines.run evaluate --split test_id --model flan_t5_small --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
```

ID 对三个模型执行。OOD 使用 `selection.json` 中的开发集主模型。该文件冻结数据、源码、模型权重及开发选择。输出包括训练进度、每轮开发结果、候选诊断、修复轨迹和指标。
