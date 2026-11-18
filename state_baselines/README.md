# 状态策略基线

当前 `evaluate` 入口会执行早先实现的规则修复分支。该修复方案未经用户确认，相关结果仅用于复现历史运行。正式修复方案待用户指定。

三模型共享 `data/policy_v3`。NewPart1 的 Flan-T5-base 将验证答案首个判别 token 的交叉熵与后续 token 平均交叉熵各计 50%。有效答案的后续监督为结束符。模型逐步选择四个动作，解码时不屏蔽非法动作。对每个初始计划比较直接输出、模型自验证加规则修复、执行器验证加同一规则修复。修复使用提供的首错位置，最多回退两步并桥接三步。

在仓库根目录运行。配置与阶段顺序见 [实验计划](../docs/phase1_state_policy_baseline_plan.md)，正式训练和后续评测进展见 [实验记录](../docs/state_policy_experiment_log.md)。

```powershell
python -m state_baselines.run audit --executor EXECUTOR --out RUN
python -m state_baselines.run pilot --model flan_t5_small --pretrained PRETRAINED_SNAPSHOT --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
python -m state_baselines.run train --model flan_t5_small --pretrained PRETRAINED_SNAPSHOT --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
```

对 `flan_t5_base`、`bart_base` 可分别运行 pilot 与 train。原始权重目录应为固定 revision 命名的快照，或包含记录固定 revision 的 `upstream_revision.txt`。每个 pilot 用32条训练样例验证两次优化器更新和开发推理。正式训练独立加载原始权重。

```powershell
python -m state_baselines.run select --executor EXECUTOR --out RUN
python -m state_baselines.run evaluate --split test_id --model flan_t5_small --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
```

ID 对三个模型执行。OOD 使用 `selection.json` 中的开发集主模型。该文件冻结数据、源码、模型权重及开发选择。输出包括训练进度、每轮开发结果、候选诊断、修复轨迹和指标。

## NewPart1 的单模型复现

使用 `google/flan-t5-base` 的 revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`，原始 `data/policy_v3` 与冻结执行器。台式机实验使用 RTX 3050、PyTorch 2.11.0+cu128、Transformers 4.57.6。为复现指定新的 `RUN` 目录。

```powershell
python -m state_baselines.run audit --executor EXECUTOR --out RUN
python -m state_baselines.run train --model flan_t5_base --pretrained PRETRAINED_SNAPSHOT --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
python -m state_baselines.run select --model flan_t5_base --executor EXECUTOR --out RUN
python -m state_baselines.run evaluate --split test_id --model flan_t5_base --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
python -m state_baselines.run evaluate --split test_ood --model flan_t5_base --executor EXECUTOR --out RUN --ledger SHARED_LEDGER
```

省略 `select` 的 `--model` 时按原规则比较三个模型。NewPart1 结果及已保存逐例证据见 [实验记录](../docs/state_policy_experiment_log.md)。证据中的选模清单属于当时的实验入口。复现时由本入口为新运行生成选模清单。

## 逐步验证数据与扫描

从原训练集的每条验证候选取一个状态。取首个非法动作、合法候选的最后一个动作，或空候选的停止状态。提示包含地图、目标、当前位置和待查动作。生成的 `train.jsonl` 只包含新的 V 样例。联合训练入口将它们与原 G 样例合并。两版的实测结果和原始数据文件见 [实验记录](../docs/step_verification_v1_v2.md)。

```powershell
python -m state_baselines.prepare_step_verification --executor EXECUTOR --out outputs/step-verification-v1
```

`audit.json` 保存原文件哈希、按类数据分布和典型扫描轨迹。开发集与 ID/OOD 测试集直接读取原候选，用执行器反馈在 CPU 上检查扫描逻辑。

`state_baselines.step_verification.verify_steps(verifier, grids, plans)` 接收实现 `predict` 的验证器，每次检查各候选的下一步。`valid` 和 `goal_not_reached` 都允许检查后续动作。其余非法原因立即结束，并以当前全局步数返回。候选结束时，未到目标报告为 `N+1`。局部训练标签的第 1 步是当前动作，第 2 步是动作后的停止点。程序维护坐标，正式验证反馈来自传入的模型。

Flan-T5-base 沿用 NewPart1 的拆分损失。首个 `valid/invalid` token 的交叉熵占 50%，其余 token 的平均交叉熵占 50%。`valid` 的其余部分只有 EOS。

以下命令执行逐步 V 的联合微调实验。每次使用同一个 `--step-train` 文件。训练入口按原顺序保留 2,000 条 G 样例，并逐条替换 2,000 条 V 样例。原开发、测试文件直接用于完整候选扫描。联合微调更新共享权重，单独保存检查点。`evaluate --frozen-plans` 可加载同一测试集上旧 G 的规划记录，隔离生成结果变化对验证比较的影响。

```powershell
$stepTrain = 'outputs/step-verification-v1/train.jsonl'
python -m state_baselines.run audit --executor EXECUTOR --out RUN --step-train $stepTrain
python -m state_baselines.run train --model flan_t5_base --pretrained PRETRAINED_SNAPSHOT --executor EXECUTOR --out RUN --ledger SHARED_LEDGER --step-train $stepTrain
python -m state_baselines.run select --model flan_t5_base --executor EXECUTOR --out RUN --step-train $stepTrain
foreach ($split in 'dev', 'test_id', 'test_ood') {
    python -m state_baselines.run evaluate --split $split --model flan_t5_base --executor EXECUTOR --out RUN --ledger SHARED_LEDGER --step-train $stepTrain
}
```

## 双标签训练数据

```powershell
python -m state_baselines.prepare_dual_label_data --executor EXECUTOR --out outputs/step-verification-v2
```

`train.jsonl` 按原顺序保存 4,000 条记录。2,000 条 G 保留原动作监督，`state_verification` 标注在执行动作前停在 `policy_position` 的结果。`local_stop_target` 中的第 1 步指长度为零的局部动作串的结束位置。2,000 条独立 V 保留原有效样例和三类动作错误。对原未完成样例，取同一训练记录 `target_plan` 实际到达目标的最后一步，按 `valid` 监督。`source_plan_field` 记录动作来源，`audit.json` 列出标签计数和文件哈希。

在上节 `audit`、`train`、`select` 和 `evaluate` 命令中使用 `--dual-train outputs/step-verification-v2/train.jsonl`，并指定新的 `--out RUN`。每条 G 记录使用原动作损失加 0.5 倍状态验证损失。状态验证以当前坐标和局部停止为输入，沿用拆分损失。`training.json` 分别记录每轮的主任务损失、状态损失和带权总损失。每轮按 4,000 条原始记录采样并更新 250 次。

固定 G 对照使用 [证据包](../deliverables/step_verification_v1_v2_evidence.zip) 中 `frozen_inputs` 目录的两份原始记录。执行 `evaluate --split test_id` 时依次附加 `--frozen-plans PATH/frozen_inputs/firstpart_id.jsonl --frozen-plans PATH/frozen_inputs/newpart1_id.jsonl`。
