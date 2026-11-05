# CoT 与无解奖励试跑（2026-11-05）

分支：`tree-verifier-reachability`，原名 `codex/tree-verifier-reachability`。本次没有修改稳定 main 或默认策略。

## 参考核查

参考 [Hugging Face 项目](https://huggingface.co/EnjiXiong/AIAA4051-FinalProject-PPNL)。实际读取其 `grid-path-planning/data_utils.py`、`train_rl.py`、`reward.py`：CoT 用逐步坐标作为监督目标；RL 脚本只接受 vanilla/structured，明确不支持 cot。其策略损失实际是单次更新的 REINFORCE 形式加组内标准化和 KL，不能因命名 GRPO 就认为已使用多次裁剪更新。模型卡高分还包含执行器搜索，不是无解识别成绩。

本项目仅借鉴坐标监督与组内奖励思路，独立增加 NO_PATH 和严格 FINAL 分隔符，使用已有带 clipping 的 GRPO 实现。没有真实人类偏好，准确名称为 CoT SFT + RLVR，不是 RLHF，也不是原项目的直接复现。

## 已执行协议

入口 `tools/run_reachability_cot.py`。固定已有 T5-base 全量微调底座，仅训练 q/v LoRA。使用此前隔离的 16 张训练地图、8 张开发地图；不读取人工偏好字段，直接由 BFS 生成独立监督目标。可达目标为坐标轨迹加动作，无解目标为起点连通分量加 NO_PATH。推理输入不提供 BFS 结果。

先监督训练 3 遍，再对 16 个训练地图各采样 4 个回答，每组做 2 次 GRPO 更新；输出上限 256 token。KL 参考为禁用 adapter 的原底座，而非 SFT adapter。奖励只判断最终答案正确性，不证明生成的坐标解释正确。四对公开诊断地图未训练；仅开发集决定是否保留 RL，相同分数保留 SFT。此次是额外数据的小样本探索，不并入原 4,000 条配额主实验，不能声称符合原正式训练预算的替代结果。

## 结果与处理

未训练、仅 CoT SFT、CoT + GRPO 在开发集的正确无解识别和可达规划均为 0/4，正确 FINAL 格式均为 0/8。SFT 和 GRPO 在八例公开诊断上同样失败。16 个 RL 采样组的 64 个回答全部奖励 -1，组内优势为零，只有 KL 项可能产生梯度。因此本轮没有无解识别增强证据，不能把未误拒解释为分类能力良好。

训练规模太小且输出格式未学会，不能从这次失败推断 CoT/强化学习普遍无效。后续若继续，应先在开发集验证格式学习和正负奖励覆盖，再考虑 RL；需独立预注册更充分的训练数据预算和无 CoT 同预算对照。本轮停止升级，默认系统不变，保存负结果用于比较。

本地 `outputs/reachability_cot_grpo_001/` 包含协议、SFT/GRPO adapter、完整采样及报告；可分享 JSON 位于 `docs/experiments/reachability_cot_grpo/`。
