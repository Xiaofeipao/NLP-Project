# Python 模块分类

为保留既有脚本、checkpoint 溯源和导入路径，公共模块不做大范围移动；按职责导航：

- 环境与数据：`core.py`、`data.py`、`prompts.py`。
- 全量微调：`modeling.py`、`optimization.py`、`cli.py`。
- GVR LoRA：`task_adapters.py`、`gvr_training.py`、`gvr_objectives.py`、`gvr_preflight.py`。
- 逐步策略与搜索：`policy.py`、`policy_data.py`、`policy_v3_data.py`。
- 运行控制：`harness.py`、`harness_backends.py`、`gvr_harness.py`、`full_finetune_gvr.py`。
- 前缀修复：`prefix_state_repair.py`（GVR trace）、`full_prefix_repair.py`（原全量模型 trace）、`prefix_evaluation.py`（统一冻结入口）。
- 奖励训练：`rlvr.py`（REINFORCE，不是 GRPO）。
- 指标与复核：`evaluation.py`、`eval_harness.py`、`gvr_evaluation.py`、`gvr_audit.py`、`experiment_freeze.py`。

方法和历史结果的边界见 [项目导航](../docs/project_map.md)。
