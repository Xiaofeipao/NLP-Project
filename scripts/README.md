# 命令入口分类

新增 [methods/](../methods/README.md) 分类入口，原脚本继续作为实现来源。转发保留参数、退出码、冻结和预算检查。

- 数据与模型下载：`fetch_official.py`、`fetch_models.py`。
- 原基线：`run_study.py`，以及 `python -m ppnl.cli`。
- 逐步策略：`run_policy_study.py`、`run_policy_v3.py`、`evaluate_policy.py`、`plan_grid.py`。
- Agent Harness：`run_harness.py`、`compare_harness.py`。
- 全量模型 GVR Harness：`run_full_finetune_gvr.py`（单网格试验）。
- GVR LoRA：`run_gvr.py`；测试要求 `freeze_experiment.py` 产生的锁。
- 两类前缀：`run_prefix_experiment.py`；旧两个前缀脚本仅转发到此入口。
- RLVR：`run_rlvr.py`、`verify_rlvr.py`，独立于 GVR 的 V/R LoRA。
- 输入消融：`preview_prompt_formats.py` 仅做 CPU 预览，训练仍用 `run_gvr.py`。
- 历史策略评测：`run_eval_harness.py`；它的原有检查不是新的前缀冻结协议。
- 结果整理：`summarize_*.py`、`export_gvr_artifacts.py`、`export_rlvr_evidence.py`。导出权重需遵守项目当前发布约定。

新前缀实验请按 [项目导航](../docs/project_map.md) 操作。低层 Python API 与单网格调试入口不能替代受冻结约束的测试集评估。
