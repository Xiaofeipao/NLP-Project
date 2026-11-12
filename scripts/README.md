# 命令入口分类

新增 [methods/](../methods/README.md) 分类入口，原脚本继续作为实现来源。转发保留参数、退出码、冻结和预算检查。

- 数据与模型下载：`fetch_official.py`、`fetch_models.py`。
- 原基线：`run_study.py`，以及 `python -m ppnl.cli`。
- 逐步策略：`run_policy_study.py`、`run_policy_v3.py`、`evaluate_policy.py`、`plan_grid.py`。
- Agent Harness：`run_harness.py`、`compare_harness.py`。
- 全量模型 GVR Harness：`run_full_finetune_gvr.py`（单网格试验）。
- 全量多任务微调后的批量评估与消融：`run_post_training.py`，方法矩阵见 `configs/post_training.json`。
- GVR LoRA：`run_gvr.py`；测试要求 `freeze_experiment.py` 产生的锁。
- 两类前缀：`run_prefix_experiment.py`；旧两个前缀脚本仅转发到此入口。
- RLVR：`run_rlvr.py`、`verify_rlvr.py`，独立于 GVR 的 V/R LoRA。
- 输入消融：`preview_prompt_formats.py` 仅做 CPU 预览，训练仍用 `run_gvr.py`。
- 历史策略评测：`run_eval_harness.py`；它的原有检查不是新的前缀冻结协议。
- 结果整理：`summarize_*.py`、`export_gvr_artifacts.py`、`export_rlvr_evidence.py`。导出权重需遵守项目当前发布约定。

新前缀实验请按 [项目导航](../docs/project_map.md) 操作。低层 Python API 与单网格调试入口不能替代受冻结约束的测试集评估。

旧 GVR 批量入口 `smoke_gvr_repair.py`、`run_gvr_development.py`、`run_gvr_repair_test.py`、`run_gvr_optional_development.py` 已停用；运行时会在加载模型/CUDA 前给出迁移说明，不会自动切换实验协议。`CachedGeneration` 保留为兼容接口。五个原始脚本（含 `freeze_gvr_repair.py`）的 Git 原始字节保存在 [历史归档](../docs/legacy_gvr_drivers_23183e2.zip)，内含逐文件 SHA-256 清单；[归档校验值](../docs/legacy_gvr_drivers_23183e2.sha256) 可用于验证 ZIP 本身。复现历史结果应使用完整提交 `23183e2` 及对应数据、模型和冻结记录。

`freeze_gvr_repair.py` 仅维护旧报告格式；它不接收新消融报告，新入口自行写入 `protocol.json`。
