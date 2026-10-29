# Project 6：G／V／R LoRA 实施与证据状态

## 当前状态（2026-10-29）

已实现固定数据索引、独立训练入口、LoRA 路由、持久化整路径循环和 B1–B4／候选评测入口。CPU 边界测试已运行。**新方案未训练、未评测出模型成绩**；仓库 README 中的百分比是历史实验记录，不能当作新方案结果。

P0 核查：仓库 `main@b433523` 含 `data/derived` 原始任务数据及 manifest，训练 800／2,000／1,200、开发 40／100／60，ID 与 OOD 各 200 条。本机已按固定上游 revision 下载官方执行器，SHA256 `375462625e1c133d5b469037c03a18cbc33160a31b2ba88a0da89a127364a6e4` 与数据 manifest 一致；`vendor/` 按仓库规则不提交。仓库未包含 `outputs/`、模型 checkpoint、`outputs/gpu_ledger.json` 或负责人确认记录。因此剩余 GPU 预算、历史权重哈希和已报告指标尚不能独立核实；默认配置拒绝开始新训练。下载执行器不能补齐历史账本或权重。

## 文件与边界

| 能力 | 文件 | 约束 |
|---|---|---|
| 冻结数据 | `ppnl/data.py`、`data/derived/gvr_task_index.json` | 直接索引原记录和布局；不生成新训练例 |
| 训练 | `ppnl/gvr_training.py`、`ppnl/task_adapters.py` | G 从固定预训练版本训练；V／R 分别从 G 启动，`q/v` LoRA；各只训练本任务 |
| 控制器 | `ppnl/gvr_harness.py` | `direct`、`self`、`executor_one`、`executor_three`；模型请求先入账；身份不符拒绝恢复 |
| 评分 | `ppnl/gvr_evaluation.py` | 独立检查最终候选；固定候选 Verify／Repair 与端到端分开汇报 |
| 入口 | `scripts/run_gvr.py`、`config.gvr.json` | 新输出目录，不覆盖历史结果 |

训练依赖固定 `transformers==4.57.6` 与 `peft==0.18.1`。PEFT 的版本选择依据其 [发布说明](https://github.com/huggingface/peft/releases/tag/v0.18.1) 和 [多适配器 API](https://huggingface.co/docs/peft/v0.18.0.rc0/en/package_reference/peft_model)。在实际 Python 3.11／CUDA 环境中仍需做载入和保存重载验证。当前本机仅有 Python 3.14 和 Transformers 5.12.1，不能据此宣称目标环境的模型测试通过。

## 启动条件与命令

1. 从原实验机器取得完整 `outputs/`、权重、共用 GPU ledger，并核对源数据与 checkpoint 哈希、各阶段曝光次数及历史总耗时。把负责人对 `google/flan-t5-small` 固定 revision 的确认记录归档；在 `config.json` 中据实填写 `manager_confirmed`。
2. 核对后据实更新 `config.gvr.json` 的三个 P0 证据位，设置统一 ledger 路径。不要复制一份新 ledger 来“重置”20 GPU 小时。新 G、V、R 各最多两轮；任何 pilot 单独记账。若课程按累计不同样本计，当前索引复用原 4,000 条，不引入新样本。
3. 在目标 Python 3.11／CUDA 环境安装依赖，然后按顺序运行：

```powershell
python scripts/run_gvr.py index
python scripts/run_gvr.py pilot --stage generation
python scripts/run_gvr.py train --stage generation
python scripts/run_gvr.py pilot --stage verification
python scripts/run_gvr.py train --stage verification
python scripts/run_gvr.py pilot --stage repair
python scripts/run_gvr.py train --stage repair
python scripts/run_gvr.py verify
python scripts/fetch_official.py
python scripts/run_gvr.py evaluate --split dev
python scripts/run_gvr.py evaluate --split test_id
python scripts/run_gvr.py evaluate --split test_ood
```

开发集用于模型选择；测试集是复用测试集。评测前核对 V／R 的 `base_identity.json` 与 G 权重哈希一致。评测保留模型原始反馈，格式错误计入失败；自验证分支不调用执行器做决策。所有模式从同一网格生成，评测器检查首个候选逐字一致。B4 相比 B3 使用更多调用，须连同模型调用、执行器调用和耗时一起呈报。

## 尚待完成的实际实验交付

训练后需要检查 LoRA 匹配模块数、可训练参数、底座冻结、保存重载与关闭适配器输出一致性；用正式 GPU 输出填入成功率、可行率、最优率、Verify Macro-F1、首错定位、Repair 成功率、按错误类型样本数与两个真实修复案例。课程报告与展示材料只有真实结果产生后才能完成。A 与 B 同时改变训练结构；若未做可选 C 全参数任务分支，不能把 A/B 差异单独归因于 LoRA。
