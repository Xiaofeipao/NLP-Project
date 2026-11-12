# PPNL：自然语言网格路径规划

本项目研究小型语言模型在单目标网格中的生成、验证和修复能力。模型输出 `up / down / left / right`；严格执行器报告 success（合法且到达目标）、feasibility（所有动作合法）和 optimality（成功且等于 BFS 最短路）。

## 当前主线

**固定已有多任务全量微调权重，比较后续 Harness 的处理方式。** 同一个 Flan-T5-small checkpoint 提供 G、V、R；用配置切换自验证、执行器反馈、修复轮数、动作约束、回溯及反馈文本。规则搜索提供移除神经评分的对照。

2026-11-12 台式机复跑已完成：13 个配置、完整 dev/ID/OOD、176 项测试通过。端到端 C4 成功率 92%，回溯 96.5%，对应最优率 43%/43.5%；Manhattan 规则对照为 100%/99.5%。完整三率、错误分组、成本和逐例轨迹见下方复跑报告。

| 内容 | 入口 |
| --- | --- |
| 按课程顺序核对要求、设计、历史结果 | [Project 6 整理](docs/project6_overview.md) |
| 当前处理流程、V/执行器边界、消融含义 | [固定权重后的 Harness](docs/post_training.md) |
| 本轮台式机 GPU 实跑 | [复跑报告](docs/post_training_run_20261112.md) |
| 配对方法配置 | [post_training.json](configs/post_training.json) |
| 数据来源、错误构造、指标分母 | [数据](data/README.md) · [协议](docs/experiment_protocol.md) |
| 历史模型与任务范围 | [方法目录](methods/README.md) · [项目地图](docs/project_map.md) |

## 运行

环境见 [安装说明](docs/environment.md)。先恢复指定权重并获取官方执行器：

```powershell
git lfs pull --include="artifacts/historical_models/flan_t5_small/train/best/**"
python scripts/fetch_official.py
python -m unittest discover -s tests -q
python scripts/run_post_training.py preflight --spec configs/post_training.json
python scripts/run_post_training.py run --spec configs/post_training.json --out outputs/post_training/run_001
```

中断后使用完全相同参数并加 `--resume`。输出目录冻结代码、模型、数据和配置；变更协议后需新建输出目录。`--limit 2 --splits dev` 可运行独立小规模检查，结果标为 smoke。跨机器可用 `--checkpoint`、`--dataset-root`、`--executor`、`--ledger` 覆盖资源路径。

默认 dev 为 G40/V100/R60，ID/OOD 各 V200/R100；端到端使用相同的 200 个唯一环境，初始 G 和 planning 会话只算一次。所有方法使用同一批初始路径，不重新训练。GPU 账本沿用历史消耗，不随运行目录重置。

单网格调试仍可使用 `python -m methods.gvr run -- --help`；批量实验统一为 `python -m methods.gvr study -- --help`。四个旧批处理脚本已退出默认执行路径，原始源码和迁移说明见 [脚本目录](scripts/README.md)。

## 代码职责

| 模块 | 职责 |
| --- | --- |
| `ppnl/core.py`、`data.py` | 严格执行器、官方核对、按环境划分和错误注入 |
| `ppnl/modeling.py`、`prompts.py` | 共享多任务模型、任务 prompt、GPU 预算 |
| `ppnl/post_training.py` | 方法矩阵、配对缓存、冻结身份、统一汇总 |
| `ppnl/gvr_harness.py` | 状态机、反馈来源、修复轮数、会话日志与恢复 |
| `ppnl/full_finetune_gvr.py`、`constrained_decoding.py` | G/V/R 适配、R 的反馈与搜索消融 |
| `ppnl/evaluation.py` | 路径指标和验证指标 |
| `ppnl/prefix_state_repair.py` | 前缀与重启的公共补全逻辑 |

V 是模型预测，可能误判；执行器是确定性程序。self 使用 G→V→R，修复后再次 V；executor 方法使用 G→执行器→R，不在推理中调用 V。B4 使用四动作/EOS 束搜索，C4 还使用状态约束；C4 可能返回合法非最短路径，不能把高成功率等同于最优规划。

## 历史实验与范围

课程主线先按相同 4,000 条多任务样本、两轮训练比较 Flan-T5-small 与 BART-base，再由开发集选择 Flan-T5-small。已有历史权重见 [历史产物](artifacts/historical_models/README.md)。本轮复用该主模型；历史基线和当前复跑分别标注环境、协议及结果。

逐步 policy/Agent、LoRA、RL 和其他模型的研究保留为独立扩展：它们并不都使用同一个 G/V/R checkpoint。来源和结论见 [三模型 campaign](docs/gpu_campaign_20261103.md)、[Agent Harness](docs/harness.md)、[GVR LoRA](docs/gvr_gpu_run.md)、[前缀实验](docs/project_map.md)。C4 已集成到当前代码，可选回溯也已实现；研究分支是否仍存在不代表实现缺失。

Poster 包含不同训练规模和提示实验，其百分比不能与本仓库 200 个环境的结果直接合并，见 [poster 对照](docs/poster_guide.md)。早期 README 的逐日记录可在 [历史提交](https://github.com/Xiaofeipao/NLP-Project/blob/23183e2/README.md) 查看。

官方执行器只用于核对可比语义；本地扩展明确处理未知动作词、首错位置和未达目标。最优性按动作数计算。数据集已被历史实验使用，本轮冻结是可复现协议，不宣称为全新的未见测试。
