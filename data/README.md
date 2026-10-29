# 数据来源与目录说明

本目录包含官方直接下载的原始文件，以及项目加工的训练、开发和测试数据。模型权重、缓存和运行日志不在此目录，也不随数据上传。

## 官方原始数据：`official/`

来源：[论文作者的 llms-as-path-planners 仓库](https://github.com/MohamedAghzal/llms-as-path-planners/tree/5fd0376fc50ae9860bc986085165ebac5675fd44/ppnl-spatial-temporal-reasoning/single_goal)。固定上游版本为 `5fd0376fc50ae9860bc986085165ebac5675fd44`。

- `1_train_set_6x6_samples.json`
- `1dev_set_6x6_samples.json`
- `1_goals_test_seen_6x6_samples.json`
- `1goals_unseen_6x6_samples.json`

保留原始下载内容，字段包括 `world`、`nl_description`、`solution_coordinates`、`agent_as_a_point` 和 `agent_has_direction`。各派生目录 manifest 的 `official_sources` 记录原文件 SHA256。数据来源和权利归属上游作者，使用时请同时查阅上游说明及引用要求。

## 多任务样本：`derived/`

由 `ppnl/data.py` 处理官方网格，按障碍物布局重新划分，构造路径生成、验证和修复任务。这里不是直接沿用官方文件名中的 train/test 划分。

- `train.jsonl`：4,000 条，生成 800、验证 2,000、修复 1,200。
- `dev.jsonl`：200 条，生成 40、验证 100、修复 60。
- `test_id.jsonl`、`test_ood.jsonl`：各 200 条，使用配对网格，候选路径错误数量不同。
- `manifest.json`：来源、布局划分、样本计数、执行器哈希和构造规则。

## 逐步策略样本：`policy_v2/` 和 `policy_v3/`

`policy_v2/` 将原规划/修复训练样本转换为状态到下一动作的监督。`policy_v3/` 增加训练关键状态采样及多最优动作标签。两个目录保留验证任务，测试文件与 `derived/` 相同，不将测试目标转换为训练样本。

## 使用与复现

从项目根目录运行。样本 JSONL 的每一行是一个完整 JSON 对象。发布的 manifest 使用仓库相对路径，移除了机器安装目录；此路径元数据调整不改变样本、布局划分或原始文件哈希，但会改变 manifest 本身的字节指纹。历史训练数据指纹对应当时的 manifest，不能直接视为发布版指纹。

官方执行器仍通过脚本获取：

```powershell
python scripts/fetch_official.py
python -m ppnl.cli audit
```

下载脚本校验已有官方文件，并获取 `vendor/ppnl/executor-point-sg.py`。如需重新构造数据，应使用新的 `dataset_root`，避免覆盖已发布数据。`.gitattributes` 禁用数据文件的自动换行转换，以保留原始字节及校验值。
