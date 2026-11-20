# 单目标网格中的单步 G/V

本分支以 [逐步验证 v1 与双标签 v2](step_verification_v1_v2.md) 为基线，研究同一 Flan-T5-base 如何在共享网格状态上预测下一方向和判断单个候选动作。正式实验使用 `data/step_g_paired_actions`。

## 任务定位

| 层次 | 输入 | 输出及用途 |
| --- | --- | --- |
| G · 单步生成 | 网格大小、当前位置、目标、墙格 | 四方向中任选一个最短路方向。若有多个等长最优方向，训练对它们的概率和取负对数。 |
| V · 单步验证 | 同一状态、允许的四个动作词、待检查的一个动作 | 输出一个结果标签。各方向共享状态，独立构造 V 样本。 |
| 程序 · 生成与验路径 | 地图、当前状态和已走动作，或一条给定的候选路径 | 维护坐标、调度 G/V、拆分路径、收集首个错误步号。回撤和重选由程序调度。 |

V 的五个标签如下。

| 标签 | 当前动作的含义 |
| --- | --- |
| `valid` | 动作合法，且落到目标 |
| `goal_not_reached` | 动作合法，落到其他格 |
| `obstacle_collision` | 目的格有墙 |
| `out_of_bounds` | 目的格越界 |
| `invalid_action_token` | 输入不属于 `up / down / left / right` |

在线生成时，程序把当前状态交给 G，将候选动作交给 V。若 V 判为合法未到，推进状态并继续生成。已核验前缀上的一步被判为 `valid` 时可结束生成。验证给定路径时，程序把它拆成连续的“动作前状态＋一个动作”，逐步调用同一个 V。即使中途经过目标，也继续检查后续动作。全程动作合法但终点未到时，沿用原基准的“动作数＋1”步号。

修复按规则处理。先定位首个失败动作，回到该动作的执行前状态并删除失败动作，从其余候选方向里重选，然后回到 G/V 生成流程。路径只是在到达前提前结束时，从合法前缀的末尾继续。正式监督只有 `g_next` 和 `v_step`。空路径、路径结束和重选由外层逻辑处理。已发布的训练结果覆盖单步、G 贪心生成和固定候选路径验证。

## 数据和核对

正式训练包含 2,000 条 G 和 2,000 条 V，开发各 100 条。V 五类各 400 条，物理动作对应的四个标签内每个方向各 100 条。复用 G 和旧 V 的同一状态，只替换 `Last move`，标签由执行规则重算。一个状态的四个动作仍是四条独立输入。标签、数据来源和 G 的最优动作集合分别保存在监督或元数据字段中。

用 302 条新动作题等量替换同类同方向的旧题。同一状态同时问到“到达”和“合法未到”的从 52 个增到 200 个，四方向齐全的从 155 个增到 173 个。固定总量后，原来的换墙、改目标及改边界对照有 30/100 组完整进入正式训练，其余保存在诊断库。训练来源的 8,000 条 G 派生题和 5,872 条旧 V 派生题均按“状态＋动作”标记是否来自正式 V。训练入口只加载 `train.jsonl`。开发和固定单错、双错路径保持冻结。

| 文件 | 用途 |
| --- | --- |
| [最终数据](../data/step_g_paired_actions/) | 正式划分、统一模板、动作组和审计 |
| [早期数据、来源与版本](../data/README.md) | 从原 v2 到最终配对策略的构建资料 |
| [数据调整与核对](../artifacts/step_verification/step_g_paired_actions/数据调整与核对.md) | 旧 V 的四方向真实样例、类别分布、重复、捷径和分区核对 |
| [构建与训练代码](../step_data_v3/) | BFS 目标、单步渲染、数据选择、两种损失及测试 |
| [三轮实验](../artifacts/step_verification/step_g_paired_actions/retrain_3ep/三轮对照结果.md) | 逐轮 V 各类召回、误判、G 和固定路径评分 |
| [历史对照与原始证据](../artifacts/step_verification/README.md) | 各阶段构建、训练日志、逐例预测和检查点索引 |

## 两种 V 损失

文本方案对所有类别统一使用 `0.5 × 首 token 交叉熵 + 0.5 × 后续非填充 token 的平均交叉熵`。五分类方案在 T5 编码器上做带掩码的均值池化，用线性头输出五个分数，对正确类别计算一次交叉熵。两组共享 G 的多最优动作损失。

RTX 3050 实测中，两组从同一份初始主干各训练三轮。第 3 轮开发 V 文本方案为 57/100、宏 F1 0.5221，五分类为 40/100、宏 F1 0.2800。文本第 2 轮误判为 `valid` 13 条，第 3 轮 31 条。五分类第 3 轮没有预测到达，全部物理动作被判为合法未到。在 10 个可一步到达的开发 G 状态上，文本第 2 轮有 1 个状态能只接受正确方向，两种第 3 轮模型均为 0 个。

## 复现

实际训练环境为 Python 3.12.10、PyTorch 2.11.0+cu128、Transformers 5.20.0。以下命令从项目根目录运行。预训练目录使用 `google/flan-t5-base` 的 `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` 权重，并在 `upstream_revision.txt` 中记录该版本。重建训练数据另需 SciPy 1.18.0，执行器由 `scripts/fetch_official.py` 获取。训练读取已发布数据时无需再次构建。

```powershell
python -m unittest step_data_v3.test_paired_actions step_data_v3.test_five_class step_data_v3.test_g_probes step_data_v3.test_last_move step_data_v3.test_single
python scripts/fetch_official.py
$base = 'path/to/flan_t5_base'
python -m step_data_v3.build_paired_actions --executor vendor/ppnl/executor-point-sg.py --tokenizer "$base/tokenizer.json" --out data/step_g_paired_rebuild
$opts = @('--data','data/step_g_paired_actions','--pretrained',$base,'--epochs','3','--microbatch','4','--no-pilot','--save-every-epoch')
python -m step_data_v3.train_gpu @opts --out outputs/step_g_text
python -m step_data_v3.train_gpu @opts --verifier-head --out outputs/step_g_five_class
```

冻结数据的生成依赖与正式训练入口随分支发布。现成 ZIP 的生成时间早于三轮实验，复现三轮训练使用本分支的训练代码。
