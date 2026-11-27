# Part 1 · 单目标 6×6 网格基线

第一组使用同一个模型完成 G（选下一步）、C（算落点）、V（判断落点）。Flan-T5-base 和 BART-base 仅在开发集比较。封存主模型后，评估 hope_dataset 修订版双错验证集 200 条，再评估单错测试集 200 条。[hope_dataset 数据修订](https://github.com/Xiaofeipao/NLP-Project/commit/d88f85b872560f9313ac4319cdd5fa86151c0780)[结果与逐题记录](../../artifacts/part1_baseline/Part1_基线结果.md)

## 数据、提示与训练

4,000 个训练状态来自 652 种地图布局。每个状态有一道 G 题，四个方向各有一道 C 题和一道 V 题。其中 400 个状态各附一个非法动作词。这些视图合计 G 4,000、C 16,400、V 16,400。五类 V 标签依次为合法未到 9,571、越界 2,767、撞墙 2,323、到达 1,339、非法词 400。数据制备逐题核对金标、四方向和最短方向集合。[状态清单](../../artifacts/part1_baseline/data/state_ids.json) · [数据核对](../../artifacts/part1_baseline/data/state_audit.json) · [协议](../../artifacts/part1_baseline/data/protocol.json)

| 任务 | 模型可见输入 | 答案与损失 |
|---|---|---|
| G | Current、Goal、Walls | 所有并列最短方向都正确。对完整正确动作的概率求和后取负对数，推理按四个方向得分排序。 |
| C | Current、Action | 输出一步后的坐标或 `no_coordinate`。对目标串含 EOS 的 token 交叉熵求和。 |
| V | 落点作为 Current、Goal、Walls | 输出五类之一。首个内容 token 上做一次五分类交叉熵，推理使用相同五个分数。 |

三个任务各自共用同一份训练和测试提示。训练 V 使用离线标注的落点。推理时它收到 C 的预测。第一组以 `C → GV → GV` 重复 125 次，每组更新使用 32 个状态，C 遍历一次，G 和 V 各遍历两次。两种模型均使用 375 次更新、种子 42、AdamW、学习率 0.0001、权重衰减 0.01、微批 4、梯度裁剪 1 和 BF16。每遍 GV 的损失中，G 占一半，各 V 类占 10%，类内按候选词均分。Flan 和 BART 的五个标签均在各自解码前缀之后取分。[G 提示](../path_text_gv/prompts/coordinate_gcv/G.txt) · [C 提示](../path_text_gv/prompts/coordinate_gcv/C.txt) · [V 提示](../path_text_gv/prompts/coordinate_gcv/V.txt) · [损失](model.py) · [训练入口](train.py)

## 完整路径的输出与评分

C、V 从起点依次检查候选路径，步号从 1 开始。首次预测撞墙、越界或非法动作词，流程用该步序号和 V 的类型输出 `invalid | step=<index> | reason=<error_type>`。中途走到目标仍继续检查后续动作。走完整条路径时，合法到达输出 `valid`，合法未到输出第 `k+1` 步的 `goal_not_reached`。例如，四步都能走但最终没到，应输出 `invalid | step=5 | reason=goal_not_reached`。模型异常输出或未提交路径分别计为验证或修复失败。[逐步推理](policy.py) · [评分](evaluate.py)

**整行正确率**按输出和金标逐字比较。**结构化宏 F1**按五种语义类别统计，某类只有整行相同才获得 TP。步号错而原因对会给该类同时增加一个 FP 和一个 FN。该类 F1 为 `2 × TP / (金标题数 + 预测题数)`，对该集出现的金标类别等权平均。开发集覆盖五类，单错覆盖四类，双错覆盖三类。误报 `valid` 计为该题错误。产生错误类型和步号的过程只使用题面与模型输出，评分才读取金标。

开发集有 100 条候选（50 有效、50 无效），覆盖 98 个起点、目标和墙布局的不同组合。选型对直接生成成功率、结构化宏 F1、无效候选修复成功率取等权平均，并列时依次看结构化宏 F1、修复率和预设优先项 Flan。两套数据保留原题号、地图、顺序及首错金标，合计替换 29 处大写方向词。数据制备按替换清单核对每个动作位置，再用执行器校验金标。验证指标只评分首错。四组布局（训练、开发、单错、双错）经核对互不重合。[选型记录](../../artifacts/part1_baseline/selection.json) · [单错与双错的构造](../../data/random_errors_20261123/README.md)

## 生成与修复

直接生成时，G 选动作、C 算落点，预测到目标或用完 36 步时停止。候选路径修复保留首错前缀，由 G 重选，C、V 迭代，最多再提议 36 次。生成路径另做模型反馈与执行器反馈两种修复，两者使用同一条初始路径和提议上限。执行器只在指定的反馈对照、离线制备和评分时使用。[推理](policy.py) · [评测入口](study.py)

直接生成对照与完整端到端流程都从空路径开始，以相同的全部评测环境为分母，计算可行率、成功率和最优率。可行要求提交路径的每一步都合法，成功还要求到达目标，最优还要求与该环境的最短路径等长。给定候选路径的修复评测以全部原本无效的候选路径为分母，计算修复后可行率和修复成功率。[分流程结果](../../artifacts/part1_baseline/Part1_基线结果.md) · [逐项计数](../../artifacts/part1_baseline/project_metrics.json)

## 复现

台式机运行环境为 Python 3.12.10、PyTorch 2.11.0+cu128、Transformers 5.20.0、tokenizers 0.23.3。`$bases` 含两个固定版本的预训练目录，`$history` 为原第一组运行根目录。带 `--history` 可复用已验证的 Flan 检查点，省略该选项可从固定的原始权重训练。首次运行先用 `python scripts/fetch_official.py` 获取固定版本的 PPNL 执行器。代码按 SHA-256 验证训练数据、盲测输入、权重和已完成的评测回执。

```powershell
python -m experiments.part1_baseline.protocol
python -m experiments.part1_baseline.audit
python -m experiments.part1_baseline.study --bases $bases --history $history
python -m experiments.part1_baseline.study --recheck
python -m experiments.part1_baseline.audit --results
python -m experiments.part1_baseline.predict --task verify --grid examples/grid.json --candidate "left up"
python -m unittest discover -s tests -p test_part1_baseline.py -v
```

默认预测入口加载开发集选出的模型。`--task generate` 返回动作串，`--task verify` 返回验证整行，`--task repair` 返回修复路径，`--trace` 附逐步记录。[预测入口](predict.py)

`--recheck` 从空目录重新进行开发集选型、双错验证和单错测试，核对两次的每份预测和评分。`audit --results` 另算课程的有效/无效二分类宏 F1、首错定位、候选修复和三个环境级流程的指标。[课程指标明细](../../artifacts/part1_baseline/project_metrics.json) · [错误分析](../../artifacts/part1_baseline/第一组_错误分析.md)

## 修复时软惩罚重复访问

在预测入口为修复或完整流程加上 `--soft-revisit`，例如
`python -m experiments.part1_baseline.predict --task repair --grid examples/grid.json --candidate "left up" --soft-revisit --trace`。

保留 Hope 已拒方向过滤。对剩余方向按 G 的排名（从 0 开始）加上 C 预测落点历史访问次数的两倍，选分数最小的动作，同分沿用 G 的排名。历史包括起点、已接受前缀和续走中接受的落点。每次最多向 V 提交一个动作，每题仍最多提交 36 次；排序可能增加 C 的预测。直接生成与原默认修复策略保留。

在 hope_dataset 提交 d88f85b 的单错和双错候选上，固定 Hope 的 Flan-T5-base 权重后，原修复成功数分别为 165/200 和 141/200，软惩罚为 179/200 和 170/200。规则应用在推理阶段。
