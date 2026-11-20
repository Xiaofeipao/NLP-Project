# PPNL 单目标网格训练数据

本版以 `StepVerification-v1-v2` 的 `9eeb494` 为基础。训练集有 2,000 条 G 和 2,000 条 V，开发集各 100 条。G 学习选下一步，反复调用即可生成完整路径。V 同时学习验证一步和验证整条路径。独立监督的 R 样本数为 0。

## G 怎样保留多条最短路

从目标反向搜索每个格子的最短距离。一个动作可执行，且能让剩余距离恰好减 1，就纳入 `optimal_actions`。下一步的损失为 `-log(sum(P(a) for a in optimal_actions))`，在四个方向上归一化。`target` 是可读的代表动作。训练时使用完整集合。在 2,000 个 G 状态中，911 个有一个最优方向，1,079 个有两个，10 个有三个。531 个状态必须先增大与目标的曼哈顿距离。

例如实际样本 `train-2516` 从 `(5,2)` 去 `(4,1)`，地图的障碍为 `(1,2)` 和 `(3,1)`。该行的 `target` 为 `left`，`optimal_actions` 为 `["up","left"]`。相同状态下，V 对完整候选 `up` 报第 2 步提前结束，对 `left up` 报 `valid`。三行可按同一 `state_id` 对照。

## V 的监督和类别分布

| 训练任务 | 有效 | 未到目标 | 撞墙 | 越界 | 非法词 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单步 V | 340 | 380 | 160 | 160 | 160 | 1,200 |
| 整条路径 V | 400 | 100 | 100 | 100 | 100 | 800 |

单步 V 回答“执行这一动作后立即结束，是否成功”。300 次合法移动后仍未到达，局部错误步号为 2。空动作表示原地停止，在目标处的 80 次为有效，在其他位置的 80 次报第 1 步提前结束。移动违规统一报局部第 1 步。逐行标签分别记录动作合法性、可定义时的后继位置和是否到达。对空动作，动作合法性字段取空值。对违规动作，后继位置及到达字段取空值。

整条路径 V 检查所有动作，以全局步号标记首错。400 条无效候选均来自对一条真实有效的最短路做一次替换、插入、删除或截断。截断算一次操作，可能移除多个动作。400 条有效候选含 200 条最短路、67 条往返绕行、67 条四步绕行、66 条不重复访问格子的较长路线。训练中的有效路径允许绕行。

## 复用状态与防捷径检查

地图尺寸、坐标约定、当前位置、目标和排序后的障碍列表使用统一模板。`states.jsonl` 按 `state_id` 存一份状态，训练和部署调用 `schema.render`。模型输入只有状态、任务、必要的候选和固定输出要求。真值、样本 ID、来源和最优动作集合放在输入之外。训练可直接读取已渲染的 `train.jsonl`。

有 705 个 G 状态与 1,058 条 V 共用。其中 700 个 G 状态是按 V 标签主动选入的，有效、未到目标、撞墙、越界、非法词各选 140 个。一个状态还可以关联其他 V，全部 1,058 条复用 V 的实际构成为有效 375、未到目标 173、撞墙 165、越界 175、非法词 170。每条监督分别占用一个配额。

局部验证中，120 对固定动作只改目标，80 对在原地停止并只改目标，80 对只增减动作目的格上的一面墙。另有 80 对边界位置对照和 80 对动作词对照。360 条局部样本直接追溯到原始候选的具体步号，另有 120 条用原始步号构造了目标改变的对照。旧的固定规则“空串报未到、遇到 jump 报非法、其余全判有效”在新版局部训练样本上精确答对 365/1,200。这是规则回放结果。

## 文件与验收

| 文件 | 用途 |
| --- | --- |
| `data/step_semantic_v3/train.jsonl`、`dev.jsonl` | 按原环境划分重采样，覆盖全部 652 张训练布局和 76 张开发布局 |
| `test_id.jsonl`、`test_ood.jsonl` | 各 200 条。按原顺序保留地图、候选、标签、编辑和参考路径，使用新版整条路径验证模板。两个测试集共用原来的 79 张测试布局 |
| `states.jsonl`、`audit.json`、`manifest.json` | 状态登记、逐类验收结果和 SHA-256。训练读取器只加载所选分区 |
| `step_data_v3/build.py`、`sampling.py`、`audit.py` | 生成样本，回放原始执行器，逐条核验最优动作集合、标签、成对条件和环境划分 |
| `step_data_v3/model_io.py`、`smoke.py` | 接入 v2 的 Flan-T5-base 训练和推理接口，以及离线 CPU 集成测试 |

原执行器核验到达和撞墙。原执行器会忽略未知词，因此非法词和首错位置沿用项目的严格词表与逐步回放规则。种子为 42，开发集使用独立随机流。数据集已连续重建两次并核对全部发布文件的字节。测试集用作对照，开发集的样本组成经过重建。旧分支与本版的开发分数需要用相同样本重新评估才能直接比较。

## 在 v2 上复现与接入

从 `StepVerification-v1-v2` 仓库根目录运行。将 `$executor` 指向原版 `executor-point-sg.py`。重建命令使用独立的输出目录。

```powershell
$executor = 'D:/UserData/Documents/ChatGPT/NLP_Project/vendor/ppnl/executor-point-sg.py'
python -m step_data_v3.audit --data data/step_semantic_v3 --source data/derived --executor $executor
python -m step_data_v3.build --source data/derived --executor $executor --out data/step_semantic_v3_rebuilt
```

数据和模型输入接口直接复用 v2 的网格、最短距离、序列读取器和损失实现。下面的接口接受在 v2 分支上从选定 checkpoint 加载的 `ppnl.modeling.Seq2Seq` 实例，配置使用 `policy_loss="optimal_actions"`。

```python
from torch.utils.data import DataLoader
from step_data_v3.model_io import prepare_training, batch_loss, StatePolicy, verify_plans

features = prepare_training(runner, "data/step_semantic_v3")
loader = DataLoader(features, batch_size=1, shuffle=True, collate_fn=runner.collate)
# 在训练循环中用 batch_loss(runner, batch) 取得每个混合批次的损失。
# 保存模型时使用 runner.model.save_pretrained(...)，模板指纹随配置保存。
# 部署用 StatePolicy(runner).scores(...)、verify_plans(runner, grids, plans)。
```

训练批次中，G 使用集合损失，V 沿用 v2 Flan-T5-base 对首个判定 token 和后续诊断各赋 50% 权重的损失。G/V 复用体现在显式样本中，原先 G 上恒定的停止辅助标签已移除。旧训练参数 `--dual-train` 会重建旧版监督，接入新版请使用上面的模型接口。局部验证可调用同文件的 `verify_moves`。

验收使用 Flan-T5-base 分词器和一个随机初始化的小型 T5 进行 CPU 前向、反向与生成，训练和推理的三个模板已逐字核对。对同一批状态与候选分词，G 的平均输入长度由 166.92 降至 64.65 个 token，单步 V 由 189.84 降至 130.05，整条路径 V 相对仓库原有整段模板由 144.40 降至 119.26。全部 4,600 条目标文本分词后都能精确还原。新数据的任务成功率、首错定位和修复成功率仍需在同预算微调后测量。课程第二部分还涉及模型修复。已有规则修复可作对照，用同一语言模型按反馈做提示修复也无需单设 R 训练样本。
