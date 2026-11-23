# Flan-T5-base 单步 G/V 的 64 条验证样本替换实验

4,000 条共享状态版的训练集、第 3 轮完整权重，以及原数据 A 与共享状态版在新单错、双错测试集上的结果已发布。[新训练数据](../../data/step_g_shared_4000/) · [新模型权重](models/shared4000-epoch3/) · [新测试集结果与逐例预测](shared4000/)

在 64 条替换实验中，B 组使用替换了 64 条 V 样本的数据，第 3 轮入选。A 组在原数据上重新训练，第 1 轮入选。两个模型均从同一份 Google Flan-T5-base 权重全参数微调，包含分词器和两项任务的权重。[B 组模型](models/B-epoch3/) · [A 组模型](models/A-epoch1/) · [基座模型卡](https://huggingface.co/google/flan-t5-base) · [模型许可证](models/LICENSE)

## 训练数据

| 版本 | 训练集 | G 行数 | V 行数 | 不同训练状态 |
|---|---|---:|---:|---:|
| A 原数据 | [查看训练集](../../data/step_g_direction_controls/train.jsonl) | 2,000 | 2,000 | 2,264 |
| B 等额替换 | [查看训练集](../../data/step_g_direction_controls_swap64/train.jsonl) | 2,000 | 2,000 | 2,264 |

每行含一个训练输入和监督目标。地图为 6×6，坐标从 0 开始，按行、列排列。G 根据墙、当前位置和目标预测一个最短下一步，`optimal_actions` 列出所有并列最短方向。V 接收相同的状态描述和一个候选动作。每组的五个 V 标签各 400 条。在到达、合法未到、撞墙、越界这四类中，每个方向各 100 条。第 5 类是非法动作词。[A 组文件清单](../../data/step_g_direction_controls/manifest.json) · [B 组文件清单](../../data/step_g_direction_controls_swap64/manifest.json)

B 组将 32 条撞墙和 32 条越界样本换成同类、同方向的样本。每类四个方向各替换 8 条。G 的输入和最短方向集合一致。A、B 两组使用相同的初始权重、训练代码和逐轮行索引顺序。两组使用相同的开发及测试文件。替换后，训练中同时含“撞墙、合法未到”的状态从 147 个增至 163 个，同时含“越界、合法未到”的状态从 106 个增至 122 个。[64 条替换逐行对照](replaced_verification_examples.jsonl)

例如，当前位置为 `(4,5)`，目标为 `(5,5)`，墙在 `(3,5)`。G 的最短下一步是 `down`。对于 V，`down` 对应 `valid`，`up` 对应 `obstacle_collision`，`left` 对应 `goal_not_reached`，`right` 对应 `out_of_bounds`。

## 训练和评测

两组均训练 3 轮，随机种子为 42，训练行逐轮打乱，使用 AdamW、学习率 `1e-4`、权重衰减 `0.01`、微批次 4、有效批次 32 和 BF16。G 的损失为所有最短方向的总概率取负对数。V 只对解码器首位的五个标签分数计算一次交叉熵。训练环境使用 PyTorch 2.11.0+cu128、Transformers 5.20.0 和 safetensors 0.8.0。发布记录包含基座版本和每份权重的 SHA-256。[配置与结果](results.json)

整路开发集含 50 条有效路径和 50 条无效路径。V 从 `valid`、`obstacle_collision`、`out_of_bounds`、`invalid_action_token` 和 `goal_not_reached` 中选一个标签。评测启用程序终点检查。动作的合法落点等于目标时保留全部五个分数，其他情况屏蔽 `valid`。固定候选逐步验证，首个非法动作按从 1 开始的步号报告。合法未到在中途可以继续，末步仍未到达时报告动作数加 1。

| 模型 | 整路开发宏 F1 | 有效路径放行 / 50 | 无效路径误放行 / 50 | 首错位置正确 / 50 |
|---|---:|---:|---:|---:|
| A 第 1 轮 | 0.9900 | 50 | 1 | 40 |
| B 第 3 轮 | 0.6128 | 17 | 2 | 27 |

[逐轮结果](results.json)列出 A、B 从第 0.5 轮到第 3 轮的 G 最短首步、整路到达、V 五类召回和整路误放行。两组同为第 2 轮时，G 都答对 54/100 个首步、走通 3/100 条路径，V 均答对 0/20 个“合法未到”动作。

先按固定候选路径有效／无效二分类宏 F1 选轮次，再按首错位置和位置、类型联合命中打破平局。开发路径及逐例预测均可按 ID 对照。[完整路径开发源文件](../../data/derived/dev.jsonl) · [两份入选权重的开发预测](evaluation/selected_dev_predictions.jsonl)

B 第 3 轮还在原有的单错和双错测试集上评估，每组 200 条。对重复出现的相同状态与动作统一打分再重放，整路宏 F1 均为 0.6769。100 条无效路径中，单错首错位置命中 59 条，双错命中 63 条。两组共用 100 条合法控制路径。[单错逐例结果](evaluation/B_epoch3_test_id.jsonl) · [双错逐例结果](evaluation/B_epoch3_test_ood.jsonl)

## 与后续 4,000 个共享状态版本的对照

两次实验均使用同一 Flan-T5-base 基座和提示模板，G 接受并列最短方向，V 对五个标签的首 token 分数计算交叉熵。两版都使用种子 42、学习率 `1e-4`、AdamW、BF16、微批次 4，每轮更新 125 次，最多 3 轮。推理均使用程序终点检查。

| 训练细节 | 本分支 A/B | 后续共享状态版 |
|---|---|---|
| 一次更新取样 | 随机打散的 32 条独立 G 或 V 输入 | 32 个状态，每个状态的 G 与全部 V 判断一起更新 |
| 每轮监督量 | 2,000 个 G、2,000 个 V | 4,000 个 G、16,000 个四方向 V 判断，加上同一批状态中附带的 400 个非法词判断 |
| 损失汇总 | 各行等权，依靠样本配额平衡任务和类别 | 显式给 G、V 各一半总权重，V 五类各占 V 权重的 20%，类内按方向或非法词均衡 |
| 选择检查点 | 每半轮评一次完整开发路径 | 每整轮评一次完整开发路径 |

按整轮加总，旧版和共享状态版都给 G、V 各一半权重，V 五类各占五分之一。共享状态版改变的是每次更新看到的判断和每条判断的相对权重。

### 新单错、双错测试集

两个测试集各有 200 条无效候选路径。每列单位为条，分母为 200。V 的首错联合命中要求类型和步号均正确。G 从同一张地图的起点独立生成路径，按执行结果统计到达。三个检查点使用相同的测试文件和终点检查。[冻结测试集](../../data/random_errors_20261123/) · [逐类结果、混淆矩阵与逐条预测](shared4000/)

| 检查点 | 单错 V 检出 | 单错 V 首错联合命中 | 单错 G 到达 | 双错 V 检出 | 双错 V 首错联合命中 | 双错 G 到达 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原数据 A 第 1 轮（开发集选中） | 119 | 83 | 10 | 152 | 51 | 16 |
| 原数据 A 第 3 轮 | 189 | 76 | 67 | 193 | 95 | 43 |
| 共享状态第 3 轮（开发集选中） | 199 | 63 | 115 | 200 | 95 | 113 |

同取第 3 轮，G 在 400 张地图上到达的数量由 110 增至 228。V 首错类型和位置同时正确的数量在单错集由 76 降至 63，在双错集均为 95。两个新测试集只含无效候选路径。含有效路径的开发集另有记录。[同轮对照](shared4000/fixed_epoch_comparison.json) · [开发集选中模型对照](shared4000/paired_summary.json) · [共享状态版逐轮开发结果](shared4000/new_model/train_result.json)

## 获取权重与运行一个 V 样本

在本分支的工作目录，用 Git LFS 获取所需的模型文件。每个 `model.safetensors` 为 990,345,064 字节。

```bash
git lfs pull --include="artifacts/flan_swap64_ab/models/B-epoch3/model.safetensors"
```

下例在仓库根目录运行，直接读取训练文件中的真实提示。数据中的目标坐标均为界内可通行的格子。

```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = Path("artifacts/flan_swap64_ab/models/B-epoch3")
data_dir = Path("data/step_g_direction_controls_swap64")
with (data_dir / "train.jsonl").open(encoding="utf-8") as stream:
    row = next(r for r in map(json.loads, stream) if r["id"] == "train-balanced-V-0055")
with (data_dir / "states.jsonl").open(encoding="utf-8") as stream:
    state = next(s for s in map(json.loads, stream) if s["id"] == row["state_id"])

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).eval()
inputs = tokenizer(row["input"], return_tensors="pt")
start = torch.tensor([[model.config.decoder_start_token_id]])
labels = model.config.ppnl_verifier_labels
with torch.inference_mode():
    logits = model(**inputs, decoder_input_ids=start).logits[0, 0]
    scores = logits[model.config.ppnl_verifier_token_ids].clone()
delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}.get(row["candidate"])
if delta is None or [v + step for v, step in zip(state["current"], delta)] != state["goal"]:
    scores[labels.index("valid")] = -torch.inf
print(row["candidate"], labels[scores.argmax().item()], row["target"])
```

输出依次是候选方向、模型预测和参考标签。在 CPU 上用 Transformers 5.12.1 以 FP32 加载两份发布权重，这个样本的输出都是 `down goal_not_reached valid`。候选动作向下恰好到达终点，两份模型在该输入上都漏判了到达。上表的正式评测使用 GPU 的 BF16 批量打分。

原始配置中 `tie_word_embeddings` 为真，权重文件分别保存了数值不同的 `shared.weight` 与 `lm_head.weight`。上述训练和加载版本都保留了这两个张量，加载日志会提示这项差异。
