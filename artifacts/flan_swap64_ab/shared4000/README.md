# 4,000 条共享状态样例的 Flan 复测

同取第 3 轮，G 在 400 张新地图上完整到达的数量从 110 提升到 228。V 的首错类型与位置同时正确，单错集从 76/200 降到 63/200，双错集维持 95/200。换方向后，模型更能识别撞墙，但也更容易把合法的中间步骤提前判成撞墙。

## 数据、权重与逐条预测

这份模型用 4,000 条共享状态样例微调。每条都有一个 G 和四个方向的 V，其中 400 条在同一行另带一个非法词判断。训练文件按状态保存，`generation` 给出全部并列最短方向，`compass` 给出四个候选的提示和标签，`lexical` 保存该行附带的非法词判断。[训练集](../../../data/step_g_shared_4000/train.jsonl) · [状态表](../../../data/step_g_shared_4000/states.jsonl) · [数据校验值](../../../data/step_g_shared_4000/manifest.json) · [冻结测试集](../../../data/random_errors_20261123/)

共享状态版按开发集选中第 3 轮。完整的 Flan-T5-base 检查点包含 G/V 共用的权重、配置和分词器。原始权重为 990,345,064 字节。Transformers 分片索引覆盖全部 282 个张量。前 26 个分片使用 Git LFS，最后两个随普通 Git 检出。加载时将 28 个分片与索引放在同一目录。[模型目录](../models/shared4000-epoch3/) · [分片索引](../models/shared4000-epoch3/model.safetensors.index.json) · [逐文件校验值与原始权重 SHA-256](checkpoint_manifest.json) · [许可证](../models/LICENSE)

| 检查点 | 单错 V / G 逐例预测 | 双错 V / G 逐例预测 | 混淆矩阵及计数 |
| --- | --- | --- | --- |
| 原数据 A 第 1 轮（开发集选中） | [V](original/single_predictions.jsonl) · [G](original/single_generation.jsonl) | [V](original/double_predictions.jsonl) · [G](original/double_generation.jsonl) | [结果](original/scores.json) |
| 原数据 A 第 3 轮（同轮对照） | [V](original_epoch3/single_predictions.jsonl) · [G](original_epoch3/single_generation.jsonl) | [V](original_epoch3/double_predictions.jsonl) · [G](original_epoch3/double_generation.jsonl) | [结果](original_epoch3/scores.json) |
| 共享状态第 3 轮（开发集选中） | [V](new/single_predictions.jsonl) · [G](new/single_generation.jsonl) | [V](new/double_predictions.jsonl) · [G](new/double_generation.jsonl) | [结果](new/scores.json) |

按样例 `id` 可关联同一地图的测试输入、参考答案、V 首错预测和 G 生成路径。机器可读结果用仓库路径或权重 SHA-256 标识检查点。

## 训练与测试口径

正式训练集有 **4,000 条**。3,600 行各有一个 G 和四个方向 V。另 400 行同样有一个 G 和四个方向 V，并在本行包含一个非法动作词判断。G 接受所有并列最短方向。V 的 16,000 个物理方向判断中，到达 1,339、合法未到 9,571、撞墙 2,323、越界 2,767。400 个词表外判断来自 20 个词，各 20 次。每轮 G 和 V 各占总损失的一半，V 的五个类别各占 V 权重的 20%。

台式机 RTX 3050 完成三轮训练，共 375 次参数更新，约 94 分钟。同用 Flan-T5-base 初始权重、种子 42、学习率 0.0001、BF16 和大小为 4 的前向微批。新训练每次更新取 32 条共享状态并使用其全部监督项，原训练每次更新取 32 条已有的单任务样例。这是两个训练方案的整体比较，数据覆盖、每次更新包含的判断数及类别加权共同变化。[数据核对](共享状态数据核对.md) · [新训练记录](new_model/train_result.json)

测试固定为提交 `123bf5a1` 的单错和双错各 200 条。新、原两组训练地图与测试地图的精确布局重合数均为零。每条完整路径只根据单步结果依次判断首错。候选动作、地图、目标和当前位置进入模型，测试答案、错误组合和参考路径仅用于核对评分。终点检查对两模型同样生效。两个模型的受限读出和首错扫描都只产生约定的 `valid` 或 `invalid | step=<index> | reason=<error_type>`。单错 5 条、双错 7 条路径含超过 128 个 token 的提示，两模型评测上限同设为实测最长的 133，所有输入完整保留。三个参评检查点均重现各自训练时记录的 100 条完整开发路径预测。

按开发集预定规则，原数据选中第 1 轮，共享数据选中第 3 轮。另列原数据第 3 轮，以对照此前按第 3 轮报告的历史结果。每个测试集里 200 条均为无效路径，因此整路 V 主要看首错类型和位置。合法路径误拒率需用含有效路径的数据计算。G 在同一批地图上只看起始状态及后续自行走到的状态，贪心生成时按真实位置到达目标停止。[测试分支](https://github.com/Xiaofeipao/NLP-Project/tree/123bf5a1db4da2a676d4a83cf80974a888cd11db/data/random_errors_20261123) · [新测试集审计](inputs/test_audit.json)

## 整条路径

各列单位均为条，每行分母均为 200。

| 测试集 | 训练数据、轮次 | V 检出无效 | V 首错类型正确 | V 首错位置正确 | V 类型与位置均正确 | G 首步最短 | G 完整到达 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单错 | 原数据第 1 轮，开发集选中 | 119 | 84 | 92 | 83 | 69 | 10 |
| 单错 | 原数据第 3 轮 | 189 | 101 | 79 | 76 | 147 | 67 |
| 单错 | 共享数据第 3 轮，开发集选中 | 199 | 104 | 64 | 63 | 155 | 115 |
| 双错 | 原数据第 1 轮，开发集选中 | 152 | 55 | 62 | 51 | 98 | 16 |
| 双错 | 原数据第 3 轮 | 193 | 125 | 104 | 95 | 119 | 43 |
| 双错 | 共享数据第 3 轮，开发集选中 | 200 | 138 | 98 | 95 | 155 | 113 |

两套开发集选中模型的双错首错联合命中相差 44 条。同取第 3 轮，双错首错联合命中相等。模型能检出路径无效，与能否指出真正的首错，是两个不同的评分项目。新模型在单错集检出 199 条，但有 129 条把错误报得过早，其中 124 条提前报的是撞墙。[开发集选中模型逐条对照](paired_summary.json) · [固定第 3 轮对照](fixed_epoch_comparison.json)

## 按真实首错拆分

后三列均要求错误类型和步号同时正确。双错中的提前结束只在 EOS 出现，67 条含提前结束的路径中它都排在第二处。

| 测试集 | 真实首错 | 样例数 | 原第 1 轮 | 原第 3 轮 | 新第 3 轮 |
| --- | --- | ---: | ---: | ---: | ---: |
| 单错 | 撞墙 | 50 | 0 | 2 | 7 |
| 单错 | 越界 | 50 | 0 | 26 | 20 |
| 单错 | 非法动作词 | 50 | 33 | 22 | 22 |
| 单错 | 提前结束 | 50 | 50 | 26 | 14 |
| 双错 | 撞墙 | 68 | 0 | 8 | 22 |
| 双错 | 越界 | 65 | 0 | 46 | 38 |
| 双错 | 非法动作词 | 67 | 51 | 41 | 35 |
| 双错 | 提前结束 | 0 | — | — | — |

双错同轮次比较，撞墙多命中 14 条，越界少 8 条，非法动作词少 6 条。单错的 50 条真实撞墙路径中，新模型有 47 条报出了撞墙，但只有 7 条也报对了步号。36 条报对类型却报早了，4 条报对类型却报晚了。[新模型单错逐条输出](new/single_predictions.jsonl) · [新模型双错逐条输出](new/double_predictions.jsonl)

## 同一状态换方向

在这 400 条路径各自的起点固定地图和目标，四个方向分别送入同一 V 模板。两方向对照要求同一状态下两个结果都判断正确，分母是实际存在的方向对。

| 判断项目 | 总数 | 原第 1 轮 | 原第 3 轮 | 新第 3 轮 |
| --- | ---: | ---: | ---: | ---: |
| 撞墙与合法未到同时正确 | 560 | 1 | 47 | 111 |
| 越界与合法未到同时正确 | 571 | 0 | 223 | 249 |
| 到达与合法未到同时正确 | 101 | 100 | 65 | 44 |
| 单个合法未到方向命中 | 990 | 989 | 633 | 454 |
| 单个撞墙方向命中 | 285 | 1 | 42 | 223 |
| 同一状态四方向全部正确 | 400 | 53 | 77 | 74 |

新模型学到了更多真实撞墙，但辨认合法未到的能力较弱。这个取舍在同状态方向对照和整路提前报错中均能观察到。[新模型四方向结果](new/matched_starts.json) · [原模型第 1 轮](original/matched_starts.json) · [原模型第 3 轮](original_epoch3/matched_starts.json)


## 获取并加载共享状态权重

在本分支的仓库根目录运行以下命令。

```bash
git lfs pull --include="artifacts/flan_swap64_ab/models/shared4000-epoch3/model-*.safetensors"
```

```python
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

checkpoint = Path("artifacts/flan_swap64_ab/models/shared4000-epoch3")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True).eval()
labels = model.config.ppnl_verifier_labels
first_ids = model.config.ppnl_verifier_token_ids
assert [tokenizer(label, add_special_tokens=False).input_ids[0] for label in labels] == first_ids
```

单步 V 比较上述五个首 token 的分数。整路评测逐步检查候选，动作无法合法落到目标时屏蔽 `valid`，到达目标时仍由模型在五类中选择。首错扫描输出 `valid` 或 `invalid | step=<index> | reason=<error_type>`。[评估设置](protocol.json) · [训练配置](new_model/preflight.json)

发布时使用 Transformers 5.12.1 和 PyTorch 2.12.1 在 CPU 上离线加载此检查点，以同一状态完成一次 G 和四个 V 的前向计算。表中的整路分数取自 GPU 上的 BF16 评测。

Transformers 会根据索引自动加载全部分片。需要核验台式机保存的原始单文件时，可在仓库根目录运行下面的命令。加上 `--output /path/to/model.safetensors` 可以另存一份与训练机逐字节一致的文件。[校验与还原脚本](restore_checkpoint.py)

```bash
python artifacts/flan_swap64_ab/shared4000/restore_checkpoint.py
```
