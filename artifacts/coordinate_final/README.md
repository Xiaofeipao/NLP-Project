# 坐标版 Flan-T5 单动作验证

这次训练用 4,000 个共享状态，含 4,000 条 G 和 16,400 条 V。G 预测下一步最短方向。V 用动作前坐标、目标、墙和候选动作预测一个标签。三个已保存的检查点都包含 Flan-T5-base 模型配置和分词器。

## 检查点与训练记录

| 轮次 | 检查点 | 开发集 V 正确数 | 开发集 V 宏 F1 | 训练记录 |
| --- | --- | ---: | ---: | --- |
| 1 | [权重及分词器](models/checkpoint-epoch-1/) | 57/100 | 0.5575 | [训练与设置](runs/four_field_epochs_1_2/training/train_result.json) |
| 2 | [权重及分词器](models/checkpoint-epoch-2/) | 79/100 | 0.7657 | [测试与混淆矩阵](runs/four_field_epochs_1_2/metrics.json) |
| 3 | [权重及分词器](models/checkpoint-epoch-3/) | 87/100 | 0.8660 | [训练记录](runs/four_field_epoch_3/training/train_result.json) · [测试与混淆矩阵](runs/four_field_epoch_3/metrics.json) |

前两轮从原始 Flan 权重开始，连续更新 250 次。第 3 轮从第 2 轮权重接着训练 125 次，重新初始化了 AdamW。三个检查点的校验值见 [模型清单](models/manifest.json)。

模型权重由 Git LFS 管理。获取该目录的完整权重后，可以用 Transformers 从各检查点目录分别加载模型和分词器。

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --branch FINAL --single-branch https://github.com/Xiaofeipao/NLP-Project.git
cd NLP-Project
git lfs pull --include="artifacts/coordinate_final/models/**"
```

## 同一检查点下的提示对照

两组固定第 3 轮坐标版权重，使用同一批单错、双错动作。每个测试集各 200 条路径，拆为 1,198 和 1,365 个候选动作。台式机 RTX 3050 使用 BF16、批次大小 4。两组均从同一份分数得到模型原判和开启终点检查后的结果。

| V 提示 | 测试集 | 原判准确率 | 原判宏 F1 | 加终点检查准确率 | 加终点检查宏 F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| [B · 2-shot](prompts/b_worked_2_shot.txt) | 单错 | 69.95% | **0.6323** | 80.22% | **0.7210** |
| [五示例](prompts/worked_5_shot.txt) | 单错 | **78.71%** | 0.5990 | **84.56%** | 0.6500 |
| B · 2-shot | 双错 | 67.99% | **0.6381** | 75.75% | **0.7249** |
| 五示例 | 双错 | **72.60%** | 0.5730 | **78.10%** | 0.6369 |

五示例版提高总体准确率，同时降低了真正撞墙和越界的召回率。B 的 2-shot 在两套测试上的五类宏 F1 均更高。[完整混淆矩阵与逐类指标](runs/two_vs_five_shots_gpu/metrics.json) · [逐动作预测](runs/two_vs_five_shots_gpu/predictions/)

更早的对照比较了原四行输入、A 的直接标签示例和 B 的落点推导示例。[首轮提示对比](runs/initial_prompt_comparison/metrics.json) · [逐动作预测](runs/initial_prompt_comparison/predictions/)

## 数据与评分

原始数据见 [训练集和开发集](../../data/step_g_shared_4000/) 以及 [单错误集和双错误集](../../data/random_errors_20261123/)。各实验目录的 `inputs-and-code.zip` 保存实际用到的输入、源码和校验清单。`results.zip` 是完成后回收的预测和训练记录。

候选路径从起点开始逐步更新坐标。动作合法就更新当前位置，撞墙、越界或动作词非法时留在原格。所有候选动作都逐步评分。终点检查用 `Current + delta(Action)` 计算落点，仅在该动作能合法到达目标时保留 `valid`。测试不调用执行器。[坐标计算源码](runs/two_vs_five_shots_gpu/source/workspace/step_data_v3/coordinate_paths.py) · [评估源码](runs/two_vs_five_shots_gpu/source/workspace/step_data_v3/evaluate_coordinates.py) · [测试坐标核对](coordinate_audit.json)

## B 提示重训

从原始 Flan 权重发起三轮计划，随后按指示暂停。第 1 轮最后记录为 60/125 次更新，当时还未生成检查点。[训练包与暂停记录](runs/two_shot_retrain_paused/)

The three Git LFS checkpoint uploads are in progress. The experiment files and test results are available in this commit.
