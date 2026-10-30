# GVR 原始实现对照与训练修复

日期：2026-10-30。基于 `8e05270`。范围为 S2、Reference-assisted；采用 Pattern 复用，以官方修复原理编写本项目实现。

## 需求与验收

核对原 PPNL、锁定版本的 Transformers/PEFT 实现，修复可复现的训练错误。验收要求：累积梯度与完整批次等价；真实 T5 标签右移正确；真实 LoRA 更新不改变底座；命名适配器保存、加载和切换保持输出一致。

保留完整路径输出、冻结数据、模型版本、学习率、损失定义、样本配额与预算。旧权重和指标不重写。此次没有运行新的正式训练，不能宣称端到端成功率已经改善。

## 对照结果与文件边界

| 能力 | 代码核对结果 | 处理 |
|---|---|---|
| 目标分词、padding、decoder 右移 | 使用官方 DataCollatorForSeq2Seq；真实 T5 测试通过 | 保留 |
| 梯度累积 | 将不同有效 token 数的批次均值等权平均，与完整批次梯度不一致 | 新增 `ppnl/optimization.py`，接入普通训练和 GVR 训练 |
| 自定义 Verify 损失 | 按样本平均，归一化单位应是样本数 | 按窗口实际样本数加权，不使用 token 权重 |
| 冻结 G、独立 LoRA、保存切换 | 真实 T5/PEFT 测试通过 | 保留加载实现，补充集成测试 |
| pilot 兼容性 | 梯度计算改变会使旧 pilot 与新正式训练不一致 | 在报告/训练签名中加入 `loss_units_v1`；拒绝沿用旧 pilot |

实施顺序：窗口归一化函数 → 两条训练入口 → pilot 签名 → 数学回归和真实模型测试。涉及 `ppnl/optimization.py`、`ppnl/modeling.py`、`ppnl/gvr_training.py` 及相应测试。

## 已确认的错误

设两个小批次的平均损失为 L1、L2，有效标签 token 数为 n1、n2。旧代码反传 `(L1 + L2) / 2`；token 平均目标对应的完整批次损失应为 `(n1 * L1 + n2 * L2) / (n1 + n2)`。

修复只在 CPU 缓存一个累积窗口，提前统计单位总数，再逐个批次送到 GPU 反传。最后一个不完整窗口同样按实际数量归一化。自定义 sequence/status 损失和逐步动作目标按样本数归一化。

使用已发布分词器、冻结训练集、seed 42、batch 8 检查第一轮：G 的 50 个窗口中 47 个 token 数不等，R 的 75 个窗口中 69 个不等；旧版 token-mean V 的 125 个窗口中 113 个不等。单个小批次被旧算法过度加权的最大倍数分别约为 1.2143、1.2703、2.4。新候选 V 已采用按样本平均的目标，其当前完整批次不受该 token 权重问题影响。

这证明错误会影响当前数据上的训练梯度，不证明它足以解释重复动作或 0% 成功率。

## 参考与来源

访问日期均为 2026-10-30。

- [PPNL T5 原始入口](https://github.com/MohamedAghzal/llms-as-path-planners/blob/5fd0376fc50ae9860bc986085165ebac5675fd44/ppnl-spatial-temporal-reasoning/train/T5/main.py) 与 [实际脚本使用的配置](https://github.com/MohamedAghzal/llms-as-path-planners/blob/5fd0376fc50ae9860bc986085165ebac5675fd44/ppnl-spatial-temporal-reasoning/train/T5/configs/train_deepspeed.json)：固定 revision；配置使用 t5-base、Adafactor、3072 轮上限。未在该 revision 文件树中发现许可证，因此仅研究结构，没有复制源码。该配置与本项目模型和课程预算不同。
- [Transformers 4.57.6 翻译示例](https://github.com/huggingface/transformers/blob/v4.57.6/examples/pytorch/translation/run_translation_no_trainer.py)：Apache-2.0；用于核对目标分词、collator 和生成输入。旧示例中的按批次均值累积不是本次修复依据。
- [Hugging Face：Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)：采用按窗口非 padding token 总数归一化的数学原理；没有复制文章中的因果语言模型 shift 代码，T5 的 decoder 右移继续由官方组件负责。
- [PEFT 0.18.1 Seq2Seq LoRA 示例](https://github.com/huggingface/peft/blob/v0.18.1/examples/conditional_generation/peft_lora_seq2seq.ipynb)：Apache-2.0；用于核对 Seq2Seq LoRA、冻结参数、保存和推理流程。新增代码为本项目独立实现，无外部源码拷贝。

## 验证与接续

在独立环境使用 Transformers 4.57.6、PEFT 0.18.1、PyTorch 2.12.1 CPU，完整运行 **77 项测试，全部通过，无跳过**。其中真实模型测试使用本地随机初始化的小 T5 和本地分词器，不下载预训练模型，不消耗 GPU 预算；它验证实现契约，不评估规划效果。

在已安装项目依赖的环境、仓库根目录运行：

```powershell
python -m unittest tests.test_accumulation tests.test_seq2seq_integration tests.test_gvr_objectives -q
python -m unittest discover -s tests -q
```

新测试覆盖：变长 token 梯度、旧算法反例、按样本损失的末尾短批次、窗口尾部、真实 T5 标签/EOS/decoder 移位、真实 T5 完整批次梯度、LoRA 参数更新与底座不变、两套命名适配器重载和禁用恢复。

下次正式训练需在独立输出目录重新做匹配的 pilot，并遵守既有数据和 GPU 账本要求。旧实验可用对应旧提交复现；回滚本次修复不需要迁移数据或权重。
