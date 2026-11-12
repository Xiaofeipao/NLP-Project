# Poster 与现有仓库导览

本页以课程 poster *Evaluating Language Models for Natural-Language Grid Path Planning*（Leyi Sheng、Guorun Wang、Xiaofeng Li）为阅读框架，说明现有代码、数据和实验产物能支持哪些陈述。Poster 是用户提供的独立 PDF，**没有作为仓库文件发布**；下文引用其图表时均标作“poster 结果”。仓库报告采用另一套实验协议，不能视为对 poster 数字的复现。

## 研究问题与任务

Poster 的核心问题是：模型输出的路径能否在网格中真正执行，以及学到的是可迁移的规划过程还是特定分布的动作模式。输入是自然语言网格描述、起点、终点及障碍物；输出是由 `up`、`down`、`left`、`right` 组成的动作序列。确定性执行器逐步检查越界和碰撞，BFS 提供最短路参照。

仓库对应实现：[网格与执行器](../ppnl/core.py)、[数据构造](../ppnl/data.py)、[指标与评测](../ppnl/evaluation.py)。现有指标协议见 [实验协议](experiment_protocol.md)。可行不等于成功：合法地停在目标之外仍是失败；成功也不一定最优。

## Poster 方法与仓库对应关系

| Poster 中的方法或材料 | Poster 设置 | 仓库现状与入口 |
| --- | --- | --- |
| T5-small、T5-base、BART-base 微调 | 6×6 单目标最短路径，16k 训练、2k 开发；生成整条动作序列 | 仓库有同名架构的[全量多任务基线](../methods/baseline/README.md)，但使用 4,000 条生成／验证／修复训练记录；不是这组单任务 checkpoint。 |
| GPT-5-mini Zero-shot / Few-shot / CoT / ReAct | 四种无参数更新提示策略，ReAct 接收位置／碰撞反馈 | 当前仓库没有这四组 GPT-5-mini 提示实验的实现、逐条输出或评测报告。`ppnl/prompts.py` 是本地模型的输入格式模板，不是 poster 的 CoT/ReAct 实验。 |
| 执行器评测 | Success、Feasibility、Optimality | [指标定义](experiment_protocol.md)与研究问题相通；比较数值前仍须核对执行器、动作解析、数据与样本量。 |
| 后续系统实验 | Poster 未报告 GVR、LoRA、GRPO、逐步策略、前缀修复 | 这些是仓库[方法目录](../methods/README.md)中的后续方向，不能反向归为 poster 的模型成绩。 |

Poster 使用 seen 6×6、unseen 5×5、unseen 7×7 和更多障碍物条件；目前公开的[派生数据](../data/README.md)来自官方 6×6 单目标网格，按障碍物布局重划分，ID/OOD 各 200 条，并配对使用同一批环境来比较候选错误条件。仓库的 OOD 标签不表示 poster 的 7×7 测试。

## Poster 报告了什么

- Figure 3 报告 T5-base 在 seen 6×6 上的成功率为 **95.8%**，并指出更大的 unseen 7×7 网格上性能明显下降。这里仅转述 poster；仓库没有与该设置匹配的权重、样本清单和预测轨迹。
- Table 1 是 GPT-5-mini 的四种提示策略，每个策略／条件取 **100 个样本**。在 unseen 7×7 上，Zero-shot 成功率 **74%**、Few-shot **72%**、CoT **48%**、ReAct **21%**；表中所有策略／条件的可行率为 **100%**。这支持“动作合法不保证到达目标”的结论，但不构成仓库代码的复现结果。
- Poster Figure 1 明示其框架图中的数值为示意值；应以 Figure 3 和 Table 1 的标注为准，不能把示意图作为独立测量。

## 仓库已发布的结果

当前 [三模型 GPU 实验记录](gpu_campaign_20261103.md) 使用 `google-t5/t5-small`、`google-t5/t5-base`、`facebook/bart-base`，在多任务数据和不同推理流程上评测。全量多任务基线的 ID 整路径直接生成成功数依次为 **2/200、3/200、0/200**；单独训练的逐步策略加访问次数约束均为 **200/200**，最优数分别为 **172/200、170/200、165/200**。这些结果来自不同任务和运行约束，不能与 poster 的 95.8% 做模型优劣排序；逐步策略的收益也不能算作原始整路径模型的能力。

完整配置、负结果和开发集选模规则在[实验记录](gpu_campaign_20261103.md)，逐项指标在[结果 JSON](gpu_campaign_results.json)，可下载权重与轨迹在[产物目录](../artifacts/campaign_20261103/README.md)。该轮只有一个随机种子，ID/OOD 测试已重复用于探索性分析；现有[冻结说明](evaluation_freeze.md)明确它们不是新的盲测。

## 复现与引用边界

- 想复核**仓库**数值：先读[数据说明](../data/README.md)和[实验协议](experiment_protocol.md)，再按[环境说明](environment.md)安装依赖；从[方法目录](../methods/README.md)选择入口，按[产物说明](../artifacts/campaign_20261103/README.md)取得对应权重与报告。
- 想复核**poster**数值：还需要其 16k/2k 数据划分、5×5/7×7 样本清单、三个单任务 checkpoint、GPT-5-mini 提示词／模型版本、调用配置与逐条输出。当前仓库未提供这些材料，因此只能将 poster 的表格作为独立展示结果引用。
- 课程展示可按“问题 → 方法 → 执行器指标 → 结果 → 泛化限制”叙述；任何图表都应标明来自 poster 还是当前仓库，并附样本量、split 和推理预算。
