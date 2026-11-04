# 三模型 GPU 实验：预先确定的方案与运行记录

**完成日期：2026-11-04。本轮训练与预定评测已全部完成，99/99 份报告齐全。** 机器可读完整结果见 [gpu_campaign_results.json](gpu_campaign_results.json)。根据后续备份要求，33 个有效权重与运行证据纳入 [Git LFS 发布目录](../artifacts/campaign_20261103/README.md)，本地原始副本继续保留；弱方法保留为负向消融，历史默认配置未被覆盖。

## 最终结果与处置

以下成功数的分母均为 200 个 ID 测试环境，最优率也以全部 200 个环境为分母。完整开发集、ID/OOD 候选任务指标在 JSON 报告中；不能把独立修复任务的成功率当作端到端成功率。

### 逐步规划：主要有效方向

- T5-small：裸策略 169/200；禁止重复访问 195/200；仅合法动作的 Harness 170/200；加入最少访问次数约束后 200/200，最短路径 172/200（86%）。
- T5-base：裸策略 163/200；禁止重复访问 192/200；仅合法动作的 Harness 163/200；加入最少访问次数约束后 200/200，最短路径 170/200（85%）。
- BART-base：裸策略 164/200；禁止重复访问 190/200；仅合法动作的 Harness 169/200；加入最少访问次数约束后 200/200，最短路径 165/200（82.5%）。

这些非搜索策略统一最多 36 步。它们是单独训练的动作策略，不是 GRPO 或 GVR LoRA 的成绩。纯 A* 对照也达到 200/200 且全部最短，因此带搜索的高分不作为语言模型独立规划能力的证据。本轮与历史 16 次调用等不同预算的结果不作直接因果比较。

开发集预先选择的方法分别为：T5-small 的 legal_no_revisit、T5-base 与 BART 的 harness_visit。T5-small 开发集两项成功率和最优率并列，按既定规则选较简单的方法，其测试成功数是 195/200。上面的 harness_visit 200/200 是预定消融的观察结果，不用测试成绩倒改开发选择，也不自动升级全局默认配置。

### 整路径基线与 GVR：保留正式对照和负结果

以下按 direct / self / executor_one / executor_three 顺序列成功数。

- 原始多任务全量微调：T5-small 为 2 / 2 / 4 / 4；T5-base 为 3 / 2 / 6 / 7；BART-base 为 0 / 0 / 0 / 0。
- vanilla GVR：T5-small 为 0 / 0 / 0 / 4；T5-base 为 1 / 1 / 1 / 1；BART-base 为 0 / 0 / 0 / 0。
- structured GVR：T5-small 为 1 / 1 / 1 / 3；T5-base 为 0 / 0 / 0 / 0；BART-base 为 0 / 1 / 1 / 1。
- 验证辅助损失 GVR：三个模型的端到端成功数均与对应 vanilla 相同，没有构成整体升级。开发诊断中，BART 验证器从全部判非法变成全部判合法，macro-F1 仍为 0.333；该配置保留为退化消融。

T5-small 原始基线的自验证把全部生成路径判为合法；执行器多轮修复中有 196/200 个案例因重复候选停止。这解释了“验证模块能够输出合法格式”为什么不等于“能够可靠自我纠错”。原始基线与失败配置均保留，不删掉低分来重写比较结论。

### RLVR：有小幅变化，尚无可靠规划提升

以同一 G 底座、输入格式、48-token 解码上限比较无 adapter / REINFORCE / GRPO：

- T5-small：0 / 0 / 4 个成功。
- T5-base：1 / 1 / 0 个成功。
- BART-base：0 / 1 / 1 个成功。

T5-small 与 BART 的 GRPO 开发成功率从 0/40 到 1/40，属于可继续研究的小样本候选，不足以取代逐步策略。T5-base 的 GRPO 只有奖励改善，没有开发成功数提升，保留为负向消融。REINFORCE 的三个配置也没有开发成功率提升，不作为默认升级。测试中的偶然提高不改变这个开发选择规则。

### 前缀修复：成功率高，但保留绕路会损害最优率

九种生成来源（每个模型的原始全量、vanilla GVR、structured GVR）在 prefix_visit 下全部达到 200/200；但其最优率分别为：T5-small 33.5% / 30.5% / 31%，T5-base 33.5% / 35.5% / 27.5%，BART 37% / 34.5% / 37%。

restart_visit 丢弃前缀，从起点调用同一逐步策略，同样全部成功，最优率分别为 86%、85%、82.5%。因此高成功率主要来自后续策略与访问约束，不能归因于原始路径生成。纯启发式 prefix_visit_heuristic 也全部成功，应作为显式符号控制保留。当前不把前缀保留作为默认升级。

### 覆盖范围和证据边界

共完成 33 个评测配置 × 开发/ID/OOD 三个划分 = 99 份报告，包括 3 个全量 Harness、9 个 GVR、9 个 RL/无 RL 对照、3 个逐步策略和 9 个前缀来源。训练、pilot、重载检查另有独立报告。CPU 回归测试 141 项通过；最终共享 GPU 账本累计约 6.669 小时，包含历史运行，不等于本轮新增训练时间。

单随机种子、重复使用的探索性测试、多个候选分别训练，不属于新的盲测。ID/OOD 网格配对、候选损坏分布不同；纯生成、逐步策略和生成路径的修复成绩相同不证明新地图分布泛化。所有负结果均保留；推荐研究重点是逐步动作训练和运行约束，GRPO、结构化输入、辅助损失和保留前缀继续作为有明确边界的消融。

本轮使用用户指定的 `google-t5/t5-small`、`google-t5/t5-base`、`facebook/bart-base`。它们不是此前的 Flan-T5-small。固定版本分别为 `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`、`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`、`aadd2ab0ae0c8268c7c9693540e9904811f36177`。

所有新文件写入逻辑目录 `outputs/campaign_20261103/`；本轮曾因输出盘容量不足导致 T5-base 保存失败，失败日志保留，损坏权重弃用，迁移输出存储后从头重训。没有删除历史数据、已完成 checkpoint 或原始成绩。只清理了 Python 缓存及代码审查的临时副本。运行前 pull 确认没有远端新提交。

## 固定矩阵

- 三个原始多任务全量微调基线：相同 4,000 条训练记录、2 epochs、batch 4、累积 4、学习率 0.0001；各做 16 条样本的独立 pilot，不把 pilot 参数作为正式初始化。
- 每个模型各做 vanilla / structured 的 GVR 对照：G 800 条、V 2,000 条、R 1,200 条；G 全量微调，V/R 独立 LoRA r=8、alpha=16，adapter 学习率 0.001，均按开发任务指标选 epoch。本轮三架构统一 token-mean loss，没有混用此前 Flan 的首状态 token 辅助损失。
- 每个模型单独训练同架构逐步策略，使用既有 `data/policy_v3` 的固定记录，比较裸策略、动作约束、Harness 访问约束及搜索／纯 A*。BART 的策略动作标签移除 BOS 以对齐一步 decoder 评分；验证文本标签保留 BOS。
- 逐步策略的 raw、legal_no_revisit、harness_legal、harness_visit 统一最多 36 步。legal_no_revisit 禁止进入访问过的位置；harness_visit 则优先选择访问次数最少的合法位置，允许必要的回访。搜索对照另按 36 次节点展开计量，不声称和神经调用成本相等。
- 每个模型基于开发集选定的 G 输入格式，分别做独立 REINFORCE LoRA 和 GRPO LoRA；每项 64 组、每组 4 条、最长 48 token。REINFORCE 每组 1 次更新，GRPO 每组 2 次；分别是 64 与 128 optimizer updates，不声称两者计算成本相同。
- RL 额外评测同一底座、同一输入格式、同一 48-token 上限的无 adapter 对照，避免把解码长度差异误认为 RL 效果。
- 每个模型增加验证损失消融：保持开发集选定格式的 G/R 不变，只重新训练 V，比较 token-mean 与 sequence-mean 加 verdict 辅助损失。BART 辅助项定位到 BOS 后的判别 token。这同时改变归一化和辅助项，不能把差异全部归因于辅助项。
- 三个全量模型和六个 GVR 输入配置分别运行完整 Harness：direct、self、executor_one、executor_three。前缀修复使用同架构逐步策略，再做 raw/legal/visit/heuristic/restart 五种对照。

各训练候选独立使用原始训练/开发划分；本轮是多方案探索，不把跨候选累计训练量包装成“一次不超过三遍的课程主实验”。共同 20 GPU 小时账本继续累计失败与成功运行，所有测试仍标为重复使用的探索性测试。

## GRPO 实现边界

之前只有 REINFORCE；本轮才新增 outcome-GRPO 的序列到序列适配。依据 [DeepSeekMath](https://arxiv.org/abs/2402.03300)，组内奖励按均值和总体标准差归一化，采样时固定旧策略 log-prob；同一批完成串做两次更新，使用 token 概率比、0.2 clipping 和冻结底座的 KL 项。保留原论文的逐序列 token 均值归一化，不等同于后续 DAPO/Dr.GRPO。没有 critic 或人工偏好数据，奖励仍是执行器奖励。

采样和重新计算似然都关闭 dropout；temperature=1、top-k=0、top-p=1，关闭 BART 强制 BOS/EOS 改写，EOS 本身计入损失，EOS 后内容和 padding 被屏蔽。LoRA 根据架构匹配 T5 的 q/v 或 BART 的 q_proj/v_proj。新增测试验证 clipping 两种优势符号、常奖励组、EOS mask、真实 BART 冻结边界和动作标签对齐。

## 选择与回退规则

开发集选择 G 的输入格式：生成成功率、最优率，完全并列时保留 vanilla。RL epoch/step 仍先看开发成功率，再看奖励，但只有成功率改善才作为部署升级证据；奖励改善而成功率不变只作为负结果／消融。无效方法保留记录，默认历史最佳配置不被自动替换。测试成绩不用于追加训练或删掉失败实验。

原基线 CLI 的历史开发口径是验证环境去重，不能和 GVR 的 40 个 generation 开发样本直接比较。本轮完整 Harness 统一采用 40/100/60 开发任务划分，最终测试对齐同序 200 个 ID / 200 个 OOD 环境。

每次测试前冻结当前模型、输入配置、源码、数据、执行器与完成的开发证据；测试输出保存锁哈希。RL 只评估其训练的 generation 任务，不假装 RL adapter 是 V/R 模型。新 checkpoint 保留本地，不自动发布权重。

## 入口与状态

`scripts/run_gpu_campaign.py` 串行执行基线、六套 GVR 和六项 RL train/dev；随后 `scripts/run_campaign_status_ablation.py` 执行三项验证损失消融；`scripts/run_campaign_policies.py` 训练并冻结同架构逐步策略；`scripts/evaluate_gpu_campaign.py` 对全量 Harness、GVR、RL generation 和前缀消融做冻结评测。通用单项评估在 `scripts/evaluate_campaign_model.py`。

实际阶段记录见本地 `training_status.json`、`evaluation_status.json` 及 `logs/`，两个状态均已完成。最终汇总见本文开头及 `gpu_campaign_results.json`。原始失败的磁盘保存日志单独保留，不纳入成功权重。
