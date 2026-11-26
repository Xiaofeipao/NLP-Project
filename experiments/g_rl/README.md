# Hope：G 的四动作奖励优化

本分支在 Hope 的共享 G/C/V 模型上增加 G 奖励目标，收录已完成的监督对照、精确期望奖励与采样 RLOO 实验。所有共享参数继续训练，C/V 保留原监督目标，没有冻结 C/V。

## 方法与实现

训练入口：[`run_hope_joint_gcv.py`](run_hope_joint_gcv.py)；奖励与数据核验：[`hope_joint_data.py`](hope_joint_data.py)。调用原 `experiments.part1_baseline` 的 `Network.loss`、`Network.logits` 和 `protocol.schedule`，未修改原训练器、模型、提示词或推理策略。

每个训练状态利用已有四方向 C/V 标注，独立核对落点、撞墙和越界，并用 BFS 核对全部并列最短方向：最短方向 +1、合法但非最短方向 -1、撞墙 -5、越界 -5。当前实现是固定的离散规则奖励，没有按首错位置或墙面距离连续加权；也没有额外 hidden-state 约束损失。

令四方向完整动作串的对数概率得分为 s，π=softmax(s)。总损失保留原监督损失，再加系数 0.2 的 G 奖励项，并沿用 G 样本权重与原归一化：

- `supervised`：额外项系数为 0。
- `expected`：额外项为 −Σ π(a|state)r(state,a)，四动作枚举后直接反向传播。
- `rloo`：每个状态按当前策略有放回采样 8 次；每次奖励减其余 7 次平均奖励，额外项为 −mean(stopgrad(advantage) × logπ(sample))。

三组均执行同样的额外四方向前向，通过 `fork_rng` 隔离额外随机数。梯度更新仍作用于全部共享参数，因此 V 仍可能因共享训练出现退化。

## 已完成实验的配置与结果

原实验代码来源 `Hope@9ebecebe54e94e3bd474a5522c23df86ad66852d`；数据来源 `hope_dataset@d88f85b872560f9313ac4319cdd5fa86151c0780`，已与修订后发布数据逐题核对一致。分支基于最新 Hope，仅新增本实验目录和结果；当前实验未把后来 V 的 few-shot 更新用于训练或评测。

三组从同一原始 Flan-T5-base 初始化，4,000 状态、36,800 视图，seed42，AdamW lr1e-4，微批 **16**、有效32状态、375次更新；C遍历一次，G/V各两次。125/250/375 更新分别记录开发结果，最终固定375检查点。没有使用测试分数选择奖励或检查点。

| 组别 | G 开发三轮 | 最终 C/V | 单错端到端成功 | 单错候选修复成功 | 双错端到端成功 |
|---|---|---|---|---|---|
| 监督对照 | 47% / 63% / 69% | 100% / 100% | 74.5% | 84% | 74.5% |
| 期望奖励 | 53% / 67% / 75% | 100% / 100% | 84.5% | 88% | 79.5% |
| RLOO | 52% / 67% / 73% | 100% / 99% | 83% | 83% | 84% |

G开发分母100；单错测试与双错验证各200。端到端从空路径生成再用模型验证修复；候选修复从给定错误路径开始，二者分开统计。最优率、逐类指标与完整逐题轨迹见[实验产物](../../artifacts/g_rl/20261126/)。

期望奖励相对同配置监督组 G 新增正确6题且无新增错误；RLOO新增正确5题、新增错误1题。单错整行验证两组均从94.5%降至93.5%，新增退化归因见 `new_verification_regressions.json`。不能声称全部指标都提高或规则已经严格内化。这是单种子试验，尚未验证跨种子稳定性。

**固定研究基线仍为组员截图与对应发布权重，G=71%、单错端到端成功79.5%、最优79%。** 本轮微批16监督对照69%仅用于本轮 RL 消融。与截图的完整对比见[固定基线对照](../../artifacts/g_rl/fixed_baseline_comparison/)。后来原版微批4复现实验不包含奖励，不能冒充本批RL实验对照。

## 运行

按 Hope Part1 的环境安装依赖，并下载固定原始 Flan-T5-base（不是 Hope 微调权重），目录内必须包含 `upstream_revision.txt`。从仓库根目录运行，路径替换为实际目录：

```bash
python scripts/fetch_official.py
python experiments/g_rl/run_hope_joint_gcv.py --repo . --dataset data/random_errors_20261123 --base /path/to/original/flan_t5_base --out outputs/g_rl/run_001 --microbatch 16 --data-only
python experiments/g_rl/run_hope_joint_gcv.py --repo . --dataset data/random_errors_20261123 --base /path/to/original/flan_t5_base --out outputs/g_rl/run_001 --microbatch 16
python experiments/g_rl/analyze_hope_joint_gcv.py --out outputs/g_rl/run_001
```

`--data-only` 不加载模型，核验4,000状态及16,000个方向。默认主命令并发训练三组，原运行设备为H100 80GB；其他显存配置可按 `--arms supervised`、`--arms expected`、`--arms rloo` 分别运行同一目录，三组完成后不带 `--arms` 再执行主命令汇总。未完成的训练目录不自动续训，应另选输出目录。

训练只访问训练状态奖励。100条路径开发题用于过程评估；双错验证、单错测试在训练完成后由原 Hope 评测器评估。`CODE_COMMIT` 记录原实验来源，训练脚本SHA和最终权重SHA见各组回执。

## 产物说明

本提交包含代码、配置回执、三轮开发预测、9组路径评测预测/评分、数据来源审计、固定基线对比和模型权重哈希。执行过的两个训练源文件保持原字节；分析脚本仅将旧机器绝对路径改为 `--out` 参数。模型权重和优化器未纳入本提交，仍保留于原远端实验目录；按本说明可重新训练。
