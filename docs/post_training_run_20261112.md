# 固定多任务权重的台式机 GPU 复跑（2026-11-12）

本轮按已冻结的 13 个配置完成 dev、ID 和 OOD 全部评估。复用已有 Flan-T5-small 多任务全量微调权重，未重新训练。设计、控制流程和消融边界见 [后处理说明](post_training.md)，课程整体要求见 [Project 6 整理](project6_overview.md)。

最终运行于 **2026-11-12 19:00:29（新加坡时间）**完成，退出码 0。最明确的结论是状态约束带来主要收益，有限回溯进一步提高成功率；模型 V 的精确诊断仍弱，详细错误反馈不是当前约束搜索成功的必要条件。Manhattan 规则对照的成功率和最优率仍然更高。

## 1. 代码整理与验收

- 单网格运行保留原 GVR 状态机，批量实验改为一个配置驱动入口；四份重复批处理退为明确迁移提示，约 379 行重复活动代码移出运行路径。原代码附 [校验归档](legacy_gvr_drivers_23183e2.zip)。
- 全量前缀恢复委托公共 rollout，保留旧 trace 与预算语义；统一成本字段并消除失败调用残留的 telemetry。
- 台式机依赖环境执行 `python -m unittest discover -s tests -q`：176 项通过。覆盖自验证不接收金标、配对缓存、环境身份、防篡改、成本、规则对照、前缀兼容和退休入口。
- 结果完成后另用独立执行/BFS/指标公式复算：429 项主路径指标、1,391 项错误分组指标、V 指标和 ID/OOD 共享记录全部一致；32 份运行源码与数据哈希一致。6,740 个会话、52,645 条事件通过哈希链检查，2,435 个唯一合法 token 路径与官方成功/碰撞语义一致。见 [独立审计](post_training_20261112/audit.json)。
- 首先运行每任务 2 条 dev smoke，随后按原矩阵全量运行。运行中的输出从 HDD 复制至 SSD 并以相同协议恢复；551 个已有会话通过日志完整性检查，中断边界为执行器反馈后的 decide 阶段，没有丢弃神经调用。
- 随后一次 SSH 连接中断影响了原后台进程。恢复时核验了 1,827 个会话，保留一次尚未返回的 `backtrack_none` 修复请求计费；改由独立 WMI 进程完成余下评估。模型、数据和方法矩阵全程未改变。

## 2. 实验身份与数据

GPU：**NVIDIA GeForce RTX 3050（6GB）**；Python 3.12.10、PyTorch 2.11.0+cu128、CUDA runtime 12.8、Transformers 4.57.6。峰值 PyTorch 分配显存 321.1 MiB，不包含桌面与驱动占用。

权重 `model.safetensors` SHA256：`0b7be3b12238b363ba8c1a9304f687c6ae6f9edbecceff0076cebe7c260d62fa`。训练来源、开发集主模型选择保持历史证据；本轮固定 seed 42、普通贪心 128 token、搜索 beam 4 / 36 动作 / 148 评分前缀。

运行协议：`159ce983c1711f2be0c71acb5c7251d3886408add79cd9492d69f7e9b21a8bfd`。全部 13 个配置在第一次推理前固定，本轮没有根据 dev 或 test 成绩筛除方法或调参；测试 receipt 绑定已完成的 dev 报告。测试集此前已使用，因此本轮标记为复跑与探索性消融。

| 划分 | 独立 V | 固定无效候选 R | G→Harness 环境 |
| --- | ---: | ---: | ---: |
| dev | 100 | 60 | 40 |
| test_id | 200 | 100 | 200 |
| test_ood | 200 | 100 | 200 |

ID/OOD 共用同样的 200 个规划环境及初始 G、planning 会话。改变的是预设候选的一次/两次错误注入，因此两者的候选 V/R 可比较，而端到端只是一组 200 环境证据。所有三率的分母均为全部被评路径；最优率不以成功路径为分母。

## 3. 端到端与候选修复

每格为 **success / feasibility / optimality**。direct 在候选栏原样保留无效输入；self 使用模型反馈，executor 使用严格执行器反馈。

| 方法 | 从 G 开始，200 环境 | ID 候选，100 条 | OOD 候选，100 条 |
| --- | --- | --- | --- |
| `direct` | 0% / 0% / 0% | 0% / 58% / 0% | 0% / 30% / 0% |
| `self` | 0% / 0% / 0% | 15% / 57% / 15% | 6% / 45% / 6% |
| `executor_one` | 0% / 0% / 0% | 19% / 46% / 19% | 7% / 45% / 7% |
| `executor_three` | 0% / 0% / 0% | 19% / 31% / 19% | 11% / 24% / 11% |
| `syntax_beam` | 0% / 0% / 0% | 25% / 56% / 25% | 8% / 51% / 8% |
| `constrained` | 92% / 100% / 43% | 98% / 100% / 96% | 94% / 100% / 83% |
| `backtrack` | 96.5% / 100% / 43.5% | 99% / 100% / 96% | 98% / 100% / 84% |
| `constrained_none` | 92.5% / 100% / 42.5% | 99% / 100% / 96% | 94% / 100% / 85% |
| `constrained_generic` | 92.5% / 100% / 42.5% | 98% / 100% / 96% | 97% / 100% / 84% |
| `backtrack_none` | 96.5% / 100% / 43% | 99% / 100% / 96% | 98% / 100% / 86% |
| `backtrack_generic` | 96.5% / 100% / 43% | 99% / 100% / 96% | 99% / 100% / 84% |
| `uniform` | 95.5% / 100% / 37% | 95% / 100% / 35% | 95% / 100% / 35% |
| `manhattan` | 100% / 100% / 99.5% | 100% / 100% / 99% | 100% / 100% / 99% |

这组数据支持以下判断：

1. **主要收益来自状态约束。** B4 的端到端成功率仍是 0%，C4 为 92%；候选修复也从 ID 25% / OOD 8% 提升至 98% / 94%。仅限制输出动作词表和增加束搜索不足以解决问题。
2. **回溯提高了可达成功，但对最短路帮助很小。** C4→backtrack 从 184/200 提升到 193/200，而最优路径仅从 86/200 提升到 87/200。OOD 修复从 94/100 到 98/100；不能把成功率提升写成最优规划能力同步提高。
3. **详细首错文本的贡献有限。** 端到端 backtrack 的 original/none/generic 都为 96.5%。OOD C4 的 generic 为 97%，高于 original 的 94%；backtrack_generic 为 99%，original 为 98%。这些差异只有少量案例，未做多种子统计推断；可以说本轮未显示详细诊断的稳定优势，不能说反馈永远无用。
4. **迭代普通修复收益有限且会破坏可行性。** 三轮普通修复使 OOD 成功率从 7% 到 11%，但 feasibility 从 45% 降至 24%；ID 成功率仍为 19%。从 G 开始仍全部失败。
5. **学习评分没有超过强规则对照。** C4 比 uniform 的 ID/OOD 修复最优率高很多（96%/83% 对 35%/35%），但端到端成功率反而较低（92% 对 95.5%）；Manhattan 同时达到 100% 成功率和 99.5% 端到端最优率。模型贡献应限定为本组条件下相对等权评分的路径质量差异，不能声称优于启发式搜索。

报告组织上宜保留三种基本流程（direct、self、executor），以 C4 为主要改进、backtrack 为进一步改进；B4、反馈文本、uniform/manhattan 作为定位贡献来源的消融和对照。13 个配置不是 13 项独立创新，也未证明每个方法都需要同时调用 G/V/R。

开发集作为整张矩阵的检查结果一并保留：

| 方法 | G40：三率 | R60：三率 |
| --- | --- | --- |
| `direct` | 0% / 2.5% / 0% | 0% / 61.6667% / 0% |
| `self` | 0% / 2.5% / 0% | 11.6667% / 55% / 11.6667% |
| `executor_one` | 0% / 0% / 0% | 16.6667% / 51.6667% / 16.6667% |
| `executor_three` | 0% / 0% / 0% | 18.3333% / 36.6667% / 18.3333% |
| `syntax_beam` | 0% / 0% / 0% | 13.3333% / 58.3333% / 13.3333% |
| `constrained` | 90% / 100% / 35% | 100% / 100% / 95% |
| `backtrack` | 97.5% / 100% / 35% | 100% / 100% / 95% |
| `constrained_none` | 92.5% / 100% / 35% | 98.3333% / 100% / 93.3333% |
| `constrained_generic` | 90% / 100% / 35% | 100% / 100% / 93.3333% |
| `backtrack_none` | 100% / 100% / 35% | 98.3333% / 100% / 93.3333% |
| `backtrack_generic` | 100% / 100% / 35% | 100% / 100% / 93.3333% |
| `uniform` | 92.5% / 100% / 30% | 95% / 100% / 38.3333% |
| `manhattan` | 100% / 100% / 95% | 100% / 100% / 100% |

## 4. V 与实际错误类型

| 划分 | Macro-F1 | 格式准确率 | 首错定位 | 首错类型 | 定位+类型 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev | 0.629073 | 100% | 14% | 50% | 14% |
| test_id | 0.718988 | 100% | 8% | 53% | 7% |
| test_ood | 0.774994 | 100% | 20% | 54% | 11% |

所有测试环境的 G 输出都被 self 的 V 接受，self 端到端实际只增加一次 V、未触发 R，最终仍为 0%。独立候选上的 Macro-F1 约 0.72/0.77，不能据此推断 V 在模型生成分布上同样可靠。对碰撞和越界的类型识别均为 0%，显示精确诊断仍是明显短板；executor 方法的高成绩也不能记为学习式 V 的成绩。

OOD 的 Macro-F1 较高也不等于空间推理泛化更好：无效候选中 `invalid_action_token` 的占比从 ID 的 20% 变为 OOD 的 45%，错误组成不同。分组结果必须与总体分数一起阅读。

定位/类型/联合准确率以所有金标无效候选为分母。下面按实际执行首错分组；同一类型只有 invalid 类，故不使用该子集的二分类 Macro-F1 做整体能力比较。

| 划分 / 首错类型 | n | V 定位 | V 类型 | E1 修复 | C4 修复 | 回溯修复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test_id / `goal_not_reached` | 58 | 12.069% | 65.5172% | 17.2414% | 96.5517% | 98.2759% |
| test_id / `invalid_action_token` | 20 | 5% | 75% | 30% | 100% | 100% |
| test_id / `obstacle_collision` | 14 | 0% | 0% | 7.14286% | 100% | 100% |
| test_id / `out_of_bounds` | 8 | 0% | 0% | 25% | 100% | 100% |
| test_ood / `goal_not_reached` | 30 | 13.3333% | 70% | 0% | 86.6667% | 93.3333% |
| test_ood / `invalid_action_token` | 45 | 28.8889% | 73.3333% | 8.88889% | 97.7778% | 100% |
| test_ood / `obstacle_collision` | 14 | 21.4286% | 0% | 14.2857% | 100% | 100% |
| test_ood / `out_of_bounds` | 11 | 0% | 0% | 9.09091% | 90.9091% | 100% |

## 5. 成本与消融的解释范围

下面是测试 200 环境中共享初始 G **之后**的实际记录；同一行不是 G 的重复采样。rule rows 与 neural rows 分列。普通生成的 decoder rows=0 表示内部计数未暴露。活动秒数含方法内 CPU/GPU 调用，不含全部日志持久化开销。

| 方法 | 神经请求 | 神经评分前缀 | 规则评分前缀 | 活动秒 | ID R 活动秒 | OOD R 活动秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct` | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| `self` | 200 | 0 | 0 | 8.85 | 36.38 | 42.31 |
| `executor_one` | 200 | 0 | 0 | 39.94 | 11.00 | 11.11 |
| `executor_three` | 600 | 0 | 0 | 143.48 | 31.38 | 33.17 |
| `syntax_beam` | 200 | 29000 | 0 | 154.72 | 77.58 | 77.44 |
| `constrained` | 200 | 15172 | 0 | 89.20 | 43.15 | 40.13 |
| `backtrack` | 200 | 29457 | 0 | 204.56 | 102.47 | 101.68 |
| `constrained_none` | 200 | 15137 | 0 | 87.86 | 42.69 | 41.35 |
| `constrained_generic` | 200 | 15195 | 0 | 89.22 | 42.98 | 41.51 |
| `backtrack_none` | 201 | 29457 | 0 | 205.91 | 103.51 | 102.86 |
| `backtrack_generic` | 200 | 29457 | 0 | 206.38 | 104.15 | 103.90 |
| `uniform` | 0 | 0 | 16096 | 0.23 | 0.09 | 0.11 |
| `manhattan` | 0 | 0 | 11084 | 0.10 | 0.12 | 0.08 |

完整 dev+ID+OOD 共 240 个唯一初始 G、6927 次神经请求。共享 G、候选 V、planning 与候选 R 的详细分账见 [结果 JSON](post_training_20261112/results.json)。同环境 ID/OOD 的 planning 成本不能再相加。

6927 次是持久化计费请求数，其中包含 SSH 中断时一次未返回的 R 请求；`backtrack_none` 因而为 201 次而非 200 次。该未完成调用的内部评分行数、token 和活动秒数无法从完成记录还原，所以细粒度计算量不覆盖这一次丢弃工作；整段保守占用时间已计入 GPU 账本。其余比较中，backtrack 原反馈相对 C4 的神经评分前缀约为 **1.94 倍**、活动时间约为 **2.29 倍**（204.56 对 89.20 秒），以此获得 4.5 个百分点的端到端成功率提升。

独立日志审计数得 **6926 次完成响应**，与 6927 次计费请求的差额恰好对应上述中断尝试。没有混入独立 smoke 的请求数。

B4→C4 隔离状态约束，C4→backtrack 隔离回溯；共同 148 行上限并不使实际计算量相等。ordinary→B4 同时改变文本语法与搜索预算，是方法对照。none/generic 只改反馈文本；执行器门控和状态约束仍在。规则对照匹配无回溯 C4，不能把回溯相对规则的收益单独归因于神经评分。

## 6. 逐例轨迹

### 6.1 模型 V 错误接受，约束修复成功：`test_id-15`

坐标从 0 开始，起点 (0, 5)，目标 (1, 0)，BFS 最短长度 6。`S` 为起点、`G` 为目标、`#` 为障碍。

```text
. . . . . S
G . . . . #
. . . . . .
. . . . . .
. . . . . .
. # . . . .
```

原候选：`down left down left left left left`。真实诊断：`invalid | step=1 | reason=obstacle_collision`。

候选构造记录：`[{"kind": "insertion", "original_index": 1, "before": "left", "after": "down"}]`。此例为一次错误注入。

- `self`：`down left down left left left left`；invalid | step=1 | reason=obstacle_collision；停止原因 `feedback_valid`。
  V 输出：`valid`。
- `executor_one`：`down left down left left left left left`；invalid | step=1 | reason=obstacle_collision；停止原因 `repair_limit`。
  R 反馈：`invalid | step=1 | reason=obstacle_collision`；解码状态 `None`。
- `constrained`：`left left down left left left`；valid；停止原因 `feedback_valid`。
  R 反馈：`invalid | step=1 | reason=obstacle_collision`；解码状态 `complete`。
- `backtrack`：`left left down left left left`；valid；停止原因 `feedback_valid`。
  R 反馈：`invalid | step=1 | reason=obstacle_collision`；解码状态 `complete`。

constrained 的逐步执行：

| 步 | 动作 | 从 → 到 | 状态 |
| --- | --- | --- | --- |
| 1 | left | [0, 5] → [0, 4] | 合法 |
| 2 | left | [0, 4] → [0, 3] | 合法 |
| 3 | down | [0, 3] → [1, 3] | 合法 |
| 4 | left | [1, 3] → [1, 2] | 合法 |
| 5 | left | [1, 2] → [1, 1] | 合法 |
| 6 | left | [1, 1] → [1, 0] | 合法 |

### 6.2 双错误候选：普通修复失败与 C4 修复：`test_ood-1`

坐标从 0 开始，起点 (3, 4)，目标 (2, 5)，BFS 最短长度 2。`S` 为起点、`G` 为目标、`#` 为障碍。

```text
. . . . . .
. . . . . .
. # # . . G
. . # . S .
. . . . . .
. . . . . .
```

原候选：`left up jump`。真实诊断：`invalid | step=3 | reason=invalid_action_token`。

候选构造记录：`[{"kind": "insertion", "original_index": 1, "before": "up", "after": "left"}, {"kind": "invalid_token", "original_index": 2, "before": "right", "after": "jump"}]`。两次注入不等于执行时能同时观测两处失败；执行器只报告首错。

- `self`：`up up`；invalid | step=3 | reason=goal_not_reached；停止原因 `repair_limit`。
  V 输出：`invalid | step=1 | reason=invalid_action_token`。
  R 反馈：`invalid | step=1 | reason=invalid_action_token`；解码状态 `None`。
  V 输出：`invalid | step=2 | reason=goal_not_reached`。
- `executor_one`：`up up`；invalid | step=3 | reason=goal_not_reached；停止原因 `repair_limit`。
  R 反馈：`invalid | step=3 | reason=invalid_action_token`；解码状态 `None`。
- `constrained`：`up right`；valid；停止原因 `feedback_valid`。
  R 反馈：`invalid | step=3 | reason=invalid_action_token`；解码状态 `complete`。
- `backtrack`：`up right`；valid；停止原因 `feedback_valid`。
  R 反馈：`invalid | step=3 | reason=invalid_action_token`；解码状态 `complete`。

constrained 的逐步执行：

| 步 | 动作 | 从 → 到 | 状态 |
| --- | --- | --- | --- |
| 1 | up | [3, 4] → [2, 4] | 合法 |
| 2 | right | [2, 4] → [2, 5] | 合法 |

### 6.3 从 G 开始：回溯恢复搜索分支：`test_id-3`

坐标从 0 开始，起点 (4, 5)，目标 (5, 1)，BFS 最短长度 5。`S` 为起点、`G` 为目标、`#` 为障碍。

```text
. . . # . .
. . . . . .
. . . . . .
. . . . . .
. . . . . S
. G . # . .
```

原候选：`up up up up up right right right right right`。真实诊断：`invalid | step=5 | reason=out_of_bounds`。

- `self`：`up up up up up right right right right right`；invalid | step=5 | reason=out_of_bounds；停止原因 `feedback_valid`。
  V 输出：`valid`。
- `executor_one`：`up up up up right right right right right right right`；invalid | step=5 | reason=out_of_bounds；停止原因 `repair_limit`。
  R 反馈：`invalid | step=5 | reason=out_of_bounds`；解码状态 `None`。
- `constrained`：`up up up up left down left left left down right right right down down left left left left up up up up right right`；invalid | step=26 | reason=goal_not_reached；停止原因 `repair_limit`。
  R 反馈：`invalid | step=5 | reason=out_of_bounds`；解码状态 `dead_end`。
- `backtrack`：`up up up up left down left left left down right right right down down left left left down`；valid；停止原因 `feedback_valid`。
  R 反馈：`invalid | step=5 | reason=out_of_bounds`；解码状态 `complete`。

backtrack 的逐步执行：

| 步 | 动作 | 从 → 到 | 状态 |
| --- | --- | --- | --- |
| 1 | up | [4, 5] → [3, 5] | 合法 |
| 2 | up | [3, 5] → [2, 5] | 合法 |
| 3 | up | [2, 5] → [1, 5] | 合法 |
| 4 | up | [1, 5] → [0, 5] | 合法 |
| 5 | left | [0, 5] → [0, 4] | 合法 |
| 6 | down | [0, 4] → [1, 4] | 合法 |
| 7 | left | [1, 4] → [1, 3] | 合法 |
| 8 | left | [1, 3] → [1, 2] | 合法 |
| 9 | left | [1, 2] → [1, 1] | 合法 |
| 10 | down | [1, 1] → [2, 1] | 合法 |
| 11 | right | [2, 1] → [2, 2] | 合法 |
| 12 | right | [2, 2] → [2, 3] | 合法 |
| 13 | right | [2, 3] → [2, 4] | 合法 |
| 14 | down | [2, 4] → [3, 4] | 合法 |
| 15 | down | [3, 4] → [4, 4] | 合法 |
| 16 | left | [4, 4] → [4, 3] | 合法 |
| 17 | left | [4, 3] → [4, 2] | 合法 |
| 18 | left | [4, 2] → [4, 1] | 合法 |
| 19 | down | [4, 1] → [5, 1] | 合法 |

此例说明回溯能从失败分支恢复到目标，但返回了 19 步路径，而最短路只有 5 步。它支持“提高成功率”的结论，同时直接展示了“成功不等于最优”的限制。

## 7. GPU 账本与可复核产物

历史两份账本共同前缀为 240 条，分叉后按完整记录去重合并为 261 条、7.841063 小时；没有把其中任一分支覆盖另一分支。本轮新增 4 条记录，合计 4016.397 秒（1.1157 小时），已记录累计 8.9567 小时。

| 操作 | 状态 | 占用墙钟秒 |
| --- | --- | ---: |
| `post_training:4ed341a2c10b` | completed | 59.266 |
| `post_training:159ce983c171` | interrupted_for_ssd | 528.422 |
| `post_training:159ce983c171` | interrupted_ssh_disconnect | 870.678 |
| `post_training:159ce983c171` | completed | 2558.031 |

HDD 中断条目按 driver 实测墙钟补记，包含启动/导入，采用保守计量。原 HDD 副本保留，SSD 续跑没有改变协议；强制退出未自动完成预算上下文的 finally，因此补账前确认了原账本无该条目。

SSH 中断条目按续跑 driver 的开始时间至恢复检测时刻计算，包含进程启动和断开后的检测间隔，是保守占用上界。最终 2558.031 秒条目正常落账；最终完整运行成功，不应把先前的两次运行中断当作算法失败案例。

后续历史 optional-dev 报告未附完整账本，因此上面的历史累计是已收集记录之和，不能宣称覆盖了所有历史 GPU 使用；本轮 smoke、中断和续跑均已记录。GPU 墙钟包含模型加载、CPU 评分和日志写入，不等同于 GPU 满负载时间。

- [机器可读结果](post_training_20261112/results.json) · [轨迹与测试 receipt](post_training_20261112/traces.zip)
- [已部署源码快照](post_training_20261112/source_snapshot.zip) · [协议可读副本](post_training_20261112/protocol_public.json)
- [GPU 账本](post_training_20261112/gpu_ledger.json) · [恢复校验](post_training_20261112/resume_validation.json) · [中断边界](post_training_20261112/interruption_boundary.json)
- [SSH 恢复记录](post_training_20261112/ssh_recovery.json) · [最终运行状态](post_training_20261112/run_status.json)
- [全部文件 SHA256](post_training_20261112/SHA256.json)

协议公开副本仅将机器绝对 checkpoint 路径改成仓库相对路径，保留原协议哈希；它不作为续跑 manifest。原始协议、每步 journal 和未修改输出另存桌面运行目录及本地 raw archive。源码快照保存实际部署字节；文档后续编辑不改变本次推理代码身份。
