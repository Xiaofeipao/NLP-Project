# Prompt 标准化对照

## 比较范围

本轮以 GitHub StepVerification-v1-v2@9eeb494fe1f775ad033ce2f668e22517cd194aee 的 step_prompt 为基准。输入是地图尺寸、动作前的位置、目标、障碍列表和单个动作。原 Prompt 解释坐标方向、合法抵达、非法动作、未到目标和局部输出格式。

三模板使用同一份冻结 NewPart1 Base（c28ae2c1），分别比较冻结权重推理和同预算 V 专用 LoRA。数据是 GitHub 证据包 v1/train.jsonl 中的 2,000 条 V 样例。训练 ID、位置、动作、target、正负构成完全相同。每组两轮、有效 batch 16、250 次更新、q/v rank 8、alpha 16、学习率 3e-4，相同随机种子和 batch 序列，保持 50:50 判别／后续文本损失。

GitHub 原 v1 从原始预训练底座联合全量训练 G/V。本轮在共同的冻结 NewPart1 上继续适配 V，锁定 GitHub Prompt 文本，在同数据和适配预算下测文字变化的增量。

## 三种模板

| 模板 | 调整 |
| --- | --- |
| GitHub 原版 | 用发布的 step_prompt 生成，逐字核对 2,000 条原训练输入。 |
| 精简固定版 compact | 固定字段顺序、JSON 坐标与排序后的障碍列表，输出格式只声明一次。错误原因使用简短释义，局部步号单独解释。 |
| 约束补齐版 explicit | 固定字段，补四方向位移、行列边界、检查顺序和输出枚举。检查必要约束更明确时的效果。 |

精简模板示例如下，实际运行逐例填入字段。

    Verify one action. Zero-based [row,column]; up decreases row, right increases column.
    Grid: 6x6
    Position: [4,3]
    Goal: [4,4]
    Obstacles: [[1,3],[3,0],[3,5],[4,5]]
    Action: "right"
    null means stop. Reply valid only for legal arrival or stopping at goal.
    Otherwise reply invalid | step=N | reason=R.
    R: invalid_action_token (unknown), out_of_bounds (outside), obstacle_collision (blocked), goal_not_reached (legal but unfinished).
    N=1 for illegal action or null; N=2 for legal unfinished move.

模板仅表达已知地图和任务规则。原 GitHub 逐步扫描、局部 R 和未完成定义沿用原逻辑，非空候选没有额外 STOP。模型回复 valid 或 goal_not_reached 且还有动作时继续扫描，其他错误或不合格式回复使扫描停止。全局错误步号由扫描器提供，最后是否完成由模型回复决定。

## 对照与评估

冻结同一权重比较三种推理模板，回答只改输入是否有效。分别使用三模板训练，比较匹配模板的成绩，回答同数据、同预算适配后的增量。另在 GitHub 模板与 compact 之间交叉推理，测模板不匹配的影响。只根据开发固定候选和开发 G 的验证指标选轮次。

报告 Macro-F1、格式率、漏检、误拒、首错与类型联合命中、原失败修好数和原合法路径修坏数。评测原 6×6 单错和双错、组员与 H100 的各两批固定 G，以及既有 7×7 和密集 6×6 候选。训练和评测布局核对无交集，既有测试此前曾被项目使用，本轮标记探索性比较。

## 实际结果

三套各训练完两轮，均由开发结果选第 1 轮，入选权重经历 125 次更新。训练脚本仍计入完整 250 次更新及第 2 轮开发评测成本。

| 同预算微调，训练／推理模板一致 | 原 GitHub 模板 | 精简固定版 | 约束补齐版 |
| --- | ---: | ---: | ---: |
| 训练输入平均 token | 190.8 | 160.5 | 242.5 |
| 训练输入最大 token | 201 | 171 | 253 |
| 开发固定候选 Macro-F1 | 52.38% | 51.65% | 60.11% |
| 开发 G 路径 Macro-F1 | 46.80% | 46.80% | 44.48% |
| 开发 G 合法误拒 | 0/168 | 0/168 | 15/168 |
| ID 单错 Macro-F1 | 52.38% | 54.00% | 54.81% |
| OOD 双错 Macro-F1 | 73.33% | 73.33% | 73.01% |
| ID／OOD 漏检，各 100 条 | 80 / 50 | 78 / 50 | 76 / 49 |
| ID／OOD 误拒，各 100 条有效 | 0 / 0 | 0 / 0 | 2 / 2 |
| ID／OOD 首错与原因联合命中，各 100 条 | 20 / 45 | 20 / 45 | 20 / 45 |
| 同一局部 R 修好 ID／OOD，各 100 条 | 10 / 26 | 10 / 26 | 10 / 26 |
| 组员两批 G 修好 | 0/31 · 0/21 | 0/31 · 0/21 | 0/31 · 0/21 |
| H100 两批 G 修好 | 0/30 · 0/29 | 0/30 · 0/29 | 0/30 · 0/29 |

精简版长度减少约 15.9%。固定单错的状态判别多检出 2 条，联合定位和最终修好数均保持相同。双错也未改善。约束补齐版在固定开发候选上高一些，但更长，在开发 G 路径上有误拒。按预设开发固定候选与开发 G Macro-F1 之和排序，约束版为 1.0459、原版 0.9918、精简版 0.9844。这个选择准则没有要求误拒为 0。相对于任务实际需求，仍需同时看合法路径被误判的风险。

## 交叉推理与冻结模型

| 训练模板 → 推理模板 | ID Macro-F1 | OOD Macro-F1 | 合法误拒，ID/OOD 共享 100 条 |
| --- | ---: | ---: | ---: |
| 原模板 → 原模板 | 52.38% | 73.33% | 0 |
| 原模板 → 精简模板 | 54.80% | 72.68% | 4 |
| 精简模板 → 精简模板 | 54.00% | 73.33% | 0 |
| 精简模板 → 原模板 | 33.33% | 33.33% | 100 |

交叉结果表明需要把训练模板版本与适配器绑定。原模板训练模型切到精简模板，在部分地图上有小幅变动，但双错 Macro-F1 和误拒不支持统一替换。精简模型反向换回原模板时会把共享的 100 条合法路径全部判成无效。

原样冻结的 NewPart1 没有局部任务适配。原模板、精简模板和约束版在固定单／双错上 Macro-F1 都是 33.33%，均把 100 条有效候选误拒；精简版的一些原因和定位成绩更低。这里是整路径模型直接读取局部提示的零样本对照，GitHub 已训练逐步 v1 的 37.75%／39.59% 属于另一份训练权重。

## 对标准化的建议与交付

可使用固定字段顺序、JSON 坐标、排序障碍和显式模板版本减少输入波动。若首要目的为缩短输入，精简版提供约 15.9% token 节约。当前单种子证据仅显示小幅状态判别变化，首错诊断和实际修复没有提高。不能把上轮结构化验证头得到的高分归因于本轮文字精简。v1 的有限局部训练覆盖和原扫描器的结束判断仍是固定条件，单改文字没有消除循环失败。

独立模板模块 prompt_templates.py 只依赖 Python 标准库，verification_prompt(grid, position, action, style="compact") 默认使用精简版。style="github" 生成原标准，style="explicit" 使用约束补齐版。模块的三种输出与冻结示例逐字一致。应用模板时同时选对应的 LoRA 权重。现有 GitHub 默认模板未覆盖。

远端最终权重和结果位于 /home/apulis-dev/outputs/NLP-Project-NewPart1-6939390/prompt-study-20261119-batched，适配器为 adapter_github、adapter_compact、adapter_explicit。代码在 /home/apulis-dev/code/NLP-Project-StepVerification-9eeb494/run_prompt_study.py。三套输入、目标、全部候选、局部回复和最终修复均有归档。最初未去重的冻结推理被停止，保留过程文件并将 311 秒费用计入账本。正式对照将重复提示去重，三组统一推理 batch 64，完整运行约 293.3 秒。该推理批次设置与 GitHub 原运行有区别，各实验组之间保持一致。

本地重算 18,328 条候选的验证与修复指标，并核对 86,000 条按组保存的局部回复记录数量。三份训练文件除 input 外逐字段相同。训练只使用原训练布局，测试曾被复用，结果属于探索性评估。数据、指标和源代码散列校验通过。

[完整指标](../deliverables/newpart1_text_verifier/prompt/results.json) · [独立复算](../deliverables/newpart1_text_verifier/prompt/audit.json) · [逐例归档](../deliverables/newpart1_text_verifier/prompt/evidence.tar.gz) · [训练与评测源码](../deliverables/newpart1_text_verifier/prompt/source.tar.gz) · [独立模板模块](../experiments/newpart1_text_verifier/prompt_templates.py) · [原模板示例](../deliverables/newpart1_text_verifier/prompt/Prompt_github_示例.txt) · [精简模板示例](../deliverables/newpart1_text_verifier/prompt/Prompt_compact_示例.txt) · [约束模板示例](../deliverables/newpart1_text_verifier/prompt/Prompt_explicit_示例.txt) · [GPU 台账](../deliverables/newpart1_text_verifier/prompt/gpu_ledger.json)
