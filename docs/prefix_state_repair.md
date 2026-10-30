# 从首次失败位置逐步修复生成路径

此方案把已保存的 G/V/R 真实生成路径作为同一批输入。官方执行器检查整条路径；失败时只保留首次失败前已执行的合法动作，从该位置让开发集选出的逐步模型逐动作续写。若候选已成功，原样保留；若路径先到目标又在后续出错，则在首次到达目标处停止。所有最终路径重新交给官方执行器评分。

这是一项推理流程改进，沿用现有逐步策略权重，不把生成器、验证器或修复 LoRA 的训练结果混称为本方案的能力。当前仓库中的 artifacts/gvr/evaluation/<split>/pipeline_traces.jsonl 提供原始生成路径及整路径执行器修复对照；data/derived/ 提供相同环境。开发集只选 40 个 Generate 环境，ID/OOD 各 200 个。ID/OOD 的网格配对相同，因此两组端到端结果不能当作独立的网格泛化试验。

## 方法与消融

| 名称 | 实际行为 |
| --- | --- |
| direct | 保存的生成器初始路径 |
| executor_one / executor_three | 保存的整路径模型在执行器反馈下修复 1/3 轮 |
| prefix_raw | 保留合法前缀，逐步模型不加动作约束 |
| prefix_legal | 同上，每步只允许网格内无碰撞动作 |
| prefix_visit | 同上，优先访问次数最少的合法邻居；死路允许回退 |
| prefix_visit_heuristic | 相同前缀、约束和预算，使用纯目标距离偏好代替模型打分 |
| restart_visit | 使用与 prefix_visit 相同的逐步模型和约束，但从原起点规划 |

每种方法在同一条初始生成路径上运行。前缀方法的动作数上限包含保留的前缀；从头规划方法使用相同的总动作数上限。执行器仅提供真实反馈、已执行状态和最终评分；推理阶段不读取参考最短路、测试集目标路径或 BFS 标签。合法动作过滤和访问次数规则是显式符号约束，需与模型贡献分开解释。prefix_visit_heuristic 是必要的非神经对照：若它与模型同样好，提升主要来自流程/约束。restart_visit 则检验保留原路径是否带来收益。整路径修复与逐步模型训练任务不同，不能把它们的差异归因于单一机制。

## 本机已完成的纯符号对照

在归档的真实生成路径上，CPU 执行 prefix_visit_heuristic 得到：

| 划分 | 成功 | 最优 | 平均续写步数 |
| --- | ---: | ---: | ---: |
| 开发集 | 40/40 | 42.5% | 5.475 |
| ID | 200/200 | 38.5% | 5.21 |
| OOD | 200/200 | 38.5% | 5.21 |

这些结果保存在 artifacts/prefix_state_repair/ 下各划分的 results.json 与 traces.jsonl，由官方执行器核对。ID/OOD 使用相同网格和相同生成路径，所以数字相同并非两次独立生成验证。此对照没有加载或调用语言模型。它表明这组小网格的成功率已被简单符号控制做到上限；真实逐步模型仍需比较最优率、路径长度、决策次数及实际推理耗时。

## 运行

从仓库根目录先检查归档生成路径、数据和执行器：

~~~powershell
python scripts/fetch_official.py
python scripts/run_prefix_state_repair.py --split dev --check-only
python scripts/run_prefix_state_repair.py --split test_id --check-only
python scripts/run_prefix_state_repair.py --split test_ood --check-only
~~~

无需模型或 GPU 可单独复现纯符号对照，输出必须使用新的目录：

~~~powershell
python scripts/run_prefix_state_repair.py --split test_id --symbolic-only --out outputs/prefix_state_repair_symbolic/test_id
~~~

真实模型评测需要 config.policy.best.json 指向的逐步策略 checkpoint、对应的 policy_selection.json、兼容的 CUDA PyTorch 和共享 GPU 账本。该策略权重及选择文件位于历史 outputs/planning_v3/multi/，目前未随 Git 仓库发布。已有 GPU 账本须继续使用；若新环境没有账本，应先从归档的历史账本恢复并核对累计值，不能从零重新计费。

~~~powershell
python scripts/run_prefix_state_repair.py --split dev --out outputs/prefix_state_repair/dev
python scripts/run_prefix_state_repair.py --split test_id --out outputs/prefix_state_repair/test_id
python scripts/run_prefix_state_repair.py --split test_ood --out outputs/prefix_state_repair/test_ood
~~~

每个输出目录必须是新目录。results.json 保存成功率、可行性、最优率、平均逐步决策数，以及数据、生成轨迹、策略权重、配置、执行器和实现哈希；traces.jsonl 保存初始路径、真实失败反馈、合法前缀、各步位置/动作和最终反馈。使用后续 G/V/R 生成器时，可通过 --initial-traces 指向其同格式逐条路径。更换策略 checkpoint 时，默认要求它与开发集选模文件的权重哈希一致；显式研究新权重时才使用 --allow-unselected-weights。

已完成 CPU 边界测试、三个 split 的归档输入检查及正式纯符号对照，尚未在此工作区运行真实逐步模型评测。当前工作区缺少逐步策略 checkpoint、开发集选模文件和 GPU 账本，本机也没有 CUDA 设备。任何使用固定或模拟动作分数得到的 CPU 数值仅验证流程，不作为模型实验结果。
