# Part3 方法比较与消融

本目录接续Part2，对应题目“提出改进后的验证或修复方法”及聚焦消融。不是另起一个数据集，也不是课程报告章节Part 3。代码沿用FirstPart的阶段入口与协议/工作流分离，底层复用 `ppnl/method_sweep.py`。详见 [逻辑关系](../docs/part1_part2_part3.md)、[固定协议](../docs/seven_methods_protocol.md)、[全部结果](../docs/seven_methods_results.md)。

## 已完成

七类有限代表方案、21配置、三条等追加训练预算分支，开发/ID/OOD全部完成。代表方案包括结构化提示、CoT提示、迭代修复、验证器引导动作选择、受约束动作选择、分类辅助目标、输入表示增强；不是穷举所有算法或超参数。

研究配置在测试前由开发集选出 `terminal_visit`：成功98.5%、可行100%、最优85%，单/双错误候选修复均78%。迭代没有提升，保留一轮。若将完整规则V控制也纳入总体排序，最好的是 `executor_visit_control`，候选修复均100%，生成仍98.5%；不能将规则V的100%说成模型V的成绩。

`best_research.json`保留原冻结研究选择；`best_overall.json`为包含规则控制的总体开发最佳。追加训练的分类辅助目标在ID有部分收益，OOD不稳定，未替换默认。普通追加训练和输入增强控制、CoT格式失败均保存，不删除失败证据。

## 复核和运行

```powershell
python -m part3.run audit
python -m part3.run verify-results
python -m part3.run plan --check-only
python -m part3.run plan --grid examples/grid.json --out outputs/part3/overall_001
python -m part3.run plan --config part3/best_research.json --grid examples/grid.json --out outputs/part3/research_001
```

前两个命令无需权重/GPU。plan需要本地实体权重；配置中的历史路径/哈希仅适用于本机这轮产物，独立重训应建立新的配置并绑定新权重哈希，不能冒充同一checkpoint。

重新执行全部训练和实验：

```powershell
python -m part3.run run --checkpoint LOCAL_PRIMARY_CHECKPOINT --out outputs/part3/new_study --ledger SHARED_LEDGER
```

将调用相同21配置和开发选优规则，新目录记录配置、代码/数据/权重哈希、训练样本IDs、选择文件、逐例轨迹。默认使用 `part2/protocol.json`。三训练支从共同两轮权重独立出发，只更新V；每支追加一遍1,984条样本，排除pilot已用满遍数的16例。G/R权重保持固定。

CLI没有替换旧main的默认规划器，也没有删除FirstPart或历史研究。此次打包只改善目录、可复现入口和结果归属，不重新声称跑了一轮GPU。


## 后续 R 最优率实验

新增删环、Top3、四动作概率、软循环惩罚与硬禁止的10方案，在错误集用CPU完成。开发选择Top3+软惩罚+删环，固定原G轨迹回放200环境为100%成功、92.5%最优。详见 [R最优率实验](../docs/r_variants_results.md)。独立运行：`python -m part3.run_r_variants --out outputs/r_variants/new_run --ledger SHARED_LEDGER --device cpu`；依赖同一份本地权重及本地G失败轨迹，CPU不扣GPU推理时间。此新研究未替换旧best_overall配置，也未push。


## 新地图单错/双错 × 原始/增强R

冻结两种R、4种V来源和7种规则，在400配对泛化环境上评测，详见[结果](../docs/generalization_r_pair_results.md)。双错数据仅测试，生成命令：`python -m scripts.build_generalization_double --output outputs/double_rebuild`。GPU评测：`python -m part3.run_generalization_r_pair --out outputs/generalization_r_pair/new_run`；CPU复核/报告：`python -m part3.report_generalization_r_pair --run outputs/generalization_r_pair/run_001`。单错V复用已核验历史输出，双错V重新推理，两种R均重新执行，无新训练或测试选型。


## NewPart1拆分损失迁移实验

同伴源码6939390和本地兼容训练均隔离保存，未覆盖旧主模型。见[主要发现](../docs/newpart1_findings.md)和[全部结果](../docs/newpart1_transfer_results.md)。原始/兼容源码包及patch在deliverables，checkpoint仍需本地准备，权重不发布。GPU命令：`python -m part3.run_newpart1_transfer --checkpoint LOCAL_SPLIT_LOSS_CHECKPOINT --out NEW_RUN --compat-no-checkpoint`。无权重复核：解压`deliverables/newpart1_transfer_evidence.zip`后，`python -m part3.report_newpart1_transfer --run EXTRACTED/evaluation --public-evidence`。完整规则均列出，不按测试选新方法。
