# 第二部分本地 GPU 实验结果

2026-11-16。训练及开发、ID/OOD评测全部完成，退出码0。本轮代码与无权重证据包随后发布，模型权重仅本地保留。设计见 [协议](phase2_feedback_protocol.md)。[指标JSON](phase2_feedback_results.json)及[审计回执](phase2_feedback_audit.json)另有仓库副本。原始指标、逐例轨迹和审计保存于 `outputs/phase2_feedback/run_001/`。

## 实验身份

FirstPart 分支未发布本轮主模型的 LFS 权重路径；按用户授权从固定预训练 revision 独立复现 Flan-T5-base。2轮500更新，按开发结果选择第2轮。开发生成成功168/191=87.96%，最优率87.96%，验证Macro-F1 62.97%、定位18%、类型34%。不能与同伴原权重差值混称方法提升。

权重 SHA256：`199332793094904518fff0a0c3597d082d12caebbe7378d911be3f0a09803f0a`。权重位于 `outputs/phase2_reproduction/run/train/flan_t5_base/best/`，仅本地保留。GPU pilot 76.95秒、训练2518.41秒、后续评测88.80秒；保守累计账本17.8925小时（含公共历史重复计费），低于20小时。

## 模块与边界

G为已训练的下一动作策略；V为模型诊断或明确的符号校验；R为反馈引导的模型续走，不是已训练的整路径文本R。候选与反馈决定保留前缀，下一动作由模型评分选择，最多36个新增动作。raw R没有mask/search；visit R过滤非法动作并优先最少访问位置，再用模型评分选动作。没有BFS/A*补路或成功路径缩短S。reason不输入动作策略，不能用本实验证明模型理解详细原因文本。

## 生成后完整流程

同一200环境、同一初始G。三率分母均200；ID/OOD地图配对，不重复计算两套生成成绩。

|方法|Feasibility|Success|Optimality|
|---|---:|---:|---:|
|direct|100.0%|85.5%|85.0%|
|self|100.0%|85.5%|85.0%|
|vocabulary|100.0%|85.5%|85.0%|
|terminal|100.0%|85.5%|85.0%|
|executor|100.0%|85.5%|85.0%|
|self_visit|100.0%|85.5%|85.0%|
|terminal_visit|100.0%|98.5%|85.0%|
|executor_visit|100.0%|98.5%|85.0%|

只改V的terminal仍85.5%；只改R的self_visit也85.5%；两者一起98.5%，救回26/29条原失败路径，仍有3条预算内未到目标。原始29条失败全部合法未到目标，模型V全部误接受；同时模型V将27条成功路径误拒绝，故终点校验虽然找齐失败，仍不能消除所有误拒绝。最终各组合法候选退化数均为0，但不能解释为V没有误拒绝。

## 固定非法候选修复

下表分母各100条原非法候选。控制候选另外100条，没有算入修复分母。

|方法|ID修复成功|OOD修复成功|
|---|---:|---:|
|direct|0.0%|0.0%|
|self|20.0%|28.0%|
|vocabulary|29.0%|50.0%|
|terminal|75.0%|73.0%|
|executor|94.0%|92.0%|
|self_visit|21.0%|30.0%|
|terminal_visit|78.0%|78.0%|
|executor_visit|100.0%|100.0%|

## 验证指标

|方式|ID Macro-F1|ID定位|ID类型|OOD Macro-F1|OOD定位|OOD类型|
|---|---:|---:|---:|---:|---:|---:|
|self|62.81%|15.00%|35.00%|78.21%|14.00%|53.00%|
|vocabulary|62.81%|30.00%|35.00%|78.21%|53.00%|53.00%|
|terminal|87.88%|78.00%|78.00%|89.94%|77.00%|75.00%|
|executor|100.00%|100.00%|100.00%|100.00%|100.00%|100.00%|

## 案例与失败分析

### 发现错误后仍需摆脱循环

案例 test_id-14，网格 `{'rows': 6, 'cols': 6, 'obstacles': [[0, 4], [2, 2], [5, 1]], 'start': [4, 0], 'goal': [0, 2]}`，最短距离 6。
初始路径：`right right up right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right`。
模型诊断：`valid`；执行器诊断：`invalid | step=37 | reason=goal_not_reached`。
- terminal：action_budget，最终 `invalid | step=73 | reason=goal_not_reached`，72步；输出 `right right up right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right`。
- self_visit：accepted，最终 `invalid | step=37 | reason=goal_not_reached`，36步；输出 `right right up right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right`。
- terminal_visit：policy_reaches_goal，最终 `valid`，40步；输出 `right right up right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right up up left up`。

### 约束续走仍会耗尽预算

案例 test_id-32，网格 `{'rows': 6, 'cols': 6, 'obstacles': [[0, 1], [1, 1], [2, 3], [3, 0], [5, 2]], 'start': [4, 0], 'goal': [1, 0]}`，最短距离 5。
初始路径：`right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left`。
模型诊断：`valid`；执行器诊断：`invalid | step=37 | reason=goal_not_reached`。
- terminal：action_budget，最终 `invalid | step=73 | reason=goal_not_reached`，72步；输出 `right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left`。
- self_visit：accepted，最终 `invalid | step=37 | reason=goal_not_reached`，36步；输出 `right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left`。
- terminal_visit：action_budget，最终 `invalid | step=73 | reason=goal_not_reached`，72步；输出 `right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left down right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right left right`。

## 结论与限制

准确反馈在固定非法候选修复中有效：相同raw R，模型反馈ID/OOD仅20%/28%，执行器反馈94%/92%。终点校验将其提高至75%/73%；加访问约束后78%/78%。在自生成循环路径上，仅准确诊断不足以成功，必须同时改变恢复动作选择；因此没有单一“反馈总是有效”结论。

这些是已查看测试集上的探索性结果，单seed，未运行新增7×7/密集6×6。完整执行器诊断的验证100%是定义上的规则真值能力，不是学习提升。访问约束的收益是符号动作约束与模型评分的系统收益；本轮未提供其uniform/Manhattan匹配对照，不能单独归因于模型。

四个新CPU边界测试通过，FirstPart工作流3项检查通过；独立回放复核352项指标、官方执行器路径语义与代码哈希一致。训练和后续GPU全部完成。
