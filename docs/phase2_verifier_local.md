> 前置CPU回放历史记录：这里使用同伴原Flan-T5-base预测。后续独立重训与GPU结果见 [Part2报告](phase2_feedback_results.md)，不要混合数值。回放所需三份phase1证据ZIP位于远端FirstPart的deliverables目录。

# 第二部分验证器改进本地实验

2026-11-16。以 origin/FirstPart 的 f00fbdf 为基线，主模型为 Flan-T5-base epoch 2。保留此前本地修改；没有合并整个分支、提交或 push。证据从该提交的三个 phase1 ZIP 读取，未运行其中的脚本。

## 失败与设计

模型验证器倾向输出 valid。固定 ID 候选误接受 85/100，OOD 误接受 61/100；生成路径中 31 个失败均为合法动作未到终点，全部被误接受。设计复用执行器状态跟踪，避免继续用错误模型诊断控制修复触发。

四个固定模式：model 原输出；vocabulary 在尚未发生空间错误的前缀上识别非法 token；terminal 在此基础上对全程合法路径检查终点；trace 使用完整执行器诊断。partial 模式遇到更早碰撞/越界时保留模型判断，避免把后续 token 错当首错。trace 是纯符号 V 上界，结果接近真值是按定义成立，不是神经能力提升，也不是新算法创新。

## 结果

dev

- model: Macro-F1 0.5559, 首错定位 8.0%, 类型 24.0%, 无效误接受 38。
- vocabulary: Macro-F1 0.6011, 首错定位 28.0%, 类型 30.0%, 无效误接受 35。
- terminal: Macro-F1 0.8990, 首错定位 80.0%, 类型 80.0%, 无效误接受 10。
- trace: Macro-F1 1.0000, 首错定位 100.0%, 类型 100.0%, 无效误接受 0。

test_id

- model: Macro-F1 0.4813, 首错定位 8.0%, 类型 15.0%, 无效误接受 85。
- vocabulary: Macro-F1 0.5864, 首错定位 24.0%, 类型 28.0%, 无效误接受 72。
- terminal: Macro-F1 0.8887, 首错定位 78.0%, 类型 78.0%, 无效误接受 22。
- trace: Macro-F1 1.0000, 首错定位 100.0%, 类型 100.0%, 无效误接受 0。

test_ood

- model: Macro-F1 0.6637, 首错定位 23.0%, 类型 34.0%, 无效误接受 61。
- vocabulary: Macro-F1 0.7630, 首错定位 50.0%, 类型 50.0%, 无效误接受 45。
- terminal: Macro-F1 0.8990, 首错定位 77.0%, 类型 75.0%, 无效误接受 20。
- trace: Macro-F1 1.0000, 首错定位 100.0%, 类型 100.0%, 无效误接受 0。

generated_id

- model: Macro-F1 0.4580, 首错定位 0.0%, 类型 0.0%, 无效误接受 31。
- vocabulary: Macro-F1 0.4580, 首错定位 0.0%, 类型 0.0%, 无效误接受 31。
- terminal: Macro-F1 1.0000, 首错定位 100.0%, 类型 100.0%, 无效误接受 0。
- trace: Macro-F1 1.0000, 首错定位 100.0%, 类型 100.0%, 无效误接受 0。

## 具体错误案例

### test_id-1

网格：{'rows': 6, 'cols': 6, 'obstacles': [[2, 1], [2, 2], [3, 2]], 'start': [3, 4], 'goal': [2, 5]}。
候选：`down right`。
模型：`valid`；真值：`invalid | step=3 | reason=goal_not_reached`。
终点模式：`invalid | step=3 | reason=goal_not_reached`；完整跟踪：`invalid | step=3 | reason=goal_not_reached`。

### test_id-15

网格：{'rows': 6, 'cols': 6, 'obstacles': [[1, 5], [5, 1]], 'start': [0, 5], 'goal': [1, 0]}。
候选：`down left down left left left left`。
模型：`valid`；真值：`invalid | step=1 | reason=obstacle_collision`。
终点模式：`valid`；完整跟踪：`invalid | step=1 | reason=obstacle_collision`。

## 边界与后续

本轮使用保存的模型输出做 CPU 回放，官方执行器逐条核实标签，协议在回放前写入，记录输入和代码哈希。CPU 时间与 GPU 分开；新增 GPU 调用/训练步数均为 0。dev100、固定 ID200/OOD200、生成 ID200；生成与候选来自相同地图，不能相加称独立环境。已见测试属于探索性分析，不据此选模型或调参。

新增 traced_verification_prompt 为模型提供逐步执行观测，可用于比较原提示与工具辅助提示；观测直接含错误信息，不能叫纯语言模型推理。此入口尚未实测模型输出。未发现 FirstPart 主模型实体权重；本机 GPU 可用，但不能拿历史 Flan-T5-small 或不同 Flan-T5-base 权重替代。新分支证据仅含训练摘要和预测，权重需从原训练机器恢复并核验 SHA256 26385be1eca841d0dfa91895597a9080e56e30cf922cec093f30df54013737db。

正式 R 方案在同伴 FirstPart 报告仍标为待定，本轮不把其历史规则修复成绩当正式基线。不重训不重跑 R，因此没有新增规划 Success/Feasibility/Optimality 结果。验证触发更准确不必然使 R 成功，必须后续配对测同一修复器。

复现：python scripts/evaluate_phase2_verifier.py --out outputs/phase2_verifier/run_new；来源需先取回 FirstPart 证据 ZIP 到 outputs/phase2_verifier/source。两项首错优先级/终点与绕路边界测试通过。
