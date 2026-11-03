# 数据隔离、开发集选择与测试前冻结

## 适用范围

本规范借鉴 HotpotQA 抽样要求中的固定划分、原始 ID、互斥检查及开发集选模原则，应用于现有 PPNL 网格任务。HotpotQA 不加入主实验，不套用它的 easy/medium/hard 标签或 300/150/200 配额；参考文件测试总数的 1000/200 矛盾也不影响现有配额。

当前沿用 seed=42、固定上游版本和历史布局分组划分。训练集 4000 条（G/V/R 为 800/2000/1200），开发集 200 条（40/100/60），ID/OOD 各 200 条。训练样本曾在各布局组内有放回抽取，因此不能宣称历史数据遵守 HotpotQA 的“题目无放回抽样”；同一训练环境跨任务复用是已知设计。不得为了符合新表述重写旧数据或覆盖历史结果。

## 隔离与审计

`ppnl/experiment_freeze.py::audit_splits` 在 CPU 上检查：

- 重算真实布局 ID，禁止 train/dev/test 布局交叉；不同起终点也不能绕过布局隔离。
- 样本 ID 在各 split 内及跨 split 不重叠；来源 ID 在 train/dev/test 间不重叠。
- 每个样本属于 manifest 声明的布局组，任务数量与 manifest 一致，已有任务索引未漂移。
- ID/OOD 有意使用相同、同序的布局/起点/终点，二者来源可重叠；它们是候选错误条件的配对比较，不能当作两组独立规划测试。
- 导出全部样本 ID、原始来源 ID、布局 ID、随机种子、上游 revision、各数据文件 SHA256 和任务索引。

运行（仅 CPU，不加载模型）：

```powershell
python scripts/freeze_experiment.py audit --protocol config.gvr.next.json --out outputs/protocol_audit/splits.json
```

输出必须使用新路径，不覆盖旧证据。训练/验证/评测 CLI 执行审计，preflight 也将隔离检查纳入是否就绪的判断。独立 Python API 和历史运行脚本不因此自动获得冻结保障；新实验统一使用下述正式入口。

## 选择、冻结和执行

1. 训练集用于权重或技能优化。开发集用于 checkpoint、损失权重、学习率、路由规则、调用预算及停止条件选择。
2. 用开发集检查所选 G/V/R，生成身份匹配、关闭 adapter 与裸 G 一致的 `verify_dev.json`。该文件是选模检查证据，具体选择理由另记在冻结说明中。
3. 完成所有代码和配置修改后创建冻结文件。冻结包含 config/protocol 指纹、G/V/R 全目录文件（包括 tokenizer/config）指纹、全部 `ppnl/` 和 `scripts/` Python 源码、执行器、数据划分和开发集验证证据。
4. 使用同一锁执行 ID/OOD 测试。更改任一绑定输入必须重新走开发集选择和新实验记录，不得修改旧锁或覆盖旧结果。

以下命令是未来新实验的流程示例。现有结果目录已存在，须改用独立新输出目录；`verify` 需要 GPU，其余审计/冻结/check 不执行 GPU 运算。

```powershell
python scripts/run_gvr.py verify --protocol config.gvr.next.json
python scripts/freeze_experiment.py freeze --protocol config.gvr.next.json --selection-report outputs/gvr_next/verify_dev.json --note "Describe development-only selection and fixed inference settings" --out outputs/gvr_next/freeze.json
python scripts/freeze_experiment.py check --protocol config.gvr.next.json --out outputs/gvr_next/freeze.json
python scripts/run_gvr.py evaluate --protocol config.gvr.next.json --split test_id --freeze outputs/gvr_next/freeze.json
python scripts/run_gvr.py evaluate --protocol config.gvr.next.json --split test_ood --freeze outputs/gvr_next/freeze.json
```

`run_gvr.py evaluate` 对 test_id/test_ood 强制要求 `--freeze`，在 CUDA 初始化前检查绑定输入。成功后在结果目录保存 `experiment_freeze.json`，关联锁哈希和测试性质。dev 评测、历史结果 CPU 重算不要求测试锁。哈希用于发现意外漂移，不是防篡改签名，也不能自动判断人是否用过测试结果选模。

## 历史测试的性质

当前 ID/OOD 已参与失败分析和后续优化，始终标记为 `reused_exploratory`。现在冻结只能保存当前状态，不能让历史测试重新变成盲测，也不能倒签为过去运行时的锁。原报告、权重、原始日志和指标保持原样。

若后续需要确认性结论，必须另建版本化 holdout：在未用于训练、选模或失败分析的布局中按预先声明的规则抽样，不与现有任一布局重叠；记录完整 ID、样本数、种子、来源及排除清单，再冻结所有条件。若没有合格未使用布局，应明确披露并另找数据来源，不能把已见样本改名当新测试。目前的实现专用于历史配对划分，不创建或声称已有新的盲测集。

## 技能图：下一步独立消融，尚未实现或运行

主实验继续使用现有 G/V/R 与 Harness。技能图另开实验，不混入当前基线或改变历史成绩。

- 对照：现有固定 Harness；明确规则的固定技能图；仅用训练轨迹更新并由开发集选择的技能图。
- 节点候选：整路径生成、验证、修复、逐步移动。边描述触发条件、输入输出约定和失败退出条件；环境网格图不等于技能图。
- 固定图与演化图使用相同技能节点、模型 checkpoint、合法动作约束、工具权限及模型/工具调用上限。若引入逐步策略，应另加使用同一策略的固定路由对照，披露额外模型。
- 演化只使用训练轨迹提出/修改技能或边，开发集选版本。测试中允许按冻结规则路由，但不跨样本写入学习记忆或更新技能库。
- 冻结技能内容、边、版本、更新算法和推理规则。真正实现时须把技能文件纳入冻结清单；当前 GVR 锁尚不支持外部技能资产，不能直接拿来宣称技能图已冻结。
- 报告配对成功率、最优率、失败类型、模型/工具调用和资源消耗。增加预算、额外模型或搜索时单列对照，不能将收益全部归给技能图。

这是一份后续实验计划，不是新增技能算法已经改善规划能力的证据。
