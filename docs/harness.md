# PPNL Agent Harness（2026-10-23）

首次实现时按用户要求只编写代码和测试。随后用户恢复 GPU 并授权开跑：47 项回归测试已全部通过，逐步策略已在示例网格上用 5 次调用成功；整路径反馈修复完成 4 轮后预算耗尽，提案均越界。该单样例结果不能视为总体成功率。旧模型、冻结指标和旧评测报告保留。

本机实测识别为 RTX 5060 Ti 16GB，NVIDIA 驱动 610.88，`torch_py311` 中 PyTorch 2.12.1+cu132，CUDA 张量运算及真实模型推理正常。首次测试受到执行沙箱临时目录权限限制，沙箱外重跑全部通过；该权限问题不属于 NVIDIA 故障。试验记录分别位于 `outputs/agent_harness/20261023_policy_2326/` 与 `outputs/agent_harness/20261023_repair_2326/`。

统一评测也已完成：dev（98 个去重网格）、test_id/test_ood（各 200 个）均与冻结指标逐字段一致，差异为 0。ID 的纯逐步策略成功率 85%，合法动作约束 97.5%，最优性保持的神经引导 A* 成功率及最优率均为 100%；纯 A* 对照同样为 100%。报告位于 `outputs/harness/20261023_gpu_validation/report.md`，逐条规划与修复轨迹在同目录的 `detail/`。这里的 `original_whole_plan` 行复用已有 baseline 轨迹，其余注册方法重新执行；没有重新训练或选择 checkpoint。

## 网上所说的 Harness

- [Microsoft Agent Framework：Agent Harness](https://learn.microsoft.com/en-us/agent-framework/concepts/harness)：模型外部的执行环境，管理工具、上下文、执行循环及状态。
- [Cloudflare：Agent harnesses](https://developers.cloudflare.com/agents/harnesses/)：负责给模型输入、执行工具、反馈结果，以及继续或停止。
- [Anthropic：Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)：区分 agent harness 与 evaluation harness；评测要检查环境结果，不能仅相信模型声称完成。
- [Anthropic：Effective harnesses for long-running agents](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents)：将工作进度持久化，让后续会话可以接续工作。
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)：语言模型基准评测框架。这里没有安装或冒充这个框架；本网格任务使用项目自身的执行器和多步交互协议。

旧 `ppnl/harness.py` 的评测注册表和指标对比仍然有用，移至 `ppnl/eval_harness.py`；旧命令移至 `scripts/run_eval_harness.py`，旧测试移至 `tests/test_eval_harness.py`。新的同名入口用于真正的 agent 运行循环，避免把两种 Harness 混称。

## 新运行层

`ppnl/harness.py` 只用标准库和确定性网格执行器。循环是：持久化并扣除一次模型调用预算 → 生成受限工具请求 → 检查参数和剩余预算 → 执行并反馈 → 持久化状态 → 判断停止条件。

工具白名单只有 `move(action)`、`verify_plan(plan)`、`submit_plan(plan)` 和 `finish()`。没有 shell、文件写入、网络或任意 Python 工具。`verify_plan` 只检查，不提交；`submit_plan` 只提交完整有效的路径；`move` 只提交物理上可行的一步。撞墙或错误提案保留为反馈，环境位置不变。到达目标后自动停止；`finish` 本身不能产生成功。命令行运行必须加载与数据 manifest 哈希一致的官方执行器。

`ppnl/harness_backends.py` 有两种基础适配器，以及下文介绍的 `repair_policy` 组合恢复适配器：

- `policy`：复用训练好的 `NeuralPolicy`，每轮输出一个动作。`legal_mask` 明示是否过滤非法邻居，不使用搜索、不自动避开所有访问过的位置，也没有 BFS 兜底。这个模型读取当前网格状态，不会因为包在 Harness 中就自动学会理解反馈文本。
- `repair`：复用原模型的 generation/repair 模板，整路径提案失败后，将首次错误反馈交给下一轮修复。Flan-T5/BART 不原生调用这些工具，工具请求由适配器构造。

配置分别在 `configs/harness/policy.json` 和 `configs/harness/repair.json`。repair 示例使用原 Flan-T5 整路径 checkpoint；若比较 BART，应建立独立配置并指向已训练的 BART checkpoint，同时保持数据与预算一致。此轮没有重新选择模型。

## 有 GPU 后使用

从项目根目录、原 `torch_py311` 环境执行。以下 trial_001 命令为使用示例；本次实际试验使用上文记录的独立目录。

```powershell
# 只校验文件、配置、版本和哈希；不加载模型、不创建试验目录
python scripts/run_harness.py run --spec configs/harness/policy.json --out outputs/agent_harness/policy_trial_001 --dry-run

# 单个隔离 trial；不训练
python scripts/run_harness.py run --spec configs/harness/policy.json --out outputs/agent_harness/policy_trial_001
python scripts/run_harness.py inspect --run outputs/agent_harness/policy_trial_001
python scripts/run_harness.py resume --run outputs/agent_harness/policy_trial_001

# 原评测功能已改名；GPU 评测示例
python scripts/run_eval_harness.py --config config.policy.best.json --split dev
```

每个新 trial 必须使用 `outputs/agent_harness/` 下的新目录。产物包括不可覆盖的 `manifest.json`、保存输入的 `spec.json`、追加式 `events.jsonl` 和原子替换的 `state.json`。记录配置、checkpoint 文件、关键代码和 ML 包版本的指纹。恢复时重新校验这些输入，改变配置、权重、网格或执行器需要新 trial。`inspect` 不导入模型库。

日志为事实来源：即使 `state.json` 落后，也从日志的完整已提交状态恢复。每条事件记录序号及前一条哈希；损坏、截断或不完整尾行会明确报错，不静默跳过。OS 文件锁防止同时运行同一 trial，进程退出后锁释放。哈希用于发现意外损坏，不是针对能重写所有文件的攻击者的防篡改签名。

模型请求在调用前记账；中断后可再次请求模型，因此属于至少一次模型调用语义，不能保证外部计费的恰好一次。环境工具只修改内存状态，并以完整日志事件提交，已提交动作不会因快照落后而再次应用。模型调用途中崩溃时，该次请求仍计入次数，但崩溃前尚未落盘的耗时无法恢复。

预算包括模型调用数、工具调用数、动作提案总量、单提案长度、保留的历史条数和活动耗时。历史保留最近 N 条，不能恢复为无限上下文。活动耗时在模型/工具调用边界检查，不能抢占正在阻塞的同步生成，不是进程级硬超时；模型加载耗时由现有项目 GPU ledger 记录。多个 GPU trial 仍应串行运行，现有 ledger 没有跨进程预留机制。

## 失败恢复优化（2026-10-23）

新增 `repair_policy` 模式：保留原整路径模型最多两次提案机会，仍未成功时切换到开发集选定的逐步策略。恢复从真实已提交位置开始，失败提案不移动环境；切换条件取自持久化状态，重启后不会重新获得两轮修复额度。每轮只调用一个模型，调用预算同时覆盖两个阶段。这个改进增加了第二个已训练模型，并没有改善原整路径 checkpoint 的参数。

可选 `avoid_revisits` 在合法邻居中优先选择访问次数最少的位置，再按神经评分选动作；无未访问邻居时仍允许回退。它是明确的执行约束，不运行图搜索，也不能保证成功或最优。原 `repair` 和默认 `policy` 行为不变。

新配置 `configs/harness/repair_policy.json` 的总预算为 16 次模型/工具调用，原失败示例现已用 7 次调用成功：2 次失败提案 + 5 次恢复动作。原先 4 次预算与新预算不同，不能仅靠这个示例归因；下面的配对比较为全部方法统一预算，并加入纯逐步策略对照。

```powershell
python scripts/run_harness.py run --spec configs/harness/repair_policy.json --out outputs/agent_harness/recovery_trial_001
python scripts/compare_harness.py --split dev --limit 20 --calls 16 --out outputs/agent_harness/recovery_compare_001
```

`compare_harness.py` 按源文件顺序取前 N 个去重网格，输入模型的只有网格和执行反馈，不使用参考路径或候选错误标签。三个方法使用相同样本与总预算，恢复模式与纯策略对照使用相同的合法动作/访问次数约束。每个 trial 清空评分缓存、单独记日志，模型权重在各自组内复用以减少加载成本；既有冻结基线不会覆盖。指标的活动耗时不含模型加载和落盘开销；`gpu_ledger` 另记整组资源消耗。

新增策略合约测试后，测试总数为 52，全部通过。

首轮配对验证已完成，记录在 `outputs/agent_harness/20261023_recovery_dev20/results.json`：开发集按源顺序的前 20 个去重网格、每个方法最多 16 次调用，纯整路径修复为 0/20 成功、平均 16 次调用；恢复模式为 20/20、平均 7.5 次调用；相同约束的纯逐步策略也是 20/20、平均 5.5 次调用。后两者最优率均为 80%，不能称为最优路径保证。对应累计活动耗时约为 480.3、10.5、3.1 秒，组间模型加载和 GPU warmup 有差异，不视为严格系统性能基准。

因此新增直接逐步模式配置 `configs/harness/policy_revisit.json`，适合优先使用；兼容旧整路径入口时可用 `repair_policy.json`。这组开发集结果没有显示“先整路径再恢复”优于直接逐步策略。正式使用前仍应查看完整 ID 评测，不能将 20 个开发样本的 100% 推广为总体成功率。

完整 ID 验证与消融现已完成，全部方法均使用相同 200 个网格、16 次调用上限和相同已训练的逐步模型：

- 只有合法动作过滤、关闭访问次数约束：170/200（85%）成功，平均调用 6.025 次。记录：`outputs/agent_harness/20261023_policy_no_revisit_id200/results.json`。
- 启用访问次数约束、直接逐步规划：199/200（99.5%）成功，平均调用 5.025 次，最优率 86.5%。相对于上述同预算消融，提高 14.5 个百分点。
- 先尝试两次整路径、再启用相同逐步恢复：198/200（99%）成功，平均调用 7.01 次，最优率 86.5%。最后两种结果在 `outputs/agent_harness/20261023_recovery_id200/results.json`。

直接逐步模式仍有一个样本在 16 次调用内未到目标，不能宣称完整成功或最优性保证。恢复模式多消耗两次失败提案额度，导致另一个样本也预算耗尽。访问次数约束在开发阶段确定，ID 验证中没有调整阈值或预算。这是现有权重上的执行流程改进，不是新的训练结果，也不替换旧冻结基线中的 `legal_mask_no_search`（该实现会排除所有已访问邻居，行为及原预算不同）。

```powershell
python scripts/run_harness.py run --spec configs/harness/policy_revisit.json --out outputs/agent_harness/policy_revisit_trial_001
python scripts/compare_harness.py --split test_id --limit 0 --calls 16 --methods repair_policy policy --out outputs/agent_harness/recovery_id_001
```

配对评测中的路径指标针对最终已提交路径；失败整路径提案不提交，因此空前缀也可能具有 100% 可行率。必须同时看成功率、`mean_invalid_calls` 和日志中的原始提案，不能把这个可行率解释为原模型输出合法。

## 与旧基线比较的限制

新运行方式会拒绝非法动作并允许再次提案，行为与旧“遇到非法动作即失败”的纯策略基线不同。不能把其成功率直接替换旧基线，也不能将执行器或动作约束贡献全算为模型能力。应以独立 trial、相同预算和固定样本做消融，报告成功率、提案失败数、模型/工具调用数、耗时、最终路径最优性及中断恢复结果。上述结果属于系统流程改进，没有重新训练原整路径模型。

旧 evaluation harness 同时修正：Markdown 报告被写成 JSON 字符串、分 split 耗时累计、成功率差值被其他差值遮盖、修复 trace 缺少真实输出，以及已有输出目录可能覆盖的问题。正式评测缺少官方 manifest 时现在直接报错；只有显式设置 `require_official=False` 的单元测试 fixture 可用本地执行器。旧跨方法共享评分缓存的计时方式保留，依然只作增量参考，不能比较冷启动性能。

runtime 合约测试覆盖工具边界、环境判定成功、预算、错误后恢复、旧快照、身份变化、日志截断和目录保护。初版共 47 项测试通过，恢复策略改进后共 52 项通过。执行命令：

```powershell
python -m unittest discover -s tests -v
```

静态语法检查与运行时测试均已完成；模型试验结果以上文日志为准，不将模型的越界提案计为成功。
