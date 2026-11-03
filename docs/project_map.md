# 项目方法、代码与分支导航

## main 是什么

`main` 是统一维护的完整代码版本：包含数据、全量微调基线、逐步策略、GVR LoRA、Harness、前缀修复和独立 RLVR 试验，以及测试和使用说明。它不是“只有 baseline”的分支，也不是某一个模型。稳定指本次 CPU 回归检查通过、入口和数据约束明确；不代表每种方法都训练过、成功率高，或 GPU 端到端已重新验证。

选择方法靠入口和配置，选择历史版本靠 Git 标签。不要把“代码在同一分支”理解为“一个实验同时使用所有方法”。原有模型与结果不会因整理代码自动更新。

## 方法分类

- **全量微调多任务基线**：Flan-T5-small / BART-base，用同一模型完成生成、验证、修复。入口 `scripts/run_study.py`、`ppnl/cli.py`；模型训练在 `ppnl/modeling.py`。已有历史运行。google-t5/t5-small 和 t5-base 不能算作已经完成的模型。
- **全量微调 + GVR Harness**：`ppnl/full_finetune_gvr.py` 适配原多任务 checkpoint，`scripts/run_full_finetune_gvr.py` 管理单网格试验；共用 `ppnl/gvr_harness.py`。仅增加推理控制，没有新微调或 LoRA，不把历史 baseline 分数冒充新 Harness 分数。
- **GVR LoRA + Harness**：G 使用 800 条生成样本全参数训练；V/R 分别在冻结 G 上训练独立 q/v LoRA。`ppnl/gvr_training.py`、`task_adapters.py`、`gvr_harness.py` 与 `scripts/run_gvr.py`；`config.gvr.json` / `config.gvr.next.json`。已有历史试验，端到端成功率仍为 0，不能引用逐步策略的 99.5%。
- **逐步策略 + Agent Harness**：`ppnl/policy.py`、`harness.py`、`harness_backends.py`；`scripts/run_harness.py` 与 `configs/harness/`。策略模型选择下一步，执行器检查动作；可加合法动作和访问次数约束。它不是 GVR LoRA。历史 99.5% 来自这一类。
- **前缀修复**：保留生成路径中已执行的合法前缀，再由另一个已训练的逐步策略补全。`ppnl/full_prefix_repair.py` 读取原全量多任务模型的 trace；`ppnl/prefix_state_repair.py` 读取 GVR trace 并提供消融。GVR 生成时关闭 V/R LoRA，因此“LoRA 前缀分支”指 GVR 实验来源，不表示补全动作由 LoRA 输出。两类统一使用 `ppnl/prefix_evaluation.py` / `scripts/run_prefix_experiment.py`，配置在 `configs/prefix/`。
- **RLVR REINFORCE + LoRA**：`ppnl/rlvr.py`、`scripts/run_rlvr.py`、`scripts/verify_rlvr.py` 与 `configs/rlvr.json`。奖励来自执行器，没有人工偏好；每组采样一次 on-policy 更新、leave-one-out baseline、采样 KL。不是完整 GRPO，也没有 PPO 裁剪。历史开发集成功率 0/40，尚无测试集成绩。
- **结构化输入消融**：`ppnl/prompts.py` 与 `configs/gvr_format_vanilla.json` / `configs/gvr_format_structured.json`。改变输入表达，不增加模型推理步骤；仅完成长度预览和代码测试，尚未训练结构化模型。不是 CoT。
- **未实现方向**：完整 GRPO、坐标跟踪 CoT、技能图独立消融。HotpotQA 不加入主实验。

Harness 是运行控制层；LoRA 是参数训练方式；REINFORCE 是强化学习算法。这三个概念不互相替代。数据和评测公用 `core.py`、`data.py`、`evaluation.py`；冻结公用 `experiment_freeze.py`，而不是为每个分支复制一套评分规则。

## 前缀评估的规范入口

先检查 `configs/prefix/lora_gvr.json` 或 `configs/prefix/full_finetune.json` 中实际存在的 trace 路径。两种文件格式不同，不能混用。神经策略必须有匹配权重的 `policy_selection.json`；不再允许跳过选模检查。

```powershell
python scripts/run_prefix_experiment.py check --spec configs/prefix/lora_gvr.json
python scripts/run_prefix_experiment.py evaluate --spec configs/prefix/lora_gvr.json --split dev --out outputs/prefix_review/dev
python scripts/run_prefix_experiment.py freeze --spec configs/prefix/lora_gvr.json --dev-report outputs/prefix_review/dev/results.json --out outputs/prefix_review/freeze.json
python scripts/run_prefix_experiment.py evaluate --spec configs/prefix/lora_gvr.json --split test_id --freeze outputs/prefix_review/freeze.json --out outputs/prefix_review/test_id
```

`check` 和 `freeze` 不做模型推理；`evaluate` 神经模式需要 GPU。符号对照在 GVR spec 设置 `symbolic_only: true`，同样先 dev、再 freeze、后 test。冻结绑定配置、数据划分、三个 trace 文件的字节哈希、全部模型/tokenizer 文件、选模文件、源码及执行器；计算测试 trace 哈希不读取其指标来选模。测试入口在 CUDA 前检查冻结，结果附锁哈希。旧脚本 `run_prefix_state_repair.py` / `run_main_prefix_state_repair.py` 转发到同一 CLI，参数已统一为 `--spec`。

历史全量基线 dev trace 来自验证候选环境去重；GVR dev 来自 40 条 generation 样本。二者不应直接比较 dev 成功率，正式配对比较要保证同一环境集合。ID/OOD 仍是重复使用的探索性测试，冻结不恢复盲测属性。

## 本次修复的行为

1. `restart_visit` 无条件从原起点重新规划，包含原候选已经合法、前缀已经到终点、原前缀超预算的情况；`kept_prefix` 为实际保留内容，另存 `observed_prefix`。
2. 所有修复输出受总动作预算约束；超长前缀截断并逐例记录，不再中断整批。原始 direct 作为历史基线保留原输出，不宣称它使用了相同动作上限。
3. 两类前缀测试统一强制冻结，开发证据不能由测试报告代替。
4. RLVR 在 CUDA 前核对 checkpoint 输入格式，训练、开发评估与重载均使用同一模板；重载还检查原底座身份。旧 vanilla 默认行为保留。
5. 策略记录请求状态数、已有缓存命中数、真实 forward 批次数和新评分状态数；决策次数不再被解释为真实 GPU forward 次数。

这次只修代码、测试和组织方式，没有新 GPU 训练、没有新成功率，旧冻结因源码变化需要重新生成，历史报告保持原样。

## 分支命名与旧版本

当前发布采用一个共享实现。以下工作分支在整理完成时从同一测试通过的提交建立，初始代码相同；名称指定后续开发负责的方向，不能用于当前消融比较。后续方法变化在对应分支开发、测试后再合入 main。

- `main`：完整项目的默认集成版本。
- `codex/full-finetune-harness`：维护全量微调模型的 Harness 适配。
- `codex/lora-gvr-harness`：维护 GVR 独立 LoRA 与整路径 Harness。
- `codex/lora-prefix-repair`：维护 GVR trace 的前缀修复和消融。
- `codex/full-finetune-prefix-repair`：维护原全量微调 trace 的前缀修复。

不创建 GRPO 分支，因为尚未实现完整 GRPO；RLVR 的独立配置与输出目录已能隔离当前试验。

旧名称被替换前，提交保存为以下标签。标签保留旧代码和旧缺陷，只供复核历史结果，不作为推荐运行入口。

- `archive/20261103/original-main`：`b433523`，原 main。
- `archive/20261103/gvr-structured`：`800092c`，原 `codex/project6-gvr-lora-harness`。
- `archive/20261103/lora-prefix`：`1fae824`，原 `codex/prefix-state-repair`、`lora+harness`、`lora+prefix` 三个同提交名称。
- `archive/20261103/full-harness`：`f0abf7b`，原 `codex/full-finetune-gvr-harness` 和误命名的 `prefix-state-repair`。
- `archive/20261103/full-prefix`：`fc6e181`，原 `codex/main-prefix-state-repair`。

例如 `git show archive/20261103/lora-prefix:ppnl/prefix_state_repair.py` 可查看旧实现，无需保留重复分支。标签不包含忽略的本地新权重；权重仍留在本地，已有公开 LFS 产物不变。
