# PPNL 单目标路径规划 验证与修复

新增 Project 6 G／V／R 的生成底座、两套 LoRA、训练记录和逐条实验结果已收录至 [产物目录](artifacts/gvr/README.md)，权重需执行 `git lfs pull` 获取。ID／OOD 候选修复为 12%／6%，四种端到端流程均为 0%；开发集按修正配额重算后修复为 7/60（11.7%）。入口与审计说明见 [GVR 协议](docs/gvr_protocol.md)，完整结果见 [实验记录](docs/gvr_gpu_run.md)。下文原有数值为此前项目的历史实验记录。

依据工作区 `NLP Project.docx` 实现。项目包含数据构造、多任务微调、两个模型的开发集比较、直接生成、自验证修复、执行器反馈修复、迭代修复改进及移除迭代的消融实验。

逐步代码讲解与面试说明见 [Interview.md](Interview.md)，涵盖数据构建、训练、策略、搜索、Harness、评测、测试和结果口径。Git 仓库包含官方原始数据与项目加工数据，来源和结构见 [数据说明](data/README.md)；模型权重、运行输出与缓存不包含在仓库中。

**2026-10-18 已在 CUDA GPU 环境中完成两个模型的正式微调、开发集选择、相同 ID 测试，以及主模型的 ID/OOD 完整管线与消融；14 项回归测试通过。**

**后续规划改进也已完成：原相同 200 个测试环境上，逐步神经策略成功率 76%，加合法动作约束为 95%，神经引导搜索为 100%；纯 A* 对照也为 100%。当前共有 22 项回归测试通过。** 新模型不覆盖原基线，详见 [规划改进](docs/planning_improvement.md)，结果在 `outputs/planning_v2/results.json`。

**最新一轮已完成两个候选的开发集比较：选中的多最优动作模型在相同测试集上，无搜索成功率提高到 85%，动作约束为 97.5%，最优性保持搜索的成功率和最优率均为 100%；34 项测试通过。** 默认规划入口现在读取 `config.policy.best.json`。详见 [最新改进与消融](docs/planning_v3.md)，完整结果在 `outputs/planning_v3/results.json`。强纯 A* 对照也为 100%，且展开更少，不能将搜索成功全部归因于模型。

使用改进后的规划器：

```powershell
conda activate ppnl
python scripts/plan_grid.py --grid examples/grid.json
```

默认使用开发集选定的混合搜索方法。若只看不搜索的逐步模型，可加 `--method state_policy`；只加动作约束可用 `--method legal_mask_no_search`。

## Agent Harness 与统一评测

**2026-10-23：根据官方资料补充 Agent Harness，优化后全部 52 项测试通过；CUDA 正常。** 新入口 `scripts/run_harness.py` 管理模型—工具—执行器反馈循环、预算、日志及中断恢复；支持逐步策略、整路径反馈修复与组合恢复。完整 ID 200 个网格、同样 16 次调用上限下，直接逐步策略增加访问次数约束后，从同预算消融的 85% 提高到 99.5%；组合恢复为 99%。最优率为 86.5%，没有最优性保证。使用说明、来源、消融结果与限制见 [Harness 文档](docs/harness.md)。

原评测入口已改名为 `scripts/run_eval_harness.py`，用于对比规划器和冻结指标：

后续已加入失败两轮后切换逐步模型的恢复模式，以及避免重复访问的直接逐步配置。52 项测试全部通过；在相同 16 次调用上限的 20 个开发网格上，纯修复成功率 0%，恢复模式与直接逐步模式均为 100%，后者平均调用更少。新配置为 `configs/harness/repair_policy.json` 和 `configs/harness/policy_revisit.json`；这是流程与约束改进，原模型权重不变。配对验证入口为 `scripts/compare_harness.py`。

优先使用已验证的直接逐步模式：

```powershell
python scripts/run_harness.py run --spec configs/harness/policy_revisit.json --out outputs/agent_harness/my_policy_trial
```

```powershell
conda activate ppnl
python scripts/run_eval_harness.py --config config.policy.best.json
python scripts/run_eval_harness.py --symbolic-only --methods symbolic_astar_goal_tiebreak
```

- 覆盖 `original_whole_plan`、`state_policy`、`legal_mask_no_search`、`neural_best_first`、`symbolic_astar_no_model`、`neural_optimal_astar`、`symbolic_astar_goal_tiebreak`；`--methods` 可只跑子集。
- 复用 `ppnl.policy` 的规划器和 `ppnl.evaluation.plan_metrics`，不另写指标；`repair_prefix` 由回归测试保证与 `scripts/evaluate_policy.py` 逐位一致。
- 只写入 `outputs/harness/<时间戳>/`，拒绝写进基线实验目录，冻结的 `eval_*` 产物不会被覆盖。
- 沿用既有护栏：权重 SHA256 与 `policy_selection.json` 不符时默认报错（`--allow-new-weights` 才放行），官方执行器 SHA256 必须与 manifest 一致，搜索预算耗尽返回失败而不使用隐藏最短路兜底。
- 每次运行产出 `results.json`、`report.md`、`report.html` 和 `detail/` 下逐条 trace；`--symbolic-only` 不需要 GPU，也不写 GPU ledger。

2026-10-22 实测：dev/test_id/test_ood 三个 split 上七个方法全部与冻结指标 0 个字段不符；test_id 的 `state_policy` 85.0%、`legal_mask_no_search` 97.5%、`neural_optimal_astar` 100% 均复现。OOD 规划耗时明显偏低，因为三个 split 共用同一份状态评分缓存，耗时只作增量参考。

## 文件入口

- `config.json`：数据上限、checkpoint、训练预算和修复轮数。
- `ppnl/core.py`：单目标网格、BFS、首次失败定位和官方执行器适配。
- `ppnl/data.py`：导入官方环境，按障碍物布局划分，构造合法路径和受控错误。
- `ppnl/modeling.py`：PyTorch 微调、开发集 loss 选最佳 epoch、pilot、GPU 预算记录。
- `ppnl/evaluation.py`：验证、修复、端到端比较、改进和消融、逐步轨迹。
- `ppnl/harness.py`：Agent Harness，工具白名单、环境状态、预算与持久化恢复。
- `ppnl/harness_backends.py`：逐步策略和整路径修复的模型适配器。
- `ppnl/eval_harness.py`：评测方法注册表、冻结指标对比、Markdown/HTML 报告。
- `scripts/run_harness.py`：Agent Harness 的 run/resume/inspect 入口。
- `scripts/run_eval_harness.py`：原统一评测入口。
- `ppnl/cli.py`：命令行入口。
- `scripts/fetch_official.py`：按固定版本下载官方单目标数据和执行器。
- `tests/`：执行器、指标和完整修复管线回归测试。
- `docs/environment.md`：通用环境配置与运行验证说明。
- `docs/experiment_protocol.md`：指标定义、实验安排和报告材料说明。

命令均在本目录 PowerShell 中执行。模型命令要求 CUDA；数据处理和测试无需 GPU。

## 环境

建议使用独立的 Python 3.11 环境，并根据目标 GPU 安装兼容的 CUDA 版 PyTorch。其余 NLP 依赖由 `requirements.txt` 管理，环境名 `ppnl` 仅为示例：

```powershell
conda create -n ppnl python=3.11
conda activate ppnl
# 先按目标平台安装兼容的 CUDA 版 PyTorch，再安装项目依赖
python -m pip install -r requirements.txt
python -m ppnl.cli environment
```

也可使用 `environment.yml` 作为环境模板，并根据目标硬件调整 PyTorch 构建。训练代码使用原生 PyTorch，不需要 datasets、accelerate 或 Gurobi。CPU 可用于数据处理与逻辑测试，模型训练和推理要求 CUDA。环境配置和验证步骤见 [环境说明](docs/environment.md)。

## 已完成的小规模 GPU 验证

初期 Flan-T5-small 和 BART-base 仅用于技术试跑，该阶段没有保存正式训练 checkpoint 或选择主模型。每个模型对三种任务各 8 条训练样本更新 3 步，再用 4 个开发集候选检验完整评估流程；没有使用测试集调参。

原始结果在 `outputs/smoke_20261018/<模型>/smoke/smoke.json`，逐步执行记录在同级 `dev_pipeline/`。Flan-T5-small 的固定样本平均 loss 从 3.6186 到 2.1980，BART 从 6.4922 到 2.2328。训练/推理/微基准中的峰值 allocated 显存分别约 1.64 GiB、2.68 GiB。这些数值仅用于确认训练链路正常，不能作为正式效果结论；仅 3 次更新后模型仍存在输出格式错误和路径失败。

加入一次性批量分词、编码缓存、8 对齐 padding、锁页内存和异步 CPU→GPU 传输。微基准在相同驻留模型和 batch 上比较 FP32 与 BF16，1 次预热后计时 3 次前后向，不更新参数。Flan-T5-small 分别约 64.6/120.2 ms，BART 约 35.7/37.8 ms。因此默认保持 FP32（允许 high matmul precision），不假设 BF16 必然加速。该微基准样本很少，只适用于当前机器和形状。

复现试跑时，可修改 `config.smoke.json` 的 `output_root` 为新目录，再运行：

```powershell
python scripts/fetch_models.py --config config.smoke.json
python -m ppnl.cli --config config.smoke.json smoke --model flan_t5_small
python -m ppnl.cli --config config.smoke.json smoke --model bart_base
```

`config.smoke.json` 保留本次使用的 BF16 设置以便复现；每次试跑同时测量 FP32/BF16。它使用本地模型缓存、只允许小规模更新，不绕过正式 `pilot`/`train` 的 checkpoint 确认要求。更换输出目录不会清空共用的 GPU ledger。

## 正式实验已完成

正式基线训练采用当前两个 checkpoint，并通过以下脚本执行： `scripts/run_study.py`。两个模型均从原始预训练权重开始，以相同的 4,000 条多任务样本训练 2 轮；每个模型处理 8,000 次样本、完成 500 次 AdamW 更新。FP32、batch 8、梯度累积 2、学习率 1e-4、seed 42 保持一致。正式训练前的 pilot 和更早的技术 smoke 单独记入预算，没有复用其权重。

- Flan-T5-small：两轮开发集 loss 为 0.4927、0.4655；ID 验证 Macro-F1 0.7141，首错定位准确率 8%，候选修复成功率 19%，修复可行性 46%，修复最优率 19%。最佳权重在 `outputs/flan_t5_small/train/best/`。
- BART-base：两轮开发集 loss 为 0.5033、0.4332；ID 验证 Macro-F1 0.7246，首错定位准确率 45%，候选修复成功率 10%，修复可行性 34%，修复最优率 10%。最佳权重在 `outputs/bart_base/train/best/`。
- 两个模型在相同 200 个 ID 环境上，直接生成成功率均为 0%。这是实际结果，表明当前固定预算下的多任务微调仍未学好从零规划；Flan-T5-small 的生成路径集中于重复模板。不能仅根据 loss 下降宣称规划成功。
- 按开发集预先确定的综合分数选择 Flan-T5-small 为主模型（0.2897 vs 0.2374），没有依据测试集改选。
- 主模型 OOD 双错误测试：验证 Macro-F1 0.7700，首错定位 20%，一次候选修复成功率 7%，最多三轮修复 11%；ID 一次与三轮均为 19%。从直接生成开始的各端到端管线仍为 0%，不能把候选修复的改进混称为端到端改进。

全流程墙钟约 419 秒；共用 ledger 累计 GPU 占用墙钟约 402 秒（含先前 smoke，约 0.112 GPU 小时），不是显卡处于满负载计算的时间。没有根据测试成绩继续训练或调参。

正式评估统一使用贪心解码，允许重复动作。BART 原始摘要配置带 `no_repeat_ngram_size=3`，已在正式任务评估时显式覆盖为 0，并同步到交付 checkpoint 的 generation_config；回归测试覆盖这一点。训练过程中的 `dev_exact_match` 只作诊断，不用于模型选择；BART 该诊断值生成于覆盖此上游参数之前，正式比较以 `eval_dev` 和 `eval_test_id` 的任务指标为准。

完整机器可读结果在 `outputs/study_results.json`，含两个模型的指标、实际 checkpoint 版本、权重 SHA256、训练配置、主模型完整实验和预算。`outputs/study_status.json` 记录 11 个成功阶段；`outputs/logs/` 保留原始日志。`outputs/config_snapshot.json` 保存本次正式配置，`outputs/source_manifest.json` 是启动时的源码哈希，`outputs/source_manifest_final.json` 是修正解码及记录来源版本后的最终源码哈希。修正没有改变已完成的梯度计算或重训权重。

重现整个流程可在配置中换一个新的 output_root 后运行：

```powershell
python scripts/run_study.py
python scripts/summarize_study.py
```

现有输出目录下，runner 会复用已经成功的阶段；修改 checkpoint、数据或训练设置时必须使用新的 output_root，不应复用旧结果。

## 训练前确认 checkpoint

文档的基线名称含 Flan-T5，但所给 T5 链接是 `google-t5/t5-small` / `google-t5/t5-base`。二者不能混用。

正式 checkpoint 为 `google/flan-t5-small` 和 `facebook/bart-base`，均固定到指定 commit。配置中通过 `user_confirmed: true` 与 `confirmation_note` 记录实验 checkpoint 的确认状态；`manager_confirmed` 保留独立含义。对于另选 checkpoint，可通过 `manager_confirmed` 或明确的 `user_confirmed` 与 confirmation_note 记录授权；二者皆无时 pilot 和训练会报错。

pilot 会记录远端解析得到的 commit，正式训练沿用该 commit。未来如更换配置、数据或 checkpoint，应使用新的输出目录重做 pilot。不要重复训练同一输出目录或覆盖已经冻结的模型选择。

## 正式实验的执行顺序

### 1 下载官方数据

```powershell
python scripts/fetch_official.py
```

只下载官方单目标 6×6 文件与执行器，不下载多目标工具。上游固定版本为 `5fd0376fc50ae9860bc986085165ebac5675fd44`。下载清单和 SHA256 保存到 `vendor/ppnl/download_manifest.json`。重复运行会校验并复用已有文件。

### 2 CPU 构造与核查

```powershell
python -m unittest discover -s tests -v
python -m ppnl.cli prepare --official-data data/official --executor vendor/ppnl/executor-point-sg.py
python -m ppnl.cli audit
```

构造 `data/derived/` 后可执行 audit；prepare 拒绝覆盖已有数据。要重新构造，请在配置中使用新的 dataset_root。

训练共 4,000 条任务样本：生成 800、验证 2,000、修复 1,200。开发集共 200 条：40、100、60。ID 和 OOD 测试集各 200 条。上限按三种任务样本总数计，没有给每种任务各分配 4,000 条。

同一障碍物布局上的所有起终点组合进入同一个 split。ID/OOD 共用同一组未见测试起终点，便于隔离错误数量的影响。每个测试集包含 100 个合法控制样本和 100 个非法样本；ID 非法样本只有一次错误注入，OOD 非法样本含两次独立位置的错误注入。合法控制样本包含最优和可行非最优路径。

### 3 两个模型各运行一次 pilot

```powershell
python -m ppnl.cli pilot --model flan_t5_small
python -m ppnl.cli pilot --model bart_base
```

检查各自 `outputs/<模型>/pilot/pilot.json`：checkpoint、输入/目标最大长度、显存峰值、吞吐、GPU 型号和版本。发现过长输入直接失败，不静默截断网格描述。默认 float32，优先可复现；如果显存不足或时间预算不合适，先统一调整两个模型的配置并重新安排实验。

pilot 仅进行一次部分数据遍历；正式训练最多两轮，总计不超过三次遍历。pilot 不保存用于正式评估的模型，也不复用 pilot 权重开始训练。pilot 属于 GPU 预算。本次技术 smoke 另有记录，不作为正式基线；正式提交时应披露这些额外技术验证更新。

### 4 匹配数据和训练预算的基线

```powershell
python -m ppnl.cli train --model flan_t5_small
python -m ppnl.cli evaluate --model flan_t5_small --split dev
python -m ppnl.cli train --model bart_base
python -m ppnl.cli evaluate --model bart_base --split dev
python -m ppnl.cli select
```

每个模型只训练一个共享生成/验证/修复的多任务 checkpoint。各模型以开发集 token 平均 loss 选择 epoch，再用相同开发环境上的生成成功率、验证 Macro-F1 和修复成功率的均值选择主模型。测试集不参与选择。

先对两个模型运行相同 ID 测试基线：

```powershell
python -m ppnl.cli evaluate --model flan_t5_small --split test_id
python -m ppnl.cli evaluate --model bart_base --split test_id
```

### 5 主模型完整实验

下面 `$primary` 从开发集选择文件读取，不按测试集成绩手动选：

```powershell
$primary = (Get-Content outputs/selection.json -Raw | ConvertFrom-Json).primary
python -m ppnl.cli evaluate --model $primary --split test_id --full
python -m ppnl.cli evaluate --model $primary --split test_ood --full
```

完整实验包含直接生成、一次自验证修复、一次执行器反馈修复、最多三轮执行器反馈修复改进，以及只保留第一轮的消融。主模型对预设 ID/OOD 候选路径的修复也执行同样的迭代与消融，不能只看自然生成路径上的改进。

## 输出

- `data/derived/manifest.json`：布局划分、来源 ID、文件哈希、构造规则、各任务计数、错误类型计数、CPU 构造时间。
- `outputs/gpu_ledger.json`：pilot、训练、评估的 GPU 占用墙钟时间，累计上限 20 小时，在 batch 边界检查停止。单个长 batch 或保存操作可能略超边界。
- `outputs/selection.json`：冻结的开发集分数和主模型。
- `outputs/<模型>/eval_*/metrics.json`：指标及定义。
- `candidate_traces.jsonl`：候选路径的金标、模型诊断、单轮修复、迭代修复轨迹。
- `pipeline_traces.jsonl`：直接生成、自反馈和执行器反馈的实际逐步记录。
- `case_studies.json`：从真实输出中选择的失败或修复案例，供撰写至少两个案例分析。

没有运行的阶段不会产生指标文件。报告应依据实际输出填写，不把代码默认配置当作实验结论。

## 官方兼容性说明

官方 `is_goal` 被用作合法动作序列的成功/碰撞真值核对。适配器只加载该函数，避免上游脚本在 import 时执行命令行，也避免引入其 pandas 依赖。官方无法区分越界和未到达目标，因此由本地逐步执行器提供这两种失败类型和失败步。

官方会忽略未知动作，并尝试拆开粘连动作；本项目为了满足文档的 `invalid_action_token` 要求，采用严格的空白分隔四动作语法。未知 token、粘连 token 和带标点动作不被悄悄修正。这是明确记录的格式扩展。所有合法动作候选和模型输出都与官方 `is_goal` 核对；发现语义不一致时中止，不发布指标。

项目最优性采用 BFS 的动作数量，而不是上游脚本中去空格后的字符数。可行性只要求执行期间无语法、越界、碰撞错误；未到目标可以可行但不成功。详细定义见实验协议。

官方资料：[PPNL 仓库](https://github.com/MohamedAghzal/llms-as-path-planners/tree/5fd0376fc50ae9860bc986085165ebac5675fd44/ppnl-spatial-temporal-reasoning)、[Transformers 4.57 文档](https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/trainer)。
