# Project 6：G／V／R LoRA 实施与证据状态

## 当前状态（2026-10-29）

已实现固定数据索引、独立训练入口、LoRA 路由、持久化整路径循环和 B1–B4／候选评测入口。用户报告已在另一 GPU 环境完成 G／V／R 正式训练、58 项测试通过、GPU 正常；ID／OOD 候选修复成功率为 12%／6%，四种端到端流程均为 0%，验证器全部预测 `valid`，生成和修复输出存在重复动作。这些是**用户提供的实验摘要**，不是本仓库可独立重算的结果：本地没有权重、GPU 账本和逐条轨迹。不得把它们与此前基线结果混称。

P0 核查：仓库 `main@b433523` 含 `data/derived` 原始任务数据及 manifest，训练 800／2,000／1,200、开发 40／100／60，ID 与 OOD 各 200 条。本机已按固定上游 revision 下载官方执行器，SHA256 `375462625e1c133d5b469037c03a18cbc33160a31b2ba88a0da89a127364a6e4` 与数据 manifest 一致；`vendor/` 按仓库规则不提交。仓库未包含本轮 `outputs/`、模型 checkpoint、`outputs/gpu_ledger.json` 或负责人确认记录。因此剩余 GPU 预算、权重哈希和用户报告的指标尚不能独立核实；默认配置仍拒绝在此环境启动新训练。

### 本轮逻辑审计

原评测入口错误地把开发集 100 条 Verify 与 60 条 Repair 一起算验证、把两类非法候选一起算修复，还用全部任务的网格算 Generate。已改为开发集分别使用固定 100／60／40 条；ID／OOD 测试仍使用 200 个候选、其中 100 个非法候选用于修复。控制器现在先处理刚返回的反馈再判断是否需要下一次模型调用，重复候选立即停止。新增训练前监督标签与单错误约束核对、每轮开发集预测分布、`all_valid_warning`、端到端 `false_valid_accepts`、停止原因与重复动作诊断。重复动作计数仅是诊断，不将合法连续动作错误判为无效。

这些修复能纠正指标口径和停止行为，**不能凭代码保证模型不再全判 `valid` 或端到端成功率提高**。自验证分支继续严格使用模型反馈；若验证器误判 `valid`，控制器按原研究协议停止，并在独立评分中记为失败。

## 文件与边界

| 能力 | 文件 | 约束 |
|---|---|---|
| 冻结数据 | `ppnl/data.py`、`data/derived/gvr_task_index.json` | 直接索引原记录和布局；不生成新训练例 |
| 训练 | `ppnl/gvr_training.py`、`ppnl/task_adapters.py` | G 从固定预训练版本训练；V／R 分别从 G 启动，`q/v` LoRA；各只训练本任务 |
| 控制器 | `ppnl/gvr_harness.py` | `direct`、`self`、`executor_one`、`executor_three`；模型请求先入账；身份不符拒绝恢复 |
| 评分 | `ppnl/gvr_evaluation.py` | 独立检查最终候选；固定候选 Verify／Repair 与端到端分开汇报 |
| 入口 | `scripts/run_gvr.py`、`config.gvr.json` | 新输出目录，不覆盖历史结果 |

训练依赖固定 `transformers==4.57.6` 与 `peft==0.18.1`。PEFT 的版本选择依据其 [发布说明](https://github.com/huggingface/peft/releases/tag/v0.18.1) 和 [多适配器 API](https://huggingface.co/docs/peft/v0.18.0.rc0/en/package_reference/peft_model)。在实际 Python 3.11／CUDA 环境中仍需做载入和保存重载验证。当前本机仅有 Python 3.14 和 Transformers 5.12.1，不能据此宣称目标环境的模型测试通过。

## 启动条件与命令

下一位运行者应从 GitHub 分支 `codex/project6-gvr-lora-harness` 克隆，在**仓库根目录**执行所有命令。`scripts/run_gvr.py` 按当前工作目录解析 `config.json`、`config.gvr.json` 及数据路径。默认 `--out` 路径与 `config.gvr.json` 的 checkpoint 路径配套；若改用 `--out`，须同步修改配置中的对应路径。

### 1. 无 GPU 的交接检查

```powershell
git clone --branch codex/project6-gvr-lora-harness https://github.com/Xiaofeipao/NLP-Project.git
cd NLP-Project
python scripts/fetch_official.py
python scripts/run_gvr.py index
python scripts/run_gvr.py preflight
python -m unittest discover -s tests -q
```

`index` 在已有索引匹配时只核查，不重写；不匹配时拒绝覆盖。`preflight` 返回 `checks`、已记录 GPU 秒数、`ready_for_training`、`ready_for_evaluation`。刚克隆的仓库预期两项 `ready` 都是 `false`，因为 `outputs/` 与确认记录不在 Git 中。运行测试无需 GPU；首次运行前需按 `environment.yml` 建立 Python 3.11 环境，或安装 CUDA 版 PyTorch 后执行 `pip install -r requirements.txt`。

若原 GPU 环境仍有 `candidate_traces.jsonl`、`pipeline_traces.jsonl`，可在仓库根目录**不加载模型、不消耗 GPU**地重算修正后的指标，输出到新文件，不覆盖原 `metrics.json`：

```powershell
python scripts/run_gvr.py recompute --split dev
python scripts/run_gvr.py recompute --split test_id
python scripts/run_gvr.py recompute --split test_ood
```

默认读取 `outputs/gvr/evaluation/<split>/`，也可用 `--evaluation-dir` 指向已保存的评测目录。输出 `metrics_recomputed.json` 记录数据和轨迹哈希；没有原始轨迹时不能重算，也不应据摘要倒推逐条结论。

### 2. 补齐训练前证据

从原实验机器取得**完整** `outputs/`，至少包括历史训练产物、模型 checkpoint、`outputs/gpu_ledger.json` 与对应配置快照。逐项核对数据／权重哈希、各训练阶段曝光次数及历史总 GPU 占用；确认共用账本仍低于 20 小时。取得负责人对 `google/flan-t5-small` 固定 revision 的确认记录后，在 `config.json` 中据实填写 `manager_confirmed` 和 `confirmation_note`，并在 `config.gvr.json` 中据实设置三个 P0 证据位。不要创建空账本或复制账本来重置预算。再次运行 `preflight`，要求 `ready_for_training: true`。若课程按累计不同样本计，当前索引复用原 4,000 条，不引入新样本。

### 3. 目标 GPU 环境训练与评测

目标环境需 Python 3.11、CUDA 版 PyTorch、`transformers==4.57.6`、`peft==0.18.1`。先完成每阶段 pilot 并查看其 `training.json`，再运行对应正式训练；G 冻结后才运行 V／R。命令：

```powershell
python scripts/run_gvr.py pilot --stage generation
python scripts/run_gvr.py train --stage generation
python scripts/run_gvr.py pilot --stage verification
python scripts/run_gvr.py train --stage verification
python scripts/run_gvr.py pilot --stage repair
python scripts/run_gvr.py train --stage repair
python scripts/run_gvr.py verify
python scripts/run_gvr.py evaluate --split dev
python scripts/run_gvr.py evaluate --split test_id
python scripts/run_gvr.py evaluate --split test_ood
```

预期产物：`outputs/gvr/{generation,verification,repair}/{pilot,train}/training.json`；G 权重在 `outputs/gvr/generation/train/best/`，V／R 权重分别在 `outputs/gvr/{verification,repair}/train/best/{verification,repair}/`。`verify` 检查保存重载、关闭适配器时 G 输出一致，并对固定开发集 40／100／60 条逐任务输出预测分布与成功率，保存至 `outputs/gvr/verify_dev.json`；它需要 GPU，也计入共用账本。评测指标在 `outputs/gvr/evaluation/<split>/metrics.json`，逐条轨迹在同目录 `candidate_traces.jsonl`、`pipeline_traces.jsonl`，案例在 `case_studies.json`。`preflight` 的 `ready_for_evaluation` 只有这些权重和配置齐备后才会为 `true`。

输出目录拒绝覆盖；中断后不要原路径重复启动训练。先保存失败日志并核算 ledger，再决定使用新输出目录和相应配置路径。新 G、V、R 各最多两轮；pilot 单独记账。

开发集用于模型选择；测试集是复用测试集。评测前核对 V／R 的 `base_identity.json` 与 G 权重哈希一致。评测保留模型原始反馈，格式错误计入失败；自验证分支不调用执行器做决策。所有模式从同一网格生成，评测器检查首个候选逐字一致。B4 相比 B3 使用更多调用，须连同模型调用、执行器调用和耗时一起呈报。

## 尚待完成的实际实验交付

训练后需要检查 LoRA 匹配模块数、可训练参数、底座冻结、保存重载与关闭适配器输出一致性；用正式 GPU 输出填入成功率、可行率、最优率、Verify Macro-F1、首错定位、Repair 成功率、按错误类型样本数与两个真实修复案例。课程报告与展示材料只有真实结果产生后才能完成。A 与 B 同时改变训练结构；若未做可选 C 全参数任务分支，不能把 A/B 差异单独归因于 LoRA。
