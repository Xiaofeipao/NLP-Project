# NLP Project：代码逐步讲解与面试说明

本文按程序的执行顺序说明项目：从原始网格数据，到训练模型，再到规划、验证、修复和 Harness 运行。说明依据当前代码及已保存实验结果整理；本文编写不触发重新训练或评测。

## 1. 项目解决什么问题

输入是一张网格地图，包括地图行列数、障碍物坐标、起点和终点。某些任务还提供一条待检查的路径。允许动作只有 `up down left right`。

项目有三个基础任务：

- **生成 generation**：给地图、起点、终点，让模型输出路径，例如 `left left down left left`。
- **验证 verification**：给地图和候选路径，让模型判断是否正确；失败时给出第一次错误的位置和原因，例如 `invalid | step=4 | reason=out_of_bounds`。
- **修复 repair**：给地图、错误路径和反馈，让模型输出修正后的路径。

后面的逐步策略和 Harness 都围绕这三个任务发展。项目包含三层：原始整路径模型、后续训练的逐步策略，以及管理调用与环境状态的 Harness。模型评分、程序约束和显式搜索的贡献必须分别说明。

## 2. 目录与文件职责

- `ppnl/`：核心代码，包含数据、模型、规划、评测和 Harness。
- `scripts/`：命令行入口和实验调度脚本。
- `configs/harness/`：不同 Harness 运行模式的配置。
- `tests/`：自动化测试。
- `data/`：官方原始数据与项目加工数据，已纳入 Git；来源和结构见 `data/README.md`。
- `vendor/`：下载的官方执行器，不纳入 Git，可通过脚本获取。
- `outputs/`：训练权重、指标、日志、逐条轨迹，不纳入 Git。
- `docs/`：实验设计、环境和改进说明。
- `examples/`：单张地图示例。
- `.cache/`：模型缓存，不纳入 Git。

代码、配置、权重、实验结果是四种不同东西。改配置不等于改模型参数；加入 Harness 约束也不等于重新训练模型。

## 3. 环境规则：`ppnl/core.py`

所有规划方法最终服从这个文件的规则。

### 3.1 `ACTIONS`：动作与坐标

```python
ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
```

坐标采用从 0 开始的 `(row, column)`。例如 `(4, 5)` 执行 `left` 到 `(4, 4)`，执行 `up` 到 `(3, 5)`。模型输出文字，执行器按此映射进行真实坐标更新。

### 3.2 `Grid`：统一地图对象

保存 `rows`、`cols`、`obstacles`、`start`、`goal`。`__post_init__()` 检查行列数为正、坐标不越界、起终点不同且不在障碍物上。`frozen=True` 限制创建后直接修改字段。

- `inside()`：检查坐标是否在地图内部。
- `layout_id`：根据尺寸和障碍物生成标识，不包含起终点。同一障碍布局的不同起终点仍属于同一布局，用于防止数据划分泄漏。
- `world()`：转成矩阵，0 为空地、1 为障碍、2 为起点、3 为终点。
- `from_world()`：将官方矩阵转成 `Grid`，检查矩形格式、单起点和单终点。
- `from_dict()`：把 JSON 字典转为 `Grid`。
- `description()`：将结构化地图转成自然语言供整路径模型使用。

### 3.3 `Verdict`：执行结论

字段包括 `valid`、`feasible`、`step`、`reason`、`trace`。

- `feasible=True`：动作合法，没有越界或碰撞。
- `valid=True`：完整路径执行结束后，确实停在目标。

合法移动但尚未到目标时，可以同时有 `feasible=True` 和 `valid=False`。`label()` 把结论写成统一文字格式。

### 3.4 `execute()`：确定性执行路径

逐个检查动作词、下一位置边界和障碍物，最后检查是否到目标。四类错误为 `invalid_action_token`、`out_of_bounds`、`obstacle_collision`、`goal_not_reached`。

错误步数从 1 开始。如果执行了 3 个合法动作但没有到目标，结束位置记作第 4 步失败，即 `动作数 + 1`。必须检查完整路径：中途经过目标，后来又离开，最终不算成功。

`trace` 逐步记录出发位置、动作、目标位置和错误原因，支持反馈、恢复和案例分析。

### 3.5 `shortest_path()`：BFS 最短路

这是确定性广度优先搜索，不是神经网络。用于构造标签、判断可达性和事后评测最优性。最新无搜索逐步 Harness 不会调用它来偷偷代替模型规划。

### 3.6 `parse_verdict()`：严格解析模型回答

只接受 `valid` 或规定格式的错误标签。不符合格式的长篇回答不会被猜测解释为正确判断。

### 3.7 `OfficialOracle`：核对官方规则

这里只提取官方脚本的 `is_goal` 函数，避免导入整个脚本触发命令行。对合法动作路径核对本地执行器与官方函数，一旦不一致就报错。

官方函数会忽略某些未知动作；项目显式增加严格动作词校验，因此 `jump` 被判为非法。这是项目规则扩展。`jsonable()` 是把数据类转换为可序列化字典的辅助函数。

## 4. 下载脚本

### 4.1 `scripts/fetch_official.py`

下载固定 commit 对应的官方 6×6 单目标数据和执行器，并记录 SHA256。已有文件哈希正确则复用；与固定版本不同则拒绝覆盖。固定版本和哈希用于实验复现。

### 4.2 `scripts/fetch_models.py`

读取配置中的 checkpoint 和 revision，通过 `snapshot_download()` 获取模型参数、tokenizer、词表和配置。只下载，不微调，不要求 CUDA。

## 5. 数据构建：`ppnl/data.py`

### 5.1 读写函数

`read_jsonl()` 读取一行一个样本的数据；`write_json()` 写单个 JSON；`write_jsonl()` 写逐行样本。

### 5.2 `load_official()`

读取指定官方 6×6 单目标文件，转换 `Grid`，按布局、起点和终点去重，使用 BFS 检查可达性，保留符合要求的地图，并记录原始文件与记录索引。

### 5.3 `corrupt()`

从正确路径制造替换、删除、插入、非法动作和提前停止等错误，同时记录修改位置、修改前后内容。例如将 `right right down` 的第二个动作改成 `left`。

### 5.4 `edit_distance()`

计算动作 token 的编辑距离。修改两次不一定导致最终序列有两次编辑距离，尤其是重复动作可能使操作抵消，因此还要检查最终结果。

### 5.5 `nonoptimal()`

在正确路径前加入一次合法可逆绕行，如 `up down`，使路径仍能成功但比最短路多两步。验证任务应判断环境中的真实有效性，不能只比较输出是否等于参考最短路径。

### 5.6 `make_candidate()`

正样本生成后交执行器验证。负样本先制造错误，再检查编辑距离和真实执行结果；碰巧仍然有效则重试。标签来自实际执行，不靠修改操作猜测。

### 5.7 `build_dataset()`

按障碍物布局划分训练、开发和测试，比例大致为 80%/10%/10%。原始正式配置为：

- 训练集：生成 800、验证 2000、修复 1200，合计 4000。
- 开发集：生成 40、验证 100、修复 60，合计 200。
- ID、OOD 测试集：各 200 条。

`row_for()` 是内部样本构造函数，组织地图、来源、候选路径、执行标签、错误记录和目标路径。`manifest.json` 保存来源、布局划分、数量、执行器哈希及构建规则。

### 5.8 ID 与 OOD

两个测试集使用配对地图，主要差别是候选路径错误：ID 主要一次修改，OOD 两处独立修改并检查编辑距离为 2；两者均有正确候选对照。因此这里的 OOD 主要是错误组合变复杂，不是地图尺寸变化。不依赖候选路径的生成任务在两个集合上可以得到相同分数。

### 5.9 `audit_dataset()`

检查训练/开发/测试布局不交叉、数量不超限、样本 ID 不重复、布局标识与地图一致。它做数据审计，不训练模型。

## 6. 提示词：`ppnl/prompts.py`

`prompt(task, grid, candidate="", feedback="")` 根据任务组织文字：

- generation：地图描述和只输出动作的要求。
- verification：再加候选路径和判断输出格式。
- repair：再加错误反馈，要求返回修正后的完整路径。

训练和推理复用模板，减少输入形式差异。这个文件不判断对错，只组织输入。

## 7. 配置文件

`config.json` 控制原始正式实验：

- `seed`：数据采样和训练随机种子。
- `dataset_root`、`output_root`：数据和输出位置。
- `models`：模型及固定 revision。原始基线为 `google/flan-t5-small` 和 `facebook/bart-base`。
- `gpu_ledger`、`gpu_hours_cap`：时间账本与额度。
- `repair_rounds`：评测中的修复轮数。

正式训练主要参数：2 epochs、batch size 8、梯度累积 2、学习率 0.0001、weight decay 0.01、输入上限 512 token、输出上限 128 token、FP32。通常有效批量为 8×2=16，最后不足的窗口单独处理。

其他文件用途：

- `config.smoke.json`：技术试跑、独立输出目录、BF16。
- `config.policy.json`：早期逐步策略训练。
- `config.policy.v3.single.json`：单动作损失对照。
- `config.policy.v3.multi.json`：多最优动作集合损失。
- `config.policy.v3.v2_reference.json`：复用旧策略权重作参考。
- `config.policy.best.json`：开发集选定的最新策略，默认部署方法为 `neural_optimal_astar`。

默认 A* 部署入口与最新 Harness 的直接逐步模式是不同算法。

## 8. 模型与训练：`ppnl/modeling.py`

### 8.1 `require_cuda()`、`gpu_budget()`、`confirmation()`

`require_cuda()` 检查 GPU 可用性，不能用时明确报错。

`gpu_budget()` 读取时间账本、计算余额、提供检查回调，并在成功或失败结束时记账。记录的是操作占用墙钟时间，不是 GPU 指令的精确计算时间；按串行运行设计，没有跨进程并发额度预留。

`confirmation()` 核对 checkpoint 已得到用户或管理方确认且有说明，只负责实验身份检查。

### 8.2 `Seq2Seq` 初始化

统一封装 Flan-T5/BART：设置种子、精度，加载 tokenizer 和权重，移动到 CUDA，设置生成规则、批处理整理器及长度限制。动作分类模式还检查四个动作是否各对应单 token。

### 8.3 `check_lengths()`

输入或目标超过上限直接报错，不静默截断，避免地图、目标或路径尾部丢失。

### 8.4 `encode_rows()`

在 CPU 上把 `input` 和 `target` 转成 token ID，并复用编码。动作集合训练附加 `policy_actions`，表示四个动作中哪些可接受。

### 8.5 `collate()` 与 `autocast()`

`collate()` 将不同长度样本补齐、整理标签及动作掩码，组成一个批次。`autocast()` 根据配置选择 BF16 混合精度或普通 FP32 上下文。

### 8.6 `loss()`

普通整路径模型使用文本生成损失。新版逐步策略使用动作分类/动作集合损失，并对混合的验证样本保留文本损失。

### 8.7 `predict()`

批量编码、检查长度、关闭梯度、确定性贪心生成，再解码文字。不采样、不使用多 beam。显式设置 `no_repeat_ngram_size=0` 和 `repetition_penalty=1.0`，因为 `right right right` 可以是合法路径，不能沿用普通文本的禁重复规则。无意义循环由后面的执行约束处理。

### 8.8 `train()`

顺序是：核对身份/CUDA/输出目录，读数据和 pilot 记录，加载模型，建立 DataLoader 和 AdamW，前向算损失，反向传播，梯度累积与裁剪，更新参数，每轮评估开发集 loss，保存 loss 最低的 checkpoint，再重载最佳 checkpoint 做生成诊断并写训练报告。

```python
loss.backward()   # 产生梯度
optimizer.step()  # 更新模型参数
```

不足完整累积窗口的最后一组批次单独归一化。训练日志保存 epoch、批次、更新次数、loss、耗时、显存、精度、模型版本及数据指纹。

### 8.9 `dataset_fingerprint()`、pilot 和 smoke

`dataset_fingerprint()` 根据 manifest、训练和开发数据形成指纹。

pilot 是小规模训练流程与资源检查，正式训练核对其配置、数据及模型版本。`smoke()` 测试小批次更新、loss、梯度、显存、生成/验证/修复链路及 FP32/BF16 微型耗时，不保存正式训练 checkpoint，分数不作为正式基线。正式训练从原预训练权重开始，不接着 smoke 的临时参数训练。

## 9. 原始模型评测：`ppnl/evaluation.py`

### 9.1 `plan_metrics()`

- `success_rate`：成功到目标的比例。
- `feasibility`：无非法动作、越界和碰撞的比例。
- `optimality`：成功且等于最短长度的比例，分母为全部样本。
- `path_efficiency_on_success`：成功样本中“最短长度/实际长度”的平均。

最短路 5 步、实际 7 步且成功：成功为真，最优为假，效率为 5/7。BFS 仅在这里事后评分。

### 9.2 `verification_metrics()`

计算两类 Macro-F1、格式准确率、首次错误定位、错误原因准确率及二者联合准确率。格式不合法也受惩罚。

### 9.3 `diagnose()` 与 `evaluate()`

`diagnose()` 整理路径、判定和 trace。`evaluate()` 先评测候选验证、错误候选修复和去重地图上的直接生成。开启 `full=True` 后增加：

- 自验证修复：模型自己判断再根据自己的反馈修复。
- 执行器反馈修复：使用确定性错误反馈。
- 多轮迭代修复：只让仍失败的样本继续。
- 单轮消融：保留第一轮，观察多轮的额外收益。
- 给定错误候选的迭代修复和按错误类型分组统计。

内部 `check_plans()` 递归核对记录中的路径与官方执行器。输出 `metrics.json`、`candidate_traces.jsonl`、`pipeline_traces.jsonl` 和 `case_studies.json`。案例优先保留真实失败，便于追查原输出、错误位置、反馈和修复结果。

## 10. 原始命令与完整实验调度

`ppnl/cli.py` 提供 `environment`、`prepare`、`audit`、`smoke`、`pilot`、`train`、`evaluate`、`select`。它解析参数，再调用核心函数。

原始 `select` 取开发集生成成功率、验证 Macro-F1、修复成功率的平均，要求至少两个模型、训练配置与数据一致，并冻结选择。当前保存的主模型为 Flan-T5-small。

`scripts/run_study.py` 串行组织两个模型的 pilot、正式训练、开发评测、模型选择、相同 ID 测试，以及主模型 ID/OOD 完整管线。每阶段保存子进程日志、返回码与耗时；已有阶段可复用，底层仍检查相关身份与指纹。

## 11. 第一次逐步改进：`ppnl/policy_data.py`

`prepare_policy()` 把原生成/修复样本改成“当前状态预测下一步”：找正确路径，随机选一个位置，执行前缀得到当前位置，输入当前位置与地图，目标是下一个动作。例如 `right right down` 在第二步前的位置以 `right` 为标签。

每个原规划/修复样本只采一个状态，不把全部步骤展开增加样本。验证任务保留原样。测试候选只复制审计，不转换为训练数据。

## 12. 第二次数据改进：`ppnl/policy_v3_data.py`

`distances_to_goal()` 从目标反向 BFS，得到可达位置及距离。`prepare()` 从可达非目标状态采样，不局限于一条参考路线。

`critical()` 检查朝目标方向直接移动是否会撞障碍。训练中有一定概率优先抽取这类困难状态，开发集不使用同样的困难偏置。

`optimal_moves()` 列出所有使最短距离减少 1 的动作，一个位置可以同时允许 `down` 和 `right`。输入仍是地图、坐标及局部状态；BFS 距离只构造训练/开发标签，不送入推理输入。

## 13. 多最优动作损失：`action_set_loss()`

代码在 `ppnl/modeling.py`。

`single_action` 只认一个标签，可能惩罚另一个同样最优的动作。`optimal_actions` 接受全部最优动作，目标为：

```text
loss = -log(所有正确动作的概率之和)
```

概率在四个动作分数间归一化。策略样本用动作损失，验证样本用文本损失，再按样本合并。它改变训练目标，确实训练了不同参数；最近仅改 Harness 的那轮没有再训练权重。

## 14. 逐步评分与搜索：`ppnl/policy.py`

### 14.1 `neighbors()` 与 `state_prompt()`

`neighbors()` 列出物理合法的动作及下一位置，不判断最优。

`state_prompt()` 提供当前位置、目标、行列差、方向、合法动作、文字地图，以及可选的每个动作“空地/障碍/越界”状态。坐标差和合法性由程序计算，不能把这些辅助信息全归为模型能力。

### 14.2 `NeuralPolicy.scores()`

编码当前状态，只计算解码器第一个输出位置，取四个动作 token 的 logits，不生成完整长路径。先校验动作各对应单 token。按布局、目标、当前位置缓存分数。

Harness 记录的是策略请求次数，缓存命中时不一定再次执行 GPU 前向计算。

### 14.3 `greedy_plans()`

反复“观察→评分→选最高分→移动”。不约束时从四个动作直接选，可能越界或碰撞。旧版有约束时只选合法且未访问邻居；没有未访问邻居就停，不主动回退。达到目标或步数预算时结束。

### 14.4 `search_plans()`

维护优先队列，将步数、神经动作偏好和曼哈顿距离结合，展开看起来有希望的状态。神经分数直接加入代价，不能保证动作数最短。不使用神经分数时是纯符号对照。预算不足返回失败，不隐藏调用最短路补答案。

### 14.5 `optimal_search_plans()`

A* 主优先级为 `f=g+h`：已走步数加曼哈顿距离。神经分数仅在主优先级相等时打破平局，不改变主要路径代价。另有纯符号的目标距离平局对照。

这保留相应最短路性质，但展开预算不足仍可能失败。模型负责评分、搜索负责候选展开、环境代码负责合法性，三者贡献不同。

## 15. 逐步实验脚本

`scripts/run_policy_study.py` 转换早期状态动作数据、审计、保存配置快照、pilot、训练并评测 dev/ID/OOD。配置改变要求新的实验目录。

`scripts/run_policy_v3.py` 比较旧参考、单动作损失、多最优动作损失。先看开发集的无约束成功率、有约束成功率、无约束最优率，选择候选后测试，输出 `config.policy.best.json`。

`scripts/evaluate_policy.py` 统一评测逐步与搜索方法，保存指标、规划轨迹、修复轨迹和开发选择。`repair_prefix()` 保留实际正确执行的前缀，从失败前的真实位置重规划后缀；它不同于从起点重新生成整条路径。测试前核对开发阶段冻结的权重身份。

## 16. 单地图入口：`scripts/plan_grid.py`

读取地图和配置，选算法，加载必要模型，规划并执行检查，打印结果。默认读 `config.policy.best.json`，当前部署方法为神经平局排序的最优性保持 A*。纯符号方法不加载模型。

所以 `plan_grid.py` 默认算法与 `run_harness.py + policy_revisit.json` 的无搜索逐步运行不同。

## 17. Agent 运行层：`ppnl/harness.py`

### 17.1 `Budgets`

限制模型请求、工具调用、动作提案总量、活动耗时、历史条数、单提案动作数。工具中的 `plan_tokens` 指空格分隔动作数量，不是模型 tokenizer 的 token 数；模型生成上限由模型配置控制。

### 17.2 `fingerprint()`、`create_session()` 与状态

`fingerprint()` 对规范化 JSON 生成 SHA256。`create_session()` 保存地图、预算、backend 和执行器身份到新 trial 目录，拒绝覆盖。

状态含：`status`、已提交的 `plan`、最近提案 `candidate`、模型/工具请求数、动作尝试数、活动时间、失败计数、`feedback`、最近历史。

**candidate 不等于 plan**：提出错误路径不代表环境已经移动。

### 17.3 `_tool()` 的四个工具

- `move(action)`：追加一步，物理可行即可提交，未到目标也允许。
- `verify_plan(plan)`：仅验证，不提交。
- `submit_plan(plan)`：完整有效且到目标才提交。
- `finish()`：主动停止，本身不能产生成功。

仅接受白名单工具和严格参数，没有 shell、网络或任意 Python 工具。动作预算先检查再执行。

合法走一步但未到目标时，反馈可为 `invalid ... goal_not_reached`；这里表示当前前缀尚未完成任务，不表示这一步非法。提交移动看 `feasible`，完成任务看 `valid`。

### 17.4 `_position()` 与 `run_session()`

`_position()` 根据已提交路径重建真实位置。`run_session()` 的循环：获取锁，读取日志，核对身份，检查预算，先持久化并扣除一次模型请求，构建上下文，调用 backend，校验工具，执行、记录反馈、提交状态，然后判断成功/主动停止/预算耗尽。

backend 异常会记为可恢复错误。模型调用前计数，即使中断也不会完全忘记这次请求。

### 17.5 `_commit()`、`read_session()` 与原子快照

`events.jsonl` 是事实来源，`state.json` 是便于查看的快照。每次先追加完整事件并 flush/fsync，再原子更新快照。快照落后时仍可从日志恢复。

`read_session()` 检查 manifest、事件顺序、前一事件哈希、当前事件哈希和完整尾行。损坏时明确报错，不静默跳过。哈希主要发现意外变化，不是对能重写所有文件的攻击者的数字签名。

`_atomic_json()` 先写临时文件再替换；`_lock()` 防止并发运行同一个 trial。时间预算在边界检查，不能抢占阻塞中的同步调用。中断恢复可能重新请求模型，不保证模型调用恰好一次，但已提交环境状态从日志恢复。

## 18. 模型适配：`ppnl/harness_backends.py`

### 18.1 `file_hash()` 与 `backend_identity()`

记录 checkpoint 文件、配置、关键代码、ML 包版本的哈希/身份。组合模式同时记录第二个模型。恢复时不能悄悄替换算法、配置或权重。

### 18.2 `RepairBackend`

第一轮 generation，后续把候选路径和执行反馈写入 repair prompt，再提出 `submit_plan`。Flan-T5/BART 不原生输出这里的工具协议，是适配代码构造工具请求。

### 18.3 `PolicyBackend`

先得到四个动作分数。`legal_mask` 过滤非法邻居；`avoid_revisits` 从已提交路径重建访问次数，只保留访问次数最少的合法邻居，再按神经分数挑选动作。

它不是绝对禁止访问旧位置：全部邻居都访问过时仍允许回退。因此比旧版“无未访问邻居就停”更灵活，也不保证成功或最短。

### 18.4 `RepairPolicyBackend`

默认给整路径模型两次机会，未成功就切换逐步策略，共用总预算。阶段取决于持久化状态，重启不重置额度；每次 `next_call()` 只请求一个模型。

它使用第二个已训练 checkpoint，不是原整路径参数突然变强。实测纯逐步往往更好，因为省下两次失败提案，把预算留给实际移动。

### 18.5 `load_backend()`

按配置加载对应 checkpoint 和适配器。组合模式额外加载逐步模型；查看日志时不调用这个加载函数。

## 19. Harness 配置与入口

`scripts/run_harness.py` 支持 `run`、`resume`、`inspect`。`prepare()` 核对配置、地图、checkpoint 和官方执行器。`--dry-run` 检查本地文件和身份，不加载模型。

- `configs/harness/policy.json`：逐步策略与合法动作过滤，原示例配置，36 次预算。
- `configs/harness/repair.json`：整路径反馈修复，4 次预算。
- `configs/harness/repair_policy.json`：两次整路径机会后逐步恢复，总预算 16。
- `configs/harness/policy_revisit.json`：直接逐步加访问次数约束，总预算 16，当前优先使用。
- `configs/harness/ablation_no_revisit.json`：关闭访问次数约束，配合比较脚本做消融。实际方法由 `--methods` 选择。

运行例子，均在项目根目录执行：

```powershell
conda activate ppnl
python scripts/run_harness.py run --spec configs/harness/policy_revisit.json --out outputs/agent_harness/my_trial
python scripts/run_harness.py inspect --run outputs/agent_harness/my_trial
python scripts/run_harness.py resume --run outputs/agent_harness/my_trial
```

## 20. 两套评测入口

### 20.1 `ppnl/eval_harness.py` 与 `scripts/run_eval_harness.py`

统一运行既有规划方法并与冻结指标比对。覆盖原整路径 baseline、无约束逐步、合法未访问约束、神经代价搜索、纯 A*、神经平局排序 A*、纯符号目标距离平局排序 A*。

`Method` 保存名称、是否需模型、说明和运行函数；`resolve_methods()` 校验方法选择；`split_rows()` 读取/去重地图；`weights_dir()` 和哈希函数核对权重；`baseline_direct_plans()` 复用原 baseline 轨迹；`repair_prefix()` 重建合法前缀；`_evaluate_split()` 评测规划和前缀修复；`_compare()` 逐字段对比冻结指标；`run()` 管理官方执行器、预算和产物；`console_table()`、`render_markdown()`、`render_html()` 格式化报告，百分比/差值辅助函数负责展示。

原整路径行可能复用已保存轨迹，不是每次重新生成。跨方法、split 共享评分缓存，所以耗时是增量参考，不等于独立冷启动比较。正式评测要求官方 manifest，已有输出目录拒绝覆盖。

### 20.2 `scripts/compare_harness.py`

比较完整交互的 `repair`、`repair_policy`、`policy`。按布局/起点/目标去重，取源顺序前 N 个或全部，统一调用预算，每个样本独立 trial，清空评分缓存，调用真实 `run_session()`，官方复核状态后汇总指标。

`clear_cache()` 防止跨 trial 缓存污染。各组内部复用已加载权重减少加载开销。输入模型只有网格和反馈，不使用参考路径；BFS 只在事后打分。保存样本 ID、配置、逐条状态和权重身份。

指标针对最终已提交路径。错误提案未提交时，空路径仍可能可行，但任务不成功；必须结合成功率、失败提案数和原始日志，不能只看 feasibility。

## 21. 实际例子的执行链

`examples/grid.json` 为 6×6 地图，起点 `(4,5)`，目标 `(5,1)`，障碍 `(4,2)`、`(5,4)`。

直接逐步模式依次：读配置和地图→加载模型/tokenizer→新建 trial→按已提交路径重建位置→四动作评分→合法性和访问次数约束→提出 `move`→执行器验证→提交状态并写日志→重复至完成。

第一轮工具请求例子：

```python
{"name": "move", "arguments": {"action": "left"}}
```

实际成功路径 `left left down left left`，位置依次是 `(4,5)`、`(4,4)`、`(4,3)`、`(5,3)`、`(5,2)`、`(5,1)`。由执行器确认后变为 success。

组合恢复模式先出现两次失败整路径提案，再做五次移动，共七次请求。

## 22. 汇总、测试、环境和辅助文件

### 22.1 三个汇总脚本

- `scripts/summarize_study.py`：核对原两模型的训练配置、数据、选择与产物，输出原始实验汇总。
- `scripts/summarize_policy.py`：核对早期策略、原基线权重和测试候选是否一致，汇总规划结果。
- `scripts/summarize_policy_v3.py`：核对新版候选、开发选择、权重和旧结果完整性，汇总改进。

它们不训练模型，除收集分数外，还确认汇总时的权重和数据确实是当时评测的一份。

### 22.2 九个测试文件

- `tests/test_core.py`：执行规则、失败定位、完整路径、错误构造与官方差分。
- `tests/test_decoding.py`：允许合法重复动作的生成设置。
- `tests/test_metrics.py`：指标口径与格式错误惩罚。
- `tests/test_pipeline.py`：完整生成/验证/修复管线。
- `tests/test_policy.py`：策略、约束、搜索预算、前缀修复。
- `tests/test_policy_v3.py`：多最优标签、困难状态和损失梯度。
- `tests/test_eval_harness.py`：注册、指标对比、报告和输出保护。
- `tests/test_harness.py`：工具、预算、提交语义、日志、身份与恢复。
- `tests/test_harness_backends.py`：阶段切换、访问次数约束和回退。

最近一次运行共 52 项全部通过。部分使用固定评分或脚本化替身，隔离验证程序逻辑；通过不代表真实模型在所有环境都成功。

### 22.3 环境与文档

`requirements.txt` 管理主要 Python 依赖，CUDA Torch 需按运行平台单独安装。`environment.yml` 提供环境模板，使用时应核对 PyTorch 与目标 GPU 的兼容性。`ppnl/__init__.py` 仅包说明。数据已纳入版本管理；`.gitignore` 排除输出、缓存、虚拟环境与本地记录等。

`README.md` 是项目入口；`docs/environment.md` 说明环境；`docs/experiment_protocol.md` 说明评测与实验安排；`docs/planning_improvement.md`、`docs/planning_v3.md` 和 `docs/harness.md` 分别记录各轮改进。

## 23. 将代码与实际结果对应

最近同一批 200 个 ID 网格、相同 16 次请求上限：

- 仅合法动作过滤、关闭访问次数约束：170/200，85%，平均 6.025 次请求。
- 直接逐步加访问次数约束：199/200，99.5%，平均 5.025 次请求。
- 先整路径提案再恢复：198/200，99%，平均 7.01 次请求。

三组逐步模型权重相同，最近提升主要来自 `PolicyBackend` 的执行约束，不是重训。此比较也不能直接替换旧冻结 `legal_mask_no_search` 指标，因为旧代码会排除全部已访问邻居，且原预算不同。

直接逐步模式最优率 86.5%，即 173 条成功且最短、26 条成功但非最短、1 条未成功。它没有最短路保证。

- 优先少请求、避免整路径重复失败：`policy_revisit.json`。
- 研究原模型反馈修复与失败恢复：`repair_policy.json`。
- 需要显式搜索和最短路性质：`plan_grid.py` 的 `neural_optimal_astar`，同时保留纯 A* 对照。

历史结果位置为 `outputs/agent_harness/20261023_recovery_id200/results.json` 和 `outputs/agent_harness/20261023_policy_no_revisit_id200/results.json`。这些属于本地实验产物，按 `.gitignore` 不上传；本文件与 `docs/harness.md` 保留结果口径和说明。

## 24. 新增 G／V／R 独立 LoRA 流程

这一版研究整条路径的“生成、验证、修复”，与第 23 节逐步动作策略是独立方案。它不通过逐步合法动作过滤提升成功率，也没有搜索或最短路兜底。

`ppnl/data.py` 新增 `task_index` 与 `task_index_manifest`：从原有 train／dev 数据按任务筛选，检查数量、ID 唯一性和布局归属；`data/derived/gvr_task_index.json` 保存源文件哈希及确切 ID，防止运行时悄悄更换样本。它不生成额外训练数据。

`ppnl/gvr_preflight.py` 检查固定索引、官方执行器哈希、共用 GPU 账本、checkpoint 确认、证据声明及权重文件是否齐备。训练条件和评测条件分别汇报。沿用原项目的有效用户或负责人确认；默认配置不会替运行者声明历史证据已经核实。

`ppnl/gvr_training.py` 的 `train_stage` 是三个任务共用的训练函数。G 从固定版本的预训练 Flan-T5-small 开始，只使用 800 条 generation 数据全参数训练；V 和 R 分别重新加载冻结后的 G，只使用各自的 2,000／1,200 条记录训练独立 LoRA。每个阶段先做小规模 pilot，再正式训练最多两轮，以本任务开发集 loss 选权重。pilot 不给正式训练提供更新后的参数。三阶段正式训练共处理 8,000 次样本、完成 500 次更新；这不是额外 8,000 个不同训练样本。

`ppnl/task_adapters.py` 负责 PEFT 配置、权重哈希和适配器切换。LoRA 注入 T5 注意力的 q／v 投影，检查所有可训练参数都属于当前适配器；V、R 的 `base_identity.json` 必须指向同一 G 权重。生成时关闭全部适配器；验证或修复时只启用对应适配器。载入后由 `verify` 在一个固定开发样本上检查关闭适配器与单独加载 G 的输出是否一致，并保存验证记录。

`ppnl/gvr_harness.py` 复用已有日志、文件锁与状态恢复能力，运行四种整路径流程：`direct` 只生成；`self` 使用模型自己的验证反馈再决定是否修复；`executor_one` 使用真实执行反馈最多修复一次；`executor_three` 最多修复三次。模型请求先持久化计数，再调用模型；达到预算、反馈有效、重复候选或修复次数上限时停止。`self` 的停止决策不读取执行器真值，错误地相信模型“valid”时也会停止；最终独立评分仍判失败。

`ppnl/gvr_evaluation.py` 分开评测固定候选验证／修复与从零生成的端到端流程。它按网格去重，并检查四种流程的首个生成候选逐字一致；最终路径由独立执行器评分。结果包括成功率、可行率、最优率、首错定位、按错误类型指标以及调用和 token 数。当前开发集分别使用 Verify 100、Repair 60、Generate 40 条。历史运行曾错误合并为 160 个验证候选和 191 个生成网格；新增 `ppnl/gvr_audit.py` 从实际保存的预测重算正确子集，同时保留原始口径作为证据。它不会产生新预测或重演更新后的控制器。

`scripts/run_gvr.py` 提供 `index`、`preflight`、`pilot`、`train`、`verify`、`evaluate`、`recompute` 七类命令；`config.gvr.json` 指定权重位置、证据位和 LoRA 参数。训练／评测拒绝覆盖非空结果目录。当前 `verify` 增加完整开发集诊断；旧运行的一条样本检查保存在 `verification.json`，两者需要区别。`tests/test_gvr.py` 还覆盖任务分流、监督标签、停止预算和 CPU 重算。

`scripts/export_gvr_artifacts.py` 将权重和实验记录导出到 `artifacts/gvr/`，移除训练记录中的设备型号，规范相对路径，逐字节验证权重和归档日志。`manifest.json` 记录文件与来源 SHA256；日志无损压缩且省略运行期锁文件。大权重由 Git LFS 管理，完整使用方法见 [产物说明](artifacts/gvr/README.md)。

运行条件和命令见 [GVR 协议](docs/gvr_protocol.md)，实际 GPU 结果见 [本轮实验记录](docs/gvr_gpu_run.md)。旧逐步 Harness 的 99.5% 不能作为这个新流程的成绩；两种方法的训练、控制器与模型调用方式不同。

## 25. GVR 训练候选与实际结果

`ppnl/gvr_objectives.py` 新增可选的逐样本归一化与首个判别 token 损失，并提供只使用开发集指标的 checkpoint 排序。`ppnl/gvr_training.py` 读取阶段设置，将实际学习率、最佳 epoch 和训练签名保存到报告；签名检查防止 LoRA 或目标变化后沿用旧 pilot。原配置的默认目标与选模方式保持不变。

`config.gvr.next.json` 将候选设置放在独立输出目录，证据位默认关闭。`scripts/run_gvr.py` 提前拒绝已有输出，`ppnl/gvr_audit.py` 拒绝冻结 split 外的网格。候选已实际训练和评测：ID／OOD 验证 Macro-F1 为 0.6499／0.7326，固定候选修复为 12%／4%，端到端仍为 0%。修复模型第 2 轮 loss 更低但成功率下降，实际选中第 1 轮，说明选模按任务指标工作。设计与完整结果见 [候选设计](docs/gvr_next.md) 和 [实跑报告](docs/gvr_next_run.md)。

运行中 Windows 状态快照替换被短暂占用，`ppnl/harness.py` 的 `_atomic_json` 现对 `PermissionError` 做有限重试；永久错误仍抛出并保留日志与临时文件。进度监控改读权威日志。原失败目录保留，完成会话校验后复用，一个中断会话继续执行。训练前 69 项测试通过，新增恢复检查后共 72 项测试通过；这些运行问题的修复没有改变模型权重或评分规则。

## 26. 数据隔离和测试前冻结

`ppnl/experiment_freeze.py` 在 CPU 上重算布局并检查样本 ID、来源 ID、manifest 配额和任务索引，导出完整划分清单。训练/开发/测试按布局隔离；ID/OOD 有意共用环境，因此只要求这两组环境同序配对，不把它们说成独立测试。

`scripts/freeze_experiment.py` 提供 audit、freeze、check。冻结将所选 G/V/R 的权重、tokenizer、配置、源码、执行器、数据及开发集验证报告绑定为哈希记录。`scripts/run_gvr.py` 在测试推理前验证 `--freeze`，缺失或漂移时在 CUDA 初始化前停止。成功结果保存锁哈希，方便追溯所用设置。

冻结解决输入漂移，不能证明过去没有看过测试集。现有测试继续标为探索性结果，当前本地冻结只是事后保存。技能图作为下一步独立消融计划，HotpotQA 不加入主实验。完整规范及未来留出集要求见 [实验冻结说明](docs/evaluation_freeze.md)。

## 27. 强化学习尝试：RLVR，而非人工反馈RLHF

目前没有人工偏好数据，所以本轮用执行器提供可验证奖励。`ppnl/rlvr.py` 对到达目标给主要奖励，对非法动作与冗长路径惩罚，对失败路径提供小幅距离进展塑形。它不读取参考答案；曼哈顿进展只是辅助训练信号，不是成功证明。

`scripts/run_rlvr.py` 复用生成底座，新增独立q/v LoRA。每个训练网格采样四条路径，以其余样本的平均奖励作baseline，通过REINFORCE增加高于baseline的输出概率，并用冻结底座的采样KL约束漂移。只做一次on-policy更新，没有critic、PPO裁剪或完整GRPO流程。`configs/rlvr.json` 固定64步、学习率0.00005、r=8、KL权重0.02及48-token解码上限。

实际完成256条采样、64次更新。训练采样有2条成功，但开发集始终0/40；奖励从−1.1691升至−1.1488，不能称为规划成功率提高。开发集选中的step64 LoRA经 `scripts/verify_rlvr.py` 重载后，40条预测完全一致。3项新增CPU测试验证奖励和梯度。没有跑测试集，没有替换原基线。详细设置、局限和本地产物说明见 [RLVR实跑报告](docs/rlvr_run.md)。

## 28. 实验记录如何发布和复核

`scripts/export_rlvr_evidence.py` 将本地RLVR运行中的五次开发集预测、64组训练采样、选模报告、重载身份和来源记录整理到 `artifacts/rlvr/`，保留数据字节并提供SHA256清单；来源中的机器绝对路径转换为仓库相对路径。它不导出新权重、缓存或整个outputs目录，重复导出会拒绝覆盖已有证据。公开的 [非权重证据](artifacts/rlvr/README.md) 支持逐条核查，加载新模型仍需要本地生成底座和adapter。

RLVR运行使用当时的 `8e05270` 代码及归档的RL新增源码；随后同步的梯度累积修复属于后续代码版本，不能写成此前试验已使用该修复。同步同伴更新后，整合了冻结规范、RLVR和真实T5/LoRA边界检查，完整90项回归测试通过。这些测试验证实现与保存重载，不代表规划成功率提高。

## 29. 参考项目借鉴：结构化输入候选

`ppnl/prompts.py` 新增可选结构化环境描述，将网格、坐标规则、起点、终点和障碍物列为字段。生成、验证和修复仍使用原有任务说明和输出格式，默认自然语言模板保持原样。`format_training_rows` 只转换输入，不改变ID、标签、参考路径或样例数；验证输入明确忽略金标反馈，修复输入使用调用者传入的反馈。

`ppnl/gvr_training.py` 先检查固定原始数据，再转换输入，训练签名记录结构化设置。新底座和LoRA保存格式身份；`ppnl/task_adapters.py` 检查匹配关系并在推理时复用同一模板。`scripts/run_gvr.py` 的裸底座／关闭adapter检查也使用相同输入格式。冻结机制要求开发集检查证据的输入格式与当前协议一致，防止拿旧模板证据替新实验背书。

两份配对配置为 `configs/gvr_format_vanilla.json` 和 `configs/gvr_format_structured.json`，分别写入新目录，其他实验参数一致。`scripts/preview_prompt_formats.py` 只用CPU和本地tokenizer检查train/dev；实际最长训练输入从159降到147token，完整95项CPU测试通过。尚未训练或评测结构化模型，不能声称规划成功率提高。坐标跟踪CoT尚未启用，参考项目的长训练预算和RL实现未移植。完整流程见 [结构化输入说明](docs/structured_inputs.md)。


## 30. 整理后 main、方法和分支的准确含义

本节是代码整理后的面试说明；详细操作见 [项目导航](docs/project_map.md)。

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


## 31. 三模型 GPU 扩展与首次 outcome-GRPO

前述“尚未实现 GRPO”是上一轮代码整理时的状态。本轮按用户要求加入真正具有组内标准化优势、固定旧策略 log-prob、概率比裁剪和多次更新的 outcome-GRPO，并保留 REINFORCE 对照。它是序列到序列模型适配，不是原研究模型的复现，也没有人工偏好。实现细节、固定预算、BART BOS 对齐、磁盘失败恢复和所有负结果保留规则见 [三模型实验记录](docs/gpu_campaign_20261103.md)。当前实际进度以本轮输出中的完成报告为准。

### 31.1 新入口如何串起来

1. `scripts/run_gpu_campaign.py` 先训练三个原始多任务基线，然后给每个模型分别训练 vanilla / structured 的 G、V、R。G 全参数训练，V/R 冻结 G 并各自训练 LoRA；最后从开发集选择 G 的输入格式，分别训练 REINFORCE 和 GRPO adapter。模型保存后还要通过重载验证。
2. `scripts/run_campaign_status_ablation.py` 保留选定的 G/R，只重训 V，对照普通 token loss 与逐样本归一化加判别 token 辅助损失。后者同时改变两个因素，所以不能把所有差异归因于判别 token。
3. `scripts/run_campaign_policies.py` 从三个预训练底座分别训练逐步动作策略。这是独立的全参数训练，不是给 G/V/R 的 LoRA 加一个循环。使用开发集选择策略配置后，冻结模型、数据、源码和评测配置，再运行测试。
4. `scripts/evaluate_gpu_campaign.py` 依次运行全量基线 Harness、三类 GVR 配置、RL 与无 RL 对照，最后用相同架构的逐步策略做前缀修复。每条生成路径都分别进入 direct、self、executor_one、executor_three，避免不同初始路径导致不公平比较。
5. `scripts/summarize_gpu_campaign.py` 汇总预定的 99 份开发/测试报告。缺失报告明确标记 missing，不能把“脚本已启动”写成“实验已完成”。公开汇总只包含成绩和协议，不包含本地权重或机器身份。

### 31.2 GRPO 代码到底增加了什么

`ppnl/rlvr.py` 的 `group_advantages` 把同一个问题的 4 个采样奖励按组均值和标准差归一化；同奖励组的优势为零。`completion_mask` 保留 EOS 本身，排除 EOS 后的 padding。`grpo_objective` 计算新旧策略的 token 概率比，对正负优势分别执行 PPO 形式的裁剪，并加入冻结底座的采样 KL 项。`scripts/run_rlvr.py` 固定采样时的旧策略概率，在同一组样本上做两次更新，避免每次都重新计算旧策略而令裁剪失效。

REINFORCE 对照使用 leave-one-out baseline，每组只更新一次。两者均采样 64 组，但 optimizer updates 分别是 64 和 128，计算预算并不相同。奖励来自网格执行器，没有人类偏好标签，因此称为 RLVR 更准确。奖励变高但成功数不变，只能说明奖励目标改善，不能说规划能力变强。

### 31.3 为什么 BART 需要专门处理

`ppnl/task_adapters.py` 根据模型架构选择注意力投影：T5 用 `q/v`，BART 用 `q_proj/v_proj`。BART 文本目标带 BOS，而动作策略直接给第一个 decoder 输出打分，所以 `ppnl/modeling.py` 只对动作训练标签移除 BOS，验证文本标签仍保留。相应地，`ppnl/gvr_objectives.py` 在 BART 的 BOS 后定位 valid/invalid 判别 token；T5 则使用第一个目标 token。

### 31.4 怎样解释逐步策略消融

`raw` 从四个动作直接选最高分；`legal_no_revisit` 排除非法动作和所有已访问位置；`harness_legal` 只排除非法动作；`harness_visit` 优先选择访问次数最少的合法位置，必要时允许回访。这四项统一最多 36 步。后两项通过持久化 Harness 调用执行器，它们没有搜索整张图。

`neural_best_first`、`neural_optimal_astar` 和 `symbolic_astar` 是额外搜索对照。纯 A* 本身就能解决这些小网格，所以搜索版本的 100% 不能当作语言模型独立规划的证据。所有方法均同时报告成功率与最优率：到达终点可以成功，但绕路不会算最优。

这些实验不覆盖旧默认配置，也不通过删除失败产物来美化结果。辅助损失等负结果保留为消融；新增 checkpoint 只保留在本地。ID/OOD 的生成网格配对、候选错误不同，因此纯生成和逐步策略的相同成绩不构成新的地图分布泛化证据。

### 31.5 本轮实际结果（2026-11-04 完成）

后续发布更新：根据新的备份要求，本轮 33 个有效权重文件、99 份报告和执行轨迹纳入 [Git LFS 产物目录](artifacts/campaign_20261103/README.md)。前文“只保留本地”描述训练完成时的决定，此次发布后不再适用；原始本地副本仍保留。公开元数据经过脱敏，原始/导出哈希分别记录，不能把迁移后的历史锁直接当作新运行的冻结证据。

同时补充发布 [9 个历史有效权重](artifacts/historical_models/README.md)，包含早期多任务基线、改进 GVR、Flan-T5 逐步策略和旧 REINFORCE；已公开的原 GVR 权重不重复复制。历史模型的预算与代码版本仍按其原报告解释。

99/99 份预定开发/测试报告全部完成；33 份冻结记录与当前源码匹配，66 份测试报告的冻结回执匹配。CPU 回归测试 141 项通过。完整数值见 [本轮报告](docs/gpu_campaign_20261103.md) 与 [指标 JSON](docs/gpu_campaign_results.json)。

在各 200 个 ID 测试环境上，T5-small、T5-base、BART-base 的裸逐步策略分别成功 169、163、164 个；加入 Harness 最少访问次数约束后均成功 200 个，最短路径分别为 172、170、165 个。这是独立的逐步策略训练和推理约束的结果，不能写成“GRPO + LoRA 达到 100%”。统一预算为 36 步，与旧版不同调用预算的结果分开报告。

同底座、同 48-token 上限的 GRPO 生成成功数分别为 4、0、1，无 RL 对照为 0、1、0。效果有限，T5-base 还出现下降；不升级为默认方法。REINFORCE、结构化输入和验证辅助损失也保留为消融，不因个别测试数字变化而倒改开发选模。

前缀保留配合访问约束虽全部成功，最优率仅为 27.5%–37%；丢弃前缀从起点重启则为 82.5%–86%。这说明成功主要依赖后续逐步策略，而原始路径中的绕路会被保留下来。纯启发式前缀控制也全部成功，因此不能把前缀修复高分全部归功于模型。

开发集预先选定的是 T5-small 的 legal_no_revisit（测试 195/200），以及 T5-base/BART 的 harness_visit（各 200/200）。T5-small 的 harness_visit 测试 200/200 属于事先列出的消融结果；不把这个事后观察包装成开发集事先选定的方案。所有新增权重保留本地，历史默认配置与分支归档不变。
