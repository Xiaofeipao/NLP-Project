# GVR 端到端改进：台式机执行结果

日期：2026-11-10。源码基线 `86b52edc3de69481352bec5d0a15c1aa372a3e2b`；实现提交为 `0978c1a`，与本报告同属 `codex/gvr-constrained-repair-study` 分支。实验在台式机 `E:\YanbingChen\NLP-Project` 运行；冻结文件中的 64 个源码哈希与该提交的工作树逐一一致。本文件为结果记录；原始 JSON/JSONL、冻结文件、账本及逐文件 SHA-256 见同目录 `evidence.zip`。压缩包 SHA-256 为 `e3f8574fa3da7094887929f7b1ed17d434171a8cb9fb445e262492716aeb8ba4`。

目录：一、实验合同与实现；二、开发集；三、测试集；四、归因与局限；五、案例；六、预算与复核；七、复现入口。

## 一 实验合同与实现

基于历史开发选中的 Flan-T5-small 全量多任务权重。真实权重 307,867,048 字节，SHA-256 为 `0b7be3b12238b363ba8c1a9304f687c6ae6f9edbecceff0076cebe7c260d62fa`，与 Git LFS 对象一致。训练记录与数据未增加；使用原训练 4,000、开发 200、ID/OOD 测试各 200 条，固定 seed 42。测试集此前已被项目使用，因此本次是**重复使用测试的探索性评估**。

新方法 C4 仅在执行器反馈的 R 步骤使用四宽度动作 beam、36 动作上限、合法移动、位置去重和目标处 EOS 门控；G/V 及旧流程仍走原解码。B4 使用相同模型、宽度、动作预算和动作语法，关闭执行状态约束。U4/H4 使用相同解码器与约束，以固定动作顺序／曼哈顿距离替代语言模型分数。H4 是强规则对照，输出选择偏向短路径，不能视为只改变一个 LM 变量的消融。推理时 C4 不调用 BFS/A* 或参考最短路；评测才用最短路计算最优性。

台式机环境为 RTX 3050（6 GB）、Python 3.12.10、PyTorch 2.11.0+cu128、Transformers 4.57.6、PEFT 0.18.1。PyTorch 版本不同于仓库 `environment.yml` 的 2.7.0；该偏差需在课程报告复现信息中明示。项目独立虚拟环境通过 `.pth` 读取台式机已有 CUDA PyTorch，其余依赖安装在 `E:\YanbingChen\NLP-Project\.venv`。`pip check` 无缺失依赖。

## 二 开发集：先过门槛，再冻结

开发集生成 40 个环境、验证 100 条、修复 60 条，配额未改变。原四流程 direct/self/executor-one/executor-three 的端到端成功数全为 **0/40**，验证 Macro-F1 为 **0.6291**。真实 G 路径 40 条均初始失败。实验修复结果如下：

| 方法 | 真实 G 后端到端 | 60 条合成候选修复 |
| --- | ---: | ---: |
| 原单轮 E1 | 0/40 | 10/60 |
| 仅动作语法 B4 | 0/40 | 8/60 |
| 执行约束 C4 | **36/40** | **60/60** |
| 固定规则 U4 | 37/40 | 57/60 |
| 曼哈顿规则 H4 | **40/40** | **60/60** |

C4 通过预设的四项工程门槛：真实生成至少 8/40；相对 E1 净增至少 4；相对 B4 净增至少 2；合成修复成功数相对 E1 下降不超过 3。初始 G 路径与旧流程逐例一致。之后生成开发决策和冻结文件，冻结 SHA-256 为 `1b6285e3f1fa03d611f3839916fdf0b4cc84130fdaf075392e47d4fa57e6877c`。改变 beam 宽度的校验尝试被拒绝，测试前无锁也无法运行。

## 三 测试集：同一批 200 个生成环境

测试 ID 的 200 条生成环境与测试 OOD 的地图、起终点逐例相同；OOD 只改变候选错误分布。ID 物理生成 G 200 次，OOD 物理生成 G 0 次，复用同一轨迹。两张表的端到端结果不能合并成 400 个独立样本。

| 方法 | 端到端成功 | 路径最优率 | 成功路径平均最短路效率 |
| --- | ---: | ---: | ---: |
| direct / self / E1 / E3 | 0/200 | 0% | — |
| B4 | 0/200 | 0% | — |
| C4 | **184/200（92%）** | 43% | 0.691 |
| U4 | 191/200（95.5%） | 37% | 0.655 |
| H4 | **200/200（100%）** | 99.5% | 0.999 |

旧验证器 Macro-F1：ID **0.7190**，OOD **0.7750**。候选修复使用各 100 条非法候选，ID 是单错误，OOD 是双独立错误：

| 方法 | ID 修复成功 | OOD 修复成功 | ID / OOD 最优率 |
| --- | ---: | ---: | ---: |
| E1 | 19/100 | 7/100 | 19% / 7% |
| B4 | 25/100 | 8/100 | 25% / 8% |
| C4 | **98/100** | **94/100** | 96% / 83% |
| U4 | 95/100 | 95/100 | 35% / 35% |
| H4 | **100/100** | **100/100** | 99% / 99% |

全部生成候选都经严格本地执行器及官方执行器交叉检查。C4 失败的 16 条均标记 `dead_end`，没有空输出；成功路径平均 8.66 步，H4 为 4.55 步。C4 的可行率 100% 包含“合法但未到目标”的失败路径，不能拿它代替成功率。

## 四 配对归因与局限

C4 对 E1 有 **184 胜、0 负**，对同宽度 B4 也有 **184 胜、0 负**。按 79 种障碍布局作配对 cluster bootstrap（seed 42、2,000 次），C4−E1 的成功率差 95% 区间约 **+87.6～+95.6 个百分点**；C4−B4 约 **+87.9～+95.6 个百分点**。这支持执行状态约束模块的作用；测试重复使用带来的选择偏差不会被区间消除。

与规则方法相比，C4 对 U4 有 **9 胜、16 负**，对 H4 有 **0 胜、16 负**；C4−H4 的布局聚类区间约 **−12.3～−4.4 个百分点**。因此不能声称语言模型动作排序是成功所必需，也不能把 C4 的 92% 当成优于规则规划的结果。约束解码改善了 GVR 流程，强规则 H4 仍是更好的当前性能上界。C4 的长绕路与死路问题是下一步应解释的模型排序缺陷；不应在已查看的测试集上改 beam、长度惩罚或 checkpoint 来追分。

## 五 两个具体案例

成功但绕路：6×6，障碍 `(5,3)`，起点 `(3,2)`，目标 `(5,0)`。原 G 为 `up up up up up right right right right right`，第 4 步越界。C4 输出 24 步合法到达路径，H4 仅用 `down down left left` 四步到达。这个例子展示端到端成功的同时，也暴露 C4 的路径质量问题。完整动作与执行字段见 `test_id/generation_traces.jsonl`，来源 ID `1_train_set_6x6_samples.json:20`。

失败死路：6×6，障碍 `(0,3),(5,3)`，起点 `(4,5)`，目标 `(5,1)`。原 G 第 5 步越界；C4 输出 25 步未到终点的合法前缀，状态为 `dead_end`。H4 的 `left left left down left` 五步到达。来源 ID `1_train_set_6x6_samples.json:15766`。该案例说明 beam=4 即使排除了非法动作，模型排序仍可能把所有保留分支带入死路。

## 六 预算、验证和后续工作

归档账本起点为 240 次、6.6690 GPU 小时。本研究新增 7 次账本记录，共 **1,569.05 秒＝0.436 小时**：两次 smoke 33.80 秒、开发阶段 325.16 秒、两次测试 1,210.09 秒。全项目现为 247 次、**7.1049 GPU 小时**，低于 20 小时上限；本研究低于新增 4 小时上限。模型分配显存峰值约 321 MB，系统显存占用观测约 1.3 GB。没有新增训练步数。

隔离工作树完整 CPU 回归 **151 项运行：150 项通过、1 项因缺少本地官方执行器文件而跳过**；台式机新路径相关 **11 项通过**，官方执行器哈希与数据清单一致，实际 GPU 推理、开发评估及 ID/OOD 测试均完成。证据包内 19 个文件的哈希均复核通过。代码与报告在同一审阅分支，尚未合并主分支。

课程报告宜把 C4 作为受约束解码的 GVR 系统改进，完整公开 H4 的更强成绩。若后续目标是提升模型本身对反馈的利用，应先审计 R 子集剩余训练轮次，再设计同一模板下的局部状态监督；在确认课程配额前不追加训练。正式提交前还需保留负责人对 Flan-T5/T5 精确 checkpoint 的确认记录，并说明测试集已重复使用。

## 七 复现入口

在固定源码提交的干净工作目录中，先获取 `artifacts/historical_models/flan_t5_small/train/best/model.safetensors` 的真实 Git LFS 对象，并放入与 `data/derived/manifest.json` 哈希一致的官方执行器文件。使用隔离的 CUDA Python 环境；本次台式机采用上述 PyTorch/Transformers 版本。核对历史 GPU 用量后，将项目归档的 `artifacts/campaign_20261103/gpu_ledger.json` 复制为新的 `outputs/gpu_ledger.json`，不要把本证据包内已经包含本次运行的末态账本再当作起点。

以下命令从项目根目录执行，输出目录必须尚不存在；`preflight` 和 `--check-only` 不加载 CUDA。冻结之后不要修改 `ppnl/` 或 `scripts/` 源码，测试入口会先校验冻结身份。

```powershell
python scripts/smoke_gvr_repair.py preflight
python scripts/smoke_gvr_repair.py run
python scripts/smoke_gvr_repair.py run --limit 40 --out outputs/gvr_repair_study/dev_generation
python scripts/smoke_gvr_repair.py run --task repair --limit 60 --out outputs/gvr_repair_study/dev_repair
python scripts/run_gvr_development.py
python scripts/freeze_gvr_repair.py
python scripts/run_gvr_repair_test.py --split test_id --check-only
python scripts/run_gvr_repair_test.py --split test_id
python scripts/run_gvr_repair_test.py --split test_ood --check-only
python scripts/run_gvr_repair_test.py --split test_ood
```
