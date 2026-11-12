# harness

持久化模型与工具循环、预算、恢复、逐步策略。

入口：`python -m methods.harness <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.harness --describe`。

可用命令：`run`、`compare`、`evaluate`。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。底层实现保留原路径，防止历史导入及冻结证据失效。


## 恢复历史高最优率入口（2026-11-12）

按用户要求恢复以历史规划方法为主的使用入口。近期 GVR 约束解码/回溯仍为独立实验，不替换这里的规划器。底层旧算法本来仍在，因此不回滚整个仓库或撤销后续正确性修复。

- 最高最优率路线：`python -m methods.harness plan -- --grid examples/grid.json`。复用原 `neural_optimal_astar`、历史多最优动作 Flan-T5-small 权重与 36 次节点展开上限。历史 ID 成功率/最优率均为 100%，纯 A* 对照也为 100%；不是无搜索模型成绩。展开预算内找不到解仍会失败，不保证所有新地图成功。
- 原逐步 Agent Harness：`python -m methods.harness run -- run --out outputs/agent_harness/restored_001`。默认读取 `configs/harness/policy.json`，每次只选一步，合法动作过滤 + 最少访问次数约束，最多 16 次模型/工具调用；历史 ID 成功 199/200，最短 173/200（86.5%）。未加入整路径修复或回溯搜索。

两条入口指向已发布的 `artifacts/historical_models/planning_v3/multi/flan_t5_small/train/best`；需要 Git LFS 实体权重和官方执行器。本轮只恢复入口和有效权重路径，不重新训练，不把旧成绩写成新运行结果，也不宣称对新增 7×7/密集 6×6 测试集已有成绩。此次按用户偏好采用历史观察结果，不重写原开发集选择记录。

校验：25 项 Harness CPU 回归通过，入口参数与配置 dry-run 通过；dry-run 不代表加载过实体权重。当前历史权重文件为 Git LFS 指针，实际 GPU 运行前需执行：

```powershell
git lfs pull --include="artifacts/historical_models/planning_v3/multi/flan_t5_small/train/best/**"
```
