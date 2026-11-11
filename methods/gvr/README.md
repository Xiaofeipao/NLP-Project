# gvr

全量模型的生成、验证、修复流程；不是独立训练算法。

入口：`python -m methods.gvr <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.gvr --describe`。

可用命令：`run`。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。底层实现保留原路径，防止历史导入及冻结证据失效。

## 两种修复解码（本地接入）

### 新增候选：已编码，尚未运行

`--repair-decoding backtrack --max-score-rows 148` 开启有限预算回溯：当前 beam 无可扩展节点时，恢复最近被裁剪的同深度备选。沿用合法动作、不回访和终点 EOS 条件。148 只是试验预算示例，不是调优值。`--repair-decoding constrained --max-score-rows 148` 是同 scorer-row 上限对照；相同行数不等于相同 token/FLOPs/耗时，仍需报告 decoder_input_tokens_processed 和时间。预算耗尽不能输出 NO_PATH，也无最优性保证。

`--repair-feedback original|none|generic` 分别提供原错误诊断、空反馈、只说明候选错误的通用反馈，默认 original。仅 executor_one/three 允许从 CLI 开启消融；执行器仍负责判分和停止，只改变 R 看到的文本。身份记录绑定选项，历史记录 repair_feedback_used 保存实际模型反馈。该消融尚不包含跨案例打乱。

普通解码默认与既有 beam 默认不变。新增选项只接入单网格入口；旧批量研究脚本仍是原 C4/B4 协议，尚不能声称已完成新方法的同预算全测试。先定义开发实验矩阵和新冻结，再开展正式对照。新增测试文件已写但按要求未运行，历史 158 项通过不覆盖本次修改。

`run` 子命令新增 `--repair-decoding ordinary|constrained`，默认 ordinary，沿用原普通解码。constrained 仅用于 executor_one / executor_three 模式、T5 全量 checkpoint；G/V 不变，R 使用合法动作、路径内不回访、终点 EOS 门控的 beam 解码。默认 beam=4、最多 36 动作，可通过 --beam-width / --max-actions 显式设置。这不保证最短路，也不能用 dead_end 推断 NO_PATH。

```powershell
python -m methods.gvr run -- run --checkpoint artifacts/campaign_20261103/baselines/t5_small/train/best --grid examples/grid.json --mode executor_one --model-config artifacts/campaign_20261103/baselines.json --repair-decoding ordinary --out outputs/full_finetune_gvr/ordinary_trial
python -m methods.gvr run -- run --checkpoint artifacts/campaign_20261103/baselines/t5_small/train/best --grid examples/grid.json --mode executor_one --model-config artifacts/campaign_20261103/baselines.json --repair-decoding constrained --out outputs/full_finetune_gvr/constrained_trial
```

追加 --dry-run 可在 CPU 核查配置、模型身份和官方执行器。约束参数存入 spec 与会话身份，resume 读取原参数，不允许静默切换。共用原执行器与 score 入口；新增 forward_batches 等计数，不能只按外层 R 调用次数断言计算量相同。

规则对照与正式研究保持独立：scripts/smoke_gvr_repair.py、run_gvr_development.py、freeze_gvr_repair.py、run_gvr_repair_test.py 及 docs/gvr_repair_study/。单网格入口不代替冻结测试。历史报告属于同伴原提交 0978c1a 的实验，不能视作本地整合后的新成绩；当前源码改变后需重新开发检查和冻结，不能绕过旧锁。历史会话的代码身份也可能不兼容，应保留原提交复现，不改写旧证据。
