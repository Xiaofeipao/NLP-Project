# gvr

全量模型的生成、验证、修复流程；不是独立训练算法。

入口：`python -m methods.gvr <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.gvr --describe`。

可用命令：`run`。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。底层实现保留原路径，防止历史导入及冻结证据失效。

## 两种修复解码（本地接入）

`run` 子命令新增 `--repair-decoding ordinary|constrained`，默认 ordinary，沿用原普通解码。constrained 仅用于 executor_one / executor_three 模式、T5 全量 checkpoint；G/V 不变，R 使用合法动作、路径内不回访、终点 EOS 门控的 beam 解码。默认 beam=4、最多 36 动作，可通过 --beam-width / --max-actions 显式设置。这不保证最短路，也不能用 dead_end 推断 NO_PATH。

```powershell
python -m methods.gvr run -- run --checkpoint artifacts/campaign_20261103/baselines/t5_small/train/best --grid examples/grid.json --mode executor_one --model-config artifacts/campaign_20261103/baselines.json --repair-decoding ordinary --out outputs/full_finetune_gvr/ordinary_trial
python -m methods.gvr run -- run --checkpoint artifacts/campaign_20261103/baselines/t5_small/train/best --grid examples/grid.json --mode executor_one --model-config artifacts/campaign_20261103/baselines.json --repair-decoding constrained --out outputs/full_finetune_gvr/constrained_trial
```

追加 --dry-run 可在 CPU 核查配置、模型身份和官方执行器。约束参数存入 spec 与会话身份，resume 读取原参数，不允许静默切换。共用原执行器与 score 入口；新增 forward_batches 等计数，不能只按外层 R 调用次数断言计算量相同。

规则对照与正式研究保持独立：scripts/smoke_gvr_repair.py、run_gvr_development.py、freeze_gvr_repair.py、run_gvr_repair_test.py 及 docs/gvr_repair_study/。单网格入口不代替冻结测试。历史报告属于同伴原提交 0978c1a 的实验，不能视作本地整合后的新成绩；当前源码改变后需重新开发检查和冻结，不能绕过旧锁。历史会话的代码身份也可能不兼容，应保留原提交复现，不改写旧证据。
