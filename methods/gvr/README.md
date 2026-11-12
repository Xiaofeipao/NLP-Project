# gvr

全量模型的生成、验证、修复流程；不是独立训练算法。

入口：`python -m methods.gvr <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.gvr --describe`。

可用命令：`run`（单网格调试）、`study`（固定权重后的批量配对消融）。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。

## 当前默认批量入口

```powershell
python -m methods.gvr study -- preflight --spec configs/post_training.json
python -m methods.gvr study -- run --spec configs/post_training.json --out outputs/post_training/run_001
```

同一多任务全量权重的 13 个配置覆盖 ordinary、B4、C4、backtrack、反馈文本删减和规则评分。完整设计、V/执行器分工、指标分母和恢复规则见 [后处理说明](../../docs/post_training.md)，本轮 GPU 证据见 [复跑报告](../../docs/post_training_run_20261112.md)。

## 单网格修复选项

`--repair-decoding backtrack --max-score-rows 148` 开启有限预算回溯：当前 beam 无可扩展节点时，恢复最近被裁剪的同深度备选。沿用合法动作、不回访和终点 EOS 条件。148 只是试验预算示例，不是调优值。`--repair-decoding constrained --max-score-rows 148` 是同 scorer-row 上限对照；相同行数不等于相同 token/FLOPs/耗时，仍需报告 decoder_input_tokens_processed 和时间。预算耗尽不能输出 NO_PATH，也无最优性保证。

`--repair-feedback original|none|generic` 分别提供原错误诊断、空反馈、只说明候选错误的通用反馈，默认 original。仅 executor_one/three 允许从 CLI 开启消融；执行器仍负责判分和停止，只改变 R 看到的文本。身份记录绑定选项，历史记录 repair_feedback_used 保存实际模型反馈。该消融尚不包含跨案例打乱。

单网格 ordinary 仍为默认值；批量 study 由独立配置显式列出方法。测试与 GPU 验证以本轮报告记录为准，历史成绩保留原协议。

`run` 支持 `--repair-decoding ordinary|constrained|backtrack`，默认 ordinary；B4 由批量 study 的 `syntax_beam` 方法配置提供。约束仅用于 executor_one / executor_three 模式、T5 全量 checkpoint；G/V 不变，R 使用合法动作、路径内不回访、终点 EOS 门控的 beam 解码。默认 beam=4、最多 36 动作，可通过 --beam-width / --max-actions 显式设置。这不保证最短路，也不能用 dead_end 推断 NO_PATH。

```powershell
python -m methods.gvr run -- run --checkpoint artifacts/campaign_20261103/baselines/t5_small/train/best --grid examples/grid.json --mode executor_one --model-config artifacts/campaign_20261103/baselines.json --repair-decoding ordinary --out outputs/full_finetune_gvr/ordinary_trial
python -m methods.gvr run -- run --checkpoint artifacts/campaign_20261103/baselines/t5_small/train/best --grid examples/grid.json --mode executor_one --model-config artifacts/campaign_20261103/baselines.json --repair-decoding constrained --out outputs/full_finetune_gvr/constrained_trial
```

追加 --dry-run 可在 CPU 核查配置、模型身份和官方执行器。约束参数存入 spec 与会话身份，resume 读取原参数，不允许静默切换。共用原执行器与 score 入口；新增 forward_batches 等计数，不能只按外层 R 调用次数断言计算量相同。

旧批处理已归档，迁移说明见 [脚本目录](../../scripts/README.md)。新 study 集成 uniform/manhattan 对照，并自行冻结协议。单网格入口不替代批量评估。历史会话需使用原提交复现，不改写旧 identity 或把新 schema 喂给旧冻结器。
