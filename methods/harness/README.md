# harness

持久化模型与工具循环、预算、恢复、逐步策略。

入口：`python -m methods.harness <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.harness --describe`。

可用命令：`run`、`compare`、`evaluate`。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。底层实现保留原路径，防止历史导入及冻结证据失效。
