# rl

执行器奖励训练：REINFORCE 或 GRPO，可使用 LoRA；不是人工反馈 RLHF。

入口：`python -m methods.rl <command> -- <原入口参数>`。只查看配置和模块归属：`python -m methods.rl --describe`。

可用命令：`run`、`verify`。详细映射见 [manifest.json](manifest.json)。配置引用仓库原文件，不复制训练参数或改变默认值。底层实现保留原路径，防止历史导入及冻结证据失效。
