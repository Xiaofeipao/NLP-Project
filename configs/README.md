# 配置分类

- `harness/`：逐步 Agent Harness、整路径修复、组合恢复及前缀保留。
- `prefix/`：两类前缀评估，统一 dev → freeze → test；`full_finetune.json` / `lora_gvr.json` 指定生成 trace 来源，不改变补全策略的训练方式。
- `gvr_format_vanilla.json` / `gvr_format_structured.json`：配对输入格式消融，未训练。
- `rlvr.json`：独立 LoRA REINFORCE 试验，显式 `prompt_format`，不是 GRPO。
- 根目录 `config.json`：原多任务基线；`config.gvr*.json`：独立 G/V/R；`config.policy*.json`：逐步策略历史实验。

保留根目录配置路径以兼容历史命令和身份记录；不能把不同 config 的指标合并为同一个模型的结果。
