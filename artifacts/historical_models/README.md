# 历史有效模型补充备份

此目录补齐以前只保存在本地的 9 个已选定权重文件及现有紧凑训练/评测记录，区别于 `artifacts/campaign_20261103` 的三模型新实验。

- `bart_base`、`flan_t5_small`：早期两个多任务全量微调基线。
- `gvr_next`：早期改进 GVR 的生成底座、验证 LoRA、修复 LoRA，共 3 个权重。
- `planning_v2`：早期 Flan-T5 逐步策略。
- `planning_v3/multi`、`planning_v3/single`：多最优动作与单动作训练对照，包含此前使用的逐步策略权重。
- `rlvr_reinforce_001`：早期 REINFORCE LoRA，底座为已发布的 `artifacts/gvr/models/generation`。这是旧 REINFORCE，不是新 GRPO。

原始 GVR 的 3 个权重已经在 `artifacts/gvr/models`，不重复备份。此目录不重新宣称旧实验使用了本轮相同的预算、代码或冻结规范；历史代码和归档标签见项目导航，成绩应与各自原始报告对应。

```bash
git lfs pull --include="artifacts/historical_models/**,artifacts/gvr/models/**"
```

权重、tokenizer、模型配置一并保留，LoRA 的底座路径已迁移到对应发布目录。使用 Transformers 加载完整底座，再用 PEFT 加载相应 adapter。`manifest.json` 记录每个文件的原始/导出哈希，权重字节不变，机器路径和设备型号已模糊化。本目录只保留紧凑历史记录，不重复打包旧版逐步 session 树；本轮完整轨迹位于新实验目录。
