# 按方法组织的入口

main 保留完整项目；分支只承载未合并实验。此目录是可执行方法层，公共实现仍放在 ppnl，不复制算法。

- baseline：全量多任务基线训练与评测。
- gvr：全量模型的生成、验证、修复流程。
- harness：模型与工具循环、预算、日志、恢复。
- lora：G 底座及独立 V/R adapter 训练。
- rl：REINFORCE 或 GRPO 的执行器奖励训练。
- policy：逐步动作模型及搜索对照。
- prefix：保留前缀与重启的消融。

GVR 是任务流程、LoRA 是微调方式、Harness 是运行控制、GRPO 是训练算法，它们可以组合，并非互斥模型。

## 运行

在仓库根目录执行；`--` 后参数原样转交历史入口：

```powershell
python -m methods.baseline --describe
python -m methods.baseline run -- --help
python -m methods.baseline run -- --config config.json train --model t5_small
python -m methods.gvr run -- --help
python -m methods.harness run -- --help
python -m methods.lora run -- preflight --protocol config.gvr.json
python -m methods.rl run -- --help
python -m methods.policy run -- --help
python -m methods.prefix run -- --help
```

各目录 manifest.json 是命令、配置路径及实现模块的统一映射。配置引用原 JSON，不复制参数。`--describe` 不加载 CUDA。退出码和原参数检查保持不变；跨方法批量实验仍由 scripts/run_gpu_campaign.py 等编排脚本管理。

## 兼容与比较

原 config*.json、configs/ 和 ppnl/scripts 文件路径保留，避免破坏历史导入与源码冻结。公共执行器 core.py、数据 data.py、指标 evaluation.py、预算 modeling.py、冻结 experiment_freeze.py 共用，不各写一套。旧命令仍有效。

configs/rlvr.json 的历史默认是 REINFORCE；GRPO 需独立 spec 明确 algorithm=grpo。统一入口不改变训练预算、数据划分、开发选模或测试冻结，也不自动使不同 checkpoint 的结果可直接比较。

tree-verifier-reachability 与 gvr-constrained-repair-study 仍是未合并实验，不包含在本次 main 整理中。已删除的方法名称分支不代表删除对应代码。
