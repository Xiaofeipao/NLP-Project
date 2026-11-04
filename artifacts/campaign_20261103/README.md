# 三模型 GPU 实验产物

本目录发布 2026-11-04 完成的三模型实验：33 个有效权重文件、99 份开发/测试报告、训练与重载记录、RL rollouts、逐案例预测和压缩执行日志。完整分析见 [实验报告](../../docs/gpu_campaign_20261103.md)，校验与数据版本见 `manifest.json`。

## 下载

模型 `.safetensors` 和执行轨迹 `.zip` 使用 Git LFS。安装 Git LFS 后，在仓库根目录运行：

```bash
git lfs install
git lfs pull --include="artifacts/campaign_20261103/**"
```

普通网页显示的三行 pointer 不是模型本体；下载需要 LFS 对象。所有权重约 6.35 GiB，另需少量报告和轨迹空间。发布工作区可以保留 pointer，原始训练权重另存本地；其他克隆可以按需下载指定模型。

## 内容与加载

- `baselines/<model>/train/best`：三个全量多任务基线。
- `gvr/<model>/<vanilla|structured>/generation/train/best`：六个全量 G；同目录族 `verification/train/best/verification` 和 `repair/train/best/repair` 是独立 LoRA。`status` 仅提供辅助损失 V，共享选定格式的 G/R。
- `policies/<model>/<model>/train/best`：三个独立逐步策略，不是 GVR 或 RL adapter。
- `rl/<model>/<reinforce|grpo>/best`：六个 RL LoRA；必须先加载该实验对应的 G 底座。各模型的 `*_format_selection.json` 记录选定输入格式。
- `full_harness/`、`gvr/*/*/evaluation/`、`policies/`、`rl_evaluation/`、`prefix/`：开发/ID/OOD 的报告与预测。`sessions.zip` 为对应 `sessions/` 目录的无损归档，排除临时锁文件。
- `source/`：与本轮评测对应的 `ppnl/` 和 `scripts/` 源码快照。数据位于仓库 `data/derived` 和 `data/policy_v3`，版本哈希列入清单，不重复复制数据。

全量模型可用 Transformers 的 `AutoModelForSeq2SeqLM.from_pretrained` 加载；LoRA 使用 PEFT，将相应 G 作为底座后调用 `PeftModel.from_pretrained`。adapter 配置中的底座路径已改为本发布目录中的相对路径。tokenizer 与必要配置一并提供。

## 可复核范围

权重字节不变，LFS OID 等于原权重 SHA-256。清单记录普通文件的原始和导出后哈希；机器型号、绝对路径等元数据已模糊化，adapter 底座路径已重定位。历史冻结报告仍记录原运行身份，因此脱敏/迁移后的副本不是可原样继续测试的冻结锁；新运行必须重新做开发核验与冻结。原始本地报告未被修改。

执行轨迹 zip 成员原样保存，可解压到其同级 `sessions/` 目录。清单记录归档哈希；session 内部日志仍可进行原来的完整性检查。部分配置/记录中的 `outputs/campaign_20261103` 指原运行逻辑目录，复现时应明确迁移输出和 checkpoint 路径，不要覆盖本目录的历史证据。

低分模型作为消融保留。GRPO 等结果不能引用逐步策略的 200/200；ID/OOD 网格配对、只有候选损坏分布不同。测试是单随机种子、重复使用的探索性测试，不是新盲测。

未包含缓存、损坏保存、重复临时副本、未选中的中间权重。失败保存的日志保留用于说明恢复过程。导出程序为 `tools/export_campaign.py`，置于冻结源码范围外，避免改动本轮源码哈希。
