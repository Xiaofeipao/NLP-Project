# GVR 权重与实验产物

此目录保存实际完成的三阶段训练产物：冻结生成底座 G、验证 LoRA V、修复 LoRA R，以及原始预测、CPU 重算指标和日志。ID／OOD 候选修复为 12%／6%，四种端到端流程均为 0%；验证器全部预测 `valid`。它们保留为负结果，未替换已有逐步规划器。

## 获取权重

在仓库根目录执行 `git lfs pull`。三个 `.safetensors` 文件均由 Git LFS 管理；生成底座约 308 MB，两套 LoRA 各约 1.39 MB。LoRA 必须配合本目录的 `models/generation/`，不能直接替换成原始预训练模型。

预训练来源为 [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)，固定 revision 为 `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`。训练、任务样本和预算设置见仓库配置与 [实验报告](../../docs/gvr_gpu_run.md)。

## 文件说明

- `models/generation/`：完整的微调生成底座、配置和分词器。
- `models/verification/`、`models/repair/`：独立 LoRA 权重、适配器配置及底座身份哈希。
- `training/<stage>/{pilot,train}.json`：训练记录；移除了设备型号，来源路径统一为相对路径。
- `evaluation/<split>/metrics.json`：历史运行原始指标，开发集旧分流口径有误。
- `evaluation/<split>/metrics_recomputed.json`：按新审计代码重算的指标，应优先引用。开发集配额为 Verify 100、Repair 60、Generate 40，修复为 7/60（11.7%）。ID／OOD 各 200 个网格、100 个无效候选。
- `candidate_traces.jsonl`、`pipeline_traces.jsonl`：原始逐条预测与评分，保留实际调用次数。
- `sessions.zip`：逐条会话的完整状态、manifest 与哈希链日志；逐字节保留，省略运行期 `.lock` 文件。解压后仍可核对历史日志链。
- `verification.json`：原运行保存重载检查；只在一个固定开发样本上比较关闭适配器与独立 G 的输出，不代表执行过新代码的完整 `verify_dev`。
- `gpu_ledger.json`：保留历史累计预算的完整账本快照。
- `historical_summary.json`：原运行摘要；开发集旧口径以重算文件为准。
- `historical_run.patch`：原训练／推理运行相对 `96b9320` 的代码差异。当前仓库还包含后续控制器和评测修复，历史轨迹不能视为重新运行这些修复后的结果。
- `manifest.json`：每个导出文件的 SHA256、来源 SHA256、转换说明及权重身份。

## 不加载模型重算结果

准备官方执行器后，在仓库根目录执行：

```powershell
python scripts/fetch_official.py
python scripts/run_gvr.py recompute --split dev --evaluation-dir artifacts/gvr/evaluation/dev --out outputs/gvr_audit/dev.json
python scripts/run_gvr.py recompute --split test_id --evaluation-dir artifacts/gvr/evaluation/test_id --out outputs/gvr_audit/test_id.json
python scripts/run_gvr.py recompute --split test_ood --evaluation-dir artifacts/gvr/evaluation/test_ood --out outputs/gvr_audit/test_ood.json
```

这些命令只重算已保存的预测，不产生新的模型输出，也不重演新控制器。

## 使用导出的模型重新评测

安装 `requirements.txt` 和兼容的 CUDA PyTorch 后，先取得 LFS 权重及官方执行器。新克隆尚无账本时，应从本目录复制历史账本后再继续计费；已有账本保留原文件，不覆盖或清零：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
if (-not (Test-Path outputs/gpu_ledger.json)) {
    Copy-Item artifacts/gvr/gpu_ledger.json outputs/gpu_ledger.json
}
$env:HF_HUB_OFFLINE = '1'
python scripts/run_gvr.py verify --protocol artifacts/gvr/protocol.json --config artifacts/gvr/config.json
python scripts/run_gvr.py evaluate --split dev --protocol artifacts/gvr/protocol.json --config artifacts/gvr/config.json
```

协议使用仓库内导出权重，新的运行输出到 `outputs/gvr_replay/`。证据位默认关闭；准备新训练时仍应核对预算和训练所需预训练缓存。此次发布没有启动新训练或 GPU 评测。
