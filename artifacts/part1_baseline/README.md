# Part 1 · G/C/V 均匀穿插基线

| 交付内容 | 文件 |
|---|---|
| 完整指标 | [开发集选型、单错测试、双错验证、生成与修复](Part1_基线结果.md) |
| 第一组错误分析 | [G 方向偏差、首错与修复失败](第一组_错误分析.md) |
| 逐题与复现 | [预测及评分](scores/) · [两次 GPU 重跑回执](rerun_confirmation.json) · [离线核对](validation.json) |
| 权重 | [Flan-T5-base](models/expanded/flan_t5_base/) · [BART-base](models/expanded/bart_base/) |
| 数据 | [4,000 个训练状态](data/state_ids.json) · [协议与哈希](data/protocol.json) · [单错测试金标](../../data/random_errors_20261123/single_error_200.jsonl) · [双错验证金标](../../data/random_errors_20261123/double_error_200.jsonl) |

单错测试集和双错验证集各 200 条，来自 hope_dataset 的 d88f85b 修订。单错修改了 9 道题，双错修改了 19 道题，共替换 29 处大写方向词。Flan 与 BART 仍用原 100 条开发题选型。[hope_dataset 数据修订](https://github.com/Xiaofeipao/NLP-Project/commit/d88f85b872560f9313ac4319cdd5fa86151c0780) · [数据协议与哈希](data/protocol.json) · [替换清单](../../data/random_errors_20261123/illegal_word_replacement_20261126.json)

## 下载与验证

在仓库根目录运行以下命令。Git LFS 会按当前分支下载这两个检查点，共约 1.55 GB。运行环境见[训练、推理和复现说明](../../experiments/part1_baseline/README.md)。

```powershell
git switch Hope
git lfs pull --include="artifacts/part1_baseline/models/**" --exclude=""
python scripts/fetch_official.py
python -m experiments.part1_baseline.audit --results --work artifacts/part1_baseline
python -m experiments.part1_baseline.diagnose
python -m experiments.part1_baseline.predict --task verify --grid examples/grid.json --candidate "left up"
```

训练后的检查点来自 Google Flan-T5-base 和 Meta BART-base。起始模型的版本、训练数据与最终权重的 SHA-256 见[冻结协议](data/protocol.json)和[选型记录](selection.json)。[Flan-T5-base 原模型](https://huggingface.co/google/flan-t5-base/tree/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2) · [BART-base 原模型](https://huggingface.co/facebook/bart-base/tree/aadd2ab0ae0c8268c7c9693540e9904811f36177) · [Apache 2.0 许可证](models/LICENSE)
