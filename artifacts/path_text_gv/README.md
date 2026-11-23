# Flan-T5 完整动作历史 G/V

本目录保存 2026 年 11 月 24 日完成的台式机 RTX 3050 实验。G 读取地图、起点、目标和迄今的全部动作，输出任一最短路径的下一步方向。V 读取同样的信息，动作列表包含待判的最后一步。它自行生成 `valid` 或 `invalid | step=<index> | reason=<error_type>`。规则执行器用于离线制作标签，模型训练和推理使用冻结文本。

## 数据、代码与结果

| 内容 | 位置 |
|---|---|
| 训练、开发、来源和数据核查 | [data/](data/) |
| 封存的单错误、双错误、历史对照及 G/V 联合评测输入和金标 | [heldout/](heldout/) |
| 串行作业、训练、损失和评测代码 | [code/](code/) |
| 复算后的指标、混淆矩阵及逐题原始输出 | [实验结果](results/结果汇总.md) |
| 核查与提交记录、归档输入校验值 | [核查记录](核查与提交记录.md) · [SHA-256 清单](package.json) |

4,000 个训练 case 对应 4,000 条 G、16,400 条 V。每个 case 覆盖四方向，400 个 case 另含一个非法词候选。三种 V 损失共用第 1 轮，各自训练第 2、3 轮。测试前按 V 五类整行正确率等权平均选检查点，三组均选中第 2 轮。

| 入选检查点 | V 末步开发整行正确 | V 五类等权整行正确率 | 单错整路首错全对 | 双错整路首错全对 | 单错截到首错后答对 | 双错截到首错后答对 |
|---|---:|---:|---:|---:|---:|---:|
| 字段加权 | 103/820 | 38.27% | 9/200 | 7/200 | 39/200 | 70/200 |
| DPO | 226/820 | 13.54% | 18/200 | 0/200 | 21/200 | 14/200 |
| MRT | 10/820 | 10.00% | 7/200 | 10/200 | 10/200 | 21/200 |

末步测试以金标截取合法前缀及待判末步。整路测试要求模型在一次生成中自己报出首错。固定地图、路径长度及末动作，仅改变历史时，七个正式检查点中最高成绩为字段加权第 3 轮的 2/171 对两条均答对。三组在 40 条 G/V 联合开发题上各得 0/40。G 提议下一步，V 报合法未到就继续，报到达、撞墙、越界或给出错误格式则停止。每题最多生成 24 步。这次没有回撤或重试。原始训练与测试日志见 [results/logs/](results/logs/)。

## 下载完整权重

权重正在上传至 Git LFS。当前提交包含检查点目录、运行配置、分词器和 SHA-256 记录。

模型目录包含 Hugging Face 配置、分词器、完整 `model.safetensors` 和对应的 SHA-256。分支保存 [共同第 1 轮](results/warmup/epoch1/model/)、字段加权 [第 2 轮](results/field/epoch2/model/) 和 [第 3 轮](results/field/epoch3/model/)、DPO [第 2 轮](results/dpo/epoch2/model/) 和 [第 3 轮](results/dpo/epoch3/model/)、MRT [第 2 轮](results/mrt/epoch2/model/) 和 [第 3 轮](results/mrt/epoch3/model/)，以及 [烟雾测试](results/smoke/weights/) 权重。每个权重 990,345,064 字节。发布的是推理检查点，优化器中间状态留在训练机。

安装 Git LFS 后，可以只下载选中的权重。例如在 PowerShell 中下载字段加权第 2 轮。

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone --branch Luck --single-branch https://github.com/Xiaofeipao/NLP-Project.git
Set-Location NLP-Project
git lfs pull --include="artifacts/path_text_gv/results/field/epoch2/model/model.safetensors"
```

使用 `AutoModelForSeq2SeqLM.from_pretrained` 和 `AutoTokenizer.from_pretrained` 读取该模型目录。本次训练使用 Python 3.12.10、PyTorch 2.11.0+cu128、Transformers 5.20.0、Tokenizers 0.23.3 和 Safetensors 0.8.0。若从基座复跑，可在支持 CUDA BF16 的 Windows 环境下，用冻结的 Flan-T5-base 原模型目录运行以下命令。基座 revision 是 `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`，使用 `settings.json` 和 `manifest.json` 核对模型与 tokenizer 哈希。

```powershell
python artifacts/path_text_gv/code/run_serial.py --root artifacts/path_text_gv --base <Flan-T5-base本地目录>
```
