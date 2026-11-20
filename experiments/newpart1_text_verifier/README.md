# NewPart1 文本验证扩展

本扩展接在 StepVerification-v1-v2@9eeb494 后，继承 NewPart1@6939390 的损失与状态策略代码。实验均使用本组独立重训的 NewPart1 Flan-T5-base，权重 SHA256 为 c28ae2c1b6e4b94c16030a9d42c5278134c808c292596c97af9d42938804aae0。模型与适配器实体保存在远端个人输出目录。

## 结果

| 同一个局部规则 R，错误候选各 100 条 | 单错修好 | 双错修好 | 对共享 100 条有效候选误拒 |
| --- | ---: | ---: | ---: |
| 原数据、GitHub 原模板 | 10 | 26 | 0 |
| 同预算精简 Prompt | 10 | 26 | 0 |
| 五类等权损失 | 33 | 37 | 100 |
| 保持 valid/invalid 各半的二级重加权 | 10 | 26 | 0 |
| 二级均衡数据 | 30 | 36 | 97 |

精简 Prompt 平均减少 15.9% 输入 token，未增加固定错例修好数。均衡和格式增强也未带来稳定的文本 V 语义收益。真实 G 失败的原格式修复率仍为 0。成对语义测试更改目标、障碍、边界或停止位置，各模型都没有在任何一类中做到一对两边同时答对。报告同时列出误拒和规则 R 的贡献。

实验是固定底模后的 V 专用 LoRA：相同训练预算下比较改动。GitHub 原 v1 从预训练底模联合全量微调 G/V，两者权重和训练方法各自记录。已有测试集反复使用，结果为探索性。

[Prompt 实验报告](../../docs/newpart1_prompt_study.md) · [类别与捷径报告](../../docs/newpart1_shortcut_study.md) · [文件散列清单](../../deliverables/newpart1_text_verifier/manifest.json)

## 目录

训练脚本沿用实际 GPU 运行源码，保留字节级哈希。run_prompt_study.py 比较三模板以及交叉推理。run_shortcut_study.py 比较重加权、均衡数据和格式扰动。run_hierarchical_study.py 是依据首轮开发退化，补做的二分类质量各半实验。prompt_templates.py 提供可直接调用的三种提示。

audit_prompt_study.py 和 audit_shortcut_study.py 在 CPU 上重放公开证据，核对标签、验证指标及修复指标。audit_lexical_shortcuts.py 复算完全不读地图、只看动作词的多数标签基线。

## CPU 重放

从仓库根目录运行。CPU 环境只需要 Python 标准库；这些复算脚本不加载训练模型。PowerShell 中先把仓库根目录加入导入路径。

```powershell
$env:PYTHONPATH = "$PWD"
python experiments/newpart1_text_verifier/audit_prompt_study.py deliverables/newpart1_text_verifier/prompt
python experiments/newpart1_text_verifier/audit_shortcut_study.py deliverables/newpart1_text_verifier/shortcut
python experiments/newpart1_text_verifier/audit_shortcut_study.py deliverables/newpart1_text_verifier/hierarchical
python experiments/newpart1_text_verifier/audit_lexical_shortcuts.py outputs/lexical-recomputed
```

CPU 重放结果为 Prompt 的 18,328 条候选记录、86,000 条局部回复数量；两轮数据实验合计 49,203 条候选记录和 13,200 条探针回复。数据标签和源码散列另有核验。

## GPU 重新训练

使用项目训练依赖、PEFT、可用的 CUDA GPU 和冻结的官方执行器。NewPart1 底座先按 [单模型训练说明](../../state_baselines/README.md) 生成，或使用与本报告散列一致的现有检查点。不同重训权重可能得到不同分数。

以下 Bash 命令从仓库根目录运行。CHECKPOINT 指向完整模型与 tokenizer 所在目录；EXECUTOR 指向冻结的 executor-point-sg.py。RUN 指向新的个人输出目录，LEDGER 采用累计 GPU 台账。预算检查在进程之间串行执行。

```bash
export PYTHONPATH="$PWD/experiments/newpart1_text_verifier:$PWD"
CHECKPOINT=/path/to/newpart1/train/flan_t5_base/best
EXECUTOR=/path/to/executor-point-sg.py
RUN=outputs/newpart1-text-extension
LEDGER=outputs/shared_gpu_ledger.json
mkdir -p "$RUN/frozen_inputs"
tar -xzf deliverables/newpart1_text_verifier/prompt/evidence.tar.gz -C "$RUN/frozen_inputs"

python experiments/newpart1_text_verifier/run_prompt_study.py \
  --checkpoint "$CHECKPOINT" --pool "$RUN/frozen_inputs" \
  --evidence deliverables/step_verification_v1_v2_evidence.zip \
  --executor "$EXECUTOR" --ledger "$LEDGER" --out "$RUN/prompt"

python experiments/newpart1_text_verifier/run_shortcut_study.py \
  --checkpoint "$CHECKPOINT" --control "$RUN/prompt/adapter_github" \
  --previous "$RUN/prompt" --evidence deliverables/step_verification_v1_v2_evidence.zip \
  --executor "$EXECUTOR" --ledger "$LEDGER" --out "$RUN/shortcut"

python experiments/newpart1_text_verifier/run_hierarchical_study.py \
  --checkpoint "$CHECKPOINT" --control "$RUN/prompt/adapter_github" \
  --previous "$RUN/prompt" --evidence deliverables/step_verification_v1_v2_evidence.zip \
  --executor "$EXECUTOR" --ledger "$LEDGER" --out "$RUN/hierarchical"
```

输入解包复用存档的候选，GPU 预测由重新训练的模型完成。输出目录要求此前不存在。三份训练脚本都固定原逐步扫描：合法但未到目标可以继续检查后续动作，非空路线末尾没有额外 STOP。局部规则 R 在首错前缀后最多回退两步、桥接三步。它与增强 R36 分别报告。

原运行输出目录为 /home/apulis-dev/outputs/NLP-Project-NewPart1-6939390 下的 prompt-study-20261119-batched、shortcut-study-20261119、shortcut-hierarchical-20261119。所有新权重仍在这些个人目录中。
