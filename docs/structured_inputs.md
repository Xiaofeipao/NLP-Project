# 结构化输入候选与配对消融

## 本轮实现

借鉴参考项目将自然语言地图整理成显式字段的思路，新增可选 `prompt_format: structured`，没有复制其训练代码。它只改变模型看到的环境表示：网格尺寸、零基坐标规则、起点、终点、障碍物。候选路径和反馈仍按原任务要求提供，生成输入不含参考路线，验证输入不含金标诊断。

默认 `vanilla` 模板保持原样。固定数据文件不重写：训练先验证原记录，再在内存中转换输入，保留相同ID、目标、配额与布局隔离。验证和测试时由同一个格式化函数构造输入，自验证修复继续使用模型原始反馈，不追加执行器正确答案。

本轮只改变输入格式，坐标跟踪CoT暂未启用；它还需要独立的训练目标、严格输出解析和评分协议，不能直接从输出中筛掉错误文字后宣称成功。参考项目的多尺度数据、15/40轮训练和RL损失均未移植。

## 配置及兼容性

- `configs/gvr_format_vanilla.json`：使用原输入格式的对照。
- `configs/gvr_format_structured.json`：结构化输入候选。

两份配置使用相同数据、阶段损失、学习率、轮数、seed及解码设置，分别写入独立新目录。它们基于当前代码进行配对，不能仅拿新结构化运行与旧代码训练的历史结果比较并归因于输入格式。两份配置的训练证据位默认关闭；是否执行仍需核对完整累计样例、遍历和GPU预算，不自动增加课程配额。

新生成底座保存 `prompt_format.json`，LoRA身份记录保存对应格式；加载时拒绝结构化协议搭配旧自然语言底座或不匹配的adapter。结构化输入也加入pilot训练签名；已有pilot不能跨格式复用。`verify_dev.json` 记录输入格式，冻结时检查开发集证据的格式与所选模型相同。原历史权重未附格式元数据时按vanilla处理。

## 已做的CPU验证

```powershell
python scripts/preview_prompt_formats.py --tokenizer outputs/gvr_next/generation/train/best --out outputs/prompt_formats_preview_20261103.json
```

该命令只打开train/dev，加载本地tokenizer，不加载模型、不调用CUDA、不执行训练。输出路径必须不存在。

实测训练4000条、开发200条，任务配额不变；最长输入token数分别为：训练vanilla159、structured147，开发vanilla155、structured143，均低于512上限。完整95项CPU回归测试通过。新增测试覆盖旧模板兼容、输入不泄露目标／验证金标、输入转换不修改原数据、未知配置及checkpoint格式不匹配拒绝。

这些是长度与实现检查，不是模型质量评测；目前没有结构化版本的成功率。

## 后续运行流程

未来运行时，先为新实验核对账本与证据，再分别使用对应协议执行 `scripts/run_gvr.py` 的 G→V→R pilot/train、verify 和开发集evaluate。每阶段都必须新训练，不能把旧LoRA挂到新的生成底座。按开发集选定格式后，用 `scripts/freeze_experiment.py` 创建新锁，test_id/test_ood评测显式传入 `--freeze`。

原始数据、原报告和旧权重保留；此次源码变更会使旧冻结锁校验失败，应保留旧锁作为历史证据，不能改写它绕过漂移检查。由于历史测试已用于分析，后续复用测试仍标记探索性结果。

参考来源：[PPNL参考项目](https://huggingface.co/EnjiXiong/AIAA4051-FinalProject-PPNL/blob/63830e4f388b47c5a32d6bb833cb53683e3e96cd/grid-path-planning/data_utils.py)。仅借鉴输入字段化思路。
