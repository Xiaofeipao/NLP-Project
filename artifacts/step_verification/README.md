# 单步 G/V 实验归档

| 路径 | 内容 |
| --- | --- |
| [v2 训练数据初始核对](v2_training_data_review_20261120/) | 原 v2 数据的审核脚本与统计 |
| [早期语义数据](step_semantic_v3/) | G/V 数据模板、样例和审计 |
| [统一单步验证](step_semantic_single/) · [最后一步输入](step_last_move/) | 中间数据版本及复现记录 |
| [类别平衡与损失历史对照](step_g_balanced/) | 原 token 平均、首 token 等权、续训和五分类头的证据 |
| [同状态四方向数据](step_g_paired_actions/) | 最终构建核对、载入验证和实验说明 |
| [两组各三轮](step_g_paired_actions/retrain_3ep/) | 逐轮开发预测、固定路径预测、四方向诊断及五分类梯度核对 |
| [交付包](../../deliverables/) | 各阶段的独立数据包 |

## 三轮实验权重

两组共六个原始 `model.safetensors` 文件，存放在台式机 GPU 的 `retrain_3ep` 运行目录。[检查点索引](step_g_paired_actions/retrain_3ep/checkpoints.json) 记录每个文件的相对路径、字节数和 SHA-256。[检查点配置与分词器](step_g_paired_actions/retrain_3ep/checkpoints/) 与索引相对应，可把同名原始权重复制进对应目录供本地加载。

[任务定位、样本和完整复现](../../docs/step_verification_gv.md)

