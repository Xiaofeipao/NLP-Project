# 单步验证数据

训练集共 4,000 条。G 的 2,000 条输入、代表动作和完整最优动作集合沿用上一版。V 的 2,000 条都用同一个模板，每条输入包含地图、当前位置、目标和一个动作。动作字段取 JSON 的 `null` 时表示扫描器发出的 `STOP`。候选路径中拼写出来的 `"STOP"` 按普通词检查。

| V 输出 | 含义 | 训练条数 |
| --- | --- | ---: |
| `valid` | 动作可执行，或在目标处停止 | 1,000 |
| `obstacle_collision` | 撞墙 | 250 |
| `out_of_bounds` | 越界 | 250 |
| `invalid_action_token` | 动作词不在上下左右四个方向中 | 250 |
| `goal_not_reached` | 停止时尚未到达目标 | 250 |

V 的唯一训练标签是 `target`。例如，从 `(0,0)` 去 `(0,2)`，中间没有障碍。第一步 `right` 的标签是 `valid`。如果路径到这里就结束，随后 `STOP` 的标签是 `goal_not_reached`，扫描器据此报告全局第 2 步提前结束。途中经过目标也会继续检查后续动作。G 的多方向集合损失沿用 v2，V 对一个短文本标签计算逐样本平均的 token 交叉熵。

开发集包含 100 条 G 和 100 条单步 V。训练集覆盖 652 张布局，711 个 G 状态与 820 条 V 共享。两个固定测试集各含 200 条完整候选，文件与 v2 的原始测试数据逐字节相同。用参考单步标签运行扫描器，400 条的有效性、首错步号和原因全部匹配原标签。这个结果检验标签设计和扫描逻辑，模型指标需在微调后测量。

## 使用

解压到 `StepVerification-v1-v2` 仓库根目录。数据位于 `data/step_semantic_single`。训练与推理使用 `step_data_v3.single` 的接口。

```python
from step_data_v3.single import prepare_training, batch_loss, verify_plans

features = prepare_training(runner, "data/step_semantic_single")
# 在训练循环中把 runner.collate(...) 返回的批次传给 batch_loss(runner, batch)。
# 微调后，verify_plans(runner, grids, candidates) 输出带全局首错步号的判定。
```

重建时使用 v2 的 `data/derived`、上版 `data/step_semantic_v3` 和项目的官方执行器。若需先重建上版，可以运行随包提供的 `step_data_v3.build`。

```powershell
$executor = 'D:/UserData/Documents/ChatGPT/NLP_Project/vendor/ppnl/executor-point-sg.py'
python -m step_data_v3.prepare_single --executor $executor --out data/step_semantic_single_rebuilt
python -m unittest step_data_v3.test_single
```
