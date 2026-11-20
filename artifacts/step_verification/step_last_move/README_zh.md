# 单步 G/V 数据

训练集有 2,000 条 G、2,000 条 V。开发集各 100 条。两种任务共用状态字段（地图尺寸、当前位置、目标、障碍）。每条 G 从当前状态选择下一步，训练目标包含所有能让最短距离减 1 的方向。每条 V 只接收动作之前的状态和一个真实候选词。当前缀已通过验证，最后一步得到 `valid` 就可以输出路径。

| V 的唯一输出标签 | 含义 | 训练条数 |
| --- | --- | ---: |
| `valid` | 当前动作合法且到达目标 | 400 |
| `goal_not_reached` | 当前动作合法，执行后还需继续走 | 700 |
| `obstacle_collision` | 目的格有障碍 | 450 |
| `out_of_bounds` | 目的格超出地图 | 250 |
| `invalid_action_token` | 候选词不属于四个方向 | 200 |

真实样本从 `(0,2)` 去 `(1,2)`，`(0,1)` 有障碍。同一个 G 状态对应四条独立的 V 输入。`down` 到达，`right` 可走但未到，`left` 撞墙，`up` 越界。训练集中有 160 个 G 状态以这种方式覆盖四个方向。合计 765 个 G 状态与 1,413 条 V 样本复用，其中 140 个 G 状态同时展示了最优方向和可走的非最优方向。

## 生成与验证接口

生成时，外层程序保存已经接受的前缀和当前位置，G 选动作，V 验该动作。V 返回 `valid` 时可以输出。返回 `goal_not_reached` 时，程序更新位置并继续。首错定位、回撤和重选可由外层控制流处理，数据中只有 `g_next` 和 `v_step` 两个训练任务。样本的 `source`、`donor_id` 和 `state_id` 用于复查来源与关联，模型只读取已渲染的 `input`。

已有候选路径由 `path_steps` 按动作拆成多个“动作前状态 + 当前动作”。`verify_plans` 用相同的单步接口依次判定。非法移动报第一个出错的动作步号。若最后一个真实动作合法但未到，整条路径沿用项目约定，报动作数加一。空路径由程序检查初始位置。扫描器向模型发出的每个请求都包含一个真实动作，训练和推理均无 `Action: null`。

## 捷径检查

旧版的 250 条“未到目标”训练标签全部出现在空动作上。711 个共用 G 状态中只有 16 个提供至少两个实际方向。新版所有“未到目标”都来自真实合法移动，有 226 个共用 G 状态提供至少两个实际方向。下面的固定规则与查表结果衡量数据里有哪些可利用的线索。模型准确率需要正式微调后测量。

| 可见信息 | 新版开发集的 V 判对数 |
| --- | ---: |
| 固定模板，按训练集多数类猜 | 35/100 |
| 只看动作词，按训练集同词多数类猜 | 40/100 |
| 只看字符长度、动作词长度和障碍数量 | 35/100 |
| 读取动作、坐标、网格尺寸和目标，遮住全部障碍 | 78/100 |

非法词可以按四词词表识别，越界可以由坐标和方向算出，这两条是任务本身的正确规则。遮住障碍的规则漏掉了开发集中全部 22 次撞墙。为单独检验是否读取障碍位置，训练中有 40 组地图对照。每组固定起点、目标和障碍数量，将一面墙从一个候选方向的目的格移到另一个方向的目的格，每个地图上的每个方向单独成一条 V。遮住障碍只能答对 80/160。开发集另有两组，结果为 4/8。其余对照含只改目标、只增减目的格障碍、只改边界位置的样本。开发集的 10 条非法词中有 5 条使用训练未见的拼写。

## 文件和复现

解压数据与代码包至 `StepVerification-v1-v2` 根目录。训练集位于 `data/step_last_move/train.jsonl`，`audit.json` 记录执行器、分区和成对条件验收，`shortcut_audit.json` 记录各限制信息基线的逐类结果。训练和开发布局分别为 652 张和 76 张。两个 200 条的固定测试文件与 v2 原文件逐字节一致。用正确的单步参考标签回放，400 条候选的整条判定、首错步号和原因全部对齐。

```python
from step_data_v3.last_move import prepare_training, batch_loss, verify_last, verify_plans

features = prepare_training(runner, "data/step_last_move")
# 训练循环使用 batch_loss(runner, batch)。线上单次验证使用动作执行前的状态。
label = verify_last(runner, state_before_last, last_action)
```

G 保留最优动作集合损失，V 使用一个短文本标签的交叉熵。上版 `data/step_semantic_single` 可供重建与对照。重建命令从仓库根目录运行。

```powershell
$executor = 'D:/UserData/Documents/ChatGPT/NLP_Project/vendor/ppnl/executor-point-sg.py'
python -m step_data_v3.paired_steps --executor $executor --out data/step_last_move_rebuilt
python -m unittest step_data_v3.test_last_move
```
