# 配对双错误泛化测试

仅用于可选额外评测，不参与训练、开发或超参数选择，也不并入必做6×6 ID/OOD的数量。

- `test_7x7_double.jsonl`、`test_dense_6x6_double.jsonl`各200条：100条有效控制、100条双编辑非法候选。
- 与`generalization_v1`单错集逐环境配对，地图、起终点及有效控制路径保持不变；`parent_id`连接原样例，双错ID加`_double`。
- 固定seed 20261112，从原最短路径的两个不同原始位置执行替换、删除、插入或非法词编辑；保证token编辑距离为2，执行器判非法。两次编辑不等于能观察到两次执行失败，诊断仍定位首错。
- 两种地图合计仍是原来的400个独立环境；单错、双错共800条候选记录，不能称800张独立地图。布局隔离继承原泛化集，不新增训练/开发地图。
- CPU构造耗时、源文件/生成器/数据哈希见`manifest.json`；GPU推理另记在实验输出目录的`gpu_ledger.json`。原单错文件不修改。

复现：`python -m scripts.build_generalization_double --output outputs/generalization_double_rebuild`。

评测：`python -m part3.run_generalization_r_pair --out outputs/generalization_r_pair/new_run`。
固定旧R为按反馈保留前缀再贪心续走，新R为回退36＋Top3＋软循环惩罚＋最终删环。全部测试仅报告，不选新版本。
