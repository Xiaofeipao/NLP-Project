# 本地 GPU 复现与同伴对比（2026-11-11）

发布更新：本次代码与结果纳入 main；此前“未 push”描述运行时状态。公开证据包为 gvr_local_evidence.zip，内含 26 份结果、逐例轨迹、冻结和预算记录，并记录原始/导出 SHA-256；本机路径已脱敏，导出锁不能代替原本地执行锁。普通解码保持默认，两个研究分支保留。后续研究候选见 gvr_next_methods.md，尚未执行。

按 Ponytail 的复用原则执行既有 smoke、开发、冻结、ID/OOD 入口，未新增训练或修改解码算法，未 push。精确恢复同伴 Flan-T5-small 权重，SHA-256 为 0b7be3b12238b363ba8c1a9304f687c6ae6f9edbecceff0076cebe7c260d62fa。本地源码独立冻结于 outputs/gvr_repair_study/freeze.json，不复用同伴的锁。

开发集全部指标与同伴一致：C4 生成 36/40、候选修复 60/60。测试端到端 B4/C4/U4/H4 分别为 0/184/191/200（分母 200），C4 最优率 43%。候选修复 E1/B4/C4/U4/H4 的 ID 成功数为 19/25/98/95/100，OOD 为 7/8/94/95/100（分母各 100）。旧四流程仍为 0/200。验证指标也一致。

排除耗时字段后，汇总指标在绝对容差 1e-12 下全部一致；平均路径效率的极小浮点尾差约为 1e-16。逐例输出并非全部字节相同：B4 在 ID 生成、ID 候选、OOD 候选各有 1 条路径不同，但未改变汇总指标。C4/U4/H4 在上述逐例路径比较中全部一致。差异详情、软件版本和对照原值见 gvr_local_gpu_comparison.json。不能据此宣称跨硬件运行逐位一致，也未确定 B4 差异的具体数值来源。

新 methods.gvr 入口另做同网格 ordinary/constrained 两次真实 GPU 验证：ordinary 一次修复后仍失败；constrained 到达终点，decode_status=complete。会话分别为 outputs/full_finetune_gvr/peer_repro_ordinary 和 peer_repro_constrained，均使用同一历史 Flan-T5-small 权重。

首次 ID 测试中途遇到 CUDA unknown error；失败账本和会话保留于 test_id_failed_cuda_20261111。确认 CUDA 张量操作恢复后从头重试成功，失败耗时没有从预算账本删除。完整成功输出位于 outputs/gvr_repair_study/test_id 和 test_ood；OOD 复用 ID 的同地图生成轨迹，不视作另外 200 张独立地图。

本报告只验证本地接入能复现同伴效果。H4 规则对照仍优于 C4，测试集仍属重复使用的探索性评估，不据此追加调参或升级默认模式。普通解码仍默认。
