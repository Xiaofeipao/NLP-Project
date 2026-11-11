# GVR 双解码本地接入

2026-11-11：基于 main 的方法入口整理，选择性接入同伴分支 98468f8 的实现、测试及独立研究证据；没有执行 Git 分支合并，没有 push。

新增入口参数和恢复身份绑定见 methods/gvr/README.md。普通解码默认不变，约束解码只在 executor_one/executor_three 的 R 启用，限 T5。规则对照及冻结评测不自动成为默认入口。历史源码锁绑定原提交，整合后不能直接用于新正式测试。

验证：158 项完整 CPU 回归全部通过；普通/约束入口在同一 T5-small checkpoint、示例网格和执行器配置下 dry-run 通过。回归临时目录改为项目内 .cache/test-temp，以避开系统临时目录权限错误。日志位于 .cache/gvr_integration_workspace.log。

实际 GPU 尝试未完成：require_cuda 检查失败，nvidia-smi 报访问权限不足。失败会话 outputs/full_finetune_gvr/local_constrained_20261111 保留；恢复 GPU 访问后可用 resume 继续。此次没有模型推理新成绩、没有重新训练。CPU 测试中的假模型验证不能替代真实模型效果评测。
