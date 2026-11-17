# 循环惩罚与相关工作

本项目可将具体实现描述为 **Visit-Penalized Neural Beam Search（带访问惩罚的神经束搜索）**。这是对本实现的描述性命名，不是宣称复现了某个同名标准算法，也不构成独立创新性的证明。

推理分数为累计动作log概率，减去目标格访问次数惩罚及即时回退惩罚。参数固定为1.5和2，未按测试调参。模型权重冻结，无反向传播，无奖励驱动参数更新，因此不是强化学习。完整R组合还包含反馈定位和确定性闭环消除，后者的最优率贡献必须单独报告。

## 相关方向

- Bellemare et al., **Unifying Count-Based Exploration and Intrinsic Motivation**, NeurIPS 2016。论文将伪计数转成强化学习探索奖励；与本项目利用访问频度抑制重复的动机相关，但本项目使用精确网格访问次数和推理时扣分，没有实现论文的密度模型/伪计数算法。[论文与官方摘要](https://proceedings.neurips.cc/paper_files/paper/2016/hash/afda332245e2af431fb7b672a68b659d-Abstract.html)
- Lu et al., **NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints**, NAACL 2021。其研究在推理阶段结合约束与搜索，可以作为冻结模型后约束解码的相关工作；它处理谓词逻辑词汇约束，与我们的动态网格合法性/循环惩罚并非同一实现。[ACL Anthology](https://aclanthology.org/2021.naacl-main.339/)
- Welleck et al., **Neural Text Generation with Unlikelihood Training**, ICLR 2020。通过训练目标降低重复内容概率，可以与推理时惩罚作方法层面的区别。我们没有使用其训练目标；该方法本身也不等于强化学习。[作者论文](https://arxiv.org/abs/1908.04319)

适合报告的表述：我们的修复器结合学习式动作评分、访问频次惩罚与有界束搜索，受基于计数的探索和约束解码思想启发；实验中的收益属于神经模型与符号控制的混合系统，不宣称已训练出新的强化学习策略。
