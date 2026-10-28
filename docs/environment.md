# 本机环境检查记录

## 2026-10-18 运行验证

已按用户要求直接使用 `D:\Anaconda\envs\torch_py311\python.exe`，没有完成或采用项目 `.venv`。该目录是先前被中断的初始化残留，代码与命令均不依赖它。

硬件为 NVIDIA GeForce RTX 5060 Ti，16,311 MiB 显存，驱动 610.88；`nvidia-smi` 显示驱动支持 CUDA 13.3。PyTorch 自带运行时为 CUDA 13.2，两者版本不需要相同。已实际执行 GPU 矩阵乘法，以及 Flan-T5-small/BART-base 的前后向、AdamW 更新、生成和修复流程。

当前使用 Python 3.11.15、torch 2.12.1+cu132、Transformers 4.57.6、sentencepiece 0.2.2、huggingface-hub 0.36.2、tokenizers 0.22.2、safetensors 0.8.0、protobuf 6.33.6。Transformers 从已撤回的 4.57.0 改为同系列修复版本 4.57.6。没有替换 torch 或 CUDA。

12 项原有测试、完整管线测试和重复动作解码回归测试共 14 项全部通过。官方数据构造与 split audit 通过。小规模 GPU 结果在 `outputs/smoke_20261018/`；正式双模型微调和完整实验也已完成，结果见 `outputs/study_results.json`。

额外 `pip check` 检测到环境中人脸识别相关包的依赖冲突，包括 facenet-pytorch 对 torch/numpy/Pillow 的旧版本约束，以及部分包缺少 TensorFlow。本次不安装或修改这些无关包；本项目依赖的实际导入与 GPU 链路均已通过。

## 2026-10-17 初始清单

检查日期：2026-10-17。只读取 Anaconda 的 conda-meta 和 site-packages 元数据，没有导入 torch、探测 GPU、安装依赖或运行训练。以下是包清单证据，不能代替未来的 import 检查和 GPU pilot。

- `env_isaaclab`：Python 3.11.13；torch 2.7.0+cu128；transformers 4.57.0；tokenizers 0.22.1；huggingface-hub 0.35.3；safetensors 0.6.2；protobuf 6.32.1。未发现 sentencepiece。它最接近本项目需求，但属于已有 Isaac Lab 环境，建议克隆后补包。
- `torch_py311`：Python 3.11.15；torch 2.12.1+cu132。未发现 transformers、sentencepiece。可作为另一个 Python 3.11 基础环境，但需要安装 NLP 依赖并在 GPU 主机核对 CUDA。
- `pytorch_env`：Python 3.13.5；torch 2.12.1。未发现 transformers、sentencepiece；numpy 同时出现 2.4.4 和 2.4.6 的 dist-info，清单存在残留可能，不优先选用。
- `tf_gpu`：Python 3.10.20；torch 2.11.0+cu128。未发现 transformers、sentencepiece。
- `myenvironment`：Python 3.13.5；未发现 torch、transformers、sentencepiece。
- `quant`：Python 3.9.25；未发现 torch、transformers、sentencepiece。

上述清单为最初的只读检查；2026-10-18 已按用户授权给 torch_py311 补齐依赖并完成运行验证，以本页最新记录为准。
