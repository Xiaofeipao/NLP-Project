# 环境配置与运行验证

## 基本要求

- 建议使用独立的 Python 3.11 环境；环境名 `ppnl` 仅为示例，可自行替换。
- 训练和模型推理需要支持 CUDA 的 GPU，以及与目标设备兼容的 PyTorch 构建。
- NLP 依赖由 `requirements.txt` 管理，包括固定版本的 Transformers、sentencepiece 和 protobuf。
- `environment.yml` 提供环境模板；其中的 PyTorch 版本应结合目标硬件核对，不能视为适用于所有设备的保证。
- 数据加工和不涉及真实模型的逻辑测试可以在 CPU 上执行。

## 安装与检查

```powershell
conda create -n ppnl python=3.11
conda activate ppnl
# 根据目标平台安装兼容的 CUDA 版 PyTorch，再安装项目依赖
python -m pip install -r requirements.txt
python -m ppnl.cli environment
```

使用其他虚拟环境工具也可以，项目不依赖特定的个人安装目录或环境名称。

运行模型前，先执行 `nvidia-smi` 检查设备和驱动，再验证 PyTorch：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.ones(8, device='cuda').sum().item())"
```

驱动支持的 CUDA 版本与 PyTorch 自带的 CUDA 运行时版本不必完全相同，但必须兼容。仅安装系统 CUDA 工具包不能替代兼容的 PyTorch 构建。BF16 为可选精度，需要设备支持；正式实验参数以配置文件为准。

## 项目验证

已完成 GPU 张量计算及 Flan-T5-small、BART-base 的前后向、优化器更新、生成和修复流程验证。运行前仍应在目标环境复核，不依赖某个固定 GPU 型号或驱动版本。

```powershell
python -m unittest discover -s tests -v
```

模型下载、数据准备、pilot 和正式实验顺序见项目 README。模型命令明确要求 CUDA，数据工具按需延迟导入模型依赖。遇到临时目录权限错误时，应检查运行目录及临时目录权限，与 GPU 驱动故障分开定位。
