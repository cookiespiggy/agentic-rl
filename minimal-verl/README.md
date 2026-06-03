# minimal-verl · Agentic RL 最小可运行示例

用 **TRL + Qwen2.5-0.5B** 跑通「数据 → SFT → GRPO → 评估」完整链路。  
配套教程：[Agentic RL 零基础教程](../README.md) · 第 [15 章](../15-第一个Agentic-RL训练实战.md)

## 快速开始

```bash
# 需要 Python 3.12 + uv
uv sync
uv run python scripts/00_check_env.py
```

## 目录结构

```
minimal-verl/
├── data/       # 训练数据
├── sft/        # SFT 训练脚本
├── rl/         # GRPO 训练脚本
├── eval/       # 评估脚本
├── scripts/    # 工具脚本（含环境验证）
└── docs/       # 项目进度与说明
```

## 环境要求

- Python ≥ 3.12
- Mac：推荐 Apple Silicon + MPS；Linux/Windows 用 CPU/CUDA
- 首次运行会从 Hugging Face 下载 Qwen2.5-0.5B（约 1GB）

## 当前进度

见 [docs/PROGRESS.md](./docs/PROGRESS.md)。
