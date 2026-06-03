# minimal-verl

> Agentic RL 最小可运行示例：用 TRL GRPOTrainer 给 Qwen2.5-0.5B 做中文情感分类。
>
> **目的**：跑通"人造数据 → SFT → GRPO → 评估"完整链路，**理解 Agentic RL 工具链**，不是训出生产模型。

## 30 秒看懂

| 步骤 | 输入 | 输出 | 耗时 |
|---|---|---|---|
| 1. 环境验证 | 无 | 打印 MPS 状态 + Qwen 推理速度 | 5-30 秒 |
| 2. 生成数据 | 11 条人造评论（8 训 + 3 测） | `data/sft_train.jsonl` + `sft_test.jsonl` | < 1 秒 |
| 3. SFT 训练 | Qwen2.5-0.5B + 8 条数据 | `sft_output/` (1.9GB) | 30 秒 |
| 4. GRPO 训练 | sft_output + 8 条 prompt + 奖励函数 | `grpo_output/` (1.9GB) | 30 秒 |
| 5. 评估 | 任意模型 + 3 条测试集 | 准确率 | 5 秒 |

**总耗时约 2 分钟**（不含首次下载 Qwen 模型 ~1GB）。

## 环境要求

- **macOS**（M1/M2/M3/M4/M5 Apple Silicon 均可，Intel macOS 也能跑但没 MPS 加速）
- Python 3.12
- 24GB 统一内存（16GB 也能跑，可能 OOM）
- [uv](https://github.com/astral-sh/uv) 包管理

> **不在 OrbStack/Linux VM 里跑**：PyTorch MPS 后端只在 macOS 构建，Linux wheel 没 Metal 代码。

## 3 分钟跑通

```bash
# 1. 克隆 + 装环境（首次约 1-3 分钟下载包）
cd /Users/jimmy/Workspaces/agentic-rl/minimal-verl
uv sync

# 2. 验证环境（MPS + Qwen 加载 + 推理速度）
uv run python scripts/00_check_env.py
# 期望看到: MPS available: True / 49+ tokens/s

# 3. 生成人造数据
uv run python data/make_data.py

# 4. 跑 SFT
uv run python sft/train_sft.py
# 期望看到: loss 5.52 → 0.36, 24 步, 30 秒

# 5. 跑 GRPO
uv run python rl/train_grpo.py
# 期望看到: rewards mean 0.25~1.0, 20 步, 30 秒

# 6. 评估 SFT 模型
uv run python eval/evaluate.py
# 期望看到: 准确率 2/3 = 66.7%

# 7. 评估 GRPO 模型（对比 SFT）
MODEL_PATH=grpo_output uv run python eval/evaluate.py
```

## 三方对比结果

| 模型 | 测试集准确率 | 行为 |
|---|---|---|
| 原始 Qwen2.5-0.5B | 0/3 = 0% | 不会分类，直接复读用户输入 |
| SFT（3 epoch） | 2/3 = 66.7% | 学会分类，"中性"边界模糊学不会 |
| GRPO（20 step） | 2/3 = 66.7% | 跟 SFT 一样（8 条数据下 RL 提升空间有限） |

**SFT 是"质变"**（0% → 67%），**GRPO 是"量变"**（让模型生成更自然、分布更集中）。8 条数据下 GRPO 难提升准确率，但 RL 信号传递正常（reward 在动，KL 在动）。

## 项目结构

```
minimal-verl/
├── README.md                       # 本文件
├── pyproject.toml                  # 依赖锁
├── .python-version                 # 3.12
├── .gitignore
├── scripts/
│   └── 00_check_env.py             # 环境验证
├── data/
│   ├── make_data.py                # 生成 11 条人造数据
│   ├── sft_train.jsonl             # 8 训 (ChatML messages)
│   └── sft_test.jsonl              # 3 测
├── sft/
│   └── train_sft.py                # trl.SFTTrainer (FP32 + MPS)
├── rl/
│   ├── reward.py                   # 0/1 严格奖励函数
│   └── train_grpo.py               # trl.GRPOTrainer (FP32 + MPS)
├── eval/
│   └── evaluate.py                 # 测准确率
├── docs/
│   ├── 01-为什么这么设计.md        # 设计决策 + 理由
│   ├── 02-数据流图.md              # 端到端数据流
│   └── 03-常见问题.md              # FAQ
└── sft_output/                     # SFT 训练产物（gitignore）
└── grpo_output/                    # GRPO 训练产物（gitignore）
```

## 关键技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| RL 框架 | TRL GRPOTrainer | macOS 跑不了 verl；TRL 接口一致；PyTorch 原生支持 MPS |
| 基座模型 | Qwen2.5-0.5B | 中文任务必须中文模型；0.5B 最小可训；24GB 内存充裕 |
| 任务 | 情感分类 3 分类 | 人造数据极简；模型需"理解"语义有 RL 价值；奖励定义明确 |
| 精度 | FP32 | Apple Silicon bf16 有坑；0.5B 模型 FP32 不紧张 |
| 数据格式 | ChatML messages | 现代 SFT 实践；SFT/GRPO 都能用 |

详见 [docs/01-为什么这么设计.md](docs/01-为什么这么设计.md)。

## 已知限制

1. **8 条训练数据太少** —— GRPO 训练 20 步 reward 不稳定（4 个 completion 经常全对/全错，组内无差异）
2. **"中性"边界学不会** —— 训练集 2 条中性，模型过拟合到训练集 prompt
3. **模型严重过拟合** —— SFT 3 epoch 后 loss 0.36/token acc 97.7%，**这是必然**（教学最小示例）

## 参考资料

- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Qwen2.5-0.5B 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Apple Silicon MPS 文档](https://developer.apple.com/metal/pytorch/)
- [GRPO 论文](https://huggingface.co/papers/2402.03300) (DeepSeekMath)
