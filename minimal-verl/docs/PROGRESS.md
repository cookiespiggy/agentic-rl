# Agentic RL 最小可运行示例 —— 项目进度

> **本文件用途**：项目级执行日志，跟随 git。人 / AI 共用。
> 详细决策理由在 `/Users/jimmy/Workspaces/agentic-rl/handoff.md`。

---

## TL;DR（30 秒看完）

- **项目**：用 TRL GRPOTrainer 训 Qwen2.5-0.5B 做中文情感分类（3 分类）
- **当前阶段**：环境验证完成（5/5），决策变更为 macOS 原生路线
- **下一步**：新会话建 macOS .venv + 跑环境验证脚本（确认 MPS 可用）
- **预计剩余工作量**：5-8 小时

---

## 0. 一句话目标

跑通"人造数据 → SFT → GRPO → 评估"完整链路。**不是**训练出生产模型，是**理解工具链**。

---

## 1. 当前状态

**日期**：2026-06-03
**会话**：第一会话（hondoff 第 13 节记录）
**进度**：环境验证 5/5（Linux 路线），决策变更中

| 阶段 | 状态 | 备注 |
|---|---|---|
| 0. 项目骨架 | ✅ | `minimal-verl/` 目录、pyproject.toml、.python-version、.gitignore |
| 1. 环境验证（Linux VM 路线） | ✅ | 5/5 步全过，**但发现 MPS 不可用** |
| 2. 决策变更：macOS 路线 | ✅ | 用户已拍板，handoff 已更新 |
| 3. macOS 环境验证 | ⏳ | **新会话第一件事** |
| 4. SFT 训练 | ❌ | 待开始 |
| 5. GRPO 训练 | ❌ | 待开始 |
| 6. 评估 | ❌ | 待开始 |
| 7. 文档 | ❌ | 待开始 |

---

## 2. 已完成清单

- [x] 项目目录骨架（`data/ docs/ eval/ rl/ sft/ scripts/`）
- [x] `pyproject.toml`（torch 2.5+ / transformers 4.46+ / trl 0.12+ / datasets / accelerate）
- [x] `.python-version`（3.12）
- [x] `.gitignore`（排除 .venv/、模型权重、HF 缓存）
- [x] OrbStack VM `agentic-rl`（Ubuntu 24.04 noble, arm64, 运行中）—— fallback 保留
- [x] VM 内装 uv 0.11.18 + Python 3.12.3
- [x] VM 内 `uv sync` 装好 PyTorch 2.12.0 + TRL 1.5.1 + 全部依赖
- [x] 验证 PyTorch import + TRL GRPOTrainer import（VM 内通过）
- [x] **关键发现**：Linux VM 内 MPS 不可用（handoff 第 13.2 节）
- [x] 决策变更：改走 macOS 原生 + MPS 路线
- [x] 写环境验证脚本 `scripts/00_check_env.py`

---

## 3. 待完成清单（按优先级）

### 优先级 P0：环境验证（macOS）
- [ ] macOS 终端 `cd /Users/jimmy/Workspaces/agentic-rl/minimal-verl`
- [ ] `uv sync`（重装，VM 装的是 Linux 版，macOS 版要重装）
- [ ] `uv run python scripts/00_check_env.py` 验证 **MPS available: True**
- [ ] 验证 Qwen2.5-0.5B 加载 + 中文推理
- [ ] 记录真实 MPS 性能（tokens/s）

### 优先级 P1：数据生成
- [ ] 写 `data/make_data.py`：生成 10 条人造情感分类数据
- [ ] 训练集/测试集划分（8 训 / 2 测）
- [ ] 验证数据格式（jsonl，input/label 字段）

### 优先级 P1：SFT 训练
- [ ] 写 `sft/train_sft.py`：HuggingFace transformers Trainer
- [ ] 写 `sft/run_sft.sh`：启动脚本
- [ ] 训练 3 epoch，验证 loss 下降
- [ ] 保存到 `sft_output/`

### 优先级 P2：GRPO 训练
- [ ] 写 `rl/reward.py`：情感分类奖励函数
- [ ] 写 `rl/train_grpo.py`：TRL GRPOTrainer
- [ ] 写 `rl/run_grpo.sh`
- [ ] 50 step 训练，验证 reward 上升
- [ ] 保存到 `grpo_output/`

### 优先级 P2：评估
- [ ] 写 `eval/evaluate.py`：加载 `grpo_output/`，在测试集上测正确率
- [ ] 打印分类报告

### 优先级 P3：文档
- [ ] `README.md`：3 分钟跑通指南
- [ ] `docs/01-为什么这么设计.md`
- [ ] `docs/02-数据流图.md`
- [ ] `docs/03-常见问题.md`

### 优先级 P0：交付前
- [ ] 端到端测试通过
- [ ] **关闭所有测试进程**（CLAUDE.md 规则）

---

## 4. 关键决策（精简版，详细见 handoff.md）

| 决策点 | 选择 | 备注 |
|---|---|---|
| 任务 | 中文情感分类 3 分类 | 正面/负面/中性 |
| RL 框架 | TRL GRPOTrainer | 不直接用 verl（OrbStack 跑不了） |
| 基座 | Qwen2.5-0.5B | 中文必须中文模型 |
| 运行环境 | **macOS 原生 + MPS** | 2026-06-03 决策变更 |
| 包管理 | uv | CLAUDE.md 规则 |
| Python | 3.12 | 24.04 原生 3.12 比 3.11 新 |
| 训练精度 | FP32 起步 | Apple Silicon bf16 有坑，handoff 第 11 节 |

---

## 5. 踩过的坑（避免再踩）

1. **PyTorch MPS 后端不在 Linux 构建** —— Linux wheel 没 Metal 代码，VM 路线**架构上**就拿不到 MPS
2. **uv init 会创建 `main.py` 模板** —— 我们项目不需要，删掉
3. **`.gitignore` 写时要排除模型权重**（`*.safetensors`、`*.bin`），1GB 仓库爆炸
4. **HF 缓存默认在 `~/.cache/huggingface/`** —— macOS 上没问题，但要走 macOS 原生（不是 VM 同步路径）
5. **`orb -m X bash -c 'cmd1 && cmd2'` 的 `&&` 不会跨 orb 边界** —— 必须用 `bash -c '...'` 包整个命令

---

## 6. 关键文件位置

```
/Users/jimmy/Workspaces/agentic-rl/
├── handoff.md                         # 完整背景 + 决策理由
├── minimal-verl/                      # 本项目
│   ├── .python-version                # 3.12
│   ├── pyproject.toml                 # 依赖锁
│   ├── .gitignore
│   ├── scripts/
│   │   └── 00_check_env.py            # 环境验证脚本
│   ├── data/                          # 数据生成 + jsonl
│   ├── sft/                           # SFT 训练
│   ├── rl/                            # GRPO 训练
│   ├── eval/                          # 评估
│   └── docs/                          # 给小白的设计文档
└── 24 篇教程 .md                       # 不动
```

VM（`agentic-rl`）保留作 fallback，已装好所有依赖，**不删除**。

---

## 7. 下一会话 prompt（直接复制）

```text
我要继续开发 Agentic RL 最小可运行示例项目。

项目位置: /Users/jimmy/Workspaces/agentic-rl/minimal-verl/
完整背景: /Users/jimmy/Workspaces/agentic-rl/handoff.md
项目进度: /Users/jimmy/Workspaces/agentic-rl/minimal-verl/docs/PROGRESS.md
(请先读 handoff.md 和 PROGRESS.md 理解上下文)

我的硬件: Apple M5 Pro, 24GB 内存, macOS 26.3
CLAUDE.md 规则: 中文交流 / 称呼"米宝宝" / uv 管 Python（OrbStack 规则本次破例）
用户背景: 完全小白,需要详细解释每个步骤"为什么"

关键决策(已定,不要改):
- 任务: 情感分类(中文,3 分类)
- RL 框架: TRL 的 GRPOTrainer(不直接用 verl)
- 基座模型: Qwen2.5-0.5B
- 运行环境: macOS 原生 (用 MPS 加速,不是 OrbStack VM)

【第一件事：macOS 端环境验证】
环境已在 OrbStack VM (agentic-rl) 里验证 5/5 通过,所有 import 正常,
但发现 Linux VM 无 MPS。已决策改走 macOS 原生路线。

请按以下步骤执行:
1. 在 macOS 终端 cd 到项目目录
2. uv sync (重装,VM 装的是 Linux 版的 wheel,macOS 版要重装)
3. uv run python scripts/00_check_env.py
4. 确认输出里 MPS available: True
5. 记录真实的 Qwen2.5-0.5B 推理速度 (tokens/s)
6. 如果 MPS 不可用,停下来讨论,不要硬上

【注意】
- 不要一上来就写完整训练代码,先打通环境
- 每步先解释"为什么"再执行
- 遇到任何坑立刻停下来讨论
- 称呼我"米宝宝"
```
