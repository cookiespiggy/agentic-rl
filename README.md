

# Agentic RL 零基础教程 · 从概念到 GRPO 实战

> **面向小白的 Agentic RL（智能体强化学习）系统教程** — 24 篇中文 Markdown，配套可运行的 TRL 最小示例。  
> 搜「Agentic RL 教程」「GRPO 入门」「LLM 强化学习」「verl TRL 实战」都能找到这里。

[![GitHub stars](https://img.shields.io/github/stars/cookiespiggy/agentic-rl?style=social)](https://github.com/cookiespiggy/agentic-rl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 这是什么？

**Agentic RL** 是把大语言模型（LLM）训练成**能规划、用工具、多轮交互**的智能体（Agent）的方法。  
本仓库不是论文搬运，而是一条**能跟着走的自学路线**：

| 你现在的水平 | 从这里开始 |
|-------------|-----------|
| 完全零基础，没听过 RL | [01 - 什么是 Agentic RL](./01-什么是Agentic-RL-概念与愿景.md) |
| 懂 LLM，想搞懂 SFT / RLHF / GRPO | [03 - LLM 与 Post-Training](./03-LLM与Post-Training基础.md) → [08 - GRPO 深度解析](./08-GRPO深度解析.md) |
| 想动手跑训练 | [minimal-verl/](./minimal-verl/) 最小示例 + [15 - 第一个训练实战](./15-第一个Agentic-RL训练实战.md) |

**核心理念**：小模型 + Agentic RL，在垂直任务上可以超越更大的通用 LLM。

---

## 学习路线（24 章）

### 阶段一 · 基础概念（01–05）

| 章节 | 主题 | 难度 |
|------|------|------|
| [01](./01-什么是Agentic-RL-概念与愿景.md) | Agentic RL 概念与愿景 | 入门 |
| [02](./02-基础知识-强化学习入门.md) | 强化学习入门（MDP、策略梯度） | 入门 |
| [03](./03-LLM与Post-Training基础.md) | LLM 与 Post-Training（SFT / RLHF / DPO / GRPO） | 入门 |
| [04](./04-从LLM-RL到Agentic-RL.md) | 从 LLM RL 到 Agentic RL | 入门 |
| [05](./05-Agentic-RL的应用场景.md) | 应用场景全景 | 入门 |

### 阶段二 · 核心算法（06–10）

| 章节 | 主题 | 难度 |
|------|------|------|
| [06](./06-PPO算法详解.md) | PPO 算法详解 | 中级 |
| [07](./07-DPO直接偏好优化.md) | DPO 直接偏好优化 | 中级 |
| [08](./08-GRPO深度解析.md) | **GRPO 深度解析**（2025–2026 主流） | 中级 |
| [09](./09-RLVR可验证奖励的强化学习.md) | RLVR 可验证奖励 | 中级 |
| [10](./10-Reward-Shaping奖励设计进阶.md) | Reward Shaping 奖励设计 | 中级 |

### 阶段三 · 训练工程（11–15）

| 章节 | 主题 | 难度 |
|------|------|------|
| [11](./11-训练框架选型与搭建.md) | 训练框架选型（verl / TRL） | 中级 |
| [12](./12-数据准备与SFT阶段.md) | 数据准备与 SFT | 中级 |
| [13](./13-训练环境搭建.md) | 训练环境搭建 | 中级 |
| [14](./14-奖励函数设计与实现.md) | 奖励函数设计 | 中级 |
| [15](./15-第一个Agentic-RL训练实战.md) | **第一个完整训练实战** | 实战 |

### 阶段四 · 环境与实战（16–20）

| 章节 | 主题 |
|------|------|
| [16](./16-网页导航Agent训练.md) | 网页导航 Agent |
| [17](./17-代码Agent训练.md) | 代码 Agent |
| [18](./18-搜索增强Agent训练.md) | 搜索增强 Agent |
| [19](./19-垂直领域Agent训练.md) | 垂直领域 Agent |
| [20](./20-评估与Benchmark.md) | 评估与 Benchmark |

### 阶段五 · 进阶前沿（21–24）

| 章节 | 主题 |
|------|------|
| [21](./21-异步Rollout与分布式训练.md) | 异步 Rollout 与分布式训练 |
| [22](./22-多Agent与Multi-Agent-RL.md) | 多 Agent / Multi-Agent RL |
| [23](./23-前沿论文与研究方向.md) | 前沿论文与研究方向 |
| [24](./24-学习总结与进阶路径.md) | 学习总结与进阶路径 |

---

## 动手练：minimal-verl 最小示例

[`minimal-verl/`](./minimal-verl/) 是一个**真正能跑**的精简项目，目标不是训出生产模型，而是理解整条链路：

```
人造数据 → SFT → GRPO → 评估
```

- **模型**：Qwen2.5-0.5B（Mac M 系列可用 MPS）
- **任务**：中文情感三分类
- **框架**：TRL `GRPOTrainer`（比 verl 更适合入门）

```bash
cd minimal-verl
uv sync
uv run python scripts/00_check_env.py   # 环境验证（4 步）
```

详细进度见 [minimal-verl/docs/PROGRESS.md](./minimal-verl/docs/PROGRESS.md)。

---

## 适合谁？

- 想入门 **LLM 强化学习 / RLHF / GRPO** 的开发者
- 想训练 **Agent**（工具调用、多轮交互）但不知道从哪下手
- 看过 verl、DeepSeek-R1 新闻，想搞懂背后训练流程
- 中文学习者，希望有**结构化路线**而不是零散博客

---

## 关键词（方便搜索）

`Agentic RL` · `智能体强化学习` · `GRPO` · `PPO` · `DPO` · `RLHF` · `RLVR` · `LLM Post-Training` · `verl` · `TRL` · `Qwen` · `SFT` · `Reward Shaping` · `工具调用 Agent` · `小模型 RL`

---

## 关于作者

由 [@cookiespiggy](https://github.com/cookiespiggy) 维护 — 持续更新教程与可运行示例，欢迎 Star ⭐ 和 Issue 反馈。

---

## License

MIT — 教程内容可自由学习、引用，请注明出处。
