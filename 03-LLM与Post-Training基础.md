---
title: "03 - LLM 与 Post-Training 基础"
description: "理解大语言模型的训练流程，特别是 Post-Training 阶段"
difficulty: "入门"
reading_time: "20 min"
tags: ["LLM", "Post-Training", "RLHF", "DPO", "GRPO", "RLVR"]
---

# 03 - LLM 与 Post-Training 基础

> **学习目标**：理解大语言模型的训练流程，特别是 Post-Training（后训练）阶段，这是 Agentic RL 的基础。

---

## 目录

- [3.1 LLM 训练的三个阶段](#31-llm-训练的三个阶段)
- [3.2 SFT（监督微调）详解](#32-sft监督微调详解)
- [3.3 Post-Training 方法演进](#33-post-training-方法演进)
- [3.4 从 Post-Training 到 Agentic RL](#34-从-post-training-到-agentic-rl)
- [3.5 2026 年标准训练流水线](#35-2026-年标准训练流水线)
- [3.6 术语速查](#36-术语速查)
- [3.7 小结](#37-小结)
- [练习题](#练习题)

---

## 3.1 LLM 训练的三个阶段

```
阶段一：Pre-Training（预训练）
  └── 在海量文本上学习语言知识和世界知识
  └── 输出：Base Model（基座模型），如 Qwen-7B-Base

阶段二：SFT（Supervised Fine-Tuning，监督微调）
  └── 用人工标注的 (指令, 回复) 对教模型遵循指令
  └── 输出：Chat Model（对话模型），如 Qwen-7B-Chat

阶段三：Post-Training（后训练）/ RLHF / RL
  └── 用强化学习进一步优化模型行为
  └── 输出：Aligned Model（对齐模型）
```

**Agentic RL 属于阶段三**，是在 SFT 之后的 RL 训练。

---

## 3.2 SFT（Supervised Fine-Tuning，监督微调）详解

SFT 的目标是让 Base Model 学会"对话格式"和基本指令遵循能力：

```python
# SFT 训练数据示例
training_data = [
    {
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是AI的一个分支..."}
        ]
    }
]

# SFT 的损失函数（Loss Function）
# 只计算 assistant 回复部分的 loss
loss = CrossEntropyLoss(model_output, target_tokens)
```

**SFT 的局限**：
- 模型只学到"模仿"训练数据中的回复
- 无法学会在环境中主动探索和自我纠正
- 分布偏移（Distribution Shift）：训练时看到的是人类示范，推理时需要自主决策

---

## 3.3 Post-Training 方法演进

```
时间线：
2022: RLHF（RL from Human Feedback）- InstructGPT / ChatGPT
2023: DPO（Direct Preference Optimization）- 简化版 RLHF
2024: GRPO（Group Relative Policy Optimization）- 无需 Critic 模型
2025: RLVR（RL with Verifiable Rewards）- 用可验证奖励替代人类偏好
2026: Agentic RL - 在多轮环境交互中训练 Agent
```

### 3.3.1 RLHF（Reinforcement Learning from Human Feedback）

```
步骤 1：收集人类偏好数据
  → 给定 prompt，生成多个回复，人类标注哪个更好

步骤 2：训练 Reward Model（奖励模型）
  → 学习人类的偏好判断

步骤 3：用 PPO 训练 Policy Model
  → Policy = LLM，Reward = 奖励模型的打分
```

**问题**：需要大量人类标注数据，训练流程复杂（要同时加载 Policy、Reference、Reward、Critic 四个模型）。

### 3.3.2 DPO（Direct Preference Optimization）

核心思想：跳过 Reward Model，直接用偏好对（Preference Pairs）训练 Policy。

```
DPO Loss = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

其中：
- y_w = preferred response（人类偏好的回复）
- y_l = rejected response（被拒绝的回复）
- π_ref = reference model（参考模型，通常是 SFT 模型）
- β = 温度超参数
```

**优点**：简单，不需要额外的 Reward Model
**缺点**：只能用于单轮对话，无法处理多轮 Agent 交互

### 3.3.3 GRPO（Group Relative Policy Optimization）

DeepSeek 提出的算法，**当前 Agentic RL 的主流选择**。

```
核心思想：
对同一个 prompt，生成一组（Group）回复，
用组内的相对质量排序来替代绝对奖励，
无需单独的 Critic/Value Model。
```

我们将在 [08 - GRPO 深度解析](./08-GRPO深度解析.md) 中详细学习。

### 3.3.4 RLVR（RL with Verifiable Rewards）

```
核心思想：用客观可验证的奖励替代人类偏好

例如：
- 数学题：答案正确 = 1，错误 = 0（自动验证）
- 代码题：测试用例通过 = 1，失败 = 0（自动验证）
- Agent 任务：任务完成 = 1，失败 = 0（环境反馈）
```

**RLVR 是 Agentic RL 的重要基础**——Agent 的奖励可以来自环境的客观反馈。

---

## 3.4 从 Post-Training 到 Agentic RL

| 维度 | 传统 Post-Training (RLHF) | Agentic RL |
|------|---------------------------|------------|
| **交互模式** | 单轮 (prompt → response) | 多轮 (multi-turn dialogue + actions) |
| **奖励来源** | 人类偏好 / Reward Model | 环境客观反馈 / Verifiable Rewards |
| **动作空间** | 只生成文本 | 文本 + 工具调用 + 代码执行 |
| **决策过程** | 一次性决策 | 序列决策 (Sequential Decision-Making) |
| **数学框架** | MDP（简化为单步） | POMDP（完整的多步序列） |
| **训练数据** | 静态偏好对 | 动态交互轨迹（Trajectories） |

---

## 3.5 2026 年标准训练流水线

根据最新实践（2026），训练一个 Tool-Using Agent 的标准三阶段：

```
Stage 1: SFT（监督微调）
  └── 教模型基本的工具调用格式（如 JSON 格式的 function call）
  └── 数据来源：人工标注的工具调用示范

Stage 2: RL Training（强化学习训练）
  └── 使用 GRPO / PPO 在真实环境中训练
  └── 模型通过试错学习何时调用工具、如何组合工具

Stage 3: Evaluation & Iteration（评估与迭代）
  └── 在 Benchmark 上评估
  └── 分析失败案例，改进环境/奖励/数据
```

---

## 3.6 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Pre-Training | 预训练 |
| Supervised Fine-Tuning (SFT) | 监督微调 |
| Post-Training | 后训练 |
| RLHF | 基于人类反馈的强化学习 |
| DPO (Direct Preference Optimization) | 直接偏好优化 |
| GRPO (Group Relative Policy Optimization) | 组相对策略优化 |
| RLVR (RL with Verifiable Rewards) | 基于可验证奖励的强化学习 |
| Reward Model | 奖励模型 |
| Reference Model | 参考模型 |
| Preference Pairs | 偏好对 |
| Distribution Shift | 分布偏移 |
| Alignment | 对齐 |

---

## 3.7 小结

- LLM 训练分三个阶段：Pre-Training → SFT → Post-Training
- Agentic RL 属于 Post-Training 阶段，在 SFT 之后进行
- 从 RLHF → DPO → GRPO → RLVR，训练方法越来越简化、越来越客观
- 2026 年标准流水线：SFT → RL (GRPO) → Eval
- 下一步：理解从 LLM RL 到 Agentic RL 的范式转变 → [04 - 从 LLM RL 到 Agentic RL](./04-从LLM-RL到Agentic-RL.md)

---

## 练习题

1. **流程排序**：将以下 LLM 训练阶段按正确顺序排列，并简要说明每个阶段的作用：
   - Post-Training / RL
   - Pre-Training（预训练）
   - SFT（监督微调）

2. **方法对比**：RLHF 和 DPO 的核心区别是什么？为什么 DPO 不适合多轮 Agent 交互场景？

3. **概念理解**：GRPO 相比 PPO 的主要优势是什么？为什么它被认为是"当前 Agentic RL 的主流选择"？

> **参考答案**：1. Pre-Training → SFT → Post-Training，见 3.1 节；2. DPO 不需要独立的 Reward Model，但只能处理单轮偏好对；3. GRPO 不需要 Critic/Value Model，通过组内相对质量替代绝对奖励，训练更稳定。
