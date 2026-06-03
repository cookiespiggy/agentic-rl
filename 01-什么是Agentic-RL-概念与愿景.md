---
title: "01 - 什么是 Agentic RL：概念与愿景"
description: "理解 Agentic RL 的定义、与传统 RL 的区别，以及为什么它是训练小模型超越大模型的关键路径"
difficulty: "入门"
reading_time: "15 min"
tags: ["Agentic RL", "概念", "愿景", "综述"]
---

# 01 - 什么是 Agentic RL：概念与愿景

> **学习目标**：理解 Agentic RL（智能体强化学习）的定义、与传统 RL 的区别，以及为什么它是训练小模型超越大模型的关键路径。

---

## 1.1 一句话定义

**Agentic RL（Agentic Reinforcement Learning，智能体强化学习）** 是一种将大语言模型（LLM）从"被动的文本生成器"转变为"自主决策的智能体（Agent）"的训练范式。

核心思想：让模型在与环境（Environment）的多轮交互中，通过试错（Trial-and-Error）和强化学习信号，学会**规划、使用工具、记忆、推理和自我改进**。

---

## 1.2 为什么需要 Agentic RL？


### 1.2.1 当前 LLM Agent 的局限

传统的 LLM Agent 通常依赖以下方式：

```text
用户输入 → Prompt Engineering（提示词工程） → LLM → 工具调用 → 输出
```

**问题：**
- **Prompt Engineering 是脆弱的**：稍微改变措辞，输出可能完全不同
- **缺乏任务规划能力**：LLM 本身并不擅长多步骤（Multi-step）任务规划
- **没有试错学习**：LLM 无法从失败中学习并改进策略

| 传统 LLM | Agentic LLM |
|-----------|-------------|
| 单轮输入 → 单轮输出 | 多轮交互，持续决策 |
| 只能生成文本 | 可以调用工具、执行动作 |
| 被动响应 | 主动规划、自主行动 |
| 无状态（Stateless） | 有记忆（Memory），可跨轮次保持上下文 |
| 依赖 Prompt Engineering | 通过 RL 内化（Internalize）行为策略 |

**关键洞察**：2025-2026 年的多项研究表明，通过 Agentic RL 训练的小模型（如 7B 参数），在特定垂直领域的 Agent 任务上，可以超越未经 RL 训练的更大模型（如 70B+）。

> NVIDIA Research（2025）在论文《Small Language Models are the Future of Agentic AI》[Belcak et al., arXiv:2506.02153] 中指出："Small Language Models（SLMs，小语言模型）是 Agentic AI 的未来"——因为在 Agent 场景中，模型执行的是**少量专业化任务**，而非通用对话。


### 1.2.2 Agentic RL 的优势

```text
用户输入 → Agent（由 RL 训练的策略网络） → 环境交互 → 奖励信号 → 策略更新
```

**优势：**
- ✅ **自主学习工具使用**：通过奖励信号学会何时、如何使用工具
- ✅ **多步规划能力**：RL 天然擅长优化长期奖励（Long-term Reward）
- ✅ **持续改进**：每次环境交互都是学习机会
- ✅ **小模型大能力**：在垂直领域，1B-7B 的 RL 训练模型可以超越 70B+ 的通用 LLM

---

## 1.3 Agentic RL 的核心框架

```text
┌─────────────────────────────────────────────────┐
│                   Agent（智能体）                 │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Policy   │  │  Memory  │  │  Tool Use    │  │
│  │（策略模型）│  │（记忆模块）│  │（工具使用）  │  │
│  └─────┬─────┘  └──────────┘  └──────┬───────┘  │
│        │                             │           │
│        ▼                             ▼           │
│  ┌─────────────────────────────────────────┐     │
│  │         Action（动作：生成文本/调用工具）  │     │
│  └────────────────────┬────────────────────┘     │
└───────────────────────┼─────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Environment    │
              │   （环境）        │
              │ - 网页浏览器     │
              │ - 代码解释器     │
              │ - 数据库         │
              │ - API 服务       │
              └────────┬─────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Observation +   │
              │  Reward          │
              │（观测 + 奖励）    │
              └──────────────────┘
```

---

## 1.4 关键术语速查表

| 英文术语 | 中文注释 | 说明 |
|---------|---------|------|
| **Agent** | 智能体 | 能自主决策和行动的 AI 系统 |
| **Environment** | 环境 | Agent 交互的外部世界 |
| **Policy** | 策略 | Agent 根据观测选择动作的规则（即模型本身） |
| **Reward** | 奖励 | 环境对 Agent 动作的反馈信号 |
| **Rollout** | 展开/采样 | Agent 与环境完整交互的一轮轨迹 |
| **Trajectory** | 轨迹 | 一次 Rollout 中所有 (状态, 动作, 奖励) 的序列 |
| **Multi-turn** | 多轮 | Agent 与环境多次交互 |
| **Tool Use** | 工具使用 | Agent 调用外部工具（搜索、代码执行等） |
| **Planning** | 规划 | Agent 制定多步骤行动计划 |
| **POMDP** | 部分可观测马尔可夫决策过程 | Agentic RL 的数学形式化框架 |
| **MDP** | 马尔可夫决策过程 | 经典 RL 的数学框架 |

---

## 1.5 学习目标路线图

本教程系列共 24 篇，分为 5 个阶段：

```text
阶段一（基础概念）: 01-05 → 建立 RL 和 LLM 的基础认知                📖 进行中
阶段二（核心算法）: 06-10 → 掌握 PPO、DPO、GRPO 等核心算法         📖 待学习
阶段三（训练工程）: 11-15 → 数据准备、环境搭建、奖励设计、实战训练   📖 待学习
阶段四（环境与实战）: 16-20 → 各类训练环境和垂直领域项目实战         📖 待学习
阶段五（进阶前沿）: 21-24 → 多智能体、前沿论文、小模型策略         📖 待学习
```

---

## 1.6 推荐阅读

1. **Survey 综述**：[The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.02547) — 综合 500+ 篇工作的权威 Survey（arXiv:2509.02547, 2026.04 更新 v5）
2. **NVIDIA Position Paper**：Belcak et al., [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153)（arXiv:2506.02153, 2025）
3. **GitHub Repo**：[Agentic-RL-Training-Recipes](https://github.com/blacksnail789521/Agentic-RL-Training-Recipes) — 精选训练方案合集

---

## 1.7 小结

- Agentic RL = RL + Agent + LLM，让语言模型学会像人一样在环境中自主行动
- 它与传统 LLM RL（如 RLHF）的核心区别在于：**多轮交互**、**工具使用**、**环境反馈**
- 小模型 + Agentic RL 在垂直领域有巨大潜力超越大模型
- 下一步：学习 RL 的基础知识 → [02 - 基础知识：强化学习入门](./02-基础知识-强化学习入门.md)

---

## 练习题

1. **概念理解**：Agentic RL 与传统 RLHF 的核心区别是什么？请从交互模式、奖励来源、动作空间三个角度说明。

2. **场景判断**：以下哪些场景适合用 Agentic RL 来训练？（多选）
   - A. 训练一个聊天机器人进行日常对话
   - B. 训练一个 Agent 在电商网站上自动完成商品比价和下单
   - C. 训练一个代码 Agent 阅读 Issue、修改代码、运行测试
   - D. 训练一个文本分类模型判断邮件是否为垃圾邮件

3. **框架分析**：回忆 1.3 节的 Agentic RL 核心框架图，说明 Policy、Memory、Tool Use 三个模块各自的作用，以及它们之间如何协同工作。

> **参考答案**：1. 见 1.3 节表格对比；2. B、C（需要多轮交互和工具使用的场景）；3. Policy 决定下一步动作，Memory 维护历史上下文，Tool Use 扩展 Agent 能力边界。
