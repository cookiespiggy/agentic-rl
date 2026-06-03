---
title: "04 - 从 LLM RL 到 Agentic RL 的范式转变"
description: "深入理解 Agentic RL 与传统 LLM RL 的本质区别，掌握 POMDP 框架和 Agentic 能力的核心维度"
difficulty: "进阶"
reading_time: "25 min"
tags: ["Agentic RL", "POMDP", "范式转变", "规划", "工具使用"]
---

# 04 - 从 LLM RL 到 Agentic RL 的范式转变

> **学习目标**：深入理解 Agentic RL 与传统 LLM RL 的本质区别，掌握 POMDP 框架和 Agentic 能力的核心维度。

---

## 目录

- [4.1 范式转变的核心](#41-范式转变的核心)
- [4.2 六大 Agentic 能力维度](#42-六大-agentic-能力维度)
- [4.3 POMDP 框架详解](#43-pomdp-框架详解)
- [4.4 Rollout 的结构差异](#44-rollout-的结构差异)
- [4.5 Agentic RL 的挑战](#45-agentic-rl-的挑战)
- [4.6 术语速查](#46-术语速查)
- [4.7 小结](#47-小结)
- [练习题](#练习题)

---

## 4.1 范式转变的核心

Survey 论文《[The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.02547)》明确指出：

> 传统 LLM RL（如 RLHF）本质上是一个**退化的单步 MDP（Degenerate Single-step MDP）**——输入 prompt，输出 response，一步结束。
> Agentic RL 是一个**完整的、时间扩展的 POMDP**——多轮交互、部分可观测、持续决策。

```
传统 LLM RL:
  prompt → [LLM] → response → reward → done

Agentic RL:
  obs_1 → [LLM] → action_1 → obs_2 → [LLM] → action_2 → ... → obs_T → action_T → reward → done
  ↑___________________________多轮交互循环_______________________________↑
```

---

## 4.2 六大 Agentic 能力维度

Survey 论文将 Agentic 能力分为六大类：

### 4.2.1 Planning（规划）

Agent 制定多步骤行动计划的能力。

```
示例：用户要求"帮我订一张明天去北京的机票"

Agent 规划：
  Step 1: 搜索航班信息（调用航班搜索 API）
  Step 2: 筛选符合条件的航班
  Step 3: 比较价格和时间
  Step 4: 推荐最佳航班给用户
  Step 5: 确认并下单（调用订票 API）
```

### 4.2.2 Tool Use（工具使用）

Agent 调用外部工具来扩展自身能力。

```
常见工具类型：
- 搜索引擎（Search Engine）
- 代码解释器（Code Interpreter）
- 数据库查询（SQL / NoSQL）
- API 调用（REST API / Function Calling）
- 文件系统（File System）
- 浏览器（Browser）
```

### 4.2.3 Memory（记忆）

跨轮次、跨会话保持信息的能力：

| 记忆类型 | 英文 | 说明 |
|---------|------|------|
| 短期记忆 | Short-term Memory | 当前对话的上下文 |
| 长期记忆 | Long-term Memory | 跨会话的知识积累 |
| 工作记忆 | Working Memory | 当前任务相关的临时信息 |
| 情景记忆 | Episodic Memory | 过去交互的经验 |

### 4.2.4 Reasoning（推理）

Agent 进行逻辑推理和判断的能力：
- 多步推理（Multi-step Reasoning）
- 自我反思（Self-Reflection）
- 因果推理（Causal Reasoning）

### 4.2.5 Self-Improvement（自我改进）

Agent 从自身经验中学习和改进：
- 自我对弈（Self-Play）
- 自我纠正（Self-Correction）
- 经验回放（Experience Replay）

### 4.2.6 Perception（感知）

处理多模态输入的能力：
- 图像理解（Image Understanding）
- 视频理解（Video Understanding）
- 音频处理（Audio Processing）

---

## 4.3 POMDP 框架详解

Agentic RL 的数学形式化：

```
POMDP = (S, A, P, R, γ, Ω, O)

S = State Space（状态空间）
A = Action Space（动作空间）
P = Transition Function（状态转移函数）
R = Reward Function（奖励函数）
γ = Discount Factor（折扣因子）
Ω = Observation Space（观测空间）
O = Observation Function（观测函数）：O(o|s',a) 在执行动作 a 转移到 s' 后观测到 o 的概率
```

**为什么是 POMDP 而不是 MDP？**

在 Agent 场景中：
- 网页 Agent 只能看到当前页面 HTML（Observation），不知道整个网站结构（State）
- 搜索 Agent 只能看到返回的 top-K 结果，不知道完整的搜索索引
- 代码 Agent 只能看到运行输出，不知道底层系统状态

---

## 4.4 Rollout（展开）的结构差异

### 传统 RLHF 的 Rollout

```
Prompt: "什么是机器学习？"
  ↓
[LLM generates response]
  ↓
Response: "机器学习是..."
  ↓
Reward Model scores: 0.85
  ↓
Done.（单轮结束）
```

### Agentic RL 的 Rollout

```
Task: "查找最新的Python版本并写一个Hello World"
  ↓
Turn 1: Agent → "我需要搜索Python最新版本"
  → Tool Call: search("latest Python version")
  → Tool Result: "Python 3.13 is the latest..."
  ↓
Turn 2: Agent → "最新版本是3.13，现在写代码"
  → Tool Call: code_interpreter("print('Hello World')")
  → Tool Result: "Hello World\n"
  ↓
Turn 3: Agent → "任务完成。Python最新版本是3.13，Hello World代码..."
  ↓
Reward: 1.0（任务成功）
  ↓
Done.（多轮结束）
```

每次 Rollout 产生一条 **Trajectory（轨迹）**：
```
τ = (o_1, a_1, r_1, o_2, a_2, r_2, ..., o_T, a_T, r_T)
```

---

## 4.5 Agentic RL 的挑战

| 挑战 | 英文术语 | 说明 |
|------|---------|------|
| 奖励稀疏 | Sparse Reward | 多轮交互后才有最终奖励 |
| 信用分配 | Credit Assignment | 哪些步骤对最终结果有贡献？ |
| 环境延迟 | Environment Latency | 工具调用等待时间导致 GPU 空闲 |
| 探索空间大 | Large Exploration Space | 可能的行动序列组合爆炸 |
| 奖励黑客 | Reward Hacking | 模型找到"作弊"方式获得高奖励 |
| 训练不稳定 | Training Instability | 多轮交互的方差大，训练难收敛 |
| 评估困难 | Evaluation Difficulty | Agent 行为多样，难以自动化评估 |

---

## 4.6 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Degenerate Single-step MDP | 退化的单步马尔可夫决策过程 |
| Temporally Extended POMDP | 时间扩展的部分可观测马尔可夫决策过程 |
| Agentic Capabilities | 智能体能力 |
| Planning | 规划 |
| Tool Use | 工具使用 |
| Memory (Short/Long-term) | 记忆（短期/长期） |
| Reasoning | 推理 |
| Self-Improvement | 自我改进 |
| Perception | 感知 |
| Trajectory | 轨迹 |
| Credit Assignment | 信用分配 |
| Reward Hacking | 奖励黑客 |

---

## 4.7 小结

- Agentic RL 将 LLM RL 从单步 MDP 扩展为多步 POMDP
- 六大核心能力：规划、工具使用、记忆、推理、自我改进、感知
- 多轮交互带来新挑战：信用分配、稀疏奖励、环境延迟
- 下一步：了解 Agentic RL 的应用场景 → [05 - Agentic RL 的应用场景](./05-Agentic-RL的应用场景.md)

---

## 练习题

1. **对比分析**：请用自己的话解释为什么传统 RLHF 被称为"退化的单步 MDP"，而 Agentic RL 是"时间扩展的 POMDP"？

2. **案例识别**：以下属于六大 Agentic 能力中的哪一种？
   - Agent 说："我需要先搜索航班信息，再比较价格，最后下单"
   - Agent 说："让我查一下用户昨天的购买记录"
   - Agent 说："我之前的方案不对，让我换一种方法试试"

3. **挑战思考**：在多轮交互中，如果 Agent 在第 1 步搜索了错误的信息，导致第 5 步的任务失败——这对应 Agentic RL 中的什么挑战？如何缓解？

> **参考答案**：1. 见 4.1 节对比图；2. 规划 → 记忆 → 自我改进；3. 对应"信用分配"挑战，可通过过程奖励（Process Reward）或优势函数（Advantage）来追溯各步骤的贡献。
