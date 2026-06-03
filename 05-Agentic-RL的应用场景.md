---
title: "05 - Agentic RL 的应用场景"
description: "了解 Agentic RL 在哪些领域有应用，找到你最感兴趣的垂直领域"
difficulty: "入门"
reading_time: "15 min"
tags: ["应用场景", "Web导航", "代码生成", "搜索增强", "数学推理"]
---

# 05 - Agentic RL 的应用场景

> **学习目标**：了解 Agentic RL 在哪些领域有应用，找到你最感兴趣的垂直领域。

---

## 目录

- [5.1 应用场景全景图](#51-应用场景全景图)
- [5.2 重点场景详解](#52-重点场景详解)
- [5.3 垂直领域小模型的机会](#53-垂直领域小模型的机会)
- [5.4 术语速查](#54-术语速查)
- [5.5 小结](#55-小结)
- [练习题](#练习题)

---

## 5.1 应用场景全景图

```
Agentic RL 应用场景
├── Web Navigation（网页导航）
│   ├── 网页浏览与信息提取
│   ├── 表单填写与操作
│   └── 多步骤网页任务
├── Code Generation & SWE（代码生成与软件工程）
│   ├── 代码编写与调试
│   ├── 自动修复 Bug
│   └── 仓库级别代码理解
├── Search & QA（搜索与问答）
│   ├── 多轮搜索与推理
│   ├── 知识密集型问答
│   └── 检索增强生成（RAG）
├── Mathematics & Science（数学与科学）
│   ├── 多步数学推理
│   ├── 定理证明
│   └── 科学实验设计
├── Data Analysis（数据分析）
│   ├── SQL 查询生成
│   ├── 数据可视化
│   └── 报表生成
├── Robotics & Embodied AI（机器人与具身智能）
│   ├── 导航与移动
│   ├── 物体操作
│   └── 人机协作
└── Game Playing（游戏）
    ├── 棋类游戏
    ├── 电子游戏
    └── 沙盒世界
```

---

## 5.2 重点场景详解

### 5.2.1 Web Navigation（网页导航）

**目标**：训练 Agent 在真实网页上完成任务（填表、购物、信息查询等）。

**代表性工作**：
- **WebRL**（ICLR 2025）：使用 Self-Evolving Curriculum（自进化课程）和 Outcome Reward Model（结果奖励模型）训练 Web Agent
- **WebAgent-R1**：使用 RL 训练 LLM 在浏览器中自主操作

**训练环境**：
- BrowserGym：标准化的浏览器交互环境
- MiniWoB / WebArena / VisualWebArena

**奖励设计**：
- 任务完成 = 1，未完成 = 0
- 可选：步骤效率奖励（用更少步骤完成 = 更高奖励）

### 5.2.2 Software Engineering（软件工程）

**目标**：训练 Agent 阅读 Issue、理解代码库、编写和提交修复代码。

**代表性工作**：
- **ReTool**（ICLR 2025）：强化学习增强的代码工具使用
- **SWE-agent**：自动修复 GitHub Issue 的 Agent

**训练环境**：
- SWE-bench：基于真实 GitHub 仓库的代码修复 Benchmark
- CodeSandbox：安全的代码执行沙盒

### 5.2.3 Search-Augmented Reasoning（搜索增强推理）

**目标**：训练 Agent 在推理过程中主动搜索外部知识。

**代表性工作**：
- **Search-R1**（2025）：RL 训练的多轮搜索与推理框架
- **R1-Searcher**：让模型学会何时搜索、搜索什么

**训练环境**：
- 自定义搜索引擎 API
- Reasoning Gym：推理训练数据集

### 5.2.4 Mathematics（数学推理）

**目标**：训练 Agent 进行多步数学推理，可使用计算工具。

**代表性工作**：
- **DAPO**：开源 RL 算法，在数学推理上取得 SOTA
- **DeepSeek-R1**：通过 RL 涌现出推理链（Chain-of-Thought）

**训练数据**：
- GSM8K：小学数学应用题
- MATH：竞赛级数学题
- AIME / IMO：高级数学竞赛

---

## 5.3 垂直领域小模型的机会

这是本教程的核心主题——**用小模型在垂直领域超越大模型**。

### 5.3.1 为什么小模型在垂直领域有优势？

| 维度 | 大模型 (70B+) | 小模型 (7B-14B) + Agentic RL |
|------|-------------|--------------------------|
| 推理成本 | 高 | 低（可部署在边缘设备） |
| 延迟 | 高 | 低 |
| 专业知识 | 泛化但不深入 | 通过 RL 在特定领域深度优化 |
| 部署灵活性 | 需要多卡 | 单卡甚至 CPU 可部署 |
| 更新迭代 | 慢（重新训练成本高） | 快（可频繁 RL 迭代） |

### 5.3.2 适合小模型 + Agentic RL 的场景

1. **客服 Agent**：在特定产品的知识库上训练
2. **数据分析 Agent**：在特定数据库/业务上训练
3. **代码审查 Agent**：在特定代码库上训练
4. **医疗问答 Agent**：在特定医学领域训练
5. **金融分析 Agent**：在特定金融工具上训练

### 5.3.3 成功案例参考

- **Mock Worlds, Real Skills**（Lyu et al., arXiv:2601.22511, ACL 2026）：提出 SynthAgent 框架，在模拟环境中合成多样化工具使用数据训练小模型，证明 8B-14B 模型通过 Agentic RL 可以在多个 Agent 基准上超越 32B 模型
- **NVIDIA SLM Agents**（Belcak et al., arXiv:2506.02153, 2025）：NVIDIA Research 立场论文，系统论述了 SLM 在 Agent 场景下的成本、延迟和灵活性优势

---

## 5.4 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Web Navigation | 网页导航 |
| Software Engineering (SWE) | 软件工程 |
| Search-Augmented Reasoning | 搜索增强推理 |
| Self-Evolving Curriculum | 自进化课程学习 |
| Outcome Reward Model | 结果奖励模型 |
| Code Sandbox | 代码沙盒 |
| BrowserGym | 浏览器交互环境 |
| SWE-bench | 软件工程基准测试 |
| Chain-of-Thought (CoT) | 思维链 |
| Small Language Model (SLM) | 小语言模型 |
| Edge Deployment | 边缘部署 |

---

## 5.5 小结

- Agentic RL 有广泛的应用场景：网页导航、代码、搜索、数学、游戏等
- 垂直领域的小模型 + Agentic RL 是超越大模型的可行路径
- 选择你感兴趣的领域，后续教程将以实战为导向
- 下一步：开始学习核心算法 → [06 - PPO 算法详解](./06-PPO算法详解.md)

---

## 练习题

1. **场景匹配**：将以下应用场景与其适合的奖励设计配对：
   - ① 网页导航  &nbsp;&nbsp; A. 测试用例通过率
   - ② 软件工程  &nbsp;&nbsp; B. 答案正确性 + 计算工具使用
   - ③ 数学推理  &nbsp;&nbsp; C. 任务完成 + 步骤效率
   - ④ 搜索增强  &nbsp;&nbsp; D. 最终答案正确 + 搜索策略合理性

2. **商业分析**：为什么在垂直领域中，小模型（7B）+ Agentic RL 可能比通用大模型（70B+）更适合生产部署？请列出至少 3 个理由。

3. **案例思考**：如果让你用 Agentic RL 训练一个"客服 Agent"，你会选择哪个场景作为切入点？你的奖励函数如何设计？

> **参考答案**：1. ①-C、②-A、③-B、④-D；2. 见 5.3.1 节表格：推理成本低、延迟低、可频繁迭代、可边缘部署；3. 开放题，合理即可。
