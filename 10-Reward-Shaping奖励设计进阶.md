# 10 - Reward Shaping 奖励设计进阶

> **学习目标**：掌握 Reward Shaping（奖励塑形）的高级技巧，解决 Agentic RL 中稀疏奖励和信用分配的核心难题。

---

## 10.1 为什么需要 Reward Shaping？

在 Agentic RL 中，最常见的问题是 **Sparse Reward（稀疏奖励）**：

```
Task: "在GitHub上修复一个Bug"

Agent 执行了 20 步操作：
  Step 1-19: 各种操作（读代码、搜索、编辑、运行测试...）
  Step 20: 提交修复 → 测试通过 → reward = 1.0

问题：前 19 步都没有奖励信号，模型不知道哪些步骤是对的
```

**Reward Shaping** 通过添加额外的中间奖励信号，引导模型更快学习。

---

## 10.2 奖励函数的分层设计

### 10.2.1 三层奖励架构

```
Layer 3: Task Reward（任务奖励）
  └→ 任务完成/失败，这是最核心的信号
  └→ 例：成功 +1.0，失败 0.0

Layer 2: Progress Reward（进度奖励）
  └→ 衡量 Agent 离目标有多近
  └→ 例：调用了正确的 API +0.2，获得了有用的搜索结果 +0.1

Layer 1: Behavior Reward（行为奖励）
  └→ 鼓励良好的行为模式
  └→ 例：正确的工具调用格式 +0.05，有效的思考步骤 +0.05
```

### 10.2.2 实际奖励函数示例

```python
def layered_reward(trajectory):
    """三层奖励函数"""
    reward = 0.0
    
    # Layer 3: 任务结果
    if task_completed(trajectory):
        reward += 1.0
    elif partial_progress(trajectory):
        reward += 0.3  # 部分进展
    
    # Layer 2: 进度奖励
    for step in trajectory.steps:
        if step.type == "tool_call" and step.successful:
            reward += 0.05  # 成功的工具调用
        if step.type == "search" and step.has_relevant_results:
            reward += 0.1  # 有用的搜索结果
    
    # Layer 1: 行为奖励
    if has_thinking_process(trajectory):
        reward += 0.05  # 有思考过程
    if not has_unnecessary_steps(trajectory):
        reward += 0.05  # 没有不必要的步骤
    
    # 惩罚
    if has_errors(trajectory):
        reward -= 0.1 * count_errors(trajectory)
    
    return reward
```

---

## 10.3 信用分配（Credit Assignment）

### 10.3.1 问题描述

在一条多轮轨迹中，哪些步骤对最终成功有贡献？

```
轨迹：
  T1: 搜索 "Python sort" → 获得文档   ← 关键步骤
  T2: 阅读文档 → 理解 sorted() 用法   ← 关键步骤
  T3: 搜索 "Python file IO" → 无关结果 ← 无关步骤
  T4: 写代码 sorted(data) → 正确       ← 关键步骤
  T5: 输出答案 → 正确                   ← 最终步骤

问题：T3 是无关步骤，不应该获得奖励
```

### 10.3.2 解决方案

**方案一：Turn-level Reward**

对每一轮独立给奖励：

```python
turn_rewards = {
    "T1": 0.2,   # 正确的搜索策略
    "T2": 0.2,   # 有效信息提取
    "T3": -0.05, # 无关搜索（轻微惩罚）
    "T4": 0.3,   # 正确的代码生成
    "T5": 0.3,   # 正确的最终答案
}
```

**方案二：Discounted Return（折扣回报）**

越靠近最终结果的步骤获得越高的权重：

```python
def discounted_credit(trajectory, final_reward, gamma=0.9):
    """计算每个步骤的折扣信用"""
    credits = []
    for i, step in enumerate(trajectory.steps):
        # 距离最终步骤越远，折扣越大
        distance = len(trajectory.steps) - 1 - i
        credit = final_reward * (gamma ** distance)
        credits.append(credit)
    return credits
```

---

## 10.4 TIPS: Turn-level Reward Shaping

**TIPS（Turn-level Information-Potential Reward Shaping）** 是 2026 年的重要工作：

```
核心思想：
为搜索增强的 LLM Agent 提供 Turn-level 的奖励信号，
利用每轮搜索返回的信息量（Information Potential）来评估该轮的质量。

具体做法：
1. 每轮搜索后，评估搜索结果与任务的相关性
2. 相关性高 → 高 turn reward
3. 相关性低 → 低/负 turn reward
4. 最终答案正确性 → outcome reward
```

**效果**：在 QA 任务上显著优于纯 Outcome Reward 的 RLVR 基线。

---

## 10.5 Process Reward Model（PRM，过程奖励模型）

除了最终结果，还可以训练一个评估"推理/行动过程质量"的模型：

```python
# PRM 训练数据示例
prm_data = [
    {
        "trajectory": [step1, step2, step3],
        "step_labels": [
            {"step": step1, "quality": "good"},   # 正确的搜索
            {"step": step2, "quality": "neutral"}, # 中性操作
            {"step": step3, "quality": "good"},   # 正确的推理
        ]
    }
]

# PRM 输出每步的质量分数
step_rewards = prm.predict(trajectory)
# → [0.8, 0.5, 0.9]
```

---

## 10.6 奖励归一化（Reward Normalization）

奖励的数值范围对训练稳定性至关重要：

```python
import numpy as np

def exponential_moving_average(values, alpha=0.1):
    """指数移动平均"""
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return np.array(ema)

def exponential_moving_std(values, alpha=0.1):
    """指数移动标准差"""
    ema = exponential_moving_average(values, alpha)
    squared_diff = (np.array(values) - ema) ** 2
    return np.sqrt(exponential_moving_average(squared_diff.tolist(), alpha))

def normalize_rewards(rewards, method="group"):
    """奖励归一化"""
    if method == "group":
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8
        return (rewards - mean) / std
    
    elif method == "running":
        running_mean = exponential_moving_average(rewards)
        running_std = exponential_moving_std(rewards)
        return (rewards - running_mean) / (running_std + 1e-8)
    
    elif method == "clip":
        return np.clip(rewards, -1.0, 1.0)
```

---

## 10.7 奖励设计最佳实践

### 10.7.1 原则

1. **简洁优先**：先用最简单的 Outcome Reward 跑通，再逐步增加
2. **避免过拟合**：奖励不要太过特化，否则模型只在训练环境有效
3. **防范 Hacking**：始终检查模型是否在"钻空子"
4. **可验证性**：尽量用客观可计算的奖励，减少主观判断

### 10.7.2 调试清单

```
□ 奖励的分布是否合理？（均值、方差）
□ 是否存在所有样本奖励相同的情况？（GRPO 会梯度为0）
□ 模型是否在重复某种能获得奖励但实际无效的行为？
□ 奖励信号的频率是否足够？（太稀疏 → 训练慢）
□ KL 散度是否在合理范围内？（太大 → 模型偏离太远）
```

---

## 10.8 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Reward Shaping | 奖励塑形 |
| Sparse / Dense Reward | 稀疏 / 稠密奖励 |
| Credit Assignment | 信用分配 |
| Process Reward Model (PRM) | 过程奖励模型 |
| Outcome Reward Model (ORM) | 结果奖励模型 |
| Discounted Return | 折扣回报 |
| Reward Normalization | 奖励归一化 |
| Reward Hacking | 奖励黑客 |

---

## 10.9 小结

- Reward Shaping 通过添加中间奖励信号解决稀疏奖励问题
- 三层奖励架构：Task + Progress + Behavior
- 信用分配是多轮 Agent 训练的关键问题
- TIPS 和 PRM 提供了 Turn-level 的奖励设计思路
- 下一步：进入训练工程阶段 → [11 - 训练框架选型与搭建](./11-训练框架选型与搭建.md)
