# 09 - RLVR 可验证奖励的强化学习

> **学习目标**：理解 RLVR（Reinforcement Learning with Verifiable Rewards）范式，这是 Agentic RL 中奖励设计的核心理念。

---

## 9.1 什么是 RLVR？

**RLVR（Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习）** 是 2025-2026 年兴起的一种训练范式：

> 用**客观可验证的外部信号**作为奖励，替代人类偏好标注或训练 Reward Model。

```
传统 RLHF：
  prompt → generate → Reward Model scores → train

RLVR：
  prompt → generate → 客观验证（代码运行/答案检查/环境反馈） → train
```

---

## 9.2 RLVR 的核心优势

| 维度 | RLHF | RLVR |
|------|------|------|
| 奖励来源 | 人类标注 / Reward Model | 客观验证程序 |
| 成本 | 高（人工标注昂贵） | 低（自动化验证） |
| 奖励质量 | 受人类主观性影响 | 客观、确定、可重复 |
| 可扩展性 | 差（标注瓶颈） | 强（可大规模自动化） |
| 奖励黑客风险 | 中（Reward Model 可被欺骗） | 低（难以欺骗客观验证） |

---

## 9.3 RLVR 的奖励类型

### 9.3.1 答案验证（Answer Verification）

```python
# 数学题：答案正确性验证
def math_reward(model_answer, ground_truth):
    """数学题的奖励函数"""
    extracted = extract_answer(model_answer)  # 从回复中提取答案
    if extracted == ground_truth:
        return 1.0
    else:
        return 0.0
```

**代表数据集**：GSM8K、MATH、AIME

### 9.3.2 代码验证（Code Verification）

```python
# 代码题：测试用例验证
def code_reward(generated_code, test_cases):
    """代码生成的奖励函数"""
    try:
        for test_input, expected_output in test_cases:
            actual_output = execute(generated_code, test_input)
            if actual_output != expected_output:
                return 0.0
        return 1.0  # 所有测试通过
    except Exception:
        return 0.0  # 运行错误
```

### 9.3.3 环境验证（Environment Verification）

```python
# Agent 任务：环境状态验证
def agent_reward(trajectory, success_condition):
    """Agent 任务的奖励函数"""
    final_state = trajectory.get_final_state()
    
    if success_condition(final_state):
        return 1.0  # 任务成功
    else:
        return 0.0  # 任务失败
```

### 9.3.4 格式验证（Format Verification）

```python
# 验证模型是否按正确格式输出
def format_reward(response):
    """格式正确性奖励"""
    reward = 0.0
    if has_think_tags(response):      # 有  标签
        reward += 0.1
    if has_tool_call_format(response): # 正确的工具调用格式
        reward += 0.1
    if not has_hallucination(response): # 无明显幻觉（实际需外部知识库校验，此处仅示意）
        reward += 0.1
    return reward
```

---

## 9.4 RLVR + GRPO 的标准流水线

```
┌──────────────────────────────────────────────────────────┐
│                   RLVR + GRPO Pipeline                    │
│                                                          │
│  1. 从数据集取一个 batch 的 prompts                        │
│       ↓                                                  │
│  2. 对每个 prompt，用当前策略采样 G 个回复                  │
│       ↓                                                  │
│  3. 用 Verifiable Reward 函数给每个回复打分                │
│       ↓                                                  │
│  4. 计算组内相对 Advantage                                │
│       ↓                                                  │
│  5. 用 GRPO Loss 更新策略                                 │
│       ↓                                                  │
│  6. 重复                                                  │
└──────────────────────────────────────────────────────────┘
```

---

## 9.5 RLVR 在 Agentic RL 中的特殊形式

### 9.5.1 Outcome Reward（结果奖励）

最简单也最常用的形式：只看最终任务是否完成。

```
Agent 任务完成 → reward = 1.0
Agent 任务失败 → reward = 0.0
```

### 9.5.2 Turn-level Reward（轮次级奖励）

为每一轮交互提供中间奖励：

```
Turn 1: Agent 正确选择了工具 → +0.2
Turn 2: Agent 正确解析了结果 → +0.3
Turn 3: Agent 给出了正确答案 → +0.5（最终奖励）
Total: 1.0
```

### 9.5.3 Step-wise Reward（步骤级奖励）

更细粒度的奖励，每一步都有反馈：

```
Step 1: 正确识别任务类型 → +0.1
Step 2: 调用正确的 API → +0.15
Step 3: 正确处理返回值 → +0.1
Step 4: 正确推理 → +0.15
Step 5: 输出正确答案 → +0.5
```

> **TIPS（Turn-level Information-Potential Reward Shaping）**（Tao et al., 2026）证明了 Turn-level Reward 在搜索增强 Agent 中显著优于纯 Outcome Reward。

> ⚠️ **Credit Assignment 问题**：Step-wise Reward 面临的核心挑战是——中间步骤正确但最终结果错误时，奖励应如何分配？实际中常用「最终结果 + 中间步骤」的混合方案，或用 Process Reward Model（PRM）对每一步做独立评估。

---

## 9.6 Reward Hacking（奖励黑客）及其防御

### 9.6.1 常见的 Reward Hacking 行为

```
1. 格式欺骗：模型输出正确格式但内容错误
   → 防御：同时检查格式和内容

2. 工具滥用：模型不断调用同一工具刷分
   → 防御：限制工具调用次数，加入效率惩罚

3. 重复回答：模型重复输出已知正确答案
   → 防御：检测重复，降低重复奖励

4. 简短逃避：模型给出极简回复以避免出错
   → 防御：设置最小长度奖励
```

### 9.6.2 防御策略

```python
def robust_reward(response, ground_truth, trajectory=None):
    """鲁棒的奖励函数"""
    reward = 0.0
    
    # 1. 核心正确性
    if verify_correctness(response, ground_truth):
        reward += 1.0
    else:
        return -0.5  # 错误惩罚
    
    # 2. 效率奖励（用更少步骤完成）
    if trajectory:
        num_steps = len(trajectory.steps)
        efficiency = max(0, 1 - num_steps / max_steps)
        reward += 0.1 * efficiency
    
    # 3. 格式奖励
    if has_correct_format(response):
        reward += 0.05
    
    # 4. 惩罚重复
    if has_repetition(response):
        reward -= 0.2
    
    return reward
```

---

## 9.7 实际案例：用 RLVR 训练数学推理 Agent

```python
# 简化的 RLVR 训练循环
from verl import GRPOTrainer

def math_reward_fn(completions, ground_truths):
    """数学题 RLVR 奖励函数"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        answer = extract_final_answer(completion)
        
        if answer is None:
            rewards.append(-0.5)  # 格式错误（未输出答案标签），给负分
        elif verify_math_answer(answer, gt):
            rewards.append(1.0)   # 答案正确
        else:
            rewards.append(0.0)   # 答案错误
    
    return rewards

# 使用 verl 框架训练
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B",
    reward_fn=math_reward_fn,
    dataset="gsm8k_train",
    num_samples_per_prompt=8,  # GRPO 的 G
    learning_rate=1e-6,
    kl_coeff=0.01,
)
trainer.train()
```

---

## 9.8 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| RLVR (RL with Verifiable Rewards) | 基于可验证奖励的强化学习 |
| Verifiable Reward | 可验证奖励 |
| Outcome Reward | 结果奖励 |
| Process Reward | 过程奖励 |
| Turn-level Reward | 轮次级奖励 |
| Step-wise Reward | 步骤级奖励 |
| Reward Hacking | 奖励黑客（模型钻奖励漏洞） |
| Reward Shaping | 奖励塑形 |

---

## 9.9 小结

- RLVR 用客观可验证的奖励替代人类偏好，是 Agentic RL 的核心范式
- 奖励设计直接影响训练效果：Outcome Reward 简单但稀疏，Turn-level Reward 更密集但设计更复杂
- 需要防范 Reward Hacking
- 下一步：学习 Reward Shaping → [10 - Reward Shaping 奖励设计进阶](./10-Reward-Shaping奖励设计进阶.md)
