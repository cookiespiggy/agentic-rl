# 06 - PPO 算法详解

> **学习目标**：深入理解 Proximal Policy Optimization（PPO，近端策略优化），这是 Agentic RL 的基石算法之一。

---

## 6.1 PPO 是什么？

PPO 由 OpenAI 于 2017 年提出，是目前最常用的 On-Policy（同策略）RL 算法之一。

**核心思想**：每次策略更新都限制在一个"信任区域"内，避免一次更新太多导致训练崩溃。

**类比理解**：学习骑自行车——不要一次做太大的改变（比如突然放手），而是每次小幅调整姿势。

---

## 6.2 PPO 的数学推导

### 6.2.1 策略梯度基础

标准策略梯度：

```
∇J(θ) = E[∇log π_θ(a|s) · A(s,a)]

其中 A(s,a) = Advantage（优势函数）
表示"执行动作 a 比平均水平好多少"
```

### 6.2.2 Importance Sampling（重要性采样）

为了重用数据，引入重要性采样比率：

```
r(θ) = π_θ(a|s) / π_θ_old(a|s)

r(θ) > 1 → 新策略比旧策略更倾向于选择这个动作
r(θ) < 1 → 新策略比旧策略更不倾向于选择这个动作
r(θ) = 1 → 新旧策略相同
```

### 6.2.3 PPO-Clip 目标函数

```
L_CLIP(θ) = E[min(r(θ) · A(s,a), clip(r(θ), 1-ε, 1+ε) · A(s,a))]

其中 ε 是超参数（通常 0.1-0.2），控制策略更新的幅度

clip(r(θ), 1-ε, 1+ε) 的含义：
- 如果 r(θ) > 1+ε，则截断为 1+ε（不允许增加太多）
- 如果 r(θ) < 1-ε，则截断为 1-ε（不允许减少太多）
```

**直觉理解**：
- 如果某个动作好（A > 0），增加其概率，但增加幅度不超过 ε
- 如果某个动作差（A < 0），减少其概率，但减少幅度不超过 ε

---

## 6.3 PPO 在 LLM 中的应用

### 6.3.1 四个模型组件

```
┌─────────────────┐   ┌─────────────────┐
│  Policy Model   │   │ Reference Model │
│  π_θ（策略模型）  │   │ π_ref（参考模型） │
│  正在训练的模型   │   │ 冻结的SFT模型    │
└─────────────────┘   └─────────────────┘

┌─────────────────┐   ┌─────────────────┐
│  Reward Model   │   │  Value Model    │
│  R_φ（奖励模型）  │   │  V_ψ（价值模型） │
│  评估回复质量     │   │  Critic（评论家） │
└─────────────────┘   └─────────────────┘
```

### 6.3.2 PPO 训练循环

```
for each iteration:
    1. Rollout Phase（展开阶段）
       - Policy Model 生成回复
       - Reward Model 给每个回复打分
    
    2. Compute Advantage（计算优势）
       - Value Model 估计每个 token 的价值
       - 计算 GAE（Generalized Advantage Estimation，广义优势估计）
    
    3. Update Policy（更新策略）
       - 用 PPO-Clip 目标函数更新 Policy Model
       - KL 散度约束防止偏离 Reference Model 太远
    
    4. Update Value Model（更新价值模型）
       - 用实际奖励更新 Value Model
```

### 6.3.3 KL 散度惩罚

L_total = L_CLIP(θ) + β · KL(π_θ || π_ref)

KL 散度防止 Policy 偏离 Reference Model 太远
β 控制约束强度（通常 0.01~0.1）

> 另一种常见做法是将 KL 作为奖励的惩罚项（Reward_adjusted = R - β·KL），
> 两种方式本质等价。Loss 层面加 KL 更便于梯度传播。

---



## 6.4 PPO 在 Agentic RL 中的特殊考虑

### 6.4.1 多轮交互的 Token 级别 Advantage

在 Agentic RL 中，Rollout 包含多轮对话和工具调用：

```
Turn 1: [user question] → [agent thought + tool call] → [tool result]
Turn 2: [observation] → [agent thought + tool call] → [tool result]
Turn 3: [observation] → [final answer]
```

**关键问题**：如何给每一轮、每一个 token 分配 Advantage？

**解决方案**：
- **Token-level Advantage**：使用 GAE 逐 token 计算
- **Turn-level Advantage**：对每一轮使用独立的奖励
- **Outcome Advantage**：整个轨迹共享最终结果奖励

> 以上三种 Advantage 粒度为文档作者归纳，实际应用中可根据场景混合使用。

### 6.4.2 Masking（掩码）

在多轮 Agent 交互中，需要 Mask 掉以下 token 的 Loss：
- System Prompt tokens
- User message tokens
- Tool result tokens（由环境返回，不是模型生成的）
- Padding tokens（填充标记）

只对 **Agent 生成的 token** 计算 Loss 和 Advantage。

---

## 6.5 PPO 的优缺点

| 优点 | 缺点 |
|------|------|
| 训练稳定，广泛验证 | 需要加载 4 个模型，显存占用大（7B 模型约需 120GB+） |
| 支持多轮交互 | 超参数敏感（ε, β, 学习率等） |
| 有理论保证 | 实现复杂，调参困难 |
| 适合 Agentic RL | 训练速度慢（On-Policy） |

---

## 6.6 代码骨架（伪代码）

```python
import torch

def ppo_update(policy, reference, reward_model, value_model, optimizer, trajectories, config):
    """PPO 更新步骤
    
    Args:
        policy: 策略模型（Policy Model）
        reference: 参考模型（Reference Model，冻结的 SFT 模型）
        reward_model: 奖励模型（Reward Model）
        value_model: 价值模型（Value Model / Critic）
        optimizer: 策略模型的优化器
        trajectories: Rollout 收集的轨迹数据
        config: 训练配置
    """
    
    for epoch in range(config.ppo_epochs):
        for batch in trajectories.batches(config.batch_size):
            # 1. 计算当前策略的概率
            log_probs = policy.log_prob(batch.states, batch.actions)
            old_log_probs = batch.old_log_probs
            
            # 2. 计算重要性采样比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 3. 计算优势（Advantage）
            # 注：LLM 中 Value Model 的输入是已生成的 token sequence 而非环境状态
            advantages = compute_gae(batch.rewards, value_model(batch.sequences))
            
            # 4. PPO-Clip 目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.eps, 1 + config.eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 5. KL 散度惩罚
            kl_div = compute_kl(policy, reference, batch.states)
            total_loss = policy_loss + config.beta * kl_div
            
            # 6. 更新策略
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

---

## 6.7 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Proximal Policy Optimization (PPO) | 近端策略优化 |
| Importance Sampling Ratio | 重要性采样比率 |
| Advantage Function | 优势函数 |
| GAE (Generalized Advantage Estimation) | 广义优势估计 |
| KL Divergence | KL散度（衡量分布差异） |
| Clipping | 截断/裁剪 |
| Trust Region | 信任区域 |
| On-Policy | 同策略 |

---

## 6.8 小结

- PPO 通过限制每次更新的幅度，实现稳定的策略优化
- 在 LLM RL 中需要加载 4 个模型，显存和计算开销大
- 在 Agentic RL 中需要处理多轮交互的 Token-level Advantage
- 下一步：学习 DPO 和 GRPO → [07 - DPO 直接偏好优化](./07-DPO直接偏好优化.md)
