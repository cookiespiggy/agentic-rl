# 08 - GRPO 深度解析

> **学习目标**：深入理解 GRPO（Group Relative Policy Optimization，组相对策略优化），这是当前 Agentic RL 最主流的算法。

---

## 8.1 GRPO 的起源

GRPO 由 **DeepSeek** 团队在 DeepSeek-R1 中提出，核心理念：
- 不需要 Critic/Value Model（价值模型），节省显存
- 通过 **组内相对比较** 来计算 Advantage，简洁高效
- 继承了 PPO 的 Online 探索和 DPO 的模型简洁性

**DeepSeek-R1-Zero 的训练结果震惊了 AI 界**：通过纯 RL 训练（不经过 SFT），模型"涌现"出了思维链（Chain-of-Thought）推理能力。
完整版 DeepSeek-R1 在此基础上加入了冷启动 SFT 数据（数千条 curated CoT 数据），进一步提升了推理质量和可读性。

---

## 8.2 GRPO 的核心思想

### 8.2.1 Group Sampling（组采样）

对每个 prompt，生成一组 G 个回复（通常 G=4~16）：

```
Prompt: "求解方程 2x + 5 = 13"
  ↓ 采样 G=4 个回复

y_1: "x = 4"               → reward_1 = 1.0（正确）
y_2: "x = 5"               → reward_2 = 0.0（错误）
y_3: "x = 4, 验证: 2×4+5=13" → reward_3 = 1.0（正确）
y_4: "不知道怎么做"          → reward_4 = 0.0（放弃）
```

### 8.2.2 Relative Advantage（相对优势）

**关键创新**：用组内的均值和标准差来归一化奖励，得到每个回复的相对优势。

```
对于组内第 i 个回复：

A_i = (r_i - mean(r_1,...,r_G)) / std(r_1,...,r_G)

> 当组内所有奖励值相同时 std≈0，需添加微小常数 ε（如 1e-8）防止除零。

其中：
- r_i = 第 i 个回复的奖励
- mean = 组内奖励均值
- std = 组内奖励标准差
```

**示例**：
```
奖励：[1.0, 0.0, 1.0, 0.0]
均值 = 0.5, 标准差 = 0.5

A_1 = (1.0 - 0.5) / 0.5 = +1.0  → 比平均好
A_2 = (0.0 - 0.5) / 0.5 = -1.0  → 比平均差
A_3 = (1.0 - 0.5) / 0.5 = +1.0  → 比平均好
A_4 = (0.0 - 0.5) / 0.5 = -1.0  → 比平均差
```

### 8.2.3 GRPO 目标函数

```
L_GRPO = E[-1/G Σ_i min(ρ_i · A_i, clip(ρ_i, 1-ε, 1+ε) · A_i) + β · KL(π_θ || π_ref)]

其中：
- ρ_i = π_θ(y_i|x) / π_θ_old(y_i|x)  重要性采样比率
- A_i = 组内相对优势
- ε = 截断参数（通常 0.2）
- β = KL 惩罚系数
```

---

## 8.3 GRPO vs PPO 对比

| 维度 | GRPO | PPO |
|------|------|-----|
| 需要的模型 | 2个 (Policy + Reference) | 4个 (Policy + Ref + Reward + Value) |
| Advantage 计算 | 组内相对排序 | GAE（需要 Value Model） |
| 显存占用 | 低 | 高（约2倍） |
| 实现复杂度 | 简单 | 复杂 |
| 训练速度 | 快 | 慢 |
| Agentic RL 适用性 | 优秀 | 优秀 |

---

## 8.4 GRPO 在 Agentic RL 中的应用

### 8.4.1 Trajectory-level GRPO（轨迹级 GRPO）

对同一个任务，采样多条完整轨迹：

```
Task: "在Amazon上找一本Python书"

Trajectory 1: search("Python book") → click(first result) → done
  → reward = 1.0（成功）

Trajectory 2: browse("amazon.com") → search("Python") → scroll → click → done
  → reward = 1.0（成功，但步骤多）

Trajectory 3: search("Python tutorial") → click(wrong page) → stuck
  → reward = 0.0（失败）

Trajectory 4: search("Python book amazon") → click(first) → done
  → reward = 1.0（成功）

Group advantage: [0.58, 0.58, -1.73, 0.58]   # 精确计算值
```

### 8.4.2 Token-level GRPO

在多轮交互中，对每个 token 计算组内相对优势：

```
Turn 1 tokens: [agent thinks about search strategy]
Turn 2 tokens: [agent calls search tool]
Turn 3 tokens: [agent processes result and answers]

每个 token 的 advantage 来自其所在轨迹的组内相对排名
```

---

## 8.5 GRPO 的改进变体

### 8.5.1 DAPO（Dynamic Advantage Policy Optimization）

ByteDance 在 DeepSeek-R1 基础上的改进：
- 动态过滤全对/全错的 prompt（避免梯度为 0）
- Token-level loss 替代 Sequence-level loss
- 更大的 KL 截断值
- 动态采样策略

### 8.5.2 Tree-GRPO（ICLR 2026）

阿里巴巴高德团队提出的改进：
- 使用 **Tree Search（树搜索）** 替代独立的链式 Rollout
- 多个轨迹共享前缀，减少重复计算
- 更高效的探索策略

### 8.5.3 GiGPO（Group-in-Group Policy Optimization）

专为 LLM Agent 设计的改进：
- 两层 Group：Task Group（任务组）+ Response Group（回复组）
- 既考虑任务间的难度差异，也考虑组内的回复质量

### 8.5.4 Training-Free GRPO（OpenReview 2026）

- 无需训练即可实现 GRPO 效果的推理时方法
- 在 Inference 时进行组内投票/排序

---

## 8.6 GRPO 代码实现

```python
import torch
import torch.nn.functional as F

def compute_grpo_loss(policy, policy_old, reference, reward_function, prompts, num_samples=4, eps=0.2, beta=0.01):
    """GRPO 损失计算
    
    Args:
        policy: 当前策略模型（Policy Model）
        policy_old: 旧策略模型（冻结副本，用于计算重要性采样比率）
                  注：建议每 N 步（如 500 步）从 policy 同步一次权重
        reference: 参考模型（Reference Model，冻结的 SFT 模型）
        reward_function: 外部奖励函数（如答案验证、环境反馈等）
        prompts: 提示词列表
        num_samples: 每个 prompt 的采样数量 G
        eps: PPO-Clip 截断范围
        beta: KL 惩罚系数
    """
    
    losses = []
    for prompt in prompts:
        # 1. 对每个 prompt 采样 G 个回复
        responses = []
        rewards = []
        for _ in range(num_samples):
            response = policy.generate(prompt)
            reward = reward_function(prompt, response)  # 外部奖励
            responses.append(response)
            rewards.append(reward)
        
        # 2. 计算组内相对优势
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        advantages = (rewards_tensor - mean_r) / (std_r + 1e-8)
        
        # 3. 计算每个回复的策略比率
        for i, response in enumerate(responses):
            log_prob_new = policy.log_prob(prompt, response)
            log_prob_old = policy_old.log_prob(prompt, response)
            ratio = torch.exp(log_prob_new - log_prob_old)
            
            # 4. PPO-Clip 目标
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages[i]
            policy_loss = -torch.min(surr1, surr2)
            
            # 5. KL 惩罚
            kl = compute_kl_divergence(policy, reference, prompt, response)
            loss = policy_loss + beta * kl
            losses.append(loss)
    
    return torch.stack(losses).mean()
```

---

## 8.7 关键超参数指南

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| `num_samples (G)` | 4~16 | 每个 prompt 的采样数量 |
| `eps (ε)` | 0.2 | PPO 截断范围 |
| `beta (β)` | 0.01~0.1 | KL 惩罚系数 |
| `learning_rate` | 1e-6 ~ 5e-7 | 学习率（小模型可用大一点） |
| `temperature` | 0.8~1.0 | 生成温度（控制探索） |
| `max_new_tokens` | 512~2048 | 最大生成长度 |

---

## 8.8 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Group Relative Policy Optimization (GRPO) | 组相对策略优化 |
| Group Sampling | 组采样 |
| Relative Advantage | 相对优势 |
| Importance Sampling Ratio | 重要性采样比率 |
| DAPO | 动态优势策略优化 |
| Tree-GRPO | 树搜索 GRPO |
| GiGPO | 组中组策略优化 |
| Emergent Ability | 涌现能力 |

---

## 8.9 小结

- GRPO 是当前 Agentic RL 的主流算法，用组内相对比较替代 Value Model
- 比 PPO 节省约 50% 显存，实现更简单
- DAPO、Tree-GRPO、GiGPO 等变体针对不同场景做了优化
- 下一步：学习 RLVR → [09 - RLVR 可验证奖励的强化学习](./09-RLVR可验证奖励的强化学习.md)
