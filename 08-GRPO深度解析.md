# 08 - GRPO 深度解析

> **学习目标**：深入理解 GRPO（Group Relative Policy Optimization，组相对策略优化），这是当前 Agentic RL 最主流的算法。

---

## 目录

- [8.1 GRPO 的起源](#81-grpo-的起源)
- [8.2 GRPO 的核心思想](#82-grpo-的核心思想)
- [8.3 GRPO vs PPO 对比](#83-grpo-vs-ppo-对比)
- [8.4 GRPO 在 Agentic RL 中的应用](#84-grpo-在-agentic-rl-中的应用)
- [8.5 GRPO 的改进变体](#85-grpo-的改进变体)
- [8.6 GRPO 代码实现](#86-grpo-代码实现)
- [8.7 关键超参数指南](#87-关键超参数指南)
- [8.8 术语速查](#88-术语速查)
- [8.9 小结](#89-小结)
- [练习题](#练习题)

---

## 8.1 GRPO 的起源

GRPO 由 **DeepSeek** 团队**首次提出于 DeepSeekMath**（[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)），并在 **DeepSeek-R1**（[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）中完成**大规模应用**。核心理念：
- 不需要 Critic/Value Model（价值模型），省掉一个同规模的网络
- 通过 **组内相对比较** 来计算 Advantage，简洁高效
- 继承了 PPO 的 Online 探索特性；模型上的简洁性来自 **省掉 Critic/Value Model**，与 DPO 无关

**DeepSeek-R1-Zero 的训练结果震惊了 AI 界**：通过纯 RL 训练（不经过 SFT），模型"涌现"出了思维链（Chain-of-Thought）推理能力。
完整版 DeepSeek-R1 在此基础上加入了冷启动 SFT 数据（**约 80 万条，about 800k samples** 的 curated CoT 数据），进一步提升了推理质量和可读性。

---

## 8.2 GRPO 的核心思想

### 8.2.1 Group Sampling（组采样）

对每个 prompt，生成一组 G 个回复：

> **G 的取值**：DeepSeekMath / DeepSeek-R1 原文用的是 **G=64**——组越大，组内相对比较的统计量越稳，但 rollout 成本随 G 线性上升。**4~16 是后续工程实践中的折中推荐**，不是论文设定；组太小时组内标准差本身就不稳定，优势信号噪声大。

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
- std = 组内奖励的**样本标准差**（ddof=1），即 verl / TRL 里 `torch.std()` 的默认行为
```

**示例**：
```
奖励：[1.0, 0.0, 1.0, 0.0]
均值 = 0.5, 样本标准差（ddof=1）= sqrt(1.0 / 3) ≈ 0.5774

A_1 = (1.0 - 0.5) / 0.5774 ≈ +0.866  → 比平均好
A_2 = (0.0 - 0.5) / 0.5774 ≈ -0.866  → 比平均差
A_3 = (1.0 - 0.5) / 0.5774 ≈ +0.866  → 比平均好
A_4 = (0.0 - 0.5) / 0.5774 ≈ -0.866  → 比平均差
```

> **口径提醒**：如果用**总体标准差**（ddof=0，NumPy `np.std()` 的默认），上面的数值会变成 ±1.0（[1.0, 1.0, 0.0, 1.0] 那组则是 [0.58, 0.58, -1.73, 0.58]）。**verl 与 TRL 的 GRPO 优势归一化用的都是样本标准差（ddof=1）**，本章代码与该口径对齐。两种口径只差一个 `sqrt(G/(G-1))` 的缩放因子，不改变同一组内优势的正负号与相对大小关系。

### 8.2.3 GRPO 目标函数

GRPO 的原始形式（DeepSeekMath eq.10）按 **token** 展开，并且在**每条序列内部**做长度归一化：

```
L_GRPO = E_{x, {o_i}₁^G ~ π_old(·|x)} [ (1/G) Σ_{i=1}^{G} (1/|o_i|) Σ_{t=1}^{|o_i|}
             { min( ρ_{i,t} · Â_{i,t},  clip(ρ_{i,t}, 1-ε, 1+ε) · Â_{i,t} )
               - β · D̂_{i,t}^{KL} } ]

其中：
- o_i = 第 i 条回复（序列），|o_i| = 它的 token 数
- ρ_{i,t} = π_θ(o_{i,t}|x,o_{i,<t}) / π_old(o_{i,t}|x,o_{i,<t})   重要性采样比率
- Â_{i,t} = 广播到第 t 个 token 上的组内相对优势（见 8.4.2）
- D̂_{i,t}^{KL} = token 级 KL 的 k3 估计量（见下）
- ε = 截断参数（通常 0.2）
- β = KL 惩罚系数（verl 里对应 algorithm.kl_ctrl.kl_coef）
```

**token 级 KL（k3 估计量）**：KL 惩罚是**逐 token** 累加的，常用 Schulman 的 k3 无偏估计量：

```
D̂_{i,t}^{KL} = π_ref(o_{i,t}|x,o_{i,<t}) / π_θ(o_{i,t}|x,o_{i,<t})
             - log( π_ref(o_{i,t}|x,o_{i,<t}) / π_θ(o_{i,t}|x,o_{i,<t}) ) - 1
```

（verl 的 `low_var_kl` 估计量与这个 k3 等价，都是对 `D_KL(π_θ || π_ref)` 的低方差无偏估计。）

**两个容易忽略的点**：

1. **`1/|o_i|` 长度归一化**——每条序列先在自己内部按 token 求平均，再对 G 条序列求平均。副作用是：**长序列里的单个 token 对梯度的贡献天然更小**，不同长度的轨迹被拉到同一量级。
2. **序列级平均的偏置**——"先按序列内平均、再按序列平均" 意味着 **每条序列等权**，短序列的 token 被放大、长序列的 token 被摊薄。DAPO 之所以要改成 **token-level policy gradient loss**（把 batch 内所有 token 铺平后统一求平均），就是为了消除这个序列级平均带来的偏置。

**符号对照**（与第 06 章 PPO 对齐）：本章的 ρ_i（逐 token 写就是 ρ_{i,t}）就是第 06 章的 **r(θ)**，两者是同一个重要性采样比率，只是记号不同；本章的 **A_i** 对应第 06 章 PPO 里的优势函数 **A(s,a)**（也就是第 02 章 §2.7 Policy Gradient 中的优势项）——区别只在于 GRPO 用组内统计量替代了 Critic 给出的估值。

---

## 8.3 GRPO vs PPO 对比

| 维度 | GRPO | PPO |
|------|------|-----|
| 需要的模型 | 2个 (Policy + Reference) | 4个 (Policy + Ref + Reward + Value) |
| Advantage 计算 | 组内相对排序 | GAE（需要 Value Model） |
| 显存占用 | 低（**仅指模型权重**） | 高（约2倍，多出的主要是 Value Model） |
| 实现复杂度 | 简单 | 复杂 |
| 训练速度 | 快 | 慢 |
| Agentic RL 适用性 | 优秀 | 优秀 |

> **显存口径**：GRPO 相对 PPO 省下的是 **Critic/Value Model 的模型权重**（以及它那份梯度与 Adam 状态）。但训练显存的大头往往是 **policy 自身的梯度 + Adam 优化器状态 + rollout 引擎的 KV cache**，这部分 GRPO 一分没省。所以"省一半显存"只在**权重口径**下成立，不能当成整卡显存的实测结论——真要省显存请用 gradient checkpointing / offload（见 [24 - 学习总结与进阶路径](./24-学习总结与进阶路径.md)）。

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

Group advantage: [0.5, 0.5, -1.5, 0.5]   # 样本标准差（ddof=1）下的精确计算值
```

> 若改用**总体标准差**（ddof=0），同一组奖励会得到 [0.58, 0.58, -1.73, 0.58]；本章统一按 verl/TRL 的 ddof=1 口径。

### 8.4.2 Token-level GRPO

"Token-level" 在这里其实指**两件不同的事**，读的时候别混成一件事：

#### (1) token-level advantage —— 优势广播

在多轮交互中，每条轨迹先在组内拿到一个**标量**优势 A_i，再把它**广播**到该轨迹的每一个 token 上：

```
Turn 1 tokens: [agent thinks about search strategy]
Turn 2 tokens: [agent calls search tool]
Turn 3 tokens: [agent processes result and answers]

每个 token 的 advantage 来自其所在轨迹的组内相对排名
同一条轨迹内的所有 token 共享同一个标量 A_i
```

也就是说：**轨迹级比较 → 序列级标量 → 广播到 token**。GRPO 并**没有**给每个 token 单独打分，token 之间不存在"谁比谁好"。

#### (2) token-level loss —— 损失聚合

真正按 token 计算的是**损失的聚合方式**（见 8.2.3 的目标函数）：

```
序列级平均（GRPO 原式）： (1/G) Σ_i [ (1/|o_i|) Σ_t loss_{i,t} ]
token 级平均（DAPO）：     (1 / Σ_i |o_i|) Σ_i Σ_t loss_{i,t}
```

差别在于：序列级平均让**每条序列等权**（一条 10 token 的轨迹和一条 1000 token 的轨迹各占 1/G），token 级平均则让**每个 token 按其占比说话**。DAPO 指出前者会引入长度偏置，因此改用后者。

---

## 8.5 GRPO 的改进变体

### 8.5.1 DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）

ByteDance 在 DeepSeek-R1 基础上的改进（[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)），论文给出**四项技术**：

- **Clip-Higher**：把截断范围的上下界**解耦**（下界 ε_low=0.2，上界 ε_up=0.28），让低概率 token 也有更大的上升空间，鼓励探索
- **Dynamic Sampling（动态采样）**：**过滤掉组内全对或全错的 prompt**——这些 prompt 的优势全为 0、不产生梯度，还会稀释有效 batch
- **Token-Level Policy Gradient Loss**：用 token 级平均替代序列级平均（见 8.4.2）
- **Overlong Reward Shaping（超长响应奖励整形）**：对超出长度上限的响应按超长幅度施加惩罚，防止模型靠"写长一点蒙对"取巧

> **关于 KL**：DAPO 的做法是**移除 KL 散度项**（Removing KL Divergence），而不是放大 KL 截断值——它把 KL 惩罚系数 β 直接置 0，靠 clip 本身来限制策略偏移。

### 8.5.2 Tree-GRPO（ICLR 2026）

阿里巴巴高德团队提出的改进：
- 使用 **Tree Search（树搜索）** 替代独立的链式 Rollout
- 多个轨迹共享前缀，减少重复计算
- 更高效的探索策略

### 8.5.3 GiGPO（Group-in-Group Policy Optimization）

专为 LLM Agent 设计的改进（[arXiv:2505.10978](https://arxiv.org/abs/2505.10978)），两层分组：

- **episode-level 组**：同一个任务下的一批完整轨迹，按最终结果做组内比较（这一层就是 GRPO）
- **step-level anchor state grouping（锚点状态分组）**：把**跨轨迹中"遇到相同环境状态"的动作事后归到一组**，用锚点状态的回报做组内比较
- 核心创新：**不需要额外 rollout**，就把 GRPO 只有 episode 级的信用分配扩展到了 **step 级**

### 8.5.4 Training-Free GRPO（ICLR 2026 投稿）

[arXiv:2510.08191](https://arxiv.org/abs/2510.08191)，OpenReview 上的 venue 是 **"Submitted to ICLR 2026"**——是**投稿状态，不是录用**。

- 不更新任何模型参数，靠推理阶段的采样与信号注入逼近 GRPO 的效果
- 机制是**用组内的语义优势替代数值优势**：不比奖励分数，而是比同一 prompt 下各候选回复的语义质量
- 通过**多轮蒸馏 token prior** 把组内语义优势回灌到采样分布上（**不是**推理时做组内投票/排序）

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
        std_r = rewards_tensor.std()  # ddof=1（样本标准差），与 verl / TRL 的实现一致
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
| `num_samples (G)` | 4~16 | 每个 prompt 的采样数量。**DeepSeekMath / DeepSeek-R1 原文用 G=64**，4~16 是后续工程实践的折中（见 8.2.1） |
| `eps (ε)` | 0.2 | PPO 截断范围 |
| `beta (β)` | 0.01~0.1 | KL 惩罚系数（DAPO 直接置 0） |
| `learning_rate` | 1e-6 ~ 5e-7 | 学习率（小模型可用大一点） |
| `temperature` | 0.8~1.0 | 生成温度（控制探索） |
| `max_new_tokens`（TRL / transformers）| 512~2048 | 最大生成长度 |
| `response_length`（**verl**）| 512~2048 | 同一个东西，verl 的键名不同 |

---

## 8.8 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Group Relative Policy Optimization (GRPO) | 组相对策略优化 |
| Group Sampling | 组采样 |
| Relative Advantage | 相对优势 |
| Importance Sampling Ratio (ρ_i / r(θ)) | 重要性采样比率（本章 ρ_i = 第 06 章的 r(θ)） |
| Advantage Function (A(s,a) / A_i) | 优势函数（GRPO 里记作 A_i，用组内相对排名估计） |
| Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) | 解耦 Clip 与动态采样策略优化 |
| Tree-GRPO | 树搜索 GRPO |
| Group-in-Group Policy Optimization (GiGPO) | 组中组策略优化（episode 级 + step 级锚点分组） |
| Emergent Ability | 涌现能力 |

---

## 8.9 小结

- GRPO 是当前 Agentic RL 的主流算法，用组内相对比较替代 Value Model
- 相比 PPO 省掉的是 Critic 的**模型权重**（训练显存的大头——policy 梯度与 Adam 状态——一分没省），实现更简单
- DAPO、Tree-GRPO、GiGPO 等变体针对不同场景做了优化
- 下一步：学习 RLVR → [09 - RLVR 可验证奖励的强化学习](./09-RLVR可验证奖励的强化学习.md)

---

## 练习题

1. **数值计算**：同一个 prompt 下采样 4 条回复，奖励分别是 `[1.0, 1.0, 0.0, 1.0]`。用**样本标准差（ddof=1）**计算这 4 条回复的归一化优势。如果误用了**总体标准差（ddof=0）**，结果会变成什么样？两者差别从哪来？

2. **概念辨析**：GRPO 里的 "token-level" 至少可以指两件事。请分别说明「token-level advantage」和「token-level loss」各指什么，并回答：DAPO 改的是哪一个？为什么不能改另一个？

3. **公式理解**：目标函数里的 `1/|o_i|` 起什么作用？它会带来什么副作用，DAPO 的 token-level loss 又是怎么消掉这个副作用的？

4. **超参判断**：论文原文用 G=64，工程上常用 4~16。假设 rollout 成本随 G 线性增长、而训练显存不变。当奖励是 0/1 二值且组内全对或全错的概率不低时，把 G 从 64 降到 8 会分别给「优势信号的方差」和「DAPO 的 Dynamic Sampling 有效性」带来什么影响？什么情况下应该把 G 调回去？

> **参考答案**：1. ddof=1 时均值 0.75、样本标准差 0.5，优势为 **[0.5, 0.5, -1.5, 0.5]**；ddof=0 时总体标准差约 0.433，优势为 **[0.58, 0.58, -1.73, 0.58]**。差别来自 ddof：分母是 G-1 还是 G，两者只差一个 `sqrt(G/(G-1))` 的缩放，不改变优势的正负号与组内排序。verl / TRL 用的是 ddof=1（见 8.2.2、8.6）。2. token-level advantage 指**优势广播**——每条轨迹先在组内拿到一个标量 A_i，再广播到该轨迹的每个 token，token 之间并不单独比较（见 8.4.2(1)）；token-level loss 指**损失聚合**——按 token 平均而不是按序列平均（见 8.4.2(2)）。DAPO 改的是**损失聚合**：优势本身来自组内相对排名，改它就等于改了信用分配的语义。3. `1/|o_i|` 是序列内部的长度归一化，把不同长度轨迹的 token 梯度拉到同一量级；副作用是"先序列内平均、再序列间平均"让**每条序列等权**，短序列的 token 被放大、长序列被摊薄。DAPO 把 batch 内所有 token 铺平后统一平均（`(1/Σ|o_i|) ΣΣ loss`），于是每个 token 按其占比说话，长度偏置消失（见 8.2.3、8.4.2）。4. G 变小 → 组内标准差估计更不稳，**优势信号的方差变大**；同时"组内全对/全错"的概率上升，这些 prompt 的优势全为 0，**Dynamic Sampling 要过滤掉的废样本占比变大**，有效 batch 被进一步稀释。因此当任务本身就接近全对或全错时，要么调大 G（成本线性上升），要么先提高任务难度/奖励分辨率，再谈开 DAPO 的动态采样。
