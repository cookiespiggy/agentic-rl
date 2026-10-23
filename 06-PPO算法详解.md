# 06 - PPO 算法详解

> **学习目标**：深入理解 Proximal Policy Optimization（PPO，近端策略优化），这是 Agentic RL 的基石算法之一。

---

## 目录

- [6.1 PPO 是什么？](#61-ppo-是什么)
- [6.2 PPO 的数学推导](#62-ppo-的数学推导)
- [6.3 PPO 在 LLM 中的应用](#63-ppo-在-llm-中的应用)
- [6.4 PPO 在 Agentic RL 中的特殊考虑](#64-ppo-在-agentic-rl-中的特殊考虑)
- [6.5 PPO 的优缺点](#65-ppo-的优缺点)
- [6.6 代码骨架（伪代码）](#66-代码骨架伪代码)
- [6.7 术语速查](#67-术语速查)
- [6.8 小结](#68-小结)
- [练习题](#练习题)

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

其中 ε 是超参数（通常 0.1-0.2），控制概率比允许偏离 1 的范围
（也就是单次更新最多能把策略推动多远）

clip(r(θ), 1-ε, 1+ε) 的含义：
- 如果 r(θ) > 1+ε，则截断为 1+ε（比率的上界）
- 如果 r(θ) < 1-ε，则截断为 1-ε（比率的下界）
```

**直觉理解**（关键：被截断的是**比率 r(θ)**，不是概率本身）：

约束对象是重要性采样比率 `r(θ) = π_θ(a|s) / π_old(a|s)`——"新策略把这个动作的概率放大了几倍"，
而不是"概率变了多少"。ε=0.2 允许新策略把某个动作的概率最多放大到 1.2 倍、最小压到 0.8 倍；
概率从 0.01 涨到 0.012（比值 1.2）完全合法，从 0.01 涨到 0.05（比值 5）才会被截断。

**两侧并不对称**，这正是 clip 真正在做的事：

- **A > 0（好动作）**：一旦 r > 1+ε，`min(r·A, clip(r)·A)` 取到常数 `(1+ε)·A`，**梯度变成 0**——
  策略已经把这个动作的概率抬得够多，不再继续往上推。
- **A < 0（坏动作）**：一旦 r < 1-ε，`min(r·A, clip(r)·A)` 取到常数 `(1-ε)·A`，**梯度同样变成 0**——
  策略已经把这个动作的概率压得够多，不再继续往下压。
- **反方向不受限制**：好动作的概率被压低（r < 1-ε，A > 0）或坏动作的概率被抬高（r > 1+ε，A < 0）时，
  取到的是**未截断**的那一项，梯度仍然非零。clip 只拦"顺着 advantage 方向继续加码"，不拦"反向纠偏"。

**这就是隐式的 KL 早停（KL early-stop）机制**：当策略朝被鼓励的方向跑出信任区域后，截断目标变成常数、
不再提供梯度，等价于在偏离旧策略过多之前**主动停止这一维的更新**。这与 6.3.4 的显式 KL 惩罚是两道互补的保险。

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

### 6.3.3 GAE（广义优势估计）

**TD 残差（one-step TD error）**：

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

δ_t 是"这一步实际拿到的奖励 + 对未来的新估计"与"原本的估计"之间的差：
当 V(s_t) 相对真实价值偏大（过于乐观）时 δ_t 偏负，偏小（过于悲观）时 δ_t 偏正。

**GAE（Generalized Advantage Estimation，广义优势估计）**：

```
A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

λ 是 GAE 的衰减系数，**控制偏差-方差权衡**：

| λ | 含义 | 偏差 | 方差 |
|---|------|------|------|
| λ = 0 | A_t = δ_t，退化成 TD(0) | 最大（只信一步预测） | 最小（最平滑） |
| λ = 1 | A_t ≈ G_t − V(s_t)，退化成蒙特卡洛 | 最小（无偏） | 最大（噪声全进梯度） |
| 0 < λ < 1 | 几何加权混合，越远的 δ 权重按 (γλ)^l 衰减 | 可调 | 可调 |

**直觉**：λ 决定"允许多信任未来的估计"。λ 越接近 1，优势越接近真实回报，但梯度噪声越大、训练越不稳；
λ 越接近 0，优势越平滑稳定，但偏差越大。LLM RL 里常见取 γ = 1、λ ≈ 0.95。

**价值函数损失（Critic 的训练目标）**：

```
L_V = E[ (V(s_t) − G_t)^2 ]
```

其中 G_t 是回报目标（Return）。理论上可以取完整的蒙特卡洛回报 `G_t = Σ_k γ^k · r_{t+k}`，
但实践中更常用 **bootstrap 后的 λ-回报**（也就是上面算出来的 `A_t + V(s_t)`）当回归目标——
它的偏差更小、方差更可控。Critic 本质上就是一个回归模型：
把回报目标当标签去拟合 V(s_t)。

**LLM 场景的特殊之处：序列级奖励 vs token 级 value**：

在 LLM RL 里，奖励通常只有**一个标量**（整条回复正确 +1），
而 Value Model 输出的是**每个 token 一个**的价值，两者粒度对不上。标准做法是：

```
reward（序列级奖励，只挂在最后一个 token 上）：
  [0, 0, 0, ..., 0, R]

value（token 级估计）：
  [V_1, V_2, ..., V_T]
```

即 **r_t = 0 (t < T)，r_T = R，终止状态 V(s_{T+1}) = 0**，
然后在时间轴上**从后往前**做 bootstrap：

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

这样"奖励是真的、value 是估的"这个区分仍然保留（δ_T = R − V(s_T) 直接用到真实奖励），
而靠价值估计把最终结果**向前传递**给前面的 token，使每个 token 都拿到非平凡的优势。
另一种更简单的做法是把同一个标量奖励广播到所有 token（Response-level broadcast），
实现更省事但方差更大。

（如果连 Critic 都不训，用组内相对优势替代 GAE，就是 [08 - GRPO 深度解析](./08-GRPO深度解析.md) 的做法。）

### 6.3.4 KL 散度惩罚

```
L_total(θ) = -L_CLIP(θ) + β · KL(π_θ || π_ref)
```

这里统一成 **loss 形式**（与 [08 - GRPO 深度解析](./08-GRPO深度解析.md) 一致）：
6.2.3 的 `L_CLIP` 是一个**要最大化**的量，取负号后才是待最小化的 loss；
`KL(π_θ || π_ref)` 是非负的散度，乘 β 后作为惩罚项。

KL 散度防止 Policy 偏离 Reference Model 太远
β 控制约束强度（通常 0.01~0.1）

> ⚠️ **两种 KL 的落点不同，不能互相替代**：
> - **Loss 层（token 级 k3 估计器）**：KL 直接加进 loss，在每个 token 位置逐点计算、逐点回传梯度；
> - **Reward 层（序列级）**：把 KL 折进奖励（`Reward_adjusted = R − β·KL`），奖励是序列级标量，
>   KL 只对整条序列给出一个数。
>
> 二者**并不等价**：token 级 KL 能对每个位置给出不同大小的约束（哪几个 token 偏离得最狠就压哪几个），
> 序列级 KL 只能对整条轨迹给一个统一折扣，两者的梯度路径也不一样。
> 选哪种取决于你的 KL 预算和对单 token 离群点的容忍度——**不要假设两者"本质等价"**。

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
            # 序列级奖励挂在最后一个 token 上，其余 token 的 reward 为 0（见 6.3.3）
            advantages = compute_gae(batch.rewards, value_model(batch.sequences))
            
            # 4. PPO-Clip 目标（loss 形式：-L_CLIP）
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.eps, 1 + config.eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 5. KL 散度惩罚（token 级 k3 估计器，见 6.3.4）
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
| TD Residual (δ_t) | TD 残差（一步时序差分误差） |
| GAE (Generalized Advantage Estimation) | 广义优势估计 |
| Value Function Loss | 价值函数损失（Critic 的回归目标） |
| KL Divergence | KL散度（衡量分布差异） |
| Clipping | 截断/裁剪 |
| Trust Region | 信任区域 |
| On-Policy | 同策略 |

---

## 6.8 小结

- PPO 通过限制每次更新的幅度，实现稳定的策略优化
- 截断约束的是**概率比 r(θ)**，两侧不对称：顺 advantage 方向跑出信任区域后梯度归零（隐式 KL 早停）
- GAE 用 λ 在偏差与方差之间取舍，价值函数用 L_V = E[(V(s_t) − G_t)²] 训练
- 在 LLM RL 中需要加载 4 个模型，显存和计算开销大
- 在 Agentic RL 中需要处理多轮交互的 Token-level Advantage
- 下一步：学习 DPO 和 GRPO → [07 - DPO 直接偏好优化](./07-DPO直接偏好优化.md)

---

## 练习题

1. **手算 GAE**：一条 3 步轨迹，token 级奖励 `r = [0, 0, 1]`，价值估计 `V = [0.4, 0.6, 0.8]`，
   取 γ = 1、λ = 0.5，终止状态 `V(s_4) = 0`。请算出三个 TD 残差 δ_t 和三个优势 A_t。

2. **λ 的作用**：为什么说 λ = 1 时"无偏但方差大"、λ = 0 时"方差小但偏差大"？
   如果训练时优势估计出现剧烈震荡，你会怎么调 λ？

3. **clip 机制**：某个动作的 A < 0、当前 r(θ) = 0.5、ε = 0.2。
   `min(r·A, clip(r)·A)` 取到的是哪一支？此时梯度是否为 0？这说明 clip 到底拦住了什么、
   又没拦住什么？

4. **序列级奖励 vs token 级 value**：在 LLM RL 里，奖励只有一个标量而 Value Model 是逐 token 输出。
   请说明标准做法是什么，以及"把同一个标量奖励广播到所有 token"这种简化做法会付出什么代价。

> **参考答案**：1. δ_1 = 0 + 1×0.6 − 0.4 = 0.2，δ_2 = 0 + 1×0.8 − 0.6 = 0.2，δ_3 = 1 + 1×0 − 0.8 = 0.2；
>    A_3 = δ_3 = 0.2，A_2 = δ_2 + (γλ)δ_3 = 0.2 + 0.5×0.2 = 0.3，
>    A_1 = δ_1 + 0.5·δ_2 + 0.25·δ_3 = 0.2 + 0.1 + 0.05 = 0.35（公式见 6.3.3）。
> 2. λ = 1 时 A_t = Σ γ^l δ_{t+l} ≈ G_t − V(s_t)，直接用真实回报做基准，因此无偏，但蒙特卡洛回报的噪声全部进入梯度；
>    λ = 0 时 A_t = δ_t，只依赖一步 TD 预测，噪声小但完全继承 V 的估计偏差。震荡时应**调小 λ**（更平滑）或调小 γ，
>    而不是动 β（见 6.3.3、6.3.4）。
> 3. A < 0 且 r = 0.5 < 1-ε = 0.8。因为 A 为负，`0.8·A < 0.5·A`（乘同一个负数，大数反而更小），
>    所以 `min(r·A, clip(r)·A)` 取到的是**常数** `0.8·A = (1-ε)·A`，**梯度为 0**——
>    策略不再继续压低这个动作的概率。
>    clip 拦住的是"顺着 advantage 方向继续加码"（A>0 时 r>1+ε 停止上推、A<0 时 r<1-ε 停止下压），
>    拦不住反方向（A>0 时概率被压低、A<0 时概率被抬高，梯度仍非零），这就是隐式 KL 早停机制（见 6.2.3）。
> 4. 标准做法是**把序列级奖励放在最后一个 token 上**（r_t = 0 (t<T)、r_T = R、V(s_{T+1}) = 0），
>    再按 δ_t = r_t + γV(s_{t+1}) − V(s_t)、A_t = Σ(γλ)^l δ_{t+l} 从后往前 bootstrap，
>    让"奖励是真的、value 是估的"的区分保留下来（见 6.3.3）。
>    广播做法把同一个标量塞给每个 token，代价是所有 token 的优势高度相关、梯度方差变大，
>    而且丢掉了"哪一步更该被鼓励"的时序信息。
