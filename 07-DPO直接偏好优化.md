# 07 - DPO 直接偏好优化

> **学习目标**：理解 DPO（Direct Preference Optimization）如何简化 RLHF 流程，以及它在 Agentic RL 中的局限与扩展。

---

## 7.1 DPO 的动机

PPO 的问题：
1. 需要训练单独的 Reward Model（奖励模型）
2. 需要加载 4 个模型（Policy + Reference + Reward + Critic）
3. 训练不稳定，超参数多

**DPO 的核心思想**：跳过 Reward Model，直接用 Preference Pairs（偏好对）优化 Policy。

---

## 7.2 DPO 的数学推导

### 7.2.1 从 RLHF 到 DPO

RLHF 的目标：
```
max_π E[r(x,y)] - β·KL(π || π_ref)
```

DPO 证明了：最优策略可以用 Closed-Form（闭式解）表示为 Reward 的函数：
```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y) / β)
```

反过来，Reward 可以用 Policy 表示：
```
r(x,y) = β · log(π(y|x) / π_ref(y|x)) + β·log Z(x)
```
其中 Z(x) 是配分函数（Partition Function）。代入 Bradley-Terry 偏好模型时，
Z(x) 在 pairwise 比较中会消去，因此最终 DPO Loss 中不含 Z(x)。

### 7.2.2 DPO 损失函数

将上述代入 Bradley-Terry 偏好模型，得到 DPO Loss：

```
L_DPO = -E_{(x,y_w,y_l) ~ D}[log σ(β · (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))]

其中：
- y_w = winning response（人类偏好的回复，获胜方）
- y_l = losing response（被拒绝的回复，失败方）
- π = policy model（正在训练的策略模型）
- π_ref = reference model（参考模型，通常是SFT模型）
- β = temperature hyperparameter（温度超参数，通常 0.1）
```

**直觉理解**：
- DPO 鼓励 π 相对于 π_ref 提高 y_w 的概率，降低 y_l 的概率
- 不需要单独的 Reward Model！

---

## 7.3 DPO 的训练数据

DPO 需要 **Preference Pairs（偏好对）**：

```python
preference_data = [
    {
        "prompt": "解释量子力学",
        "chosen": "量子力学是物理学的一个分支，它描述了...",  # y_w（好的回复）
        "rejected": "量子力学很复杂，我不太确定..."           # y_l（差的回复）
    },
    # ... 更多样本
]
```

**数据来源**：
1. 人工标注（昂贵但质量高）
2. 模型自评（Self-Play，成本低但可能有偏差）
3. 规则过滤（如代码运行通过/失败）

---

## 7.4 DPO 的变体

### 7.4.1 IPO（Identity Preference Optimization）

解决 DPO 的过拟合问题。

### 7.4.2 KTO（Kahneman-Tversky Optimization）

不需要偏好对，只需要 "好" 和 "差" 的单独样本。

### 7.4.3 ORPO（Odds Ratio Preference Optimization）

将 SFT 和 Preference Optimization 合并到一个阶段。

### 7.4.4 SimPO（Simple Preference Optimization）

用序列平均 Log Probability 替代显式的 Reference Model。

---

## 7.5 DPO 在 Agentic RL 中的局限

| 局限 | 说明 |
|------|------|
| 只支持单轮 | DPO 假设单次 (prompt, response)，不处理多轮交互 |
| 需要偏好对 | 构造高质量的 Agent 偏好对困难 |
| 无探索 | DPO 是 Offline（离线）方法，不能在环境中探索 |
| 信用分配缺失 | 无法区分多轮交互中各步骤的贡献 |

### 扩展 DPO 到 Agentic 场景的尝试

- **AgenticDPO**（2025）：将 DPO 扩展到多轮场景，通过 Turn-level Preference（轮次级偏好）来训练
- **Trajectory-level DPO**：用完整轨迹的质量排序来替代单轮偏好

> 但这些扩展仍不如 GRPO/PPO 在 Agentic RL 中主流。

---

## 7.6 DPO 代码示例

```python
from trl import DPOTrainer, DPOConfig

# 准备数据
dataset = load_dataset("preference_pairs")

# 配置
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
)

# 初始化 Trainer
trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 训练
trainer.train()
```

---

## 7.7 DPO vs PPO 对比

| 维度 | DPO | PPO |
|------|-----|-----|
| 模型数量 | 2 (Policy + Reference) | 4 (Policy + Ref + Reward + Value) |
| 训练数据 | 偏好对 (Preference Pairs) | 在线 Rollout 数据 |
| 训练方式 | Offline（离线） | Online（在线） |
| 多轮支持 | 弱 | 强 |
| 稳定性 | 较高 | 中等（需调参） |
| Agentic 适用性 | 低 | 高 |

---

## 7.8 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Direct Preference Optimization (DPO) | 直接偏好优化 |
| Preference Pairs | 偏好对 |
| Bradley-Terry Model | 布拉德利-特里偏好模型 |
| Closed-Form Solution | 闭式解 |
| Offline RL | 离线强化学习 |
| Online RL | 在线强化学习 |
| Chosen / Rejected | 选中/拒绝（偏好对中的好/差回复） |

---

## 7.9 小结

- DPO 跳过 Reward Model，直接用偏好对训练 Policy，简洁高效
- 但 DPO 本质是 Offline 方法，在多轮 Agentic 场景中能力有限
- GRPO 结合了 DPO 的简洁和 PPO 的在线探索能力
- 下一步：深入学习 GRPO → [08 - GRPO 深度解析](./08-GRPO深度解析.md)
