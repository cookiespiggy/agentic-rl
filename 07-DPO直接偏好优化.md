# 07 - DPO 直接偏好优化

> **学习目标**：理解 DPO（Direct Preference Optimization）如何简化 RLHF 流程，以及它在 Agentic RL 中的局限与扩展。

---

## 目录

- [7.1 DPO 的动机](#71-dpo-的动机)
- [7.2 DPO 的数学推导](#72-dpo-的数学推导)
- [7.3 DPO 的训练数据](#73-dpo-的训练数据)
- [7.4 DPO 的变体](#74-dpo-的变体)
- [7.5 DPO 在 Agentic RL 中的局限](#75-dpo-在-agentic-rl-中的局限)
- [7.6 DPO 代码示例](#76-dpo-代码示例)
- [7.7 DPO vs PPO 对比](#77-dpo-vs-ppo-对比)
- [7.8 术语速查](#78-术语速查)
- [7.9 小结](#79-小结)
- [练习题](#练习题)

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
from datasets import Dataset, load_dataset   # load_dataset 用于换成你自己的偏好数据
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# 准备数据：HF 数据集需要含 prompt / chosen / rejected 三列
#
# 方式一：换成你自己的偏好数据（HF Hub 上的真实数据集名）
# dataset = load_dataset("your_org/your_preference_dataset", split="train")
#
# 方式二：先用下面这份最小样例把流程跑通（数据来自 7.3 节）
dataset = Dataset.from_list([
    {
        "prompt": "解释量子力学",
        "chosen": "量子力学是物理学的一个分支，它描述了...",
        "rejected": "量子力学很复杂，我不太确定...",
    },
])

# 加载模型与分词器
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
policy_model = AutoModelForCausalLM.from_pretrained(model_name)
reference_model = AutoModelForCausalLM.from_pretrained(model_name)  # 冻结的参考模型

# 配置
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
)

# 初始化 Trainer
# ⚠️ TRL 1.x 用 processing_class= 传分词器，签名里已经没有 tokenizer= 了
trainer = DPOTrainer(
    model=policy_model,
    ref_model=reference_model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# 训练
trainer.train()
```

> ⚠️ 版本提示：TRL 1.x 起 `DPOTrainer` 的分词器参数是 `processing_class=`（不是 `tokenizer=`），
> 传 `tokenizer=` 会直接报 `TypeError: unexpected keyword argument 'tokenizer'`。
> 上面的数据源是**占位**：`your_org/your_preference_dataset` 并不存在，
> 换成你自己的偏好数据（或先用上面的最小样例跑通）才能真正训起来。

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

---

## 练习题

1. **推导复盘**：DPO 的最优策略闭式解里有一个配分函数 Z(x)。为什么最后 DPO Loss 里 Z(x) 不见了？
   这件事对"用 DPO 训练"意味着什么实际好处？

2. **数值计算**：β = 0.1，某条样本上
   `log(π(y_w|x)/π_ref(y_w|x)) = −0.5`，`log(π(y_l|x)/π_ref(y_l|x)) = −0.8`。
   请算出 DPO Loss 的值，并说明它相对 0.693（= −log 0.5，即"完全学不动"）是什么关系。

3. **β 的作用**：把 β 从 0.1 调到 0.01，模型行为会有什么变化？为什么 β 既是"温度"又是"约束强度"？

4. **多轮场景**：DPO 在一条 5 轮的工具调用轨迹上，只有"整条轨迹成功/失败"这一个偏好标签。
   请说明为什么 DPO 无法从这个标签里区分"第 2 轮的工具调用选错了"和"第 4 轮参数写错了"，
   并给出至少两种可能的补救思路。

> **参考答案**：1. Z(x) 只依赖 prompt 而不依赖具体回复，在 Bradley-Terry 的 pairwise 比较
>    `π(y_w|x)/π(y_l|x)` 里分子分母同时含 `1/Z(x)`，会**精确消去**（见 7.2.1）。
>    实际好处是：训练时不需要估计配分函数，也不需要采样或拟合一个显式的 Reward Model——
>    这正是 DPO 比 RLHF 简单一大截的原因。
> 2. 括号内 = `−0.5 − (−0.8) = 0.3`，乘 β = 0.1 得 **0.03**；
>    Loss = `−log σ(0.03)` ≈ `−log 0.5075` ≈ **0.678**。
>    比 0.693 略小，说明"选中的更好"已经带来一点点正向梯度，但信号很弱——
>    这正是 DPO 需要把 β 设小、学习率设到 5e-7 量级的原因（见 7.6）。
> 3. β 越小，括号内的差被压缩得越厉害，loss 越贴近 0.693，梯度越小、模型更新越保守；
>    β 越大，偏好信号被放大得越厉害，但同时 `π(y|x)/π_ref(y|x)` 的隐式正则（等价于 KL 惩罚）也越松，
>    模型更容易偏离参考模型、也更容易被偏好数据里的噪声带偏。
>    所以 β 同时扮演"温度"（软化概率比）和"约束强度"（限制偏离 π_ref）两个角色。
> 4. 因为 DPO 的 loss 只看 `log π(y_w|x) − log π(y_l|x)` 这一个**序列级**标量差，
>    梯度均匀地摊到序列的所有 token 上，**没有任何机制把信用分配到具体的某一步**（见 7.5 的"信用分配缺失"）。
>    补救思路至少包括：① **AgenticDPO** 那样构造 Turn-level 偏好（每轮都标一条偏好对）；
>    ② 用 **Trajectory-level DPO** 配合多条轨迹做排序，让"哪条轨迹好"间接约束关键步骤；
>    ③ 改用 **GRPO/PPO** 这类在线方法，让 advantage 通过 TD/组内相对量落到具体 token 上（见 08、09 章）。
