# 10 - Reward Shaping 奖励设计进阶

> **学习目标**：掌握 Reward Shaping（奖励塑形）的高级技巧，解决 Agentic RL 中稀疏奖励和信用分配的核心难题。

---

## 目录

- [10.1 为什么需要 Reward Shaping？](#101-为什么需要-reward-shaping)
- [10.2 奖励函数的分层设计](#102-奖励函数的分层设计)
- [10.3 信用分配（Credit Assignment）](#103-信用分配credit-assignment)
- [10.4 TIPS: Turn-level Information-Potential Reward Shaping](#104-tips-turn-level-information-potential-reward-shaping)
- [10.5 Process Reward Model（PRM，过程奖励模型）](#105-process-reward-modelprm过程奖励模型)
- [10.6 奖励归一化（Reward Normalization）](#106-奖励归一化reward-normalization)
- [10.7 奖励设计最佳实践](#107-奖励设计最佳实践)
- [10.8 术语速查](#108-术语速查)
- [10.9 小结](#109-小结)
- [练习题](#练习题)

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
  └→ 例：成功的工具调用 +0.05，获得了有用的搜索结果 +0.1

Layer 1: Behavior Reward（行为奖励）
  └→ 鼓励良好的行为模式
  └→ 例：正确的工具调用格式 +0.05，有效的思考步骤 +0.05
```

### 10.2.2 实际奖励函数示例

```python
def layered_reward(trajectory):
    """三层奖励函数

    trajectory 采用 dict 风格（本教程统一约定）：
      {"steps": [...], "actions": [...], "final_state": {...}}
    每个 step 也是 dict：{"type": ..., "successful": ..., ...}
    """
    reward = 0.0
    
    # Layer 3: 任务结果
    if task_completed(trajectory):
        reward += 1.0
    elif partial_progress(trajectory):
        reward += 0.3  # 部分进展
    
    # Layer 2: 进度奖励
    for step in trajectory.get("steps", []):
        if step.get("type") == "tool_call" and step.get("successful"):
            reward += 0.05  # 成功的工具调用
        if step.get("type") == "search" and step.get("has_relevant_results"):
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

> **轨迹对象约定**：本教程统一用 **dict 风格**描述轨迹——
> `trajectory.get("steps")` / `trajectory.get("actions")` / `trajectory.get("final_state")`，
> 每个 step 同样是 dict（如 `step.get("type")`、`step.get("successful")`）。
> 这样轨迹可以直接序列化、跨进程传递，也方便用 `**kwargs` 交给奖励函数（见 09.6.2）。

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
    steps = trajectory.get("steps", [])
    for i, step in enumerate(steps):
        # 距离最终步骤越远，折扣越大
        distance = len(steps) - 1 - i
        credit = final_reward * (gamma ** distance)
        credits.append(credit)
    return credits
```

---

## 10.4 TIPS: Turn-level Information-Potential Reward Shaping

**TIPS（Turn-level Information-Potential Reward Shaping）**（Xie et al., 2026，arXiv:2603.22293）
是 2026 年的重要工作：

```
核心思想：
为搜索增强的 LLM Agent 提供 Turn-level 的奖励信号。

具体做法（potential-based reward shaping）：
1. 用 teacher model 对**正确答案的似然增量**来刻画状态 s 的"信息势"：
       Φ(s) = teacher model 给出的"正确答案似然"相对于基线的增量
2. 每一轮搜索后，用势函数的差分作为这一轮的 turn reward：
       turn_reward = Φ(s') − Φ(s)
3. 最终答案的正确性 → outcome reward（照常叠加）

关键性质：policy-invariant
```

**为什么是 policy-invariant**：经典结论（Ng et al., 1999）指出，形如
`r'(s,a,s') = r(s,a,s') + γ·Φ(s') − Φ(s)` 的奖励整形**不改变最优策略**。
把整形项沿一条轨迹求和，势函数会**望远镜式相消**（γ=1 时精确等于 `Φ(s_T) − Φ(s_0)`），
整条轨迹的回报只多了一个**与策略无关的常数偏置**——
不同策略之间的相对优劣没变，最优策略自然也没变，变的只是回报的尺度。

> ⚠️ 这跟"每轮搜索后评估搜索结果相关性 → 相关性高给高 turn reward"是**两回事**：
> 后者是对**特定动作**加分，会改变最优策略（也因此更容易被 hack）；
> 前者只加与动作无关的势函数差分，理论上不引入新的最优解。
> 注意：TIPS 里的 potential 不是"这一轮搜索返回内容与任务的相关程度"，而是**teacher model 对正确答案的似然增量**。

**效果**：在 QA 任务上优于纯 Outcome Reward 的 RLVR 基线。

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

### 10.5.1 一个常见误区：PRM 能否换成外部判断服务？

既然 PRM 本质是"给每一步打个分"，而闭集判断模型（如 System One 系列）正好擅长这类单步判断——那能不能**不训 PRM，直接调外部服务**？

**答案：训练期不行，推理期可以。**

原因是延迟量级不匹配。PRM 在训练循环里被调用的次数是：

```text
调用次数 ≈ batch_size × group_size × rollout_steps × 训练步数
```

以 `batch_size=64`、`group_size=8`、`rollout_steps=8`、`训练步数=1000` 估算，约 **409 万次**。若每次调用耗时 200 ms，训练时间将增加数百小时——**完全不可行**。

**结论：训练期的过程奖励必须本地可计算**，也就是必须训一个 PRM（或写规则）。外部服务进不了训练循环。

但反过来，**模型训练完成、上线推理之后**，如果还需要过程质量判断，此时延迟预算宽松得多（单次百毫秒可接受），就可以把这一步外置。

> 完整的延迟核算与实践建议见 [14.3 训练期硬约束](./14-奖励函数设计与实现.md)；判别与策略的边界划分见 [25 - 判别能力外置](./25-判别能力外置-什么时候不该用RL.md)。

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
    rewards = np.asarray(rewards, dtype=float)

    if method == "group":
        mean = np.mean(rewards)
        # 组内标准差用 ddof=1（样本标准差），对齐 verl core_algos.py 与 TRL 的 torch.std()
        # ⚠️ G < 2 时 ddof=1 会返回 NaN（Degrees of Freedom <= 0），
        #    所以单样本组必须退化为 ddof=0，否则整组优势全变成 NaN。
        ddof = 1 if len(rewards) > 1 else 0
        std = np.std(rewards, ddof=ddof) + 1e-8
        return (rewards - mean) / std
    
    elif method == "running":
        running_mean = exponential_moving_average(rewards)
        running_std = exponential_moving_std(rewards)
        return (rewards - running_mean) / (running_std + 1e-8)
    
    elif method == "clip":
        return np.clip(rewards, -1.0, 1.0)
```

> ⚠️ **ddof 口径必须写清楚**：
> - `np.std()` 默认 **ddof=0**（总体标准差），`torch.std()` 默认 **ddof=1**（样本标准差）。
> - verl 的 GRPO 优势归一化（`verl/trainer/ppo/core_algos.py:321,525`）与 TRL
>   用的都是 **ddof=1**，本章代码与之对齐。
> - **但 ddof=1 有边界**：G < 2 时 `np.std(..., ddof=1)` 返回 **NaN**
>   （`Degrees of Freedom <= 0`），整组优势会全部变成 NaN、训练直接崩。
>   真实实现必须对 **G=1 的组退化为 ddof=0**（上面代码里的 `ddof` 三元式就是干这个的），
>   或者干脆丢弃 G=1 的样本。
>
> 顺带一提：GRPO 用的是**组内相对优势**（减去组内均值再除以组内标准差），
> 不是"减去全局 baseline"——这正是它不需要 Critic 的原因，见 [08 - GRPO 深度解析](./08-GRPO深度解析.md)。

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
- TIPS 用 **potential-based shaping**（Φ(s′) − Φ(s)）提供 Turn-level 奖励，policy-invariant
- PRM 提供了另一条 Turn/Step-level 奖励的思路，但训练期必须本地可计算
- 奖励归一化要注意 **ddof 口径**（verl/TRL 用 ddof=1，G<2 会 NaN）
- 下一步：进入训练工程阶段 → [11 - 训练框架选型与搭建](./11-训练框架选型与搭建.md)

---

## 练习题

1. **手算归一化**：一组 4 条轨迹的奖励是 `[1.0, 0.0, 1.0, 0.0]`。
   分别用 ddof=1 和 ddof=0 算归一化结果，并说明 verl / TRL 选的是哪个、
   为什么这个选择会改变优势的数值尺度。

2. **ddof 的坑**：G=1 时 `np.std(rewards, ddof=1)` 会返回什么？为什么？
   真实实现应该怎么处理？请写出那行代码。

3. **三层奖励设计**：给一个"在电商网站上下单"的 Agent 设计 Layer 1 / 2 / 3 奖励。
   哪一层最容易被 hack？为什么？给出对应的防御。

4. **policy-invariant 的边界**：TIPS 的 turn reward = Φ(s′) − Φ(s)。
   为什么这种 shaping 不会改变最优策略？如果把 Φ 换成"这一轮搜索返回内容与任务的相关程度"，
   结论还成立吗？

> **参考答案**：1. mean = 0.5。ddof=1：样本标准差 = `√(1.0/3)` = **0.5774**，
>    归一化结果 = **[+0.866, −0.866, +0.866, −0.866]**；
>    ddof=0：总体标准差 = `√(1.0/4)` = **0.5**，归一化结果 = **[+1.0, −1.0, +1.0, −1.0]**。
>    verl（`core_algos.py:321,525`）与 TRL 用的是 **ddof=1**（`torch.std()` 默认），
>    本章代码与之对齐。ddof 变小 → 优势被放大 → 有效学习率变高（见 10.6）。
> 2. 返回 **NaN**，并伴随 `RuntimeWarning: Degrees of Freedom <= 0`——样本标准差要除以 `G−1`，
>    G=1 时分母为 0。真实实现要么对 G=1 退化为 ddof=0，要么丢弃这类样本：
>    `ddof = 1 if len(rewards) > 1 else 0`，然后 `std = np.std(rewards, ddof=ddof)`（见 10.6 的代码）。
> 3. 开放题，参考 10.2 的模板：Layer 3 任务结果（下单成功 +1.0 / 失败 0）；
>    Layer 2 进度（选对了商品 +0.1、成功调用库存接口 +0.05、填对了地址 +0.05）；
>    Layer 1 行为（工具调用格式正确 +0.05、思考过程存在 +0.05）。
>    **最容易 hack 的是 Layer 1**——它与任务成败无关，模型只要学会"摆出正确格式"
>    就能白拿分，即使最后根本没能下单。防御：把行为奖励的额度压小、设上限、
>    或者只在 Layer 3 成功的轨迹上才发放（"成功前提下的加分"）。
> 4. 经典结论（Ng et al., 1999）：`r'(s,a,s') = r(s,a,s') + γΦ(s') − Φ(s)` 是 **policy-invariant** 的，
>    因为沿轨迹求和时势函数项**望远镜式相消**（γ=1 时 = `Φ(s_T) − Φ(s_0)`），
>    整条轨迹只多了一个与策略无关的常数偏置，最优策略不变。
>    换成"搜索结果相关性"就不成立了：相关性依赖于**具体执行了哪次搜索**，
>    属于"对特定动作加分"，会改变不同动作之间的相对回报，也就可能改变最优策略，
>    同时给了模型一个可以操纵的目标（多搜、搜关键词作弊）——这是更容易被 hack 的 shaping。
