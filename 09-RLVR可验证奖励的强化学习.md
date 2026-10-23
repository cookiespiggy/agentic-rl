# 09 - RLVR 可验证奖励的强化学习

> **学习目标**：理解 RLVR（Reinforcement Learning with Verifiable Rewards）范式，这是 Agentic RL 中奖励设计的核心理念。

---

## 目录

- [9.1 什么是 RLVR？](#91-什么是-rlvr)
- [9.2 RLVR 的核心优势](#92-rlvr-的核心优势)
- [9.3 RLVR 的奖励类型](#93-rlvr-的奖励类型)
- [9.4 RLVR + GRPO 的标准流水线](#94-rlvr--grpo-的标准流水线)
- [9.5 RLVR 在 Agentic RL 中的特殊形式](#95-rlvr-在-agentic-rl-中的特殊形式)
- [9.6 Reward Hacking（奖励黑客）及其防御](#96-reward-hacking奖励黑客及其防御)
- [9.7 实际案例：用 RLVR 训练数学推理 Agent](#97-实际案例用-rlvr-训练数学推理-agent)
- [9.8 术语速查](#98-术语速查)
- [9.9 小结](#99-小结)
- [练习题](#练习题)

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
| 奖励黑客风险 | 中（Reward Model 可被欺骗） | 仍然存在（客观验证同样能被绕过，见 9.6） |

> ⚠️ **别把"可验证"当成"不可 hack"**：验证程序本身也是攻击面。模型可以**改写测试文件**、
> **对验证器写特判**、**在输出里塞干扰文本骗过答案抽取器**——验证逻辑只要落到了
> 模型可写、可读、可影响的文件系统或文本流上，就存在被利用的路径。
> 完整清单与防御见 [9.6 节](#96-reward-hacking奖励黑客及其防御)。

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
    """Agent 任务的奖励函数

    trajectory 采用 dict 风格（本教程统一约定）：
      {"steps": [...], "actions": [...], "final_state": {...}}
    """
    final_state = trajectory.get("final_state")
    
    if success_condition(final_state):
        return 1.0  # 任务成功
    else:
        return 0.0  # 任务失败
```

> **轨迹对象约定**：本教程统一用 **dict 风格**描述轨迹——
> `trajectory.get("steps")` / `trajectory.get("actions")` / `trajectory.get("final_state")`，
> 每个 step 同样是 dict（如 `step.get("type")`）。这样轨迹可以直接序列化、跨进程传递，
> 也方便用 `**kwargs` 交给奖励函数（见 9.6.2）。

### 9.3.4 格式验证（Format Verification）

```python
# 验证模型是否按正确格式输出
def format_reward(response):
    """格式正确性奖励"""
    reward = 0.0
    if has_think_tags(response):      # 有 <think> 标签
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

> **TIPS（Turn-level Information-Potential Reward Shaping）**（Xie et al., 2026，arXiv:2603.22293）
> 用 **potential-based reward shaping（基于势函数的奖励塑形）** 给搜索增强 Agent 提供 Turn-level 奖励：
> 取 teacher model 对**正确答案的似然增量**作势函数 Φ，turn reward = Φ(s′) − Φ(s)。
> 这类 shaping 是 **policy-invariant** 的——把势函数差分沿一条轨迹求和会望远镜式相消，
> 整条轨迹的回报只多了一个**与策略无关的常数偏置**，最优策略因此不变（详见 [10.4 节](./10-Reward-Shaping奖励设计进阶.md)）。
> 实验上它优于纯 Outcome Reward 的 RLVR 基线。

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

5. 测试篡改（RLVR 特有）：代码 Agent 直接改写仓库里的测试文件，让断言"通过"
   → 防御：测试文件只读挂载、每个 episode 后重置工作区、训练前校验测试文件哈希

6. 验证器特判（RLVR 特有）：针对验证程序写特判（硬编码答案、识别评测器特征）
   → 防御：验证器与训练数据同分布、定期轮换验证器、答案规范化后再比对

7. 抽取器欺骗（RLVR 特有）：输出里塞干扰文本，让答案抽取器抽到错误的那一段
   → 防御：抽取规则收紧、抽取失败即判错、抽取器版本化并保留审计日志
```

> 前 4 条是**通用**的 RL 奖励黑客，后 3 条是 RLVR 特有的——因为"客观验证"本身
> 也要落在一个模型可以影响的文件系统或文本流上。9.2 节表格里"奖励黑客风险：仍然存在"
> 指的就是后 3 条。

### 9.6.2 防御策略

```python
def _coerce_to_text(completion) -> str:
    """把 TRL 的 completion 格式转成字符串。

    对话模式下 completions 是 list[list[dict]]（每个元素是消息列表），
    需要取 assistant 消息的 content。
    """
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
    return str(completion)


def robust_reward(prompts, completions, max_steps=10, **kwargs):
    """鲁棒的奖励函数

    TRL 奖励函数契约：reward_func(prompts=..., completions=..., completion_ids=..., **reward_kwargs)
    其中 reward_kwargs 是**数据集列名**——标准答案通常叫 ground_truth / answer，
    从 kwargs 里取；max_steps 是步数预算，缺省 10。
    """
    ground_truths = kwargs.get("ground_truth", kwargs.get("answer", []))
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths] * len(completions)

    # max_steps 至少取 1，保证下面的除法永远不除零
    step_budget = max(1, int(max_steps))

    rewards = []
    for i, completion in enumerate(completions):
        response = _coerce_to_text(completion)
        ground_truth = ground_truths[i] if i < len(ground_truths) else None

        reward = 0.0

        # 1. 核心正确性
        if ground_truth is not None and verify_correctness(response, ground_truth):
            reward += 1.0
        else:
            reward = -0.5  # 错误惩罚

        # 2. 效率奖励（用更少步骤完成）
        num_steps = len(kwargs.get("steps", []) or [])
        efficiency = max(0.0, 1.0 - num_steps / step_budget)
        reward += 0.1 * efficiency

        # 3. 格式奖励
        if has_correct_format(response):
            reward += 0.05

        # 4. 惩罚重复
        if has_repetition(response):
            reward -= 0.2

        rewards.append(reward)

    return rewards
```

> ⚠️ **两个容易踩的坑**：
> 1. `max_steps` 一定要有默认值并做 `max(1, ...)` 兜底——否则一旦上游没传，
>    `num_steps / max_steps` 会抛 `ZeroDivisionError`（或除以 `None`）。
> 2. 奖励函数**不能**用 `ground_truths` 这类必需位置参数：TRL 只传
>    `prompts` / `completions` / `completion_ids` 加上数据集列名，
>    标准答案要从 `**kwargs` 里按列名取。

---

## 9.7 实际案例：用 RLVR 训练数学推理 Agent

```python
# 简化的 RLVR 奖励函数
# TRL 契约：reward_func(prompts=..., completions=..., completion_ids=..., **reward_kwargs)
# 标准答案是数据集的一列，从 kwargs 里按列名取（见 9.6.2）
def math_reward(prompts, completions, **kwargs):
    """数学题 RLVR 奖励函数"""
    ground_truths = kwargs.get("ground_truth", kwargs.get("answer", []))

    rewards = []
    for completion, gt in zip(completions, ground_truths):
        text = _coerce_to_text(completion)   # 对话模式下 completions 是 list[list[dict]]
        answer = extract_final_answer(text)

        if answer is None:
            rewards.append(-0.5)  # 格式错误（未输出答案标签），给负分
        elif verify_math_answer(answer, gt):
            rewards.append(1.0)   # 答案正确
        else:
            rewards.append(0.0)   # 答案错误

    return rewards
```

用 verl 训练（GRPO 在 verl 里就是 PPO trainer 换一个 advantage 估计器）：

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=8 \
    data.prompt_key=prompt \
    data.train_batch_size=128 \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.default_local_dir=checkpoints/rlvr-math
```

> ⚠️ **verl 里没有 `GRPOTrainer` 这个类**——`verl/__init__.py` 的 `__all__` 只有 `DataProto` 和 `__version__`。
> 入口是 `python -m verl.trainer.main_ppo`（`verl/trainer/` 下**没有** `main.py`），
> GRPO 通过 `algorithm.adv_estimator=grpo` 打开，KL 系数是 `algorithm.kl_ctrl.kl_coef`。
>
> `train_batch_size × rollout.n` 须能被 `n_gpus` 整除（`verl/utils/config.py`），
> 上面 128 × 8 = 1024，在 8 卡 / 16 卡上都成立。
>
> 把 `math_reward` 注册成 verl 的自定义 reward function 时，**配置键各版本有差异**，
> 以你所用版本的 config 为准；奖励函数的**签名约定**（`prompts` / `completions` / `**kwargs`）
> 才是跨框架通用的那一层。
>
> ⚠️ verl 训练脚本需要 CUDA 环境，本仓库未实机验证。

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
- 可验证奖励**同样会被 hack**（改测试文件、写特判、骗过答案抽取器），需要防御
- 下一步：学习 Reward Shaping → [10 - Reward Shaping 奖励设计进阶](./10-Reward-Shaping奖励设计进阶.md)

---

## 练习题

1. **手算组内优势**：同一个 prompt 采样了 4 条轨迹，验证器给出的奖励是 `[1, 1, 0, 1]`。
   用**样本标准差（ddof=1）**归一化，算出 4 个 advantage；再用总体标准差（ddof=0）算一遍，
   比较两者的差异，并说明 verl / TRL 用的是哪一个。

2. **零梯度的组**：上面那组如果奖励变成 `[1, 1, 1, 1]`（全对）或 `[0, 0, 0, 0]`（全错），
   GRPO 的梯度会怎样？为什么？实际训练时怎么处理这种情况？

3. **可验证奖励也会被 hack**：请举出一个"答案检查程序完全正确、但模型仍然能骗到满分"的例子，
   并给出至少两条防御措施。

4. **TIPS 的机制**：TIPS 用 teacher model 对正确答案的**似然增量**作势函数 Φ，
   turn reward = Φ(s′) − Φ(s)。若 Φ(s) = 0.3、Φ(s′) = 0.8，这一轮的 turn reward 是多少？
   为什么说这种 shaping 是 policy-invariant 的？

> **参考答案**：1. ddof=1：mean = 0.75，样本标准差 = **0.5**，advantage = `[0.5, 0.5, −1.5, 0.5]`；
>    ddof=0：总体标准差 = **0.433**，advantage = `[0.58, 0.58, −1.73, 0.58]`。
>    verl（`verl/trainer/ppo/core_algos.py:321,525`）与 TRL 用的都是 **ddof=1**
>    （`torch.std()` 的默认就是样本标准差），见 10.6 节。
> 2. 组内奖励全相等时，mean = 标准差的中心，减去 mean 后全部为 0（标准差为 0），
>    **advantage 全为 0 → 该组对梯度的贡献为 0**，等于白跑一遍 rollout。
>    常见处理：动态采样过滤掉全对/全错的 prompt（DAPO 的 Dynamic Sampling），
>    或把这类 prompt 挪到下一批重采（见 9.4 的流水线）。
> 3. 例子：代码题的测试文件在同一个可写工作区里，模型直接**改写测试文件**让断言通过；
>    或对答案抽取器做特判（硬编码答案、在输出里塞干扰文本骗过抽取器）。
>    防御：测试文件只读挂载 / 每 episode 后重置工作区 / 校验测试文件哈希；
>    抽取失败即判错、抽取器版本化留审计日志；定期轮换验证器（见 9.6.1）。
> 4. turn reward = **Φ(s′) − Φ(s) = 0.8 − 0.3 = 0.5**。
>    policy-invariant 的原因：经典的 potential-based shaping（Ng et al., 1999）满足
>    `r'(s,a,s') = r(s,a,s') + γ·Φ(s') − Φ(s)`。
>    把 shaping 沿一条轨迹求和，Φ 项会**望远镜式相消**（γ=1 时精确等于 Φ(s_T) − Φ(s_0)），
>    只剩一个与动作序列无关的常数。也就是说整形只是给整条轨迹的回报加了一个
>    **与策略无关的常数偏置**，不改变不同策略之间的相对优劣，最优策略因此不变——变的只是回报尺度。
