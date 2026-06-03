# 15 - 第一个 Agentic RL 训练实战

> **学习目标**：从零开始完成一个完整的 Agentic RL 训练项目——用 GRPO 训练一个数学推理 Agent。

---

## 15.1 项目概述

**目标**：训练一个 3B 参数的模型，使用计算器工具解决数学问题。

```
Task Flow（任务流程）:
  User: "计算 234 * 567 + 89"
  Agent: 让我来计算... [思考过程]
  Agent: 调用 calculator("234 * 567 + 89")
  Tool: 返回 132767
  Agent: 最终答案是 132767
```

**技术栈**：
- 基座模型：Qwen2.5-3B
- 训练框架：verl
- 算法：GRPO（Group Relative Policy Optimization）
- 数据集：GSM8K（小学数学应用题）

---

## 15.2 Step 1: 环境准备

```bash
# 1. 安装依赖
pip install verl ray wandb datasets transformers

# 2. 准备 GSM8K 数据集（小学数学应用题）
python -c "
from datasets import load_dataset
ds = load_dataset('gsm8k', 'main')
print(f'训练集: {len(ds[\"train\"])} 条')
print(f'测试集: {len(ds[\"test\"])} 条')
print(f'示例: {ds[\"train\"][0]}')
"
```

**数据预处理脚本** (`prepare_data.py`)：

```python
"""将 GSM8K 数据转换为 verl 所需格式"""
import json
from datasets import load_dataset

def extract_gsm8k_answer(answer_text):
    """从 GSM8K 的 answer 字段中提取最终数值答案"""
    import re
    match = re.search(r'####\s*(\d+)', answer_text)
    if match:
        return int(match.group(1).replace(',', ''))
    return None

def prepare_gsm8k_for_verl(split="train", output_path="data/gsm8k_train.jsonl"):
    """准备 verl 训练数据"""
    ds = load_dataset("gsm8k", "main")
    
    with open(output_path, "w") as f:
        for item in ds[split]:
            answer = extract_gsm8k_answer(item["answer"])
            if answer is None:
                continue
            
            verl_sample = {
                "messages": [
                    {"role": "system", "content": "你是一个数学助手。请分析问题，必要时使用 calculator 工具计算，最终给出答案。"},
                    {"role": "user", "content": item["question"]}
                ],
                "ground_truth": answer,
                "data_source": "gsm8k",
                "agent_name": "math_agent",
            }
            f.write(json.dumps(verl_sample, ensure_ascii=False) + "\n")
    
    print(f"已保存到 {output_path}")

# 准备训练集和测试集
prepare_gsm8k_for_verl("train", "data/gsm8k_train.jsonl")
prepare_gsm8k_for_verl("test", "data/gsm8k_test.jsonl")
```

```bash
# 执行数据准备
mkdir -p data
python prepare_data.py
```

---

## 15.3 Step 2: 定义计算器工具

`calculator_tool.py`：

```python
"""计算器工具 —— 供 Agent 在 RL 训练中调用"""
import re
import ast
import operator

# 允许的运算符白名单（安全过滤）
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def safe_eval(expression: str) -> float:
    """安全地计算数学表达式（不使用 eval，防止代码注入）"""
    
    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"不允许的运算符: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"不允许的运算符: {op_type.__name__}")
        else:
            raise ValueError(f"不支持的表达式节点: {type(node).__name__}")
    
    tree = ast.parse(expression, mode='eval')
    return _eval_node(tree.body)


class CalculatorTool:
    """计算器工具"""
    
    name = "calculator"
    description = "计算数学表达式。输入一个数学表达式字符串，返回计算结果。"
    
    def execute(self, expression: str) -> str:
        """执行计算
        
        Args:
            expression: 数学表达式，如 "234 * 567 + 89"
        
        Returns:
            计算结果字符串，或错误信息
        """
        try:
            # 清理输入：去除空格、逗号等
            cleaned = expression.replace(",", "").replace(" ", "")
            result = safe_eval(cleaned)
            
            # 如果是整数，返回整数格式
            if result == int(result):
                return str(int(result))
            return str(round(result, 6))
        except Exception as e:
            return f"计算错误: {str(e)}"


# 测试
if __name__ == "__main__":
    calc = CalculatorTool()
    print(calc.execute("234 * 567 + 89"))       # 132767
    print(calc.execute("(123 + 456) * 789"))     # 456771
    print(calc.execute("100 / 3"))               # 33.333333
    print(calc.execute("__import__('os')"))       # 计算错误（安全拦截）
```

---

## 15.4 Step 3: 定义奖励函数

`reward_function.py`：

```python
"""数学推理 Agent 的奖励函数"""
import re

def extract_answer(text: str) -> str:
    """从模型回复中提取最终数字答案
    
    尝试多种提取方式：
    1. \boxed{} 格式
    2. "答案是: XXX" 格式  
    3. "最终答案: XXX" 格式
    4. 回复中最后一个数字
    """
    # 方式1: \boxed{答案}
    match = re.search(r'\\boxed\{(.+?)\}', text)
    if match:
        return match.group(1).strip()
    
    # 方式2: "答案是" / "最终答案" / "answer is"
    match = re.search(r'(?:答案|最终答案|answer)[是为：:\s]+(-?[\d,]+\.?\d*)', text, re.I)
    if match:
        return match.group(1).replace(",", "").strip()
    
    # 方式3: 最后一个独立的数字
    numbers = re.findall(r'(-?[\d,]+\.?\d*)', text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    
    return None


def math_agent_reward(completions: list, ground_truths: list, **kwargs) -> list:
    """数学推理 Agent 奖励函数
    
    奖励组成：
    - 正确性奖励: 答案正确 +1.0，错误 0.0
    - 工具使用奖励: 使用了 calculator 工具 +0.1
    - 思考过程奖励: 有推理步骤（<think> 标签） +0.05
    
    Args:
        completions: 模型生成的回复列表
        ground_truths: 正确答案列表
    
    Returns:
        每个回复对应的奖励值列表
    """
    rewards = []
    
    for completion, gt in zip(completions, ground_truths):
        reward = 0.0
        
        # 1. 正确性奖励（核心信号）
        answer = extract_answer(completion)
        if answer is not None:
            try:
                if abs(float(answer) - float(gt)) < 1e-6:
                    reward += 1.0
            except ValueError:
                pass  # 答案无法转为数字
        
        # 2. 工具使用奖励（鼓励使用 calculator）
        if "calculator" in completion.lower():
            reward += 0.1
        
        # 3. 思考过程奖励（鼓励结构化思考）
        if "<think>" in completion and "</think>" in completion:
            reward += 0.05
        
        # 4. 长度惩罚（防止生成长度爆炸）
        response_length = len(completion)
        if response_length > 500:
            reward -= 0.01 * ((response_length - 500) / 500)
        
        rewards.append(reward)
    
    return rewards


# 测试
if __name__ == "__main__":
    # 模拟测试
    completions = [
        "<think>需要计算 234*567+89</think>\ncalculator('234*567+89')\n答案是 132767",
        "我觉得答案大概是 100000",
        "<think>分步计算</think>\ncalculator('234*567')\ncalculator('132678+89')\n最终答案是 132767",
    ]
    ground_truths = [132767, 132767, 132767]
    
    rewards = math_agent_reward(completions, ground_truths)
    for i, (c, r) in enumerate(zip(completions, rewards)):
        print(f"回复 {i+1}: 奖励 = {r}")
    # 预期输出:
    # 回复 1: 奖励 = 1.15 (正确 + 工具 + 思考)
    # 回复 2: 奖励 = 0.0  (错误，无工具，无思考)
    # 回复 3: 奖励 = 1.15 (正确 + 工具 + 思考)
```

---

## 15.5 Step 4: 编写训练脚本

`run_math_agent.sh`（使用 verl 框架）：

```bash
#!/bin/bash
# run_math_agent.sh —— 用 GRPO 训练数学推理 Agent

set -e

# 基础配置
MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN_DATA="data/gsm8k_train.jsonl"
TEST_DATA="data/gsm8k_test.jsonl"
OUTPUT_DIR="checkpoints/math_agent_v1"
PROJECT_NAME="math_agent_grpo"

# 启动训练
python -m verl.trainer.main \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.max_new_tokens=1024 \
    actor_rollout_ref.rollout.mode=async \
    \
    algorithm=grpo \
    algorithm.kl_coeff=0.01 \
    \
    trainer.total_epochs=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=qwen2_5_3b_gsm8k \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "训练完成！模型保存在 ${OUTPUT_DIR}"
```

**关键超参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `rollout.n` | 8 | GRPO 每组采样数（G），越大信号越稳定但越慢 |
| `rollout.temperature` | 0.8 | 生成温度，控制探索程度 |
| `rollout.max_new_tokens` | 1024 | 最大生成长度 |
| `kl_coeff` | 0.01 | KL 惩罚系数，防止偏离 SFT 模型太远 |
| `total_epochs` | 5 | 训练轮数 |

---

## 15.6 Step 5: 启动训练

```bash
# 确保已登录 wandb（用于实验追踪）
wandb login

# 启动训练
bash run_math_agent.sh
```

---

## 15.7 Step 6: 监控训练进度

在 wandb 面板中观察以下关键指标：

```python
import wandb

# 关键监控指标：
# - reward_mean: 平均奖励趋势（应该逐步上升）
# - kl_divergence: KL散度（不要太大，说明策略没有偏离太远）
# - response_length: 回复长度变化（观察模型是否学会使用工具格式）
# - tool_use_rate: 工具使用率（应该逐步上升）
# - group_advantage_std: 组内优势标准差（如果接近 0 说明梯度消失）

# 训练过程中的健康指标：
# ✓ reward_mean 稳步上升
# ✓ kl_divergence 保持在 0.1 以下
# ✓ tool_use_rate 从低逐步升高到 0.5+
# ✓ 没有出现 reward 突然崩溃
```

---

## 15.8 Step 7: 评估模型

`evaluate_model.py`：

```python
"""评估训练后的数学推理 Agent"""
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

# 加载训练后的模型
model_path = "checkpoints/math_agent_v1"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# 系统提示（与训练时一致）
SYSTEM_PROMPT = "你是一个数学助手。请分析问题，必要时使用 calculator 工具计算，最终给出答案。"

def generate_response(question: str, max_new_tokens=512) -> str:
    """生成 Agent 回复"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )
    
    # 只取新生成的 token
    new_tokens = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def extract_answer(text: str) -> float:
    """提取最终数字答案（与 reward_function.py 保持一致）"""
    # 方式1: \boxed{答案}
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    # 方式2: "答案是"/"最终答案"/"answer is"
    match = re.search(r'(?:答案|最终答案|answer)[是为：:\s]+(-?[\d,]+\.?\d*)', text, re.I)
    if match:
        return float(match.group(1).replace(",", ""))
    # 方式3: 最后一个数字
    numbers = re.findall(r'(-?[\d,]+\.?\d*)', text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None

# 在 GSM8K 测试集上评估
ds = load_dataset("gsm8k", "main")["test"]
correct = 0
total = min(100, len(ds))  # 评估前 100 条

print(f"开始评估，共 {total} 条...")
for i in range(total):
    question = ds[i]["question"]
    answer_text = ds[i]["answer"]
    
    # 提取正确答案（GSM8K 格式: #### 数字）
    match = re.search(r'####\s*(\d+)', answer_text)
    if not match:
        continue
    ground_truth = int(match.group(1).replace(",", ""))
    
    # 模型生成回复
    response = generate_response(question)
    predicted = extract_answer(response)
    
    # 判断正确性
    is_correct = predicted is not None and abs(predicted - ground_truth) < 1e-6
    if is_correct:
        correct += 1
    
    if (i + 1) % 10 == 0:
        print(f"  进度: {i+1}/{total}, 正确率: {correct/(i+1):.2%}")

print(f"\n最终正确率: {correct}/{total} = {correct/total:.2%}")
```

```bash
# 运行评估
python evaluate_model.py
```

---

## 15.9 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 奖励全为 0 | 模型不会调用工具 | 检查 SFT 数据格式，确保工具调用格式正确 |
| KL 散度飙升 | 学习率太大 | 降低 `learning_rate`（如从 1e-6 降到 5e-7） |
| 训练崩溃 | 奖励方差过大 | 增加 GRPO group size（`rollout.n` 从 8 提高到 16） |
| 模型不调用工具 | 工具格式不对 | 检查 tokenizer 是否正确注册了工具调用模板 |
| 奖励全部相同 | 任务难度单一 | 增加数据集中任务难度的多样性 |
| 生成长度爆炸 | 缺乏长度惩罚 | 添加 `max_new_tokens` 限制，或增加长度惩罚项 |
| 训练速度很慢 | GPU 空闲等待工具 | 使用 `rollout.mode=async` 启用异步 Rollout |

---

## 15.10 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| End-to-End Training | 端到端训练 |
| Checkpoint | 检查点（模型快照） |
| WandB Dashboard | 实验追踪面板 |
| KL Divergence | KL散度（策略偏离程度） |
| Async Rollout | 异步展开（避免工具等待导致 GPU 空闲） |

---

## 15.11 小结

- 完成了第一个完整的 Agentic RL 训练项目
- 关键步骤：准备数据 → 定义工具 → 设计奖励 → 训练 → 评估
- 下一步：学习网页导航 Agent 训练 → [16 - 网页导航Agent训练](./16-网页导航Agent训练.md)
