# 12 - 数据准备与 SFT 阶段

> **学习目标**：学会准备 Agentic RL 的训练数据，并完成 SFT（Supervised Fine-Tuning，监督微调）阶段。

---

## 12.1 整体数据流水线

```
原始数据 → 数据清洗 → SFT 数据构建 → SFT 训练 → Agent 格式数据 → RL 训练
```

---

## 12.2 SFT 数据格式

### 12.2.1 基本对话格式（ChatML）

```python
sample = {
    "messages": [
        {"role": "system", "content": "你是一个专业的客服助手"},
        {"role": "user", "content": "我的订单什么时候到？"},
        {"role": "assistant", "content": "让我帮您查询订单状态..."}
    ]
}
```

### 12.2.2 工具调用格式（Tool Calling Format）

SFT 阶段需要教模型正确的工具调用格式：

```python
tool_call_sample = {
    "messages": [
        {"role": "system", "content": "你是一个助手，可以使用工具帮助用户。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
        {
            "role": "assistant",
            "content": '我来查询天气。\n\n<think>\n让我查一下北京的天气。\n</think>\n\n{"name": "get_weather", "arguments": {"city": "北京"}}'
        },
        {"role": "tool", "content": '{"temperature": "28C", "condition": "晴"}'},
        {"role": "assistant", "content": "北京今天天气晴朗，气温28度。"}
    ],
    "tools": [
        {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "city": {"type": "string", "description": "城市名称"}
            }
        }
    ]
}
```

### 12.2.3 多轮 Agent 格式

```python
# 多轮 Agent 交互数据
agent_sample = {
    "messages": [
        {"role": "system", "content": "你是一个数据分析助手"},
        {"role": "user", "content": "分析销售数据中的趋势"},
        {"role": "assistant", "content": "我需要先查询数据...\n\n<think>\n让我查询销售数据表。\n</think>\n\n{\"name\": \"run_sql\", \"arguments\": {\"query\": \"SELECT * FROM sales\"}}"},
        {"role": "tool", "content": '[{"month": "Jan", "sales": 100}, ...]'},
        {"role": "assistant", "content": "数据获取成功，现在进行可视化...\n\n<think>\n数据已获取，我来画一个趋势图。\n</think>\n\n{\"name\": \"plot_chart\", \"arguments\": {\"type\": \"line\"}}"},
        {"role": "tool", "content": "图表已生成: chart_001.png"},
        {"role": "assistant", "content": "根据分析，销售趋势如下..."}
    ],
    "agent_name": "data_analyst"  # verl 需要的字段，标识使用哪个 Agent
}
```

---

## 12.3 数据构建方法

### 12.3.1 方法一：人工标注（高质量，高成本）

```
优点：质量最高，格式最准确
缺点：成本高，速度慢
适用：初始种子数据（Seed Data），几百条即可
```

### 12.3.2 方法二：大模型蒸馏（Distillation，推荐）

用大模型（如 GPT-4、Claude）生成训练数据给小模型：

```python
import json

def generate_sft_data_with_teacher(teacher_model, task_description, num_samples=100):
    """用大模型（Teacher Model，教师模型）生成 SFT 训练数据"""
    
    system_prompt = """你是一个数据生成专家。
    请根据任务描述，生成高质量的 Agent 交互数据。
    包含正确的工具调用格式。"""
    
    response = teacher_model(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"为以下任务生成10条训练数据：{task_description}"}
        ]
    )
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)
```

**蒸馏流程**：
```
Teacher Model (GPT-4, 70B+)
    ↓ 生成高质量 Agent 轨迹
Student Model (Qwen-7B, 目标小模型)
    ↓ 用 SFT 学习 Teacher 的行为
    ↓ 然后用 RL 进一步优化
```

### 12.3.3 方法三：Self-Instruct（自指令生成）

让模型自己生成问题和回答：

```python
def generate_instructions(existing_data, num_new=5):
    """基于已有数据生成新指令"""
    new_instructions = []
    for _ in range(num_new):
        new_instructions.append(f"为以下示例生成类似的任务：{existing_data[0]}")
    return new_instructions

def filter_by_quality(data_list, min_length=20):
    """质量过滤：去除过短或无效的样本"""
    return [d for d in data_list if len(d.get("response", "")) >= min_length]

def self_instruct(base_model, tokenizer, seed_examples, num_iterations=5):
    """自指令数据生成"""
    all_data = list(seed_examples)
    
    for i in range(num_iterations):
        # 1. 基于已有数据生成新的指令
        new_instructions = generate_instructions(all_data)
        
        # 2. 用模型回答新指令
        for instruction in new_instructions:
            inputs = tokenizer(instruction, return_tensors="pt")
            outputs = base_model.generate(**inputs, max_new_tokens=512)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_data.append({"instruction": instruction, "response": response})
        
        # 3. 质量过滤
        all_data = filter_by_quality(all_data)
    
    return all_data
```

---

## 12.4 数据质量过滤

### 12.4.1 过滤规则

```python
import hashlib
import json
import re

def has_tool_call(sample):
    """检查样本是否包含工具调用"""
    for msg in sample["messages"]:
        if msg["role"] == "assistant" and re.search(r'\{["\']name["\']', msg["content"]):
            return True
    return False

def has_tool_response(sample):
    """检查工具调用后是否有对应的 tool response"""
    roles = [m["role"] for m in sample["messages"]]
    for i, r in enumerate(roles):
        if r == "assistant" and i + 1 < len(roles) and roles[i + 1] == "tool":
            return True
    return False

def is_duplicate(sample, existing_list, threshold=0.9):
    """基于内容哈希的简单去重"""
    content = "".join(m["content"] for m in sample["messages"])
    h = hashlib.md5(content.encode()).hexdigest()
    seen = set()
    for s in existing_list:
        ch = "".join(m["content"] for m in s["messages"])
        seen.add(hashlib.md5(ch.encode()).hexdigest())
    return h in seen

def extract_json_from_tool_call(content):
    """从工具调用内容中提取 JSON 字符串"""
    match = re.search(r'\{.*"name".*"arguments".*\}', content, re.DOTALL)
    return match.group(0) if match else ""

def validate_tool_call_format(sample):
    """验证工具调用格式是否正确"""
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            if '<think>' in content or '</think>' in content:
                try:
                    json_str = extract_json_from_tool_call(content)
                    if json_str:
                        json.loads(json_str)
                except json.JSONDecodeError:
                    return False
    return True

def filter_sft_data(data_list):
    """SFT 数据质量过滤"""
    filtered = []
    
    for sample in data_list:
        # 1. 长度过滤
        total_len = sum(len(m["content"]) for m in sample["messages"])
        if total_len > 8192 or total_len < 50:
            continue
        
        # 2. 格式检查
        if not validate_tool_call_format(sample):
            continue
        
        # 3. 工具调用有效性
        if has_tool_call(sample) and not has_tool_response(sample):
            continue
        
        # 4. 去重（基于内容哈希）
        if is_duplicate(sample, filtered):
            continue
        
        filtered.append(sample)
    
    return filtered
```

### 12.4.2 数据平衡

```python
import random
from collections import Counter

def get_tool_type(sample):
    """推断样本使用的工具类型"""
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            match = re.search(r'"name"\s*:\s*"(\w+)"', msg["content"])
            if match:
                return match.group(1)
    return "unknown"

def balance_dataset(data_list):
    """平衡数据集的类别分布"""
    tool_counts = Counter(get_tool_type(d) for d in data_list)
    print(f"工具分布: {tool_counts}")
    
    target_count = len(data_list) // len(tool_counts)
    
    balanced = []
    for tool_type in tool_counts:
        subset = [d for d in data_list if get_tool_type(d) == tool_type]
        if len(subset) > target_count:
            subset = random.sample(subset, target_count)
        balanced.extend(subset)
    
    return balanced
```

---

## 12.5 SFT 训练

### 12.5.1 使用 verl 进行 SFT

```bash
# verl 提供的 SFT 脚本示例
bash examples/sft/run_qwen2_5_3b_sft.sh
```

### 12.5.2 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 1e-5 ~ 5e-6 | SFT 学习率 |
| epochs | 2-5 | 训练轮数 |
| batch_size | 4-8 | 批大小 |
| max_seq_length | 4096-8192 | 最大序列长度 |
| warmup_ratio | 0.05 | 预热比例 |
| weight_decay | 0.01 | 权重衰减 |

### 12.5.3 SFT 后的验证

```python
from transformers import AutoTokenizer

def has_correct_format(generated):
    """检查回复格式是否包含 </think> 结束标记"""
    return '</think>' in generated

def has_valid_tool_call(generated):
    """检查回复是否包含合法的 JSON 工具调用"""
    return re.search(r'\{["\']name["\']\s*:\s*["\']\w+["\']', generated) is not None

def evaluate_sft_model(model, tokenizer, test_data, max_length=2048):
    """评估 SFT 模型"""
    metrics = {
        "format_accuracy": 0,
        "tool_call_accuracy": 0,
        "response_quality": 0,
    }
    
    for sample in test_data:
        input_text = sample["messages"][-1]["content"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = model.generate(**inputs, max_new_tokens=512)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        expected = sample["messages"][-1]["content"]
        
        if has_correct_format(generated):
            metrics["format_accuracy"] += 1
        
        if '<think>' in expected and '</think>' in expected:
            if has_valid_tool_call(generated):
                metrics["tool_call_accuracy"] += 1
    
    n = len(test_data)
    for key in metrics:
        metrics[key] /= n
    
    return metrics
```

---

## 12.6 verl 中的数据预处理

### 12.6.1 数据集准备脚本

```python
# examples/data_preprocess/my_dataset.py
import json

def prepare_for_verl(input_path, output_path):
    """将数据转换为 verl 所需格式"""
    
    with open(input_path) as f:
        data = json.load(f)
    
    verl_data = []
    for sample in data:
        verl_sample = {
            "messages": sample["messages"],
            "agent_name": "tool_agent_loop",  # 使用工具 Agent Loop
            "data_source": "my_domain",
        }
        verl_data.append(verl_sample)
    
    with open(output_path, 'w') as f:
        json.dump(verl_data, f)

prepare_for_verl("raw_data.json", "verl_data.json")
```

---

## 12.7 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| ChatML (Chat Markup Language) | 聊天标记语言 |
| Distillation | 蒸馏（大模型知识迁移到小模型） |
| Teacher Model / Student Model | 教师模型 / 学生模型 |
| Self-Instruct | 自指令生成 |
| Undersampling / Oversampling | 下采样 / 过采样 |
| Seed Data | 种子数据 |
| Data Preprocessing | 数据预处理 |

---

## 12.8 小结

- SFT 是 Agentic RL 的第一步，教模型基本的工具调用格式
- 数据质量 > 数据数量，100 条高质量数据 > 10000 条低质量数据
- 大模型蒸馏（Distillation）是最高效的数据构建方式
- 下一步：搭建训练环境 → [13 - 训练环境搭建](./13-训练环境搭建.md)
