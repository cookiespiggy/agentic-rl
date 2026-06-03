# 20 - 评估与 Benchmark

> **学习目标**：掌握 Agentic RL 模型的评估方法和主流 Benchmark。

---

## 20.1 评估维度

### 20.1.1 能力维度

| 维度 | 英文 | 评估内容 |
|------|------|---------|
| 任务完成率 | Task Success Rate | Agent 是否完成了任务 |
| 步骤效率 | Step Efficiency | 完成任务用了多少步 |
| 工具使用准确率 | Tool Use Accuracy | 是否调用了正确的工具 |
| 推理质量 | Reasoning Quality | 思考过程是否合理 |
| 探索效率 | Exploration Efficiency | 在未知环境中发现有效策略的速度 |
| 成本效率 | Cost Efficiency | 完成任务消耗的 token / API 调用次数 |
| 信息获取率 | Information Gain | 每次交互获取的有用信息量 |
| 鲁棒性 | Robustness | 面对异常情况的处理能力 |
| 泛化能力 | Generalization | 在未见任务上的表现 |

### 20.1.2 评估指标

```python
def task_completed(model, task):
    """检查 Agent 是否成功完成任务"""
    trajectory = model.generate_trajectory(task["prompt"])
    if "ground_truth" not in task:
        return True
    return task["ground_truth"] in str(trajectory)


def count_steps(model, task):
    """统计 Agent 完成任务所用的步数"""
    trajectory = model.generate_trajectory(task["prompt"])
    return len(trajectory.get("turns", []))


def correct_tool_used(model, task):
    """检查 Agent 是否使用了正确的工具"""
    trajectory = model.generate_trajectory(task["prompt"])
    expected_tool = task.get("expected_tool")
    if expected_tool is None:
        return True
    actual_tools = [c["tool"] for c in trajectory.get("tool_calls", [])]
    return expected_tool in actual_tools


def evaluate_agent(model, test_tasks):
    metrics = {}
    
    # 任务成功率（Task Success Rate）
    successes = sum(1 for t in test_tasks if task_completed(model, t))
    metrics["success_rate"] = successes / len(test_tasks)
    
    # 平均步骤数（Average Steps）
    all_steps = [count_steps(model, t) for t in test_tasks]
    metrics["avg_steps"] = sum(all_steps) / len(all_steps)
    
    # 工具使用准确率（Tool Use Accuracy）
    correct_tools = sum(1 for t in test_tasks if correct_tool_used(model, t))
    metrics["tool_accuracy"] = correct_tools / len(test_tasks)
    
    # Token 消耗总量（Cost Efficiency）
    all_tokens = [model.count_tokens(t) for t in test_tasks]
    metrics["avg_tokens"] = sum(all_tokens) / len(all_tokens)
    
    return metrics
```

---

## 20.2 主流 Benchmark

### 20.2.1 Web Navigation（网页导航）

| Benchmark | 任务数 | 难度 | 说明 |
|-----------|-------|------|------|
| MiniWoB | 100+ | 简单 | 简化网页任务 |
| WebArena | 800+ | 中等 | 真实网站模拟 |
| VisualWebArena | 900+ | 较难 | 多模态网页 |
| OSWorld | 300+ | 困难 | 操作系统级任务 |
| AgentBench | 300+ | 中等 | 多维度 LLM Agent 评测（含 Web/OS/DB） |
| GAIA | 466 | 困难 | 通用 AI 助手多步推理 |

### 20.2.2 Code（代码）

| Benchmark | 任务数 | 说明 |
|-----------|-------|------|
| HumanEval | 164 | 函数级代码生成 |
| SWE-bench | 2000+ | 仓库级代码修复 |
| MBPP | 1000 | 基础编程问题 |
| ToolBench | 16000+ | 工具使用综合评估（含 49 类工具） |

### 20.2.3 Search & QA（搜索问答）

| Benchmark | 任务数 | 说明 |
|-----------|-------|------|
| HotpotQA | 10000+ | 多跳推理问答 |
| 2WikiMultiHopQA | 1000+ | 多源知识问答 |
| BrowseComp | 100+ | 复杂浏览任务 |
| MINT | 100+ | 多轮交互工具使用评测 |

### 20.2.4 Mathematics（数学）

| Benchmark | 任务数 | 说明 |
|-----------|-------|------|
| GSM8K | 8500 | 小学数学应用题 |
| MATH | 12000 | 竞赛级数学 |
| AIME | 30/年 | 高级数学竞赛 |

### 20.2.5 General Agent（通用智能体）

| Benchmark | 任务数 | 说明 |
|-----------|-------|------|
| MT-Bench | 80 | 多轮对话质量评估 |
| ALFWorld | 100+ | 文本游戏任务（客厅/厨房/卧室等场景） |
| AgentInstruct | 1800+ | 多领域 Agent 指令数据集 |

---

## 20.3 评估实践

### 20.3.1 评估流水线

```
Model Checkpoint（模型检查点）
    ↓
Load Model + Tokenizer（加载模型和分词器）
    ↓
Run Inference on Test Set（在测试集上推理）
    ↓
Compute Metrics（计算指标）
    ↓
Compare with Baselines（与基线对比）
    ↓
Analyze Failures（分析失败案例）
```

### 20.3.2 自动化评估集成（lm-evaluation-harness）

```python
"""使用 lm-evaluation-harness 进行标准化评估"""
# 安装: pip install lm-eval

def evaluate_with_lm_eval(model_args: str, tasks: list, batch_size: int = 8):
    """集成 lm-eval 评估
    
    Args:
        model_args: 模型参数如 "pretrained=Qwen/Qwen2.5-7B"
        tasks: 任务列表如 ["gsm8k", "miniwob"]
        batch_size: 批处理大小
    """
    import lm_eval
    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=0,
        device="cuda",
    )
    
    # 格式化输出
    for task, metric in results["results"].items():
        print(f"{task}: acc={metric.get('acc,none', 'N/A')}")
    
    return results


# 使用示例
evaluate_with_lm_eval(
    model_args="pretrained=Qwen/Qwen2.5-7B",
    tasks=["gsm8k", "miniwob_click_test"],
)
```

### 20.3.3 A/B 测试与统计显著性检验

```python
"""A/B 测试与 Bootstrap 显著性检验"""
import numpy as np
from scipy import stats


def bootstrap_test(scores_a: list, scores_b: list, n_iterations: int = 10000):
    """Bootstrap 检验两个模型的性能差异是否显著
    
    Args:
        scores_a: 模型 A 在测试集上的得分列表
        scores_b: 模型 B 在测试集上的得分列表
        n_iterations: 重采样次数
    
    Returns:
        p_value: 差异显著性水平
    """
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    combined = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)
    
    count = 0
    for _ in range(n_iterations):
        np.random.shuffle(combined)
        diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if diff >= observed_diff:
            count += 1
    
    p_value = count / n_iterations
    return p_value


# 使用示例
baseline_scores = [0.75, 0.80, 0.72, 0.78, 0.76]  # SFT-only
grpo_scores = [0.85, 0.88, 0.82, 0.90, 0.86]       # SFT + GRPO
p_val = bootstrap_test(baseline_scores, grpo_scores)
print(f"p-value = {p_val:.4f}")  # p < 0.05 表示差异显著
print("结论:", "GRPO 显著优于 Baseline" if p_val < 0.05 else "差异不显著")
```

对比不同训练策略的效果:
- **Baseline**: SFT-only model
- **Experiment A**: SFT + GRPO
- **Experiment B**: SFT + PPO
- 在同一测试集上对比 metrics，使用 Bootstrap 检验统计显著性

---

## 20.4 评估环境部署指南

### 20.4.1 环境搭建

| 环境 | 部署方式 | 资源需求 | 备注 |
|------|---------|---------|------|
| MiniWoB | `pip install miniwob` | 2 CPU, 4GB RAM | 本地浏览器模拟 |
| WebArena | Docker Compose | 8 CPU, 16GB RAM | 需申请 API key |
| SWE-bench | `pip install swebench` | 4 CPU, 8GB RAM | 需配置 Docker 沙盒 |
| BrowserGym | `pip install browsergym` | 2 CPU, 4GB RAM | 支持 MiniWoB/WebArena |

### 20.4.2 环境检查清单

- [ ] Docker 已安装并正常运行
- [ ] 对应的 API key 已配置（环境变量或 .env 文件）
- [ ] 网络代理设置正确（部分环境需科学上网）
- [ ] 测试集已下载到本地 `data/` 目录
- [ ] 基准测试可复现（运行官方示例确认环境正常）

### 20.4.3 跨 Benchmark 交叉验证

```python
"""多个 Benchmark 交叉验证，防止过拟合特定评测"""

CROSS_VALIDATION_TASKS = {
    "web_nav": ["miniwob", "webarena", "browsergym"],
    "code": ["humaneval", "mbpp", "swebench"],
    "search": ["hotpotqa", "2wikimultihopqa"],
}

def cross_validate(model, task_groups: dict):
    """在多个 Benchmark 上评估，计算综合得分"""
    all_results = {}
    for group_name, tasks in task_groups.items():
        group_scores = []
        for task in tasks:
            score = evaluate_agent(model, [{"prompt": task}])
            group_scores.append(score["success_rate"])
        all_results[group_name] = {
            "scores": group_scores,
            "mean": sum(group_scores) / len(group_scores),
            "std": __import__("statistics").stdev(group_scores) if len(group_scores) > 1 else 0,
        }
    return all_results
```

---

## 20.5 常见评估陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|---------|
| 数据泄露 | 测试集出现在训练集中 | 严格划分训练/测试集 |
| 评估偏差 | 只测简单任务 | 包含不同难度的测试 |
| 单一指标 | 只看成功率 | 多维度评估 |
| 过拟合评测 | 针对特定Benchmark优化 | 多个Benchmark交叉验证 |

---

## 20.7 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Benchmark | 基准测试 |
| Task Success Rate | 任务成功率 |
| Step Efficiency | 步骤效率 |
| Cost Efficiency | 成本效率 |
| Exploration Efficiency | 探索效率 |
| Generalization | 泛化能力 |
| A/B Testing | A/B 测试 |
| Bootstrap Test | 自助法显著性检验 |
| Data Leakage | 数据泄露 |

---

## 20.8 小结

- 评估需要多维度：成功率、效率、工具使用准确率等
- 选择合适的 Benchmark 来衡量你的垂直领域
- 注意评估陷阱：数据泄露、评估偏差
- 下一步：进入进阶主题 → [21 - Async Rollout与异步训练](./21-异步Rollout与分布式训练.md)
