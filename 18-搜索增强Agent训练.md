# 18 - 搜索增强 Agent 训练

> **学习目标**：训练一个能在推理过程中主动搜索外部知识的 Agent。

---

## 18.1 Search-R1 概述

**Search-R1**（2025）是搜索增强 RL 的代表性工作：

```
核心思想：
让 LLM 在推理过程中学会"何时搜索"和"搜索什么"

传统 RAG:
  query → 检索 → 生成

Search-R1:
  query → 推理 → [决定搜索] → 检索 → 继续推理 → [可能再搜索] → 最终答案
```

---

## 18.2 搜索环境设计

### 18.2.1 搜索 API 封装

> 基础的搜索环境接口设计参见 [13 - 训练环境搭建](./13-训练环境搭建.md)。下面是专为 Search-R1 训练设计的增强版本，增加了搜索次数限制和多引擎支持。

```python
class SearchEnvironment:
    """搜索引擎交互环境"""
    
    def __init__(self, search_engine="serpapi", max_results=5):
        self.search_engine = search_engine
        self.max_results = max_results
        self.search_count = 0
        self.max_searches = 3  # 限制搜索次数
    
    def search(self, query: str) -> list:
        """执行搜索"""
        self.search_count += 1
        if self.search_count > self.max_searches:
            return [{"error": "Search limit reached"}]
        
        results = self._call_search_api(query)
        return results[:self.max_results]
    
    def _call_search_api(self, query):
        """调用搜索API"""
        if self.search_engine == "serpapi":
            return self._serpapi_search(query)
        elif self.search_engine == "bing":
            return self._bing_search(query)
        else:
            return self._mock_search(query)
    
    def _mock_search(self, query):
        """模拟搜索（用于测试）"""
        return [
            {"title": f"Result 1 for {query}", "snippet": "..."},
            {"title": f"Result 2 for {query}", "snippet": "..."},
        ]
    
    def reset(self):
        self.search_count = 0
```

### 18.2.2 搜索工具定义

```python
SEARCH_TOOL_SPEC = {
    "name": "search",
    "description": "Search the web for information. Use this when you need external knowledge.",
    "parameters": {
        "query": {
            "type": "string",
            "description": "The search query"
        }
    }
}
```

---

## 18.3 训练数据格式

```python
search_agent_sample = {
    "messages": [
        {"role": "system", "content": "你是一个知识问答助手。当不确定答案时，请使用搜索工具查找信息。"},
        {"role": "user", "content": "世界上最高的山峰是哪座？"},
        {
            "role": "assistant",
            "content": "让我搜索一下这个信息。\n\n<think>\n世界上最高的山峰是哪座？让我搜索确认一下。\n</think>\n"
        },
        {"role": "tool", "content": '[{"title": "珠穆朗玛峰", "snippet": "珠穆朗玛峰是喜马拉雅山脉的主峰，海拔8848.86米，是世界最高峰。"}]'},
        {
            "role": "assistant",
            "content": "根据搜索结果，世界上最高的山峰是**珠穆朗玛峰**（Mount Everest），海拔8848.86米。"
        }
    ],
    "data_source": "qa_dataset",
    "agent_name": "search_agent"
}
```

---

## 18.4 搜索奖励设计

```python
def fuzzy_match(answer: str, ground_truth: str) -> bool:
    """模糊匹配答案（支持数字容差和子串匹配）"""
    if not answer or not ground_truth:
        return False
    # 尝试数字匹配（允许 1% 相对误差）
    try:
        a, b = float(answer.replace(",", "")), float(ground_truth.replace(",", ""))
        if abs(a - b) / (abs(b) + 1e-9) < 0.01:
            return True
    except ValueError:
        pass
    # 字符串子串匹配
    return str(ground_truth).lower() in str(answer).lower()


def search_agent_reward(completions: list, ground_truths: list,
                        trajectories: list = None, **kwargs) -> list:
    """搜索 Agent 奖励函数

    Args:
        completions: Agent 生成的回复列表
        ground_truths: 正确答案列表
        trajectories: 完整交互轨迹列表

    Returns:
        每个回复对应的奖励值列表
    """
    rewards = []

    for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
        reward = 0.0
        traj = trajectories[i] if trajectories else []

        # 1. 最终答案正确性
        answer = extract_answer(completion)
        if answer and fuzzy_match(answer, gt):
            reward += 1.0

        # 2. 搜索质量评估
        search_steps = [s for s in traj if isinstance(s, dict) and s.get("type") == "search"]
        for step in search_steps:
            query = step.get("query", "")
            results = step.get("results", [])

            if is_good_search_query(query, gt):
                reward += 0.05

            if has_relevant_results(results, gt):
                reward += 0.05

        # 3. 效率奖励（不要过度搜索）
        if len(search_steps) <= 2:
            reward += 0.05

        # 4. 惩罚无搜索就给答案
        if not search_steps and answer != gt:
            reward -= 0.1

        rewards.append(max(reward, -0.5))

    return rewards
```

---

## 18.5 Search-R1 训练脚本

```bash
#!/bin/bash
# run_search_r1.sh

MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN_DATA="data/search_train.parquet"
TEST_DATA="data/search_test.parquet"
OUTPUT_DIR="checkpoints/search_agent_v1"
PROJECT_NAME="search_agent_rl"

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
    trainer.experiment_name=qwen2_5_3b_search \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    tools.search.enabled=True \
    tools.search.max_searches=3 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}
```

---

## 18.6 搜索质量优化

### 18.6.1 搜索查询改写

```python
def evaluate_search_quality(trajectory):
    """评估搜索查询质量"""
    for step in trajectory:
        if step["type"] == "search":
            query = step["query"]
            # 好的搜索查询应该：
            # 1. 具体且有针对性
            # 2. 包含关键词
            # 3. 不是太长也不是太短
            if len(query.split()) < 2:
                return "too_short"
            if len(query) > 200:
                return "too_long"
    return "good"
```

### 18.6.2 搜索结果整合

```python
def synthesize_search_results(results_list):
    """整合多次搜索的结果"""
    synthesized = {
        "key_facts": [],
        "sources": [],
        "contradictions": []
    }
    
    for results in results_list:
        for result in results:
            facts = extract_facts(result["snippet"])
            synthesized["key_facts"].extend(facts)
            synthesized["sources"].append(result["title"])
    
    # 检测矛盾信息
    synthesized["contradictions"] = detect_contradictions(synthesized["key_facts"])
    
    return synthesized
```

---

## 18.7 术语速查

| 英文术语 | 中文注释 |
|---------|---------|
| Search-R1 | 搜索增强的 RL 框架 |
| RAG (Retrieval-Augmented Generation) | 检索增强生成 |
| Search Query | 搜索查询 |
| Search Result Relevance | 搜索结果相关性 |
| Multi-turn Search | 多轮搜索 |

---

## 18.8 小结

- Search-R1 让 LLM 学会在推理过程中主动搜索
- 搜索奖励设计需要平衡正确性、搜索质量和效率
- 搜索查询质量是 Agent 能力的关键指标
- 下一步：训练垂直领域 Agent → [19 - 垂直领域Agent训练](./19-垂直领域Agent训练.md)
