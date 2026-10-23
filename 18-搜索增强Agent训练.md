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
"""搜索环境（Search-R1 专用增强版）

基础接口参见 [13 - 训练环境搭建](./13-训练环境搭建.md)。这里额外加了「轨迹记录」，
供 §18.4 的奖励函数使用——TRL 只把 (prompts, completions, **kwargs) 传给奖励函数，
没有 trajectories 参数，所以只能由环境侧把每一步写进模块级记录器。
"""


class SearchTrajectoryRecorder:
    """按 rollout 缓存搜索步骤。

    说明：这是一个教学简化版，只缓存**最近一条** rollout 的步骤。
    生产环境请改用 verl 的 AgentLoop，它会把轨迹挂在样本的 extra_info 上。
    """

    def __init__(self):
        self._steps: list = []

    def reset(self):
        self._steps = []

    def record(self, query: str, results: list):
        self._steps.append({"type": "search", "query": query, "results": results})

    def take(self) -> list:
        """取出并清空已记录的步骤"""
        steps, self._steps = self._steps, []
        return steps


# 模块级单例：环境负责写，奖励函数负责读
TRAJ_RECORDER = SearchTrajectoryRecorder()


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
        results = results[:self.max_results]
        TRAJ_RECORDER.record(query, results)  # ← 供 §18.4 的奖励函数读取
        return results
    
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

> 这份 tool spec 就是 §18.5.1 里 `tools/search_tool.yaml` 的 `tool_schema` 部分——
> verl 侧直接用 YAML 版本，不要同时维护两份。

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

### 18.3.1 本章数据的来源

> **本章不提供数据生成脚本。** 搜索语料（网页快照、搜索引擎 API 配额）和人工标注的
> 多跳问答**无法离线获取**，仓库里也没有可直接跑的数据。下面的字段清单和样例只是
> **契约**：你必须自己按这个格式把数据准备好，放到 `data/search_train.parquet`、
> `data/search_test.parquet`（训练脚本读的就是这两个路径）。
>
> 数据通常来自三处：
> 1. **HotpotQA / 2WikiMultiHopQA**（见 [20 - 评估与Benchmark](./20-评估与Benchmark.md)）
>    ——多跳问答有标准答案和支撑段落，最容易改造成本章格式；
> 2. 搜索引擎 API 抓回来的网页快照 —— 用来生成「搜索结果」；
> 3. 大模型蒸馏 —— 让 Teacher 模型按你的格式产出 `messages`（方法见
>    [19 - 垂直领域Agent训练](./19-垂直领域Agent训练.md) §19.6.2）。

### 18.3.2 数据集字段

`data/search_train.parquet` 每行是一条样本，字段如下：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `messages` | `list[dict]` | ✅ | 对话历史，角色按 `system → user → assistant → tool → assistant` 交替 |
| `answer` | `str` | ✅ | **正确答案**（ground truth）。奖励函数从这里取，不做位置参数 |
| `data_source` | `str` | ✅ | 数据来源标记，例如 `qa_dataset`、`hotpotqa` |
| `agent_name` | `str` | ✅ | Agent 角色名，例如 `search_agent` |
| `extra_info` | `dict` | ❌ | 附加信息（难度标签、支撑段落 id、来源 URL 等），训练时不参与打分 |

> `data.prompt_key=messages`（见 §18.5 的训练脚本）告诉 verl 从哪一列取 prompt；
> `answer` 列则通过 **kwargs 原样传给奖励函数。

### 18.3.3 示例记录

```python
# 第 1 条：单跳，答案就是搜索结果里的一句话
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
    "answer": "珠穆朗玛峰",
    "data_source": "qa_dataset",
    "agent_name": "search_agent",
    "extra_info": {"difficulty": "easy", "num_hops": 1},
}

# 第 2 条：多跳，需要先搜一个实体、再搜它的属性
search_agent_sample_multihop = {
    "messages": [
        {"role": "system", "content": "你是一个知识问答助手。当不确定答案时，请使用搜索工具查找信息。"},
        {"role": "user", "content": "《星际穿越》的导演出生于哪一年？"},
        {
            "role": "assistant",
            "content": "先查《星际穿越》的导演是谁。\n\n<think>\n需要两步：先定位导演，再查出生年份。\n</think>\n"
        },
        {"role": "tool", "content": '[{"title": "星际穿越 (电影)", "snippet": "《星际穿越》由克里斯托弗·诺兰执导。"}]'},
        {
            "role": "assistant",
            "content": "导演是克里斯托弗·诺兰，再查一下他的出生年份。\n\n<think>\n导演是诺兰。\n</think>\n"
        },
        {"role": "tool", "content": '[{"title": "克里斯托弗·诺兰", "snippet": "克里斯托弗·诺兰（Christopher Nolan），1970年7月30日出生于伦敦。"}]'},
        {
            "role": "assistant",
            "content": "《星际穿越》的导演克里斯托弗·诺兰出生于 **1970 年**。"
        }
    ],
    "answer": "1970",
    "data_source": "hotpotqa",
    "agent_name": "search_agent",
    "extra_info": {"difficulty": "medium", "num_hops": 2},
}
```

---

## 18.4 搜索奖励设计

### 18.4.1 统一契约

奖励函数的调用约定在本系列里是**统一**的（首次声明见
[14 - 奖励函数设计与实现](./14-奖励函数设计与实现.md) §14.1），本章沿用：

```
reward_func(prompts, completions, **kwargs) -> list[float]
```

- `prompts`：`list[list[dict]]`，数据集里的原始对话；
- `completions`：`list[list[dict]]`（对话模式）或 `list[str]`——**不是** `list[str]`，
  所以要先用 `_coerce_to_text` 转成文本；
- `kwargs`：数据集其余**列名**。本章约定 `answer`（正确答案）、`trajectories`（轨迹，可选）；
- ground truth 一律从 `kwargs` 取，**不作为位置参数**；
- `trajectories=` / `sandbox=` / `envs=` 这些**不是 TRL 的参数**，要么来自数据集列，
  要么通过 §18.2.1 的模块级 `TRAJ_RECORDER` 传入。

> **与 ch13 的关系**：ch13 的 `SearchEnvironment.compute_reward(final_answer, ground_truth)`
> 是「环境内部的单次打分」；本章的 `search_agent_reward` 是**训练器级**的奖励函数，
> 签名按上面的统一约定改写（ch13 那个方法签名不变，两者是不同层次的东西）。

### 18.4.2 实现

```python
"""搜索 Agent 奖励函数

统一契约（见 §18.4.1 与 ch14）：
    reward_func(prompts, completions, **kwargs) -> list[float]
"""
import re


def _coerce_to_text(completion) -> str:
    """把 TRL 的 completion 统一成文本。

    trl 1.5.1 对话模式下 completions 是 list[list[dict]]
    （形如 [{"role": "assistant", "content": "..."}]），不是 list[str]。
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        if isinstance(last, str):
            return last
    return str(completion)


def _keywords(text: str) -> list:
    """把 ground truth 切成关键词（按空白/中英文标点）"""
    return [w for w in re.split(r"[\s,，。.、:：;；()（）\[\]【】]+", text or "") if w]


def extract_answer(text: str) -> str:
    """从回复正文里提取最终答案。

    优先取「答案：/最终答案：/Answer:」之后的第一行；
    都没有时退回最后一段非空文本（先剥掉 <think>...</think>）。
    """
    text = (text or "").strip()
    match = re.search(r"(?:最终答案|答案|Answer)\s*(?:是)?\s*[:：]\s*(.+)", text, re.IGNORECASE)
    if match:
        lines = match.group(1).strip().splitlines()
        return lines[0].strip() if lines else ""
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else body


def is_good_search_query(query: str, ground_truth: str) -> bool:
    """判断搜索查询质量：长度适中，且至少命中 ground truth 的一个关键词。"""
    q = (query or "").strip()
    if not q:
        return False
    tokens = q.split()
    if len(tokens) < 2 or len(q) > 200:
        return False
    keywords = _keywords(ground_truth)
    if not keywords:
        return True
    return any(kw.lower() in q.lower() for kw in keywords)


def has_relevant_results(results: list, ground_truth: str) -> bool:
    """判断搜索结果里是否出现了与 ground truth 相关的内容。"""
    if not results:
        return False
    keywords = _keywords(ground_truth)
    if not keywords:
        return True
    blob = " ".join(
        f"{r.get('title', '')} {r.get('snippet', '')}"
        for r in results if isinstance(r, dict)
    ).lower()
    return any(kw.lower() in blob for kw in keywords)


def extract_facts(snippet: str) -> list:
    """从搜索摘要里抽取候选事实（简化版：按句号/换行切句）。"""
    if not snippet:
        return []
    facts = []
    for sent in re.split(r"[。！？!?\n]+", snippet):
        sent = sent.strip()
        if len(sent) >= 4:
            facts.append(sent)
    return facts


def detect_contradictions(facts: list) -> list:
    """检测事实里的简单矛盾。

    简化启发式：同一关键词后面出现两个不同的数值 → 记一条矛盾。
    真实系统里这一步基本要交给 LLM 判断。
    """
    contradictions = []
    seen = {}
    for fact in facts:
        pairs = re.findall(
            r"([A-Za-z一-鿿]{2,})\s*(?:为|是|约)?\s*([0-9]+(?:\.[0-9]+)?)",
            fact,
        )
        for kw, num in pairs:
            prev = seen.get(kw)
            if prev is None:
                seen[kw] = num
            elif prev != num:
                contradictions.append({"keyword": kw, "values": [prev, num], "fact": fact})
    return contradictions


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


def search_agent_reward(prompts, completions, **kwargs) -> list[float]:
    """搜索 Agent 奖励函数

    Args:
        prompts: list[list[dict]]，原始对话（本函数只用来对齐长度）
        completions: list[list[dict]] | list[str]
        kwargs: 数据集其余列。本章约定：
            answer       -> list[str]，正确答案（ground truth）
            trajectories -> list[list[dict]]，搜索轨迹（可选；
                           verl 里由 AgentLoop 挂在样本的 extra_info 上）

    Returns:
        list[float]，每个 completion 一个奖励。
    """
    answers = kwargs.get("answer")
    if answers is None:
        answers = kwargs.get("answers") or kwargs.get("ground_truths") or [None] * len(completions)

    # 轨迹来源二选一：数据集列优先；否则用模块级记录器（TRL 场景）
    trajectories = kwargs.get("trajectories")
    if trajectories is None:
        trajectories = [TRAJ_RECORDER.take() for _ in completions]

    rewards = []

    for i, completion in enumerate(completions):
        text = _coerce_to_text(completion)
        gt = answers[i] if i < len(answers) and answers[i] else ""
        traj = trajectories[i] if i < len(trajectories) and trajectories[i] else []

        reward = 0.0

        # 1. 最终答案正确性
        answer = extract_answer(text)
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

> **这个奖励没有 1.0 上界**：答对 +1.0、每次好查询 +0.05、每次相关结果 +0.05、效率 +0.05
> 是**累加**的，最多能到 1.35 左右。它和别的章里归一化到 [0,1] 的奖励**不可横向比较**。
> GRPO 只关心同一 prompt 组内的相对大小，绝对值不影响训练。

---

## 18.5 Search-R1 训练脚本

### 18.5.1 工具定义文件

verl 的多轮工具配置走两个路径：`multi_turn.tool_config_path` 指向一个 YAML
（`class_name` / `config` / `tool_schema` 三段），`multi_turn.function_tool_path`
指向一个 OpenAI 格式的 function schema JSON。

```yaml
# tools/search_tool.yaml —— multi_turn.tool_config_path 指向本文件
tools:
  - class_name: "search_tools.SearchTool"   # 你自己的 BaseTool 子类（继承 verl.tools.base.BaseTool）
    config:
      max_results: 5
      max_searches: 3
    tool_schema:
      type: "function"
      function:
        name: "search"
        description: "Search the web for information. Use this when you need external knowledge."
        parameters:
          type: "object"
          properties:
            query:
              type: "string"
              description: "The search query"
          required: ["query"]
```

> 第二个文件 `tools/search_tool_functions.json`（`multi_turn.function_tool_path` 指向它）
> 是一份 OpenAI 格式的 function schema 列表：

```json
[
  {
    "type": "function",
    "function": {
      "name": "search",
      "description": "Search the web for information. Use this when you need external knowledge.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
      }
    }
  }
]
```

> **注意**：verl **没有**「按工具名点开关」的配置段（常见的 `tools.<name>.enabled`
> 写法在 verl 里不存在）。多轮工具只有
> `actor_rollout_ref.rollout.multi_turn.{enable,tool_config_path,function_tool_path}`
> 三个键。另外 `function_tool_path` 在部分版本里被声明为路径**列表**，
> 那时要写成 `['tools/search_tool_functions.json']`——以你本地
> `verl/trainer/config/_generated_ppo_trainer.yaml` 为准。

### 18.5.2 训练脚本

```bash
#!/bin/bash
# run_search_r1.sh
#
# ⚠️ verl 训练需要 CUDA 环境；本仓库没有 CI 在跑这段脚本，下面的键只做过离线核对。
#
# --- verl 必填项（这 4 个键务必显式写出）---
#   actor_rollout_ref.actor.optim.lr=1e-6
#       该键在不同 verl 版本的有效默认值不同（v0.9.1 是 1e-6，更早版本是 1e-3），
#       所以不要依赖默认值，必须显式写。
#   data.prompt_key=messages
#       数据集里对话列的列名，字段契约见 §18.3.2。
#   data.train_batch_size=32
#   actor_rollout_ref.rollout.n=8
#       verl 的整除约束：train_batch_size × rollout.n 须被 n_gpus 整除
#       （verl/utils/config.py:123-124），这里 32 × 8 = 256，8 卡时每卡 32 条。
#       ⚠️ 这和 TRL 的规则**不同**：TRL 要求 generation_batch_size
#       （= per_device_train_batch_size × num_processes × steps_per_generation）
#       能被 num_generations 整除。两套规则不要混用。

MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN_DATA="data/search_train.parquet"
TEST_DATA="data/search_test.parquet"
OUTPUT_DIR="checkpoints/search_agent_v1"
PROJECT_NAME="search_agent_rl"
TOOL_CONFIG="tools/search_tool.yaml"
FUNCTION_TOOL_CONFIG="tools/search_tool_functions.json"

python -m verl.trainer.main_ppo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.prompt_key=messages \
    data.train_batch_size=32 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.mode=async \
    \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.01 \
    \
    trainer.total_epochs=3 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=qwen2_5_3b_search \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
    actor_rollout_ref.rollout.multi_turn.function_tool_path=${FUNCTION_TOOL_CONFIG} \
    \
    trainer.default_local_dir=${OUTPUT_DIR}
```

> `trainer.total_epochs=3` 与全系列统一口径一致（SFT 5 / GRPO 3）：
> RL 阶段数据量小、epoch 多了容易在少数 prompt 上过拟合。

> **checkpoint 落盘位置**：verl 会把权重写到
> `${OUTPUT_DIR}/global_step_<N>/actor`（不是 `${OUTPUT_DIR}/actor`）。

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
    """整合多次搜索的结果

    extract_facts / detect_contradictions 的实现见 §18.4.2。
    """
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
