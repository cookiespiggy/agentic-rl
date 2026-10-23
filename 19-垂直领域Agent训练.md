# 19 - 垂直领域 Agent 训练

> **学习目标**：学会为特定垂直领域训练专业化的 Agent，掌握从数据构建到 RL 训练的完整实战流程，实现小模型超越大模型。

---

## 19.1 为什么垂直领域是机会

NVIDIA Research（2025）指出：小语言模型（SLM，Small Language Model）在 Agent 场景中比 LLM 更适合，因为 Agent 执行的是**少量专业化任务**。

核心论点:
- 大模型（70B+）：泛化能力强但推理成本高
- 小模型（3B-14B）+ Agentic RL：在特定领域深度优化，成本低，部署灵活

---

## 19.2 垂直领域训练方法论

### 19.2.1 四步法

```
Step 1: 领域数据收集（Domain Data Collection）
  → 收集领域内的真实任务
  → 标注高质量的 Agent 交互数据

Step 2: 领域 SFT（Domain-Specific SFT）
  → 用领域数据微调基座模型
  → 教会模型领域术语和工具调用格式
  → 学习率 5e-5、5 个 epoch（全系列统一口径，见 §19.2.1）

Step 3: 领域 RL（Domain RL Training）
  → 在领域环境中用 GRPO 训练
  → 通过领域特定的奖励函数引导学习
  → 学习率 1e-6、3 个 epoch（全系列统一口径）

Step 4: 评估与迭代（Evaluation & Iteration）
  → 在领域 Benchmark 上评估
  → 分析失败案例，改进数据和奖励
```

> **学习率 / epoch 的统一口径**（本系列各章一致）：SFT 用 **5e-5**（数据量大时可下调到 **1e-5**）、**5 个 epoch**；GRPO 用 **1e-6**、**3 个 epoch**。
> RL 阶段的学习率比 SFT 低一到两个数量级是正常的——
> GRPO 的梯度信号来自组内相对优势，方差比 SFT 的监督损失大得多。

### 19.2.2 领域选择建议

适合小模型 + Agentic RL 的垂直领域:

| 领域 | 适合原因 | 典型任务 | 奖励可验证性 |
|------|---------|---------|-------------|
| 客服 Agent | 任务明确，可验证 | 查询订单、退换货 | 中（规则匹配） |
| 数据分析 Agent | SQL 可自动验证 | 数据查询、报表生成 | **高**（SQL 结果对比） |
| 医疗问答 Agent | 知识密集型 | 症状分析、用药建议 | 低（需专家评估） |
| 金融分析 Agent | 计算密集型 | 财报分析、风险评估 | 高（数值验证） |
| 代码审查 Agent | 测试可验证 | Bug 检测、代码优化 | **高**（测试用例） |

> **建议**：优先选择**奖励可自动验证**的领域，这类场景最契合 RLVR 范式，训练效果最好。

### 19.2.3 本章数据的来源

> **本章不提供数据生成脚本。** 客服工单、企业 SQL schema、订单库这些**无法离线获取**，
> 仓库里也没有现成数据。下面给的是**契约**：你必须自己按这个格式准备好数据，放到
> `data/customer_service_train.jsonl` / `data/customer_service_test.jsonl`（客服领域）和
> `data/data_analysis_train.jsonl` / `data/data_analysis_test.jsonl`（数据分析领域）。
>
> 数据通常来自三处：
> 1. **真实业务脱敏导出**——客服工单、订单表、销售库（这是最正的做法，但要注意合规）；
> 2. **大模型蒸馏**——让 Teacher 模型按你的格式产出 `messages`
>    （脚本见 §19.6.2，`gpt-4` 换成任意可用模型）；
> 3. **半自动构造**——用规则脚本拼出「用户问题 → 工具调用 → 工具返回 → 回复」的模板。

#### 19.2.3.1 基础字段（两个领域通用）

`.jsonl` 每行一条样本：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `messages` | `list[dict]` | ✅ | 对话历史，角色按 `system → user → assistant → tool → assistant` 交替 |
| `data_source` | `str` | ✅ | 领域标记，例如 `customer_service`、`data_analysis` |
| `agent_name` | `str` | ✅ | Agent 角色名，例如 `cs_agent`、`data_analyst` |

#### 19.2.3.2 奖励函数需要的字段（**每个字段都要有**）

两个奖励函数都会从 **kwargs 里的数据集列**读 ground truth（统一契约见
[14 - 奖励函数设计与实现](./14-奖励函数设计与实现.md) §14.1，
本章的约定见 §19.3.2 / §19.4.2）。**下面这张表就是「造数据时必须填的清单」**：

| 字段 | 类型 | 被谁读 | 含义 | 缺失会怎样 |
|------|------|--------|------|-----------|
| `expected_answer` | `str` | `customer_service_reward` → `check_problem_solved` | 客服回复里应该出现的**标准答案句** | 该样本永远拿不到问题解决分（+1.0） |
| `required_keys` | `list[str]` | `check_problem_solved` | 回复中**必须包含**的关键信息（订单号、金额、日期…），按 70%/90% 命中率判分 | 退化成「只要出现标准答案句就算对」，判分变松 |
| `key_facts` | `list[str]` | `customer_service_reward` → `has_accurate_info` | 回复里应当**准确出现**的事实片段，80% 命中算准确 | `has_accurate_info` 恒返回 True，信息准确性这 +0.1 变成白送 |
| `expected_tools` | `list[str]` | `customer_service_reward` | 期望被调用的**工具名**序列，如 `["query_order", "process_refund"]` | 工具正确性 +0.1 拿不到；多调的工具还会被扣分 |
| `solvable` | `bool` | `customer_service_reward` | 该请求**能否自助解决**。`true` 时调用 `transfer_to_human` 要扣 -0.2 | 退化为「转人工永远不罚」，学到的策略会乱转人工 |
| `expected_sql_results` | `list[dict]` | `data_analysis_reward` | 期望的 SQL 结果，每项形如 `{"rows": [...], "columns": [...], "row_count": N}` | SQL 正确性完全拿不到分，奖励只剩格式分 |

#### 19.2.3.3 样例记录

**客服领域**（`data/customer_service_train.jsonl` 的一行）：

```python
customer_service_train_record = {
    # ---- 基础字段 ----
    "messages": [
        {"role": "system", "content": "你是XX商城的智能客服助手。……"},
        {"role": "user", "content": "我的订单 ORD-20260520-088 到哪了？已经等了一周了。"},
        {"role": "assistant", "content": "<think>用户问物流状态，先查订单。</think>\n\n"
                                        "{\"name\": \"query_order\", \"arguments\": {\"order_id\": \"ORD-20260520-088\"}}"},
        {"role": "tool", "content": "{\"status\": \"shipped\", \"estimated_delivery\": \"2026-05-27\"}"},
        {"role": "assistant", "content": "您的订单已发货，预计 5 月 27 日送达，快递单号 SF1234567890。"},
    ],
    "data_source": "customer_service",
    "agent_name": "cs_agent",
    # ---- 奖励函数需要的字段（一个都不能少）----
    "expected_answer": "已发货",
    "required_keys": ["ORD-20260520-088", "SF1234567890", "2026-05-27"],
    "key_facts": ["已发货", "5月27日", "SF1234567890"],
    "expected_tools": ["query_order"],
    "solvable": True,
}
```

**数据分析领域**（`data/data_analysis_train.jsonl` 的一行）：

```python
data_analysis_train_record = {
    "messages": [
        {"role": "system", "content": "你是一个数据分析助手。……"},
        {"role": "user", "content": "帮我分析一下各地区上个月的销售情况"},
        {"role": "assistant", "content": "<think>按地区汇总上月销售额。</think>\n\n"
                                        "{\"name\": \"run_sql\", \"arguments\": {\"query\": \"SELECT region, SUM(amount) FROM sales GROUP BY region\"}}"},
        {"role": "tool", "content": "{\"success\": true, \"columns\": [\"region\", \"total_sales\"], \"rows\": [...]}"},
        {"role": "assistant", "content": "华东地区销售额最高，占比 37.7%。"},
    ],
    "data_source": "data_analysis",
    "agent_name": "data_analyst",
    # ---- 奖励函数需要的字段 ----
    "expected_sql_results": [
        {
            "rows": [
                {"region": "华东", "total_sales": 580000},
                {"region": "华南", "total_sales": 420000},
                {"region": "华北", "total_sales": 350000},
                {"region": "西部", "total_sales": 180000},
            ],
            "columns": ["region", "total_sales"],
            "row_count": 4,
        }
    ],
    "expected_tools": ["run_sql"],
    "solvable": True,
}
```

> `expected_sql_results` 的 `rows` 要和 `DataAnalysisEnvironment.run_sql()` 真正返回的
> `rows` **逐字段相等**才算「精确匹配」——生成期望结果时，最稳的做法是拿同一个
> SQLite 文件跑一遍你写的标准 SQL，把返回值原样存进去。

---

## 19.3 案例一：客服 Agent 训练

### 19.3.1 环境与工具设计

```python
"""客服 Agent 的工具和环境定义"""
from typing import Optional


class ToolCallLog:
    """模块级工具调用日志。

    TRL 不会给奖励函数传 tool_logs 参数，所以工具集在记录调用时写进这里，
    奖励函数（§19.3.2）算完一条就取走。
    生产环境请改用 verl 的 AgentLoop，它会把轨迹挂在样本的 extra_info 上。
    """

    def __init__(self):
        self._logs: list = []

    def reset(self):
        self._logs = []

    def record(self, entry: dict):
        self._logs.append(entry)

    def take(self) -> list:
        logs, self._logs = self._logs, []
        return logs


# 模块级单例：工具集负责写，奖励函数负责读
TOOL_LOG = ToolCallLog()


class CustomerServiceTools:
    """客服 Agent 的工具集"""
    
    def __init__(self, order_db, knowledge_base):
        self.order_db = order_db
        self.knowledge_base = knowledge_base
        self.call_log = []  # 记录工具调用
    
    def _log(self, entry: dict):
        """同时写入实例日志和模块级日志（后者供奖励函数读）"""
        self.call_log.append(entry)
        TOOL_LOG.record(entry)
    
    def query_order(self, order_id: str) -> dict:
        """查询订单信息
        
        Args:
            order_id: 订单编号，如 "ORD-20260101-001"
        """
        self._log({"tool": "query_order", "order_id": order_id})
        order = self.order_db.get(order_id)
        if order:
            return {
                "status": order["status"],       # pending/shipped/delivered
                "estimated_delivery": order.get("eta"),
                "tracking_number": order.get("tracking"),
                "items": order["items"],
            }
        return {"error": f"Order {order_id} not found"}
    
    def process_refund(self, order_id: str, reason: str) -> dict:
        """处理退款
        
        Args:
            order_id: 订单编号
            reason: 退款原因
        """
        self._log({"tool": "process_refund", "order_id": order_id})
        order = self.order_db.get(order_id)
        if not order:
            return {"error": "Order not found"}
        if order["status"] != "delivered":
            return {"error": "Can only refund delivered orders"}
        return {
            "refund_id": f"REF-{order_id}",
            "amount": order["total"],
            "status": "processing",
            "estimated_days": 3,
        }
    
    def search_faq(self, question: str) -> str:
        """搜索常见问题知识库
        
        Args:
            question: 用户问题
        """
        self._log({"tool": "search_faq", "question": question})
        # 简单的关键词匹配（实际项目中可用向量检索）
        best_match = None
        best_score = 0
        for faq in self.knowledge_base:
            score = sum(1 for kw in faq["keywords"] if kw in question.lower())
            if score > best_score:
                best_score = score
                best_match = faq
        if best_match and best_score > 0:
            return best_match["answer"]
        return "未找到相关信息，建议转人工客服。"
    
    def transfer_to_human(self, reason: str) -> dict:
        """转接人工客服
        
        Args:
            reason: 转接原因
        """
        self._log({"tool": "transfer_to_human", "reason": reason})
        return {
            "status": "transferred",
            "queue_position": 3,
            "estimated_wait": "5 minutes",
        }
    
    def reset(self):
        self.call_log = []
```

### 19.3.2 奖励函数

```python
"""客服 Agent 奖励函数

统一契约（与 [14 - 奖励函数设计与实现](./14-奖励函数设计与实现.md) §14.1 一致，
本章两个案例沿用同一套）：

    reward_func(prompts, completions, **kwargs) -> list[float]

- prompts:     list[list[dict]]，数据集里的原始对话
- completions: list[list[dict]]（对话模式）或 list[str]——先用 _coerce_to_text 转文本
- kwargs:      数据集其余**列名**。本章约定：
                 expected_answer / required_keys / key_facts /
                 expected_tools / solvable   → ground truth，从 kwargs 取
                 tool_logs                     → 工具调用日志（可选）
- ground truth 一律从 kwargs 取，**不作为位置参数**。
- `tool_logs=` / `sandbox=` / `envs=` 这些**不是 TRL 的参数**：
  要么做成一列数据集字段（kwargs），要么通过模块级对象传入。
  本章的客服工具集自带 call_log（见 §19.3.1），所以用模块级对象更自然。
"""


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



# ToolCallLog / TOOL_LOG 的定义见 §19.3.1（工具集那一节），这里直接复用。


def customer_service_reward(prompts, completions, **kwargs) -> list[float]:
    """客服 Agent 的多组件奖励函数

    奖励组成：
    - 问题解决: 问题被正确解决 +1.0
    - 工具使用正确性: 调用了正确的工具
    - 客户满意度: 回复礼貌、信息准确
    - 不必要转人工惩罚: 可以自行解决却转人工 -0.2
    - 效率奖励: 用更少轮次解决

    Args:
        prompts: list[list[dict]]，原始对话
        completions: list[list[dict]] | list[str]
        kwargs: 数据集其余列，见上方说明
    """
    gts = kwargs.get("ground_truths")
    if gts is None:
        # 逐样本的 ground truth 字段散在多个列里，TRL 会按列名分别传进来
        gts = [
            {
                "expected_answer": kwargs.get("expected_answer", [None] * len(completions))[i],
                "required_keys": kwargs.get("required_keys", [None] * len(completions))[i],
                "key_facts": kwargs.get("key_facts", [None] * len(completions))[i],
                "expected_tools": kwargs.get("expected_tools", [None] * len(completions))[i],
                "solvable": kwargs.get("solvable", [None] * len(completions))[i],
            }
            for i in range(len(completions))
        ]

    tool_logs = kwargs.get("tool_logs")
    if tool_logs is None:
        tool_logs = [TOOL_LOG.take() for _ in completions]

    rewards = []

    for i, completion in enumerate(completions):
        completion = _coerce_to_text(completion)
        gt = gts[i] if i < len(gts) and gts[i] else {}
        reward = 0.0
        log = tool_logs[i] if i < len(tool_logs) else None
        
        # === 1. 问题解决（核心信号）===
        if check_problem_solved(completion, gt, log):
            reward += 1.0
        
        # === 2. 工具使用正确性 ===
        if log:
            expected_tools = gt.get("expected_tools") or []
            actual_tools = [call["tool"] for call in log]
            
            # 正确工具调用
            for tool in expected_tools:
                if tool in actual_tools:
                    reward += 0.1
            
            # 不必要的工具调用
            unnecessary = set(actual_tools) - set(expected_tools)
            reward -= 0.05 * len(unnecessary)
        
        # === 3. 信息准确性 ===
        if has_accurate_info(completion, gt):
            reward += 0.1
        
        # === 4. 不必要转人工惩罚 ===
        if log and "transfer_to_human" in [c["tool"] for c in log]:
            if gt.get("solvable", True):  # 本可以自行解决
                reward -= 0.2
        
        # === 5. 格式奖励 ===
        if "<think>" in completion and "</think>" in completion:
            reward += 0.05
        
        rewards.append(max(reward, -0.5))
    
    return rewards


def check_problem_solved(completion: str, gt: dict, log: list = None) -> bool:
    """检查客户问题是否被正确解决（关键信息匹配）"""
    expected_answer = gt.get("expected_answer", "")
    required_keys = gt.get("required_keys", [])
    if not expected_answer:
        return False
    # 方式 1: 完整答案匹配
    if expected_answer.lower() in completion.lower():
        # 如果有必要的关键信息，进一步验证
        if required_keys:
            match_count = sum(1 for k in required_keys if k.lower() in completion.lower())
            return match_count / len(required_keys) >= 0.7
        return True
    # 方式 2: 关键信息匹配（即使完整答案未出现）
    if required_keys:
        match_count = sum(1 for k in required_keys if k.lower() in completion.lower())
        return match_count / len(required_keys) >= 0.9
    return False


def has_accurate_info(completion: str, gt: dict) -> bool:
    """检查回复中的信息是否准确"""
    facts = gt.get("key_facts", [])
    if not facts:
        return True
    match_count = sum(1 for f in facts if f.lower() in completion.lower())
    return match_count / len(facts) >= 0.8
```

> 这个奖励同样是**累加式**的（问题解决 +1.0、每个正确工具 +0.1、信息准确 +0.1、
> 格式 +0.05），**没有 1.0 上界**——多工具场景能到 1.25 左右。
> 它与 §19.4.2 的数据分析奖励一样，**不可与其他章的归一化奖励横向比较**。
> 本函数需要的字段清单见 §19.2.3.2。

### 19.3.3 SFT 数据示例

```python
customer_service_sft_sample = {
    "messages": [
        {
            "role": "system",
            "content": """你是XX商城的智能客服助手。

## 可用工具
1. query_order(order_id) - 查询订单状态和物流信息
2. process_refund(order_id, reason) - 处理退款（仅限已签收订单）
3. search_faq(question) - 搜索常见问题知识库
4. transfer_to_human(reason) - 转接人工客服

## 工作原则
- 优先自己解决用户问题
- 只有在确实无法解决时才转人工
- 回复要简洁、准确、有礼貌
- 涉及退款操作前，先确认订单状态"""
        },
        {"role": "user", "content": "我的订单 ORD-20260520-088 到哪了？已经等了一周了。"},
        {
            "role": "assistant",
            "content": """<think>
用户询问订单 ORD-20260520-088 的物流状态。我需要先查询这个订单的信息。
</think>

我来帮您查询订单状态。

{"name": "query_order", "arguments": {"order_id": "ORD-20260520-088"}}"""
        },
        {"role": "tool", "content": "{\"status\": \"shipped\", \"estimated_delivery\": \"2026-05-27\", \"tracking_number\": \"SF1234567890\", \"items\": [\"蓝牙耳机 x1\"]}"},
        {
            "role": "assistant",
            "content": """<think>
订单状态为 shipped（已发货），预计送达日期是 2026-05-27，快递单号 SF1234567890。
当前日期是 2026-05-25，预计后天到达。
</think>

您好！已为您查询到订单 ORD-20260520-088 的信息：

- **商品**：蓝牙耳机 x1
- **状态**：已发货，运输中
- **预计送达**：5月27日（后天）
- **快递单号**：SF1234567890

您的订单正在路上，预计后天到达。如有其他问题请随时联系我！"""
        }
    ],
    "data_source": "customer_service",
    "agent_name": "cs_agent"
}
```

---

## 19.4 案例二：数据分析 Agent

### 19.4.1 环境与工具设计

```python
"""数据分析 Agent 的工具和环境"""
import sqlite3
import json


class _EnvHolder:
    """模块级环境持有者。

    TRL 不会给奖励函数传 envs= 参数，所以环境实例放在这里，
    奖励函数（§19.4.2）再取出来跑 SQL。
    生产环境请改用 verl 的 AgentLoop，它会把轨迹挂在样本的 extra_info 上。
    """

    def __init__(self):
        self._env = None

    def set(self, env):
        self._env = env

    def get(self):
        return self._env


SQL_ENV = _EnvHolder()


class DataAnalysisEnvironment:
    """数据分析 Agent 的 RL 训练环境
    
    提供 SQL 查询、图表生成和报告导出功能。
    SQL 查询结果可以自动验证，完美契合 RLVR 范式。
    """
    
    def __init__(self, db_path: str, schema_info: str):
        self.db_path = db_path
        self.schema_info = schema_info
        self.query_log = []
        self._conn = None  # 连接池缓存
        SQL_ENV.set(self)  # 注册给奖励函数用（见 §19.4.2）
    
    def run_sql(self, query: str) -> dict:
        """执行 SQL 查询
        
        Args:
            query: SQL 查询语句
        """
        self.query_log.append(query)
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(query)
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            return {
                "success": True,
                "columns": columns,
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "rows": [], "row_count": 0}
    
    def plot_chart(self, chart_type: str, data: dict, title: str = "") -> dict:
        """生成图表（模拟）
        
        Args:
            chart_type: 图表类型（bar/line/pie）
            data: 图表数据 {"labels": [...], "values": [...]}
            title: 图表标题
        """
        return {
            "chart_id": f"chart_{len(self.query_log)}",
            "type": chart_type,
            "title": title,
            "status": "generated",
        }
    
    def export_report(self, content: str, format: str = "markdown") -> dict:
        """导出分析报告
        
        Args:
            content: 报告内容
            format: 输出格式（markdown/csv）
        """
        return {
            "report_id": f"report_{len(self.query_log)}",
            "format": format,
            "word_count": len(content),
            "status": "exported",
        }
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取或创建数据库连接（连接池复用）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def get_schema(self) -> str:
        """获取数据库 schema 信息"""
        return self.schema_info
    
    def reset(self):
        self.query_log = []
    
    def close(self):
        """关闭数据库连接"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
```

### 19.4.2 SQL 验证奖励函数

```python
"""数据分析 Agent 奖励函数 —— SQL 自动验证

签名沿用本章的统一契约（见 §19.3.2 的说明与
[14 - 奖励函数设计与实现](./14-奖励函数设计与实现.md) §14.1）：

    reward_func(prompts, completions, **kwargs) -> list[float]

- ground truth 从 kwargs 取（本章用 `expected_sql_results` 列）；
- `envs=` **不是 TRL 的参数** → 本章用模块级 `SQL_ENV` 传入（见 §19.4.1）。

> **与 ch14 的 `sql_agent_reward` 的关系**：ch14 那版是**单条**打分
> （`sql_agent_reward(completion, expected_result, db_connection)`，
> 直接拿 db 连接跑一条 SQL），适合讲清楚「SQL 可以自动验证」这件事；
> 本章这版是**训练器级**的——签名单一化、按 `completions` 批量打分、
> 环境通过模块级对象传入、并且支持一条回复里写多条 SQL。
> 两者的判断逻辑（跑 SQL → 和期望结果比对）是一致的。
"""


def data_analysis_reward(prompts, completions, **kwargs) -> list[float]:
    """数据分析 Agent 奖励函数
    
    核心优势：SQL 查询结果可以**自动验证**！
    执行 SQL → 对比期望结果 → 自动计算奖励
    这完美契合 RLVR（可验证奖励的强化学习）范式。
    
    奖励组成：
    - SQL 正确性: 查询结果与期望结果匹配
    - 分析深度: 多次查询获取不同维度的数据
    - 报告质量: 最终报告包含数据支撑
    - 格式规范: 正确使用工具调用格式

    Args:
        prompts: list[list[dict]]，原始对话
        completions: list[list[dict]] | list[str]
        kwargs: 数据集其余列。本章约定：
            expected_sql_results -> list[list[dict]]，期望的 SQL 结果
    """
    rewards = []
    
    expected_results_all = kwargs.get("expected_sql_results") or [[] for _ in completions]
    
    for i, completion in enumerate(completions):
        completion = _coerce_to_text(completion)
        reward = 0.0
        env = SQL_ENV.get()
        
        # === 1. SQL 正确性（核心，可自动验证）===
        sql_queries = extract_sql_queries(completion)
        expected_results = (
            expected_results_all[i] if i < len(expected_results_all) else []
        )
        
        if sql_queries and env:
            for j, query in enumerate(sql_queries):
                if j < len(expected_results):
                    actual = env.run_sql(query)
                    expected = expected_results[j]
                    
                    if actual["success"]:
                        # 精确匹配
                        if actual["rows"] == expected.get("rows", []):
                            reward += 0.5
                        # 列名匹配（部分正确）
                        elif actual["columns"] == expected.get("columns", []):
                            reward += 0.2
                        # 行数匹配
                        elif actual["row_count"] == expected.get("row_count", -1):
                            reward += 0.1
        elif sql_queries:
            # 无环境时，至少奖励写出了 SQL
            reward += 0.1 * len(sql_queries)
        
        # === 2. 分析深度 ===
        if len(sql_queries) >= 2:
            reward += 0.1  # 多维度分析
        if any("GROUP BY" in q.upper() for q in sql_queries):
            reward += 0.05  # 使用了聚合分析
        
        # === 3. 报告质量 ===
        if has_data_insights(completion):
            reward += 0.1
        
        # === 4. 格式奖励 ===
        if "<think>" in completion and "</think>" in completion:
            reward += 0.05
        
        rewards.append(max(reward, -0.5))
    
    return rewards


def extract_sql_queries(text: str) -> list:
    """从 Agent 回复中提取所有 SQL 查询"""
    import re
    queries = []
    
    # 方式 1: 从工具调用中提取
    matches = re.findall(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    for m in matches:
        query = m.replace("\\n", " ").replace('\\"', '"')
        if any(kw in query.upper() for kw in ["SELECT", "INSERT", "UPDATE"]):
            queries.append(query)
    
    # 方式 2: 从 SQL 代码块中提取
    matches = re.findall(r'```sql\s*(.*?)```', text, re.DOTALL)
    queries.extend(matches)
    
    return queries


def has_data_insights(text: str) -> bool:
    """检查回复是否包含数据洞察"""
    insight_keywords = [
        "趋势", "增长", "下降", "占比", "同比", "环比",
        "最高", "最低", "平均", "总计", "分析表明",
    ]
    return sum(1 for kw in insight_keywords if kw in text) >= 2
```

> **这个奖励没有 1.0 上界。** 它按**命中查询数累加**：每条 SQL 精确匹配 +0.5
> （列名匹配 +0.2、行数匹配 +0.1），再加上多维分析 +0.1、聚合分析 +0.05、
> 报告质量 +0.1、格式 +0.05。所以一条写了 3 条正确 SQL 的回复能拿到 1.6 以上。
>
> **它和其他章的归一化奖励不可横向比较**——别的章的奖励基本落在 [0,1]，
> 这里不是。**本奖励的语义就这样，不做归一化**；GRPO 只用同一 prompt 组内的
> 相对大小，绝对值不影响训练，但**跨章比较 reward 曲线是错的**。

### 19.4.3 SFT 数据示例

```python
data_analysis_sft_sample = {
    "messages": [
        {
            "role": "system",
            "content": """你是一个数据分析助手。你可以查询公司数据库并生成分析报告。

## 数据库 Schema
- sales(id, product, region, amount, date)
- customers(id, name, tier, signup_date)

## 可用工具
1. run_sql(query) - 执行 SQL 查询
2. plot_chart(chart_type, data, title) - 生成图表
3. export_report(content, format) - 导出报告

## 工作流程
1. 理解用户的分析需求
2. 编写 SQL 查询获取数据
3. 分析数据并总结洞察
4. 如有需要，生成图表或报告"""
        },
        {"role": "user", "content": "帮我分析一下各地区上个月的销售情况"},
        {
            "role": "assistant",
            "content": """<think>
用户想要分析各地区上月销售情况。
我需要：
1. 按地区汇总上月销售额
2. 找出销售额最高和最低的地区
先查询汇总数据。
</think>

我来查询各地区的销售数据。

{"name": "run_sql", "arguments": {"query": "SELECT region, SUM(amount) as total_sales, COUNT(*) as order_count FROM sales WHERE date >= '2026-05-01' AND date < '2026-06-01' GROUP BY region ORDER BY total_sales DESC"}}"""
        },
        {"role": "tool", "content": "{\"success\": true, \"columns\": [\"region\", \"total_sales\", \"order_count\"], \"rows\": [{\"region\": \"华东\", \"total_sales\": 580000, \"order_count\": 1200}, {\"region\": \"华南\", \"total_sales\": 420000, \"order_count\": 980}, {\"region\": \"华北\", \"total_sales\": 350000, \"order_count\": 760}, {\"region\": \"西部\", \"total_sales\": 180000, \"order_count\": 420}]}"},
        {
            "role": "assistant",
            "content": """</think>
## 上月（2026年5月）各地区销售分析

| 地区 | 销售额 | 订单数 | 占比 |
|------|--------|--------|------|
| 华东 | ¥580,000 | 1,200 | 37.7% |
| 华南 | ¥420,000 | 980 | 27.3% |
| 华北 | ¥350,000 | 760 | 22.7% |
| 西部 | ¥180,000 | 420 | 11.7% |

**关键洞察：**
1. **华东地区**销售额最高（¥58万），占总销售额的 37.7%
2. **西部地区**销售额最低，但有增长潜力
3. 各地区客单价相近（约 ¥430-480），说明定价策略较为统一

{"name": "plot_chart", "arguments": {"chart_type": "bar", "data": {"labels": ["华东", "华南", "华北", "西部"], "values": [580000, 420000, 350000, 180000]}, "title": "2026年5月各地区销售额"}}"""
        }
    ],
    "data_source": "data_analysis",
    "agent_name": "data_analyst"
}
```

---

## 19.5 通用训练脚本

```bash
#!/bin/bash
# run_domain_agent.sh —— 垂直领域 Agent 通用训练脚本
#
# ⚠️ verl 训练需要 CUDA 环境；本仓库没有 CI 在跑这段脚本，下面的键只做过离线核对。
#
# --- verl 必填项（这 4 个键务必显式写出）---
#   actor_rollout_ref.actor.optim.lr=1e-6
#       该键在不同 verl 版本的有效默认值不同（v0.9.1 是 1e-6，更早版本是 1e-3），
#       所以不要依赖默认值，必须显式写。
#   data.prompt_key=messages
#       数据集里对话列的列名；字段契约见 §19.2.3。
#   data.train_batch_size=32
#   actor_rollout_ref.rollout.n=8
#       verl 的整除约束：train_batch_size × rollout.n 须被 n_gpus 整除
#       （verl/utils/config.py:123-124），这里 32 × 8 = 256，8 卡时每卡 32 条。
#       ⚠️ 这和 TRL 的规则**不同**：TRL 要求 generation_batch_size
#       （= per_device_train_batch_size × num_processes × steps_per_generation）
#       能被 num_generations 整除。两套规则不要混用。

set -e

# === 配置区 ===
DOMAIN="${1:-customer_service}"  # 默认客服领域
VALID_DOMAINS=("customer_service" "data_analysis" "custom")
if [[ ! " ${VALID_DOMAINS[*]} " =~ " ${DOMAIN} " ]]; then
    echo "错误: 不支持的领域 '${DOMAIN}'。支持: ${VALID_DOMAINS[*]}"
    exit 1
fi

MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN_DATA="data/${DOMAIN}_train.jsonl"
TEST_DATA="data/${DOMAIN}_test.jsonl"
OUTPUT_DIR="checkpoints/${DOMAIN}_agent_v1"

echo "===== 训练 ${DOMAIN} Agent ====="
echo "模型: ${MODEL_PATH}"
echo "训练数据: ${TRAIN_DATA}"

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
    trainer.project_name=${DOMAIN}_agent_rl \
    trainer.experiment_name=qwen2_5_3b_${DOMAIN} \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "${DOMAIN} Agent 训练完成！模型保存在 ${OUTPUT_DIR}"
```

> `trainer.total_epochs=3`：RL 阶段是小数据集 + 少量 epoch 更好——**epoch 过多容易在
> 少数 prompt 上过拟合**，奖励曲线看着还在涨但泛化已经掉了。全系列统一口径是
> **SFT 5 / GRPO 3**。
>
> **checkpoint 落盘位置**：`${OUTPUT_DIR}/global_step_<N>/actor`（不是 `${OUTPUT_DIR}/actor`）。

---

## 19.6 小模型超越大模型的策略

### 19.6.1 蒸馏 + RL 策略

```
Phase 1: Teacher Model (GPT-4, 70B) 生成高质量 Agent 轨迹
Phase 2: Student Model (7B) 通过 SFT 学习 Teacher 的行为
Phase 3: Student Model 通过 RL 在领域环境中进一步优化
Phase 4: Student Model 在特定任务上超越 Teacher
```

### 19.6.2 蒸馏数据生成脚本

```python
"""用大模型（Teacher）生成垂直领域 SFT 数据"""
import json
import re
from openai import OpenAI


def generate_domain_sft_data(
    teacher_model: str = "gpt-4",
    domain: str = "customer_service",
    task_descriptions: list = None,
    num_samples_per_task: int = 5,
    output_path: str = "data/domain_sft_distilled.jsonl",
):
    """用大模型生成垂直领域 Agent 交互数据
    
    Args:
        teacher_model: 教师模型名称
        domain: 领域名称
        task_descriptions: 任务描述列表
        num_samples_per_task: 每个任务生成的样本数
        output_path: 输出文件路径
    """
    client = OpenAI()
    
    system_prompt = f"""你是一个 {domain} 领域的 Agent 数据生成专家。
请根据任务描述，生成高质量的 Agent 多轮交互数据。

要求：
1. 包含正确的工具调用格式（JSON 格式）
2. 包含 <think>...</think> 思考过程
3. 对话自然、信息准确
4. 输出为 JSON 格式的 messages 数组"""
    
    all_samples = []
    
    for task_desc in task_descriptions:
        for _ in range(num_samples_per_task):
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"为以下任务生成一条 Agent 交互数据：\n{task_desc}"},
                ],
                temperature=0.8,
            )
            
            try:
                content = response.choices[0].message.content
                # 移除可能的 Markdown 代码块标记
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                sample = json.loads(content)
                sample["data_source"] = domain
                sample["agent_name"] = f"{domain}_agent"
                all_samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    # 保存
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"已生成 {len(all_samples)} 条蒸馏数据，保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    tasks = [
        "用户询问订单 ORD-001 的物流状态，订单已发货",
        "用户要求退款，订单已签收，符合退款条件",
        "用户询问退换货政策",
        "用户投诉商品质量问题，需要转人工",
    ]
    generate_domain_sft_data(
        task_descriptions=tasks,
        num_samples_per_task=10,
        output_path="data/cs_sft_distilled.jsonl",
    )
```

### 19.6.3 关键成功因素

1. **高质量的领域数据**（Quality over Quantity）：100 条高质量 > 10000 条低质量
2. **精确的奖励函数**：避免 Reward Hacking，确保奖励信号与真实能力对齐
3. **充分的训练迭代**：RL 阶段按全系列统一口径跑 **3 个 epoch**（SFT 阶段 5 个）；
   epoch 不是越多越好，小数据集上 epoch 多了容易过拟合，要看验证集曲线而不是训练 reward
4. **持续分析和改进失败案例**：每轮训练后分析 bad case，改进数据/奖励

---

## 19.7 术语速查

| 英文术语 | 中文注释 |
|---------|----------|
| Domain-Specific Agent | 领域专用 Agent |
| Distillation | 蒸馏（大模型知识迁移到小模型） |
| Domain Data Collection | 领域数据收集 |
| Customer Service Agent | 客服 Agent |
| Data Analysis Agent | 数据分析 Agent |
| Teacher Model / Student Model | 教师模型 / 学生模型 |
| SQL Verification | SQL 自动验证（可验证奖励的典型应用） |

---

## 19.8 小结

- 垂直领域是小模型超越大模型的最佳赛道
- 蒸馏 + RL 是核心训练方法：大模型生成数据 → 小模型 SFT → RL 优化
- 优先选择奖励可自动验证的领域（如 SQL、代码），最契合 RLVR 范式
- 下一步：评估方法 → [20 - 评估与Benchmark](./20-评估与Benchmark.md)
