# 27 - 任务建模：把业务流程压成可训练 Schema

> **学习目标**：把“业务口头需求”转成机器可训练、可评估、可部署的 schema 规范。

---

## 27.1 任务建模的核心原则

垂直领域判别模型 的成败，70% 取决于 schema 是否建模正确。  
这里有三个硬原则：

1. **单字段单职责**：一个字段只回答一个问题
2. **标签可执行**：每个选项都能直接映射到系统动作
3. **边界可判定**：标注员能稳定区分相邻类别

---

## 27.2 从业务流程到字段设计

以“企业客服分流”举例，业务方常说：

- “先判断是不是账单问题”
- “再看复杂度”
- “低把握要升级人工”

把口头流程翻译成 schema：

| 业务问题 | 字段 | 类型 | 输出示例 |
|---|---|---|---|
| 属于哪个队列 | `intent` | `choice` | `billing/support/sales/general` |
| 处理难度多大 | `complexity` | `score` | `0.0~1.0` |
| 是否要升级 | `needs_escalation` | `bool` | `true/false` |

---

## 27.3 v0 Schema 模板（可直接改）

```json
{
  "schema_id": "your-domain-router",
  "version": "0.1.0",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "把请求路由到最匹配队列",
      "criteria": {
        "billing": "扣费、退款、发票、账单",
        "support": "故障、报错、功能异常",
        "sales": "报价、套餐、采购",
        "general": "以上都不属于"
      }
    },
    "complexity": {
      "type": "score",
      "instructions": "评估任务复杂度（0低-1高）",
      "criteria": ["low", "medium", "high"]
    },
    "needs_escalation": {
      "type": "bool",
      "instructions": "该请求是否应升级到更强模型或人工复核"
    }
  }
}
```

---

## 27.4 标签设计的反模式（必须避开）

### 反模式 1：把动作和语义混在一个标签里

坏例子：`billing_need_human_now`  
问题：语义（billing）和策略（need_human_now）耦合，后续策略难改。

### 反模式 2：标签太细但样本太少

如果你有 20 个 intent 标签，但每类只有几十条数据，模型必然不稳。

### 反模式 3：标签不可执行

标签命中后，系统还要“再猜”下一步动作，说明 schema 设计失败。

---

## 27.5 本章交付物（必须产出）

在你的仓库里创建（或按此结构记录）：

```text
schema/
└── v0.json
```

并满足以下验收：

- 至少 1 个 `choice` 字段
- 至少 1 个 `score/bool` 字段
- 每个 `choice` 选项都有业务解释
- 标签命中后可直接映射动作

---

## 对应工程锚点（minimal-decision-bench）

本章不是抽象概念，直接对应到：

- Schema 文件：[`minimal-decision-bench/schemas/v1.json`](./minimal-decision-bench/schemas/v1.json)
- Schema 校验：[`minimal-decision-bench/src/minimal_decision_bench/schema.py`](./minimal-decision-bench/src/minimal_decision_bench/schema.py)
- 单测验证：[`minimal-decision-bench/tests/test_schema_and_routing.py`](./minimal-decision-bench/tests/test_schema_and_routing.py)

建议你做 1 次“读完即验证”：

```bash
cd minimal-decision-bench
uv run pytest -q tests/test_schema_and_routing.py
```

---

## 27.6 本章小结

- schema 是 垂直领域判别模型 的“任务合同”
- 合同写不好，后面的训练、评估、路由都会漂
- 下一章进入数据协议：如何保证标注一致性和难例覆盖

下一章：**[28 - 数据工程 I：标注协议与难例覆盖](./28-数据工程I-标注协议与难例覆盖.md)**。
