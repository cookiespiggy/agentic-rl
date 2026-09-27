"""关键词规则基线（rules_v0）。

对应主仓库 29 章：在训练任何模型前，先建立可解释、可回放、可对比的规则基线。
它的意义不是替代模型，而是提供**最小可用下限**——回答「模型到底比简单规则好多少」。

设计约束：

1. 关键词从 `schemas/v1.json` 的 `intent.criteria` 反推，不依赖训练数据；
2. 每条决策带 `trace`，记录命中了哪些规则（可解释、可回放）；
3. 置信度是**未校准的固定值**，不是概率。这正是规则基线在 ECE 上必然很差的原因，
   也是 31 章「让置信度真正可用」要解决的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 从 schema criteria 展开：billing=扣费/退款/发票/账单，support=报错/故障/不可用/接口异常，
# sales=报价/套餐/采购/升级咨询。
#
# 关键词范围刻意定在「规范词 + 明显形态变体」，**不含重度口语枚举**。
# 理由：把「多收了钱」「便宜点」「怎么卖」这类转述全部枚举进去，规则会退化成一张
# 训练集抄写表，那测的就不是「规则 vs 模型」而是「枚举耐心」。真实的规则基线就是
# 在显式样本上很强、在需要消歧与语义泛化的样本上崩掉——这正是本工程要暴露的。
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "扣费", "扣款", "扣走", "退款", "退钱", "发票", "开票", "账单", "对账",
        "续费", "优惠券", "已支付", "抬头", "收款", "多收", "消费明细",
        "划走", "余额", "收据", "凭证",
    ),
    "support": (
        "报错", "错误", "故障", "失败", "不可用", "异常", "超时", "白屏",
        "卡住", "卡在", "卡死", "打不开", "用不了", "不能用", "加载不出来",
        "error", "500", "timeout", "崩溃", "挂了", "没反应", "没效果",
        "收不到", "不一致", "很慢", "转圈", "延迟", "进不去", "点不动",
        "看不到", "不好使",
    ),
    "sales": (
        "报价", "询价", "套餐", "采购", "试用", "定制", "折扣", "优惠",
        "部署", "年付", "月付", "版本", "续约", "企业版", "团队版", "小团队",
        "价格", "价位", "多少钱", "预算", "方案", "成本", "便宜", "商务",
        "合作", "花多少",
    ),
}

# 优先级：多诉求句按「直接经济损失 > 功能不可用 > 商务动作 > 一般咨询」取最高。
INTENT_PRIORITY: tuple[str, ...] = ("billing", "support", "sales", "general")

ESCALATION_KEYWORDS: tuple[str, ...] = (
    "影响生产", "线上", "生产环境", "紧急", "尽快", "全部", "所有", "整个",
)

_COMPLEXITY_HIGH_MARKERS: tuple[str, ...] = ("影响生产", "线上", "紧急", "尽快", "全部", "所有")
_COMPLEXITY_LOW_MARKERS: tuple[str, ...] = ("怎么", "在哪里", "如何", "能不能", "有没有", "吗？")

# 规则没有概率语义，只能给固定值。这两个常数刻意分开，便于 ECE 暴露问题。
CONFIDENCE_MATCHED = 0.90
CONFIDENCE_FALLBACK = 0.35


@dataclass(frozen=True)
class RuleDecision:
    intent: str
    confidence: float
    complexity: float
    needs_escalation: bool
    trace: tuple[str, ...]


def decide(text: str, intent_labels: list[str] | None = None) -> RuleDecision:
    """对单条文本给出规则决策。"""
    labels = list(intent_labels) if intent_labels else list(INTENT_PRIORITY)
    hits = {intent: [kw for kw in kws if kw in text] for intent, kws in INTENT_KEYWORDS.items()}
    matched = [intent for intent in INTENT_PRIORITY if hits.get(intent)]

    if matched:
        # 按优先级取第一个命中的类别；同时记录其它命中项，便于回放多诉求句
        intent = matched[0]
        trace = tuple(f"{name}:{'/'.join(hits[name])}" for name in matched)
        confidence = CONFIDENCE_MATCHED
    else:
        intent = "general" if "general" in labels else labels[-1]
        trace = ("fallback:general",)
        confidence = CONFIDENCE_FALLBACK

    complexity = rule_complexity(text)
    escalation = any(kw in text for kw in ESCALATION_KEYWORDS)
    if escalation:
        trace = (*trace, "escalation:keyword")
    return RuleDecision(
        intent=intent,
        confidence=confidence,
        complexity=complexity,
        needs_escalation=escalation,
        trace=trace,
    )


def rule_complexity(text: str) -> float:
    """规则版复杂度打分：高扰动词加分，低扰动词减分，基线 0.5。"""
    score = 0.5
    if any(kw in text for kw in _COMPLEXITY_HIGH_MARKERS):
        score += 0.25
    if any(kw in text for kw in _COMPLEXITY_LOW_MARKERS):
        score -= 0.15
    return round(min(0.99, max(0.01, score)), 4)


def infer_rows(
    rows: list[dict],
    label_key: str = "intent_id",
    intent_labels: list[str] | None = None,
) -> dict[str, Any]:
    """与 trainers.infer_task 同构的返回，方便规则基线走同一套指标与报告代码。

    intent_labels 必须与训练脚本用的 schema 标签顺序一致，否则 id 会错位。
    """
    intent_labels = list(intent_labels) if intent_labels else _intent_labels_from(rows, label_key)
    label_to_id = {name: i for i, name in enumerate(intent_labels)}

    labels: list[Any] = []
    preds: list[Any] = []
    confidences: list[float] = []
    prob_vectors: list[list[float]] = []
    traces: list[list[str]] = []

    n = len(intent_labels)
    for row in rows:
        decision = decide(row["text"], intent_labels)
        labels.append(row["labels"][label_key])
        preds.append(label_to_id.get(decision.intent, 0))
        confidences.append(decision.confidence)
        rest = (1.0 - decision.confidence) / max(1, n - 1)
        prob_vectors.append(
            [decision.confidence if i == preds[-1] else rest for i in range(n)]
        )
        traces.append(list(decision.trace))

    return {
        "labels": labels,
        "preds": preds,
        "confidences": confidences,
        "probs": prob_vectors,
        "traces": traces,
    }


def infer_escalation(rows: list[dict]) -> dict[str, Any]:
    """升级判定的规则分支，返回与 infer_task 二分类同构的结构。"""
    labels: list[Any] = []
    preds: list[Any] = []
    confidences: list[float] = []
    prob_vectors: list[list[float]] = []
    for row in rows:
        decision = decide(row["text"])
        positive = decision.needs_escalation
        labels.append(row["labels"]["needs_escalation_id"])
        preds.append(int(positive))
        prob = CONFIDENCE_MATCHED if positive else 1.0 - CONFIDENCE_MATCHED
        confidences.append(max(prob, 1.0 - prob))
        prob_vectors.append([1.0 - prob, prob])
    return {
        "labels": labels,
        "preds": preds,
        "confidences": confidences,
        "probs": prob_vectors,
    }


def infer_complexity(rows: list[dict]) -> dict[str, Any]:
    labels: list[Any] = []
    preds: list[Any] = []
    confidences: list[float] = []
    prob_vectors: list[list[float]] = []
    for row in rows:
        value = rule_complexity(row["text"])
        labels.append(row["labels"]["complexity"])
        preds.append(value)
        confidences.append(value)
        prob_vectors.append([value])
    return {
        "labels": labels,
        "preds": preds,
        "confidences": confidences,
        "probs": prob_vectors,
    }


def _intent_labels_from(rows: list[dict], label_key: str) -> list[str]:
    """从行数据的 intent 字段恢复标签顺序（与 schema 顺序一致）。"""
    seen: list[str] = []
    for row in rows:
        name = row["labels"].get("intent")
        if name is not None and name not in seen:
            seen.append(name)
    if seen:
        # 按优先级排序，保证与 INTENT_PRIORITY 一致
        ordered = [name for name in INTENT_PRIORITY if name in seen]
        ordered += [name for name in seen if name not in ordered]
        return ordered
    return list(INTENT_PRIORITY)
