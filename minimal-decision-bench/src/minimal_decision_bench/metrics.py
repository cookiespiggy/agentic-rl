from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
)


def intent_metrics(y_true: list[str], y_pred: list[str], top_conf: list[float]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # 小折里某个类可能一次都没被预测到；显式给 0 而不是让 sklearn 抛警告
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ece_top_confidence": ece(y_true, y_pred, top_conf),
        "mean_confidence": float(np.mean(top_conf)),
    }


def escalation_metrics(
    y_true: list[bool], y_pred: list[bool], y_prob_true: list[float]
) -> dict[str, float]:
    """二分类升级任务指标。

    参数 y_prob_true 必须是 P(needs_escalation=True)，即正类概率。
    不能传 top-class 置信度：模型预测「不需要升级」且置信 0.9 时，
    max prob 也是 0.9，会让 Brier 失去意义。
    """
    y_true_i = np.array(y_true, dtype=int)
    y_pred_i = np.array(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true_i, y_pred_i)),
        "f1": float(f1_score(y_true_i, y_pred_i, zero_division=0)),
        "brier": float(brier_score_loss(y_true_i, y_prob_true)),
    }


def complexity_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    return {"mae": float(mean_absolute_error(y_true, y_pred))}


def ece(y_true: list[str], y_pred: list[str], conf: list[float], bins: int = 10) -> float:
    true = np.array(y_true)
    pred = np.array(y_pred)
    c = np.array(conf)
    corr = (true == pred).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    val = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (c >= lo) & (c < hi if i < bins - 1 else c <= hi)
        if mask.any():
            val += np.abs(corr[mask].mean() - c[mask].mean()) * mask.mean()
    return float(val)

