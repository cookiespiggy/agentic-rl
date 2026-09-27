"""选择性预测（selective prediction）指标：置信度能不能**定位错误**。

**为什么需要单独一套指标**：ECE 只回答「平均而言概率与正确率是否对齐」，它**不回答**
「判错的时候置信度是不是更低」。而后者才是分流/级联能不能做的前提。

一个模型可以 ECE 很好但完全无法定位错误——比如所有样本都输出 0.9，ECE 只反映总体
准确率与 0.9 的差距，但置信度对「这条会不会错」零信息。

本模块提供四个指标：

| 指标 | 回答的问题 |
|---|---|
| `error_detection_auroc` | 把置信度当「判对概率」的排序器，AUC 是多少（0.5 = 完全无信息） |
| `risk_coverage` | 只保留最自信的 x% 流量时，准确率是多少 |
| `aurc` | 风险-覆盖曲线的面积（越小越好，0.5 相当于无信息） |
| `confidence_gap` | 判错样本 vs 判对样本的置信度分布差距 |

工程结论（见 `reports/confidence_diagnosis.json`）：三个分支的 `error_detection_auroc`
都在 0.6 附近，**置信度定位错误的能力很弱**——这正是 11 章的级联实验里
「帕累托前沿只剩 encoder-only 一个点」的原因。
"""

from __future__ import annotations

from statistics import fmean, median

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_COVERAGES = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def error_detection_auroc(confidences: list[float], correct: list[bool]) -> float:
    """把置信度当「这条判对」的打分器，算 AUROC。

    等价于 P(判对样本的置信度 > 判错样本的置信度)。**0.5 表示置信度对错误零信息**，
    这种情况下任何基于置信度的门控都不可能有效。

    只有单一类别（全对或全错）时无定义，返回 `float("nan")`。
    """
    labels = [1 if c else 0 for c in correct]
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, confidences))


def risk_coverage(
    confidences: list[float],
    correct: list[bool],
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> list[dict]:
    """按置信度从高到低保留前 c 比例，报告该子集上的准确率（= 1 - 风险）。

    这是选择性预测的标准视角：**「只服务最自信的那部分流量，质量能提多少」**。
    如果曲线几乎是平的，说明置信度分不出难易，选择性服务没有收益。
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences 与 correct 长度必须一致")
    if not confidences:
        return []

    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    flags = np.array([bool(correct[i]) for i in order], dtype=float)

    out: list[dict] = []
    for coverage in coverages:
        k = max(1, min(len(flags), round(coverage * len(flags))))
        out.append(
            {
                "coverage": round(coverage, 4),
                "n": k,
                "accuracy": round(float(flags[:k].mean()), 4),
                "risk": round(1.0 - float(flags[:k].mean()), 4),
            }
        )
    return out


def aurc(
    confidences: list[float],
    correct: list[bool],
    steps: int = 100,
) -> float:
    """风险-覆盖曲线下的面积（Area Under Risk-Coverage）。

    风险 = 该覆盖下的错误率。**无信息时 AURC ≈ 整体错误率**，越小越好；
    如果 AURC 与整体错误率几乎相等，说明置信度排序没有把难样本排到前面。
    """
    if not confidences:
        return float("nan")
    curve = risk_coverage(
        confidences,
        correct,
        tuple(i / steps for i in range(1, steps + 1)),
    )
    risks = [point["risk"] for point in curve]
    return round(fmean(risks), 4)


def confidence_gap(confidences: list[float], correct: list[bool]) -> dict:
    """判对 vs 判错的置信度分布对比。

    `gap` 是两组均值的差。gap 很小、或者 `error_above_correct_median`（判错样本里
    置信度高于「判对样本中位数」的比例）接近 0.5，都说明置信度无法区分对错。
    """
    ok = [c for c, flag in zip(confidences, correct) if flag]
    bad = [c for c, flag in zip(confidences, correct) if not flag]
    if not ok or not bad:
        return {"n_correct": len(ok), "n_error": len(bad)}

    ok_median = median(ok)
    above = sum(1 for c in bad if c > ok_median) / len(bad)
    return {
        "n_correct": len(ok),
        "n_error": len(bad),
        "correct_mean": round(fmean(ok), 4),
        "error_mean": round(fmean(bad), 4),
        "gap": round(fmean(ok) - fmean(bad), 4),
        "correct_median": round(ok_median, 4),
        "error_median": round(median(bad), 4),
        "error_above_correct_median": round(above, 4),
        "error_max": round(max(bad), 4),
    }


def diagnose(
    confidences: list[float],
    correct: list[bool],
    ece_value: float | None = None,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> dict:
    """一次算齐四个指标，便于各分支横向比较。"""
    result = {
        "n": len(confidences),
        "base_accuracy": round(sum(correct) / len(correct), 4) if correct else float("nan"),
        "error_detection_auroc": round(error_detection_auroc(confidences, correct), 4),
        "aurc": aurc(confidences, correct),
        "risk_coverage": risk_coverage(confidences, correct, coverages),
        "confidence_gap": confidence_gap(confidences, correct),
    }
    if ece_value is not None:
        result["ece_top_confidence"] = round(ece_value, 4)
    # AUROC 与 AURC 的关系：AUROC 越接近 0.5，AURC 越接近 base 错误率
    result["aurc_vs_base_error"] = (
        round(result["aurc"] - (1 - result["base_accuracy"]), 4)
        if correct
        else None
    )
    return result
