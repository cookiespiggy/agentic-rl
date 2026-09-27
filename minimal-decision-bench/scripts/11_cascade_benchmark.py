"""分流与级联基准：把「三选一」升级成「怎么组合」。

**背景**：这个 bench 的其他脚本评估的都是三条路线各自独立跑。但真实系统是**组合**——
高置信度走便宜的分支，低置信度升级到贵的。这条路一直没测过，而 `FINDINGS.md` 里已经
写下了「先用规则兜住简单流量、模型只处理难的那部分往往更划算」这条**未经验证**的断言。
本脚本把它查清楚。

评估四种策略：

| 策略 | 说明 |
|---|---|
| `*-only` | 单分支基线（encoder / qwen_lora / rules） |
| `cascade-enc2qwen` | encoder 置信度 < t 时升级到 Qwen，扫描 t |
| `rule-first` | 规则命中则用规则，否则走 encoder |
| `oracle-cascade` | **上界**：完美知道 encoder 哪些样本错，只把错的升级给 Qwen |

`oracle-cascade` 是关键参照：如果连「完美门控」都比单分支好不了多少，那任何基于
置信度的门控都不可能有效——结论就是确定的，不必再调阈值。

成本模型（级联的第二级只对**未命中**的样本执行）：

```text
cascade-enc2qwen : L_enc + P(conf < t) × L_qwen
rule-first       : L_rules + P(规则未命中) × L_enc
oracle-cascade   : L_enc + P(encoder 判错) × L_qwen
```

数据来自 `reports/cross_validation.json` 的逐折逐样本预测（**不需要重新训练**）。

输出：`reports/cascade_benchmark.json` + `inference/cascade_v1.md`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.io_utils import write_json
from minimal_decision_bench.metrics import (
    complexity_metrics,
    escalation_metrics,
    intent_metrics,
)

MODEL_BRANCHES = ("encoder", "qwen_lora", "rules")
THRESHOLD_GRID = [round(i / 100, 2) for i in range(101)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--cv-report", default="cross_validation.json")
    p.add_argument("--latency-report", default="latency_benchmark.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cv_path = ROOT / "reports" / args.cv_report
    if not cv_path.exists():
        raise SystemExit(f"缺少 {cv_path.name}，请先跑 scripts/08_cross_validate.py")

    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    pairs = _load_pairs(cv)
    if not pairs:
        raise SystemExit("cross_validation.json 里没有可配对的逐样本记录")

    latency = _load_latency(args.latency_report)
    grid = _grid(args.threshold_step)

    report: dict = {
        "n_samples": len(pairs),
        "folds": cv.get("folds"),
        "latency_ms": latency,
        "threshold_grid": len(grid),
        "baselines": {},
        "cascades": {},
        "conclusion": {},
    }

    print(f"配对了 {len(pairs)} 条样本，延迟口径：{latency}")
    for branch in MODEL_BRANCHES:
        report["baselines"][branch] = _evaluate(
            [pair[branch] for pair in pairs.values()],
            latency[branch],
        )
        b = report["baselines"][branch]
        print(
            f"  {branch:11s} acc={b['intent']['accuracy']:.4f} "
            f"f1={b['intent']['f1_macro']:.4f} 成本={b['cost_ms']:.2f} ms"
        )

    report["cascades"]["cascade-enc2qwen"] = _sweep_enc2qwen(pairs, latency, grid)
    report["cascades"]["rule-first"] = _rule_first(pairs, latency)
    report["cascades"]["oracle-cascade"] = _oracle(pairs, latency)

    for name, entry in report["cascades"].items():
        if "points" in entry:
            best = entry["pareto"]
            print(
                f"  {name:18s} 帕累托点 {len(best)} 个；"
                f"最优质量 {max(p['intent']['f1_macro'] for p in best):.4f}"
            )
        else:
            print(
                f"  {name:18s} acc={entry['intent']['accuracy']:.4f} "
                f"f1={entry['intent']['f1_macro']:.4f} 成本={entry['cost_ms']:.2f} ms"
            )

    report["conclusion"] = _conclude(report)
    out_json = ROOT / "reports" / "cascade_benchmark.json"
    write_json(out_json, report)
    out_md = ROOT / "inference" / "cascade_v1.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"\nsaved {out_json.relative_to(ROOT)}")
    print(f"saved {out_md.relative_to(ROOT)}")
    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))


# --- 数据装配 ---------------------------------------------------------------


def _load_pairs(cv: dict) -> dict[tuple[str, str], dict]:
    """把逐折逐样本预测按 (fold, id) 配对，方便跨分支取同一条样本。"""
    pairs: dict[tuple[str, str], dict] = {}
    for fold, entry in cv.get("results", {}).items():
        for branch in MODEL_BRANCHES:
            branch_entry = entry.get("branches", {}).get(branch)
            if not branch_entry:
                continue
            for record in branch_entry.get("records", []):
                pairs.setdefault((fold, record["id"]), {})[branch] = record
    return {k: v for k, v in pairs.items() if len(v) == len(MODEL_BRANCHES)}


def _load_latency(name: str) -> dict[str, float]:
    path = ROOT / "reports" / name
    if not path.exists():
        raise SystemExit(f"缺少 {name}，请先跑 scripts/06_benchmark_latency.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        branch: float(data["branches"][branch]["end_to_end_serial"]["p50_ms"])
        for branch in MODEL_BRANCHES
    }


def _grid(step: float) -> list[float]:
    if step <= 0:
        raise SystemExit("--threshold-step 必须为正")
    n = round(1.0 / step)
    return [round(i * step, 4) for i in range(n + 1)]


# --- 评估 -------------------------------------------------------------------


def _evaluate(records: list[dict], cost_ms: float) -> dict:
    """给一组逐样本记录算三项指标与成本。"""
    if not records:
        return {}
    return {
        "n": len(records),
        "intent": intent_metrics(
            [r["intent_true"] for r in records],
            [r["intent_pred"] for r in records],
            [r["intent_confidence"] for r in records],
        ),
        "needs_escalation": escalation_metrics(
            [r["escalation_true"] for r in records],
            [r["escalation_pred"] for r in records],
            [r["escalation_prob_true"] for r in records],
        ),
        "complexity": complexity_metrics(
            [r["complexity_true"] for r in records],
            [r["complexity_pred"] for r in records],
        ),
        "cost_ms": round(cost_ms, 4),
    }


def _sweep_enc2qwen(
    pairs: dict, latency: dict[str, float], grid: list[float]
) -> dict:
    """扫描置信度阈值：encoder 置信度 < t 的样本升级给 Qwen。"""
    points: list[dict] = []
    for t in grid:
        chosen: list[dict] = []
        escalated = 0
        for pair in pairs.values():
            enc = pair["encoder"]
            if enc["intent_confidence"] < t:
                chosen.append(pair["qwen_lora"])
                escalated += 1
            else:
                chosen.append(enc)
        rate = escalated / len(pairs)
        point = _evaluate(
            chosen, latency["encoder"] + rate * latency["qwen_lora"]
        )
        point.update({"threshold": t, "escalation_rate": round(rate, 4)})
        points.append(point)
    return {"points": points, "pareto": _pareto(points)}


def _rule_first(pairs: dict, latency: dict[str, float]) -> dict:
    """规则命中则用规则，否则走 encoder。

    规则没有概率语义，「命中」优先看 `rule_trace`（08 脚本会落盘）；若报告来自旧版本
    没有该字段，则退化为置信度判断——规则只有 `CONFIDENCE_MATCHED`(0.90) 与
    `CONFIDENCE_FALLBACK`(0.35) 两个取值，> 0.5 即命中。
    """
    chosen: list[dict] = []
    hit = 0
    for pair in pairs.values():
        rule = pair["rules"]
        trace = rule.get("rule_trace")
        if trace is None:
            matched = rule["intent_confidence"] > 0.5
        else:
            matched = bool(trace) and not any(t.startswith("fallback") for t in trace)
        if matched:
            chosen.append(rule)
            hit += 1
        else:
            chosen.append(pair["encoder"])
    rate = hit / len(pairs)
    result = _evaluate(chosen, latency["rules"] + (1 - rate) * latency["encoder"])
    result["rule_hit_rate"] = round(rate, 4)
    return result


def _oracle(pairs: dict, latency: dict[str, float]) -> dict:
    """上界：完美门控——只把 encoder 判错的样本升级给 Qwen。"""
    chosen: list[dict] = []
    upgraded = 0
    for pair in pairs.values():
        enc = pair["encoder"]
        if enc["intent_pred"] != enc["intent_true"]:
            chosen.append(pair["qwen_lora"])
            upgraded += 1
        else:
            chosen.append(enc)
    rate = upgraded / len(pairs)
    result = _evaluate(chosen, latency["encoder"] + rate * latency["qwen_lora"])
    result["upgrade_rate"] = round(rate, 4)
    result["note"] = "完美门控上界：现实中做不到，用来判断「门控值不值得做」"
    return result


def _pareto(points: list[dict]) -> list[dict]:
    """质量越高越好、成本越低越好的非支配点集。

    再按 (成本, 质量) 去重：一段区间内的阈值可能给出完全相同的级联行为
    （例如 encoder 的最低置信度是 0.6，所有 t ≤ 0.6 的阈值都等价于 encoder-only），
    不去重会让前沿被几十个重复点淹没。
    """
    kept: list[dict] = []
    for point in points:
        dominated = any(
            other["intent"]["f1_macro"] >= point["intent"]["f1_macro"]
            and other["cost_ms"] <= point["cost_ms"]
            and (
                other["intent"]["f1_macro"] > point["intent"]["f1_macro"]
                or other["cost_ms"] < point["cost_ms"]
            )
            for other in points
        )
        if not dominated:
            kept.append(point)

    unique: dict[tuple[float, float], dict] = {}
    for point in sorted(kept, key=lambda p: p["threshold"]):
        key = (round(point["cost_ms"], 4), round(point["intent"]["f1_macro"], 6))
        unique.setdefault(key, point)
    return sorted(unique.values(), key=lambda p: p["cost_ms"])


# --- 结论 -------------------------------------------------------------------


def _conclude(report: dict) -> dict:
    base = report["baselines"]
    enc = base["encoder"]
    qwen = base["qwen_lora"]
    rules = base["rules"]
    oracle = report["cascades"]["oracle-cascade"]
    rule_first = report["cascades"]["rule-first"]
    sweep = report["cascades"]["cascade-enc2qwen"]["points"]

    best_quality = max(p["intent"]["f1_macro"] for p in sweep)
    best_point = max(sweep, key=lambda p: p["intent"]["f1_macro"])
    cheapest_same_quality = min(
        (p for p in sweep if p["intent"]["f1_macro"] >= enc["intent"]["f1_macro"]),
        key=lambda p: p["cost_ms"],
        default=None,
    )

    out = {
        "encoder_only_f1": round(enc["intent"]["f1_macro"], 4),
        "qwen_only_f1": round(qwen["intent"]["f1_macro"], 4),
        "rules_only_f1": round(rules["intent"]["f1_macro"], 4),
        "oracle_cascade_f1": round(oracle["intent"]["f1_macro"], 4),
        "oracle_headroom_over_encoder": round(
            oracle["intent"]["f1_macro"] - enc["intent"]["f1_macro"], 4
        ),
        "oracle_upgrade_rate": oracle["upgrade_rate"],
        "best_gated_f1": round(best_quality, 4),
        "best_gated_threshold": best_point["threshold"],
        "rule_first_f1": round(rule_first["intent"]["f1_macro"], 4),
        "rule_hit_rate": rule_first["rule_hit_rate"],
    }
    if cheapest_same_quality:
        out["cheapest_at_encoder_quality"] = {
            "threshold": cheapest_same_quality["threshold"],
            "cost_ms": cheapest_same_quality["cost_ms"],
            "f1": round(cheapest_same_quality["intent"]["f1_macro"], 4),
        }

    # 判定门控是否值得做
    if out["oracle_headroom_over_encoder"] <= 0.005:
        out["verdict"] = (
            "门控不值得做：**完美门控**相对 encoder-only 的提升都不超过 0.5 个百分点，"
            "说明「encoder 判错的样本恰好是 Qwen 擅长的」这个前提不成立，"
            "任何基于置信度的门控都不可能有效。"
        )
    elif best_quality <= enc["intent"]["f1_macro"]:
        out["verdict"] = (
            "置信度门控不值得做：可实现的门控质量不超过 encoder-only。"
            f"但完美门控还有 {out['oracle_headroom_over_encoder']:.4f} 的空间，"
            "说明 encoder 的置信度不足以定位它自己会错的地方（校准问题，见 31 章）。"
        )
    else:
        out["verdict"] = (
            f"门控有价值：最优阈值 {best_point['threshold']} 下质量 "
            f"{best_quality:.4f}，高于 encoder-only 的 {enc['intent']['f1_macro']:.4f}。"
        )
    return out


def _to_markdown(report: dict) -> str:
    lat = report["latency_ms"]
    lines = [
        "# 分流与级联基准 v1",
        "",
        f"- 样本数：{report['n_samples']}（来自 {report['folds']} 折交叉验证的逐样本预测，**未重新训练**）",
        (
            f"- 延迟口径（端到端 P50）：encoder `{lat['encoder']} ms`　"
            f"qwen_lora `{lat['qwen_lora']} ms`　rules `{lat['rules']} ms`"
        ),
        "",
        "> 级联的成本模型：第二级只对**未命中**的样本执行。",
        "",
        "## 单分支基线",
        "",
        "| 分支 | intent acc | intent F1 | 成本 ms |",
        "|---|---|---|---|",
    ]
    for name, entry in report["baselines"].items():
        lines.append(
            f"| `{name}` | {entry['intent']['accuracy']:.4f} | "
            f"{entry['intent']['f1_macro']:.4f} | {entry['cost_ms']} |"
        )

    oracle = report["cascades"]["oracle-cascade"]
    rule_first = report["cascades"]["rule-first"]
    lines += [
        "",
        "## 级联与分流",
        "",
        "| 策略 | intent acc | intent F1 | 成本 ms | 触发率 |",
        "|---|---|---|---|---|",
        (
            f"| `oracle-cascade`（**上界**） | {oracle['intent']['accuracy']:.4f} | "
            f"{oracle['intent']['f1_macro']:.4f} | {oracle['cost_ms']} | "
            f"升级 {oracle['upgrade_rate']:.4f} |"
        ),
        (
            f"| `rule-first` | {rule_first['intent']['accuracy']:.4f} | "
            f"{rule_first['intent']['f1_macro']:.4f} | {rule_first['cost_ms']} | "
            f"命中 {rule_first['rule_hit_rate']:.4f} |"
        ),
        "",
        "## 置信度门控：帕累托前沿",
        "",
        "| 阈值 | 升级率 | intent F1 | 成本 ms |",
        "|---|---|---|---|",
    ]
    for point in report["cascades"]["cascade-enc2qwen"]["pareto"]:
        lines.append(
            f"| {point['threshold']} | {point['escalation_rate']:.4f} | "
            f"{point['intent']['f1_macro']:.4f} | {point['cost_ms']} |"
        )

    concl = report["conclusion"]
    lines += [
        "",
        "## 结论",
        "",
        (
            f"- encoder-only F1 `{concl['encoder_only_f1']}`　"
            f"Qwen-only F1 `{concl['qwen_only_f1']}`　规则-only F1 `{concl['rules_only_f1']}`"
        ),
        (
            f"- **完美门控上界** F1 `{concl['oracle_cascade_f1']}`"
            f"（相对 encoder-only 只高 `{concl['oracle_headroom_over_encoder']}`，"
            f"升级率 `{concl['oracle_upgrade_rate']}`）"
        ),
        (
            f"- 可实现的置信度门控最优 F1 `{concl['best_gated_f1']}`"
            f"（阈值 `{concl['best_gated_threshold']}`）"
        ),
        f"- `rule-first` F1 `{concl['rule_first_f1']}`（规则命中率 `{concl['rule_hit_rate']}`）",
        "",
        f"**判定：** {concl['verdict']}",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
