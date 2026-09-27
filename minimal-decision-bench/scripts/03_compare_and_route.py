"""三方对比：encoder-only vs Qwen LoRA vs 关键词规则基线。

规则基线（rules_v0）在这里生成——它不需要模型，只需要数据与 schema。
对应主仓库 29 章：「模型到底比简单规则好多少？」这个问题必须有答案，
否则整条论证链缺少下界。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench import rules
from minimal_decision_bench.data_builder import full_pool_complexity_floor
from minimal_decision_bench.io_utils import read_json, read_jsonl, write_json
from minimal_decision_bench.metrics import (
    complexity_metrics,
    escalation_metrics,
    intent_metrics,
)
from minimal_decision_bench.routing import route_by_confidence
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.trainers import branch_paths, prediction_records

BRANCHES = ("encoder", "qwen_lora", "rules")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default="", help="读取/写出带后缀的报告，用于多种子运行")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(ROOT / "schemas" / "v1.json")
    routing = schema.routing
    labels = list(schema.intent_labels())
    label_to_id = {k: i for i, k in enumerate(labels)}
    id_to_label = {i: k for k, i in label_to_id.items()}

    test_rows = _encode(read_jsonl(ROOT / "data" / "test.jsonl"), label_to_id)
    _write_rules_baseline(test_rows, labels, id_to_label, args.run_tag)

    metrics = {
        "encoder": read_json(branch_paths(ROOT, "encoder", args.run_tag)[1]),
        "qwen_lora": read_json(branch_paths(ROOT, "qwen_lora", args.run_tag)[1]),
        "rules": read_json(ROOT / "reports" / f"rules_metrics{args.run_tag}.json"),
    }

    summary = {
        "intent_f1_macro": {b: metrics[b]["intent"]["f1_macro"] for b in BRANCHES},
        "intent_ece": {b: metrics[b]["intent"]["ece_top_confidence"] for b in BRANCHES},
        "escalation_f1": {b: metrics[b]["needs_escalation"]["f1"] for b in BRANCHES},
        "complexity_mae": {b: metrics[b]["complexity"]["mae"] for b in BRANCHES},
    }

    # 路由样例一律来自落盘的逐样本预测，不使用硬编码数字。
    route_examples: list[dict] = []
    missing: list[str] = []
    reason_counts: dict[str, dict[str, int]] = {}
    for branch in BRANCHES:
        path = branch_paths(ROOT, branch, args.run_tag)[2]
        if not path.exists():
            missing.append(path.name)
            continue
        records = read_json(path)
        reason_counts[branch] = _route_reason_counts(records, routing)
        for name, record in _pick_examples(records, routing):
            decision = route_by_confidence(
                intent_confidence=record["intent_confidence"],
                complexity=record["complexity_pred"],
                escalation_prob=record["escalation_prob_true"],
                cheap_threshold=float(routing["cheap_threshold"]),
                mid_threshold=float(routing["mid_threshold"]),
            )
            # name 与 reason 必须一致（_pick_examples 已按 reason 分组保证）
            assert name == decision.reason, (name, decision.reason)
            route_examples.append(
                {
                    "branch": branch,
                    "name": name,
                    "text": record["text"],
                    "intent_confidence": round(record["intent_confidence"], 4),
                    "complexity": round(record["complexity_pred"], 4),
                    "escalation_prob": round(record["escalation_prob_true"], 4),
                    "tier": decision.tier,
                    "reason": decision.reason,
                }
            )

    if missing:
        print(
            "[warn] 缺少预测文件 "
            + ", ".join(missing)
            + "；对应分支不产出路由样例（请先跑 01/02 脚本）。"
        )

    report = {
        "comparison": summary,
        "branches": list(BRANCHES),
        "run_tag": args.run_tag,
        # complexity MAE 要对齐下限而不是 0：标签含不可学成分。
        # 下限从全模板池算（数据集属性），不随本报告的切分口径变化。
        "complexity_floor": full_pool_complexity_floor(),
        "routing_examples_source": (
            "model_predictions" if not missing else "model_predictions(partial)"
        ),
        "routing_examples": route_examples,
        # 各档位的样本数：用来判断某个 reason 是否根本没有样本（此时不产样例）
        "routing_reason_counts": reason_counts,
    }
    out_path = ROOT / "reports" / f"comparison_report{args.run_tag}.json"
    write_json(out_path, report)
    print(f"saved {out_path.relative_to(ROOT)}")
    _print_summary(summary)


def _write_rules_baseline(
    rows: list[dict],
    labels: list[str],
    id_to_label: dict[int, str],
    run_tag: str = "",
) -> None:
    """生成规则基线的指标与逐样本预测。"""
    intent_out = rules.infer_rows(rows, "intent_id", intent_labels=labels)
    esc_out = rules.infer_escalation(rows)
    comp_out = rules.infer_complexity(rows)

    metrics = {
        "branch": "rules-v0",
        "note": "关键词规则基线，置信度为未校准固定值，ECE 必然很差",
        "intent": intent_metrics(
            [id_to_label[i] for i in intent_out["labels"]],
            [id_to_label[i] for i in intent_out["preds"]],
            intent_out["confidences"],
        ),
        "needs_escalation": escalation_metrics(
            [bool(v) for v in esc_out["labels"]],
            [bool(v) for v in esc_out["preds"]],
            [float(p[1]) for p in esc_out["probs"]],
        ),
        "complexity": complexity_metrics(comp_out["labels"], comp_out["preds"]),
    }
    write_json(ROOT / "reports" / f"rules_metrics{run_tag}.json", metrics)
    records = prediction_records(rows, intent_out, esc_out, comp_out, id_to_label)
    for record, trace in zip(records, intent_out["traces"], strict=True):
        record["rule_trace"] = trace
    write_json(ROOT / "reports" / f"rules_predictions{run_tag}.json", records)
    print(f"saved reports/rules_metrics{run_tag}.json（规则基线）")


def _pick_examples(records: list[dict], routing: dict) -> list[tuple[str, dict]]:
    """按**真实路由结果**分组挑选代表样本。

    关键约束：返回的 name 必须等于 `route_by_confidence` 给出的 `reason`。
    早期版本按「置信度最接近中档」等启发式挑样本，挑出来的可能实际路由成
    `premium`，于是出现 `name=medium_conf` 但 `reason=high_risk` 的自相矛盾——
    读者会以为这两个字段在描述同一件事。

    某一类路由在测试集里没有样本时直接跳过，不硬凑。
    """
    if not records:
        return []
    cheap_threshold = float(routing["cheap_threshold"])
    mid_threshold = float(routing["mid_threshold"])

    by_reason: dict[str, list[dict]] = {}
    for record in records:
        decision = route_by_confidence(
            intent_confidence=record["intent_confidence"],
            complexity=record["complexity_pred"],
            escalation_prob=record["escalation_prob_true"],
            cheap_threshold=cheap_threshold,
            mid_threshold=mid_threshold,
        )
        by_reason.setdefault(decision.reason, []).append(record)

    picked: list[tuple[str, dict]] = []
    for reason in ("high_conf_low_risk", "medium_conf", "high_risk"):
        pool = by_reason.get(reason)
        if not pool:
            continue
        # 取组内置信度中位数附近的一条作为代表
        confidences = sorted(r["intent_confidence"] for r in pool)
        median = confidences[len(confidences) // 2]
        best = min(pool, key=lambda r: abs(r["intent_confidence"] - median))
        picked.append((reason, best))
    return picked


def _route_reason_counts(records: list[dict], routing: dict) -> dict[str, int]:
    """各路由档位的样本数——用来判断某一档是否根本没有样本。"""
    counts: dict[str, int] = {}
    for record in records:
        decision = route_by_confidence(
            intent_confidence=record["intent_confidence"],
            complexity=record["complexity_pred"],
            escalation_prob=record["escalation_prob_true"],
            cheap_threshold=float(routing["cheap_threshold"]),
            mid_threshold=float(routing["mid_threshold"]),
        )
        counts[decision.reason] = counts.get(decision.reason, 0) + 1
    return counts


def _encode(rows: list[dict], label_to_id: dict[str, int]) -> list[dict]:
    out = []
    for r in rows:
        labels = r["labels"].copy()
        labels["intent_id"] = label_to_id[labels["intent"]]
        labels["needs_escalation_id"] = int(labels["needs_escalation"])
        out.append({"id": r["id"], "text": r["text"], "labels": labels})
    return out


def _print_summary(summary: dict) -> None:
    print(f"\n{'指标':<22}{'encoder':>12}{'qwen_lora':>12}{'rules':>12}")
    for key, values in summary.items():
        row = "".join(f"{values[b]:>12.4f}" for b in BRANCHES)
        print(f"{key:<22}{row}")


if __name__ == "__main__":
    main()
