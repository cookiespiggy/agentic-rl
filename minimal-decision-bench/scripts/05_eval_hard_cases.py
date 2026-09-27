"""难例评测：消费 data/hard_cases.jsonl，产出错误剖面。

对应主仓库 28 章「标注协议与难例覆盖」与 29 章「规则基线与错误剖面」。
hard_cases 是人工设计的边界样本（多诉求混合、语义模糊），用来暴露
整体指标掩盖掉的失败模式——尤其是「该升级却判成不升级」这类业务高危错误。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench import rules
from minimal_decision_bench.io_utils import read_jsonl, write_json
from minimal_decision_bench.routing import route_by_confidence
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.trainers import (
    DEFAULT_MAX_LENGTH,
    branch_paths,
    infer_task,
)

BRANCHES = ("encoder", "qwen_lora", "rules")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(ROOT / "schemas" / "v1.json")
    labels = list(schema.intent_labels())
    label_to_id = {k: i for i, k in enumerate(labels)}
    id_to_label = {i: k for k, i in label_to_id.items()}

    rows = read_jsonl(ROOT / "data" / "hard_cases.jsonl")
    enc = _encode(rows, label_to_id)

    report: dict = {"n_cases": len(enc), "run_tag": args.run_tag, "branches": {}}
    for branch in BRANCHES:
        if branch == "rules":
            intent_out = rules.infer_rows(enc, "intent_id", intent_labels=labels)
            esc_out = rules.infer_escalation(enc)
            comp_out = rules.infer_complexity(enc)
        else:
            artifact_dir = branch_paths(ROOT, branch, args.run_tag)[0]
            if not (artifact_dir / "intent").exists():
                print(f"[warn] 缺少 {artifact_dir}/intent，跳过 {branch}（请先跑 01/02 脚本）")
                continue
            intent_out = infer_task(
                enc, str(artifact_dir / "intent"), "intent_id", False,
                num_labels=len(labels), max_length=DEFAULT_MAX_LENGTH,
            )
            esc_out = infer_task(
                enc, str(artifact_dir / "escalation"), "needs_escalation_id", False,
                num_labels=2, max_length=DEFAULT_MAX_LENGTH,
            )
            comp_out = infer_task(
                enc, str(artifact_dir / "complexity"), "complexity", True,
                num_labels=1, max_length=DEFAULT_MAX_LENGTH,
            )
        report["branches"][branch] = _profile(
            enc, intent_out, esc_out, comp_out, id_to_label, schema.routing
        )

    out = ROOT / "reports" / f"hard_cases_report{args.run_tag}.json"
    write_json(out, report)
    print(f"saved {out.relative_to(ROOT)}")
    _print_report(report)


def _profile(
    enc: list[dict],
    intent_out: dict,
    esc_out: dict,
    comp_out: dict,
    id_to_label: dict[int, str],
    routing: dict,
) -> dict:
    cases: list[dict] = []
    for i, row in enumerate(enc):
        intent_true = id_to_label[row["labels"]["intent_id"]]
        intent_pred = id_to_label[intent_out["preds"][i]]
        esc_true = bool(row["labels"]["needs_escalation_id"])
        esc_pred = bool(esc_out["preds"][i])
        esc_prob = float(esc_out["probs"][i][1])
        comp_true = float(row["labels"]["complexity"])
        comp_pred = float(comp_out["preds"][i])
        decision = route_by_confidence(
            intent_confidence=float(intent_out["confidences"][i]),
            complexity=comp_pred,
            escalation_prob=esc_prob,
            cheap_threshold=float(routing["cheap_threshold"]),
            mid_threshold=float(routing["mid_threshold"]),
        )
        cases.append(
            {
                "id": row["id"],
                "text": row["text"],
                "category": row.get("category", "unknown"),
                "intent_true": intent_true,
                "intent_pred": intent_pred,
                "intent_ok": intent_true == intent_pred,
                "intent_confidence": round(float(intent_out["confidences"][i]), 4),
                "escalation_true": esc_true,
                "escalation_pred": esc_pred,
                "escalation_prob_true": round(esc_prob, 4),
                "escalation_ok": esc_true == esc_pred,
                "complexity_true": comp_true,
                "complexity_pred": round(comp_pred, 4),
                "complexity_abs_error": round(abs(comp_true - comp_pred), 4),
                "routing_tier": decision.tier,
                "routing_reason": decision.reason,
            }
        )

    n = len(cases)
    summary = {
        "intent_accuracy": round(sum(c["intent_ok"] for c in cases) / n, 4),
        "escalation_accuracy": round(sum(c["escalation_ok"] for c in cases) / n, 4),
        "intent_misclassified": [
            {
                "text": c["text"],
                "true": c["intent_true"],
                "pred": c["intent_pred"],
                "confidence": c["intent_confidence"],
            }
            for c in cases
            if not c["intent_ok"]
        ],
        # 漏报升级（该升级却判成不升级）是业务上最危险的错误
        "escalation_false_negative": [
            {"text": c["text"], "prob_true": c["escalation_prob_true"]}
            for c in cases
            if c["escalation_true"] and not c["escalation_pred"]
        ],
        "escalation_false_positive": [
            {"text": c["text"], "prob_true": c["escalation_prob_true"]}
            for c in cases
            if not c["escalation_true"] and c["escalation_pred"]
        ],
        "complexity_mae": round(
            sum(c["complexity_abs_error"] for c in cases) / n, 4
        ),
        "complexity_note": (
            "难例的复杂度标签是人工判断，而训练标签由文本哈希推导——模型学的是随机数，"
            "所以难例上的 complexity MAE 不构成能力证据"
        ),
        "tier_distribution": _counter(c["routing_tier"] for c in cases),
        "by_category": _by_category(cases),
    }
    return {"cases": cases, "summary": summary}


def _by_category(cases: list[dict]) -> dict:
    """按失败模式拆开统计——整体准确率会掩盖「哪一类难例在崩」。"""
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["category"], []).append(case)

    out: dict = {}
    for category, rows in sorted(groups.items()):
        size = len(rows)
        out[category] = {
            "n": size,
            "intent_accuracy": round(sum(r["intent_ok"] for r in rows) / size, 4),
            "escalation_accuracy": round(sum(r["escalation_ok"] for r in rows) / size, 4),
            "escalation_missed": sum(
                1 for r in rows if r["escalation_true"] and not r["escalation_pred"]
            ),
        }
    return out


def _counter(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _print_report(report: dict) -> None:
    for branch, data in report["branches"].items():
        s = data["summary"]
        print(f"\n=== {branch} ===")
        print(
            f"  intent acc={s['intent_accuracy']:.4f}  "
            f"escalation acc={s.get('escalation_accuracy', float('nan')):.4f}  "
            f"tier={s['tier_distribution']}"
        )
        print(
            f"  错分 {len(s['intent_misclassified'])} 条 / "
            f"漏报升级 {len(s['escalation_false_negative'])} 条 / "
            f"误报升级 {len(s['escalation_false_positive'])} 条"
        )
        by_cat = s.get("by_category") or {}
        if by_cat:
            print("  按失败模式：")
            for category, entry in by_cat.items():
                print(
                    f"    {category:16s} n={entry['n']:2d} "
                    f"intent={entry['intent_accuracy']:.2f} "
                    f"escalation={entry['escalation_accuracy']:.2f} "
                    f"漏报升级={entry['escalation_missed']}"
                )
        for c in data["cases"]:
            flag = "OK " if c["intent_ok"] else "ERR"
            print(
                f"  [{flag}] {c['category']:14s} {c['text'][:24]:26s} "
                f"intent {c['intent_true']}->{c['intent_pred']} "
                f"esc {c['escalation_true']!s:5s}->{c['escalation_pred']!s:5s}"
                f"({c['escalation_prob_true']:.2f}) tier={c['routing_tier']}"
            )


def _encode(rows: list[dict], label_to_id: dict[str, int]) -> list[dict]:
    out = []
    for r in rows:
        labels = r["labels"].copy()
        labels["intent_id"] = label_to_id[labels["intent"]]
        labels["needs_escalation_id"] = int(labels["needs_escalation"])
        out.append(
            {
                "id": r["id"],
                "text": r["text"],
                "category": r.get("category", "unknown"),
                "labels": labels,
            }
        )
    return out


if __name__ == "__main__":
    main()
