"""多种子结果汇总：把各 seed 的指标聚成 mean ± std，并给出差距/波动比值。

n=3 不做显著性检验，只回答一个问题：两条分支的差距是否明显大于种子自身波动。
如果差距小于一个合并标准差，就不能说「A 比 B 好」，只能说「本次运行 A 更高」。

规则基线（rules_v0）是确定性的，std 恒为 0，作为下界参照。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.io_utils import read_json, write_json

BRANCHES = ("encoder", "qwen_lora", "rules")
MODEL_BRANCHES = ("encoder", "qwen_lora")
METRICS = (
    ("intent", "f1_macro"),
    ("intent", "ece_top_confidence"),
    ("needs_escalation", "f1"),
    ("needs_escalation", "brier"),
    ("complexity", "mae"),
)
# 这些指标越小越好，判断优劣时方向相反
LOWER_IS_BETTER = {"intent.ece_top_confidence", "needs_escalation.brier", "complexity.mae"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--out-tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    report: dict = {
        "seeds": seeds,
        "note": (
            "n 很小，不做显著性检验；只看差距是否大于种子波动。"
            "规则基线是确定性的，std 恒为 0。"
            "⚠️ 这是**单次切分**口径：test 只有 24 条模板，分支间细微差距不可信。"
            "主口径请看 reports/cross_validation.json（5 折交叉验证），"
            "单次切分曾给出过与交叉验证方向相反的结论。"
        ),
        "primary_caliber": "reports/cross_validation.json",
        "branches": {},
        "gaps_vs_rules": {},
        "gap_encoder_vs_qwen": {},
    }

    for branch in BRANCHES:
        per_metric: dict = {}
        for group, key in METRICS:
            values = []
            for seed in seeds:
                path = ROOT / "reports" / f"{branch}_metrics_s{seed}.json"
                if path.exists():
                    values.append(float(read_json(path)[group][key]))
            if values:
                per_metric[f"{group}.{key}"] = {
                    "mean": round(statistics.fmean(values), 4),
                    "std": round(statistics.pstdev(values), 4),
                    "n": len(values),
                    "values": [round(v, 4) for v in values],
                }
        report["branches"][branch] = per_metric

    rules = report["branches"].get("rules", {})
    for metric in rules:
        for branch in MODEL_BRANCHES:
            model = report["branches"].get(branch, {}).get(metric)
            if not model:
                continue
            report["gaps_vs_rules"][f"{branch}|{metric}"] = _gap(model, rules[metric], metric)

    for metric in report["branches"].get("encoder", {}):
        enc = report["branches"]["encoder"].get(metric)
        qwen = report["branches"].get("qwen_lora", {}).get(metric)
        if enc and qwen:
            report["gap_encoder_vs_qwen"][metric] = _gap(enc, qwen, metric)

    out = ROOT / "reports" / f"multiseed_report{args.out_tag}.json"
    write_json(out, report)
    print(f"saved {out.relative_to(ROOT)}")
    _print(report)


def _gap(left: dict, right: dict, metric: str) -> dict:
    """left 相对 right 的差距，以及差距是否大于波动。"""
    gap = left["mean"] - right["mean"]
    pooled = math.sqrt((left["std"] ** 2 + right["std"] ** 2) / 2)
    ratio = abs(gap) / pooled if pooled > 0 else float("inf")
    lower_is_better = metric in LOWER_IS_BETTER
    better = "left" if (gap > 0) != lower_is_better else "right"
    return {
        "left_mean": left["mean"],
        "right_mean": right["mean"],
        "gap_left_minus_right": round(gap, 4),
        "pooled_std": round(pooled, 4),
        "gap_over_pooled_std": round(ratio, 2) if ratio != float("inf") else None,
        "separated": ratio > 1.0,
        "better_side": better,
        "lower_is_better": lower_is_better,
    }


def _print(report: dict) -> None:
    print(f"\nseeds = {report['seeds']}")
    header = f"{'metric':34s}{'encoder':>19s}{'qwen_lora':>19s}{'rules':>10s}"
    print(header)
    print("-" * len(header))
    for metric in report["branches"].get("encoder", {}):
        cells = []
        for branch in BRANCHES:
            entry = report["branches"].get(branch, {}).get(metric)
            cells.append(
                f"{entry['mean']:>11.4f}±{entry['std']:<7.4f}" if entry else f"{'-':>19s}"
            )
        print(f"{metric:34s}{''.join(cells)}")

    print("\n模型相对规则基线的提升（gap / 模型自身 std）")
    for key, gap in report["gaps_vs_rules"].items():
        branch, metric = key.split("|")
        verdict = "显著" if gap["separated"] else "未超出波动"
        sign = "+" if gap["gap_left_minus_right"] > 0 else ""
        print(
            f"  {branch:11s} {metric:32s} "
            f"gap={sign}{gap['gap_left_minus_right']:.4f}  "
            f"ratio={gap['gap_over_pooled_std']}  {verdict}"
        )


if __name__ == "__main__":
    main()
