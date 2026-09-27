"""置信度诊断：**置信度能不能定位错误？**

补上 `FINDINGS.md` 与 31 章里点名的那个缺失指标。ECE 只回答「平均而言概率与正确率
是否对齐」，不回答「判错时置信度是不是更低」——而后者才是分流/级联能不能做的前提。

脚本对三个分支各算：

- `error_detection_auroc`：把置信度当「判对概率」的排序器，AUC 是多少（0.5 = 零信息）
- `aurc`：风险-覆盖曲线面积（与整体错误率几乎相等 ⇒ 排序没把难样本排前面）
- `risk_coverage`：只服务最自信的 x% 时准确率多少（含相对基线的增益）
- `confidence_gap`：判对 vs 判错的置信度分布差距

并按难度层拆开，最后给出「ECE 与 AUROC 的排序是否一致」——用来证明**ECE 不是充分
验收标准**。

数据来自 `reports/cross_validation.json` 的逐样本预测（**不需要重新训练**）。

输出：`reports/confidence_diagnosis.json` + `inference/confidence_diagnosis_v1.md`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.io_utils import write_json
from minimal_decision_bench.selective import diagnose

BRANCHES = ("encoder", "qwen_lora", "rules")
LAYERS = ("explicit", "conflict", "synonym")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cv-report", default="cross_validation.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cv_path = ROOT / "reports" / args.cv_report
    if not cv_path.exists():
        raise SystemExit(f"缺少 {cv_path.name}，请先跑 scripts/08_cross_validate.py")

    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    records = _pooled_records(cv)
    if not records:
        raise SystemExit("cross_validation.json 里没有逐样本记录")

    report: dict = {
        "n_samples": len(records["encoder"]),
        "note": (
            "ECE 衡量「平均校准」，本报告衡量「能否定位错误」。"
            "两者不是一回事——一个模型可以 ECE 很好却完全无法定位错误。"
        ),
        "branches": {},
        "per_layer": {},
        "verdict": {},
    }

    print(f"样本数 {report['n_samples']}")
    for branch in BRANCHES:
        rows = records[branch]
        confidences = [r["intent_confidence"] for r in rows]
        correct = [r["intent_pred"] == r["intent_true"] for r in rows]
        ece_value = _ece(cv, branch)
        report["branches"][branch] = diagnose(confidences, correct, ece_value)
        entry = report["branches"][branch]
        print(
            f"  {branch:11s} acc={entry['base_accuracy']:.4f} "
            f"ECE={entry.get('ece_top_confidence')} "
            f"AUROC={entry['error_detection_auroc']:.4f} AURC={entry['aurc']:.4f}"
        )

    for branch in ("encoder", "qwen_lora"):
        report["per_layer"][branch] = {}
        for layer in LAYERS:
            rows = [r for r in records[branch] if r.get("layer") == layer]
            if not rows:
                continue
            confidences = [r["intent_confidence"] for r in rows]
            correct = [r["intent_pred"] == r["intent_true"] for r in rows]
            report["per_layer"][branch][layer] = diagnose(confidences, correct)

    report["verdict"] = _verdict(report)
    out_json = ROOT / "reports" / "confidence_diagnosis.json"
    write_json(out_json, report)
    out_md = ROOT / "inference" / "confidence_diagnosis_v1.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"\nsaved {out_json.relative_to(ROOT)}")
    print(f"saved {out_md.relative_to(ROOT)}")
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))


def _pooled_records(cv: dict) -> dict[str, list[dict]]:
    pooled: dict[str, list[dict]] = {branch: [] for branch in BRANCHES}
    for entry in cv.get("results", {}).values():
        for branch in BRANCHES:
            branch_entry = entry.get("branches", {}).get(branch)
            if branch_entry:
                pooled[branch].extend(branch_entry.get("records", []))
    return pooled


def _ece(cv: dict, branch: str) -> float | None:
    summary = cv.get("summary", {}).get("branches", {}).get(branch, {})
    return summary.get("pooled", {}).get("intent", {}).get("ece_top_confidence")


def _verdict(report: dict) -> dict:
    branches = report["branches"]
    encoder = branches["encoder"]
    inf = float("inf")

    # ECE 更好的分支，AUROC 是否也更好？排序不一致就说明 ECE 不是充分标准。
    ece_rank = sorted(
        BRANCHES, key=lambda b: branches[b].get("ece_top_confidence", inf)
    )
    auroc_rank = sorted(BRANCHES, key=lambda b: -branches[b]["error_detection_auroc"])

    def gain_at(branch: str, coverage: float) -> float | None:
        for point in branches[branch]["risk_coverage"]:
            if abs(point["coverage"] - coverage) < 1e-9:
                return round(point["accuracy"] - branches[branch]["base_accuracy"], 4)
        return None

    auroc = encoder["error_detection_auroc"]
    if auroc <= 0.55:
        strength = "几乎为零"
    elif auroc <= 0.70:
        strength = "较弱"
    elif auroc <= 0.85:
        strength = "中等"
    else:
        strength = "较强"

    base_error = round(1 - encoder["base_accuracy"], 4)
    gain_90 = gain_at("encoder", 0.9)
    agree = ece_rank == auroc_rank

    parts = [
        (
            f"encoder 的置信度对错误的定位能力是**{strength}**"
            f"（AUROC {auroc}，0.5 = 零信息）；"
            f"AURC {encoder['aurc']} 低于整体错误率 {base_error}"
            f"（差 {encoder['aurc_vs_base_error']:+}），"
            f"说明只服务最自信的 90% 流量时准确率能提 {gain_90:+}。"
        )
    ]
    if agree:
        parts.append("ECE 与 AUROC 的排序一致，本数据无法据此否定 ECE 的代表性。")
    else:
        parts.append(
            f"**但 ECE 与 AUROC 排序不一致**：ECE 最好的是 `{ece_rank[0]}`"
            f"（{branches[ece_rank[0]].get('ece_top_confidence')}），"
            f"而定位能力最强的是 `{auroc_rank[0]}`"
            f"（AUROC {branches[auroc_rank[0]]['error_detection_auroc']}）——"
            "**校准好不等于能定位错误，ECE 不是充分验收标准。**"
        )
    parts.append(
        "规则的 AUROC 只有 "
        f"{branches['rules']['error_detection_auroc']}（≈ 0.5），"
        "因为它的置信度只有两个固定取值，**结构性无法定位错误**——"
        "这解释了 11 章里 `rule-first` 为什么必然拖累质量。"
    )
    return {
        "ece_best": ece_rank[0],
        "auroc_best": auroc_rank[0],
        "ece_and_auroc_agree": agree,
        "encoder_auroc": auroc,
        "encoder_auroc_strength": strength,
        "encoder_aurc": encoder["aurc"],
        "encoder_base_error": base_error,
        "aurc_vs_base_error": encoder["aurc_vs_base_error"],
        "gain_at_90pct_coverage": {
            branch: gain_at(branch, 0.9) for branch in BRANCHES
        },
        "conclusion": "".join(parts),
    }


def _to_markdown(report: dict) -> str:
    lines = [
        "# 置信度诊断：能否定位错误 v1",
        "",
        f"- 样本数：{report['n_samples']}（来自交叉验证的逐样本预测，**未重新训练**）",
        "",
        f"> {report['note']}",
        "",
        "## 四个指标横向对比",
        "",
        "| 分支 | 准确率 | ECE | 错误检测 AUROC | AURC | AURC − 整体错误率 |",
        "|---|---|---|---|---|---|",
    ]
    for branch, entry in report["branches"].items():
        lines.append(
            f"| `{branch}` | {entry['base_accuracy']:.4f} | "
            f"{entry.get('ece_top_confidence')} | "
            f"{entry['error_detection_auroc']:.4f} | {entry['aurc']:.4f} | "
            f"{entry['aurc_vs_base_error']:+.4f} |"
        )

    lines += [
        "",
        "## 风险-覆盖曲线（只服务最自信的 x%）",
        "",
        "| 分支 | 覆盖 | 样本数 | 准确率 | 相对基线增益 |",
        "|---|---|---|---|---|",
    ]
    for branch, entry in report["branches"].items():
        base = entry["base_accuracy"]
        for point in entry["risk_coverage"]:
            lines.append(
                f"| `{branch}` | {point['coverage']:.0%} | {point['n']} | "
                f"{point['accuracy']:.4f} | {point['accuracy'] - base:+.4f} |"
            )

    lines += [
        "",
        "## 判对 vs 判错的置信度分布",
        "",
        "| 分支 | 判对均值 | 判错均值 | 差距 | 判错样本里置信度高于判对中位数的比例 |",
        "|---|---|---|---|---|",
    ]
    for branch, entry in report["branches"].items():
        gap = entry["confidence_gap"]
        if "gap" not in gap:
            continue
        lines.append(
            f"| `{branch}` | {gap['correct_mean']} | {gap['error_mean']} | "
            f"{gap['gap']} | {gap['error_above_correct_median']} |"
        )

    lines += ["", "## 按难度层拆开（encoder / Qwen）", ""]
    for branch, layers in report["per_layer"].items():
        lines += [
            f"### {branch}",
            "",
            "| 难度层 | 准确率 | AUROC | AURC |",
            "|---|---|---|---|",
        ]
        for layer, entry in layers.items():
            lines.append(
                f"| `{layer}` | {entry['base_accuracy']:.4f} | "
                f"{entry['error_detection_auroc']:.4f} | {entry['aurc']:.4f} |"
            )
        lines.append("")

    verdict = report["verdict"]
    lines += [
        "## 判定",
        "",
        (
            f"- ECE 最好的分支：`{verdict['ece_best']}`　"
            f"AUROC 最好的分支：`{verdict['auroc_best']}`　"
            f"**排序一致：{'是' if verdict['ece_and_auroc_agree'] else '否'}**"
        ),
        f"- 90% 覆盖下的准确率增益：{verdict['gain_at_90pct_coverage']}",
        "",
        f"**结论：** {verdict['conclusion']}",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
