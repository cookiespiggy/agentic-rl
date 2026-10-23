"""差距统计：把「两条分支差多少」拆成**描述性比值**与**显著性检验**两件事。

这个脚本回答一个很具体的问题：文档里那句「encoder 和 Qwen 统计上无法区分」，
到底是怎么算出来的、站得住吗？

背景
----
仓库里有两套「差距」口径，经常被混用：

1. **描述性 gap 比**（`scripts/07_aggregate_multiseed.py` 同款公式）：
   ``|mean_a - mean_b| / sqrt((std_a² + std_b²) / 2)``
   这里的 std 是**总体标准差（ddof=0）**，且数据源是 JSON 里**已四舍五入**的
   ``summary.branches[b].per_fold[m].mean/.std``。
   它回答的是「差距是不是波动的很多倍」——**不是显著性检验**，不能说「显著」。

2. **显著性检验**：本脚本做的三层检验 + Holm 多重比较校正。

为什么要三层
------------
5 折交叉验证的**折之间不独立**：每折都在同一批 120 个模板上训 4/5，测试集虽不相交，
但训练集高度重叠。所以朴素配对 t 检验（``stats.ttest_rel``）会**高估显著性**。
本脚本因此同时给出：

- L1 朴素配对 t（参考值）
- L2 **Nadeau-Bengio 依赖校正 t**（权威值）
      ``SE = s · sqrt(1/n + n_test/n_train)``，5 折下 ``n_test/n_train = 1/4``
      → 因子 ``sqrt(0.2 + 0.25) = 0.6708``
      注意 ``s`` 必须是**样本标准差（ddof=1）**——用 ddof=0 会让结论反转。
- L3 **精确 Wilcoxon 符号秩检验**（假设最少；n=5 时最小可达双侧 p = 2/32 = 0.0625 > 0.05）

结论以 **L2 + Holm** 为准。

用法
----
    python scripts/13_gap_stats.py

产出 ``reports/gap_stats.json``（不重训，纯后处理，秒级）。
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

# 统一指标名 -> (results[fid] 里的路径, per_fold 里的键)
#
# 注意这两套键名**不一样**（是上游脚本的历史遗留），必须显式映射：
#   results[fid].branches[b].metrics  ->  intent.ece_top_confidence / needs_escalation.*
#   summary.branches[b].per_fold       ->  intent.ece / escalation.*
METRICS: dict[str, tuple[str, str]] = {
    "intent_f1": ("intent.f1_macro", "intent.f1_macro"),
    "intent_ece": ("intent.ece_top_confidence", "intent.ece"),
    "escalation_f1": ("needs_escalation.f1", "escalation.f1"),
    "escalation_brier": ("needs_escalation.brier", "escalation.brier"),
    "complexity_mae": ("complexity.mae", "complexity.mae"),
}

# 越小越好的指标
LOWER_IS_BETTER = {"intent_ece", "escalation_brier", "complexity_mae"}

BRANCHES = ("encoder", "qwen_lora", "rules")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default="reports/cross_validation.json",
        help="交叉验证报告路径（默认 reports/cross_validation.json）",
    )
    p.add_argument("--out", default="reports/gap_stats.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = ROOT / args.source
    if not src.exists():
        print(f"缺少 {args.source}，先跑 make cv")
        return
    cv = read_json(src)
    results = cv.get("results") or {}
    if not results:
        print(f"{args.source} 里没有 results 段")
        return

    folds = sorted(results, key=int)
    n_folds = len(folds)

    # 逐折取值（全精度）
    per_fold: dict[str, dict[str, list[float]]] = {}
    for unified, (results_path, _pf_key) in METRICS.items():
        group, key = results_path.split(".", 1)
        for branch in BRANCHES:
            vals: list[float] = []
            for fid in folds:
                node = results[fid].get("branches", {}).get(branch, {}).get("metrics", {})
                task = "intent" if group == "intent" else (
                    "needs_escalation" if group == "needs_escalation" else "complexity"
                )
                try:
                    vals.append(float(node[task][key]))
                except (KeyError, TypeError):
                    vals = []
                    break
            if vals:
                per_fold.setdefault(unified, {})[branch] = vals

    comparisons = {
        "encoder_vs_qwen": _compare(per_fold, "encoder", "qwen_lora", cv, n_folds),
        "model_vs_rules": {
            branch: _compare(per_fold, branch, "rules", cv, n_folds)
            for branch in ("encoder", "qwen_lora")
        },
    }

    # Holm 校正只对「encoder vs Qwen」这一组主比较做（5 个指标）
    holm = _holm(
        {name: rec["nadeau_bengio"]["p"] for name, rec in comparisons["encoder_vs_qwen"]["metrics"].items()}
    )
    for name, rec in comparisons["encoder_vs_qwen"]["metrics"].items():
        rec["holm_significant"] = holm.get(name, False)

    report = {
        "caliber": "cross_validation",
        "source": args.source,
        "folds": n_folds,
        "significance_policy": {
            "authoritative": "nadeau_bengio_corrected_paired_t + holm",
            "layers": {
                "L1": "朴素配对 t（stats.ttest_rel）——参考值，依赖假设不成立时会高估显著性",
                "L2": (
                    "Nadeau-Bengio 依赖校正 t：SE = s·sqrt(1/n + n_test/n_train)，"
                    "5 折下 n_test/n_train = 1/4；s 取样本标准差（ddof=1）"
                ),
                "L3": "精确 Wilcoxon 符号秩（method='exact'）；n=5 时最小可达双侧 p = 2/32 = 0.0625",
                "multiplicity": "Holm 校正，m = 指标数",
            },
            "nadeau_bengio_se_factor": round(math.sqrt(1 / n_folds + 1 / (n_folds - 1)), 4),
            "std_ddof_for_nadeau_bengio": 1,
            "holm_m": len(METRICS),
            "survivors": sorted(n for n, ok in holm.items() if ok),
        },
        "descriptive_ratio_policy": {
            "caliber": "总体标准差（ddof=0）",
            "source": "summary.branches[b].per_fold[m].mean/.std（JSON 里已四舍五入的值）",
            "warning": "这是描述性比值，不是显著性检验——不能读成「显著」",
        },
        "comparisons": comparisons,
    }

    out = ROOT / args.out
    write_json(out, report)
    print(f"saved {out.relative_to(ROOT)}")
    _print(report)


def _compare(
    per_fold: dict,
    left: str,
    right: str,
    cv: dict,
    n_folds: int,
) -> dict:
    """left vs right 的逐指标统计。"""
    from scipy import stats as sps

    metrics_out: dict = {}
    p_values: dict[str, float] = {}

    for unified, (_results_path, pf_key) in METRICS.items():
        a_vals = per_fold.get(unified, {}).get(left)
        b_vals = per_fold.get(unified, {}).get(right)
        if not a_vals or not b_vals or len(a_vals) != len(b_vals):
            continue

        diffs = [a - b for a, b in zip(a_vals, b_vals)]
        mean_d = statistics.fmean(diffs)
        sd = statistics.stdev(diffs)  # ddof=1 —— 必须是样本标准差
        lower_better = unified in LOWER_IS_BETTER

        # L1 朴素配对 t
        t_stat, p_naive = sps.ttest_rel(a_vals, b_vals)

        # L2 Nadeau-Bengio 依赖校正 t
        se_factor = math.sqrt(1 / n_folds + 1 / (n_folds - 1))
        nb_t = mean_d / (sd * se_factor) if sd > 0 else float("inf")
        nb_p = float(2 * sps.t.sf(abs(nb_t), n_folds - 1))

        # L3 精确 Wilcoxon
        try:
            w_p = float(sps.wilcoxon(a_vals, b_vals, alternative="two-sided", method="exact").pvalue)
        except (TypeError, ValueError):
            w_p = None

        # 方向一致：encoder 更优的折数
        better = sum(
            1 for d in diffs if (d < 0 if lower_better else d > 0)
        )
        ties = sum(1 for d in diffs if d == 0)

        metrics_out[unified] = {
            "per_fold": {left: a_vals, right: b_vals},
            "lower_is_better": lower_better,
            "mean_diff": round(mean_d, 6),
            "paired_t": {"t": round(float(t_stat), 4), "p": round(float(p_naive), 5)},
            "nadeau_bengio": {
                "t": round(float(nb_t), 4),
                "p": round(nb_p, 5),
                "se_factor": round(se_factor, 4),
                "std_ddof": 1,
            },
            "wilcoxon_exact": {"p": None if w_p is None else round(w_p, 5)},
            "direction_consistent": f"{better}/{n_folds}",
            "ties": ties,
            "better_side": left if better * 2 > n_folds else right,
            "descriptive_ratio_vs_pooled_std": _descriptive_ratio(cv, left, right, pf_key),
            "holm_significant": False,
        }
        p_values[unified] = nb_p

    return {
        "caliber": "cross_validation",
        "left": left,
        "right": right,
        "metrics": metrics_out,
    }


def _descriptive_ratio(cv: dict, left: str, right: str, pf_key: str) -> float | None:
    """描述性 gap 比：|mean_a - mean_b| / sqrt((std_a² + std_b²)/2)。

    数据源刻意用 ``summary...per_fold[m].mean/.std``（JSON 里已四舍五入的值），
    而不是从 ``.values`` 重算——后者会得到 0.32/0.06，与已发布的 0.33/0.05 不符。
    """
    branches = cv.get("summary", {}).get("branches", {})
    a = branches.get(left, {}).get("per_fold", {}).get(pf_key)
    b = branches.get(right, {}).get("per_fold", {}).get(pf_key)
    if not a or not b:
        return None
    pooled = math.sqrt((a["std"] ** 2 + b["std"] ** 2) / 2)
    if pooled <= 0:
        return None
    return round(abs(a["mean"] - b["mean"]) / pooled, 2)


def _holm(p_values: dict[str, float]) -> dict[str, bool]:
    """Holm-Bonferroni 逐步降序校正。返回每个指标的「是否显著」。"""
    m = len(p_values)
    if m == 0:
        return {}
    order = sorted(p_values, key=lambda k: p_values[k])
    out: dict[str, bool] = {}
    still_alive = True
    for rank, key in enumerate(order):
        threshold = 0.05 / (m - rank)
        ok = still_alive and p_values[key] <= threshold
        if not ok:
            still_alive = False
        out[key] = ok
    return out


def _print(report: dict) -> None:
    print(f"\n口径 = {report['caliber']}（{report['folds']} 折）")
    print("权威判据 = Nadeau-Bengio 依赖校正 t + Holm")
    print(f"SE 因子 = {report['significance_policy']['nadeau_bengio_se_factor']}（s 用 ddof=1）\n")

    comp = report["comparisons"]["encoder_vs_qwen"]
    header = f"{'metric':22}{'t(naive)':>10}{'p':>9}{'t(NB)':>9}{'NB p':>9}{'wilcox':>9}{'dir':>6}{'ratio':>8}"
    print(header)
    print("-" * len(header))
    for name, rec in comp["metrics"].items():
        w = rec["wilcoxon_exact"]["p"]
        print(
            f"{name:22}"
            f"{rec['paired_t']['t']:>10.3f}{rec['paired_t']['p']:>9.4f}"
            f"{rec['nadeau_bengio']['t']:>9.3f}{rec['nadeau_bengio']['p']:>9.4f}"
            f"{(w if w is not None else float('nan')):>9.4f}"
            f"{rec['direction_consistent']:>6}"
            f"{(rec['descriptive_ratio_vs_pooled_std'] if rec['descriptive_ratio_vs_pooled_std'] is not None else float('nan')):>8.2f}"
        )

    survivors = report["significance_policy"]["survivors"]
    m = report["significance_policy"]["holm_m"]
    print(f"\nHolm（m={m}）后存活：{survivors or '（无）'}")
    if survivors:
        print(f"→ 有 {len(survivors)} 个指标在依赖校正 + Holm 后达到显著性门槛：{survivors}")
    else:
        print(f"→ 没有指标在依赖校正 + Holm（m={m}）后达到显著性门槛。")

    # 下面这段从数据推导，不写死——数据变了，文案会跟着变，不会悄悄说谎。
    brier = comp["metrics"].get("escalation_brier")
    if brier:
        folds = brier["direction_consistent"]
        naive = brier["paired_t"]["p"]
        nb = brier["nadeau_bengio"]["p"]
        wil = brier["wilcoxon_exact"]["p"]
        if "escalation_brier" in survivors:
            print(f"  escalation Brier 在 Holm 后存活（朴素 p={naive}，校正 p={nb}）——"
                  f"可以说「在 {folds} 折上方向一致」")
        else:
            print(f"  escalation Brier 方向 {folds} 一致，但朴素 p={naive} -> 依赖校正 p={nb}"
                  f" -> Wilcoxon p={wil}，**未证实**，不能宣称显著。")

    print("\n模型 vs 规则（描述性比值，ddof=0 口径）：")
    for branch, cmp_ in report["comparisons"]["model_vs_rules"].items():
        for name, rec in cmp_["metrics"].items():
            r = rec["descriptive_ratio_vs_pooled_std"]
            print(f"  {branch:11s} {name:22s} ratio={r}")


if __name__ == "__main__":
    main()
