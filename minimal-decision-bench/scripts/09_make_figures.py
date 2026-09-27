"""从 reports/ 生成教程用图表。

产出到仓库根目录的 assets/，命名沿用 `章节号-序号-主题` 约定。

    python scripts/09_make_figures.py

需要的报告（缺失则跳过对应图，不报错）：

- reports/latency_benchmark.json      -> 26-01 质量 vs 成本
- reports/cross_validation.json       -> 28-01 难度分层、31-01 折间稳定性
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT.parent / "assets"

# 每张图的源报告：用来检查「报告已更新但图没重生成」这种静默过期。
#
# 注意：matplotlib 输出**不可字节复现**（PNG/SVG 里带创建时间等元数据），
# 所以不能用「重新生成后 diff」做守卫，只能比时间戳。
SOURCES: dict[str, tuple[str, ...]] = {
    "26-01-quality-vs-cost": ("latency_benchmark.json", "multiseed_report.json"),
    "28-01-difficulty-layers": ("cross_validation.json",),
    "31-01-cv-stability": ("cross_validation.json",),
    "31-02-confidence-diagnosis": ("confidence_diagnosis.json",),
}

# 三个分支统一配色，保证跨图一致
COLORS = {"encoder": "#185FA5", "qwen_lora": "#534AB7", "rules": "#854F0B"}
LABELS = {"encoder": "encoder-only", "qwen_lora": "Qwen3.5-0.8B LoRA", "rules": "关键词规则"}
BRANCHES = ("rules", "encoder", "qwen_lora")

CJK_FONTS = (
    "PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC",
    "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
    "SimHei", "Arial Unicode MS",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--formats", default="svg,png", help="逗号分隔，例如 svg 或 svg,png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--check",
        action="store_true",
        help="只检查图表是否比源报告旧，不生成（供 make check / CI 使用）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        sys.exit(1 if _report_stale() else 0)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    _setup_fonts()
    ASSETS.mkdir(parents=True, exist_ok=True)

    made = []
    made += _figure_quality_vs_cost(formats, args.dpi)
    made += _figure_difficulty_layers(formats, args.dpi)
    made += _figure_cv_stability(formats, args.dpi)
    made += _figure_confidence_diagnosis(formats, args.dpi)

    if not made:
        print("没有生成任何图表：先跑 03/06/08 脚本产出报告")
        return
    print(f"\n共生成 {len(made)} 个文件 -> {ASSETS}")
    for path in made:
        print(f"  {path.name}")


def _report_stale() -> int:
    """报告比图新说明图过期了。返回过期数量（供退出码使用）。"""
    stale: list[str] = []
    for stem, sources in SOURCES.items():
        figure = ASSETS / f"{stem}.png"
        if not figure.exists():
            continue
        reports = [
            ROOT / "reports" / name
            for name in sources
            if (ROOT / "reports" / name).exists()
        ]
        if not reports:
            continue
        newest = max(path.stat().st_mtime for path in reports)
        if newest > figure.stat().st_mtime:
            stale.append(
                f"{stem}（源报告更新于 {time.strftime('%m-%d %H:%M', time.localtime(newest))}）"
            )
    if stale:
        print("以下图表比源报告旧，请重跑 scripts/09_make_figures.py：")
        for item in stale:
            print(f"  {item}")
    else:
        print("图表与源报告时间一致")
    return len(stale)


def _setup_fonts() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            print(f"使用中文字体：{name}")
            break
    else:
        print("[warn] 未找到中文字体，中文标签可能显示为方块")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.autolayout"] = True


def _load(name: str) -> dict | None:
    path = ROOT / "reports" / name
    if not path.exists():
        print(f"[skip] 缺少 {path.name}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, stem: str, formats: list[str], dpi: int) -> list[Path]:
    out = []
    for fmt in formats:
        path = ASSETS / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        out.append(path)
    plt.close(fig)
    return out


# --- 26-01 质量 vs 成本 ------------------------------------------------------


def _figure_quality_vs_cost(formats: list[str], dpi: int) -> list[Path]:
    latency = _load("latency_benchmark.json")
    multiseed = _load("multiseed_report.json")
    if not latency or not multiseed:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for branch in BRANCHES:
        e2e = latency["branches"][branch]["end_to_end_serial"]["p50_ms"]
        f1 = multiseed["branches"][branch]["intent.f1_macro"]
        ax.errorbar(
            e2e, f1["mean"], yerr=f1["std"], fmt="o", markersize=11,
            color=COLORS[branch], ecolor=COLORS[branch], elinewidth=1.4,
            capsize=5, label=LABELS[branch], zorder=3,
        )
        # 偏移单位是 points；encoder 误差棒向上延伸，标注放下面避免压线
        offset = (0, -24) if branch == "encoder" else (0, 16)
        ax.annotate(
            f"{f1['mean']:.3f}", (e2e, f1["mean"]),
            textcoords="offset points", xytext=offset, ha="center",
            fontsize=10, color=COLORS[branch], fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("端到端 P50 延迟（毫秒，对数轴）")
    ax.set_ylabel("intent F1 (macro)")
    ax.set_title("质量与成本的取舍：三条判别路线\n（误差棒为多种子标准差）", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0.5, 0.92)
    return _save(fig, "26-01-quality-vs-cost", formats, dpi)


# --- 28-01 难度分层 ----------------------------------------------------------


def _figure_difficulty_layers(formats: list[str], dpi: int) -> list[Path]:
    cv = _load("cross_validation.json")
    if not cv:
        return []
    layers = _layers(cv)
    if not layers:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.26
    xs = range(len(layers))
    for i, branch in enumerate(BRANCHES):
        values = []
        for layer in layers:
            entry = cv["summary"]["branches"].get(branch, {}).get("per_layer", {}).get(layer)
            values.append(entry["intent_accuracy"] if entry else 0.0)
        positions = [x + (i - 1) * width for x in xs]
        bars = ax.bar(positions, values, width, label=LABELS[branch],
                      color=COLORS[branch], alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [f"{layer}\n({_layer_desc(layer)})" for layer in layers], fontsize=10
    )
    ax.set_ylabel("intent 准确率（交叉验证 pooled）")
    ax.set_title("模板难度分层上的三方表现\n（conflict 层让规则崩掉，模型的相对优势最大）", fontsize=12)
    # 留出图例空间，避免与 1.00 的柱子重叠
    ax.set_ylim(0, 1.34)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    return _save(fig, "28-01-difficulty-layers", formats, dpi)


def _layer_desc(layer: str) -> str:
    return {
        "explicit": "含本类词",
        "conflict": "含他类词",
        "synonym": "无类别词",
    }.get(layer, layer)


def _layers(cv: dict) -> list[str]:
    for branch in BRANCHES:
        per_layer = cv["summary"]["branches"].get(branch, {}).get("per_layer")
        if per_layer:
            return list(per_layer)
    return []


# --- 31-01 折间稳定性 --------------------------------------------------------


def _figure_cv_stability(formats: list[str], dpi: int) -> list[Path]:
    cv = _load("cross_validation.json")
    if not cv:
        return []
    fold_ids = sorted(cv["results"], key=int)
    if len(fold_ids) < 2:
        return []

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [int(f) + 1 for f in fold_ids]
    for branch in BRANCHES:
        values = []
        for fid in fold_ids:
            entry = cv["results"][fid]["branches"].get(branch)
            values.append(entry["metrics"]["intent"]["f1_macro"] if entry else float("nan"))
        stats = cv["summary"]["branches"].get(branch, {}).get("per_fold", {}).get("intent.f1_macro")
        mean = stats["mean"] if stats else sum(values) / len(values)
        ax.plot(xs, values, "o-", color=COLORS[branch], label=LABELS[branch],
                markersize=7, linewidth=1.6)
        ax.axhline(mean, color=COLORS[branch], linestyle="--", linewidth=0.9, alpha=0.55)

    ax.set_xlabel("交叉验证折")
    ax.set_ylabel("intent F1 (macro)")
    ax.set_title("折间稳定性：每条模板都被评测一次\n（虚线为各分支均值）", fontsize=12)
    ax.set_xticks(xs)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="center right")
    return _save(fig, "31-01-cv-stability", formats, dpi)


# --- 31-02 置信度能否定位错误 -------------------------------------------------


def _figure_confidence_diagnosis(formats: list[str], dpi: int) -> list[Path]:
    diag = _load("confidence_diagnosis.json")
    if not diag:
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # 左：风险-覆盖曲线（只服务最自信的 x% 时准确率如何）
    for branch in BRANCHES:
        entry = diag["branches"].get(branch)
        if not entry:
            continue
        points = entry["risk_coverage"]
        ax1.plot(
            [p["coverage"] for p in points],
            [p["accuracy"] for p in points],
            "o-", color=COLORS[branch], label=LABELS[branch], markersize=6,
        )
        ax1.axhline(
            entry["base_accuracy"], color=COLORS[branch],
            linestyle="--", linewidth=0.8, alpha=0.45,
        )
    ax1.set_xlabel("覆盖率（只服务最自信的 x%）")
    ax1.set_ylabel("该子集上的 intent 准确率")
    ax1.set_title("风险-覆盖曲线（虚线为各自全量基线）", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.35)
    ax1.legend(frameon=False, loc="lower left", fontsize=9)
    ax1.set_ylim(0.5, 1.02)

    # 右：判对 vs 判错的置信度均值——差距越小越难定位错误
    names = [
        branch
        for branch in BRANCHES
        if "gap" in diag["branches"].get(branch, {}).get("confidence_gap", {})
    ]
    xs = list(range(len(names)))
    width = 0.36
    ok_bars = ax2.bar(
        [x - width / 2 for x in xs],
        [diag["branches"][b]["confidence_gap"]["correct_mean"] for b in names],
        width, label="判对", color="#0F6E56", alpha=0.85,
    )
    bad_bars = ax2.bar(
        [x + width / 2 for x in xs],
        [diag["branches"][b]["confidence_gap"]["error_mean"] for b in names],
        width, label="判错", color="#A32D2D", alpha=0.85,
    )
    ax2.bar_label(ok_bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.bar_label(bad_bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([LABELS[b] for b in names], fontsize=9)
    ax2.set_ylabel("intent 置信度（均值）")
    ax2.set_title("判对 vs 判错的置信度差距", fontsize=11)
    ax2.set_ylim(0, 1.18)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)

    return _save(fig, "31-02-confidence-diagnosis", formats, dpi)


if __name__ == "__main__":
    main()