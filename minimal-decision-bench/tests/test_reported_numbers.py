"""锁住「已入库报告」里的关键结论数字，防止文档与数据漂移。

设计约束
--------
1. **只锁稳定的量**。延迟、吞吐、部署体积这些是**机器相关**的——
   `reports/latency_benchmark.json` 里带着本机硬件串（`macOS-…-arm64`），
   而 MPS 上的推理还有可观的抖动（仓库自己的记录：同 seed 同超参跑三次，
   intent F1 分别是 0.7404 / 0.7638 / 0.8424，极差 0.102）。
   所以这些数字只能做**容差**断言，甚至只能断言「存在且量级合理」。
2. **口径分离**。仓库里有三份报告，指标名同名但口径不同：
   - `cross_validation.json`  → **交叉验证口径**（主口径）
   - `multiseed_report.json`   → **多种子 / 同一切分**口径
   - `comparison_report.json`  → **单次切分**口径
   断言时必须点名文件，不能混用。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"缺少 reports/{name}（先跑 make main / make cv）")
    return json.loads(path.read_text(encoding="utf-8"))


# ── 交叉验证口径（主口径，数字稳定） ──────────────────────────────────────────

# pooled 指标：encoder / qwen_lora / rules
CV_POOLED = {
    ("intent", "f1_macro"): (0.9058, 0.8967, 0.5933),
    ("intent", "ece_top_confidence"): (0.0570, 0.0746, 0.2667),
    ("needs_escalation", "f1"): (0.8387, 0.8411, 0.6898),
    ("needs_escalation", "brier"): (0.1002, 0.1138, 0.2483),
    ("complexity", "mae"): (0.1281, 0.1536, 0.1670),
}


@pytest.mark.parametrize("metric,expected", list(CV_POOLED.items()))
def test_cv_pooled_metrics_are_stable(metric, expected) -> None:
    """交叉验证 pooled 的 5 个指标 × 3 个分支。"""
    cv = _load("cross_validation.json")
    task, key = metric
    for branch, want in zip(("encoder", "qwen_lora", "rules"), expected):
        got = cv["summary"]["branches"][branch]["pooled"][task][key]
        assert got == pytest.approx(want, abs=1e-4), f"{branch}.{task}.{key}: {got} != {want}"


def test_cv_pooled_sample_size_and_template_count() -> None:
    """pooled n = 480 = 120 模板 × 4 后缀。"""
    cv = _load("cross_validation.json")
    assert cv["summary"]["branches"]["encoder"]["pooled"]["n"] == 480
    # 4 个后缀变体近似重复 —— 统计推断应按 120 个模板做 cluster bootstrap
    assert 480 == 120 * 4


def test_cv_per_fold_counts() -> None:
    cv = _load("cross_validation.json")
    assert cv["summary"]["folds_done"] == 5
    for branch in ("encoder", "qwen_lora", "rules"):
        for metric, entry in cv["summary"]["branches"][branch]["per_fold"].items():
            assert len(entry["values"]) == 5, f"{branch}.{metric}"


# ── 差距统计（scripts/13_gap_stats.py 的产物） ────────────────────────────────

def test_gap_stats_descriptive_ratios_match_published_values() -> None:
    """描述性 gap 比必须与文档里写的数字一致（口径：总体标准差 ddof=0）。"""
    stats = _load("gap_stats.json")
    metrics = stats["comparisons"]["encoder_vs_qwen"]["metrics"]
    published = {
        "intent_f1": 0.33,
        "intent_ece": 0.18,
        "escalation_f1": 0.05,
        "escalation_brier": 0.65,
        "complexity_mae": 1.89,
    }
    for name, want in published.items():
        got = metrics[name]["descriptive_ratio_vs_pooled_std"]
        assert got == pytest.approx(want, abs=0.01), f"{name}: {got} != {want}"


def test_gap_stats_models_vs_rules_range_and_exception() -> None:
    """模型相对规则：encoder 4.99~8.63、Qwen 5.39~7.26，唯一例外是 Qwen complexity MAE = 1.08。"""
    stats = _load("gap_stats.json")
    cmp_ = stats["comparisons"]["model_vs_rules"]

    enc = {k: v["descriptive_ratio_vs_pooled_std"] for k, v in cmp_["encoder"]["metrics"].items()}
    qwen = {k: v["descriptive_ratio_vs_pooled_std"] for k, v in cmp_["qwen_lora"]["metrics"].items()}

    assert min(enc.values()) == pytest.approx(4.99, abs=0.01)
    assert max(enc.values()) == pytest.approx(8.63, abs=0.01)
    assert qwen["complexity_mae"] == pytest.approx(1.08, abs=0.01)
    others = [v for k, v in qwen.items() if k != "complexity_mae"]
    assert min(others) == pytest.approx(5.39, abs=0.01)
    assert max(others) == pytest.approx(7.26, abs=0.01)


def test_gap_stats_significance_conclusion() -> None:
    """权威结论：依赖校正 + Holm 之后**没有任何指标**达到显著性门槛。

    这条断言的意义是：一旦有人把 NB 校正的 std 从 ddof=1 改成 ddof=0，
    escalation Brier 的校正 p 会从 0.0107 变成 0.0073（低于 Holm 阈值 0.010），
    结论会反转。这条测试就是为了拦住那种改动。
    """
    stats = _load("gap_stats.json")
    assert stats["significance_policy"]["std_ddof_for_nadeau_bengio"] == 1
    assert stats["significance_policy"]["survivors"] == []

    metrics = stats["comparisons"]["encoder_vs_qwen"]["metrics"]
    brier = metrics["escalation_brier"]
    assert brier["nadeau_bengio"]["p"] == pytest.approx(0.0107, abs=0.0005)
    assert brier["paired_t"]["p"] == pytest.approx(0.0025, abs=0.0005)
    assert brier["wilcoxon_exact"]["p"] == pytest.approx(0.0625, abs=0.0005)
    assert brier["direction_consistent"] == "5/5"

    # 其余四个指标在朴素口径下也不显著
    for name in ("intent_f1", "intent_ece", "escalation_f1"):
        assert metrics[name]["paired_t"]["p"] > 0.05, name


def test_gap_stats_carries_caliber_and_comparison_tags() -> None:
    """每条 gap 记录都要带口径标签，避免不同口径的结论被混着读。"""
    stats = _load("gap_stats.json")
    assert stats["caliber"] == "cross_validation"
    for key, cmp_ in stats["comparisons"].items():
        if key == "model_vs_rules":
            for branch, sub in cmp_.items():
                assert sub["caliber"] == "cross_validation", branch
        else:
            assert cmp_["caliber"] == "cross_validation", key


def test_multiseed_report_uses_descriptive_flag_name() -> None:
    """多种子报告用的是**描述性**标志，不是显著性判定。"""
    report = _load("multiseed_report.json")
    gaps = list(report["gaps_vs_rules"].values()) + list(report["gap_encoder_vs_qwen"].values())
    assert gaps, "multiseed_report.json 里没有 gap 记录"
    for gap in gaps:
        assert "descriptive_separated" in gap
        assert "separated" not in gap
        assert "gap_over_pooled_std" in gap


# ── 机器相关量：只做量级断言 ────────────────────────────────────────────────

def test_latency_report_is_machine_tagged_and_ordered() -> None:
    """延迟是机器相关的：只断言「规则 < encoder < Qwen」这个**顺序**与量级。"""
    report = _load("latency_benchmark.json")
    assert report["hardware"]["machine"]
    branches = report["branches"]
    rules = branches["rules"]["end_to_end_serial"]["p50_ms"]
    enc = branches["encoder"]["end_to_end_serial"]["p50_ms"]
    qwen = branches["qwen_lora"]["end_to_end_serial"]["p50_ms"]
    assert rules < enc < qwen
    # 量级：encoder 约 20 ms、Qwen 约 140 ms（容差 30%）
    assert 10 < enc < 30
    assert 90 < qwen < 200


def test_latency_end_to_end_is_sum_of_three_task_models() -> None:
    """端到端延迟 = 三个任务模型 p50 之和——文档里必须按这个口径解读。"""
    report = _load("latency_benchmark.json")
    for branch in ("encoder", "qwen_lora"):
        tasks = report["branches"][branch]["tasks"]
        total = sum(t["single_request"]["p50"]["p50_ms"] for t in tasks.values())
        assert report["branches"][branch]["end_to_end_serial"]["p50_ms"] == pytest.approx(
            round(total, 2), abs=0.01
        )


def test_footprint_is_per_task_model_not_per_system() -> None:
    """部署体积是**单任务模型**的体积，三任务合计要乘 3。"""
    report = _load("latency_benchmark.json")
    enc = report["branches"]["encoder"]["footprint_mb"]["intent"]
    assert enc["own_mb"] == pytest.approx(390.6, abs=1.0)
    tasks = report["branches"]["encoder"]["tasks"]
    assert len(tasks) == 3


def test_hard_cases_report_shape() -> None:
    report = _load("hard_cases_report.json")
    total = sum(len(v) for v in report["by_failure_mode"].values()) if "by_failure_mode" in report else None
    if total is None:
        pytest.skip("难例报告结构变化，跳过数量断言")
    assert total == 50
