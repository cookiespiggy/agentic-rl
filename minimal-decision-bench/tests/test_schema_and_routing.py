from __future__ import annotations

import collections
import math
import pathlib
import random
import sys

import pytest
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.data_builder import (
    MIN_TEMPLATES_PER_INTENT,
    _split_sizes,
    assert_no_text_leakage,
    build_dataset,
    build_folds,
    complexity_floor,
    enumerate_templates,
    enumerate_texts,
    fold_split,
    strip_suffix,
    subtype_of,
    text_complexity,
    unique_text_stats,
)
from minimal_decision_bench.metrics import escalation_metrics
from minimal_decision_bench.routing import route_by_confidence
from minimal_decision_bench.rules import (
    CONFIDENCE_FALLBACK,
    CONFIDENCE_MATCHED,
    decide,
    infer_escalation,
    infer_rows,
)
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.selective import (
    aurc,
    confidence_gap,
    diagnose,
    error_detection_auroc,
    risk_coverage,
)
from minimal_decision_bench.trainers import (
    ATTENTION_PROJECTION_NAMES,
    _guess_target_modules,
    branch_paths,
)


def test_schema_load() -> None:
    schema = load_schema(ROOT / "schemas" / "v1.json")
    assert schema.schema_id == "saas-router-v1"
    assert "billing" in schema.intent_labels()


def test_route_rules() -> None:
    assert route_by_confidence(0.91, 0.2, 0.2, 0.85, 0.6).tier == "cheap"
    assert route_by_confidence(0.7, 0.4, 0.3, 0.85, 0.6).tier == "mid"
    assert route_by_confidence(0.8, 0.9, 0.8, 0.85, 0.6).tier == "premium"


# --- 数据切分：防泄漏守卫 ---------------------------------------------------


def test_no_text_leakage_between_splits() -> None:
    """回归测试：旧实现让 test 的 72 条样本 100% 与 train 文本重合。"""
    train, dev, test = build_dataset(seed=42, reps_per_text=1)
    assert_no_text_leakage(train, dev, test)

    tr = {row["text"] for row in train}
    dv = {row["text"] for row in dev}
    te = {row["text"] for row in test}
    assert not (tr & te)
    assert not (tr & dv)
    assert not (dv & te)


def test_no_template_leakage_between_splits() -> None:
    """回归测试：仅按文本切分仍会漏，测试集的底层模板可能全在训练集里。"""
    train, dev, test = build_dataset(seed=42, reps_per_text=1)
    tr = {strip_suffix(row["text"]) for row in train}
    dv = {strip_suffix(row["text"]) for row in dev}
    te = {strip_suffix(row["text"]) for row in test}
    assert not (tr & te)
    assert not (tr & dv)
    assert not (dv & te)
    # 每个 split 都必须非空，否则评测无从谈起
    assert tr and dv and te


def test_split_covers_all_templates() -> None:
    total_templates = sum(len(v) for v in enumerate_templates().values())
    train, dev, test = build_dataset(seed=42, reps_per_text=1)
    covered = {strip_suffix(row["text"]) for row in train + dev + test}
    assert len(covered) == total_templates
    # 展开后的文本数量 = 模板数 × 后缀数
    assert len(enumerate_texts()["billing"]) == total_templates // 4 * 4


def test_same_text_has_stable_labels() -> None:
    """回归测试：旧实现每次重采样 complexity，同一文本标签自相矛盾。"""
    train, dev, test = build_dataset(seed=42, reps_per_text=3)
    seen: dict[str, tuple[float, bool]] = {}
    for row in train + dev + test:
        signature = (row["labels"]["complexity"], row["labels"]["needs_escalation"])
        assert seen.setdefault(row["text"], signature) == signature


def test_text_complexity_is_deterministic_and_in_range() -> None:
    text = "API 一直报 500，线上不可用。"
    assert text_complexity("support", text) == text_complexity("support", text)
    value = text_complexity("support", text)
    assert 0.0 <= value <= 1.0


# --- 指标语义：Brier 必须用 P(正类) ----------------------------------------


def test_escalation_brier_uses_prob_true() -> None:
    # 若误传 top-class 置信度（例如全预测 False 却传 0.9），brier 会变成 0.81
    perfect_positive = escalation_metrics([True, True], [True, True], [1.0, 1.0])
    perfect_negative = escalation_metrics([False, False], [False, False], [0.0, 0.0])
    assert perfect_positive["brier"] == pytest.approx(0.0)
    assert perfect_negative["brier"] == pytest.approx(0.0)


# --- LoRA 目标模块：必须覆盖混合架构 ----------------------------------------


def test_attention_target_names_cover_hybrid_arch() -> None:
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"):
        assert name in ATTENTION_PROJECTION_NAMES


def test_guess_target_modules_covers_linear_attention() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class FakeHybrid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.full_attn = nn.ModuleDict(
                {n: nn.Linear(4, 4) for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
            )
            self.linear_attn = nn.ModuleDict(
                {"in_proj_qkv": nn.Linear(4, 4), "out_proj": nn.Linear(4, 4)}
            )
            self.mlp = nn.ModuleDict({"down_proj": nn.Linear(4, 4)})

    matched = _guess_target_modules(FakeHybrid())
    assert "q_proj" in matched
    assert "in_proj_qkv" in matched
    assert "out_proj" in matched
    # MLP 投影不应被当作注意力层
    assert "down_proj" not in matched


def test_unique_text_stats_reports_counts() -> None:
    rows = [
        {"text": "a"},
        {"text": "a"},
        {"text": "a 比较着急。"},
        {"text": "b 比较着急。"},
    ]
    stats = unique_text_stats(rows)
    assert stats["rows"] == 4
    assert stats["unique_texts"] == 3
    assert stats["unique_templates"] == 2


# --- 切分尺寸与多种子路径 ---------------------------------------------------


def test_split_sizes_never_starves_a_split() -> None:
    for n in range(MIN_TEMPLATES_PER_INTENT, 25):
        n_train, n_dev, n_test = _split_sizes(n, (0.6, 0.2, 0.2))
        assert n_train >= 1 and n_dev >= 1 and n_test >= 1
        assert n_train + n_dev + n_test == n


def test_split_sizes_rejects_tiny_pool() -> None:
    with pytest.raises(ValueError):
        _split_sizes(MIN_TEMPLATES_PER_INTENT - 1, (0.6, 0.2, 0.2))


def test_branch_paths_isolates_run_tags() -> None:
    plain = branch_paths(ROOT, "encoder")
    tagged = branch_paths(ROOT, "encoder", "_s43")
    assert plain[0].name == "encoder"
    assert tagged[0].name == "encoder_s43"
    assert plain[1].name == "encoder_metrics.json"
    assert tagged[1].name == "encoder_metrics_s43.json"
    # 多种子运行绝不能覆盖头条结果
    assert plain[1] != tagged[1]


# --- 规则基线（rules_v0） ---------------------------------------------------


def test_rules_priority_is_billing_then_support_then_sales() -> None:
    assert decide("退款接口一直超时").intent == "billing"
    assert decide("报价页面打不开").intent == "support"
    assert decide("我们公司想买，什么价位").intent == "sales"


def test_rules_fallback_to_general_and_lower_confidence() -> None:
    decision = decide("我该怎么设置头像？")
    assert decision.intent == "general"
    assert decision.confidence == CONFIDENCE_FALLBACK
    assert decide("我被重复扣费了").confidence == CONFIDENCE_MATCHED
    assert "fallback:general" in decision.trace


def test_rules_trace_is_recorded_for_audit() -> None:
    decision = decide("续费之后系统就一直报错。")
    # 多诉求句要能回放出两个类别都命中了
    assert any(item.startswith("billing:") for item in decision.trace)
    assert any(item.startswith("support:") for item in decision.trace)


def test_rules_documented_failure_mode_on_conflict() -> None:
    """设计内的失败：规则分不清「诉求」与「背景」，会把背景词当诉求。

    这条用例把失败模式固定下来，防止有人无意中「修好」它而掩盖了
    规则基线的真实上限。
    """
    # 真值是 support（用户要修系统），规则按优先级判成 billing
    assert decide("续费之后系统就一直报错。").intent == "billing"


def test_rules_output_matches_infer_task_shape() -> None:
    rows = [
        {
            "text": "API 一直报 500，线上不可用",
            "labels": {"intent": "support", "intent_id": 1, "complexity": 0.8,
                       "needs_escalation": True, "needs_escalation_id": 1},
        }
    ]
    out = infer_rows(rows, "intent_id", intent_labels=["billing", "support", "sales", "general"])
    assert set(out) >= {"labels", "preds", "confidences", "probs"}
    assert out["preds"] == [1]
    assert len(out["probs"][0]) == 4
    esc = infer_escalation(rows)
    # 「线上」命中升级关键词
    assert esc["preds"] == [1]
    assert esc["probs"][0][1] == pytest.approx(CONFIDENCE_MATCHED)


def test_rule_baseline_band_guards_dataset_separability() -> None:
    """守卫：规则基线既不能满分（数据太简单），也不能过低（数据不合理）。

    这是「数据集有没有区分度」的回归测试——没有它，模板很容易被改回
    「关键词与类别一一对应」的状态，整个 bench 又会失去意义。
    """
    _, _, test = build_dataset(seed=42, reps_per_text=1)
    y_true = [row["labels"]["intent"] for row in test]
    y_pred = [decide(row["text"]).intent for row in test]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert 0.45 < f1 < 0.85, f"规则基线 F1={f1:.4f}，数据集区分度异常"


def test_rule_baseline_layer_profile() -> None:
    """规则应在显式样本上强、在关键词冲突样本上弱。"""
    _, _, test = build_dataset(seed=42, reps_per_text=1)
    stats: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for row in test:
        layer = subtype_of(row["labels"]["intent"], strip_suffix(row["text"]))
        stats[layer][0] += 1
        stats[layer][1] += decide(row["text"]).intent == row["labels"]["intent"]

    explicit = stats["explicit"][1] / stats["explicit"][0]
    conflict = stats["conflict"][1] / stats["conflict"][0]
    assert explicit > 0.9, f"显式层规则应接近满分，实际 {explicit:.2f}"
    assert conflict < explicit - 0.3, f"冲突层规则应显著更差，实际 {conflict:.2f}"


# --- 交叉验证折分配 ---------------------------------------------------------


def test_build_folds_covers_every_template_exactly_once() -> None:
    folds = build_folds(n_folds=5, seed=42)
    all_templates = {t for ts in enumerate_templates().values() for t in ts}
    covered: list[str] = []
    for fold in folds:
        covered.extend(t for ts in fold.values() for t in ts)
    assert len(covered) == len(all_templates)
    assert set(covered) == all_templates


def test_build_folds_is_stratified() -> None:
    """每折在 (intent × 难度层) 上的构成必须一致，否则折间不可比。"""
    folds = build_folds(n_folds=5, seed=42)
    counts = []
    for fold in folds:
        counter: dict[tuple[str, str], int] = collections.Counter()
        for intent, templates in fold.items():
            for template in templates:
                counter[(intent, subtype_of(intent, template))] += 1
        counts.append(counter)
    assert all(counter == counts[0] for counter in counts), "各折难度构成不一致"
    assert len(counts[0]) == 4 * 3, "应覆盖 4 类 intent × 3 个难度层"


def test_fold_split_keeps_train_test_disjoint() -> None:
    folds = build_folds(n_folds=5, seed=42)
    for k in range(5):
        train, test = fold_split(folds, k)
        train_templates = {t for ts in train.values() for t in ts}
        test_templates = {t for ts in test.values() for t in ts}
        assert train_templates and test_templates
        assert not (train_templates & test_templates), f"fold {k} 出现模板泄漏"


# --- 复杂度标签下限 ---------------------------------------------------------


def test_complexity_floor_ordering() -> None:
    """复杂度标签有不可约噪声，三档参照必须严格递减。"""
    train, dev, test = build_dataset(seed=42, reps_per_text=1)
    floor = complexity_floor(train + dev + test)
    assert floor["random_guess_mae"] > floor["intent_only_mae"] > floor["oracle_mae"]
    # 下限必须明显大于 0——否则说明标签里没有不可学成分，这个函数就失去意义
    assert floor["oracle_mae"] > 0.05


def test_complexity_floor_reveals_intent_dominance() -> None:
    """可学信号应主要来自 intent：只知道 intent 就能拿下大部分可压缩空间。"""
    train, dev, test = build_dataset(seed=42, reps_per_text=1)
    floor = complexity_floor(train + dev + test)
    total = floor["random_guess_mae"] - floor["oracle_mae"]
    from_intent = floor["random_guess_mae"] - floor["intent_only_mae"]
    assert from_intent / total > 0.6, "intent 应贡献大部分可学信号"


# --- 选择性预测（置信度能否定位错误） ---------------------------------------


def test_error_detection_auroc_extremes() -> None:
    """完美分离 / 完全反向 / 零信息三种极端情形。"""
    # 判对的置信度都高于判错的 → 完美分离
    assert error_detection_auroc([0.9, 0.9, 0.2, 0.2], [True, True, False, False]) == 1.0
    # 反向 → 0
    assert error_detection_auroc([0.2, 0.2, 0.9, 0.9], [True, True, False, False]) == 0.0
    # 全同置信度 → 无排序信息
    flat = error_detection_auroc([0.5] * 4, [True, False, True, False])
    assert math.isnan(flat) or abs(flat - 0.5) < 0.01


def test_error_detection_auroc_nan_when_single_class() -> None:
    assert math.isnan(error_detection_auroc([0.9, 0.8], [True, True]))


def test_risk_coverage_is_monotonic_for_informative_confidence() -> None:
    """置信度真能区分难易时，覆盖率越低准确率越高。"""
    confidences = [0.9] * 8 + [0.1] * 2
    correct = [True] * 8 + [False] * 2
    curve = risk_coverage(confidences, correct, (0.5, 0.9, 1.0))
    accuracies = [point["accuracy"] for point in curve]
    assert accuracies == sorted(accuracies, reverse=True)
    assert curve[0]["accuracy"] == 1.0
    assert curve[-1]["accuracy"] == 0.8


def test_aurc_equals_base_error_when_confidence_uninformative() -> None:
    """无信息时 AURC 应约等于整体错误率——这正是「排序没把难样本排前面」的判据。

    用与正确性**独立**的随机置信度构造（而不是全同置信度：全同时排序退化成输入顺序，
    小覆盖率下会被交替序列带偏，测不出这个性质）。
    """
    rng = random.Random(0)
    correct = [rng.random() < 0.5 for _ in range(200)]
    confidences = [rng.random() for _ in range(200)]
    base_error = 1 - sum(correct) / len(correct)
    assert abs(aurc(confidences, correct) - base_error) < 0.05
    assert abs(error_detection_auroc(confidences, correct) - 0.5) < 0.12


def test_confidence_gap_reports_overlap() -> None:
    gap = confidence_gap([0.9, 0.85, 0.8, 0.2], [True, True, True, False])
    assert gap["n_correct"] == 3
    assert gap["n_error"] == 1
    assert gap["gap"] > 0
    assert gap["error_above_correct_median"] == 0.0


def test_diagnose_bundles_all_metrics() -> None:
    result = diagnose([0.9, 0.8, 0.3, 0.2], [True, True, False, False], ece_value=0.05)
    for key in (
        "error_detection_auroc",
        "aurc",
        "risk_coverage",
        "confidence_gap",
        "base_accuracy",
        "ece_top_confidence",
    ):
        assert key in result
    assert result["base_accuracy"] == 0.5
