"""章节里的关键数字必须与已入库的报告一致。

**口径纪律（最重要的一点）**
---------------------------
仓库里有三份报告，指标名同名但口径不同：

| 文件 | 口径 | 用途 |
|---|---|---|
| `cross_validation.json` | **5 折交叉验证** | **主口径**，文档主表都用它 |
| `multiseed_report.json` | 多种子 / 同一切分 | 补充参考，数字明显不同 |
| `comparison_report.json` | 单次切分 | 只在「单次切分 vs 交叉验证」对比里出现 |

所以断言必须**点名文件**，不能笼统地说「intent F1 应该是 0.9x」。

这个测试做两件事：
1. **存在性**：文档该写的数字写没写、写得对不对；
2. **不存在性**：文档**不该**再出现的旧表述还在不在（比如已修正的「五个数量级」）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "minimal-decision-bench"
REPORTS = BENCH / "reports"


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"缺少 reports/{name}（先跑 make main / make cv）")
    return json.loads(path.read_text(encoding="utf-8"))


def _chapter(name: str) -> str:
    matches = list(ROOT.glob(name))
    if not matches:
        pytest.skip(f"找不到章节 {name}")
    return matches[0].read_text(encoding="utf-8")


# ── 交叉验证口径：文档主表里的数字 ───────────────────────────────────────────

# 章节 -> [(标签, CV pooled 里的路径, 期望值)]
#
# 注意：只列**确实应该出现绝对值**的章节。
# ch25 讲的是「判别 vs 策略的边界」，它引用的是描述性比值（0.33 等）而不是
# 绝对 F1，所以 ch25 不在这里断言绝对值——见 test_descriptive_ratio_0_33_is_in_docs。
CV_FACTS = {
    "26-*.md": [
        ("intent F1 encoder", ("encoder", "intent", "f1_macro"), 0.9058),
        ("intent F1 qwen", ("qwen_lora", "intent", "f1_macro"), 0.8967),
        ("intent ECE encoder", ("encoder", "intent", "ece_top_confidence"), 0.0570),
    ],
    "29-*.md": [
        ("complexity MAE encoder", ("encoder", "complexity", "mae"), 0.1281),
        ("complexity MAE qwen", ("qwen_lora", "complexity", "mae"), 0.1536),
    ],
    "31-*.md": [
        ("intent ECE encoder", ("encoder", "intent", "ece_top_confidence"), 0.0570),
        ("intent ECE qwen", ("qwen_lora", "intent", "ece_top_confidence"), 0.0746),
    ],
    "32-*.md": [
        ("int8 体积", None, 98.3),
    ],
}


@pytest.mark.parametrize("pattern,facts", list(CV_FACTS.items()))
def test_chapter_quality_numbers_match_cv_report(pattern, facts) -> None:
    """章节里写的质量数字必须等于交叉验证报告里的值。"""
    cv = _load("cross_validation.json")
    text = _chapter(pattern)
    for label, path, expected in facts:
        if path is not None:
            branch, task, key = path
            got = cv["summary"]["branches"][branch]["pooled"][task][key]
            assert got == pytest.approx(expected, abs=1e-4), f"{label}: 报告里是 {got}"
        # 数字要出现在章节里（允许 4 位或 2 位小数等写法）
        candidates = {f"{expected:.4f}", f"{expected:.2f}", f"{expected:g}", str(expected)}
        assert any(c in text for c in candidates), f"{label}: {expected} 没出现在 {pattern}"


def test_latency_numbers_match_benchmark_report() -> None:
    """32 章的延迟数字必须与延迟基准报告一致。"""
    report = _load("latency_benchmark.json")
    text = _chapter("32-*.md")
    for branch in ("encoder", "qwen_lora", "rules"):
        value = report["branches"][branch]["end_to_end_serial"]["p50_ms"]
        assert f"{value:.2f}" in text, f"{branch} 端到端 P50 {value:.2f} 不在 32 章里"


def test_latency_end_to_end_is_three_models_sum() -> None:
    """端到端延迟 = 三个任务模型之和，文档必须按这个口径写。"""
    report = _load("latency_benchmark.json")
    for branch in ("encoder", "qwen_lora"):
        tasks = report["branches"][branch]["tasks"]
        total = sum(t["single_request"]["p50"]["p50_ms"] for t in tasks.values())
        assert report["branches"][branch]["end_to_end_serial"]["p50_ms"] == pytest.approx(
            round(total, 2), abs=0.01
        )


# ── 描述性 gap 比（ddof=0 口径，与显著性检验是两回事） ───────────────────────

def test_descriptive_ratio_0_33_is_in_docs() -> None:
    """encoder-vs-Qwen 的 intent F1 描述性比值 0.33。"""
    stats = _load("gap_stats.json")
    ratio = stats["comparisons"]["encoder_vs_qwen"]["metrics"]["intent_f1"][
        "descriptive_ratio_vs_pooled_std"
    ]
    assert ratio == pytest.approx(0.33, abs=0.01)
    for pattern in ("README.md", "25-*.md", "31-*.md"):
        text = _chapter(pattern)
        assert "0.33" in text, f"{pattern} 里找不到描述性比值 0.33"


def test_models_vs_rules_exception_1_08_is_present() -> None:
    """模型相对规则的描述性比值区间 4.99~8.63，唯一例外是 Qwen complexity MAE = 1.08。

    文档里凡是写「5~9 倍标准差」的地方，都必须同时写明这个例外。
    """
    stats = _load("gap_stats.json")
    qwen = stats["comparisons"]["model_vs_rules"]["qwen_lora"]["metrics"]
    assert qwen["complexity_mae"]["descriptive_ratio_vs_pooled_std"] == pytest.approx(
        1.08, abs=0.01
    )
    for pattern in ("26-*.md", "29-*.md", "33-*.md"):
        text = _chapter(pattern)
        assert "1.08" in text, f"{pattern} 里找不到例外值 1.08"


# ── 不该再出现的旧表述 ──────────────────────────────────────────────────────

# (章节 glob, 不该出现的片段, 为什么)
FORBIDDEN = [
    ("32-*.md", "五个数量级", "实际约 4.5 个数量级（31,669×）"),
    ("25-*.md", "五个数量级", "实际约 4.5 个数量级"),
    ("26-*.md", "五个数量级", "实际约 4.5 个数量级"),
    ("29-*.md", "五个数量级", "实际约 4.5 个数量级"),
    ("33-*.md", "五个数量级", "实际约 4.5 个数量级"),
    ("28-*.md", "7~43 倍标准差", "与全仓 CV 主口径不一致；应写 5~9 并注明另一种子口径"),
    ("20-*.md", "miniwob_click_test", "不是 lm-evaluation-harness 的真实任务名"),
    ("15-*.md", "456771", "算错了，(123+456)*789 = 456831"),
    ("08-*.md", "Dynamic Advantage Policy Optimization", "DAPO 的全称不是这个"),
    ("09-*.md", "Tao et al", "TIPS 的作者是 Xie et al."),
    ("23-*.md", "2503.14442", "DAPO 的 arXiv 号是 2503.14476"),
    ("24-*.md", "2503.14442", "DAPO 的 arXiv 号是 2503.14476"),
]


@pytest.mark.parametrize("pattern,forbidden,why", FORBIDDEN)
def test_forbidden_old_statements_are_gone(pattern, forbidden, why) -> None:
    text = _chapter(pattern)
    assert forbidden not in text, f"{pattern} 里还留着「{forbidden}」——{why}"


def test_no_headline_claims_post_holm_significance() -> None:
    """任何章节都不得宣称「Holm 后显著」——依赖校正后没有任何指标存活。"""
    for path in sorted(ROOT.glob("[0-9][0-9]-*.md")) + [ROOT / "README.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert "Holm 后显著" not in text, f"{path.name} 里出现了「Holm 后显著」"
        assert "Holm 校正后显著" not in text, f"{path.name} 里出现了「Holm 校正后显著」"


def test_inference_docs_match_reports() -> None:
    """inference/*.md 里的关键数字与 reports/*.json 一致。"""
    pairs = [
        ("confidence_diagnosis_v1.md", "confidence_diagnosis.json"),
        ("cascade_v1.md", "cascade_benchmark.json"),
        ("quantization_v1.md", "quantization_benchmark.json"),
        ("benchmark_v1.md", "latency_benchmark.json"),
    ]
    for md_name, json_name in pairs:
        md = BENCH / "inference" / md_name
        js = REPORTS / json_name
        if not md.exists() or not js.exists():
            pytest.skip(f"缺少 {md_name} / {json_name}")
        text = md.read_text(encoding="utf-8")
        data = json.loads(js.read_text(encoding="utf-8"))
        if json_name == "confidence_diagnosis.json":
            for branch in ("encoder", "qwen_lora", "rules"):
                auroc = data["branches"][branch]["error_detection_auroc"]
                assert f"{auroc:.4f}" in text, f"{md_name}: {branch} AUROC {auroc} 不在 md 里"


# ── 章节学习结构（06–10 与 02–05 对齐） ─────────────────────────────────────

CHAPTERS_WITH_EXERCISES = ["06", "07", "08", "09", "10"]


@pytest.mark.parametrize("prefix", CHAPTERS_WITH_EXERCISES)
def test_chapters_have_toc_exercises_and_answers(prefix) -> None:
    """每章要有 `## 目录`、`## 练习题` 和行内 `> **参考答案**：`。

    注意：**不要** YAML frontmatter（commit 072bdd8 已有意移除，GitHub 会渲染成表格）；
    **不要** `## 参考答案` 标题（全仓用的是行内引用块）。
    """
    matches = list(ROOT.glob(f"{prefix}-*.md"))
    assert matches, f"找不到 {prefix} 章"
    text = matches[0].read_text(encoding="utf-8")
    assert text.count("\n## 目录") == 1, f"{prefix} 章的 `## 目录` 数量不对"
    assert text.count("\n## 练习题") == 1, f"{prefix} 章的 `## 练习题` 数量不对"
    assert "> **参考答案**：" in text, f"{prefix} 章缺行内参考答案"
    assert "\n## 参考答案" not in text, f"{prefix} 章不该有 `## 参考答案` 标题"
    assert not text.startswith("---\n"), f"{prefix} 章不该有 YAML frontmatter"


def test_heading_numbers_are_unique_and_contiguous() -> None:
    """所有章的 `## N.M` 编号唯一且连续。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_heading_numbers", ROOT / "tools" / "check_heading_numbers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    problems = []
    for path in sorted(ROOT.glob("[0-9][0-9]-*.md")):
        problems.extend(module.check_file(path))
    assert not problems, "小节编号问题：\n" + "\n".join(problems)
