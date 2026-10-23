#!/usr/bin/env python3
"""检查 `inference/*.md` 与 `reports/*.json` 的关键数字是否一致。

背景
----
`inference/*.md` 和 `reports/*.json` 是**同一个脚本的同一次运行**写出来的
（见 06/10/11/12 各脚本），所以它们必须一致。但报告后来被重跑过（换机器、
换 MPS 版本），md 没跟着更新——于是出现了「同一个数在两个文件里不一样」的情况。

做法：**按行比对**。对每个 md，找以某个分支名开头的表格行，检查目标数字是否出现在该行里。
这样不依赖列序，比列号更耐改版。

用法：
    python tools/check_inference_mirror.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "minimal-decision-bench"


def _get(data: dict, dotted: str):
    node = data
    for part in dotted.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _rows(md_text: str, branch: str) -> list[str]:
    """取出 md 里以 `branch` 开头（允许反引号）的表格行。"""
    pattern = re.compile(r"^\|\s*`?" + re.escape(branch) + r"`?\s*\|(.*)$", re.MULTILINE)
    return [m.group(1) for m in pattern.finditer(md_text)]


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


# md 文件 -> (源 json, [(标签, md 里定位用的行首, json 取值路径)])
MIRRORS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "benchmark_v1.md": (
        "latency_benchmark.json",
        [
            ("encoder 端到端 P50", None,
             "branches.encoder.end_to_end_serial.p50_ms"),
            ("qwen 端到端 P50", None,
             "branches.qwen_lora.end_to_end_serial.p50_ms"),
        ],
    ),
    "cascade_v1.md": (
        "cascade_benchmark.json",
        [
            ("encoder intent F1", "encoder",
             "baselines.encoder.intent_f1_macro"),
            ("qwen intent F1", "qwen_lora",
             "baselines.qwen_lora.intent_f1_macro"),
            ("rules intent F1", "rules",
             "baselines.rules.intent_f1_macro"),
        ],
    ),
    "quantization_v1.md": (
        "quantization_benchmark.json",
        [
            ("int8 体积", "onnx-cpu-int8", "variants.onnx-cpu-int8.footprint_mb"),
            ("int8 单请求 P50", "onnx-cpu-int8", "variants.onnx-cpu-int8.single_request.p50.p50_ms"),
        ],
    ),
    "confidence_diagnosis_v1.md": (
        "confidence_diagnosis.json",
        [
            ("encoder AUROC", "encoder", "branches.encoder.error_detection_auroc"),
            ("qwen AUROC", "qwen_lora", "branches.qwen_lora.error_detection_auroc"),
            ("rules AUROC", "rules", "branches.rules.error_detection_auroc"),
            ("encoder AURC", "encoder", "branches.encoder.aurc"),
        ],
    ),
}


def _variants(data: dict) -> dict:
    """量化报告的变体表在 `variants` 还是别的键下，取一下。"""
    for key in ("variants", "results", "branches"):
        node = data.get(key)
        if isinstance(node, dict):
            return node
    return {}


def main() -> None:
    problems: list[str] = []
    checked = 0

    for md_name, (json_name, specs) in MIRRORS.items():
        md_path = BENCH / "inference" / md_name
        json_path = BENCH / "reports" / json_name
        if not md_path.exists() or not json_path.exists():
            print(f"[skip] {md_name} / {json_name} 不存在")
            continue

        md_text = md_path.read_text(encoding="utf-8")
        data = json.loads(json_path.read_text(encoding="utf-8"))

        for label, row_key, dotted in specs:
            value = _get(data, dotted)
            if value is None:
                # 量化报告的结构不同，单独取
                parts = dotted.split(".")
                if parts[0] == "variants":
                    variant = _variants(data).get(parts[1], {})
                    value = variant
                    if not isinstance(value, dict) or "footprint_mb" not in variant:
                        continue
                else:
                    continue

            checked += 1
            wanted = {f"{float(value):.2f}", f"{float(value):.4f}", f"{float(value):g}",
                      str(value), f"{float(value):.1f}", f"{float(value):.3f}"}

            if row_key is None:
                # 全文里找一句带这个数字的结论句
                ok = any(
                    f"{float(value):.2f}" in line or f"{float(value):g}" in line
                    for line in md_text.splitlines()
                    if "P50" in line and "端到端" in line
                )
                if not ok:
                    problems.append(
                        f"{md_name}: 「{label}」应为 {value}（源 reports/{json_name}），md 里找不到"
                    )
                continue

            rows = _rows(md_text, row_key)
            if not rows:
                problems.append(f"{md_name}: 找不到以 `{row_key}` 开头的表格行")
                continue
            all_numbers: set[str] = set()
            for row in rows:
                all_numbers |= _numbers_in(row)
            if not (wanted & all_numbers):
                problems.append(
                    f"{md_name}: 「{label}」应为 {value}（源 reports/{json_name}），"
                    f"但 `{row_key}` 行里是 {sorted(all_numbers)}"
                )

    if problems:
        print(f"发现 {len(problems)} 处 inference 文档与源报告不一致：\n")
        for p in problems:
            print(f"  {p}")
        print("\n修复：python minimal-decision-bench/scripts/14_render_inference_docs.py")
        sys.exit(1)

    print(f"inference 镜像一致性检查通过（{checked} 个关键数字）")


if __name__ == "__main__":
    main()
