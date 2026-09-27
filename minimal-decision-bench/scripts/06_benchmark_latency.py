"""推理延迟与吞吐基准（对应主仓库 32 章）。

固定压测口径，否则对比无效：
- 请求长度：test 集 token 长度的 p50 / p95 各一组
- 单请求延迟：预热后重复 N 次，报 P50 / P95 / P99
- 吞吐：batch 1 / 8 / 32，报批延迟、单样本延迟与 QPS
- 端到端：一次请求要串行跑 intent + escalation + complexity 三个模型，故给出三者之和
- 硬件与精度写入报告，跨机器对比前必须核对

规则基线（rules_v0）没有模型，只测关键词扫描本身的耗时，作为成本下界。

输出：reports/latency_benchmark.json + inference/benchmark_v1.md
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench import rules
from minimal_decision_bench.io_utils import read_json, read_jsonl, write_json
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.trainers import (
    DEFAULT_MAX_LENGTH,
    batched_logits,
    branch_paths,
    load_classifier,
)

MODEL_BRANCHES = ("encoder", "qwen_lora")
BRANCHES = ("encoder", "qwen_lora", "rules")
TASKS = (
    ("intent", "intent_id", False),
    ("escalation", "needs_escalation_id", False),
    ("complexity", "complexity", True),
)
BATCH_SIZES = (1, 8, 32)
REPEATS_SINGLE = 30
REPEATS_BATCH = 5
WARMUP = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default="")
    p.add_argument("--repeats-single", type=int, default=REPEATS_SINGLE)
    p.add_argument("--repeats-batch", type=int, default=REPEATS_BATCH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(ROOT / "schemas" / "v1.json")
    num_labels_map = {
        "intent": len(list(schema.intent_labels())),
        "escalation": 2,
        "complexity": 1,
    }
    texts = [r["text"] for r in read_jsonl(ROOT / "data" / "test.jsonl")]

    report: dict = {
        "protocol": {
            "max_length": DEFAULT_MAX_LENGTH,
            "repeats_single": args.repeats_single,
            "repeats_batch": args.repeats_batch,
            "warmup": WARMUP,
            "batch_sizes": list(BATCH_SIZES),
            "length_buckets": "test 集 token 长度的 p50 / p95",
            "device": _device_name(),
            "note": "批大小用于近似并发档位；MPS 上无法真并发，跨机器对比前必须核对硬件",
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "branches": {},
    }

    for branch in MODEL_BRANCHES:
        artifact_dir = branch_paths(ROOT, branch, args.run_tag)[0]
        if not (artifact_dir / "intent").exists():
            print(f"[warn] 缺少 {artifact_dir}/intent，跳过 {branch}（请先跑 01/02 脚本）")
            continue
        print(f"\n=== benchmarking {branch} ===")
        branch_report: dict = {"tasks": {}, "footprint_mb": {}}
        per_task_p50: list[float] = []

        for task_name, _, is_regression in TASKS:
            model_dir = artifact_dir / task_name
            model, tokenizer, device = load_classifier(
                str(model_dir), num_labels_map[task_name], is_regression
            )
            branch_report["footprint_mb"][task_name] = _inference_footprint_mb(model_dir)
            buckets = _length_buckets(tokenizer, texts)

            def forward(batch: list[str], _m=model, _t=tokenizer, _d=device) -> None:
                batched_logits(_m, _t, batch, _d, DEFAULT_MAX_LENGTH)

            task_report = _measure(forward, buckets, args)
            branch_report["tasks"][task_name] = task_report
            per_task_p50.append(task_report["single_request"]["p50"]["p50_ms"])
            print(
                f"  {task_name:11s} p50={task_report['single_request']['p50']['p50_ms']:8.2f} ms "
                f"batch32_qps={task_report['batch_throughput']['32']['qps']:8.1f}"
            )
            del model

        branch_report["end_to_end_serial"] = {
            "p50_ms": round(sum(per_task_p50), 2),
            "note": "三个任务模型串行执行的总延迟",
        }
        report["branches"][branch] = branch_report

    report["branches"]["rules"] = _benchmark_rules(texts, args)

    report["comparison"] = _comparison(report["branches"])
    out_json = ROOT / "reports" / f"latency_benchmark{args.run_tag}.json"
    write_json(out_json, report)
    out_md = ROOT / "inference" / f"benchmark_v1{args.run_tag}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"\nsaved {out_json.relative_to(ROOT)}")
    print(f"saved {out_md.relative_to(ROOT)}")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


def _measure(fn: Callable[[list[str]], None], buckets: dict, args) -> dict:
    """对给定的前向函数测单请求延迟与批吞吐。"""
    task_report: dict = {"length_buckets": {k: v["tokens"] for k, v in buckets.items()}}
    for bucket_name, bucket in buckets.items():
        timings = _time_call(
            lambda b=bucket["text"]: fn([b]), args.repeats_single
        )
        task_report.setdefault("single_request", {})[bucket_name] = {
            "repeats": args.repeats_single,
            **_percentiles(timings),
        }
    task_report["batch_throughput"] = _measure_batch(fn, buckets["p50"]["text"], args.repeats_batch)
    return task_report


def _measure_batch(fn: Callable[[list[str]], None], text: str, repeats: int) -> dict:
    out: dict = {}
    for batch_size in BATCH_SIZES:
        batch = [text] * batch_size
        timings = _time_call(lambda b=batch: fn(b), repeats)
        batch_ms = statistics.median(timings)
        out[str(batch_size)] = {
            "batch_ms": round(batch_ms, 3),
            "per_sample_ms": round(batch_ms / batch_size, 3),
            "qps": round(1000.0 / batch_ms * batch_size, 1),
            "repeats": repeats,
        }
    return out


def _benchmark_rules(texts: list[str], args) -> dict:
    """规则基线没有模型：只测关键词扫描本身。三个任务共用同一次扫描。"""
    print("\n=== benchmarking rules ===")
    buckets = _rule_buckets(texts)

    def scan(batch: list[str]) -> None:
        for text in batch:
            rules.decide(text)

    task_report = _measure(scan, buckets, args)
    p50 = task_report["single_request"]["p50"]["p50_ms"]
    print(f"  rules       p50={p50:8.4f} ms batch32_qps={task_report['batch_throughput']['32']['qps']:8.1f}")
    return {
        "tasks": {name: task_report for name, _, _ in TASKS},
        "footprint_mb": {"intent": 0.0, "escalation": 0.0, "complexity": 0.0},
        "end_to_end_serial": {
            "p50_ms": round(p50, 4),
            "note": "规则一次扫描同时产出三个任务的判定，无需串行跑三次",
        },
    }


def _rule_buckets(texts: list[str]) -> dict:
    """规则不看 tokenizer，用字符长度近似 p50 / p95 长度档。"""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    p50_i = order[len(order) // 2]
    p95_i = order[min(len(order) - 1, int(len(order) * 0.95))]
    return {
        "p50": {"text": texts[p50_i], "tokens": len(texts[p50_i])},
        "p95": {"text": texts[p95_i], "tokens": len(texts[p95_i])},
    }


def _time_call(fn: Callable[[], None], repeats: int) -> list[float]:
    for _ in range(WARMUP):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "p50_ms": round(_quantile(ordered, 0.50), 4),
        "p95_ms": round(_quantile(ordered, 0.95), 4),
        "p99_ms": round(_quantile(ordered, 0.99), 4),
        "mean_ms": round(statistics.fmean(values), 4),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _length_buckets(tokenizer, texts: list[str]) -> dict:
    lengths = [
        len(tokenizer(t, truncation=True, max_length=DEFAULT_MAX_LENGTH)["input_ids"])
        for t in texts
    ]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    p50_i = order[len(order) // 2]
    p95_i = order[min(len(order) - 1, int(len(order) * 0.95))]
    return {
        "p50": {"text": texts[p50_i], "tokens": lengths[p50_i]},
        "p95": {"text": texts[p95_i], "tokens": lengths[p95_i]},
    }


def _dir_size_mb(path: Path, skip_checkpoints: bool = False) -> float:
    total = 0
    for p in path.rglob("*"):
        if not p.is_file() or p.name.startswith("optimizer"):
            continue
        rel = p.relative_to(path)
        if skip_checkpoints and any(part.startswith("checkpoint-") for part in rel.parts):
            continue
        total += p.stat().st_size
    return round(total / 1024 / 1024, 1)


def _inference_footprint_mb(model_dir: Path) -> dict:
    """推理部署体积。

    PEFT 分支只存了 adapter，部署时还要带上底座，否则会得出「LoRA 比 encoder 小」
    的错误结论。这里把底座一并计入。
    """
    own = _dir_size_mb(model_dir, skip_checkpoints=True)
    base_mb = 0.0
    adapter_cfg = model_dir / "adapter_config.json"
    if adapter_cfg.exists():
        base_path = Path(read_json(adapter_cfg)["base_model_name_or_path"])
        if base_path.exists():
            base_mb = _dir_size_mb(base_path, skip_checkpoints=True)
    return {
        "own_mb": own,
        "base_mb": base_mb,
        "total_mb": round(own + base_mb, 1),
    }


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except (ImportError, RuntimeError, AttributeError):
        return "unknown"


def _comparison(branches: dict) -> dict:
    missing = [b for b in BRANCHES if b not in branches]
    if missing:
        return {"note": f"缺少分支 {missing}，跑完再对比"}
    enc = branches["encoder"]
    qwen = branches["qwen_lora"]
    rule = branches["rules"]
    enc_p50 = enc["end_to_end_serial"]["p50_ms"]
    qwen_p50 = qwen["end_to_end_serial"]["p50_ms"]
    rule_p50 = rule["end_to_end_serial"]["p50_ms"]
    return {
        "end_to_end_p50_ms": {
            "encoder": enc_p50,
            "qwen_lora": qwen_p50,
            "rules": rule_p50,
        },
        "latency_ratio_vs_rules": {
            "encoder": round(enc_p50 / rule_p50, 1) if rule_p50 else None,
            "qwen_lora": round(qwen_p50 / rule_p50, 1) if rule_p50 else None,
        },
        "batch32_qps": {
            "encoder": enc["tasks"]["intent"]["batch_throughput"]["32"]["qps"],
            "qwen_lora": qwen["tasks"]["intent"]["batch_throughput"]["32"]["qps"],
            "rules": rule["tasks"]["intent"]["batch_throughput"]["32"]["qps"],
        },
        "intent_model_footprint_mb": {
            "encoder": enc["footprint_mb"]["intent"]["total_mb"],
            "qwen_lora": qwen["footprint_mb"]["intent"]["total_mb"],
            "rules": 0.0,
        },
    }


def _to_markdown(report: dict) -> str:
    p = report["protocol"]
    lines = [
        "# 推理延迟与吞吐基准 v1",
        "",
        f"- 设备：`{p['device']}`　max_length={p['max_length']}",
        f"- 单请求重复 {p['repeats_single']} 次（预热 {p['warmup']} 次不计入），批吞吐重复 {p['repeats_batch']} 次",
        f"- 批大小 {p['batch_sizes']}（用于近似并发档位）",
        f"- 硬件：{report['hardware']['platform']} / {report['hardware']['machine']}",
        "",
        "> 跨机器对比前必须核对硬件与精度；MPS 上无法真并发，批大小只是近似。",
        "",
    ]
    for branch, data in report["branches"].items():
        lines += [f"## {branch}", ""]
        lines += [
            "### 单请求延迟（毫秒）",
            "",
            "| 任务 | 长度档 | tokens | P50 | P95 | P99 |",
            "|---|---|---|---|---|---|",
        ]
        for task, tr in data["tasks"].items():
            for bucket, m in tr["single_request"].items():
                lines.append(
                    f"| {task} | {bucket} | {tr['length_buckets'][bucket]} | "
                    f"{m['p50_ms']} | {m['p95_ms']} | {m['p99_ms']} |"
                )
        lines += ["", "### 批吞吐（p50 长度档）", "", "| 任务 | 批大小 | 批延迟 ms | 单样本 ms | QPS |", "|---|---|---|---|---|"]
        for task, tr in data["tasks"].items():
            for bs, m in tr["batch_throughput"].items():
                lines.append(
                    f"| {task} | {bs} | {m['batch_ms']} | {m['per_sample_ms']} | {m['qps']} |"
                )
        lines += [
            "",
            (
                f"**端到端串行总延迟 P50 = {data['end_to_end_serial']['p50_ms']} ms**"
                f"　（{data['end_to_end_serial']['note']}）"
            ),
            "",
            f"模型磁盘占用（不含 optimizer）：`{data['footprint_mb']}` MB",
            "",
        ]
    cmp_ = report.get("comparison", {})
    if "end_to_end_p50_ms" in cmp_:
        e2e = cmp_["end_to_end_p50_ms"]
        qps = cmp_["batch32_qps"]
        fp = cmp_["intent_model_footprint_mb"]
        ratio = cmp_["latency_ratio_vs_rules"]
        lines += [
            "## 三方对比",
            "",
            "| 指标 | encoder-only | Qwen LoRA | 关键词规则 |",
            "|---|---|---|---|",
            f"| 端到端 P50 | {e2e['encoder']} ms | {e2e['qwen_lora']} ms | {e2e['rules']} ms |",
            f"| 相对规则的倍数 | {ratio['encoder']}x | {ratio['qwen_lora']}x | 1x |",
            f"| intent 批 32 QPS | {qps['encoder']} | {qps['qwen_lora']} | {qps['rules']} |",
            f"| 部署体积 | {fp['encoder']} MB | {fp['qwen_lora']} MB | 0 MB（无需模型） |",
            "",
            "**成本是「判别外置」论证的关键一半**：质量上三方互有胜负（见 26 章与 README），",
            "但推理成本差一到两个量级。规则基线的存在还说明另一件事——",
            "**先证明模型确实比规则好，再谈模型选型**；否则最该省的不是模型，是模型本身。",
            "",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
