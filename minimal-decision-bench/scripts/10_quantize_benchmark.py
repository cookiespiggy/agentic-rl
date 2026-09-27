"""量化与 ONNX 推理基准（补 32 章标题里承诺、但一直未覆盖的「量化」部分）。

对比同一份 encoder 的多种推理形态：

| 变体 | 说明 |
|---|---|
| `torch-mps-fp32` | 当前基线（MPS + fp32） |
| `torch-cpu-fp32` | 同模型在 CPU 上的表现，用于**隔离设备差异** |
| `torch-cpu-int8` | PyTorch 动态量化（Linear 权重转 int8） |
| `onnx-cpu-fp32` | ONNX Runtime CPU |
| `onnx-cpu-int8` | ONNX Runtime 动态量化 |

**必须分组比较，否则结论是误导性的**：

- A 组（设备差异）：`torch-mps-fp32` vs `torch-cpu-fp32`——量化与 ONNX 都是 CPU 侧优化，
  拿它们直接跟 MPS 基线比，测出来的大半是设备差异而不是量化收益；
- B 组（同一设备上的形态差异）：CPU 上的 fp32 / int8 / ONNX 互比——这才是量化的真实收益。

每个变体独立 try/except：某个后端不可用不影响其它变体出数。

输出：`reports/quantization_benchmark.json` + `inference/quantization_v1.md`
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.io_utils import read_jsonl, write_json
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.trainers import DEFAULT_MAX_LENGTH, ensure_pad_token

ARTIFACT = ROOT / "artifacts" / "encoder" / "intent"
ONNX_DIR = ROOT / "artifacts" / "onnx"
REPEATS_SINGLE = 30
REPEATS_BATCH = 5
WARMUP = 3
BATCH_SIZES = (1, 8, 32)
TASK = "intent"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats-single", type=int, default=REPEATS_SINGLE)
    p.add_argument("--repeats-batch", type=int, default=REPEATS_BATCH)
    p.add_argument(
        "--force-export",
        action="store_true",
        help="即使 ONNX 已存在也重新导出（默认复用，导出一次约 10 秒）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not ARTIFACT.exists():
        raise SystemExit(f"缺少 {ARTIFACT}，请先跑 scripts/01_train_encoder.py")
    _warn_stale_temp_files()

    texts = [r["text"] for r in read_jsonl(ROOT / "data" / "test.jsonl")]

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ARTIFACT), trust_remote_code=True, use_fast=False
    )
    ensure_pad_token(tokenizer)
    buckets = _length_buckets(tokenizer, texts)

    report: dict = {
        "task": TASK,
        "protocol": {
            "max_length": DEFAULT_MAX_LENGTH,
            "repeats_single": args.repeats_single,
            "repeats_batch": args.repeats_batch,
            "warmup": WARMUP,
            "batch_sizes": list(BATCH_SIZES),
            "length_buckets": {k: v["tokens"] for k, v in buckets.items()},
            "note": (
                "量化与 ONNX 都是 CPU 侧优化，不能直接与 MPS 基线比；"
                "报告分「设备差异」与「同设备形态差异」两组"
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "onnxruntime": _ort_version(),
        },
        "variants": {},
        "unavailable": {},
    }

    variants = [
        ("torch-mps-fp32", lambda: _torch_variant(torch, "mps")),
        ("torch-cpu-fp32", lambda: _torch_variant(torch, "cpu")),
        ("torch-cpu-int8", lambda: _torch_int8_variant(torch)),
        ("onnx-cpu-fp32", lambda: _onnx_variant(False, args, texts)),
        ("onnx-cpu-int8", lambda: _onnx_variant(True, args, texts)),
    ]

    for name, factory in variants:
        print(f"\n=== {name} ===")
        try:
            predict, meta = factory()
        except Exception as exc:  # noqa: BLE001 - 逐变体降级，不让单点失败毁掉整份报告
            print(f"  不可用：{type(exc).__name__}: {exc}")
            report["unavailable"][name] = f"{type(exc).__name__}: {exc}"
            continue

        single = _measure_single(predict, buckets, args.repeats_single)
        batch = _measure_batch(predict, buckets["p50"]["text"], args.repeats_batch)
        report["variants"][name] = {
            **meta,
            "single_request": single,
            "batch_throughput": batch,
        }
        print(
            f"  p50={single['p50']['p50_ms']:9.3f} ms  "
            f"batch32_qps={batch['32']['qps']:10.1f}  size={meta['size_mb']} MB"
        )
        if "agreement" in meta:
            print(
                f"  量化一致性：argmax 一致率 {meta['agreement']['argmax_agreement']}，"
                f"最大 logit 偏差 {meta['agreement']['max_abs_logit_diff']}"
            )

    report["comparison"] = _comparison(report["variants"])
    out_json = ROOT / "reports" / "quantization_benchmark.json"
    write_json(out_json, report)
    out_md = ROOT / "inference" / "quantization_v1.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"\nsaved {out_json.relative_to(ROOT)}")
    print(f"saved {out_md.relative_to(ROOT)}")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


# --- 变体构造 ---------------------------------------------------------------


def _torch_variant(torch, device_name: str):
    """原生 PyTorch 推理（MPS 或 CPU）。"""
    from minimal_decision_bench.trainers import load_classifier

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用")
    model, tok, _ = load_classifier(str(ARTIFACT), _num_labels(), False)
    device = torch.device(device_name)
    model = model.to(device)
    model.eval()

    def predict(batch: list[str]):
        encoded = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            return model(**encoded).logits.detach().cpu().numpy()

    return predict, {"size_mb": _dir_size_mb(ARTIFACT)}


def _torch_int8_variant(torch):
    """PyTorch 动态量化：Linear 权重转 int8（只支持 CPU）。"""
    quantize_dynamic = getattr(torch.quantization, "quantize_dynamic", None)
    if quantize_dynamic is None:
        raise RuntimeError("当前 torch 没有 torch.quantization.quantize_dynamic")

    from minimal_decision_bench.trainers import load_classifier

    model, tok, _ = load_classifier(str(ARTIFACT), _num_labels(), False)
    model = model.to("cpu").eval()
    with torch.no_grad():
        quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    def predict(batch: list[str]):
        encoded = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        )
        with torch.no_grad():
            return quantized(**encoded).logits.detach().cpu().numpy()

    # 内存中量化，磁盘体积不变
    return predict, {"size_mb": _dir_size_mb(ARTIFACT)}


def _onnx_variant(int8: bool, args: argparse.Namespace, texts: list[str]):
    """ONNX Runtime 推理，可选动态量化。"""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    fp32_path = _onnx_fp32_path(args)
    onnx_path = _quantize_onnx(fp32_path) if int8 else fp32_path

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    tokenizer = AutoTokenizer.from_pretrained(
        str(ARTIFACT), trust_remote_code=True, use_fast=False
    )
    ensure_pad_token(tokenizer)

    def predict(batch: list[str]):
        encoded = tokenizer(
            batch, return_tensors="np", padding=True, truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        )
        feed = {
            name: encoded[name].astype(np.int64)
            for name in input_names
            if name in encoded
        }
        return session.run(None, feed)[0]

    meta: dict = {"size_mb": _dir_size_mb(onnx_path)}
    if int8:
        # 量化必须同时报精度损失，不能只报「更快更小」
        meta["agreement"] = _quantization_agreement(fp32_path, onnx_path, texts)
    return predict, meta


_FP32_CACHE: list[Path] = []


def _onnx_fp32_path(args: argparse.Namespace) -> Path:
    """导出只做一次——两个 ONNX 变体共用同一份 fp32 模型。"""
    if not _FP32_CACHE:
        _FP32_CACHE.append(_export_onnx(args))
    return _FP32_CACHE[0]


def _export_onnx(args: argparse.Namespace) -> Path:
    """把 encoder 导出成 ONNX（动态 batch/seq 轴）。"""
    import torch
    from transformers import AutoTokenizer

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    target = ONNX_DIR / "encoder_intent.onnx"
    if target.exists() and not args.force_export:
        print(f"  复用已有 ONNX -> {target.name}（{_dir_size_mb(target)} MB）")
        return target

    from minimal_decision_bench.trainers import load_classifier

    model, _, _ = load_classifier(str(ARTIFACT), _num_labels(), False)
    model = model.to("cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(ARTIFACT), trust_remote_code=True, use_fast=False
    )
    ensure_pad_token(tokenizer)
    dummy = tokenizer(
        ["导出占位"], return_tensors="pt", padding=True, truncation=True,
        max_length=DEFAULT_MAX_LENGTH,
    )
    # MacBERT 是 BERT 系，forward 需要 token_type_ids；不显式声明会导出成固定零张量
    inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "token_type_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(target),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
        )
    print(f"  已导出 ONNX -> {target.name}（{_dir_size_mb(target)} MB）")
    return target


def _quantize_onnx(fp32_path: Path) -> Path:
    """ONNX 动态量化（权重 int8）。

    onnxruntime 会在输入同目录写一个 `{input_stem}-inferred.onnx` 临时文件，并在结束时
    删掉它。这里有两个坑：

    1. **名字冲突**：若同名临时文件已存在（上次运行被中断留下），onnxruntime 会先去删它，
       而在带删除保护的环境里这一步可能被拦下，整个量化就失败。所以每次运行都用**带 PID
       的唯一暂存名**，保证临时文件名不与任何历史文件冲突。
    2. **清理被拦**：只要量化产物已落盘，清理临时文件失败就不该算失败——这里容忍它。
    """
    import os
    import shutil
    import subprocess

    from onnxruntime.quantization import QuantType  # noqa: F401 - 子进程里用到

    target = fp32_path.with_name(fp32_path.stem + "_int8.onnx")
    if target.exists():
        print(f"  复用已有量化模型 -> {target.name}（{_dir_size_mb(target)} MB）")
        return target

    staging = fp32_path.with_name(f"{fp32_path.stem}_q{os.getpid()}.onnx")
    shutil.copy2(fp32_path, staging)

    # 量化放在**子进程**里做：onnxruntime 的临时文件清理在带删除保护的环境里可能
    # 直接终止进程。隔离到子进程后，失败会变成一个可捕获的错误，而不是让整份报告
    # 连写都没写就丢掉。
    code = (
        "from onnxruntime.quantization import QuantType, quantize_dynamic;"
        f"quantize_dynamic({str(staging)!r}, {str(target)!r}, weight_type=QuantType.QInt8)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0 and not (target.exists() and target.stat().st_size > 0):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(
            f"量化子进程失败（exit {proc.returncode}）：" + " | ".join(tail)
        )
    if proc.returncode != 0:
        print(f"  [warn] 子进程 exit {proc.returncode}，但量化产物已生成，继续")

    print(f"  已量化 ONNX -> {target.name}（{_dir_size_mb(target)} MB）")
    return target


def _warn_stale_temp_files() -> None:
    """提示遗留的 `*-inferred.onnx` 临时文件（它们会触发上文的删除冲突）。"""
    if not ONNX_DIR.exists():
        return
    stale = sorted(ONNX_DIR.glob("*-inferred.onnx"))
    if stale:
        total = sum(p.stat().st_size for p in stale) / 1024 / 1024
        print(
            f"[warn] 发现 {len(stale)} 个遗留临时文件（共 {total:.0f} MB）："
            + ", ".join(p.name for p in stale)
        )
        print("       建议清理：rm -f artifacts/onnx/*-inferred.onnx")


def _quantization_agreement(fp32_path: Path, int8_path: Path, texts: list[str]) -> dict:
    """校验量化没把模型改坏：逐样本比对 int8 与 fp32 的 argmax 是否一致。

    量化基准只报「更快更小」是不负责任的——必须同时报精度损失。这里用
    **预测一致率**做最低限度校验（不重跑全量指标，成本可控）。
    """
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ARTIFACT), trust_remote_code=True, use_fast=False
    )
    ensure_pad_token(tokenizer)

    def run(path: Path):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = {i.name for i in session.get_inputs()}
        out = []
        for text in texts:
            encoded = tokenizer(
                [text], return_tensors="np", padding=True, truncation=True,
                max_length=DEFAULT_MAX_LENGTH,
            )
            feed = {n: encoded[n].astype(np.int64) for n in names if n in encoded}
            out.append(session.run(None, feed)[0][0])
        return np.asarray(out)

    fp32, int8 = run(fp32_path), run(int8_path)
    agree = float((fp32.argmax(axis=1) == int8.argmax(axis=1)).mean())
    return {
        "n": len(texts),
        "argmax_agreement": round(agree, 4),
        "max_abs_logit_diff": round(float(np.abs(fp32 - int8).max()), 4),
    }


# --- 计时 -------------------------------------------------------------------


def _measure_single(predict, buckets: dict, repeats: int) -> dict:
    out: dict = {}
    for name, bucket in buckets.items():
        timings = _time_call(lambda t=bucket["text"]: predict([t]), repeats)
        out[name] = {"repeats": repeats, **_percentiles(timings)}
    return out


def _measure_batch(predict, text: str, repeats: int) -> dict:
    out: dict = {}
    for size in BATCH_SIZES:
        batch = [text] * size
        timings = _time_call(lambda b=batch: predict(b), repeats)
        median = statistics.median(timings)
        out[str(size)] = {
            "batch_ms": round(median, 3),
            "per_sample_ms": round(median / size, 3),
            "qps": round(1000.0 / median * size, 1),
            "repeats": repeats,
        }
    return out


def _time_call(fn, repeats: int) -> list[float]:
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
        "p50_ms": round(_quantile(ordered, 0.50), 3),
        "p95_ms": round(_quantile(ordered, 0.95), 3),
        "p99_ms": round(_quantile(ordered, 0.99), 3),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


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


# --- 辅助 -------------------------------------------------------------------


def _num_labels() -> int:
    schema = load_schema(ROOT / "schemas" / "v1.json")
    return len(list(schema.intent_labels()))


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        return round(path.stat().st_size / 1024 / 1024, 1)
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / 1024 / 1024, 1)


def _ort_version() -> str:
    try:
        import onnxruntime

        return onnxruntime.__version__
    except ImportError:
        return "not installed"


def _comparison(variants: dict) -> dict:
    """分两组比较：设备差异 vs 同设备形态差异。"""
    out: dict = {"note": "量化/ONNX 是 CPU 侧优化，与 MPS 基线不可直接比"}

    def p50(name: str):
        entry = variants.get(name)
        return entry["single_request"]["p50"]["p50_ms"] if entry else None

    mps, cpu_fp32 = p50("torch-mps-fp32"), p50("torch-cpu-fp32")
    if mps and cpu_fp32:
        out["group_a_device"] = {
            "torch-mps-fp32_p50_ms": mps,
            "torch-cpu-fp32_p50_ms": cpu_fp32,
            "mps_speedup_over_cpu": round(cpu_fp32 / mps, 2),
        }

    group_b: dict = {}
    for name in ("torch-cpu-fp32", "torch-cpu-int8", "onnx-cpu-fp32", "onnx-cpu-int8"):
        entry = variants.get(name)
        if not entry:
            continue
        group_b[name] = {
            "p50_ms": entry["single_request"]["p50"]["p50_ms"],
            "batch32_qps": entry["batch_throughput"]["32"]["qps"],
            "size_mb": entry["size_mb"],
        }
    if cpu_fp32 and group_b:
        base = group_b.get("torch-cpu-fp32", {}).get("p50_ms")
        for name, entry in group_b.items():
            if base:
                entry["speedup_vs_torch_cpu_fp32"] = round(base / entry["p50_ms"], 2)
    out["group_b_same_device"] = group_b
    return out


def _to_markdown(report: dict) -> str:
    p = report["protocol"]
    lines = [
        "# 量化与 ONNX 推理基准 v1",
        "",
        f"- 任务：`{report['task']}`　max_length={p['max_length']}",
        f"- 单请求重复 {p['repeats_single']} 次（预热 {p['warmup']} 次不计入），批吞吐重复 {p['repeats_batch']} 次",
        f"- 硬件：{report['hardware']['platform']} / {report['hardware']['machine']}",
        f"- torch `{report['hardware']['torch']}`　onnxruntime `{report['hardware']['onnxruntime']}`",
        "",
        f"> ⚠️ {p['note']}",
        "",
        "## 单请求延迟（毫秒，p50 长度档）",
        "",
        "| 变体 | P50 | P95 | P99 | 体积 MB |",
        "|---|---|---|---|---|",
    ]
    for name, entry in report["variants"].items():
        m = entry["single_request"]["p50"]
        lines.append(
            f"| `{name}` | {m['p50_ms']} | {m['p95_ms']} | {m['p99_ms']} | {entry['size_mb']} |"
        )

    lines += ["", "## 批吞吐", "", "| 变体 | 批大小 | 批延迟 ms | 单样本 ms | QPS |", "|---|---|---|---|---|"]
    for name, entry in report["variants"].items():
        for size, m in entry["batch_throughput"].items():
            lines.append(
                f"| `{name}` | {size} | {m['batch_ms']} | {m['per_sample_ms']} | {m['qps']} |"
            )

    agreement = {
        name: entry["agreement"]
        for name, entry in report["variants"].items()
        if "agreement" in entry
    }
    if agreement:
        lines += [
            "",
            "## 量化一致性校验（int8 vs fp32）",
            "",
            "| 变体 | 样本数 | argmax 一致率 | 最大 logit 偏差 |",
            "|---|---|---|---|",
        ]
        for name, a in agreement.items():
            lines.append(
                f"| `{name}` | {a['n']} | {a['argmax_agreement']} | {a['max_abs_logit_diff']} |"
            )
        lines += [
            "",
            "> 量化基准只报「更快更小」是不负责任的，必须同时报精度损失。",
            "",
        ]

    cmp_ = report.get("comparison", {})
    lines += ["", "## 分组对比", ""]
    if "group_a_device" in cmp_:
        a = cmp_["group_a_device"]
        lines += [
            "### A 组：设备差异（同 fp32）",
            "",
            (
                f"- MPS `{a['torch-mps-fp32_p50_ms']} ms` vs CPU `{a['torch-cpu-fp32_p50_ms']} ms`"
                f" → **MPS 快 {a['mps_speedup_over_cpu']} 倍**"
            ),
            "",
        ]
    if "group_b_same_device" in cmp_:
        lines += [
            "### B 组：同设备（CPU）上的形态差异",
            "",
            "| 变体 | P50 ms | 相对 torch-cpu-fp32 | 批32 QPS | 体积 MB |",
            "|---|---|---|---|---|",
        ]
        for name, entry in cmp_["group_b_same_device"].items():
            speedup = entry.get("speedup_vs_torch_cpu_fp32")
            lines.append(
                f"| `{name}` | {entry['p50_ms']} | {speedup}x | "
                f"{entry['batch32_qps']} | {entry['size_mb']} |"
            )
        lines.append("")

    if report.get("unavailable"):
        lines += ["## 不可用的变体", "", "（完整原因见 `reports/quantization_benchmark.json`）", ""]
        for name, reason in report["unavailable"].items():
            short = reason if len(reason) <= 120 else reason[:117] + "..."
            lines.append(f"- `{name}`：{short}")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
