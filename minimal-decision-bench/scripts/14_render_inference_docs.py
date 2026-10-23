"""从**已入库的** reports/*.json 重新渲染 inference/*.md。

为什么需要这个脚本
----------------
`inference/*.md` 和 `reports/*.json` 是同一个脚本的同一次运行写出来的
（见 06/10/11/12 各脚本的 `write_json(...)` + `write_text(_to_markdown(report))`）。
所以它们**必须一致**——但现实是：

- 报告后来重跑过（换机器、换 MPS 版本），json 更新了，md 没同步；
- 报告被纳入版本控制后，读者 clone 下来就能核对数字，
  于是 md 与 json 对不上这件事会立刻暴露。

直接重跑基准测试来刷新 md 是**错的**：那会连 reports/*.json 一起重写，
把已经发布并被文档引用的数字全部刷新一遍，文档立刻漂移。

正确做法：**只渲染，不重跑**——从磁盘上已入库的 json 重新生成 md。

用法
----
    python scripts/14_render_inference_docs.py
    python scripts/14_render_inference_docs.py --check   # 只检查是否一致
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# md 文件 -> (渲染函数所在的脚本, 源 json)
TARGETS: dict[str, tuple[str, str]] = {
    "benchmark_v1.md": ("06_benchmark_latency.py", "latency_benchmark.json"),
    "quantization_v1.md": ("10_quantize_benchmark.py", "quantization_benchmark.json"),
    "cascade_v1.md": ("11_cascade_benchmark.py", "cascade_benchmark.json"),
    "confidence_diagnosis_v1.md": ("12_confidence_diagnosis.py", "confidence_diagnosis.json"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="只检查，不写文件")
    return p.parse_args()


def _load_script(name: str):
    """按路径加载脚本模块（拿到它的 _to_markdown）。"""
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_render_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    drift: list[str] = []
    skipped: list[str] = []

    for md_name, (script_name, json_name) in TARGETS.items():
        md_path = ROOT / "inference" / md_name
        json_path = ROOT / "reports" / json_name
        if not json_path.exists():
            print(f"[skip] 缺少 reports/{json_name}")
            continue
        report = json.loads(json_path.read_text(encoding="utf-8"))
        try:
            module = _load_script(script_name)
        except ImportError as exc:  # 缺 torch / onnx 等可选依赖
            skipped.append(f"{md_name}（{script_name} 依赖缺失：{exc}）")
            if args.check:
                print(f"[skip] inference/{md_name}（{exc}）")
            continue
        rendered = module._to_markdown(report)

        if args.check:
            current = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
            if current != rendered:
                drift.append(md_name)
                print(f"[drift] inference/{md_name} 与 reports/{json_name} 不一致")
            else:
                print(f"[ok] inference/{md_name}")
            continue

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(rendered, encoding="utf-8")
        print(f"[write] inference/{md_name}（源：reports/{json_name}）")

    if skipped and not args.check:
        print(f"（跳过 {len(skipped)} 个：依赖缺失）")
    if args.check and drift:
        print(f"\n{len(drift)} 个 inference 文档与源报告不一致，跑："
              f"python scripts/14_render_inference_docs.py")
        sys.exit(1)
    if args.check:
        print("\n全部 inference 文档与源报告一致")


if __name__ == "__main__":
    main()
