from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.data_builder import (
    assert_no_text_leakage,
    build_dataset,
    build_hard_cases,
    enumerate_templates,
    full_pool_complexity_floor,
    strip_suffix,
    subtype_of,
    unique_text_stats,
)
from minimal_decision_bench.io_utils import write_json, write_jsonl
from minimal_decision_bench.rules import decide


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reps-per-text",
        type=int,
        default=1,
        help="每条唯一文本在各自 split 内扩充的样本数（1 = 不重复）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_templates = sum(len(v) for v in enumerate_templates().values())
    train, dev, test = build_dataset(seed=args.seed, reps_per_text=args.reps_per_text)

    assert_no_text_leakage(train, dev, test)

    hard = build_hard_cases()
    write_jsonl(ROOT / "data" / "train.jsonl", train)
    write_jsonl(ROOT / "data" / "dev.jsonl", dev)
    write_jsonl(ROOT / "data" / "test.jsonl", test)
    write_jsonl(ROOT / "data" / "hard_cases.jsonl", hard)

    print("dataset created")
    print(f"模板池={total_templates}（按模板切分，再展开「模板 × 后缀」）")
    print(f"train={len(train)} dev={len(dev)} test={len(test)} hard={len(hard)}")
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        stats = unique_text_stats(rows)
        print(
            f"  {name}: rows={stats['rows']} "
            f"unique_texts={stats['unique_texts']} "
            f"unique_templates={stats['unique_templates']}"
        )

    print("\n难度层分布（explicit=含本类词 / conflict=含他类词 / synonym=无类别词）")
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        counter = collections.Counter(
            subtype_of(row["labels"]["intent"], strip_suffix(row["text"])) for row in rows
        )
        total = sum(counter.values()) or 1
        detail = " ".join(f"{k}={v}({100 * v / total:.0f}%)" for k, v in sorted(counter.items()))
        print(f"  {name}: {detail}")

    print("\n规则基线（rules_v0）在各 split 上的 intent 准确率")
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        hits = sum(decide(row["text"]).intent == row["labels"]["intent"] for row in rows)
        print(f"  {name}: {hits}/{len(rows)} = {hits / len(rows):.4f}")

    print("\nleakage check: OK（train/dev/test 的文本与模板集合均两两不相交）")

    # 复杂度标签的不可约噪声下限：MAE 要对齐它，不是对齐 0。
    # 这是数据集属性，与切分口径无关，所以从全模板池算一次。
    floor = full_pool_complexity_floor()
    write_json(ROOT / "reports" / "complexity_floor.json", floor)
    print("\ncomplexity 标签下限（reports/complexity_floor.json）")
    print(f"  随机猜          MAE = {floor['random_guess_mae']}")
    print(f"  只知道 intent   MAE = {floor['intent_only_mae']}")
    print(f"  不可约下限      MAE = {floor['oracle_mae']}  <- 模型最多做到这里")
    print("  说明：可学信号主要来自 intent，complexity 不是独立能力")


if __name__ == "__main__":
    main()
