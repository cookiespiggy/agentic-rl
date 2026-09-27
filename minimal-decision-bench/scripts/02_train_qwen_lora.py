from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.io_utils import read_jsonl, write_json
from minimal_decision_bench.metrics import (
    complexity_metrics,
    escalation_metrics,
    intent_metrics,
)
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.trainers import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    TaskConfig,
    add_tuning_args,
    branch_paths,
    ensure_local_model,
    infer_task,
    prediction_records,
    train_task,
)

BRANCH = "qwen_lora"

# 调优默认值（相对修复前的 bs=1 / lr=5e-5 / 无 warmup / 1 epoch）：
# - lr 1e-4：LoRA 常见区间，且分类头是新初始化的，需要比全量微调更大的步长
# - warmup 0.1：小数据 + 小 batch 下抑制早期梯度爆炸
# - grad_accum 4：等效 batch 4，降低 bs=1 的方差
# - cosine：比 linear 在后期衰减更平滑
DEFAULT_LR = 1e-4
DEFAULT_WARMUP = 0.1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_SCHEDULER = "cosine"
DEFAULT_EPOCHS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    # MPS 上 batch_size=1 是内存约束下的选择，靠梯度累积凑等效 batch；CUDA 上可直接调大。
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # 默认全量（None = 不切片）。旧默认值 128/48/72 会让两条分支口径不一致。
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-dev-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    add_tuning_args(p)
    p.set_defaults(
        lr=DEFAULT_LR,
        warmup_ratio=DEFAULT_WARMUP,
        grad_accum=DEFAULT_GRAD_ACCUM,
        scheduler=DEFAULT_SCHEDULER,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path if args.dry_run else ensure_local_model(args.model_path)
    artifact_dir, metrics_path, predictions_path = branch_paths(ROOT, BRANCH, args.run_tag)
    schema = load_schema(ROOT / "schemas" / "v1.json")
    labels = list(schema.intent_labels())
    label_to_id = {k: i for i, k in enumerate(labels)}
    id_to_label = {i: k for k, i in label_to_id.items()}

    train = read_jsonl(ROOT / "data" / "train.jsonl")
    dev = read_jsonl(ROOT / "data" / "dev.jsonl")
    test = read_jsonl(ROOT / "data" / "test.jsonl")
    train_enc = _take(_encode_rows(train, label_to_id), args.max_train_samples)
    dev_enc = _take(_encode_rows(dev, label_to_id), args.max_dev_samples)
    test_enc = _take(_encode_rows(test, label_to_id), args.max_test_samples)

    common = {
        "model_path": model_path,
        "lora": True,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "grad_accum_steps": args.grad_accum,
        "lr_scheduler_type": args.scheduler,
        "save_checkpoints": args.save_checkpoints,
    }
    tasks = [
        TaskConfig(
            name="intent",
            output_dir=str(artifact_dir / "intent"),
            num_labels=len(labels),
            label_key="intent_id",
            **common,
        ),
        TaskConfig(
            name="escalation",
            output_dir=str(artifact_dir / "escalation"),
            num_labels=2,
            label_key="needs_escalation_id",
            **common,
        ),
        TaskConfig(
            name="complexity",
            output_dir=str(artifact_dir / "complexity"),
            num_labels=1,
            label_key="complexity",
            is_regression=True,
            **common,
        ),
    ]

    if args.dry_run:
        print("dry-run mode")
        print(f"branch={BRANCH} run_tag={args.run_tag!r}")
        print(f"model_path={model_path}")
        print(f"samples train/dev/test={len(train_enc)}/{len(dev_enc)}/{len(test_enc)}")
        print(f"artifact_dir={artifact_dir}")
        print(
            f"lr={args.lr:g} warmup={args.warmup_ratio:g} "
            f"grad_accum={args.grad_accum} sched={args.scheduler} epochs={args.epochs} "
            f"save_checkpoints={args.save_checkpoints}"
        )
        for t in tasks:
            print(f"task={t.name} output={t.output_dir} lora={t.lora}")
        return

    for t in tasks:
        train_task(train_enc, dev_enc, t)

    intent_out = infer_task(
        test_enc, str(artifact_dir / "intent"), "intent_id", False,
        num_labels=len(labels), max_length=DEFAULT_MAX_LENGTH,
    )
    esc_out = infer_task(
        test_enc, str(artifact_dir / "escalation"), "needs_escalation_id", False,
        num_labels=2, max_length=DEFAULT_MAX_LENGTH,
    )
    comp_out = infer_task(
        test_enc, str(artifact_dir / "complexity"), "complexity", True,
        num_labels=1, max_length=DEFAULT_MAX_LENGTH,
    )

    intent_true = [id_to_label[i] for i in intent_out["labels"]]
    intent_pred = [id_to_label[i] for i in intent_out["preds"]]
    esc_true = [bool(v) for v in esc_out["labels"]]
    esc_pred = [bool(v) for v in esc_out["preds"]]
    # 必须是 P(正类)，不能传 top-class 置信度
    esc_prob_true = [float(p[1]) for p in esc_out["probs"]]

    report = {
        "branch": "qwen3.5-0.8b-base-lora",
        "run_tag": args.run_tag,
        "seed": args.seed,
        "model_path": model_path,
        "hparams": {
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "grad_accum": args.grad_accum,
            "scheduler": args.scheduler,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "save_checkpoints": args.save_checkpoints,
        },
        "samples": {"train": len(train_enc), "dev": len(dev_enc), "test": len(test_enc)},
        "intent": intent_metrics(intent_true, intent_pred, intent_out["confidences"]),
        "needs_escalation": escalation_metrics(esc_true, esc_pred, esc_prob_true),
        "complexity": complexity_metrics(comp_out["labels"], comp_out["preds"]),
    }
    write_json(metrics_path, report)
    write_json(
        predictions_path,
        prediction_records(test_enc, intent_out, esc_out, comp_out, id_to_label),
    )
    print(f"saved {metrics_path.relative_to(ROOT)}")
    print(report)


def _take(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def _encode_rows(rows: list[dict], label_to_id: dict[str, int]) -> list[dict]:
    out = []
    for r in rows:
        labels = r["labels"].copy()
        labels["intent_id"] = label_to_id[labels["intent"]]
        labels["needs_escalation_id"] = int(labels["needs_escalation"])
        out.append({"id": r["id"], "text": r["text"], "labels": labels})
    return out


if __name__ == "__main__":
    main()
