"""按模板分组的 K 折交叉验证。

**为什么需要它**：单次切分下 test 只有 24 条模板 / 96 条样本，encoder 的 F1 标准差
0.034，叠加 MPS 非确定性后噪声更大——分支间的细微差距（例如 encoder 与 Qwen 的
intent F1 只差 0.034）根本无法判定。K 折让**每条模板都被评测一次**，等效样本量
翻 K 倍，而且不需要新写一条模板。

设计要点：

- 折分配按 (intent × 难度层) 分层，保证每折的难度构成一致，折间方差才有意义；
- 每折的训练集来自其余 K-1 折，测试集是该折，模板层面完全隔离（无泄漏）；
- 没有独立 dev 集，`eval_dataset` 取训练集的一个切片，仅用于记录 eval_loss
  （本流程不做早停或最优 checkpoint 选择，所以不影响结论）；
- 每折跑完立刻落盘，长时间运行中断也不会丢结果；
- 规则基线是确定性的，作为不随折变化的参照。

用法：

    python scripts/08_cross_validate.py --folds 5
    python scripts/08_cross_validate.py --folds 5 --branches encoder,rules  # 跳过较慢的 LoRA
    python scripts/08_cross_validate.py --folds 5 --eval-only               # 沿用已有模型，只重跑推理
    python scripts/08_cross_validate.py --folds 5 --report-only             # 只重算聚合
    python scripts/08_cross_validate.py --folds 5 --force                   # 忽略缓存重新训练

改指标口径或补字段时用 `--eval-only`：折模型留在 `artifacts/cv_fold*/`，几分钟就能重算，
不必再花 30 多分钟训练。
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench import rules
from minimal_decision_bench.data_builder import (
    build_folds,
    expand_templates,
    fold_split,
    full_pool_complexity_floor,
    strip_suffix,
    subtype_of,
)
from minimal_decision_bench.io_utils import read_json, write_json
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
    ensure_local_model,
    infer_task,
    prediction_records,
    train_task,
)

MODEL_BRANCHES = ("encoder", "qwen_lora")
ALL_BRANCHES = ("encoder", "qwen_lora", "rules")
TASKS = (
    ("intent", "intent_id", False, "intent"),
    ("escalation", "needs_escalation_id", False, "escalation"),
    ("complexity", "complexity", True, "complexity"),
)

# 与 01/02 脚本保持一致的超参，保证 CV 结论能迁移到主流程
ENCODER_HPARAMS = {"batch_size": 8, "learning_rate": 5e-5, "warmup_ratio": 0.0,
                   "grad_accum_steps": 1, "lr_scheduler_type": "linear"}
QWEN_HPARAMS = {"batch_size": 1, "learning_rate": 1e-4, "warmup_ratio": 0.1,
                "grad_accum_steps": 4, "lr_scheduler_type": "cosine"}
EPOCHS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--branches", default=",".join(ALL_BRANCHES))
    p.add_argument("--encoder-model-path", default="hfl/chinese-macbert-base")
    p.add_argument("--qwen-model-path", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--run-tag", default="")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="不训练，只用已有模型重跑推理（改指标口径或补字段时用，几分钟即可）",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="连推理也不跑，只从已有逐折结果重算聚合",
    )
    p.add_argument("--force", action="store_true", help="忽略已有结果，重新训练")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    branches = [b.strip() for b in args.branches.split(",") if b.strip()]
    unknown = [b for b in branches if b not in ALL_BRANCHES]
    if unknown:
        raise SystemExit(f"未知分支：{unknown}，可选 {ALL_BRANCHES}")

    schema = load_schema(ROOT / "schemas" / "v1.json")
    labels = list(schema.intent_labels())
    label_to_id = {k: i for i, k in enumerate(labels)}
    id_to_label = {i: k for k, i in label_to_id.items()}

    folds = build_folds(n_folds=args.folds, seed=args.seed)
    out_path = ROOT / "reports" / f"cross_validation{args.run_tag}.json"

    # 必须复用已有结果：--report-only / --eval-only 的用途就是「不重训」。
    # 早期版本在这里用空 state 覆盖了报告，把逐折结果清空过——别再犯。
    state: dict = {"folds": args.folds, "seed": args.seed, "branches": branches,
                   "results": {}, "summary": {}}
    if out_path.exists():
        state["results"] = read_json(out_path).get("results", {})
    state.update({"folds": args.folds, "seed": args.seed, "branches": branches})

    if args.report_only and not state["results"]:
        raise SystemExit(
            f"--report-only 需要已有的逐折结果，但 {out_path.name} 里没有。\n"
            "  若折模型还在 artifacts/cv_fold*/，用 --eval-only 只重跑推理；\n"
            "  否则去掉该参数重新训练。"
        )

    if not args.report_only:
        for k in range(args.folds):
            cached = state["results"].get(str(k))
            if cached and not args.force and not args.eval_only:
                print(f"[fold {k}] 已有结果，跳过（--force 重训 / --eval-only 只重跑推理）")
                continue
            train_t, test_t = fold_split(folds, k)
            train_rows = _encode(
                expand_templates(train_t, 1, args.seed, f"cv{k}-train"), label_to_id
            )
            test_rows = _encode(
                expand_templates(test_t, 1, args.seed, f"cv{k}-test"), label_to_id
            )
            mode = "只重跑推理" if args.eval_only else "训练 + 评测"
            print(
                f"\n=== fold {k}/{args.folds}（{mode}）=== "
                f"train={len(train_rows)} test={len(test_rows)} rows"
            )
            fold_result: dict = {"n_train": len(train_rows), "n_test": len(test_rows),
                                 "branches": {}}
            for branch in branches:
                fold_result["branches"][branch] = _evaluate(
                    branch, k, train_rows, test_rows, args, labels, id_to_label,
                    train=not args.eval_only,
                )
            state["results"][str(k)] = fold_result
            write_json(out_path, state)  # 增量落盘
            print(f"[fold {k}] 已落盘 -> {out_path.relative_to(ROOT)}")

    # 硬防护：绝不写出没有逐折结果的报告。
    # 早期版本在 --report-only 分支里用空 state 覆盖过 cross_validation.json，
    # 把 30 多分钟的训练结果清空了——这类「空结果覆盖好结果」的 bug 必须由代码挡住，
    # 不能靠人记得。
    if not state["results"]:
        raise SystemExit(
            "没有逐折结果，拒绝写出空报告（否则会覆盖已有结果）。\n"
            "  折模型在 artifacts/cv_fold*/ 时用 --eval-only；否则去掉 --report-only 重新训练。"
        )

    state["summary"] = _summarize(state, folds, id_to_label)
    write_json(out_path, state)
    print(f"\nsaved {out_path.relative_to(ROOT)}")
    _print_summary(state["summary"])


def _evaluate(
    branch: str,
    fold: int,
    train_rows: list[dict],
    test_rows: list[dict],
    args: argparse.Namespace,
    labels: list[str],
    id_to_label: dict[int, str],
    train: bool = True,
) -> dict:
    if branch == "rules":
        intent_out = rules.infer_rows(test_rows, "intent_id", intent_labels=labels)
        esc_out = rules.infer_escalation(test_rows)
        comp_out = rules.infer_complexity(test_rows)
    else:
        artifact_dir = ROOT / "artifacts" / f"cv_fold{fold}" / branch
        if train:
            _train_branch(branch, artifact_dir, train_rows, args, labels)
        elif not (artifact_dir / "intent").exists():
            raise SystemExit(
                f"--eval-only 需要已有模型，但 {artifact_dir}/intent 不存在。"
                "去掉该参数重新训练。"
            )
        intent_out = infer_task(
            test_rows, str(artifact_dir / "intent"), "intent_id", False,
            num_labels=len(labels), max_length=DEFAULT_MAX_LENGTH,
        )
        esc_out = infer_task(
            test_rows, str(artifact_dir / "escalation"), "needs_escalation_id", False,
            num_labels=2, max_length=DEFAULT_MAX_LENGTH,
        )
        comp_out = infer_task(
            test_rows, str(artifact_dir / "complexity"), "complexity", True,
            num_labels=1, max_length=DEFAULT_MAX_LENGTH,
        )

    intent_true = [id_to_label[i] for i in intent_out["labels"]]
    intent_pred = [id_to_label[i] for i in intent_out["preds"]]
    esc_true = [bool(v) for v in esc_out["labels"]]
    esc_pred = [bool(v) for v in esc_out["preds"]]
    esc_prob_true = [float(p[1]) for p in esc_out["probs"]]

    metrics = {
        "intent": intent_metrics(intent_true, intent_pred, intent_out["confidences"]),
        "needs_escalation": escalation_metrics(esc_true, esc_pred, esc_prob_true),
        "complexity": complexity_metrics(comp_out["labels"], comp_out["preds"]),
    }
    records = prediction_records(test_rows, intent_out, esc_out, comp_out, id_to_label)
    for record in records:
        record["layer"] = subtype_of(
            record["intent_true"], strip_suffix(record["text"])
        )
    if branch == "rules":
        # 规则分支额外落盘 trace：11_cascade_benchmark 要靠它判断「是否命中」
        for record, trace in zip(records, intent_out["traces"], strict=True):
            record["rule_trace"] = trace
    return {"metrics": metrics, "records": records}


def _train_branch(
    branch: str,
    artifact_dir: Path,
    train_rows: list[dict],
    args: argparse.Namespace,
    labels: list[str],
) -> None:
    if branch == "qwen_lora":
        model_path = ensure_local_model(args.qwen_model_path)
        hparams, use_lora = QWEN_HPARAMS, True
    else:
        model_path = ensure_local_model(args.encoder_model_path)
        hparams, use_lora = ENCODER_HPARAMS, False

    # eval_dataset 仅用于记录 eval_loss：取训练集切片，绝不碰测试折
    dev_rows = train_rows[: max(1, len(train_rows) // 8)]

    specs = (
        ("intent", len(labels), "intent_id", False),
        ("escalation", 2, "needs_escalation_id", False),
        ("complexity", 1, "complexity", True),
    )
    for name, num_labels, label_key, is_regression in specs:
        train_task(
            train_rows,
            dev_rows,
            TaskConfig(
                name=name,
                model_path=model_path,
                output_dir=str(artifact_dir / name),
                num_labels=num_labels,
                label_key=label_key,
                is_regression=is_regression,
                lora=use_lora,
                epochs=args.epochs,
                seed=args.seed,
                **hparams,
            ),
        )


def _summarize(state: dict, folds: list, id_to_label: dict[int, str]) -> dict:
    branches = state["branches"]
    fold_ids = sorted(state["results"], key=int)
    summary: dict = {"folds_done": len(fold_ids), "branches": {}}

    for branch in branches:
        per_fold_metrics: dict[str, list[float]] = collections.defaultdict(list)
        pooled_records: list[dict] = []
        for fid in fold_ids:
            entry = state["results"][fid]["branches"].get(branch)
            if not entry:
                continue
            metrics = entry["metrics"]
            per_fold_metrics["intent.f1_macro"].append(metrics["intent"]["f1_macro"])
            per_fold_metrics["intent.ece"].append(metrics["intent"]["ece_top_confidence"])
            per_fold_metrics["escalation.f1"].append(metrics["needs_escalation"]["f1"])
            per_fold_metrics["escalation.brier"].append(metrics["needs_escalation"]["brier"])
            per_fold_metrics["complexity.mae"].append(metrics["complexity"]["mae"])
            pooled_records.extend(entry["records"])

        summary["branches"][branch] = {
            "per_fold": {
                key: {
                    "mean": round(statistics.fmean(values), 4),
                    "std": round(statistics.pstdev(values), 4),
                    "values": [round(v, 4) for v in values],
                }
                for key, values in per_fold_metrics.items()
            },
            "pooled": _pooled_metrics(pooled_records),
            "per_layer": _per_layer_metrics(pooled_records),
            "per_intent": _per_intent_metrics(pooled_records),
        }

    # 复杂度下限是数据集属性，与切分口径无关，所以从全模板池算一次。
    # （早期版本在这里对 pooled 记录重算，数值虽然巧合一致，但语义上依赖了
    #  「CV 的 pooled 恰好覆盖全池」这个隐含前提。）
    summary["complexity_floor"] = full_pool_complexity_floor()
    return summary


def _pooled_metrics(records: list[dict]) -> dict:
    if not records:
        return {}
    return {
        "n": len(records),
        "intent": intent_metrics(
            [r["intent_true"] for r in records],
            [r["intent_pred"] for r in records],
            [r["intent_confidence"] for r in records],
        ),
        "needs_escalation": escalation_metrics(
            [r["escalation_true"] for r in records],
            [r["escalation_pred"] for r in records],
            [r["escalation_prob_true"] for r in records],
        ),
        "complexity": complexity_metrics(
            [r["complexity_true"] for r in records],
            [r["complexity_pred"] for r in records],
        ),
    }


def _per_intent_metrics(records: list[dict]) -> dict:
    """按类别拆开 precision / recall / F1。

    整体 macro-F1 会掩盖结构性问题——例如兜底类 `general` 的**召回率**最低，
    说明模型倾向把该兜底的样本强行塞进具体类别（静默错误）。
    """
    labels = sorted({record["intent_true"] for record in records})
    out: dict = {}
    for label in labels:
        tp = sum(
            1 for r in records if r["intent_true"] == label and r["intent_pred"] == label
        )
        fp = sum(
            1 for r in records if r["intent_true"] != label and r["intent_pred"] == label
        )
        fn = sum(
            1 for r in records if r["intent_true"] == label and r["intent_pred"] != label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        out[label] = {
            "n": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return out


def _per_layer_metrics(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        groups[record["layer"]].append(record)
    return {
        layer: {
            "n": len(rows),
            "intent_accuracy": round(
                sum(r["intent_pred"] == r["intent_true"] for r in rows) / len(rows), 4
            ),
            "complexity_mae": round(
                sum(abs(r["complexity_pred"] - r["complexity_true"]) for r in rows)
                / len(rows),
                4,
            ),
        }
        for layer, rows in sorted(groups.items())
    }


def _encode(rows: list[dict], label_to_id: dict[str, int]) -> list[dict]:
    out = []
    for r in rows:
        labels = r["labels"].copy()
        labels["intent_id"] = label_to_id[labels["intent"]]
        labels["needs_escalation_id"] = int(labels["needs_escalation"])
        out.append({"id": r["id"], "text": r["text"], "labels": labels})
    return out


def _print_summary(summary: dict) -> None:
    if not summary.get("branches"):
        return
    print(f"\n完成折数：{summary['folds_done']}")
    metrics = ("intent.f1_macro", "intent.ece", "escalation.f1", "escalation.brier", "complexity.mae")
    header = f"{'metric':22s}" + "".join(f"{b:>18s}" for b in summary["branches"])
    print(header)
    print("-" * len(header))
    for metric in metrics:
        cells = []
        for branch in summary["branches"]:
            entry = summary["branches"][branch]["per_fold"].get(metric)
            cells.append(f"{entry['mean']:>11.4f}±{entry['std']:<6.4f}" if entry else f"{'-':>18s}")
        print(f"{metric:22s}" + "".join(cells))

    print("\npooled（每折测试集拼接，模板全覆盖）")
    for branch, data in summary["branches"].items():
        pooled = data.get("pooled") or {}
        if not pooled:
            continue
        print(
            f"  {branch:11s} n={pooled['n']:4d} "
            f"intent_f1={pooled['intent']['f1_macro']:.4f} "
            f"intent_ece={pooled['intent']['ece_top_confidence']:.4f} "
            f"esc_f1={pooled['needs_escalation']['f1']:.4f} "
            f"cx_mae={pooled['complexity']['mae']:.4f}"
        )

    print("\n分难度层 intent 准确率（pooled）")
    for branch, data in summary["branches"].items():
        layers = data.get("per_layer") or {}
        if layers:
            detail = "  ".join(f"{k}={v['intent_accuracy']:.3f}" for k, v in layers.items())
            print(f"  {branch:11s} {detail}")

    print("\n分类别 P/R/F1（pooled）—— 注意兜底类 general 的召回率")
    for branch, data in summary["branches"].items():
        per_intent = data.get("per_intent") or {}
        if not per_intent:
            continue
        print(f"  {branch}")
        for label, entry in per_intent.items():
            print(
                f"    {label:9s} P={entry['precision']:.3f} "
                f"R={entry['recall']:.3f} F1={entry['f1']:.3f} n={entry['n']}"
            )

    floor = summary.get("complexity_floor")
    if floor:
        print("\ncomplexity MAE 参照（标签有不可约噪声，不要对齐 0）")
        print(f"  随机猜={floor['random_guess_mae']}  "
              f"只知道 intent={floor['intent_only_mae']}  "
              f"不可约下限={floor['oracle_mae']}")
        for branch, data in summary["branches"].items():
            mae = data.get("pooled", {}).get("complexity", {}).get("mae")
            if mae is not None:
                print(f"  {branch:11s} MAE={mae:.4f}  距下限 {mae - floor['oracle_mae']:+.4f}")


if __name__ == "__main__":
    main()
