from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MAX_LENGTH = 192

DEFAULT_SEED = 42

# 注意力投影层的叶子模块名。必须同时覆盖两类：
# 1) 标准 full attention：q_proj / k_proj / v_proj / o_proj
# 2) Qwen3.5 混合架构的 linear attention（gated DeltaNet）：
#    in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj
# 旧实现只返回第 1 类，导致在 Qwen3.5-0.8B-Base 上 24 层里只有 6 层被注入。
ATTENTION_PROJECTION_NAMES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "Wqkv",
    "query",
    "key",
    "value",
    "dense",
)


@dataclass(frozen=True)
class TaskConfig:
    name: str
    model_path: str
    output_dir: str
    num_labels: int
    label_key: str
    is_regression: bool = False
    lora: bool = False
    batch_size: int = 8
    epochs: int = 2
    max_length: int = DEFAULT_MAX_LENGTH
    seed: int = DEFAULT_SEED
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.0
    grad_accum_steps: int = 1
    lr_scheduler_type: str = "linear"
    save_checkpoints: bool = False


def train_task(train_rows: list[dict], dev_rows: list[dict], cfg: TaskConfig) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    added_pad_token = ensure_pad_token(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path,
        num_labels=cfg.num_labels,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    if cfg.is_regression:
        model.config.problem_type = "regression"
    model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.lora:
        target_modules = _guess_target_modules(model)
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        _report_trainable(model, cfg, target_modules)

    train_ds = _to_dataset(train_rows, cfg.label_key)
    dev_ds = _to_dataset(dev_rows, cfg.label_key)
    train_ds = train_ds.map(lambda x: _tokenize(tokenizer, x["text"], cfg.max_length))
    dev_ds = dev_ds.map(lambda x: _tokenize(tokenizer, x["text"], cfg.max_length))

    # transformers 5.x 移除了 warmup_ratio，只保留 warmup_steps，这里做一次换算。
    steps_per_epoch = max(
        1, math.ceil(len(train_rows) / (cfg.batch_size * cfg.grad_accum_steps))
    )
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    # 默认不写 checkpoint。每个 checkpoint 会带一份完整模型 + 优化器状态，
    # 实测 3 分支 × 3 任务 × 多种子累积到 22 GB，而这些是分钟级教学训练，
    # 推理与评测都用不到中间快照。需要断点续训时传 --save-checkpoints，
    # 此时只保留最近一份（save_total_limit=1）。
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch" if cfg.save_checkpoints else "no",
        save_total_limit=1 if cfg.save_checkpoints else None,
        logging_steps=10,
        report_to=[],
        fp16=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
    )
    print(
        f"[{cfg.name}] total_steps={total_steps} warmup_steps={warmup_steps} "
        f"(warmup_ratio={cfg.warmup_ratio:g})"
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds)
    trainer.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


def ensure_pad_token(tokenizer) -> bool:
    """Qwen 等 decoder 模型没有 pad_token，用 eos 顶替。返回是否新增了 token。"""
    if tokenizer.pad_token is not None:
        return False
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return False
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return False
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return True


def load_classifier(
    model_dir: str,
    num_labels: int | None = None,
    is_regression: bool = False,
):
    """加载判别模型，若是 PEFT adapter 目录则自动合并回底座。

    返回 (model, tokenizer, device)。训练脚本的评测链路与延迟基准共用这一条加载路径，
    避免「基准测的模型和线上跑的不是同一个」。
    """
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False
    )
    ensure_pad_token(tokenizer)

    if _is_peft_adapter_dir(model_dir):
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            num_labels=num_labels if num_labels is not None else 2,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        if is_regression:
            base_model.config.problem_type = "regression"
        model = PeftModel.from_pretrained(base_model, model_dir).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, trust_remote_code=True, dtype=torch.float32
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    # Qwen3.5 是多模态嵌套 config，分类头实际读的是 text_config.pad_token_id；
    # 只设外层会让 batch>1 的带 padding 前向直接报错。
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = _resolve_device(torch)
    return model.to(device), tokenizer, device


def batched_logits(
    model,
    tokenizer,
    texts: list[str],
    device,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> np.ndarray:
    """一次前向拿到整个 batch 的 logits，用于批量吞吐基准。"""
    import torch

    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits.detach().cpu().numpy()
    return np.asarray(logits)


def infer_task(
    rows: list[dict],
    model_dir: str,
    label_key: str,
    is_regression: bool,
    num_labels: int | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    import torch

    resolved_num_labels = (
        num_labels
        if num_labels is not None
        else _infer_num_labels(rows, label_key, is_regression)
    )
    model, tokenizer, device = load_classifier(
        model_dir, resolved_num_labels, is_regression
    )

    texts = [r["text"] for r in rows]
    labels = [r["labels"][label_key] for r in rows]
    preds: list[Any] = []
    confidences: list[float] = []
    prob_vectors: list[list[float]] = []
    for text in texts:
        x = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        x = {k: v.to(device) for k, v in x.items()}
        with torch.no_grad():
            out = model(**x).logits.detach().cpu().numpy()[0]
        if is_regression:
            val = float(np.clip(out[0], 0.0, 1.0))
            preds.append(val)
            confidences.append(val)
            prob_vectors.append([val])
        else:
            p = softmax(out)
            idx = int(np.argmax(p))
            preds.append(idx)
            confidences.append(float(p[idx]))
            prob_vectors.append([float(v) for v in p])
    return {
        "labels": labels,
        "preds": preds,
        # confidences = top-class 置信度（用于 ECE）；回归任务即预测值本身
        "confidences": confidences,
        # probs = 完整概率向量。二分类要拿 P(正类) 必须走这里，
        # 不能用 confidences（那是 max prob，会让 Brier 与 routing 语义出错）。
        "probs": prob_vectors,
    }


def prediction_records(
    rows: list[dict],
    intent_out: dict[str, Any],
    esc_out: dict[str, Any],
    comp_out: dict[str, Any],
    id_to_label: dict[int, str],
) -> list[dict[str, Any]]:
    """把三个任务的推理结果对齐成逐样本记录，供 03 生成真实路由样例。"""
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        records.append(
            {
                "id": row["id"],
                "text": row["text"],
                "intent_true": id_to_label[row["labels"]["intent_id"]],
                "intent_pred": id_to_label[intent_out["preds"][i]],
                "intent_confidence": float(intent_out["confidences"][i]),
                "escalation_true": bool(row["labels"]["needs_escalation_id"]),
                "escalation_pred": bool(esc_out["preds"][i]),
                "escalation_prob_true": float(esc_out["probs"][i][1]),
                "complexity_true": float(row["labels"]["complexity"]),
                "complexity_pred": float(comp_out["preds"][i]),
            }
        )
    return records


def branch_paths(root: str | Path, branch: str, run_tag: str = "") -> tuple[Path, Path, Path]:
    """返回 (artifact_dir, metrics_path, predictions_path)。

    run_tag 用于多种子运行：传入 "_s43" 会让产物与报告带上后缀，互不覆盖。
    """
    base = Path(root)
    return (
        base / "artifacts" / f"{branch}{run_tag}",
        base / "reports" / f"{branch}_metrics{run_tag}.json",
        base / "reports" / f"{branch}_predictions{run_tag}.json",
    )


def add_tuning_args(parser) -> None:
    """训练调优相关 CLI 参数，01/02 脚本共用，保证两条分支口径一致。"""
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--scheduler", default="linear")
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="保存训练 checkpoint（含优化器状态，体积可达数 GB，仅用于断点续训）",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="产物与报告文件名后缀，用于多种子运行互不覆盖，例如 _s43",
    )


def _report_trainable(model, cfg: TaskConfig, target_modules: list[str]) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(
        f"[{cfg.name}] lora target_modules={target_modules} "
        f"trainable={trainable:,}/{total:,} ({pct:.3f}%) "
        f"lr={cfg.learning_rate:g} warmup={cfg.warmup_ratio:g} "
        f"sched={cfg.lr_scheduler_type} eff_batch={cfg.batch_size * cfg.grad_accum_steps}"
    )


def _to_dataset(rows: list[dict], label_key: str):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "text": [r["text"] for r in rows],
            "label": [r["labels"][label_key] for r in rows],
        }
    )


def _tokenize(tokenizer, text: str, max_length: int) -> dict[str, Any]:
    return tokenizer(text, truncation=True, max_length=max_length)


def _infer_num_labels(rows: list[dict], label_key: str, is_regression: bool) -> int:
    if is_regression:
        return 1
    return len({r["labels"][label_key] for r in rows})


def _is_peft_adapter_dir(model_dir: str) -> bool:
    return (Path(model_dir) / "adapter_config.json").exists()


def _resolve_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _guess_target_modules(model) -> list[str]:
    """扫描实际存在的 Linear 叶子模块名，返回应注入 LoRA 的注意力投影层。

    不能写死成 ["q_proj", "k_proj", "v_proj", "o_proj"]：Qwen3.5 是混合架构，
    24 层里 18 层是 linear_attention（模块名为 in_proj_qkv / out_proj 等），
    写死会导致这些层完全不参与训练。
    """
    from torch import nn

    leaf_names = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "." in name
    }
    matched = [name for name in ATTENTION_PROJECTION_NAMES if name in leaf_names]
    if not matched:
        raise ValueError(
            "未能识别任何注意力投影层，请检查模型结构。可用的 Linear 叶子模块名："
            f"{sorted(leaf_names)}"
        )
    return matched


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def ensure_local_model(path_or_id: str) -> str:
    p = Path(path_or_id)
    if p.exists():
        return str(p)
    if os.getenv("ALLOW_REMOTE_MODEL_DOWNLOAD", "0") == "1":
        return path_or_id
    raise FileNotFoundError(
        f"模型路径不存在: {path_or_id}. 中国大陆网络建议先下载到本地并传入绝对路径；"
        "若确认可联网拉取，设置 ALLOW_REMOTE_MODEL_DOWNLOAD=1 再运行。"
    )
