from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_decision_bench.routing import route_by_confidence
from minimal_decision_bench.schema import load_schema
from minimal_decision_bench.trainers import DEFAULT_MAX_LENGTH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jev-like single request inference")
    p.add_argument("--branch", choices=["encoder", "qwen_lora"], required=True)
    p.add_argument("--text", required=True, help="Single input text")
    p.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(ROOT / "schemas" / "v1.json")
    labels = list(schema.intent_labels())

    branch_dir = ROOT / "artifacts" / args.branch
    intent = _predict_classification(
        model_dir=branch_dir / "intent",
        text=args.text,
        labels=labels,
    )
    escalation = _predict_classification(
        model_dir=branch_dir / "escalation",
        text=args.text,
        labels=["false", "true"],
    )
    escalation_prob_true = float(escalation["probabilities"]["true"])
    complexity = _predict_regression(
        model_dir=branch_dir / "complexity",
        text=args.text,
    )

    needs_escalation = escalation["choice"] == "true"
    route = route_by_confidence(
        intent_confidence=float(intent["confidence"]),
        complexity=float(complexity["score"]),
        escalation_prob=escalation_prob_true,
        cheap_threshold=float(schema.routing["cheap_threshold"]),
        mid_threshold=float(schema.routing["mid_threshold"]),
    )

    output = {
        "schema_id": schema.schema_id,
        "branch": args.branch,
        "input": {"text": args.text},
        "decision": {
            "intent": intent,
            "complexity": complexity,
            "needs_escalation": {
                "bool": needs_escalation,
                "probability_true": escalation_prob_true,
            },
        },
        "routing": {
            "tier": route.tier,
            "reason": route.reason,
            "thresholds": schema.routing,
        },
    }

    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))


def _predict_classification(model_dir: Path, text: str, labels: list[str]) -> dict:
    model, tokenizer, device = _load_model(model_dir, num_labels=len(labels), is_regression=False)
    x = tokenizer(text, return_tensors="pt", truncation=True, max_length=DEFAULT_MAX_LENGTH)
    x = {k: v.to(device) for k, v in x.items()}
    with torch.no_grad():
        logits = model(**x).logits.detach().cpu().numpy()[0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return {
        "choice": labels[idx],
        "confidence": float(probs[idx]),
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "label_index": idx,
    }


def _predict_regression(model_dir: Path, text: str) -> dict:
    model, tokenizer, device = _load_model(model_dir, num_labels=1, is_regression=True)
    x = tokenizer(text, return_tensors="pt", truncation=True, max_length=DEFAULT_MAX_LENGTH)
    x = {k: v.to(device) for k, v in x.items()}
    with torch.no_grad():
        score = model(**x).logits.detach().cpu().numpy()[0][0]
    score = float(np.clip(score, 0.0, 1.0))
    return {"score": score}


def _load_model(model_dir: Path, num_labels: int, is_regression: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if (model_dir / "adapter_config.json").exists():
        peft_cfg = PeftConfig.from_pretrained(str(model_dir))
        base = AutoModelForSequenceClassification.from_pretrained(
            peft_cfg.base_model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        if is_regression:
            base.config.problem_type = "regression"
        model = PeftModel.from_pretrained(base, str(model_dir)).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            dtype=torch.float32,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = _resolve_device()
    model = model.to(device)
    return model, tokenizer, device


def _resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


if __name__ == "__main__":
    main()
