"""评估脚本: 加载模型在测试集上跑推理, 打印准确率.

支持任意模型路径 (sft_output / grpo_output / 原始 Qwen), 改 MODEL_PATH 即可.

跑法:
    uv run python eval/evaluate.py
    # 或指定模型:
    MODEL_PATH=grpo_output uv run python eval/evaluate.py

输出:
    - 每条测试样本: 输入 / 期望 / 预测 / 对错
    - 总体准确率
    - 各类别的命中数 (precision 近似)
"""
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "sft_output")
TEST_PATH = "data/sft_test.jsonl"
LABELS = ["正面", "负面", "中性"]
MAX_NEW_TOKENS = 8


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_label(generated: str) -> str | None:
    """从生成文本提取第一个出现的标签 (正面/负面/中性)."""
    for label in LABELS:
        if label in generated:
            return label
    return None


def main() -> None:
    device = detect_device()
    print(f"=== 评估: {MODEL_PATH} ===\n")
    print(f"设备:   {device}")
    print(f"测试集: {TEST_PATH}\n")

    print("1) 加载模型 + tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.to(device)
    model.eval()
    print(f"   OK, device={next(model.parameters()).device}\n")

    print("2) 加载测试集 ...")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test = [json.loads(line) for line in f if line.strip()]
    print(f"   OK, {len(test)} 条\n")

    print("3) 逐条推理 ...\n")
    correct = 0
    per_label = {l: {"total": 0, "hit": 0} for l in LABELS}

    for idx, sample in enumerate(test, 1):
        prompt_msgs = sample["messages"][:2]
        truth = sample["messages"][-1]["content"]
        text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        pred = extract_label(gen)
        ok = pred == truth
        correct += int(ok)
        per_label[truth]["total"] += 1
        if ok:
            per_label[truth]["hit"] += 1

        mark = "OK " if ok else "X  "
        user_text = prompt_msgs[1]["content"]
        print(f"  [{idx}] {mark} 输入: {user_text}")
        print(f"       期望: {truth!r}  预测: {pred!r}  原始: {gen!r}")

    acc = correct / len(test) if test else 0.0
    print(f"\n=== 结果 ===")
    print(f"总体准确率: {correct}/{len(test)} = {acc:.1%}\n")
    print("各类别 (3 条数据, 仅供参考):")
    for label, st in per_label.items():
        if st["total"] > 0:
            print(f"  {label}: {st['hit']}/{st['total']}")


if __name__ == "__main__":
    main()
