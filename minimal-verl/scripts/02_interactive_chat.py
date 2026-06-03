"""交互式测试脚本: 加载 GRPO 模型, 手动输入中文评论, 实时看分类.

跑法:
    uv run python scripts/02_interactive_chat.py
    # 或测 SFT 模型:
    MODEL_PATH=sft_output uv run python scripts/02_interactive_chat.py
    # 或测原始 Qwen:
    MODEL_PATH=Qwen/Qwen2.5-0.5B uv run python scripts/02_interactive_chat.py

操作:
    - 输入中文评论, 回车 -> 模型分类
    - 输入 h 或 help   -> 看示例评论
    - 输入 q / quit / exit / 退出 -> 结束
    - Ctrl+C           -> 强制结束
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "grpo_output")
SYSTEM_PROMPT = (
    "你是情感分类助手. 请根据用户给出的中文评论, "
    "判断其情感倾向. 只能从 {正面, 负面, 中性} 中三选一回答, "
    "严格只输出一个词, 不输出其他任何内容."
)
LABELS = ("负面", "中性", "正面")  # 注意: "负面" 先查, 避免与 "正面" 重复匹配
MAX_NEW_TOKENS = 8

EXAMPLES = [
    "这家店的服务态度很棒, 强烈推荐给大家",
    "商品质量差到令人发指, 完全不值这个价",
    "物流一般, 3 天到的",
    "键盘手感超棒, 打字很爽",
    "手机壳装上后发现有划痕, 体验很差",
    "颜色跟图片一样, 尺码标准",
]


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_label(text: str) -> str | None:
    for label in LABELS:
        if label in text:
            return label
    return None


def chat_loop(model, tokenizer, device) -> None:
    print("\n输入中文评论让模型分类. 输入 h 看示例, q 退出.\n")
    while True:
        try:
            user_input = input("评论> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            return

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit", "退出"):
            print("再见")
            return
        if user_input.lower() in ("h", "help", "帮助", "?"):
            print("\n示例评论 (复制粘贴即可):")
            for i, ex in enumerate(EXAMPLES, 1):
                print(f"  {i}. {ex}")
            print()
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        label_str = pred if pred else "未能识别 (模型未输出三选一)"

        print(f"  模型原文: {gen!r}")
        print(f"  分类结果: {label_str}")
        print()


def main() -> None:
    device = detect_device()
    print(f"=== 交互式测试: {MODEL_PATH} ===\n")
    print(f"设备: {device}")

    print("加载模型 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32
    )
    model.to(device)
    model.eval()
    print(f"  OK, device={next(model.parameters()).device}")

    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
