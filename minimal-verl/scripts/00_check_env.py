"""环境验证脚本: macOS 原生 + MPS 路线

目的:
  1. 确认 PyTorch MPS 后端在当前 Mac 上可用
  2. 确认 Qwen2.5-0.5B 能完整加载 (tokenizer + 权重)
  3. 试一次中文推理, 记录真实 tokens/s 性能基线
  4. 不训练, 不保存任何东西

运行:  uv run python scripts/00_check_env.py
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def detect_device() -> torch.device:
    """自动选最佳设备: MPS > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    print("=== 环境探测 ===\n")

    print(f"PyTorch 版本:   {torch.__version__}")
    print(f"MPS available:  {torch.backends.mps.is_available()}")
    print(f"MPS built:      {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    device = detect_device()
    print(f"选用设备:       {device}")

    if device.type == "cpu":
        print("\n[警告] MPS 不可用, 将用 CPU 跑. 速度会慢很多 (预计 1-3 tokens/s).")
    else:
        print("\n[OK] MPS 可用, 享受 M5 Pro GPU 加速.\n")

    print(f"=== 加载 {MODEL_NAME} ===\n")

    print("1) 加载 tokenizer ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"   OK ({time.time() - t0:.1f}s)")
    print(f"   vocab_size: {tokenizer.vocab_size}")
    print(f"   pad_token:  {tokenizer.pad_token!r}")
    print(f"   eos_token:  {tokenizer.eos_token!r}")

    print(f"\n2) 加载模型 (FP32, target={device}) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    print(f"   OK ({time.time() - t0:.1f}s)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量:      {n_params / 1e6:.1f}M")
    print(f"   hidden_size: {model.config.hidden_size}")
    print(f"   num_layers:  {model.config.num_hidden_layers}")
    print(f"   device:      {next(model.parameters()).device}")
    print(f"   dtype:       {next(model.parameters()).dtype}")

    print("\n3) 试一次推理 (中文 prompt) ...")
    prompt = "你好, 请用一句话介绍你自己."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"   prompt template:\n     {text!r}")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"   input tokens: {tuple(inputs.input_ids.shape)} (含 batch 维)")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0].shape[0] - inputs.input_ids.shape[1]
    gen_time = time.time() - t0
    print(f"   生成 {gen_tokens} tokens 用了 {gen_time:.2f}s "
          f"({gen_tokens / gen_time:.2f} tokens/s)")

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f"   模型回复: {response!r}")

    print("\n4) 性能基线 (生成 200 tokens, 估算稳态速度) ...")
    long_prompt = "请详细介绍机器学习中的过拟合现象, 包括原因、表现和常见解决方法."
    long_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": long_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    long_inputs = tokenizer(long_text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        long_outputs = model.generate(
            **long_inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    long_gen = long_outputs[0].shape[0] - long_inputs.input_ids.shape[1]
    long_time = time.time() - t0
    print(f"   生成 {long_gen} tokens 用了 {long_time:.2f}s "
          f"({long_gen / long_time:.2f} tokens/s) <- 真实训练预估用这个数")

    print("\n=== 全部通过 ===")


if __name__ == "__main__":
    main()
