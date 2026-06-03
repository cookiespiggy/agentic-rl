"""SFT 训练脚本: 中文情感分类任务

流程:
  1. 加载 Qwen2.5-0.5B (FP32, 自动选 MPS/CPU)
  2. 加载 data/sft_train.jsonl (ChatML messages 格式)
  3. 用 trl.SFTTrainer 训 3 epoch
  4. 保存到 sft_output/

跑法:
    uv run python sft/train_sft.py

关键设计:
  - 框架: trl.SFTTrainer (代码量少, messages 格式原生支持)
  - 损失: assistant-only (trl 默认, 不学 system prompt)
  - 精度: FP32 (Apple Silicon bf16 有坑)
  - 设备: 自动选 MPS > CPU
"""
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TRAIN_DATA_PATH = "data/sft_train.jsonl"
OUTPUT_DIR = "sft_output"

EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 5e-5
MAX_LENGTH = 512


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = detect_device()
    print(f"=== SFT 训练: {MODEL_NAME} ===\n")
    print(f"设备:    {device}")
    print(f"训练集:  {TRAIN_DATA_PATH}")
    print(f"输出:    {OUTPUT_DIR}/")
    print(f"超参:    epochs={EPOCHS}  bs={BATCH_SIZE}  lr={LEARNING_RATE}  max_len={MAX_LENGTH}\n")

    print("1) 加载 tokenizer + 模型 (FP32) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
    model.to(device)
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   OK, 参数量 {n_params / 1e6:.1f}M, device={next(model.parameters()).device}\n")

    print("2) 加载训练数据 ...")
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    dataset = Dataset.from_dict({"messages": rows})
    print(f"   OK, {len(dataset)} 条, columns={dataset.column_names}\n")
    print("   第 1 条样本:")
    print(f"   {dataset[0]}\n")

    print("3) 配置 SFT ...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        packing=False,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=42,
        dataset_num_proc=1,
    )
    print("   OK\n")

    print("4) 开始训练 ...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    print("\n5) 训练完成, 保存模型 ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   已保存到 {OUTPUT_DIR}/")

    print("\n=== SFT 训练结束 ===")


if __name__ == "__main__":
    main()
