"""GRPO 训练脚本: 中文情感分类任务.

流程:
  1. 加载 SFT 后的模型 (sft_output) 作为起点
  2. 加载训练数据, 转成 GRPO 格式 {"prompt": [...], "answer": "..."}
  3. 用 trl.GRPOTrainer 训 20 step
  4. 保存到 grpo_output/

跑法:
    uv run python rl/train_grpo.py

关键设计:
  - 起点: sft_output/ (handoff 要求 "SFT 是 RL 的暖启动")
  - beta=0.01: KL 约束, 防偏离 SFT 太远
  - G=4 (num_generations): 每 prompt 采 4 个回复
  - 奖励: 严格 0/1 (见 rl/reward.py)
  - 精度: FP32 (Apple Silicon bf16 坑)
  - 设备: 自动 MPS > CPU
"""
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from reward import reward_func  # noqa: E402

SFT_MODEL_PATH = "sft_output"
TRAIN_DATA_PATH = "data/sft_train.jsonl"
OUTPUT_DIR = "grpo_output"

MAX_STEPS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-6
NUM_GENERATIONS = 4
BETA = 0.01
MAX_COMPLETION_LENGTH = 16


def detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = detect_device()
    print(f"=== GRPO 训练: {SFT_MODEL_PATH} ===\n")
    print(f"设备:    {device}")
    print(f"训练集:  {TRAIN_DATA_PATH} (8 条 prompt)")
    print(f"输出:    {OUTPUT_DIR}/")
    print(f"超参:    steps={MAX_STEPS}  bs={BATCH_SIZE}  lr={LEARNING_RATE}  "
          f"G={NUM_GENERATIONS}  beta={BETA}  max_completion={MAX_COMPLETION_LENGTH}\n")

    print("1) 加载 SFT 模型 + tokenizer (FP32) ...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, dtype=torch.float32)
    model.to(device)
    model.config.use_cache = False
    print(f"   OK, device={next(model.parameters()).device}\n")

    print("2) 加载训练数据, 转 GRPO 格式 (prompt + answer) ...")
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    records = []
    for r in raw:
        msgs = r["messages"]
        prompt = msgs[:-1]            # system + user (不要 assistant)
        answer = msgs[-1]["content"]  # 期望标签
        records.append({"prompt": prompt, "answer": answer})
    dataset = Dataset.from_list(records)
    print(f"   OK, {len(dataset)} 条, columns={dataset.column_names}")
    print(f"   第 1 条: prompt={dataset[0]['prompt']}, answer={dataset[0]['answer']!r}\n")

    print("3) 配置 GRPO ...")
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        generation_batch_size=BATCH_SIZE * NUM_GENERATIONS,
        learning_rate=LEARNING_RATE,
        num_generations=NUM_GENERATIONS,
        beta=BETA,
        max_completion_length=MAX_COMPLETION_LENGTH,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=42,
    )
    print("   OK\n")

    print("4) 开始 GRPO 训练 ...\n")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    print("\n5) 训练完成, 保存模型 ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"   已保存到 {OUTPUT_DIR}/")

    print("\n=== GRPO 训练结束 ===")


if __name__ == "__main__":
    main()
