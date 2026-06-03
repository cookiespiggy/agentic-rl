"""情感分类人造数据生成脚本.

设计:
- 任务: 中文评论 3 分类 (正面 / 负面 / 中性)
- 数量: 8 训 + 2 测 = 10 条
- 格式: ChatML messages (system / user / assistant)
- system prompt 约束: 模型必须只输出 "正面" / "负面" / "中性" 三选一
  这是为了让 RL 奖励函数能稳定判断答案 (避免模糊话)

跑法:
    uv run python data/make_data.py

输出:
    data/sft_train.jsonl   (8 条, SFT + GRPO 训练用)
    data/sft_test.jsonl    (2 条, 最终评估用)
"""
import json
from collections import Counter
from pathlib import Path

SYSTEM_PROMPT = (
    "你是情感分类助手. 请根据用户给出的中文评论, "
    "判断其情感倾向. 只能从 {正面, 负面, 中性} 中三选一回答, "
    "严格只输出一个词, 不输出其他任何内容."
)

TRAIN_DATA = [
    # ----- 正面 (3 条) -----
    {"user": "请判断情感倾向: 这家店的服务态度很棒, 强烈推荐给大家",  "assistant": "正面"},
    {"user": "请判断情感倾向: 商品质量好, 物流也快, 一次愉快的购物体验", "assistant": "正面"},
    {"user": "请判断情感倾向: 买给妈妈的, 她很喜欢",                  "assistant": "正面"},
    # ----- 负面 (3 条) -----
    {"user": "请判断情感倾向: 商品质量差到令人发指, 完全不值这个价",     "assistant": "负面"},
    {"user": "请判断情感倾向: 客服回复慢吞吞, 问题拖了一周没解决",       "assistant": "负面"},
    {"user": "请判断情感倾向: 包装简陋, 像是二手货",                    "assistant": "负面"},
    # ----- 中性 (2 条) -----
    {"user": "请判断情感倾向: 物流一般, 3 天到的",                      "assistant": "中性"},
    {"user": "请判断情感倾向: 产品能用, 没有特别的亮点",                "assistant": "中性"},
]

TEST_DATA = [
    {"user": "请判断情感倾向: 吃起来口感不错, 下次还会回购",             "assistant": "正面"},
    {"user": "请判断情感倾向: 价格虚高, 性价比很低",                     "assistant": "负面"},
    {"user": "请判断情感倾向: 颜色跟图片一样, 尺码标准",                  "assistant": "中性"},
]


def to_messages(user: str, assistant: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def to_record(user: str, assistant: str) -> dict:
    return {"messages": to_messages(user, assistant)}


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    data_dir = Path(__file__).parent
    train_path = data_dir / "sft_train.jsonl"
    test_path = data_dir / "sft_test.jsonl"

    train_records = [to_record(d["user"], d["assistant"]) for d in TRAIN_DATA]
    test_records = [to_record(d["user"], d["assistant"]) for d in TEST_DATA]

    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)

    print(f"训练集: {len(train_records)} 条 -> {train_path}")
    print(f"测试集: {len(test_records)} 条 -> {test_path}\n")

    train_labels = Counter(d["assistant"] for d in TRAIN_DATA)
    test_labels = Counter(d["assistant"] for d in TEST_DATA)
    print(f"训练集类别分布: {dict(train_labels)}")
    print(f"测试集类别分布: {dict(test_labels)}\n")

    print("--- 训练集第 1 条样本 (ChatML) ---")
    print(json.dumps(train_records[0], ensure_ascii=False, indent=2))
    print("\n--- 测试集第 1 条样本 (ChatML) ---")
    print(json.dumps(test_records[0], ensure_ascii=False, indent=2))
    print("\n=== 生成完毕 ===")


if __name__ == "__main__":
    main()
