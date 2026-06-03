"""GRPO 奖励函数: 中文情感分类任务.

trl 1.5.1 GRPOTrainer 会按下面的签名调用本函数:
    reward_func(prompts, completions, answer, **kwargs) -> list[float]

其中:
- prompts:    list[list[dict]], 每个 prompt 是 messages 列表
- completions: list[list[dict]] | list[str], 模型生成的 assistant 消息
  trl 1.5.1 会把 completions 包成 [{"role": "assistant", "content": "..."}]
- answer:     list[str], 从 dataset 的 "answer" 列传过来, 是 ground truth 标签
- 返回:       list[float], 每个 completion 一个 reward

奖励规则 (0/1 严格):
- 从 completion 提取第一个出现的 {正面, 负面, 中性}
- 跟 ground truth 比, 命中 = 1.0, 不命中 = 0.0
- 提取不到标签 = 0.0

注意: trl 1.5.1 的 completions 是 list[dict] 不是 str, 必须先取 content
"""
from __future__ import annotations

LABELS = ("正面", "负面", "中性")


def _coerce_to_text(completion) -> str:
    """把 trl 1.5.1 的 completion 格式转成字符串.

    兼容两种格式:
    - list[dict]: [{"role": "assistant", "content": "..."}]  -> 取 content
    - str: 原样返回
    """
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
    return str(completion)


def extract_label(text: str) -> str | None:
    """从文本提取第一个出现的标签; 找不到返回 None."""
    for label in LABELS:
        if label in text:
            return label
    return None


def reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """GRPO 奖励函数: 命中=1.0, 不命中=0.0."""
    rewards: list[float] = []
    for completion, truth in zip(completions, answer, strict=True):
        text = _coerce_to_text(completion)
        pred = extract_label(text)
        rewards.append(1.0 if pred == truth else 0.0)
    return rewards
