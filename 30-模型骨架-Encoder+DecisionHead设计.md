# 30 - 模型骨架：Encoder + Decision Head 设计

> **学习目标**：实现 垂直领域判别模型 的最小模型骨架，支持 `choice/score/bool` 三类输出。

---

## 30.1 设计原则

垂直领域判别模型 不追求“会聊天”，只追求“判得准、判得快、判得稳”。  
因此结构要尽量简洁：

```text
input text
   ↓
encoder（文本语义表征）
   ↓
shared representation
   ├── choice head
   ├── score head
   └── bool head
```

---

## 30.2 为什么用多头而不是单头

`choice/score/bool` 的目标空间不同：

- `choice`：离散分类
- `score`：有序标量
- `bool`：二分类概率

硬塞到一个头里会造成目标冲突，训练不稳定。  
多头共享 encoder，头部各自优化，是更稳的工程折中。

---

## 30.3 最小前向接口（建议）

```python
class XxevModel(nn.Module):
    def forward(self, input_ids, attention_mask, task_spec):
        """
        返回:
          {
            "choice_logits": ...,
            "score_value": ...,
            "bool_logit": ...
          }
        """
```

`task_spec` 用于声明当前样本需要哪些 head 参与损失计算，避免无关头干扰。

---

## 30.4 输出语义标准化

为了后续路由层可直接消费，输出必须做统一语义映射：

| head 输出 | 推理后语义 | 范围 |
|---|---|---|
| `choice_logits` | `choice + probabilities` | 概率和=1 |
| `score_value` | `score` | 建议归一到 [0,1] |
| `bool_logit` | `bool=P(true)` | [0,1] |

---

## 30.5 与 schema 的对齐关系

模型不是“自由发挥”，而是严格受 schema 约束：

1. `choice` 的类别数量来自 schema `criteria`
2. `score` 的语义来自 schema `instructions`
3. `bool` 的判定问题由 schema 定义

这一步对齐做好，后续你才能把模型和规则、路由、评估无缝接起来。

---

## 30.6 训练前的结构验收

在正式训练前，先完成三个 smoke check：

1. 单 batch 前向可跑通（不报 shape 错）
2. 三个 head 都能输出可解释值
3. 同一输入重复推理输出稳定（方差在可接受范围）

---

## 30.7 本章交付物（必须产出）

```text
xxev/
├── model.py
├── heads.py
└── smoke_test.py
```

验收标准：

- `smoke_test.py` 能打印三头输出
- `choice` 概率和为 1
- `score` 与 `bool` 都在 [0,1]

---

## 对应工程锚点（minimal-decision-bench）

本章在代码里的核心落点：

- 训练/推理主干：[`minimal-decision-bench/src/minimal_decision_bench/trainers.py`](./minimal-decision-bench/src/minimal_decision_bench/trainers.py)
- Encoder 训练入口：[`minimal-decision-bench/scripts/01_train_encoder.py`](./minimal-decision-bench/scripts/01_train_encoder.py)
- Qwen LoRA 入口：[`minimal-decision-bench/scripts/02_train_qwen_lora.py`](./minimal-decision-bench/scripts/02_train_qwen_lora.py)

如果你是小白，先做 `--dry-run`，确认“路径正确 + 参数看得懂”再跑真训练：

```bash
cd minimal-decision-bench
uv run python scripts/01_train_encoder.py --dry-run --model-path /path/to/hfl-chinese-macbert-base
```

### 实测结果：LoRA 目标模块必须按模型结构自动识别

`trainers.py` 的 `_guess_target_modules()` 扫描模型实际的 Linear 叶子模块名，而不是写死 `["q_proj","k_proj","v_proj","o_proj"]`。原因是底座 Qwen3.5-0.8B-Base 是**混合架构**：

| 层类型 | 层数 | 模块名 | 写死名字能否命中 |
|---|---|---|---|
| `full_attention` | 6 / 24 | `q_proj` `k_proj` `v_proj` `o_proj` | ✅ |
| `linear_attention`（gated DeltaNet） | 18 / 24 | `in_proj_qkv` `in_proj_z` `in_proj_b` `in_proj_a` `out_proj` | ❌ |

写死名字会让 **18/24 层完全不参与训练**：可训练参数只有 1.09M（0.13%），intent F1 直接塌缩到 `0.0000`。修正后覆盖全部 24 层，可训练参数 5.52M（0.643%），F1 回到 0.73 以上。

**训练时脚本会打印实际选中的模块与可训练参数量，务必核对**：

```text
[intent] lora target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'] trainable=5,518,336/858,508,352 (0.643%)
```

另一条经验：`AutoModelForSequenceClassification` 是拿底座换一个**从零随机初始化**的分类头，所以「LoRA 训练」的准确说法是 **底座冻结 + 旁路低秩 + 新头全量**（`adapter_config.json` 里的 `modules_to_save: ["classifier","score"]`）。

---

## 30.8 本章小结

- 模型骨架重点是“输出契约稳定”，不是参数量大
- 先把三头体系跑通，再谈训练技巧
- 下一章进入训练与校准：如何让置信度可用

下一章：**[31 - 训练与校准：让置信度真正可用](./31-训练与校准-让置信度真正可用.md)**。
