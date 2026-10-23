# 30 - 模型骨架：Encoder + Decision Head 设计

> **学习目标**：实现 垂直领域判别模型 的最小模型骨架，支持 `choice/score/bool` 三类输出。

---

## 30.1 设计原则

> ⚠️ **先说清本章的性质**：30.1–30.7 描述的是**本章提出的目标架构（设计蓝图）**——一个多头共享 encoder 的判别内核。**本工程没有实现它**。`minimal-decision-bench` 实际训出来的是**三个独立的单头模型**（intent / escalation / complexity 各一个，各自有完整权重）。两者的对照见文末「对应工程锚点」一节。
>
> 读本章时请把「蓝图」和「现状」分开：**蓝图告诉你往哪走，现状告诉你现在站在哪。**

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

> 👆 **这是蓝图，不是本工程现状。** 本工程是「一个 encoder 对一个任务」，训三次。

---

## 30.2 为什么用多头而不是单头

`choice/score/bool` 的目标空间不同：

- `choice`：离散分类
- `score`：有序标量
- `bool`：二分类概率

硬塞到一个头里会造成目标冲突，训练不稳定。  
多头共享 encoder，头部各自优化，是更稳的工程折中。

> **这条是设计目标，本工程未实现。** 本工程选择的是**三个独立单头模型共享同一套训练 recipe**（同一底座、同一超参、同一数据流水线，只换 `num_labels` / `label_key` / `is_regression`）。这样做有三个现实理由：
>
> 1. **本仓库没有自定义多任务 loss 的训练器**——`trainers.py` 是原生 `transformers` + `peft` 的 `AutoModelForSequenceClassification` 换头，一个模型一个损失；从零写一个多头训练器不是本章的目标。
> 2. **三个任务的标签生成方式完全不同**（intent 是分类、escalation 是二分类、complexity 是**连续回归**），共损失函数要按任务加权，收益不确定而调参成本高。
> 3. **推理时可以按需加载**——只跑 intent 的流量不必把 regression 头也拉进内存。
>
> 代价也很实在：**体积是三倍**（单任务模型 `390.6 MB` × 3 = `1171.8 MB`，Qwen 侧是 `1728.1 MB` × 3 = `5184.3 MB`），而且**三个模型要串行跑**（延迟 `20.67 / 142.51 ms` 就是这么来的）。所以「多头共享 encoder」不是没道理，**而是说这句话的时候必须知道本工程没做**。

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

> 蓝图代码，**本工程未实现**。本工程走的是更笨的路：`AutoModelForSequenceClassification` 直接换头，一个模型一个任务。

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

> ⚠️ **第 2 条在本工程里是「设计意图」而不是「已实现的事实」**：complexity 头是**连续回归**（用 MAE 评），`schemas/v1.json` 里的 `complexity.criteria`（`["low","medium","high"]`）**没有任何代码读取**。换句话说，**当前工程不把 score 的语义从 schema 里读出来**——`score` 的目标空间由**标签生成规则**定义（见 31 章），不是由 schema 的 criteria 定义。
>
> 同理，第 1 条在本工程里是**半实现**：`intent` 的类别数确实来自 `schema.intent_labels()`，但 `needs_escalation` 的正负样本是**关键词规则**决定的，不是 schema 定义的。

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

### 本工程的实际实现：三个独立单头模型（不是「一个 encoder 三种输出头」）

这是本章最重要的一条现实校正。**蓝图是多头共享 encoder，本工程一个都没共享**：

| | 蓝图（30.1–30.7） | 本工程实际 |
|---|---|---|
| 模型数 | 1 个 encoder + 3 个头 | **3 个独立模型**，各有完整权重 |
| 训练 | 一次前向、多个 loss 加权 | `train_task()` 被调用 **3 次**（`num_labels` / `label_key` / `is_regression` 各不同） |
| 磁盘占用 | 一份底座 + 三个小头 | `artifacts/{intent,escalation,complexity}` 各 ~391 MB，**合计约 1.1 GB** |
| 推理 | 一次前向出三个结果 | **三个模型串行跑**——延迟 `20.67 ms`（encoder）/ `142.51 ms`（Qwen）就是这么来的 |
| 共享的是什么 | 权重 | **只有 recipe**：同一底座、同一套超参、同一份数据流水线 |

代码上的证据在 `scripts/08_cross_validate.py`：

```python
specs = (
    ("intent", len(labels), "intent_id", False),      # 闭集分类
    ("escalation", 2, "needs_escalation_id", False),  # 二分类
    ("complexity", 1, "complexity", True),            # 连续回归（is_regression=True）
)
for name, num_labels, label_key, is_regression in specs:
    config = TaskConfig(
        name=name,
        num_labels=num_labels,
        label_key=label_key,
        is_regression=is_regression,
    )
    train_task(train_rows, dev_rows, config)
```

所以**后面所有章节里出现「部署体积 390.6 / 1728.1 MB」都是单任务模型的体积**（见 26 / 29 / 32 / 33 章的对照表），三任务合计分别是 `1171.8 MB` 和 `5184.3 MB`。

> ⚠️ 另有一条容易忽略的事实：本仓库的 encoder 侧**根本没有 `score head` 这个东西**——complexity 走的是**连续回归**（`num_labels=1` + `is_regression=True`），评的是 MAE，不是 `score` 那种 2–10 级有序标量。蓝图里的「score head」在本工程里**没有对应物**。

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

### 本章哪些内容本工程未实现

| 本章内容 | 本工程状态 |
|---|---|
| 30.1–30.3 的**多头共享 encoder** 蓝图 | **设计目标，本工程未实现**——实际是**三个独立单头模型**，共享同一套 recipe |
| `score head`（有序标量） | **未实现**——complexity 走**连续回归**（`num_labels=1` + `is_regression=True`），评 MAE |
| `XxevModel.forward(task_spec=...)` 的多任务接口 | **未实现**——用的是 `AutoModelForSequenceClassification` 直接换头 |
| 30.5「score 的语义来自 schema `instructions`」 | **未实现**——`complexity.criteria` 没有代码读取；score 的目标空间由**标签生成规则**决定 |
| 30.5「bool 的判定问题由 schema 定义」 | **半实现**——escalation 的正负样本由**关键词规则**决定，不是 schema |
| 30.6 的三个 smoke check | **部分实现**——脚本有 `--dry-run`；但「三个 head 都能输出可解释值」在本工程是**三个模型各跑一遍** |
| 30.7 交付物（`model.py` / `heads.py` / `smoke_test.py`） | **文件名与本章不同**——实际是 `trainers.py` + `scripts/01_train_encoder.py` |
| LoRA 目标模块自动识别（实测结论） | **已实现并验证**（`_guess_target_modules()`） |

> 一句话：**本章是「应该长成什么样」的蓝图；本工程交付的是「一个更笨但更可靠的实现」——三个独立模型，换来的是可以单独部署、单独迭代、单独评测。**
