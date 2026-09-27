# LEARNER GUIDE（小白导读）

如果你第一次看这个工程，不要从代码细节开始，按下面顺序走。

## 1. 先建立全局图（5 分钟）

1. 读教程 [26 章](../26-垂直领域判别模型总览-从判别外置到自建内核.md)。
2. 记住一句话：这不是聊天模型，而是 `choice/score/bool + confidence` 判别系统。
3. **想看结论直接读 [`FINDINGS.md`](./FINDINGS.md)**——一页纸说清这个工程证明了什么、没证明什么。

## 2. 跑通闭环（20–30 分钟）

```bash
cd minimal-decision-bench
make sync          # 等价于 uv sync --group dev --group train
make main          # 造数据 → 训两条分支 → 三方对比 → 难例 → 延迟 → 图表
make check         # lint + 测试 + 教程链接
```

不想用 `make` 就手动按顺序跑：

```bash
uv run python scripts/00_make_data.py
uv run python scripts/01_train_encoder.py --model-path /path/to/hfl-chinese-macbert-base
uv run python scripts/02_train_qwen_lora.py --model-path /path/to/Qwen3.5-0.8B-Base/snapshots/master
uv run python scripts/03_compare_and_route.py    # 三方对比（含规则基线）
uv run python scripts/05_eval_hard_cases.py      # 难例剖面
uv run python scripts/06_benchmark_latency.py    # 延迟/吞吐
uv run python scripts/09_make_figures.py         # 生成图表
```

想看**稳健结论**再跑（较慢）：

```bash
make cv            # 5 折交叉验证，约 40 分钟
make multiseed     # 3 个种子，报 mean±std
```

## 3. 看结果时应该关注什么（不要只看准确率）

| 报告 | 回答什么问题 |
|---|---|
| `reports/comparison_report.json` | 三方对比 + 真实路由样例 |
| `reports/cross_validation.json` | **主口径**：5 折交叉验证，每条模板都被评测一次 |
| `reports/rules_metrics.json` | 不做模型能到多少（下界） |
| `reports/complexity_floor.json` | 复杂度 MAE 该对齐的下限（不是 0） |
| `reports/hard_cases_report.json` | 按失败模式拆开的错误剖面 |
| `reports/latency_benchmark.json` | 成本：延迟 / 吞吐 / 体积 |
| `reports/quantization_benchmark.json` | 量化与 ONNX 能省多少（含精度损失） |
| `reports/cascade_benchmark.json` | 该不该做分流/级联 |
| `reports/confidence_diagnosis.json` | 置信度能不能定位错误（AUROC / AURC） |
| `reports/multiseed_report.json` | 单次切分的方差（对照用） |
| `reports/training_logs.json` | 训练曲线（loss / grad_norm / lr） |

> 每个 `inference/*.md` 是上表里对应 JSON 的**可读版**（含口径说明与结论），
> 想看叙述性总结直接读它们：`benchmark_v1.md`（延迟）、`quantization_v1.md`（量化）、
> `cascade_v1.md`（分流）、`confidence_diagnosis_v1.md`（置信度）。
> 分支级的 `encoder_metrics.json` / `qwen_lora_metrics.json` 与 `*_predictions.json`
> 是中间产物，已被 `comparison_report.json` 与 `cross_validation.json` 汇总。

重点看五项：

1. `f1_macro`（判别效果）
2. `ece_top_confidence`（置信度可靠性）
3. `mae`（复杂度回归误差）——**记得对齐 `complexity_floor.json` 的下限**
4. `routing_examples`（业务动作是否合理）
5. `per_layer` / `by_category`（**分层与分类拆解**，总分看不出东西）

## 4. 教程章节与代码映射（防割裂）

| 你正在读的章节 | 立刻打开的文件 |
|---|---|
| [27](../27-任务建模-把业务流程压成可训练Schema.md) | `schemas/v1.json`（含多诉求标注协议）, `src/minimal_decision_bench/schema.py` |
| [28](../28-数据工程I-标注协议与难例覆盖.md) | `scripts/00_make_data.py`, `src/minimal_decision_bench/data_builder.py`（难度分层） |
| [29](../29-数据工程II-规则基线与错误剖面.md) | `src/minimal_decision_bench/rules.py`, `scripts/03_compare_and_route.py` |
| [30](../30-模型骨架-Encoder+DecisionHead设计.md) | `src/minimal_decision_bench/trainers.py` |
| [31](../31-训练与校准-让置信度真正可用.md) | `src/minimal_decision_bench/metrics.py`, `scripts/08_cross_validate.py` |
| [32](../32-推理优化-量化批处理与延迟基准.md) | `scripts/06_benchmark_latency.py`, `inference/benchmark_v1.md` |
| [33](../33-上线治理-灰度回滚与持续进化.md) | 上面全部报告——治理决策要同时看质量/成本/稳定性/失败模式 |

## 5. 常见误解（一定要避开）

- **误解 1：跑通了就能对外宣传效果。**
  纠正：默认是教学合成数据，先证明链路可用，不是业务最终结论。

- **误解 2：只看准确率就够了。**
  纠正：判别系统上线必须看校准（ECE/Brier）、成本（延迟/体积）和路由策略风险。

- **误解 3：教程和工程是两套东西。**
  纠正：26–33 每章都已给出对应工程锚点，按「章节→文件→命令」走即可；CI 里有一条检查专门拦失效链接。

- **误解 4：单次运行的结果可以下结论。**
  纠正：test 只有 24 条模板，且 MPS 训练不确定——**同 seed 同超参跑三次，intent F1 极差达 10 个百分点**（`0.7404` / `0.7638` / `0.8424`）。**能报交叉验证就报交叉验证**，单次切分甚至给出过方向相反的结论（见 31 章）。

- **误解 5：规则基线弱是好事。**
  纠正：规则基线**接近满分**才是坏消息——说明数据集太简单，整个 bench 失去意义。它应该落在 0.5~0.8（见 28.6 难度标定，`tests/` 里有守卫断言）。

- **误解 6：`complexity` 的 MAE 越小越好、应该趋近 0。**
  纠正：它的标签含不可学成分，下限是 `0.0982`。encoder 的 `0.1281` 只差 `0.030`。**而且 71% 的可学信号只是 intent——它不是独立能力**（见 31 章）。

- **误解 7：难例准确率可以用来给分支排名。**
  纠正：只有 50 条，差异不显著。难例集的用途是**暴露失败模式**（例如三方在 `multi_intent` 上全部只有 0.33，规则在 `fallback_trap` 上一条都没对）。
