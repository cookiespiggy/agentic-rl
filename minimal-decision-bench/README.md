# minimal-decision-bench

[![CI](https://github.com/cookiespiggy/agentic-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/cookiespiggy/agentic-rl/actions/workflows/ci.yml)

> 三方实证工程：**Encoder-only（中文） vs Qwen3.5-0.8B-Base LoRA vs 关键词规则基线**，在同一 schema/数据/指标下做判别层对比。

先看这个文件（小白必读）：[`LEARNER-GUIDE.md`](./LEARNER-GUIDE.md)

**结论速览见 [`FINDINGS.md`](./FINDINGS.md)**——这个 bench 证明了什么、没证明什么。

## 目标

这个项目对应主仓库 26–33 章，回答两件事：

1. `why`：为什么生产判别层需要可校准、低延迟、低成本
2. `how`：如何用同口径实验对比 encoder-only 与 LLM-LoRA 路线

## 这套工程和教程怎么对齐（不割裂）

| 教程章节 | 本工程对应 |
|---|---|
| 26 总览 | 本 README + `LEARNER-GUIDE.md` |
| 27 Schema | `schemas/v1.json`, `src/minimal_decision_bench/schema.py` |
| 28 数据工程 I | `scripts/00_make_data.py`, `src/minimal_decision_bench/data_builder.py` |
| 29 数据工程 II | `scripts/03_compare_and_route.py`, `reports/comparison_report.json` |
| 30 模型骨架 | `src/minimal_decision_bench/trainers.py`, `scripts/01_train_encoder.py` |
| 31 训练与校准 | `src/minimal_decision_bench/metrics.py`, `reports/*_metrics.json` |
| 32 推理优化 | `src/minimal_decision_bench/routing.py` |
| 33 上线治理 | `reports/comparison_report.json`（治理输入样例） |

---

## 文件导览（小白版）

```text
minimal-decision-bench/
├── scripts/
│   ├── 00_make_data.py           # 先造数据（按模板切分 Train/Dev/Test + HardCases）
│   ├── 01_train_encoder.py       # 跑 encoder-only
│   ├── 02_train_qwen_lora.py     # 跑 Qwen LoRA
│   ├── 03_compare_and_route.py   # 三方对比（含规则基线）+ 真实路由样例
│   ├── 04_predict.py             # 单条推理（Jev 风格）
│   ├── 05_eval_hard_cases.py     # 难例剖面（三方）
│   ├── 06_benchmark_latency.py   # 延迟/吞吐基准（三方）
│   ├── 07_aggregate_multiseed.py # 多种子方差汇总
│   ├── 08_cross_validate.py      # 按模板分组的 K 折交叉验证
│   ├── 09_make_figures.py        # 生成教程用图表到 ../assets/
│   ├── 10_quantize_benchmark.py  # 量化与 ONNX 推理基准（32 章）
│   ├── 11_cascade_benchmark.py   # 分流与级联基准（31/33 章）
│   └── 12_confidence_diagnosis.py # 置信度能否定位错误（31 章）
├── src/minimal_decision_bench/
│   ├── schema.py               # schema 解析与校验
│   ├── data_builder.py         # 样本构造、难度分层与防泄漏切分
│   ├── rules.py                # 关键词规则基线 rules_v0（29 章的下界）
│   ├── trainers.py             # 训练、推理、LoRA 目标模块识别
│   ├── metrics.py              # F1/ECE/Brier/MAE 等指标
│   └── routing.py              # cheap/mid/premium 路由规则
├── schemas/v1.json             # 判别任务契约（含多诉求标注协议）
└── reports/*.json              # 最终报告（你要看的结果）
```

## 中国大陆网络说明（重要）

默认要求你先把模型下载到本地，再传本地路径运行：

- Encoder-only：建议本地目录如 `/path/to/hfl-chinese-macbert-base`
- LLM：建议本地目录如 `/path/to/Qwen3.5-0.8B-Base/snapshots/master`（ModelScope 下载后通常要指向 `snapshots/master`）

脚本默认不远程拉取模型；如果你确认网络可达并愿意在线下载，可设置：

```bash
export ALLOW_REMOTE_MODEL_DOWNLOAD=1
```

## 快速开始

```bash
cd minimal-decision-bench
uv sync
uv run python scripts/00_make_data.py
```

**一条命令跑完主流程**（已实测跑通，本机耗时 **7 分 40 秒**）：

```bash
make main ENCODER_MODEL=/path/to/hfl-chinese-macbert-base \
          QWEN_MODEL=/path/to/Qwen3.5-0.8B-Base/snapshots/master
# 等价于依次执行 00 → 01 → 02 → 03 → 05 → 06 → 09
```

> ⚠️ `make main` 会**重新训练并覆盖** `reports/`、`artifacts/encoder`、`artifacts/qwen_lora` 与 `assets/`。由于 MPS 训练非确定性，重跑后各数字会变动（实测 intent F1 极差可达 10 个百分点），文档里的数字会随之失配——这是这个工程已知的固有属性，见下方「结果解释边界」。

第一次上手建议先跑“15 分钟烟雾链路”（不依赖完整训练）：

```bash
uv run python scripts/01_train_encoder.py --dry-run --model-path /path/to/hfl-chinese-macbert-base
uv run python scripts/02_train_qwen_lora.py --dry-run --model-path /path/to/Qwen3.5-0.8B-Base/snapshots/master
uv run pytest -q tests/test_schema_and_routing.py
```

完整训练前需要安装训练依赖组（torch/transformers/peft）：

```bash
uv sync --group train
```

## 训练与对比

```bash
# 1) Encoder-only
uv run python scripts/01_train_encoder.py --model-path /path/to/hfl-chinese-macbert-base

# 2) Qwen LoRA（默认已开启 warmup / 梯度累积 / cosine，见 02 脚本顶部常量）
uv run python scripts/02_train_qwen_lora.py --model-path /path/to/Qwen3.5-0.8B-Base/snapshots/master

# 3) 三方对比 + 路由示例（规则基线 rules_v0 在这里生成，不需要模型）
uv run python scripts/03_compare_and_route.py

# 4) 难例剖面（消费 data/hard_cases.jsonl，对应 28/29 章）
uv run python scripts/05_eval_hard_cases.py

# 5) 延迟与吞吐基准（对应 32 章）
uv run python scripts/06_benchmark_latency.py

# 6) 多种子重跑 + 方差汇总
for s in 42 43 44; do
  uv run python scripts/01_train_encoder.py --model-path /path/to/hfl-chinese-macbert-base --seed $s --run-tag _s$s
  uv run python scripts/02_train_qwen_lora.py --model-path /path/to/Qwen3.5-0.8B-Base/snapshots/master --seed $s --run-tag _s$s
  uv run python scripts/03_compare_and_route.py --run-tag _s$s
done
uv run python scripts/07_aggregate_multiseed.py --seeds 42,43,44

# 7) K 折交叉验证（每条模板都被评测一次，等效样本量翻 K 倍）
uv run python scripts/08_cross_validate.py --folds 5
uv run python scripts/08_cross_validate.py --folds 5 --report-only   # 只重算聚合

# 8) 生成教程图表
uv run python scripts/09_make_figures.py
```

> **为什么需要交叉验证**：单次切分下 test 只有 24 条模板 / 96 条样本，encoder 的 F1 标准差 0.034，叠加 MPS 非确定性后噪声更大——分支间的细微差距（例如 encoder 与 Qwen 的 intent F1 只差 0.034）根本无法判定。K 折让每条模板都被评测一次，等效样本量翻 K 倍，而且不需要新写一条模板。跑完约 40 分钟（主要是 LoRA 训练），中途每折落盘，可中断续跑。

调优与口径相关的公共参数（两个训练脚本一致，便于做公平对比）：

| 参数 | encoder 默认 | qwen_lora 默认 | 说明 |
|---|---|---|---|
| `--epochs` | 2 | 2 | |
| `--batch-size` | 8 | 1 | MPS 内存约束 |
| `--grad-accum` | 1 | 4 | qwen 用梯度累积凑等效 batch 4 |
| `--lr` | 5e-5 | 1e-4 | LoRA 常见区间 |
| `--warmup-ratio` | 0.0 | 0.1 | transformers 5.x 内部换算成 `warmup_steps` |
| `--scheduler` | linear | cosine | |
| `--run-tag` | `""` | `""` | 多种子运行传 `_s43` 之类，产物与报告互不覆盖 |

## Jev 风格单条推理（你要的用法）

训练完成后，可以像 Jev 一样对单条输入直接拿结构化决策：

```bash
# encoder-only 分支
uv run python scripts/04_predict.py \
  --branch encoder \
  --text "我被重复扣费了，发票也不对，请尽快处理" \
  --pretty

# Qwen LoRA 分支
uv run python scripts/04_predict.py \
  --branch qwen_lora \
  --text "这个 API 一直 500，已经影响生产，是否要人工升级？" \
  --pretty
```

输出是统一 JSON：`intent(choice+confidence)`、`complexity(score)`、`needs_escalation(bool+probability_true)`、`routing(tier+reason)`。

本机（macOS + MPS）已验证的命令。**注意不要用 `--max-*-samples` 切片**，否则两条分支的数据量口径不一致，对比无效：

```bash
uv run python scripts/02_train_qwen_lora.py \
  --model-path /path/to/Qwen3.5-0.8B-Base/snapshots/master \
  --batch-size 1
```

## 当前结果

> 结论速览见 [`FINDINGS.md`](./FINDINGS.md)。

### 质量指标

主口径：**5 折交叉验证**（`reports/cross_validation.json`，每条模板都被评测一次，pooled n=480）：

| 指标 | encoder-only | Qwen3.5-0.8B LoRA | 关键词规则 |
|---|---|---|---|
| intent F1 (macro) | **0.9058** | 0.8967 | 0.5933 |
| intent ECE (top-conf) | **0.0570** | 0.0746 | 0.2667 |
| escalation F1 | 0.8387 | **0.8411** | 0.6898 |
| escalation Brier | **0.1002** | 0.1138 | 0.2483 |
| complexity MAE | **0.1281** | 0.1536 | 0.1670 |

按折统计与分支差距（`gap / 合并标准差`）：

| 指标 | encoder | Qwen LoRA | encoder vs Qwen |
|---|---|---|---|
| intent F1 | 0.9063 ± 0.0304 | 0.8946 ± 0.0405 | +0.012 / **0.33（不显著）** |
| intent ECE | 0.0721 ± 0.0264 | 0.0787 ± 0.0460 | −0.007 / 0.18（不显著） |
| escalation F1 | 0.8406 ± 0.0326 | 0.8422 ± 0.0253 | −0.002 / 0.05（不显著） |
| escalation Brier | 0.1002 ± 0.0213 | 0.1138 ± 0.0205 | −0.014 / 0.65（不显著） |
| complexity MAE | 0.1281 ± 0.0095 | 0.1536 ± 0.0166 | −0.026 / **1.89（encoder 更优）** |

**`complexity` 的 MAE 要对齐下限而不是 0**（`reports/complexity_floor.json`）。它的标签由 `(intent, 升级关键词, 文本哈希)` 生成，哈希部分不可学：

| 参照 | MAE |
|---|---|
| 随机猜（全局均值） | 0.1860 |
| 只知道 intent | 0.1147 |
| 知道 intent + 关键词（**不可约下限**） | **0.0982** |

所以 encoder 的 `0.1281` 距下限只有 `+0.030`，Qwen 是 `+0.055`。**71% 的可学信号只是 intent**——`complexity` 不是独立能力，它的价值在演示「一个 encoder 服务三种输出头」与「标签有不可约噪声时指标要对齐下限」。

### 难例剖面（50 条，11 类失败模式）

| 失败模式 | n | encoder | Qwen LoRA | 规则 |
|---|---|---|---|---|
| `multi_intent` 多诉求混合 | 6 | **0.33** | **0.33** | **0.33** |
| `fallback_trap` 只是问信息却被当成诉求 | 4 | 0.25 | 0.25 | **0.00** |
| `vague` 语义模糊 | 6 | 0.33 | 0.50 | **0.83** |
| `adversarial` 反问/讽刺/错别字 | 4 | 0.50 | 0.25 | 0.50 |
| `extreme_length` 极短或极长 | 4 | 0.50 | **0.75** | **0.75** |
| `negation` 否定陷阱 | 4 | 0.50 | 0.50 | **0.75** |
| `coverage` 常规覆盖 | 8 | **0.75** | 0.62 | 0.50 |
| `typo_noise` 错别字/噪声 | 4 | 0.75 | **1.00** | **1.00** |
| `synonym` 同义转述 | 5 | **1.00** | 0.40 | 0.20 |
| `jargon` 领域黑话/中英混杂 | 3 | **1.00** | 0.67 | 0.33 |
| `boundary` 复杂度边界 | 2 | 1.00 | 0.50 | 1.00 |

整体：encoder intent acc `0.600` / Qwen `0.520` / 规则 `0.540`；漏报升级 encoder 10 条、Qwen 12 条、规则 13 条（规则零误报）。

**`multi_intent` 与 `fallback_trap` 是三方共同死穴**——前者因为 120 条训练模板全是单一诉求；后者因为「有强类别信号但用户只是问信息」，规则在这类上**一条都没对**。三方在不同架构上犯同一个错，指向的是**标注覆盖缺口**，不是模型能力。

> ⚠️ 50 条难例**仍不能用来给分支排名**（encoder 0.60 vs Qwen 0.52，n=50 时差异不显著）。它的用途是暴露失败模式。

### 分流与级联（`reports/cascade_benchmark.json`）

真实系统不是「三选一」而是组合。`scripts/11_cascade_benchmark.py` 用交叉验证的逐样本预测（480 条，**未重新训练**）测了这条路：

| 策略 | intent F1 | 成本 ms | 触发率 |
|---|---|---|---|
| `encoder-only` | **0.9058** | 20.67 | — |
| `qwen-only` | 0.8967 | 142.51 | — |
| `rules-only` | 0.5933 | 0.0045 | — |
| `rule-first` | 0.7080 | 6.21 | 规则命中 70% |
| `cascade-enc2qwen`（101 个阈值） | **0.9058**（最优） | ≥ 20.67 | — |
| `oracle-cascade`（**完美门控上界**） | 0.9537 | 34.03 | 升级 9.4% |

**置信度门控完全无效**：扫完 101 个阈值，帕累托前沿只剩 `encoder-only` 一个点——每次升级都是更贵更差。

**机制解释**（`reports/confidence_diagnosis.json`——这个问题 ECE 回答不了）：

| 分支 | 准确率 | ECE | 错误检测 AUROC | AURC |
|---|---|---|---|---|
| encoder | 0.9062 | **0.0570** | 0.7608 | 0.0345 |
| Qwen LoRA | 0.8979 | 0.0746 | **0.8511** | **0.0234** |
| 关键词规则 | 0.5917 | 0.2667 | **0.5224** | 0.4088 |

- encoder 的定位能力是**中等**（AUROC `0.7608`；AURC `0.0345` 明显低于整体错误率 `0.0938`）——**不是「置信度完全没有信息」**；
- **但 ECE 最好的 encoder 在定位能力上反而不如 Qwen**——**校准好 ≠ 能定位错误，ECE 不是充分验收标准**；
- 规则的 AUROC 只有 `0.5224`（≈ 0.5），风险-覆盖曲线是**平的**，结构性无法定位错误；
- 真正的原因是**被挑出来的难样本，第二个分支也做不对**。升级链的价值取决于「第二个分支在第一个分支的错误上是否更强」。

![置信度能否定位错误](../assets/31-02-confidence-diagnosis.png)

### 成本指标

（`reports/latency_benchmark.json`，MPS / fp32）：

| 指标 | encoder-only | Qwen LoRA | 关键词规则 |
|---|---|---|---|
| 端到端 P50 | 20.67 ms | 142.51 ms | **0.0045 ms** |
| 相对规则的倍数 | 4593x | 31669x | 1x |
| intent 批 32 QPS | 1114.2 | 24.6 | **380762** |
| 部署体积（含底座） | 390.6 MB | 1728.1 MB | 0（无需模型） |

**三条结论：**

1. **模型确实比规则好，且提升显著。** 五个质量指标上，两条模型相对规则的差距都是 **5~9 倍标准差**（唯一例外：Qwen 的 complexity MAE 只有 1.08 倍）。**先把这条立住，再谈模型选型。**
2. **两条模型路线彼此分不出高下。** 除 complexity MAE（encoder 更优，比值 1.89）外，intent F1 / ECE / escalation F1 / Brier 四项的差距都远小于 1 倍标准差。**所以选 encoder 的理由是成本低 7 倍，不是质量更高。**
3. **成本跨五个数量级。** 规则快 encoder 约 4600 倍、快 Qwen 约 32000 倍。所以真正的问题不是「encoder 还是 LLM」，而是「模型相对规则提升的那点 F1，值不值 4600 倍延迟」。

### 单次切分 vs 交叉验证

单次切分（test 24 条模板 / 96 样本，多种子 42/43/44）会给出**方向相反**的结论：

| 口径 | encoder | Qwen LoRA | 结论 |
|---|---|---|---|
| 单次切分 intent F1 | 0.7891 ± 0.0345 | **0.8229 ± 0.0068** | Qwen 领先 0.034 |
| 5 折 CV intent F1 | **0.9063 ± 0.0304** | 0.8946 ± 0.0405 | 差 0.012，**不显著** |

单次切分只有 72 条模板用于训练、24 条用于评测；交叉验证每折用 96 条训练，且 120 条模板**全部**被评测。前者不仅噪声大，还会把结论指向错误方向。

> ⚠️ 门槛对照：26 章定的 `F1 >= 0.85` 与 `ECE <= 0.06`。CV 口径下 encoder 的 F1 `0.9058` **已达标**，但 ECE `0.0570` 只是勉强过线；Qwen 两项都未达标（F1 0.8967 / ECE 0.0746）。门槛不是随便定的，达不到是常态。

> ⚠️ **MPS 上训练不是确定性的，而且幅度远超预期。** 同 seed、同超参跑三次 encoder，intent F1 分别是 `0.7404` / `0.7638` / `0.8424`——**极差 `0.102`（10.2 个百分点）**。这比「种子差异」大得多，也远大于交叉验证的折间标准差（`0.0304`）。所以单次运行的数字**不能当作点估计**；分支间差距小于 1 倍标准差时不要过度解读，能报交叉验证就报交叉验证。

## 修复记录（相对历史版本）

| 问题 | 修复 |
|---|---|
| **规则基线能打平所有模型**（模板用词与类别一一对应） | 模板池 48 → **120 条**，按 explicit / conflict / synonym 三层组织各 33%；规则基线 test F1 从 `1.0000` 降到 `0.6142`，模型重新有发挥空间 |
| 数据泄漏：test 100% 与 train 文本重合 | 切分单位从「样本」提到「**模板**」；`assert_no_text_leakage` 双守卫 |
| **缺少规则基线**（29 章整章在讲，工程一行没有） | 新增 `src/minimal_decision_bench/rules.py`，接入 03/05/06/07 全部报告，成为三方对比的第三列 |
| LoRA 只覆盖 6/24 层（Qwen3.5 混合架构） | `_guess_target_modules` 改为扫描实际 Linear 叶子模块名，覆盖 `in_proj_qkv/z/b/a`、`out_proj`；可训练参数 1.09M(0.13%) → 5.52M(0.643%) |
| Brier / routing 误用 top-class 置信度当 P(正类) | `infer_task` 返回完整概率向量 `probs`，escalation 指标与路由改用 `P(True)` |
| 两条分支口径不一致（336 条 vs 128 条、2 epoch vs 1） | 两个脚本共用 `--max-*-samples`，默认全量 |
| `03` 的 routing_examples 是硬编码数字 | 改为从 `reports/<branch>_predictions.json` 的真实预测生成 |
| complexity 标签每次重采样，同文本自相矛盾 | 改为按文本哈希确定性推导 |
| 难例集 `hard_cases.jsonl` 只写不读 | 新增 `scripts/05_eval_hard_cases.py` 消费它，产出错误剖面 |
| 没有延迟/吞吐基准 | 新增 `scripts/06_benchmark_latency.py`（32 章口径） |
| 单次运行就下结论 | 新增 `scripts/07_aggregate_multiseed.py`，多种子报 mean±std |
| 多诉求句标注无协议 | `schemas/v1.json` 写入 annotation_protocol（诉求优先于背景，多诉求按损失优先级） |
| 训练/推理 `max_length` 不一致（192 vs 256） | 统一为 `DEFAULT_MAX_LENGTH` |

量化效果：**泄漏撑起了 encoder 约 11pp 的 intent F1**（1.0000 → 0.8911）；**LoRA 修复把 Qwen 分支从完全塌缩救回来**（intent F1 0.0000 → 0.7328）。

## 结果解释边界（发布前必读）

- 本工程默认数据集（`data/*.jsonl`）是教学用合成数据；
- 对外发布“模型效果结论”前，请替换为真实业务数据并固定评测协议后复现；
- 数据切分按**模板**隔离。历史版本先复制样本再随机切分，导致 test 的 72 条样本 100% 与 train 文本重合。**注意：只按「模板+后缀」的文本切分是不够的**，测试集的底层模板仍可能全在训练集里（实测过：改完文本级切分后 encoder F1 仍是 1.0）；
- 模板池 120 条（每类 30 条，分 `explicit` / `conflict` / `synonym` 三层各 10 条），train/dev/test 各 72/24/24 条模板，切分比例 60/20/20。**难度已标定**：规则基线 F1 `0.5933`，落在「模型有发挥空间」的区间（见主仓库 28.6）；
- **`complexity` 不是独立能力。** 标签由 `(intent, 升级关键词, 文本哈希)` 生成，哈希部分不可学，71% 的可学信号只是 intent。所以它的 MAE 要对齐 `reports/complexity_floor.json` 里的下限（`0.0982`），不要对齐 0；
- **难例集只有 30 条，且真值分布不均衡**（support 13 / general 7 / billing 5 / sales 5）。它的用途是暴露失败模式，**不能用来给分支排名**；
- Qwen 分支已调优（`lr=1e-4`、`warmup_ratio=0.1`、梯度累积 4、cosine），不再是历史版本里的 `batch_size=1 + 无 warmup`；
- 训练在 MPS 上**不确定，且幅度很大**：同 seed 同超参跑三次 encoder，intent F1 是 `0.7404` / `0.7638` / `0.8424`（极差 `0.102`）。分支间差距小于 1 倍标准差时不要过度解读。

## 无模型烟雾验证（不触发重训练）

```bash
uv run python scripts/01_train_encoder.py --dry-run --model-path /path/to/hfl-chinese-macbert-base
uv run python scripts/02_train_qwen_lora.py --dry-run --model-path /path/to/Qwen3.5-0.8B-Base/snapshots/master
uv run pytest -q tests/test_schema_and_routing.py
uv run ruff check src scripts tests
python ../tools/check_chapter_links.py     # 教程章节的相对链接是否还有效
```

CI（`.github/workflows/ci.yml`）跑的就是这几条：ruff + pytest + 章节链接检查。依赖 torch 的那条用例用 `importorskip` 自动跳过，所以 CI 不必拉 torch。

## 输出

```text
data/
  train.jsonl dev.jsonl test.jsonl hard_cases.jsonl
artifacts/
  encoder/{intent,escalation,complexity}
  qwen_lora/{intent,escalation,complexity}
  encoder_s43/...  qwen_lora_s43/...    # --run-tag 产生的多种子产物
  cv_fold0..4/...                       # 交叉验证的折模型（可用 --eval-only 复用）
inference/
  benchmark_v1.md                       # 延迟/吞吐基准报告（32 章）
  quantization_v1.md                    # 量化与 ONNX 基准报告（32 章）
  cascade_v1.md                         # 分流与级联基准报告（31/33 章）
  confidence_diagnosis_v1.md            # 置信度能否定位错误（31 章）
reports/
  encoder_metrics.json
  qwen_lora_metrics.json
  rules_metrics.json             # 关键词规则基线（29 章的下界）
  complexity_floor.json          # 复杂度标签的不可约下限（MAE 要对齐它，不是 0）
  encoder_predictions.json       # 逐样本预测，供 03 生成真实路由样例
  qwen_lora_predictions.json
  rules_predictions.json         # 含 rule_trace，记录命中了哪些规则
  comparison_report.json         # 三方对比
  hard_cases_report.json         # 难例剖面（28/29 章）
  latency_benchmark.json         # 延迟/吞吐基准（32 章）
  quantization_benchmark.json    # 量化与 ONNX 基准（32 章）
  cascade_benchmark.json         # 分流与级联基准（31/33 章）
  confidence_diagnosis.json      # 置信度能否定位错误：AUROC/AURC/风险-覆盖（31 章）
  multiseed_report.json          # 多种子 mean±std 与差距/波动比值
  cross_validation.json          # K 折交叉验证：pooled + 每折 mean±std + 分难度层
  training_logs.json             # 各任务的 loss / grad_norm / lr 曲线（从 checkpoint 抽出的留档）
  # 带 _s42/_s43/_s44 后缀的是 --run-tag 产生的多种子报告（metrics + predictions），
  # 由 07 汇总成 multiseed_report.json，不逐条列出
```

图表由 `09_make_figures.py` 生成到仓库根的 `assets/`，命名沿用 `章节号-序号-主题` 约定（如 `26-01-quality-vs-cost.svg`）。

## 磁盘占用说明

训练默认**不写 checkpoint**（`save_strategy="no"`）。原因：每个 checkpoint 会带一份完整模型 + 优化器状态，实测 3 分支 × 3 任务 × 多种子累积到 22 GB，而推理和评测都用不到中间快照。

- **默认**：`artifacts/{branch}/{task}/` 只有 `model.safetensors`（或 adapter）、config、tokenizer，单任务约 391 MB；
- **需要断点续训**时加 `--save-checkpoints`，此时 `save_total_limit=1`，只保留最近一份；
- 训练曲线（`loss` / `grad_norm` / `learning_rate`）已留档在 `reports/training_logs.json`（36 份），不依赖 checkpoint。

跑完全部流程后 `artifacts/` 约 **11 GB**，构成如下（都是推理需要的，没有中间产物）：

| 目录 | 体积 | 说明 |
|---|---|---|
| `cv_fold0..4/` | 6.5 GB | 交叉验证的 5 折模型——**留着可以用 `--eval-only` 免重训重算指标**（实测省 30 分钟） |
| `encoder_s43/` `encoder_s44/` | 2.2 GB | 多种子的 encoder |
| `encoder/` | 1.1 GB | 主流程的 encoder |
| `onnx/` | 489 MB | ONNX fp32 + int8 模型 |
| `qwen_lora*` | 363 MB | 三条 Qwen LoRA adapter |

历史遗留的 checkpoint 可用下面的命令清理：

```bash
find artifacts -type d -name "checkpoint-*" -prune -exec rm -rf {} +
find artifacts \( -name "optimizer.pt" -o -name "scheduler.pt" -o -name "rng_state.pth" \) -delete
```

清理前后：`artifacts/` 从 **26 GB → 3.8 GB**，而部署体积报告（encoder 390.6 MB / Qwen 1728.1 MB）与所有指标**一字未变**——因为 `06_benchmark_latency.py` 的体积统计本来就排除了这些文件。
