# Agentic RL 零基础教程 · 从概念到 GRPO 实战

> **面向小白的 Agentic RL（智能体强化学习）系统教程** — 33 篇中文 Markdown + 两套**定位不同**的可运行工程：
> 判别模型三方对照实证（[`minimal-decision-bench/`](./minimal-decision-bench/)，26–33 章数字的来源）与训练工具链最小示例（[`minimal-verl/`](./minimal-verl/)，TRL 栈 / macOS MPS / 单步任务，**不演示 Agentic RL 能力**）。
>
> 搜「Agentic RL 教程」「GRPO 入门」「LLM 强化学习」「verl TRL 实战」「Jev 与 RL 的边界」「System One 判别模型」「判别能力外置」都能找到这里。

[![GitHub stars](https://img.shields.io/github/stars/cookiespiggy/agentic-rl?style=social)](https://github.com/cookiespiggy/agentic-rl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/cookiespiggy/agentic-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/cookiespiggy/agentic-rl/actions/workflows/ci.yml)

> **只想看结论？** → [`minimal-decision-bench/FINDINGS.md`](./minimal-decision-bench/FINDINGS.md)：一页纸说清「这个实证工程证明了什么、没证明什么」。

> **质量分层（先看清楚再往下读）**：本仓库两部分的**证据等级不一样**，不要一视同仁。
>
> | 章节 | 性质 | 证据强度 |
> |---|---|---|
> | **01–24** | **知识整理 + 示意片段** | 框架 API 是按 verl `0.9.1` / TRL `1.5.1` 的**真实源码**写的，但**训练脚本不在 CI 里执行**——本机无 CUDA，verl 训练、BrowserGym/Docker/SQL 环境**永久无法实机验证**。正确性靠**离线 lint（config key 快照对照）+ 人工核对上游源码**保证 |
> | **25–33** | **有可复现实验支撑** | **实测**数字来自 [`minimal-decision-bench/`](./minimal-decision-bench/) **入库的 `reports/*.json`**，CI 跑 lint + 单测 + 章节链接检查；口径与统计结论见各章末尾的「本章哪些内容本工程未实现」。**例外**：25 章的 Jev 产品指标是**厂商公开口径，本仓库未独立验证** |
>
> 两套工程**不是平行的两条路**，定位不同：
> - [`minimal-decision-bench/`](./minimal-decision-bench/) —— **判别模型的实证工程**，本 README「实证工程」一节所有数字的来源。
> - [`minimal-verl/`](./minimal-verl/) —— **训练工具链的最小示例**（TRL 栈 / macOS MPS / 单步任务），它验证的是 *SFT → GRPO → eval 这条链路能不能跑通*，**不是 Agentic RL 能力示例**。它自己的 README 写得很直白：「情感分类本身不需要 Agentic RL……**它是『工具链验证示例』，不是『Agentic RL 能力示例』**」。

---

## 📚 同作者的姊妹篇：如果你还不会写 Python / 没做过 RAG

<table>
<tr>
<td width="50%" valign="top">

**《Java 转 AI 的 90 天》** — 写给 Java 后端工程师的 AI 转型路线

[在线阅读](https://cookiespiggy.github.io/java-to-ai-in-90-days/) · [GitHub](https://github.com/cookiespiggy/java-to-ai-in-90-days)

**本仓库 vs 那一本，区别在哪：**

| | 《Java 转 AI 的 90 天》 | **本仓库** |
|---|---|---|
| 讲什么 | **应用层**：怎么把模型接进业务 | **训练层**：怎么让模型变强 |
| 适合谁 | 会 Java，没碰过 AI | 已经能稳定上线 RAG/Agent |
| RAG | [13 篇完整项目](https://github.com/cookiespiggy/java-to-ai-in-90-days/tree/main/projects/rag-project)，top1 **7/7** | 只讲检索原理 |
| 本地推理 | [23 篇实测](https://cookiespiggy.github.io/java-to-ai-in-90-days/23-local-inference/)，本地 4B 零 API 费用 | [32 章量化与延迟基准](./32-推理优化-量化批处理与延迟基准.md) |
| 实测数据 | 每篇都留了真实输出和踩坑记录 | 有三方对照实证 |

**推荐路径：先读那本打底 → 再回来读本仓库拔高。**

</td>
</tr>
</table>

---

## 这是什么？

**Agentic RL** 是把大语言模型（LLM）训练成**能规划、用工具、多轮交互**的智能体（Agent）的方法。  
本仓库不是论文搬运，而是一条**能跟着走的自学路线**：

| 你现在的水平 | 从这里开始 |
|-------------|-----------|
| 完全零基础，没听过 RL | [01 - 什么是 Agentic RL](./01-什么是Agentic-RL-概念与愿景.md) |
| 懂 LLM，想搞懂 SFT / RLHF / GRPO | [03 - LLM 与 Post-Training](./03-LLM与Post-Training基础.md) → [08 - GRPO 深度解析](./08-GRPO深度解析.md) |
| 想动手跑训练 | [minimal-verl/](./minimal-verl/) 最小示例 + [15 - 第一个训练实战](./15-第一个Agentic-RL训练实战.md) |
| **搜「Jev」进来的** | [25 - 判别能力外置：Jev 出现后，什么时候不该用 RL](./25-判别能力外置-什么时候不该用RL.md) |
| 想从零造垂类判断模型 | [26-33 垂直领域判别模型实现线](./26-垂直领域判别模型总览-从判别外置到自建内核.md) |

**核心理念**：小模型 + Agentic RL，在垂直任务上可以超越更大的通用 LLM。

---

## 实证工程：判别模型到底该怎么选（26–33 章配套）

[`minimal-decision-bench/`](./minimal-decision-bench/) 不是示例代码，是一套**可复现的三方对照实验**——同一份 schema、同一份数据、同一套指标，把三条判别路线放在一起比：

| 路线 | 角色 |
|---|---|
| 关键词规则 `rules_v0` | **下界**：不做模型能到多少 |
| encoder-only（中文 MacBERT，102M） | **推荐路线** |
| Qwen3.5-0.8B LoRA | **对照**：用 LLM 做同一件事 |

**它得出的结论里有几条是反直觉的**（完整数据见 [`FINDINGS.md`](./minimal-decision-bench/FINDINGS.md)）：

| 结论 | 证据 |
|---|---|
| 质量上 encoder 与 LLM **没有一项指标达到统计显著** | 5 个指标上，依赖校正 + Holm 校正后**没有一个**存活。其中 **escalation Brier 方向 5/5 折一致**（encoder 更优，约 12%），但依赖校正 p=`0.0107`、精确 Wilcoxon p=`0.0625`——**定性为「值得注意但未证实」**，不宣称统计显著 |
| 但成本差一个量级 | 端到端 P50 `20.67 ms` vs `142.51 ms`（**三个任务模型串行求和**）；部署体积**单任务** `390.6 MB` vs `1728.1 MB`，**三任务合计** `1171.8 MB` vs `5184.3 MB` |
| **量化后 CPU 比 GPU 还快** | ONNX int8 `4.04 ms` vs MPS fp32 `6.78 ms`（快 1.68 倍），体积压到 1/4（`98.3 MB`） |
| **置信度门控（级联）完全无效** | 扫 101 个阈值，帕累托前沿只剩 `encoder-only` **一个点** |
| **数据集难度决定结论** | 早期模板下规则基线 F1 是 `1.0000`（比两个模型都高）——那时证明的是「不需要模型」 |

> **统计口径**：上面「0.33 倍标准差」这类数字是**描述性比值**（总体标准差 ddof=0），**不是显著性检验**。真正的推断要走三层检验（朴素配对 t → Nadeau–Bengio 依赖校正 t → 精确 Wilcoxon）叠 Holm 校正——本仓库跑完后的结论就是上面第一行：**没有任何指标存活**。详见 [31 章](./31-训练与校准-让置信度真正可用.md) 与 `reports/gap_stats.json`。

规模：**15 个脚本 / 23 份入库报告 / 44 条测试 / 4 张图 / CI 全绿**。

```bash
cd minimal-decision-bench
make sync && make main   # 造数据 → 训两条分支 → 三方对比 → 难例 → 延迟 → 图表（本机 7 分 40 秒）
make cv                  # 5 折交叉验证（约 40 分钟，想要稳健结论再跑）
make check               # lint + 测试 + 教程链接
```

> **发布边界**：默认数据是教程合成集，用于证明工程可跑通与结论方向；对外发布业务结论前，请替换为真实数据并固定评测协议后复现。

---

## 小白先看：从教程到代码的 30 分钟闭环

如果你第一次接触这个仓库，按下面顺序走，不会迷路：

1. 先读 [26](./26-垂直领域判别模型总览-从判别外置到自建内核.md) 理解目标与边界。
2. 打开 [`minimal-decision-bench/LEARNER-GUIDE.md`](./minimal-decision-bench/LEARNER-GUIDE.md) 看“章节到代码”导航图。
3. 在 `minimal-decision-bench/` 里执行 `make main`（等价于依次跑 `00_make_data.py → 01_train_encoder.py → 02_train_qwen_lora.py → 03_compare_and_route.py → 05_eval_hard_cases.py → 06_benchmark_latency.py → 09_make_figures.py`）。想看稳健结论再跑 `make cv`。
4. 带着报告回读 27–33，逐章对照“为什么这样设计”。

26–33 与 `minimal-decision-bench` 的一一对应如下：

| 教程章节 | 你要理解什么 | 直接对应的工程文件 |
|---|---|---|
| [26](./26-垂直领域判别模型总览-从判别外置到自建内核.md) | 系统边界与发布边界 | [`README.md`](./minimal-decision-bench/README.md), [`LEARNER-GUIDE.md`](./minimal-decision-bench/LEARNER-GUIDE.md) |
| [27](./27-任务建模-把业务流程压成可训练Schema.md) | schema 建模 | [`schemas/v1.json`](./minimal-decision-bench/schemas/v1.json), [`src/minimal_decision_bench/schema.py`](./minimal-decision-bench/src/minimal_decision_bench/schema.py) |
| [28](./28-数据工程I-标注协议与难例覆盖.md) | 数据协议与 hard cases | [`scripts/00_make_data.py`](./minimal-decision-bench/scripts/00_make_data.py), [`src/minimal_decision_bench/data_builder.py`](./minimal-decision-bench/src/minimal_decision_bench/data_builder.py) |
| [29](./29-数据工程II-规则基线与错误剖面.md) | 可解释基线与误差剖面意识 | [`scripts/03_compare_and_route.py`](./minimal-decision-bench/scripts/03_compare_and_route.py), `reports/comparison_report.json`（已入库） |
| [30](./30-模型骨架-Encoder+DecisionHead设计.md) | 三头任务骨架 | [`src/minimal_decision_bench/trainers.py`](./minimal-decision-bench/src/minimal_decision_bench/trainers.py), [`scripts/01_train_encoder.py`](./minimal-decision-bench/scripts/01_train_encoder.py) |
| [31](./31-训练与校准-让置信度真正可用.md) | 训练与置信度指标 | [`src/minimal_decision_bench/metrics.py`](./minimal-decision-bench/src/minimal_decision_bench/metrics.py), `reports/encoder_metrics.json`, `reports/qwen_lora_metrics.json`（已入库） |
| [32](./32-推理优化-量化批处理与延迟基准.md) | 推理路径与部署前优化 | [`src/minimal_decision_bench/routing.py`](./minimal-decision-bench/src/minimal_decision_bench/routing.py), [`scripts/03_compare_and_route.py`](./minimal-decision-bench/scripts/03_compare_and_route.py) |
| [33](./33-上线治理-灰度回滚与持续进化.md) | 上线治理与回滚策略 | `reports/comparison_report.json`（已入库，作为治理输入样例） |

---

## 学习路线（33 章）

### 阶段一 · 基础概念（01–05）

| 章节 | 主题 | 难度 |
|------|------|------|
| [01](./01-什么是Agentic-RL-概念与愿景.md) | Agentic RL 概念与愿景 | 入门 |
| [02](./02-基础知识-强化学习入门.md) | 强化学习入门（MDP、策略梯度） | 入门 |
| [03](./03-LLM与Post-Training基础.md) | LLM 与 Post-Training（SFT / RLHF / DPO / GRPO） | 入门 |
| [04](./04-从LLM-RL到Agentic-RL.md) | 从 LLM RL 到 Agentic RL | 入门 |
| [05](./05-Agentic-RL的应用场景.md) | 应用场景全景 | 入门 |

### 阶段二 · 核心算法（06–10）

| 章节 | 主题 | 难度 |
|------|------|------|
| [06](./06-PPO算法详解.md) | PPO 算法详解 | 中级 |
| [07](./07-DPO直接偏好优化.md) | DPO 直接偏好优化 | 中级 |
| [08](./08-GRPO深度解析.md) | **GRPO 深度解析**（2025–2026 主流） | 中级 |
| [09](./09-RLVR可验证奖励的强化学习.md) | RLVR 可验证奖励 | 中级 |
| [10](./10-Reward-Shaping奖励设计进阶.md) | Reward Shaping 奖励设计 | 中级 |

### 阶段三 · 训练工程（11–15）

| 章节 | 主题 | 难度 |
|------|------|------|
| [11](./11-训练框架选型与搭建.md) | 训练框架选型（verl / TRL） | 中级 |
| [12](./12-数据准备与SFT阶段.md) | 数据准备与 SFT | 中级 |
| [13](./13-训练环境搭建.md) | 训练环境搭建 | 中级 |
| [14](./14-奖励函数设计与实现.md) | 奖励函数设计 | 中级 |
| [15](./15-第一个Agentic-RL训练实战.md) | **第一个完整训练实战** | 实战 |

### 阶段四 · 环境与实战（16–20）

| 章节 | 主题 |
|------|------|
| [16](./16-网页导航Agent训练.md) | 网页导航 Agent |
| [17](./17-代码Agent训练.md) | 代码 Agent |
| [18](./18-搜索增强Agent训练.md) | 搜索增强 Agent |
| [19](./19-垂直领域Agent训练.md) | 垂直领域 Agent |
| [20](./20-评估与Benchmark.md) | 评估与 Benchmark |

### 阶段五 · 进阶前沿（21–24）

| 章节 | 主题 |
|------|------|
| [21](./21-异步Rollout与分布式训练.md) | 异步 Rollout 与分布式训练 |
| [22](./22-多Agent与Multi-Agent-RL.md) | 多 Agent / Multi-Agent RL |
| [23](./23-前沿论文与研究方向.md) | 前沿论文与研究方向 |
| [24](./24-学习总结与进阶路径.md) | 学习总结与进阶路径 |

### 阶段六 · 时代补充（25）

| 章节 | 主题 |
|------|------|
| [25](./25-判别能力外置-什么时候不该用RL.md) | **判别能力外置：Jev 出现后，什么时候不该用 RL** |

### 阶段七 · 垂直领域判别模型 实现线（26–33）

| 章节 | 主题 |
|------|------|
| [26](./26-垂直领域判别模型总览-从判别外置到自建内核.md) | 垂直领域判别模型 总览：从判别外置到自建内核 |
| [27](./27-任务建模-把业务流程压成可训练Schema.md) | 任务建模：把业务流程压成可训练 Schema |
| [28](./28-数据工程I-标注协议与难例覆盖.md) | 数据工程 I：标注协议与难例覆盖 |
| [29](./29-数据工程II-规则基线与错误剖面.md) | 数据工程 II：规则基线与错误剖面 |
| [30](./30-模型骨架-Encoder+DecisionHead设计.md) | 模型骨架：Encoder + Decision Head 设计 |
| [31](./31-训练与校准-让置信度真正可用.md) | 训练与校准：让置信度真正可用 |
| [32](./32-推理优化-量化批处理与延迟基准.md) | 推理优化：量化、批处理与延迟基准 |
| [33](./33-上线治理-灰度回滚与持续进化.md) | 上线治理：灰度、回滚与持续进化 |

---

## Jev 与 RL 的边界（从「Jev」搜过来的话，先看这一节）

2026 年 9 月，TypeSafe AI 发布首个 **System One** 模型 **Jev**——它不生成文本，只返回类型化决策（`choice` / `score` / `noul`），单次延迟 70–500 ms。

> 口径说明：Jev 术语里常写 `noul`；本仓库 26–33 章的工程实现统一落到 `bool` 字段（如 `needs_escalation: true/false`），语义等价。

这带来一个绕不开的问题：**判别类任务还需要自己 RL 吗？**

本教程的立场是：**判别类可以外置，策略类必须自己训。**

> **一句话回答**：Jev 替代的不是 RL，而是"用 RL 去做判别任务"这个做法。它本身就是 RL 的产物——置信度校准能力来自 RLCD（Reinforcement Learning for Calibrated Decisions），所以理解 RL，才能理解它的边界在哪里失效。

| 任务类型 | 该用什么 | 本教程对应章节 |
|---------|---------|--------------|
| 单步、闭集判断（分类 / 路由 / 闸门） | 判断模型或规则引擎 | 见 [25](./25-判别能力外置-什么时候不该用RL.md) |
| 单步、需要数值比较 | 规则引擎（代码） | 见 [25.5](./25-判别能力外置-什么时候不该用RL.md) |
| **多步轨迹、信用分配、编排** | **必须 Agentic RL** | **16–22 章** |
| 开放式生成 | SFT + RL | [12](./12-数据准备与SFT阶段.md)、[15](./15-第一个Agentic-RL训练实战.md) |

**为什么本教程没有过时——三条理由：**

1. **训练循环里用不了外部判断服务。** 奖励函数每次迭代被调用数百万次，延迟预算在毫秒级；外部判断服务单次 70–500 ms，量级不匹配。详见 [14.3 训练期硬约束](./14-奖励函数设计与实现.md)。
2. **判断模型不碰策略问题。** 它是单步、近视的——没有轨迹概念，不做信用分配，不涉及多 Agent 协调。而这正是本教程 16–22 章的全部内容。
3. **判断模型本身就是 RL 的产物。** 其置信度校准能力来自 RLCD（Reinforcement Learning for Calibrated Decisions）。理解 RL，才能理解它的能力边界在哪里失效。

**一句话**：本教程教的不是"怎么训一个分类器"，而是"怎么让模型学会在一条轨迹上做对决策"。前者可以被 Schema 替代，后者不行。

**本仓库的定位**：大多数 Jev 内容讲的是"它多快、多便宜、怎么接入"。而"它替代了什么、没替代什么"这个问题，需要同时理解 RL 与判断模型才能回答——本仓库补的正是这一块。

---

## 动手练：minimal-verl 最小示例

[`minimal-verl/`](./minimal-verl/) 是一个**真正能跑**的精简项目，目标不是训出生产模型，而是理解整条链路：

```
人造数据 → SFT → GRPO → 评估
```

- **模型**：Qwen2.5-0.5B（Mac M 系列可用 MPS）
- **任务**：中文情感三分类
- **框架**：TRL `GRPOTrainer`（比 verl 更适合入门）

> ⚠️ **它和 `minimal-decision-bench/` 不是平行的两条路，别混为一谈。**
>
> | | `minimal-verl/` | `minimal-decision-bench/` |
> |---|---|---|
> | 栈 | **TRL**（`GRPOTrainer`）、macOS / **MPS**、**单步任务** | PyTorch / sklearn，**判别模型**（分类 + 回归） |
> | 验证什么 | **工具链**能不能跑通：SFT → GRPO → eval | **判别路线**该怎么选：规则 vs encoder vs LLM |
> | 不验证什么 | **不验证 Agentic RL 能力**——情感三分类是单步闭集任务，按 [25 章](./25-判别能力外置-什么时候不该用RL.md) 的判据本来就不需要 RL | 不验证策略类任务 |
>
> `minimal-verl/` 自己的 README 写得很直白：「**情感分类本身不需要 Agentic RL**……**它是『工具链验证示例』，不是『Agentic RL 能力示例』**」。它练的是**工具**，不是**能力**。

```bash
cd minimal-verl
uv sync
uv run python scripts/00_check_env.py   # 环境验证（5 步）
```

详细进度见 `minimal-verl/docs/PROGRESS.md`（内部协作文档，不入库）。

判别模型的工程闭环实践（对应 26–33 章）见上文「实证工程」一节——那里有完整命令与关键结论。

> 中国大陆网络建议先设置镜像并使用本地模型目录：
>
> `UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`

---

## 适合谁？

- 想入门 **LLM 强化学习 / RLHF / GRPO** 的开发者
- 想训练 **Agent**（工具调用、多轮交互）但不知道从哪下手
- 看过 verl、DeepSeek-R1 新闻，想搞懂背后训练流程
- 想知道**判别类任务到底要不要自己训**、判断模型与 RL 的边界在哪的工程决策者
- 中文学习者，希望有**结构化路线**而不是零散博客

---

## 关键词（方便搜索）

`Agentic RL` · `智能体强化学习` · `GRPO` · `PPO` · `DPO` · `RLHF` · `RLVR` · `LLM Post-Training` · `verl` · `TRL` · `Qwen` · `SFT` · `Reward Shaping` · `工具调用 Agent` · `小模型 RL` · **`Jev`** · `TypeSafe` · `System One` · `判别模型` · `闭集判断` · `判别能力外置` · `RLCD` · `能力边界` · `梯度扫描` · `判别模型 benchmark` · `encoder 微调` · `MacBERT` · `规则基线` · `置信度校准` · `ECE` · `交叉验证` · `模型量化` · `ONNX Runtime` · `级联分流` · `推理延迟基准`

---

## 关于作者

由 [@cookiespiggy](https://github.com/cookiespiggy) 维护 — 持续更新教程与可运行示例，欢迎 Star ⭐ 和 Issue 反馈。

---

## License

MIT — 教程内容可自由学习、引用，请注明出处。
