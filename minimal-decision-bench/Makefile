# minimal-decision-bench 常用命令入口
#
#   make help          查看全部目标
#   make main          主流程：造数据 → 训两条分支 → 三方对比 → 难例 → 延迟 → 图表
#   make check         跑 CI 里的那几项（lint + 测试 + 教程链接）
#
# 模型路径可通过环境变量覆盖：
#   make encoder ENCODER_MODEL=/path/to/model

ENCODER_MODEL ?= /path/to/hfl-chinese-macbert-base
QWEN_MODEL ?= /path/to/Qwen3.5-0.8B-Base/snapshots/master
# uv 通常已在 PATH；官方安装脚本会放到 ~/.local/bin，这里做个兜底
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: help sync data encoder qwen compare hard latency quantize cascade confidence \
        multiseed cv figures figures-check main check test lint links secrets checkpoints-clean

help:  ## 查看全部目标
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk -F':.*## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

sync:  ## 安装依赖（dev + train 组）
	$(UV) sync --group dev --group train

data:  ## 生成数据集（按模板切分 + 难度分层 + 泄漏自检）
	$(UV) run python scripts/00_make_data.py

encoder:  ## 训练 encoder-only 分支
	$(UV) run python scripts/01_train_encoder.py --model-path $(ENCODER_MODEL)

qwen:  ## 训练 Qwen LoRA 分支
	$(UV) run python scripts/02_train_qwen_lora.py --model-path $(QWEN_MODEL)

compare:  ## 三方对比 + 生成规则基线
	$(UV) run python scripts/03_compare_and_route.py

hard:  ## 难例剖面（三方）
	$(UV) run python scripts/05_eval_hard_cases.py

latency:  ## 延迟/吞吐基准（三方）
	$(UV) run python scripts/06_benchmark_latency.py

quantize:  ## 量化与 ONNX 推理基准（需要 onnxruntime + onnx）
	$(UV) run python scripts/10_quantize_benchmark.py

cascade:  ## 分流与级联基准（用交叉验证的逐样本预测，不需重新训练）
	$(UV) run python scripts/11_cascade_benchmark.py

confidence:  ## 置信度诊断：AUROC / AURC / 风险-覆盖（不需重新训练）
	$(UV) run python scripts/12_confidence_diagnosis.py

multiseed:  ## 多种子重跑并汇总（3 个 seed，较慢）
	@for s in 42 43 44; do \
	  echo "=== seed $$s ==="; \
	  $(UV) run python scripts/01_train_encoder.py --model-path $(ENCODER_MODEL) --seed $$s --run-tag _s$$s; \
	  $(UV) run python scripts/02_train_qwen_lora.py --model-path $(QWEN_MODEL) --seed $$s --run-tag _s$$s; \
	  $(UV) run python scripts/03_compare_and_route.py --run-tag _s$$s; \
	done
	$(UV) run python scripts/07_aggregate_multiseed.py --seeds 42,43,44

cv:  ## K 折交叉验证（5 折，约 40 分钟）
	$(UV) run python scripts/08_cross_validate.py --folds 5 \
	  --encoder-model-path $(ENCODER_MODEL) --qwen-model-path $(QWEN_MODEL)

figures:  ## 生成教程图表到 ../assets/
	$(UV) run python scripts/09_make_figures.py

main: data encoder qwen compare hard latency figures  ## 主流程（不含慢的多种子与 CV）

check: lint test links secrets figures-check  ## CI 里的全部检查 + 密钥扫描 + 图表新鲜度

secrets:  ## 扫描 git 跟踪文件与提交历史里是否混进凭据
	python ../tools/check_no_secrets.py --history

figures-check:  ## 检查图表是否比源报告旧（图由 reports/ 生成，容易静默过期）
	$(UV) run python scripts/09_make_figures.py --check

test:  ## 跑测试
	$(UV) run pytest -q tests

lint:  ## 静态检查
	$(UV) run ruff check src scripts tests

links:  ## 检查教程章节的相对链接
	python ../tools/check_chapter_links.py

checkpoints-clean:  ## 清理训练中间产物（checkpoint 与优化器状态）
	find artifacts -type d -name "checkpoint-*" -prune -exec rm -rf {} +
	find artifacts \( -name "optimizer.pt" -o -name "scheduler.pt" -o -name "rng_state.pth" \) -delete
