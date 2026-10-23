#!/usr/bin/env python3
"""检查教程里的 Python 代码块：能不能解析、有没有未定义的名字。

**为什么要 fence-aware**
----------------------
朴素的「```python … ```」正则会在**源码里含字面量 ```** 的块上截断
（本仓库的 14 章、17 章奖励函数示例里就有），从而报出假的语法错误。
所以这里按行扫，自己配对开闭围栏。

**「未定义名」的判定**
--------------------
代码块常常是**片段**（比如只展示奖励函数体），上下文里的辅助函数定义在别的块里。
所以未定义名只做**提示**，并且：
- 跳过 Python 内置名、常见第三方库名（torch / numpy / pandas / transformers / trl …）
- 跳过块内自己定义的名字
- 同名出现在**同一章的其他块**里 → 不算未定义
- 明确标了 `pseudocode` 的块跳过语法检查

用法：
    python tools/check_chapter_python.py            # 严格模式，失败退出码 1
    python tools/check_chapter_python.py --warnings  # 只报未定义名，不失败
"""

from __future__ import annotations

import argparse
import ast
import builtins
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE = re.compile(r"^\s*```(\w*)\s*$")

# 常见第三方库与框架名（章节片段里经常直接用）
KNOWN_NAMES = {
    "torch", "numpy", "np", "pandas", "pd", "json", "os", "sys", "re", "math",
    "random", "time", "logging", "typing", "dataclasses", "collections",
    "itertools", "functools", "pathlib", "Path", "datetime", "subprocess",
    "shutil", "tempfile", "textwrap", "copy", "warnings", "contextlib",
    "asyncio", "abc", "enum", "statistics", "csv", "sqlite3", "hashlib",
    "argparse", "yaml", "requests", "httpx", "transformers", "datasets",
    "trl", "peft", "accelerate", "sklearn", "matplotlib", "pytest",
    "List", "Dict", "Tuple", "Optional", "Any", "Union", "Callable",
    "Iterable", "Sequence", "Iterator", "Set", "Mapping", "Self",
    "AutoTokenizer", "AutoModelForCausalLM", "AutoConfig", "AutoModel",
    "Trainer", "TrainingArguments", "DPOTrainer", "DPOConfig",
    "GRPOTrainer", "GRPOConfig", "SFTTrainer", "SFTConfig",
    "PreTrainedTokenizer", "PreTrainedModel", "Dataset", "load_dataset",
    "torch", "nn", "DataLoader", "Tensor",
}

# 这些名字在各章里是「约定俗成」的，跨章复用，不算未定义
CROSS_CHAPTER_OK = {
    "main", "evaluate_agent", "extract_answer", "reward_func", "reward_fn",
    "compute_grpo_loss", "safe_execute", "DockerCodeSandbox", "BaseEnvironment",
    "CalculatorTool", "WebNavEnvironment", "SearchEnvironment",
    "DataAnalysisEnvironment", "SelfPlayTrainer", "CTDEMultiAgentTrainer",
    "_coerce_to_text", "coerce_to_text",
    # minimal-decision-bench 的真实 API 名（章节引用的是工程里的代码）
    "TaskConfig", "train_task", "train_rows", "dev_rows", "labels",
    "initialize_tools_from_config", "load_all_tools", "ToolType",
}

# 章节里**有意只给片段**的辅助函数：正文会说明它们「假设已存在」或在前文定义过。
# 逐个列出，避免把它们当成真 bug。
FRAGMENT_HELPERS = {
    # 06 PPO：GAE 与 KL 的实现留给读者（正文给了完整公式）
    "compute_gae", "compute_kl",
    # 08 GRPO：KL 估计器的实现细节不是本章重点
    "compute_kl_divergence",
    # 09 RLVR：一组奖励函数的判定辅助函数
    "execute", "has_hallucination", "has_think_tags", "has_tool_call_format",
    "has_correct_format", "has_repetition", "verify_correctness",
    "extract_final_answer", "verify_math_answer",
    # 10 Reward Shaping：过程奖励的分解辅助函数
    "count_errors", "has_errors", "has_thinking_process", "has_unnecessary_steps",
    "partial_progress", "task_completed", "prm",
    # 10 里的示意变量
    "step1", "step2", "step3",
    # 14：奖励函数签名里的 kwargs
    "reward_kwargs",
    # 28：sklearn 的评估片段
    "f1_score", "y_pred", "y_true",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--warnings", action="store_true", help="未定义名只提示，不失败")
    return p.parse_args()


def iter_python_blocks(text: str):
    """产出 (起始行号, 代码, 是否伪代码)。

    只处理**显式标了 python / py** 的块。
    未标注语言的块（75 个）里绝大多数是 ASCII 流程图、终端输出、中文伪代码——
    那些不是 Python，硬解析只会报出一堆假错误。
    """
    lines = text.splitlines()
    in_block = False
    lang = ""
    start = 0
    buf: list[str] = []
    for i, line in enumerate(lines, 1):
        m = FENCE.match(line)
        if m:
            if not in_block:
                in_block = True
                lang = m.group(1).lower()
                start = i
                buf = []
            else:
                if lang in {"python", "py", "pseudocode", "pseudo"}:
                    yield start, "\n".join(buf), lang.startswith("pseudo")
                in_block = False
                lang = ""
                buf = []
            continue
        if in_block:
            buf.append(line)


def collect_defined(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def collect_used(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def main() -> None:
    args = parse_args()
    files = sorted(ROOT.glob("[0-9][0-9]-*.md"))

    syntax_errors: list[str] = []
    undefined: list[str] = []
    total_blocks = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        chapter_defined: set[str] = set()
        blocks = list(iter_python_blocks(text))
        for _, code, _ in blocks:
            chapter_defined |= collect_defined(code)

        for start, code, is_pseudo in blocks:
            total_blocks += 1
            if is_pseudo:
                continue
            try:
                ast.parse(code)
            except SyntaxError as exc:
                syntax_errors.append(
                    f"{path.name}:{start}: 语法错误：{exc.msg}（行 {exc.lineno}）"
                )
                continue
            used = collect_used(code)
            defined = collect_defined(code)
            for name in sorted(used - defined - chapter_defined):
                if name in builtins.__dict__:
                    continue
                if name in KNOWN_NAMES or name in CROSS_CHAPTER_OK:
                    continue
                if name in FRAGMENT_HELPERS:
                    continue
                if name.isupper():
                    continue  # 常量
                undefined.append(f"{path.name}:{start}: 未定义名 `{name}`")

    if syntax_errors:
        print(f"发现 {len(syntax_errors)} 处语法错误：\n")
        for p in syntax_errors:
            print(f"  {p}")
        print()
    if undefined:
        print(f"发现 {len(undefined)} 处疑似未定义名（可能是有意的片段，请人工确认）：\n")
        for p in undefined:
            print(f"  {p}")
        print()

    if syntax_errors:
        sys.exit(1)
    if undefined and not args.warnings:
        print("（加 --warnings 只提示不失败）")
        sys.exit(1)

    print(f"章节 Python 代码块检查通过（{total_blocks} 个块，0 语法错误）")


if __name__ == "__main__":
    main()
