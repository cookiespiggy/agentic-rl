#!/usr/bin/env python3
"""离线校验教程里的 verl 训练命令：Hydra 覆盖项是不是 verl 真实存在的 key。

**为什么要这个工具**：verl 的配置面很大，而且命令写在 .md 的 ```bash 代码块里，
不装 verl 就没法用 `verl.trainer.main_ppo --help` 之类的方式核对。装 verl 又要 CUDA。

做法：把 verl v0.9.1 的**预合成**配置（`verl/trainer/config/_generated_ppo_trainer.yaml`）
的 key 路径抽成快照存在 `tools/verl_config_keys.yaml`，这里只做纯文本比对：
- 入口必须是 `python -m verl.trainer.main_ppo`（verl 没有 `verl/trainer/main.py`）
- 每个 `key.path=value` 覆盖项的 key 路径必须在快照里
- 自由透传的 dict 节点（如 `actor_rollout_ref.rollout.engine_kwargs.sglang.*`）
  只要**父节点存在**就放行——它下面的键是运行时透传给引擎的

另外检查每条训练命令是否带齐四个必填项（少了会静默用错默认值）：
- `actor_rollout_ref.actor.optim.lr`（verl 各版本的有效默认值不同）
- `data.prompt_key`（数据列名是 `messages` 时必须显式指定）
- `data.train_batch_size`（默认 1024，配 `rollout.n` 会放大成上万条 completion）
- `actor_rollout_ref.rollout.n`（组大小）

用法：
    python tools/check_verl_configs.py            # 校验并打印
    python tools/check_verl_configs.py --quiet    # 只在有问题时输出
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "tools" / "verl_config_keys.yaml"

# 入口必须是这个（verl/trainer/ 下没有 main.py）
ENTRYPOINT = "python -m verl.trainer.main_ppo"
BAD_ENTRYPOINTS = (
    "python -m verl.trainer.main",   # 精确匹配：后面必须是 _ppo 或行尾
)
# 判据：出现 "verl.trainer.main" 且其后不是 "_ppo" → 错误

# 必须显式写出的四个键
REQUIRED_KEYS = (
    "actor_rollout_ref.actor.optim.lr",
    "data.prompt_key",
    "data.train_batch_size",
    "actor_rollout_ref.rollout.n",
)

# 明确记录「verl 没有这些键」——教程里如果还写着它们，就是抄了旧文档或别家框架
FORBIDDEN_KEYS = (
    "algorithm.kl_coeff",
    "actor_rollout_ref.rollout.max_new_tokens",
    "actor_rollout_ref.strategy",
    "ray.num_workers",
    "ray.max_restarts",
    "ray.health_check_interval",
    "ray.restart_delay",
    "trainer.save_limit",
    "trainer.save_total_limit",
    "trainer.logger.wandb",
    "rollout.vllm.tensor_parallel_size",
    "rollout.sglang.mem_fraction_static",
)

# `tools.xxx.enabled` 这种「根本没有 tools 配置段」的写法
TOOLS_PATTERN = re.compile(r"(?<![\w.])tools\.[\w.]+\.enabled")

# 从 markdown 里提取 ```bash 代码块（fence-aware：按行扫，遇到 ``` 开关）
FENCE = re.compile(r"^\s*```(\w*)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true", help="只在有问题时输出")
    return p.parse_args()


def iter_bash_blocks(text: str):
    """产出 (起始行号, 代码内容)。只取 bash/sh/shell 标注的块。"""
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
                if lang in {"bash", "sh", "shell", ""} and any(
                    "verl.trainer" in b for b in buf
                ):
                    yield start, "\n".join(buf)
                in_block = False
                lang = ""
            continue
        if in_block:
            buf.append(line)


def load_snapshot() -> tuple[set[str], set[str]]:
    if not SNAPSHOT.exists():
        print(f"缺少 {SNAPSHOT.relative_to(ROOT)}")
        sys.exit(2)
    try:
        import yaml
    except ImportError:  # pragma: no cover
        print("需要 pyyaml：pip install pyyaml")
        sys.exit(2)
    doc = yaml.safe_load(SNAPSHOT.read_text(encoding="utf-8"))
    return set(doc.get("known_keys") or []), set(doc.get("leaf_keys") or [])


def check_command(block: str, known: set[str], leaves: set[str]) -> list[str]:
    """返回这条命令里的问题列表（空列表 = 合法）。"""
    problems: list[str] = []
    if "verl.trainer.main" not in block:
        return problems

    # 入口必须精确是 main_ppo：出现 verl.trainer.main 而后面不是 _ppo 就是错的
    for m in re.finditer(r"verl\.trainer\.main\b(?!_ppo)", block):
        problems.append(f"入口应为 `{ENTRYPOINT}`，不是 `verl.trainer.main`")
        break

    overrides: dict[str, str] = {}
    seen_entrypoint = False
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "verl.trainer.main" in line:
            seen_entrypoint = True
            continue
        if not seen_entrypoint:
            # 入口之前是 shell 变量赋值，不是 Hydra 覆盖项
            continue
        m = re.match(r"^([A-Za-z_][\w.]*)=(.+)$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        # shell 变量（纯大写 + 下划线）不是配置键
        if "." not in key and key.upper() == key:
            continue
        overrides[key] = value

    for key in overrides:
        if key in known or key in leaves:
            continue
        # 自由透传：父节点存在且其下没有叶子定义（引擎透传参数）
        parent = key.rsplit(".", 1)[0]
        if parent and parent in known and not any(
            leaf.startswith(parent + ".") for leaf in leaves
        ):
            continue
        problems.append(f"verl 里没有这个配置键：{key}")

    for bad in FORBIDDEN_KEYS:
        if bad in overrides:
            problems.append(f"verl 里没有这个配置键：{bad}")

    if TOOLS_PATTERN.search(block):
        problems.append("verl 没有 tools.* 配置段；工具接入用 actor_rollout_ref.rollout.multi_turn.*")

    for required in REQUIRED_KEYS:
        if required not in overrides:
            problems.append(f"缺少必填项：{required}")

    return problems


def main() -> None:
    args = parse_args()
    known, leaves = load_snapshot()

    total_commands = 0
    all_problems: list[tuple[str, int, list[str]]] = []

    for md in sorted(ROOT.glob("[0-9][0-9]-*.md")):
        text = md.read_text(encoding="utf-8")
        for start, block in iter_bash_blocks(text):
            if "verl.trainer.main" not in block:
                continue
            total_commands += 1
            problems = check_command(block, known, leaves)
            if problems:
                all_problems.append((md.name, start, problems))

    if all_problems:
        print(f"发现 {len(all_problems)} 处 verl 配置问题（共检查 {total_commands} 条训练命令）：\n")
        for name, line, problems in all_problems:
            print(f"  {name}:{line}")
            for p in problems:
                print(f"    - {p}")
            print()
        sys.exit(1)

    if not args.quiet:
        print(f"verl 配置检查通过（{total_commands} 条训练命令，{len(known)} 个已知 key）")


if __name__ == "__main__":
    main()
