#!/usr/bin/env python3
"""检查各章 `## N.M` 小节编号：**唯一**且**连续**。

为什么两个都要查
----------------
- **唯一**：`## 25.6` 出现两次，读者按编号跳转会迷路。
- **连续**：`20.5` 之后直接是 `20.7`，说明中间那节被删了但没重排。

这两个问题曾经同时存在于本仓库（ch20 缺 20.6、ch25 重复 25.6、ch26 缺 26.6），
所以这个检查是必要的。

非数字标题（`## 目录`、`## 练习题`、章节末尾的 `## 参考答案` 之类）**不参与**编号检查。

用法：
    python tools/check_heading_numbers.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 只匹配二级标题里的 "N.M" 形式
HEADING = re.compile(r"^##\s+(\d+)\.(\d+)\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    numbers: list[tuple[int, int, int]] = []  # (major, minor, line)
    in_fence = False
    fence = re.compile(r"^\s*```")
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if fence.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(line)
        if m:
            numbers.append((int(m.group(1)), int(m.group(2)), lineno))

    if not numbers:
        return problems

    # 章节号应与文件名前缀一致
    prefix = path.name.split("-", 1)[0]
    if prefix.isdigit():
        expected_major = int(prefix)
        wrong_major = sorted({n for n, _, _ in numbers if n != expected_major})
        if wrong_major:
            problems.append(
                f"{path.name}: 小节编号的章号与文件名不符（文件是 {expected_major} 章，"
                f"正文出现 {wrong_major}）"
            )

    # 唯一性 + 连续性（按 major 分组）
    by_major: dict[int, list[tuple[int, int]]] = {}
    for major, minor, lineno in numbers:
        by_major.setdefault(major, []).append((minor, lineno))

    for major, entries in sorted(by_major.items()):
        minors = [m for m, _ in entries]
        seen: dict[int, int] = {}
        for minor, lineno in entries:
            if minor in seen:
                problems.append(
                    f"{path.name}:{lineno}: 小节编号 {major}.{minor} 重复"
                    f"（首次出现在第 {seen[minor]} 行）"
                )
            else:
                seen[minor] = lineno
        ordered = sorted(set(minors))
        expected = list(range(ordered[0], ordered[0] + len(ordered)))
        if ordered != expected:
            missing = sorted(set(expected) - set(ordered))
            problems.append(
                f"{path.name}: 小节编号不连续，{major}.{ordered[0]}~{major}.{ordered[-1]} "
                f"之间缺 {missing}"
            )

    return problems


def main() -> None:
    args = parse_args()
    files = sorted(ROOT.glob("[0-9][0-9]-*.md"))
    all_problems: list[str] = []
    for path in files:
        all_problems.extend(check_file(path))

    if all_problems:
        print(f"发现 {len(all_problems)} 处小节编号问题：\n")
        for p in all_problems:
            print(f"  {p}")
        sys.exit(1)

    if not args.quiet:
        print(f"小节编号检查通过（{len(files)} 章，唯一且连续）")


if __name__ == "__main__":
    main()
