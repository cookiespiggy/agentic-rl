"""检查根目录教程章节里的相对链接是否指向真实存在的文件。

教程与工程「咬合」最大的风险是链接腐烂：一旦调整工程目录结构，章节里的
「对应工程锚点」链接就会静默失效，读者点过去是 404，而没人会主动发现。
这个脚本把它变成 CI 能拦住的事。

用法：
    python tools/check_chapter_links.py          # 检查全部根目录 md
    python tools/check_chapter_links.py 26 27 28 # 只检查指定章节
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 同时覆盖 ./ 与 ../ 两种相对路径；图片链接 `![alt](path)` 也会被匹配到
LINK_PATTERN = re.compile(r"\]\((\.{1,2}/[^)\s]+)\)")
# 这些文件是历史交接文档，允许存在指向旧路径的链接
SKIP_FILES = {"handoff.md"}


def main() -> int:
    prefixes = sys.argv[1:]
    targets = sorted(
        path
        for path in ROOT.glob("*.md")
        if path.name not in SKIP_FILES
        and (not prefixes or path.name.split("-", 1)[0] in prefixes)
    )

    checked = 0
    broken: list[str] = []
    for md in targets:
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for raw in LINK_PATTERN.findall(line):
                target = raw.split("#", 1)[0]
                if not target:
                    continue
                checked += 1
                resolved = (md.parent / target).resolve()
                if not resolved.exists():
                    broken.append(f"{md.name}:{lineno}  ->  {target}")

    print(f"扫描 {len(targets)} 个文件，检查 {checked} 个相对链接")
    if broken:
        print(f"\n失效链接 {len(broken)} 个：")
        for item in broken:
            print(f"  {item}")
        return 1
    print("全部有效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
