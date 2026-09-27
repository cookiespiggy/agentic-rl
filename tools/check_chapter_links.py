"""检查根目录教程章节里的相对链接是否有效。

教程与工程「咬合」最大的风险是链接腐烂：一旦调整工程目录结构，章节里的
「对应工程锚点」链接就会静默失效，读者点过去是 404，而没人会主动发现。

做两层检查：

1. **存在性**——目标文件在当前工作区里存在；
2. **入库状态**——目标**被 git 跟踪**。

第 2 条是必须的：`reports/`、`artifacts/`、`data/*.jsonl` 这些生成物被 gitignore，
本地跑完脚本后它们确实存在，**但新克隆里没有**——只查存在性会在本地通过、在 CI 失败。
（这个坑真实发生过：CI 的 `docs` job 因为 10 个指向 `reports/*.json` 的链接而失败，
本地却一直是绿的。）

git 不可用时（例如从 tarball 运行）自动跳过第 2 条，只做存在性检查。

用法：
    python tools/check_chapter_links.py          # 检查全部根目录 md
    python tools/check_chapter_links.py 26 27 28 # 只检查指定章节
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 同时覆盖 ./ 与 ../ 两种相对路径；图片链接 `![alt](path)` 也会被匹配到
LINK_PATTERN = re.compile(r"\]\((\.{1,2}/[^)\s]+)\)")
# 这些文件是历史交接文档，允许存在指向旧路径的链接
SKIP_FILES = {"handoff.md"}


def nfc(text: str) -> str:
    """统一成 NFC。

    macOS 文件名是 NFD（字母 + 组合符），而 git 存的是 NFC。不归一化会产生大量
    假阳性——实测直接把两者做集合比较，会让 119 个正常链接被误判为「未入库」。
    """
    return unicodedata.normalize("NFC", text)


def tracked_paths() -> set[str] | None:
    """git 已跟踪的相对路径集合；git 不可用时返回 None。"""
    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {nfc(p) for p in out.split("\0") if p}


def is_tracked(rel: str, resolved: Path, tracked: set[str]) -> bool:
    """目录要求「其下有任意被跟踪的文件」，文件要求「本身被跟踪」。"""
    rel = nfc(rel)
    if resolved.is_dir():
        prefix = rel.rstrip("/") + "/"
        return any(p.startswith(prefix) for p in tracked)
    return rel in tracked


def main() -> int:
    prefixes = sys.argv[1:]
    targets = sorted(
        path
        for path in ROOT.glob("*.md")
        if path.name not in SKIP_FILES
        and (not prefixes or path.name.split("-", 1)[0] in prefixes)
    )

    tracked = tracked_paths()
    checked = 0
    broken: list[str] = []
    unversioned: list[str] = []

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
                    continue
                if tracked is None:
                    continue
                try:
                    rel = str(resolved.relative_to(ROOT))
                except ValueError:
                    continue  # 指向仓库之外，不管
                if not is_tracked(rel, resolved, tracked):
                    unversioned.append(f"{md.name}:{lineno}  ->  {target}")

    print(f"扫描 {len(targets)} 个文件，检查 {checked} 个相对链接")
    if tracked is None:
        print("（git 不可用，跳过「入库状态」检查）")

    if broken:
        print(f"\n失效链接 {len(broken)} 个（目标不存在）：")
        for item in broken:
            print(f"  {item}")
    if unversioned:
        print(
            f"\n未入库链接 {len(unversioned)} 个"
            "（本地存在但未被 git 跟踪，新克隆里会 404）："
        )
        for item in unversioned:
            print(f"  {item}")
        print("  提示：生成物（reports/ artifacts/ data/*.jsonl）不该写成 markdown 链接，")
        print("        改成 `路径` 代码格式并注明「生成物，不入库」。")

    if broken or unversioned:
        return 1
    print("全部有效且均已入库")
    return 0


if __name__ == "__main__":
    sys.exit(main())
