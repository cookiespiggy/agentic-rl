"""扫描 git 跟踪的文件里是否混进了凭据。

**为什么需要它**：`.gitignore` 只能拦住「不该进仓库的文件」，拦不住「写进代码里的密钥」。
而密钥一旦进了提交历史，即使后续删掉也仍然留在历史里，只能靠轮换密钥解决。

只扫 **git 跟踪的文件**，这是关键：

- `.venv/` 里的第三方包会大量误报（ruff 二进制里有 `sk-...` 字面量、torch 的
  `RECORD` 里是 base64 哈希、scipy 的 `.so` 里有 `ghs_ge...` 这类符号名），
  而这些目录本来就不入库；
- 全历史扫描（`--history`）会遍历每个提交的每个 blob，能发现「曾经提交过又删掉」
  的密钥——只扫当前工作区是发现不了的。

误报豁免：在对应行加 `allow-secret-scan` 注释即可。

用法：
    python tools/check_no_secrets.py            # 只扫当前跟踪的文件
    python tools/check_no_secrets.py --history  # 再扫全部提交历史
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 凭据形态。刻意写窄，避免把普通文本当成密钥。
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GitHub OAuth/PAT", re.compile(r"\bgh[opsru]_[A-Za-z0-9]{36,}")),
    ("GitHub fine-grained PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}")),
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("OpenAI 风格 key", re.compile(r"\bsk-[A-Za-z0-9]{32,}")),
    ("AWS Access Key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abpsr]-[A-Za-z0-9-]{20,}")),
    ("私钥块", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
)

SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".whl", ".so", ".dylib",
    ".safetensors", ".bin", ".onnx", ".pyc",
}
ALLOW_MARKER = "allow-secret-scan"


def nfc(text: str) -> str:
    """macOS 文件名是 NFD，git 存 NFC；不归一化会找不到文件。"""
    return unicodedata.normalize("NFC", text)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def scan(text: str, label: str) -> list[str]:
    findings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        for name, pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                value = match.group()
                masked = f"{value[:6]}…{value[-4:]}" if len(value) > 12 else "…"
                findings.append(f"{label}:{lineno}  [{name}] {masked}")
    return findings


def scan_worktree() -> list[str]:
    findings: list[str] = []
    for rel in sorted(nfc(p) for p in git("ls-files").splitlines()):
        path = ROOT / rel
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue  # 二进制或读不了，跳过
        findings.extend(scan(text, rel))
    return findings


def scan_history() -> list[str]:
    findings: list[str] = []
    revs = git("rev-list", "--all").split()
    blobs: dict[str, str] = {}
    for rev in revs:
        for entry in git("ls-tree", "-r", "-z", rev).split("\0"):
            if not entry:
                continue
            meta, path = entry.split("\t", 1)
            blobs.setdefault(meta.split()[2], path)

    for sha, path in blobs.items():
        if Path(path).suffix.lower() in SKIP_SUFFIXES:
            continue
        content = subprocess.run(
            ["git", "cat-file", "-p", sha], cwd=ROOT, capture_output=True
        ).stdout
        try:
            text = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        findings.extend(scan(text, f"{path}@{sha[:8]}"))
    return findings


def main() -> int:
    check_history = "--history" in sys.argv

    findings = scan_worktree()
    scope = "git 跟踪的文件"
    if check_history:
        findings += scan_history()
        scope = "git 跟踪的文件 + 全部提交历史"

    print(f"扫描 {scope}")
    if findings:
        print(f"\n发现 {len(findings)} 处疑似凭据：")
        for item in findings:
            print(f"  ⚠ {item}")
        print("\n  处置建议：立刻轮换该凭据，再从历史里清除")
        print("  （只删文件不够——密钥仍在历史里；误报可加 `allow-secret-scan` 注释豁免）")
        return 1

    print("未发现凭据 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
