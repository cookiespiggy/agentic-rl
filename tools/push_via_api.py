#!/usr/bin/env python3
"""通过 GitHub Git Data API 把本地领先的提交推送到远端（不使用 git push）。

流程（对应 Git Data API 的标准四步）：

    1. GET  /git/ref/heads/main            取远端当前 SHA，校验它 == 本地 HEAD^
    2. POST /git/blobs                     为每个变更文件创建 blob
    3. POST /git/trees   base_tree=远端树   基于远端树叠加本次变更
    4. POST /git/commits parents=[远端SHA]  创建提交
    5. PATCH /git/refs/heads/main          更新分支引用（这一步才是「推送」）

token 从环境变量 GH_TOKEN 读取，**绝不打印**。

`--dry-run` 会完成校验并上传 blob，但**不建树、不提交、不更新引用**——未挂到引用上的
blob 是悬空对象，GitHub 会自行回收，因此这一步是安全的。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "cookiespiggy"
REPO = "agentic-rl"
BRANCH = "main"
API = "https://api.github.com"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="上传 blob 但不建树/不提交/不更新引用（悬空 blob 会被回收，安全）",
    )
    p.add_argument(
        "--range",
        dest="range_spec",
        help="推送一个提交区间，例如 26f8dcb..HEAD。按顺序逐个提交，"
             "保留每个提交原本的作者/时间。区间基点要与远端内容一致。",
    )
    return p.parse_args()


def token() -> str:
    value = os.environ.get("GH_TOKEN", "").strip()
    if not value:
        sys.exit("GH_TOKEN 为空。请先：export GH_TOKEN=$(/opt/homebrew/bin/gh auth token)")
    return value


def api(tok: str, path: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        sys.exit(f"API 失败 {method} {path} -> HTTP {exc.code}\n{detail}")


def git(*args: str) -> str:
    """只读的本地 git 内省（rev-parse / ls-tree / diff / log），不做任何推送。

    固定 `core.quotepath=false`：否则含中文的路径会被 C 风格转义成
    `"24-\\345\\255\\246..."`，按该字符串读文件会 FileNotFoundError。
    """
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def git_bytes(*args: str) -> bytes:
    """只读的本地 git 内省，返回**原始字节**（PNG 之类的二进制文件不能走文本通道）。"""
    return subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        capture_output=True,
        check=True,
    ).stdout


def commit_meta(sha: str) -> tuple[str, str, str, str, str, str]:
    """取某个提交的作者/提交者与时间。"""
    raw = git("log", "-1", "--format=%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI", sha)
    return tuple(raw.split("\x1f"))  # type: ignore[return-value]


def tree_modes(sha: str) -> dict[str, str]:
    """某提交里每个文件的 git mode。"""
    modes: dict[str, str] = {}
    for line in git("ls-tree", "-r", sha).splitlines():
        meta, path = line.split("\t", 1)
        modes[path] = meta.split()[0]
    return modes


def push_range(tok: str, root: Path, args) -> None:
    """按顺序推送一个提交区间（base..HEAD），逐个保留原本的作者与时间。

    为什么需要这个模式：原脚本一次只推一个提交（HEAD 相对 HEAD^），
    而本地往往领先远端多个提交——要保留「一批一批改、方便回滚」的历史，
    就得按顺序把每个提交都建出来，而不是把最终状态压成一个提交。
    """
    base_spec, _, head_spec = args.range_spec.partition("..")
    base = git("rev-parse", base_spec)
    head = git("rev-parse", head_spec)

    commits = git("rev-list", "--reverse", f"{base}..{head}").splitlines()
    commits = [c for c in commits if c]
    if not commits:
        print("区间里没有提交，无需推送")
        return

    ref = api(tok, f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    remote_sha = ref["object"]["sha"]
    remote_commit = api(tok, f"/repos/{OWNER}/{REPO}/git/commits/{remote_sha}")

    print(f"远端 {BRANCH:8s} {remote_sha[:12]}  ({(remote_commit['message'].splitlines() or [''])[0][:56]})")
    print(f"本地区间      {base[:12]}..{head[:12]}  共 {len(commits)} 个提交\n")

    # 区间基点必须与远端「同内容」：tree 相同即内容相同。
    base_tree = git("rev-parse", f"{base}^{{tree}}")
    if base_tree != remote_commit["tree"]["sha"]:
        sys.exit(
            f"区间基点 {base[:12]} 的 tree ({base_tree[:12]}) 与远端 tree "
            f"({remote_commit['tree']['sha'][:12]}) 不一致，拒绝推送。\n"
            "  说明区间起点选错了，或远端有本地没有的内容。"
        )
    print(f"  ✓ 基点内容与远端一致（tree {base_tree[:12]}）\n")

    # blob 缓存：同一份内容只上传一次（git blob SHA 是内容的哈希，可直接做 key）
    blob_cache: dict[str, str] = {}
    prev_sha = remote_sha
    prev_tree = remote_commit["tree"]["sha"]
    mapping: list[tuple[str, str]] = []

    for idx, sha in enumerate(commits, 1):
        parent = git("rev-parse", f"{sha}^")
        subject = git("log", "-1", "--format=%s", sha)
        print(f"[{idx}/{len(commits)}] {sha[:12]}  {subject[:62]}")

        # 用 -z（NUL 分隔）拿路径：含中文的路径在默认输出里会被 git 加引号并转义，
        # 按那个字符串读文件必然 FileNotFoundError。
        raw = git("diff", "--name-status", "-z", parent, sha)
        fields = [f for f in raw.split("\0") if f]
        changes: list[tuple[str, str]] = []
        i = 0
        while i < len(fields):
            status = fields[i]
            if status.startswith("R") or status.startswith("C"):
                # 重命名/复制：两个路径，第二个是新路径
                changes.append(("M", fields[i + 2]))
                i += 3
                continue
            changes.append((status, fields[i + 1]))
            i += 2
        modes = tree_modes(sha)

        entries: list[dict] = []
        for status, path in changes:
            if status == "D":
                entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
                continue
            content = git_bytes("cat-file", "blob", f"{sha}:{path}")
            # 用 git blob 的哈希（内容寻址）做缓存 key，同一份内容只上传一次
            cache_key = hashlib.sha1(
                b"blob %d\0" % len(content) + content
            ).hexdigest()
            blob_sha = blob_cache.get(cache_key)
            if blob_sha is None:
                blob = api(
                    tok,
                    f"/repos/{OWNER}/{REPO}/git/blobs",
                    "POST",
                    {"content": base64.b64encode(content).decode(), "encoding": "base64"},
                )
                blob_sha = blob["sha"]
                blob_cache[cache_key] = blob_sha
            entries.append(
                {
                    "path": path,
                    "mode": modes.get(path, "100644"),
                    "type": "blob",
                    "sha": blob_sha,
                }
            )

        tree = api(
            tok,
            f"/repos/{OWNER}/{REPO}/git/trees",
            "POST",
            {"base_tree": prev_tree, "tree": entries},
        )
        # 安全断言：远端建出来的树必须和本地该提交的树**逐字节一致**。
        # 不一致就说明变更集算错了（比如误用了工作区内容而不是该提交的内容），
        # 这时候必须停——推上去会得到一段内容错乱的历史。
        local_tree_of_commit = git("rev-parse", f"{sha}^{{tree}}")
        if tree["sha"] != local_tree_of_commit:
            sys.exit(
                f"\n远端 tree {tree['sha'][:12]} != 本地 tree {local_tree_of_commit[:12]}"
                f"（提交 {sha[:12]}）——变更集算错了，中止推送。\n"
                "  常见原因：读的是工作区内容而不是该提交的内容。"
            )

        an, ae, ad, cn, ce, cd = commit_meta(sha)
        commit = api(
            tok,
            f"/repos/{OWNER}/{REPO}/git/commits",
            "POST",
            {
                "message": git("log", "-1", "--format=%B", sha),
                "tree": tree["sha"],
                "parents": [prev_sha],
                "author": {"name": an, "email": ae, "date": ad},
                "committer": {"name": cn, "email": ce, "date": cd},
            },
        )
        prev_sha = commit["sha"]
        prev_tree = tree["sha"]
        mapping.append((sha, commit["sha"]))
        same = "✓ SHA 一致" if commit["sha"] == sha else "△ SHA 不同（内容相同）"
        print(f"    tree {tree['sha'][:12]}  commit {commit['sha'][:12]}  {same}")

    if args.dry_run:
        print("\n[dry-run] 已上传 blob 并建树/建提交（未更新引用）。")
        return

    api(
        tok,
        f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        "PATCH",
        {"sha": prev_sha, "force": True},
    )
    print(f"\n  ✓ 已更新 refs/heads/{BRANCH} -> {prev_sha[:12]}")

    diffs = [(a, b) for a, b in mapping if a != b]
    if not diffs:
        print(f"\n推送完成，远端 SHA 与本地完全一致：{prev_sha}")
    else:
        print(f"\n推送完成：{prev_sha[:12]} -> {OWNER}/{REPO}@{BRANCH}")
        print(f"  ⚠️ {len(diffs)}/{len(mapping)} 个提交的 SHA 与本地不同（内容相同）。")
        print("     原因：GitHub 把提交时间规范成 UTC，本地存的是 +08:00 偏移——")
        print("     同一时刻、不同字符串，SHA 必然不同。")
        print("     本地需要同步引用，否则后续 git push 会被拒：")


def main() -> None:
    args = parse_args()
    tok = token()
    root = Path(git("rev-parse", "--show-toplevel"))
    os.chdir(root)

    if args.range_spec:
        push_range(tok, root, args)
        return

    local_sha = git("rev-parse", "HEAD")
    parent_sha = git("rev-parse", "HEAD^")

    print(f"本地 HEAD      {local_sha[:12]}")
    print(f"本地 HEAD^     {parent_sha[:12]}")

    ref = api(tok, f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    remote_sha = ref["object"]["sha"]
    print(f"远端 {BRANCH:8s} {remote_sha[:12]}")

    local_tree = git("rev-parse", "HEAD^{tree}")
    if remote_sha == parent_sha:
        print("  ✓ 远端 HEAD == 本地 HEAD^，可以快进推送\n")
    else:
        # 远端可能已经有「同内容但不同 SHA」的提交（例如上一次用 API 推送时没带
        # 原始作者/时间）。这种情况下允许重建提交以对齐 SHA。
        remote_commit = api(tok, f"/repos/{OWNER}/{REPO}/git/commits/{remote_sha}")
        same_tree = remote_commit["tree"]["sha"] == local_tree
        same_parent = remote_commit["parents"][0]["sha"] == parent_sha
        if same_tree and same_parent:
            print(
                "  ℹ 远端已有同内容提交（tree 与 parent 一致，仅 SHA 不同）——"
                "将重建提交以对齐 SHA\n"
            )
        else:
            sys.exit(
                "远端 HEAD 既不是本地 HEAD^，也不是「同内容不同 SHA」的提交，拒绝推送。\n"
                "  说明远端有本地没有的内容，需要先同步（本脚本不做 merge/rebase）。"
            )

    # 变更文件清单（本地提交相对其父提交）
    raw = git("diff", "--name-status", f"{parent_sha}", local_sha)
    changes = [line.split("\t", 1) for line in raw.splitlines() if line.strip()]
    print(f"本次变更 {len(changes)} 个文件")

    # 每个文件的 git mode（100644 / 100755 / 120000）
    modes: dict[str, str] = {}
    for line in git("ls-tree", "-r", local_sha).splitlines():
        meta, path = line.split("\t", 1)
        modes[path] = meta.split()[0]

    entries: list[dict] = []
    total_bytes = 0
    for status, path in changes:
        file_path = root / path
        if status == "D":
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        content = file_path.read_bytes()
        total_bytes += len(content)
        blob = api(
            tok,
            f"/repos/{OWNER}/{REPO}/git/blobs",
            "POST",
            {"content": base64.b64encode(content).decode(), "encoding": "base64"},
        )
        entries.append(
            {
                "path": path,
                "mode": modes.get(path, "100644"),
                "type": "blob",
                "sha": blob["sha"],
            }
        )
        print(f"  blob {status}  {path}  ({len(content)} B)")

    print(f"\n上传 {total_bytes / 1024:.0f} KB，共 {len(entries)} 个条目")

    remote_commit = api(tok, f"/repos/{OWNER}/{REPO}/git/commits/{remote_sha}")
    base_tree = remote_commit["tree"]["sha"]

    if args.dry_run:
        print("\n[dry-run] 已创建全部 blob，但不创建 tree/commit、不更新引用。")
        print(f"[dry-run] 若继续，将基于 base_tree {base_tree[:12]} 建树并更新 {BRANCH}。")
        return

    tree = api(
        tok,
        f"/repos/{OWNER}/{REPO}/git/trees",
        "POST",
        {"base_tree": base_tree, "tree": entries},
    )
    print(f"\n  ✓ tree    {tree['sha'][:12]}")

    message = git("log", "-1", "--format=%B")

    # 原样带上本地提交的作者/提交者与时间。
    # 不带的话 GitHub 会用账号名和「当前时间」生成提交，导致 tree/parent 相同但 SHA
    # 不同——本地与远端会「同内容但分叉」，之后 git push 会被拒。
    meta = git(
        "log", "-1", "--format=%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI"
    ).split("\x1f")
    author_name, author_email, author_date, committer_name, committer_email, committer_date = meta

    commit = api(
        tok,
        f"/repos/{OWNER}/{REPO}/git/commits",
        "POST",
        {
            "message": message,
            "tree": tree["sha"],
            "parents": [remote_sha],
            "author": {
                "name": author_name,
                "email": author_email,
                "date": author_date,
            },
            "committer": {
                "name": committer_name,
                "email": committer_email,
                "date": committer_date,
            },
        },
    )
    print(f"  ✓ commit  {commit['sha'][:12]}")

    api(
        tok,
        f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        "PATCH",
        {"sha": commit["sha"], "force": True},
    )
    print(f"  ✓ 已更新 refs/heads/{BRANCH}")

    if commit["sha"] == local_sha:
        print(f"\n推送完成，SHA 与本地完全一致：{commit['sha']}")
    else:
        print(
            f"\n推送完成：{commit['sha'][:12]} -> {OWNER}/{REPO}@{BRANCH}\n"
            f"  ⚠️ SHA 与本地 {local_sha[:12]} 不同（内容相同）。"
            "本地需要同步引用，否则后续 git push 会被拒。"
        )


if __name__ == "__main__":
    main()
