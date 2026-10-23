#!/usr/bin/env bash
# 全量回归验收：把计划 §8 的 22 项检查一次跑完。
#
#   bash tools/run_regression.sh
#
# 任何一项失败即退出码非零。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
FAILED_ITEMS=()

run() {
  local name="$1"; shift
  local dir="$1"; shift
  echo ""
  echo "─── $name ───"
  if (cd "$dir" && "$@") >/tmp/_reg_out 2>&1; then
    tail -3 /tmp/_reg_out
    PASS=$((PASS + 1))
  else
    cat /tmp/_reg_out
    FAIL=$((FAIL + 1))
    FAILED_ITEMS+=("$name")
  fi
}

B=minimal-decision-bench

run "1. ruff lint"              "$B" uv run ruff check src scripts tests
run "2. 未定义名（F821）"        "$B" uv run ruff check --select F821 src scripts tests
run "3. 单元测试"                "$B" uv run pytest -q tests
run "4. 教程链接"                "."  python3 tools/check_chapter_links.py
run "5. 凭据扫描"                "."  python3 tools/check_no_secrets.py --history
run "6. 图表新鲜度（内容寻址）"   "$B" uv run python scripts/09_make_figures.py --check
run "7. 差距统计后处理"           "$B" uv run python scripts/13_gap_stats.py
run "8. inference 镜像一致性"    "."  python3 tools/check_inference_mirror.py
run "9. inference 文档重渲染"     "$B" uv run python scripts/14_render_inference_docs.py --check
run "11. 小节编号唯一且连续"      "."  python3 tools/check_heading_numbers.py
run "13. verl 必填参数 + config"  "."  python3 tools/check_verl_configs.py
run "14. 章节 Python 代码块"      "."  python3 tools/check_chapter_python.py
run "22. minimal-verl 测试"       "minimal-verl" python3 -m unittest discover -s tests

echo ""
echo "─── 10. 报告已入库 ───"
N=$(git ls-files "$B/reports" | wc -l | tr -d ' ')
echo "已跟踪的报告数：$N"
if [ "$N" -ge 14 ]; then echo "✓ ≥ 14"; PASS=$((PASS + 1)); else echo "✗ 少于 14"; FAIL=$((FAIL + 1)); FAILED_ITEMS+=("10. 报告已入库"); fi

echo ""
echo "─── 12. 内部文档未入库 ───"
if git ls-files | grep -qE "handoff\.md|PROGRESS\.md"; then
  echo "✗ 内部文档被跟踪了"; FAIL=$((FAIL + 1)); FAILED_ITEMS+=("12. 内部文档未入库")
else
  echo "✓ 内部文档均未入库"; PASS=$((PASS + 1))
fi

echo ""
echo "─── 15. HTML 敏感标签未被吞 ───"
python3 - <<'PY'
import pathlib, re, sys
# 检查的是「标签被 HTML 清洗吞掉」这个**具体**故障：
# 原文写着「有 <think> 标签」，落盘后变成「有  标签」（标签没了，留下两个空格）。
# 用「介词 + 多个空格 + 标签/tag」的签名来检测，比数配对可靠——
# 因为正文里本来就有「输出里有 <think> 标签」这种不成对的用法。
PAT = re.compile(r"[有含带输][\u4e00-\u9fff]{0,2}\s{2,}(标签|tag)", re.IGNORECASE)
bad = []
for md in sorted(pathlib.Path(".").glob("[0-9][0-9]-*.md")):
    for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        if PAT.search(line):
            bad.append(f"{md.name}:{i}: 疑似标签被吞——{line.strip()[:70]}")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("✓ 没有被吞的标签")
PY
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); FAILED_ITEMS+=("15. HTML 标签成对"); fi

echo ""
echo "─── 18. ch06-10 学习结构 ───"
python3 - <<'PY'
import pathlib, sys
bad = []
for p in ["06", "07", "08", "09", "10"]:
    md = list(pathlib.Path(".").glob(f"{p}-*.md"))
    if not md:
        bad.append(f"{p}: 找不到章节"); continue
    t = md[0].read_text(encoding="utf-8")
    if t.count("\n## 目录") != 1: bad.append(f"{p}: `## 目录` 数量不对")
    if t.count("\n## 练习题") != 1: bad.append(f"{p}: `## 练习题` 数量不对")
    if "> **参考答案**：" not in t: bad.append(f"{p}: 缺行内参考答案")
    if "\n## 参考答案" in t: bad.append(f"{p}: 不该有 `## 参考答案` 标题")
    if t.startswith("---\n"): bad.append(f"{p}: 不该有 YAML frontmatter")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("✓ 06-10 结构齐")
PY
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); FAILED_ITEMS+=("18. ch06-10 学习结构"); fi

echo ""
echo "─── 20. Makefile 契约 ───"
python3 - <<'PY'
import pathlib, sys
t = pathlib.Path("minimal-decision-bench/Makefile").read_text(encoding="utf-8")
bad = []
main_line = [l for l in t.splitlines() if l.startswith("main:")]
if not main_line:
    bad.append("main 目标缺失")
else:
    line = main_line[0]
    for need in ("cascade", "confidence", "figures"):
        if need not in line: bad.append(f"main 目标缺 {need}")
    if line.index("figures") < line.index("confidence"):
        bad.append("figures 必须排在 confidence 之后")
if "--group quantize" not in t:
    bad.append("sync 目标没有装 quantize 依赖组")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("✓ Makefile 契约")
PY
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); FAILED_ITEMS+=("20. Makefile 契约"); fi

echo ""
echo "─── 21. 统计口径一致性 ───"
python3 - <<'PY'
import json, pathlib, re, sys
p = pathlib.Path("minimal-decision-bench/reports/gap_stats.json")
if not p.exists():
    print("缺少 gap_stats.json"); sys.exit(1)
d = json.loads(p.read_text(encoding="utf-8"))
bad = []
if d.get("caliber") != "cross_validation": bad.append("gap_stats 口径不对")
if d["significance_policy"]["survivors"] != []: bad.append("不应有指标在 Holm 后存活")
if d["significance_policy"]["std_ddof_for_nadeau_bengio"] != 1: bad.append("NB 校正的 std 必须是 ddof=1")
ms = pathlib.Path("minimal-decision-bench/reports/multiseed_report.json")
if ms.exists():
    m = json.loads(ms.read_text(encoding="utf-8"))
    for k, g in list(m["gaps_vs_rules"].items()) + list(m["gap_encoder_vs_qwen"].items()):
        if "descriptive_separated" not in g: bad.append(f"{k}: 缺 descriptive_separated")
        if "separated" in g: bad.append(f"{k}: 仍有过时的 separated 字段")
src = pathlib.Path("minimal-decision-bench/scripts/07_aggregate_multiseed.py").read_text(encoding="utf-8")
# 「显著性检验」是合法用法；这里要抓的是把描述性比值说成「显著」的那句文案
if re.search(r'"显著"|\'显著\'|显著\s*(if|else)', src):
    bad.append("07 脚本的文案里还有把比值说成「显著」的措辞")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("✓ 口径一致")
PY
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); FAILED_ITEMS+=("21. 统计口径一致性"); fi

echo ""
echo "════════════════════════════════════"
echo "通过 $PASS 项，失败 $FAIL 项"
if [ "$FAIL" -gt 0 ]; then
  echo "失败项："
  for f in "${FAILED_ITEMS[@]}"; do echo "  - $f"; done
  exit 1
fi
echo "全部通过 ✓"
