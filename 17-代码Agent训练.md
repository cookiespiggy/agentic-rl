# 17 - 代码 Agent 训练

> **学习目标**：训练一个能编写、运行和调试代码的 Code Agent，掌握从沙盒环境搭建到 RL 训练的完整流程。

---

## 17.1 代码 Agent 概述

**目标**：训练 Agent 自主完成编程任务——编写代码、运行测试、阅读错误、修复 Bug。

```
输入：编程任务描述 + 测试用例

Agent 动作（Actions）:
- write_code(code): 编写代码
- run_code(code): 运行代码并获取输出
- read_error(error): 阅读错误信息
- edit_code(line, new_code): 修改指定行代码
- finish(code): 提交最终代码

输出：通过所有测试用例的代码
```

---

## 17.2 代码沙盒环境

### 17.2.1 Docker 隔离执行（推荐）

```python
"""安全的代码执行沙盒 —— Docker 隔离版"""
import subprocess
import tempfile
import os


class DockerCodeSandbox:
    """基于 Docker 的安全代码执行沙盒
    
    特性：
    - 网络隔离：禁止外部网络访问
    - 内存限制：256MB
    - 超时限制：30 秒
    - 文件系统隔离：只读挂载 + 临时写入目录
    """
    
    def __init__(self, image: str = "python:3.11-slim",
                 timeout: int = 30, memory_limit: str = "256m"):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
    
    def execute(self, code: str, stdin_input: str = "") -> dict:
        """在 Docker 容器中执行 Python 代码
        
        Args:
            code: 要执行的 Python 代码
            stdin_input: 标准输入（可选）
        
        Returns:
            执行结果字典：{stdout, stderr, return_code, success}
        """
        # 将代码写入临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='/tmp'
        ) as f:
            f.write(code)
            code_file = f.name
        
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network=none",           # 网络隔离
                    f"--memory={self.memory_limit}",
                    "--cpus=1",                 # 限制 CPU
                    "-v", f"{code_file}:/app/code.py:ro",  # 只读挂载
                    self.image,
                    "python", "/app/code.py"
                ],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timeout ({self.timeout}s)",
                "return_code": -1,
                "success": False,
            }
        except FileNotFoundError:
            # Docker 未安装，降级到子进程模式
            return self._fallback_execute(code, stdin_input)
        finally:
            os.unlink(code_file)
    
    @staticmethod
    def _limit_resources():
        """子进程资源限制（内存 256MB，CPU 30 秒）"""
        import resource
        try:
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        except (resource.error, ValueError):
            pass
    
    def _fallback_execute(self, code: str, stdin_input: str = "") -> dict:
        """降级方案：子进程隔离执行（含资源限制）"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=True
        ) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python", f.name],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    preexec_fn=self._limit_resources,
                )
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": result.returncode == 0,
                }
            except subprocess.TimeoutExpired:
                return {"stdout": "", "stderr": "Timeout", "return_code": -1, "success": False}
    
    def run_tests(self, code: str, test_cases: list) -> dict:
        """运行多个测试用例
        
        Args:
            code: 待测代码
            test_cases: [{"input": "2 3", "expected_output": "5"}, ...]
        
        Returns:
            {passed: int, total: int, results: list, pass_rate: float}
        """
        results = []
        passed = 0
        
        for i, tc in enumerate(test_cases):
            result = self.execute(code, stdin_input=tc.get("input", ""))
            
            # 对比输出（去除尾部空白和换行）
            actual = result["stdout"].strip()
            expected = tc.get("expected_output", "").strip()
            is_correct = (actual == expected) and result["success"]
            
            if is_correct:
                passed += 1
            
            results.append({
                "test_id": i,
                "passed": is_correct,
                "actual_output": actual[:500],
                "expected_output": expected[:500],
                "stderr": result["stderr"][:500] if not result["success"] else "",
            })
        
        return {
            "passed": passed,
            "total": len(test_cases),
            "results": results,
            "pass_rate": passed / len(test_cases) if test_cases else 0.0,
        }
```

### 17.2.2 子进程隔离（轻量级）

```python
"""轻量级代码沙盒 —— 子进程隔离版（适合快速实验）"""
import subprocess
import tempfile


class SubprocessSandbox:
    """基于子进程的代码沙盒，无需 Docker
    
    适用于本地开发和小模型实验。
    注意：安全性不如 Docker 隔离，不要在生产环境使用。
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def execute(self, code: str, stdin_input: str = "") -> dict:
        """执行 Python 代码"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(code)
            f.flush()
            
            try:
                result = subprocess.run(
                    ["python", f.name],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": result.returncode == 0,
                }
            except subprocess.TimeoutExpired:
                return {"stdout": "", "stderr": "Timeout", "return_code": -1, "success": False}
```

---

## 17.3 代码 Agent 训练流水线

### 17.3.1 SFT 数据格式

```python
# Code Agent 的 SFT 训练数据示例
code_agent_sft_sample = {
    "messages": [
        {
            "role": "system",
            "content": """你是一个编程助手（Code Agent）。
你的任务是根据题目要求编写 Python 代码，并通过运行测试来验证正确性。

## 可用工具
- run_code(code): 运行 Python 代码，返回 stdout 和 stderr
- submit(code): 提交最终代码

## 工作流程
1. 分析题目要求
2. 编写代码
3. 运行测试验证
4. 如果有错误，阅读错误并修复
5. 所有测试通过后提交"""
        },
        {
            "role": "user",
            "content": "编写一个函数，接收两个整数列表，返回它们的交集（去重后排序）。\n\n测试用例：\n输入: [1,2,3,4] [3,4,5,6]\n输出: [3, 4]"
        },
        {
            "role": "assistant",
            "content": """<think>
题目要求：两个整数列表的交集，去重后排序。
思路：用 set 取交集，然后 sorted 排序。
先写代码，然后运行测试。
</think>

我来编写代码并运行测试。

{"name": "run_code", "arguments": {"code": "def intersection(lst1, lst2):\\n    return sorted(list(set(lst1) & set(lst2)))\\n\\n# 测试\\nprint(intersection([1,2,3,4], [3,4,5,6]))"}}"""
        },
        {"role": "tool", "content": "stdout: [3, 4]\nstderr: \nreturn_code: 0\nsuccess: true"},
        {
            "role": "assistant",
            "content": """</think>
测试通过！输出 [3, 4] 与预期一致。提交最终代码。

{"name": "submit", "arguments": {"code": "def intersection(lst1, lst2):\\n    return sorted(list(set(lst1) & set(lst2)))"}}"""
        }
    ],
    "data_source": "coding_tasks",
    "agent_name": "code_agent"
}
```

### 17.3.2 RL 阶段奖励函数

```python
"""Code Agent RL 奖励函数"""
import re


def code_agent_reward(completions: list, ground_truths: list,
                      sandbox=None, **kwargs) -> list:
    """代码 Agent 的多组件奖励函数
    
    奖励组成：
    - 测试通过率（Test Pass Rate）: passed / total_tests
    - 全部通过额外奖励: +0.2
    - 自我修复能力: 首次失败后修复成功 +0.1
    - 代码质量: 有注释 +0.05
    - 格式奖励: 正确使用 think 标签 +0.05
    
    Args:
        completions: Agent 生成的回复列表
        ground_truths: 测试用例列表
        sandbox: 代码沙盒实例（用于运行测试）
    """
    rewards = []
    
    for completion, test_cases in zip(completions, ground_truths):
        reward = 0.0
        
        # === 1. 测试通过率（核心信号）===
        code = extract_code(completion)
        if code and sandbox:
            test_result = sandbox.run_tests(code, test_cases)
            pass_rate = test_result["pass_rate"]
            reward += pass_rate  # 0.0 ~ 1.0
            
            # 全部通过额外奖励
            if pass_rate == 1.0:
                reward += 0.2
        elif code:
            # 无沙盒时尝试简单验证
            reward += 0.1  # 至少写出了代码
        
        # === 2. 自我修复能力 ===
        if has_self_debugging(completion):
            reward += 0.1
        
        # === 3. 代码质量 ===
        if code and has_comments(code):
            reward += 0.05
        
        # === 4. 格式奖励 ===
        if "<think>" in completion and "</think>" in completion:
            reward += 0.05
        
        # 惩罚：没有产出任何代码
        if not code:
            reward -= 0.3
        
        rewards.append(max(reward, -0.5))
    
    return rewards


def extract_code(text: str) -> str:
    """从 Agent 回复中提取代码"""
    import json
    
    # 方式 1: 从工具调用的 JSON 结构中提取（优先使用 json 解析）
    for pattern in [
        r'{"name":\s*"(?:submit|run_code)".*?"code":\s*".*?"\}\}',
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                code = data.get("arguments", {}).get("code", "")
                if code.strip():
                    return code
            except json.JSONDecodeError:
                pass
    
    # 方式 2: 从松散 "code" 字段提取（含转义处理）
    match = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match:
        code = match.group(1)
        code = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        if code.strip():
            return code
    
    # 方式 3: 从 Markdown 代码块中提取
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 方式 4: 从 ``` 代码块中提取
    match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""


def has_self_debugging(text: str) -> bool:
    """检查是否有自我调试行为（运行→发现错误→修复→再运行）"""
    # 检查是否有多次 run_code 调用
    run_count = text.count("run_code")
    has_error_reading = any(kw in text.lower() for kw in ["error", "traceback", "错误", "修复", "fix"])
    return run_count >= 2 and has_error_reading


def has_comments(code: str) -> bool:
    """检查代码是否包含注释"""
    return "#" in code or '"""' in code or "'''" in code
```

---

## 17.4 SWE-bench 实战

SWE-bench 是代码 Agent 的标准评测基准：
- 基于真实 GitHub Issue（2000+ 任务）
- Agent 需要：读 Issue → 理解代码库 → 编写修复 → 通过测试

### 17.4.1 SWE-bench 评估流水线

```python
"""SWE-bench 评估脚本（简化版）"""
import subprocess
import os


class SWEBenchEvaluator:
    """SWE-bench 评估器
    
    对每个任务：
    1. 克隆目标仓库
    2. 切换到指定 commit
    3. 应用模型生成的 patch
    4. 运行测试套件
    5. 记录通过/失败
    """
    
    def __init__(self, workspace_dir: str = "/tmp/swe_bench"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
    
    def evaluate_single(self, task: dict, model_patch: str) -> dict:
        """评估单个 SWE-bench 任务
        
        Args:
            task: SWE-bench 任务信息
                {
                    "instance_id": "django__django-12345",
                    "repo": "django/django",
                    "base_commit": "abc123",
                    "test_patch": "...",  # 测试补丁
                    "FAIL_TO_PASS": ["tests.test_xxx"],  # 需要从失败变通过的测试
                }
            model_patch: 模型生成的代码补丁（unified diff 格式）
        
        Returns:
            {"resolved": bool, "test_results": dict}
        """
        repo_dir = os.path.join(self.workspace_dir, task["instance_id"])
        
        # 1. 准备仓库（简化示意）
        self._setup_repo(task, repo_dir)
        
        # 2. 应用模型补丁
        patch_applied = self._apply_patch(repo_dir, model_patch)
        if not patch_applied:
            return {"resolved": False, "reason": "patch_apply_failed"}
        
        # 3. 运行 FAIL_TO_PASS 测试
        fail_to_pass = task.get("FAIL_TO_PASS", [])
        test_results = self._run_tests(repo_dir, fail_to_pass)
        
        # 4. 判断是否解决
        resolved = all(
            test_results.get(test, False) for test in fail_to_pass
        )
        
        return {
            "resolved": resolved,
            "instance_id": task["instance_id"],
            "test_results": test_results,
        }
    
    def _setup_repo(self, task: dict, repo_dir: str):
        """克隆仓库并切换到目标 commit"""
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", "--depth=100",
                 f"https://github.com/{task['repo']}.git", repo_dir],
                capture_output=True, timeout=120,
            )
        subprocess.run(
            ["git", "checkout", task["base_commit"]],
            cwd=repo_dir, capture_output=True,
        )
    
    def _apply_patch(self, repo_dir: str, patch: str) -> bool:
        """应用 unified diff 格式的补丁"""
        try:
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=patch, capture_output=True, text=True, cwd=repo_dir,
            )
            if result.returncode != 0:
                return False
            
            subprocess.run(
                ["git", "apply", "-"],
                input=patch, capture_output=True, text=True, cwd=repo_dir,
            )
            return True
        except Exception:
            return False
    
    @staticmethod
    def _detect_test_framework(repo_dir: str) -> str:
        """检测项目使用的测试框架"""
        config_files = {
            "pytest": ["pytest.ini", "pyproject.toml", "setup.cfg"],
            "unittest": ["unittest.cfg"],
        }
        for framework, files in config_files.items():
            for f in files:
                if os.path.exists(os.path.join(repo_dir, f)):
                    with open(os.path.join(repo_dir, f)) as fh:
                        if framework in fh.read().lower():
                            return framework
        # 默认使用 pytest
        return "pytest"
    
    def _run_tests(self, repo_dir: str, test_ids: list) -> dict:
        """运行指定测试并返回结果（自动检测测试框架）"""
        framework = self._detect_test_framework(repo_dir)
        results = {}
        for test_id in test_ids:
            try:
                if framework == "unittest":
                    cmd = ["python", "-m", "unittest", test_id]
                else:
                    cmd = ["python", "-m", "pytest", test_id, "-x", "--tb=short"]
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=repo_dir, timeout=120,
                )
                results[test_id] = (result.returncode == 0)
            except subprocess.TimeoutExpired:
                results[test_id] = False
        return results
```

### 17.4.2 训练脚本

```bash
#!/bin/bash
# run_code_agent.sh —— 用 GRPO 训练 Code Agent

set -e

MODEL_PATH="Qwen/Qwen2.5-7B"  # 代码任务建议 7B 起步
TRAIN_DATA="data/code_agent_train.jsonl"
TEST_DATA="data/code_agent_test.jsonl"
OUTPUT_DIR="checkpoints/code_agent_v1"

python -m verl.trainer.main \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.max_new_tokens=2048 \
    actor_rollout_ref.rollout.mode=async \
    \
    algorithm=grpo \
    algorithm.kl_coeff=0.01 \
    \
    trainer.total_epochs=5 \
    trainer.project_name=code_agent_rl \
    trainer.experiment_name=qwen2_5_7b_code \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    tools.code_sandbox.enabled=True \
    tools.code_sandbox.backend=docker \
    tools.code_sandbox.timeout=30 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "Code Agent 训练完成！模型保存在 ${OUTPUT_DIR}"
```

---

## 17.5 代表性工作

| 工作 | 机构 | 特点 |
|------|------|------|
| **SWE-agent** | Princeton | 专为代码修复设计的 Agent 框架 |
| **ReTool** | ICLR 2025 | RL 增强的代码工具使用 |
| **OpenHands** | 开源社区 | 开源的通用代码 Agent 平台 |

---

## 17.6 术语速查

| 英文术语 | 中文注释 |
|---------|----------|
| Code Sandbox | 代码沙盒（安全隔离的代码执行环境） |
| Docker Isolation | Docker 隔离（容器级安全隔离） |
| SWE-bench | 软件工程基准测试（基于真实 GitHub Issue） |
| Test Pass Rate | 测试通过率 |
| Self-Debugging | 自我调试（运行→发现错误→修复→再运行） |
| Unified Diff | 统一差异格式（补丁文件的标准格式） |
| FAIL_TO_PASS | 需从失败变通过的测试（SWE-bench 评估指标） |

---

## 17.7 小结

- 代码 Agent 需要安全的代码执行环境（Docker 隔离 > 子进程隔离）
- 奖励设计核心：测试通过率 + 自我修复能力 + 代码质量
- SWE-bench 是代码 Agent 的标准评测基准，代码任务建议 7B 模型起步
- 下一步：搜索 Agent → [18 - 搜索增强Agent训练](./18-搜索增强Agent训练.md)
