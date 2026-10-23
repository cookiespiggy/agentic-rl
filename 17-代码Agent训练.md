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
import base64
import subprocess
import os


class DockerCodeSandbox:
    """基于 Docker 的安全代码执行沙盒
    
    特性：
    - 网络隔离：禁止外部网络访问
    - 内存限制：256MB
    - 超时限制：30 秒
    - 文件系统隔离：容器内独立文件系统，代码写到容器自己的 /tmp
    
    代码怎么送进容器？**不要用 `-v 宿主文件:/app/code.py`**——
    macOS 上 Docker Desktop 默认不共享 /tmp，宿主文件在容器里会变成空文件，
    报错是 `can't find '__main__' module`（很难一眼看出是挂载没生效）。
    这里改成把代码 base64 后塞进环境变量，再让容器内的 bootstrap 脚本
    解码写到容器自己的 /tmp——不依赖任何 bind mount。
    """
    
    BOOTSTRAP = (
        "import base64, os, sys\n"
        "open('/tmp/code.py', 'wb').write(base64.b64decode(os.environ['CODE_B64']))\n"
        "os.execv(sys.executable, [sys.executable, '/tmp/code.py'])\n"
    )
    
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
        env_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "-i",
                    "--network=none",           # 网络隔离
                    f"--memory={self.memory_limit}",
                    "--cpus=1",                 # 限制 CPU
                    "-e", f"CODE_B64={env_code}",
                    self.image,
                    "python", "-c", self.BOOTSTRAP,
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
        """降级方案：子进程隔离执行（含资源限制）
        
        比 Docker 弱：没有网络隔离，也没有独立的文件系统视图。
        只适合本地开发调试，**不要**在训练里长期用。
        """
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            code_file = f.name
        try:
            result = subprocess.run(
                ["python", code_file],
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
        finally:
            try:
                os.unlink(code_file)
            except OSError:
                pass
    
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

> **超参口径（全书统一）**：SFT 用 **5e-5**（数据量大时可下调到 **1e-5**）、**5 个 epoch**；GRPO 用 **1e-6**、**3 个 epoch**（见 §17.4.2）。别把两阶段的设置混用。

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

奖励函数的**调用契约**统一见 [14 章 §14.1](./14-奖励函数设计与实现.md#141-奖励函数契约先读这一节)：签名是 `reward_fn(prompts, completions, **kwargs)`，标准答案从 `kwargs["ground_truth"]` 取（数据集列名），`completions` 在对话模式下是 `list[list[dict]]`，需要先用 `_coerce_to_text` 归一化。

**沙盒从哪来？** TRL 不会传 `sandbox=`——它只传数据集的列名。沙盒是一个进程级的对象（还持有镜像、超时、内存限制），所以用**模块级单例**持有：奖励函数直接用，训练脚本负责在启动前把它建好。

```python
"""Code Agent RL 奖励函数"""
import re

from reward_utils import _coerce_to_text  # 见 14 章 §14.1.2
from sandbox import DockerCodeSandbox    # 见 §17.2.1

# === 模块级沙盒单例 ===
# TRL 只传数据集列名，不会传 sandbox=（见 14 章 §14.1.2）。
# 沙盒是进程级对象（持有镜像 / 超时 / 内存限制），所以放模块级。
SANDBOX = DockerCodeSandbox(image="python:3.11-slim", timeout=30, memory_limit="256m")


def code_agent_reward(prompts, completions, **kwargs) -> list[float]:
    """代码 Agent 的多组件奖励函数
    
    原签名是 code_agent_reward(completions, ground_truths, sandbox)，
    现统一到 TRL 契约（原签名改名，语义不变）。
    
    奖励组成：
    - 测试通过率（Test Pass Rate）: passed / total_tests
    - 全部通过额外奖励: +0.2
    - 自我修复能力: 首次失败后修复成功 +0.1
    - 代码质量: 有注释 +0.05
    - 格式奖励: 正确使用 think 标签 +0.05
    
    Args:
        prompts: prompt 列表（对话模式下是 list[list[dict]]）
        completions: Agent 生成的回复；对话模式下是 list[list[dict]]
        kwargs: 数据集列。ground_truth 是测试用例列表
    
    Returns:
        每个回复对应的奖励值
    """
    test_case_sets = kwargs["ground_truth"]
    rewards = []
    
    for completion, test_cases in zip(completions, test_case_sets):
        text = _coerce_to_text(completion)
        reward = 0.0
        
        # === 1. 测试通过率（核心信号）===
        code = extract_code(text)
        if code and SANDBOX:
            test_result = SANDBOX.run_tests(code, test_cases)
            pass_rate = test_result["pass_rate"]
            reward += pass_rate  # 0.0 ~ 1.0
            
            # 全部通过额外奖励
            if pass_rate == 1.0:
                reward += 0.2
        elif code:
            # 无沙盒时尝试简单验证
            reward += 0.1  # 至少写出了代码
        
        # === 2. 自我修复能力 ===
        if has_self_debugging(text):
            reward += 0.1
        
        # === 3. 代码质量 ===
        if code and has_comments(code):
            reward += 0.05
        
        # === 4. 格式奖励 ===
        if "<think>" in text and "</think>" in text:
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
        
        # 空集合陷阱：all([]) 是 True，
        # 一个没有 FAIL_TO_PASS 的任务会被判成「已解决」——必须显式挡掉
        if not fail_to_pass:
            return {
                "resolved": False,
                "instance_id": task["instance_id"],
                "reason": "no_fail_to_pass_tests",
                "test_results": {},
            }
        
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
        """克隆仓库并切换到目标 commit
        
        两个必须处理的坑：
        1) 浅克隆（--depth=100）里可能没有 base_commit。
           先 fetch 那个 commit 本身，失败再退回完整历史。
        2) checkout 必须带 check=True。
           否则 commit 不在浅克隆里时它会静默失败，
           补丁就打到了一棵错误的树上，评估结果完全不可信。
        """
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", "--no-checkout",
                 f"https://github.com/{task['repo']}.git", repo_dir],
                capture_output=True, timeout=300, check=True,
            )
        
        # 浅克隆里取不到 base_commit 时，直接按 commit 抓一次
        fetched = subprocess.run(
            ["git", "fetch", "--depth=1", "origin", task["base_commit"]],
            cwd=repo_dir, capture_output=True, text=True,
        )
        if fetched.returncode != 0:
            # 该 commit 不在浅历史里，退回完整克隆
            subprocess.run(
                ["git", "fetch", "--unshallow"],
                cwd=repo_dir, capture_output=True, text=True,
            )
        
        # check=True：checkout 失败必须抛出来，不能静默继续
        subprocess.run(
            ["git", "checkout", task["base_commit"]],
            cwd=repo_dir, capture_output=True, text=True, check=True,
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

代码 Agent 的工具是「在沙盒里跑代码」，同样走 verl 的 `multi_turn` 段接入：YAML 声明工具的 schema，Python 文件给实现。

```yaml
# tools/code_tool.yaml
- class_name: CodeSandboxTool
  config:
    image: python:3.11-slim
    timeout: 30
    memory_limit: 256m
  tool_schema:
    type: function
    function:
      name: run_code
      description: 在隔离沙盒中执行一段 Python 代码，返回 stdout / stderr / return_code
      parameters:
        type: object
        properties:
          code:
            type: string
            description: 要执行的 Python 代码
        required:
          - code
```

```python
# tools/code_tool.py —— multi_turn.function_tool_path 指向的文件
def run_code(code: str) -> dict:
    """在 Docker 沙盒中执行代码。

    实现见 §17.2.1 的 DockerCodeSandbox。
    """
    from sandbox import DockerCodeSandbox
    # ... 实际工程里应该复用同一个沙盒实例，而不是每次新建
    ...
```

训练脚本：

```bash
#!/bin/bash
# run_code_agent.sh —— 用 GRPO 训练 Code Agent

set -e

MODEL_PATH="Qwen/Qwen2.5-7B"  # 代码任务建议 7B 起步
TRAIN_DATA="data/code_agent_train.jsonl"
TEST_DATA="data/code_agent_test.jsonl"
OUTPUT_DIR="checkpoints/code_agent_v1"

# 说明（注释写在命令外面——bash 里续行中的 # 会把后面的参数吃掉）：
# 1) actor_rollout_ref.actor.optim.lr 必须显式写出：该键在不同 verl 版本的有效默认值
#    不同（v0.9.1 为 1e-6，更早版本为 1e-3），不要依赖默认值。GRPO 统一用 1e-6。
# 2) data.prompt_key=messages：本章数据集的对话列名是 messages。
# 3) data.train_batch_size=8：verl 的整除约束是
#    「train_batch_size × rollout.n 须被 n_gpus 整除」（verl/utils/config.py:123-124），
#    这里 8 × 4 = 32，可被 1/2/4/8 卡整除。
#    注意这与 TRL 的约束不同——TRL 要求 generation_batch_size 能被 num_generations 整除。

python -m verl.trainer.main_ppo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.prompt_key=messages \
    data.train_batch_size=8 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.mode=async \
    \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=tools/code_tool.yaml \
    actor_rollout_ref.rollout.multi_turn.function_tool_path=tools/code_tool.py \
    \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.01 \
    \
    trainer.total_epochs=3 \
    trainer.project_name=code_agent_rl \
    trainer.experiment_name=qwen2_5_7b_code \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "Code Agent 训练完成！模型保存在 ${OUTPUT_DIR}"
```

**两个容易踩的坑**：

- **入口是 `python -m verl.trainer.main_ppo`**，不是 `verl.trainer.main`——`verl/trainer/` 下没有 `main.py`。
- **verl 没有 `tools.*.enabled` 这种配置段。** 沙箱是工具，接入方式是 `actor_rollout_ref.rollout.multi_turn.*`；给沙箱单独开一个 `tools.code_sandbox` 配置块并在里面写 `enabled=True`，会被 Hydra 判为未知配置项。沙箱本身的行为（镜像、超时、内存上限）由你自己的 `DockerCodeSandbox` 决定，不在 verl 的配置里。

### 17.4.3 checkpoint 落在哪

verl 按 step 分目录存 checkpoint：

```text
checkpoints/code_agent_v1/
└── global_step_150/          # trainer.save_freq=1，每步都存
    └── actor/                # actor 权重（HF 格式，可直接 from_pretrained）
```

---

## 17.5 本章数据的来源

⚠️ **本章的训练数据必须由你自己准备。** SWE-bench 的实例要从官方仓库拉（需要联网 + 磁盘），自建编程任务集也要自己造题并跑测试——**无法离线获取**。所以本章不提供「跑一下就能生成数据」的脚本。下面给出的是训练脚本实际消费的**数据格式**（`data/code_agent_train.jsonl`）和几条示例记录。

**字段清单**（每行一个 JSON 对象）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | list[dict] | 对话历史。**这是 `data.prompt_key` 指向的列**（训练脚本里写的是 `data.prompt_key=messages`）。含 `user`（题目 + 测试用例）、`assistant`（思考 + 工具调用）、`tool`（运行结果） |
| `ground_truth` | list[dict] | 测试用例列表。奖励函数用 `kwargs["ground_truth"]` 读它（见 [14 章 §14.1](./14-奖励函数设计与实现.md#141-奖励函数契约先读这一节)），每项形如 `{"input": ..., "expected_output": ...}` |
| `data_source` | str | 数据来源标记，用于分组统计（如 `"coding_tasks"`） |
| `agent_name` | str | Agent 类型标记（如 `"code_agent"`） |

**示例记录**：

```jsonl
{"messages": [{"role": "user", "content": "写一个函数求两个列表的交集（去重后排序）。"}], "ground_truth": [{"input": "[1,2,3,4] [3,4,5,6]", "expected_output": "[3, 4]"}], "data_source": "coding_tasks", "agent_name": "code_agent"}
{"messages": [{"role": "user", "content": "写一个函数判断字符串是否为回文，忽略大小写和非字母字符。"}], "ground_truth": [{"input": "A man, a plan, a canal: Panama", "expected_output": "True"}, {"input": "hello", "expected_output": "False"}], "data_source": "coding_tasks", "agent_name": "code_agent"}
{"messages": [{"role": "user", "content": "写一个函数返回列表中第二大的元素，元素不足两个时返回 None。"}], "ground_truth": [{"input": "[3, 1, 4, 4, 5]", "expected_output": "4"}, {"input": "[1]", "expected_output": "None"}], "data_source": "coding_tasks", "agent_name": "code_agent"}
```

> 完整的 `messages` 样例（带 `assistant` / `tool` 往返）见 §17.3.1。

---

## 17.6 代表性工作

| 工作 | 机构 | 特点 |
|------|------|------|
| **SWE-agent** | Princeton | 专为代码修复设计的 Agent 框架 |
| **ReTool** | ICLR 2025 | RL 增强的代码工具使用 |
| **OpenHands** | 开源社区 | 开源的通用代码 Agent 平台 |

---

## 17.7 术语速查

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

## 17.8 小结

- 代码 Agent 需要安全的代码执行环境（Docker 隔离 > 子进程隔离）
- **模型生成的代码绝不在 trainer 进程内 `exec()`**——必须走容器隔离（[14 章 §14.2.2](./14-奖励函数设计与实现.md) 也强调这一点）
- 奖励设计核心：测试通过率 + 自我修复能力 + 代码质量；奖励函数签名统一为 `reward_fn(prompts, completions, **kwargs)`（见 [14 章 §14.1](./14-奖励函数设计与实现.md#141-奖励函数契约先读这一节)）
- 沙盒是进程级对象，用**模块级单例**持有（TRL 不会传 `sandbox=`）
- SWE-bench 是代码 Agent 的标准评测基准，代码任务建议 7B 模型起步
- 下一步：搜索 Agent → [18 - 搜索增强Agent训练](./18-搜索增强Agent训练.md)
