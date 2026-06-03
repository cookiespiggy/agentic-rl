# 16 - 网页导航 Agent 训练

> **学习目标**：学会训练一个能在真实网页上完成任务的 Web Navigation Agent，掌握从环境搭建到 RL 训练的完整流程。

---

## 16.1 网页导航 Agent 概述

**目标**：训练 Agent 在真实网页上自主完成任务（如填表、购物、信息查询等）。

```
输入：用户的网页操作任务
  例: "在 Amazon 上搜索 Python 入门书籍，将第一本加入购物车"

Agent 动作空间（Action Space）:
- click(element_id): 点击元素
- type(element_id, text): 输入文本
- scroll(direction): 滚动页面
- navigate(url): 导航到 URL
- go_back(): 返回上一页
- finish(result): 完成并返回结果

输出：完成/失败 + 操作轨迹（Trajectory）
```

---

## 16.2 环境搭建

### 16.2.1 BrowserGym 安装

```bash
# 安装 BrowserGym
pip install browsergym

# 安装浏览器驱动
playwright install chromium

# 验证安装
python -c "
import gymnasium as gym
import browsergym
env = gym.make('browsergym/miniwow/enter_text', task_kwargs={'task': 'Enter hello'})
obs, info = env.reset()
print(f'Observation keys: {obs.keys()}')
print('BrowserGym 安装成功!')
"
```

### 16.2.2 WebNavEnvironment 封装

```python
"""网页导航 Agent 的 RL 训练环境封装"""
import gymnasium as gym
import browsergym
import re
from typing import Optional


def format_web_obs(obs: dict) -> str:
    """将网页观测格式化为 LLM 可理解的文本"""
    parts = []

    if "url" in obs:
        parts.append(f"[Current URL]: {obs['url']}")
    if "title" in obs:
        parts.append(f"[Page Title]: {obs['title']}")

    if "text_content" in obs:
        text = obs["text_content"][:2000]
        parts.append(f"[Page Content]:\n{text}")

    if "interactive_elements" in obs:
        parts.append("\n[Interactive Elements]:")
        for elem in obs["interactive_elements"][:30]:
            elem_str = f"  - id={elem.get('id', 'N/A')}, " \
                       f"tag={elem.get('tag', 'N/A')}, " \
                       f"text=\"{elem.get('text', '')[:50]}\""
            parts.append(elem_str)

    if "task" in obs:
        parts.append(f"\n[Task]: {obs['task']}")

    return "\n".join(parts)


class WebNavEnvironment:
    """Web Navigation RL 环境
    
    将 BrowserGym 封装为标准的 RL 环境接口，
    支持 verl 框架的多轮 Agent 训练。
    """
    
    def __init__(self, env_name: str = "browsergym/miniwow/enter_text",
                 max_steps: int = 15, task_kwargs: Optional[dict] = None):
        """
        Args:
            env_name: BrowserGym 环境名称
            max_steps: 每个 episode 最大交互步数
            task_kwargs: 传递给环境的任务参数
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.task_kwargs = task_kwargs or {}
        self.env = None
        self.current_step = 0
        self.trajectory = []  # 记录完整轨迹
    
    def reset(self) -> dict:
        """重置环境，返回初始观测"""
        if self.env is not None:
            self.env.close()
        
        self.env = gym.make(self.env_name, task_kwargs=self.task_kwargs)
        obs, info = self.env.reset()
        self.current_step = 0
        self.trajectory = {"steps": [], "max_steps": self.max_steps}
        
        return {
            "observation": self._format_obs(obs),
            "info": info,
        }
    
    def step(self, action_str: str) -> tuple:
        """执行动作
        
        Args:
            action_str: Agent 生成的动作字符串，如 "click('a12')"
        
        Returns:
            (observation, reward, terminated, info)
        """
        self.current_step += 1
        
        # 解析 Agent 的动作
        browsergym_action = self._parse_action(action_str)
        
        # 执行动作
        try:
            obs, env_reward, terminated, truncated, info = self.env.step(browsergym_action)
        except Exception as e:
            # 动作执行失败
            self.trajectory["steps"].append({
                "step": self.current_step,
                "action": action_str,
                "result": f"error: {str(e)}",
                "reward": 0,
            })
            return self._format_obs({}), -0.1, True, {"error": str(e)}
        
        # 记录轨迹
        self.trajectory["steps"].append({
            "step": self.current_step,
            "action": action_str,
            "observation": self._format_obs(obs)[:500],
            "reward": env_reward,
            "result": "success",
        })
        
        # 超过最大步数强制结束
        if self.current_step >= self.max_steps:
            truncated = True
        
        terminated = terminated or truncated
        
        return (
            self._format_obs(obs),
            env_reward,
            terminated,
            info,
        )
    
    def _format_obs(self, obs: dict) -> str:
        return format_web_obs(obs)
    
    def _parse_action(self, action_str: str) -> str:
        """将 Agent 的动作字符串解析为 BrowserGym 动作格式
        
        支持的格式:
          click('a12')        → 点击元素 a12
          type('a5', 'hello') → 在元素 a5 输入 hello
          scroll(down)        → 向下滚动
          goto('http://...')  → 导航到 URL
          go_back()           → 返回上一页
          finish('done')      → 完成任务
        """
        action_str = action_str.strip()
        
        # 用正则解析动作，支持多种引号格式
        patterns = [
            (r"^(click|type|scroll|goto|go_back|fill|finish|noop)\(.*?\)$", lambda m: m.group(0)),
        ]
        for pattern, handler in patterns:
            match = re.match(pattern, action_str, re.DOTALL)
            if match:
                return handler(match)
        
        # 尝试从文本中提取动作调用
        match = re.search(r"(click|type|scroll|goto|go_back|fill|finish|noop)\(.*?\)", action_str)
        if match:
            return match.group(0)
        
        # 默认返回空操作
        return 'noop()'
    
    def close(self):
        """关闭环境"""
        if self.env is not None:
            self.env.close()
```

---

## 16.3 动作空间定义

Agent 需要知道所有可用的网页操作。以下是嵌入 System Prompt 的动作提示模板：

```python
# Web Agent 的 System Prompt（动作提示模板）
WEB_AGENT_SYSTEM_PROMPT = """你是一个网页导航助手（Web Navigation Agent）。
你的任务是在网页上完成用户指定的操作。

## 可用动作

在每一步，你必须从以下动作中选择一个执行：

1. click(element_id) - 点击指定元素
2. type(element_id, text) - 在指定输入框中输入文本
3. scroll(down) 或 scroll(up) - 向上或向下滚动页面
4. goto(url) - 导航到指定的 URL
5. go_back() - 返回浏览器上一页
6. finish(result) - 任务完成，返回结果

## 输出格式

每一步请按以下格式输出：

<think>
分析当前页面状态，决定下一步动作
</think>

动作: <action>

## 注意事项
- 仔细观察页面中的可交互元素（按钮、链接、输入框等）
- 使用元素的 id（如 a12, b5）来指定操作目标
- 如果页面加载缓慢，请耐心等待
- 如果某操作失败，尝试其他方法完成任务
- 完成任务后务必调用 finish(result)
"""


def format_web_observation_for_prompt(observation: str, available_actions: list = None) -> str:
    """将观测格式化为 Agent 的对话消息"""
    msg = f"当前页面状态：\n{observation}"
    if available_actions:
        msg += "\n\n可选动作:\n"
        for i, action in enumerate(available_actions, 1):
            msg += f"  {i}. {action}\n"
    return msg
```

---

## 16.4 奖励设计

```python
"""Web Navigation Agent 奖励函数"""
import re


def web_agent_reward(completions: list, ground_truths: list,
                     trajectories: list = None, **kwargs) -> list:
    """网页导航 Agent 的多组件奖励函数
    
    奖励组成：
    - 任务完成度（Outcome Reward）: 成功 +1.0，失败 0.0
    - 步骤效率奖励: 步骤越少完成越好
    - 无效操作惩罚: 每次无效操作扣分
    - 格式奖励: 正确使用 think 标签和动作格式
    
    Args:
        completions: Agent 生成的回复列表
        ground_truths: 期望结果列表（可以是任务配置）
        trajectories: 完整交互轨迹列表
    
    Returns:
        每个回复对应的奖励值
    """
    rewards = []
    
    for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
        reward = 0.0
        traj = trajectories[i] if trajectories else None
        
        # === Layer 3: 任务结果（Outcome Reward）===
        task_success = check_task_completion(completion, gt, traj)
        if task_success:
            reward += 1.0
        
        # === Layer 2: 效率与进度奖励 ===
        if traj:
            steps_list = traj.get("steps", [])
            num_steps = len(steps_list)
            max_steps = traj.get("max_steps", 15)
            
            # 步骤效率：用更少步骤完成获得更高奖励
            if task_success:
                efficiency = max(0, 1 - num_steps / max_steps)
                reward += 0.1 * efficiency
            
            # 无效操作惩罚
            invalid_count = count_invalid_actions(traj)
            reward -= 0.05 * invalid_count
        
        # === Layer 1: 行为奖励 ===
        # 思考过程奖励
        if "<think>" in completion and "</think>" in completion:
            reward += 0.05
        
        # 动作格式正确性
        if has_valid_action_format(completion):
            reward += 0.02
        
        # 惩罚：动作格式完全错误
        if not has_valid_action_format(completion) and not task_success:
            reward -= 0.1
        
        # 限制奖励范围
        reward = max(reward, -0.5)
        rewards.append(reward)
    
    return rewards


def check_task_completion(completion: str, ground_truth: str,
                          trajectory: dict = None) -> bool:
    """检查任务是否完成"""
    # 方式 1: Agent 主动调用 finish()
    if "finish(" in completion:
        # 提取 finish 中的结果
        match = re.search(r"finish\(['\"]?(.+?)['\"]?\)", completion)
        if match:
            result = match.group(1).strip().lower()
            gt_lower = str(ground_truth).strip().lower()
            # 精确匹配或包含匹配
            if result == gt_lower or gt_lower in result:
                return True
    
    # 方式 2: 环境返回的奖励信号
    if trajectory:
        env_rewards = [s.get("reward", 0) for s in trajectory.get("steps", [])]
        if env_rewards and max(env_rewards) >= 1.0:
            return True
    
    return False


def count_invalid_actions(trajectory: dict) -> int:
    """统计无效操作次数"""
    invalid = 0
    for step in trajectory.get("steps", []):
        result = step.get("result", "")
        if "error" in result.lower() or "invalid" in result.lower():
            invalid += 1
    return invalid


def has_valid_action_format(completion: str) -> bool:
    """检查是否包含合法的动作格式"""
    valid_patterns = [
        r"click\(['\"]?[a-zA-Z0-9_]+['\"]?\)",
        r"type\(['\"]?[a-zA-Z0-9_]+['\"]?,\s*['\"].+?['\"]\)",
        r"scroll\((down|up)\)",
        r"goto\(['\"]https?://.+?['\"]\)",
        r"go_back\(\)",
        r"finish\(.+?\)",
    ]
    for pattern in valid_patterns:
        if re.search(pattern, completion):
            return True
    return False
```

---

## 16.5 SFT 数据与训练配置

### 16.5.1 SFT 训练数据示例

```python
# web_agent_sft_data.jsonl 中的样本
web_sft_sample = {
    "messages": [
        {"role": "system", "content": WEB_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": "Search for 'Python programming' on the search box and click the first result."},
        {
            "role": "assistant",
            "content": """<think>
用户要求我搜索 "Python programming" 并点击第一个结果。
当前页面有一个搜索框 (id=a3) 和一个搜索按钮 (id=a4)。
我应该先在搜索框中输入文本，然后点击搜索按钮。
</think>

动作: type('a3', 'Python programming')"""
        },
        {"role": "tool", "content": "Action executed. Page updated.\n\n[Interactive Elements]:\n  - id=a1, tag=a, text=\"Home\"\n  - id=a3, tag=input, text=\"Python programming\"\n  - id=a4, tag=button, text=\"Search\"\n  - id=a10, tag=a, text=\"Python.org - Welcome to Python\"\n  - id=a11, tag=a, text=\"Python Tutorial - W3Schools\""},
        {
            "role": "assistant",
            "content": """<think>
搜索已完成，页面显示了搜索结果。
第一个结果是 "Python.org - Welcome to Python" (id=a10)。
我需要点击它。
</think>

动作: click('a10')"""
        },
        {"role": "tool", "content": "Action executed. Navigated to https://python.org"},
        {
            "role": "assistant",
            "content": """<think>
已成功导航到 Python 官方网站。任务完成。
</think>

动作: finish('Successfully searched for Python programming and clicked the first result (Python.org)')"""
        }
    ],
    "data_source": "miniwow",
    "agent_name": "web_nav_agent"
}
```

### 16.5.2 verl 训练脚本

```bash
#!/bin/bash
# run_web_agent.sh —— 用 GRPO 训练 Web Navigation Agent

set -e

MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN_DATA="data/web_agent_train.jsonl"
TEST_DATA="data/web_agent_test.jsonl"
OUTPUT_DIR="checkpoints/web_agent_v1"
PROJECT_NAME="web_agent_grpo"

python -m verl.trainer.main \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.max_new_tokens=512 \
    actor_rollout_ref.rollout.mode=async \
    \
    algorithm=grpo \
    algorithm.kl_coeff=0.01 \
    \
    trainer.total_epochs=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=qwen2_5_3b_miniwow \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    tools.browser.enabled=True \
    tools.browser.env_name=browsergym/miniwow/enter_text \
    tools.browser.max_steps=15 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "Web Agent 训练完成！模型保存在 ${OUTPUT_DIR}"
```

---

## 16.6 从 MiniWoB 到 WebArena

| 环境 | 复杂度 | 说明 | 建议 |
|------|--------|------|------|
| MiniWoB | 简单 | 简化的网页任务（表单填写等） | **入门首选** |
| WebArena | 中等 | 真实网站模拟（电商、论坛等） | 验证后迁移 |
| VisualWebArena | 较难 | 多模态网页任务（需要理解图片） | 进阶挑战 |
| WorkArena | 企业级 | 企业 Web 应用操作 | 产业应用 |

**课程学习（Curriculum Learning）策略**：

```
Phase 1: MiniWoB 简单任务（表单填写、按钮点击）
  → 验证基础动作能力
  
Phase 2: MiniWoB 复杂任务（多步骤表单、组合操作）
  → 验证多步规划能力
  
Phase 3: WebArena 真实场景（电商购物、论坛发帖）
  → 验证真实环境泛化能力
```

建议：先在 MiniWoB 上验证训练流程，确认 reward 正常上升后，再迁移到更复杂的环境。

---

## 16.7 评估脚本

```python
"""Web Navigation Agent 评估脚本"""
import gymnasium as gym
import browsergym
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def evaluate_web_agent(model_path: str, env_name: str,
                       num_episodes: int = 50, max_steps: int = 15):
    """在指定 BrowserGym 环境上评估 Web Agent
    
    Args:
        model_path: 训练后的模型路径
        env_name: BrowserGym 环境名
        num_episodes: 评估 episode 数量
        max_steps: 每个 episode 最大步数
    
    Returns:
        评估指标字典
    """
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # 创建环境
    env = gym.make(env_name)
    
    metrics = {"success": 0, "total_steps": 0, "episodes": 0}
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        conversation = [
            {"role": "system", "content": WEB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": info.get("task", "Complete the task")},
        ]
        
        episode_steps = 0
        success = False
        
        for step in range(max_steps):
            # 格式化观测并添加到对话
            obs_text = format_web_obs(obs)
            conversation.append({"role": "tool", "content": obs_text})
            
            # 模型生成动作
            inputs = tokenizer.apply_chat_template(
                conversation, tokenize=True, return_tensors="pt", add_generation_prompt=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=256, temperature=0.3)
            
            action_text = tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True
            )
            
            # 解析动作并执行
            action = extract_action(action_text)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            
            # 更新对话
            conversation.append({"role": "assistant", "content": action_text})
            
            if terminated or truncated:
                if reward >= 1.0:
                    success = True
                break
        
        metrics["success"] += int(success)
        metrics["total_steps"] += episode_steps
        metrics["episodes"] += 1
        
        if (ep + 1) % 10 == 0:
            sr = metrics["success"] / metrics["episodes"]
            avg_steps = metrics["total_steps"] / metrics["episodes"]
            print(f"  Episode {ep+1}/{num_episodes} | "
                  f"Success Rate: {sr:.2%} | Avg Steps: {avg_steps:.1f}")
    
    env.close()
    
    return {
        "success_rate": metrics["success"] / metrics["episodes"],
        "avg_steps": metrics["total_steps"] / metrics["episodes"],
        "total_episodes": metrics["episodes"],
    }


def extract_action(text: str) -> str:
    """从模型输出中提取动作"""
    import re
    # 尝试提取 "动作: xxx" 格式
    match = re.search(r'动作[:：]\s*(.+)', text)
    if match:
        return match.group(1).strip()
    # 尝试直接提取函数调用
    match = re.search(r'(click|type|scroll|goto|go_back|finish|noop)\(.*?\)', text)
    if match:
        return match.group(0)
    return "noop()"
```

---

## 16.8 术语速查

| 英文术语 | 中文注释 |
|---------|----------|
| Web Navigation | 网页导航 |
| BrowserGym | 浏览器 Gym 环境 |
| MiniWoB | 简化网页任务环境 |
| WebArena | 网页竞技场 |
| Action Space | 动作空间 |
| Interactive Elements | 可交互元素 |
| Curriculum Learning | 课程学习（由易到难的训练策略） |
| Trajectory | 轨迹（Agent 与环境完整交互记录） |

---

## 16.9 小结

- 网页导航 Agent 需要浏览器环境（BrowserGym）和标准化的动作空间
- 奖励设计需要兼顾任务完成度、步骤效率和行为规范性
- 从 MiniWoB 入门，通过课程学习逐步迁移到 WebArena 等复杂环境
- 下一步：代码 Agent → [17 - 代码Agent训练](./17-代码Agent训练.md)
