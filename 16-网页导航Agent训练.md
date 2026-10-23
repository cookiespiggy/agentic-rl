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

# MiniWoB 任务是独立包，要单独装
pip install browsergym-miniwob

# 安装浏览器驱动
playwright install chromium

# 验证安装
python -c "
import gymnasium as gym
import browsergym
# env id 用连字符：browsergym/miniwob/enter-text（不是 enter_text）
# 任务参数叫 goal（不是 task）
env = gym.make('browsergym/miniwob/enter-text', task_kwargs={'goal': 'Enter hello'})
obs, info = env.reset()
print(f'Observation keys: {list(obs.keys())}')
print(f'Goal: {obs[\"goal_object\"]}')   # 任务目标在 obs 里，info 里没有 task 键
print('BrowserGym 安装成功!')
"
```

**BrowserGym 的三个常见记错**：

| 记错的写法 | 正确写法 |
|-----------|---------|
| `browsergym/miniwob/enter_text` | `browsergym/miniwob/enter-text`（连字符） |
| `task_kwargs={'task': ...}` | `task_kwargs={'goal': ...}` |
| `obs['text_content']` | `obs['page_content']` |
| `elem['id']` | `elem['bid']`（browser id） |
| `info['task']` | `obs['goal_object']`（`info` 里没有任务目标） |

### 16.2.2 WebNavEnvironment 封装

```python
"""网页导航 Agent 的 RL 训练环境封装"""
import gymnasium as gym
import browsergym
import re
from typing import Optional


def format_web_obs(obs: dict) -> str:
    """将网页观测格式化为 LLM 可理解的文本

    BrowserGym 的 obs 键：
    - page_content      页面文本（不是 text_content）
    - goal_object       任务目标（不在 info 里）
    - interactive_elements[i]["bid"]  元素标识（不是 id）
    """
    parts = []

    if "url" in obs:
        parts.append(f"[Current URL]: {obs['url']}")
    if "title" in obs:
        parts.append(f"[Page Title]: {obs['title']}")

    if "page_content" in obs:
        text = obs["page_content"][:2000]
        parts.append(f"[Page Content]:\n{text}")

    if "interactive_elements" in obs:
        parts.append("\n[Interactive Elements]:")
        for elem in obs["interactive_elements"][:30]:
            elem_str = f"  - bid={elem.get('bid', 'N/A')}, " \
                       f"tag={elem.get('tag', 'N/A')}, " \
                       f"text=\"{elem.get('text', '')[:50]}\""
            parts.append(elem_str)

    if "goal_object" in obs:
        parts.append(f"\n[Task]: {obs['goal_object']}")

    return "\n".join(parts)


class WebNavEnvironment:
    """Web Navigation RL 环境
    
    将 BrowserGym 封装为标准的 RL 环境接口，
    支持 verl 框架的多轮 Agent 训练。
    """
    
    def __init__(self, env_name: str = "browsergym/miniwob/enter-text",
                 max_steps: int = 15, task_kwargs: Optional[dict] = None):
        """
        Args:
            env_name: BrowserGym 环境名称（注意是连字符）
            max_steps: 每个 episode 最大交互步数
            task_kwargs: 传递给环境的任务参数（键名是 goal）
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.task_kwargs = task_kwargs or {}
        self.env = None
        self.current_step = 0
        self.trajectory = {}  # 记录完整轨迹（dict 风格，见 14 章 §14.2.3）
    
    def reset(self) -> tuple:
        """重置环境，返回 (observation, info)
        
        和 gymnasium 的 reset 约定一致。
        """
        if self.env is not None:
            self.env.close()
        
        self.env = gym.make(self.env_name, task_kwargs=self.task_kwargs)
        obs, info = self.env.reset()
        self.current_step = 0
        self.trajectory = {
            "steps": [],
            "actions": [],
            "max_steps": self.max_steps,
            "final_state": {},
        }
        
        return self._format_obs(obs), info
    
    def step(self, action_str: str) -> tuple:
        """执行动作
        
        Args:
            action_str: Agent 生成的动作字符串，如 "click('a12')"
        
        Returns:
            gymnasium 五元组 (observation, reward, terminated, truncated, info)
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
            self.trajectory["actions"].append({"error": str(e), "success": False})
            return self._format_obs({}), -0.1, True, False, {"error": str(e)}
        
        # 记录轨迹
        self.trajectory["steps"].append({
            "step": self.current_step,
            "action": action_str,
            "observation": self._format_obs(obs)[:500],
            "reward": env_reward,
            "result": "success",
        })
        self.trajectory["actions"].append({"success": True})
        self.trajectory["final_state"] = obs
        
        # 超过最大步数强制结束
        if self.current_step >= self.max_steps:
            truncated = True
        
        return (
            self._format_obs(obs),
            env_reward,
            terminated,
            truncated,
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

奖励函数的**调用契约**统一见 [14 章 §14.1](./14-奖励函数设计与实现.md#141-奖励函数契约先读这一节)：签名是 `reward_fn(prompts, completions, **kwargs)`，标准答案从 `kwargs["ground_truth"]` 取（数据集列名），`completions` 在对话模式下是 `list[list[dict]]`，需要先用 `_coerce_to_text` 归一化。

**轨迹从哪来？** TRL 不会传 `trajectories=`——它只传数据集的列名。轨迹是 rollout 时才产生的运行时状态，所以走**模块级仓库**：环境侧每跑完一个 episode 就把轨迹写进去，奖励函数按索引取。这样既不污染数据集，也不用改训练器的调用约定。

```python
"""Web Navigation Agent 奖励函数"""
import re

from reward_utils import _coerce_to_text  # 见 14 章 §14.1.2

# === 模块级轨迹仓库 ===
# TRL 只传数据集列名，不会传 trajectories=（见 14 章 §14.1.2）。
# 环境侧在 episode 结束后调 record_trajectory() 写入，
# 奖励函数再按索引取回。
TRAJECTORY_STORE: dict = {}


def record_trajectory(sample_index: int, trajectory: dict) -> None:
    """环境侧调用：登记第 sample_index 条轨迹"""
    TRAJECTORY_STORE[sample_index] = trajectory


def clear_trajectories() -> None:
    """每个 batch 开始前清空"""
    TRAJECTORY_STORE.clear()


def web_agent_reward(prompts, completions, **kwargs) -> list[float]:
    """网页导航 Agent 的多组件奖励函数
    
    原签名是 web_agent_reward(completions, ground_truths, trajectories)，
    现统一到 TRL 契约（原签名改名，语义不变）。
    
    奖励组成：
    - 任务完成度（Outcome Reward）: 成功 +1.0，失败 0.0
    - 步骤效率奖励: 步骤越少完成越好
    - 无效操作惩罚: 每次无效操作扣分
    - 格式奖励: 正确使用 think 标签和动作格式
    
    Args:
        prompts: prompt 列表（对话模式下是 list[list[dict]]）
        completions: Agent 生成的回复；对话模式下是 list[list[dict]]
        kwargs: 数据集列。ground_truth 是期望结果，
            trajectories 可选——不传就从模块级 TRAJECTORY_STORE 取
    
    Returns:
        每个回复对应的奖励值
    """
    ground_truths = kwargs["ground_truth"]
    trajectories = kwargs.get("trajectories")
    rewards = []
    
    for i, (completion, gt) in enumerate(zip(completions, ground_truths)):
        text = _coerce_to_text(completion)
        reward = 0.0
        traj = trajectories[i] if trajectories else TRAJECTORY_STORE.get(i)
        
        # === Layer 3: 任务结果（Outcome Reward）===
        task_success = check_task_completion(text, gt, traj)
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
        if "<think>" in text and "</think>" in text:
            reward += 0.05
        
        # 动作格式正确性
        if has_valid_action_format(text):
            reward += 0.02
        
        # 惩罚：动作格式完全错误
        if not has_valid_action_format(text) and not task_success:
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

> **SFT 超参（全书统一）**：学习率 **5e-5**、**5** 个 epoch。GRPO 阶段的学习率是 **1e-6**、**3** 个 epoch（见 §16.5.2）。别把两阶段的设置混用。

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

多轮 Agent 训练在 verl 里靠 `multi_turn` 段开启。工具怎么接进去？两个配置项：

- `multi_turn.tool_config_path` —— 指向一个 YAML，声明有哪些工具、每个工具的入参 schema；
- `multi_turn.function_tool_path` —— 指向一个 Python 文件，里面是工具的实现函数。

YAML 的格式是 verl 自己的（`class_name` / `config` / `tool_schema` 三段）：

```yaml
# tools/web_tool.yaml
- class_name: BrowserTool
  config:
    env_name: browsergym/miniwob/enter-text
    max_steps: 15
  tool_schema:
    type: function
    function:
      name: browser_action
      description: 在网页上执行一个动作。动作格式：click('a12') / type('a5', 'hello') / scroll(down) / goto('http://...') / go_back() / finish('结果')
      parameters:
        type: object
        properties:
          action:
            type: string
            description: 要执行的动作字符串
        required:
          - action
```

```python
# tools/web_tool.py —— multi_turn.function_tool_path 指向的文件
def browser_action(action: str) -> str:
    """执行一个网页动作，返回执行后的页面摘要。

    实现见 §16.2.2 的 WebNavEnvironment。
    """
    # ... 调 BrowserGym，执行 action，返回新的观测文本
    ...
```

训练脚本：

```bash
#!/bin/bash
# run_web_agent.sh —— 用 GRPO 训练 Web Navigation Agent

set -e

MODEL_PATH="Qwen/Qwen2.5-3B"
TRAIN_DATA="data/web_agent_train.jsonl"
TEST_DATA="data/web_agent_test.jsonl"
OUTPUT_DIR="checkpoints/web_agent_v1"
PROJECT_NAME="web_agent_grpo"

# 说明（注释写在命令外面——bash 里续行中的 # 会把后面的参数吃掉）：
# 1) actor_rollout_ref.actor.optim.lr 必须显式写出：该键在不同 verl 版本的有效默认值
#    不同（v0.9.1 为 1e-6，更早版本为 1e-3），不要依赖默认值。GRPO 统一用 1e-6。
# 2) data.prompt_key=messages：本章数据集的对话列名是 messages。
# 3) data.train_batch_size=8：verl 的整除约束是
#    「train_batch_size × rollout.n 须被 n_gpus 整除」（verl/utils/config.py:123-124），
#    这里 8 × 8 = 64，可被 1/2/4/8 卡整除。
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.mode=async \
    \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=tools/web_tool.yaml \
    actor_rollout_ref.rollout.multi_turn.function_tool_path=tools/web_tool.py \
    \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.01 \
    \
    trainer.total_epochs=3 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=qwen2_5_3b_miniwob \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "Web Agent 训练完成！模型保存在 ${OUTPUT_DIR}"
```

**两个容易踩的坑**：

- **入口是 `python -m verl.trainer.main_ppo`**，不是 `verl.trainer.main`——`verl/trainer/` 下没有 `main.py`。
- **verl 没有 `tools.*.enabled` 这种配置段。** 工具是挂在 `actor_rollout_ref.rollout.multi_turn.*` 下的；给浏览器单独开一个 `tools.browser` 配置块并在里面写 `enabled=True`，会被 Hydra 直接判为未知配置项而报错。

### 16.5.3 checkpoint 落在哪

verl 按 step 分目录存 checkpoint，不是直接放在 `default_local_dir` 根目录：

```text
checkpoints/web_agent_v1/
└── global_step_150/          # trainer.save_freq=1，每步都存
    └── actor/                # actor 权重（HF 格式，可直接 from_pretrained）
```

所以评估时加载的路径是 `checkpoints/web_agent_v1/global_step_150/actor`。

---

## 16.6 本章数据的来源

⚠️ **本章的训练数据必须由你自己准备。** MiniWoB / WebArena 的任务实例要么需要联网抓取、要么需要本地跑浏览器才能采到，**无法离线获取**，所以本章不提供「跑一下就能生成数据」的脚本。下面给出的是训练脚本实际消费的**数据格式**（`data/web_agent_train.jsonl`）和几条示例记录，照着这个格式产出自己的数据即可。

**字段清单**（每行一个 JSON 对象）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | list[dict] | 对话历史。**这是 `data.prompt_key` 指向的列**（训练脚本里写的是 `data.prompt_key=messages`）。至少包含一条 `user` 消息；有历史轨迹时还会包含 `assistant` / `tool` 消息 |
| `ground_truth` | str | 期望结果。奖励函数用 `kwargs["ground_truth"]` 读它（见 [14 章 §14.1](./14-奖励函数设计与实现.md#141-奖励函数契约先读这一节)）。对网页任务来说，通常是任务目标文本或期望的结果串 |
| `data_source` | str | 数据来源标记，用于分组统计（如 `"miniwob"`） |
| `agent_name` | str | Agent 类型标记（如 `"web_nav_agent"`） |

**示例记录**：

```jsonl
{"messages": [{"role": "user", "content": "Click the button named Submit."}], "ground_truth": "submit", "data_source": "miniwob", "agent_name": "web_nav_agent"}
{"messages": [{"role": "user", "content": "Enter 'hello' in the text box."}], "ground_truth": "hello", "data_source": "miniwob", "agent_name": "web_nav_agent"}
{"messages": [{"role": "user", "content": "Search for 'Python programming' and click the first result."}], "ground_truth": "Python.org", "data_source": "miniwob", "agent_name": "web_nav_agent"}
```

> 多轮的 `messages` 里，`assistant` 消息放 Agent 的思考 + 动作，`tool` 消息放环境返回的页面观测——格式示例见 §16.5.1。

---

## 16.7 从 MiniWoB 到 WebArena

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

## 16.8 评估脚本

**每个任务只跑一条轨迹，所有指标都从这条轨迹上算。** 不要为了算「成功率」「平均步数」「格式合规率」各生成一次——那样同一个任务会跑出三条不同的轨迹，指标之间没法互相印证，而且采样成本翻三倍。

```python
"""Web Navigation Agent 评估脚本"""
import gymnasium as gym
import browsergym
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def evaluate_web_agent(model_path: str, env_name: str,
                       num_episodes: int = 50, max_steps: int = 15,
                       task_kwargs: dict = None):
    """在指定 BrowserGym 环境上评估 Web Agent
    
    Args:
        model_path: 训练后的模型路径
            （verl 的布局是 <default_local_dir>/global_step_<N>/actor）
        env_name: BrowserGym 环境名
        num_episodes: 评估 episode 数量
        max_steps: 每个 episode 最大步数
        task_kwargs: 传给 BrowserGym 的任务参数（键名是 goal）
    
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
    env = gym.make(env_name, task_kwargs=task_kwargs or {})
    
    metrics = {"success": 0, "total_steps": 0, "total_reward": 0.0,
               "invalid_actions": 0, "episodes": 0}
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        # 任务目标在 obs["goal_object"] 里，info 里没有 task 键
        conversation = [
            {"role": "system", "content": WEB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": obs.get("goal_object", "Complete the task")},
        ]
        
        episode_steps = 0
        # 下面这些都从同一条轨迹上统计，不重新生成
        max_env_reward = 0.0
        invalid_actions = 0
        
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
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as e:
                invalid_actions += 1
                terminated = True
                reward = -0.1
                truncated = False
            
            episode_steps += 1
            max_env_reward = max(max_env_reward, reward)
            
            # 更新对话
            conversation.append({"role": "assistant", "content": action_text})
            
            if terminated or truncated:
                break
        
        # --- 三个指标复用上面这同一条轨迹 ---
        # 1) 成功：整个 episode 里环境给过的最高奖励 >= 1.0
        metrics["success"] += int(max_env_reward >= 1.0)
        # 2) 步数：这条轨迹走了多少步
        metrics["total_steps"] += episode_steps
        # 3) 奖励：这条轨迹上环境给的奖励之和
        metrics["total_reward"] += max_env_reward
        # 4) 无效操作：同一条轨迹里出错的次数
        metrics["invalid_actions"] += invalid_actions
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
        "avg_reward": metrics["total_reward"] / metrics["episodes"],
        "invalid_actions_per_episode": metrics["invalid_actions"] / metrics["episodes"],
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

## 16.9 术语速查

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

## 16.10 小结

- 网页导航 Agent 需要浏览器环境（BrowserGym）和标准化的动作空间
- BrowserGym 的几个易错点：env id 用连字符、任务参数叫 `goal`、页面文本在 `page_content`、元素标识是 `bid`、任务目标在 `obs["goal_object"]`
- 奖励设计需要兼顾任务完成度、步骤效率和行为规范性；奖励函数签名统一为 `reward_fn(prompts, completions, **kwargs)`（见 [14 章 §14.1](./14-奖励函数设计与实现.md#141-奖励函数契约先读这一节)）
- 轨迹不是 TRL 会传的参数，用**模块级仓库**在环境侧写入、奖励侧按索引取
- 评估时**一个任务只跑一条轨迹**，多个指标从这条轨迹上统计
- 从 MiniWoB 入门，通过课程学习逐步迁移到 WebArena 等复杂环境
- 下一步：代码 Agent → [17 - 代码Agent训练](./17-代码Agent训练.md)
