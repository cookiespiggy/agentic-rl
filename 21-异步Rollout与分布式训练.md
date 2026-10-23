# 21 - 异步 Rollout 与分布式训练

> **学习目标**：理解 Agentic RL 中的异步 Rollout 机制和分布式训练架构，掌握 verl 的 Server-based 架构和性能优化技巧。

---

## 21.1 为什么需要异步 Rollout

### 21.1.1 同步 Rollout 的问题

在传统同步训练（Synchronous Rollout）中:
1. GPU 生成回复 → 等待环境返回工具调用结果 → 继续生成
2. 工具调用可能需要 0.1-10 秒（API 调用、网页加载等）
3. 等待期间 GPU 完全空闲，利用率极低

### 21.1.2 异步解决方案

verl 的 Async Rollout（异步展开）:
- 推理引擎（Inference Engine）和 Agent（智能体）分离
- Agent 发起推理请求后立即去做其他事（如调用工具）
- 使用 asyncio 协程机制，多个 Rollout 并发执行

```
同步: GPU [生成A] --- [等待工具] --- [生成A] --- [等待工具]
                   ↓ 空闲浪费                ↓ 空闲浪费

异步: GPU [生成A] [生成B] [生成C] [生成D] ...
      Tool         [工具A返回] [工具B返回] [工具C返回] ...
                   ↑ GPU 持续工作，利用率大幅提升
```

### 21.1.3 性能对比

```python
"""同步 vs 异步 Rollout 性能模拟"""
import asyncio
import time


# === 同步 Rollout ===
def sync_rollout(num_rollouts: int, gen_time: float = 0.5, tool_time: float = 2.0):
    """同步 Rollout：逐个执行，等待工具时 GPU 空闲"""
    total_time = 0
    for _ in range(num_rollouts):
        total_time += gen_time    # GPU 生成
        total_time += tool_time   # 等待工具（GPU 空闲！）
    return total_time

# === 异步 Rollout ===
async def async_single_rollout(gen_time: float = 0.5, tool_time: float = 2.0):
    """单个异步 Rollout"""
    await asyncio.sleep(gen_time)  # GPU 生成
    await asyncio.sleep(tool_time)  # 等待工具（其他 Rollout 可以利用 GPU）
    return True

async def async_rollout(num_rollouts: int, gen_time: float = 0.5, tool_time: float = 2.0):
    """异步 Rollout：并发执行，GPU 不等待"""
    tasks = [async_single_rollout(gen_time, tool_time) for _ in range(num_rollouts)]
    start = time.time()
    await asyncio.gather(*tasks)
    return time.time() - start


# 性能对比
num_rollouts = 8
sync_time = sync_rollout(num_rollouts)
async_time = asyncio.run(async_rollout(num_rollouts))

print(f"同步 Rollout ({num_rollouts} 条): {sync_time:.1f}s")
print(f"异步 Rollout ({num_rollouts} 条): {async_time:.1f}s")
print(f"加速比: {sync_time / async_time:.1f}x")
# 典型输出: 同步 20.0s / 异步 2.5s → 加速比 8.0x
# （同步 = 8 × (0.5 生成 + 2.0 等工具) = 20.0s；异步 8 条并发，总耗时 ≈ 2.5s）
```

---

## 21.2 verl 的 Server-based 架构

### 21.2.1 系统组件

| 组件 | 角色 | 说明 |
|------|------|------|
| `AgentLoopBase` | 客户端（Client） | 执行 Agent 逻辑、工具调用（`verl/experimental/agent_loop/agent_loop.py`） |
| `LLMServerClient` | 推理网关 | 提供 generate 接口给 AgentLoop（`verl/workers/rollout/llm_server.py`） |
| `FullyAsyncLLMServerClient` | 全异步服务端 | 连接推理引擎的 DP group，做真正的「生成与训练解耦」 |

> verl v0.9.1 里**没有** `AsyncServer` 这个类，也没有 `AgentLoop` 这个基类名——
> 真实的类名是 `AgentLoopBase`（Agent 循环基类）、`LLMServerClient`（同步/半异步网关）
> 和 `FullyAsyncLLMServerClient`（全异步网关）。

### 21.2.2 Agent Loop 核心实现

```python
"""verl Agent Loop 核心逻辑（简化版）

展示异步 Rollout 的关键流程：
Agent 发起推理请求 → 工具调用 → 多轮交互 → 收集轨迹

注意：这是**按 verl 的设计写的教学简化版**，不是 verl 源码。
verl 真实的基类叫 `AgentLoopBase`（见 verl/experimental/agent_loop/agent_loop.py），
推理网关叫 `LLMServerClient` / `FullyAsyncLLMServerClient`
（见 verl/workers/rollout/llm_server.py）。
"""
import asyncio
from typing import Callable, Optional


class AgentLoop:
    """Agent Loop 实现（简化版，参考 verl 的 AgentLoopBase）
    
    核心设计：
    1. Agent 和推理引擎（Server）分离
    2. 使用 asyncio 实现并发 Rollout
    3. 工具调用不阻塞 GPU 推理
    """
    
    def __init__(self, llm_client, tools: dict, max_turns: int = 10):
        """
        Args:
            llm_client: 推理引擎客户端（连接 LLMServerClient / FullyAsyncLLMServerClient）
            tools: 可用工具字典 {"tool_name": callable}
            max_turns: 最大交互轮数
        """
        self.llm_client = llm_client
        self.tools = tools
        self.max_turns = max_turns
    
    async def run_single_rollout(self, prompt: str, ground_truth=None, max_retries: int = 2) -> dict:
        """执行单条异步 Rollout
         
        Args:
            prompt: 用户输入
            ground_truth: 正确答案（用于计算奖励）
            max_retries: 工具调用解析失败时的最大重试次数
         
        Returns:
            完整轨迹信息
        """
        conversation = [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": prompt},
        ]
        
        trajectory = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "turns": [],
            "token_ids": [],       # 所有生成的 token
            "log_probs": [],       # 对数概率（用于 advantage 计算）
            "tool_calls": [],
        }
        
        for turn in range(self.max_turns):
            # 1. 异步请求推理引擎生成回复
            #    关键：这里不阻塞，其他 Rollout 可以并发使用 GPU
            try:
                response = await self.llm_client.generate_async(
                    messages=conversation,
                    max_new_tokens=512,
                    temperature=0.8,
                )
            except Exception as e:
                # 生成失败时记录错误并终止
                trajectory["turns"].append({
                    "turn": turn,
                    "type": "error",
                    "content": f"Generation failed: {str(e)}",
                })
                break
            
            # 记录生成的 token 信息
            trajectory["token_ids"].extend(response["token_ids"])
            trajectory["log_probs"].extend(response["log_probs"])
            
            assistant_msg = response["text"]
            conversation.append({"role": "assistant", "content": assistant_msg})
            
            # 2. 解析是否有工具调用（带重试）
            tool_call = self._parse_tool_call(assistant_msg)
            retries = 0
            while tool_call is None and retries < max_retries:
                # 要求模型重新生成（修复格式）
                retry_msg = "工具调用格式错误，请使用正确的 JSON 格式重新调用。"
                conversation.append({"role": "user", "content": retry_msg})
                response = await self.llm_client.generate_async(
                    messages=conversation,
                    max_new_tokens=256,
                    temperature=0.8,
                )
                assistant_msg = response["text"]
                conversation.append({"role": "assistant", "content": assistant_msg})
                tool_call = self._parse_tool_call(assistant_msg)
                retries += 1
            
            if tool_call is None:
                # 重试后仍无有效工具调用 → Agent 给出了最终回复
                trajectory["turns"].append({
                    "turn": turn,
                    "type": "final_answer",
                    "content": assistant_msg,
                })
                break
            
            # 3. 执行工具调用（异步，不阻塞 GPU）
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            
            tool_result = await self._execute_tool_async(tool_name, tool_args)
            
            trajectory["turns"].append({
                "turn": turn,
                "type": "tool_call",
                "tool": tool_name,
                "args": tool_args,
                "result": tool_result,
            })
            trajectory["tool_calls"].append({
                "tool": tool_name,
                "success": "error" not in str(tool_result).lower(),
            })
            
            # 将工具结果添加到对话
            conversation.append({
                "role": "tool",
                "content": str(tool_result),
            })
        
        return trajectory
    
    async def _execute_tool_async(self, tool_name: str, args: dict) -> str:
        """异步执行工具调用"""
        if tool_name not in self.tools:
            return f"Error: tool '{tool_name}' not found"
        
        try:
            # 在线程池中执行同步的工具函数，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.tools[tool_name](**args)
            )
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _parse_tool_call(self, text: str) -> Optional[dict]:
        """从模型输出中解析工具调用"""
        import json
        import re
        
        # 尝试解析 JSON 格式的工具调用
        match = re.search(r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\}', text)
        if match:
            try:
                return {
                    "name": match.group(1),
                    "arguments": json.loads(match.group(2)),
                }
            except json.JSONDecodeError:
                pass
        return None


class AsyncRolloutManager:
    """异步 Rollout 管理器
    
    并发执行多个 Rollout，最大化 GPU 利用率。
    """
    
    def __init__(self, agent_loop: AgentLoop, concurrency: int = 32):
        self.agent_loop = agent_loop
        self.concurrency = concurrency  # 并发 Rollout 数量
    
    async def run_batch(self, prompts: list, ground_truths: list = None) -> list:
        """批量异步 Rollout
        
        Args:
            prompts: prompt 列表
            ground_truths: 正确答案列表
        
        Returns:
            轨迹列表
        """
        if ground_truths is None:
            ground_truths = [None] * len(prompts)
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def limited_rollout(prompt, gt):
            async with semaphore:
                return await self.agent_loop.run_single_rollout(prompt, gt)
        
        tasks = [
            limited_rollout(p, gt) for p, gt in zip(prompts, ground_truths)
        ]
        
        trajectories = await asyncio.gather(*tasks)
        return list(trajectories)
```

### 21.2.3 关键设计：为什么不用标准 Chat Completion API

```
标准 Chat Completion API 的问题：
1. Token ↔ Text 转换可能不可逆
   → tokenizer.encode(tokenizer.decode(tokens)) != tokens
   
2. 训练阶段需要精确的 token-level log probability
   → Chat API 只返回文本，丢失了 token 级信息
   
3. Advantage 计算依赖精确的 log_probs
   → 如果 token 对不上，advantage 计算不准确
   → 直接影响训练效果

verl 的解决方案：
- LLMServerClient 直接操作 token 级别
- 返回 (token_ids, log_probs) 而非纯文本
- 保证 advantage 计算的准确性
```

---

## 21.3 分布式训练

### 21.3.1 FSDP（Fully Sharded Data Parallel，全分片数据并行）

```python
"""FSDP 分布式训练配置示例"""
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def setup_fsdp(model, config):
    """配置 FSDP 分布式训练
    
    FSDP 将模型参数分片到多个 GPU 上，
    每个 GPU 只持有模型的一部分参数。
    
    Args:
        model: PyTorch 模型
        config: 训练配置
    """
    # 自动包装策略：按 Transformer 层分片
    # 常见模型对应的层类型：
    #   Qwen2 / Qwen2.5:   transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer
    #   Llama / Llama3:    transformers.models.llama.modeling_llama.LlamaDecoderLayer
    #   Mistral:           transformers.models.mistral.modeling_mistral.MistralDecoderLayer
    #   DeepSeek:          transformers.models.deepseek.modeling_deepseek.DeepSeekDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # 支持混合精度训练
    )
    
    return model


# FSDP 配置（YAML 格式，verl 使用）
#
# ⚠️ 下面这段是**示意（非 verl 配置键）**，用的是 PyTorch FSDP 的原生参数名，
# 只用来说明「verl 的 fsdp 策略背后对应哪些 FSDP 概念」。
# verl 自己**没有** fsdp.sharding_strategy / fsdp.cpu_offload /
# fsdp.mixed_precision / fsdp.backward_prefetch 这些顶层键；
# verl v0.9.1 里相关的是：
#   actor_rollout_ref.actor.strategy=fsdp
#   actor_rollout_ref.ref.strategy=fsdp
#   actor_rollout_ref.actor.fsdp_config.*     # offload / prefetch 等子项
#   actor_rollout_ref.ref.fsdp_config.*        # 参考模型同上
# 具体的子项名字以你本地的 verl/trainer/config/_generated_ppo_trainer.yaml 为准。
fsdp_config = """
# 示意（非 verl 配置键）—— PyTorch FSDP 的原生参数
sharding_strategy: FULL_SHARD       # 全分片
cpu_offload: false                  # 是否将非活跃参数卸载到 CPU
mixed_precision: bf16               # 混合精度（推荐 bf16）
backward_prefetch: BACKWARD_PRE     # 反向传播预取
forward_prefetch: true              # 前向传播预取
"""
```

### 21.3.2 Ray 集群配置

```python
"""Ray 集群初始化（verl 使用 Ray 作为分布式计算框架）"""
import ray


def init_ray_cluster(num_gpus: int = 4, num_cpus: int = 32):
    """初始化 Ray 集群
    
    Ray 管理多个 Worker（工作节点）：
    - Rollout Workers: 负责生成和工具调用
    - Training Workers: 负责模型更新
    - Inference Workers: 负责推理引擎
    
    Args:
        num_gpus: GPU 数量
        num_cpus: CPU 数量
    """
    ray.init(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        # 内存配置
        object_store_memory=10 * 1024**3,  # 10GB 对象存储
        # 日志配置
        log_to_driver=True,
        ignore_reinit_error=True,
    )
    
    print(f"Ray 集群已初始化: {num_gpus} GPUs, {num_cpus} CPUs")
    print(f"可用资源: {ray.available_resources()}")


# Ray Worker 配置（YAML 格式）
#
# ⚠️ 下面这段是**示意（非 verl 配置键）**。verl v0.9.1 里
# `ray_kwargs.ray_init` 只有 `num_cpus` 和 `runtime_env.py_executable`，
# **没有** `num_workers` / `max_restarts` / `health_check_interval` /
# `restart_delay` 这四个键，也没有 `ray.rollout.*` / `ray.inference.*` 这类节点。
# 重启策略属于 **Ray 自己**的范畴（`ray.init(max_restarts=..., ...)` 或
# Ray Cluster 的 autoscaler 配置），不在 verl 的 YAML 里配。
ray_config = """
# 示意（非 verl 配置键）—— 这些数字由你自己的 Ray 集群 / 推理服务决定
num_workers: 4                # Worker 数量（通常 = GPU 数量）
num_gpus_per_worker: 1        # 每个 Worker 的 GPU 数
num_cpus_per_worker: 8        # 每个 Worker 的 CPU 数

# Rollout Workers（负责异步 Rollout）
rollout:
  num_replicas: 2             # Rollout 副本数
  max_concurrent: 32          # 每个副本最大并发 Rollout 数

# Inference Workers（负责推理引擎）
inference:
  engine: sglang              # 推理引擎（sglang 或 vllm）
  tensor_parallel_size: 2     # 张量并行度
  max_num_seqs: 256           # 最大并发序列数
"""

"""
num_replicas 与 max_concurrent 的配比原则：

  总并发 Rollout 数 = num_replicas × max_concurrent

  计算依据：
  1. 推理引擎吞吐 = max_num_seqs / avg_generation_time
  2. Rollout 并发量应匹配推理引擎吞吐，避免过载或空闲
  3. 工具调用耗时越长，需要的 max_concurrent 越大

  经验公式：
    max_concurrent ≈ 推理引擎吞吐 × 平均工具调用耗时

  示例（7B 模型，avg_gen=0.5s, avg_tool=2.0s）:
    - 推理引擎吞吐 ≈ 256 / 0.5 = 512 seq/s
    - max_concurrent ≈ 512 × 2.0 / num_replicas = 512
    - 实际建议取 32-64（受显存和 CPU 线程限制）

  推荐值：
  | 模型大小 | num_replicas | max_concurrent |
  |---------|-------------|---------------|
  | 3B      | 4           | 64            |
  | 7B      | 2           | 32            |
  | 14B     | 1           | 16            |
"""
```

---

## 21.4 性能优化技巧

### 21.4.1 推理引擎选择

| 引擎 | 优势 | 适用场景 | 安装 |
|------|------|---------|------|
| vLLM | 稳定、兼容性好 | 通用场景 | `pip install vllm` |
| SGLang | 性能更优、支持 RadixAttention | 高吞吐场景 | `pip install sglang[all]` |

```python
"""推理引擎配置对比

⚠️ 张量并行度的键是 `rollout.tensor_model_parallel_size`（顶层，**不是**挂在
vllm 子节点下的写法）；引擎私有的参数一律走 `rollout.engine_kwargs.<engine>.*`
（vLLM / SGLang 各自透传）。
"""

# vLLM 配置
vllm_config = """
actor_rollout_ref:
  rollout:
    name: vllm                 # 引擎名（rollout.name）
    tensor_model_parallel_size: 2   # 张量并行度
    engine_kwargs:
      vllm:
        gpu_memory_utilization: 0.8  # GPU 显存使用率
        max_num_seqs: 256             # 最大并发序列
        swap_space: 4                 # CPU swap 空间 (GB)
"""

# SGLang 配置（推荐，性能更优）
sglang_config = """
actor_rollout_ref:
  rollout:
    name: sglang
    tensor_model_parallel_size: 2
    engine_kwargs:
      sglang:
        mem_fraction_static: 0.8     # 静态显存分配比例
        max_running_requests: 256    # 最大并发请求
        enable_radix_cache: true     # RadixAttention 缓存（减少重复计算）
"""
```

> `engine_kwargs` 下面挂的是**引擎原生的启动参数**，verl 只做透传——
> 所以这里写什么由 sglang / vllm 的版本决定，键名不对会直接启动失败。
> 引擎本身是用 `rollout.name` 选的（`vllm` / `sglang`），
> 这个键请以你本地的 `verl/trainer/config/_generated_ppo_trainer.yaml` 为准。

### 21.4.2 Load Balancing（负载均衡）

```python
"""verl Stream Mode 负载均衡调度（概念示意）"""
import asyncio
from collections import deque


class LoadBalancer:
    """verl 的负载均衡器
    
    将 Rollout 请求均匀分配到多个推理引擎实例，
    减少长尾请求（Long-tail Requests）对性能的影响。
    
    Stream Mode 调度策略：
    - 维护一个请求队列
    - 每个推理引擎实例从队列中取请求
    - 完成快的实例自动获取更多请求
    """
    
    def __init__(self, num_engines: int):
        self.num_engines = num_engines
        self.engine_loads = [0] * num_engines  # 每个引擎的当前负载
        self.request_queue = deque()
    
    def get_least_loaded_engine(self) -> int:
        """获取负载最低的引擎"""
        min_load = min(self.engine_loads)
        engine_id = self.engine_loads.index(min_load)
        return engine_id
    
    async def dispatch(self, requests: list) -> list:
        """将请求分发到各引擎"""
        results = []
        tasks = []
        
        for req in requests:
            engine_id = self.get_least_loaded_engine()
            self.engine_loads[engine_id] += 1
            # 异步分发到对应引擎
            tasks.append(self._send_to_engine(engine_id, req))
        
        results = await asyncio.gather(*tasks)
        
        # 更新负载
        for i in range(self.num_engines):
            self.engine_loads[i] = 0
        
        return results
    
    async def _send_to_engine(self, engine_id: int, request):
        """发送请求到指定推理引擎
        
        通过 aiohttp 连接 SGLang / vLLM 推理引擎的 HTTP 接口。
        """
        import aiohttp
        
        # 推理引擎 API 地址（实际部署时替换）
        engine_urls = [
            "http://localhost:30000/generate",  # SGLang
            "http://localhost:8000/v1/completions",  # vLLM
        ]
        url = engine_urls[engine_id % len(engine_urls)]
        
        payload = {
            "prompt": request["prompt"],
            "max_new_tokens": request.get("max_new_tokens", 512),
            "temperature": request.get("temperature", 0.8),
            "n": 1,
            "stream": False,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                return {"engine_id": engine_id, "result": result}
```

### 21.4.3 Batch Size 优化

> ⚠️ **verl 里没有 `rollout_batch_size` 这个键。** 一次迭代实际生成多少条轨迹，
> 由下面两个键的**乘积**决定：
>
> | 概念 | verl 里的键 |
> |------|-------------|
> | 每次迭代取多少个 prompt | `data.train_batch_size` |
> | 每个 prompt 采样几条回复（GRPO 组大小） | `actor_rollout_ref.rollout.n` |
> | 实际生成的轨迹条数 | `train_batch_size × rollout.n` |
>
> 常见的「per-GPU batch size」直觉在 verl 里对应 `data.train_batch_size / n_gpus`
> （再乘上 `rollout.n` 才是每卡实际生成的条数）。
> 注意 verl 的整除约束：`train_batch_size × rollout.n` 须被 `n_gpus` 整除
> （`verl/utils/config.py:123-124`）——这和 TRL 的 `% num_generations` 规则**不同**。

```
推荐配置公式（按 verl 的真实键写）:
  data.train_batch_size = n_gpus × per_gpu_batch_size
  actor_rollout_ref.rollout.n = G            # GRPO 组大小

per_gpu_batch_size 推荐值：
  - 3B 模型:  8-16
  - 7B 模型:  4-8
  - 14B 模型: 2-4

示例（4 × A100 80G, 7B 模型）：
  data.train_batch_size = 4 × 8 = 32
  actor_rollout_ref.rollout.n = 8
  → 每次迭代处理 32 / 8 = 4 个不同的 prompt
  → 每个 prompt 采样 8 个回复
  → 总共生成 32 × 8 = 256 条轨迹用于训练
  → 整除校验：256 % 4 == 0 ✓
```

> **硬件需求**以 [24 - 学习总结与进阶路径](./24-学习总结与进阶路径.md) §24.6.1
> 的表为准（3B：SFT 1× RTX 4090 24GB，GRPO 2× RTX 4090 或 1× A100 80GB；
> 24GB 单卡能跑 SFT，GRPO 需要 ≥48GB 显存或开 offload）。

---

## 21.5 完整的分布式训练脚本

```bash
#!/bin/bash
# run_distributed.sh —— 多 GPU 分布式 Agentic RL 训练
#
# ⚠️ verl 训练需要 CUDA 环境；本仓库没有 CI 在跑这段脚本，下面的键只做过离线核对。
#
# --- verl 必填项（这 4 个键务必显式写出）---
#   actor_rollout_ref.actor.optim.lr=1e-6
#       该键在不同 verl 版本的有效默认值不同（v0.9.1 是 1e-6，更早版本是 1e-3），
#       所以不要依赖默认值，必须显式写。
#       （分片策略也是 actor/ref 各自一个键：actor_rollout_ref.actor.strategy、
#        actor_rollout_ref.ref.strategy，没有 actor_rollout_ref.strategy 这种顶层键）
#   data.prompt_key=messages
#       数据集里对话列的列名。
#   data.train_batch_size=32
#   actor_rollout_ref.rollout.n=8
#       verl 的整除约束：train_batch_size × rollout.n 须被 n_gpus 整除
#       （verl/utils/config.py:123-124），这里 32 × 8 = 256，8 卡时每卡 32 条。
#       ⚠️ 这和 TRL 的规则**不同**：TRL 要求 generation_batch_size
#       （= per_device_train_batch_size × num_processes × steps_per_generation）
#       能被 num_generations 整除。两套规则不要混用。
#
# --- bash 小坑（写这个脚本时踩过）---
#   反斜杠续行（`\`）里的 `#` **不是注释**，会被当成普通参数传给命令。
#   所以下面这条命令内部一行注释都没有，注释全部写在命令开始之前。

set -e

# === 集群配置 ===
NUM_GPUS=8
NUM_GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR="localhost"
MASTER_PORT=29500

# === 模型与数据 ===
MODEL_PATH="Qwen/Qwen2.5-7B"
TRAIN_DATA="data/train.jsonl"
TEST_DATA="data/test.jsonl"
OUTPUT_DIR="checkpoints/distributed_v1"
# enable / disable（从上次 checkpoint 恢复）
RESUME_MODE=${RESUME_MODE:-"disable"}

# === 启动分布式训练 ===
# trainer.max_actor_ckpt_to_keep=3：最多保留 3 个 actor checkpoint
# trainer.logger 是**列表**（不是可以挂子键的对象），WandB 的项目名/实验名
# 走顶层的 trainer.project_name / trainer.experiment_name
# ⚠️ 列表值要整体加引号：trainer.logger='[console,wandb,tensorboard]'，
#    写成 trainer.logger=['console', 'wandb', 'tensorboard'] 会被 bash 拆成
#    三个参数，Hydra 直接解析失败
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.main_ppo \
    \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.prompt_key=messages \
    data.train_batch_size=32 \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.01 \
    \
    trainer.total_epochs=3 \
    trainer.project_name=agentic_rl_distributed \
    trainer.experiment_name=qwen2_5_7b_8gpu \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.logger='[console,wandb,tensorboard]' \
    trainer.default_local_dir=${OUTPUT_DIR}

echo "分布式训练完成！模型保存在 ${OUTPUT_DIR}/global_step_<N>/actor"
```

> **容错 / 重启策略不在 verl 的 YAML 里。** verl v0.9.1 的
> `ray_kwargs.ray_init` 只有 `num_cpus` 和 `runtime_env.py_executable`，
> **没有** `num_workers` / `max_restarts` / `health_check_interval` /
> `restart_delay` 这四个键。这些是 **Ray 自己的**参数：
> 在 `ray.init(max_restarts=N, ...)` 里传，或者写进 Ray Cluster 的 autoscaler 配置，
> 再由 `ray_kwargs.ray_init.runtime_env` 把环境带进 worker。
>
> **checkpoint 落盘位置**：verl 把权重写到 `${OUTPUT_DIR}/global_step_<N>/actor`。
> `trainer.max_actor_ckpt_to_keep=3` 表示最多保留 3 个 actor checkpoint；
> verl 保留 checkpoint 数量的键只有 `trainer.max_actor_ckpt_to_keep`，没有
> `save_limit` / `save_total_limit` 那种写法（后者写带空格的 `50 GB` 还会直接让
> Hydra 解析失败）。
>
> **梯度累积**也不在 `trainer.*` 下：verl 没有
> `trainer.per_device_train_batch_size` / `trainer.gradient_accumulation_steps`
> （那是 HuggingFace `Trainer` 的键）。verl 用 `data.train_batch_size` 和
> `actor_rollout_ref.actor.fsdp_config.*` 自行切 micro-batch。
>
> `trainer.total_epochs=3` 与全系列统一口径一致（SFT 5 / GRPO 3）。

---

## 21.6 术语速查

| 英文术语 | 中文注释 |
|---------|----------|
| Async Rollout | 异步展开（多 Rollout 并发执行） |
| Synchronous Rollout | 同步展开（逐个执行，等待工具时 GPU 空闲） |
| Server-based Architecture | 基于服务的架构（推理引擎与 Agent 分离） |
| FSDP (Fully Sharded Data Parallel) | 全分片数据并行（PyTorch 原生分布式） |
| DeepSpeed ZeRO | 微软的分布式训练优化（ZeRO-1/2/3） |
| Load Balancing | 负载均衡（请求均匀分配到推理引擎） |
| Long-tail Requests | 长尾请求（少数慢请求拖慢整体性能） |
| Tensor Parallelism | 张量并行（将模型层分配到多 GPU） |
| RadixAttention | SGLang 的前缀缓存优化技术 |
| torchrun | PyTorch 分布式训练启动器 |

---

## 21.7 小结

- 异步 Rollout 解决了 Agent 训练中 GPU 空闲的问题：§21.1.3 的脚本实测 **8.0×**（同步 20.0 s vs 异步 2.5 s）
- verl 的 Server-based 架构分离了推理引擎和 Agent 逻辑，使用 asyncio 并发
- FSDP + Ray 是分布式训练的核心基础设施
- 推理引擎推荐 SGLang（性能更优），Batch Size 需根据 GPU 和模型大小调整
- 下一步：前沿方向 → [22 - 多Agent与Multi-Agent RL](./22-多Agent与Multi-Agent-RL.md)
