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
# 典型输出: 加速比 3-5x
```

---

## 21.2 verl 的 Server-based 架构

### 21.2.1 系统组件

| 组件 | 角色 | 说明 |
|------|------|------|
| AgentLoop | 客户端（Client） | 执行 Agent 逻辑、工具调用 |
| LLMServerClient | 推理网关 | 提供 generate 接口给 AgentLoop |
| AsyncServer | 服务端（Server） | 连接推理引擎的 DP group |

### 21.2.2 Agent Loop 核心实现

```python
"""verl Agent Loop 核心逻辑（简化版）

展示异步 Rollout 的关键流程：
Agent 发起推理请求 → 工具调用 → 多轮交互 → 收集轨迹
"""
import asyncio
from typing import Callable, Optional


class AgentLoop:
    """verl 的 Agent Loop 实现（简化版）
    
    核心设计：
    1. Agent 和推理引擎（Server）分离
    2. 使用 asyncio 实现并发 Rollout
    3. 工具调用不阻塞 GPU 推理
    """
    
    def __init__(self, llm_client, tools: dict, max_turns: int = 10):
        """
        Args:
            llm_client: 推理引擎客户端（连接 AsyncServer）
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
fsdp_config = """
actor_rollout_ref:
  strategy: fsdp
  fsdp:
    sharding_strategy: FULL_SHARD       # 全分片
    cpu_offload: false                   # 是否将非活跃参数卸载到 CPU
    mixed_precision: bf16                # 混合精度（推荐 bf16）
    backward_prefetch: BACKWARD_PRE      # 反向传播预取
    forward_prefetch: true               # 前向传播预取
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
ray_config = """
ray:
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
"""推理引擎配置对比"""

# vLLM 配置
vllm_config = """
actor_rollout_ref:
  rollout:
    engine: vllm
    vllm:
      tensor_parallel_size: 2      # 张量并行度
      gpu_memory_utilization: 0.8  # GPU 显存使用率
      max_num_seqs: 256            # 最大并发序列
      swap_space: 4                # CPU swap 空间 (GB)
"""

# SGLang 配置（推荐，性能更优）
sglang_config = """
actor_rollout_ref:
  rollout:
    engine: sglang
    sglang:
      tensor_parallel_size: 2
      mem_fraction_static: 0.8     # 静态显存分配比例
      max_running_requests: 256    # 最大并发请求
      enable_radix_cache: true     # RadixAttention 缓存（减少重复计算）
"""
```

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

```
推荐配置公式：
  rollout_batch_size = num_gpus × per_gpu_batch_size

per_gpu_batch_size 推荐值：
  - 3B 模型:  8-16
  - 7B 模型:  4-8
  - 14B 模型: 2-4

示例（4 × A100 80G, 7B 模型）：
  rollout_batch_size = 4 × 8 = 32
  GRPO group_size = 8
  → 每次迭代处理 32 / 8 = 4 个不同的 prompt
  → 每个 prompt 采样 8 个回复
  → 总共生成 32 条轨迹用于训练
```

---

## 21.5 完整的分布式训练脚本

```bash
#!/bin/bash
# run_distributed.sh —— 多 GPU 分布式 Agentic RL 训练

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
RESUME_MODE=${RESUME_MODE:-"disable"}  # enable / disable（从上次 checkpoint 恢复）

# === 启动分布式训练 ===
torchrun \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m verl.trainer.main \
    \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${TEST_DATA} \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.strategy=fsdp \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.max_new_tokens=1024 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.engine=sglang \
    \
    algorithm=grpo \
    algorithm.kl_coeff=0.01 \
    \
    trainer.total_epochs=5 \
    trainer.project_name=agentic_rl_distributed \
    trainer.experiment_name=qwen2_5_7b_8gpu \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    \
    trainer.per_device_train_batch_size=4 \
    trainer.gradient_accumulation_steps=2 \
    \
    trainer.default_local_dir=${OUTPUT_DIR} \
    \
    # === Checkpoint / 容错 ===
    trainer.resume_mode=${RESUME_MODE} \
    trainer.save_freq=1 \
    trainer.save_limit=3 \              # 最多保留 3 个 checkpoint
    trainer.save_total_limit=50 GB \    # checkpoint 磁盘限额
    \
    # === 容错配置 ===
    ray.num_workers=4 \
    ray.max_restarts=3 \                # Worker 崩溃后最大重启次数
    ray.health_check_interval=30 \      # 健康检查间隔（秒）
    ray.restart_delay=10 \              # 崩溃后等待秒数再重启
    \
    # === 训练监控 ===
    trainer.logger=['console', 'wandb', 'tensorboard'] \
    trainer.logger.wandb.project=agentic_rl_distributed \
    trainer.logger.wandb.name=qwen2_5_7b_8gpu

echo "分布式训练完成！模型保存在 ${OUTPUT_DIR}"
```

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

- 异步 Rollout 解决了 Agent 训练中 GPU 空闲的问题，典型加速 3-5 倍
- verl 的 Server-based 架构分离了推理引擎和 Agent 逻辑，使用 asyncio 并发
- FSDP + Ray 是分布式训练的核心基础设施
- 推理引擎推荐 SGLang（性能更优），Batch Size 需根据 GPU 和模型大小调整
- 下一步：前沿方向 → [22 - 多Agent与Multi-Agent RL](./22-多Agent与Multi-Agent-RL.md)
