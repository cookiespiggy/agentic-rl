"""多轮信息收集环境: 策略类 Agentic RL 任务.

与 rl/reward.py 的情感分类任务（判别类）相对，本模块提供一个**策略类**任务：
Agent 必须通过多轮追问集齐必要槽位，才能给出最终方案。

为什么这个任务需要 RL 而不是 Schema:
- "问什么、按什么顺序问、什么时候该停" 是**策略**问题，不是判断问题
- 同一目标存在多条路径（先问城市还是先问人数），有探索空间
- 最终成败取决于**整条轨迹**，不取决于任何单步判断

设计说明与改造步骤见 docs/04-多步任务改造方案.md。
"""
from __future__ import annotations

REQUIRED_SLOTS: tuple[str, ...] = ("city", "people", "time")

# 奖励常量（密集模式）
R_ASK_VALID = 0.2      # 问到未收集的必要槽位
R_ASK_INVALID = -0.2   # 重复追问或问无关槽位
R_ANSWER_OK = 0.4      # 集齐后给出方案
R_ANSWER_EARLY = -0.5  # 信息不全就抢答
R_ILLEGAL = -0.3       # 非法动作


class InfoCollectionEnv:
    """多轮信息收集环境.

    参数:
        case:     {"request": str, "slots": {"city": ..., "people": ..., "time": ...}}
        max_turns: 单局最大轮数
    """

    def __init__(self, case: dict, max_turns: int = 5) -> None:
        self.case = case
        self.max_turns = max_turns
        self.collected: dict[str, str] = {}
        self.turns = 0

    def step(self, action: dict) -> tuple[dict, float, bool]:
        """执行一个动作, 返回 (observation, reward, done)."""
        self.turns += 1
        kind = action.get("type")

        if kind == "ask":
            slot = action.get("slot")
            if slot in REQUIRED_SLOTS and slot not in self.collected:
                self.collected[slot] = self.case["slots"][slot]
                return self._obs(), R_ASK_VALID, False
            return self._obs(), R_ASK_INVALID, False

        if kind == "answer":
            complete = len(self.collected) == len(REQUIRED_SLOTS)
            return self._obs(), (R_ANSWER_OK if complete else R_ANSWER_EARLY), True

        return self._obs(), R_ILLEGAL, False

    def succeeded(self) -> bool:
        """是否集齐全部必要槽位."""
        return len(self.collected) == len(REQUIRED_SLOTS)

    def _obs(self) -> dict:
        return {
            "turns": self.turns,
            "collected": dict(self.collected),
            "remaining": [s for s in REQUIRED_SLOTS if s not in self.collected],
        }


def episode_return(actions: list[dict], case: dict) -> float:
    """稀疏奖励模式: 整条轨迹只在结束时给一次奖励.

    成功 = 1.0, 失败 = 0.0。

    与密集模式的区别: 密集模式每步即时反馈, 会削弱信用分配
    （模型可能学会"多问几次拿小奖励"而非"尽快完成"）；
    稀疏模式把信用分配问题完整暴露出来, 更接近真实 RLVR 场景。
    """
    env = InfoCollectionEnv(case)
    for action in actions:
        _, _, done = env.step(action)
        if done:
            break
    return 1.0 if env.succeeded() else 0.0
