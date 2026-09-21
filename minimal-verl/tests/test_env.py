"""rl/env.py 的单元测试.

运行:
    python tests/test_env.py

不依赖 pytest, 只用标准库 unittest。
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "rl"))

from env import REQUIRED_SLOTS, InfoCollectionEnv, episode_return  # noqa: E402

CASE = {
    "request": "帮我订个餐厅",
    "slots": {"city": "上海", "people": "4", "time": "今晚7点"},
}


class TestInfoCollectionEnv(unittest.TestCase):
    def test_ask_valid_slot(self):
        env = InfoCollectionEnv(CASE)
        _, reward, done = env.step({"type": "ask", "slot": "city"})
        self.assertAlmostEqual(reward, 0.2)
        self.assertFalse(done)

    def test_ask_duplicate_slot(self):
        env = InfoCollectionEnv(CASE)
        env.step({"type": "ask", "slot": "city"})
        _, reward, _ = env.step({"type": "ask", "slot": "city"})
        self.assertAlmostEqual(reward, -0.2)

    def test_ask_irrelevant_slot(self):
        env = InfoCollectionEnv(CASE)
        _, reward, _ = env.step({"type": "ask", "slot": "weather"})
        self.assertAlmostEqual(reward, -0.2)

    def test_answer_early_penalised(self):
        env = InfoCollectionEnv(CASE)
        env.step({"type": "ask", "slot": "city"})
        _, reward, done = env.step({"type": "answer", "content": "x"})
        self.assertAlmostEqual(reward, -0.5)
        self.assertTrue(done)

    def test_answer_after_all_slots(self):
        env = InfoCollectionEnv(CASE)
        for slot in REQUIRED_SLOTS:
            env.step({"type": "ask", "slot": slot})
        _, reward, done = env.step({"type": "answer", "content": "上海今晚7点4人"})
        self.assertAlmostEqual(reward, 0.4)
        self.assertTrue(done)
        self.assertTrue(env.succeeded())

    def test_illegal_action(self):
        env = InfoCollectionEnv(CASE)
        _, reward, _ = env.step({"type": "jump"})
        self.assertAlmostEqual(reward, -0.3)

    def test_observation_shape(self):
        env = InfoCollectionEnv(CASE)
        obs, _, _ = env.step({"type": "ask", "slot": "city"})
        self.assertEqual(obs["turns"], 1)
        self.assertEqual(obs["collected"], {"city": "上海"})
        self.assertEqual(sorted(obs["remaining"]), ["people", "time"])


class TestEpisodeReturn(unittest.TestCase):
    def test_success(self):
        actions = [
            {"type": "ask", "slot": "city"},
            {"type": "ask", "slot": "people"},
            {"type": "ask", "slot": "time"},
            {"type": "answer", "content": "ok"},
        ]
        self.assertAlmostEqual(episode_return(actions, CASE), 1.0)

    def test_answer_too_early(self):
        actions = [
            {"type": "ask", "slot": "city"},
            {"type": "answer", "content": "ok"},
        ]
        self.assertAlmostEqual(episode_return(actions, CASE), 0.0)

    def test_only_duplicates(self):
        actions = [{"type": "ask", "slot": "city"}] * 3
        self.assertAlmostEqual(episode_return(actions, CASE), 0.0)

    def test_empty_trajectory(self):
        self.assertAlmostEqual(episode_return([], CASE), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
