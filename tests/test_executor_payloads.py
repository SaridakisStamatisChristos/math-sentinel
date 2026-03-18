from __future__ import annotations

import unittest

from engine.actions import Action, ActionType
from engine.executor import StateExecutor
from engine.state import ReasoningState


class _StructuredRegistry:
    def call(self, name: str, arg: str, state: ReasoningState | None = None) -> dict[str, object]:
        return {
            "ok": True,
            "result_text": "derived_fact",
            "result_payload": {"dependencies": ["dep_a"], "obligations": ["finish_subgoal"]},
            "goal_progress": 0.8,
            "solved": True,
            "answer": "final_answer",
            "risk": 0.1,
        }


class ExecutorPayloadTests(unittest.TestCase):
    def test_executor_normalizes_payloads_and_updates_state(self) -> None:
        state = ReasoningState(task_id="task_1", domain="toy", problem_text="p", goal="g")
        executor = StateExecutor(_StructuredRegistry())

        child, info = executor.apply(state, Action(type=ActionType.APPLY, tool="toy_tool", content="p"))

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "final_answer")
        self.assertIn("derived_fact", child.derived_facts)
        self.assertIn("finish_subgoal", child.obligations)
        self.assertIn("dep_a", child.dependency_refs)
        self.assertGreater(child.terminal_confidence, 0.0)
        self.assertGreater(info["goal_progress"], 0.0)


if __name__ == "__main__":
    unittest.main()
