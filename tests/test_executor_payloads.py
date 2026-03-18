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
            "result_payload": {
                "dependencies": ["dep_a"],
                "obligations": ["finish_subgoal"],
                "resolved_obligations": ["old_obligation"],
                "evidence": ["fact from tool"],
                "candidate_answer": "final_answer",
            },
            "goal_progress": 0.8,
            "solved": True,
            "answer": "final_answer",
            "risk": 0.1,
        }


class _FailingStructuredRegistry:
    def call(self, name: str, arg: str, state: ReasoningState | None = None) -> dict[str, object]:
        return {
            "ok": False,
            "result_text": "draft failed",
            "result_payload": {
                "failed_patch_candidates": [{"path": "app.py", "score": 0.5}],
                "state_metadata": {"failed_patch_attempt_count": 1},
            },
            "risk": 0.9,
        }


class ExecutorPayloadTests(unittest.TestCase):
    def test_executor_normalizes_payloads_and_updates_state(self) -> None:
        state = ReasoningState(task_id="task_1", domain="toy", problem_text="p", goal="g", obligations=["old_obligation"])
        executor = StateExecutor(_StructuredRegistry())

        child, info = executor.apply(state, Action(type=ActionType.APPLY, tool="toy_tool", content="p"))

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "final_answer")
        self.assertIn("derived_fact", child.derived_facts)
        self.assertIn("finish_subgoal", child.obligations)
        self.assertNotIn("old_obligation", child.obligations)
        self.assertIn("dep_a", child.dependency_refs)
        self.assertIn("fact from tool", child.evidence_refs)
        self.assertEqual(child.metadata.get("candidate_answer"), "final_answer")
        self.assertGreater(child.terminal_confidence, 0.0)
        self.assertGreater(info["goal_progress"], 0.0)

    def test_executor_preserves_failed_tool_payloads_for_analysis(self) -> None:
        state = ReasoningState(task_id="task_2", domain="toy", problem_text="p", goal="g")
        executor = StateExecutor(_FailingStructuredRegistry())

        child, info = executor.apply(state, Action(type=ActionType.APPLY, tool="draft_patch", content="p"))

        self.assertEqual(info["valid_step"], 0.0)
        self.assertEqual(child.metadata.get("failed_patch_attempt_count"), 1)
        self.assertEqual(len(child.tool_payloads), 1)
        self.assertFalse(child.tool_payloads[0]["ok"])


if __name__ == "__main__":
    unittest.main()
