from __future__ import annotations

import unittest

from domains.math.backend import MathReasoningDomain
from engine.actions import Action, ActionType
from engine.executor import StateExecutor
from engine.state import ReasoningState


class _Registry:
    def call(self, name: str, arg: str, state: ReasoningState | None = None) -> dict[str, object]:
        if name == "echo":
            return {"ok": True, "result": arg, "solved": True, "answer": arg, "goal_progress": 1.0}
        return {"ok": False, "result": f"unknown tool: {name}"}


class EngineCoreTests(unittest.TestCase):
    def test_state_executor_uses_custom_answer_judge(self) -> None:
        state = ReasoningState(
            task_id="task_1",
            domain="toy",
            problem_text="Return yes",
            goal="Answer using domain policy",
            expected_answer="ignored",
            metadata={"accepted": "yes"},
        )
        executor = StateExecutor(_Registry(), answer_judge=lambda s, candidate: candidate == str(s.metadata.get("accepted")))

        child, info = executor.apply(state, Action(type=ActionType.ANSWER, content="yes"))

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "yes")
        self.assertEqual(info["proof_completion"], 1.0)

    def test_math_backend_creates_solver_ready_executor(self) -> None:
        backend = MathReasoningDomain()
        task = backend.manual_task("arithmetic", "Compute: 2 + 3")
        state = backend.make_state(task)
        repair = backend.fallback_repairs(state)[0]

        child, info = backend.create_executor().apply(state, repair)

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "5")
        self.assertGreater(info["goal_progress"], 0.0)


if __name__ == "__main__":
    unittest.main()
