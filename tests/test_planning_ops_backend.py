from __future__ import annotations

import unittest

from domains.planning_ops.backend import PlanningOpsReasoningDomain


class PlanningOpsBackendTests(unittest.TestCase):
    def test_project_plan_solves_via_fallback_tool(self) -> None:
        backend = PlanningOpsReasoningDomain()
        task = backend.manual_task(
            "project_plan",
            "Create a valid project plan.\nTasks:\n- design (duration=1, priority=3, deps=none)\n- build (duration=2, priority=4, deps=design)\n- test (duration=1, priority=2, deps=build)\nReturn the ordered task plan.",
            "design -> build -> test",
        )
        state = backend.make_state(task)

        child, info = backend.create_executor().apply(state, backend.fallback_repairs(state)[0])

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "design -> build -> test")
        self.assertGreater(info["goal_progress"], 0.0)

    def test_shopping_and_day_plans_are_normalized_for_answer_checking(self) -> None:
        backend = PlanningOpsReasoningDomain()

        shopping_task = backend.manual_task(
            "shopping_plan",
            "Choose the best shopping plan under budget=5.\nItems:\n- bread (cost=2, priority=5)\n- milk (cost=3, priority=4)\n- fruit (cost=4, priority=6)\n- tea (cost=1, priority=2)\nReturn the chosen items in input order separated by commas.",
            "bread, milk",
        )
        self.assertTrue(backend.evaluate_answer(shopping_task, "bread,milk"))

        day_task = backend.manual_task(
            "day_plan",
            "Create the best day plan under time_limit=3.\nTasks:\n- design (duration=1, priority=2, deps=none)\n- build (duration=2, priority=4, deps=design)\n- review (duration=2, priority=3, deps=none)\nReturn the ordered task plan.",
            "design -> build",
        )
        self.assertTrue(backend.evaluate_answer(day_task, "design->build"))


if __name__ == "__main__":
    unittest.main()
