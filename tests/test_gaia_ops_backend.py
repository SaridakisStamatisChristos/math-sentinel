from __future__ import annotations

import unittest

from domains.gaia_ops.backend import GaiaOpsReasoningDomain


class GaiaOpsBackendTests(unittest.TestCase):
    def test_recommended_tool_repair_solves_csv_case(self) -> None:
        backend = GaiaOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        state = backend.make_state(task)
        executor = backend.create_executor()

        repair = backend.fallback_repairs(state)[1]
        child, info = executor.apply(state, repair)

        self.assertEqual(child.status, "solved")
        self.assertEqual(child.final_answer, "22")
        self.assertGreaterEqual(float(info["goal_progress"]), 1.0)

    def test_benchmark_tasks_cover_multiple_reasoning_families(self) -> None:
        backend = GaiaOpsReasoningDomain()
        families = {task.domain for task in backend.benchmark_tasks()}

        self.assertEqual(families, {"gaia_csv_reasoning", "gaia_json_reasoning", "gaia_schedule_reasoning"})

    def test_manual_task_matches_fixture_case_for_sample_flow(self) -> None:
        backend = GaiaOpsReasoningDomain()

        task = backend.manual_task(
            "gaia_csv_reasoning",
            "Use the files in the workspace to answer this question: what is the total sales amount for the east region in sales.csv? Return only the number.",
        )

        self.assertIn("fixture_dir", task.meta)
        state = backend.make_state(task)
        self.assertIn("sales.csv", state.metadata["workspace_files"])
