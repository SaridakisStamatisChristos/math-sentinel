from __future__ import annotations

import unittest

from domains.gaia_ops.backend import GaiaOpsReasoningDomain


class GaiaOpsBackendTests(unittest.TestCase):
    def test_evidence_driven_fallback_loop_solves_csv_case_without_oracle_tool_hints(self) -> None:
        backend = GaiaOpsReasoningDomain()
        task = backend.benchmark_tasks()[0]
        task.meta.pop("oracle_evidence_file", None)
        task.meta.pop("oracle_tool", None)
        task.meta.pop("oracle_input", None)
        state = backend.make_state(task)
        executor = backend.create_executor()

        info = {}
        for _ in range(6):
            repair = backend.fallback_repairs(state)[0]
            state, info = executor.apply(state, repair)
            if state.metadata.get("candidate_answer"):
                answer_action = backend.fallback_repairs(state)[0]
                state, info = executor.apply(state, answer_action)
            if state.status == "solved":
                break

        self.assertEqual(state.status, "solved")
        self.assertEqual(state.final_answer, "22")
        self.assertIn("22", " ".join(state.evidence_refs + state.derived_facts))
        self.assertGreaterEqual(float(info["goal_progress"]), 0.8)
        self.assertEqual(state.metadata.get("target_file"), "sales.csv")

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
        self.assertEqual(state.metadata.get("target_file"), "sales.csv")
